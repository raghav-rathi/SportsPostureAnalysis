"""Phase 2 (M2A): Telemetry v2.0 Builder, <35 ms Instant CPU HITL Recompute & Demo Session Seeder.

Exploits the CPU/GPU seam:
  - Raw 3D detections (`raw_nlf.npz`) are cached once from GPU (or fixture).
  - Re-calibrating MCC 4-click landmarks, switching FPS (30/60/120/240), nudging causal events
    (BFC < FFC <= UAH < BR), changing bowling arm, or updating workload context runs strictly
    on local CPU in <35 ms with ZERO GPU round-trip!
  - Explicitly flags `needs_gpu_refit_for_K = True` whenever a new 4-click calibration changes
    focal length by >15% relative to the intrinsics originally passed into NLF.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from ..constants import JOINT_IDX, MCC_PITCH_LANDMARKS_M, SMPL24_JOINTS
from ..schemas import (
    BiomechanicalFactSheet,
    BowlingArm,
    DeliveryEvents,
    EventTimestamp,
    ObservabilityGrade,
    PitchCalibration,
    VideoAudit,
    WorkloadContext,
)
from ..stage1_pan_tracker import track_camera_rotation_3dof_video, transform_cam_to_pitch_world_per_frame
from ..stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    solve_two_wicket_calibration,
    transform_cam_to_pitch_world,
)
from ..stage2_ball_release import estimate_ball_release_speed_video
from ..stage2_dual_sync_fusion import quantify_metrics_from_fused_covariance
from ..stage2_event_spotting import detect_delivery_events
from ..stage2_isb_anatomy import extract_isb_anatomy_kinematics
from ..stage3_kinematics import extract_fast_bowling_kinematics
from ..stage4_uncertainty import METRIC_SPECS, _clamp_ci, quantify_and_gate_metrics
from ..stage5_evaluator import build_fact_sheet, evaluate_repeatability
from ..stage6_coach_llm import generate_grounded_report, render_report_markdown
from ..synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)
from .db import StudioDatabase
from .video_proxy import render_synthetic_proxy_video


def project_mcc_wireframe_2d(cal: PitchCalibration) -> Dict[str, List[float]]:
    """Project all canonical MCC Law 6/7/8 pitch landmarks into 2D image coordinates for live wireframe overlay."""
    K = np.asarray(cal.camera_matrix_K, dtype=np.float64).reshape(3, 3)
    R = np.asarray(cal.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(cal.translation_t, dtype=np.float64).reshape(3)
    out: Dict[str, List[float]] = {}
    for name, pt_w in MCC_PITCH_LANDMARKS_M.items():
        pc = R @ np.asarray(pt_w, dtype=np.float64) + t
        if pc[2] <= 0.2:
            continue
        uv = K @ pc
        out[name] = [round(float(uv[0] / uv[2]), 1), round(float(uv[1] / uv[2]), 1)]
    return out


def apply_causal_event_overrides(
    events: DeliveryEvents,
    overrides: Dict[str, Any],
    fps: float,
    n_frames: int,
    source_frame_indices: Optional[List[int]] = None,
) -> DeliveryEvents:
    """Apply user-nudged event frames with strict causal clamping `0 <= BFC < FFC <= UAH < BR <= n-1`."""
    if not overrides:
        return events

    bfc_f = float(overrides["bfc_frame"]) if overrides.get("bfc_frame") is not None else events.bfc.frame_float
    ffc_f = float(overrides["ffc_frame"]) if overrides.get("ffc_frame") is not None else events.ffc.frame_float
    br_f = float(overrides["br_frame"]) if overrides.get("br_frame") is not None else events.br.frame_float
    uah_f = (
        float(overrides["uah_frame"])
        if overrides.get("uah_frame") is not None
        else (events.uah.frame_float if events.uah is not None else None)
    )

    # Strict causal clamp
    max_i = float(max(n_frames - 1, 3))
    br_f = float(np.clip(br_f, 2.0, max_i))
    ffc_f = float(np.clip(ffc_f, 1.0, br_f - 0.5))
    bfc_f = float(np.clip(bfc_f, 0.0, ffc_f - 0.5))
    if uah_f is not None:
        uah_f = float(np.clip(uah_f, ffc_f, br_f - 0.25))

    def _update_ev(old: EventTimestamp, new_ff: float, was_overridden: bool) -> EventTimestamp:
        if not was_overridden and abs(new_ff - old.frame_float) < 1e-4:
            return old
        idx = int(np.clip(round(new_ff), 0, max(n_frames - 1, 0)))
        src_i = int(source_frame_indices[idx]) if (source_frame_indices and 0 <= idx < len(source_frame_indices)) else idx
        return EventTimestamp(
            frame_float=round(new_ff, 3),
            frame_idx=idx,
            source_frame_idx=src_i,
            timestamp_s=round(new_ff / max(fps, 1.0), 4),
            timing_uncertainty_ms=round(500.0 / max(fps, 1.0), 2),
            confidence=0.98,
            detection_method="human_verified_override" if was_overridden else old.detection_method,
        )

    new_bfc = _update_ev(events.bfc, bfc_f, overrides.get("bfc_frame") is not None)
    new_ffc = _update_ev(events.ffc, ffc_f, overrides.get("ffc_frame") is not None)
    new_br = _update_ev(events.br, br_f, overrides.get("br_frame") is not None)
    new_uah = (
        _update_ev(events.uah or new_ffc, uah_f, overrides.get("uah_frame") is not None)
        if uah_f is not None
        else events.uah
    )

    stride_ms = max((new_ffc.timestamp_s - new_bfc.timestamp_s) * 1000.0, 1.0)
    ffc_br_ms = max((new_br.timestamp_s - new_ffc.timestamp_s) * 1000.0, 1.0)
    # If human explicitly verified BR, mark delivery complete unless explicitly forced incomplete
    is_comp = True if overrides.get("br_frame") is not None else events.is_complete_delivery

    return events.model_copy(
        update={
            "bfc": new_bfc,
            "ffc": new_ffc,
            "uah": new_uah,
            "br": new_br,
            "delivery_stride_duration_ms": round(stride_ms, 1),
            "ffc_to_br_duration_ms": round(ffc_br_ms, 1),
            "is_complete_delivery": is_comp,
        }
    )


def build_telemetry_v2(
    joints_world_m: np.ndarray,
    joints2d_px: np.ndarray,
    joint_uncertainties_m: np.ndarray,
    joint_in_frame: np.ndarray,
    kin_series: Dict[str, np.ndarray],
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    proxy_fps: float = 30.0,
    all_tracks: Optional[List[Dict[str, Any]]] = None,
    camera_poses_per_frame: Optional[List[Dict[str, Any]]] = None,
    isb_summary: Optional[Dict[str, Any]] = None,
    needs_gpu_refit_for_K: bool = False,
) -> Dict[str, Any]:
    """Build versioned `telemetry.json` (`schema_version = "2.0"`) for 60-120 fps browser scrubbing."""
    n = joints_world_m.shape[0]
    src_map = audit.source_frame_indices if (audit.source_frame_indices and len(audit.source_frame_indices) == n) else list(range(n))
    pts_ms = [round(i * 1000.0 / max(proxy_fps, 1.0), 3) for i in range(n)]
    t_rel_ffc_ms = [round((i - events.ffc.frame_float) * 1000.0 / max(audit.fps, 1.0), 2) for i in range(n)]

    # Ground-zeroed 3D world skeleton for the Three.js Pitch Deck so bowler stands cleanly on Y=0
    y_ground = float(np.percentile(joints_world_m[:, [JOINT_IDX["lank"], JOINT_IDX["rank"]], 1], 8)) if n > 0 else 0.0
    jw_render = joints_world_m.copy()
    jw_render[:, :, 1] -= y_ground

    # Round floats compactly (3 decimals = 1 mm precision!) to keep telemetry payload crisp & fast (<250 KB)
    return {
        "schema_version": "2.0",
        "joint_names": list(SMPL24_JOINTS),
        "num_frames": n,
        "analysis_fps": audit.fps,
        "proxy_fps": proxy_fps,
        "source_frame_indices": src_map,
        "frame_pts_ms": pts_ms,
        "time_rel_ffc_ms": t_rel_ffc_ms,
        "width": audit.width,
        "height": audit.height,
        "y_ground_offset_m": round(y_ground, 4),
        "joints3d_world_m": np.round(jw_render, 4).tolist(),
        "joints2d_px": np.round(joints2d_px, 1).tolist(),
        "joint_uncertainties_m": np.round(joint_uncertainties_m, 4).tolist(),
        "joint_in_frame": joint_in_frame.astype(bool).tolist(),
        "kin_series": {k: np.round(np.asarray(v, dtype=np.float64), 2).tolist() for k, v in kin_series.items()},
        "mcc_wireframe_2d": project_mcc_wireframe_2d(cal) if cal.is_intrinsics_calibrated else {},
        "camera_position_world_m": [round(float(x), 3) for x in cal.camera_position_world_m],
        "camera_poses_per_frame": camera_poses_per_frame,
        "all_tracks": all_tracks or [],
        "isb_anatomy": isb_summary,
        "needs_gpu_refit_for_K": bool(needs_gpu_refit_for_K),
    }


def recompute_delivery_cpu(
    data_dir: str | Path,
    bowler_id: str,
    overrides: Optional[Dict[str, Any]] = None,
    video_path: Optional[str | Path] = None,
    proxy_fps: float = 30.0,
) -> Dict[str, Any]:
    """Execute instant (<35 ms) CPU HITL recompute over cached `raw_nlf.npz` + `audit.json`."""
    t0 = time.perf_counter()
    ddir = Path(data_dir)
    npz_path = ddir / "raw_nlf.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cached 3D trajectory not found: {npz_path}")

    ov = dict(overrides or {})
    z = np.load(npz_path, allow_pickle=False)
    j3d_cam = z["joints3d_cam_m"].astype(np.float64)
    j2d = z["joints2d_px"].astype(np.float64)
    j_unc = z["joint_uncertainties_m"].astype(np.float64)
    j_in = z["joint_in_frame"].astype(bool)
    valid_mask = z["valid_mask"].astype(bool)
    boxes = z["bowler_boxes_xyxy"].astype(np.float64) if "bowler_boxes_xyxy" in z else None
    bml87_cam = z["bml_movi_87_cam_m"].astype(np.float64) if "bml_movi_87_cam_m" in z else None
    bml87_unc = z["bml_movi_87_unc_m"].astype(np.float64) if "bml_movi_87_unc_m" in z else None
    cov_world_fused = z["cov_world_fused"].astype(np.float64) if "cov_world_fused" in z else None
    dual_baseline_sin = float(z["dual_baseline_sin"]) if "dual_baseline_sin" in z else None
    all_tracks = json.loads(str(z["all_tracks_json"])) if "all_tracks_json" in z else []

    audit = VideoAudit.model_validate_json((ddir / "audit.json").read_text(encoding="utf-8"))
    base_cal = PitchCalibration.model_validate_json((ddir / "base_calibration.json").read_text(encoding="utf-8"))
    base_meta = json.loads((ddir / "meta.json").read_text(encoding="utf-8")) if (ddir / "meta.json").exists() else {}
    if dual_baseline_sin is None and "dual_baseline_sin" in base_meta:
        dual_baseline_sin = float(base_meta["dual_baseline_sin"])

    # 1. FPS override (3-tier slow-mo resolver pill!)
    if ov.get("capture_fps_override") is not None:
        new_fps = float(ov["capture_fps_override"])
        audit = audit.model_copy(
            update={
                "fps": round(new_fps, 3),
                "duration_s": round(audit.num_frames / max(new_fps, 1.0), 3),
                "fps_grade": "high_speed_240" if new_fps >= 200 else ("slow_mo_120" if new_fps >= 100 else ("standard_60" if new_fps >= 50 else "low_25")),
                "est_arm_deg_per_frame": round(2300.0 / max(new_fps, 1.0), 2),
                "est_ball_m_per_frame": round(38.89 / max(new_fps, 1.0), 3),
                "timebase_status": "user_fps_override" if audit.timebase_unreliable else audit.timebase_status,
            }
        )

    arm = BowlingArm(ov["bowling_arm"]) if ov.get("bowling_arm") else BowlingArm(base_meta.get("bowling_arm", "right"))

    # 2. Calibration override (4-click MCC PnP or two-wicket pixels)
    cal = base_cal
    needs_gpu_refit_for_K = False
    nlf_fx_used = float(base_meta.get("nlf_fx_used", base_cal.camera_matrix_K[0][0]))
    if ov.get("mcc_landmarks_px") and len(ov["mcc_landmarks_px"]) >= 4:
        lm_map = {k: (float(v[0]), float(v[1])) for k, v in ov["mcc_landmarks_px"].items()}
        cal = calibrate_from_mcc_landmarks(
            lm_map,
            width=audit.width,
            height=audit.height,
            is_panning_camera=base_cal.is_panning_camera,
        )
        new_fx = float(cal.camera_matrix_K[0][0])
        if abs(new_fx - nlf_fx_used) / max(nlf_fx_used, 1.0) > 0.15:
            needs_gpu_refit_for_K = True
    elif ov.get("two_wickets_px"):
        tw = ov["two_wickets_px"]
        cal = solve_two_wicket_calibration(
            h_near_px=float(tw["h_near_px"]),
            base_near_y_px=float(tw["base_near_y_px"]),
            h_far_px=float(tw["h_far_px"]),
            base_far_y_px=float(tw["base_far_y_px"]),
            width=audit.width,
            height=audit.height,
            is_panning_camera=base_cal.is_panning_camera,
        )
        new_fx = float(cal.camera_matrix_K[0][0])
        if abs(new_fx - nlf_fx_used) / max(nlf_fx_used, 1.0) > 0.15:
            needs_gpu_refit_for_K = True

    # 3. Optional 3-DOF Rotational Pan Tracking (M2B) when enabled + MCC calibrated + video available
    pan_summary: Optional[Dict[str, Any]] = None
    pan_res_verified = None
    camera_poses_per_frame: Optional[List[Dict[str, Any]]] = None
    if ov.get("enable_pan_tracking") and cal.is_intrinsics_calibrated and video_path and Path(video_path).exists():
        src_indices = audit.source_frame_indices or list(range(j3d_cam.shape[0]))
        pan_res = track_camera_rotation_3dof_video(
            video_path=video_path,
            source_frame_indices=src_indices,
            K=np.asarray(cal.camera_matrix_K, dtype=np.float64),
            R_anchor=np.asarray(cal.rotation_matrix_R, dtype=np.float64),
            t_anchor=np.asarray(cal.translation_t, dtype=np.float64),
            anchor_frame=int(ov.get("pan_anchor_frame", len(src_indices) // 2)),
            bowler_boxes_xyxy=boxes,
        )
        pan_summary = {
            "is_verified": pan_res.is_verified,
            "median_residual_px": pan_res.median_residual_px,
            "p90_residual_px": pan_res.p90_residual_px,
            "max_pan_deg": pan_res.max_pan_deg,
            "sigma_pan_deg": pan_res.sigma_pan_deg,
            "verification_reason": pan_res.verification_reason,
        }
        if pan_res.is_verified:
            pan_res_verified = pan_res
            j3d_world = transform_cam_to_pitch_world_per_frame(j3d_cam, pan_res.R_per_frame, pan_res.t_per_frame)
            # Never claim metric calibration if K shifted >15% without a GPU NLF refit OR base cal wasn't metric!
            metric_ok = bool(cal.is_metric_calibrated and not needs_gpu_refit_for_K)
            cal = cal.model_copy(
                update={
                    "is_panning_camera": False,
                    "is_metric_calibrated": metric_ok,
                    "heading_sigma_deg": round(math.hypot(cal.heading_sigma_deg, pan_res.sigma_pan_deg), 2),
                    "calibration_notes": list(cal.calibration_notes) + [pan_res.verification_reason],
                }
            )
            camera_poses_per_frame = [
                {"frame": i, "R": np.round(pan_res.R_per_frame[i], 5).tolist(), "t": np.round(pan_res.t_per_frame[i], 4).tolist()}
                for i in range(len(pan_res.R_per_frame))
            ]
        else:
            j3d_world = transform_cam_to_pitch_world(j3d_cam, cal)
    else:
        j3d_world = transform_cam_to_pitch_world(j3d_cam, cal)

    # 4. Event spotting + optional human causal-clamped event overrides
    events = detect_delivery_events(
        j3d_world,
        fps=audit.fps,
        bowling_arm=arm,
        joint_in_frame=j_in,
        source_frame_indices=audit.source_frame_indices or None,
    )
    if ov.get("event_overrides"):
        events = apply_causal_event_overrides(
            events,
            ov["event_overrides"],
            fps=audit.fps,
            n_frames=j3d_world.shape[0],
            source_frame_indices=audit.source_frame_indices or None,
        )

    # 5. Stage 3 kinematics + optional M2B ISB bml_movi_87 rigid-body anatomy
    kin = extract_fast_bowling_kinematics(
        j3d_world,
        events,
        fps=audit.fps,
        bowling_arm=arm,
        joint_uncertainties_m=j_unc,
        joint_in_frame=j_in,
        camera_R_w2c=np.asarray(cal.rotation_matrix_R, dtype=np.float64),
    )

    isb_summary: Optional[Dict[str, Any]] = None
    if bml87_cam is not None and bml87_cam.shape[0] == j3d_cam.shape[0]:
        # De-rotate bml_movi_87 per-frame when 3-DOF pan tracking is verified (P0-2 #4)
        if pan_res_verified is not None:
            bml87_world = transform_cam_to_pitch_world_per_frame(bml87_cam, pan_res_verified.R_per_frame, pan_res_verified.t_per_frame)
        else:
            bml87_world = transform_cam_to_pitch_world(bml87_cam, cal)
        isb = extract_isb_anatomy_kinematics(
            bml87_world,
            events,
            bowling_arm=arm,
            markers_unc_87=bml87_unc,
            cal=cal,
        )
        isb_summary = {
            "scalars": isb.scalars,
            "metrics": {k: m.model_dump() for k, m in isb.metrics.items()},
            "n_eff_pelvis": isb.n_eff_pelvis,
            "n_eff_thorax": isb.n_eff_thorax,
            "n_eff_foot": isb.n_eff_foot,
            "rho_assumed": isb.rho_assumed,
            "front_foot_strike_mode": isb.front_foot_strike_mode,
        }

    # 6. Stage 4 uncertainty & Stage 5 FactSheet (zero unvalidated 0-100 composites!)
    workload_ctx: Optional[WorkloadContext] = None
    if ov.get("workload"):
        workload_ctx = WorkloadContext.model_validate(ov["workload"])

    cal_overridden = bool(ov.get("mcc_landmarks_px") or ov.get("two_wickets_px"))
    if (
        cov_world_fused is not None
        and cov_world_fused.shape[:2] == (j3d_world.shape[0], 24)
        and (not cal_overridden)
        and pan_res_verified is None
    ):
        metrics = quantify_metrics_from_fused_covariance(
            kin, audit, cal, events, cov_world_fused, float(dual_baseline_sin or 0.0)
        )
    else:
        metrics = quantify_and_gate_metrics(kin, audit, cal, events=events, dual_baseline_sin=dual_baseline_sin)

    # Propagate 3-DOF rotational pan translational drift D * sigma_pan_rad into world translation metrics (P1-2)
    if pan_res_verified is not None and pan_res_verified.sigma_pan_deg > 0.0:
        cam_pos = np.asarray(cal.camera_position_world_m, dtype=np.float64)
        pelv_mean = np.mean(j3d_world[:, JOINT_IDX["pelv"]], axis=0)
        cam_dist_m = max(float(np.linalg.norm(cam_pos - pelv_mean)), 2.0)
        drift_m = cam_dist_m * math.radians(pan_res_verified.sigma_pan_deg)
        dt_run = max(min(0.22, events.bfc.frame_float / max(audit.fps, 1.0)), 1.0 / max(audit.fps, 1.0))
        pan_extras = {
            "runup_speed_mps": drift_m / max(dt_run, 0.04),
            "stride_length_pct_stature": 100.0 * drift_m / max(kin.estimated_stature_m, 1.0),
            "follow_through_drift_deg": math.degrees(math.atan2(drift_m, 1.2)),
        }
        for pk, s_extra in pan_extras.items():
            me = metrics.get(pk)
            if me is None or me.suppressed or me.value is None:
                continue
            spec = METRIC_SPECS[pk]
            sup_thresh = float(spec[3])
            new_cal = math.hypot(float(me.sigma_calibration), s_extra)
            new_tot = math.hypot(float(me.sigma_total), s_extra)
            if new_tot > sup_thresh:
                metrics[pk] = me.model_copy(
                    update={
                        "value": None,
                        "ci_low": None,
                        "ci_high": None,
                        "sigma_calibration": round(new_cal, 2),
                        "sigma_total": round(new_tot, 2),
                        "observability": ObservabilityGrade.UNMEASURABLE,
                        "suppressed": True,
                        "suppression_reason": (
                            f"Suppressed: 3-DOF pan translational drift uncertainty (+-{new_tot:.2f} {me.unit} at D={cam_dist_m:.1f} m) "
                            f"exceeds reliability limit ({sup_thresh:.1f} {me.unit})."
                        ),
                    }
                )
            else:
                ci_lo, ci_hi = _clamp_ci(pk, float(me.value) - 1.96 * new_tot, float(me.value) + 1.96 * new_tot)
                metrics[pk] = me.model_copy(
                    update={
                        "ci_low": round(ci_lo, 2),
                        "ci_high": round(ci_hi, 2),
                        "sigma_calibration": round(new_cal, 2),
                        "sigma_total": round(new_tot, 2),
                    }
                )

    # Wire M2D calibrated side-on wrist-ROI optical ball speed estimator when source/proxy video exists (P1-1)
    if video_path and Path(video_path).exists():
        wri_i = JOINT_IDX["rwri" if arm == BowlingArm.RIGHT else "lwri"]
        wri_unc_br = float(j_unc[min(events.br.frame_idx, len(j_unc) - 1), wri_i])
        ball_res = estimate_ball_release_speed_video(
            video_path=video_path,
            joints2d_px=j2d,
            joints_world_m=j3d_world,
            audit=audit,
            cal=cal,
            events=events,
            bowling_arm=arm,
            wrist_unc_m=max(wri_unc_br, 0.12),
        )
        metrics["ball_release_speed_kph"] = ball_res.metric

    fs = build_fact_sheet(
        bowler_id=bowler_id,
        bowling_arm=arm,
        action_class=kin.action_class,
        pose_backend=str(base_meta.get("pipeline_mode", "single_view_studio")),
        audit=audit,
        cal=cal,
        events=events,
        metrics=metrics,
        workload=workload_ctx,
    )
    report = generate_grounded_report(fs)
    report_md = render_report_markdown(fs, report)

    telemetry = build_telemetry_v2(
        joints_world_m=kin.joints_world_smoothed if kin.joints_world_smoothed is not None else j3d_world,
        joints2d_px=j2d,
        joint_uncertainties_m=j_unc,
        joint_in_frame=j_in,
        kin_series=kin.series,
        audit=audit,
        cal=cal,
        events=events,
        proxy_fps=proxy_fps,
        all_tracks=all_tracks,
        camera_poses_per_frame=camera_poses_per_frame,
        isb_summary=isb_summary,
        needs_gpu_refit_for_K=needs_gpu_refit_for_K,
    )

    # Persist updated fact_sheet.json & telemetry.json locally
    (ddir / "fact_sheet.json").write_text(fs.model_dump_json(indent=2), encoding="utf-8")
    (ddir / "telemetry.json").write_text(json.dumps(telemetry), encoding="utf-8")
    (ddir / "coaching_report.md").write_text(report_md, encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "fact_sheet": fs,
        "report": report,
        "report_markdown": report_md,
        "telemetry": telemetry,
        "isb_summary": isb_summary,
        "pan_summary": pan_summary,
        "recompute_ms": round(elapsed_ms, 2),
        "needs_gpu_refit_for_K": needs_gpu_refit_for_K,
    }


def compute_session_comparison(deliveries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Wire Pillar 3 `evaluate_repeatability` + FFC-aligned multi-ball spaghetti envelopes & fatigue drift."""
    fact_sheets: List[BiomechanicalFactSheet] = []
    labels: List[str] = []
    curves_by_metric: Dict[str, List[Tuple[str, np.ndarray, np.ndarray]]] = {
        "front_knee_deg": [],
        "thorax_lat_flex_deg": [],
        "trunk_flex_deg": [],
        "shoulder_yaw_deg": [],
    }
    drift_series: Dict[str, List[Dict[str, Any]]] = {
        "front_knee_br_deg": [],
        "thorax_lat_flex_br_deg": [],
        "shoulder_counter_rotation_deg": [],
        "trunk_flexion_delta_deg": [],
    }

    for idx, d in enumerate(deliveries):
        if not d.get("fact_sheet_json"):
            continue
        fs = BiomechanicalFactSheet.model_validate_json(d["fact_sheet_json"])
        if not fs.events.is_complete_delivery:
            continue
        fact_sheets.append(fs)
        lbl = d.get("label") or f"Ball {idx + 1}"
        labels.append(lbl)

        for mk in drift_series:
            me = fs.metrics.get(mk)
            if me and (not me.suppressed) and me.value is not None:
                drift_series[mk].append({"ball_index": idx + 1, "label": lbl, "value": me.value, "ci_low": me.ci_low, "ci_high": me.ci_high})

        tel_path = Path(d["data_dir"]) / "telemetry.json"
        if tel_path.exists():
            tel = json.loads(tel_path.read_text(encoding="utf-8"))
            t_ms = np.asarray(tel.get("time_rel_ffc_ms", []), dtype=np.float64)
            ks = tel.get("kin_series", {})
            for ck in curves_by_metric:
                if ck in ks and len(ks[ck]) == len(t_ms) and len(t_ms) >= 5:
                    curves_by_metric[ck].append((lbl, t_ms, np.asarray(ks[ck], dtype=np.float64)))

    rep_pillar = evaluate_repeatability(fact_sheets) if len(fact_sheets) >= 2 else None

    # Resample multi-ball waveforms onto shared FFC-aligned grid [-300 ms, +220 ms] for mean +- 1 SD spaghetti plot
    t_grid = np.linspace(-300.0, 220.0, 105)
    spaghetti: Dict[str, Any] = {}
    for ck, items in curves_by_metric.items():
        if not items:
            continue
        interp_rows = []
        per_ball = []
        for lbl, t_arr, y_arr in items:
            yi = np.interp(t_grid, t_arr, y_arr, left=np.nan, right=np.nan)
            interp_rows.append(yi)
            per_ball.append({"label": lbl, "values": [None if not math.isfinite(v) else round(float(v), 2) for v in yi]})
        mat = np.asarray(interp_rows, dtype=np.float64)
        with np.errstate(all="ignore"):
            mu = np.nanmean(mat, axis=0)
            sd = np.nanstd(mat, axis=0)
        spaghetti[ck] = {
            "time_ms": [round(float(x), 1) for x in t_grid],
            "mean": [None if not math.isfinite(v) else round(float(v), 2) for v in mu],
            "sd": [None if not math.isfinite(v) else round(float(v), 2) for v in sd],
            "balls": per_ball,
        }

    # Simple linear slope across spell balls for within-spell fatigue drift indicator
    fatigue_slopes: Dict[str, Optional[float]] = {}
    for mk, pts in drift_series.items():
        if len(pts) >= 3:
            xs = np.array([p["ball_index"] for p in pts], dtype=np.float64)
            ys = np.array([p["value"] for p in pts], dtype=np.float64)
            slope = float(np.polyfit(xs, ys, 1)[0])
            fatigue_slopes[mk] = round(slope, 2)
        else:
            fatigue_slopes[mk] = None

    return {
        "completed_deliveries_count": len(fact_sheets),
        "delivery_labels": labels,
        "repeatability": rep_pillar.model_dump() if rep_pillar is not None else None,
        "spaghetti_curves": spaghetti,
        "spell_drift_points": drift_series,
        "spell_drift_slope_per_ball": fatigue_slopes,
    }


def seed_demo_session(db: StudioDatabase, sessions_root: Path, fixtures_dir: Optional[Path] = None) -> str:
    """Seed an immediately usable offline demo session (240 fps Dual-View Synthetic + Cached Real Steyn & McGrath)."""
    existing = db.list_sessions()
    if existing:
        return existing[0]["id"]

    bowler = db.create_bowler("Demo Pace Bowler (Studio Showcase)", bowling_arm="right", age_band="adult", bowler_id="bwl_demo")
    session = db.create_session(
        bowler_id=bowler["id"],
        title="Phase 2 Showcase — Calibrated 240 fps Spell & Real-Clip Fixtures",
        notes="Includes MCC-calibrated 240 fps Side-On & Rear-On views for Dual-Phone 3D Fusion plus verified v4 real-clip fixtures.",
        session_id="ses_demo",
    )
    today = datetime.now(timezone.utc).date()
    db.add_workload_log(bowler["id"], (today - timedelta(days=5)).isoformat(), 36, "net", "High-intensity target bowling")
    db.add_workload_log(bowler["id"], (today - timedelta(days=3)).isoformat(), 42, "match", "Club 7-over spell")
    db.add_workload_log(bowler["id"], (today - timedelta(days=1)).isoformat(), 24, "net", "Technical front-leg brace drill")

    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg", "popping_crease_off",
        "popping_crease_leg", "stump_middle_base", "stump_middle_top",
    ]
    demo_specs = [
        ("del_demo_side_240", "Ball 1 — Side-On 240 fps (MCC Calibrated)", 90.0, 5.0, 171.0, 36.0, 18.0),
        ("del_demo_rear_240", "Ball 1 — Rear 240 fps (Dual-Sync Mate)", 0.0, 6.0, 171.0, 36.0, 18.0),
        ("del_demo_side_b2", "Ball 2 — Side-On 240 fps (Mild Fatigue)", 90.0, 5.0, 165.0, 40.0, 22.0),
    ]
    for did, lbl, az, el, fk_br, lf_br, scr in demo_specs:
        ddir = sessions_root / session["id"] / did
        ddir.mkdir(parents=True, exist_ok=True)
        params = SyntheticDeliveryParams(
            bowler_arm=BowlingArm.RIGHT,
            fps=240.0,
            front_knee_br_deg=fk_br,
            thorax_lat_flex_br_deg=lf_br,
            shoulder_counter_rot_deg=scr,
        )
        jw, _ = generate_synthetic_delivery(params)
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        j2d, j_cam = project_world_points(jw, K, R, t)
        lm_px = project_mcc_landmarks(K, R, t, lm_keys, noise_px=0.5)
        cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)

        n = jw.shape[0]
        audit = VideoAudit(
            video_path=f"demo:{did}",
            width=1280,
            height=720,
            fps=240.0,
            container_fps=240.0,
            num_frames=n,
            raw_num_frames=n,
            duration_s=round(n / 240.0, 3),
            median_laplacian_var=320.0,
            blur_grade="crisp",
            blur_roi_method="bowler_roi",
            fps_grade="high_speed_240",
            est_arm_deg_per_frame=round(2300.0 / 240.0, 2),
            est_ball_m_per_frame=round(38.89 / 240.0, 3),
            timebase_status="verified_native",
            source_frame_indices=list(range(n)),
        )
        np.savez_compressed(
            ddir / "raw_nlf.npz",
            joints3d_cam_m=j_cam.astype(np.float32),
            joints2d_px=j2d.astype(np.float32),
            joint_uncertainties_m=np.full((n, 24), 0.025, dtype=np.float32),
            joint_in_frame=np.ones((n, 24), dtype=np.bool_),
            valid_mask=np.ones(n, dtype=np.bool_),
        )
        (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps({"bowling_arm": "right", "pipeline_mode": "single_view_studio", "nlf_fx_used": float(K[0, 0])}),
            encoding="utf-8",
        )
        proxy_mp4 = render_synthetic_proxy_video(
            j2d, 1280, 720, ddir / "scrub_proxy.mp4", proxy_fps=30.0, title=lbl, mcc_wireframe_2d=project_mcc_wireframe_2d(cal)
        )
        res = recompute_delivery_cpu(ddir, bowler_id=lbl, proxy_fps=30.0)
        db.upsert_delivery(
            delivery_id=did,
            session_id=session["id"],
            label=lbl,
            data_dir=ddir,
            video_path=str(proxy_mp4),
            proxy_path=str(proxy_mp4),
            pipeline_mode="single_view_studio",
            start_frame=0,
            end_frame=n,
            status="ready",
            fact_sheet_json=res["fact_sheet"].model_dump_json(),
            overrides={"mcc_landmarks_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in lm_px.items()}},
        )

    # Also seed real-clip fixtures (`steyn` and `mcgrath`) if tests/fixtures/*.npz are present
    if fixtures_dir and fixtures_dir.exists():
        for fname, did, lbl in [
            ("steyn_nlf_fixture.npz", "del_real_steyn", "Real Clip — Dale Steyn Net (25.2 fps Two-Wicket)"),
            ("mcgrath_nlf_fixture.npz", "del_real_mcgrath", "Real Clip — Glenn McGrath Reel (Incomplete Gate Demo)"),
        ]:
            fp = fixtures_dir / fname
            if not fp.exists():
                continue
            z = np.load(fp, allow_pickle=False)
            meta = json.loads(str(z["meta_json"]))
            ddir = sessions_root / session["id"] / did
            ddir.mkdir(parents=True, exist_ok=True)
            j_cam = z["joints3d_cam_m"].astype(np.float64)
            cal = PitchCalibration.model_validate(meta["calibration"])
            K = np.asarray(cal.camera_matrix_K, dtype=np.float64).reshape(3, 3)
            # Project 3D camera joints to 2D pixels for crisp fixture overlay
            z_safe = np.maximum(j_cam[:, :, 2:3], 0.2)
            j2d = np.einsum("ij,tkj->tki", K, j_cam)[:, :, :2] / z_safe
            n = j_cam.shape[0]
            fps = float(meta["fps"])
            audit = VideoAudit(
                video_path=f"fixture:{fname}",
                width=int(meta["width"]),
                height=int(meta["height"]),
                fps=fps,
                container_fps=float(meta["container_fps"]),
                num_frames=n,
                raw_num_frames=n,
                duration_s=round(n / max(fps, 1.0), 3),
                median_laplacian_var=433.7 if "steyn" in fname else 209.2,
                blur_grade="crisp",
                blur_roi_method="bowler_roi",
                fps_grade="low_25",
                est_arm_deg_per_frame=round(2300.0 / max(fps, 1.0), 2),
                est_ball_m_per_frame=round(38.89 / max(fps, 1.0), 3),
                timebase_status=str(meta["timebase_status"]),
                timebase_unreliable=bool(meta["timebase_unreliable"]),
                held_frame_ratio=float(meta["held_frame_ratio"]),
                longest_held_run_frames=int(meta["longest_held_run_frames"]),
                source_frame_indices=list(range(n)),
            )
            np.savez_compressed(
                ddir / "raw_nlf.npz",
                joints3d_cam_m=j_cam.astype(np.float32),
                joints2d_px=j2d.astype(np.float32),
                joint_uncertainties_m=z["joint_uncertainties_m"].astype(np.float32),
                joint_in_frame=z["joint_in_frame"].astype(np.bool_),
                valid_mask=z["valid_mask"].astype(np.bool_),
            )
            (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
            (ddir / "base_calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
            (ddir / "meta.json").write_text(
                json.dumps({"bowling_arm": meta["bowling_arm"], "pipeline_mode": "real_clip_fixture", "nlf_fx_used": float(K[0, 0])}),
                encoding="utf-8",
            )
            proxy_mp4 = render_synthetic_proxy_video(j2d, int(meta["width"]), int(meta["height"]), ddir / "scrub_proxy.mp4", proxy_fps=25.0, title=lbl)
            res = recompute_delivery_cpu(ddir, bowler_id=meta["bowler_id"], proxy_fps=25.0)
            db.upsert_delivery(
                delivery_id=did,
                session_id=session["id"],
                label=lbl,
                data_dir=ddir,
                video_path=str(proxy_mp4),
                proxy_path=str(proxy_mp4),
                pipeline_mode="real_clip_fixture",
                start_frame=0,
                end_frame=n,
                status="ready",
                fact_sheet_json=res["fact_sheet"].model_dump_json(),
            )

    return session["id"]
