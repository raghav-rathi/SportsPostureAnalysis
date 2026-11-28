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
from scipy.ndimage import gaussian_filter1d, maximum_filter1d, minimum_filter1d

from ..constants import JOINT_IDX, MCC_PITCH_LANDMARKS_M, SMPL24_JOINTS
from ..schemas import (
    BiomechanicalFactSheet,
    BowlingArm,
    CameraViewClass,
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
from ..stage2_dual_sync_fusion import fuse_dual_view_3d_trajectories, quantify_metrics_from_fused_covariance
from ..stage2_event_spotting import detect_delivery_events
from ..stage2_isb_anatomy import extract_isb_anatomy_kinematics
from ..stage3_kinematics import extract_fast_bowling_kinematics
from ..stage3_v5_metrics import (
    compute_3d_com_trajectory,
    extract_and_quantify_v5_biomechanics,
    synthesize_bml_movi_87_from_smpl24,
)
from ..stage4_uncertainty import METRIC_SPECS, _clamp_ci, quantify_and_gate_metrics
from ..stage5_evaluator import build_fact_sheet, evaluate_repeatability, hydrate_coaching_story
from ..stage6_coach_llm import generate_grounded_report, render_report_markdown
from ..stage6_stroboscopic import build_multi_joint_trails
from ..synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)
from .db import StudioDatabase
from .video_proxy import build_all_intra_scrub_proxy, render_synthetic_proxy_video

_DEMO_SEED_VERSION = "4.0"

# Exact 63 pulldown-deduped frame indices in `uiZN5FHWjJk.mp4` matching `tests/fixtures/steyn_nlf_fixture.npz`
_STEYN_SOURCE_FRAME_INDICES: List[int] = [
    25, 26, 27, 28, 29, 31, 32, 33, 34, 35,
    37, 38, 39, 40, 41, 43, 44, 45, 46, 47,
    49, 50, 51, 52, 53, 55, 56, 57, 58, 59,
    61, 62, 63, 64, 65, 67, 68, 69, 70, 71,
    73, 74, 75, 76, 77, 79, 80, 81, 82, 83,
    85, 86, 87, 88, 89, 91, 92, 93, 94, 95,
    97, 98, 99,
]


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
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    com_world_m: Optional[np.ndarray] = None,
    sequence_series: Optional[Dict[str, np.ndarray]] = None,
    wrist_rel_speed_mps: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Build versioned `telemetry.json` (`schema_version = "2.0"`, minor `2.1`) for 60-120 fps browser scrubbing."""
    n = joints_world_m.shape[0]
    src_map = audit.source_frame_indices if (audit.source_frame_indices and len(audit.source_frame_indices) == n) else list(range(n))
    pts_ms = [round(i * 1000.0 / max(proxy_fps, 1.0), 3) for i in range(n)]
    t_rel_ffc_ms = [round((i - events.ffc.frame_float) * 1000.0 / max(audit.fps, 1.0), 2) for i in range(n)]

    is_crease_calibrated = bool(
        cal.method in ("mcc_law7_pnp", "dual_3d_ray_fusion_dual_mcc_pnp")
        and cal.is_metric_calibrated
        and (not cal.is_panning_camera)
    )
    pitch_display_mode = "calibrated_mcc" if is_crease_calibrated else "bowler_centered"

    com_raw = com_world_m if (com_world_m is not None and len(com_world_m) == n) else compute_3d_com_trajectory(joints_world_m)
    foot_idxs = [JOINT_IDX["lank"], JOINT_IDX["rank"], JOINT_IDX["ltoe"], JOINT_IDX["rtoe"]]
    sole_r_m = 0.028

    if n == 0:
        y_ground = 0.0
        jw_render = joints_world_m.copy()
        com_render = com_raw.copy()
        cam_pos_render = [round(float(x), 3) for x in cal.camera_position_world_m]
    elif is_crease_calibrated:
        # True MCC Law 7 crease-calibrated clip: preserve exact Pitch World (X, Z) so popping/return creases
        # and stumps stay metrically truthful; zero only the static ground-sole plane over [BFC, BR].
        i_bfc_g = int(np.clip(events.bfc.frame_idx, 0, n - 1))
        i_br_g = int(np.clip(events.br.frame_idx, 0, n - 1))
        stance_feet_y = (
            np.nanmin(joints_world_m[i_bfc_g : i_br_g + 1, foot_idxs, 1], axis=1)
            if i_br_g >= i_bfc_g
            else np.nanmin(joints_world_m[:, foot_idxs, 1], axis=1)
        )
        y_ground = float(np.nanpercentile(stance_feet_y, 5) - sole_r_m)
        jw_render = joints_world_m.copy()
        jw_render[:, :, 1] -= y_ground
        com_render = com_raw.copy()
        com_render[:, 1] -= y_ground
        cam_pos_render = [round(float(x), 3) for x in cal.camera_position_world_m]
    else:
        # Non-crease-calibrated clip (`two_wicket_geometry`, `self_calibrated`, or panning camera):
        # Stage 3/4/5 scientific metrics are ALREADY computed on raw `joints_world_m`. For visual 3D display:
        # 1) Track rolling lower-envelope ground under feet so grounded steps kiss Y=0 without flattening jumps.
        # 2) Low-pass filter shared monocular root (X, Z) jitter and center FFC pelvis at (0, 0), applying the
        #    EXACT same per-frame rigid translation `shift[t]` to all 24 joints & COM (100% angle-preserving!).
        foot_min_y = np.nanmin(joints_world_m[:, foot_idxs, 1], axis=1)
        win_jump = max(7, int(round(0.45 * max(float(audit.fps), 25.0))))
        # 1D morphological opening (`max(min(y))`) strips airborne bound peaks (<0.45 s) with ZERO downward slope bias!
        floor_open = maximum_filter1d(
            minimum_filter1d(foot_min_y, size=win_jump, mode="nearest"),
            size=win_jump,
            mode="nearest",
        )
        floor_smooth = gaussian_filter1d(floor_open, sigma=max(1.5, 0.06 * max(float(audit.fps), 25.0)), mode="nearest")
        dy_t = np.minimum(floor_smooth, foot_min_y) - sole_r_m

        pelv_xz = joints_world_m[:, JOINT_IDX["pelv"], :][:, [0, 2]]
        sig_root = max(2.2, 0.22 * max(float(audit.fps), 25.0))
        pelv_xz_smooth = gaussian_filter1d(pelv_xz, sigma=sig_root, axis=0, mode="nearest")
        i_anchor = int(np.clip(events.ffc.frame_idx if events.is_complete_delivery else n // 2, 0, n - 1))
        pelv_xz_disp = pelv_xz_smooth - pelv_xz_smooth[i_anchor : i_anchor + 1, :]
        dx_t = pelv_xz[:, 0] - pelv_xz_disp[:, 0]
        dz_t = pelv_xz[:, 1] - pelv_xz_disp[:, 1]
        shift_t = np.stack([dx_t, dy_t, dz_t], axis=1)

        jw_render = joints_world_m - shift_t[:, None, :]
        com_render = com_raw - shift_t
        y_ground = float(dy_t[i_anchor])
        dx_anc = float(pelv_xz_smooth[i_anchor, 0])
        dz_anc = float(pelv_xz_smooth[i_anchor, 1])
        cam_pos_render = [
            round(float(cal.camera_position_world_m[0] - dx_anc), 3),
            round(float(cal.camera_position_world_m[1]), 3),
            round(float(cal.camera_position_world_m[2] - dz_anc), 3),
        ]

    # Project raw 3D COM into 2D camera pixels (camera extrinsics stay locked to raw Pitch World)
    K_mat = np.asarray(cal.camera_matrix_K, dtype=np.float64).reshape(3, 3)
    R_mat = np.asarray(cal.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    t_vec = np.asarray(cal.translation_t, dtype=np.float64).reshape(3)
    com_cam = (R_mat @ com_raw.T).T + t_vec[None, :]
    com_uvh = (K_mat @ com_cam.T).T
    com_2d_px = com_uvh[:, :2] / np.maximum(com_uvh[:, 2:3], 0.2)

    trails = build_multi_joint_trails(
        joints_world_render_m=jw_render,
        joints2d_px=joints2d_px,
        com_world_render_m=com_render,
        com_2d_px=com_2d_px,
        wrist_rel_speed_mps=wrist_rel_speed_mps,
        events=events,
        fps=audit.fps,
        bowling_arm=bowling_arm,
    )

    # Round floats compactly (3-4 decimals = <=1 mm precision!) to keep telemetry payload crisp & fast
    return {
        "schema_version": "2.0",
        "telemetry_minor_version": "2.1",
        "is_crease_calibrated": is_crease_calibrated,
        "pitch_display_mode": pitch_display_mode,
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
        "com_world_m": np.round(com_render, 4).tolist(),
        "com_2d_px": np.round(com_2d_px, 1).tolist(),
        "joint_uncertainties_m": np.round(joint_uncertainties_m, 4).tolist(),
        "joint_in_frame": joint_in_frame.astype(bool).tolist(),
        "kin_series": {k: np.round(np.asarray(v, dtype=np.float64), 2).tolist() for k, v in kin_series.items()},
        "sequence_series": (
            {k: np.round(np.asarray(v, dtype=np.float64), 2).tolist() for k, v in sequence_series.items()}
            if sequence_series
            else {}
        ),
        "trails": trails,
        "mcc_wireframe_2d": project_mcc_wireframe_2d(cal) if is_crease_calibrated else {},
        "camera_position_world_m": cam_pos_render,
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
    interp_gaps = z["interp_gap_frames"].astype(np.int32) if "interp_gap_frames" in z else None
    held_mask = z["held_frame_mask"].astype(bool) if "held_frame_mask" in z else None
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

    pipeline_mode = str(base_meta.get("pipeline_mode", "single_view_studio"))
    is_synthetic_proxy = (
        pipeline_mode in ("single_view_studio", "real_clip_fixture")
        or str(audit.video_path).startswith(("demo:", "fixture:"))
    )
    is_noiseless_synthetic = bool(base_meta.get("is_noiseless_synthetic", False))
    default_unc_scale = 0.35 if pipeline_mode in ("single_view_gpu_worker", "single_view_local_nlf", "real_clip_fixture") else 1.0
    unc_scale = float(base_meta.get("unc_scale_factor", default_unc_scale))

    # 1. FPS override (3-tier slow-mo resolver pill — clears timebase_unreliable ONLY when clip is not frozen! Bug #8 / P1-1)
    if ov.get("capture_fps_override") is not None:
        new_fps = float(ov["capture_fps_override"])
        can_clear_tb = (audit.held_frame_ratio < 0.08) and (audit.longest_held_run_frames <= 2)
        audit = audit.model_copy(
            update={
                "fps": round(new_fps, 3),
                "duration_s": round(audit.num_frames / max(new_fps, 1.0), 3),
                "fps_grade": "high_speed_240" if new_fps >= 200 else ("slow_mo_120" if new_fps >= 100 else ("medium_60" if new_fps >= 50 else ("standard_30" if new_fps >= 28 else "low_25"))),
                "est_arm_deg_per_frame": round(2300.0 / max(new_fps, 1.0), 2),
                "est_ball_m_per_frame": round(38.89 / max(new_fps, 1.0), 3),
                "timebase_status": "user_fps_override" if (audit.timebase_unreliable and can_clear_tb) else audit.timebase_status,
                "timebase_unreliable": False if can_clear_tb else audit.timebase_unreliable,
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

    # 3. Optional 3-DOF Rotational Pan Tracking (M2B) when enabled + MCC calibrated + REAL textured video available (Bug #4)
    pan_summary: Optional[Dict[str, Any]] = None
    pan_res_verified = None
    camera_poses_per_frame: Optional[List[Dict[str, Any]]] = None
    if ov.get("enable_pan_tracking") and is_synthetic_proxy:
        pan_summary = {
            "is_verified": False,
            "median_residual_px": 0.0,
            "p90_residual_px": 0.0,
            "max_pan_deg": 0.0,
            "sigma_pan_deg": 0.0,
            "verification_reason": (
                "3-DOF background pan tracking requires real video footage with background texture "
                "(not applicable to synthetic stick-figure demo or cached fixture proxies)."
            ),
        }
        j3d_world = transform_cam_to_pitch_world(j3d_cam, cal)
    elif ov.get("enable_pan_tracking") and cal.is_intrinsics_calibrated and video_path and Path(video_path).exists():
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

    # 4. Event spotting + optional human causal-clamped event overrides (passes held_frame_mask on edited/frozen clips! Bug #6 / P1-1)
    is_frozen_clip = (
        audit.timebase_status == "edited_or_frozen"
        or audit.held_frame_ratio >= 0.08
        or audit.longest_held_run_frames >= 3
    )
    held_use = held_mask if (held_mask is not None and len(held_mask) == j3d_world.shape[0] and is_frozen_clip) else None
    events = detect_delivery_events(
        j3d_world,
        fps=audit.fps,
        bowling_arm=arm,
        joint_in_frame=j_in,
        source_frame_indices=audit.source_frame_indices or None,
        held_frame_mask=held_use,
    )
    if ov.get("event_overrides"):
        events = apply_causal_event_overrides(
            events,
            ov["event_overrides"],
            fps=audit.fps,
            n_frames=j3d_world.shape[0],
            source_frame_indices=audit.source_frame_indices or None,
        )

    # 5. Stage 3 kinematics (Bug #1: unc_scale_factor=0.35; Bug #2 & P1-2: skip monocular Rician debiasing on noiseless or dual-fused skeletons)
    kin = extract_fast_bowling_kinematics(
        j3d_world,
        events,
        fps=audit.fps,
        bowling_arm=arm,
        joint_uncertainties_m=j_unc,
        joint_in_frame=j_in,
        interp_gap_frames=interp_gaps,
        camera_R_w2c=None if cov_world_fused is not None else np.asarray(cal.rotation_matrix_R, dtype=np.float64),
        unc_scale_factor=unc_scale,
        debias_near_straight=(not is_noiseless_synthetic) and (cov_world_fused is None),
    )

    # Stage 3 Kinematic panning cross-check (pipeline.py parity! Bug #6)
    if (not cal.is_panning_camera) and (pan_res_verified is None) and (not is_synthetic_proxy):
        pelv_2d_x = j2d[valid_mask, JOINT_IDX["pelv"], 0]
        if len(pelv_2d_x) >= 5:
            x_span_frac = float(np.ptp(pelv_2d_x)) / max(audit.width, 1)
            is_side_ish = cal.view_class in (
                CameraViewClass.SIDE_ON_RIGHT,
                CameraViewClass.SIDE_ON_LEFT,
                CameraViewClass.OPEN_SIDE_45,
            )
            if is_side_ish and x_span_frac < 0.22 and abs(kin.scalars.get("runup_speed_mps", 0.0)) < 1.3:
                cal = cal.model_copy(
                    update={
                        "is_panning_camera": True,
                        "is_metric_calibrated": False,
                        "calibration_notes": list(cal.calibration_notes)
                        + [
                            f"Kinematic pan check: bowler stayed centered (2D X span={x_span_frac:.2f}w, "
                            f"apparent speed={kin.scalars.get('runup_speed_mps', 0.0):.2f} m/s); "
                            "ground-referenced translation suppressed."
                        ],
                    }
                )

    isb_obj = None
    bml87_world_use: Optional[np.ndarray] = None
    isb_summary: Optional[Dict[str, Any]] = None
    if bml87_cam is not None and bml87_cam.shape[0] == j3d_cam.shape[0]:
        # De-rotate bml_movi_87 per-frame when 3-DOF pan tracking is verified (P0-2 #4)
        if pan_res_verified is not None:
            bml87_world = transform_cam_to_pitch_world_per_frame(bml87_cam, pan_res_verified.R_per_frame, pan_res_verified.t_per_frame)
        else:
            bml87_world = transform_cam_to_pitch_world(bml87_cam, cal)
        bml87_world_use = bml87_world
        isb = extract_isb_anatomy_kinematics(
            bml87_world,
            events,
            bowling_arm=arm,
            markers_unc_87=bml87_unc,
            cal=cal,
        )
        isb_obj = isb
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
        metrics = quantify_and_gate_metrics(
            kin, audit, cal, events=events, unc_scale_factor=unc_scale, dual_baseline_sin=dual_baseline_sin
        )

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

    # Wire M2D calibrated side-on wrist-ROI optical ball speed estimator on real video footage (skip stick-figure demos!)
    if video_path and Path(video_path).exists() and (not is_synthetic_proxy) and (not str(audit.video_path).startswith("dual:")):
        wri_i = JOINT_IDX["rwri" if arm == BowlingArm.RIGHT else "lwri"]
        wri_unc_br = float(j_unc[min(events.br.frame_idx, len(j_unc) - 1), wri_i]) * unc_scale
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
    elif (
        str(audit.video_path).startswith("dual:")
        and events.is_complete_delivery
        and base_meta.get("cached_ball_speed_metric")
        and ov.get("capture_fps_override") is None
        and not (isinstance(ov.get("event_overrides"), dict) and ov["event_overrides"].get("br_frame") is not None)
    ):
        from ..schemas import MetricEstimate

        metrics["ball_release_speed_kph"] = MetricEstimate.model_validate(base_meta["cached_ball_speed_metric"])

    use_fused_cov = (
        cov_world_fused is not None
        and cov_world_fused.shape[:2] == (j3d_world.shape[0], 24)
        and (not cal_overridden)
        and pan_res_verified is None
    )
    v5_bundle = extract_and_quantify_v5_biomechanics(
        kin=kin,
        audit=audit,
        cal=cal,
        events=events,
        workload=workload_ctx,
        isb=isb_obj,
        markers_world_87=bml87_world_use,
        unc_scale_factor=unc_scale,
        dual_baseline_sin=dual_baseline_sin,
        cov_world_fused=cov_world_fused if use_fused_cov else None,
    )

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
        v5_bundle=v5_bundle,
    )
    has_real_vid = bool(base_meta.get("has_real_video", False))
    s_kind = "real" if (is_synthetic_proxy and has_real_vid) else ("synthetic" if is_synthetic_proxy else None)
    story = hydrate_coaching_story(
        fs,
        delivery_id=ddir.name,
        is_sample=bool(is_synthetic_proxy),
        sample_kind=s_kind,
    )
    fs = fs.model_copy(update={"coaching_story": story})
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
        bowling_arm=arm,
        com_world_m=v5_bundle.com_world_m,
        sequence_series=v5_bundle.sequence_series,
        wrist_rel_speed_mps=v5_bundle.wrist_rel_speed_mps,
    )

    # Persist updated fact_sheet.json & telemetry.json locally
    (ddir / "fact_sheet.json").write_text(fs.model_dump_json(indent=2), encoding="utf-8")
    (ddir / "telemetry.json").write_text(json.dumps(telemetry), encoding="utf-8")
    (ddir / "coaching_report.md").write_text(report_md, encoding="utf-8")

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "fact_sheet": fs,
        "story": story,
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


def seed_demo_session(
    db: StudioDatabase,
    sessions_root: Path,
    fixtures_dir: Optional[Path] = None,
    public_clips_dir: Optional[Path] = None,
) -> str:
    """Seed Studio v5 Demo Session:
    - Real Dale Steyn rear-on net video (`uiZN5FHWjJk.mp4`) + Real Indoor 120 fps front-on video (`ZpBNq8cIRoY.mp4`)
    - 4-Ball MCC-calibrated 240 fps spell (`Ball 1` fresh -> `Ball 4` late-spell fatigue) + Rear Dual-Sync Mate + ISB `bml_movi_87`
    - 28-day realistic ACWR & governing-body workload history (`108 balls/7d`, Coupled ACWR ~1.01, EWMA ~1.03)
    """
    existing = db.list_sessions()
    demo_ses_dir = sessions_root / "ses_demo"
    marker_file = demo_ses_dir / ".demo_seed_version"
    v5_marker_file = demo_ses_dir / ".demo_v5_version"
    has_demo_ses = any(s.get("id") == "ses_demo" for s in existing)
    any_deliveries = any(int(s.get("delivery_count", 0)) > 0 for s in existing)

    if any_deliveries:
        if not has_demo_ses:
            return existing[0]["id"]
        if (
            marker_file.exists()
            and marker_file.read_text(encoding="utf-8").strip() == _DEMO_SEED_VERSION
            and v5_marker_file.exists()
            and v5_marker_file.read_text(encoding="utf-8").strip() == "5.2"
        ):
            return existing[0]["id"]
        # Upgrade pre-v5 `ses_demo` in place while keeping `.demo_seed_version == "4.0"` backward compatibility!

    if not db.get_bowler("bwl_demo"):
        bowler = db.create_bowler("Demo Pace Bowler (Studio Showcase)", bowling_arm="right", age_band="adult", bowler_id="bwl_demo")
    else:
        bowler = db.get_bowler("bwl_demo") or {"id": "bwl_demo"}

    if not db.get_session("ses_demo"):
        session = db.create_session(
            bowler_id=bowler["id"],
            title="Studio v5 Showcase — 4-Ball 240 fps Spell & Real Net Footage",
            notes="Includes real Dale Steyn & Indoor 120 fps videos plus a 4-ball MCC-calibrated 240 fps spell with ISB 87-marker anatomy.",
            session_id="ses_demo",
        )
    else:
        session = db.get_session("ses_demo") or {"id": "ses_demo"}

    if len(db.list_workload_logs(bowler["id"])) < 6:
        # Clear partial 3-row v4 seed if upgrading so 28-day ACWR series is clean & non-duplicated
        for old_w in db.list_workload_logs(bowler["id"]):
            db.delete_workload_log(bowler["id"], str(old_w["id"]))
        today = datetime.now(timezone.utc).date()
        seeded_logs = [
            (26, 30, "net", "Pre-spell rhythm & crease alignment", 6, 18, 1),
            (24, 42, "match", "Saturday club 7-over spell", 7, 30, 1),
            (21, 36, "net", "Target corridor bowling", 6, 24, 1),
            (19, 30, "net", "Technical front-arm gather session", 6, 18, 1),
            (17, 36, "match", "Midweek T20 + warm-up overs", 7, 24, 2),
            (14, 42, "net", "High-intensity new-ball spell", 8, 30, 2),
            (12, 24, "net", "Short-run front-leg brace drill", 5, 18, 1),
            (10, 42, "match", "League 7-over spell", 8, 30, 2),
            (8, 36, "net", "Yorker & bouncer execution", 7, 24, 1),
            (5, 36, "net", "High-intensity target bowling", 7, 24, 1),
            (3, 42, "match", "Club 7-over spell", 8, 30, 2),
            (1, 30, "net", "Technical front-leg brace & tempo check", 6, 18, 1),
        ]
        for d_ago, n_balls, s_type, note_txt, rpe_v, max_sp, sore_v in seeded_logs:
            db.add_workload_log(
                bowler["id"],
                (today - timedelta(days=d_ago)).isoformat(),
                n_balls,
                s_type,
                note_txt,
                rpe=rpe_v,
                max_spell_balls=max_sp,
                soreness_lumbar=sore_v,
                fatigue_level=2,
            )

    clips_root = (
        Path(public_clips_dir)
        if public_clips_dir is not None
        else (Path(__file__).resolve().parents[3] / "data" / "public_clips")
    )

    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg", "popping_crease_off",
        "popping_crease_leg", "stump_middle_base", "stump_middle_top",
    ]
    def _sstep(t_val: float, t0: float, t1: float) -> float:
        x = float(np.clip((t_val - t0) / max(t1 - t0, 1e-6), 0.0, 1.0))
        return x * x * (3.0 - 2.0 * x)

    def _apply_demo_proximal_to_distal_uncoil(jw_in: np.ndarray, p_in: SyntheticDeliveryParams) -> np.ndarray:
        """Apply realistic post-FFC proximal-to-distal pelvis (+28 ms) -> thorax (+64 ms) rotational uncoiling
        while preserving exact BFC alignment, SCR minimum, knee angle, and shoulder-to-wrist arm kinematics.
        """
        out = jw_in.copy()
        n_f = len(out)
        ffc_t, br_t = float(p_in.ffc_time_s), float(p_in.br_time_s)
        lhip_i, rhip_i = JOINT_IDX["lhip"], JOINT_IDX["rhip"]
        lsho_i, rsho_i = JOINT_IDX["lsho"], JOINT_IDX["rsho"]
        lcla_i, rcla_i = JOINT_IDX["lcla"], JOINT_IDX["rcla"]
        pelv_i, neck_i, spi3_i = JOINT_IDX["pelv"], JOINT_IDX["neck"], JOINT_IDX["spi3"]
        r_arm_chain = [JOINT_IDX["relb"], JOINT_IDX["rwri"], JOINT_IDX["rhan"]]
        l_arm_chain = [JOINT_IDX["lelb"], JOINT_IDX["lwri"], JOINT_IDX["lhan"]]

        for fi in range(n_f):
            t_cur = fi / float(p_in.fps)
            if t_cur <= ffc_t - 0.012:
                continue
            u_orig = _sstep(t_cur, ffc_t, br_t + 0.12)
            # 1) Pelvis: fast hip uncoiling across [FFC - 10 ms, FFC + 66 ms] (peaks at FFC + 28 ms ~580 deg/s, then brakes!)
            # Pivot rear hip `rhip` around the braced lead-hip socket `lhip` so `lhip` & front-knee brace stay 100% invariant!
            u_pelv_fast = _sstep(t_cur, ffc_t - 0.010, ffc_t + 0.066)
            d_pelv_rad = math.radians(42.0 * u_pelv_fast - 35.0 * u_orig)
            if abs(d_pelv_rad) > 1e-6:
                c_p, s_p = math.cos(d_pelv_rad), math.sin(d_pelv_rad)
                lhip_pt = out[fi, lhip_i]
                rel_r = out[fi, rhip_i] - lhip_pt
                out[fi, rhip_i, 0] = lhip_pt[0] + c_p * rel_r[0] + s_p * rel_r[2]
                out[fi, rhip_i, 2] = lhip_pt[2] - s_p * rel_r[0] + c_p * rel_r[2]

            # 2) Thorax: delayed chest whip across [FFC + 16 ms, FFC + 104 ms] (peaks at FFC + 60 ms ~1020 deg/s!)
            u_sh_fast = _sstep(t_cur, ffc_t + 0.016, ffc_t + 0.104)
            d_sh_rad = math.radians(75.0 * (u_sh_fast - u_orig))
            if abs(d_sh_rad) > 1e-6:
                axis = out[fi, neck_i] - out[fi, pelv_i]
                axis = axis / max(float(np.linalg.norm(axis)), 1e-6)
                # Rodrigues rotation around +axis (+Y_world x +Z = +X -> +yaw right-hand rule!)
                k_vec = axis
                c_s, s_s = math.cos(d_sh_rad), math.sin(d_sh_rad)

                def _rot_vec(v_in: np.ndarray) -> np.ndarray:
                    return v_in * c_s + np.cross(k_vec, v_in) * s_s + k_vec * float(np.dot(k_vec, v_in)) * (1.0 - c_s)

                old_rsho = out[fi, rsho_i].copy()
                old_lsho = out[fi, lsho_i].copy()
                neck_pt = out[fi, neck_i]
                spi3_pt = out[fi, spi3_i]
                out[fi, rsho_i] = neck_pt + _rot_vec(old_rsho - neck_pt)
                out[fi, lsho_i] = neck_pt + _rot_vec(old_lsho - neck_pt)
                out[fi, rcla_i] = spi3_pt + _rot_vec(out[fi, rcla_i] - spi3_pt)
                out[fi, lcla_i] = spi3_pt + _rot_vec(out[fi, lcla_i] - spi3_pt)
                # Carry upper/lower arms rigidly on their shoulder sockets so elbow/wrist angles & speeds are 100% preserved
                d_rsho = out[fi, rsho_i] - old_rsho
                d_lsho = out[fi, lsho_i] - old_lsho
                out[fi, r_arm_chain] += d_rsho[None, :]
                out[fi, l_arm_chain] += d_lsho[None, :]
        return out

    # 4-Ball 240 fps calibrated spell (Ball 1 -> Ball 4 progressive fatigue) + Ball 1 Rear Dual-Sync Mate
    demo_specs = [
        ("del_demo_side_240", "Ball 1 — Side-On 240 fps (MCC Calibrated)", 90.0, 5.0, 5.95, 160.0, 171.0, 45.0, 36.0, 18.0, 149.0, 0.00),
        ("del_demo_rear_240", "Ball 1 — Rear 240 fps (Dual-Sync Mate)", 0.0, 6.0, 5.95, 160.0, 171.0, 45.0, 36.0, 18.0, 149.0, 0.00),
        ("del_demo_side_b2", "Ball 2 — Side-On 240 fps (Mild Fatigue)", 90.0, 5.0, 5.85, 160.0, 165.0, 43.0, 40.0, 22.0, 144.0, 0.03),
        ("del_demo_side_b3", "Ball 3 — Side-On 240 fps (Mid-Spell Drift)", 90.0, 5.0, 5.72, 159.0, 157.0, 41.0, 42.0, 25.0, 138.0, 0.06),
        ("del_demo_side_b4", "Ball 4 — Side-On 240 fps (Late-Spell Knee Yield)", 90.0, 5.0, 5.55, 159.0, 143.0, 38.0, 45.0, 28.0, 131.0, 0.09),
    ]
    b1_dual_inputs: Dict[str, Dict[str, Any]] = {}
    for did, lbl, az, el, v_run, fk_ffc, fk_br, tf_br, lf_br, scr, delay_arm, z_shift_m in demo_specs:
        ddir = sessions_root / session["id"] / did
        ddir.mkdir(parents=True, exist_ok=True)
        params = SyntheticDeliveryParams(
            bowler_arm=BowlingArm.RIGHT,
            fps=240.0,
            runup_speed_mps=v_run,
            front_knee_ffc_deg=fk_ffc,
            front_knee_br_deg=fk_br,
            trunk_flexion_br_deg=tf_br,
            thorax_lat_flex_br_deg=lf_br,
            shoulder_counter_rot_deg=scr,
            delayed_arm_ffc_deg=delay_arm,
        )
        jw, _ = generate_synthetic_delivery(params)
        jw = _apply_demo_proximal_to_distal_uncoil(jw, params)
        # Apply realistic over-the-wicket stump corridor offset (+0.58 m + ball-to-ball lateral spread)
        jw[:, :, 2] += 0.58 + z_shift_m
        # Apply clean +14 cm open "train-tracks" front-foot corridor on Ball 1..3, narrowing on fatigued Ball 4
        if did != "del_demo_side_b4":
            jw[:, [JOINT_IDX["lhip"], JOINT_IDX["lkne"], JOINT_IDX["lank"], JOINT_IDX["ltoe"]], 2] -= 0.07

        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=9.8, fov_deg=46.0)
        j2d, j_cam = project_world_points(jw, K, R, t)
        lm_px = project_mcc_landmarks(K, R, t, lm_keys, noise_px=0.5)
        cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)

        # Synthesize camera-space ISB 87 bony markers so ISB Pelvis/Thorax/Foot modules run cleanly
        m87_world = synthesize_bml_movi_87_from_smpl24(jw, BowlingArm.RIGHT, anterior_pelvic_tilt_deg=16.0, front_foot_splay_deg=11.0)
        m87_cam = np.einsum("ij,tkj->tki", R, m87_world) + t[None, None, :]

        n = jw.shape[0]
        j_in_arr = np.ones((n, 24), dtype=np.bool_)
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
            joint_in_frame=j_in_arr,
            valid_mask=np.ones(n, dtype=np.bool_),
            bml_movi_87_cam_m=m87_cam.astype(np.float32),
            bml_movi_87_unc_m=np.full((n, 87), 0.022, dtype=np.float32),
        )
        (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps(
                {
                    "bowling_arm": "right",
                    "pipeline_mode": "single_view_studio",
                    "nlf_fx_used": float(K[0, 0]),
                    "unc_scale_factor": 1.0,
                    "is_noiseless_synthetic": True,
                    "has_real_video": False,
                }
            ),
            encoding="utf-8",
        )
        res = recompute_delivery_cpu(ddir, bowler_id=lbl, proxy_fps=30.0)
        proxy_mp4 = render_synthetic_proxy_video(
            j2d,
            1280,
            720,
            ddir / "scrub_proxy.mp4",
            proxy_fps=30.0,
            title=lbl,
            mcc_wireframe_2d=project_mcc_wireframe_2d(cal),
            joint_in_frame=j_in_arr,
            bowling_arm="right",
            br_frame_idx=int(res["fact_sheet"].events.br.frame_idx),
        )
        frames_cache = ddir / "frames"
        if frames_cache.exists():
            for fp_old in frames_cache.glob("*"):
                fp_old.unlink(missing_ok=True)
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
            isb_summary=res["isb_summary"],
            overrides={"mcc_landmarks_px": {k: [round(v[0], 1), round(v[1], 1)] for k, v in lm_px.items()}},
        )
        if did in ("del_demo_side_240", "del_demo_rear_240"):
            b1_dual_inputs[did] = {
                "jw": transform_cam_to_pitch_world(j_cam, cal),
                "cal": cal,
                "audit": audit,
                "unc": np.full((n, 24), 0.025, dtype=np.float64),
                "in_frame": j_in_arr,
                "K": K,
                "R": R,
                "t": t,
            }

        # Seed Dual-Phone 240 fps Showcase (`del_demo_dual_240`) immediately after Ball 1 mates so it precedes Ball 2..4!
        if did == "del_demo_rear_240" and "del_demo_side_240" in b1_dual_inputs:
            in_a = b1_dual_inputs["del_demo_side_240"]
            in_b = b1_dual_inputs["del_demo_rear_240"]
            dual_did = "del_demo_dual_240"
            dual_lbl = "Spell Ball 1 — Dual-Phone 240 fps (Full 3D Sequence)"
            ddir_dual = sessions_root / session["id"] / dual_did
            ddir_dual.mkdir(parents=True, exist_ok=True)
            fused_demo = fuse_dual_view_3d_trajectories(
                bowler_id=dual_lbl,
                bowling_arm=BowlingArm.RIGHT,
                joints_world_a=in_a["jw"],
                cal_a=in_a["cal"],
                audit_a=in_a["audit"],
                joints_world_b=in_b["jw"],
                cal_b=in_b["cal"],
                audit_b=in_b["audit"],
                unc_a=in_a["unc"],
                unc_b=in_b["unc"],
                in_frame_a=in_a["in_frame"],
                in_frame_b=in_b["in_frame"],
                unc_scale_a=1.0,
                unc_scale_b=1.0,
            )
            Ka_d, Ra_d, ta_d = in_a["K"], in_a["R"], in_a["t"]
            jw_f = fused_demo.joints_world_fused
            n_f_d = len(jw_f)
            j_cam_f = np.einsum("ij,tkj->tki", Ra_d, jw_f) + ta_d[None, None, :]
            z_safe_f = np.maximum(j_cam_f[:, :, 2:3], 0.2)
            j2d_f = np.einsum("ij,tkj->tki", Ka_d, j_cam_f)[:, :, :2] / z_safe_f
            m87_w_f = synthesize_bml_movi_87_from_smpl24(
                jw_f, BowlingArm.RIGHT, anterior_pelvic_tilt_deg=16.0, front_foot_splay_deg=11.0
            )
            m87_cam_f = np.einsum("ij,tkj->tki", Ra_d, m87_w_f) + ta_d[None, None, :]

            np.savez_compressed(
                ddir_dual / "raw_nlf.npz",
                joints3d_cam_m=j_cam_f.astype(np.float32),
                joints2d_px=j2d_f.astype(np.float32),
                joint_uncertainties_m=fused_demo.joint_uncertainties_fused.astype(np.float32),
                joint_in_frame=fused_demo.joint_in_frame_fused.astype(np.bool_),
                valid_mask=np.ones(n_f_d, dtype=np.bool_),
                cov_world_fused=fused_demo.cov_world_fused.astype(np.float32),
                dual_baseline_sin=np.array(float(fused_demo.baseline_sin), dtype=np.float64),
                bml_movi_87_cam_m=m87_cam_f.astype(np.float32),
                bml_movi_87_unc_m=np.full((n_f_d, 87), 0.016, dtype=np.float32),
            )
            (ddir_dual / "audit.json").write_text(fused_demo.fact_sheet.audit.model_dump_json(indent=2), encoding="utf-8")
            (ddir_dual / "base_calibration.json").write_text(
                fused_demo.fact_sheet.calibration.model_dump_json(indent=2), encoding="utf-8"
            )
            (ddir_dual / "meta.json").write_text(
                json.dumps(
                    {
                        "bowling_arm": "right",
                        "pipeline_mode": "dual_3d_ray_fusion",
                        "dual_baseline_sin": float(fused_demo.baseline_sin),
                        "nlf_fx_used": float(Ka_d[0, 0]),
                        "unc_scale_factor": 1.0,
                        "is_noiseless_synthetic": True,
                        "has_real_video": False,
                        "source_delivery_ids": ["del_demo_side_240", "del_demo_rear_240"],
                    }
                ),
                encoding="utf-8",
            )
            res_dual = recompute_delivery_cpu(ddir_dual, bowler_id=dual_lbl, proxy_fps=30.0)
            proxy_dual = render_synthetic_proxy_video(
                j2d_f,
                1280,
                720,
                ddir_dual / "scrub_proxy.mp4",
                proxy_fps=30.0,
                title=dual_lbl,
                mcc_wireframe_2d=project_mcc_wireframe_2d(fused_demo.fact_sheet.calibration),
                joint_in_frame=fused_demo.joint_in_frame_fused,
                bowling_arm="right",
                br_frame_idx=int(res_dual["fact_sheet"].events.br.frame_idx),
            )
            frames_cache_d = ddir_dual / "frames"
            if frames_cache_d.exists():
                for fp_old in frames_cache_d.glob("*"):
                    fp_old.unlink(missing_ok=True)
            db.upsert_delivery(
                delivery_id=dual_did,
                session_id=session["id"],
                label=dual_lbl,
                data_dir=ddir_dual,
                video_path=str(proxy_dual),
                proxy_path=str(proxy_dual),
                pipeline_mode="dual_3d_ray_fusion",
                start_frame=0,
                end_frame=n_f_d,
                status="ready",
                fact_sheet_json=res_dual["fact_sheet"].model_dump_json(),
                isb_summary=res_dual["isb_summary"],
            )

    # Seed real-clip fixtures (`steyn`, `indoor`, and `mcgrath`) — binding REAL `uiZN5FHWjJk.mp4` AND REAL `ZpBNq8cIRoY.mp4`!
    if fixtures_dir and fixtures_dir.exists():
        for fname, did, lbl in [
            ("steyn_nlf_fixture.npz", "del_real_steyn", "Dale Steyn — Real Net Video"),
            ("indoor_nlf_fixture.npz", "del_real_indoor", "Indoor Net 120fps — Real Front-On Video"),
            ("mcgrath_nlf_fixture.npz", "del_real_mcgrath", "Glenn McGrath — Incomplete Clip Example"),
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
            z_safe = np.maximum(j_cam[:, :, 2:3], 0.2)
            j2d = np.einsum("ij,tkj->tki", K, j_cam)[:, :, :2] / z_safe
            n = j_cam.shape[0]
            fps = float(meta["fps"])
            j_in_fix = z["joint_in_frame"].astype(np.bool_)

            real_mp4: Optional[Path] = None
            use_real_mp4 = False
            src_indices = list(range(n))
            proxy_fps_use = 25.0

            if did == "del_real_steyn":
                cand_mp4 = clips_root / "uiZN5FHWjJk.mp4"
                if cand_mp4.exists() and len(_STEYN_SOURCE_FRAME_INDICES) == n:
                    real_mp4 = cand_mp4
                    use_real_mp4 = True
                    src_indices = list(_STEYN_SOURCE_FRAME_INDICES)
                    proxy_fps_use = 25.0
            elif did == "del_real_indoor":
                cand_mp4 = clips_root / "ZpBNq8cIRoY.mp4"
                if cand_mp4.exists() and n == 264:
                    real_mp4 = cand_mp4
                    use_real_mp4 = True
                    src_indices = list(range(n))
                    proxy_fps_use = 30.0

            audit = VideoAudit(
                video_path=f"fixture:{fname}",
                width=int(meta["width"]),
                height=int(meta["height"]),
                fps=fps,
                container_fps=float(meta["container_fps"]),
                num_frames=n,
                raw_num_frames=75 if (did == "del_real_steyn" and use_real_mp4) else n,
                duration_s=round(n / max(fps, 1.0), 3),
                median_laplacian_var=433.7 if "steyn" in fname else (184.0 if "indoor" in fname else 209.2),
                blur_grade="crisp",
                blur_roi_method="bowler_roi",
                fps_grade="slow_mo_120" if fps >= 100.0 else "low_25",
                est_arm_deg_per_frame=round(2300.0 / max(fps, 1.0), 2),
                est_ball_m_per_frame=round(38.89 / max(fps, 1.0), 3),
                timebase_status=str(meta["timebase_status"]),
                timebase_unreliable=bool(meta["timebase_unreliable"]),
                held_frame_ratio=float(meta["held_frame_ratio"]),
                longest_held_run_frames=int(meta["longest_held_run_frames"]),
                source_frame_indices=src_indices,
            )
            np.savez_compressed(
                ddir / "raw_nlf.npz",
                joints3d_cam_m=j_cam.astype(np.float32),
                joints2d_px=j2d.astype(np.float32),
                joint_uncertainties_m=z["joint_uncertainties_m"].astype(np.float32),
                joint_in_frame=j_in_fix,
                valid_mask=z["valid_mask"].astype(np.bool_),
            )
            (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
            (ddir / "base_calibration.json").write_text(cal.model_dump_json(indent=2), encoding="utf-8")
            (ddir / "meta.json").write_text(
                json.dumps(
                    {
                        "bowling_arm": meta["bowling_arm"],
                        "pipeline_mode": "real_clip_fixture",
                        "nlf_fx_used": float(K[0, 0]),
                        "unc_scale_factor": 0.35,
                        "is_noiseless_synthetic": False,
                        "has_real_video": use_real_mp4,
                    }
                ),
                encoding="utf-8",
            )
            res = recompute_delivery_cpu(ddir, bowler_id=meta["bowler_id"], proxy_fps=proxy_fps_use)
            if use_real_mp4 and real_mp4 is not None:
                try:
                    proxy_mp4 = build_all_intra_scrub_proxy(
                        real_mp4,
                        ddir / "scrub_proxy.mp4",
                        source_frame_indices=src_indices,
                        proxy_fps=proxy_fps_use,
                    )
                except Exception:
                    use_real_mp4 = False
                    proxy_mp4 = render_synthetic_proxy_video(
                        j2d,
                        int(meta["width"]),
                        int(meta["height"]),
                        ddir / "scrub_proxy.mp4",
                        proxy_fps=proxy_fps_use,
                        title=lbl,
                        joint_in_frame=j_in_fix,
                        bowling_arm=str(meta["bowling_arm"]),
                        br_frame_idx=int(res["fact_sheet"].events.br.frame_idx) if res["fact_sheet"].events.is_complete_delivery else None,
                    )
            else:
                proxy_mp4 = render_synthetic_proxy_video(
                    j2d,
                    int(meta["width"]),
                    int(meta["height"]),
                    ddir / "scrub_proxy.mp4",
                    proxy_fps=proxy_fps_use,
                    title=lbl,
                    joint_in_frame=j_in_fix,
                    bowling_arm=str(meta["bowling_arm"]),
                    br_frame_idx=int(res["fact_sheet"].events.br.frame_idx) if res["fact_sheet"].events.is_complete_delivery else None,
                )
            frames_cache = ddir / "frames"
            if frames_cache.exists():
                for fp_old in frames_cache.glob("*"):
                    fp_old.unlink(missing_ok=True)
            db.upsert_delivery(
                delivery_id=did,
                session_id=session["id"],
                label=lbl,
                data_dir=ddir,
                video_path=str(real_mp4) if (use_real_mp4 and real_mp4 is not None) else str(proxy_mp4),
                proxy_path=str(proxy_mp4),
                pipeline_mode="real_clip_fixture",
                start_frame=0,
                end_frame=n,
                status="ready",
                fact_sheet_json=res["fact_sheet"].model_dump_json(),
            )

    demo_ses_dir.mkdir(parents=True, exist_ok=True)
    marker_file.write_text(_DEMO_SEED_VERSION, encoding="utf-8")
    v5_marker_file.write_text("5.2", encoding="utf-8")
    return session["id"]
