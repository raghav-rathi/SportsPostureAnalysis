"""End-to-End Fast-Bowling Biomechanics Pipeline & Multi-Angle Fusion Orchestrator.

Fixes EXP-007:
  - B2: Automatic Stage 0 timebase audit + regular pulldown deduplication (`source_frame_indices`)
        threaded cleanly through NLF extraction and Stage 6 visualizers + bowler-ROI blur measurement.
  - B3 / B16: Runs MCC Pitch PnP or Two-Wicket Geometry calibration BEFORE NLF so calibrated
              `intrinsic_matrix` and `world_up_vector` feed directly into `detect_smpl_batched`.
  - B4: Never discards `is_panning_camera` or skips kinematic pan checks on landmark calibration paths.
  - B12 / Section 7.6: Supports empirical GPU Repeatability Intervals (`base`, `flip`, `aug3` + MC) on real runs.
  - B19: Validates delivery-timing compatibility, applies `_clamp_ci`, and checks azimuth baseline in dual-view fusion.
  - B23 / B26: Uses `constants.JOINT_IDX` and records pinned NLF checkpoint SHA-256 hash.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .constants import JOINT_IDX
from .schemas import (
    BiomechanicalFactSheet,
    BowlerActionClass,
    BowlingArm,
    CameraViewClass,
    CoachingReport,
    IntervalType,
    MetricEstimate,
    ObservabilityGrade,
    PitchCalibration,
    WorkloadContext,
)
from .stage0_audit import (
    analyze_frame_cadence,
    audit_video,
    detect_camera_motion,
    measure_bowler_roi_blur,
)
from .stage1_pitch_calibration import (
    auto_detect_two_wickets,
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    solve_two_wicket_calibration,
    transform_cam_to_pitch_world,
    world_up_from_calibration,
)
from .stage2_event_spotting import detect_delivery_events
from .stage2_nlf_3d import (
    TrackedBowler3D,
    extract_bowler_3d_nlf,
    extract_bowler_3d_nlf_ensemble,
)
from .stage3_kinematics import classify_action_portus, extract_fast_bowling_kinematics
from .stage4_uncertainty import _clamp_ci, compute_repeatability_metrics, quantify_and_gate_metrics
from .stage5_evaluator import build_fact_sheet
from .stage6_coach_llm import generate_grounded_report, render_report_markdown
from .stage6_visualizer import (
    render_annotated_freeze_video,
    render_biomechanics_dashboard,
    render_keyframe_panel,
)


def analyze_delivery_video(
    video_path: str | Path,
    bowler_id: str,
    nlf_model_path: str | Path,
    output_dir: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    bowling_arm: Optional[BowlingArm] = None,
    mcc_landmarks_px: Optional[Dict[str, Tuple[float, float]]] = None,
    two_wickets_px: Optional[Dict[str, float]] = None,
    auto_two_wickets: bool = True,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    workload: Optional[WorkloadContext] = None,
    capture_fps_override: Optional[float] = None,
    num_aug: int = 1,
    compute_repeatability: bool = True,
    device: str = "cuda",
) -> Tuple[BiomechanicalFactSheet, CoachingReport, Dict[str, Path]]:
    """Run the full 7-stage fast-bowling biomechanics pipeline on a single video clip."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0A: Timebase Integrity, Pulldown Deduplication, Freeze-Frame & Cut Audit (EXP-007 B2)
    audit = audit_video(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        capture_fps_override=capture_fps_override,
        dedup_regular_pulldown=True,
    )
    src_indices = audit.source_frame_indices

    # Stage 0B: Pre-NLF 4-DOF Camera Motion Check (pan, tilt, zoom — EXP-007 B14)
    is_panning_pre, pan_frac_pre, motion_metrics_pre = detect_camera_motion(
        video_path=video_path,
        source_frame_indices=src_indices,
        fps_hint=audit.fps,
    )

    # Stage 1A: Pre-NLF Pitch Calibration (MCC PnP or Two-Wicket Geometry -> calibrated K & world_up BEFORE NLF!)
    cal_pre: Optional[PitchCalibration] = None
    if mcc_landmarks_px is not None and len(mcc_landmarks_px) >= 4:
        cal_pre = calibrate_from_mcc_landmarks(
            image_landmarks_px=mcc_landmarks_px,
            width=audit.width,
            height=audit.height,
            is_panning_camera=is_panning_pre,
        )
    elif two_wickets_px is not None:
        cal_pre = solve_two_wicket_calibration(
            h_near_px=float(two_wickets_px["h_near_px"]),
            base_near_y_px=float(two_wickets_px["base_near_y_px"]),
            h_far_px=float(two_wickets_px["h_far_px"]),
            base_far_y_px=float(two_wickets_px["base_far_y_px"]),
            width=audit.width,
            height=audit.height,
            roll_deg=float(two_wickets_px.get("roll_deg", 0.0)),
            near_x_px=two_wickets_px.get("near_x_px"),
            far_x_px=two_wickets_px.get("far_x_px"),
            is_panning_camera=is_panning_pre,
            from_front=bool(two_wickets_px.get("from_front", False)),
        )
    elif auto_two_wickets:
        cal_pre = auto_detect_two_wickets(
            video_path=video_path,
            source_frame_indices=src_indices,
            is_panning_camera=is_panning_pre,
        )

    K_nlf = np.asarray(cal_pre.camera_matrix_K, dtype=np.float64) if cal_pre is not None else None
    up_nlf = world_up_from_calibration(cal_pre) if cal_pre is not None else None

    # Stage 2A: Streaming 3D Pose Extraction (with calibrated K & world_up if available, + repeatability ensemble)
    ensemble_tracked: List[TrackedBowler3D] = []
    if compute_repeatability and num_aug == 1:
        tracked, ensemble_tracked = extract_bowler_3d_nlf_ensemble(
            video_path=video_path,
            model_path=nlf_model_path,
            source_frame_indices=src_indices,
            intrinsic_matrix=K_nlf,
            world_up_vector=up_nlf,
            bowling_arm_override=bowling_arm,
            anchor_xy_norm=anchor_xy_norm,
            device=device,
        )
    else:
        tracked = extract_bowler_3d_nlf(
            video_path=video_path,
            model_path=nlf_model_path,
            source_frame_indices=src_indices,
            num_aug=num_aug,
            intrinsic_matrix=K_nlf,
            world_up_vector=up_nlf,
            bowling_arm_override=bowling_arm,
            anchor_xy_norm=anchor_xy_norm,
            device=device,
        )
        ensemble_tracked = [tracked]

    # Stage 0C: Refine Blur Strictly Inside Bowler ROI (eliminating Reel caption sharpness inversion!)
    # Refresh blur warnings so bowler_roi crisp/severe_blur status never contradicts pre-ROI warnings (P1-3).
    roi_lap, roi_blur = measure_bowler_roi_blur(video_path, src_indices, tracked.bowler_boxes_xyxy)
    if roi_lap > 0.0:
        refreshed_warnings = [
            w for w in audit.warnings
            if "Low interior sharpness" not in w and "motion blur inside bowler ROI" not in w
        ]
        if roi_blur == "severe_blur":
            refreshed_warnings.append(
                f"Severe motion blur inside bowler ROI (Laplacian var={roi_lap:.1f}); "
                "3D landmark uncertainty inflated and low-SNR metrics gated."
            )
        elif roi_blur == "moderate_blur":
            refreshed_warnings.append(
                f"Moderate motion blur inside bowler ROI (Laplacian var={roi_lap:.1f}); "
                "3D pose intervals widened for shutter-speed blur."
            )
        audit = audit.model_copy(
            update={
                "median_laplacian_var": roi_lap,
                "blur_grade": roi_blur,
                "blur_roi_method": "bowler_roi",
                "warnings": refreshed_warnings,
            }
        )

    # Bowler-masked 4-DOF camera motion check (pan + tilt + zoom)
    is_panning, pan_frac, motion_metrics = detect_camera_motion(
        video_path=video_path,
        bowler_boxes_xyxy=tracked.bowler_boxes_xyxy,
        source_frame_indices=src_indices,
        fps_hint=audit.fps,
    )

    # Stage 1B: Finalize Calibration (NEVER discard is_panning_camera on landmark/two-wicket paths! EXP-007 B4)
    if cal_pre is not None:
        cal = cal_pre.model_copy(
            update={
                "is_panning_camera": bool(is_panning),
                "is_metric_calibrated": bool(cal_pre.is_intrinsics_calibrated and (not is_panning)),
                "camera_motion_metrics": motion_metrics,
            }
        )
    else:
        pelv_cam = tracked.joints3d_cam_m[:, JOINT_IDX["pelv"]]
        neck_cam = tracked.joints3d_cam_m[:, JOINT_IDX["neck"]]
        up_cam = neck_cam - pelv_cam
        cal = calibrate_from_3d_motion_and_gravity(
            pelvis_traj_cam_m=pelv_cam,
            up_vectors_cam=up_cam,
            width=audit.width,
            height=audit.height,
            joints3d_cam_m=tracked.joints3d_cam_m,
            joint_in_frame=tracked.joint_in_frame,
            joint_uncertainties_m=tracked.joint_uncertainties_m,
            valid_mask=tracked.valid_mask,
            bowling_arm_is_right=(tracked.detected_bowling_arm == BowlingArm.RIGHT),
            is_panning_camera=is_panning,
            background_upright_vectors=tracked.background_upright_vectors,
            K_override=K_nlf,
        ).model_copy(update={"camera_motion_metrics": motion_metrics})

    joints_world = transform_cam_to_pitch_world(tracked.joints3d_cam_m, cal)

    # Pass per-frame held mask when clip was NOT deduplicated (edited/frozen Reels) so freeze frames never trigger BR
    held_mask_events: Optional[np.ndarray] = None
    if audit.timebase_status == "edited_or_frozen":
        cad_box = analyze_frame_cadence(video_path, start_frame=start_frame, end_frame=end_frame)
        hm = cad_box["held_mask"]
        if len(hm) == len(joints_world):
            held_mask_events = hm

    # Stage 2B: Strict Release Sanity Gates & Causal Event Spotting (EXP-007 B1, B7)
    events = detect_delivery_events(
        joints_world=joints_world,
        fps=audit.fps,
        bowling_arm=tracked.detected_bowling_arm,
        joint_in_frame=tracked.joint_in_frame,
        source_frame_indices=src_indices,
        held_frame_mask=held_mask_events,
    )

    # Stage 3: 3D Fast-Bowling Kinematics with Debiasing & Local Ground Reference
    kin = extract_fast_bowling_kinematics(
        joints_world=joints_world,
        events=events,
        fps=audit.fps,
        bowling_arm=tracked.detected_bowling_arm,
        joint_uncertainties_m=tracked.joint_uncertainties_m,
        joint_in_frame=tracked.joint_in_frame,
        interp_gap_frames=tracked.interp_gap_frames,
        camera_R_w2c=np.asarray(cal.rotation_matrix_R, dtype=np.float64),
        unc_scale_factor=0.35,  # calibrated scaling for raw NLF self-gating uncertainties
    )

    # Kinematic panning cross-check (runs even on single-frame landmark calibration! EXP-007 B4)
    if not cal.is_panning_camera:
        pelv_2d_x = tracked.joints2d_px[tracked.valid_mask, JOINT_IDX["pelv"], 0]
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
                        "calibration_notes": cal.calibration_notes
                        + [
                            f"Kinematic pan check: bowler stayed centered (2D X span={x_span_frac:.2f}w, "
                            f"apparent speed={kin.scalars.get('runup_speed_mps', 0.0):.2f} m/s); "
                            "ground-referenced translation suppressed."
                        ],
                    }
                )

    # Stage 4: Empirical GPU Repeatability Intervals (when ensemble available & delivery complete) or Analytical Jacobian
    interval_type: IntervalType = "analytical_uncertainty"
    if len(ensemble_tracked) >= 2 and events.is_complete_delivery:
        ens_worlds: List[np.ndarray] = [joints_world]
        for tr_v in ensemble_tracked[1:]:
            if cal_pre is not None:
                ens_worlds.append(transform_cam_to_pitch_world(tr_v.joints3d_cam_m, cal))
            else:
                p_v = tr_v.joints3d_cam_m[:, JOINT_IDX["pelv"]]
                u_v = tr_v.joints3d_cam_m[:, JOINT_IDX["neck"]] - p_v
                cal_v = calibrate_from_3d_motion_and_gravity(
                    pelvis_traj_cam_m=p_v,
                    up_vectors_cam=u_v,
                    width=audit.width,
                    height=audit.height,
                    joints3d_cam_m=tr_v.joints3d_cam_m,
                    joint_in_frame=tr_v.joint_in_frame,
                    joint_uncertainties_m=tr_v.joint_uncertainties_m,
                    valid_mask=tr_v.valid_mask,
                    bowling_arm_is_right=(tracked.detected_bowling_arm == BowlingArm.RIGHT),
                    is_panning_camera=cal.is_panning_camera,
                    background_upright_vectors=tr_v.background_upright_vectors,
                    K_override=K_nlf,
                )
                ens_worlds.append(transform_cam_to_pitch_world(tr_v.joints3d_cam_m, cal_v))
        metrics = compute_repeatability_metrics(
            base_kin=kin,
            ensemble_world_trajs=ens_worlds,
            audit=audit,
            cal=cal,
            events=events,
            n_mc=64,
        )
        interval_type = "repeatability_interval"
    else:
        metrics = quantify_and_gate_metrics(
            kin=kin,
            audit=audit,
            cal=cal,
            events=events,
            unc_scale_factor=0.35,
            interval_type="analytical_uncertainty",
        )

    # Stage 5: Definition-Matched Cohort Evaluator & Interval-Cleared Coaching Cues
    fact_sheet = build_fact_sheet(
        bowler_id=bowler_id,
        bowling_arm=tracked.detected_bowling_arm,
        action_class=kin.action_class,
        pose_backend=tracked.backend_name,
        audit=audit,
        cal=cal,
        events=events,
        metrics=metrics,
        workload=workload,
        interval_type=interval_type,
        metadata={
            "start_frame": start_frame,
            "end_frame": end_frame,
            "nlf_model_sha256": tracked.model_sha256,
            "nlf_used_calibrated_intrinsics": tracked.used_calibrated_intrinsics,
            "repeatability_ensemble_size": len(ensemble_tracked),
            "num_tracks_seen": tracked.num_tracks_seen,
            "valid_detection_ratio": round(float(np.mean(tracked.valid_mask)), 3),
            "estimated_stature_m": round(kin.estimated_stature_m, 3),
            "background_pan_fraction": pan_frac,
            "camera_motion_metrics": motion_metrics,
            "standing_bg_gravity_frames": len(tracked.background_upright_vectors),
        },
    )

    # Stage 6A & 6B: Grounded Report & Zero-Drift Visuals
    report = generate_grounded_report(fact_sheet)
    md_text = render_report_markdown(fact_sheet, report)

    fact_path = out_dir / "fact_sheet.json"
    report_path = out_dir / "coaching_report.md"
    traj_path = out_dir / "trajectory_3d.npz"
    keyframe_path = out_dir / "keyframes_annotated.jpg"
    dashboard_path = out_dir / "biomechanics_dashboard.png"
    annotated_video_path = out_dir / "annotated_freeze_analysis.mp4"

    fact_path.write_text(fact_sheet.model_dump_json(indent=2), encoding="utf-8")
    report_path.write_text(md_text, encoding="utf-8")

    npz_payload = {
        "joints3d_cam_m": tracked.joints3d_cam_m,
        "joints3d_world_m": joints_world,
        "joints2d_px": tracked.joints2d_px,
        "joint_uncertainties_m": tracked.joint_uncertainties_m,
        "joint_in_frame": tracked.joint_in_frame,
        "valid_mask": tracked.valid_mask,
        "interp_gap_frames": tracked.interp_gap_frames,
        "source_frame_indices": np.asarray(src_indices, dtype=np.int32),
    }
    for s_key, s_arr in kin.series.items():
        npz_payload[f"series_{s_key}"] = np.asarray(s_arr, dtype=np.float32)
    np.savez_compressed(traj_path, **npz_payload)

    render_keyframe_panel(
        video_path=video_path,
        start_frame=start_frame,
        joints2d_px=tracked.joints2d_px,
        fact_sheet=fact_sheet,
        out_path=keyframe_path,
        source_frame_indices=src_indices,
    )
    render_biomechanics_dashboard(
        joints_world=joints_world,
        kin=kin,
        fact_sheet=fact_sheet,
        out_path=dashboard_path,
    )
    render_annotated_freeze_video(
        video_path=video_path,
        start_frame=start_frame,
        joints2d_px=tracked.joints2d_px,
        fact_sheet=fact_sheet,
        out_path=annotated_video_path,
        source_frame_indices=src_indices,
    )

    artifacts = {
        "fact_sheet": fact_path,
        "report_md": report_path,
        "trajectory_npz": traj_path,
        "keyframes_jpg": keyframe_path,
        "dashboard_png": dashboard_path,
        "annotated_video_mp4": annotated_video_path,
    }
    return fact_sheet, report, artifacts


def fuse_dual_view_fact_sheets(
    side_fs: BiomechanicalFactSheet,
    rear_fs: BiomechanicalFactSheet,
    bowler_id: Optional[str] = None,
) -> BiomechanicalFactSheet:
    """Select/fuse complementary metrics from two views with delivery-timing check & physical CI clamping (B19)."""
    if side_fs.bowling_arm != rear_fs.bowling_arm:
        raise ValueError(
            f"Cannot fuse clips with mismatched bowling arms: side={side_fs.bowling_arm.value} "
            f"vs rear={rear_fs.bowling_arm.value}."
        )

    # Check camera baseline separation (prevents collinear views masquerading as orthogonal dual-view!)
    az_diff_rad = math.radians(side_fs.calibration.azimuth_deg - rear_fs.calibration.azimuth_deg)
    baseline_sin = abs(math.sin(az_diff_rad))
    fusion_notes = [
        f"Metric-level dual-view selection/fusion (camera azimuth separation={abs(side_fs.calibration.azimuth_deg - rear_fs.calibration.azimuth_deg):.1f} deg, |sin(d_az)|={baseline_sin:.2f})."
    ]
    if baseline_sin < 0.35:
        fusion_notes.append(
            "Warning: camera views are nearly collinear (|sin(d_az)| < 0.35); complementary plane observability is limited."
        )

    # Timing compatibility check (B19: warn when clips appear to be from different deliveries/timing scales)
    if side_fs.events.is_complete_delivery and rear_fs.events.is_complete_delivery:
        d_stride = abs(side_fs.events.delivery_stride_duration_ms - rear_fs.events.delivery_stride_duration_ms)
        d_ffc_br = abs(side_fs.events.ffc_to_br_duration_ms - rear_fs.events.ffc_to_br_duration_ms)
        if d_stride > 80.0 or d_ffc_br > 55.0:
            fusion_notes.append(
                f"Warning: timing mismatch between views (|d_stride|={d_stride:.0f} ms, |d_ffc_br|={d_ffc_br:.0f} ms); "
                "clips may not be from the exact same delivery or have mismatched FPS."
            )

    obs_rank = {
        ObservabilityGrade.HIGH: 4,
        ObservabilityGrade.MODERATE: 3,
        ObservabilityGrade.LOW: 2,
        ObservabilityGrade.UNMEASURABLE: 1,
        ObservabilityGrade.REFUSED: 0,
    }
    fused_metrics: Dict[str, MetricEstimate] = {}
    all_keys = list(dict.fromkeys(list(side_fs.metrics.keys()) + list(rear_fs.metrics.keys())))
    for k in all_keys:
        m1 = side_fs.metrics.get(k)
        m2 = rear_fs.metrics.get(k)
        if m1 is None:
            fused_metrics[k] = m2  # type: ignore[assignment]
            continue
        if m2 is None:
            fused_metrics[k] = m1
            continue
        if m1.observability == ObservabilityGrade.REFUSED or m2.observability == ObservabilityGrade.REFUSED:
            fused_metrics[k] = m1
            continue
        if not m1.suppressed and m2.suppressed:
            fused_metrics[k] = m1
        elif m1.suppressed and not m2.suppressed:
            fused_metrics[k] = m2
        elif not m1.suppressed and not m2.suppressed:
            if obs_rank[m1.observability] > obs_rank[m2.observability]:
                fused_metrics[k] = m1
            elif obs_rank[m2.observability] > obs_rank[m1.observability]:
                fused_metrics[k] = m2
            else:
                s1 = max(m1.sigma_total or 5.0, 0.1)
                s2 = max(m2.sigma_total or 5.0, 0.1)
                w1, w2 = 1.0 / (s1 * s1), 1.0 / (s2 * s2)
                v_fused = (w1 * m1.value + w2 * m2.value) / (w1 + w2)  # type: ignore[operator]
                s_fused = float(np.sqrt(1.0 / (w1 + w2)))
                half = 1.96 * s_fused
                ci_lo, ci_hi = _clamp_ci(k, float(v_fused - half), float(v_fused + half))
                fused_metrics[k] = m1.model_copy(
                    update={
                        "value": round(float(v_fused), 2),
                        "ci_low": round(ci_lo, 2),
                        "ci_high": round(ci_hi, 2),
                        "sigma_total": round(s_fused, 2),
                    }
                )
        else:
            fused_metrics[k] = m1

    fused_cal = side_fs.calibration.model_copy(
        update={
            "view_class": CameraViewClass.DUAL_VIEW_FUSED,
            "is_metric_calibrated": side_fs.calibration.is_metric_calibrated or rear_fs.calibration.is_metric_calibrated,
            "is_intrinsics_calibrated": side_fs.calibration.is_intrinsics_calibrated or rear_fs.calibration.is_intrinsics_calibrated,
            "is_panning_camera": side_fs.calibration.is_panning_camera and rear_fs.calibration.is_panning_camera,
            "calibration_notes": side_fs.calibration.calibration_notes + rear_fs.calibration.calibration_notes + fusion_notes,
        }
    )

    sho_m = fused_metrics.get("shoulder_alignment_bfc_deg")
    pelv_m = fused_metrics.get("pelvis_alignment_bfc_deg")
    scr_m = fused_metrics.get("shoulder_counter_rotation_deg")
    if sho_m and not sho_m.suppressed and sho_m.value is not None and scr_m and not scr_m.suppressed and scr_m.value is not None:
        p_val = pelv_m.value if (pelv_m and not pelv_m.suppressed and pelv_m.value is not None) else sho_m.value
        fused_action = classify_action_portus(sho_m.value, p_val, scr_m.value)
    else:
        fused_action = BowlerActionClass.UNKNOWN

    fused_audit = side_fs.audit if side_fs.audit.fps >= rear_fs.audit.fps else rear_fs.audit
    fused_events = side_fs.events if side_fs.events.is_complete_delivery else rear_fs.events

    return build_fact_sheet(
        bowler_id=bowler_id or side_fs.bowler_id,
        bowling_arm=side_fs.bowling_arm,
        action_class=fused_action,
        pose_backend=f"{side_fs.pose_backend}+metric_level_dual_fusion",
        audit=fused_audit,
        cal=fused_cal,
        events=fused_events,
        metrics=fused_metrics,
        workload=side_fs.workload_context,
        interval_type=side_fs.interval_type,
        metadata={
            "fused_from": [side_fs.bowler_id, rear_fs.bowler_id],
            "side_fps": side_fs.audit.fps,
            "rear_fps": rear_fs.audit.fps,
            "azimuth_baseline_sin": round(baseline_sin, 3),
        },
    )
