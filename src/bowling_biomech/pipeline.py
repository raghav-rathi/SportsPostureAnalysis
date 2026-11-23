"""End-to-End Fast-Bowling Biomechanics Pipeline & Multi-Angle Fusion Orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from .schemas import (
    BiomechanicalFactSheet,
    BowlerActionClass,
    BowlingArm,
    CameraViewClass,
    CoachingReport,
    MetricEstimate,
    ObservabilityGrade,
    WorkloadContext,
)
from .stage0_audit import audit_video, detect_camera_panning
from .stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
)
from .stage2_event_spotting import detect_delivery_events
from .stage2_nlf_3d import extract_bowler_3d_nlf
from .stage3_kinematics import classify_action_portus, extract_fast_bowling_kinematics
from .stage4_uncertainty import quantify_and_gate_metrics
from .stage5_evaluator import build_fact_sheet
from .stage6_coach_llm import generate_grounded_report, render_report_markdown
from .stage6_visualizer import (
    render_annotated_freeze_video,
    render_biomechanics_dashboard,
    render_keyframe_panel,
)
from .synthetic_ground_truth import JOINT_IDX


def analyze_delivery_video(
    video_path: str | Path,
    bowler_id: str,
    nlf_model_path: str | Path,
    output_dir: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    bowling_arm: Optional[BowlingArm] = None,
    mcc_landmarks_px: Optional[Dict[str, Tuple[float, float]]] = None,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    workload: Optional[WorkloadContext] = None,
    capture_fps_override: Optional[float] = None,
    num_aug: int = 1,
    device: str = "cuda",
) -> Tuple[BiomechanicalFactSheet, CoachingReport, Dict[str, Path]]:
    """Run the full 7-stage fast-bowling biomechanics pipeline on a single video clip."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0: Sub-Clip Video Quality & Physics Audit
    audit = audit_video(video_path, start_frame=start_frame, end_frame=end_frame)
    if capture_fps_override is not None and capture_fps_override > 0:
        eff_fps = float(capture_fps_override)
        audit = audit.model_copy(
            update={
                "fps": round(eff_fps, 3),
                "duration_s": round(audit.num_frames / eff_fps, 3),
                "fps_grade": (
                    "high_speed_240" if eff_fps >= 200
                    else ("slow_mo_120" if eff_fps >= 100 else ("medium_60" if eff_fps >= 48 else "standard_30"))
                ),
                "est_arm_deg_per_frame": round(2300.0 / eff_fps, 2),
                "est_ball_m_per_frame": round(38.89 / eff_fps, 3),
                "warnings": [w for w in audit.warnings if "Frame rate is" not in w]
                + [f"Using caller-specified true capture FPS={eff_fps:.1f} fps (slow-motion playback override)."],
            }
        )

    # Stage 2A: Frontier 3D Pose + Visibility + Uncertainty Extraction (NLF on CUDA)
    tracked = extract_bowler_3d_nlf(
        video_path=video_path,
        model_path=nlf_model_path,
        start_frame=start_frame,
        end_frame=end_frame,
        num_aug=num_aug,
        bowling_arm_override=bowling_arm,
        anchor_xy_norm=anchor_xy_norm,
        device=device,
    )

    # Stage 0B: Background Camera Panning Detection (letterbox + watermark + bowler masked)
    is_panning, pan_frac = detect_camera_panning(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        bowler_boxes_xyxy=tracked.bowler_boxes_xyxy,
    )

    # Stage 1: Pitch Geometry Calibration (MCC Law 7 PnP or Upright-Gravity Self-Calibration)
    if mcc_landmarks_px is not None and len(mcc_landmarks_px) >= 4:
        cal = calibrate_from_mcc_landmarks(
            image_landmarks_px=mcc_landmarks_px,
            width=audit.width,
            height=audit.height,
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
        )

    joints_world = transform_cam_to_pitch_world(tracked.joints3d_cam_m, cal)

    # Stage 2B: Causal Multi-Cue Sub-Frame Event Spotting (BFC < FFC <= UAH < BR < FT)
    events = detect_delivery_events(
        joints_world=joints_world,
        fps=audit.fps,
        bowling_arm=tracked.detected_bowling_arm,
        joint_in_frame=tracked.joint_in_frame,
    )

    # Stage 3: Peer-Reviewed 3D Fast-Bowling Kinematics
    kin = extract_fast_bowling_kinematics(
        joints_world=joints_world,
        events=events,
        fps=audit.fps,
        bowling_arm=tracked.detected_bowling_arm,
        joint_uncertainties_m=tracked.joint_uncertainties_m,
        joint_in_frame=tracked.joint_in_frame,
    )

    # Kinematic panning cross-check: on a side/45 view, if bowler 2D pelvis stays centered in frame
    # while measured ground run-up speed is near zero, flag as uncompensated panning/stationary drill!
    if (not cal.is_metric_calibrated) and (not cal.is_panning_camera):
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
                        "calibration_notes": cal.calibration_notes
                        + [
                            f"Kinematic pan check: bowler stayed centered (2D X span={x_span_frac:.2f}w, "
                            f"apparent speed={kin.scalars.get('runup_speed_mps', 0.0):.2f} m/s); "
                            "ground-referenced run-up/stride suppressed."
                        ],
                    }
                )

    # Stage 4: Camera-View Observability Matrix, Crop Gating, & 3-Source Uncertainty Gate
    metrics = quantify_and_gate_metrics(kin=kin, audit=audit, cal=cal, events=events)

    # Stage 5: Archetype-Conditioned Fast-Bowling Evaluator
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
        metadata={
            "start_frame": start_frame,
            "end_frame": end_frame,
            "num_tracks_seen": tracked.num_tracks_seen,
            "valid_detection_ratio": round(float(np.mean(tracked.valid_mask)), 3),
            "estimated_stature_m": round(kin.estimated_stature_m, 3),
            "background_pan_fraction": pan_frac,
            "standing_bg_gravity_frames": len(tracked.background_upright_vectors),
        },
    )

    # Stage 6A & 6B: Grounded Coaching Report & Truthful Visuals
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
    np.savez_compressed(
        traj_path,
        joints3d_cam_m=tracked.joints3d_cam_m,
        joints3d_world_m=joints_world,
        joints2d_px=tracked.joints2d_px,
        joint_uncertainties_m=tracked.joint_uncertainties_m,
        joint_in_frame=tracked.joint_in_frame,
        valid_mask=tracked.valid_mask,
    )
    render_keyframe_panel(
        video_path=video_path,
        start_frame=start_frame,
        joints2d_px=tracked.joints2d_px,
        fact_sheet=fact_sheet,
        out_path=keyframe_path,
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
    """Fuse complementary side-on and rear-on FactSheets via observability + inverse-variance weighting."""
    if side_fs.bowling_arm != rear_fs.bowling_arm:
        raise ValueError(
            f"Cannot fuse clips with mismatched bowling arms: side={side_fs.bowling_arm.value} "
            f"vs rear={rear_fs.bowling_arm.value}."
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
                fused_metrics[k] = m1.model_copy(
                    update={
                        "value": round(float(v_fused), 2),
                        "ci_low": round(float(v_fused - half), 2),
                        "ci_high": round(float(v_fused + half), 2),
                        "sigma_total": round(s_fused, 2),
                    }
                )
        else:
            fused_metrics[k] = m1

    fused_cal = side_fs.calibration.model_copy(
        update={
            "view_class": CameraViewClass.DUAL_VIEW_FUSED,
            "is_metric_calibrated": side_fs.calibration.is_metric_calibrated or rear_fs.calibration.is_metric_calibrated,
            "is_panning_camera": side_fs.calibration.is_panning_camera and rear_fs.calibration.is_panning_camera,
            "calibration_notes": side_fs.calibration.calibration_notes
            + rear_fs.calibration.calibration_notes
            + ["Fused complementary side-on and rear-on views."],
        }
    )

    # Recompute action_class directly from fused transverse alignment metrics
    sho_m = fused_metrics.get("shoulder_alignment_bfc_deg")
    pelv_m = fused_metrics.get("pelvis_alignment_bfc_deg")
    scr_m = fused_metrics.get("shoulder_counter_rotation_deg")
    if sho_m and not sho_m.suppressed and sho_m.value is not None:
        p_val = pelv_m.value if (pelv_m and not pelv_m.suppressed and pelv_m.value is not None) else sho_m.value
        s_val = scr_m.value if (scr_m and not scr_m.suppressed and scr_m.value is not None) else 0.0
        fused_action = classify_action_portus(sho_m.value, p_val, s_val)
    else:
        fused_action = BowlerActionClass.UNKNOWN

    # Use higher effective FPS between the two clips for audit metadata
    fused_audit = side_fs.audit if side_fs.audit.fps >= rear_fs.audit.fps else rear_fs.audit

    return build_fact_sheet(
        bowler_id=bowler_id or side_fs.bowler_id,
        bowling_arm=side_fs.bowling_arm,
        action_class=fused_action,
        pose_backend=f"{side_fs.pose_backend}+dual_view_fusion",
        audit=fused_audit,
        cal=fused_cal,
        events=side_fs.events if side_fs.events.is_complete_delivery else rear_fs.events,
        metrics=fused_metrics,
        workload=side_fs.workload_context,
        metadata={
            "fused_from": [side_fs.bowler_id, rear_fs.bowler_id],
            "side_fps": side_fs.audit.fps,
            "rear_fps": rear_fs.audit.fps,
        },
    )
