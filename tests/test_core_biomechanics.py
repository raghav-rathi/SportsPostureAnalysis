"""Expanded pytest suite verifying all P0/P1/P2 audit findings:
  - Non-circular transverse alignment vectors for BOTH Right and Left arms + np.unwrap
  - Coplanar (Y=0 creases only) & 3D MCC Law 7 PnP calibration
  - Upright-gravity self-calibration (preventing underground camera elevation)
  - Causal event spotting (BFC < FFC <= UAH < BR) + truncated-clip detection
  - Out-of-frame cropped-joint suppression + panning-camera suppression + permanent ICC refusal
  - Two-sided evaluator scoring, interior-knot non-monotonic CI, and dual-view arm mismatch guards
"""

from __future__ import annotations

import numpy as np
import pytest

from bowling_biomech.pipeline import fuse_dual_view_fact_sheets
from bowling_biomech.schemas import (
    BowlerActionClass,
    BowlingArm,
    CameraViewClass,
    ObservabilityGrade,
    VideoAudit,
)
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import (
    _transverse_alignment_deg,
    extract_fast_bowling_kinematics,
)
from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
from bowling_biomech.stage5_evaluator import build_fact_sheet
from bowling_biomech.stage6_coach_llm import generate_grounded_report, render_report_markdown
from bowling_biomech.synthetic_ground_truth import (
    JOINT_IDX,
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)


def _dummy_audit(fps: float = 240.0, blur_grade: str = "crisp") -> VideoAudit:
    return VideoAudit(
        video_path="synthetic.mp4",
        width=1280,
        height=720,
        fps=fps,
        num_frames=int(1.2 * fps),
        duration_s=1.2,
        median_laplacian_var=250.0 if blur_grade == "crisp" else 25.0,
        blur_grade=blur_grade,
        fps_grade="high_speed_240" if fps >= 200 else ("medium_60" if fps >= 48 else "standard_30"),
        est_arm_deg_per_frame=round(2300.0 / fps, 2),
        est_ball_m_per_frame=round(38.89 / fps, 3),
    )


def test_transverse_alignment_non_circular_vectors_right_and_left_arm():
    """Verify Portus (2004) transverse convention directly from explicit physical 3D shoulder vectors."""
    # Right-arm: +X down pitch, +Y up, +Z = bowler's right when facing +X
    # 1. Front-On (270 deg): lsho=(0, 1.4, -0.2), rsho=(0, 1.4, +0.2) => lr=(0, 0, +0.4)
    # 2. Right-arm Side-On (180 deg): lsho=(+0.2, 1.4, 0), rsho=(-0.2, 1.4, 0) => lr=(-0.4, 0, 0)
    # 3. Right-arm Counter-rotated 30 deg past Side-On (150 deg): lr=(-0.3464, 0, -0.2) => atan2(-0.3464, -0.2) = -120 => 150 deg
    l_pts = np.array([[0.0, 1.4, -0.2], [0.2, 1.4, 0.0], [0.1732, 1.4, 0.1]], dtype=np.float64)
    r_pts = np.array([[0.0, 1.4, 0.2], [-0.2, 1.4, 0.0], [-0.1732, 1.4, -0.1]], dtype=np.float64)
    ang_r = _transverse_alignment_deg(l_pts, r_pts, is_right_arm=True, ref_idx=0)
    assert abs(ang_r[0] - 270.0) < 0.5
    assert abs(ang_r[1] - 180.0) < 0.5
    assert abs(ang_r[2] - 150.0) < 0.5

    # Left-arm Side-On (180 deg): right shoulder points down pitch (+X), left shoulder points back (-X) => lr=(+0.4, 0, 0)
    l_pts_l = np.array([[0.0, 1.4, -0.2], [-0.2, 1.4, 0.0]], dtype=np.float64)
    r_pts_l = np.array([[0.0, 1.4, 0.2], [0.2, 1.4, 0.0]], dtype=np.float64)
    ang_l = _transverse_alignment_deg(l_pts_l, r_pts_l, is_right_arm=False, ref_idx=0)
    assert abs(ang_l[0] - 270.0) < 0.5
    assert abs(ang_l[1] - 180.0) < 0.5

    # Verify continuous unwrapping without 360-deg jump across +-pi branch cut
    thetas = np.linspace(-3.0, -3.5, 10)  # crosses -pi continuously
    l_wrap = np.zeros((10, 3))
    r_wrap = np.column_stack([np.sin(thetas), np.zeros(10), np.cos(thetas)])
    ang_wrap = _transverse_alignment_deg(l_wrap, r_wrap, is_right_arm=True, ref_idx=0)
    assert np.max(np.abs(np.diff(ang_wrap))) < 10.0


@pytest.mark.parametrize(
    "az_true,el_true,expected_view",
    [
        (0.0, 6.0, CameraViewClass.REAR_BEHIND_BOWLER),
        (90.0, 5.0, CameraViewClass.SIDE_ON_RIGHT),
        (45.0, 8.0, CameraViewClass.OPEN_SIDE_45),
        (8.0, 24.0, CameraViewClass.BROADCAST_ELEVATED),
    ],
)
def test_mcc_law7_pnp_calibration_accuracy(az_true, el_true, expected_view):
    K, R, t, _ = make_virtual_camera(azimuth_deg=az_true, elevation_deg=el_true, distance_m=12.0, fov_deg=55.0)
    keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg",
        "stump_middle_base", "stump_middle_top",
    ]
    lm_px = project_mcc_landmarks(K, R, t, keys, noise_px=0.6, rng=np.random.default_rng(7))
    cal = calibrate_from_mcc_landmarks(lm_px, width=1280, height=720, K=K)

    assert cal.is_metric_calibrated
    assert cal.view_class == expected_view
    assert cal.reprojection_error_px is not None and cal.reprojection_error_px < 2.5
    assert abs(cal.azimuth_deg - az_true) < 2.5
    assert abs(cal.elevation_deg - el_true) < 2.5


def test_coplanar_5point_creases_only_ippe_pnp():
    """Verify coplanar N>=5 ground-only crease landmarks solve cleanly above ground via IPPE."""
    K, R, t, _ = make_virtual_camera(azimuth_deg=40.0, elevation_deg=15.0, distance_m=14.0, fov_deg=42.0)
    coplanar_keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg",
        "popping_crease_center",
    ]
    lm_px = project_mcc_landmarks(K, R, t, coplanar_keys, noise_px=0.4, rng=np.random.default_rng(11))
    cal = calibrate_from_mcc_landmarks(lm_px, width=1280, height=720, K=K)
    assert cal.camera_position_world_m[1] > 0.5
    assert abs(cal.azimuth_deg - 40.0) < 3.0
    assert abs(cal.elevation_deg - 15.0) < 3.0


def test_motion_gravity_self_calibration_not_underground_and_not_metric():
    params = SyntheticDeliveryParams(fps=120.0)
    j_world, _ = generate_synthetic_delivery(params)
    K, R, t, _ = make_virtual_camera(azimuth_deg=0.0, elevation_deg=5.0, distance_m=11.0, fov_deg=55.0)
    _, j_cam = project_world_points(j_world, K, R, t)

    pelv_cam = j_cam[:, JOINT_IDX["pelv"]]
    up_cam = j_cam[:, JOINT_IDX["neck"]] - pelv_cam
    cal = calibrate_from_3d_motion_and_gravity(
        pelv_cam, up_cam, width=1280, height=720, fov_deg=55.0, joints3d_cam_m=j_cam
    )
    assert cal.method == "motion_gravity_self_cal"
    assert cal.is_metric_calibrated is False
    assert cal.elevation_deg >= -1.0  # Never underground!
    assert cal.view_class == CameraViewClass.REAR_BEHIND_BOWLER


@pytest.mark.parametrize("arm", [BowlingArm.RIGHT, BowlingArm.LEFT])
def test_synthetic_240fps_event_spotting_and_kinematics_both_arms(arm):
    params = SyntheticDeliveryParams(
        bowler_arm=arm,
        fps=240.0,
        runup_speed_mps=5.9,
        front_knee_ffc_deg=158.0,
        front_knee_br_deg=170.0,
        trunk_flexion_ffc_deg=12.0,
        trunk_flexion_br_deg=44.0,
        thorax_lat_flex_br_deg=36.0,
        shoulder_counter_rot_deg=18.0,
        delayed_arm_ffc_deg=146.0,
    )
    j_world, gt = generate_synthetic_delivery(params)
    events = detect_delivery_events(j_world, fps=240.0, bowling_arm=arm)

    assert events.is_complete_delivery is True
    assert events.bfc.frame_float < events.ffc.frame_float
    assert events.uah is not None
    assert events.ffc.frame_float <= events.uah.frame_float < events.br.frame_float
    assert abs(events.bfc.frame_float - gt["bfc_frame"]) <= 4.0
    assert abs(events.ffc.frame_float - gt["ffc_frame"]) <= 3.0
    assert abs(events.br.frame_float - gt["br_frame"]) <= 3.0

    kin = extract_fast_bowling_kinematics(j_world, events, fps=240.0, bowling_arm=arm)
    assert 1.75 <= kin.estimated_stature_m <= 1.98
    assert abs(kin.scalars["runup_speed_mps"] - gt["runup_speed_mps"]) < 0.35
    assert abs(kin.scalars["front_knee_br_deg"] - gt["front_knee_br_deg"]) < 3.0
    assert abs(kin.scalars["trunk_flexion_delta_deg"] - gt["trunk_flexion_delta_deg"]) < 3.5
    assert abs(kin.scalars["thorax_lat_flex_br_deg"] - gt["thorax_lat_flex_br_deg"]) < 3.0
    assert abs(kin.scalars["shoulder_counter_rotation_deg"] - gt["shoulder_counter_rotation_deg"]) < 4.5
    assert abs(kin.scalars["delayed_arm_ffc_deg"] - gt["delayed_arm_ffc_deg"]) < 5.0
    assert kin.action_class != BowlerActionClass.MIXED


def test_cropped_ankle_and_panning_and_truncated_clip_suppression():
    params = SyntheticDeliveryParams(fps=120.0)
    j_world, _ = generate_synthetic_delivery(params)
    events = detect_delivery_events(j_world, fps=120.0)

    # Simulate front ankle cropped outside frame (like indoor sofa video!)
    vis = np.ones((j_world.shape[0], 24), dtype=bool)
    vis[:, JOINT_IDX["lank"]] = False
    kin = extract_fast_bowling_kinematics(j_world, events, fps=120.0, joint_in_frame=vis)

    K_s, R_s, t_s, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    lm_s = project_mcc_landmarks(
        K_s, R_s, t_s,
        ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"],
    )
    cal_pan = calibrate_from_mcc_landmarks(lm_s, 1280, 720, K=K_s).model_copy(
        update={"is_metric_calibrated": False, "is_panning_camera": True}
    )
    m = quantify_and_gate_metrics(kin, _dummy_audit(fps=120.0), cal_pan, events=events)

    assert m["front_knee_br_deg"].suppressed is True
    assert "cropped" in (m["front_knee_br_deg"].suppression_reason or "").lower()
    assert m["runup_speed_mps"].suppressed is True
    assert m["stride_length_pct_stature"].suppressed is True
    assert m["icc_elbow_legality_verdict"].observability == ObservabilityGrade.REFUSED

    # Truncated clip ending before overhead ball release (first 62% of frames, cut off near FFC)
    j_trunc = j_world[: int(0.60 * len(j_world))]
    ev_trunc = detect_delivery_events(j_trunc, fps=120.0)
    assert ev_trunc.is_complete_delivery is False
    kin_t = extract_fast_bowling_kinematics(j_trunc, ev_trunc, fps=120.0)
    m_t = quantify_and_gate_metrics(kin_t, _dummy_audit(fps=120.0), cal_pan, events=ev_trunc)
    assert m_t["front_knee_br_deg"].suppressed is True
    assert m_t["trunk_flexion_delta_deg"].suppressed is True


def test_observability_gate_60fps_velocity_high_and_ci_clamping():
    params = SyntheticDeliveryParams(fps=60.0, front_knee_br_deg=176.0)
    j_world, _ = generate_synthetic_delivery(params)
    events = detect_delivery_events(j_world, fps=60.0)
    kin = extract_fast_bowling_kinematics(j_world, events, fps=60.0)
    K_s, R_s, t_s, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    lm_s = project_mcc_landmarks(
        K_s, R_s, t_s,
        ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"],
    )
    cal_side = calibrate_from_mcc_landmarks(lm_s, 1280, 720, K=K_s)
    m = quantify_and_gate_metrics(kin, _dummy_audit(fps=60.0), cal_side, events=events)

    # High-velocity arm metrics must be suppressed at 60 fps (<90 fps)
    assert m["delayed_arm_ffc_deg"].suppressed is True
    assert m["peak_arm_angular_vel_dps"].suppressed is True
    # Physical CI clamping: internal knee angle CI must never exceed 180.0 deg!
    assert m["front_knee_br_deg"].ci_high is not None and m["front_knee_br_deg"].ci_high <= 180.0
    # Zero leaked enum string "velocity_high-plane"
    for me in m.values():
        assert "velocity_high-plane" not in (me.suppression_reason or "")


def test_evaluator_two_sided_curves_dual_fusion_and_icc_caveat_preserved():
    p_healthy = SyntheticDeliveryParams(
        fps=240.0, front_knee_br_deg=170.0, trunk_flexion_br_deg=44.0,
        thorax_lat_flex_br_deg=36.0, shoulder_counter_rot_deg=15.0,
    )
    p_highload = SyntheticDeliveryParams(
        fps=240.0, front_knee_br_deg=172.0, trunk_flexion_br_deg=46.0,
        thorax_lat_flex_br_deg=53.0, shoulder_counter_rot_deg=38.0,
    )
    p_ipsi_drift = SyntheticDeliveryParams(
        fps=240.0, front_knee_br_deg=168.0, trunk_flexion_br_deg=42.0,
        thorax_lat_flex_br_deg=-18.0, follow_through_drift_deg=-32.0,
    )

    K_s, R_s, t_s, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    K_r, R_r, t_r, _ = make_virtual_camera(azimuth_deg=0.0, elevation_deg=6.0)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    cal_s = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_s, R_s, t_s, lm_keys), 1280, 720, K=K_s)
    cal_r = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_r, R_r, t_r, lm_keys), 1280, 720, K=K_r)
    aud = _dummy_audit(fps=240.0)

    def _run_fused(p: SyntheticDeliveryParams, bid: str):
        j, _ = generate_synthetic_delivery(p)
        ev = detect_delivery_events(j, fps=240.0, bowling_arm=p.bowler_arm)
        kin = extract_fast_bowling_kinematics(j, ev, fps=240.0, bowling_arm=p.bowler_arm)
        fs_s = build_fact_sheet(bid, p.bowler_arm, kin.action_class, "synthetic", aud, cal_s, ev, quantify_and_gate_metrics(kin, aud, cal_s, ev))
        fs_r = build_fact_sheet(bid, p.bowler_arm, kin.action_class, "synthetic", aud, cal_r, ev, quantify_and_gate_metrics(kin, aud, cal_r, ev))
        return fuse_dual_view_fact_sheets(fs_s, fs_r, bowler_id=bid)

    fs_healthy = _run_fused(p_healthy, "HealthyElite")
    fs_load = _run_fused(p_highload, "ExplosiveHighLoad")
    fs_ipsi = _run_fused(p_ipsi_drift, "IpsilateralDrift")

    assert fs_healthy.quadrant.quadrant_label == "High Efficiency + High Durability"
    assert fs_load.quadrant.quadrant_label == "High Efficiency + Elevated Mechanical Load"
    assert fs_healthy.lumbar_load_screening.score > fs_load.lumbar_load_screening.score + 25.0
    # Ipsilateral negative lateral bend + negative leg-side drift MUST score substantially lower than healthy!
    assert fs_ipsi.lumbar_load_screening.score < fs_healthy.lumbar_load_screening.score - 15.0

    # Verify mismatched arm fusion raises ValueError
    p_left = SyntheticDeliveryParams(bowler_arm=BowlingArm.LEFT, fps=240.0)
    fs_left = _run_fused(p_left, "LeftArm")
    with pytest.raises(ValueError, match="mismatched bowling arms"):
        fuse_dual_view_fact_sheets(fs_healthy, fs_left)

    # Verify REFUSED ICC verdict is ALWAYS present in caveats even when many metrics are suppressed
    j, _ = generate_synthetic_delivery(p_highload)
    ev = detect_delivery_events(j, fps=30.0)
    kin30 = extract_fast_bowling_kinematics(j, ev, fps=30.0)
    fs_rear30 = build_fact_sheet("Rear30", BowlingArm.RIGHT, kin30.action_class, "synthetic", _dummy_audit(30.0), cal_r, ev, quantify_and_gate_metrics(kin30, _dummy_audit(30.0), cal_r, ev))
    rep = generate_grounded_report(fs_rear30)
    assert any("ICC 15-Degree Elbow Legality" in c for c in rep.capture_quality_caveats)
    assert "icc_elbow_legality_verdict" not in rep.cited_metric_keys
    md = render_report_markdown(fs_rear30, rep)
    assert "REFUSED" in md


def test_detect_camera_panning_static_vs_panning_and_angle_upgrade_gating(tmp_path):
    import cv2
    from bowling_biomech.stage0_audit import detect_camera_panning

    rng = np.random.default_rng(7)
    h, w = 240, 320
    # Rich textured background pattern (e.g., nets, doors, stands)
    bg_wide = rng.integers(40, 210, size=(h, w + 160, 3), dtype=np.uint8)
    bg_wide = cv2.GaussianBlur(bg_wide, (3, 3), 0.8)

    # 1. Static camera with moving bright TV / person block (must NOT false-positive as panning!)
    static_path = tmp_path / "static_with_tv.mp4"
    vw = cv2.VideoWriter(str(static_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(20):
        frame = bg_wide[:, :w].copy()
        # Moving foreground block (simulating TV/non-striker) in top-right corner
        x0 = 210 + (i % 6) * 4
        frame[20:70, x0 : x0 + 40] = (230, 180, 90)
        vw.write(frame)
    vw.release()
    is_pan_s, frac_s = detect_camera_panning(static_path)
    assert is_pan_s is False, f"Static camera false-positive: pan_frac={frac_s}"

    # 2. Panning camera with black letterbox bars + static white watermark (MUST detect panning!)
    pan_path = tmp_path / "panning_letterbox.mp4"
    vw2 = cv2.VideoWriter(str(pan_path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(20):
        shift = i * 4
        frame = bg_wide[:, shift : shift + w].copy()
        frame[:30, :] = 0   # top letterbox
        frame[-30:, :] = 0  # bottom letterbox
        frame[36:56, 20:140] = 250  # bright static watermark banner
        vw2.write(frame)
    vw2.release()
    is_pan_p, frac_p = detect_camera_panning(pan_path)
    assert is_pan_p is True and frac_p >= 0.09, f"Panning camera missed: pan_frac={frac_p}"

    # 3. Rear-on blurry clip (az ~ 2 deg) with suppressed coronal metric must NOT suggest "Add a Behind-the-Bowler (Rear) Camera Angle"
    params = SyntheticDeliveryParams(fps=25.0)
    j_world, _ = generate_synthetic_delivery(params)
    ev = detect_delivery_events(j_world, fps=25.0)
    kin = extract_fast_bowling_kinematics(j_world, ev, fps=25.0)
    K_r, R_r, t_r, _ = make_virtual_camera(azimuth_deg=2.0, elevation_deg=6.0)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    cal_r = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_r, R_r, t_r, lm_keys), 1280, 720, K=K_r)
    m = quantify_and_gate_metrics(kin, _dummy_audit(fps=25.0), cal_r, events=ev)
    # Force pelvic drop suppressed as if by 25 fps blur
    m["pelvic_drop_ffc_deg"] = m["pelvic_drop_ffc_deg"].model_copy(update={"suppressed": True})
    fs = build_fact_sheet("RearBlur", BowlingArm.RIGHT, kin.action_class, "synthetic", _dummy_audit(25.0), cal_r, ev, m)
    upgrade_titles = [c.title for c in fs.capture_upgrades]
    assert not any("Rear" in t for t in upgrade_titles), f"Misleading rear-angle cue on rear clip: {upgrade_titles}"

