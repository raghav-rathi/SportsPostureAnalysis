"""Comprehensive Unit & Regression Test Suite for Cricket Fast-Bowling Biomechanics (Phase 1.5).

Covers non-circular vector conventions, MCC Laws 6/7/8 PnP & Two-Wicket geometry, larger-side NLF
intrinsics, Stage 0 pulldown/freeze/pan/tilt/zoom checks, Stage 2B release sanity gates (including
slingy bowlers & frozen stills), Stage 3 local ground & wrap safety, Stage 4 Jacobian & branch order,
Stage 5 interval-aware cues & retired composites, and permanent ICC 15-degree refusal.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bowling_biomech.constants import JOINT_IDX, MCC_STUMP_CENTER_HALF_SPACING_M
from bowling_biomech.pipeline import fuse_dual_view_fact_sheets
from bowling_biomech.schemas import (
    BowlerActionClass,
    BowlingArm,
    CameraViewClass,
    ObservabilityGrade,
    VideoAudit,
    WorkloadContext,
)
from bowling_biomech.stage0_audit import audit_video, detect_camera_panning
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    default_camera_matrix,
    solve_two_wicket_calibration,
    transform_cam_to_pitch_world,
    world_up_from_calibration,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import (
    _transverse_alignment_deg,
    classify_action_portus,
    extract_fast_bowling_kinematics,
)
from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
from bowling_biomech.stage5_evaluator import build_fact_sheet, evaluate_repeatability
from bowling_biomech.stage6_coach_llm import generate_grounded_report, render_report_markdown
from bowling_biomech.synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)


def _dummy_audit(fps: float = 240.0, unreliable: bool = False) -> VideoAudit:
    return VideoAudit(
        video_path="synthetic.mp4",
        width=1280,
        height=720,
        fps=fps,
        num_frames=int(round(1.15 * fps)),
        duration_s=1.15,
        median_laplacian_var=240.0,
        blur_grade="crisp",
        fps_grade="high_speed_240" if fps >= 200 else ("slow_mo_120" if fps >= 100 else "standard_30"),
        est_arm_deg_per_frame=2300.0 / fps,
        est_ball_m_per_frame=38.89 / fps,
        timebase_unreliable=unreliable,
        timebase_status="edited_or_frozen" if unreliable else "verified_native",
    )


def test_transverse_alignment_non_circular_and_b18_wrap_safety():
    """Non-circular vector test for right & left arms PLUS EXP-007 B18 straddle wrap test."""
    # Right-arm: Front-on (270), Side-on (180), Semi-open (210), Over-rotated (150)
    pts_r = [
        (np.array([0.0, 1.4, -0.2]), np.array([0.0, 1.4, 0.2]), 270.0),
        (np.array([0.2, 1.4, 0.0]), np.array([-0.2, 1.4, 0.0]), 180.0),
        (np.array([0.173205, 1.4, -0.1]), np.array([-0.173205, 1.4, 0.1]), 210.0),
        (np.array([0.173205, 1.4, 0.1]), np.array([-0.173205, 1.4, -0.1]), 150.0),
    ]
    for l_pt, r_pt, expected in pts_r:
        ang = _transverse_alignment_deg(l_pt[None, :], r_pt[None, :], is_right_arm=True, ref_idx=0)[0]
        assert abs(ang - expected) < 0.25

    # Left-arm mirror
    pts_l = [
        (np.array([0.0, 1.4, -0.2]), np.array([0.0, 1.4, 0.2]), 270.0),
        (np.array([-0.2, 1.4, 0.0]), np.array([0.2, 1.4, 0.0]), 180.0),
        (np.array([-0.173205, 1.4, -0.1]), np.array([0.173205, 1.4, 0.1]), 210.0),
    ]
    for l_pt, r_pt, expected in pts_l:
        ang = _transverse_alignment_deg(l_pt[None, :], r_pt[None, :], is_right_arm=False, ref_idx=0)[0]
        assert abs(ang - expected) < 0.25

    # EXP-007 B18: Pelvis at 92 deg & shoulder at 86 deg straddling 90 deg must give +6 deg, NEVER -354 deg!
    l_p = np.array([[0.0, 1.0, 0.0]])
    # 92 deg -> rad = -178 deg -> lr = [sin(-178), 0, cos(-178)] = [-0.0349, 0, -0.9994]
    r_p = np.array([[math.sin(math.radians(-178.0)), 1.0, math.cos(math.radians(-178.0))]])
    # 86 deg -> rad = +176 deg -> lr = [sin(176), 0, cos(176)] = [+0.0698, 0, -0.9976]
    r_s = np.array([[math.sin(math.radians(176.0)), 1.4, math.cos(math.radians(176.0))]])
    l_s = np.array([[0.0, 1.4, 0.0]])
    p_yaw = _transverse_alignment_deg(l_p, r_p, is_right_arm=True, ref_idx=0)
    s_yaw = _transverse_alignment_deg(l_s, r_s, is_right_arm=True, ref_idx=0, anchor_target_deg=float(p_yaw[0]))
    sep = float(p_yaw[0] - s_yaw[0])
    assert abs(sep - 6.0) < 0.2, f"B18 wrap regression: sep={sep}"

    # Portus (2001/2004) AIS cutoffs
    assert classify_action_portus(205.0, 205.0, 10.0) == BowlerActionClass.SIDE_ON
    assert classify_action_portus(225.0, 220.0, 10.0) == BowlerActionClass.SEMI_OPEN
    assert classify_action_portus(250.0, 248.0, 10.0) == BowlerActionClass.FRONT_ON
    assert classify_action_portus(225.0, 220.0, 32.0) == BowlerActionClass.MIXED


@pytest.mark.parametrize(
    "az,el,expected_view",
    [
        (0.0, 6.0, CameraViewClass.REAR_BEHIND_BOWLER),
        (90.0, 5.0, CameraViewClass.SIDE_ON_RIGHT),
        (45.0, 8.0, CameraViewClass.OPEN_SIDE_45),
        (8.0, 24.0, CameraViewClass.BROADCAST_ELEVATED),
    ],
)
def test_mcc_pnp_and_b4_pan_preservation(az, el, expected_view):
    K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg",
        "stump_middle_base", "stump_middle_top",
    ]
    img_pts = project_mcc_landmarks(K, R, t, lm_keys, noise_px=0.0)
    cal = calibrate_from_mcc_landmarks(img_pts, 1280, 720, K=K)
    assert cal.reprojection_error_px is not None and cal.reprojection_error_px < 0.5
    assert cal.view_class == expected_view
    assert cal.is_metric_calibrated is True
    assert cal.is_intrinsics_calibrated is True

    # EXP-007 B4: Landmark path MUST preserve is_panning_camera=True and suppress metric ground translation!
    cal_pan = calibrate_from_mcc_landmarks(img_pts, 1280, 720, K=K, is_panning_camera=True)
    assert cal_pan.is_panning_camera is True
    assert cal_pan.is_metric_calibrated is False
    assert cal_pan.is_intrinsics_calibrated is True


def test_b16_larger_side_intrinsics_and_b24_two_wicket_closed_form():
    # EXP-007 B16: Portrait 720x1280 and landscape 1280x720 must share identical larger-side focal length
    K_land = default_camera_matrix(1280, 720, 55.0)
    K_port = default_camera_matrix(720, 1280, 55.0)
    assert abs(K_land[0, 0] - K_port[0, 0]) < 1e-6

    # EXP-007 B24: Regulation Law 8 stump center half-spacing
    assert abs(MCC_STUMP_CENTER_HALF_SPACING_M - 0.096) < 0.002

    # Two-wicket closed-form calibration reproducing Steyn geometry (EXP-007 Section 3.3)
    cal_tw = solve_two_wicket_calibration(
        h_near_px=294.0, base_near_y_px=1060.0,
        h_far_px=170.0, base_far_y_px=810.0,
        width=1920, height=1080,
        is_panning_camera=True,
    )
    assert cal_tw is not None
    assert cal_tw.is_intrinsics_calibrated is True
    assert cal_tw.is_metric_calibrated is False  # panning preserved
    assert 10500.0 < cal_tw.camera_matrix_K[0][0] < 12300.0
    up_cam = world_up_from_calibration(cal_tw)
    assert up_cam[1] < -0.95  # world UP points along -Y in OpenCV camera frame


def test_coplanar_5point_creases_only_ippe_pnp():
    K, R, t, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=6.0, distance_m=11.0, fov_deg=55.0)
    coplanar_keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg", "popping_crease_center",
    ]
    img_pts = project_mcc_landmarks(K, R, t, coplanar_keys, noise_px=0.0)
    cal = calibrate_from_mcc_landmarks(img_pts, 1280, 720, K=K)
    assert cal.reprojection_error_px is not None and cal.reprojection_error_px < 1.0
    assert cal.camera_position_world_m[1] > 0.5
    assert abs(cal.azimuth_deg - 90.0) < 3.0


def test_self_cal_consistency_b15_and_local_release_height_b5():
    """Verify B15 (cam_pos == -R.T @ t strictly) and B5 (local front-foot ground fixes sprint-lean release height)."""
    params = SyntheticDeliveryParams(fps=120.0, sprint_lean_deg=14.0)
    j_world, _ = generate_synthetic_delivery(params)
    K, R, t, _ = make_virtual_camera(azimuth_deg=45.0, elevation_deg=8.0, distance_m=12.0, fov_deg=55.0)
    _, j_cam = project_world_points(j_world, K, R, t)
    cal = calibrate_from_3d_motion_and_gravity(
        j_cam[:, JOINT_IDX["pelv"]],
        j_cam[:, JOINT_IDX["neck"]] - j_cam[:, JOINT_IDX["pelv"]],
        1280,
        720,
        joints3d_cam_m=j_cam,
    )
    assert cal.is_metric_calibrated is False
    R_est = np.asarray(cal.rotation_matrix_R)
    t_est = np.asarray(cal.translation_t)
    # EXP-007 B15: camera_position_world_m MUST remain algebraically consistent with -R.T @ t
    assert np.allclose(cal.camera_position_world_m, -R_est.T @ t_est, atol=1e-5)

    # Local ground reference keeps release_height error small even under 14 deg sprint lean!
    jw_est = transform_cam_to_pitch_world(j_cam, cal)
    ev = detect_delivery_events(jw_est, fps=120.0)
    kin_est = extract_fast_bowling_kinematics(jw_est, ev, fps=120.0)
    ev_gt = detect_delivery_events(j_world, fps=120.0)
    kin_gt = extract_fast_bowling_kinematics(j_world, ev_gt, fps=120.0)
    assert abs(kin_est.scalars["release_height_m"] - kin_gt.scalars["release_height_m"]) < 0.08


@pytest.mark.parametrize("arm", [BowlingArm.RIGHT, BowlingArm.LEFT])
def test_synthetic_240fps_non_teleporting_kinematics_both_arms(arm):
    params = SyntheticDeliveryParams(
        bowler_arm=arm, fps=240.0, front_knee_br_deg=170.0, trunk_flexion_br_deg=46.0,
        thorax_lat_flex_br_deg=38.0, shoulder_counter_rot_deg=18.0, delayed_arm_ffc_deg=146.0,
    )
    j_world, gt = generate_synthetic_delivery(params)
    # Verify non-teleporting foot contact (EXP-007 S3: max 1-frame ankle step < 0.08 m at 240 fps!)
    f_ank_i = JOINT_IDX["lank" if arm == BowlingArm.RIGHT else "rank"]
    max_step = float(np.max(np.linalg.norm(np.diff(j_world[:, f_ank_i], axis=0), axis=1)))
    assert max_step < 0.08, f"Teleporting front ankle detected: max_step={max_step:.3f} m"

    events = detect_delivery_events(j_world, fps=240.0, bowling_arm=arm)
    assert events.is_complete_delivery is True
    assert events.bfc.frame_float < events.ffc.frame_float < events.br.frame_float
    if events.uah is not None:
        assert events.ffc.frame_float <= events.uah.frame_float < events.br.frame_float
    assert abs(events.br.frame_float - gt["br_frame"]) < 2.5

    kin = extract_fast_bowling_kinematics(j_world, events, fps=240.0, bowling_arm=arm)
    assert abs(kin.scalars["front_knee_br_deg"] - gt["front_knee_br_deg"]) < 3.0
    assert abs(kin.scalars["trunk_flexion_delta_deg"] - gt["trunk_flexion_delta_deg"]) < 3.5
    assert abs(kin.scalars["thorax_lat_flex_br_deg"] - gt["thorax_lat_flex_br_deg"]) < 3.0
    assert abs(kin.scalars["shoulder_counter_rotation_deg"] - gt["shoulder_counter_rotation_deg"]) < 4.5
    assert kin.action_class != BowlerActionClass.MIXED


def test_b1_release_sanity_gates_reject_incomplete_and_frozen_and_pass_slingy():
    params = SyntheticDeliveryParams(fps=120.0)
    j_world, _ = generate_synthetic_delivery(params)

    # 1. Truncated clip ending ~45 deg before vertical (like McGrath Reel) MUST return is_complete_delivery=False!
    cut_idx = int(round((params.ffc_time_s + 0.055) * 120.0))
    ev_trunc = detect_delivery_events(j_world[:cut_idx], fps=120.0)
    assert ev_trunc.is_complete_delivery is False

    # 2. Frozen clip around release MUST be rejected!
    j_frozen = j_world.copy()
    br_i = int(round(params.br_time_s * 120.0))
    j_frozen[br_i - 4 :] = j_frozen[br_i - 4]  # freeze before release
    ev_frz = detect_delivery_events(j_frozen, fps=120.0)
    assert ev_frz.is_complete_delivery is False

    # 3. Slingy round-arm fast bowler (wrist 0.14 m above shoulder, extended 0.54 m laterally +Z) MUST PASS!
    j_sling = j_world.copy()
    sho = j_sling[:, JOINT_IDX["rsho"]]
    elb = j_sling[:, JOINT_IDX["relb"]]
    wri = j_sling[:, JOINT_IDX["rwri"]]
    # Rotate arm 62 deg outward into +Z (right-arm outward abduction) while preserving sagittal circumduction X
    ang = math.radians(62.0)
    for arr_j in (elb, wri):
        v = arr_j - sho
        vy_new = v[:, 1] * math.cos(ang) - v[:, 2] * math.sin(ang)
        vz_new = v[:, 1] * math.sin(ang) + v[:, 2] * math.cos(ang)
        arr_j[:, 1] = sho[:, 1] + vy_new
        arr_j[:, 2] = sho[:, 2] + vz_new
    ev_sling = detect_delivery_events(j_sling, fps=120.0, bowling_arm=BowlingArm.RIGHT)
    assert ev_sling.is_complete_delivery is True


def test_b5_b17_b8_translation_gating_branch_order_and_interval_aware_cues():
    params = SyntheticDeliveryParams(fps=120.0, front_knee_br_deg=174.0)
    j_world, _ = generate_synthetic_delivery(params)
    events = detect_delivery_events(j_world, fps=120.0)

    # Cropped ankle + panning -> front_knee, runup, stride, AND follow_through_drift all suppressed (B5)
    vis = np.ones((j_world.shape[0], 24), dtype=bool)
    vis[:, JOINT_IDX["lank"]] = False
    kin = extract_fast_bowling_kinematics(j_world, events, fps=120.0, joint_in_frame=vis)

    K_s, R_s, t_s, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    lm_s = project_mcc_landmarks(
        K_s, R_s, t_s,
        ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"],
    )
    cal_pan = calibrate_from_mcc_landmarks(lm_s, 1280, 720, K=K_s, is_panning_camera=True)
    m = quantify_and_gate_metrics(kin, _dummy_audit(fps=120.0), cal_pan, events=events)
    assert m["front_knee_br_deg"].suppressed is True
    assert m["runup_speed_mps"].suppressed is True
    assert m["stride_length_pct_stature"].suppressed is True
    assert m["follow_through_drift_deg"].suppressed is True
    assert m["icc_elbow_legality_verdict"].observability == ObservabilityGrade.REFUSED

    # EXP-007 B17: High-view metric with sigma at ~88% of suppress threshold MUST be graded LOW (not MODERATE)!
    cal_side = calibrate_from_mcc_landmarks(lm_s, 1280, 720, K=K_s, is_panning_camera=False)
    kin_full = extract_fast_bowling_kinematics(j_world, events, fps=120.0, joint_uncertainties_m=np.full((len(j_world), 24), 0.115))
    m_hi = quantify_and_gate_metrics(kin_full, _dummy_audit(fps=120.0), cal_side, events=events, empirical_sigmas={"front_knee_br_deg": 10.2})
    assert m_hi["front_knee_br_deg"].suppressed is False
    assert m_hi["front_knee_br_deg"].observability == ObservabilityGrade.LOW

    # EXP-007 B8: Interval-aware cues NEVER fire when interval straddles threshold!
    m_straddle = dict(m_hi)
    m_straddle["front_knee_br_deg"] = m_hi["front_knee_br_deg"].model_copy(
        update={"value": 146.0, "ci_low": 136.0, "ci_high": 156.0, "observability": ObservabilityGrade.HIGH, "suppressed": False}
    )
    fs_str = build_fact_sheet("StraddleTest", BowlingArm.RIGHT, kin_full.action_class, "synthetic", _dummy_audit(120.0), cal_side, events, m_straddle)
    assert not any("Front-Knee Flexion" in c.title for c in fs_str.limiters)
    assert any("Front-Knee Yielding" in c.title for c in fs_str.inconclusive_observations)
    # Retired unvalidated 0-100 composites
    assert fs_str.kinetic_efficiency.score is None
    assert fs_str.lumbar_load_screening.score is None

    # When interval strictly clears (< 150), limiter DOES fire!
    m_clear = dict(m_hi)
    m_clear["front_knee_br_deg"] = m_hi["front_knee_br_deg"].model_copy(
        update={"value": 136.0, "ci_low": 127.0, "ci_high": 145.0, "observability": ObservabilityGrade.HIGH, "suppressed": False}
    )
    fs_clr = build_fact_sheet(
        "ClearTest", BowlingArm.RIGHT, kin_full.action_class, "synthetic", _dummy_audit(120.0), cal_side, events, m_clear,
        workload=WorkloadContext(age_years=16, bowls_per_week=210),
    )
    assert any("Front-Knee Flexion" in c.title for c in fs_clr.limiters)
    assert fs_clr.workload_guidance is not None and "U17" in fs_clr.workload_guidance.age_band


def test_b2_timebase_and_b14_pan_tilt_zoom_detection(tmp_path):
    import cv2

    rng = np.random.default_rng(7)
    h, w = 240, 320
    bg_wide = rng.integers(35, 215, size=(h + 240, w + 1600, 3), dtype=np.uint8)
    bg_wide = cv2.GaussianBlur(bg_wide, (3, 3), 0.8)

    # 1. Static camera with moving bright TV block -> False
    static_p = tmp_path / "static_tv.mp4"
    vw = cv2.VideoWriter(str(static_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(24):
        fr = bg_wide[:h, :w].copy()
        x0 = 210 + (i % 6) * 4
        fr[20:70, x0 : x0 + 40] = (230, 180, 90)
        vw.write(fr)
    vw.release()
    is_p, _ = detect_camera_panning(static_p)
    assert is_p is False

    # 2. Zoom-in 60% -> MUST detect (EXP-007 B14 S8!)
    zoom_p = tmp_path / "zoom.mp4"
    vw = cv2.VideoWriter(str(zoom_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    base = bg_wide[:h, :w]
    for i in range(30):
        sc = 1.0 + 0.60 * (i / 29.0)
        cw, ch = int(round(w / sc)), int(round(h / sc))
        x0, y0 = (w - cw) // 2, (h - ch) // 2
        vw.write(cv2.resize(base[y0 : y0 + ch, x0 : x0 + cw], (w, h)))
    vw.release()
    is_z, _ = detect_camera_panning(zoom_p)
    assert is_z is True

    # 3. Vertical tilt -> MUST detect (EXP-007 B14 S8!)
    tilt_p = tmp_path / "tilt.mp4"
    vw = cv2.VideoWriter(str(tilt_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(30):
        vw.write(bg_wide[i * 4 : i * 4 + h, :w])
    vw.release()
    is_t, _ = detect_camera_panning(tilt_p)
    assert is_t is True

    # 4. Long 300-frame steady horizontal pan -> MUST detect (EXP-007 B14 S8!)
    long_pan_p = tmp_path / "long_pan.mp4"
    vw = cv2.VideoWriter(str(long_pan_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    for i in range(300):
        vw.write(bg_wide[:h, i * 4 : i * 4 + w])
    vw.release()
    is_lp, _ = detect_camera_panning(long_pan_p)
    assert is_lp is True

    # 5. EXP-007 B2: Regular 1-in-6 pulldown video (25 in 30 fps) -> pulldown_deduped & ~25 fps!
    pull_p = tmp_path / "pulldown.mp4"
    vw = cv2.VideoWriter(str(pull_p), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (w, h))
    src_i = 0
    for i in range(60):
        if i > 0 and i % 6 == 5:
            pass  # repeat previous frame
        else:
            src_i += 1
        vw.write(bg_wide[:h, src_i * 5 : src_i * 5 + w])
    vw.release()
    aud_p = audit_video(pull_p)
    assert aud_p.timebase_status == "pulldown_deduped"
    assert aud_p.regular_repeat_cadence == 6
    assert 24.0 <= aud_p.fps <= 26.0
    assert len(aud_p.source_frame_indices) == aud_p.num_frames < 60

    # 6. Frozen/edited clip (longest run >= 8) -> edited_or_frozen & timebase_unreliable=True!
    frz_p = tmp_path / "frozen.mp4"
    vw = cv2.VideoWriter(str(frz_p), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (w, h))
    for i in range(40):
        shift = min(i, 20) * 5
        vw.write(bg_wide[:h, shift : shift + w])
    vw.release()
    aud_f = audit_video(frz_p)
    assert aud_f.timebase_status == "edited_or_frozen"
    assert aud_f.timebase_unreliable is True
    assert aud_f.longest_held_run_frames >= 8


def test_dual_fusion_arm_mismatch_repeatability_and_icc_caveat_preserved():
    p_a = SyntheticDeliveryParams(fps=240.0, front_knee_br_deg=170.0, thorax_lat_flex_br_deg=34.0)
    K_s, R_s, t_s, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    K_r, R_r, t_r, _ = make_virtual_camera(azimuth_deg=0.0, elevation_deg=6.0)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    cal_s = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_s, R_s, t_s, lm_keys), 1280, 720, K=K_s)
    cal_r = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_r, R_r, t_r, lm_keys), 1280, 720, K=K_r)
    aud = _dummy_audit(fps=240.0)

    def _run_fs(p: SyntheticDeliveryParams, cal, bid: str):
        j, _ = generate_synthetic_delivery(p)
        ev = detect_delivery_events(j, fps=p.fps, bowling_arm=p.bowler_arm)
        kin = extract_fast_bowling_kinematics(j, ev, fps=p.fps, bowling_arm=p.bowler_arm)
        return build_fact_sheet(bid, p.bowler_arm, kin.action_class, "synthetic", aud, cal, ev, quantify_and_gate_metrics(kin, aud, cal, ev))

    fs_s = _run_fs(p_a, cal_s, "BowlerA")
    fs_r = _run_fs(p_a, cal_r, "BowlerA")
    fused = fuse_dual_view_fact_sheets(fs_s, fs_r)
    assert fused.metrics["front_knee_br_deg"].suppressed is False
    assert fused.metrics["thorax_lat_flex_br_deg"].suppressed is False
    assert fused.metrics["front_knee_br_deg"].ci_high <= 180.0

    # Arm mismatch raises ValueError
    fs_l = _run_fs(SyntheticDeliveryParams(bowler_arm=BowlingArm.LEFT, fps=240.0), cal_r, "LeftA")
    with pytest.raises(ValueError, match="mismatched bowling arms"):
        fuse_dual_view_fact_sheets(fs_s, fs_l)

    # Multi-delivery repeatability works cleanly
    rep_p = evaluate_repeatability([fused, fused, fused])
    assert rep_p is not None and len(rep_p.contributing_metrics) >= 1

    # Permanent ICC refusal never truncated in CoachingReport or Markdown
    rep = generate_grounded_report(fs_r)
    assert any("ICC 15-Degree Elbow Extension Legality" in c for c in rep.capture_quality_caveats)
    assert "icc_elbow_legality_verdict" not in rep.cited_metric_keys
    md = render_report_markdown(fs_r, rep)
    assert "ICC 15-Degree Elbow Extension Legality" in md and "REFUSED" in md


@pytest.mark.parametrize("clip_name", ["steyn", "indoor", "mcgrath", "ambrose"])
def test_cached_real_clip_regression_fixtures(clip_name: str):
    """EXP-007 Roadmap Item 6: Fast CPU regression gate against cached real-video NLF 3D trajectories."""
    import json
    from pathlib import Path
    from bowling_biomech.schemas import PitchCalibration

    fix_path = Path(__file__).parent / "fixtures" / f"{clip_name}_nlf_fixture.npz"
    assert fix_path.exists(), f"Missing real-clip fixture: {fix_path}"
    z = np.load(fix_path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))

    jw = z["joints3d_world_m"].astype(np.float64)
    j_unc = z["joint_uncertainties_m"].astype(np.float64)
    j_in = z["joint_in_frame"].astype(bool)
    arm = BowlingArm(meta["bowling_arm"])
    fps = float(meta["fps"])
    cal = PitchCalibration.model_validate(meta["calibration"])

    audit = VideoAudit(
        video_path=f"fixture:{clip_name}",
        width=int(meta["width"]),
        height=int(meta["height"]),
        fps=fps,
        container_fps=float(meta["container_fps"]),
        num_frames=int(jw.shape[0]),
        raw_num_frames=int(jw.shape[0]),
        duration_s=round(jw.shape[0] / max(fps, 1.0), 3),
        median_laplacian_var=250.0,
        blur_grade="crisp",
        fps_grade="low_25" if fps < 45 else "good_120",
        est_arm_deg_per_frame=2300.0 / max(fps, 1.0),
        est_ball_m_per_frame=38.9 / max(fps, 1.0),
        timebase_status=str(meta["timebase_status"]),
        timebase_unreliable=bool(meta["timebase_unreliable"]),
        held_frame_ratio=float(meta["held_frame_ratio"]),
        longest_held_run_frames=int(meta["longest_held_run_frames"]),
    )

    ev = detect_delivery_events(jw, fps=fps, bowling_arm=arm, joint_in_frame=j_in)
    assert ev.is_complete_delivery is bool(meta["expected_complete"])

    kin = extract_fast_bowling_kinematics(
        jw, ev, fps=fps, bowling_arm=arm, joint_uncertainties_m=j_unc, joint_in_frame=j_in
    )
    metrics = quantify_and_gate_metrics(kin, audit, cal, events=ev)
    fs = build_fact_sheet(meta["bowler_id"], arm, kin.action_class, "fixture", audit, cal, ev, metrics)

    # Permanent ICC refusal invariant on every real clip
    assert fs.metrics["icc_elbow_legality_verdict"].observability == ObservabilityGrade.REFUSED
    assert fs.metrics["icc_elbow_legality_verdict"].suppressed is True
    # Retired unvalidated 0-100 composites must never reappear
    assert fs.kinetic_efficiency.score is None
    assert fs.lumbar_load_screening.score is None

    if clip_name in ("mcgrath", "ambrose"):
        # Truncated/edited archival Reel clips MUST suppress 100% of delivery metrics and cues (EXP-007 B1/B2)
        reported = [k for k, m in fs.metrics.items() if not m.suppressed]
        assert reported == [], f"{clip_name} leaked metrics despite incomplete delivery: {reported}"
        assert fs.strengths == [] and fs.limiters == []
    elif clip_name == "steyn":
        assert abs(ev.br.frame_float - float(meta["expected_br_frame"])) <= 1.0
        assert fs.metrics["thorax_lat_flex_br_deg"].suppressed is False
        assert 26.0 <= fs.metrics["thorax_lat_flex_br_deg"].value <= 35.0
        assert fs.metrics["follow_through_drift_deg"].suppressed is True
        assert fs.metrics["runup_speed_mps"].suppressed is True
    elif clip_name == "indoor":
        assert abs(ev.br.frame_float - float(meta["expected_br_frame"])) <= 1.5
        assert fs.metrics["front_knee_br_deg"].suppressed is True
        assert fs.metrics["follow_through_drift_deg"].suppressed is True
        assert fs.metrics["peak_arm_angular_vel_dps"].suppressed is True
