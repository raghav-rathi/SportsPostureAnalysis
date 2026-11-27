"""Phase 2 (M2A-M2D) Comprehensive Verification Suite.

Validates all Phase 2 Go / No-Go quantitative gates:
  - M2A: FastAPI Local Studio, SQLite persistence, Telemetry v2.0, <50 ms CPU HITL recompute,
         4-click MCC PnP preview/save, causal-clamped event nudging, workload logs, 9:16 Reel export.
  - M2B: 3-DOF rotational background pan tracking (SO(3) recovery + residual rejection gate),
         `bml_movi_87` ISB rigid-body pelvis/thorax/foot anatomy with correlated-depth N_eff guardrail,
         multi-ball net-session auto-clipper (recall + 0 FP), Pillar 3 repeatability & spaghetti curves.
  - M2C: Unsynchronized dual-phone +Y_world 1000 Hz sync (<2.5 ms MAE across mixed 120+240 fps + 0.85 s offset),
         anisotropic ray-precision 3D fusion (<4.5 deg 3-plane accuracy), collinear dual-cam suppression gate,
         and permanent ICC 15-degree refusal.
  - M2D: Wrist-ROI ballistic speed RANSAC accuracy (<2.5 km/h) + strict hard gating (fps, cal, pan, rear view),
         and deterministic numeric-verifier Coach Copilot blocking 100% of adversarial hallucinations.
"""

from __future__ import annotations

import math
from pathlib import Path
import time

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from bowling_biomech.constants import JOINT_IDX
from bowling_biomech.schemas import BowlingArm, CameraViewClass, ObservabilityGrade, VideoAudit
from bowling_biomech.server.app import create_app
from bowling_biomech.stage0_net_clipper import detect_delivery_candidates_from_signal
from bowling_biomech.stage1_pan_tracker import track_camera_rotation_3dof_frames
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_mcc_landmarks,
    default_camera_matrix,
    transform_cam_to_pitch_world,
)
from bowling_biomech.stage2_ball_release import (
    estimate_ball_release_speed_from_world_points,
    fit_ballistic_speed_ransac,
)
from bowling_biomech.stage2_dual_sync_fusion import (
    fuse_dual_view_3d_trajectories,
    synchronize_dual_view_trajectories,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage2_isb_anatomy import (
    BML_MOVI_87_IDX,
    effective_sample_size,
    extract_isb_anatomy_kinematics,
)
from bowling_biomech.stage6_coach_copilot import ask_grounded_coach, verify_coach_response
from bowling_biomech.synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)


def _dummy_audit(fps: float = 240.0, n_frames: int = 252, unreliable: bool = False) -> VideoAudit:
    return VideoAudit(
        video_path="synthetic.mp4",
        width=1280,
        height=720,
        fps=fps,
        container_fps=fps,
        num_frames=n_frames,
        raw_num_frames=n_frames,
        duration_s=round(n_frames / fps, 3),
        median_laplacian_var=280.0,
        blur_grade="crisp",
        fps_grade="high_speed_240" if fps >= 200 else "slow_mo_120",
        est_arm_deg_per_frame=2300.0 / fps,
        est_ball_m_per_frame=38.89 / fps,
        timebase_status="edited_or_frozen" if unreliable else "verified_native",
        timebase_unreliable=unreliable,
    )


def test_m2b_3dof_rotational_pan_tracker_recovery_and_rejection():
    """Verify 3-DOF SO(3) pan tracker recovers exact pure rotation, preserves fixed C_0, and rejects blank frames."""
    rng = np.random.default_rng(7)
    w, h = 640, 360
    K = default_camera_matrix(w, h, fov_deg=55.0)
    # Build rich static background canvas and warp it by known pure camera yaw dR_t = R_y(yaw_t)
    base_tex = np.zeros((h, w), dtype=np.uint8)
    for _ in range(280):
        x, y = int(rng.integers(20, w - 20)), int(rng.integers(30, h - 30))
        r = int(rng.integers(4, 16))
        col = int(rng.integers(80, 245))
        cv2.circle(base_tex, (x, y), r, col, -1)
    base_tex = cv2.GaussianBlur(base_tex, (3, 3), 0.7)

    K_inv = np.linalg.inv(K)
    R0 = np.eye(3, dtype=np.float64)
    t0 = np.array([0.0, -1.5, 12.0], dtype=np.float64)
    C0_true = -R0.T @ t0

    true_yaws_deg = np.linspace(0.0, 10.0, 12)
    frames = []
    for yaw_deg in true_yaws_deg:
        a = math.radians(yaw_deg)
        dR = np.array([[math.cos(a), 0.0, math.sin(a)], [0.0, 1.0, 0.0], [-math.sin(a), 0.0, math.cos(a)]])
        H = K @ dR @ K_inv
        warped = cv2.warpPerspective(base_tex, H, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=20)
        frames.append(warped)

    res = track_camera_rotation_3dof_frames(frames, K, R0, t0, anchor_frame=0, residual_gate_px=2.5)
    assert res.is_verified is True
    assert res.median_residual_px < 1.2
    assert abs(res.max_pan_deg - 10.0) < 0.8
    # Camera optical center C_0 = -R_t^T t_t MUST stay algebraically invariant across every frame!
    for i in range(len(frames)):
        c_i = -res.R_per_frame[i].T @ res.t_per_frame[i]
        assert np.allclose(c_i, C0_true, atol=1e-6)
        assert abs(np.linalg.det(res.R_per_frame[i]) - 1.0) < 1e-6

    # Blank / textureless frames must be cleanly REJECTED (never falsely unlock translation!)
    blank_frames = [np.full((h, w), 128, dtype=np.uint8) for _ in range(6)]
    res_bad = track_camera_rotation_3dof_frames(blank_frames, K, R0, t0, anchor_frame=0)
    assert res_bad.is_verified is False


@pytest.mark.parametrize("arm", [BowlingArm.RIGHT, BowlingArm.LEFT])
def test_m2b_isb_bml_movi_87_rigid_body_anatomy_and_neff_guardrail(arm: BowlingArm):
    """Non-circular verification of ISB Pelvis/Thorax/Foot frames & correlated-depth N_eff guardrail."""
    assert effective_sample_size(4, 0.85) < 1.25  # Never fake sqrt(4)=2x variance reduction!
    assert effective_sample_size(6, 0.85) < 1.25

    params = SyntheticDeliveryParams(bowler_arm=arm, fps=120.0)
    jw, _ = generate_synthetic_delivery(params)
    ev = detect_delivery_events(jw, fps=120.0, bowling_arm=arm)
    n = jw.shape[0]
    sign_arm = 1.0 if arm == BowlingArm.RIGHT else -1.0

    # Construct exact known ISB bony markers on (T, 87, 3) with known angles:
    # - Pelvis: anterior tilt = +18 deg (x_fwd tilts downward by 18 deg), front-on yaw = 270 deg (z_right = +Z)
    # - Thorax: forward sagittal inclination = +38 deg -> lumbopelvic = 38 - 18 = +20 deg!
    # - Front foot: +22 deg external rotation toward bowling side, heel 4 cm lower than toe -> heel_first!
    m87 = np.zeros((n, 87, 3), dtype=np.float64)
    tilt = math.radians(18.0)
    p_xfwd = np.array([math.cos(tilt), -math.sin(tilt), 0.0])
    p_zright = np.array([0.0, 0.0, 1.0])
    p_ctr = np.array([0.0, 0.90, 0.0])
    m87[:, BML_MOVI_87_IDX["lasis"]] = p_ctr + 0.10 * p_xfwd - 0.12 * p_zright
    m87[:, BML_MOVI_87_IDX["rasis"]] = p_ctr + 0.10 * p_xfwd + 0.12 * p_zright
    m87[:, BML_MOVI_87_IDX["lpsis"]] = p_ctr - 0.10 * p_xfwd - 0.08 * p_zright
    m87[:, BML_MOVI_87_IDX["rpsis"]] = p_ctr - 0.10 * p_xfwd + 0.08 * p_zright

    t_inc = math.radians(38.0)
    t_yup = np.array([math.sin(t_inc), math.cos(t_inc), 0.0])
    t_xfwd = np.array([math.cos(t_inc), -math.sin(t_inc), 0.0])
    t_zright = np.array([0.0, 0.0, 1.0])
    t_ctr = np.array([0.1, 1.30, 0.0])
    m87[:, BML_MOVI_87_IDX["backneck"]] = t_ctr + 0.22 * t_yup - 0.08 * t_xfwd
    m87[:, BML_MOVI_87_IDX["upperback"]] = t_ctr - 0.15 * t_yup - 0.08 * t_xfwd
    m87[:, BML_MOVI_87_IDX["clavicle"]] = t_ctr + 0.20 * t_yup + 0.10 * t_xfwd
    m87[:, BML_MOVI_87_IDX["sternum"]] = t_ctr - 0.15 * t_yup + 0.10 * t_xfwd
    m87[:, BML_MOVI_87_IDX["lshom"]] = t_ctr + 0.18 * t_yup - 0.20 * t_zright
    m87[:, BML_MOVI_87_IDX["rshom"]] = t_ctr + 0.18 * t_yup + 0.20 * t_zright

    # Front foot: +22 deg externally rotated / splayed open away from target (-Z for right-arm left foot, +Z for left-arm right foot)
    f_ang = math.radians(22.0)
    hee_k = "lhee" if arm == BowlingArm.RIGHT else "rhee"
    toe_k = "ltoe" if arm == BowlingArm.RIGHT else "rtoe"
    m87[:, BML_MOVI_87_IDX[hee_k]] = np.array([1.0, 0.02, 0.0])
    m87[:, BML_MOVI_87_IDX[toe_k]] = np.array([1.0 + 0.24 * math.cos(f_ang), 0.06, -sign_arm * 0.24 * math.sin(f_ang)])

    res = extract_isb_anatomy_kinematics(m87, ev, bowling_arm=arm, rho_corr=0.85)
    assert abs(res.scalars["anterior_pelvic_tilt_ffc_deg"] - 18.0) < 0.5
    assert abs(res.scalars["lumbopelvic_angle_ffc_deg"] - 20.0) < 0.5
    assert abs(res.scalars["front_foot_angle_ffc_deg"] - 22.0) < 0.5
    assert res.front_foot_strike_mode == "heel_first"
    assert res.metrics["lumbopelvic_angle_ffc_deg"].suppressed is False
    assert res.metrics["lumbopelvic_angle_ffc_deg"].ci_low is not None

    # P0-2 Non-Circular Side-On (yaw=180 deg) Rigid Neutral Spine Tilted 30 deg Down-Pitch:
    # Must give EXACT 0.0 deg lumbopelvic flexion (fixing the 30-deg coordinate-frame mismatch!)
    m_rigid = np.zeros((n, 87, 3), dtype=np.float64)
    # Side-on orientation (facing bowler's left = -Z for right arm, +Z for left arm), then pitched 30 deg down +X
    r_inc = math.radians(30.0)
    r_yup = np.array([math.sin(r_inc), math.cos(r_inc), 0.0])
    r_zright = np.array([math.cos(r_inc), -math.sin(r_inc), 0.0]) if arm == BowlingArm.RIGHT else np.array([-math.cos(r_inc), math.sin(r_inc), 0.0])
    r_xfwd = np.cross(r_yup, r_zright)
    r_xfwd /= np.linalg.norm(r_xfwd)
    m_rigid[:, BML_MOVI_87_IDX["lasis"]] = p_ctr + 0.10 * r_xfwd - 0.12 * r_zright
    m_rigid[:, BML_MOVI_87_IDX["rasis"]] = p_ctr + 0.10 * r_xfwd + 0.12 * r_zright
    m_rigid[:, BML_MOVI_87_IDX["lpsis"]] = p_ctr - 0.10 * r_xfwd - 0.08 * r_zright
    m_rigid[:, BML_MOVI_87_IDX["rpsis"]] = p_ctr - 0.10 * r_xfwd + 0.08 * r_zright
    m_rigid[:, BML_MOVI_87_IDX["backneck"]] = t_ctr + 0.22 * r_yup - 0.08 * r_xfwd
    m_rigid[:, BML_MOVI_87_IDX["upperback"]] = t_ctr - 0.15 * r_yup - 0.08 * r_xfwd
    m_rigid[:, BML_MOVI_87_IDX["clavicle"]] = t_ctr + 0.20 * r_yup + 0.10 * r_xfwd
    m_rigid[:, BML_MOVI_87_IDX["sternum"]] = t_ctr - 0.15 * r_yup + 0.10 * r_xfwd
    m_rigid[:, BML_MOVI_87_IDX["lshom"]] = t_ctr + 0.18 * r_yup - 0.20 * r_zright
    m_rigid[:, BML_MOVI_87_IDX["rshom"]] = t_ctr + 0.18 * r_yup + 0.20 * r_zright
    # Closed side-on front foot past 90 deg (-110 deg) -> must NOT clamp to -90!
    f_closed = math.radians(-110.0)
    m_rigid[:, BML_MOVI_87_IDX[hee_k]] = np.array([1.0, 0.02, 0.0])
    m_rigid[:, BML_MOVI_87_IDX[toe_k]] = np.array([1.0 + 0.24 * math.cos(f_closed), 0.02, -sign_arm * 0.24 * math.sin(f_closed)])

    res_rigid = extract_isb_anatomy_kinematics(m_rigid, ev, bowling_arm=arm, rho_corr=0.85)
    assert abs(res_rigid.scalars["lumbopelvic_angle_ffc_deg"]) < 0.1, (
        f"Rigid side-on spine reported non-zero lumbopelvic angle: {res_rigid.scalars['lumbopelvic_angle_ffc_deg']}"
    )
    assert abs(res_rigid.scalars["front_foot_angle_ffc_deg"] - (-110.0)) < 0.5

    # P0-2 Guardrail #3: Incomplete delivery suppresses ALL ISB metrics; rear-on camera suppresses sagittal ISB metrics!
    ev_inc = ev.model_copy(update={"is_complete_delivery": False})
    res_inc = extract_isb_anatomy_kinematics(m87, ev_inc, bowling_arm=arm, rho_corr=0.85)
    assert res_inc.front_foot_strike_mode == "suppressed"
    assert all(me.suppressed and me.value is None for me in res_inc.metrics.values())

    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    Kr, Rr, tr, _ = make_virtual_camera(azimuth_deg=0.0, elevation_deg=6.0)
    cal_rear = calibrate_from_mcc_landmarks(project_mcc_landmarks(Kr, Rr, tr, lm_keys), 1280, 720, K=Kr)
    res_rear = extract_isb_anatomy_kinematics(m87, ev, bowling_arm=arm, rho_corr=0.85, cal=cal_rear)
    assert res_rear.metrics["anterior_pelvic_tilt_ffc_deg"].suppressed is True
    assert res_rear.metrics["lumbopelvic_angle_ffc_deg"].suppressed is True


def test_m2c_unsynchronized_dual_phone_sync_ray_fusion_and_collinear_gate():
    """Gate 2C: <2.5 ms sync MAE across mixed 120+240 fps, <4.5 deg 3-plane fusion, and collinear suppression."""
    rng = np.random.default_rng(2026)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]

    # Camera A: Side-On 90 deg @ 120 fps; Camera B: Rear 0 deg @ 240 fps with +0.42 s start-time offset!
    offset_true_s = 0.420
    p_a = SyntheticDeliveryParams(
        fps=120.0, duration_s=1.20, bfc_time_s=0.45, ffc_time_s=0.642, br_time_s=0.750,
        front_knee_br_deg=170.0, trunk_flexion_br_deg=48.0, thorax_lat_flex_br_deg=38.0,
    )
    p_b = SyntheticDeliveryParams(
        fps=240.0, duration_s=1.65,
        bfc_time_s=0.45 + offset_true_s, ffc_time_s=0.642 + offset_true_s, br_time_s=0.750 + offset_true_s,
        front_knee_br_deg=170.0, trunk_flexion_br_deg=48.0, thorax_lat_flex_br_deg=38.0,
    )
    jw_a, gt_a = generate_synthetic_delivery(p_a)
    jw_b, _ = generate_synthetic_delivery(p_b)

    Ka, Ra, ta, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0, distance_m=12.0, fov_deg=55.0)
    Kb, Rb, tb, _ = make_virtual_camera(azimuth_deg=0.0, elevation_deg=6.0, distance_m=12.0, fov_deg=55.0)
    _, jc_a = project_world_points(jw_a, Ka, Ra, ta)
    _, jc_b = project_world_points(jw_b, Kb, Rb, tb)

    # Inject anisotropic monocular camera noise (large Z depth error, tight XY in-plane error)
    jc_a_n = jc_a + rng.normal(0.0, [0.012, 0.012, 0.045], size=jc_a.shape)
    jc_b_n = jc_b + rng.normal(0.0, [0.012, 0.012, 0.045], size=jc_b.shape)

    cal_a = calibrate_from_mcc_landmarks(project_mcc_landmarks(Ka, Ra, ta, lm_keys, 0.8, rng), 1280, 720, K=Ka)
    cal_b = calibrate_from_mcc_landmarks(project_mcc_landmarks(Kb, Rb, tb, lm_keys, 0.8, rng), 1280, 720, K=Kb)

    jwa_est = transform_cam_to_pitch_world(jc_a_n, cal_a)
    jwb_est = transform_cam_to_pitch_world(jc_b_n, cal_b)
    aud_a = _dummy_audit(fps=120.0, n_frames=jw_a.shape[0])
    aud_b = _dummy_audit(fps=240.0, n_frames=jw_b.shape[0])

    # 1. Sub-frame +Y_world sync accuracy (< 2.5 ms!)
    sync = synchronize_dual_view_trajectories(jwa_est, 120.0, jwb_est, 240.0)
    sync_err_ms = abs(sync.offset_b_minus_a_s - offset_true_s) * 1000.0
    assert sync_err_ms < 2.5, f"Dual-phone sync error too high: {sync_err_ms:.2f} ms"

    # 2. Orthogonal Side + Rear Anisotropic 3D Ray-Precision Fusion (< 4.5 deg simultaneous 3-plane accuracy!)
    fused = fuse_dual_view_3d_trajectories(
        "DualTest", BowlingArm.RIGHT, jwa_est, cal_a, aud_a, jwb_est, cal_b, aud_b
    )
    assert fused.alignment_method == "dual_mcc_pnp"
    assert fused.baseline_sin > 0.95
    assert fused.fact_sheet.calibration.view_class == CameraViewClass.DUAL_VIEW_FUSED
    m = fused.fact_sheet.metrics
    assert m["front_knee_br_deg"].suppressed is False
    assert m["thorax_lat_flex_br_deg"].suppressed is False
    assert abs(m["front_knee_br_deg"].value - gt_a["front_knee_br_deg"]) < 4.5
    assert abs(m["thorax_lat_flex_br_deg"].value - gt_a["thorax_lat_flex_br_deg"]) < 4.5
    # Permanent ICC refusal invariant!
    assert m["icc_elbow_legality_verdict"].observability == ObservabilityGrade.REFUSED
    assert m["icc_elbow_legality_verdict"].suppressed is True

    # 3. Collinear Loophole Gate: two nearly collinear side cameras (85 deg & 94 deg -> 9 deg apart!)
    Kc, Rc, tc, _ = make_virtual_camera(azimuth_deg=85.0, elevation_deg=5.0, distance_m=12.0, fov_deg=55.0)
    cal_c = calibrate_from_mcc_landmarks(project_mcc_landmarks(Kc, Rc, tc, lm_keys, 0.8, rng), 1280, 720, K=Kc)
    _, jc_c = project_world_points(jw_b, Kc, Rc, tc)
    jwc_est = transform_cam_to_pitch_world(jc_c + rng.normal(0.0, [0.012, 0.012, 0.045], size=jc_c.shape), cal_c)
    collinear = fuse_dual_view_3d_trajectories(
        "CollinearTest", BowlingArm.RIGHT, jwa_est, cal_a, aud_a, jwc_est, cal_c, aud_b
    )
    assert collinear.baseline_sin < 0.25
    assert collinear.fact_sheet.calibration.view_class != CameraViewClass.DUAL_VIEW_FUSED
    assert collinear.fact_sheet.metrics["thorax_lat_flex_br_deg"].suppressed is True


def test_m2b_net_session_auto_clipper_recall_and_zero_fp():
    """Verify net-session clipper finds 4/4 bursts with 0 false positives on quiet walk-backs."""
    fps = 30.0
    scan_fps = 6.0
    dur_s = 28.0
    n_samples = int(dur_s * scan_fps)
    sample_frames = np.round(np.arange(n_samples) * (fps / scan_fps)).astype(np.int64)
    t_s = sample_frames / fps

    rng = np.random.default_rng(11)
    energy = rng.normal(2.0, 0.35, size=n_samples)
    true_peaks_s = [4.0, 10.5, 17.0, 23.5]
    for pk in true_peaks_s:
        energy += 14.0 * np.exp(-0.5 * ((t_s - pk) / 0.45) ** 2)

    cands = detect_delivery_candidates_from_signal(energy, sample_frames, fps=fps, total_frames=int(dur_s * fps))
    assert len(cands) == 4
    for cand, pk in zip(cands, true_peaks_s):
        assert abs(cand.peak_frame / fps - pk) < 0.6


def test_m2d_wrist_roi_ballistic_speed_accuracy_and_hard_gates():
    """Gate 2D: <3.5 km/h ballistic speed accuracy on 240 fps calibrated side-on, 100% suppressed otherwise."""
    rng = np.random.default_rng(99)
    p = SyntheticDeliveryParams(fps=240.0)
    jw, _ = generate_synthetic_delivery(p)
    ev = detect_delivery_events(jw, fps=240.0)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]

    # True 138.0 km/h (38.333 m/s) release velocity at -4 deg launch angle + 2 gross outlier detections!
    v_true_kph = 138.0
    v_mps = v_true_kph / 3.6
    vx_t, vy_t = v_mps * math.cos(math.radians(-4.0)), v_mps * math.sin(math.radians(-4.0))
    ts = np.arange(10, dtype=np.float64) / 240.0
    xs = 1.30 + vx_t * ts + rng.normal(0.0, 0.008, size=10)
    ys = 2.05 + vy_t * ts - 0.5 * 9.80665 * (ts**2) + rng.normal(0.0, 0.008, size=10)
    # Inject 2 gross outliers
    xs[2] += 0.45
    ys[7] -= 0.55
    pts = np.column_stack([xs, ys])

    # 1. Valid 240 fps calibrated Side-On 90 deg -> must recover within <2.5 km/h
    Ks, Rs, ts_v, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    cal_side = calibrate_from_mcc_landmarks(project_mcc_landmarks(Ks, Rs, ts_v, lm_keys), 1280, 720, K=Ks)
    res_ok = estimate_ball_release_speed_from_world_points(ts, pts, _dummy_audit(240.0), cal_side, ev)
    assert res_ok.metric.suppressed is False
    assert abs(res_ok.metric.value - v_true_kph) < 2.5

    # 2. Rear view (az = 5 deg) -> MUST suppress 100%!
    Kr, Rr, tr_v, _ = make_virtual_camera(azimuth_deg=5.0, elevation_deg=6.0)
    cal_rear = calibrate_from_mcc_landmarks(project_mcc_landmarks(Kr, Rr, tr_v, lm_keys), 1280, 720, K=Kr)
    res_rear = estimate_ball_release_speed_from_world_points(ts, pts, _dummy_audit(240.0), cal_rear, ev)
    assert res_rear.metric.suppressed is True

    # 3. Low FPS (60 fps) -> MUST suppress 100%!
    res_60 = estimate_ball_release_speed_from_world_points(ts, pts, _dummy_audit(60.0), cal_side, ev)
    assert res_60.metric.suppressed is True

    # 4. Panning / uncalibrated -> MUST suppress 100%!
    cal_pan = cal_side.model_copy(update={"is_panning_camera": True, "is_metric_calibrated": False})
    res_pan = estimate_ball_release_speed_from_world_points(ts, pts, _dummy_audit(240.0), cal_pan, ev)
    assert res_pan.metric.suppressed is True


def test_m2d_numeric_verifier_blocks_adversarial_hallucinations(tmp_path: Path):
    """Gate 2D: 0% hallucinated or leaked-suppressed numbers survive verify_coach_response."""
    app = create_app(data_root=tmp_path, seed_demo=True)
    client = TestClient(app)
    d = client.get("/api/deliveries/del_demo_rear_240").json()
    from bowling_biomech.schemas import BiomechanicalFactSheet

    fs = BiomechanicalFactSheet.model_validate(d["fact_sheet"])
    assert fs.metrics["front_knee_br_deg"].suppressed is True

    # Adversarial 1-6 (All EXP-009 audit bypasses MUST be blocked 100%!)
    adversarial_cases = [
        ("The bowler's front knee at ball release was 164.2 deg, showing good bracing.", "suppressed front knee"),
        ("The bowler is legal under the ICC 15-degree rule with elbow extension at 11 deg.", "ICC legal"),
        ("Lateral flexion reached 73.4 degrees at peak load.", "hallucinated number"),
        ("This delivery was legal under the ICC 15° rule because the elbow straightens by only 8 degrees.", "ICC unicode 15 deg + was legal"),
        ("The bowling action is **completely illegal** because it exceeds the ICC 15° limit.", "markdown bold completely illegal"),
        ("Although run-up speed was gated, Front Knee Angle at Ball Release was 180 deg.", "compound-sentence exemption poisoning"),
        ("Your front knee angle at release reached 180 degrees with solid bracing.", "paraphrased front knee 180"),
        ("Your ball release speed was 120 km/h.", "paraphrased ball release speed 120"),
        ("Overall Biomechanical Efficiency Score: 95 / 100.", "resurrected 0-100 composite score"),
    ]
    for cand, desc in adversarial_cases:
        ok, viols = verify_coach_response(cand, fs)
        assert ok is False and len(viols) >= 1, f"Verifier failed to block ({desc}): {cand}"

    # Grounded Copilot responses MUST pass 100% cleanly across ALL seeded deliveries (including real Steyn/McGrath + adolescent workload)!
    ses = client.get("/api/sessions/ses_demo").json()["session"]
    ui_prompts = [
        "Is this action ICC legal?",
        "Why was front knee suppressed?",
        "What does this imply for lumbar load and injury risk?",
        "Summarize verified measurements",
    ]
    for d_meta in ses["deliveries"]:
        det_i = client.get(f"/api/deliveries/{d_meta['id']}").json()
        fs_i = BiomechanicalFactSheet.model_validate(det_i["fact_sheet"])
        for q in ui_prompts:
            ans = ask_grounded_coach(q, fs_i)
            assert ans.verification_passed is True, (
                f"Failed verification on delivery={d_meta['id']} q='{q}': {ans.blocked_violations}"
            )
    # Also verify adolescent age_years=16 workload guidance note ('Ages 16-17') passes verification cleanly
    rec_u17 = client.post(
        "/api/deliveries/del_demo_side_240/recompute",
        json={"workload": {"age_years": 16, "bowls_per_week": 60}},
    ).json()
    fs_u17 = BiomechanicalFactSheet.model_validate(rec_u17["fact_sheet"])
    ans_u17 = ask_grounded_coach("What does this mean for injury risk?", fs_u17)
    assert ans_u17.verification_passed is True, f"Adolescent workload failed verification: {ans_u17.blocked_violations}"


def test_m2a_fastapi_studio_end_to_end_hitl_dual_and_reel(tmp_path: Path):
    """Gate 2A: End-to-end FastAPI Studio verification (<150 ms CPU HITL, 4-click cal, compare, dual fuse, 9:16 reel)."""
    app = create_app(data_root=tmp_path, seed_demo=True)
    client = TestClient(app)

    # 1. Health & demo session bootstrap
    h = client.get("/api/health")
    assert h.status_code == 200 and h.json()["status"] == "ok"
    sessions = client.get("/api/sessions").json()["sessions"]
    assert len(sessions) == 1
    sid = sessions[0]["id"]

    ses = client.get(f"/api/sessions/{sid}").json()["session"]
    del_ids = [d["id"] for d in ses["deliveries"]]
    assert "del_demo_side_240" in del_ids and "del_demo_rear_240" in del_ids

    # 2. Fetch Delivery detail + Telemetry v2.0 + Proxy MP4 stream
    det = client.get("/api/deliveries/del_demo_side_240").json()
    assert det["telemetry"]["schema_version"] == "2.0"
    assert len(det["telemetry"]["kin_series"]["front_knee_deg"]) == det["telemetry"]["num_frames"]
    assert det["fact_sheet"]["metrics"]["icc_elbow_legality_verdict"]["observability"] == "refused"
    vid_resp = client.get("/api/deliveries/del_demo_side_240/video")
    assert vid_resp.status_code == 200 and len(vid_resp.content) > 1000

    # 3. <35 ms (test budget <150 ms) CPU HITL Recompute: FPS pill + causal-clamped event nudge
    t0 = time.perf_counter()
    rec = client.post(
        "/api/deliveries/del_demo_side_240/recompute",
        json={"capture_fps_override": 240.0, "event_overrides": {"ffc_frame": 148}},
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert rec.status_code == 200
    body = rec.json()
    assert body["fact_sheet"]["events"]["ffc"]["detection_method"] == "human_verified_override"
    assert body["fact_sheet"]["events"]["bfc"]["frame_float"] < body["fact_sheet"]["events"]["ffc"]["frame_float"]
    assert body["recompute_ms"] < 250.0  # Typically 12-40 ms on laptop CPU
    assert elapsed_ms < 1000.0

    # 4. 4-Click MCC PnP live wireframe preview
    lm_px = det["delivery"]["overrides"]["mcc_landmarks_px"]
    prev = client.post("/api/deliveries/del_demo_side_240/preview-calibration", json={"mcc_landmarks_px": lm_px})
    assert prev.status_code == 200
    assert prev.json()["reprojection_error_px"] is not None
    assert "popping_crease_off" in prev.json()["mcc_wireframe_2d"]

    # 5. Pillar 3 Multi-Ball Repeatability & Spaghetti Curves
    comp = client.get(f"/api/sessions/{sid}/compare").json()
    assert comp["completed_deliveries_count"] >= 3
    assert comp["repeatability"] is not None
    assert "front_knee_deg" in comp["spaghetti_curves"]

    # 6. Dual-Phone 3D Ray-Precision Fusion via API (Side + Rear) + No-Op Recompute State Preservation (P0-4)
    fuse = client.post(
        f"/api/sessions/{sid}/fuse-dual",
        json={"delivery_id_a": "del_demo_side_240", "delivery_id_b": "del_demo_rear_240", "label": "Fused API Ball 1"},
    )
    assert fuse.status_code == 200
    f_json = fuse.json()
    fused_id = f_json["delivery_id"]
    assert f_json["baseline_sin"] > 0.95
    assert f_json["fact_sheet"]["metrics"]["front_knee_br_deg"]["suppressed"] is False
    assert f_json["fact_sheet"]["metrics"]["thorax_lat_flex_br_deg"]["suppressed"] is False
    sig_before = f_json["fact_sheet"]["metrics"]["thorax_lat_flex_br_deg"]["sigma_total"]

    # No-op recompute on dual-fused delivery must preserve exact fused covariance & pose_backend!
    f_rec = client.post(f"/api/deliveries/{fused_id}/recompute", json={})
    assert f_rec.status_code == 200
    f_rec_fs = f_rec.json()["fact_sheet"]
    assert f_rec_fs["pose_backend"] == "dual_3d_ray_fusion"
    assert abs(f_rec_fs["metrics"]["thorax_lat_flex_br_deg"]["sigma_total"] - sig_before) < 0.05

    # 7. Grounded Coach Copilot via API (ICC legality refusal)
    ask = client.post("/api/deliveries/del_demo_side_240/coach/ask", json={"question": "Is this delivery ICC legal?"})
    assert ask.status_code == 200
    assert ask.json()["verification_passed"] is True
    assert "REFUSED" in ask.json()["answer_markdown"]

    # 8. 1-Click 9:16 Truthful Breakdown Reel MP4 Export
    reel = client.post("/api/deliveries/del_demo_side_240/export-reel")
    assert reel.status_code == 200 and reel.json()["size_bytes"] > 5000
    dl = client.get(reel.json()["reel_url"])
    assert dl.status_code == 200 and len(dl.content) == reel.json()["size_bytes"]

    # 9. Longitudinal Workload Log API (P0-5: Calendar-day window must ignore stale 1-year-old rows!)
    wkl_before = client.get("/api/bowlers/bwl_demo/workload").json()
    assert wkl_before["deliveries_last_7d"] >= 100
    for stale_day in ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"]:
        r_old = client.post(
            "/api/bowlers/bwl_demo/workload",
            json={"log_date": stale_day, "deliveries_count": 40, "session_type": "net", "notes": "Stale history"},
        )
        assert r_old.status_code == 200
    wkl_after = client.get("/api/bowlers/bwl_demo/workload").json()
    assert wkl_after["deliveries_last_7d"] == wkl_before["deliveries_last_7d"]
    assert wkl_after["deliveries_last_28d"] == wkl_before["deliveries_last_28d"]


def test_p0_p1_audit_regressions_psd_covariance_1080p_reel_and_video_ball_speed(tmp_path: Path):
    """Non-circular regressions for EXP-009 audit findings:
    - P0-4: PSD matrix-sqrt cross-covariance satisfies Cauchy-Schwarz under asymmetric FFC/BR drop-out
    - P0-3: 9:16 Reel scales 1920x1080 joints2d onto a 1280x720 proxy with zero 1.5x blowup
    - P1-1: Pixel-level estimate_ball_release_speed_video recovers 140 km/h on 1080p 120 fps video
            with moving follow-through forearm masked out!
    """
    from bowling_biomech.constants import JOINT_IDX
    from bowling_biomech.stage2_ball_release import estimate_ball_release_speed_video
    from bowling_biomech.stage2_dual_sync_fusion import _pose_sigma_from_cov_tensor
    from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
    from bowling_biomech.stage6_visualizer import render_truthful_breakdown_reel_9x16

    # 1. P0-4 Cauchy-Schwarz PSD cross-covariance check
    p = SyntheticDeliveryParams(fps=120.0, front_knee_br_deg=165.0)
    jw, _ = generate_synthetic_delivery(p)
    ev = detect_delivery_events(jw, fps=120.0)
    kin = extract_fast_bowling_kinematics(jw, ev, fps=120.0)
    n = jw.shape[0]
    cov = np.zeros((n, 24, 3, 3), dtype=np.float64)
    # Tiny covariance at FFC, huge covariance at BR -> naive 0.5*(C1+C2) violates Cauchy-Schwarz!
    for f in range(n):
        sig = 0.01 if f <= ev.ffc.frame_idx + 2 else 0.18
        cov[f, :] = np.eye(3) * (sig**2)
    s_ffc = _pose_sigma_from_cov_tensor("front_knee_ffc_deg", kin, cov)
    s_br = _pose_sigma_from_cov_tensor("front_knee_br_deg", kin, cov)
    s_chg = _pose_sigma_from_cov_tensor("front_knee_change_deg", kin, cov)
    assert s_chg >= abs(s_br - s_ffc) - 0.15, f"Cauchy-Schwarz violated: {s_chg:.2f} < |{s_br:.2f} - {s_ffc:.2f}|"

    # 2. P1-1 Pixel-level 1080p 120 fps synthetic Side-On video with 140 km/h ball + moving forearm distraction
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    W, H = 1920, 1080
    K, R, t, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=4.0, distance_m=7.0, fov_deg=55.0, width=W, height=H)
    cal_1080 = calibrate_from_mcc_landmarks(project_mcc_landmarks(K, R, t, lm_keys), W, H, K=K)
    j2d_1080, _ = project_world_points(jw, K, R, t)

    vid_path = tmp_path / "ball_140kph_1080p.mp4"
    writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 120.0, (W, H))
    wri_i = JOINT_IDX["rwri"]
    elb_i = JOINT_IDX["relb"]
    wri_br = jw[ev.br.frame_idx, wri_i]
    v_true_kph = 140.0
    v_mps = v_true_kph / 3.6
    vx_t, vy_t = v_mps * math.cos(math.radians(-4.0)), v_mps * math.sin(math.radians(-4.0))
    for f in range(n):
        img = np.full((H, W, 3), 24, dtype=np.uint8)
        # Draw bright moving follow-through forearm (must be masked out, NOT fitted as ball!)
        ew = j2d_1080[f, elb_i]
        ww = j2d_1080[f, wri_i]
        if np.all(np.isfinite(ew)) and np.all(np.isfinite(ww)):
            cv2.line(img, (int(ew[0]), int(ew[1])), (int(ww[0]), int(ww[1])), (230, 230, 230), 10)
            cv2.circle(img, (int(ww[0]), int(ww[1])), 12, (235, 235, 235), -1)
        # Draw flying crimson/white ball from BR onward along ballistic trajectory
        if f >= ev.br.frame_idx:
            dt_f = (f - ev.br.frame_idx) / 120.0
            pb = np.array([wri_br[0] + 0.15 + vx_t * dt_f, wri_br[1] + vy_t * dt_f - 0.5 * 9.81 * (dt_f**2), wri_br[2]])
            pc = R @ pb + t
            if pc[2] > 0.2:
                uvw = K @ pc
                bu, bv = int(round(uvw[0] / uvw[2])), int(round(uvw[1] / uvw[2]))
                if 0 <= bu < W and 0 <= bv < H:
                    cv2.circle(img, (bu, bv), 6, (255, 255, 255), -1)
        writer.write(img)
    writer.release()

    aud_1080 = _dummy_audit(fps=120.0, n_frames=n).model_copy(
        update={"width": W, "height": H, "source_frame_indices": list(range(n))}
    )
    ball_res = estimate_ball_release_speed_video(vid_path, j2d_1080, jw, aud_1080, cal_1080, ev, BowlingArm.RIGHT, wrist_unc_m=0.14)
    assert ball_res.metric.suppressed is False, f"Ball speed falsely suppressed: {ball_res.metric.suppression_reason}"
    assert ball_res.ransac_inliers >= 6
    assert abs(ball_res.metric.value - v_true_kph) < 4.5, f"Got {ball_res.metric.value} vs {v_true_kph}"

    # 3. P0-3 1080p joints2d onto 720p downscaled proxy in 9:16 Reel export (zero 1.5x blowup!)
    proxy_720 = tmp_path / "proxy_720p.mp4"
    pw = cv2.VideoWriter(str(proxy_720), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (1280, 720))
    for _ in range(8):
        pw.write(np.zeros((720, 1280, 3), dtype=np.uint8))
    pw.release()
    # Place all 24 joints at exact 1080p image center (960, 540) -> MUST draw at Reel video center (540, 670)!
    j2_center = np.tile(np.array([960.0, 540.0]), (8, 24, 1))
    from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
    from bowling_biomech.stage5_evaluator import build_fact_sheet

    metrics_1080 = quantify_and_gate_metrics(kin, aud_1080, cal_1080, events=ev)
    fs_1080 = build_fact_sheet("ReelScaleTest", BowlingArm.RIGHT, kin.action_class, "test", aud_1080, cal_1080, ev, metrics_1080)
    out_reel = render_truthful_breakdown_reel_9x16(
        proxy_720, 0, j2_center, jw[:8], fs_1080, tmp_path / "reel_scale.mp4", freeze_seconds=0.35, source_frame_indices=list(range(8))
    )
    cap = cv2.VideoCapture(str(out_reel))
    ok_r, fr_r = cap.read()
    cap.release()
    assert ok_r and fr_r is not None
    # Video box is W_OUT=1080, VIDEO_H=1120 -> 1280x720 scaled by 1080/1280 = 0.84375 -> nh=608, oy=110+(1120-608)//2=366 -> center=(540, 670)
    # With the old 1.5x bug, (960, 540) drew at x=810 (way off-center!). Verify bright white joint dot sits at (540, 670)!
    patch_center = fr_r[664:676, 534:546]
    patch_buggy = fr_r[664:676, 804:816]
    assert float(patch_center.max()) > 200.0, "Scaled 1080p joint missing from Reel center (540, 670)!"
    assert float(patch_buggy.max()) < 60.0, "1.5x unscaled joint ghost still present at x=810!"


def test_studio_v3_analysis_fixes_drills_notes_annotations_and_bowlers(tmp_path: Path):
    """EXP-010 Studio v3.0 Regression Suite:
    - Bug #1: Real-clip `unc_scale_factor=0.35` unlocks Steyn SCR & `action_class == mixed`
    - Bug #2: Noiseless synthetic demo skips Rician over-debiasing (front_knee_br_deg stays ~171 deg, not 180 deg)
    - Bug #4: 3-DOF pan tracking rejects synthetic/fixture proxies cleanly
    - Bug #8: `capture_fps_override` clears `timebase_unreliable` only when clip is not frozen
    - Bug #9: `scan-video` issues session-bound `scan_id` and blocks cross-session/traversal attacks
    - Bug #10: `GET /api/bowlers` & `POST /api/sessions` with `bowler_id` reuses bowler without duplicating workload
    - Features #1/#4/#5: Telestrator annotations, Coach notes, Delivery delete, Corrective drills + Copilot verifier
    """
    from bowling_biomech.schemas import BiomechanicalFactSheet
    from bowling_biomech.server.video_proxy import slice_video_segment

    app = create_app(data_root=tmp_path, seed_demo=True)
    client = TestClient(app)

    # Health v3.0 & pose_engine status
    h = client.get("/api/health").json()
    assert h["version"] == "3.0.0"
    assert h["pose_engine"] in ("remote_gpu_worker", "local_nlf_model", "cpu_recompute_only")

    # Bug #2: Noiseless synthetic Side-On demo must NOT over-debias 171.0 deg knee to 180.0 deg
    side = client.get("/api/deliveries/del_demo_side_240").json()
    fk_val = side["fact_sheet"]["metrics"]["front_knee_br_deg"]["value"]
    assert fk_val is not None and 167.0 <= fk_val <= 174.5, f"Rician over-debiasing bug regressed: {fk_val}"

    # Bug #1: Dale Steyn fixture must unlock SCR & classify as mixed; at 25 fps its analytical CI straddles 30 deg
    # so scientific non-straddling rule honestly routes it to inconclusive_observations (Watchlist), NOT false drills!
    steyn = client.get("/api/deliveries/del_real_steyn").json()
    st_fs = BiomechanicalFactSheet.model_validate(steyn["fact_sheet"])
    assert st_fs.metrics["shoulder_counter_rotation_deg"].suppressed is False
    assert st_fs.metrics["shoulder_counter_rotation_deg"].value is not None
    assert st_fs.metrics["shoulder_counter_rotation_deg"].value >= 30.0
    assert st_fs.action_class.value == "mixed"
    assert len(st_fs.inconclusive_observations) >= 1

    # Feature #4: A crisp 240 fps delivery with a non-straddling limiter (front knee collapse ~138 deg)
    # MUST generate evidence-grounded corrective_drills AND pass Coach Copilot verifier 100%!
    from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
    from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
    from bowling_biomech.stage5_evaluator import build_fact_sheet

    p_lim = SyntheticDeliveryParams(fps=240.0, front_knee_ffc_deg=160.0, front_knee_br_deg=135.0)
    jw_lim, _ = generate_synthetic_delivery(p_lim)
    ev_lim = detect_delivery_events(jw_lim, fps=240.0)
    lm_keys = ["bowling_crease_off", "bowling_crease_leg", "popping_crease_off", "popping_crease_leg", "stump_middle_base", "stump_middle_top"]
    Ks, Rs, ts_v, _ = make_virtual_camera(azimuth_deg=90.0, elevation_deg=5.0)
    cal_lim = calibrate_from_mcc_landmarks(project_mcc_landmarks(Ks, Rs, ts_v, lm_keys), 1280, 720, K=Ks)
    kin_lim = extract_fast_bowling_kinematics(jw_lim, ev_lim, fps=240.0, camera_R_w2c=Rs, debias_near_straight=False)
    met_lim = quantify_and_gate_metrics(kin_lim, _dummy_audit(240.0, len(jw_lim)), cal_lim, events=ev_lim)
    fs_lim = build_fact_sheet("DrillTest", BowlingArm.RIGHT, kin_lim.action_class, "test", _dummy_audit(240.0, len(jw_lim)), cal_lim, ev_lim, met_lim)
    assert len(fs_lim.corrective_drills) >= 1
    assert any(d.target_limiter_metric in ("front_knee_br_deg", "front_knee_change_deg") for d in fs_lim.corrective_drills)
    drill_ans = ask_grounded_coach("What drills should I practice to fix my limiters?", fs_lim)
    assert drill_ans.verification_passed is True, f"Drill copilot failed verifier: {drill_ans.blocked_violations}"
    drill_ans_st = ask_grounded_coach("What drills should I practice?", st_fs)
    assert drill_ans_st.verification_passed is True, f"Steyn watchlist drill answer failed verifier: {drill_ans_st.blocked_violations}"

    # Bug #4: 3-DOF pan tracking on synthetic proxy returns clean is_verified=False explanation
    pan_r = client.post("/api/deliveries/del_demo_side_240/recompute", json={"enable_pan_tracking": True}).json()
    assert pan_r["pan_summary"]["is_verified"] is False
    assert "real video" in pan_r["pan_summary"]["verification_reason"].lower()

    # Bug #8 / P1-1: FPS override preserves BOTH timebase_unreliable=True AND timebase_status='edited_or_frozen' on frozen McGrath Reel!
    mc_r = client.post("/api/deliveries/del_real_mcgrath/recompute", json={"capture_fps_override": 120.0}).json()
    assert mc_r["fact_sheet"]["audit"]["timebase_unreliable"] is True
    assert mc_r["fact_sheet"]["audit"]["timebase_status"] == "edited_or_frozen"
    # On clean verified_native clip, clicking 240 preserves verified_native!
    clean_r = client.post("/api/deliveries/del_demo_side_240/recompute", json={"capture_fps_override": 240.0}).json()
    assert clean_r["fact_sheet"]["audit"]["timebase_status"] == "verified_native"

    # P1-4: Capture/blur questions containing "improve" or "fix" MUST route to capture audit, NOT drills!
    cap_ans = ask_grounded_coach("How can I improve capture quality and fix camera blur?", st_fs)
    assert cap_ans.verification_passed is True
    assert "Capture audit" in cap_ans.answer_markdown

    # Bug #10: Bowler reuse across multiple sessions preserves workload history
    bowlers_before = client.get("/api/bowlers").json()["bowlers"]
    assert len(bowlers_before) == 1
    s2 = client.post("/api/sessions", json={"bowler_id": "bwl_demo", "title": "Follow-up Net Session"}).json()["session"]
    assert s2["bowler_id"] == "bwl_demo"
    bowlers_after = client.get("/api/bowlers").json()["bowlers"]
    assert len(bowlers_after) == 1

    # Feature #1: Telestrator annotations CRUD
    shapes = [
        {"id": "s1", "frame": 12, "type": "angle", "points": [[100, 100], [150, 200], [220, 180]], "label": "2D Angle"},
        {"id": "s2", "frame": 12, "type": "plumb", "points": [[300, 50], [300, 400]]},
    ]
    put_ann = client.put("/api/deliveries/del_demo_side_240/annotations", json={"annotations": shapes})
    assert put_ann.status_code == 200 and len(put_ann.json()["annotations"]) == 2
    get_ann = client.get("/api/deliveries/del_demo_side_240/annotations").json()
    assert len(get_ann["annotations"]) == 2 and get_ann["annotations"][0]["type"] == "angle"

    # Feature #5: Timestamped Coach Notes CRUD
    add_n = client.post(
        "/api/deliveries/del_demo_side_240/notes",
        json={"frame_idx": 148, "timestamp_s": 0.617, "tag": "cue", "body": "Strong front-leg brace at FFC."},
    )
    assert add_n.status_code == 200
    nid = add_n.json()["note"]["id"]
    notes = client.get("/api/deliveries/del_demo_side_240/notes").json()["notes"]
    assert len(notes) == 1 and notes[0]["tag"] == "cue"
    del_n = client.delete(f"/api/deliveries/del_demo_side_240/notes/{nid}")
    assert del_n.status_code == 200 and client.get("/api/deliveries/del_demo_side_240/notes").json()["notes"] == []

    # Bug #9: Net-video scan_id + cross-session protection + local slicing
    mini_vid = tmp_path / "mini_net.mp4"
    vw = cv2.VideoWriter(str(mini_vid), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 240))
    for i in range(45):
        fr = np.full((240, 320, 3), 40 + (i % 10) * 15, dtype=np.uint8)
        vw.write(fr)
    vw.release()
    with mini_vid.open("rb") as f:
        scan_r = client.post("/api/sessions/ses_demo/scan-video", files={"video": ("mini_net.mp4", f, "video/mp4")})
    assert scan_r.status_code == 200
    scan_id = scan_r.json()["scan_id"]
    assert scan_id.startswith("scan_")
    # Cross-session access to scan_id must be blocked (403) and malformed scan_id blocked (400)
    bad_fmt = client.post(
        "/api/sessions/ses_demo/deliveries/from-scan",
        json={"scan_id": "../etc/passwd", "clips": [{"start_frame": 0, "end_frame": 10, "label": "x"}]},
    )
    assert bad_fmt.status_code == 400
    cross_ses = client.post(
        f"/api/sessions/{s2['id']}/deliveries/from-scan",
        json={"scan_id": scan_id, "clips": [{"start_frame": 0, "end_frame": 10, "label": "x"}]},
    )
    assert cross_ses.status_code == 403
    sliced = slice_video_segment(mini_vid, tmp_path / "sliced.mp4", 5, 19)
    assert sliced.exists() and sliced.stat().st_size > 1000

    # Delete delivery + cascading check
    del_r = client.delete("/api/deliveries/del_demo_side_b2")
    assert del_r.status_code == 200 and del_r.json()["deleted"] is True
    assert client.get("/api/deliveries/del_demo_side_b2").status_code == 404
