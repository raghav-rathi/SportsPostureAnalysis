"""Studio v5 Comprehensive Non-Circular Biomechanics, Guardrail & API Regression Suite.

Covers all 11 Studio v5 modules while enforcing AGENTS.md §1.2 non-negotiable guardrails:
  - Module 1: Proximal-to-distal kinematic sequence verdicts, sub-frame lags, & <90 fps / timebase gates
  - Module 2: Exact 14-segment de Leva (1996) COM partition-of-unity, translation equivariance, & pan gates
  - Module 3 & 4: Non-circular right-arm & left-arm chirality for head fall-away & lead-arm flare/drop
  - Module 5 & 6: Crease corridor, strict MCC Law 7 popping-crease gate, release tunnel, & tempo waterfall
  - Module 7–11: Stroboscopic compositor (real Steyn/Indoor + 240 fps spell), 3D archetypes, ACWR workload,
    spell bullseye & fatigue drift, CSV/JSON export licensing headers, Coach Copilot numeric verifier on
    v5 metrics, and takeaway capture-tip separation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
from fastapi.testclient import TestClient
import numpy as np
import pytest

from bowling_biomech.constants import JOINT_IDX
from bowling_biomech.schemas import (
    BiomechanicalFactSheet,
    BowlingArm,
    DeliveryEvents,
    EventTimestamp,
    ObservabilityGrade,
    PitchCalibration,
    VideoAudit,
)
from bowling_biomech.server.app import create_app
from bowling_biomech.stage1_pitch_calibration import calibrate_from_mcc_landmarks
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
from bowling_biomech.stage3_v5_metrics import (
    compute_3d_com_trajectory,
    extract_and_quantify_v5_biomechanics,
    get_de_leva_smpl24_weights,
)
from bowling_biomech.stage5_evaluator import hydrate_coaching_story
from bowling_biomech.stage6_coach_copilot import ask_grounded_coach, verify_coach_response
from bowling_biomech.synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
)
from bowling_biomech.workload_acwr import compute_bowler_workload_analytics


def _make_test_audit(fps: float = 240.0, n_frames: int = 240, timebase_unreliable: bool = False) -> VideoAudit:
    return VideoAudit(
        video_path="test:v5",
        width=1280,
        height=720,
        fps=fps,
        container_fps=fps,
        num_frames=n_frames,
        raw_num_frames=n_frames,
        duration_s=round(n_frames / fps, 3),
        median_laplacian_var=280.0,
        blur_grade="crisp",
        blur_roi_method="bowler_roi",
        fps_grade="high_speed_240" if fps >= 200 else ("slow_mo_120" if fps >= 100 else "standard_30"),
        est_arm_deg_per_frame=round(2300.0 / fps, 2),
        est_ball_m_per_frame=round(38.89 / fps, 3),
        timebase_status="edited_or_frozen" if timebase_unreliable else "verified_native",
        timebase_unreliable=timebase_unreliable,
        source_frame_indices=list(range(n_frames)),
    )


def _make_calibrated_view(az_deg: float = 90.0, el_deg: float = 6.0, is_panning: bool = False, method: str = "mcc_law7_pnp") -> PitchCalibration:
    lm_keys = [
        "bowling_crease_off",
        "bowling_crease_leg",
        "popping_crease_off",
        "popping_crease_leg",
        "stump_middle_base",
        "stump_middle_top",
    ]
    K, R, t, _ = make_virtual_camera(azimuth_deg=az_deg, elevation_deg=el_deg, distance_m=9.5, fov_deg=46.0)
    lm_px = project_mcc_landmarks(K, R, t, lm_keys, noise_px=0.2)
    cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)
    if is_panning or method != "mcc_law7_pnp":
        is_metric = (not is_panning) and (method in ("mcc_law7_pnp", "two_wicket_geometry"))
        cal = cal.model_copy(
            update={
                "method": method,  # type: ignore[arg-type]
                "is_panning_camera": is_panning,
                "is_metric_calibrated": is_metric,
                "is_intrinsics_calibrated": method in ("mcc_law7_pnp", "two_wicket_geometry"),
            }
        )
    return cal


def test_v5_de_leva_com_partition_of_unity_and_panning_gates() -> None:
    """Module 2: Exact partition-of-unity, translation equivariance, braking impulse, and pan gating."""
    for sex in ("male", "female"):
        alpha = get_de_leva_smpl24_weights(sex)
        assert alpha.shape == (24,)
        assert abs(float(np.sum(alpha)) - 1.0) < 1e-13
        core_mass = (
            alpha[JOINT_IDX["neck"]]
            + alpha[JOINT_IDX["lhip"]]
            + alpha[JOINT_IDX["rhip"]]
            + alpha[JOINT_IDX["lkne"]]
            + alpha[JOINT_IDX["rkne"]]
        )
        assert 0.60 <= float(core_mass) <= 0.90

    # Non-circular translation equivariance: COM(P + c) == COM(P) + c for arbitrary 3D vector c
    rng = np.random.default_rng(20261127)
    pts = rng.normal(loc=0.0, scale=1.2, size=(60, 24, 3))
    shift = np.array([3.45, -1.12, 0.87], dtype=np.float64)
    com_a = compute_3d_com_trajectory(pts, sex="male")
    com_b = compute_3d_com_trajectory(pts + shift[None, None, :], sex="male")
    np.testing.assert_allclose(com_b - com_a, np.broadcast_to(shift, com_a.shape), atol=1e-12)

    # Calibrated static side-on vs panning side-on gating test
    params = SyntheticDeliveryParams(bowler_arm=BowlingArm.RIGHT, fps=240.0, runup_speed_mps=6.0, stride_length_m=1.56)
    jw, _ = generate_synthetic_delivery(params)
    n = len(jw)
    audit = _make_test_audit(fps=240.0, n_frames=n)
    ev = detect_delivery_events(jw, fps=240.0, bowling_arm=BowlingArm.RIGHT)
    kin = extract_fast_bowling_kinematics(
        jw, ev, fps=240.0, bowling_arm=BowlingArm.RIGHT, joint_uncertainties_m=np.full((n, 24), 0.022)
    )

    cal_static = _make_calibrated_view(az_deg=90.0, is_panning=False)
    v5_static = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit, cal=cal_static, events=ev)
    assert not v5_static.com_braking.com_ffc_speed_mps.suppressed
    assert not v5_static.com_braking.com_braking_delta_vx_mps.suppressed
    assert (v5_static.com_braking.com_braking_delta_vx_mps.value or 0.0) > 0.8
    assert not v5_static.com_braking.front_leg_plant_angle_ffc_deg.suppressed
    assert 15.0 <= float(v5_static.com_braking.front_leg_plant_angle_ffc_deg.value or 0.0) <= 45.0

    # Under handheld panning (`is_panning_camera=True`), horizontal COM speeds MUST be suppressed,
    # while single-instant `front_leg_plant_angle_ffc_deg` and local-ground `com_vertical_sink_pct_stature` remain observable!
    cal_pan = _make_calibrated_view(az_deg=90.0, is_panning=True)
    v5_pan = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit, cal=cal_pan, events=ev)
    assert v5_pan.com_braking.com_ffc_speed_mps.suppressed
    assert v5_pan.com_braking.com_ffc_speed_mps.value is None
    assert v5_pan.com_braking.com_br_speed_mps.suppressed
    assert v5_pan.com_braking.com_braking_delta_vx_mps.suppressed
    assert v5_pan.com_braking.com_braking_efficiency_pct.suppressed
    assert not v5_pan.com_braking.front_leg_plant_angle_ffc_deg.suppressed
    assert not v5_pan.com_braking.com_vertical_sink_pct_stature.suppressed


def _smooth_sigmoid(x: np.ndarray) -> np.ndarray:
    """Smooth analytical sigmoid `tanh(1.20 * x)` on `[-1, 1]` with exact peak derivative at `x = 0`."""
    return np.tanh(1.20 * x)


@pytest.mark.parametrize("arm", [BowlingArm.RIGHT, BowlingArm.LEFT])
def test_v5_kinematic_sequence_verdicts_and_fps_timebase_gates(arm: BowlingArm) -> None:
    """Module 1: Right & left arm proximal-to-distal vs out-of-order verdicts, plus <90 fps & timebase gates."""
    params = SyntheticDeliveryParams(bowler_arm=arm, fps=240.0, runup_speed_mps=5.9)
    jw, _ = generate_synthetic_delivery(params)
    n = len(jw)
    audit_240 = _make_test_audit(fps=240.0, n_frames=n)
    # Mirror open-side 45 deg camera azimuth (+45 deg for right-arm, -45 deg for left-arm)
    az_open = 45.0 if arm == BowlingArm.RIGHT else -45.0
    cal_45 = _make_calibrated_view(az_deg=az_open, el_deg=8.0)

    ev = detect_delivery_events(jw, fps=240.0, bowling_arm=arm)
    kin = extract_fast_bowling_kinematics(
        jw, ev, fps=240.0, bowling_arm=arm, joint_uncertainties_m=np.full((n, 24), 0.020)
    )

    # 1. Inject non-circular analytical S-curve yaw ramps on `pelvis_yaw_deg` & `shoulder_yaw_deg`
    # Optimal sequence: pelvis yaw (~620 deg/s) peaks at (BR - 58 ms), thorax (~980 deg/s) at (BR - 28 ms), arm/wrist into BR!
    t_s = np.arange(n, dtype=np.float64) / 240.0
    t_br = ev.br.frame_float / 240.0
    kin.series["pelvis_yaw_deg"] = 210.0 + 19.0 * (1.0 + _smooth_sigmoid((t_s - (t_br - 0.058)) / 0.028))
    kin.series["shoulder_yaw_deg"] = 200.0 + 26.0 * (1.0 + _smooth_sigmoid((t_s - (t_br - 0.028)) / 0.024))

    v5_opt = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit_240, cal=cal_45, events=ev)
    ks_opt = v5_opt.kinematic_sequence
    assert ks_opt.verdict == "optimal_proximal_to_distal"
    assert not ks_opt.pelvis_to_thorax_lag_ms.suppressed
    assert 18.0 <= float(ks_opt.pelvis_to_thorax_lag_ms.value or 0.0) <= 42.0
    assert (ks_opt.thorax_to_arm_lag_ms.value or 0.0) > 8.0
    assert (ks_opt.whip_gain_arm_over_thorax.value or 0.0) > 1.5

    # 2. Out-of-order sequence: chest uncoils 42 ms BEFORE pelvis!
    kin.series["pelvis_yaw_deg"] = 210.0 + 19.0 * (1.0 + _smooth_sigmoid((t_s - (t_br - 0.018)) / 0.028))
    kin.series["shoulder_yaw_deg"] = 200.0 + 26.0 * (1.0 + _smooth_sigmoid((t_s - (t_br - 0.060)) / 0.024))
    v5_ooo = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit_240, cal=cal_45, events=ev)
    assert v5_ooo.kinematic_sequence.verdict == "out_of_order"
    assert float(v5_ooo.kinematic_sequence.pelvis_to_thorax_lag_ms.value or 0.0) < -25.0

    # 3. Unreliable timebase gate at 240 fps -> MUST suppress sequence completely
    audit_bad_tb = _make_test_audit(fps=240.0, n_frames=n, timebase_unreliable=True)
    v5_bad_tb = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit_bad_tb, cal=cal_45, events=ev)
    assert v5_bad_tb.kinematic_sequence.verdict == "unmeasurable"
    assert v5_bad_tb.kinematic_sequence.pelvis_to_thorax_lag_ms.suppressed
    assert v5_bad_tb.v5_metrics["seq_arm_peak_dps"].suppressed

    # 4. Low frame-rate gate (<90 fps -> 30 fps) -> MUST suppress sequence completely
    params_30 = SyntheticDeliveryParams(bowler_arm=arm, fps=30.0)
    jw_30, _ = generate_synthetic_delivery(params_30)
    ev_30 = detect_delivery_events(jw_30, fps=30.0, bowling_arm=arm)
    kin_30 = extract_fast_bowling_kinematics(
        jw_30, ev_30, fps=30.0, bowling_arm=arm, joint_uncertainties_m=np.full((len(jw_30), 24), 0.025)
    )
    v5_30 = extract_and_quantify_v5_biomechanics(
        kin=kin_30, audit=_make_test_audit(fps=30.0, n_frames=len(jw_30)), cal=cal_45, events=ev_30
    )
    assert v5_30.kinematic_sequence.verdict == "unmeasurable"
    assert v5_30.v5_metrics["seq_pelvis_peak_dps"].suppressed
    assert v5_30.v5_metrics["lead_arm_pulldown_peak_vel_dps"].suppressed


@pytest.mark.parametrize("arm", [BowlingArm.RIGHT, BowlingArm.LEFT])
def test_v5_non_circular_chirality_head_lead_arm_and_crease_gates(arm: BowlingArm) -> None:
    """Modules 3, 4, 5, 6: Non-circular 3D coordinate vector sign checks for right & left arm + Law 7 crease gate."""
    sign_arm = 1.0 if arm == BowlingArm.RIGHT else -1.0
    f_ank_name = "lank" if arm == BowlingArm.RIGHT else "rank"
    r_ank_name = "rank" if arm == BowlingArm.RIGHT else "lank"
    nb_elb_name = "lelb" if arm == BowlingArm.RIGHT else "relb"

    # Build an explicit hand-constructed 120-frame trajectory in Pitch World Frame (+X pitch, +Y up, +Z right)
    # WITHOUT relying on `generate_synthetic_delivery` for head offset, lead-elbow flare, or foot corridor!
    n = 120
    fps = 120.0
    jw = np.zeros((n, 24, 3), dtype=np.float64)
    for t_i in range(n):
        x_base = 0.04 * t_i - 2.0
        jw[t_i, JOINT_IDX["pelv"]] = [x_base, 0.95, 0.0]
        jw[t_i, JOINT_IDX["lhip"]] = [x_base, 0.92, -0.10]
        jw[t_i, JOINT_IDX["rhip"]] = [x_base, 0.92, +0.10]
        jw[t_i, JOINT_IDX["spi1"]] = [x_base, 1.12, 0.0]
        jw[t_i, JOINT_IDX["spi2"]] = [x_base, 1.28, 0.0]
        jw[t_i, JOINT_IDX["spi3"]] = [x_base, 1.40, 0.0]
        jw[t_i, JOINT_IDX["neck"]] = [x_base, 1.50, 0.0]
        jw[t_i, JOINT_IDX["head"]] = [x_base, 1.70, 0.0]
        jw[t_i, JOINT_IDX["lsho"]] = [x_base, 1.46, -0.18]
        jw[t_i, JOINT_IDX["rsho"]] = [x_base, 1.46, +0.18]
        jw[t_i, JOINT_IDX["lkne"]] = [x_base, 0.50, -0.10]
        jw[t_i, JOINT_IDX["rkne"]] = [x_base, 0.50, +0.10]
        jw[t_i, JOINT_IDX["lank"]] = [x_base, 0.08, -0.10]
        jw[t_i, JOINT_IDX["rank"]] = [x_base, 0.08, +0.10]
        jw[t_i, JOINT_IDX["ltoe"]] = [x_base + 0.14, 0.03, -0.10]
        jw[t_i, JOINT_IDX["rtoe"]] = [x_base + 0.14, 0.03, +0.10]
        b_s = 1.0 if arm == BowlingArm.RIGHT else -1.0
        b_elb = "relb" if arm == BowlingArm.RIGHT else "lelb"
        b_wri = "rwri" if arm == BowlingArm.RIGHT else "lwri"
        b_han = "rhan" if arm == BowlingArm.RIGHT else "lhan"
        nb_wri = "lwri" if arm == BowlingArm.RIGHT else "rwri"
        nb_han = "lhan" if arm == BowlingArm.RIGHT else "rhan"
        jw[t_i, JOINT_IDX[b_elb]] = [x_base, 1.74, b_s * 0.20]
        jw[t_i, JOINT_IDX[b_wri]] = [x_base, 1.98, b_s * 0.22]
        jw[t_i, JOINT_IDX[b_han]] = [x_base + 0.05, 2.04, b_s * 0.22]
        jw[t_i, JOINT_IDX[nb_elb_name]] = [x_base + 0.15, 1.46, -b_s * 0.18]
        jw[t_i, JOINT_IDX[nb_wri]] = [x_base + 0.28, 1.46, -b_s * 0.18]
        jw[t_i, JOINT_IDX[nb_han]] = [x_base + 0.34, 1.46, -b_s * 0.18]

    # Set exact analytical geometry across BFC=45, FFC=68, UAH=75, BR=81:
    # 1) Back foot at BFC=45 at Z = +sign_arm * 0.05 m; Front foot from FFC..BR at X = 1.12 m (10 cm behind 1.22 m crease!),
    #    Z = -sign_arm * 0.13 m -> exact open corridor = -sign_arm * (-0.13*s - 0.05*s) = +0.18 m = +18.0 cm ("train_tracks_open"!)
    jw[38:54, JOINT_IDX[r_ank_name], 2] = sign_arm * 0.05
    jw[62:92, JOINT_IDX[f_ank_name], 0] = 1.12
    jw[62:92, JOINT_IDX[f_ank_name], 2] = -sign_arm * 0.13

    # 2) At BR=81, lean head contralaterally (`-sign_arm * Z`) to Z = -sign_arm * 0.43 m
    #    -> offset past front ankle (-sign_arm * 0.13 m) is EXACTLY +0.30 m = +30.0 cm!
    jw[75:88, JOINT_IDX["head"], 2] = -sign_arm * 0.43

    # 3) At BR=81, flare lead elbow contralaterally outside ipsilateral ribcage (`-sign_arm * 0.14 m` -> Z_rib)
    jw[75:88, JOINT_IDX[nb_elb_name], 2] = -sign_arm * 0.36

    ev = DeliveryEvents(
        bfc=EventTimestamp(
            frame_idx=45, frame_float=45.0, timestamp_s=45.0 / fps, confidence=0.95, timing_uncertainty_ms=5.0, detection_method="analytical_test"
        ),
        ffc=EventTimestamp(
            frame_idx=68, frame_float=68.0, timestamp_s=68.0 / fps, confidence=0.95, timing_uncertainty_ms=5.0, detection_method="analytical_test"
        ),
        uah=EventTimestamp(
            frame_idx=75, frame_float=75.0, timestamp_s=75.0 / fps, confidence=0.92, timing_uncertainty_ms=5.0, detection_method="analytical_test"
        ),
        br=EventTimestamp(
            frame_idx=81, frame_float=81.0, timestamp_s=81.0 / fps, confidence=0.96, timing_uncertainty_ms=4.0, detection_method="analytical_test"
        ),
        ft_peak=EventTimestamp(
            frame_idx=96, frame_float=96.0, timestamp_s=96.0 / fps, confidence=0.90, timing_uncertainty_ms=8.0, detection_method="analytical_test"
        ),
        delivery_stride_duration_ms=round((68.0 - 45.0) * 1000.0 / fps, 2),
        ffc_to_br_duration_ms=round((81.0 - 68.0) * 1000.0 / fps, 2),
        is_complete_delivery=True,
    )
    kin = extract_fast_bowling_kinematics(
        jw, ev, fps=fps, bowling_arm=arm, joint_uncertainties_m=np.full((n, 24), 0.015)
    )
    audit = _make_test_audit(fps=fps, n_frames=n)

    # Rear-on view (0 deg) resolves coronal head fall-away, lead-elbow flare, and foot-plant corridor!
    cal_rear = _make_calibrated_view(az_deg=0.0, el_deg=6.0, method="mcc_law7_pnp")
    v5_rear = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit, cal=cal_rear, events=ev)

    assert not v5_rear.head_stability.head_lateral_offset_from_ankle_br_cm.suppressed
    assert abs(float(v5_rear.head_stability.head_lateral_offset_from_ankle_br_cm.value or 0.0) - 30.0) < 3.0
    assert float(v5_rear.head_stability.head_coronal_angle_br_deg.value or 0.0) > 20.0
    assert not v5_rear.lead_arm.lead_elbow_ribcage_flare_br_pct_stature.suppressed
    assert float(v5_rear.lead_arm.lead_elbow_ribcage_flare_br_pct_stature.value or 0.0) > 8.0
    assert v5_rear.crease_tunnel.corridor_verdict == "train_tracks_open"
    assert abs(float(v5_rear.crease_tunnel.foot_plant_corridor_offset_cm.value or 0.0) - 18.0) < 3.0
    # Rear view MUST suppress sagittal `front_heel_crease_margin_cm` (`sin_side < 0.50`)!
    assert v5_rear.crease_tunnel.front_heel_crease_margin_cm.suppressed

    # Side-on MCC Law 7 view (90 deg) MUST report `front_heel_crease_margin_cm` (~ +14.5 cm with -4.5 cm ankle->heel)
    # while `two_wicket_geometry` MUST suppress `front_heel_crease_margin_cm`!
    cal_side_mcc = _make_calibrated_view(az_deg=90.0, el_deg=6.0, method="mcc_law7_pnp")
    v5_side_mcc = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit, cal=cal_side_mcc, events=ev)
    assert not v5_side_mcc.crease_tunnel.front_heel_crease_margin_cm.suppressed
    assert 10.0 <= float(v5_side_mcc.crease_tunnel.front_heel_crease_margin_cm.value or 0.0) <= 19.0

    cal_side_tw = _make_calibrated_view(az_deg=90.0, el_deg=6.0, method="two_wicket_geometry")
    v5_side_tw = extract_and_quantify_v5_biomechanics(kin=kin, audit=audit, cal=cal_side_tw, events=ev)
    assert v5_side_tw.crease_tunnel.front_heel_crease_margin_cm.suppressed
    assert v5_side_tw.crease_tunnel.front_heel_crease_margin_cm.value is None

    # Tempo waterfall sanity check: BFC->FFC = 23 frames = 191.7 ms, FFC->BR = 13 frames = 108.3 ms -> ratio = 1.77
    assert abs(float(v5_side_mcc.tempo_rhythm.tempo_stride_duration_ms.value or 0.0) - 191.67) < 1.0
    assert abs(float(v5_side_mcc.tempo_rhythm.tempo_braking_to_br_ms.value or 0.0) - 108.33) < 1.0
    assert abs(float(v5_side_mcc.tempo_rhythm.tempo_gather_to_whip_ratio.value or 0.0) - (23.0 / 13.0)) < 0.03
    assert len(v5_side_mcc.tempo_rhythm.waterfall_phases) >= 3


def test_v5_studio_endpoints_strobe_archetypes_spell_workload_and_verifier(tmp_path: Path) -> None:
    """End-to-end integration test across Modules 7–11, TypeScript contract shapes, and Guardrail #1-#4."""
    app = create_app(data_root=tmp_path / "studio_v5_data", seed_demo=True)
    client = TestClient(app)

    # 1. Overview: Dual Real Videos (`del_real_steyn` + `del_real_indoor`) + `del_demo_dual_240` + Snapshots
    r_ov = client.get("/api/overview")
    assert r_ov.status_code == 200
    ov = r_ov.json()
    sample_ids = [s["delivery_id"] for s in ov["samples"]]
    assert sample_ids[0] == "del_real_steyn"
    assert sample_ids[1] == "del_real_indoor"
    assert sample_ids[2] == "del_demo_dual_240"
    assert ov["workload_snapshot"] is not None
    assert ov["workload_snapshot"]["balls_7d"] >= 100
    assert ov["workload_snapshot"]["acwr_zone"] == "sweet_spot"
    assert ov["spell_snapshot"] is not None
    assert ov["spell_snapshot"]["completed_count"] >= 4

    # Verify Dual-Phone 240 fps Showcase (`del_demo_dual_240`) unlocks ALL 4 kinematic sequence lanes & 22/22 checks!
    r_dual = client.get("/api/deliveries/del_demo_dual_240")
    assert r_dual.status_code == 200
    dual_payload = r_dual.json()
    dual_fs = BiomechanicalFactSheet.model_validate(dual_payload["fact_sheet"])
    assert dual_fs.kinematic_sequence is not None
    assert dual_fs.kinematic_sequence.verdict == "optimal_proximal_to_distal"
    assert len(dual_fs.kinematic_sequence.segments) == 4
    for seg in dual_fs.kinematic_sequence.segments:
        assert not seg.peak_dps.suppressed, f"Dual-fused segment {seg.segment} unexpectedly suppressed!"
        assert seg.peak_time_rel_ffc_ms is not None
    assert dual_payload["story"]["capture"]["measured_count"] == 22
    assert set(dual_payload["telemetry"]["sequence_series"].keys()) >= {
        "pelvis_omega_dps",
        "thorax_omega_dps",
        "arm_omega_dps",
        "wrist_whip_dps",
    }

    # Verify >=120 fps borderline (`watch`) copy fix: NEVER tells a 240 fps clip to "film in crisp slow-mo"!
    from bowling_biomech.stage5_evaluator import _build_checkpoint_plain_meaning

    msg_240 = _build_checkpoint_plain_meaning("front_leg_brace", "watch", "158°", "152°–164°", fps=240.0, timebase_unreliable=False)
    assert "slow-mo" not in msg_240.lower() and "2–3 more deliveries" in msg_240
    msg_30 = _build_checkpoint_plain_meaning("front_leg_brace", "watch", "158°", "146°–170°", fps=30.0, timebase_unreliable=False)
    assert "slow-mo" in msg_30.lower()

    # Verify Real Indoor 120 fps delivery (`ZpBNq8cIRoY.mp4` + `indoor_nlf_fixture.npz`)
    r_ind = client.get("/api/deliveries/del_real_indoor")
    assert r_ind.status_code == 200
    ind_data = r_ind.json()
    ind_fs = BiomechanicalFactSheet.model_validate(ind_data["fact_sheet"])
    assert ind_fs.metrics["icc_elbow_legality_verdict"].observability == ObservabilityGrade.REFUSED
    assert ind_fs.metrics["icc_elbow_legality_verdict"].suppressed is True
    # Indoor has unverified container timebase & cropped front ankle -> velocity/tempo & ankle-referenced metrics MUST be suppressed!
    assert ind_fs.v5_metrics["seq_arm_peak_dps"].suppressed is True
    assert ind_fs.v5_metrics["tempo_stride_duration_ms"].suppressed is True
    assert ind_fs.v5_metrics["head_lateral_offset_from_ankle_br_pct_stature"].suppressed is True

    # 2. Telemetry v2.1 (`com_world_m`, `com_2d_px`, `trails["2d_px" | "3d_m"]`, `sequence_series`, `isb_anatomy`)
    r_det_b1 = client.get("/api/deliveries/del_demo_side_240")
    assert r_det_b1.status_code == 200
    tel = r_det_b1.json()["telemetry"]
    assert tel["schema_version"] == "2.0"
    assert tel["telemetry_minor_version"] == "2.1"
    assert len(tel["com_world_m"]) == tel["num_frames"]
    assert len(tel["com_2d_px"]) == tel["num_frames"]
    assert "2d_px" in tel["trails"] and "3d_m" in tel["trails"]
    assert "bowling_wrist" in tel["trails"]["3d_m"] and "com" in tel["trails"]["3d_m"]
    assert "pelvis_omega_dps" in tel["sequence_series"]
    assert tel["isb_anatomy"] is not None

    # 3. Module 7: Stroboscopic JPEG across Side-On Demo, Real Steyn, and Real Indoor
    for did, exp_modes in (
        ("del_demo_side_240", {"studio_mannequin_strobe", "panoramic_composite"}),
        ("del_real_steyn", {"coronal_sequence_strip"}),
        ("del_real_indoor", {"coronal_sequence_strip"}),
    ):
        r_st = client.get(f"/api/deliveries/{did}/stroboscopic.jpg?w=960&trails=true")
        assert r_st.status_code == 200
        assert r_st.headers["content-type"] == "image/jpeg"
        assert r_st.headers["x-stroboscopic-mode"] in exp_modes
        img_arr = cv2.imdecode(np.frombuffer(r_st.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        assert img_arr is not None and img_arr.shape[0] > 240 and img_arr.shape[1] >= 480

    # 4. Module 8: Pro Archetype 3D Reference Library (Right & Left Arm + Glazier 2010 Disclaimer)
    for arm_q in ("right", "left"):
        r_arch = client.get(f"/api/archetypes?arm={arm_q}")
        assert r_arch.status_code == 200
        archs = r_arch.json()["archetypes"]
        assert [a["id"] for a in archs] == [
            "side_on_express",
            "semi_open_all_round",
            "front_on_power",
            "high_arm_bounce",
        ]
        for a in archs:
            assert a["bowling_arm"] == arm_q
            assert "Glazier" in a["disclaimer"] and "template" in a["disclaimer"].lower()
            env = a["metric_envelopes"]["front_knee_br_deg"]
            assert "low" in env and "high" in env and env["low"] < env["high"]
            assert len(a["joints3d_world_m"]) == a["num_frames"]

    # 5. Module 9: ACWR Workload & Governing-Body Directives + Low-Chronic Gate + Soreness Referral
    r_wl = client.get("/api/bowlers/bwl_demo/workload")
    assert r_wl.status_code == 200
    wl = r_wl.json()
    assert len(wl["daily_series_28d"]) == 28
    assert isinstance(wl["directive_checks"], list) and len(wl["directive_checks"]) >= 3
    assert "Lolli" in wl["medical_disclaimer"] and "not" in wl["medical_disclaimer"].lower()
    assert wl["acwr_summary"]["acwr_ewma"] is not None
    assert 0.80 <= float(wl["acwr_summary"]["acwr_ewma"]) <= 1.30

    # Log a soreness=4 entry -> must trigger `lumbar_stiffness_logged` welfare referral alert, then delete cleanly
    r_add = client.post(
        "/api/bowlers/bwl_demo/workload",
        json={
            "log_date": wl["daily_series_28d"][-1]["date"],
            "deliveries_count": 24,
            "session_type": "net",
            "rpe": 7,
            "max_spell_balls": 18,
            "soreness_lumbar": 4,
            "pain_when_bowling": True,
        },
    )
    assert r_add.status_code == 200
    added_body = r_add.json()
    alert_kinds = {a["kind"] for a in added_body["alerts"]}
    assert "lumbar_stiffness_logged" in alert_kinds
    new_log_id = added_body["entry"]["id"]
    r_del = client.delete(f"/api/bowlers/bwl_demo/workload/{new_log_id}")
    assert r_del.status_code == 200 and r_del.json()["deleted"] is True

    # Verify `<30 balls/wk` low-chronic-base guardrail suppresses ACWR division explosion
    low_base_res = compute_bowler_workload_analytics(
        {"id": "b_new", "age_band": "u15"},
        [{"log_date": wl["daily_series_28d"][-1]["date"], "deliveries_count": 48, "session_type": "match", "rpe": 7}],
    )
    assert low_base_res["acwr_summary"]["zone"] == "low_chronic_base"
    assert low_base_res["acwr_summary"]["acwr_coupled"] is None

    # 6. Module 10: Spell Analytics (World Stump Release Tunnel + Ellipse95 + Verified Ball 1->4 Knee Fatigue Drift)
    r_sp = client.get("/api/bowlers/bwl_demo/spell-analytics")
    assert r_sp.status_code == 200
    sp = r_sp.json()
    assert sp["release_tunnel"]["mode"] == "world_stump_m"
    assert len(sp["release_tunnel"]["points"]) >= 4
    pt0 = sp["release_tunnel"]["points"][0]
    assert "semi_major_m" in pt0["ellipse_95"] and "semi_minor_m" in pt0["ellipse_95"]
    fk_drift = next(d for d in sp["fatigue_drift"] if d["metric_key"] == "front_knee_br_deg")
    assert fk_drift["status"] == "verified_drift"
    assert fk_drift["slope_per_ball"] < -2.0

    # 7. Compare Extended Rows + Takeaway Capture-Tip Bugfix + CSV/JSON Export Licensing
    r_cmp = client.get("/api/compare?a=del_demo_side_240&b=del_demo_side_b4")
    assert r_cmp.status_code == 200
    cmp_json = r_cmp.json()
    assert len(cmp_json["rows"]) == 5
    assert len(cmp_json["extended_rows"]) >= 4

    for fmt, ctype in (("csv", "text/csv"), ("json", "application/json")):
        r_exp = client.get(f"/api/deliveries/del_demo_side_240/export?format={fmt}")
        assert r_exp.status_code == 200
        assert ctype in r_exp.headers["content-type"]
        txt = r_exp.text
        assert "CC-BY-NC-4.0" in txt and "REFUSED" in txt

    # 8. Deterministic Coach Copilot Numeric Verifier + Strength Takeaway Cleanliness across ALL seeded balls!
    deliv_payload = client.get("/api/deliveries").json()
    all_cards = deliv_payload.get("items") or deliv_payload.get("deliveries") or ov["recent_deliveries"]
    assert len(all_cards) >= 7
    for card in all_cards:
        did = card["id"]
        det = client.get(f"/api/deliveries/{did}").json()
        fs = BiomechanicalFactSheet.model_validate(det["fact_sheet"])
        story = hydrate_coaching_story(fs, delivery_id=did)
        if story.top_takeaway.kind in ("strength", "all_clear") and story.capture.next_clip_tip:
            assert story.top_takeaway.what_to_try != story.capture.next_clip_tip
        for ins in story.v5_insights:
            v_ok, v_viols = verify_coach_response(ins.plain_summary, fs)
            assert v_ok, f"Unverified number in v5_insight {ins.id} on {did}: {v_viols}"
        ans = ask_grounded_coach("Summarize my kinematic sequence, center-of-mass braking, and head stability.", fs)
        assert ans.verification_passed, f"Copilot failed on {did}: {ans.blocked_violations}"

    # Adversarial hallucination of suppressed `front_heel_crease_margin_cm` on rear-on Dale Steyn MUST be blocked!
    steyn_fs = BiomechanicalFactSheet.model_validate(client.get("/api/deliveries/del_real_steyn").json()["fact_sheet"])
    adv_ok, adv_viols = verify_coach_response("Your front-heel popping-crease margin was +8.4 cm behind the line.", steyn_fs)
    assert not adv_ok and len(adv_viols) > 0
