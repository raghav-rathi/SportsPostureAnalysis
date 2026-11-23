"""EXP-007 independent-audit diagnostics on the synthetic benchmark (CPU only, deterministic).

Each experiment answers one question the stock benchmark does not ask:

  S1  per-metric CI coverage, bias and suppression for ALL reported metrics (the stock benchmark scores 7 of 22
      and reports one pooled coverage number)
  S2  event-timing accuracy when BFC->FFC and FFC->BR durations follow published values instead of the two fixed
      durations that are also hard-coded as priors inside the detector
  S3  generator artefacts: the foot-position jump at BFC and FFC, and the arm rotation after release
  S4  CI coverage when the true pose noise differs from the per-joint uncertainty handed to Stage 4
  S5  CI coverage under a world-frame error (yaw of +X, tilt of gravity) of the size self-calibration produces
  S6  self-calibration axis errors under an angled run-up and a forward sprint lean
  S7  MCC PnP with unknown focal length, and the effect of the pose model assuming a different focal length
  S8  unit-level reproductions (transverse unwrap straddle, pan detector blind spots)
  S9  decision reliability: how often the coaching cues, pillar labels and action class computed from noisy
      estimates match the ones computed from the true values

Run:  python validation/audit_synthetic_diagnostics.py --out outputs/exp007_synthetic_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_synthetic_benchmark import _ar1_correlated_noise  # noqa: E402

from bowling_biomech.schemas import BowlingArm, DeliveryEvents, EventTimestamp, VideoAudit  # noqa: E402
from bowling_biomech.stage0_audit import detect_camera_panning  # noqa: E402
from bowling_biomech.stage1_pitch_calibration import (  # noqa: E402
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events  # noqa: E402
from bowling_biomech.stage3_kinematics import (  # noqa: E402
    _transverse_alignment_deg,
    classify_action_portus,
    extract_fast_bowling_kinematics,
)
from bowling_biomech.stage4_uncertainty import METRIC_SPECS, quantify_and_gate_metrics  # noqa: E402
from bowling_biomech.stage5_evaluator import build_fact_sheet  # noqa: E402
from bowling_biomech.synthetic_ground_truth import (  # noqa: E402
    JOINT_IDX,
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)

VIEWPOINTS = [
    ("side_on_90", 90.0, 5.0),
    ("open_side_45", 45.0, 8.0),
    ("rear_0", 0.0, 6.0),
    ("broadcast_elevated", 10.0, 24.0),
]
FRAME_RATES = [25.0, 30.0, 60.0, 120.0, 240.0]
LM_KEYS = [
    "bowling_crease_off", "bowling_crease_leg",
    "popping_crease_off", "popping_crease_leg",
    "stump_middle_base", "stump_middle_top",
]
STOCK_EVAL_KEYS = [
    "front_knee_ffc_deg", "front_knee_br_deg", "trunk_flexion_delta_deg", "thorax_lat_flex_br_deg",
    "shoulder_counter_rotation_deg", "delayed_arm_ffc_deg", "runup_speed_mps",
]
UPPER_BODY = [
    "spi1", "spi2", "spi3", "neck", "head", "lcla", "rcla", "lsho", "rsho",
    "lelb", "relb", "lwri", "rwri", "lhan", "rhan",
]


def _audit(fps: float, n_frames: int, duration_s: float) -> VideoAudit:
    """Same VideoAudit the stock benchmark builds."""
    return VideoAudit(
        video_path="synthetic.mp4", width=1280, height=720, fps=fps, num_frames=n_frames, duration_s=duration_s,
        median_laplacian_var=220.0 if fps >= 120 else 90.0,
        blur_grade="crisp" if fps >= 120 else "moderate_blur",
        fps_grade="high_speed_240" if fps >= 200 else ("slow_mo_120" if fps >= 100 else "standard_30"),
        est_arm_deg_per_frame=2300.0 / fps, est_ball_m_per_frame=38.89 / fps,
    )


def _event(frame_float: float, fps: float) -> EventTimestamp:
    return EventTimestamp(
        frame_float=frame_float, frame_idx=int(round(frame_float)), timestamp_s=frame_float / fps,
        timing_uncertainty_ms=0.0, confidence=0.99, detection_method="ground_truth",
    )


def _true_events(gt: Dict[str, float], fps: float) -> DeliveryEvents:
    return DeliveryEvents(
        bfc=_event(gt["bfc_frame"], fps), ffc=_event(gt["ffc_frame"], fps), br=_event(gt["br_frame"], fps),
        delivery_stride_duration_ms=(gt["ffc_frame"] - gt["bfc_frame"]) / fps * 1000.0,
        ffc_to_br_duration_ms=(gt["br_frame"] - gt["ffc_frame"]) / fps * 1000.0,
    )


def _reference_scalars(params: SyntheticDeliveryParams) -> Dict[str, float]:
    """Definition-consistent ground truth: Stage 3 run on the clean 240 fps trajectory at the true event times.

    This is the most charitable reference available. It removes any disagreement about metric definitions and
    leaves only pose noise, calibration error and event-timing error in the measured error.
    """
    p240 = SyntheticDeliveryParams(**{**params.__dict__, "fps": 240.0})
    j_clean, gt = generate_synthetic_delivery(p240)
    kin = extract_fast_bowling_kinematics(j_clean, _true_events(gt, 240.0), 240.0, bowling_arm=params.bowler_arm)
    return kin.scalars


def _stock_noise(shape: tuple, fps: float, rng: np.random.Generator, scale: float = 1.0) -> Tuple[np.ndarray, float]:
    """Exactly the stock benchmark noise (camera-frame AR(1), tau = 80 ms), optionally scaled.

    Returns the noise and the per-joint uncertainty value the stock benchmark hands to Stage 4.
    """
    rho = float(np.exp(-1.0 / max(0.08 * fps, 1.0)))
    sigma_xy = 0.015 if fps >= 120 else 0.022
    sigma_z = 0.036 if fps >= 120 else 0.052
    noise = np.stack(
        [
            _ar1_correlated_noise(shape, sigma_xy * scale, rho, rng),
            _ar1_correlated_noise(shape, sigma_xy * scale, rho, rng),
            _ar1_correlated_noise(shape, sigma_z * scale, rho, rng),
        ],
        axis=-1,
    )
    return noise, float(np.hypot(sigma_xy, sigma_z * 0.55))


def _rot_y(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


class _Tally:
    """Per-key accumulator of error, coverage and suppression."""

    def __init__(self) -> None:
        self.err: Dict[str, List[float]] = {}
        self.cov: Dict[str, List[bool]] = {}
        self.half: Dict[str, List[float]] = {}
        self.sup: Dict[str, int] = {}
        self.n: Dict[str, int] = {}

    def add(self, key: str, metric, truth: float) -> None:
        self.n[key] = self.n.get(key, 0) + 1
        if metric.suppressed or metric.value is None:
            self.sup[key] = self.sup.get(key, 0) + 1
            return
        self.err.setdefault(key, []).append(metric.value - truth)
        self.cov.setdefault(key, []).append(bool(metric.ci_low <= truth <= metric.ci_high))
        self.half.setdefault(key, []).append(0.5 * (metric.ci_high - metric.ci_low))

    def table(self) -> Dict[str, Dict[str, Optional[float]]]:
        out = {}
        for key in self.n:
            e = np.asarray(self.err.get(key, []), dtype=np.float64)
            row: Dict[str, Optional[float]] = {
                "trials": self.n[key],
                "reported": int(e.size),
                "suppressed_pct": round(100.0 * self.sup.get(key, 0) / self.n[key], 1),
                "bias": None, "rmse": None, "median_ci_half_width": None,
                "rmse_over_reported_sigma": None, "ci95_coverage_pct": None,
            }
            if e.size:
                rmse = float(np.sqrt(np.mean(e**2)))
                half = float(np.median(self.half[key]))
                row.update(
                    bias=round(float(e.mean()), 2), rmse=round(rmse, 2), median_ci_half_width=round(half, 2),
                    # 1.0 means the stated sigma matches the realised error; below 1 the interval is too wide.
                    rmse_over_reported_sigma=round(rmse / max(half / 1.96, 1e-9), 2),
                    ci95_coverage_pct=round(100.0 * float(np.mean(self.cov[key])), 1),
                )
            out[key] = row
        return out

    def pooled_coverage(self, keys: Optional[List[str]] = None) -> Optional[float]:
        vals = [c for k, v in self.cov.items() if (keys is None or k in keys) for c in v]
        return round(100.0 * float(np.mean(vals)), 1) if vals else None


def _random_params(arm: BowlingArm, fps: float, rng: np.random.Generator, **extra) -> SyntheticDeliveryParams:
    """Same seven randomised parameters, in the same draw order, as the stock benchmark."""
    return SyntheticDeliveryParams(
        bowler_arm=arm, fps=fps,
        runup_speed_mps=float(rng.uniform(4.8, 6.4)),
        front_knee_ffc_deg=float(rng.uniform(146.0, 168.0)),
        front_knee_br_deg=float(rng.uniform(150.0, 175.0)),
        trunk_flexion_br_deg=float(rng.uniform(34.0, 50.0)),
        thorax_lat_flex_br_deg=float(rng.uniform(28.0, 52.0)),
        shoulder_counter_rot_deg=float(rng.uniform(10.0, 38.0)),
        delayed_arm_ffc_deg=float(rng.uniform(120.0, 158.0)),
        **extra,
    )


def _calibrated_trial(
    params: SyntheticDeliveryParams,
    cam: Tuple[np.ndarray, np.ndarray, np.ndarray],
    rng: np.random.Generator,
    noise_scale: float = 1.0,
    world_error: Optional[np.ndarray] = None,
) -> Dict:
    """One stock-benchmark trial (oracle K, 1 px landmark noise)."""
    K, R, t = cam
    fps = params.fps
    j_world_gt, gt = generate_synthetic_delivery(params)
    _, j_cam_clean = project_world_points(j_world_gt, K, R, t)
    noise, j_unc_val = _stock_noise(j_cam_clean.shape[:2], fps, rng, scale=noise_scale)
    j_unc = np.full(j_cam_clean.shape[:2], j_unc_val)
    lm_px = project_mcc_landmarks(K, R, t, LM_KEYS, noise_px=1.0, rng=rng)
    cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)
    j_world_est = transform_cam_to_pitch_world(j_cam_clean + noise, cal)
    if world_error is not None:
        j_world_est = j_world_est @ world_error.T
    events = detect_delivery_events(j_world_est, fps=fps, bowling_arm=params.bowler_arm)
    kin = extract_fast_bowling_kinematics(
        j_world_est, events, fps=fps, bowling_arm=params.bowler_arm, joint_uncertainties_m=j_unc
    )
    audit = _audit(fps, j_world_est.shape[0], params.duration_s)
    metrics = quantify_and_gate_metrics(kin, audit, cal, events=events)
    return {"metrics": metrics, "events": events, "gt": gt, "kin": kin, "cal": cal, "audit": audit}


def s1_per_metric_coverage(n_trials: int = 12) -> Dict:
    """Stock benchmark re-run with per-metric bookkeeping for every key in METRIC_SPECS."""
    rng = np.random.default_rng(2026)
    all_keys = list(METRIC_SPECS.keys())
    by_metric = _Tally()
    by_metric_view = {v[0]: _Tally() for v in VIEWPOINTS}
    stock = _Tally()
    for vname, az, el in VIEWPOINTS:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        for fps in FRAME_RATES:
            for trial in range(n_trials):
                arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                params = _random_params(arm, fps, rng)
                res = _calibrated_trial(params, (K, R, t), rng)
                ref = _reference_scalars(params)
                for key in all_keys:
                    by_metric.add(key, res["metrics"][key], ref[key])
                    by_metric_view[vname].add(key, res["metrics"][key], ref[key])
                for key in STOCK_EVAL_KEYS:
                    stock.add(key, res["metrics"][key], res["gt"][key])
    never_scored = [k for k in all_keys if k not in STOCK_EVAL_KEYS]
    return {
        "stock_pooled_coverage_pct_param_truth": stock.pooled_coverage(),
        "pooled_coverage_pct_all_metrics_reference_truth": by_metric.pooled_coverage(),
        "metrics_never_scored_by_stock_benchmark": never_scored,
        "per_metric": by_metric.table(),
        "per_metric_by_view": {v: tl.table() for v, tl in by_metric_view.items()},
    }


def s2_event_timing_priors(n_trials: int = 60) -> Dict:
    """Event errors with the generator's fixed durations versus durations drawn from Felton et al. (2019).

    Clips are cut 150 ms after release. The generator's bowling arm keeps circling after release (see S3), and
    with a shorter FFC->BR phase its second overhead pass falls inside the 1.2 s clip. The last block shows what
    the detector does when that second pass is left in.
    """
    out: Dict[str, Dict] = {}
    K, R, t, _ = make_virtual_camera(azimuth_deg=45.0, elevation_deg=8.0, distance_m=12.0, fov_deg=55.0)
    conditions = (
        ("fixed_270_130_ms", True, (240.0, 120.0, 60.0, 30.0)),
        ("literature_192pm31_103pm11_ms", True, (240.0, 120.0, 60.0, 30.0)),
        ("literature_durations_second_arm_pass_left_in_clip", False, (60.0, 30.0)),
    )
    for label, trim, rates in conditions:
        rng = np.random.default_rng(7)
        rows = {}
        for fps in rates:
            errs: Dict[str, List[float]] = {"bfc": [], "ffc": [], "br": []}
            stride_err = []
            for trial in range(n_trials):
                arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                if label.startswith("fixed"):
                    stride_s, ffc_br_s = 0.27, 0.13
                else:
                    stride_s = float(np.clip(rng.normal(0.192, 0.031), 0.125, 0.270))
                    ffc_br_s = float(np.clip(rng.normal(0.103, 0.011), 0.080, 0.135))
                params = _random_params(
                    arm, fps, rng, bfc_time_s=0.45, ffc_time_s=0.45 + stride_s, br_time_s=0.45 + stride_s + ffc_br_s
                )
                j_world_gt, gt = generate_synthetic_delivery(params)
                if trim:
                    j_world_gt = j_world_gt[: int(math.ceil(gt["br_frame"] + 0.15 * fps)) + 1]
                _, j_cam = project_world_points(j_world_gt, K, R, t)
                noise, _ = _stock_noise(j_cam.shape[:2], fps, rng)
                lm_px = project_mcc_landmarks(K, R, t, LM_KEYS, noise_px=1.0, rng=rng)
                cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)
                ev = detect_delivery_events(transform_cam_to_pitch_world(j_cam + noise, cal), fps=fps, bowling_arm=arm)
                for name, e in (("bfc", ev.bfc), ("ffc", ev.ffc), ("br", ev.br)):
                    errs[name].append((e.frame_float - gt[f"{name}_frame"]) / fps * 1000.0)
                stride_err.append(ev.delivery_stride_duration_ms - stride_s * 1000.0)
            row = {f"{k}_mae_ms": round(float(np.mean(np.abs(v))), 1) for k, v in errs.items()}
            row.update({f"{k}_bias_ms": round(float(np.mean(v)), 1) for k, v in errs.items()})
            row.update(
                {f"{k}_err_over_40ms_pct": round(100.0 * float(np.mean(np.abs(v) > 40.0)), 1) for k, v in errs.items()}
            )
            row["stride_duration_mae_ms"] = round(float(np.mean(np.abs(stride_err))), 1)
            row["half_frame_ms_reported_as_uncertainty"] = round(500.0 / fps, 1)
            rows[str(int(fps))] = row
        out[label] = rows
    return out


def s3_generator_artefacts() -> Dict:
    """Ankle displacement across BFC and FFC at 240 fps, and bowling-arm rotation after release."""
    params = SyntheticDeliveryParams(fps=240.0)
    j, gt = generate_synthetic_delivery(params)
    out: Dict[str, Dict] = {}
    for name, joint, frame in (("ffc_front_ankle", "lank", gt["ffc_frame"]), ("bfc_rear_ankle", "rank", gt["bfc_frame"])):
        f = int(round(frame))
        step = np.linalg.norm(np.diff(j[:, JOINT_IDX[joint]], axis=0), axis=1)
        window = step[max(0, f - 12) : f + 12]
        across = float(step[f - 1])
        out[name] = {
            "step_across_contact_m": round(across, 3),
            "implied_speed_mps": round(across * 240.0, 1),
            "median_neighbouring_step_m": round(float(np.median(np.delete(window, 11))), 4),
            "height_before_contact_m": round(float(j[f - 1, JOINT_IDX[joint], 1]), 3),
            "height_at_contact_m": round(float(j[f, JOINT_IDX[joint], 1]), 3),
        }
    arm = j[:, JOINT_IDX["rwri"]] - j[:, JOINT_IDX["rsho"]]
    phi = np.degrees(np.unwrap(np.arctan2(arm[:, 0], arm[:, 1])))
    br = int(round(gt["br_frame"]))
    out["bowling_arm_after_release"] = {
        "rotation_in_first_150ms_deg": round(float(phi[br + 36] - phi[br]), 0),
        "rotation_by_clip_end_deg": round(float(phi[-1] - phi[br]), 0),
        "clip_end_after_release_ms": round((len(phi) - 1 - br) / 240.0 * 1000.0, 0),
    }
    return out


def s4_noise_scale_mismatch(n_trials: int = 12) -> Dict:
    """Coverage when the true noise is k times what the per-joint uncertainty claims (k = 1 is the stock run)."""
    out = {}
    for scale in (0.5, 1.0, 2.0, 3.0):
        rng = np.random.default_rng(2026)
        stock, full = _Tally(), _Tally()
        for vname, az, el in VIEWPOINTS:
            K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
            for fps in FRAME_RATES:
                for trial in range(n_trials):
                    arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                    params = _random_params(arm, fps, rng)
                    res = _calibrated_trial(params, (K, R, t), rng, noise_scale=scale)
                    ref = _reference_scalars(params)
                    for key in STOCK_EVAL_KEYS:
                        stock.add(key, res["metrics"][key], res["gt"][key])
                    for key in METRIC_SPECS:
                        full.add(key, res["metrics"][key], ref[key])
        tab = full.table()
        out[f"true_noise_x{scale}"] = {
            "stock_7_metric_pooled_coverage_pct": stock.pooled_coverage(),
            "all_metric_pooled_coverage_pct": full.pooled_coverage(),
            "per_metric_coverage_pct": {k: v["ci95_coverage_pct"] for k, v in tab.items() if v["reported"] >= 20},
        }
    return out


def s5_world_frame_error(n_trials: int = 12) -> Dict:
    """Coverage when the world frame carries a fixed yaw or gravity-tilt error (stock noise otherwise)."""
    cases = {
        "none": np.eye(3),
        "yaw_5deg": _rot_y(5.0),
        "yaw_10deg": _rot_y(10.0),
        "yaw_15deg": _rot_y(15.0),
        "tilt_fwd_5deg": _rot_z(-5.0),
        "tilt_side_5deg": _rot_x(5.0),
    }
    watch = [
        "thorax_lat_flex_br_deg", "thorax_lat_flex_ffc_deg", "trunk_flexion_br_deg", "trunk_flexion_delta_deg",
        "pelvis_alignment_bfc_deg", "shoulder_alignment_bfc_deg", "pelvic_drop_ffc_deg", "front_knee_br_deg",
        "release_height_m", "follow_through_drift_deg",
    ]
    out = {}
    for name, rot in cases.items():
        rng = np.random.default_rng(2026)
        tl = _Tally()
        for vname, az, el in VIEWPOINTS:
            K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
            for fps in (120.0, 240.0):
                for trial in range(n_trials):
                    arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                    params = _random_params(arm, fps, rng)
                    res = _calibrated_trial(params, (K, R, t), rng, world_error=rot)
                    ref = _reference_scalars(params)
                    for key in watch:
                        tl.add(key, res["metrics"][key], ref[key])
        tab = tl.table()
        out[name] = {
            "pooled_coverage_pct": tl.pooled_coverage(),
            "per_metric": {
                k: {"bias": v["bias"], "rmse": v["rmse"], "coverage_pct": v["ci95_coverage_pct"], "reported": v["reported"]}
                for k, v in tab.items()
            },
        }
    return out


def _realistic_approach(
    j_world: np.ndarray, gt: Dict[str, float], fps: float, arm: BowlingArm, runup_angle_deg: float, extra_lean_deg: float
) -> np.ndarray:
    """Add an angled approach line and a forward sprint lean to the pre-BFC part of a synthetic delivery.

    The stock generator runs exactly along +X with a 4 deg trunk lean. Both edits leave the delivery stride
    (BFC onward) untouched, so every BFC/FFC/BR ground-truth value stays valid.
    """
    j = j_world.copy()
    bfc = gt["bfc_frame"]
    sign_arm = 1.0 if arm == BowlingArm.RIGHT else -1.0
    pelv_x_bfc = float(j[int(round(bfc)), JOINT_IDX["pelv"], 0])
    ub = [JOINT_IDX[k] for k in UPPER_BODY]
    for i in range(j.shape[0]):
        if i >= bfc:
            break
        # Positive angle: approach from the non-bowling-arm side. Negative: from the bowling-arm side.
        # Either way the line merges with the pitch axis at BFC.
        j[i, :, 2] += -sign_arm * math.tan(math.radians(runup_angle_deg)) * (pelv_x_bfc - j[i, JOINT_IDX["pelv"], 0])
        # Sprint lean: rotate the upper body forward about the pelvis, fading out over the last 150 ms before BFC.
        fade = float(np.clip((bfc - i) / (0.15 * fps), 0.0, 1.0))
        if extra_lean_deg > 0.0 and fade > 0.0:
            pelv = j[i, JOINT_IDX["pelv"]].copy()
            j[i, ub] = (j[i, ub] - pelv) @ _rot_z(-extra_lean_deg * fade).T + pelv
    return j


def s6_self_calibration_realism(n_trials: int = 16) -> Dict:
    """Self-calibration +X and gravity errors, and the resulting metric errors, for realistic approaches."""
    watch = [
        "thorax_lat_flex_br_deg", "trunk_flexion_br_deg", "trunk_flexion_delta_deg",
        "pelvis_alignment_bfc_deg", "shoulder_alignment_bfc_deg", "front_knee_br_deg", "release_height_m",
    ]
    out = {}
    for label, angle, lean in (
        ("stock_straight_4deg_lean", 0.0, 0.0),
        ("straight_12deg_lean", 0.0, 8.0),
        ("straight_18deg_lean", 0.0, 14.0),
        ("runup_plus15deg_12deg_lean", 15.0, 8.0),
        ("runup_minus15deg_12deg_lean", -15.0, 8.0),
    ):
        per_view = {}
        for vname, az, el in VIEWPOINTS[:3]:
            rng = np.random.default_rng(11)
            K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
            head_err, g_err, tl = [], [], _Tally()
            for trial in range(n_trials):
                arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                params = _random_params(arm, 120.0, rng)
                jw, gt = generate_synthetic_delivery(params)
                jw = _realistic_approach(jw, gt, 120.0, arm, angle, lean)
                _, jc = project_world_points(jw, K, R, t)
                noise, j_unc_val = _stock_noise(jc.shape[:2], 120.0, rng)
                jc_n = jc + noise
                cal = calibrate_from_3d_motion_and_gravity(
                    jc_n[:, JOINT_IDX["pelv"]], jc_n[:, JOINT_IDX["neck"]] - jc_n[:, JOINT_IDX["pelv"]],
                    1280, 720, joints3d_cam_m=jc_n, bowling_arm_is_right=(arm == BowlingArm.RIGHT),
                )
                r_est = np.asarray(cal.rotation_matrix_R)
                # Columns of a world-to-camera rotation are the world axes expressed in the camera frame;
                # make_virtual_camera returns exactly that matrix. R.T @ v therefore gives v in true world axes.
                x_true_frame = R.T @ r_est[:, 0]
                y_true_frame = R.T @ r_est[:, 1]
                head_err.append(abs(math.degrees(math.atan2(x_true_frame[2], x_true_frame[0]))))
                g_err.append(math.degrees(math.acos(float(np.clip(y_true_frame[1], -1.0, 1.0)))))
                jw_est = transform_cam_to_pitch_world(jc_n, cal)
                ev = detect_delivery_events(jw_est, 120.0, bowling_arm=arm)
                kin = extract_fast_bowling_kinematics(
                    jw_est, ev, 120.0, bowling_arm=arm, joint_uncertainties_m=np.full(jc.shape[:2], j_unc_val)
                )
                metrics = quantify_and_gate_metrics(kin, _audit(120.0, jw_est.shape[0], params.duration_s), cal, events=ev)
                ref = _reference_scalars(params)
                for key in watch:
                    tl.add(key, metrics[key], ref[key])
            tab = tl.table()
            per_view[vname] = {
                "heading_error_of_x_axis_deg_mean": round(float(np.mean(head_err)), 1),
                "heading_error_of_x_axis_deg_max": round(float(np.max(head_err)), 1),
                "gravity_error_deg_mean": round(float(np.mean(g_err)), 1),
                "gravity_error_deg_max": round(float(np.max(g_err)), 1),
                "pooled_coverage_pct": tl.pooled_coverage(),
                "per_metric": {
                    k: {"bias": v["bias"], "rmse": v["rmse"], "coverage_pct": v["ci95_coverage_pct"], "reported": v["reported"]}
                    for k, v in tab.items()
                },
            }
        out[label] = per_view
    return out


def _camera_for_fov(az: float, el: float, fov_deg: float):
    """Camera that keeps the bowler the same size in frame as the stock 12 m / 55 deg setup."""
    dist = 12.0 * math.tan(math.radians(55.0) / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=dist, fov_deg=fov_deg) + (dist,)


def _pose_model_with_assumed_focal(j_cam_true: np.ndarray, f_true: float, f_assumed: float) -> np.ndarray:
    """Camera-frame joints a monocular pose model returns when it assumes the wrong focal length.

    Assumption (weak perspective, the usual consequence of an unknown focal length): the model reproduces
    every pixel and keeps metric root-relative depth, so the root depth scales by f_assumed / f_true.
    EXP-007 checked this on real video: re-running NLF with a different assumed FOV scales pelvis depth by the
    focal ratio and leaves image-plane motion unchanged.
    """
    root_z = j_cam_true[:, JOINT_IDX["pelv"], 2:3]
    z_assumed = root_z * (f_assumed / f_true) + (j_cam_true[..., 2] - root_z)
    xy = j_cam_true[..., :2] / j_cam_true[..., 2:3] * (f_true / f_assumed) * z_assumed[..., None]
    return np.concatenate([xy, z_assumed[..., None]], axis=-1)


def s7_mcc_focal(n_trials: int = 10) -> Dict:
    """(a) PnP with unknown K over the six-value FOV grid. (b) Pose model assuming 55 deg while PnP knows the truth."""
    out: Dict[str, Dict] = {"a_unknown_K_fov_grid": {}, "b_pose_model_assumes_55deg": {}}
    for fov in (6.0, 10.0, 16.0, 25.0, 36.0, 48.0, 62.0):
        for noise_px in (1.0, 3.0):
            rng = np.random.default_rng(5)
            az_e, el_e, pos_e, picked, fails = [], [], [], [], 0
            for trial in range(n_trials):
                az, el = [(90.0, 5.0), (45.0, 8.0), (0.0, 6.0), (10.0, 24.0)][trial % 4]
                K, R, t, cam_pos, dist = _camera_for_fov(az, el, fov)
                lm_px = project_mcc_landmarks(K, R, t, LM_KEYS, noise_px=noise_px, rng=rng)
                try:
                    cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720)
                except RuntimeError:
                    fails += 1
                    continue
                picked.append(2.0 * math.degrees(math.atan(640.0 / cal.camera_matrix_K[0][0])))
                az_e.append(abs((cal.azimuth_deg - az + 180.0) % 360.0 - 180.0))
                el_e.append(abs(cal.elevation_deg - el))
                pos_e.append(float(np.linalg.norm(np.asarray(cal.camera_position_world_m) - cam_pos)) / dist * 100.0)
            out["a_unknown_K_fov_grid"][f"true_fov_{fov:g}deg_noise_{noise_px:g}px"] = {
                "failures": fails,
                "picked_fov_deg_median": round(float(np.median(picked)), 1) if picked else None,
                "azimuth_abs_err_deg_mean": round(float(np.mean(az_e)), 1) if az_e else None,
                "elevation_abs_err_deg_mean": round(float(np.mean(el_e)), 1) if el_e else None,
                "camera_position_err_pct_of_distance_mean": round(float(np.mean(pos_e)), 1) if pos_e else None,
            }
    watch = ["runup_speed_mps", "stride_length_pct_stature", "release_height_m", "thorax_lat_flex_br_deg", "front_knee_br_deg"]
    for fov in (55.0, 25.0, 12.0):
        for vname, az, el in VIEWPOINTS[:2]:
            rng = np.random.default_rng(3)
            K, R, t, _, _ = _camera_for_fov(az, el, fov)
            f_assumed = 640.0 / math.tan(math.radians(55.0) / 2.0)
            tl = _Tally()
            grades: Dict[str, set] = {}
            for trial in range(n_trials):
                arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                params = _random_params(arm, 120.0, rng)
                jw, gt = generate_synthetic_delivery(params)
                _, jc = project_world_points(jw, K, R, t)
                jc_model = _pose_model_with_assumed_focal(jc, float(K[0, 0]), f_assumed)
                noise, j_unc_val = _stock_noise(jc.shape[:2], 120.0, rng)
                lm_px = project_mcc_landmarks(K, R, t, LM_KEYS, noise_px=1.0, rng=rng)
                cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)
                jw_est = transform_cam_to_pitch_world(jc_model + noise, cal)
                ev = detect_delivery_events(jw_est, 120.0, bowling_arm=arm)
                kin = extract_fast_bowling_kinematics(
                    jw_est, ev, 120.0, bowling_arm=arm, joint_uncertainties_m=np.full(jc.shape[:2], j_unc_val)
                )
                metrics = quantify_and_gate_metrics(kin, _audit(120.0, jw_est.shape[0], params.duration_s), cal, events=ev)
                ref = _reference_scalars(params)
                for key in watch:
                    tl.add(key, metrics[key], ref[key])
                    if not metrics[key].suppressed:
                        grades.setdefault(key, set()).add(metrics[key].observability.value)
            tab = tl.table()
            out["b_pose_model_assumes_55deg"][f"true_fov_{fov:g}deg_{vname}"] = {
                k: {
                    "bias": v["bias"], "rmse": v["rmse"], "coverage_pct": v["ci95_coverage_pct"],
                    "suppressed_pct": v["suppressed_pct"], "grades_when_reported": sorted(grades.get(k, [])),
                }
                for k, v in tab.items()
            }
    return out


def _write_video(path: Path, frames: List[np.ndarray], fps: float = 30.0) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()


def s8_unit_reproductions() -> Dict:
    out: Dict[str, Dict] = {}

    # (a) Pelvis and shoulder alignments are unwrapped and anchored independently into [90, 450).
    #     Two lines 6 deg apart that straddle the 90 deg anchor boundary come out about 354 deg apart.
    def _line(angle_deg: float) -> Tuple[np.ndarray, np.ndarray]:
        rad = math.radians(angle_deg - 270.0)
        right = np.array([[math.sin(rad), 0.0, math.cos(rad)]]) * 0.2
        return -right, right

    pl, pr = _line(93.0)
    sl, sr = _line(87.0)
    pelvis = float(_transverse_alignment_deg(pl, pr, True, ref_idx=0)[0])
    shoulder = float(_transverse_alignment_deg(sl, sr, True, ref_idx=0)[0])
    out["a_transverse_anchor_straddle"] = {
        "pelvis_alignment_deg": round(pelvis, 1),
        "shoulder_alignment_deg": round(shoulder, 1),
        "reported_hip_shoulder_separation_deg": round(pelvis - shoulder, 1),
        "true_separation_deg": 6.0,
    }

    # (b) Pan detector: horizontal median flow only, 28 evenly spaced samples.
    rng = np.random.default_rng(7)
    h, w = 240, 320
    big = cv2.GaussianBlur(rng.integers(40, 210, size=(h * 3, w * 6, 3), dtype=np.uint8), (5, 5), 1.2)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        cases = {}
        # zoom-in by 60 % over 24 frames (broadcast push-in): radial flow, median dx near zero
        frames = []
        for i in range(24):
            s = 1.0 + 0.6 * i / 23.0
            cw, ch = int(w * 2 / s), int(h * 2 / s)
            cx, cy = big.shape[1] // 2, big.shape[0] // 2
            frames.append(cv2.resize(big[cy - ch // 2 : cy + ch // 2, cx - cw // 2 : cx + cw // 2], (w, h)))
        _write_video(tmp_dir / "zoom.mp4", frames)
        cases["zoom_in_60pct"] = detect_camera_panning(tmp_dir / "zoom.mp4")
        # vertical tilt 4 px/frame
        frames = [big[40 + 4 * i : 40 + 4 * i + h, 100 : 100 + w].copy() for i in range(24)]
        _write_video(tmp_dir / "tilt.mp4", frames)
        cases["vertical_tilt_4px_per_frame"] = detect_camera_panning(tmp_dir / "tilt.mp4")
        # the same 4 px/frame horizontal pan the unit test uses, 20 frames (detected) ...
        frames = [big[100 : 100 + h, 4 * i : 4 * i + w].copy() for i in range(20)]
        _write_video(tmp_dir / "pan_short.mp4", frames)
        cases["horizontal_pan_4px_per_frame_20_frames"] = detect_camera_panning(tmp_dir / "pan_short.mp4")
        # ... and in a 300-frame clip, where the 28 samples are ~11 frames (44 px) apart
        frames = [big[100 : 100 + h, 4 * i : 4 * i + w].copy() for i in range(300)]
        _write_video(tmp_dir / "pan_long.mp4", frames)
        cases["horizontal_pan_4px_per_frame_300_frames"] = detect_camera_panning(tmp_dir / "pan_long.mp4")
    out["b_pan_detector"] = {
        k: {"is_panning": bool(v[0]), "pan_fraction_of_width": float(v[1])} for k, v in cases.items()
    }
    out["b_pan_detector"]["horizontal_pan_4px_per_frame_300_frames"]["true_pan_fraction_of_width"] = round(4 * 299 / 320, 2)
    return out


def s9_decision_reliability(n_trials: int = 12) -> Dict:
    """Do the user-facing decisions survive the benchmark's own noise?

    For each stock-benchmark trial build two fact sheets with identical suppression states and CIs: one from the
    estimated values, one with every reported value replaced by its reference truth. Then compare cues, pillar
    labels, quadrant and action class.
    """
    rng = np.random.default_rng(2026)
    cue_stats: Dict[str, Dict[str, int]] = {}
    agree = {"kinetic_label": [], "durability_label": [], "action_class": []}
    kin_abs, dur_abs = [], []
    for vname, az, el in VIEWPOINTS:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        for fps in FRAME_RATES:
            for trial in range(n_trials):
                arm = BowlingArm.RIGHT if trial % 2 == 0 else BowlingArm.LEFT
                params = _random_params(arm, fps, rng)
                res = _calibrated_trial(params, (K, R, t), rng)
                ref = _reference_scalars(params)
                m_est = res["metrics"]
                m_true = {
                    k: (m if (m.suppressed or m.value is None) else m.model_copy(update={"value": round(ref[k], 2)}))
                    for k, m in m_est.items()
                }
                true_class = classify_action_portus(
                    ref["shoulder_alignment_bfc_deg"], ref["pelvis_alignment_bfc_deg"], ref["shoulder_counter_rotation_deg"]
                )
                fs_est = build_fact_sheet("est", arm, res["kin"].action_class, "synthetic", res["audit"], res["cal"], res["events"], m_est)
                fs_true = build_fact_sheet("true", arm, true_class, "synthetic", res["audit"], res["cal"], res["events"], m_true)
                est_titles = {c.title for c in fs_est.strengths + fs_est.limiters}
                true_titles = {c.title for c in fs_true.strengths + fs_true.limiters}
                for title in est_titles | true_titles:
                    st = cue_stats.setdefault(title, {"issued": 0, "correct": 0, "should_have_been_issued": 0})
                    st["issued"] += int(title in est_titles)
                    st["correct"] += int(title in est_titles and title in true_titles)
                    st["should_have_been_issued"] += int(title in true_titles)
                if fs_est.kinetic_efficiency.score is not None and fs_true.kinetic_efficiency.score is not None:
                    agree["kinetic_label"].append(fs_est.kinetic_efficiency.status_label == fs_true.kinetic_efficiency.status_label)
                    kin_abs.append(abs(fs_est.kinetic_efficiency.score - fs_true.kinetic_efficiency.score))
                if fs_est.lumbar_load_screening.score is not None and fs_true.lumbar_load_screening.score is not None:
                    agree["durability_label"].append(
                        fs_est.lumbar_load_screening.status_label == fs_true.lumbar_load_screening.status_label
                    )
                    dur_abs.append(abs(fs_est.lumbar_load_screening.score - fs_true.lumbar_load_screening.score))
                if fs_est.action_class.value != "unknown" and fs_true.action_class.value != "unknown":
                    agree["action_class"].append(fs_est.action_class == fs_true.action_class)
    cues = {}
    for title, st in sorted(cue_stats.items()):
        cues[title] = {
            **st,
            "precision_pct": round(100.0 * st["correct"] / st["issued"], 1) if st["issued"] else None,
            "recall_pct": round(100.0 * st["correct"] / st["should_have_been_issued"], 1) if st["should_have_been_issued"] else None,
        }
    return {
        "cues": cues,
        "label_agreement_pct": {k: (round(100.0 * float(np.mean(v)), 1) if v else None) for k, v in agree.items()},
        "label_agreement_n": {k: len(v) for k, v in agree.items()},
        "kinetic_score_abs_diff_median": round(float(np.median(kin_abs)), 1) if kin_abs else None,
        "kinetic_score_abs_diff_p90": round(float(np.percentile(kin_abs, 90)), 1) if kin_abs else None,
        "durability_score_abs_diff_median": round(float(np.median(dur_abs)), 1) if dur_abs else None,
        "durability_score_abs_diff_p90": round(float(np.percentile(dur_abs, 90)), 1) if dur_abs else None,
    }


EXPERIMENTS = {
    "s1_per_metric_coverage": s1_per_metric_coverage,
    "s2_event_timing_priors": s2_event_timing_priors,
    "s3_generator_artefacts": s3_generator_artefacts,
    "s4_noise_scale_mismatch": s4_noise_scale_mismatch,
    "s5_world_frame_error": s5_world_frame_error,
    "s6_self_calibration_realism": s6_self_calibration_realism,
    "s7_mcc_focal": s7_mcc_focal,
    "s8_unit_reproductions": s8_unit_reproductions,
    "s9_decision_reliability": s9_decision_reliability,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=Path("outputs/exp007_synthetic_diagnostics.json"))
    parser.add_argument("--only", nargs="*", choices=list(EXPERIMENTS), help="run a subset")
    args = parser.parse_args()
    results = {}
    for name, fn in EXPERIMENTS.items():
        if args.only and name not in args.only:
            continue
        print(f"[{name}] running ...", flush=True)
        results[name] = fn()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
