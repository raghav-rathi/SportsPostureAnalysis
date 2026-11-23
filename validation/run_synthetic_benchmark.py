"""Synthetic Ground-Truth Metrology & Per-Metric Uncertainty Calibration Benchmark (v2).

Upgraded per EXP-007 (Roadmap Item 5 + B22):
  - Adds CLI flags (`--out`, `--trials`, `--seed`) fixing B22
  - Scores ALL 22 observable metrics individually (bias, RMSE, median interval half-width,
    realized RMSE / stated sigma sharpness ratio, and empirical coverage %) against clean ground truth
  - Randomizes release arm angle (`br_phi_deg in [-2, +16] deg`) and Felton (2019) phase timings
    instead of hardcoding the detector's own constants
  - Reports BFC, FFC, and BR timing MAE and bias explicitly
  - Evaluates a non-oracle noise stress arm (`0.5x`, `1.0x`, `2.0x` true noise vs supplied uncertainty)
  - Evaluates a realistic noisy uncalibrated self-calibration arm (`12 deg` sprint lean + `10 deg`
    angled approach + signed azimuth error)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from bowling_biomech.constants import JOINT_IDX
from bowling_biomech.schemas import BowlingArm, DeliveryEvents, EventTimestamp, VideoAudit
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
from bowling_biomech.stage4_uncertainty import METRIC_SPECS, quantify_and_gate_metrics
from bowling_biomech.synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
    project_world_points,
)


def _ar1_correlated_noise(
    shape: tuple,
    sigma_total: float,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Temporally correlated AR(1) noise along axis 0 (simulating realistic monocular depth drift)."""
    t_len = shape[0]
    out = np.zeros(shape, dtype=np.float64)
    innov_scale = sigma_total * np.sqrt(max(1.0 - rho * rho, 1e-6))
    out[0] = rng.normal(0.0, sigma_total, size=shape[1:])
    for t in range(1, t_len):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innov_scale, size=shape[1:])
    return out


def _oracle_events_from_params(p: SyntheticDeliveryParams, n_frames: int) -> DeliveryEvents:
    def _ev(t_s: float) -> EventTimestamp:
        ff = t_s * p.fps
        idx = int(np.clip(round(ff), 0, n_frames - 1))
        return EventTimestamp(
            frame_float=ff, frame_idx=idx, source_frame_idx=idx, timestamp_s=t_s,
            timing_uncertainty_ms=0.1, confidence=0.99, detection_method="oracle",
        )
    return DeliveryEvents(
        runup_start_frame=0,
        bfc=_ev(p.bfc_time_s),
        ffc=_ev(p.ffc_time_s),
        uah=None,
        br=_ev(p.br_time_s),
        ft_peak=None,
        delivery_stride_duration_ms=(p.ffc_time_s - p.bfc_time_s) * 1000.0,
        ffc_to_br_duration_ms=(p.br_time_s - p.ffc_time_s) * 1000.0,
        is_complete_delivery=True,
    )


def _signed_angle_diff_deg(a: float, b: float) -> float:
    return float(((a - b + 180.0) % 360.0) - 180.0)


def run_benchmark(
    n_trials_per_cell: int = 12,
    out_json: Path | None = None,
    seed: int = 2026,
) -> Dict:
    rng = np.random.default_rng(seed)
    viewpoints = [
        ("side_on_90", 90.0, 5.0),
        ("open_side_45", 45.0, 8.0),
        ("rear_0", 0.0, 6.0),
        ("broadcast_elevated", 10.0, 24.0),
    ]
    frame_rates = [25.0, 30.0, 60.0, 120.0, 240.0]
    all_keys = list(METRIC_SPECS.keys())
    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg",
        "stump_middle_base", "stump_middle_top",
    ]

    summary_rows = []
    per_metric_acc: Dict[str, Dict[str, List[float]]] = {
        k: {"err": [], "sigma": [], "half_ci": [], "covered": []} for k in all_keys
    }
    timing_acc: Dict[float, Dict[str, List[float]]] = {
        fps: {"bfc": [], "ffc": [], "br": []} for fps in frame_rates
    }
    false_incomplete_by_fps: Dict[str, int] = {str(int(fps)): 0 for fps in frame_rates}
    total_non_sup = 0
    total_covered = 0

    for vname, az, el in viewpoints:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        for fps in frame_rates:
            sq_errs: Dict[str, List[float]] = {k: [] for k in all_keys}
            covered_cnt = 0
            obs_cnt = 0
            sup_cnt = 0
            br_err_ms_list = []

            for trial in range(n_trials_per_cell):
                arm = BowlingArm.RIGHT if (trial % 2 == 0) else BowlingArm.LEFT
                bfc_t = float(rng.uniform(0.38, 0.45))
                stride_s = float(rng.uniform(0.155, 0.235))  # Felton 2019: 192 +- 31 ms
                ffc_br_s = float(rng.uniform(0.088, 0.128))  # Felton 2019: 103-128 ms
                params = SyntheticDeliveryParams(
                    bowler_arm=arm,
                    fps=fps,
                    bfc_time_s=bfc_t,
                    ffc_time_s=bfc_t + stride_s,
                    br_time_s=bfc_t + stride_s + ffc_br_s,
                    runup_speed_mps=float(rng.uniform(4.8, 6.4)),
                    front_knee_ffc_deg=float(rng.uniform(146.0, 168.0)),
                    front_knee_br_deg=float(rng.uniform(150.0, 175.0)),
                    trunk_flexion_br_deg=float(rng.uniform(34.0, 50.0)),
                    thorax_lat_flex_br_deg=float(rng.uniform(28.0, 52.0)),
                    shoulder_counter_rot_deg=float(rng.uniform(10.0, 38.0)),
                    delayed_arm_ffc_deg=float(rng.uniform(120.0, 158.0)),
                    br_phi_deg=float(rng.uniform(-2.0, 16.0)),
                    pelvic_drop_ffc_deg=float(rng.uniform(-6.0, 12.0)),
                    follow_through_drift_deg=float(rng.uniform(-14.0, 14.0)),
                )
                j_world_gt, _ = generate_synthetic_delivery(params)
                oracle_ev = _oracle_events_from_params(params, j_world_gt.shape[0])
                gt_kin = extract_fast_bowling_kinematics(j_world_gt, oracle_ev, fps=fps, bowling_arm=arm)
                gt_vals = gt_kin.scalars

                _, j_cam_clean = project_world_points(j_world_gt, K, R, t)
                rho = float(np.exp(-1.0 / max(0.08 * fps, 1.0)))
                sigma_xy = 0.015 if fps >= 120 else 0.022
                sigma_z = 0.036 if fps >= 120 else 0.052
                noise = np.stack(
                    [
                        _ar1_correlated_noise(j_cam_clean.shape[:2], sigma_xy, rho, rng),
                        _ar1_correlated_noise(j_cam_clean.shape[:2], sigma_xy, rho, rng),
                        _ar1_correlated_noise(j_cam_clean.shape[:2], sigma_z, rho, rng),
                    ],
                    axis=-1,
                )
                j_cam_noisy = j_cam_clean + noise
                j_unc = np.full(j_cam_clean.shape[:2], float(np.hypot(sigma_xy, sigma_z * 0.55)))

                lm_px = project_mcc_landmarks(K, R, t, lm_keys, noise_px=1.0, rng=rng)
                cal = calibrate_from_mcc_landmarks(lm_px, 1280, 720, K=K)
                j_world_est = transform_cam_to_pitch_world(j_cam_noisy, cal)

                events = detect_delivery_events(j_world_est, fps=fps, bowling_arm=arm)
                if not events.is_complete_delivery:
                    false_incomplete_by_fps[str(int(fps))] += 1
                bfc_e = (events.bfc.frame_float - oracle_ev.bfc.frame_float) / fps * 1000.0
                ffc_e = (events.ffc.frame_float - oracle_ev.ffc.frame_float) / fps * 1000.0
                br_e = (events.br.frame_float - oracle_ev.br.frame_float) / fps * 1000.0
                timing_acc[fps]["bfc"].append(bfc_e)
                timing_acc[fps]["ffc"].append(ffc_e)
                timing_acc[fps]["br"].append(br_e)
                br_err_ms_list.append(abs(br_e))

                kin = extract_fast_bowling_kinematics(
                    j_world_est,
                    events,
                    fps=fps,
                    bowling_arm=arm,
                    joint_uncertainties_m=j_unc,
                    camera_R_w2c=np.asarray(cal.rotation_matrix_R),
                    unc_scale_factor=1.0,
                )
                audit = VideoAudit(
                    video_path="synthetic.mp4",
                    width=1280,
                    height=720,
                    fps=fps,
                    num_frames=j_world_est.shape[0],
                    duration_s=params.duration_s,
                    median_laplacian_var=220.0 if fps >= 120 else 90.0,
                    blur_grade="crisp" if fps >= 120 else "moderate_blur",
                    fps_grade="high_speed_240" if fps >= 200 else ("slow_mo_120" if fps >= 100 else "standard_30"),
                    est_arm_deg_per_frame=2300.0 / fps,
                    est_ball_m_per_frame=38.89 / fps,
                )
                metrics = quantify_and_gate_metrics(kin, audit, cal, events=events, unc_scale_factor=1.0)

                for k in all_keys:
                    m = metrics[k]
                    if m.suppressed or m.value is None or k not in gt_vals:
                        sup_cnt += 1
                        continue
                    obs_cnt += 1
                    total_non_sup += 1
                    err = float(m.value - gt_vals[k])
                    sq_errs[k].append(err * err)
                    cov_flag = 1.0 if (m.ci_low is not None and m.ci_high is not None and m.ci_low <= gt_vals[k] <= m.ci_high) else 0.0
                    covered_cnt += int(cov_flag)
                    total_covered += int(cov_flag)
                    per_metric_acc[k]["err"].append(err)
                    per_metric_acc[k]["sigma"].append(float(m.sigma_total or 0.0))
                    per_metric_acc[k]["half_ci"].append(0.5 * float((m.ci_high or m.value) - (m.ci_low or m.value)))
                    per_metric_acc[k]["covered"].append(cov_flag)

            summary_rows.append(
                {
                    "view": vname,
                    "fps": fps,
                    "br_mae_ms": round(float(np.mean(br_err_ms_list)), 2),
                    "observable_evals": obs_cnt,
                    "suppressed_evals": sup_cnt,
                    "ci95_coverage_pct": round(100.0 * covered_cnt / max(obs_cnt, 1), 1),
                    "rmse_by_metric": {
                        k: (round(float(np.sqrt(np.mean(sq_errs[k]))), 2) if sq_errs[k] else None)
                        for k in ("front_knee_br_deg", "thorax_lat_flex_br_deg", "delayed_arm_ffc_deg", "stride_length_pct_stature", "runup_speed_mps")
                    },
                }
            )

    # Per-metric summary across all 22 metrics
    per_metric_summary: Dict[str, Dict] = {}
    for k in all_keys:
        errs = np.asarray(per_metric_acc[k]["err"], dtype=np.float64)
        sigs = np.asarray(per_metric_acc[k]["sigma"], dtype=np.float64)
        hcis = np.asarray(per_metric_acc[k]["half_ci"], dtype=np.float64)
        covs = np.asarray(per_metric_acc[k]["covered"], dtype=np.float64)
        if errs.size == 0:
            per_metric_summary[k] = {"n": 0}
            continue
        rmse = float(np.sqrt(np.mean(errs**2)))
        med_sig = float(np.median(sigs))
        per_metric_summary[k] = {
            "n": int(errs.size),
            "bias": round(float(np.mean(errs)), 2),
            "rmse": round(rmse, 2),
            "median_half_ci": round(float(np.median(hcis)), 2),
            "rmse_over_median_sigma": round(rmse / max(med_sig, 1e-6), 2),
            "coverage_pct": round(100.0 * float(np.mean(covs)), 1),
        }

    # Timing summary per FPS
    timing_summary = {}
    for fps, d in timing_acc.items():
        timing_summary[str(int(fps))] = {
            ev_k: {
                "mae_ms": round(float(np.mean(np.abs(vals))), 2),
                "bias_ms": round(float(np.mean(vals)), 2),
            }
            for ev_k, vals in d.items()
        }

    # Realistic noisy uncalibrated self-calibration arm (12 deg sprint lean + 10 deg angled approach + AR(1) noise!)
    self_cal_arm = []
    for vname, az, el in [("side_on_90", 90.0, 5.0), ("open_side_45", 45.0, 8.0), ("rear_0", 0.0, 6.0)]:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        az_signed_errs, rh_errs, lf_errs, rh_cov, lf_cov = [], [], [], [], []
        for trial in range(12):
            arm = BowlingArm.RIGHT if (trial % 2 == 0) else BowlingArm.LEFT
            p = SyntheticDeliveryParams(
                bowler_arm=arm, fps=120.0, sprint_lean_deg=12.0, approach_angle_deg=10.0 if trial % 2 == 0 else -10.0
            )
            jw, _ = generate_synthetic_delivery(p)
            oe = _oracle_events_from_params(p, jw.shape[0])
            gt_s = extract_fast_bowling_kinematics(jw, oe, fps=120.0, bowling_arm=arm).scalars
            _, jc = project_world_points(jw, K, R, t)
            rho = float(np.exp(-1.0 / (0.08 * 120.0)))
            jc_noisy = jc + np.stack(
                [
                    _ar1_correlated_noise(jc.shape[:2], 0.015, rho, rng),
                    _ar1_correlated_noise(jc.shape[:2], 0.015, rho, rng),
                    _ar1_correlated_noise(jc.shape[:2], 0.036, rho, rng),
                ],
                axis=-1,
            )
            j_unc = np.full(jc.shape[:2], float(np.hypot(0.015, 0.036 * 0.55)))
            cal_sc = calibrate_from_3d_motion_and_gravity(
                jc_noisy[:, JOINT_IDX["pelv"]],
                jc_noisy[:, JOINT_IDX["neck"]] - jc_noisy[:, JOINT_IDX["pelv"]],
                1280,
                720,
                joints3d_cam_m=jc_noisy,
                joint_uncertainties_m=j_unc,
                bowling_arm_is_right=(arm == BowlingArm.RIGHT),
            )
            az_signed_errs.append(abs(_signed_angle_diff_deg(cal_sc.azimuth_deg, az)))
            jw_est = transform_cam_to_pitch_world(jc_noisy, cal_sc)
            ev = detect_delivery_events(jw_est, 120.0, bowling_arm=arm)
            kin = extract_fast_bowling_kinematics(
                jw_est, ev, 120.0, bowling_arm=arm, joint_uncertainties_m=j_unc,
                camera_R_w2c=np.asarray(cal_sc.rotation_matrix_R),
            )
            aud = VideoAudit(
                video_path="syn.mp4", width=1280, height=720, fps=120.0, num_frames=jw_est.shape[0],
                duration_s=p.duration_s, median_laplacian_var=220.0, blur_grade="crisp", fps_grade="slow_mo_120",
                est_arm_deg_per_frame=2300.0 / 120.0, est_ball_m_per_frame=38.89 / 120.0,
            )
            ms = quantify_and_gate_metrics(kin, aud, cal_sc, events=ev)
            rh_errs.append(kin.scalars["release_height_m"] - gt_s["release_height_m"])
            lf_errs.append(kin.scalars["thorax_lat_flex_br_deg"] - gt_s["thorax_lat_flex_br_deg"])
            if not ms["release_height_m"].suppressed and ms["release_height_m"].ci_low is not None:
                rh_cov.append(1.0 if ms["release_height_m"].ci_low <= gt_s["release_height_m"] <= ms["release_height_m"].ci_high else 0.0)
            if not ms["thorax_lat_flex_br_deg"].suppressed and ms["thorax_lat_flex_br_deg"].ci_low is not None:
                lf_cov.append(1.0 if ms["thorax_lat_flex_br_deg"].ci_low <= gt_s["thorax_lat_flex_br_deg"] <= ms["thorax_lat_flex_br_deg"].ci_high else 0.0)

        self_cal_arm.append(
            {
                "view": vname,
                "signed_az_mae_deg": round(float(np.mean(az_signed_errs)), 2),
                "release_height_bias_m": round(float(np.mean(rh_errs)), 3),
                "release_height_rmse_m": round(float(np.sqrt(np.mean(np.square(rh_errs)))), 3),
                "release_height_reported_n": len(rh_cov),
                "lat_flex_br_rmse_deg": round(float(np.sqrt(np.mean(np.square(lf_errs)))), 2),
                "lat_flex_br_cov_pct": round(100.0 * float(np.mean(lf_cov)), 1) if lf_cov else None,
            }
        )

    # Non-oracle noise mismatch stress arm (true noise 0.5x, 1.0x, 2.0x relative to stated uncertainty)
    noise_mismatch = {}
    K_o, R_o, t_o, _ = make_virtual_camera(azimuth_deg=45.0, elevation_deg=8.0, distance_m=12.0, fov_deg=55.0)
    for mult in (0.5, 1.0, 2.0):
        cov_m, n_m = 0, 0
        for trial in range(16):
            arm = BowlingArm.RIGHT if (trial % 2 == 0) else BowlingArm.LEFT
            p = SyntheticDeliveryParams(bowler_arm=arm, fps=120.0, br_phi_deg=float(rng.uniform(0.0, 14.0)))
            jw, _ = generate_synthetic_delivery(p)
            oe = _oracle_events_from_params(p, jw.shape[0])
            gt_s = extract_fast_bowling_kinematics(jw, oe, fps=120.0, bowling_arm=arm).scalars
            _, jc = project_world_points(jw, K_o, R_o, t_o)
            rho = float(np.exp(-1.0 / (0.08 * 120.0)))
            jc_n = jc + np.stack(
                [
                    _ar1_correlated_noise(jc.shape[:2], 0.015 * mult, rho, rng),
                    _ar1_correlated_noise(jc.shape[:2], 0.015 * mult, rho, rng),
                    _ar1_correlated_noise(jc.shape[:2], 0.036 * mult, rho, rng),
                ],
                axis=-1,
            )
            j_unc_fixed = np.full(jc.shape[:2], float(np.hypot(0.015, 0.036 * 0.55)))
            cal = calibrate_from_mcc_landmarks(project_mcc_landmarks(K_o, R_o, t_o, lm_keys, noise_px=1.0, rng=rng), 1280, 720, K=K_o)
            jw_e = transform_cam_to_pitch_world(jc_n, cal)
            ev = detect_delivery_events(jw_e, 120.0, bowling_arm=arm)
            kin = extract_fast_bowling_kinematics(jw_e, ev, 120.0, bowling_arm=arm, joint_uncertainties_m=j_unc_fixed, camera_R_w2c=np.asarray(cal.rotation_matrix_R))
            aud = VideoAudit(
                video_path="syn.mp4", width=1280, height=720, fps=120.0, num_frames=jw_e.shape[0],
                duration_s=p.duration_s, median_laplacian_var=220.0, blur_grade="crisp", fps_grade="slow_mo_120",
                est_arm_deg_per_frame=2300.0 / 120.0, est_ball_m_per_frame=38.89 / 120.0,
            )
            ms = quantify_and_gate_metrics(kin, aud, cal, events=ev)
            for k in all_keys:
                m = ms[k]
                if not m.suppressed and m.value is not None and m.ci_low is not None and k in gt_s:
                    n_m += 1
                    cov_m += int(m.ci_low <= gt_s[k] <= m.ci_high)
        noise_mismatch[f"{mult}x"] = round(100.0 * cov_m / max(n_m, 1), 1)

    result = {
        "benchmark_version": "2.0.0_per_metric_non_circular",
        "noise_model": "temporally_correlated_AR1_80ms_tau_both_arms_randomized_br_and_felton_timings",
        "pooled_analytical_coverage_pct_all_22": round(100.0 * total_covered / max(total_non_sup, 1), 1),
        "total_observable_evaluations": total_non_sup,
        "false_incomplete_trials": int(sum(false_incomplete_by_fps.values())),
        "false_incomplete_by_fps": false_incomplete_by_fps,
        "per_metric_calibration": per_metric_summary,
        "event_timing_errors_ms": timing_summary,
        "noise_mismatch_coverage_pct_all_22": noise_mismatch,
        "noisy_realistic_self_cal_arm_12deg_lean_10deg_angle": self_cal_arm,
        "cells": summary_rows,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic Ground-Truth Per-Metric Metrology Benchmark")
    ap.add_argument("--out", type=Path, default=Path("outputs/synthetic_benchmark_results.json"))
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--seed", type=int, default=2026)
    args = ap.parse_args()

    res = run_benchmark(n_trials_per_cell=args.trials, out_json=args.out, seed=args.seed)
    print(f"Pooled Analytical Coverage (all 22 metrics, synthetic AR(1) oracle noise): {res['pooled_analytical_coverage_pct_all_22']}% ({res['total_observable_evaluations']} evals)")
    print(f"Non-Oracle Noise Mismatch Stress (all 22 @ 45deg 120fps): {res['noise_mismatch_coverage_pct_all_22']}")
    print("\nPer-Metric Calibration (All 22 Metrics):")
    for k, s in res["per_metric_calibration"].items():
        if s.get("n", 0) == 0:
            continue
        print(f"  {k:32s} | n={s['n']:3d} | bias={s['bias']:+6.2f} | rmse={s['rmse']:6.2f} | rmse/sig={s['rmse_over_median_sigma']:4.2f} | cov={s['coverage_pct']:5.1f}%")
    print("\nEvent Timing Errors (ms):")
    for fps_s, t_d in res["event_timing_errors_ms"].items():
        print(f"  {fps_s:>3s} fps | BFC MAE={t_d['bfc']['mae_ms']:4.1f} (bias {t_d['bfc']['bias_ms']:+5.1f}) | FFC MAE={t_d['ffc']['mae_ms']:4.1f} (bias {t_d['ffc']['bias_ms']:+5.1f}) | BR MAE={t_d['br']['mae_ms']:4.1f}")
    print("\nNoisy Realistic Self-Calibration Arm (120fps, 12 deg sprint lean, +-10 deg approach):")
    for sc in res["noisy_realistic_self_cal_arm_12deg_lean_10deg_angle"]:
        print(f"  {sc['view']:14s} | signed Az MAE={sc['signed_az_mae_deg']:4.1f} deg | RH bias={sc['release_height_bias_m']:+.3f} m (rep={sc['release_height_reported_n']}) | LatFlex RMSE={sc['lat_flex_br_rmse_deg']:4.1f} deg (cov={sc['lat_flex_br_cov_pct']}%)")


if __name__ == "__main__":
    main()
