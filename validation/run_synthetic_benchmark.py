"""Synthetic Ground-Truth Metrology & 95% Confidence Interval Calibration Benchmark.

Upgraded per independent audit (P1-4):
  - Uses temporally correlated AR(1) monocular ray-depth drift + white jitter (NOT pure white noise
    that Butterworth filtfilt artificially washes away at 240 fps!)
  - Evaluates BOTH MCC Law 7 PnP calibration AND uncalibrated Motion-Gravity Self-Calibration arms
  - Tests BOTH Right-Arm and Left-Arm fast bowlers across 4 viewpoints x 5 frame rates
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from bowling_biomech.schemas import BowlingArm, VideoAudit
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
from bowling_biomech.synthetic_ground_truth import (
    JOINT_IDX,
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
    """Generate temporally correlated AR(1) noise along axis 0 (simulating realistic monocular depth drift)."""
    t_len = shape[0]
    out = np.zeros(shape, dtype=np.float64)
    innov_scale = sigma_total * np.sqrt(max(1.0 - rho * rho, 1e-6))
    out[0] = rng.normal(0.0, sigma_total, size=shape[1:])
    for t in range(1, t_len):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innov_scale, size=shape[1:])
    return out


def run_benchmark(n_trials_per_cell: int = 12, out_json: Path | None = None) -> Dict:
    rng = np.random.default_rng(2026)
    viewpoints = [
        ("side_on_90", 90.0, 5.0),
        ("open_side_45", 45.0, 8.0),
        ("rear_0", 0.0, 6.0),
        ("broadcast_elevated", 10.0, 24.0),
    ]
    frame_rates = [25.0, 30.0, 60.0, 120.0, 240.0]
    eval_keys = [
        "front_knee_ffc_deg",
        "front_knee_br_deg",
        "trunk_flexion_delta_deg",
        "thorax_lat_flex_br_deg",
        "shoulder_counter_rotation_deg",
        "delayed_arm_ffc_deg",
        "runup_speed_mps",
    ]
    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg",
        "popping_crease_off", "popping_crease_leg",
        "stump_middle_base", "stump_middle_top",
    ]

    summary_rows = []
    total_non_sup = 0
    total_covered = 0

    for vname, az, el in viewpoints:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        for fps in frame_rates:
            sq_errs: Dict[str, List[float]] = {k: [] for k in eval_keys}
            covered_cnt = 0
            obs_cnt = 0
            sup_cnt = 0
            br_err_ms_list = []

            for trial in range(n_trials_per_cell):
                arm = BowlingArm.RIGHT if (trial % 2 == 0) else BowlingArm.LEFT
                params = SyntheticDeliveryParams(
                    bowler_arm=arm,
                    fps=fps,
                    runup_speed_mps=float(rng.uniform(4.8, 6.4)),
                    front_knee_ffc_deg=float(rng.uniform(146.0, 168.0)),
                    front_knee_br_deg=float(rng.uniform(150.0, 175.0)),
                    trunk_flexion_br_deg=float(rng.uniform(34.0, 50.0)),
                    thorax_lat_flex_br_deg=float(rng.uniform(28.0, 52.0)),
                    shoulder_counter_rot_deg=float(rng.uniform(10.0, 38.0)),
                    delayed_arm_ffc_deg=float(rng.uniform(120.0, 158.0)),
                )
                j_world_gt, gt = generate_synthetic_delivery(params)
                _, j_cam_clean = project_world_points(j_world_gt, K, R, t)

                # Temporally correlated AR(1) depth drift (survives Butterworth filtering!) + white jitter
                rho = float(np.exp(-1.0 / max(0.08 * fps, 1.0)))  # ~80 ms correlation time constant
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
                br_err_ms_list.append(abs(events.br.frame_float - gt["br_frame"]) / fps * 1000.0)

                kin = extract_fast_bowling_kinematics(
                    j_world_est, events, fps=fps, bowling_arm=arm, joint_uncertainties_m=j_unc
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
                metrics = quantify_and_gate_metrics(kin, audit, cal, events=events)

                for k in eval_keys:
                    m = metrics[k]
                    if m.suppressed or m.value is None:
                        sup_cnt += 1
                        continue
                    obs_cnt += 1
                    total_non_sup += 1
                    err = m.value - gt[k]
                    sq_errs[k].append(err * err)
                    if m.ci_low is not None and m.ci_high is not None and (m.ci_low <= gt[k] <= m.ci_high):
                        covered_cnt += 1
                        total_covered += 1

            row = {
                "view": vname,
                "fps": fps,
                "br_mae_ms": round(float(np.mean(br_err_ms_list)), 2),
                "observable_evals": obs_cnt,
                "suppressed_evals": sup_cnt,
                "ci95_coverage_pct": round(100.0 * covered_cnt / max(obs_cnt, 1), 1),
                "rmse_by_metric": {
                    k: (round(float(np.sqrt(np.mean(v))), 2) if v else None) for k, v in sq_errs.items()
                },
            }
            summary_rows.append(row)

    # Also benchmark uncalibrated motion-gravity self-calibration arm (side_on_90 & open_side_45 @ 120fps)
    self_cal_arm = []
    for vname, az, el in [("side_on_90", 90.0, 5.0), ("open_side_45", 45.0, 8.0), ("rear_0", 0.0, 6.0)]:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az, elevation_deg=el, distance_m=12.0, fov_deg=55.0)
        az_errs = []
        fk_errs = []
        lf_errs = []
        for trial in range(8):
            arm = BowlingArm.RIGHT if (trial % 2 == 0) else BowlingArm.LEFT
            p = SyntheticDeliveryParams(bowler_arm=arm, fps=120.0)
            jw, gt = generate_synthetic_delivery(p)
            _, jc = project_world_points(jw, K, R, t)
            cal_sc = calibrate_from_3d_motion_and_gravity(
                jc[:, JOINT_IDX["pelv"]],
                jc[:, JOINT_IDX["neck"]] - jc[:, JOINT_IDX["pelv"]],
                1280,
                720,
                joints3d_cam_m=jc,
                bowling_arm_is_right=(arm == BowlingArm.RIGHT),
            )
            az_errs.append(abs(abs(cal_sc.azimuth_deg) - abs(az)))
            jw_est = transform_cam_to_pitch_world(jc, cal_sc)
            ev = detect_delivery_events(jw_est, 120.0, bowling_arm=arm)
            kin = extract_fast_bowling_kinematics(jw_est, ev, 120.0, bowling_arm=arm)
            fk_errs.append((kin.scalars["front_knee_br_deg"] - gt["front_knee_br_deg"]) ** 2)
            lf_errs.append((kin.scalars["thorax_lat_flex_br_deg"] - gt["thorax_lat_flex_br_deg"]) ** 2)
        self_cal_arm.append(
            {
                "view": vname,
                "mean_az_err_deg": round(float(np.mean(az_errs)), 2),
                "fk_br_rmse_deg": round(float(np.sqrt(np.mean(fk_errs))), 2),
                "lat_flex_br_rmse_deg": round(float(np.sqrt(np.mean(lf_errs))), 2),
            }
        )

    result = {
        "noise_model": "temporally_correlated_AR1_80ms_tau_both_arms",
        "overall_ci95_coverage_pct": round(100.0 * total_covered / max(total_non_sup, 1), 1),
        "total_observable_evaluations": total_non_sup,
        "cells": summary_rows,
        "uncalibrated_self_cal_arm": self_cal_arm,
    }
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    out_path = Path("outputs/synthetic_benchmark_results.json")
    res = run_benchmark(n_trials_per_cell=12, out_json=out_path)
    print(f"Overall 95% CI Empirical Coverage (AR(1) Correlated Noise, Right+Left Arms): {res['overall_ci95_coverage_pct']}% across {res['total_observable_evaluations']} observable evaluations")
    for r in res["cells"]:
        print(
            f"  {r['view']:20s} @ {int(r['fps']):3d}fps | BR MAE={r['br_mae_ms']:5.1f}ms | "
            f"CI95 cov={r['ci95_coverage_pct']:5.1f}% | FK_BR={r['rmse_by_metric']['front_knee_br_deg']} | "
            f"LatFlex={r['rmse_by_metric']['thorax_lat_flex_br_deg']} | DelayArm={r['rmse_by_metric']['delayed_arm_ffc_deg']}"
        )
    print("\nUncalibrated Self-Calibration Arm (No Oracle K / No MCC Landmarks, 120fps):")
    for sc in res["uncalibrated_self_cal_arm"]:
        print(f"  {sc['view']:16s} | Az MAE={sc['mean_az_err_deg']} deg | FK_BR RMSE={sc['fk_br_rmse_deg']} deg | LatFlex RMSE={sc['lat_flex_br_rmse_deg']} deg")
