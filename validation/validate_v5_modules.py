"""Studio v5 Non-Circular AR(1) Monte Carlo Metrology Validation (`validation/validate_v5_modules.py`).

Evaluates empirical 95% CI coverage, RMSE, and `rmse / mean_sigma` across all Studio v5 `MetricEstimate`s
under the identical temporally correlated AR(1) camera-ray depth noise model (`tau = 80 ms`, `sigma_xy = 1.5 cm`,
`sigma_z = 3.6 cm`) used in `validation/run_synthetic_benchmark.py`, across camera views (`0°`, `45°`, `90°`),
frame rates (`120 fps`, `240 fps`), and both right & left bowling arms, while leaving the canonical 22-metric
benchmark 100% untouched.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from bowling_biomech.schemas import BowlingArm, DeliveryEvents, EventTimestamp, VideoAudit
from bowling_biomech.stage1_pitch_calibration import (
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
)
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
from bowling_biomech.stage3_v5_metrics import extract_and_quantify_v5_biomechanics
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
    """Temporally correlated AR(1) noise along axis 0 (`tau = 80 ms`)."""
    t_len = shape[0]
    out = np.zeros(shape, dtype=np.float64)
    innov_scale = sigma_total * np.sqrt(max(1.0 - rho * rho, 1e-6))
    out[0] = rng.normal(0.0, sigma_total, size=shape[1:])
    for t in range(1, t_len):
        out[t] = rho * out[t - 1] + rng.normal(0.0, innov_scale, size=shape[1:])
    return out


def _oracle_events_from_params(p: SyntheticDeliveryParams, n_frames: int, jw_gt: np.ndarray) -> DeliveryEvents:
    ev_det = detect_delivery_events(jw_gt, fps=p.fps, bowling_arm=p.bowler_arm)

    def _ev(t_s: float) -> EventTimestamp:
        ff = t_s * p.fps
        idx = int(np.clip(round(ff), 0, n_frames - 1))
        return EventTimestamp(
            frame_float=ff,
            frame_idx=idx,
            source_frame_idx=idx,
            timestamp_s=t_s,
            timing_uncertainty_ms=0.1,
            confidence=0.99,
            detection_method="oracle",
        )

    return DeliveryEvents(
        bfc=_ev(p.bfc_time_s),
        ffc=_ev(p.ffc_time_s),
        uah=ev_det.uah,
        br=_ev(p.br_time_s),
        ft_peak=ev_det.ft_peak,
        delivery_stride_duration_ms=round((p.ffc_time_s - p.bfc_time_s) * 1000.0, 2),
        ffc_to_br_duration_ms=round((p.br_time_s - p.ffc_time_s) * 1000.0, 2),
        is_complete_delivery=True,
    )


def run_v5_metrology_benchmark(n_seeds_per_cell: int = 8) -> Dict[str, Any]:
    lm_keys = [
        "bowling_crease_off",
        "bowling_crease_leg",
        "popping_crease_off",
        "popping_crease_leg",
        "stump_middle_base",
        "stump_middle_top",
    ]
    views = [("rear_0", 0.0, 6.0), ("open_45", 45.0, 8.0), ("side_90", 90.0, 5.0)]
    fps_list = [120.0, 240.0]
    arms = [BowlingArm.RIGHT, BowlingArm.LEFT]

    records_by_metric: Dict[str, List[Dict[str, float]]] = defaultdict(list)
    suppressed_counts: Dict[str, int] = defaultdict(int)
    total_trials = 0
    rng = np.random.default_rng(20261127)

    for view_name, az_deg, el_deg in views:
        K, R, t, _ = make_virtual_camera(azimuth_deg=az_deg, elevation_deg=el_deg, distance_m=12.0, fov_deg=55.0)
        cal_gt = calibrate_from_mcc_landmarks(project_mcc_landmarks(K, R, t, lm_keys, noise_px=0.0), 1280, 720, K=K)
        for fps in fps_list:
            for arm in arms:
                for _ in range(n_seeds_per_cell):
                    total_trials += 1
                    bfc_t = float(rng.uniform(0.38, 0.45))
                    stride_s = float(rng.uniform(0.155, 0.235))
                    ffc_br_s = float(rng.uniform(0.088, 0.128))
                    params = SyntheticDeliveryParams(
                        bowler_arm=arm,
                        fps=fps,
                        bfc_time_s=bfc_t,
                        ffc_time_s=bfc_t + stride_s,
                        br_time_s=bfc_t + stride_s + ffc_br_s,
                        runup_speed_mps=float(rng.uniform(5.2, 6.2)),
                        front_knee_ffc_deg=float(rng.uniform(152.0, 166.0)),
                        front_knee_br_deg=float(rng.uniform(158.0, 175.0)),
                        trunk_flexion_br_deg=float(rng.uniform(36.0, 48.0)),
                        thorax_lat_flex_br_deg=float(rng.uniform(30.0, 46.0)),
                        shoulder_counter_rot_deg=float(rng.uniform(12.0, 34.0)),
                        delayed_arm_ffc_deg=float(rng.uniform(128.0, 154.0)),
                        br_phi_deg=float(rng.uniform(-2.0, 14.0)),
                    )
                    jw_gt, _ = generate_synthetic_delivery(params)
                    jw_gt[:, :, 2] += float(rng.uniform(0.45, 0.72)) * (1.0 if arm == BowlingArm.RIGHT else -1.0)
                    n = len(jw_gt)

                    audit = VideoAudit(
                        video_path=f"bench:{view_name}:{int(fps)}:{total_trials}",
                        width=1280,
                        height=720,
                        fps=fps,
                        container_fps=fps,
                        num_frames=n,
                        raw_num_frames=n,
                        duration_s=round(n / fps, 3),
                        median_laplacian_var=260.0,
                        blur_grade="crisp",
                        blur_roi_method="bowler_roi",
                        fps_grade="high_speed_240" if fps >= 200 else "slow_mo_120",
                        est_arm_deg_per_frame=round(2300.0 / fps, 2),
                        est_ball_m_per_frame=round(38.89 / fps, 3),
                        timebase_status="verified_native",
                        source_frame_indices=list(range(n)),
                    )

                    oracle_ev = _oracle_events_from_params(params, n, jw_gt)
                    kin_gt = extract_fast_bowling_kinematics(
                        jw_gt, oracle_ev, fps=fps, bowling_arm=arm, joint_uncertainties_m=np.full((n, 24), 1e-4)
                    )
                    v5_gt = extract_and_quantify_v5_biomechanics(kin=kin_gt, audit=audit, cal=cal_gt, events=oracle_ev)

                    _, j_cam_clean = project_world_points(jw_gt, K, R, t)
                    rho = float(np.exp(-1.0 / max(0.08 * fps, 1.0)))
                    sigma_xy = 0.015
                    sigma_z = 0.036
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

                    lm_noisy = project_mcc_landmarks(K, R, t, lm_keys, noise_px=1.0, rng=rng)
                    cal_est = calibrate_from_mcc_landmarks(lm_noisy, 1280, 720, K=K)
                    jw_noisy = transform_cam_to_pitch_world(j_cam_noisy, cal_est)

                    ev_est = detect_delivery_events(jw_noisy, fps=fps, bowling_arm=arm)
                    kin_est = extract_fast_bowling_kinematics(
                        jw_noisy, ev_est, fps=fps, bowling_arm=arm, joint_uncertainties_m=j_unc
                    )
                    v5_est = extract_and_quantify_v5_biomechanics(kin=kin_est, audit=audit, cal=cal_est, events=ev_est)

                    for m_key, me_est in v5_est.v5_metrics.items():
                        if me_est.suppressed or me_est.value is None or me_est.ci_low is None or me_est.ci_high is None:
                            suppressed_counts[m_key] += 1
                            continue
                        me_ref = v5_gt.v5_metrics.get(m_key)
                        if me_ref is None or me_ref.value is None:
                            continue
                        err = float(me_est.value) - float(me_ref.value)
                        covered = 1.0 if (float(me_est.ci_low) <= float(me_ref.value) <= float(me_est.ci_high)) else 0.0
                        records_by_metric[m_key].append(
                            {
                                "err": err,
                                "covered": covered,
                                "sigma": float(me_est.sigma_total or 0.0),
                            }
                        )

    per_metric: Dict[str, Any] = {}
    all_covered: List[float] = []
    for m_key in sorted(set(records_by_metric.keys()) | set(suppressed_counts.keys())):
        recs = records_by_metric.get(m_key, [])
        n_obs = len(recs)
        n_sup = suppressed_counts.get(m_key, 0)
        if n_obs == 0:
            per_metric[m_key] = {"n_obs": 0, "n_suppressed": n_sup, "coverage_95_pct": None, "rmse": None}
            continue
        errs = np.array([r["err"] for r in recs], dtype=np.float64)
        covs = [r["covered"] for r in recs]
        sigs = np.array([r["sigma"] for r in recs], dtype=np.float64)
        all_covered.extend(covs)
        rmse = float(math.sqrt(float(np.mean(errs**2))))
        mean_sig = float(np.mean(sigs))
        per_metric[m_key] = {
            "n_obs": n_obs,
            "n_suppressed": n_sup,
            "coverage_95_pct": round(100.0 * float(np.mean(covs)), 1),
            "rmse": round(rmse, 2),
            "mean_sigma": round(mean_sig, 2),
            "rmse_over_sigma": round(rmse / max(mean_sig, 1e-6), 2),
        }

    pooled_cov = round(100.0 * float(np.mean(all_covered)), 1) if all_covered else 0.0
    return {
        "benchmark": "studio_v5_deep_modules_ar1",
        "total_deliveries": total_trials,
        "total_observable_evals": len(all_covered),
        "pooled_95_ci_coverage_pct": pooled_cov,
        "per_metric": per_metric,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--out", type=str, default="docs/results/v5/v5_modules_benchmark.json")
    args = parser.parse_args()

    res = run_v5_metrology_benchmark(n_seeds_per_cell=args.seeds)
    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(res, indent=2), encoding="utf-8")

    print(
        f"[Studio v5 AR(1) Benchmark] Deliveries={res['total_deliveries']} | "
        f"Observable Evals={res['total_observable_evals']} | "
        f"Pooled 95% CI Coverage={res['pooled_95_ci_coverage_pct']:.1f}%"
    )
    for k, m in res["per_metric"].items():
        print(
            f"  - {k:46s}: n={m['n_obs']:3d} (sup={m['n_suppressed']:2d}) | "
            f"cov={str(m['coverage_95_pct']):>5s}% | rmse={str(m['rmse']):>6s} | rmse/sig={m.get('rmse_over_sigma')}"
        )


if __name__ == "__main__":
    main()
