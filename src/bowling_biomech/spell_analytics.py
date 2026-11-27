"""Studio v5 (Module 10): Spell / Over Consistency, 3D Release Tunnel Bullseye & Fatigue Drift Analytics.

Aligned 1:1 with `web/src/types.ts` (`SpellAnalyticsResponse`, `ReleaseTunnel`, `ReleaseTunnelPoint`,
`Ellipse95`, `RepeatabilityRow`, `FatigueDriftRow`).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .constants import MCC_STUMP_CENTER_HALF_SPACING_M, MCC_STUMP_HEIGHT_M, MCC_STUMP_OUTER_WIDTH_M
from .schemas import BiomechanicalFactSheet


_TRACKED_DRIFT_SPECS = [
    ("front_knee_br_deg", "Front-leg brace at release", "deg", -1.0, "Front-knee brace softened across the spell"),
    ("release_height_m", "Ball release height", "m", -1.0, "Release height dropped across the spell"),
    ("thorax_lat_flex_br_deg", "Sideways trunk tilt at release", "deg", +1.0, "Sideways trunk tilt increased across the spell"),
    ("shoulder_counter_rotation_deg", "Shoulder counter-rotation", "deg", +1.0, "Shoulder twist crept upward across the spell"),
    ("trunk_flexion_delta_deg", "Forward chest whip", "deg", -1.0, "Forward chest drive decreased across the spell"),
    ("peak_arm_angular_vel_dps", "Peak upper-arm whip speed", "deg/s", -1.0, "Upper-arm whip speed declined across the spell"),
    ("runup_speed_mps", "Approach speed", "m/s", -1.0, "Run-up speed slowed across the spell"),
    ("tempo_gather_to_whip_ratio", "Gather-to-snap tempo ratio", "ratio", +1.0, "Delivery tempo ratio shifted across the spell"),
]


def compute_spell_analytics(
    delivery_rows: List[Dict[str, Any]],
    bowler_id: str = "bwl_demo",
    session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Compute Release-Point Tunnel Bullseye, Debiased Repeatability CV%, and Fatigue Drift across deliveries."""
    completed: List[Tuple[Dict[str, Any], BiomechanicalFactSheet]] = []
    excluded: List[Dict[str, str]] = []

    # Deduplicate constituent single-camera views when their fused `dual_3d_ray_fusion` delivery is present
    superseded_ids: Set[str] = set()
    present_ids = {str(r.get("id", "")) for r in delivery_rows}
    for row in delivery_rows:
        if str(row.get("pipeline_mode", "")) == "dual_3d_ray_fusion" and row.get("data_dir"):
            meta_p = Path(str(row["data_dir"])) / "meta.json"
            if meta_p.exists():
                try:
                    meta_obj = json.loads(meta_p.read_text(encoding="utf-8"))
                    for src_id in meta_obj.get("source_delivery_ids") or []:
                        if str(src_id) in present_ids:
                            superseded_ids.add(str(src_id))
                except Exception:
                    pass

    for row in delivery_rows:
        did = str(row.get("id", ""))
        lbl = str(row.get("label") or did)
        if did in superseded_ids:
            excluded.append({"delivery_id": did, "label": lbl, "reason_plain": "Combined into Dual-Phone 3D fused delivery."})
            continue
        if not row.get("fact_sheet_json"):
            excluded.append({"delivery_id": did, "label": lbl, "reason_plain": "Analysis not ready."})
            continue
        try:
            fs = BiomechanicalFactSheet.model_validate_json(row["fact_sheet_json"])
        except Exception:
            excluded.append({"delivery_id": did, "label": lbl, "reason_plain": "Invalid FactSheet data."})
            continue
        if not fs.events.is_complete_delivery:
            excluded.append({"delivery_id": did, "label": lbl, "reason_plain": "Incomplete clip (no verified ball release)."})
            continue
        completed.append((row, fs))

    # 1. Release-Point Bullseye ("Release Tunnel")
    world_pts: List[Dict[str, Any]] = []
    body_pts: List[Dict[str, Any]] = []
    excluded_world: List[Dict[str, str]] = list(excluded)

    for order_idx, (row, fs) in enumerate(completed):
        did = str(row["id"])
        lbl = str(row.get("label") or f"Ball {order_idx + 1}")
        all_m = fs.all_metric_estimates()
        rh = all_m.get("release_height_m")
        z_stump = all_m.get("release_lateral_from_stump_m")
        z_ank = all_m.get("release_lateral_from_ankle_m")
        br_idx = int(fs.events.br.frame_idx)

        if rh is None or rh.suppressed or rh.value is None:
            excluded_world.append({"delivery_id": did, "label": lbl, "reason_plain": "Release height withheld on this clip."})
            continue

        sig_y = float(rh.sigma_total or 0.05)
        if z_stump is not None and (not z_stump.suppressed) and z_stump.value is not None:
            sig_z = float(z_stump.sigma_total or 0.06)
            maj = round(1.96 * max(sig_z, sig_y), 3)
            mnr = round(1.96 * min(sig_z, sig_y), 3)
            ang = 0.0 if sig_z >= sig_y else 90.0
            world_pts.append(
                {
                    "delivery_id": did,
                    "label": lbl,
                    "order": order_idx + 1,
                    "lateral_m": round(float(z_stump.value), 3),
                    "height_m": round(float(rh.value), 3),
                    "sigma_lateral_m": round(sig_z, 3),
                    "sigma_height_m": round(sig_y, 3),
                    "ellipse_95": {"semi_major_m": maj, "semi_minor_m": mnr, "angle_deg": ang},
                    "frame_idx": br_idx,
                    "observability": z_stump.observability.value,
                }
            )
        else:
            excluded_world.append(
                {
                    "delivery_id": did,
                    "label": lbl,
                    "reason_plain": "Needs static pitch/stump calibration for stump-referenced release coordinates.",
                }
            )

        if z_ank is not None and (not z_ank.suppressed) and z_ank.value is not None:
            sig_za = float(z_ank.sigma_total or 0.06)
            maj_a = round(1.96 * max(sig_za, sig_y), 3)
            mnr_a = round(1.96 * min(sig_za, sig_y), 3)
            ang_a = 0.0 if sig_za >= sig_y else 90.0
            body_pts.append(
                {
                    "delivery_id": did,
                    "label": lbl,
                    "order": order_idx + 1,
                    "lateral_m": round(float(z_ank.value), 3),
                    "height_m": round(float(rh.value), 3),
                    "sigma_lateral_m": round(sig_za, 3),
                    "sigma_height_m": round(sig_y, 3),
                    "ellipse_95": {"semi_major_m": maj_a, "semi_minor_m": mnr_a, "angle_deg": ang_a},
                    "frame_idx": br_idx,
                    "observability": z_ank.observability.value,
                }
            )

    use_world = len(world_pts) >= 2 or (len(world_pts) >= 1 and len(body_pts) == 0)
    active_pts = world_pts if use_world else body_pts
    tunnel_mode = "world_stump_m" if use_world else "body_corridor_m"

    centroid = None
    raw_rad_sd: Optional[float] = None
    deb_rad_sd: Optional[float] = None
    spread_ci_cm: Optional[List[float]] = None
    group_ellipse = None

    if len(active_pts) >= 2:
        zs = np.array([p["lateral_m"] for p in active_pts], dtype=np.float64)
        ys = np.array([p["height_m"] for p in active_pts], dtype=np.float64)
        sz = np.array([p["sigma_lateral_m"] for p in active_pts], dtype=np.float64)
        sy = np.array([p["sigma_height_m"] for p in active_pts], dtype=np.float64)

        var_z_obs = float(np.var(zs, ddof=1))
        var_y_obs = float(np.var(ys, ddof=1))
        var_z_true = max(var_z_obs - float(np.mean(sz**2)), 0.0)
        var_y_true = max(var_y_obs - float(np.mean(sy**2)), 0.0)

        raw_rad = math.sqrt(var_z_obs + var_y_obs)
        deb_rad = math.sqrt(var_z_true + var_y_true)
        raw_rad_sd = round(raw_rad, 4)
        deb_rad_sd = round(deb_rad, 4)

        se_cm = max(100.0 * raw_rad / math.sqrt(2.0 * max(len(active_pts) - 1, 1)), 1.5)
        val_cm = 100.0 * deb_rad
        spread_ci_cm = [round(max(0.0, val_cm - 1.96 * se_cm), 1), round(val_cm + 1.96 * se_cm, 1)]
        centroid = {"lateral_m": round(float(np.mean(zs)), 3), "height_m": round(float(np.mean(ys)), 3)}
        rx_g = 2.4477 * math.sqrt(max(var_z_obs, 0.0004))
        ry_g = 2.4477 * math.sqrt(max(var_y_obs, 0.0004))
        group_ellipse = {
            "semi_major_m": round(max(rx_g, ry_g), 3),
            "semi_minor_m": round(min(rx_g, ry_g), 3),
            "angle_deg": 0.0 if rx_g >= ry_g else 90.0,
        }
    elif len(active_pts) == 1:
        centroid = {"lateral_m": active_pts[0]["lateral_m"], "height_m": active_pts[0]["height_m"]}

    # 2. Repeatability CV% Table & 3. Inverse-Variance Weighted Spell Fatigue Drift
    repeatability_table: List[Dict[str, Any]] = []
    fatigue_drift: List[Dict[str, Any]] = []

    for m_key, m_title, unit, fatigue_dir, fatigue_desc in _TRACKED_DRIFT_SPECS:
        pts_m: List[Dict[str, Any]] = []
        for order_idx, (row, fs) in enumerate(completed):
            me = fs.all_metric_estimates().get(m_key)
            if me is None or me.suppressed or me.value is None:
                continue
            pts_m.append(
                {
                    "delivery_id": str(row["id"]),
                    "order": order_idx + 1,
                    "label": str(row.get("label") or f"Ball {order_idx + 1}"),
                    "value": float(me.value),
                    "ci_low": float(me.ci_low if me.ci_low is not None else me.value),
                    "ci_high": float(me.ci_high if me.ci_high is not None else me.value),
                    "sigma": max(float(me.sigma_total or 1.0), 1e-3),
                }
            )

        if len(pts_m) >= 2:
            vals = np.array([p["value"] for p in pts_m], dtype=np.float64)
            sigs = np.array([p["sigma"] for p in pts_m], dtype=np.float64)
            obs_var = float(np.var(vals, ddof=1))
            meas_var = float(np.mean(sigs**2))
            deb_sd = math.sqrt(max(obs_var - meas_var, 0.0))
            mean_v = float(np.mean(vals))
            cv_pct = 100.0 * deb_sd / max(abs(mean_v), 1.0)
            meas_lim = meas_var >= 0.80 * max(obs_var, 1e-9)
            grade = "measurement_limited" if meas_lim else ("tight" if cv_pct <= 3.5 else ("moderate" if cv_pct <= 7.5 else "variable"))
            repeatability_table.append(
                {
                    "metric_key": m_key,
                    "label_plain": m_title,
                    "unit": unit,
                    "n_balls": len(pts_m),
                    "mean": round(mean_v, 2),
                    "raw_sd": round(math.sqrt(obs_var), 2),
                    "mean_measurement_sigma": round(math.sqrt(meas_var), 2),
                    "debiased_sd": round(deb_sd, 2),
                    "debiased_cv_pct": round(cv_pct, 2),
                    "grade": grade,
                }
            )

        if len(pts_m) >= 3:
            xs = np.array([float(p["order"]) for p in pts_m], dtype=np.float64)
            ys_m = np.array([p["value"] for p in pts_m], dtype=np.float64)
            ws = 1.0 / np.maximum(np.array([p["sigma"] for p in pts_m], dtype=np.float64) ** 2, 1e-6)
            w_sum = float(np.sum(ws))
            x_bar = float(np.sum(ws * xs) / w_sum)
            y_bar = float(np.sum(ws * ys_m) / w_sum)
            s_xx = float(np.sum(ws * ((xs - x_bar) ** 2)))
            if s_xx > 1e-9:
                slope = float(np.sum(ws * (xs - x_bar) * (ys_m - y_bar)) / s_xx)
                se_slope = 1.0 / math.sqrt(s_xx)
            else:
                slope, se_slope = 0.0, 999.0

            p_first, p_last = pts_m[0], pts_m[-1]
            endpoints_non_overlap = (p_last["ci_high"] < p_first["ci_low"]) or (p_last["ci_low"] > p_first["ci_high"])
            slope_sig = abs(slope) >= 1.96 * se_slope
            flagged = bool(slope_sig and endpoints_non_overlap)

            total_shift = slope * (xs[-1] - xs[0])
            if flagged:
                if slope * fatigue_dir > 0:
                    status_plain = f"{fatigue_desc} ({total_shift:+.1f} {unit} across spell, beyond measurement noise)."
                else:
                    status_plain = f"{m_title} shifted {total_shift:+.1f} {unit} across the spell (beyond measurement noise)."
            else:
                status_plain = "Variation across balls stays within single-delivery uncertainty intervals."

            fatigue_drift.append(
                {
                    "metric_key": m_key,
                    "label_plain": m_title,
                    "unit": unit,
                    "n_balls": len(pts_m),
                    "slope_per_ball": round(slope, 3),
                    "slope_se": round(se_slope, 3),
                    "spell_shift": round(total_shift, 2),
                    "status": "verified_drift" if flagged else "within_measurement_noise",
                    "status_plain": status_plain,
                    "points": [
                        {
                            "delivery_id": p["delivery_id"],
                            "order": p["order"],
                            "value": round(p["value"], 2),
                            "ci_low": round(p["ci_low"], 2),
                            "ci_high": round(p["ci_high"], 2),
                        }
                        for p in pts_m
                    ],
                }
            )

    return {
        "bowler_id": bowler_id,
        "session_id": session_id,
        "completed_count": len(completed),
        "release_spread_ci_cm": spread_ci_cm,
        "release_tunnel": {
            "mode": tunnel_mode,
            "mode_label_plain": (
                "Pitch-calibrated stump view (metres from middle stump)"
                if tunnel_mode == "world_stump_m"
                else "Body-referenced release corridor (metres from planted front foot)"
            ),
            "points": active_pts,
            "centroid": centroid,
            "raw_radial_sd_m": raw_rad_sd,
            "debiased_radial_sd_m": deb_rad_sd,
            "ellipse_95": group_ellipse,
            "stump_dims_m": {
                "height": MCC_STUMP_HEIGHT_M,
                "outer_width": MCC_STUMP_OUTER_WIDTH_M,
                "center_half_spacing": MCC_STUMP_CENTER_HALF_SPACING_M,
            },
            "excluded": excluded_world if use_world else excluded,
        },
        "repeatability_table": repeatability_table,
        "fatigue_drift": fatigue_drift,
    }
