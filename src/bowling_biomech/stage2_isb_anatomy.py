"""Phase 2 (M2B): NLF `bml_movi_87` ISB Rigid-Body Pelvis, Thorax & Foot Anatomy.

Uses NLF's built-in 87-marker anatomical head (`bml_movi_87`, zero extra weights) to fit ISB
(Wu et al., 2002/2005) orthonormal coordinate frames for:
  - Pelvis (`lasis`, `rasis`, `lpsis`, `rpsis`): true anterior-posterior depth lever ->
    sagittal anterior pelvic tilt, rigid-body pelvis transverse alignment, and
    coordinate-invariant 3D relative `lumbopelvic_angle` (`R_pelvis^T @ R_thorax`, Alway et al., 2021).
  - Thorax (`backneck` C7, `upperback` T10, `clavicle`, `sternum`, `lshom`, `rshom`):
    rigid-body thorax transverse alignment & lateral inclination.
  - Front Foot (`lhee`/`ltoe` or `rhee`/`rtoe`): unclamped `front_foot_angle_ffc_deg`
    (`0°` = down pitch `+X`, `+` = externally rotated / splayed open, `-` = closed toward bowling side)
    and heel-vs-forefoot strike indicator.

Statistical & Neuro-Symbolic Guardrails:
  1. Correlated Monocular Depth Trap: marker depth errors on the same segment are strongly
     correlated (`rho_hat = 0.85`). Uncertainty propagation strictly uses effective sample size
         N_eff = N / (1 + (N - 1) * rho_hat)
     and NEVER divides rotational uncertainty by naive `sqrt(N)`.
  2. Guardrail #3 (Zero Naked Point Estimates): `compute_isb_metric_estimates(...)` wraps every
     ISB scalar in a `MetricEstimate` carrying `(value, ci_low, ci_high, sigma_total, observability)`
     or explicit `suppressed=True` when `!events.is_complete_delivery` or camera geometry/uncertainty
     cannot observe the segment's anatomical plane.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, Optional, Tuple

import numpy as np

from .schemas import BowlingArm, CameraViewClass, DeliveryEvents, MetricEstimate, ObservabilityGrade, PitchCalibration

# Verified indices from `nlf_l_multi.torchscript` -> `m.per_skeleton_joint_names["bml_movi_87"]`
BML_MOVI_87_IDX: Dict[str, int] = {
    "backneck": 0,   # C7
    "upperback": 1,  # T10 / mid-thoracic spine
    "clavicle": 2,   # IJ / suprasternal notch
    "sternum": 3,    # PX / xiphoid process
    "umbilicus": 4,
    "lshom": 8,
    "lasis": 15,
    "lpsis": 16,
    "lhee": 21,
    "lfifthmetatarsal": 22,
    "ltoe": 23,
    "lfirstmetatarsal": 32,
    "lscapula": 34,
    "rshom": 39,
    "rasis": 46,
    "rpsis": 47,
    "rhee": 52,
    "rfifthmetatarsal": 53,
    "rtoe": 54,
    "rfirstmetatarsal": 63,
    "rscapula": 65,
}


@dataclass
class ISBAnatomyResult:
    """Rigid-body ISB kinematics & effective-N_eff uncertainties derived from `bml_movi_87`."""

    pelvis_yaw_isb_deg: np.ndarray        # (T,) unwrapped ISB pelvis transverse alignment
    thorax_yaw_isb_deg: np.ndarray        # (T,) unwrapped ISB thorax transverse alignment
    pelvic_tilt_sagittal_deg: np.ndarray  # (T,) anterior (+) / posterior (-) pelvic tilt
    lumbopelvic_angle_deg: np.ndarray     # (T,) 3D invariant relative thorax-to-pelvis flexion (+ = flexion, - = extension)
    front_foot_angle_deg: np.ndarray      # (T,) heel->toe angle vs +X (0=straight, +=externally rotated/open, -=closed)
    scalars: Dict[str, float] = field(default_factory=dict)
    metrics: Dict[str, MetricEstimate] = field(default_factory=dict)
    n_eff_pelvis: float = 1.0
    n_eff_thorax: float = 1.0
    n_eff_foot: float = 1.0
    rho_assumed: float = 0.85
    front_foot_strike_mode: str = "unknown"  # heel_first | flat_foot | forefoot_first | suppressed


def effective_sample_size(n_markers: int, rho_corr: float = 0.85) -> float:
    """Compute effective independent markers N_eff = N / (1 + (N - 1) * rho) under correlated depth errors."""
    n = max(int(n_markers), 1)
    rho = float(np.clip(rho_corr, 0.0, 0.999))
    return float(n / (1.0 + (n - 1) * rho))


def _normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, eps)


def _unwrap_portus_branch(rad_raw: np.ndarray, ref_idx: int, anchor_target_deg: Optional[float] = None) -> np.ndarray:
    deg = 270.0 + np.degrees(np.unwrap(rad_raw))
    idx_safe = int(np.clip(ref_idx, 0, len(deg) - 1))
    if anchor_target_deg is not None:
        shift = 360.0 * round((float(deg[idx_safe]) - float(anchor_target_deg)) / 360.0)
    else:
        shift = 360.0 * math.floor((float(deg[idx_safe]) - 60.0) / 360.0)
    return deg - shift


def fit_isb_pelvis_frame(
    markers_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit right-handed orthonormal ISB Pelvis frame (x_fwd, y_up, z_right) from `lasis, rasis, lpsis, rpsis` (T, 87, 3).

    Following Wu et al. (2002) in right-handed Pitch World (+X down pitch, +Y up, +Z = +X x +Y = bowler's right):
      - z_right = normalize(0.5 * ((rasis - lasis) + (rpsis - lpsis)))  (bowler's right)
      - x_raw   = mid_asis - mid_psis                                   (anterior)
      - y_up    = normalize(cross(z_right, x_raw))                      (+Z x +X = +Y superior, no 90-deg flip!)
      - x_fwd   = normalize(cross(y_up, z_right))                       (exact orthonormal anterior)
    """
    lasis = markers_world[:, BML_MOVI_87_IDX["lasis"]]
    rasis = markers_world[:, BML_MOVI_87_IDX["rasis"]]
    lpsis = markers_world[:, BML_MOVI_87_IDX["lpsis"]]
    rpsis = markers_world[:, BML_MOVI_87_IDX["rpsis"]]

    mid_asis = 0.5 * (lasis + rasis)
    mid_psis = 0.5 * (lpsis + rpsis)

    z_right = _normalize(0.5 * ((rasis - lasis) + (rpsis - lpsis)))
    x_raw = mid_asis - mid_psis
    y_up = _normalize(np.cross(z_right, x_raw))
    x_fwd = _normalize(np.cross(y_up, z_right))
    return x_fwd, y_up, z_right


def fit_isb_thorax_frame(
    markers_world: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit right-handed orthonormal ISB Thorax frame (x_fwd, y_up, z_right) from C7/T10/CLAV/STRN/LSHO/RSHO (Wu et al., 2005)."""
    c7 = markers_world[:, BML_MOVI_87_IDX["backneck"]]
    t10 = markers_world[:, BML_MOVI_87_IDX["upperback"]]
    clav = markers_world[:, BML_MOVI_87_IDX["clavicle"]]
    strn = markers_world[:, BML_MOVI_87_IDX["sternum"]]
    lsho = markers_world[:, BML_MOVI_87_IDX["lshom"]]
    rsho = markers_world[:, BML_MOVI_87_IDX["rshom"]]

    top_mid = 0.5 * (c7 + clav)
    bot_mid = 0.5 * (t10 + strn)
    y_up = _normalize(top_mid - bot_mid)

    x_ant = 0.5 * ((clav - c7) + (strn - t10))
    z_from_ant = _normalize(np.cross(x_ant, y_up))
    z_from_sho = _normalize(rsho - lsho)
    dots = np.sum(z_from_ant * z_from_sho, axis=-1, keepdims=True)
    z_from_ant = np.where(dots < 0.0, -z_from_ant, z_from_ant)
    z_right = _normalize(0.55 * z_from_sho + 0.45 * z_from_ant)

    x_fwd = _normalize(np.cross(y_up, z_right))
    z_right = _normalize(np.cross(x_fwd, y_up))
    return x_fwd, y_up, z_right


def _grade_from_sigma(sig: float, high_cut: float, mod_cut: float) -> ObservabilityGrade:
    if sig <= high_cut:
        return ObservabilityGrade.HIGH
    if sig <= mod_cut:
        return ObservabilityGrade.MODERATE
    return ObservabilityGrade.LOW


def _suppressed_isb(key: str, label: str, reason: str, sig_est: float = 0.0) -> MetricEstimate:
    return MetricEstimate(
        key=key,
        label=label,
        value=None,
        unit="deg",
        ci_low=None,
        ci_high=None,
        sigma_total=round(float(sig_est), 2),
        observability=ObservabilityGrade.UNMEASURABLE,
        interval_type="analytical_uncertainty",
        suppressed=True,
        suppression_reason=reason,
        empirical_reference="ISB rigid-body segment anatomy (bml_movi_87 marker clusters).",
        literature_source="Wu et al. (2002/2005); Alway et al. (2021)",
    )


def extract_isb_anatomy_kinematics(
    markers_world_87: np.ndarray,
    events: DeliveryEvents,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    rho_corr: float = 0.85,
    markers_unc_87: Optional[np.ndarray] = None,
    cal: Optional[PitchCalibration] = None,
) -> ISBAnatomyResult:
    """Extract ISB rigid-body Pelvis, Thorax & Front-Foot kinematics + N_eff-propagated MetricEstimates."""
    n = markers_world_87.shape[0]
    is_right = bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0

    p_xfwd, p_yup, p_zright = fit_isb_pelvis_frame(markers_world_87)
    t_xfwd, t_yup, t_zright = fit_isb_thorax_frame(markers_world_87)

    i_bfc = int(np.clip(events.bfc.frame_idx, 0, n - 1))
    i_ffc = int(np.clip(events.ffc.frame_idx, i_bfc, n - 1))
    i_br = int(np.clip(events.br.frame_idx, i_ffc, n - 1))

    # Portus (2001/2004) transverse alignment from rigid-body +Z_right vectors (same sign convention as Stage 3)
    pelv_rad = np.arctan2(sign_arm * p_zright[:, 0], p_zright[:, 2])
    thor_rad = np.arctan2(sign_arm * t_zright[:, 0], t_zright[:, 2])
    pelv_yaw = _unwrap_portus_branch(pelv_rad, ref_idx=i_bfc)
    thor_yaw = _unwrap_portus_branch(thor_rad, ref_idx=i_bfc, anchor_target_deg=float(pelv_yaw[i_bfc]))

    # Sagittal anterior pelvic tilt (+ = anterior tilt leaning forward; local pelvis anatomical plane vs world horizontal)
    pelv_horiz = np.hypot(p_xfwd[:, 0], p_xfwd[:, 2])
    pelvic_tilt_deg = np.degrees(np.arctan2(-p_xfwd[:, 1], np.maximum(pelv_horiz, 1e-6)))

    # 3D Coordinate-Invariant Relative Lumbopelvic Angle (R_pelvis^T @ R_thorax, Alway et al. 2021):
    # Projects thorax superior axis `t_yup` onto pelvis local sagittal plane `(p_xfwd, p_yup)`:
    #   + = thorax flexed anteriorly relative to pelvis; - = lumbar extension / arch; 0 = exact neutral rigid spine at ANY yaw!
    lumbopelvic_deg = np.degrees(
        np.arctan2(np.sum(t_yup * p_xfwd, axis=-1), np.sum(t_yup * p_yup, axis=-1))
    )

    # Front-foot alignment angle relative to +X down-pitch axis (UNCLAMPED (-180, 180]):
    #   0 deg = toe pointing straight down pitch (+X)
    #   + deg = externally rotated / splayed open away from target (-Z for right-arm left foot, +Z for left-arm right foot)
    #   - deg = internally rotated / closed across body (+Z for right-arm left foot, -Z for left-arm right foot)
    hee_i = BML_MOVI_87_IDX["lhee" if is_right else "rhee"]
    toe_i = BML_MOVI_87_IDX["ltoe" if is_right else "rtoe"]
    f_vec = markers_world_87[:, toe_i] - markers_world_87[:, hee_i]
    front_foot_angle_deg = np.degrees(np.arctan2(-sign_arm * f_vec[:, 2], f_vec[:, 0]))

    # Heel-vs-forefoot strike classification at FFC
    dy_strike = float(markers_world_87[i_ffc, toe_i, 1] - markers_world_87[i_ffc, hee_i, 1])
    if not events.is_complete_delivery:
        strike_mode = "suppressed"
    elif dy_strike > 0.025:
        strike_mode = "heel_first"
    elif dy_strike < -0.025:
        strike_mode = "forefoot_first"
    else:
        strike_mode = "flat_foot"

    scr_isb = float(max(0.0, float(thor_yaw[i_bfc]) - float(np.min(thor_yaw[i_bfc : i_ffc + 1]))))
    scalars = {
        "pelvis_alignment_bfc_isb_deg": round(float(pelv_yaw[i_bfc]), 2),
        "thorax_alignment_bfc_isb_deg": round(float(thor_yaw[i_bfc]), 2),
        "shoulder_counter_rotation_isb_deg": round(scr_isb, 2),
        "anterior_pelvic_tilt_ffc_deg": round(float(pelvic_tilt_deg[i_ffc]), 2),
        "lumbopelvic_angle_ffc_deg": round(float(lumbopelvic_deg[i_ffc]), 2),
        "front_foot_angle_ffc_deg": round(float(front_foot_angle_deg[i_ffc]), 2),
        "lumbopelvic_angle_br_deg": round(float(lumbopelvic_deg[i_br]), 2),
    }

    n_eff_pelvis = effective_sample_size(4, rho_corr)
    n_eff_thorax = effective_sample_size(6, rho_corr)
    n_eff_foot = effective_sample_size(2, rho_corr)

    # Propagate correlated-depth marker uncertainty via N_eff + calibration/plane gates (Guardrail #3!)
    unc = markers_unc_87 if (markers_unc_87 is not None and markers_unc_87.shape[:2] == (n, 87)) else np.full((n, 87), 0.045, dtype=np.float64)
    p_idx = [BML_MOVI_87_IDX[k] for k in ("lasis", "rasis", "lpsis", "rpsis")]
    t_idx = [BML_MOVI_87_IDX[k] for k in ("backneck", "upperback", "clavicle", "sternum", "lshom", "rshom")]
    f_idx = [hee_i, toe_i]

    # Segment anatomical lever arms (meters) at relevant events
    def _lever_pelvis_ap(fi: int) -> float:
        m_a = 0.5 * (markers_world_87[fi, BML_MOVI_87_IDX["lasis"]] + markers_world_87[fi, BML_MOVI_87_IDX["rasis"]])
        m_p = 0.5 * (markers_world_87[fi, BML_MOVI_87_IDX["lpsis"]] + markers_world_87[fi, BML_MOVI_87_IDX["rpsis"]])
        return max(float(np.linalg.norm(m_a - m_p)), 0.12)

    def _lever_pelvis_ml(fi: int) -> float:
        return max(float(np.linalg.norm(markers_world_87[fi, BML_MOVI_87_IDX["rasis"]] - markers_world_87[fi, BML_MOVI_87_IDX["lasis"]])), 0.18)

    def _lever_thorax_h(fi: int) -> float:
        top = 0.5 * (markers_world_87[fi, BML_MOVI_87_IDX["backneck"]] + markers_world_87[fi, BML_MOVI_87_IDX["clavicle"]])
        bot = 0.5 * (markers_world_87[fi, BML_MOVI_87_IDX["upperback"]] + markers_world_87[fi, BML_MOVI_87_IDX["sternum"]])
        return max(float(np.linalg.norm(top - bot)), 0.22)

    def _lever_thorax_w(fi: int) -> float:
        return max(float(np.linalg.norm(markers_world_87[fi, BML_MOVI_87_IDX["rshom"]] - markers_world_87[fi, BML_MOVI_87_IDX["lshom"]])), 0.28)

    def _lever_foot(fi: int) -> float:
        return max(float(np.linalg.norm(markers_world_87[fi, toe_i] - markers_world_87[fi, hee_i])), 0.14)

    def _seg_angle_sigma_deg(fi: int, idxs: list, n_eff: float, lever_m: float, depth_mult: float = 1.0) -> float:
        sig_m = float(np.mean(unc[fi, idxs]))
        # Effective pair displacement uncertainty under correlated depth errors (NEVER naive 1/sqrt(N)!)
        sig_eff_m = (math.sqrt(2.0) * sig_m * depth_mult) / math.sqrt(max(n_eff, 1.0))
        return math.degrees(math.atan2(sig_eff_m, lever_m))

    is_dual = (cal is not None and cal.view_class == CameraViewClass.DUAL_VIEW_FUSED)
    az_rad = math.radians(cal.azimuth_deg) if cal is not None else math.radians(90.0)
    sin_az = abs(math.sin(az_rad))
    sig_grav = float(cal.gravity_sigma_deg) if cal is not None else 2.5
    sig_head = float(cal.heading_sigma_deg) if cal is not None else 4.0

    # Foreshortening penalty for sagittal depth-axis metrics when viewed near rear-on (sin_az -> 0)
    sag_depth_mult = 1.0 if is_dual else (1.0 / max(sin_az, 0.25))
    sag_observable = is_dual or (sin_az >= 0.45)

    meta_specs = [
        ("pelvis_alignment_bfc_isb_deg", "ISB Pelvis Alignment at BFC", i_bfc, "trans_pelvis", 8.0, 14.0, 24.0, (60.0, 420.0)),
        ("thorax_alignment_bfc_isb_deg", "ISB Thorax Alignment at BFC", i_bfc, "trans_thorax", 8.0, 14.0, 24.0, (60.0, 420.0)),
        ("shoulder_counter_rotation_isb_deg", "ISB Shoulder Counter-Rotation", i_bfc, "scr", 7.0, 13.0, 22.0, (0.0, 180.0)),
        ("anterior_pelvic_tilt_ffc_deg", "ISB Anterior Pelvic Tilt at FFC", i_ffc, "sag_pelvis", 6.0, 12.0, 20.0, (-60.0, 90.0)),
        ("lumbopelvic_angle_ffc_deg", "ISB Lumbopelvic Flexion at FFC", i_ffc, "sag_lumbo", 7.0, 13.0, 22.0, (-90.0, 120.0)),
        ("lumbopelvic_angle_br_deg", "ISB Lumbopelvic Flexion at BR", i_br, "sag_lumbo", 7.0, 13.0, 22.0, (-90.0, 120.0)),
        ("front_foot_angle_ffc_deg", "ISB Front-Foot Angle at FFC", i_ffc, "foot", 8.0, 15.0, 25.0, (-180.0, 180.0)),
    ]

    metrics: Dict[str, MetricEstimate] = {}
    for key, label, fi, kind, hi_cut, mod_cut, sup_cut, bounds in meta_specs:
        if not events.is_complete_delivery:
            metrics[key] = _suppressed_isb(key, label, "Suppressed: incomplete delivery failed overhead ball-release sanity gates.")
            continue

        val = float(scalars[key])
        if kind == "trans_pelvis":
            s_pose = _seg_angle_sigma_deg(fi, p_idx, n_eff_pelvis, _lever_pelvis_ml(fi))
            sig_tot = math.hypot(s_pose, sig_head)
        elif kind == "trans_thorax":
            s_pose = _seg_angle_sigma_deg(fi, t_idx, n_eff_thorax, _lever_thorax_w(fi))
            sig_tot = math.hypot(s_pose, sig_head)
        elif kind == "scr":
            s_pose = _seg_angle_sigma_deg(fi, t_idx, n_eff_thorax, _lever_thorax_w(fi)) * 1.15
            sig_tot = math.hypot(s_pose, 0.35 * sig_head)
        elif kind == "sag_pelvis":
            s_pose = _seg_angle_sigma_deg(fi, p_idx, n_eff_pelvis, _lever_pelvis_ap(fi), depth_mult=sag_depth_mult)
            sig_tot = math.hypot(s_pose, sig_grav)
            if not sag_observable:
                metrics[key] = _suppressed_isb(
                    key, label,
                    f"Suppressed: sagittal anterior pelvic tilt requires side-on/45-deg or dual-view camera (|sin(az)|={sin_az:.2f} < 0.45).",
                    sig_est=sig_tot,
                )
                continue
        elif kind == "sag_lumbo":
            s_p = _seg_angle_sigma_deg(fi, p_idx, n_eff_pelvis, _lever_pelvis_ap(fi), depth_mult=sag_depth_mult)
            s_t = _seg_angle_sigma_deg(fi, t_idx, n_eff_thorax, _lever_thorax_h(fi), depth_mult=sag_depth_mult)
            sig_tot = math.hypot(s_p, s_t)
            if not sag_observable:
                metrics[key] = _suppressed_isb(
                    key, label,
                    f"Suppressed: sagittal lumbopelvic angle requires side-on/45-deg or dual-view camera (|sin(az)|={sin_az:.2f} < 0.45).",
                    sig_est=sig_tot,
                )
                continue
        else:  # foot
            s_pose = _seg_angle_sigma_deg(fi, f_idx, n_eff_foot, _lever_foot(fi))
            sig_tot = math.hypot(s_pose, sig_head)

        if sig_tot > sup_cut or not math.isfinite(val):
            metrics[key] = _suppressed_isb(
                key, label,
                f"Suppressed: N_eff-propagated 1-sigma uncertainty ({sig_tot:.1f} deg) exceeds {sup_cut:.0f} deg reliability gate (rho={rho_corr:.2f}).",
                sig_est=sig_tot,
            )
            continue

        lo = max(bounds[0], val - 1.96 * sig_tot)
        hi = min(bounds[1], val + 1.96 * sig_tot)
        metrics[key] = MetricEstimate(
            key=key,
            label=label,
            value=round(val, 2),
            unit="deg",
            ci_low=round(lo, 2),
            ci_high=round(hi, 2),
            sigma_total=round(sig_tot, 2),
            observability=_grade_from_sigma(sig_tot, hi_cut, mod_cut),
            interval_type="analytical_uncertainty",
            suppressed=False,
            empirical_reference="ISB rigid-body segment anatomy (bml_movi_87 marker clusters; N_eff correlated-depth uncertainty).",
            literature_source="Wu et al. (2002/2005); Alway et al. (2021)",
        )

    return ISBAnatomyResult(
        pelvis_yaw_isb_deg=pelv_yaw,
        thorax_yaw_isb_deg=thor_yaw,
        pelvic_tilt_sagittal_deg=pelvic_tilt_deg,
        lumbopelvic_angle_deg=lumbopelvic_deg,
        front_foot_angle_deg=front_foot_angle_deg,
        scalars=scalars,
        metrics=metrics,
        n_eff_pelvis=round(n_eff_pelvis, 3),
        n_eff_thorax=round(n_eff_thorax, 3),
        n_eff_foot=round(n_eff_foot, 3),
        rho_assumed=float(rho_corr),
        front_foot_strike_mode=strike_mode,
    )
