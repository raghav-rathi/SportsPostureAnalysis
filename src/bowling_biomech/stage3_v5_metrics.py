"""Studio v5 Deep Biomechanical Analysis Engine (Modules 1-6) & 4-Source Uncertainty Propagator.

Implements peer-reviewed fast-bowling analysis modules on top of Stage 3 SMPL-24 + ISB `bml_movi_87`
trajectories while preserving 0.0% regression on the canonical 22-metric AR(1) metrology benchmark:
  - Module 1: Proximal-to-Distal Kinematic Sequence Engine (Putnam, 1993; Ferdinands et al., 2013)
  - Module 2: 14-Segment Anthropometric 3D COM & Braking Impulse Analyzer (de Leva, 1996; King et al., 2016)
  - Module 3: Head Stability & Coronal Fall-Away Analyzer (Glazier, 2010)
  - Module 4: Lead (Non-Bowling) Arm Pull-Down Mechanics (Portus et al., 2004; Ferdinands et al., 2010)
  - Module 5: Crease Alignment, Foot-Plant Geometry & 3D Release-Point Tunnel (Bartlett et al., 1996; Phillips et al., 2012)
  - Module 6: Delivery Tempo & Rhythm Fingerprint (Felton et al., 2019)

Every scalar output obeys AGENTS.md §1.2 Guardrail #3: wrapped in `MetricEstimate` with full
`(value, ci_low, ci_high, sigma_total, observability)` or explicit `suppressed=True` + `suppression_reason`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .constants import JOINT_IDX, MCC_POPPING_CREASE_M
from .schemas import (
    BowlingArm,
    CameraViewClass,
    COMBrakingAnalysis,
    CreaseReleaseTunnelAnalysis,
    DeliveryEvents,
    HeadStabilityAnalysis,
    IntervalType,
    KinematicSequenceAnalysis,
    LeadArmAnalysis,
    MetricEstimate,
    ObservabilityGrade,
    PitchCalibration,
    SequenceSegmentPeak,
    SequenceVerdict,
    TempoPhaseSegment,
    TempoRhythmAnalysis,
    VideoAudit,
    WorkloadContext,
)
from .stage2_event_spotting import smooth_trajectory
from .stage2_isb_anatomy import BML_MOVI_87_IDX, ISBAnatomyResult
from .stage3_kinematics import KinematicsResult, _interp_frame
from .stage4_uncertainty import _fd_grad, _view_grade_from_geometry, _world_cov_for_joint


# ---------------------------------------------------------------------------
# Module 2: Exact 14-Segment de Leva (1996) Body Segment Parameters
# ---------------------------------------------------------------------------
# Segment tuples: (prox_spec, dist_spec, male_mass, male_com_lambda, female_mass, female_com_lambda)
# Endpoints are dictionaries of {smpl_joint_name: weight} summing to 1.0!
_DE_LEVA_14_SEGMENTS: List[Tuple[Dict[str, float], Dict[str, float], float, float, float, float]] = [
    # 1. Head + Neck: neck -> vertex (approximated as head + 0.45 * (head - neck) = 1.45*head - 0.45*neck)
    ({"neck": 1.0}, {"head": 1.45, "neck": -0.45}, 0.0694, 0.5002, 0.0668, 0.4841),
    # 2. Trunk: neck -> mid-hip (0.5*lhip + 0.5*rhip)
    ({"neck": 1.0}, {"lhip": 0.5, "rhip": 0.5}, 0.4346, 0.4310, 0.4257, 0.3782),
    # 3-4. Left / Right Upper Arm
    ({"lsho": 1.0}, {"lelb": 1.0}, 0.0271, 0.5772, 0.0255, 0.5754),
    ({"rsho": 1.0}, {"relb": 1.0}, 0.0271, 0.5772, 0.0255, 0.5754),
    # 5-6. Left / Right Forearm
    ({"lelb": 1.0}, {"lwri": 1.0}, 0.0162, 0.4574, 0.0138, 0.4559),
    ({"relb": 1.0}, {"rwri": 1.0}, 0.0162, 0.4574, 0.0138, 0.4559),
    # 7-8. Left / Right Hand
    ({"lwri": 1.0}, {"lhan": 1.0}, 0.0061, 0.7900, 0.0056, 0.7474),
    ({"rwri": 1.0}, {"rhan": 1.0}, 0.0061, 0.7900, 0.0056, 0.7474),
    # 9-10. Left / Right Thigh
    ({"lhip": 1.0}, {"lkne": 1.0}, 0.1416, 0.4095, 0.1478, 0.3612),
    ({"rhip": 1.0}, {"rkne": 1.0}, 0.1416, 0.4095, 0.1478, 0.3612),
    # 11-12. Left / Right Shank
    ({"lkne": 1.0}, {"lank": 1.0}, 0.0433, 0.4459, 0.0481, 0.4416),
    ({"rkne": 1.0}, {"rank": 1.0}, 0.0433, 0.4459, 0.0481, 0.4416),
    # 13-14. Left / Right Foot
    ({"lank": 1.0}, {"ltoe": 1.0}, 0.0137, 0.4415, 0.0129, 0.4014),
    ({"rank": 1.0}, {"rtoe": 1.0}, 0.0137, 0.4415, 0.0129, 0.4014),
]


def get_de_leva_smpl24_weights(sex: str = "male") -> np.ndarray:
    """Return exact 24-element partition-of-unity weight vector `alpha` (`sum(alpha) == 1.0`) for SMPL-24."""
    is_female = str(sex or "male").strip().lower().startswith("f")
    alpha = np.zeros(24, dtype=np.float64)
    total_mass = sum((seg[4] if is_female else seg[2]) for seg in _DE_LEVA_14_SEGMENTS)
    for prox_map, dist_map, m_m, lam_m, m_f, lam_f in _DE_LEVA_14_SEGMENTS:
        w_s = (m_f if is_female else m_m) / total_mass
        lam = lam_f if is_female else lam_m
        for j_name, coeff in prox_map.items():
            alpha[JOINT_IDX[j_name]] += w_s * (1.0 - lam) * coeff
        for j_name, coeff in dist_map.items():
            alpha[JOINT_IDX[j_name]] += w_s * lam * coeff
    # Normalize to machine precision so COM(P + c) == COM(P) + c holds to 1e-15
    alpha /= float(np.sum(alpha))
    return alpha


def compute_3d_com_trajectory(joints_world: np.ndarray, sex: str = "male") -> np.ndarray:
    """Compute `(T, 3)` 3D whole-body Center of Mass in Pitch World meters via de Leva (1996) 14-segment model."""
    alpha = get_de_leva_smpl24_weights(sex)
    return np.einsum("j,tjc->tc", alpha, np.asarray(joints_world, dtype=np.float64))


V5_PHYSICAL_CI_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "seq_pelvis_peak_dps": (0.0, 2200.0),
    "seq_thorax_peak_dps": (0.0, 2800.0),
    "seq_arm_peak_dps": (0.0, 4500.0),
    "seq_wrist_whip_peak_dps": (0.0, 5500.0),
    "seq_pelvis_to_thorax_lag_ms": (-250.0, 250.0),
    "seq_thorax_to_arm_lag_ms": (-250.0, 250.0),
    "seq_arm_to_wrist_lag_ms": (-150.0, 150.0),
    "seq_whip_gain_arm_over_thorax": (0.4, 6.5),
    "seq_pelvis_braking_decel_pct": (0.0, 100.0),
    "com_ffc_speed_mps": (0.0, 12.0),
    "com_br_speed_mps": (0.0, 12.0),
    "com_braking_delta_vx_mps": (-3.0, 8.0),
    "com_braking_efficiency_pct": (-25.0, 95.0),
    "com_vertical_sink_pct_stature": (-15.0, 35.0),
    "front_leg_plant_angle_ffc_deg": (-15.0, 65.0),
    "head_coronal_angle_br_deg": (-45.0, 85.0),
    "head_fall_away_ffc_to_br_deg": (-45.0, 75.0),
    "head_lateral_offset_from_ankle_br_pct_stature": (-30.0, 50.0),
    "head_lateral_offset_from_ankle_br_cm": (-60.0, 95.0),
    "head_vertical_drop_ffc_to_br_pct_stature": (-20.0, 35.0),
    "lead_arm_gather_peak_elev_deg": (-60.0, 90.0),
    "lead_arm_early_drop_pre_ffc_deg": (0.0, 150.0),
    "lead_arm_pulldown_peak_vel_dps": (0.0, 2800.0),
    "lead_elbow_ribcage_flare_br_pct_stature": (0.0, 45.0),
    "foot_plant_corridor_offset_cm": (-80.0, 80.0),
    "stride_alignment_angle_deg": (-45.0, 45.0),
    "front_heel_crease_margin_cm": (-80.0, 120.0),
    "release_height_pct_stature": (60.0, 140.0),
    "release_lateral_from_ankle_m": (-0.60, 1.10),
    "release_lateral_from_stump_m": (-1.80, 1.80),
    "release_extension_past_crease_m": (-1.20, 1.80),
    "tempo_stride_duration_ms": (40.0, 600.0),
    "tempo_braking_to_br_ms": (30.0, 350.0),
    "tempo_uah_to_br_whip_ms": (8.0, 180.0),
    "tempo_gather_to_whip_ratio": (0.3, 4.5),
}


def _clamp_v5_ci(key: str, lo: float, hi: float) -> Tuple[float, float]:
    bounds = V5_PHYSICAL_CI_BOUNDS.get(key)
    if bounds is None:
        return lo, hi
    lo_b, hi_b = bounds
    if lo_b is not None:
        lo = max(lo_b, lo)
        hi = max(lo_b, hi)
    if hi_b is not None:
        lo = min(hi_b, lo)
        hi = min(hi_b, hi)
    return lo, hi


def _savgol_1d(arr: np.ndarray, fps: float, span_s: float = 0.055, deriv: int = 1) -> Tuple[np.ndarray, float]:
    n = len(arr)
    win = max(5, 2 * int(round(0.5 * span_s * fps)) + 1)
    if win > n:
        win = n if (n % 2 == 1) else max(3, n - 1)
    if win >= 5 and n >= win:
        out = savgol_filter(arr, window_length=win, polyorder=2, deriv=deriv, delta=1.0 / max(fps, 1.0))
        eff_span = float(win - 1) / max(fps, 1.0)
        return out, eff_span
    if deriv == 1:
        return np.gradient(arr) * max(fps, 1.0), 2.0 / max(fps, 1.0)
    return np.gradient(np.gradient(arr) * max(fps, 1.0)) * max(fps, 1.0), 2.0 / max(fps, 1.0)


def _subframe_peak(series: np.ndarray, idx_lo: int, idx_hi: int) -> Tuple[float, float, int]:
    """Return `(peak_val, peak_frame_float, discrete_idx)` with parabolic 3-point vertex refinement."""
    n = len(series)
    lo = int(np.clip(idx_lo, 0, max(n - 1, 0)))
    hi = int(np.clip(idx_hi, lo, max(n - 1, 0)))
    sub = series[lo : hi + 1]
    if len(sub) == 0:
        return 0.0, float(lo), lo
    k_rel = int(np.argmax(sub))
    k = lo + k_rel
    val = float(series[k])
    f_pk = float(k)
    if 0 < k < n - 1:
        y0, y1, y2 = float(series[k - 1]), float(series[k]), float(series[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom < -1e-6:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            f_pk = float(k) + delta
            val = y1 - 0.25 * (y0 - y2) * delta
    return val, f_pk, k


@dataclass
class V5DeepBiomechBundle:
    """Complete output of Studio v5 Modules 1-6 ready for FactSheet, CoachingStory, and Telemetry v2.1."""

    v5_metrics: Dict[str, MetricEstimate]
    kinematic_sequence: KinematicSequenceAnalysis
    com_braking: COMBrakingAnalysis
    head_stability: HeadStabilityAnalysis
    lead_arm: LeadArmAnalysis
    crease_tunnel: CreaseReleaseTunnelAnalysis
    tempo_rhythm: TempoRhythmAnalysis
    com_world_m: np.ndarray
    sequence_series: Dict[str, np.ndarray] = field(default_factory=dict)
    wrist_rel_speed_mps: Optional[np.ndarray] = None


def extract_and_quantify_v5_biomechanics(
    kin: KinematicsResult,
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    workload: Optional[WorkloadContext] = None,
    isb: Optional[ISBAnatomyResult] = None,
    markers_world_87: Optional[np.ndarray] = None,
    unc_scale_factor: float = 1.0,
    interval_type: IntervalType = "analytical_uncertainty",
    dual_baseline_sin: Optional[float] = None,
    cov_world_fused: Optional[np.ndarray] = None,
) -> V5DeepBiomechBundle:
    """Compute and uncertainty-gate all Studio v5 deep biomechanical modules (Modules 1-6)."""
    sex = workload.sex if workload is not None else "male"
    is_female = str(sex).lower().startswith("f")
    j = kin.joints_world_smoothed
    assert j is not None, "KinematicsResult.joints_world_smoothed is required for v5 biomechanics."
    n = j.shape[0]
    fps = max(float(audit.fps), 1.0)
    is_right = kin.bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0
    stature_m = max(float(kin.estimated_stature_m), 1.20)

    is_dual = cal.view_class == CameraViewClass.DUAL_VIEW_FUSED
    b_sin = float(dual_baseline_sin) if dual_baseline_sin is not None else (1.0 if is_dual else 0.0)
    is_dual_orth = is_dual and (b_sin >= 0.45)
    R_w2c = np.asarray(cal.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    blur_mult = {"crisp": 1.0, "moderate_blur": 1.18, "severe_blur": 1.50}.get(audit.blur_grade, 1.15)
    unc_arr = kin.joint_uncertainties_array if kin.joint_uncertainties_array is not None else np.full((n, 24), 0.035)
    gaps = kin.interp_gap_frames if kin.interp_gap_frames is not None else np.zeros(n, dtype=np.int32)

    az = abs(cal.azimuth_deg) % 360.0
    az = 360.0 - az if az > 180.0 else az
    sin_side = abs(math.sin(math.radians(az)))
    cos_rear = abs(math.cos(math.radians(az)))

    # Reconstruct joint visibility mask from frame bounds / kinematics
    f_hip_i = JOINT_IDX["lhip" if is_right else "rhip"]
    f_kne_i = JOINT_IDX["lkne" if is_right else "rkne"]
    f_ank_i = JOINT_IDX["lank" if is_right else "rank"]
    r_ank_i = JOINT_IDX["rank" if is_right else "lank"]
    b_sho_i = JOINT_IDX["rsho" if is_right else "lsho"]
    b_elb_i = JOINT_IDX["relb" if is_right else "lelb"]
    b_wri_i = JOINT_IDX["rwri" if is_right else "lwri"]
    b_han_i = JOINT_IDX["rhan" if is_right else "lhan"]
    nb_sho_i = JOINT_IDX["lsho" if is_right else "rsho"]
    nb_elb_i = JOINT_IDX["lelb" if is_right else "relb"]
    pelv_i, neck_i, head_i = JOINT_IDX["pelv"], JOINT_IDX["neck"], JOINT_IDX["head"]
    lhip_i, rhip_i = JOINT_IDX["lhip"], JOINT_IDX["rhip"]
    lsho_i, rsho_i = JOINT_IDX["lsho"], JOINT_IDX["rsho"]

    bfc_f, ffc_f, br_f = events.bfc.frame_float, events.ffc.frame_float, events.br.frame_float
    i_bfc = int(np.clip(round(bfc_f), 0, n - 1))
    i_ffc = int(np.clip(round(ffc_f), i_bfc, n - 1))
    i_br = int(np.clip(round(br_f), i_ffc, n - 1))

    def _cov(ji: int, f_fl: float) -> np.ndarray:
        idx = int(np.clip(round(f_fl), 0, n - 1))
        if cov_world_fused is not None and cov_world_fused.shape[:2] == (n, 24):
            return cov_world_fused[idx, ji]
        u_val = float(_interp_frame(unc_arr[:, ji], f_fl))
        return _world_cov_for_joint(R_w2c, u_val, unc_scale_factor, blur_mult, int(gaps[idx]), is_dual_orth)

    def _cov_epi(ji: int, f_fl: float) -> np.ndarray:
        idx = int(np.clip(round(f_fl), 0, n - 1))
        if cov_world_fused is not None and cov_world_fused.shape[:2] == (n, 24):
            return cov_world_fused[idx, ji]
        u_val = float(_interp_frame(unc_arr[:, ji], f_fl))
        return _world_cov_for_joint(R_w2c, u_val, unc_scale_factor, blur_mult, int(gaps[idx]), True)

    def _pt(ji: int, f_fl: float) -> np.ndarray:
        return _interp_frame(j[:, ji], f_fl)

    def _var_fn(func: Callable[[np.ndarray], float], j_ids: List[int], f_fl: float, epi: bool = False) -> float:
        pts = np.stack([_pt(ji, f_fl) for ji in j_ids], axis=0)
        g = _fd_grad(func, pts)
        v = 0.0
        for k_i, ji in enumerate(j_ids):
            c_mat = _cov_epi(ji, f_fl) if epi else _cov(ji, f_fl)
            v += float(g[k_i] @ c_mat @ g[k_i])
        return max(v, 0.0)

    def _psd_sqrt(C: np.ndarray) -> np.ndarray:
        sym = 0.5 * (C + C.T)
        ev_c, evec = np.linalg.eigh(sym)
        return (evec * np.sqrt(np.maximum(ev_c, 0.0))) @ evec.T

    def _var_diff_fn(func: Callable[[np.ndarray], float], j_ids: List[int], f1: float, f2: float, epi: bool = False) -> float:
        pts1 = np.stack([_pt(ji, f1) for ji in j_ids], axis=0)
        pts2 = np.stack([_pt(ji, f2) for ji in j_ids], axis=0)
        g1 = _fd_grad(func, pts1)
        g2 = _fd_grad(func, pts2)
        rho = math.exp(-abs(f2 - f1) / (fps * 0.08))
        v = 0.0
        for k_i, ji in enumerate(j_ids):
            c1 = _cov_epi(ji, f1) if epi else _cov(ji, f1)
            c2 = _cov_epi(ji, f2) if epi else _cov(ji, f2)
            c12 = _psd_sqrt(c1) @ _psd_sqrt(c2)
            v += float(g1[k_i] @ c1 @ g1[k_i] + g2[k_i] @ c2 @ g2[k_i] - 2.0 * rho * (g1[k_i] @ c12 @ g2[k_i]))
        return max(v, 0.0)

    def _make_est(
        key: str,
        label: str,
        unit: str,
        raw_val: Optional[float],
        sig_pose: float,
        sig_epi: float,
        sig_temp: float,
        sig_cal: float,
        sig_scale: float,
        plane_type: str,
        sup_thresh: float,
        def_note: str,
        ref_text: str,
        source: str,
        in_frame_ok: bool = True,
        max_j_unc: float = 0.04,
        anchors: Tuple[str, ...] = ("br",),
        extra_suppress: Optional[str] = None,
    ) -> MetricEstimate:
        sig_proj = math.sqrt(max(sig_pose**2 - sig_epi**2, 0.0))
        sig_tot = math.sqrt(sig_pose**2 + sig_temp**2 + sig_cal**2 + sig_scale**2)
        view_grade = _view_grade_from_geometry(
            plane_type=plane_type,
            az_deg=cal.azimuth_deg,
            el_deg=cal.elevation_deg,
            is_dual_fused=is_dual_orth,
            is_metric_cal=cal.is_metric_calibrated,
            dual_baseline_sin=b_sin if is_dual else 1.0,
        )
        ev_map = {"bfc": events.bfc, "ffc": events.ffc, "br": events.br}
        min_conf = min(ev_map[a].confidence for a in anchors if a in ev_map) if anchors else 1.0

        suppressed = False
        reason: Optional[str] = None
        obs = view_grade

        if not events.is_complete_delivery:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: clip did not contain a verified high-speed overhead ball release."
        elif raw_val is None or not math.isfinite(raw_val):
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: required anatomical landmarks were missing or non-finite."
        elif extra_suppress is not None:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = extra_suppress
        elif min_conf < 0.30:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = f"Suppressed: anchor event detection confidence too low ({min_conf:.2f}; required foot cropped or boundary hit)."
        elif not in_frame_ok:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: required anatomical landmarks stepped outside the camera frame."
        elif max_j_unc > 0.165:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = f"Suppressed: high 3D joint localization uncertainty ({max_j_unc:.2f} m) due to occlusion or blur."
        elif view_grade == ObservabilityGrade.UNMEASURABLE:
            suppressed = True
            needed = (
                "side-on (~90 deg)" if plane_type in ("sagittal", "velocity_high", "metric_scale")
                else ("behind-the-bowler (~0 deg)" if plane_type == "coronal" else "45 deg open-side or dual-phone")
            )
            reason = f"Suppressed: camera angle ({cal.azimuth_deg:.0f} deg) cannot reliably resolve {plane_type} kinematics; film from a {needed} view."
        elif sig_tot > sup_thresh:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = f"Suppressed: total 1-sigma uncertainty (+-{sig_tot:.1f} {unit}) exceeds reliability limit ({sup_thresh:.1f} {unit})."
        elif sig_tot > 0.82 * sup_thresh:
            obs = ObservabilityGrade.LOW
        elif sig_tot > 0.65 * sup_thresh and obs == ObservabilityGrade.HIGH:
            obs = ObservabilityGrade.MODERATE

        if suppressed or raw_val is None:
            return MetricEstimate(
                key=key,
                label=label,
                unit=unit,
                value=None,
                ci_low=None,
                ci_high=None,
                interval_type=interval_type,
                sigma_total=round(sig_tot, 2),
                sigma_projective=round(sig_proj, 2),
                sigma_temporal=round(sig_temp, 2),
                sigma_epistemic=round(sig_epi, 2),
                sigma_calibration=round(sig_cal, 2),
                observability=ObservabilityGrade.UNMEASURABLE,
                suppressed=True,
                suppression_reason=reason,
                definition_note=def_note,
                empirical_reference=ref_text,
                literature_source=source,
            )

        ci_lo, ci_hi = _clamp_v5_ci(key, float(raw_val) - 1.96 * sig_tot, float(raw_val) + 1.96 * sig_tot)
        return MetricEstimate(
            key=key,
            label=label,
            unit=unit,
            value=round(float(raw_val), 2),
            ci_low=round(ci_lo, 2),
            ci_high=round(ci_hi, 2),
            interval_type=interval_type,
            sigma_total=round(sig_tot, 2),
            sigma_projective=round(sig_proj, 2),
            sigma_temporal=round(sig_temp, 2),
            sigma_epistemic=round(sig_epi, 2),
            sigma_calibration=round(sig_cal, 2),
            observability=obs,
            suppressed=False,
            suppression_reason=None,
            definition_note=def_note,
            empirical_reference=ref_text,
            literature_source=source,
        )

    v5_metrics: Dict[str, MetricEstimate] = {}
    in_fr = kin.metric_joints_in_frame
    ankle_ffc_vis = in_fr.get("front_knee_ffc_deg", True)
    ankle_br_vis = in_fr.get("release_height_m", True)
    stride_vis = in_fr.get("stride_length_pct_stature", True)

    # =========================================================================
    # MODULE 1: Proximal-to-Distal Kinematic Sequence Engine
    # =========================================================================
    use_isb_pelv = (
        isb is not None
        and len(isb.pelvis_yaw_isb_deg) == n
        and "pelvis_alignment_bfc_isb_deg" in isb.metrics
        and (not isb.metrics["pelvis_alignment_bfc_isb_deg"].suppressed)
    )
    use_isb_thor = (
        isb is not None
        and len(isb.thorax_yaw_isb_deg) == n
        and "thorax_alignment_bfc_isb_deg" in isb.metrics
        and (not isb.metrics["thorax_alignment_bfc_isb_deg"].suppressed)
    )
    pelv_yaw_src = isb.pelvis_yaw_isb_deg if use_isb_pelv and isb is not None else kin.series["pelvis_yaw_deg"]
    thor_yaw_src = isb.thorax_yaw_isb_deg if use_isb_thor and isb is not None else kin.series["shoulder_yaw_deg"]

    pelvis_omega_dps, span_ax = _savgol_1d(pelv_yaw_src, fps=fps, span_s=0.055, deriv=1)
    thorax_omega_dps, _ = _savgol_1d(thor_yaw_src, fps=fps, span_s=0.055, deriv=1)
    pelvis_alpha_dps2, _ = _savgol_1d(pelvis_omega_dps, fps=fps, span_s=0.055, deriv=1)
    thorax_alpha_dps2, _ = _savgol_1d(thorax_omega_dps, fps=fps, span_s=0.055, deriv=1)

    arm_omega_dps = np.asarray(kin.series["bowling_arm_omega_dps"], dtype=np.float64)
    arm_alpha_dps2, _ = _savgol_1d(arm_omega_dps, fps=fps, span_s=0.050, deriv=1)

    # Body-centric distal lever d_dist = b_han - b_sho (100% invariant to camera pan & scale in deg/s!)
    j_fast = smooth_trajectory(j, fps=fps, cutoff_hz=min(20.0, 0.42 * fps))
    d_dist = j_fast[:, b_han_i] - j_fast[:, b_sho_i]
    d_vel = np.column_stack([_savgol_1d(d_dist[:, ax_i], fps=fps, span_s=0.050, deriv=1)[0] for ax_i in range(3)])
    wrist_rel_speed_mps = np.linalg.norm(d_vel, axis=1)
    d_norm_sq = np.maximum(np.sum(d_dist * d_dist, axis=1), 0.22 * 0.22)
    wrist_whip_dps = np.degrees(np.linalg.norm(np.cross(d_dist, d_vel), axis=1) / d_norm_sq)
    wrist_alpha_dps2, _ = _savgol_1d(wrist_whip_dps, fps=fps, span_s=0.050, deriv=1)

    # Search propulsive uncoiling window [FFC - 45 ms, BR + 15 ms] (avoids mid-bound counter-rotation wiggles near BFC)
    i_seq_prox_lo = int(np.clip(max(i_bfc + 1, round(ffc_f - 0.045 * fps)), 0, n - 1))
    i_seq_prox_hi = int(np.clip(math.ceil(br_f + 0.015 * fps), i_seq_prox_lo, n - 1))
    i_seq_dist_lo = i_ffc
    i_seq_dist_hi = int(np.clip(math.ceil(br_f + 0.020 * fps), i_seq_dist_lo, n - 1))

    pk_pelv, f_pk_pelv, k_pelv = _subframe_peak(pelvis_omega_dps, i_seq_prox_lo, i_seq_prox_hi)
    pk_thor, f_pk_thor, k_thor = _subframe_peak(thorax_omega_dps, i_seq_prox_lo, i_seq_prox_hi)
    pk_arm, f_pk_arm, k_arm = _subframe_peak(arm_omega_dps, i_seq_dist_lo, i_seq_dist_hi)
    pk_wri, f_pk_wri, k_wri = _subframe_peak(wrist_whip_dps, i_seq_dist_lo, i_seq_dist_hi)

    seq_gate_msg: Optional[str] = None
    if audit.timebase_unreliable:
        seq_gate_msg = f"Suppressed: video timebase is unreliable ({audit.timebase_status}); kinematic sequence velocity peaks require verified frame timing."
    elif fps < 90.0:
        seq_gate_msg = f"Suppressed: effective rate is {fps:.1f} fps; resolving millisecond proximal-to-distal peak ordering requires >=100 fps slow-motion."

    def _fn_yaw_s(p: np.ndarray) -> float:
        lr = p[1] - p[0]
        return float(math.degrees(math.atan2(sign_arm * lr[0], lr[2])))

    def _fn_arm_phi_s(p: np.ndarray) -> float:
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[0], v[1])))

    rho_ax = math.exp(-max(span_ax, 2.0 / fps) / 0.08)
    deriv_mult_ax = math.sqrt(max(2.0 * (1.0 - rho_ax), 0.05)) / max(0.5 * span_ax, 0.018)
    span_arm = max(kin.savgol_window_s, 2.0 / fps)
    rho_arm = math.exp(-span_arm / 0.08)
    deriv_mult_arm = math.sqrt(max(2.0 * (1.0 - rho_arm), 0.05)) / max(0.5 * span_arm, 0.015)

    sig_pelv_ang = 0.88 * math.sqrt(_var_fn(_fn_yaw_s, [lhip_i, rhip_i], f_pk_pelv))
    sig_pelv_epi_ang = 0.88 * math.sqrt(_var_fn(_fn_yaw_s, [lhip_i, rhip_i], f_pk_pelv, epi=True))
    sig_thor_ang = 0.88 * math.sqrt(_var_fn(_fn_yaw_s, [lsho_i, rsho_i], f_pk_thor))
    sig_thor_epi_ang = 0.88 * math.sqrt(_var_fn(_fn_yaw_s, [lsho_i, rsho_i], f_pk_thor, epi=True))
    sig_arm_ang = math.sqrt(_var_fn(_fn_arm_phi_s, [b_sho_i, b_elb_i], f_pk_arm))
    sig_arm_epi_ang = math.sqrt(_var_fn(_fn_arm_phi_s, [b_sho_i, b_elb_i], f_pk_arm, epi=True))
    sig_wri_ang = math.sqrt(_var_fn(_fn_arm_phi_s, [b_sho_i, b_han_i], f_pk_wri))
    sig_wri_epi_ang = math.sqrt(_var_fn(_fn_arm_phi_s, [b_sho_i, b_han_i], f_pk_wri, epi=True))

    sig_pk_pelv = deriv_mult_ax * sig_pelv_ang
    sig_pk_thor = deriv_mult_ax * sig_thor_ang
    sig_pk_arm = deriv_mult_arm * sig_arm_ang
    sig_pk_wri = deriv_mult_arm * sig_wri_ang * 1.15

    me_pelv_pk = _make_est(
        "seq_pelvis_peak_dps", "Peak Pelvis Rotational Velocity", "deg/s", pk_pelv,
        sig_pk_pelv, deriv_mult_ax * sig_pelv_epi_ang, 12.0, 8.0, 0.0,
        "transverse", 380.0,
        "Peak Savitzky-Golay (~55 ms) global transverse pelvis rotational velocity about +Y_world during [BFC, BR].",
        "Proximal segment #1 in fast-bowling kinetic chain (~400-700 deg/s).",
        "Ferdinands et al. (2013), J. Sports Sciences; Putnam (1993)",
        in_frame_ok=in_fr.get("pelvis_alignment_bfc_deg", True),
        anchors=("bfc", "ffc"),
        extra_suppress=seq_gate_msg,
    )
    me_thor_pk = _make_est(
        "seq_thorax_peak_dps", "Peak Thorax Rotational Velocity", "deg/s", pk_thor,
        sig_pk_thor, deriv_mult_ax * sig_thor_epi_ang, 15.0, 10.0, 0.0,
        "transverse", 420.0,
        "Peak Savitzky-Golay (~55 ms) global transverse thorax/shoulder rotational velocity about +Y_world during [BFC, BR].",
        "Segment #2 in proximal-to-distal cascade (~700-1150 deg/s).",
        "Ferdinands et al. (2013); Worthington et al. (2013)",
        in_frame_ok=in_fr.get("shoulder_alignment_bfc_deg", True),
        anchors=("ffc", "br"),
        extra_suppress=seq_gate_msg,
    )
    me_arm_pk = _make_est(
        "seq_arm_peak_dps", "Peak Upper-Arm Circumduction Velocity", "deg/s", pk_arm,
        sig_pk_arm, deriv_mult_arm * sig_arm_epi_ang, 25.0, 12.0, 0.0,
        "velocity_high", 450.0,
        "Peak Savitzky-Golay (~50 ms) bowling upper-arm circumduction velocity during [FFC, BR].",
        "Segment #3 in fast-bowling whip (~1800-2600 deg/s).",
        "Worthington et al. (2013); Ferdinands et al. (2013)",
        in_frame_ok=in_fr.get("peak_arm_angular_vel_dps", True),
        anchors=("ffc", "br"),
        extra_suppress=seq_gate_msg,
    )
    me_wri_pk = _make_est(
        "seq_wrist_whip_peak_dps", "Peak Distal Hand/Wrist Whip Velocity", "deg/s", pk_wri,
        sig_pk_wri, deriv_mult_arm * sig_wri_epi_ang * 1.15, 32.0, 14.0, 0.0,
        "velocity_high", 580.0,
        "Scale- & translation-invariant distal angular whip ||(p_han - p_sho) x v_rel|| / ||p_han - p_sho||^2 in deg/s.",
        "Terminal segment #4 peaking at ball release (~2300-3400 deg/s).",
        "Putnam (1993); Ferdinands et al. (2013)",
        in_frame_ok=in_fr.get("peak_arm_angular_vel_dps", True),
        anchors=("ffc", "br"),
        extra_suppress=seq_gate_msg,
    )

    def _peak_time_sigma_ms(pk_om: float, sig_om: float) -> float:
        snr = abs(pk_om) / max(sig_om, 20.0)
        return float(math.hypot(500.0 / fps, 85.0 / max(snr - 1.15, 0.40)))

    s_t_pelv = _peak_time_sigma_ms(pk_pelv, sig_pk_pelv)
    s_t_thor = _peak_time_sigma_ms(pk_thor, sig_pk_thor)
    s_t_arm = _peak_time_sigma_ms(pk_arm, sig_pk_arm)
    s_t_wri = _peak_time_sigma_ms(pk_wri, sig_pk_wri)

    t_pelv_ffc = (f_pk_pelv - ffc_f) * 1000.0 / fps
    t_thor_ffc = (f_pk_thor - ffc_f) * 1000.0 / fps
    t_arm_ffc = (f_pk_arm - ffc_f) * 1000.0 / fps
    t_wri_ffc = (f_pk_wri - ffc_f) * 1000.0 / fps

    t_pelv_br = (f_pk_pelv - br_f) * 1000.0 / fps
    t_thor_br = (f_pk_thor - br_f) * 1000.0 / fps
    t_arm_br = (f_pk_arm - br_f) * 1000.0 / fps
    t_wri_br = (f_pk_wri - br_f) * 1000.0 / fps

    lag_pt = t_thor_ffc - t_pelv_ffc
    lag_ta = t_arm_ffc - t_thor_ffc
    lag_aw = t_wri_ffc - t_arm_ffc

    pelv_snr_ok = pk_pelv >= 2.35 * max(sig_pk_pelv, 1.0)
    thor_snr_ok = pk_thor >= 2.35 * max(sig_pk_thor, 1.0)
    arm_snr_ok = pk_arm >= 2.35 * max(sig_pk_arm, 1.0)

    lag_pt_gate = seq_gate_msg
    if lag_pt_gate is None and (me_pelv_pk.suppressed or me_thor_pk.suppressed):
        lag_pt_gate = "Suppressed: pelvis or thorax rotational peak is unobservable from this camera angle."
    elif lag_pt_gate is None and not (pelv_snr_ok and thor_snr_ok):
        lag_pt_gate = "Suppressed: proximal hip-or-chest velocity peak is below 2.4-sigma pose-noise floor on single-camera depth; use dual-camera or ISB markers to resolve millisecond lag."

    lag_ta_gate = seq_gate_msg
    if lag_ta_gate is None and (me_thor_pk.suppressed or me_arm_pk.suppressed):
        lag_ta_gate = "Suppressed: requires both thorax transverse rotation and upper-arm circumduction to be observable (45 deg open-side or dual-camera 120-240 fps)."
    elif lag_ta_gate is None and not (thor_snr_ok and arm_snr_ok):
        lag_ta_gate = "Suppressed: thorax rotational peak SNR is too low on this camera angle to localize thorax-to-arm millisecond lag."

    me_lag_pt = _make_est(
        "seq_pelvis_to_thorax_lag_ms", "Pelvis-to-Thorax Peak Timing Gap", "ms", lag_pt,
        math.hypot(s_t_pelv, s_t_thor), 0.7 * math.hypot(s_t_pelv, s_t_thor), 2.5, 0.0, 0.0,
        "transverse", 75.0,
        "Time elapsed between peak pelvis rotational speed and peak thorax rotational speed (+ = proximal-to-distal).",
        "Positive separation (~15-45 ms) indicates clean hip-leads-chest uncoiling.",
        "Ferdinands et al. (2013); Putnam (1993)",
        in_frame_ok=(not me_pelv_pk.suppressed and not me_thor_pk.suppressed),
        anchors=("bfc", "ffc"),
        extra_suppress=lag_pt_gate,
    )
    me_lag_ta = _make_est(
        "seq_thorax_to_arm_lag_ms", "Thorax-to-Arm Peak Timing Gap", "ms", lag_ta,
        math.hypot(s_t_thor, s_t_arm), 0.7 * math.hypot(s_t_thor, s_t_arm), 2.5, 0.0, 0.0,
        "velocity_high", 75.0,
        "Time elapsed between peak thorax rotational speed and peak upper-arm circumduction speed.",
        "Positive separation (~20-55 ms) confirms chest whip precedes arm acceleration.",
        "Worthington et al. (2013); Ferdinands et al. (2013)",
        anchors=("ffc", "br"),
        extra_suppress=lag_ta_gate,
    )
    me_lag_aw = _make_est(
        "seq_arm_to_wrist_lag_ms", "Arm-to-Wrist Whip Timing Gap", "ms", lag_aw,
        math.hypot(s_t_arm, s_t_wri), 0.7 * math.hypot(s_t_arm, s_t_wri), 2.0, 0.0, 0.0,
        "velocity_high", 45.0,
        "Time elapsed between upper-arm circumduction peak and terminal hand/wrist whip peak.",
        "Near-zero to positive lag (~0-18 ms) into ball release.",
        "Putnam (1993)",
        anchors=("ffc", "br"),
        extra_suppress=seq_gate_msg if seq_gate_msg else ("Suppressed: upper-arm or wrist peak unobservable." if (me_arm_pk.suppressed or me_wri_pk.suppressed) else None),
    )

    whip_gain = pk_arm / max(pk_thor, 100.0)
    sig_gain = whip_gain * math.hypot(sig_pk_arm / max(pk_arm, 150.0), sig_pk_thor / max(pk_thor, 150.0))
    whip_gain_gate = seq_gate_msg if seq_gate_msg else ("Suppressed: requires both thorax rotation and upper-arm circumduction to be observable." if (me_thor_pk.suppressed or me_arm_pk.suppressed) else None)
    me_whip_gain = _make_est(
        "seq_whip_gain_arm_over_thorax", "Arm-Over-Thorax Angular Whip Gain", "ratio", whip_gain,
        sig_gain, 0.7 * sig_gain, 0.08, 0.05, 0.0,
        "velocity_high", 1.35,
        "Ratio of peak upper-arm circumduction angular velocity to peak thorax rotational velocity.",
        "Typical proximal-to-distal speed multiplication: ~1.9-3.2x.",
        "Putnam (1993); Ferdinands et al. (2013)",
        anchors=("ffc", "br"),
        extra_suppress=whip_gain_gate,
    )

    pelv_at_br = float(_interp_frame(pelvis_omega_dps, br_f))
    pelv_decel_pct = float(np.clip(100.0 * (pk_pelv - pelv_at_br) / max(pk_pelv, 80.0), 0.0, 100.0))
    ratio_p = float(np.clip(pelv_at_br / max(pk_pelv, 150.0), -0.5, 1.2))
    sig_decel = (100.0 / max(pk_pelv, 220.0)) * math.hypot(0.60 * sig_pk_pelv, abs(ratio_p) * sig_pk_pelv)
    pelv_decel_gate = seq_gate_msg
    if pelv_decel_gate is None and (me_pelv_pk.suppressed or not pelv_snr_ok):
        pelv_decel_gate = "Suppressed: pelvis rotational velocity peak is below reliable pose-noise SNR on this camera angle."
    me_pelv_decel = _make_est(
        "seq_pelvis_braking_decel_pct", "Pelvis Pre-Release Deceleration", "%", pelv_decel_pct,
        sig_decel, 0.7 * sig_decel, 4.0, 2.0, 0.0,
        "transverse", 48.0,
        "Percentage drop in pelvis rotational velocity from its peak down to ball release.",
        "Higher proximal deceleration (>35%) transfers rotational momentum cleanly up to the thorax.",
        "Putnam (1993); Ferdinands et al. (2013)",
        anchors=("ffc", "br"),
        extra_suppress=pelv_decel_gate,
    )

    tol_ms = max(500.0 / fps, 6.0)
    if (
        (not events.is_complete_delivery)
        or seq_gate_msg is not None
        or me_pelv_pk.suppressed
        or me_thor_pk.suppressed
        or me_arm_pk.suppressed
        or me_lag_pt.suppressed
        or me_lag_ta.suppressed
    ):
        seq_verdict: SequenceVerdict = "unmeasurable"
        seq_lbl = "Needs Slow-Mo / Dual View"
        seq_sum = (
            "Proximal-to-distal peak timing requires 120–240 fps slow-motion from a diagonal open-side or dual-camera setup."
            if events.is_complete_delivery
            else "No complete overhead release was observed, so kinematic sequencing was withheld."
        )
    elif lag_pt > tol_ms and lag_ta > tol_ms and lag_aw >= -tol_ms:
        seq_verdict = "optimal_proximal_to_distal"
        seq_lbl = "Clean Proximal-to-Distal Whip"
        seq_sum = (
            f"Hips peak first, followed by chest (+{int(round(lag_pt))} ms) and bowling arm (+{int(round(lag_ta))} ms), "
            "multiplying speed cleanly into ball release."
        )
    elif lag_pt >= -tol_ms and lag_ta >= -tol_ms and lag_aw >= -tol_ms:
        seq_verdict = "simultaneous_rotational"
        seq_lbl = "Simultaneous Block Rotation"
        seq_sum = "Hips and chest peak nearly together within frame resolution rather than uncoiling in distinct waves."
    else:
        seq_verdict = "out_of_order"
        seq_lbl = "Out-of-Order Sequence"
        if lag_pt < -tol_ms:
            seq_sum = "Shoulders reached peak rotation ahead of the hips, spinning the upper body open early."
        else:
            seq_sum = "Bowling arm rushed ahead of peak chest rotation, leaking torso whip."

    seg_peaks = [
        SequenceSegmentPeak(
            segment="pelvis", label_plain="Hips (Pelvis)", peak_dps=me_pelv_pk,
            peak_time_rel_ffc_ms=None if me_pelv_pk.suppressed else round(t_pelv_ffc, 1),
            peak_time_rel_br_ms=None if me_pelv_pk.suppressed else round(t_pelv_br, 1),
            timing_sigma_ms=round(s_t_pelv, 1),
        ),
        SequenceSegmentPeak(
            segment="thorax", label_plain="Chest (Thorax)", peak_dps=me_thor_pk,
            peak_time_rel_ffc_ms=None if me_thor_pk.suppressed else round(t_thor_ffc, 1),
            peak_time_rel_br_ms=None if me_thor_pk.suppressed else round(t_thor_br, 1),
            timing_sigma_ms=round(s_t_thor, 1),
        ),
        SequenceSegmentPeak(
            segment="bowling_arm", label_plain="Upper Arm", peak_dps=me_arm_pk,
            peak_time_rel_ffc_ms=None if me_arm_pk.suppressed else round(t_arm_ffc, 1),
            peak_time_rel_br_ms=None if me_arm_pk.suppressed else round(t_arm_br, 1),
            timing_sigma_ms=round(s_t_arm, 1),
        ),
        SequenceSegmentPeak(
            segment="wrist_whip", label_plain="Wrist & Hand", peak_dps=me_wri_pk,
            peak_time_rel_ffc_ms=None if me_wri_pk.suppressed else round(t_wri_ffc, 1),
            peak_time_rel_br_ms=None if me_wri_pk.suppressed else round(t_wri_br, 1),
            timing_sigma_ms=round(s_t_wri, 1),
        ),
    ]
    kin_seq_obj = KinematicSequenceAnalysis(
        verdict=seq_verdict,
        verdict_label_plain=seq_lbl,
        summary_plain=seq_sum,
        segments=seg_peaks,
        pelvis_to_thorax_lag_ms=me_lag_pt,
        thorax_to_arm_lag_ms=me_lag_ta,
        arm_to_wrist_lag_ms=me_lag_aw,
        whip_gain_arm_over_thorax=me_whip_gain,
        pelvis_braking_decel_pct=me_pelv_decel,
    )
    for me in (me_pelv_pk, me_thor_pk, me_arm_pk, me_wri_pk, me_lag_pt, me_lag_ta, me_lag_aw, me_whip_gain, me_pelv_decel):
        v5_metrics[me.key] = me

    # =========================================================================
    # MODULE 2: 3D Center of Mass (COM) & Braking Impulse Analyzer
    # =========================================================================
    alpha_com = get_de_leva_smpl24_weights(sex)
    com_raw = compute_3d_com_trajectory(j, sex=sex)
    com_sm = smooth_trajectory(com_raw[:, None, :], fps=fps, cutoff_hz=min(12.0, 0.32 * fps))[:, 0, :]
    com_vx, span_com = _savgol_1d(com_sm[:, 0], fps=fps, span_s=0.065, deriv=1)

    def _com_cov(f_fl: float, epi: bool = False) -> np.ndarray:
        # Correlated-depth body covariance (rho_body = 0.75 prevents false 1/sqrt(24) shrinkage!)
        rho_b = 0.75
        indep_sum = np.zeros((3, 3), dtype=np.float64)
        sqrt_sum = np.zeros((3, 3), dtype=np.float64)
        for j_i in range(24):
            w_j = float(alpha_com[j_i])
            if abs(w_j) < 1e-6:
                continue
            cj = _cov_epi(j_i, f_fl) if epi else _cov(j_i, f_fl)
            indep_sum += (w_j * w_j) * cj
            sqrt_sum += w_j * _psd_sqrt(cj)
        return (1.0 - rho_b) * indep_sum + rho_b * (sqrt_sum @ sqrt_sum.T)

    c_com_ffc = _com_cov(ffc_f, epi=False)
    c_com_ffc_epi = _com_cov(ffc_f, epi=True)
    c_com_br = _com_cov(br_f, epi=False)

    vx_ffc = float(_interp_frame(com_vx, ffc_f))
    vx_br = float(_interp_frame(com_vx, br_f))
    delta_vx = vx_ffc - vx_br
    brake_eff_pct = 100.0 * delta_vx / max(vx_ffc, 1.5)

    rho_cv = math.exp(-max(span_com, 2.0 / fps) / 0.08)
    sig_vx_ffc = math.sqrt(max(2.0 * (1.0 - rho_cv) * float(c_com_ffc[0, 0]), 0.0)) / max(span_com, 0.025)
    sig_vx_ffc_epi = math.sqrt(max(2.0 * (1.0 - rho_cv) * float(c_com_ffc_epi[0, 0]), 0.0)) / max(span_com, 0.025)
    sig_vx_br = math.sqrt(max(2.0 * (1.0 - rho_cv) * float(c_com_br[0, 0]), 0.0)) / max(span_com, 0.025)
    sig_dvx = math.hypot(sig_vx_ffc, sig_vx_br) * 0.85

    trans_gate_msg: Optional[str] = None
    if audit.timebase_unreliable:
        trans_gate_msg = f"Suppressed: video timebase is unreliable ({audit.timebase_status}); COM velocities require verified timing."
    elif cal.is_panning_camera or not cal.is_metric_calibrated:
        trans_gate_msg = "Suppressed: metric COM horizontal speed requires a static camera with calibrated MCC pitch/wicket geometry."

    brake_gate_msg = trans_gate_msg
    if brake_gate_msg is None and fps < 55.0:
        brake_gate_msg = f"Suppressed: effective rate ({fps:.1f} fps) is too slow to resolve FFC->BR COM braking deceleration; record at >=60-240 fps."

    eff_gate_msg: Optional[str] = None
    if audit.timebase_unreliable:
        eff_gate_msg = f"Suppressed: unreliable video timebase ({audit.timebase_status})."
    elif cal.is_panning_camera:
        eff_gate_msg = "Suppressed: camera panning corrupts horizontal COM deceleration ratios."
    elif fps < 55.0:
        eff_gate_msg = f"Suppressed: COM braking efficiency requires >=60 fps (got {fps:.1f} fps)."
    elif vx_ffc < 1.8:
        eff_gate_msg = "Suppressed: forward COM velocity at FFC was too small to compute a meaningful braking percentage."

    me_com_ffc = _make_est(
        "com_ffc_speed_mps", "COM Horizontal Speed at FFC", "m/s", vx_ffc,
        sig_vx_ffc, sig_vx_ffc_epi, 0.12, 0.08, 0.0,
        "metric_scale", 1.25,
        "14-segment de Leva (1996) whole-body Center of Mass horizontal speed (+X) at front-foot contact.",
        "Elite male fast bowlers enter FFC at ~5.2-6.1 m/s.",
        "de Leva (1996); Worthington et al. (2013)",
        in_frame_ok=(ankle_ffc_vis and stride_vis),
        anchors=("ffc",),
        extra_suppress=trans_gate_msg,
    )
    me_com_br = _make_est(
        "com_br_speed_mps", "COM Horizontal Speed at Release", "m/s", vx_br,
        sig_vx_br, sig_vx_ffc_epi, 0.14, 0.08, 0.0,
        "metric_scale", 1.25,
        "14-segment whole-body Center of Mass horizontal speed (+X) at ball release.",
        "Lower residual COM speed relative to FFC reflects firmer front-leg braking.",
        "de Leva (1996); King et al. (2016)",
        in_frame_ok=ankle_br_vis,
        anchors=("br",),
        extra_suppress=trans_gate_msg,
    )
    me_com_dvx = _make_est(
        "com_braking_delta_vx_mps", "COM Horizontal Braking Loss (FFC->BR)", "m/s", delta_vx,
        sig_dvx, 0.75 * sig_dvx, 0.16, 0.08, 0.0,
        "metric_scale", 1.35,
        "Horizontal COM velocity reduction v_x(FFC) - v_x(BR) converted into upper-body angular momentum.",
        "Braced fast bowlers typically check ~1.3-2.4 m/s of horizontal COM velocity across [FFC, BR].",
        "Worthington et al. (2013); King, Worthington, & Ranson (2016)",
        in_frame_ok=(ankle_ffc_vis and ankle_br_vis),
        anchors=("ffc", "br"),
        extra_suppress=brake_gate_msg,
    )
    sig_eff = 100.0 * sig_dvx / max(abs(vx_ffc), 2.5)
    me_com_eff = _make_est(
        "com_braking_efficiency_pct", "Front-Leg Linear Braking Conversion", "%", brake_eff_pct,
        sig_eff, 0.75 * sig_eff, 2.8, 1.5, 0.0 if cal.is_metric_calibrated else 4.0,
        "sagittal", 24.0,
        "Percentage of FFC horizontal COM velocity checked between FFC and ball release.",
        "Firm front-leg bracers typically convert ~28-46% of horizontal COM speed; collapsing knees <18%.",
        "Worthington et al. (2013); King et al. (2016)",
        in_frame_ok=(ankle_ffc_vis and ankle_br_vis),
        anchors=("ffc", "br"),
        extra_suppress=eff_gate_msg,
    )

    # Local-ground dimensionless COM vertical sink (% stature) — immune to panning!
    y_g_bfc = float(_interp_frame(j[:, r_ank_i, 1], bfc_f))
    local_f_ank_y = j[i_ffc : i_br + 1, f_ank_i, 1]
    y_g_ffc_br = float(np.percentile(local_f_ank_y, 10)) if len(local_f_ank_y) > 0 else float(j[i_br, f_ank_i, 1])
    h_com_bfc = float(_interp_frame(com_sm[:, 1], bfc_f)) - y_g_bfc
    h_com_min = float(np.min(com_sm[i_ffc : i_br + 1, 1])) - y_g_ffc_br
    com_sink_pct = 100.0 * (h_com_bfc - h_com_min) / stature_m
    sig_sink_pct = 100.0 * math.sqrt(max(float(c_com_ffc[1, 1] + _cov(f_ank_i, ffc_f)[1, 1]), 0.0)) / stature_m
    sink_gate = "Suppressed: uncalibrated torso-only gravity (>6.5 deg) prevents vertical COM sink accuracy." if (not cal.is_intrinsics_calibrated and cal.gravity_sigma_deg > 6.5) else None
    me_com_sink = _make_est(
        "com_vertical_sink_pct_stature", "COM Vertical Sink (BFC->Braking)", "% stature", com_sink_pct,
        sig_sink_pct, 0.8 * sig_sink_pct, 0.9, 0.8 * cal.gravity_sigma_deg * 0.35, 0.0,
        "sagittal" if sin_side >= cos_rear else "coronal", 9.5,
        "Dimensionless vertical COM height drop from BFC into the [FFC, BR] braking trough (% of stature).",
        "Controlled sink (~2-7% stature) preserves tall release; >10% indicates sinking into the crease.",
        "Felton, Worthington, & King (2019/2020)",
        in_frame_ok=(ankle_ffc_vis and stride_vis),
        anchors=("bfc", "ffc"),
        extra_suppress=sink_gate,
    )

    # King, Worthington, & Ranson (2016) sagittal front-leg plant angle at FFC (single-instant -> pan-invariant!)
    com_pt_ffc = _interp_frame(com_sm, ffc_f)
    f_ank_pt_ffc = _pt(f_ank_i, ffc_f)
    plant_angle_deg = math.degrees(math.atan2(f_ank_pt_ffc[0] - com_pt_ffc[0], max(com_pt_ffc[1] - f_ank_pt_ffc[1], 0.25)))

    def _fn_plant(p: np.ndarray) -> float:
        # p[0] = com, p[1] = f_ank
        return float(math.degrees(math.atan2(p[1, 0] - p[0, 0], max(p[0, 1] - p[1, 1], 0.25))))

    g_pl = _fd_grad(_fn_plant, np.stack([com_pt_ffc, f_ank_pt_ffc], axis=0))
    sig_pl_pose = math.sqrt(max(float(g_pl[0] @ c_com_ffc @ g_pl[0] + g_pl[1] @ _cov(f_ank_i, ffc_f) @ g_pl[1]), 0.0))
    sig_pl_epi = math.sqrt(max(float(g_pl[0] @ c_com_ffc_epi @ g_pl[0] + g_pl[1] @ _cov_epi(f_ank_i, ffc_f) @ g_pl[1]), 0.0))
    me_plant_ang = _make_est(
        "front_leg_plant_angle_ffc_deg", "Front-Leg Sagittal Plant Angle at FFC", "deg", plant_angle_deg,
        sig_pl_pose, sig_pl_epi, 1.8 * (events.ffc.timing_uncertainty_ms / 10.0), cal.gravity_sigma_deg, 0.0,
        "sagittal", 11.5,
        "Sagittal angle of COM-to-front-ankle braking strut relative to vertical at FFC (King et al., 2016).",
        "Elite fast-bowling plant angle: ~28-38 deg (>42 deg = over-striding; <20 deg = landing on top of foot).",
        "King, Worthington, & Ranson (2016), J. Sports Sciences",
        in_frame_ok=ankle_ffc_vis,
        anchors=("ffc",),
    )

    if (
        not me_com_dvx.suppressed
        and me_com_dvx.value is not None
        and not me_com_eff.suppressed
        and me_com_eff.value is not None
        and not me_plant_ang.suppressed
        and me_plant_ang.value is not None
    ):
        com_sum = (
            f"Front foot plants at {int(round(me_plant_ang.value))}° and checks {me_com_dvx.value:.1f} m/s "
            f"({int(round(me_com_eff.value))}% of forward COM speed) into upper-body whip."
        )
    elif not me_plant_ang.suppressed and me_plant_ang.value is not None:
        com_sum = f"Front-leg plant angle is {int(round(me_plant_ang.value))}° at foot strike; lock a calibrated static camera to unlock metric COM braking speed."
    elif not me_com_sink.suppressed and me_com_sink.value is not None:
        com_sum = f"Vertical center-of-mass sink across plant is {int(round(me_com_sink.value))}% of height; film side-on to measure horizontal braking impulse."
    else:
        com_sum = "Center-of-mass braking metrics need a full-body side-on slow-motion view with visible creases."

    com_obj = COMBrakingAnalysis(
        summary_plain=com_sum,
        com_ffc_speed_mps=me_com_ffc,
        com_br_speed_mps=me_com_br,
        com_braking_delta_vx_mps=me_com_dvx,
        com_braking_efficiency_pct=me_com_eff,
        com_vertical_sink_pct_stature=me_com_sink,
        front_leg_plant_angle_ffc_deg=me_plant_ang,
    )
    for me in (me_com_ffc, me_com_br, me_com_dvx, me_com_eff, me_com_sink, me_plant_ang):
        v5_metrics[me.key] = me

    # =========================================================================
    # MODULE 3: Head Stability & Coronal Fall-Away Analyzer
    # =========================================================================
    def _fn_head_cor(p: np.ndarray) -> float:
        # p[0] = pelv, p[1] = head
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(-sign_arm * v[2], math.hypot(v[0], max(v[1], 0.1)))))

    head_vec = j[:, head_i] - j[:, pelv_i]
    head_cor_series = np.degrees(np.arctan2(-sign_arm * head_vec[:, 2], np.hypot(head_vec[:, 0], np.maximum(head_vec[:, 1], 0.1))))
    hc_ffc = float(_interp_frame(head_cor_series, ffc_f))
    hc_br = float(_interp_frame(head_cor_series, br_f))
    hc_delta = hc_br - hc_ffc

    head_br_vis = in_fr.get("thorax_lat_flex_br_deg", True)
    sig_hc_br = math.sqrt(_var_fn(_fn_head_cor, [pelv_i, head_i], br_f))
    sig_hc_br_epi = math.sqrt(_var_fn(_fn_head_cor, [pelv_i, head_i], br_f, epi=True))
    sig_hc_delta = math.sqrt(_var_diff_fn(_fn_head_cor, [pelv_i, head_i], ffc_f, br_f))
    sig_hc_delta_epi = math.sqrt(_var_diff_fn(_fn_head_cor, [pelv_i, head_i], ffc_f, br_f, epi=True))

    me_head_cor_br = _make_est(
        "head_coronal_angle_br_deg", "Head-Pelvis Coronal Tilt at Release", "deg", hc_br,
        sig_hc_br, sig_hc_br_epi, 1.5, cal.gravity_sigma_deg, 0.0,
        "coronal", 12.5,
        "Coronal inclination of pelvis->head axis toward the non-bowling side at ball release.",
        "Tracks whole-axis head-and-spine tilt at release (~26-42 deg typical).",
        "Glazier (2010), Sports Medicine",
        in_frame_ok=head_br_vis,
        anchors=("br",),
    )
    me_head_fall_delta = _make_est(
        "head_fall_away_ffc_to_br_deg", "Dynamic Head Fall-Away Sweep (FFC->BR)", "deg", hc_delta,
        sig_hc_delta, sig_hc_delta_epi, 2.0, 0.35 * cal.gravity_sigma_deg, 0.0,
        "coronal", 13.5,
        "Additional coronal head tilt toward the off/non-bowling side between FFC and ball release.",
        "Controlled sweep (<22 deg) keeps gaze corridor stable; >28 deg signals late coronal collapse.",
        "Glazier (2010)",
        in_frame_ok=(in_fr.get("thorax_lat_flex_ffc_deg", True) and head_br_vis),
        anchors=("ffc", "br"),
    )

    head_pt_br = _pt(head_i, br_f)
    f_ank_pt_br = _pt(f_ank_i, br_f)
    dz_head_ank_m = -sign_arm * float(head_pt_br[2] - f_ank_pt_br[2])
    dz_head_pct = 100.0 * dz_head_ank_m / stature_m
    dz_head_cm = 100.0 * dz_head_ank_m
    var_dz_ha = float(_cov(head_i, br_f)[2, 2] + _cov(f_ank_i, br_f)[2, 2])
    var_dz_ha_epi = float(_cov_epi(head_i, br_f)[2, 2] + _cov_epi(f_ank_i, br_f)[2, 2])
    sig_dz_m = math.sqrt(max(var_dz_ha, 0.0))
    sig_dz_epi_m = math.sqrt(max(var_dz_ha_epi, 0.0))

    me_head_off_pct = _make_est(
        "head_lateral_offset_from_ankle_br_pct_stature", "Head Lateral Offset Past Front Foot", "% stature", dz_head_pct,
        100.0 * sig_dz_m / stature_m, 100.0 * sig_dz_epi_m / stature_m,
        1.2, 1.3 * cal.gravity_sigma_deg, 0.0,
        "coronal", 12.0,
        "Contralateral horizontal distance of head past the planted front-ankle vertical plumb line at BR (% stature).",
        "0-10% stature = eyes stacked near front-foot corridor; >16% = falling away off-balance.",
        "Bartlett et al. (1996); Glazier (2010)",
        in_frame_ok=(head_br_vis and ankle_br_vis),
        anchors=("br",),
    )
    cm_scale_gate = None if (cal.is_metric_calibrated or cal.is_intrinsics_calibrated) else "Suppressed: metric cm head offset requires calibrated pitch/wicket intrinsics (see % stature instead)."
    me_head_off_cm = _make_est(
        "head_lateral_offset_from_ankle_br_cm", "Head Lateral Offset Past Front Foot (cm)", "cm", dz_head_cm,
        100.0 * sig_dz_m, 100.0 * sig_dz_epi_m,
        2.0, 2.2 * cal.gravity_sigma_deg, 0.0 if cal.is_metric_calibrated else 0.04 * abs(dz_head_cm),
        "coronal", 22.0,
        "Contralateral horizontal offset of head center past planted front-ankle plumb line at BR in cm (+ = falling away).",
        "0 to +18 cm = stacked head corridor; > +28 cm = pronounced coronal fall-away.",
        "Bartlett et al. (1996); Glazier (2010)",
        in_frame_ok=(head_br_vis and ankle_br_vis),
        anchors=("br",),
        extra_suppress=cm_scale_gate,
    )

    h_head_ffc = float(_pt(head_i, ffc_f)[1] - f_ank_pt_ffc[1])
    h_head_br = float(head_pt_br[1] - f_ank_pt_br[1])
    head_vdrop_pct = 100.0 * (h_head_ffc - h_head_br) / stature_m
    sig_hv_pct = 100.0 * math.sqrt(max(float(_cov(head_i, ffc_f)[1, 1] + _cov(head_i, br_f)[1, 1]), 0.0)) / stature_m
    me_head_vdrop = _make_est(
        "head_vertical_drop_ffc_to_br_pct_stature", "Head Vertical Drop (FFC->BR)", "% stature", head_vdrop_pct,
        sig_hv_pct, 0.8 * sig_hv_pct, 0.9, 0.5 * cal.gravity_sigma_deg, 0.0,
        "sagittal" if sin_side >= cos_rear else "coronal", 10.5,
        "Vertical head height reduction from FFC to BR normalized by stature.",
        "Typical forward-flexion head drop: ~4-11% stature.",
        "Worthington et al. (2013)",
        in_frame_ok=(head_br_vis and ankle_ffc_vis and ankle_br_vis),
        anchors=("ffc", "br"),
    )

    if not me_head_off_pct.suppressed and me_head_off_pct.value is not None:
        if me_head_off_pct.ci_low is not None and me_head_off_pct.ci_low > 16.0:
            head_sum = f"Head drifts contralaterally past the planted front foot ({int(round(me_head_off_pct.value))}% of height) at release."
        else:
            head_sum = f"Head stays stacked near the front-foot balance line ({int(round(me_head_off_pct.value))}% of height) through release."
    elif not me_head_cor_br.suppressed and me_head_cor_br.value is not None:
        head_sum = f"Coronal head-spine tilt reaches {int(round(me_head_cor_br.value))}° at release (keep full feet in frame to measure ankle plumb offset)."
    else:
        head_sum = "Film from behind or in front of the bowler to measure sideways head stability over the front foot."

    head_obj = HeadStabilityAnalysis(
        summary_plain=head_sum,
        head_coronal_angle_br_deg=me_head_cor_br,
        head_fall_away_ffc_to_br_deg=me_head_fall_delta,
        head_lateral_offset_from_ankle_br_pct_stature=me_head_off_pct,
        head_lateral_offset_from_ankle_br_cm=me_head_off_cm,
        head_vertical_drop_ffc_to_br_pct_stature=me_head_vdrop,
    )
    for me in (me_head_cor_br, me_head_fall_delta, me_head_off_pct, me_head_off_cm, me_head_vdrop):
        v5_metrics[me.key] = me

    # =========================================================================
    # MODULE 4: Lead (Non-Bowling) Arm Pull-Down Mechanics
    # =========================================================================
    nb_elev_series = np.asarray(kin.series["nb_arm_elev_deg"], dtype=np.float64)
    nb_omega_dps, span_nb = _savgol_1d(nb_elev_series, fps=fps, span_s=0.050, deriv=1)
    i_gather_lo = max(0, int(round(bfc_f - 0.04 * fps)))
    pk_nb_elev, f_pk_nb, _ = _subframe_peak(nb_elev_series, i_gather_lo, i_ffc)
    nb_at_ffc = float(_interp_frame(nb_elev_series, ffc_f))
    nb_early_drop = max(0.0, pk_nb_elev - nb_at_ffc)

    i_pull_lo = max(0, int(round(ffc_f - 0.02 * fps)))
    pk_nb_pull_dps, f_pk_pull, _ = _subframe_peak(-nb_omega_dps, i_pull_lo, i_br)
    pk_nb_pull_dps = max(0.0, pk_nb_pull_dps)

    def _fn_nb_el(p: np.ndarray) -> float:
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[1], max(math.hypot(v[0], v[2]), 1e-4))))

    nb_vis = in_fr.get("non_bowling_arm_elev_ffc_deg", True)
    sig_nb_pk = math.sqrt(_var_fn(_fn_nb_el, [nb_sho_i, nb_elb_i], f_pk_nb))
    sig_nb_pk_epi = math.sqrt(_var_fn(_fn_nb_el, [nb_sho_i, nb_elb_i], f_pk_nb, epi=True))
    sig_nb_drop = math.sqrt(_var_diff_fn(_fn_nb_el, [nb_sho_i, nb_elb_i], f_pk_nb, ffc_f))
    sig_nb_pull_ang = math.sqrt(_var_fn(_fn_nb_el, [nb_sho_i, nb_elb_i], f_pk_pull))
    rho_nb = math.exp(-max(span_nb, 2.0 / fps) / 0.08)
    sig_nb_pull_dps = math.sqrt(max(2.0 * (1.0 - rho_nb), 0.05)) * sig_nb_pull_ang / max(0.5 * span_nb, 0.015)

    me_nb_peak = _make_est(
        "lead_arm_gather_peak_elev_deg", "Lead-Arm Gather Peak Elevation", "deg", pk_nb_elev,
        sig_nb_pk, sig_nb_pk_epi, 2.0, cal.gravity_sigma_deg, 0.0,
        "sagittal" if sin_side >= 0.45 else "coronal", 16.5,
        "Peak elevation of non-bowling upper arm above horizontal during [BFC-40ms, FFC] (+90 = straight up).",
        "High front-arm gather (+35 to +75 deg) establishes a tall lever before front-foot plant.",
        "Portus et al. (2004); Ferdinands et al. (2010)",
        in_frame_ok=nb_vis,
        anchors=("bfc", "ffc"),
    )
    nb_om_ffc = abs(float(_interp_frame(nb_omega_dps, ffc_f)))
    sig_nb_drop_temp = max(2.5, nb_om_ffc * (float(events.ffc.timing_uncertainty_ms) / 1000.0))
    me_nb_drop = _make_est(
        "lead_arm_early_drop_pre_ffc_deg", "Lead-Arm Premature Drop Before FFC", "deg", nb_early_drop,
        sig_nb_drop, 0.75 * sig_nb_drop, sig_nb_drop_temp, 0.3 * cal.gravity_sigma_deg, 0.0,
        "sagittal" if sin_side >= 0.45 else "coronal", 20.5,
        "Drop in lead-arm elevation from gather peak down to FFC (large drop = collapsing front arm before foot plant).",
        "<=18 deg = delayed front-arm block held until FFC; >=38 deg = early front-arm collapse.",
        "Portus et al. (2004); Ferdinands et al. (2010)",
        in_frame_ok=nb_vis,
        anchors=("bfc", "ffc"),
    )
    me_nb_pull_vel = _make_est(
        "lead_arm_pulldown_peak_vel_dps", "Lead-Arm Pull-Down Peak Angular Speed", "deg/s", pk_nb_pull_dps,
        sig_nb_pull_dps, 0.75 * sig_nb_pull_dps, 20.0, 10.0, 0.0,
        "velocity_high", 420.0,
        "Peak downward Savitzky-Golay (~50 ms) angular speed of lead arm during [FFC, BR].",
        "Measures active front-side deceleration timing across [FFC, BR] (requires >=100 fps).",
        "Ferdinands et al. (2010)",
        in_frame_ok=nb_vis,
        anchors=("ffc", "br"),
        extra_suppress=seq_gate_msg,
    )

    nb_elb_br = _pt(nb_elb_i, br_f)
    rib_br = 0.5 * (_pt(nb_sho_i, br_f) + _pt(f_hip_i, br_f))
    flare_m = max(0.0, -sign_arm * float(nb_elb_br[2] - rib_br[2]))
    flare_pct = 100.0 * flare_m / stature_m
    sig_flare_pct = 100.0 * math.sqrt(max(float(_cov(nb_elb_i, br_f)[2, 2] + 0.5 * _cov(nb_sho_i, br_f)[2, 2]), 0.0)) / stature_m
    me_nb_flare = _make_est(
        "lead_elbow_ribcage_flare_br_pct_stature", "Lead-Elbow Lateral Flare at Release", "% stature", flare_pct,
        sig_flare_pct, 0.8 * sig_flare_pct, 1.0, 0.5 * cal.heading_sigma_deg, 0.0,
        "coronal", 9.5,
        "Contralateral lateral distance of lead elbow outside the ipsilateral shoulder-hip ribcage line at BR (% stature).",
        "<=8% stature = tight ribcage tuck; >=16% = wide 'flying' lead elbow pulling chest off target.",
        "Portus et al. (2004); Fast-bowling coaching kinematics",
        in_frame_ok=(nb_vis and head_br_vis),
        anchors=("br",),
    )

    if (
        not me_nb_drop.suppressed
        and me_nb_drop.value is not None
        and not me_nb_peak.suppressed
        and me_nb_peak.value is not None
    ):
        if me_nb_drop.ci_low is not None and me_nb_drop.ci_low >= 38.0:
            la_sum = f"Front arm dropped {int(round(me_nb_drop.value))}° before front-foot plant — hold the lead arm taller until landing."
        else:
            la_sum = f"Lead arm reaches {int(round(me_nb_peak.value))}° in the gather with controlled pre-plant hold ({int(round(me_nb_drop.value))}° drop)."
    elif not me_nb_flare.suppressed and me_nb_flare.value is not None:
        la_sum = f"Lead elbow tucks within {int(round(me_nb_flare.value))}% of height from the ribcage line at ball release."
    else:
        la_sum = "Lead-arm gather and ribcage pull-down require visible front-arm landmarks in slow-motion."

    lead_arm_obj = LeadArmAnalysis(
        summary_plain=la_sum,
        lead_arm_gather_peak_elev_deg=me_nb_peak,
        lead_arm_early_drop_pre_ffc_deg=me_nb_drop,
        lead_arm_pulldown_peak_vel_dps=me_nb_pull_vel,
        lead_elbow_ribcage_flare_br_pct_stature=me_nb_flare,
    )
    for me in (me_nb_peak, me_nb_drop, me_nb_pull_vel, me_nb_flare):
        v5_metrics[me.key] = me

    # =========================================================================
    # MODULE 5: Crease Alignment, Foot-Plant Geometry & 3D Release-Point Tunnel
    # =========================================================================
    r_ank_bfc = _pt(r_ank_i, bfc_f)
    f_ank_ffc = _pt(f_ank_i, ffc_f)
    dz_open_m = -sign_arm * float(f_ank_ffc[2] - r_ank_bfc[2])
    corridor_cm = 100.0 * dz_open_m
    dx_stride_m = max(float(f_ank_ffc[0] - r_ank_bfc[0]), 0.35)
    stride_align_deg = math.degrees(math.atan2(dz_open_m, dx_stride_m))

    var_dz_feet = float(_cov(r_ank_i, bfc_f)[2, 2] + _cov(f_ank_i, ffc_f)[2, 2])
    var_dz_feet_epi = float(_cov_epi(r_ank_i, bfc_f)[2, 2] + _cov_epi(f_ank_i, ffc_f)[2, 2])
    sig_corr_cm = 100.0 * math.sqrt(max(var_dz_feet, 0.0))
    sig_corr_epi_cm = 100.0 * math.sqrt(max(var_dz_feet_epi, 0.0))
    # Heading rotation sensitivity: 1 deg heading error shifts lateral foot corridor by dx_stride * sin(1 deg)
    sig_corr_cal_cm = 100.0 * dx_stride_m * math.radians(cal.heading_sigma_deg)

    foot_pan_gate: Optional[str] = None
    if cal.is_panning_camera or not cal.is_metric_calibrated:
        foot_pan_gate = "Suppressed: back-to-front foot corridor requires a static camera with calibrated MCC pitch/wicket geometry."

    me_corridor_cm = _make_est(
        "foot_plant_corridor_offset_cm", "Front-to-Back Foot Lateral Corridor", "cm", corridor_cm,
        sig_corr_cm, sig_corr_epi_cm, 2.0, sig_corr_cal_cm, 0.0,
        "coronal" if cos_rear >= 0.45 else "metric_scale", 22.0,
        "Lateral offset -sign_arm*(Z_FFC - Z_BFC) in cm (+ = open 'train-tracks' corridor; - = crossed-over).",
        "+5 to +25 cm = open 'train tracks' allowing clean hip clearance; <-5 cm = crossed-over.",
        "Bartlett et al. (1996); Portus et al. (2004)",
        in_frame_ok=stride_vis,
        anchors=("bfc", "ffc"),
        extra_suppress=foot_pan_gate,
    )
    me_stride_ang = _make_est(
        "stride_alignment_angle_deg", "Delivery Stride Alignment Angle", "deg", stride_align_deg,
        math.degrees(math.atan2(sig_corr_cm / 100.0, dx_stride_m)),
        math.degrees(math.atan2(sig_corr_epi_cm / 100.0, dx_stride_m)),
        1.2, cal.heading_sigma_deg, 0.0,
        "metric_scale" if sin_side >= 0.45 else "coronal", 11.5,
        "Horizontal angle of BFC->FFC ankle vector relative to +X pitch line (+ = open corridor, - = crossed).",
        "0 to +10 deg = parallel/open stride corridor; <-4 deg = cutting across the crease.",
        "Bartlett et al. (1996); Portus et al. (2004)",
        in_frame_ok=stride_vis,
        anchors=("bfc", "ffc"),
        extra_suppress=foot_pan_gate,
    )

    # Front-heel popping-crease margin (STRICTLY MCC Law 7 PnP + static camera + side-on!)
    if markers_world_87 is not None and markers_world_87.shape[:2] == (n, 87):
        hee_idx = BML_MOVI_87_IDX["lhee" if is_right else "rhee"]
        x_heel_ffc = float(_interp_frame(markers_world_87[:, hee_idx, 0], ffc_f))
    else:
        x_heel_ffc = float(f_ank_ffc[0]) - 0.045
    crease_margin_cm = 100.0 * (MCC_POPPING_CREASE_M - x_heel_ffc)
    reproj_err_m = float(cal.reprojection_error_px or 1.5) / max(float(cal.px_per_meter_at_crease or 180.0), 60.0)
    sig_crease_pose_cm = 100.0 * math.sqrt(max(float(_cov(f_ank_i, ffc_f)[0, 0]), 0.0))
    sig_crease_epi_cm = 100.0 * math.sqrt(max(float(_cov_epi(f_ank_i, ffc_f)[0, 0]), 0.0))
    foot_vx_arr = np.gradient(j[:, f_ank_i, 0], 1.0 / fps)
    vx_foot_ffc = float(np.max(np.abs(foot_vx_arr[max(0, i_ffc - 2) : min(n, i_ffc + 2)]))) if n >= 3 else 3.0
    sig_crease_temp_cm = 100.0 * max(vx_foot_ffc, 2.2) * (math.hypot(float(events.ffc.timing_uncertainty_ms), 5.5) / 1000.0)
    sig_crease_cal_cm = 100.0 * math.hypot(reproj_err_m, 0.055)

    crease_gate: Optional[str] = None
    if cal.method not in ("mcc_law7_pnp", "dual_3d_ray_fusion_dual_mcc_pnp") or (not cal.is_metric_calibrated) or cal.is_panning_camera:
        crease_gate = "Suppressed: front-foot popping-crease margin requires 4-click MCC Law 7 crease calibration on a static tripod."
    elif sin_side < 0.50 and not is_dual_orth:
        crease_gate = "Suppressed: front-heel crease margin requires a side-on (~90 deg) or dual-camera view of the popping crease."

    me_crease_cm = _make_est(
        "front_heel_crease_margin_cm", "Front-Heel Popping-Crease Margin (Practice)", "cm", crease_margin_cm,
        sig_crease_pose_cm, sig_crease_epi_cm, sig_crease_temp_cm, sig_crease_cal_cm, 0.0,
        "metric_scale", 22.0,
        "Estimated horizontal margin (1.22 m - X_heel_FFC) behind the popping crease (+ = behind line; practice estimate, NOT an umpire call).",
        "+2 to +14 cm = efficient crease usage without overstepping.",
        "MCC Law 7 Geometry (Practice Estimate)",
        in_frame_ok=ankle_ffc_vis,
        anchors=("ffc",),
        extra_suppress=crease_gate,
    )

    # 3D Release-Point Tunnel Metrics
    p_rel = _pt(b_han_i, br_f)
    rh_m = float(kin.scalars["release_height_m"])
    rh_pct = 100.0 * rh_m / stature_m
    sig_rh_m = math.sqrt(max(float(_cov(b_wri_i, br_f)[1, 1] + _cov(f_ank_i, br_f)[1, 1]), 0.0))
    sig_rh_epi_m = math.sqrt(max(float(_cov_epi(b_wri_i, br_f)[1, 1] + _cov_epi(f_ank_i, br_f)[1, 1]), 0.0))
    rh_gate = "Suppressed: uncalibrated intrinsics and torso-only gravity (>5.5 deg) prevent reliable release-height scaling." if ((not cal.is_intrinsics_calibrated) and cal.gravity_sigma_deg > 5.5) else None
    me_rh_pct = _make_est(
        "release_height_pct_stature", "Release Height (% Stature)", "% stature", rh_pct,
        100.0 * sig_rh_m / stature_m, 100.0 * sig_rh_epi_m / stature_m,
        1.2, 0.5 * cal.gravity_sigma_deg, 2.5,
        "sagittal" if sin_side >= cos_rear else "coronal", 11.0,
        "Ball release height above local ground divided by estimated bowler stature.",
        "High-arm fast bowlers typically release at ~106-116% of stature; slingier actions ~95-104%.",
        "Worthington et al. (2013); Phillips et al. (2012)",
        in_frame_ok=ankle_br_vis,
        anchors=("ffc", "br"),
        extra_suppress=rh_gate,
    )

    rel_lat_ank_m = sign_arm * float(p_rel[2] - f_ank_pt_br[2])
    sig_rla_m = math.sqrt(max(float(_cov(b_han_i, br_f)[2, 2] + _cov(f_ank_i, br_f)[2, 2]), 0.0))
    sig_rla_epi_m = math.sqrt(max(float(_cov_epi(b_han_i, br_f)[2, 2] + _cov_epi(f_ank_i, br_f)[2, 2]), 0.0))
    me_rel_ank = _make_est(
        "release_lateral_from_ankle_m", "Release Slot Width From Front Ankle", "m", rel_lat_ank_m,
        sig_rla_m, sig_rla_epi_m, 0.02, 0.02 * cal.gravity_sigma_deg, 0.0 if cal.is_metric_calibrated else 0.04 * abs(rel_lat_ank_m),
        "coronal", 0.22,
        "Horizontal release-hand width relative to planted front ankle (+ = toward bowling-arm side).",
        "0.02-0.22 m = high over-the-top ear-skimming slot; 0.35-0.65 m = wider round-arm slot.",
        "Phillips, Portus, Davids, & Renshaw (2012), JSMS",
        in_frame_ok=(ankle_br_vis and in_fr.get("peak_arm_angular_vel_dps", True)),
        anchors=("br",),
    )

    stump_gate: Optional[str] = None
    if cal.method not in ("mcc_law7_pnp", "two_wicket_geometry", "dual_3d_ray_fusion_dual_mcc_pnp") or cal.is_panning_camera or (not cal.is_metric_calibrated):
        stump_gate = "Suppressed: stump-referenced release width requires static calibrated wicket/crease geometry."
    rel_stump_z_m = float(p_rel[2])
    sig_rsz = math.sqrt(max(float(_cov(b_han_i, br_f)[2, 2]), 0.0))
    sig_rsz_cal = math.hypot(0.042, MCC_POPPING_CREASE_M * math.radians(cal.heading_sigma_deg))
    me_rel_stump = _make_est(
        "release_lateral_from_stump_m", "Release Lateral Offset From Middle Stump", "m", rel_stump_z_m,
        sig_rsz, math.sqrt(max(float(_cov_epi(b_han_i, br_f)[2, 2]), 0.0)), 0.025, sig_rsz_cal, 0.0,
        "coronal" if cos_rear >= 0.45 else "metric_scale", 0.25,
        "World +Z lateral release coordinate from middle stump (0,0,0) in meters.",
        "Over-the-wicket fast bowlers typically release ~0.45-0.95 m wide of middle stump.",
        "Phillips et al. (2012); MCC Law 8 Pitch Frame",
        in_frame_ok=in_fr.get("peak_arm_angular_vel_dps", True),
        anchors=("br",),
        extra_suppress=stump_gate,
    )

    rel_ext_m = float(p_rel[0]) - MCC_POPPING_CREASE_M
    sig_rext = math.sqrt(max(float(_cov(b_han_i, br_f)[0, 0]), 0.0))
    vx_hand_br = abs(float(np.gradient(j[:, b_han_i, 0], 1.0 / fps)[i_br])) if n >= 3 else 12.0
    sig_rext_temp = max(vx_hand_br * (math.hypot(float(events.br.timing_uncertainty_ms), 3.0) / 1000.0), 0.045)
    sig_rext_cal = math.hypot(reproj_err_m, 0.058)
    me_rel_ext = _make_est(
        "release_extension_past_crease_m", "Release Extension Past Popping Crease", "m", rel_ext_m,
        sig_rext, math.sqrt(max(float(_cov_epi(b_han_i, br_f)[0, 0]), 0.0)), sig_rext_temp, sig_rext_cal, 0.0,
        "metric_scale", 0.28,
        "Down-pitch release extension X_rel - 1.22 m past the popping crease.",
        "Forward trunk flexion typically projects ball release ~0.20-0.65 m in front of the popping crease.",
        "Worthington et al. (2013); MCC Law 7 Pitch Frame",
        in_frame_ok=in_fr.get("peak_arm_angular_vel_dps", True),
        anchors=("br",),
        extra_suppress=crease_gate,
    )

    if me_corridor_cm.suppressed or me_corridor_cm.value is None or me_corridor_cm.ci_low is None or me_corridor_cm.ci_high is None:
        corr_verdict = "unmeasurable"
        corr_lbl = "Needs Static Calibrated View"
        corr_sum = "Calibrate visible pitch creases on a steady tripod to measure your foot-plant corridor and crease margin."
    elif me_corridor_cm.ci_high < -3.0:
        corr_verdict = "crossed_over"
        corr_lbl = "Crossed-Over Alignment"
        corr_sum = f"Front foot planted {abs(int(round(me_corridor_cm.value)))} cm across your back-foot line, narrowing hip clearance."
    elif me_corridor_cm.ci_low > 4.0:
        corr_verdict = "train_tracks_open"
        corr_lbl = "Clean 'Train-Tracks' Corridor"
        corr_sum = f"Front foot planted {int(round(me_corridor_cm.value))} cm on an open corridor, giving the hips room to clear."
    else:
        corr_verdict = "tightrope_inline"
        corr_lbl = "Inline / Tightrope Corridor"
        corr_sum = f"Back and front feet landed nearly inline ({int(round(me_corridor_cm.value))} cm lateral offset)."

    crease_obj = CreaseReleaseTunnelAnalysis(
        corridor_verdict=corr_verdict,  # type: ignore[arg-type]
        corridor_label_plain=corr_lbl,
        summary_plain=corr_sum,
        foot_plant_corridor_offset_cm=me_corridor_cm,
        stride_alignment_angle_deg=me_stride_ang,
        front_heel_crease_margin_cm=me_crease_cm,
        release_height_pct_stature=me_rh_pct,
        release_lateral_from_ankle_m=me_rel_ank,
        release_lateral_from_stump_m=me_rel_stump,
        release_extension_past_crease_m=me_rel_ext,
    )
    for me in (me_corridor_cm, me_stride_ang, me_crease_cm, me_rh_pct, me_rel_ank, me_rel_stump, me_rel_ext):
        v5_metrics[me.key] = me

    # =========================================================================
    # MODULE 6: Delivery Tempo & Rhythm Fingerprint
    # =========================================================================
    stride_ms = float(events.delivery_stride_duration_ms)
    brake_ms = float(events.ffc_to_br_duration_ms)
    uah_br_ms = float((events.br.timestamp_s - events.uah.timestamp_s) * 1000.0) if events.uah is not None else None
    tempo_ratio = stride_ms / max(brake_ms, 25.0)

    s_bfc_ms = float(events.bfc.timing_uncertainty_ms)
    s_ffc_ms = math.hypot(float(events.ffc.timing_uncertainty_ms), 5.0)
    s_br_ms = float(events.br.timing_uncertainty_ms)
    s_uah_ms = float(events.uah.timing_uncertainty_ms) if events.uah is not None else 12.0

    sig_stride_ms = math.hypot(s_bfc_ms, s_ffc_ms)
    sig_brake_ms = math.hypot(s_ffc_ms, s_br_ms)
    sig_whip_ms = math.hypot(s_uah_ms, s_br_ms)
    # Exact shared-t_FFC Jacobian for R = (t_FFC - t_BFC) / (t_BR - t_FFC): dR/dt_FFC = (1 + R) / T_brake!
    sig_ratio = (1.0 / max(brake_ms, 30.0)) * math.sqrt(
        s_bfc_ms**2 + ((1.0 + tempo_ratio) * s_ffc_ms) ** 2 + (tempo_ratio * s_br_ms) ** 2
    )

    tb_gate: Optional[str] = None
    if audit.timebase_unreliable:
        tb_gate = f"Suppressed: edited or frozen video timebase ({audit.timebase_status}; {audit.held_frame_ratio:.0%} repeated frames) invalidates millisecond phase durations."

    me_t_stride = _make_est(
        "tempo_stride_duration_ms", "Delivery Stride Duration (BFC->FFC)", "ms", stride_ms,
        0.0, 0.0, sig_stride_ms, 0.0, 0.0,
        "sagittal" if sin_side >= cos_rear else "coronal", 65.0,
        "Elapsed time from back-foot contact to front-foot contact in milliseconds.",
        "Felton et al. (2019): 191.9 +- 31.3 ms (elite male), 172.3 +- 27.6 ms (elite female).",
        "Felton, Lister, Worthington, & King (2019), Sports Biomechanics",
        anchors=("bfc", "ffc"),
        extra_suppress=tb_gate,
    )
    me_t_brake = _make_est(
        "tempo_braking_to_br_ms", "Braking-to-Release Duration (FFC->BR)", "ms", brake_ms,
        0.0, 0.0, sig_brake_ms, 0.0, 0.0,
        "sagittal" if sin_side >= cos_rear else "coronal", 48.0,
        "Elapsed time from front-foot contact to ball release in milliseconds.",
        "Felton et al. (2019): 103.3 +- 11.2 ms (elite male), 127.7 +- 11.0 ms (elite female).",
        "Felton et al. (2019), Sports Biomechanics",
        anchors=("ffc", "br"),
        extra_suppress=tb_gate,
    )
    whip_gate = tb_gate
    if whip_gate is None and fps < 90.0:
        whip_gate = f"Suppressed: UAH->BR whip duration (~35-55 ms) requires >=100 fps slow-motion (clip is {fps:.1f} fps)."
    if whip_gate is None and uah_br_ms is None:
        whip_gate = "Suppressed: upper-arm horizontal (UAH) crossing was not isolated."
    me_t_whip = _make_est(
        "tempo_uah_to_br_whip_ms", "Final Arm-Whip Duration (UAH->BR)", "ms", uah_br_ms,
        0.0, 0.0, sig_whip_ms, 0.0, 0.0,
        "velocity_high", 28.0,
        "Elapsed time from upper-arm horizontal crossing (UAH) to ball release (BR).",
        "Fast-bowling final overhead whip typically spans ~32-55 ms.",
        "Portus et al. (2004); Felton et al. (2019)",
        anchors=("ffc", "br"),
        extra_suppress=whip_gate,
    )
    me_t_ratio = _make_est(
        "tempo_gather_to_whip_ratio", "Stride-to-Release Tempo Ratio", "ratio", tempo_ratio,
        0.0, 0.0, sig_ratio, 0.0, 0.0,
        "sagittal" if sin_side >= cos_rear else "coronal", 0.95,
        "Ratio of delivery stride duration (BFC->FFC) to braking/release phase (FFC->BR).",
        "Felton et al. (2019): ~1.86 +- 0.32 (elite male) and ~1.35 +- 0.25 (elite female) — ~2:1 gather-to-snap rhythm.",
        "Felton, Lister, Worthington, & King (2019)",
        anchors=("bfc", "ffc", "br"),
        extra_suppress=tb_gate,
    )

    waterfall: List[TempoPhaseSegment] = []
    if events.is_complete_delivery and not audit.timebase_unreliable:
        ref_stride = (145.0, 200.0) if is_female else (160.0, 224.0)
        ref_brake = (115.0, 140.0) if is_female else (92.0, 118.0)
        waterfall.append(
            TempoPhaseSegment(
                id="stride_bfc_ffc",
                label_plain="Back-foot land → front-foot plant",
                start_rel_ffc_ms=round(-stride_ms, 1),
                end_rel_ffc_ms=0.0,
                duration_ms=round(stride_ms, 1),
                uncertainty_ms=round(sig_stride_ms, 1),
                reference_range_ms=ref_stride,
            )
        )
        if events.uah is not None and uah_br_ms is not None and 0.0 < uah_br_ms < brake_ms:
            early_ms = brake_ms - uah_br_ms
            waterfall.append(
                TempoPhaseSegment(
                    id="early_braking_ffc_uah",
                    label_plain="Front-foot plant → arm level",
                    start_rel_ffc_ms=0.0,
                    end_rel_ffc_ms=round(early_ms, 1),
                    duration_ms=round(early_ms, 1),
                    uncertainty_ms=round(math.hypot(s_ffc_ms, s_uah_ms), 1),
                    reference_range_ms=(45.0, 75.0),
                )
            )
            waterfall.append(
                TempoPhaseSegment(
                    id="whip_uah_br",
                    label_plain="Arm level → ball release",
                    start_rel_ffc_ms=round(early_ms, 1),
                    end_rel_ffc_ms=round(brake_ms, 1),
                    duration_ms=round(uah_br_ms, 1),
                    uncertainty_ms=round(sig_whip_ms, 1),
                    reference_range_ms=(32.0, 56.0),
                )
            )
        else:
            waterfall.append(
                TempoPhaseSegment(
                    id="braking_ffc_br",
                    label_plain="Front-foot plant → ball release",
                    start_rel_ffc_ms=0.0,
                    end_rel_ffc_ms=round(brake_ms, 1),
                    duration_ms=round(brake_ms, 1),
                    uncertainty_ms=round(sig_brake_ms, 1),
                    reference_range_ms=ref_brake,
                )
            )
        if events.ft_peak is not None and events.ft_peak.timestamp_s > events.br.timestamp_s:
            ft_ms = (events.ft_peak.timestamp_s - events.br.timestamp_s) * 1000.0
            waterfall.append(
                TempoPhaseSegment(
                    id="follow_through",
                    label_plain="Ball release → follow-through",
                    start_rel_ffc_ms=round(brake_ms, 1),
                    end_rel_ffc_ms=round(brake_ms + ft_ms, 1),
                    duration_ms=round(ft_ms, 1),
                    uncertainty_ms=round(events.ft_peak.timing_uncertainty_ms, 1),
                    reference_range_ms=None,
                )
            )

    if not me_t_stride.suppressed and me_t_stride.value is not None and me_t_brake.value is not None and me_t_ratio.value is not None:
        tempo_sum = (
            f"Delivery stride takes {int(round(me_t_stride.value))} ms followed by a {int(round(me_t_brake.value))} ms "
            f"release snap ({me_t_ratio.value:.2f}:1 gather-to-snap tempo)."
        )
    else:
        tempo_sum = "Phase timing was withheld because this clip contains edited/repeated frames or no complete release."

    tempo_obj = TempoRhythmAnalysis(
        summary_plain=tempo_sum,
        tempo_stride_duration_ms=me_t_stride,
        tempo_braking_to_br_ms=me_t_brake,
        tempo_uah_to_br_whip_ms=me_t_whip,
        tempo_gather_to_whip_ratio=me_t_ratio,
        waterfall_phases=waterfall,
    )
    for me in (me_t_stride, me_t_brake, me_t_whip, me_t_ratio):
        v5_metrics[me.key] = me

    sequence_series = {
        "pelvis_omega_dps": pelvis_omega_dps,
        "thorax_omega_dps": thorax_omega_dps,
        "arm_omega_dps": arm_omega_dps,
        "wrist_whip_dps": wrist_whip_dps,
        "com_vx_mps": com_vx,
    }

    return V5DeepBiomechBundle(
        v5_metrics=v5_metrics,
        kinematic_sequence=kin_seq_obj,
        com_braking=com_obj,
        head_stability=head_obj,
        lead_arm=lead_arm_obj,
        crease_tunnel=crease_obj,
        tempo_rhythm=tempo_obj,
        com_world_m=com_sm,
        sequence_series=sequence_series,
        wrist_rel_speed_mps=wrist_rel_speed_mps,
    )


def synthesize_bml_movi_87_from_smpl24(
    joints_world: np.ndarray,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    anterior_pelvic_tilt_deg: float = 16.0,
    front_foot_splay_deg: float = 12.0,
) -> np.ndarray:
    """Synthesize anatomically consistent `(T, 87, 3)` ISB `bml_movi_87` bony markers from SMPL-24 world joints.

    Used strictly for Studio synthetic demo deliveries (`ses_demo` 4-ball spell) so ISB rigid-body
    pelvis, thorax, lumbopelvic, and foot-strike modules light up cleanly in offline demo mode.
    """
    n = joints_world.shape[0]
    m87 = np.zeros((n, 87, 3), dtype=np.float64)
    is_right = bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0

    pelv = joints_world[:, JOINT_IDX["pelv"]]
    lhip = joints_world[:, JOINT_IDX["lhip"]]
    rhip = joints_world[:, JOINT_IDX["rhip"]]
    spi2 = joints_world[:, JOINT_IDX["spi2"]]
    neck = joints_world[:, JOINT_IDX["neck"]]
    lsho = joints_world[:, JOINT_IDX["lsho"]]
    rsho = joints_world[:, JOINT_IDX["rsho"]]

    # Orthonormal pelvis frame from lhip/rhip + sagittal anterior tilt
    z_p = rhip - lhip
    z_p /= np.maximum(np.linalg.norm(z_p, axis=1, keepdims=True), 1e-6)
    up_w = np.broadcast_to(np.array([[0.0, 1.0, 0.0]]), z_p.shape)
    x_p_horiz = np.cross(up_w, z_p)
    x_p_horiz /= np.maximum(np.linalg.norm(x_p_horiz, axis=1, keepdims=True), 1e-6)
    y_p_horiz = np.cross(z_p, x_p_horiz)
    tilt_rad = math.radians(anterior_pelvic_tilt_deg)
    x_p = math.cos(tilt_rad) * x_p_horiz - math.sin(tilt_rad) * y_p_horiz

    m87[:, BML_MOVI_87_IDX["lasis"]] = pelv + 0.10 * x_p - 0.12 * z_p
    m87[:, BML_MOVI_87_IDX["rasis"]] = pelv + 0.10 * x_p + 0.12 * z_p
    m87[:, BML_MOVI_87_IDX["lpsis"]] = pelv - 0.09 * x_p - 0.08 * z_p
    m87[:, BML_MOVI_87_IDX["rpsis"]] = pelv - 0.09 * x_p + 0.08 * z_p

    # Orthonormal thorax frame from neck - spi2 and rsho - lsho
    y_t = neck - pelv
    y_t /= np.maximum(np.linalg.norm(y_t, axis=1, keepdims=True), 1e-6)
    z_t = rsho - lsho
    z_t = z_t - np.sum(z_t * y_t, axis=1, keepdims=True) * y_t
    z_t /= np.maximum(np.linalg.norm(z_t, axis=1, keepdims=True), 1e-6)
    x_t = np.cross(y_t, z_t)

    m87[:, BML_MOVI_87_IDX["backneck"]] = neck - 0.06 * x_t
    m87[:, BML_MOVI_87_IDX["clavicle"]] = neck - 0.03 * y_t + 0.07 * x_t
    m87[:, BML_MOVI_87_IDX["upperback"]] = spi2 - 0.08 * x_t
    m87[:, BML_MOVI_87_IDX["sternum"]] = spi2 + 0.09 * x_t
    m87[:, BML_MOVI_87_IDX["lshom"]] = lsho
    m87[:, BML_MOVI_87_IDX["rshom"]] = rsho

    # Left & right heel/toe markers with realistic heel-first touchdown geometry
    splay_r = math.radians(front_foot_splay_deg)
    f_dir = np.array([math.cos(splay_r), 0.0, -sign_arm * math.sin(splay_r)], dtype=np.float64)
    for side_prefix, is_front in (("l", is_right), ("r", not is_right)):
        ank = joints_world[:, JOINT_IDX[f"{side_prefix}ank"]]
        toe = joints_world[:, JOINT_IDX[f"{side_prefix}toe"]]
        if is_front:
            m87[:, BML_MOVI_87_IDX[f"{side_prefix}hee"]] = ank - 0.05 * f_dir - np.array([0.0, 0.015, 0.0])
            m87[:, BML_MOVI_87_IDX[f"{side_prefix}toe"]] = ank + 0.17 * f_dir + np.array([0.0, 0.018, 0.0])
        else:
            m87[:, BML_MOVI_87_IDX[f"{side_prefix}hee"]] = ank - np.array([0.05, 0.0, 0.0])
            m87[:, BML_MOVI_87_IDX[f"{side_prefix}toe"]] = toe
    return m87
