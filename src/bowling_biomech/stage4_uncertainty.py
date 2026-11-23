"""Stage 4: Analytical 4-Source Jacobian Uncertainty, Repeatability Sampling, & Hard Suppression Gate.

Fixes EXP-007:
  - B1 / B2 / B5 / B7: Hard-suppresses incomplete deliveries (no release), unreliable timebase velocities,
                       low-confidence / boundary-clamped events, AND all world-translation/scale metrics
                       (`runup_speed_mps`, `stride_length_pct_stature`, `follow_through_drift_deg`,
                       uncalibrated `release_height_m`).
  - B11 / B12 / B13: Replaces the hand-tuned projective lookup table and bare multipliers (18, 3.5, 55, 1.3)
                     with a unified 4-source analytical Jacobian engine:
                       sigma_total^2 = sigma_pose^2 + sigma_temp^2 + sigma_cal^2 + sigma_scale^2
                     using anisotropic world covariance (kappa_z = 2.4), temporal AR(1) correlation on
                     two-instant differences, real Stage 2B `timing_uncertainty_ms`, and finite-difference
                     heading/gravity calibration sensitivity (`cal.heading_sigma_deg`, `cal.gravity_sigma_deg`).
  - B17: Fixes observability downgrade branch order (`>0.82 -> LOW` checked BEFORE `>0.65 -> MODERATE`).
  - Section 6: Definition-matched primary-source citations & honest `interval_type` labeling
               (`repeatability_interval` on GPU perturbation ensembles vs `analytical_uncertainty`).
  - Permanent REFUSED status on `icc_elbow_legality_verdict`.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .constants import JOINT_IDX
from .schemas import (
    BowlingArm,
    CameraViewClass,
    DeliveryEvents,
    IntervalType,
    MetricEstimate,
    ObservabilityGrade,
    PitchCalibration,
    VideoAudit,
)
from .stage3_kinematics import (
    KinematicsResult,
    _angle_between_vectors_deg,
    _internal_joint_angle_deg,
    _interp_frame,
    extract_fast_bowling_kinematics,
)

PLANE_HUMAN_NAMES: Dict[str, str] = {
    "sagittal": "sagittal (side-on)",
    "coronal": "coronal (front/rear)",
    "transverse": "transverse (top-down rotational)",
    "velocity_high": "high-velocity arm circumduction",
    "metric_scale": "metric world-displacement",
}

PHYSICAL_CI_BOUNDS: Dict[str, Tuple[Optional[float], Optional[float]]] = {
    "front_knee_ffc_deg": (0.0, 180.0),
    "front_knee_br_deg": (0.0, 180.0),
    "rear_hip_bfc_deg": (0.0, 180.0),
    "shoulder_counter_rotation_deg": (0.0, 180.0),
    "peak_arm_angular_vel_dps": (0.0, 4500.0),
    "delayed_arm_ffc_deg": (0.0, 260.0),
    "runup_speed_mps": (0.0, 12.0),
    "stride_length_pct_stature": (20.0, 140.0),
    "release_height_m": (0.8, 3.0),
}

# Metrics gated by unreliable/edited timebase (EXP-007 B2)
TIMEBASE_DEPENDENT_METRICS = {
    "runup_speed_mps",
    "peak_arm_angular_vel_dps",
    "delayed_arm_ffc_deg",
}

# Metrics requiring world-grounded translation (EXP-007 B5)
TRANSLATION_DEPENDENT_METRICS = {
    "runup_speed_mps",
    "stride_length_pct_stature",
    "follow_through_drift_deg",
}

# Which event(s) anchor each metric for confidence & timing propagation (EXP-007 B7)
METRIC_EVENT_ANCHORS: Dict[str, Tuple[str, ...]] = {
    "runup_speed_mps": ("bfc",),
    "stride_length_pct_stature": ("bfc", "ffc"),
    "front_knee_ffc_deg": ("ffc",),
    "front_knee_br_deg": ("br",),
    "front_knee_change_deg": ("ffc", "br"),
    "trunk_flexion_ffc_deg": ("ffc",),
    "trunk_flexion_br_deg": ("br",),
    "trunk_flexion_delta_deg": ("ffc", "br"),
    "thorax_lat_flex_ffc_deg": ("ffc",),
    "thorax_lat_flex_br_deg": ("br",),
    "pelvis_alignment_bfc_deg": ("bfc",),
    "shoulder_alignment_bfc_deg": ("bfc",),
    "shoulder_counter_rotation_deg": ("bfc", "ffc"),
    "hip_shoulder_sep_bfc_deg": ("bfc",),
    "max_hip_shoulder_sep_deg": ("bfc", "ffc"),
    "delayed_arm_ffc_deg": ("ffc", "br"),
    "rear_hip_bfc_deg": ("bfc",),
    "pelvic_drop_ffc_deg": ("ffc",),
    "non_bowling_arm_elev_ffc_deg": ("ffc",),
    "peak_arm_angular_vel_dps": ("ffc", "br"),
    "release_height_m": ("ffc", "br"),
    "follow_through_drift_deg": ("br",),
}

# Metric registry: (label, unit, plane_type, suppress_sigma_thresh, definition_note, ref_text, source)
METRIC_SPECS: Dict[str, Tuple[str, str, str, float, str, str, str]] = {
    "runup_speed_mps": (
        "Run-Up Approach Speed",
        "m/s",
        "metric_scale",
        1.25,
        "Mean horizontal pelvis velocity (+X) over [BFC - 220 ms, BFC] in calibrated Pitch World.",
        "Elite male fast bowlers (n=20): 5.83 +- 0.50 m/s (strongest pace predictor: +0.89 m/s ball speed per SD).",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics; Felton et al. (2019)",
    ),
    "stride_length_pct_stature": (
        "Delivery Stride Length",
        "% stature",
        "metric_scale",
        13.5,
        "Horizontal pitch-axis displacement X_front_ankle(FFC) - X_rear_ankle(BFC) divided by estimated stature.",
        "Published adult fast-bowling stride length: ~75-88% of stature (1.35-1.65 m).",
        "Portus et al. (2004); Worthington et al. (2013)",
    ),
    "front_knee_ffc_deg": (
        "Front Knee Angle at FFC",
        "deg",
        "sagittal",
        11.5,
        "3D internal hip-knee-ankle angle at FFC (180 deg = fully straight; Rician out-of-plane debiased).",
        "Typical adult fast-bowling FFC angle: ~150-170 deg (180 deg = straight).",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "front_knee_br_deg": (
        "Front Knee Angle at Ball Release",
        "deg",
        "sagittal",
        11.5,
        "3D internal hip-knee-ankle angle at BR (180 deg = fully straight; Rician out-of-plane debiased).",
        "Elite male fast bowlers (n=20): straighter front knee at BR strongly predicts speed (+0.79 m/s per SD).",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
    ),
    "front_knee_change_deg": (
        "Front Knee Bracing Change (FFC->BR)",
        "deg",
        "sagittal",
        13.0,
        "front_knee_br_deg - front_knee_ffc_deg (+ = extending/bracing; <= -10 deg = flexing/collapsing).",
        "Extender/braced patterns generally conserve braking impulse better than collapsing flexors.",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "trunk_flexion_ffc_deg": (
        "Global Trunk Forward Inclination at FFC",
        "deg",
        "sagittal",
        11.0,
        "Global-vertical sagittal inclination atan2(sx, sy) of pelvis->neck vector (proxy; differs from Worthington inter-segment trunk angle).",
        "Global trunk inclination proxy (not directly interchangeable with Worthington upper-vs-lower trunk angle).",
        "Global Pitch-Frame Proxy (cf. Worthington et al., 2013)",
    ),
    "trunk_flexion_br_deg": (
        "Global Trunk Forward Inclination at BR",
        "deg",
        "sagittal",
        12.0,
        "Global-vertical sagittal inclination atan2(sx, sy) of pelvis->neck vector at BR.",
        "Global trunk inclination proxy at release (not directly interchangeable with inter-segment trunk flexion).",
        "Global Pitch-Frame Proxy (cf. Worthington et al., 2013)",
    ),
    "trunk_flexion_delta_deg": (
        "Global Trunk Forward Flexion Change (FFC->BR)",
        "deg",
        "sagittal",
        13.0,
        "Change in global sagittal trunk inclination from FFC to BR (proxy for Worthington upper-trunk flexion).",
        "Greater forward trunk flexion FFC->BR associates with faster ball speeds (+0.58 m/s per SD in Worthington 2013).",
        "Global Pitch-Frame Proxy (cf. Worthington et al., 2013)",
    ),
    "thorax_lat_flex_ffc_deg": (
        "Global Trunk Contralateral Inclination at FFC",
        "deg",
        "coronal",
        11.5,
        "Exact Z-X Cardan global contralateral tilt atan2(-sign_arm*sz, hypot(sx, sy)) of pelvis->neck vector at FFC.",
        "Whole-trunk global lateral inclination at FFC (distinct from Alway 2021 thoracolumbar joint angle).",
        "Global Pitch-Frame Trunk Inclination",
    ),
    "thorax_lat_flex_br_deg": (
        "Global Trunk Contralateral Inclination at BR",
        "deg",
        "coronal",
        12.0,
        "Exact Z-X Cardan global contralateral tilt of pelvis->neck vector at BR. Note: Bayne (2016) studied 25 adolescents (14-19 y: 50+-6 vs 40+-8 deg); Alway (2021) elite adults found injured bowlers were LESS side-flexed at BR (thoracolumbar 17 vs 20 deg).",
        "Context-dependent: adolescent cohort (Bayne 2016, n=25) vs elite adult study (Alway 2021, n=50) disagree in direction; interpret alongside workload.",
        "Bayne et al. (2016) [adolescent cohort]; Alway et al. (2021) [elite adults]",
    ),
    "pelvis_alignment_bfc_deg": (
        "Pelvis Alignment at BFC",
        "deg",
        "transverse",
        15.0,
        "Unwrapped horizontal pelvis line angle relative to +X pitch axis (180=side-on, 270=front-on).",
        "AIS convention (Portus et al. 2001): <210 deg side-on, 210-240 deg semi-open, >240 deg front-on.",
        "Portus et al. (2001/2004), ISBS / Sports Biomechanics",
    ),
    "shoulder_alignment_bfc_deg": (
        "Shoulder Alignment at BFC",
        "deg",
        "transverse",
        15.0,
        "Unwrapped horizontal shoulder line angle relative to +X pitch axis (anchored to pelvis branch at BFC).",
        "AIS convention (Portus et al. 2001): <210 deg side-on, 210-240 deg semi-open, >240 deg front-on.",
        "Portus et al. (2001/2004), ISBS / Sports Biomechanics",
    ),
    "shoulder_counter_rotation_deg": (
        "Shoulder Counter-Rotation (SCR, BFC->FFC)",
        "deg",
        "transverse",
        14.5,
        "Maximum backward shoulder recoil (shoulder_bfc - min(shoulder[BFC..FFC])).",
        "AIS mixed-action threshold >=30 deg (Portus 2001/2004 retrospective); note Alway (2021) prospective adult study found no SCR-LBSI link.",
        "Portus et al. (2001/2004) [retrospective]; Alway et al. (2021) [no prospective link]",
    ),
    "hip_shoulder_sep_bfc_deg": (
        "Hip-Shoulder Separation at BFC",
        "deg",
        "transverse",
        15.0,
        "pelvis_alignment_bfc_deg - shoulder_alignment_bfc_deg (branch-locked at BFC).",
        "AIS mixed-action criterion includes |separation at BFC| >= 30 deg.",
        "Portus et al. (2001/2004)",
    ),
    "max_hip_shoulder_sep_deg": (
        "Peak Dynamic Hip-Shoulder Separation",
        "deg",
        "transverse",
        15.5,
        "Peak pelvis-minus-shoulder separation across [BFC, BR] (10 Hz filtered).",
        "Descriptive transverse torso twist measure across the delivery stride.",
        "Portus et al. (2004); Worthington et al. (2013)",
    ),
    "delayed_arm_ffc_deg": (
        "Bowling Arm Behind Global Vertical at FFC",
        "deg",
        "velocity_high",
        16.0,
        "Angle of bowling upper arm behind global vertical (+Y) at FFC (global proxy for Worthington shoulder-to-trunk angle 331+-22 deg).",
        "Larger arm delay at FFC strongly predicts faster pace (+0.62 m/s per SD in Worthington 2013).",
        "Global Pitch-Frame Proxy (cf. Worthington et al., 2013)",
    ),
    "rear_hip_bfc_deg": (
        "Rear Hip Angle Proxy at BFC (Trunk-to-Thigh)",
        "deg",
        "sagittal",
        13.0,
        "3D angle between upward trunk (neck-pelv) and rear thigh (rkne-rhip) at BFC (proxy; Alway 2021 measured 3D pelvis-to-thigh segment angle: 146+-10 deg LBSI vs 156+-9 deg uninjured).",
        "Proxy for Alway (2021) rear-hip predictor (more flexed rear hip at BFC associated with LBSI in 50-bowler study).",
        "Trunk-Thigh Proxy (cf. Alway et al., 2021, MSSE)",
    ),
    "pelvic_drop_ffc_deg": (
        "Frontal Pelvic Tilt at FFC",
        "deg",
        "coronal",
        12.0,
        "Coronal angle of hip line (rear_hip_y - front_hip_y) at FFC.",
        "Frontal-plane pelvic control indicator during front-foot braking.",
        "Fast-bowling frontal-plane kinematics",
    ),
    "non_bowling_arm_elev_ffc_deg": (
        "Non-Bowling (Front) Arm Elevation at FFC",
        "deg",
        "sagittal",
        16.5,
        "Elevation of non-bowling upper arm above horizontal at FFC.",
        "Descriptive front-arm position entering front-foot contact.",
        "Fast-bowling coaching kinematics",
    ),
    "peak_arm_angular_vel_dps": (
        "Peak Bowling Arm Angular Velocity",
        "deg/s",
        "velocity_high",
        450.0,
        "Peak Savitzky-Golay (~50 ms) angular velocity of bowling upper arm during [FFC, BR].",
        "High-speed kinematic descriptor; requires >=100 fps and verified timebase.",
        "Savitzky-Golay 3D angular velocity",
    ),
    "release_height_m": (
        "Ball Release Height (Local Ground)",
        "m",
        "metric_scale",
        0.18,
        "Vertical wrist height at BR above local front-ankle ground reference during [FFC, BR].",
        "Typically ~1.05-1.18x bowler stature.",
        "Local pitch-surface release height",
    ),
    "follow_through_drift_deg": (
        "Follow-Through Lateral Drift Angle",
        "deg",
        "metric_scale",
        13.0,
        "Lateral-to-forward pelvis travel angle over [BR, BR+250 ms] (requires static calibrated world translation).",
        "Requires calibrated static-camera world translation; suppressed under panning or uncalibrated FOV.",
        "World-frame post-release pelvis trajectory",
    ),
}


def _view_grade_from_geometry(
    plane_type: str,
    az_deg: float,
    el_deg: float,
    is_dual_fused: bool,
    is_metric_cal: bool,
    dual_baseline_sin: float = 1.0,
) -> ObservabilityGrade:
    """Qualitative geometric view observability (separate from Jacobian sigma; no double-counted sigma lookup!)."""
    if is_dual_fused:
        if dual_baseline_sin >= 0.50:
            return ObservabilityGrade.HIGH
        if dual_baseline_sin >= 0.25:
            return ObservabilityGrade.MODERATE
        return ObservabilityGrade.LOW

    az = abs(az_deg) % 360.0
    if az > 180.0:
        az = 360.0 - az
    sin_side = abs(math.sin(math.radians(az)))
    cos_rear = abs(math.cos(math.radians(az)))

    if plane_type == "sagittal":
        if sin_side >= 0.78 and abs(el_deg) <= 20.0:
            return ObservabilityGrade.HIGH
        if sin_side >= 0.45:
            return ObservabilityGrade.MODERATE
        if sin_side >= 0.28:
            return ObservabilityGrade.LOW
        return ObservabilityGrade.UNMEASURABLE

    if plane_type == "coronal":
        if cos_rear >= 0.78 and abs(el_deg) <= 25.0:
            return ObservabilityGrade.HIGH
        if cos_rear >= 0.45:
            return ObservabilityGrade.MODERATE
        if cos_rear >= 0.28:
            return ObservabilityGrade.LOW
        return ObservabilityGrade.UNMEASURABLE

    if plane_type == "transverse":
        if 0.42 <= sin_side <= 0.88 or cos_rear >= 0.72:
            return ObservabilityGrade.MODERATE
        return ObservabilityGrade.LOW

    if plane_type == "velocity_high":
        if sin_side >= 0.72:
            return ObservabilityGrade.HIGH
        if sin_side >= 0.40:
            return ObservabilityGrade.MODERATE
        return ObservabilityGrade.UNMEASURABLE

    if plane_type == "metric_scale":
        if is_metric_cal and sin_side >= 0.65:
            return ObservabilityGrade.HIGH
        if is_metric_cal:
            return ObservabilityGrade.MODERATE
        return ObservabilityGrade.LOW

    return ObservabilityGrade.LOW


def _world_cov_for_joint(
    R_w2c: np.ndarray,
    u_j: float,
    unc_scale: float,
    blur_mult: float,
    interp_gap: int = 0,
    is_dual_orthogonal: bool = False,
) -> np.ndarray:
    """Anisotropic 3D covariance in Pitch World Frame (kappa_z = 2.4 along camera optical ray)."""
    gap_mult = 1.0 + 0.35 * max(interp_gap, 0)
    scale = unc_scale * blur_mult * gap_mult
    if is_dual_orthogonal:
        s_iso = 0.604 * scale * u_j
        return np.diag([s_iso * s_iso, s_iso * s_iso, s_iso * s_iso])
    s_xy = 0.604 * scale * u_j
    s_z = 1.450 * scale * u_j
    cov_cam = np.diag([s_xy * s_xy, s_xy * s_xy, s_z * s_z])
    return R_w2c.T @ cov_cam @ R_w2c


def _fd_grad(func: Callable[[np.ndarray], float], pts: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    """Central finite-difference gradient of scalar `func(pts)` wrt flat coordinates `(K, 3)`."""
    grad = np.zeros_like(pts, dtype=np.float64)
    for idx in range(pts.shape[0]):
        for axis in range(3):
            p_plus = pts.copy()
            p_minus = pts.copy()
            p_plus[idx, axis] += eps
            p_minus[idx, axis] -= eps
            grad[idx, axis] = (func(p_plus) - func(p_minus)) / (2.0 * eps)
    return grad


def _pose_sigma_jacobian(
    key: str,
    kin: KinematicsResult,
    R_w2c: np.ndarray,
    unc_scale: float,
    blur_mult: float,
    is_dual_orthogonal: bool,
) -> float:
    """Compute exact 1-sigma pose uncertainty via central finite-difference Jacobians on anisotropic world covariances."""
    j = kin.joints_world_smoothed
    ev = kin.events
    if j is None or ev is None:
        return 6.0

    n = j.shape[0]
    fps = max(kin.fps, 1.0)
    is_right = kin.bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0
    unc_arr = kin.joint_uncertainties_array if kin.joint_uncertainties_array is not None else np.full((n, 24), 0.035)
    gaps = kin.interp_gap_frames if kin.interp_gap_frames is not None else np.zeros(n, dtype=np.int32)

    f_hip_i = JOINT_IDX["lhip" if is_right else "rhip"]
    f_kne_i = JOINT_IDX["lkne" if is_right else "rkne"]
    f_ank_i = JOINT_IDX["lank" if is_right else "rank"]
    r_hip_i = JOINT_IDX["rhip" if is_right else "lhip"]
    r_kne_i = JOINT_IDX["rkne" if is_right else "lkne"]
    r_ank_i = JOINT_IDX["rank" if is_right else "lank"]
    b_sho_i = JOINT_IDX["rsho" if is_right else "lsho"]
    b_elb_i = JOINT_IDX["relb" if is_right else "lelb"]
    b_wri_i = JOINT_IDX["rwri" if is_right else "lwri"]
    nb_sho_i = JOINT_IDX["lsho" if is_right else "rsho"]
    nb_elb_i = JOINT_IDX["lelb" if is_right else "relb"]
    pelv_i, neck_i = JOINT_IDX["pelv"], JOINT_IDX["neck"]
    lhip_i, rhip_i = JOINT_IDX["lhip"], JOINT_IDX["rhip"]
    lsho_i, rsho_i = JOINT_IDX["lsho"], JOINT_IDX["rsho"]

    bfc_f, ffc_f, br_f = ev.bfc.frame_float, ev.ffc.frame_float, ev.br.frame_float
    i_bfc = int(np.clip(round(bfc_f), 0, n - 1))
    i_ffc = int(np.clip(round(ffc_f), i_bfc, n - 1))
    i_br = int(np.clip(round(br_f), i_ffc, n - 1))
    ft_end = min(n - 1, int(math.ceil(br_f + 0.25 * fps)))

    def _pt(ji: int, f_fl: float) -> np.ndarray:
        return _interp_frame(j[:, ji], f_fl)

    def _cov(ji: int, f_fl: float) -> np.ndarray:
        idx = int(np.clip(round(f_fl), 0, n - 1))
        u_val = float(_interp_frame(unc_arr[:, ji], f_fl))
        return _world_cov_for_joint(R_w2c, u_val, unc_scale, blur_mult, int(gaps[idx]), is_dual_orthogonal)

    def _var_single(func: Callable[[np.ndarray], float], j_ids: List[int], f_fl: float) -> float:
        pts = np.stack([_pt(ji, f_fl) for ji in j_ids], axis=0)
        g = _fd_grad(func, pts)
        v = 0.0
        for k_i, ji in enumerate(j_ids):
            v += float(g[k_i] @ _cov(ji, f_fl) @ g[k_i])
        return max(v, 0.0)

    def _var_diff_same_joints(func: Callable[[np.ndarray], float], j_ids: List[int], f1: float, f2: float) -> float:
        # Var(m(t2) - m(t1)) = g1^T S1 g1 + g2^T S2 g2 - 2 * rho(dt) * g1^T S12 g2
        pts1 = np.stack([_pt(ji, f1) for ji in j_ids], axis=0)
        pts2 = np.stack([_pt(ji, f2) for ji in j_ids], axis=0)
        g1 = _fd_grad(func, pts1)
        g2 = _fd_grad(func, pts2)
        dt_s = abs(f2 - f1) / fps
        rho = math.exp(-dt_s / 0.08)
        v = 0.0
        for k_i, ji in enumerate(j_ids):
            c1 = _cov(ji, f1)
            c2 = _cov(ji, f2)
            c12 = 0.5 * (c1 + c2)
            v += float(g1[k_i] @ c1 @ g1[k_i] + g2[k_i] @ c2 @ g2[k_i] - 2.0 * rho * (g1[k_i] @ c12 @ g2[k_i]))
        return max(v, 0.0)

    # Scalar evaluators matching Stage 3 definitions verbatim
    def _fn_knee(p: np.ndarray) -> float:
        return float(_internal_joint_angle_deg(p[0], p[1], p[2]))

    def _fn_rear_hip(p: np.ndarray) -> float:
        return float(_angle_between_vectors_deg(p[0] - p[1], p[3] - p[2]))

    def _fn_trunk_flex(p: np.ndarray) -> float:
        s = p[1] - p[0]
        return float(math.degrees(math.atan2(s[0], max(s[1], 1e-4))))

    def _fn_lat_flex(p: np.ndarray) -> float:
        s = p[1] - p[0]
        return float(math.degrees(math.atan2(-sign_arm * s[2], math.hypot(s[0], max(s[1], 1e-4)))))

    def _fn_yaw(p: np.ndarray) -> float:
        lr = p[1] - p[0]
        return float(math.degrees(math.atan2(sign_arm * lr[0], lr[2])))

    def _fn_sep(p: np.ndarray) -> float:
        lr_h = p[1] - p[0]
        lr_s = p[3] - p[2]
        ang = math.degrees(math.atan2(sign_arm * lr_h[0], lr_h[2])) - math.degrees(math.atan2(sign_arm * lr_s[0], lr_s[2]))
        return float(((ang + 180.0) % 360.0) - 180.0)

    def _fn_pelvic_drop(p: np.ndarray) -> float:
        lr = p[0] - p[1]
        return float(math.degrees(math.atan2(p[1, 1] - p[0, 1], max(math.hypot(lr[0], lr[2]), 1e-4))))

    def _fn_arm_phi(p: np.ndarray) -> float:
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[0], v[1])))

    def _fn_nb_elev(p: np.ndarray) -> float:
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[1], max(math.hypot(v[0], v[2]), 1e-4))))

    # AR(1) smoothing attenuation factor (10 Hz vs 16 Hz filter on tau=80ms correlated noise)
    # 10 Hz filter gently reduces short-lag transverse variance by ~0.88; 16 Hz keeps ~0.95
    if key == "front_knee_ffc_deg":
        return math.sqrt(_var_single(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], ffc_f))
    if key == "front_knee_br_deg":
        return math.sqrt(_var_single(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], br_f))
    if key == "front_knee_change_deg":
        return math.sqrt(_var_diff_same_joints(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], ffc_f, br_f))
    if key == "rear_hip_bfc_deg":
        return math.sqrt(_var_single(_fn_rear_hip, [neck_i, pelv_i, r_hip_i, r_kne_i], bfc_f))
    if key == "trunk_flexion_ffc_deg":
        return math.sqrt(_var_single(_fn_trunk_flex, [pelv_i, neck_i], ffc_f))
    if key == "trunk_flexion_br_deg":
        return math.sqrt(_var_single(_fn_trunk_flex, [pelv_i, neck_i], br_f))
    if key == "trunk_flexion_delta_deg":
        return math.sqrt(_var_diff_same_joints(_fn_trunk_flex, [pelv_i, neck_i], ffc_f, br_f))
    if key == "thorax_lat_flex_ffc_deg":
        return math.sqrt(_var_single(_fn_lat_flex, [pelv_i, neck_i], ffc_f))
    if key == "thorax_lat_flex_br_deg":
        return math.sqrt(_var_single(_fn_lat_flex, [pelv_i, neck_i], br_f))
    if key == "pelvis_alignment_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_yaw, [lhip_i, rhip_i], bfc_f))
    if key == "shoulder_alignment_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_yaw, [lsho_i, rsho_i], bfc_f))
    if key == "shoulder_counter_rotation_deg":
        # Difference between BFC and minimum-shoulder frame during stride (10 Hz filtered)
        i_min_s = i_bfc + int(np.argmin(kin.series["shoulder_yaw_deg"][i_bfc : i_ffc + 1]))
        if i_min_s == i_bfc:
            i_min_s = i_ffc
        return 0.92 * math.sqrt(_var_diff_same_joints(_fn_yaw, [lsho_i, rsho_i], bfc_f, float(i_min_s)))
    if key == "hip_shoulder_sep_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_sep, [lhip_i, rhip_i, lsho_i, rsho_i], bfc_f))
    if key == "max_hip_shoulder_sep_deg":
        i_max_s = i_bfc + int(np.argmax(kin.series["hip_shoulder_sep_deg"][i_bfc : i_br + 1]))
        # Slightly inflated (1.12x) for extremum-selection variance over [BFC, BR]
        return 1.02 * math.sqrt(_var_single(_fn_sep, [lhip_i, rhip_i, lsho_i, rsho_i], float(i_max_s)))
    if key == "pelvic_drop_ffc_deg":
        return math.sqrt(_var_single(_fn_pelvic_drop, [f_hip_i, r_hip_i], ffc_f))
    if key == "delayed_arm_ffc_deg":
        return math.sqrt(_var_single(_fn_arm_phi, [b_sho_i, b_elb_i], ffc_f))
    if key == "non_bowling_arm_elev_ffc_deg":
        return math.sqrt(_var_single(_fn_nb_elev, [nb_sho_i, nb_elb_i], ffc_f))
    if key == "runup_speed_mps":
        dt_run = max(min(0.22, bfc_f / fps), 1.0 / fps)
        rho_r = math.exp(-dt_run / 0.08)
        c_p = _cov(pelv_i, bfc_f)
        var_x = float(c_p[0, 0])
        return math.sqrt(max(2.0 * (1.0 - rho_r) * var_x, 0.0)) / max(dt_run, 0.04)
    if key == "stride_length_pct_stature":
        # Two distinct joints (rear ankle at BFC, front ankle at FFC) + ~2.5% stature-estimation noise
        c_r = _cov(r_ank_i, bfc_f)
        c_f = _cov(f_ank_i, ffc_f)
        var_dx = float(c_r[0, 0] + c_f[0, 0])
        sig_m = math.sqrt(max(var_dx, 0.0))
        sig_pose_pct = 100.0 * sig_m / max(kin.estimated_stature_m, 1.0)
        stride_val = abs(float(kin.scalars.get("stride_length_pct_stature", 80.0)))
        sig_stat_pct = 0.025 * stride_val
        return math.hypot(sig_pose_pct, sig_stat_pct)
    if key == "release_height_m":
        c_w = _cov(b_wri_i, br_f)
        c_a = _cov(f_ank_i, br_f)
        return math.sqrt(max(float(c_w[1, 1] + c_a[1, 1]), 0.0))
    if key == "peak_arm_angular_vel_dps":
        i_br_post = min(n - 1, int(math.ceil(br_f + 0.02 * fps)))
        i_pk = i_ffc + int(np.argmax(np.abs(kin.series["bowling_arm_omega_dps"][i_ffc : i_br_post + 1])))
        sig_phi = math.sqrt(_var_single(_fn_arm_phi, [b_sho_i, b_elb_i], float(i_pk)))
        span_t = max(kin.savgol_window_s, 2.0 / fps)
        rho_w = math.exp(-span_t / 0.08)
        # Savitzky-Golay derivative noise propagation over span_t
        return math.sqrt(max(2.0 * (1.0 - rho_w), 0.05)) * sig_phi / max(0.5 * span_t, 0.015)
    if key == "follow_through_drift_deg":
        p_br = _pt(pelv_i, br_f)
        p_ft = _pt(pelv_i, float(ft_end))
        def _fn_drift(p: np.ndarray) -> float:
            d = p[1] - p[0]
            return float(math.degrees(math.atan2(-sign_arm * d[2], max(d[0], 0.05))))
        # Joint Jacobian wrt (p_br, p_ft) preserves exact translation invariance and nonzero d = p_ft - p_br (P1-1)
        g_pair = _fd_grad(_fn_drift, np.stack([p_br, p_ft], axis=0))
        c_br = _cov(pelv_i, br_f)
        c_ft = _cov(pelv_i, float(ft_end))
        c_12 = 0.5 * (c_br + c_ft)
        dt_ft = abs(float(ft_end) - br_f) / fps
        rho_ft = math.exp(-dt_ft / 0.08)
        var_ft = float(
            g_pair[0] @ c_br @ g_pair[0]
            + g_pair[1] @ c_ft @ g_pair[1]
            + 2.0 * rho_ft * (g_pair[0] @ c_12 @ g_pair[1])
        )
        return math.sqrt(max(var_ft, 0.0))

    return 5.0


def _temporal_sigma_from_events(key: str, kin: KinematicsResult, events: DeliveryEvents) -> float:
    """Propagate actual Stage 2B per-event `timing_uncertainty_ms` via finite difference on Stage 3 series."""
    fps = max(kin.fps, 1.0)
    n = kin.joints_world_smoothed.shape[0] if kin.joints_world_smoothed is not None else 10
    s_bfc = events.bfc.timing_uncertainty_ms / 1000.0
    s_ffc = events.ffc.timing_uncertainty_ms / 1000.0
    s_br = events.br.timing_uncertainty_ms / 1000.0

    def _series_sens(s_name: str, f_fl: float, dt_s: float) -> float:
        arr = kin.series.get(s_name)
        if arr is None or len(arr) < 3:
            return 0.0
        df = max(dt_s * fps, 0.15)
        v_0 = float(_interp_frame(arr, f_fl))
        v_plus = float(_interp_frame(arr, f_fl + df))
        v_minus = float(_interp_frame(arr, f_fl - df))
        # RMS of left/right one-sided shifts preserves true sensitivity across contact-event kinks (BFC/FFC)
        return math.sqrt(0.5 * ((v_plus - v_0) ** 2 + (v_0 - v_minus) ** 2))

    if key == "front_knee_ffc_deg":
        return _series_sens("front_knee_deg", events.ffc.frame_float, s_ffc)
    if key == "front_knee_br_deg":
        return _series_sens("front_knee_deg", events.br.frame_float, s_br)
    if key == "front_knee_change_deg":
        return math.hypot(
            _series_sens("front_knee_deg", events.ffc.frame_float, s_ffc),
            _series_sens("front_knee_deg", events.br.frame_float, s_br),
        )
    if key == "trunk_flexion_ffc_deg":
        return _series_sens("trunk_flex_deg", events.ffc.frame_float, s_ffc)
    if key == "trunk_flexion_br_deg":
        return _series_sens("trunk_flex_deg", events.br.frame_float, s_br)
    if key == "trunk_flexion_delta_deg":
        return math.hypot(
            _series_sens("trunk_flex_deg", events.ffc.frame_float, s_ffc),
            _series_sens("trunk_flex_deg", events.br.frame_float, s_br),
        )
    if key == "thorax_lat_flex_ffc_deg":
        return _series_sens("thorax_lat_flex_deg", events.ffc.frame_float, s_ffc)
    if key == "thorax_lat_flex_br_deg":
        return _series_sens("thorax_lat_flex_deg", events.br.frame_float, s_br)
    if key == "pelvis_alignment_bfc_deg":
        return _series_sens("pelvis_yaw_deg", events.bfc.frame_float, s_bfc)
    if key == "shoulder_alignment_bfc_deg":
        return _series_sens("shoulder_yaw_deg", events.bfc.frame_float, s_bfc)
    if key == "shoulder_counter_rotation_deg":
        return _series_sens("shoulder_yaw_deg", events.bfc.frame_float, s_bfc)
    if key == "hip_shoulder_sep_bfc_deg":
        return _series_sens("hip_shoulder_sep_deg", events.bfc.frame_float, s_bfc)
    if key == "max_hip_shoulder_sep_deg":
        return 0.5 * _series_sens("hip_shoulder_sep_deg", events.bfc.frame_float, s_bfc)
    if key == "delayed_arm_ffc_deg":
        return _series_sens("bowling_arm_phi_deg", events.ffc.frame_float, s_ffc)
    if key == "rear_hip_bfc_deg":
        return _series_sens("rear_hip_deg", events.bfc.frame_float, s_bfc)
    if key == "pelvic_drop_ffc_deg":
        return _series_sens("pelvic_drop_deg", events.ffc.frame_float, s_ffc)
    if key == "non_bowling_arm_elev_ffc_deg":
        return _series_sens("nb_arm_elev_deg", events.ffc.frame_float, s_ffc)
    if key == "stride_length_pct_stature" and kin.joints_world_smoothed is not None:
        # One-sided RMS horizontal ankle sensitivity at BFC & FFC contact kinks (fixes S1's stride under-coverage!)
        is_right = kin.bowling_arm == BowlingArm.RIGHT
        f_ank_x = kin.joints_world_smoothed[:, JOINT_IDX["lank" if is_right else "rank"], 0]
        r_ank_x = kin.joints_world_smoothed[:, JOINT_IDX["rank" if is_right else "lank"], 0]
        df_b = max(s_bfc * fps, 0.2)
        df_f = max(s_ffc * fps, 0.2)
        r0 = float(_interp_frame(r_ank_x, events.bfc.frame_float))
        f0 = float(_interp_frame(f_ank_x, events.ffc.frame_float))
        d_r = math.sqrt(
            0.5 * (
                (float(_interp_frame(r_ank_x, events.bfc.frame_float + df_b)) - r0) ** 2
                + (r0 - float(_interp_frame(r_ank_x, events.bfc.frame_float - df_b))) ** 2
            )
        )
        d_f = math.sqrt(
            0.5 * (
                (float(_interp_frame(f_ank_x, events.ffc.frame_float + df_f)) - f0) ** 2
                + (f0 - float(_interp_frame(f_ank_x, events.ffc.frame_float - df_f))) ** 2
            )
        )
        return 100.0 * math.hypot(d_r, d_f) / max(kin.estimated_stature_m, 1.0)
    if key == "release_height_m" and kin.joints_world_smoothed is not None:
        is_right = kin.bowling_arm == BowlingArm.RIGHT
        wri_y = kin.joints_world_smoothed[:, JOINT_IDX["rwri" if is_right else "lwri"], 1]
        df_br = max(s_br * fps, 0.2)
        return 0.5 * abs(float(_interp_frame(wri_y, events.br.frame_float + df_br)) - float(_interp_frame(wri_y, events.br.frame_float - df_br)))

    # Fallback using local_rates * anchor timing_uncertainty_ms
    anchors = METRIC_EVENT_ANCHORS.get(key, ("br",))
    sig_t_s = max({"bfc": s_bfc, "ffc": s_ffc, "br": s_br}[a] for a in anchors)
    return abs(kin.local_rates_per_sec.get(key, 0.0)) * sig_t_s * 0.35


def _calibration_sigma_fd(kin: KinematicsResult, cal: PitchCalibration) -> Dict[str, float]:
    """Propagate calibration heading (sigma_psi) and gravity (sigma_g) uncertainties via world-frame rotation FD."""
    j = kin.joints_world_smoothed
    ev = kin.events
    if j is None or ev is None:
        return {}

    sig_psi = max(float(cal.heading_sigma_deg), 0.0)
    sig_g = max(float(cal.gravity_sigma_deg), 0.0)
    if sig_psi < 1e-3 and sig_g < 1e-3:
        return {}

    eps_deg = 2.0
    eps = math.radians(eps_deg)
    c, s = math.cos(eps), math.sin(eps)

    def _rot_y(sgn: float) -> np.ndarray:
        return np.array([[c, 0.0, sgn * s], [0.0, 1.0, 0.0], [-sgn * s, 0.0, c]], dtype=np.float64)

    def _rot_x(sgn: float) -> np.ndarray:
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -sgn * s], [0.0, sgn * s, c]], dtype=np.float64)

    def _rot_z(sgn: float) -> np.ndarray:
        return np.array([[c, -sgn * s, 0.0], [sgn * s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    # Rotate around pelvis at FFC so lever offsets reflect physical local coordinates
    pivot = j[int(np.clip(round(ev.ffc.frame_float), 0, j.shape[0] - 1)), JOINT_IDX["pelv"]]

    def _eval_rot(R_d: np.ndarray) -> Dict[str, float]:
        j_rot = (j - pivot) @ R_d.T + pivot
        k_rot = extract_fast_bowling_kinematics(j_rot, ev, fps=kin.fps, bowling_arm=kin.bowling_arm)
        return k_rot.scalars

    s_yp, s_ym = _eval_rot(_rot_y(1.0)), _eval_rot(_rot_y(-1.0))
    s_xp, s_xm = _eval_rot(_rot_x(1.0)), _eval_rot(_rot_x(-1.0))
    s_zp, s_zm = _eval_rot(_rot_z(1.0)), _eval_rot(_rot_z(-1.0))

    out: Dict[str, float] = {}
    for k in METRIC_SPECS:
        if k not in s_yp:
            continue
        d_psi = (s_yp[k] - s_ym[k]) / (2.0 * eps_deg)
        d_gx = (s_xp[k] - s_xm[k]) / (2.0 * eps_deg)
        d_gz = (s_zp[k] - s_zm[k]) / (2.0 * eps_deg)
        out[k] = math.sqrt((d_psi * sig_psi) ** 2 + (d_gx * sig_g) ** 2 + (d_gz * sig_g) ** 2)
    return out


def _clamp_ci(key: str, ci_low: float, ci_high: float) -> Tuple[float, float]:
    bounds = PHYSICAL_CI_BOUNDS.get(key)
    if bounds is None:
        return ci_low, ci_high
    lo_b, hi_b = bounds
    if lo_b is not None:
        ci_low = max(lo_b, ci_low)
        ci_high = max(lo_b, ci_high)
    if hi_b is not None:
        ci_low = min(hi_b, ci_low)
        ci_high = min(hi_b, ci_high)
    return ci_low, ci_high


def quantify_and_gate_metrics(
    kin: KinematicsResult,
    audit: VideoAudit,
    cal: PitchCalibration,
    events: Optional[DeliveryEvents] = None,
    unc_scale_factor: float = 1.0,
    interval_type: IntervalType = "analytical_uncertainty",
    dual_baseline_sin: float = 1.0,
    empirical_sigmas: Optional[Dict[str, float]] = None,
    empirical_quantiles: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, MetricEstimate]:
    """Apply 4-source Jacobian / empirical repeatability uncertainty propagation and strict scientific gating."""
    ev = events or kin.events
    is_dual = cal.view_class == CameraViewClass.DUAL_VIEW_FUSED
    is_dual_orth = is_dual and (dual_baseline_sin >= 0.50)
    R_w2c = np.asarray(cal.rotation_matrix_R, dtype=np.float64)
    blur_mult = {"crisp": 1.0, "moderate_blur": 1.18, "severe_blur": 1.50}.get(audit.blur_grade, 1.15)

    az = abs(cal.azimuth_deg) % 360.0
    if az > 180.0:
        az = 360.0 - az
    sin_side = abs(math.sin(math.radians(az)))

    cal_sigmas = _calibration_sigma_fd(kin, cal)
    estimates: Dict[str, MetricEstimate] = {}

    for key, (label, unit, plane_type, sup_thresh, def_note, ref_text, source) in METRIC_SPECS.items():
        raw_val = kin.scalars.get(key)
        if raw_val is None or not math.isfinite(raw_val):
            estimates[key] = MetricEstimate(
                key=key,
                label=label,
                unit=unit,
                interval_type=interval_type,
                observability=ObservabilityGrade.UNMEASURABLE,
                suppressed=True,
                suppression_reason="Required anatomical landmarks were missing or non-finite.",
                definition_note=def_note,
                empirical_reference=ref_text,
                literature_source=source,
            )
            continue

        j_in_frame = kin.metric_joints_in_frame.get(key, True)
        j_uncs = kin.metric_joint_uncertainties_m.get(key, (0.04,))
        max_j_unc = max(j_uncs) if j_uncs else 0.04

        view_grade = _view_grade_from_geometry(
            plane_type=plane_type,
            az_deg=cal.azimuth_deg,
            el_deg=cal.elevation_deg,
            is_dual_fused=is_dual,
            is_metric_cal=cal.is_metric_calibrated,
            dual_baseline_sin=dual_baseline_sin,
        )

        # 4-source analytical Jacobian uncertainty (zero arbitrary lookup constants!)
        # Split pose variance orthogonally into isotropic in-plane epistemic (kappa_z=1) and
        # out-of-plane projective depth excess so sigma_epistemic^2 + sigma_projective^2 == sig_pose^2 (P1-4).
        sig_pose = _pose_sigma_jacobian(key, kin, R_w2c, unc_scale_factor, blur_mult, is_dual_orth)
        sig_epi = _pose_sigma_jacobian(key, kin, R_w2c, unc_scale_factor, blur_mult, True)
        sig_proj = math.sqrt(max(sig_pose**2 - sig_epi**2, 0.0))
        sig_temp = _temporal_sigma_from_events(key, kin, ev) if ev is not None else 0.0
        sig_cal = float(cal_sigmas.get(key, 0.0))
        sig_scale = (0.04 * abs(raw_val)) if (key == "release_height_m" and not cal.is_metric_calibrated) else 0.0

        # If empirical GPU repeatability ensemble was provided, combine with projective/calibration/temporal floor
        # so shared monocular depth bias across ensemble runs cannot under-state known systematic floors (P1-4).
        rep_floor = math.sqrt((0.5 * sig_proj) ** 2 + sig_cal**2 + (0.5 * sig_temp) ** 2 + (0.22 * sig_epi) ** 2)
        if empirical_sigmas is not None and key in empirical_sigmas:
            sig_emp = float(empirical_sigmas[key])
            sig_total = math.hypot(sig_emp, rep_floor)
        else:
            sig_total = math.sqrt(sig_pose**2 + sig_temp**2 + sig_cal**2 + sig_scale**2)

        obs = view_grade
        suppressed = False
        reason = None

        # Anchor-event confidence check (EXP-007 B7)
        min_ev_conf = 1.0
        if ev is not None:
            ev_map = {"bfc": ev.bfc, "ffc": ev.ffc, "br": ev.br}
            min_ev_conf = min(ev_map[a].confidence for a in METRIC_EVENT_ANCHORS.get(key, ("br",)))

        # Gate 1: Incomplete delivery lacking valid overhead Ball Release (EXP-007 B1)
        if ev is not None and (not ev.is_complete_delivery):
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                "Suppressed: clip did not contain a verified high-speed overhead ball release "
                "(truncated, gather-only, or frozen clip)."
            )
        # Gate 1B: Unreliable / edited slow-mo / freeze-frame timebase on velocity/time metrics (EXP-007 B2)
        elif audit.timebase_unreliable and key in TIMEBASE_DEPENDENT_METRICS:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: video timebase is unreliable ({audit.timebase_status}; "
                f"{audit.held_frame_ratio:.0%} repeated frames, max run={audit.longest_held_run_frames}); "
                "velocity/timing metrics require verified frame rate."
            )
        # Gate 1C: Boundary-clamped or very-low-confidence anchor event (EXP-007 B7)
        elif min_ev_conf < 0.30:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: anchor event detection confidence too low ({min_ev_conf:.2f}; "
                "event hit window boundary or required foot was cropped)."
            )
        # Gate 2: Required anatomical landmarks cropped out of frame or severely occluded
        elif not j_in_frame:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: required anatomical landmarks were cropped outside the camera frame at the event instant."
        elif max_j_unc > 0.165:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = f"Suppressed: high 3D joint localization uncertainty ({max_j_unc:.2f} m) due to occlusion or motion blur."
        # Gate 3: ALL world-translation & uncalibrated scale metrics (EXP-007 B5)
        elif key in TRANSLATION_DEPENDENT_METRICS and (cal.is_panning_camera or not cal.is_metric_calibrated):
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                "Suppressed: world-grounded translation requires a static camera with calibrated MCC pitch/wicket "
                "geometry; camera pan or uncalibrated FOV distorts depth displacement."
            )
        elif key == "follow_through_drift_deg" and kin.ft_disp_x_m < 0.30:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: post-release forward pelvis travel ({kin.ft_disp_x_m:.2f} m) is too small "
                "to form a stable lateral-drift ratio."
            )
        elif key == "release_height_m" and (not cal.is_intrinsics_calibrated) and cal.gravity_sigma_deg > 5.5:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                "Suppressed: uncalibrated intrinsics and torso-only gravity tilt (>5.5 deg) prevent reliable "
                "metric release-height estimation."
            )
        # Gate 4: High-velocity arm metrics require >=90 fps
        elif plane_type == "velocity_high" and audit.fps < 90.0:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: effective rate is {audit.fps:.1f} fps (~{audit.est_arm_deg_per_frame:.0f} deg/frame); "
                "high-velocity arm kinematics require >=100 fps (ideally 120-240 fps)."
            )
        # Gate 5: View observability & total uncertainty threshold (EXP-007 B17 FIXED branch order!)
        elif view_grade == ObservabilityGrade.UNMEASURABLE:
            suppressed = True
            needed = (
                "side-on (~90 deg)" if plane_type in ("sagittal", "velocity_high", "metric_scale")
                else ("behind-the-bowler (~0 deg)" if plane_type == "coronal" else "45 deg open-side or dual-phone")
            )
            plane_label = PLANE_HUMAN_NAMES.get(plane_type, plane_type)
            reason = (
                f"Suppressed: camera azimuth ({cal.azimuth_deg:.0f} deg, view={cal.view_class.value}) "
                f"cannot reliably resolve {plane_label} kinematics; film from a {needed} angle."
            )
        elif sig_total > sup_thresh:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: total 1-sigma uncertainty (+-{sig_total:.1f} {unit}) exceeds reliability "
                f"limit ({sup_thresh:.1f} {unit}) due to view/FPS/calibration/blur constraints."
            )
        elif sig_total > 0.82 * sup_thresh:
            obs = ObservabilityGrade.LOW
        elif sig_total > 0.65 * sup_thresh and obs == ObservabilityGrade.HIGH:
            obs = ObservabilityGrade.MODERATE

        if empirical_quantiles is not None and key in empirical_quantiles:
            q_lo, q_hi = empirical_quantiles[key]
            # Floor repeatability interval against projective/calibration/temporal systematic bounds
            floor_h = 1.96 * rep_floor
            ci_lo = min(q_lo, raw_val - floor_h)
            ci_hi = max(q_hi, raw_val + floor_h)
        else:
            half_ci = 1.96 * sig_total
            ci_lo, ci_hi = float(raw_val - half_ci), float(raw_val + half_ci)

        ci_lo, ci_hi = _clamp_ci(key, ci_lo, ci_hi)

        estimates[key] = MetricEstimate(
            key=key,
            label=label,
            unit=unit,
            value=None if suppressed else round(float(raw_val), 2),
            ci_low=None if suppressed else round(ci_lo, 2),
            ci_high=None if suppressed else round(ci_hi, 2),
            interval_type=interval_type,
            sigma_total=round(sig_total, 2),
            sigma_projective=round(sig_proj, 2),
            sigma_temporal=round(sig_temp, 2),
            sigma_epistemic=round(sig_epi, 2),
            sigma_calibration=round(sig_cal, 2),
            observability=obs,
            suppressed=suppressed,
            suppression_reason=reason,
            definition_note=def_note,
            empirical_reference=ref_text,
            literature_source=source,
        )

    # Permanent hard refusal: ICC 15-degree elbow legality
    estimates["icc_elbow_legality_verdict"] = MetricEstimate(
        key="icc_elbow_legality_verdict",
        label="ICC 15-Degree Elbow Extension Legality",
        unit="deg",
        value=None,
        ci_low=None,
        ci_high=None,
        interval_type=interval_type,
        observability=ObservabilityGrade.REFUSED,
        suppressed=True,
        suppression_reason=(
            "PERMANENTLY REFUSED ON VIDEO: Accredited ICC bowling legality adjudication requires "
            ">=250 Hz multi-camera 3D optical motion capture with anatomical upper-arm/forearm marker "
            "clusters. Abraham, Feros, & Fox (2025, IJSSC) reported 22.71 deg elbow RMSE even on "
            "three calibrated 240 Hz iPads."
        ),
        definition_note="Permanent scientific refusal per ICC testing regulations.",
        empirical_reference="ICC limit: <=15 deg elbow extension from Upper Arm Horizontal (UAH) to Ball Release (BR).",
        literature_source="ICC Illegal Bowling Action Regulations; Abraham, Feros, & Fox (2025), IJSSC 20(6)",
    )
    return estimates


def compute_repeatability_metrics(
    base_kin: KinematicsResult,
    ensemble_world_trajs: List[np.ndarray],
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    n_mc: int = 64,
    seed: int = 2026,
) -> Dict[str, MetricEstimate]:
    """Compute honest empirical Repeatability Intervals via GPU pose ensemble + calibration/event Monte Carlo."""
    rng = np.random.default_rng(seed)
    n_vars = max(len(ensemble_world_trajs), 1)
    samples: Dict[str, List[float]] = {k: [] for k in METRIC_SPECS}

    sig_psi = math.radians(max(cal.heading_sigma_deg, 0.5))
    sig_g = math.radians(max(cal.gravity_sigma_deg, 0.5))
    fps = max(audit.fps, 1.0)
    s_bfc_f = (events.bfc.timing_uncertainty_ms / 1000.0) * fps
    s_ffc_f = (events.ffc.timing_uncertainty_ms / 1000.0) * fps
    s_br_f = (events.br.timing_uncertainty_ms / 1000.0) * fps

    n_frames = ensemble_world_trajs[0].shape[0]
    pivot_i = int(np.clip(round(events.ffc.frame_float), 0, n_frames - 1))

    for m_i in range(n_mc):
        j_w = ensemble_world_trajs[m_i % n_vars]
        # Sample world-frame heading & gravity tilt perturbation
        d_psi = float(rng.normal(0.0, sig_psi))
        d_gx = float(rng.normal(0.0, sig_g))
        d_gz = float(rng.normal(0.0, sig_g))

        cy, sy = math.cos(d_psi), math.sin(d_psi)
        cx, sx = math.cos(d_gx), math.sin(d_gx)
        cz, sz = math.cos(d_gz), math.sin(d_gz)
        R_d = (
            np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
            @ np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
            @ np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
        )
        piv = j_w[pivot_i, JOINT_IDX["pelv"]]
        j_pert = (j_w - piv) @ R_d.T + piv

        # Sample causally ordered event timing jitter
        bfc_j = float(np.clip(events.bfc.frame_float + rng.normal(0.0, s_bfc_f), 0.0, n_frames - 3.0))
        ffc_j = float(np.clip(events.ffc.frame_float + rng.normal(0.0, s_ffc_f), bfc_j + 0.5, n_frames - 2.0))
        br_j = float(np.clip(events.br.frame_float + rng.normal(0.0, s_br_f), ffc_j + 0.5, n_frames - 1.0))

        ev_j = events.model_copy(
            update={
                "bfc": events.bfc.model_copy(update={"frame_float": bfc_j, "frame_idx": int(round(bfc_j))}),
                "ffc": events.ffc.model_copy(update={"frame_float": ffc_j, "frame_idx": int(round(ffc_j))}),
                "br": events.br.model_copy(update={"frame_float": br_j, "frame_idx": int(round(br_j))}),
            }
        )
        k_j = extract_fast_bowling_kinematics(j_pert, ev_j, fps=fps, bowling_arm=base_kin.bowling_arm)
        for k in METRIC_SPECS:
            val = k_j.scalars.get(k)
            if val is not None and math.isfinite(val):
                samples[k].append(float(val))

    emp_sigmas: Dict[str, float] = {}
    emp_quants: Dict[str, Tuple[float, float]] = {}
    for k, vals in samples.items():
        if len(vals) >= 10:
            arr = np.asarray(vals, dtype=np.float64)
            emp_sigmas[k] = float(np.std(arr, ddof=1))
            emp_quants[k] = (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return quantify_and_gate_metrics(
        kin=base_kin,
        audit=audit,
        cal=cal,
        events=events,
        unc_scale_factor=0.35,  # calibrated for raw NLF gating signal on real clips (EXP-007 3.4)
        interval_type="repeatability_interval",
        empirical_sigmas=emp_sigmas,
        empirical_quantiles=emp_quants,
    )
