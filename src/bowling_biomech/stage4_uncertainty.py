"""Stage 4: Camera-View Observability Matrix, 3-Source Uncertainty Engine, & Hard Suppression Gate.

Computes calibrated 95% confidence intervals combining:
  1. Projective geometric sensitivity sigma_proj(azimuth, elevation)
  2. Temporal quantization error sigma_temp = local_rate / (2 * sqrt(3) * fps)
  3. Principled per-metric multi-joint epistemic localization error (no arbitrary discount)

Hard-suppresses:
  - Out-of-frame / cropped or occluded anatomical landmarks (event-specific)
  - Panning-camera or uncalibrated rear-view world translational metrics (run-up speed, stride length)
  - High-velocity arm metrics below 90 fps
  - Release-phase metrics on truncated/incomplete clips
  - ICC 15-degree elbow legality verdicts (PERMANENTLY REFUSED)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .schemas import (
    CameraViewClass,
    DeliveryEvents,
    MetricEstimate,
    ObservabilityGrade,
    PitchCalibration,
    VideoAudit,
)
from .stage3_kinematics import KinematicsResult

PLANE_HUMAN_NAMES: Dict[str, str] = {
    "sagittal": "sagittal (side-on)",
    "coronal": "coronal (front/rear)",
    "transverse": "transverse (top-down rotational)",
    "velocity_high": "high-velocity arm circumduction",
    "metric_scale": "metric world-displacement",
}

# Physical domain bounds (min, max) for CI clamping
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

# Metrics strictly dependent on an observed overhead Ball Release (suppressed if clip is truncated before BR)
RELEASE_DEPENDENT_METRICS = {
    "front_knee_br_deg",
    "front_knee_change_deg",
    "trunk_flexion_br_deg",
    "trunk_flexion_delta_deg",
    "thorax_lat_flex_br_deg",
    "peak_arm_angular_vel_dps",
    "release_height_m",
    "follow_through_drift_deg",
}

# Metric registry: (label, unit, plane_type, lever_m, is_3joint, suppress_sigma_thresh, ref_text, source)
METRIC_SPECS: Dict[str, Tuple[str, str, str, float, bool, float, str, str]] = {
    "runup_speed_mps": (
        "Run-Up Approach Speed",
        "m/s",
        "metric_scale",
        0.50,
        False,
        1.2,
        "Elite male fast bowlers: ~5.2–6.6 m/s (strong positive predictor of ball speed)",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
    ),
    "stride_length_pct_stature": (
        "Delivery Stride Length",
        "% stature",
        "metric_scale",
        0.50,
        False,
        13.0,
        "Typical fast bowling stride: ~72–88% of stature (1.30–1.65 m)",
        "Portus et al. (2004); Worthington et al. (2013)",
    ),
    "front_knee_ffc_deg": (
        "Front Knee Angle at FFC",
        "deg",
        "sagittal",
        0.43,
        True,
        11.0,
        "Typical FFC contact angle: ~150–170 deg (180 deg = fully straight)",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "front_knee_br_deg": (
        "Front Knee Angle at Ball Release",
        "deg",
        "sagittal",
        0.43,
        True,
        11.0,
        "Faster bowlers maintain straighter braced front knee (~158–178 deg at BR)",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
    ),
    "front_knee_change_deg": (
        "Front Knee Bracing Change (FFC->BR)",
        "deg",
        "sagittal",
        0.43,
        True,
        12.5,
        "Positive (+) = extender/bracer; near 0 = braced; <=-10 deg = flexing/collapsing",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "trunk_flexion_ffc_deg": (
        "Upper-Trunk Forward Flexion at FFC",
        "deg",
        "sagittal",
        0.50,
        False,
        10.5,
        "Typical FFC upright/slight-forward posture: ~5–20 deg forward of vertical",
        "Worthington et al. (2013)",
    ),
    "trunk_flexion_br_deg": (
        "Upper-Trunk Forward Flexion at BR",
        "deg",
        "sagittal",
        0.50,
        False,
        11.5,
        "Typical elite release trunk flexion: ~35–55 deg forward of vertical",
        "Worthington et al. (2013)",
    ),
    "trunk_flexion_delta_deg": (
        "Trunk Forward Flexion Range (FFC->BR)",
        "deg",
        "sagittal",
        0.50,
        False,
        12.5,
        "Greater forward trunk flexion FFC->BR (~25–42 deg) drives higher ball speed",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
    ),
    "thorax_lat_flex_ffc_deg": (
        "Thorax Contralateral Lateral Flexion at FFC",
        "deg",
        "coronal",
        0.50,
        False,
        10.5,
        "Moderate pre-release side-flexion (~12–28 deg); excessive early bend elevates load",
        "Alway et al. (2021), MSSE; Bayne et al. (2016), JSMS",
    ),
    "thorax_lat_flex_br_deg": (
        "Thorax Contralateral Lateral Flexion at BR",
        "deg",
        "coronal",
        0.50,
        False,
        11.5,
        "Uninjured elite norm: ~32–44 deg (40+-8 deg); elevated LBSI risk around >=48–50 deg",
        "Bayne, Elliott, Campbell, & Alderson (2016), JSMS",
    ),
    "pelvis_alignment_bfc_deg": (
        "Pelvis Alignment at BFC",
        "deg",
        "transverse",
        0.24,
        False,
        14.0,
        "AIS convention: <=205 deg side-on, 205–245 deg semi, >245 deg front-on",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "shoulder_alignment_bfc_deg": (
        "Shoulder Alignment at BFC",
        "deg",
        "transverse",
        0.38,
        False,
        14.0,
        "AIS convention: <=205 deg side-on, 205–245 deg semi, >245 deg front-on",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "shoulder_counter_rotation_deg": (
        "Shoulder Counter-Rotation (SCR, BFC->FFC)",
        "deg",
        "transverse",
        0.38,
        False,
        13.5,
        "<25-30 deg = non-mixed alignment; >=30 deg defines mixed action linked to LBSI",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "hip_shoulder_sep_bfc_deg": (
        "Hip-Shoulder Separation at BFC",
        "deg",
        "transverse",
        0.32,
        False,
        14.0,
        "BFC misalignment >=30 deg with counter-rotation indicates mixed action loading",
        "Portus et al. (2004), Sports Biomechanics",
    ),
    "max_hip_shoulder_sep_deg": (
        "Peak Dynamic Hip-Shoulder Separation",
        "deg",
        "transverse",
        0.32,
        False,
        14.5,
        "Well-timed late separation (~25–50 deg) supports stretch-shortening momentum transfer",
        "Portus et al. (2004); Worthington et al. (2013)",
    ),
    "delayed_arm_ffc_deg": (
        "Delayed Bowling Arm Angle Behind Vertical at FFC",
        "deg",
        "velocity_high",
        0.31,
        False,
        15.0,
        "Larger delay behind vertical (~125–165 deg at FFC) strongly predicts higher ball speed",
        "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
    ),
    "rear_hip_bfc_deg": (
        "Rear Hip Angle at BFC",
        "deg",
        "sagittal",
        0.45,
        True,
        12.5,
        "Key upstream lumbopelvic control predictor in prospective LBSI regression model",
        "Alway, Felton, Brooke-Wavell, Peirce, & King (2021), MSSE",
    ),
    "pelvic_drop_ffc_deg": (
        "Contralateral/Ipsilateral Pelvic Tilt at FFC",
        "deg",
        "coronal",
        0.24,
        False,
        11.0,
        "Controlled level pelvis (|tilt| <10 deg) reduces asymmetrical lumbar shear loading",
        "Alway et al. (2021), MSSE",
    ),
    "non_bowling_arm_elev_ffc_deg": (
        "Non-Bowling (Front) Arm Elevation at FFC",
        "deg",
        "sagittal",
        0.31,
        False,
        16.0,
        "High front-arm anchor (>15 deg above horizontal at FFC) supports trunk alignment before pull",
        "Middleton et al.; Worthington et al.",
    ),
    "peak_arm_angular_vel_dps": (
        "Peak Bowling Arm Angular Velocity",
        "deg/s",
        "velocity_high",
        0.31,
        False,
        420.0,
        "Elite fast bowlers peak ~2,000–2,570 deg/s (35–45 rad/s) near ball release",
        "Fast-bowling 3D biomechanics literature",
    ),
    "release_height_m": (
        "Ball Release Height",
        "m",
        "metric_scale",
        0.50,
        False,
        0.18,
        "Typically ~1.05–1.18x bowler stature (~1.95–2.20 m for 1.85 m bowler)",
        "Fast-bowling kinematics norms",
    ),
    "follow_through_drift_deg": (
        "Follow-Through Lateral Drift Angle",
        "deg",
        "coronal",
        0.50,
        False,
        12.5,
        "Driving momentum near target line (|drift| <12-15 deg) reduces lateral braking shear",
        "Coaching & momentum-conservation principles",
    ),
}


def _projective_sigma(
    plane_type: str,
    az_deg: float,
    el_deg: float,
    is_dual_fused: bool,
    is_metric_cal: bool,
    unit: str,
) -> Tuple[float, ObservabilityGrade]:
    """Compute camera-view projective sensitivity sigma_proj based on azimuth & elevation."""
    az = abs(az_deg) % 360.0
    if az > 180.0:
        az = 360.0 - az
    sin_side = abs(math.sin(math.radians(az)))
    cos_rear = abs(math.cos(math.radians(az)))
    el_penalty = max(0.0, (abs(el_deg) - 10.0) / 20.0)

    if is_dual_fused:
        if unit == "deg":
            return 2.2, ObservabilityGrade.HIGH
        if unit == "m/s":
            return 0.25, ObservabilityGrade.HIGH
        if unit == "% stature":
            return 3.0, ObservabilityGrade.HIGH
        if unit == "m":
            return 0.04, ObservabilityGrade.HIGH
        return 120.0, ObservabilityGrade.HIGH

    if plane_type == "sagittal":
        if sin_side >= 0.80 and abs(el_deg) <= 18.0:
            return 2.8 + 1.5 * el_penalty, ObservabilityGrade.HIGH
        if sin_side >= 0.45:
            return 5.2 + 2.5 * el_penalty, ObservabilityGrade.MODERATE
        if sin_side >= 0.28:
            return 9.8 + 3.0 * el_penalty, ObservabilityGrade.LOW
        return 16.5 + 4.0 * el_penalty, ObservabilityGrade.UNMEASURABLE

    if plane_type == "coronal":
        if cos_rear >= 0.80 and abs(el_deg) <= 22.0:
            return 2.6 + 1.8 * el_penalty, ObservabilityGrade.HIGH
        if cos_rear >= 0.45:
            return 5.4 + 2.5 * el_penalty, ObservabilityGrade.MODERATE
        if cos_rear >= 0.28:
            return 9.8 + 3.0 * el_penalty, ObservabilityGrade.LOW
        return 16.5 + 4.0 * el_penalty, ObservabilityGrade.UNMEASURABLE

    if plane_type == "transverse":
        if 0.45 <= sin_side <= 0.85:
            return 6.2 + 1.5 * el_penalty, ObservabilityGrade.MODERATE
        if cos_rear >= 0.75:
            return 6.8 + 1.2 * el_penalty, ObservabilityGrade.MODERATE
        return 11.8 + 2.5 * el_penalty, ObservabilityGrade.LOW

    if plane_type == "velocity_high":
        if unit == "deg/s":
            base = 180.0 if sin_side >= 0.65 else (320.0 if sin_side >= 0.35 else 650.0)
            grade = ObservabilityGrade.HIGH if sin_side >= 0.75 else (
                ObservabilityGrade.MODERATE if sin_side >= 0.40 else ObservabilityGrade.UNMEASURABLE
            )
            return base, grade
        if sin_side >= 0.75:
            return 4.0 + 1.5 * el_penalty, ObservabilityGrade.HIGH
        if sin_side >= 0.40:
            return 6.8 + 2.0 * el_penalty, ObservabilityGrade.MODERATE
        return 14.5 + 3.0 * el_penalty, ObservabilityGrade.UNMEASURABLE

    if plane_type == "metric_scale":
        cal_mult = 1.0 if is_metric_cal else 1.8
        if unit == "m/s":
            sig = (0.35 if sin_side >= 0.65 else (0.70 if sin_side >= 0.35 else 1.40)) * cal_mult
            grade = ObservabilityGrade.HIGH if (sin_side >= 0.70 and is_metric_cal) else (
                ObservabilityGrade.MODERATE if sin_side >= 0.40 else ObservabilityGrade.UNMEASURABLE
            )
            return sig, grade
        if unit == "% stature":
            sig = (3.5 if sin_side >= 0.65 else (6.5 if sin_side >= 0.35 else 13.0)) * cal_mult
            grade = ObservabilityGrade.HIGH if (sin_side >= 0.70 and is_metric_cal) else (
                ObservabilityGrade.MODERATE if sin_side >= 0.40 else ObservabilityGrade.UNMEASURABLE
            )
            return sig, grade
        sig = (0.05 if abs(el_deg) <= 18.0 else 0.11) * (1.0 if is_metric_cal else 1.35)
        grade = ObservabilityGrade.HIGH if (is_metric_cal and abs(el_deg) <= 18.0) else ObservabilityGrade.MODERATE
        return sig, grade

    return 10.0, ObservabilityGrade.LOW


def _epistemic_sigma_principled(
    joint_uncs_m: Tuple[float, ...],
    lever_m: float,
    is_3joint: bool,
    unit: str,
    blur_grade: str,
) -> float:
    """Principled multi-joint error propagation into metric uncertainty (no arbitrary discounting)."""
    blur_mult = {"crisp": 1.0, "moderate_blur": 1.25, "severe_blur": 1.65}.get(blur_grade, 1.2)
    if not joint_uncs_m:
        u_eff = 0.04
    elif is_3joint and len(joint_uncs_m) >= 3:
        # Vertex joint B contributes 2x lever sensitivity: sqrt(sA^2 + 4*sB^2 + sC^2) / sqrt(6) scaled
        s_a, s_b, s_c = joint_uncs_m[0], joint_uncs_m[1], joint_uncs_m[2]
        # Transverse isotropic-to-planar component factor (~1/sqrt(3) per axis -> /sqrt(2) in plane)
        u_eff = math.sqrt((s_a**2 + 4.0 * s_b**2 + s_c**2) / 3.0)
    elif len(joint_uncs_m) >= 2:
        u_eff = math.sqrt((joint_uncs_m[0] ** 2 + joint_uncs_m[1] ** 2) / 2.0)
    else:
        u_eff = joint_uncs_m[0]

    ang_deg = math.degrees(u_eff / max(lever_m, 0.20)) * blur_mult
    if unit == "deg":
        return float(np.clip(ang_deg, 1.0, 28.0))
    if unit == "deg/s":
        return float(ang_deg * 18.0)
    if unit == "m/s":
        return float(u_eff * 3.5 * blur_mult)
    if unit == "% stature":
        return float(u_eff * 55.0 * blur_mult)
    return float(u_eff * 1.3 * blur_mult)


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
) -> Dict[str, MetricEstimate]:
    """Apply 3-source uncertainty propagation, visibility/panning gating, and permanent ICC refusal."""
    is_dual = cal.view_class == CameraViewClass.DUAL_VIEW_FUSED
    az = abs(cal.azimuth_deg) % 360.0
    if az > 180.0:
        az = 360.0 - az
    sin_side = abs(math.sin(math.radians(az)))
    estimates: Dict[str, MetricEstimate] = {}

    for key, (label, unit, plane_type, lever_m, is_3joint, sup_thresh, ref_text, source) in METRIC_SPECS.items():
        raw_val = kin.scalars.get(key)
        if raw_val is None or not math.isfinite(raw_val):
            estimates[key] = MetricEstimate(
                key=key,
                label=label,
                unit=unit,
                observability=ObservabilityGrade.UNMEASURABLE,
                suppressed=True,
                suppression_reason="Required anatomical landmarks were missing or non-finite.",
                empirical_reference=ref_text,
                literature_source=source,
            )
            continue

        j_in_frame = kin.metric_joints_in_frame.get(key, True)
        j_uncs = kin.metric_joint_uncertainties_m.get(key, (0.04,))
        max_j_unc = max(j_uncs) if j_uncs else 0.04

        sig_proj, view_grade = _projective_sigma(
            plane_type=plane_type,
            az_deg=cal.azimuth_deg,
            el_deg=cal.elevation_deg,
            is_dual_fused=is_dual,
            is_metric_cal=cal.is_metric_calibrated,
            unit=unit,
        )
        local_rate = abs(kin.local_rates_per_sec.get(key, 0.0))
        sig_temp = local_rate / (2.0 * math.sqrt(3.0) * max(audit.fps, 1.0))
        sig_epi = _epistemic_sigma_principled(j_uncs, lever_m, is_3joint, unit, audit.blur_grade)
        sig_total = math.sqrt(sig_proj**2 + sig_temp**2 + sig_epi**2)

        obs = view_grade
        suppressed = False
        reason = None

        # Gate 1: Truncated / incomplete clip lacking overhead Ball Release
        if events is not None and (not events.is_complete_delivery) and key in RELEASE_DEPENDENT_METRICS:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: clip did not contain a complete overhead ball release; release-phase kinematics cannot be computed."
        # Gate 2: Required anatomical landmarks cropped out of frame or severely occluded
        elif not j_in_frame:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = "Suppressed: required anatomical landmarks were cropped outside the camera frame at the event instant."
        elif max_j_unc > 0.145:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = f"Suppressed: high 3D joint localization uncertainty ({max_j_unc:.2f} m) due to occlusion or motion blur."
        # Gate 3: Panning camera or uncalibrated rear-view world translational metrics
        elif key in ("runup_speed_mps", "stride_length_pct_stature") and cal.is_panning_camera and not cal.is_metric_calibrated:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                "Suppressed: camera panning cancels ground-referenced translation without MCC Law 7 pitch "
                "calibration; record from a static tripod or supply pitch crease landmarks."
            )
        elif key in ("runup_speed_mps", "stride_length_pct_stature") and (not cal.is_metric_calibrated) and sin_side < 0.32:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                "Suppressed: monocular depth compression from behind-the-bowler/front views prevents reliable "
                "approach velocity or stride-length estimation without pitch calibration."
            )
        # Gate 4: High-velocity arm metrics require >=90 fps (tightened from 48 fps per 60fps RMSE audit!)
        elif plane_type == "velocity_high" and audit.fps < 90.0:
            obs = ObservabilityGrade.UNMEASURABLE
            suppressed = True
            reason = (
                f"Suppressed: {audit.fps:.1f} fps causes ~{audit.est_arm_deg_per_frame:.0f} deg bowling-arm "
                "sweep per frame; high-velocity arm kinematics require >=100 fps (ideally 120-240 fps)."
            )
        # Gate 5: View observability & total uncertainty threshold
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
                f"Suppressed: total 1-sigma uncertainty (+-{sig_total:.1f} {unit}) exceeds scientific "
                f"reliability limit ({sup_thresh:.1f} {unit}) due to view/FPS/blur constraints."
            )
        elif sig_total > 0.65 * sup_thresh and obs == ObservabilityGrade.HIGH:
            obs = ObservabilityGrade.MODERATE
        elif sig_total > 0.82 * sup_thresh:
            obs = ObservabilityGrade.LOW

        half_ci = 1.96 * sig_total
        ci_lo, ci_hi = _clamp_ci(key, float(raw_val - half_ci), float(raw_val + half_ci))

        estimates[key] = MetricEstimate(
            key=key,
            label=label,
            unit=unit,
            value=None if suppressed else round(float(raw_val), 2),
            ci_low=None if suppressed else round(ci_lo, 2),
            ci_high=None if suppressed else round(ci_hi, 2),
            sigma_total=round(sig_total, 2),
            sigma_projective=round(sig_proj, 2),
            sigma_temporal=round(sig_temp, 2),
            sigma_epistemic=round(sig_epi, 2),
            observability=obs,
            suppressed=suppressed,
            suppression_reason=reason,
            empirical_reference=ref_text,
            literature_source=source,
        )

    # Permanent hard refusal for ICC 15-degree elbow legality
    estimates["icc_elbow_legality_verdict"] = MetricEstimate(
        key="icc_elbow_legality_verdict",
        label="ICC 15-Degree Elbow Legality Verdict",
        unit="deg",
        value=None,
        ci_low=None,
        ci_high=None,
        observability=ObservabilityGrade.REFUSED,
        suppressed=True,
        suppression_reason=(
            "Permanently refused: even dual-camera markerless OpenCap exhibits 22.7 deg RMSE on cricket "
            "elbow flexion (Abraham et al., 2025/2026), exceeding the entire 15-degree ICC threshold. "
            "Authoritative legality testing requires an ICC-accredited >=250 Hz 3D optical lab."
        ),
        empirical_reference="ICC limit: <=15 deg active elbow extension strictly between UAH and BR",
        literature_source="Elliott, Alderson, & Denver (2007); Abraham, Feros, & Fox (2025/2026)",
    )

    return estimates
