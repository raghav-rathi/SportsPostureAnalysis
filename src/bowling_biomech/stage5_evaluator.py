"""Stage 5: Evidence-Grounded, Archetype-Conditioned Fast-Bowling Evaluator.

Replaces opaque single-letter S/A+/A/B/C/D tiers with a transparent multi-pillar model:
  - Pillar 1: Kinetic & Pace Efficiency Index (0-100 +- CI) [Worthington 2013, Portus 2004, Felton 2018]
  - Pillar 2: Mechanical Load & Durability Screening (0-100 +- CI, higher = lower mechanical stress)
              [Alway 2021, Bayne 2016, Portus 2004, Glazier 2010]
  - Pillar 3: Action Repeatability Index (multi-delivery mode, >=3 deliveries)
  - Observability-weighted aggregation, two-sided knots, interior-extremum CI scanning,
    gated action_class, and complete prioritized coaching cues.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    BiomechanicalFactSheet,
    BowlerActionClass,
    BowlingArm,
    DeliveryEvents,
    MetricEstimate,
    ObservabilityGrade,
    PillarScore,
    PitchCalibration,
    PrioritizedCue,
    QuadrantPlacement,
    VideoAudit,
    WorkloadContext,
)
from .stage3_kinematics import classify_action_portus

OBS_QUALITY_WEIGHT: Dict[ObservabilityGrade, float] = {
    ObservabilityGrade.HIGH: 1.0,
    ObservabilityGrade.MODERATE: 0.78,
    ObservabilityGrade.LOW: 0.42,
    ObservabilityGrade.UNMEASURABLE: 0.0,
    ObservabilityGrade.REFUSED: 0.0,
}


def _piecewise_score(val: float, knots: List[Tuple[float, float]]) -> float:
    xs = [k[0] for k in knots]
    ys = [k[1] for k in knots]
    return float(np.clip(np.interp(val, xs, ys), 0.0, 100.0))


def _score_with_uncertainty(
    m: MetricEstimate,
    knots: List[Tuple[float, float]],
    use_abs: bool = False,
) -> Optional[Tuple[float, float]]:
    """Map a MetricEstimate through a (possibly non-monotonic) biomechanical curve with exact interval CI."""
    if m.suppressed or m.value is None:
        return None
    v_mid = abs(m.value) if use_abs else m.value
    s_mid = _piecewise_score(v_mid, knots)
    if m.ci_low is None or m.ci_high is None:
        return s_mid, 8.0

    if use_abs:
        # Interval of |x| for x in [ci_low, ci_high]
        lo = 0.0 if (m.ci_low <= 0.0 <= m.ci_high) else min(abs(m.ci_low), abs(m.ci_high))
        hi = max(abs(m.ci_low), abs(m.ci_high))
    else:
        lo, hi = m.ci_low, m.ci_high

    # Scan endpoints AND any interior knot extrema inside [lo, hi] for non-monotonic curves!
    eval_scores = [_piecewise_score(lo, knots), _piecewise_score(hi, knots), s_mid]
    for kx, ky in knots:
        if lo <= kx <= hi:
            eval_scores.append(ky)
    ci_half = max(max(eval_scores) - s_mid, s_mid - min(eval_scores), 2.0)
    return s_mid, ci_half


def _evaluate_kinetic_efficiency(
    metrics: Dict[str, MetricEstimate],
    workload: WorkloadContext,
) -> PillarScore:
    """Pillar 1: Fast-bowling pace/kinetic efficiency anchored in Worthington et al. (2013)."""
    is_female = workload.sex.lower().startswith("f")
    runup_knots = (
        [(3.2, 20.0), (4.2, 55.0), (5.0, 82.0), (5.8, 98.0)]
        if is_female
        else [(3.8, 20.0), (4.8, 52.0), (5.6, 80.0), (6.5, 98.0)]
    )

    components: List[Tuple[str, float, List[Tuple[float, float]]]] = [
        ("front_knee_br_deg", 0.28, [(120.0, 15.0), (145.0, 48.0), (160.0, 80.0), (172.0, 96.0), (180.0, 98.0)]),
        ("front_knee_change_deg", 0.12, [(-22.0, 20.0), (-10.0, 46.0), (0.0, 76.0), (8.0, 95.0), (16.0, 98.0)]),
        ("trunk_flexion_delta_deg", 0.24, [(6.0, 18.0), (20.0, 58.0), (32.0, 88.0), (42.0, 98.0)]),
        ("delayed_arm_ffc_deg", 0.18, [(65.0, 20.0), (110.0, 60.0), (140.0, 88.0), (160.0, 98.0)]),
        ("runup_speed_mps", 0.12, runup_knots),
        ("max_hip_shoulder_sep_deg", 0.06, [(5.0, 30.0), (22.0, 68.0), (38.0, 92.0), (55.0, 95.0)]),
    ]

    weighted_sum = 0.0
    eff_weight_total = 0.0
    raw_weight_total = 0.0
    var_sum = 0.0
    used: List[str] = []
    missing: List[str] = []

    for key, w, knots in components:
        m = metrics.get(key)
        if m is None:
            missing.append(key)
            continue
        res = _score_with_uncertainty(m, knots)
        if res is None:
            missing.append(key)
            continue
        sc, ci_h = res
        w_eff = w * OBS_QUALITY_WEIGHT.get(m.observability, 0.5)
        if w_eff <= 0.0:
            missing.append(key)
            continue
        weighted_sum += w_eff * sc
        eff_weight_total += w_eff
        raw_weight_total += w
        var_sum += (w_eff * ci_h) ** 2
        used.append(key)

    if eff_weight_total < 0.22:
        return PillarScore(
            name="Kinetic & Pace Efficiency",
            score=None,
            ci_half_width=None,
            status_label="Insufficient Observable Evidence",
            contributing_metrics=used,
            missing_or_suppressed_metrics=missing,
            summary=(
                "Too few primary pace-efficiency kinematics (front-knee bracing, trunk flexion, delayed arm, "
                "run-up speed) were reliably observable from this camera view and frame rate."
            ),
        )

    score = round(weighted_sum / eff_weight_total, 1)
    missing_penalty = 1.0 + 0.55 * (1.0 - min(raw_weight_total, 1.0))
    ci_half = round(min(28.0, (math.sqrt(var_sum) / eff_weight_total) * missing_penalty), 1)

    if score >= 82.0:
        status = "High Kinetic Efficiency"
    elif score >= 65.0:
        status = "Solid Kinetic Efficiency"
    elif score >= 48.0:
        status = "Moderate (Clear Pace Upside)"
    else:
        status = "Developing (Major Energy Leaks)"

    return PillarScore(
        name="Kinetic & Pace Efficiency",
        score=score,
        ci_half_width=ci_half,
        status_label=status,
        contributing_metrics=used,
        missing_or_suppressed_metrics=missing,
        summary=(
            f"Scored {score:.1f} +- {ci_half:.1f} / 100 based on {len(used)} observable performance predictors "
            f"({', '.join(used)})."
        ),
    )


def _evaluate_lumbar_durability(
    metrics: Dict[str, MetricEstimate],
    workload: WorkloadContext,
) -> PillarScore:
    """Pillar 2: Lumbar mechanical load & durability screening (two-sided curves; higher = lower stress)."""
    components: List[Tuple[str, float, List[Tuple[float, float]], bool]] = [
        # (key, weight, knots, use_abs)
        # Two-sided thorax lateral flexion at BR: ipsilateral negative bend penalized too!
        (
            "thorax_lat_flex_br_deg",
            0.34,
            [(-25.0, 35.0), (0.0, 74.0), (18.0, 92.0), (36.0, 90.0), (42.0, 76.0), (48.0, 48.0), (55.0, 22.0), (62.0, 10.0)],
            False,
        ),
        (
            "thorax_lat_flex_ffc_deg",
            0.14,
            [(-20.0, 40.0), (0.0, 82.0), (12.0, 94.0), (24.0, 86.0), (34.0, 58.0), (45.0, 24.0)],
            False,
        ),
        ("shoulder_counter_rotation_deg", 0.24, [(5.0, 95.0), (20.0, 84.0), (30.0, 52.0), (42.0, 22.0), (55.0, 10.0)], False),
        # Two-sided pelvic tilt/drop control
        ("pelvic_drop_ffc_deg", 0.14, [(-22.0, 30.0), (-12.0, 68.0), (0.0, 94.0), (8.0, 84.0), (16.0, 50.0), (24.0, 20.0)], False),
        # Symmetric |follow_through_drift_deg| penalizing heavy off-side OR leg-side drift equally
        ("follow_through_drift_deg", 0.08, [(0.0, 92.0), (10.0, 90.0), (20.0, 60.0), (35.0, 25.0), (55.0, 10.0)], True),
        ("rear_hip_bfc_deg", 0.06, [(100.0, 30.0), (125.0, 65.0), (145.0, 90.0), (168.0, 88.0)], False),
    ]

    weighted_sum = 0.0
    eff_weight_total = 0.0
    raw_weight_total = 0.0
    var_sum = 0.0
    used: List[str] = []
    missing: List[str] = []

    for key, w, knots, use_abs in components:
        m = metrics.get(key)
        if m is None:
            missing.append(key)
            continue
        res = _score_with_uncertainty(m, knots, use_abs=use_abs)
        if res is None:
            missing.append(key)
            continue
        sc, ci_h = res
        w_eff = w * OBS_QUALITY_WEIGHT.get(m.observability, 0.5)
        if w_eff <= 0.0:
            missing.append(key)
            continue
        weighted_sum += w_eff * sc
        eff_weight_total += w_eff
        raw_weight_total += w
        var_sum += (w_eff * ci_h) ** 2
        used.append(key)

    if eff_weight_total < 0.20:
        return PillarScore(
            name="Mechanical Load & Durability Screening",
            score=None,
            ci_half_width=None,
            status_label="Insufficient Observable Evidence",
            contributing_metrics=used,
            missing_or_suppressed_metrics=missing,
            summary=(
                "Primary spinal-loading views (coronal lateral flexion, pelvic drop, or transverse SCR) "
                "were not reliably observable from this camera angle. Add a behind-the-bowler or 45-degree clip."
            ),
        )

    score = round(weighted_sum / eff_weight_total, 1)
    missing_penalty = 1.0 + 0.55 * (1.0 - min(raw_weight_total, 1.0))
    ci_half = round(min(28.0, (math.sqrt(var_sum) / eff_weight_total) * missing_penalty), 1)

    if score >= 78.0:
        status = "Low Mechanical Load Flag"
    elif score >= 58.0:
        status = "Moderate Mechanical Load Flag"
    else:
        status = "Elevated Lumbar Load Flag"

    wk_note = ""
    if workload.age_years is not None and workload.age_years <= 19:
        wk_note = " Adolescent skeletal maturity context: monitor acute/chronic bowling workloads strictly."
    elif workload.bowls_per_week is not None and workload.bowls_per_week >= 180:
        wk_note = f" High weekly volume ({workload.bowls_per_week} balls/wk) amplifies per-delivery mechanical stress."

    return PillarScore(
        name="Mechanical Load & Durability Screening",
        score=score,
        ci_half_width=ci_half,
        status_label=status,
        contributing_metrics=used,
        missing_or_suppressed_metrics=missing,
        summary=(
            f"Durability index {score:.1f} +- {ci_half:.1f} / 100 ({status}) from {len(used)} observable "
            f"lumbar-loading markers ({', '.join(used)}).{wk_note}"
        ),
    )


def _place_quadrant(kinetic: PillarScore, durability: PillarScore) -> QuadrantPlacement:
    x, x_ci = kinetic.score, kinetic.ci_half_width
    y, y_ci = durability.score, durability.ci_half_width
    if x is None or y is None:
        return QuadrantPlacement(
            kinetic_efficiency_x=x,
            kinetic_efficiency_ci=x_ci,
            durability_safety_y=y,
            durability_safety_ci=y_ci,
            quadrant_label="Partial Observability (Add Complementary Camera Angle)",
            interpretation=(
                "Full 2D Performance-Durability placement requires both sagittal (side-on) kinetic "
                "evidence and coronal/transverse (rear or 45-deg) spinal-load evidence."
            ),
        )
    if x >= 70.0 and y >= 70.0:
        label = "High Efficiency + High Durability"
        interp = "Strong momentum transfer with controlled lumbar loading; focus on repeatability and fine-tuning."
    elif x >= 70.0 and y < 70.0:
        label = "High Efficiency + Elevated Mechanical Load"
        interp = (
            "Explosive pace mechanics paired with elevated lumbar stress markers; preserve natural pace "
            "drivers while prioritizing lumbopelvic stability, workload management, and lateral-bend control."
        )
    elif x < 70.0 and y >= 70.0:
        label = "Moderate Efficiency + High Durability"
        interp = (
            "Mechanically low-stress action with clear safe pace upside—highest return from improving "
            "front-knee bracing, trunk flexion range, or delayed bowling-arm sequencing."
        )
    else:
        label = "Developing Efficiency + Elevated Mechanical Load"
        interp = (
            "Both energy leaks and elevated spinal load flags detected; prioritize structural alignment "
            "and lumbopelvic control before pushing maximum velocity."
        )
    return QuadrantPlacement(
        kinetic_efficiency_x=x,
        kinetic_efficiency_ci=x_ci,
        durability_safety_y=y,
        durability_safety_ci=y_ci,
        quadrant_label=label,
        interpretation=interp,
    )


def _gate_action_class(
    raw_action_class: BowlerActionClass,
    metrics: Dict[str, MetricEstimate],
) -> BowlerActionClass:
    """Gate action classification on observable transverse alignment so unobservable views return UNKNOWN."""
    sho = metrics.get("shoulder_alignment_bfc_deg")
    pelv = metrics.get("pelvis_alignment_bfc_deg")
    scr = metrics.get("shoulder_counter_rotation_deg")

    if sho is None or sho.suppressed or sho.value is None or sho.observability == ObservabilityGrade.LOW:
        return BowlerActionClass.UNKNOWN
    if scr is not None and not scr.suppressed and scr.value is not None:
        p_val = pelv.value if (pelv is not None and not pelv.suppressed and pelv.value is not None) else sho.value
        return classify_action_portus(sho.value, p_val, scr.value)
    return raw_action_class


def _extract_prioritized_cues(
    metrics: Dict[str, MetricEstimate],
    audit: VideoAudit,
    cal: PitchCalibration,
) -> Tuple[List[PrioritizedCue], List[PrioritizedCue], List[PrioritizedCue]]:
    strengths: List[PrioritizedCue] = []
    limiters: List[PrioritizedCue] = []
    upgrades: List[PrioritizedCue] = []

    fk_br = metrics.get("front_knee_br_deg")
    fk_ch = metrics.get("front_knee_change_deg")
    if fk_br and not fk_br.suppressed and fk_br.value is not None:
        if fk_br.value >= 162.0 and not (fk_ch and not fk_ch.suppressed and fk_ch.value is not None and fk_ch.value <= -10.0):
            strengths.append(
                PrioritizedCue(
                    category="strength",
                    title="Strong Front-Leg Bracing at Release",
                    evidence_metrics=["front_knee_br_deg"] + (["front_knee_change_deg"] if fk_ch and not fk_ch.suppressed else []),
                    biomechanical_mechanism=(
                        f"Front knee angle at release ({fk_br.value:.1f} deg, 95% CI [{fk_br.ci_low:.1f}, {fk_br.ci_high:.1f}]) "
                        "provides a rigid braking lever to convert horizontal run-up momentum into upper-body rotation (Worthington 2013)."
                    ),
                    actionable_recommendation="Maintain eccentric quadriceps/hamstring braking capacity and avoid over-striding.",
                )
            )
        elif fk_br.value < 150.0:
            limiters.append(
                PrioritizedCue(
                    category="limiter",
                    title="Front-Knee Flexion / Yielding at Release",
                    evidence_metrics=["front_knee_br_deg"],
                    biomechanical_mechanism=(
                        f"Front knee at release ({fk_br.value:.1f} deg, 95% CI [{fk_br.ci_low:.1f}, {fk_br.ci_high:.1f}]) "
                        "absorbs braking impulse rather than catapulting the trunk forward, leaking potential ball speed."
                    ),
                    actionable_recommendation=(
                        "Check delivery stride length (avoid excessive over-striding), build eccentric lead-leg "
                        "stiffness (drop landings / split-stance isometric holds), and cue a firmer front-foot block."
                    ),
                )
            )

    if fk_ch and not fk_ch.suppressed and fk_ch.value is not None and fk_ch.value <= -10.0:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Front-Knee Collapse During Delivery (FFC -> BR)",
                evidence_metrics=["front_knee_change_deg"],
                biomechanical_mechanism=(
                    f"Front knee flexed by {abs(fk_ch.value):.1f} deg between FFC and BR (change={fk_ch.value:.1f} deg), "
                    "indicating a flexor-collapse braking pattern (Portus 2004) that dissipates vertical braking impulse."
                ),
                actionable_recommendation="Develop eccentric single-leg braking strength and cue driving hip extension over a braced front leg.",
            )
        )

    tf_d = metrics.get("trunk_flexion_delta_deg")
    if tf_d and not tf_d.suppressed and tf_d.value is not None:
        if tf_d.value >= 28.0:
            strengths.append(
                PrioritizedCue(
                    category="strength",
                    title="Effective Forward Trunk Flexion Catapult",
                    evidence_metrics=["trunk_flexion_delta_deg"],
                    biomechanical_mechanism=(
                        f"Upper trunk flexed forward by {tf_d.value:.1f} deg from FFC to BR (95% CI [{tf_d.ci_low:.1f}, {tf_d.ci_high:.1f}]), "
                        "one of the four primary predictors of elite fast-bowling ball speed (Worthington 2013)."
                    ),
                    actionable_recommendation="Preserve trunk flexion timing over the braced front leg without collapsing laterally.",
                )
            )
        elif tf_d.value < 20.0:
            limiters.append(
                PrioritizedCue(
                    category="limiter",
                    title="Limited Forward Trunk Flexion (FFC -> BR)",
                    evidence_metrics=["trunk_flexion_delta_deg"],
                    biomechanical_mechanism=(
                        f"Forward trunk flexion range ({tf_d.value:.1f} deg) is below elite fast-bowling ranges (~25-42 deg), "
                        "reducing linear-to-angular momentum transfer into the bowling shoulder."
                    ),
                    actionable_recommendation="Pair a braced front leg with dynamic chest-drive drills through release.",
                )
            )

    da = metrics.get("delayed_arm_ffc_deg")
    if da and not da.suppressed and da.value is not None:
        if da.value >= 132.0:
            strengths.append(
                PrioritizedCue(
                    category="strength",
                    title="Well-Delayed Bowling Arm at FFC",
                    evidence_metrics=["delayed_arm_ffc_deg"],
                    biomechanical_mechanism=(
                        f"Bowling arm remained {da.value:.1f} deg behind vertical at FFC, maximizing proximal-to-distal "
                        "whip acceleration once the front foot blocks."
                    ),
                    actionable_recommendation="Retain late arm acceleration rhythm during high-effort spells.",
                )
            )
        elif da.value < 105.0:
            limiters.append(
                PrioritizedCue(
                    category="limiter",
                    title="Early Bowling-Arm Circumduction at FFC",
                    evidence_metrics=["delayed_arm_ffc_deg"],
                    biomechanical_mechanism=(
                        f"Bowling arm was already forward ({da.value:.1f} deg behind vertical at FFC), indicating "
                        "the arm starts accelerating before the front leg blocks."
                    ),
                    actionable_recommendation="Practice gather-to-FFC separation drills keeping the bowling hand back until front-foot strike.",
                )
            )

    lf_br = metrics.get("thorax_lat_flex_br_deg")
    if lf_br and not lf_br.suppressed and lf_br.value is not None:
        if lf_br.value >= 46.0:
            limiters.append(
                PrioritizedCue(
                    category="limiter",
                    title="Elevated Contralateral Lateral Trunk Flexion at Release",
                    evidence_metrics=["thorax_lat_flex_br_deg"],
                    biomechanical_mechanism=(
                        f"Thorax lateral flexion at release ({lf_br.value:.1f} deg, 95% CI [{lf_br.ci_low:.1f}, {lf_br.ci_high:.1f}]) "
                        "sits near or above the ~48-50 deg prospective lumbar injury band (Bayne et al., 2016; uninjured norm ~32-44 deg)."
                    ),
                    actionable_recommendation=(
                        "Do NOT force an unnatural 0-degree vertical spine; instead target a controlled ~32-40 deg clearance "
                        "via anti-lateral-flexion trunk strength, hip/pelvic stability, and reviewing front-arm pull alignment."
                    ),
                )
            )
        elif 24.0 <= lf_br.value <= 41.0:
            strengths.append(
                PrioritizedCue(
                    category="strength",
                    title="Controlled Contralateral Trunk Clearance",
                    evidence_metrics=["thorax_lat_flex_br_deg"],
                    biomechanical_mechanism=(
                        f"Thorax lateral flexion at release ({lf_br.value:.1f} deg) sits cleanly within the uninjured "
                        "elite fast-bowling reference range (Bayne et al., 2016: 40 +- 8 deg)."
                    ),
                    actionable_recommendation="Maintain current lateral trunk stability as bowling intensity increases.",
                )
            )

    scr = metrics.get("shoulder_counter_rotation_deg")
    if scr and not scr.suppressed and scr.value is not None and scr.value >= 30.0:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Mixed-Action Shoulder Counter-Rotation (>=30 deg)",
                evidence_metrics=["shoulder_counter_rotation_deg"],
                biomechanical_mechanism=(
                    f"Measured shoulder counter-rotation ({scr.value:.1f} deg) meets or exceeds Portus et al. (2004)'s "
                    "30-degree mixed-action threshold associated with increased torsional lumbar stress."
                ),
                actionable_recommendation=(
                    "Align back-foot landing, hips, and shoulders coherently at BFC (either consistently semi-open or side-on) "
                    "to eliminate sharp upper-trunk recoil between BFC and FFC."
                ),
            )
        )

    pd = metrics.get("pelvic_drop_ffc_deg")
    if pd and not pd.suppressed and pd.value is not None and abs(pd.value) >= 12.0:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Elevated Pelvic Tilt / Drop at FFC",
                evidence_metrics=["pelvic_drop_ffc_deg"],
                biomechanical_mechanism=(
                    f"Pelvic tilt at FFC ({pd.value:.1f} deg) indicates reduced frontal-plane lumbopelvic control "
                    "(Alway et al., 2021), elevating asymmetrical lumbar shear during braking."
                ),
                actionable_recommendation="Target gluteus medius / single-leg lumbopelvic stability exercises under eccentric load.",
            )
        )

    ft = metrics.get("follow_through_drift_deg")
    if ft and not ft.suppressed and ft.value is not None and abs(ft.value) >= 18.0:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Excessive Lateral Follow-Through Drift",
                evidence_metrics=["follow_through_drift_deg"],
                biomechanical_mechanism=(
                    f"Post-release momentum drifted {abs(ft.value):.1f} deg off the target delivery axis, "
                    "increasing lateral braking shear and energy dissipation away from the stumps."
                ),
                actionable_recommendation="Cue driving the back leg and chest straight down the target corridor through the first 2 follow-through steps.",
            )
        )

    rh = metrics.get("rear_hip_bfc_deg")
    if rh and not rh.suppressed and rh.value is not None and rh.value < 125.0:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Collapsed Rear Hip Posture at BFC",
                evidence_metrics=["rear_hip_bfc_deg"],
                biomechanical_mechanism=(
                    f"Rear hip angle at BFC ({rh.value:.1f} deg) indicates upstream hip-flexion collapse, a key predictor "
                    "in Alway et al. (2021)'s prospective lumbar bone stress injury model."
                ),
                actionable_recommendation="Strengthen rear-leg triple extension and cue a taller, stacked gather entering BFC.",
            )
        )

    # Capture quality upgrades
    if audit.fps < 100.0:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Switch Smartphone to 120–240 fps Slow-Motion",
                evidence_metrics=["peak_arm_angular_vel_dps", "delayed_arm_ffc_deg"],
                biomechanical_mechanism=(
                    f"Current clip is {audit.fps:.1f} fps (~{audit.est_arm_deg_per_frame:.0f} deg arm sweep per frame), "
                    "creating severe temporal aliasing near release."
                ),
                actionable_recommendation="Record outdoors in native 1080p 240 fps slow-motion to reduce timing uncertainty by up to 8x.",
            )
        )
    if cal.is_panning_camera and not cal.is_metric_calibrated:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Use a Static Tripod Showing Pitch Creases",
                evidence_metrics=["runup_speed_mps", "stride_length_pct_stature"],
                biomechanical_mechanism="Camera panning hides ground-referenced run-up velocity and true BFC-to-FFC stride displacement.",
                actionable_recommendation="Mount the camera on a stationary tripod framing both popping/bowling creases and the bowler's full body.",
            )
        )
    suppressed_coronal = any(
        metrics[k].suppressed for k in ("thorax_lat_flex_br_deg", "pelvic_drop_ffc_deg") if k in metrics
    )
    suppressed_sagittal = any(
        metrics[k].suppressed for k in ("front_knee_br_deg", "trunk_flexion_delta_deg") if k in metrics
    )
    az_abs = abs(cal.azimuth_deg) % 360.0
    az_abs = 360.0 - az_abs if az_abs > 180.0 else az_abs
    # Only recommend adding a rear/side angle when the current camera geometry actually lacks that plane
    # (do NOT recommend "Add a Rear Camera" to an already rear-on clip suppressed by 25 fps blur!).
    if suppressed_coronal and abs(math.cos(math.radians(az_abs))) < 0.65:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Add a Behind-the-Bowler (Rear) Camera Angle",
                evidence_metrics=["thorax_lat_flex_br_deg", "pelvic_drop_ffc_deg"],
                biomechanical_mechanism="Side-on cameras cannot accurately resolve coronal lateral spine bend or pelvic drop.",
                actionable_recommendation="Place a tripod phone at waist height behind the umpire/bowler aligned with middle stump.",
            )
        )
    if suppressed_sagittal and abs(math.sin(math.radians(az_abs))) < 0.65:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Add a Side-On (90 deg) Camera Angle",
                evidence_metrics=["front_knee_br_deg", "trunk_flexion_delta_deg", "delayed_arm_ffc_deg"],
                biomechanical_mechanism="Rear/front cameras cannot accurately resolve sagittal front-knee bracing or forward trunk flexion.",
                actionable_recommendation="Place a tripod phone 8-12 m perpendicular to the popping crease showing the full run-up to follow-through.",
            )
        )

    return strengths[:3], limiters[:4], upgrades[:3]


def evaluate_repeatability(
    fact_sheets: List[BiomechanicalFactSheet],
) -> Optional[PillarScore]:
    """Pillar 3: Compute multi-delivery Action Repeatability Index when >=3 deliveries are provided."""
    if len(fact_sheets) < 3:
        return None
    tracked_keys = [
        ("front_knee_br_deg", 6.0),
        ("trunk_flexion_delta_deg", 6.0),
        ("thorax_lat_flex_br_deg", 5.0),
        ("stride_length_pct_stature", 4.0),
        ("release_height_m", 0.06),
    ]
    sub_scores = []
    used = []
    for k, ref_sd in tracked_keys:
        vals = [
            fs.metrics[k].value
            for fs in fact_sheets
            if k in fs.metrics and not fs.metrics[k].suppressed and fs.metrics[k].value is not None
        ]
        if len(vals) >= 3:
            sd = float(np.std(vals, ddof=1))
            sc = float(np.clip(100.0 * math.exp(-0.5 * (sd / ref_sd) ** 2), 15.0, 99.0))
            sub_scores.append(sc)
            used.append(f"{k}(SD={sd:.2f})")
    if not sub_scores:
        return None
    mean_sc = round(float(np.mean(sub_scores)), 1)
    status = "Elite Repeatability" if mean_sc >= 82.0 else ("Consistent" if mean_sc >= 65.0 else "Variable Action")
    return PillarScore(
        name="Action Repeatability Index",
        score=mean_sc,
        ci_half_width=5.0,
        status_label=status,
        contributing_metrics=used,
        missing_or_suppressed_metrics=[],
        summary=f"Across {len(fact_sheets)} deliveries, inter-delivery repeatability scored {mean_sc:.1f}/100 ({status}).",
    )


def build_fact_sheet(
    bowler_id: str,
    bowling_arm: BowlingArm,
    action_class: BowlerActionClass,
    pose_backend: str,
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    metrics: Dict[str, MetricEstimate],
    workload: Optional[WorkloadContext] = None,
    metadata: Optional[Dict] = None,
) -> BiomechanicalFactSheet:
    """Assemble the complete, strictly verified BiomechanicalFactSheet."""
    wk = workload if workload is not None else WorkloadContext()
    gated_action = _gate_action_class(action_class, metrics)
    kinetic = _evaluate_kinetic_efficiency(metrics, wk)
    durability = _evaluate_lumbar_durability(metrics, wk)
    quad = _place_quadrant(kinetic, durability)
    strengths, limiters, upgrades = _extract_prioritized_cues(metrics, audit, cal)

    return BiomechanicalFactSheet(
        bowler_id=bowler_id,
        bowling_arm=bowling_arm,
        action_class=gated_action,
        pose_backend=pose_backend,
        audit=audit,
        calibration=cal,
        events=events,
        workload_context=wk,
        metrics=metrics,
        kinetic_efficiency=kinetic,
        lumbar_load_screening=durability,
        repeatability=None,
        quadrant=quad,
        strengths=strengths,
        limiters=limiters,
        capture_upgrades=upgrades,
        metadata=metadata or {},
    )
