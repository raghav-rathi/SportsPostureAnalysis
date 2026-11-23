"""Stage 5: Definition-Matched Cohort Evaluator, Interval-Aware Cues, & Workload Guidance.

Fixes EXP-007:
  - Section 6 & 7.8: Retires unvalidated 0-100 composite kinetic/durability scores (`score=None`)
                     until externally validated against real reference outcomes; replaces them with
                     definition-matched primary-source cohort comparisons and official ECB (2024) /
                     Cricket Australia (2026) age-band workload guidance.
  - B8 (S9): Enforces strict non-straddling printed-interval gating (`ci_low >= thresh` /
             `ci_high <= thresh` AND `observability in (HIGH, MODERATE)`) on all strength/limiter cues,
             routing interval-straddling observations cleanly to `inconclusive_observations`.
  - B9 / B21: Eliminates single-metric 0.22 score loophole; upgrades `evaluate_repeatability` to
              subtract mean measurement variance before reporting inter-delivery repeatability.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from .schemas import (
    BiomechanicalFactSheet,
    BowlerActionClass,
    BowlingArm,
    CohortComparison,
    DeliveryEvents,
    IntervalType,
    MetricEstimate,
    ObservabilityGrade,
    PillarScore,
    PitchCalibration,
    PrioritizedCue,
    QuadrantPlacement,
    VideoAudit,
    WorkloadContext,
    WorkloadGuidance,
)
from .stage3_kinematics import classify_action_portus


def _normal_cdf_pct(z: float) -> float:
    return round(50.0 * (1.0 + math.erf(z / math.sqrt(2.0))), 1)


def build_cohort_comparisons(
    metrics: Dict[str, MetricEstimate],
    workload: WorkloadContext,
) -> List[CohortComparison]:
    """Compare verified non-suppressed metrics against published primary-source cohort distributions."""
    is_female = workload.sex.lower().startswith("f")
    # (key, cohort_name, mean, sd, definition_match, definition_note, source)
    specs = [
        (
            "runup_speed_mps",
            "Elite Female Fast (Felton 2019, n=20)" if is_female else "Elite Male Fast (Worthington 2013, n=20)",
            4.81 if is_female else 5.83,
            0.42 if is_female else 0.50,
            "exact",
            "Horizontal pelvis speed approaching BFC (+0.89 m/s ball speed per SD in Worthington 2013).",
            "Worthington et al. (2013) / Felton et al. (2019)",
        ),
        (
            "front_knee_br_deg",
            "Elite Male Fast (Worthington 2013, n=20)",
            165.0,
            14.0,
            "exact",
            "Internal 3D front-knee angle at BR (+0.79 m/s ball speed per SD in Worthington 2013).",
            "Worthington, King, & Ranson (2013), J. Applied Biomechanics",
        ),
        (
            "trunk_flexion_delta_deg",
            "Elite Male Fast (Worthington 2013, n=20)",
            33.0,
            7.5,
            "proxy_global_frame",
            "Global vertical trunk inclination change (proxy; Worthington measured upper-vs-lower trunk angle).",
            "Worthington et al. (2013) [global-frame proxy]",
        ),
        (
            "thorax_lat_flex_br_deg",
            "Adolescent Uninjured Cohort (Bayne 2016, ages 14-19, n=13)",
            40.0,
            8.0,
            "proxy_global_frame",
            "Whole-trunk global tilt proxy. Note: Bayne (2016) studied adolescents (injured 50+-6 vs uninjured 40+-8); Alway (2021) adult study found injured bowlers LESS side-flexed at BR.",
            "Bayne et al. (2016) [adolescents]; Alway et al. (2021) [adults]",
        ),
        (
            "rear_hip_bfc_deg",
            "Elite Adult Uninjured vs LBSI (Alway 2021, n=50)",
            156.0,
            9.0,
            "proxy_trunk_vector",
            "Trunk-to-thigh vector proxy (Alway 2021 measured 3D pelvis-to-thigh: uninjured 156+-9 vs LBSI 146+-10 deg; 88% in-sample vs 78% base rate).",
            "Alway et al. (2021), MSSE [trunk-thigh proxy]",
        ),
        (
            "shoulder_counter_rotation_deg",
            "AIS Fast-Bowling Cohort (Portus 2001/2004, n=42)",
            22.0,
            10.0,
            "exact",
            "AIS mixed-action threshold is >=30 deg (retrospective link in Portus 2004; no prospective LBSI link in Alway 2021).",
            "Portus et al. (2001/2004); Alway et al. (2021)",
        ),
    ]

    comps: List[CohortComparison] = []
    for key, cname, mean_v, sd_v, dmatch, dnote, src in specs:
        m = metrics.get(key)
        if m is None or m.suppressed or m.value is None or m.ci_low is None or m.ci_high is None:
            continue
        z = (m.value - mean_v) / max(sd_v, 1e-6)
        comps.append(
            CohortComparison(
                metric_key=key,
                label=m.label,
                value=m.value,
                ci_low=m.ci_low,
                ci_high=m.ci_high,
                unit=m.unit,
                cohort_name=cname,
                cohort_mean=mean_v,
                cohort_sd=sd_v,
                z_score=round(z, 2),
                percentile_approx=_normal_cdf_pct(z),
                definition_match=dmatch,  # type: ignore[arg-type]
                definition_note=dnote,
                source=src,
            )
        )
    return comps


def build_workload_guidance(workload: WorkloadContext) -> Optional[WorkloadGuidance]:
    """Provide official ECB (2024) / Cricket Australia (2026) age-band workload limits (no risk score)."""
    if workload.age_years is None and workload.bowls_per_week is None:
        return None
    age = workload.age_years
    if age is None:
        band, spell, day, wk = "Adult (unspecified age)", None, None, 240
    elif age <= 13:
        band, spell, day, wk = "U13 (<=13 yrs)", 30, 60, 120
    elif age <= 15:
        band, spell, day, wk = "U15 (14-15 yrs)", 36, 72, 150
    elif age <= 17:
        band, spell, day, wk = "U17 (16-17 yrs)", 42, 84, 180
    elif age <= 19:
        band, spell, day, wk = "U19 (18-19 yrs)", 42, 108, 210
    else:
        band, spell, day, wk = "Adult (20+ yrs)", None, None, 240

    notes = []
    if workload.bowls_per_week is not None and wk is not None:
        if workload.bowls_per_week > wk:
            notes.append(
                f"Reported weekly load ({workload.bowls_per_week} balls/wk) exceeds general {band} guideline "
                f"(~{wk} balls/wk); manage acute spikes gradually."
            )
        else:
            notes.append(
                f"Reported weekly load ({workload.bowls_per_week} balls/wk) is within general {band} guideline "
                f"(~{wk} balls/wk)."
            )
    else:
        notes.append(f"Reference workload guideline for {band}; avoid sudden >20-30% week-to-week spikes.")
    if workload.recent_growth_spurt:
        notes.append("Recent growth spurt flagged: adolescent bone-load tolerance is transiently lower.")

    return WorkloadGuidance(
        age_band=band,
        governing_body_source="ECB Recreational Fast Bowling Guidance (2024) & Cricket Australia Junior Guidelines (2026)",
        max_balls_per_spell=spell,
        max_balls_per_day=day,
        max_balls_per_week=wk,
        current_bowls_per_week=workload.bowls_per_week,
        status_note=" ".join(notes),
    )


def _retired_pillar(name: str, metrics: Dict[str, MetricEstimate], tracked_keys: List[str]) -> PillarScore:
    used = [k for k in tracked_keys if k in metrics and not metrics[k].suppressed and metrics[k].value is not None]
    missing = [k for k in tracked_keys if k not in used]
    return PillarScore(
        name=name,
        score=None,
        ci_half_width=None,
        status_label="Retired (Unvalidated Composite — See Verified Metrics & Cohort Norms)",
        contributing_metrics=used,
        missing_or_suppressed_metrics=missing,
        summary=(
            f"0-100 composite scores are retired until validated against external ground-truth outcomes "
            f"({len(used)}/{len(tracked_keys)} underlying metrics observable)."
        ),
    )


def _retired_quadrant() -> QuadrantPlacement:
    return QuadrantPlacement(
        kinetic_efficiency_x=None,
        kinetic_efficiency_ci=None,
        durability_safety_y=None,
        durability_safety_ci=None,
        quadrant_label="Retired (See Individual Verified Metrics & Cohort Comparisons)",
        interpretation=(
            "Composite 2D quadrant placement is disabled until underlying metrics are calibrated against "
            "external biomechanical & workload outcomes."
        ),
    )


def _gate_action_class(
    raw_action_class: BowlerActionClass,
    metrics: Dict[str, MetricEstimate],
) -> BowlerActionClass:
    """Gate action classification on HIGH/MODERATE observable transverse alignment."""
    sho = metrics.get("shoulder_alignment_bfc_deg")
    pelv = metrics.get("pelvis_alignment_bfc_deg")
    scr = metrics.get("shoulder_counter_rotation_deg")

    if sho is None or sho.suppressed or sho.value is None or sho.observability not in (ObservabilityGrade.HIGH, ObservabilityGrade.MODERATE):
        return BowlerActionClass.UNKNOWN
    if scr is not None and not scr.suppressed and scr.value is not None and scr.observability in (ObservabilityGrade.HIGH, ObservabilityGrade.MODERATE):
        p_val = pelv.value if (pelv is not None and not pelv.suppressed and pelv.value is not None) else sho.value
        return classify_action_portus(sho.value, p_val, scr.value)
    # Without reliable SCR we cannot rule out MIXED action honestly
    return BowlerActionClass.UNKNOWN


def _clears_upper(m: Optional[MetricEstimate], thresh: float) -> Tuple[bool, bool]:
    """Return (cleared_interval, straddled_point_only) for upper-tail threshold `value >= thresh`."""
    if m is None or m.suppressed or m.value is None or m.ci_low is None or m.ci_high is None:
        return False, False
    if m.observability not in (ObservabilityGrade.HIGH, ObservabilityGrade.MODERATE):
        return False, (m.value >= thresh)
    if m.ci_low >= thresh:
        return True, False
    if m.value >= thresh:
        return False, True
    return False, False


def _clears_lower(m: Optional[MetricEstimate], thresh: float) -> Tuple[bool, bool]:
    """Return (cleared_interval, straddled_point_only) for lower-tail threshold `value <= thresh`."""
    if m is None or m.suppressed or m.value is None or m.ci_low is None or m.ci_high is None:
        return False, False
    if m.observability not in (ObservabilityGrade.HIGH, ObservabilityGrade.MODERATE):
        return False, (m.value <= thresh)
    if m.ci_high <= thresh:
        return True, False
    if m.value <= thresh:
        return False, True
    return False, False


def _clears_abs_upper(m: Optional[MetricEstimate], thresh: float) -> Tuple[bool, bool]:
    """Return (cleared_interval, straddled_point_only) for two-sided `|value| >= thresh`."""
    if m is None or m.suppressed or m.value is None or m.ci_low is None or m.ci_high is None:
        return False, False
    if m.observability not in (ObservabilityGrade.HIGH, ObservabilityGrade.MODERATE):
        return False, (abs(m.value) >= thresh)
    min_abs = 0.0 if (m.ci_low <= 0.0 <= m.ci_high) else min(abs(m.ci_low), abs(m.ci_high))
    if min_abs >= thresh:
        return True, False
    if abs(m.value) >= thresh:
        return False, True
    return False, False


def _extract_prioritized_cues(
    metrics: Dict[str, MetricEstimate],
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
) -> Tuple[List[PrioritizedCue], List[PrioritizedCue], List[PrioritizedCue], List[PrioritizedCue]]:
    """Extract strictly interval-cleared strengths & limiters plus inconclusive observations & capture upgrades."""
    strengths: List[PrioritizedCue] = []
    limiters: List[PrioritizedCue] = []
    inconclusive: List[PrioritizedCue] = []
    upgrades: List[PrioritizedCue] = []

    def _add_inconclusive(title: str, m: MetricEstimate, thresh_desc: str) -> None:
        inconclusive.append(
            PrioritizedCue(
                category="inconclusive",
                title=f"Inconclusive: {title}",
                evidence_metrics=[m.key],
                biomechanical_mechanism=(
                    f"Point estimate ({m.value:.1f} {m.unit}) sits past {thresh_desc}, but its interval "
                    f"[{m.ci_low:.1f}, {m.ci_high:.1f}] (observability={m.observability.value}) straddles the threshold."
                ),
                actionable_recommendation="Record at higher FPS (120-240 fps) or with calibrated pitch landmarks to narrow uncertainty before acting.",
            )
        )

    # 1. Front knee at BR & knee change (strict interval clearance -> eliminates S9 51/53 false alarms!)
    fk_br = metrics.get("front_knee_br_deg")
    fk_ch = metrics.get("front_knee_change_deg")
    ch_collapse, ch_collapse_inc = _clears_lower(fk_ch, -10.0)
    if ch_collapse and fk_ch is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Front-Knee Collapse During Delivery (FFC -> BR)",
                evidence_metrics=["front_knee_change_deg"],
                biomechanical_mechanism=(
                    f"Front knee flexed across delivery (change={fk_ch.value:.1f} deg, interval "
                    f"[{fk_ch.ci_low:.1f}, {fk_ch.ci_high:.1f}], strictly <= -10 deg), indicating a flexor-collapse braking pattern."
                ),
                actionable_recommendation="Develop eccentric single-leg braking strength and cue driving hip extension over a braced front leg.",
            )
        )
    elif ch_collapse_inc and fk_ch is not None:
        _add_inconclusive("Front-Knee Collapse (FFC -> BR)", fk_ch, "<= -10 deg")

    fk_strong, _ = _clears_upper(fk_br, 160.0)
    fk_yield, fk_yield_inc = _clears_lower(fk_br, 150.0)
    if fk_strong and not ch_collapse and fk_br is not None:
        strengths.append(
            PrioritizedCue(
                category="strength",
                title="Strong Front-Leg Bracing at Release",
                evidence_metrics=["front_knee_br_deg"],
                biomechanical_mechanism=(
                    f"Front knee angle at release ({fk_br.value:.1f} deg, interval [{fk_br.ci_low:.1f}, {fk_br.ci_high:.1f}]) "
                    "clears 160 deg, providing a firm braking lever associated with higher ball speed (Worthington 2013)."
                ),
                actionable_recommendation="Maintain eccentric quadriceps/hamstring braking capacity and consistent stride length.",
            )
        )
    elif fk_yield and fk_br is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Front-Knee Flexion / Yielding at Release",
                evidence_metrics=["front_knee_br_deg"],
                biomechanical_mechanism=(
                    f"Front knee at release ({fk_br.value:.1f} deg, interval [{fk_br.ci_low:.1f}, {fk_br.ci_high:.1f}]) "
                    "stays strictly below 150 deg, absorbing braking impulse instead of transferring linear momentum."
                ),
                actionable_recommendation="Review stride length (avoid over-striding) and build eccentric lead-leg stiffness.",
            )
        )
    elif fk_yield_inc and fk_br is not None:
        _add_inconclusive("Front-Knee Yielding at Release", fk_br, "< 150 deg")

    # 2. Global trunk flexion change (FFC -> BR)
    tf_d = metrics.get("trunk_flexion_delta_deg")
    tf_strong, _ = _clears_upper(tf_d, 26.0)
    tf_lim, tf_lim_inc = _clears_lower(tf_d, 18.0)
    if tf_strong and tf_d is not None:
        strengths.append(
            PrioritizedCue(
                category="strength",
                title="Effective Forward Trunk Flexion (FFC -> BR)",
                evidence_metrics=["trunk_flexion_delta_deg"],
                biomechanical_mechanism=(
                    f"Global sagittal trunk inclination increased by {tf_d.value:.1f} deg (interval "
                    f"[{tf_d.ci_low:.1f}, {tf_d.ci_high:.1f}]), cleanly exceeding 26 deg."
                ),
                actionable_recommendation="Preserve forward trunk flexion timing over the braced front leg.",
            )
        )
    elif tf_lim and tf_d is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Limited Forward Trunk Flexion (FFC -> BR)",
                evidence_metrics=["trunk_flexion_delta_deg"],
                biomechanical_mechanism=(
                    f"Global sagittal trunk flexion change ({tf_d.value:.1f} deg, interval [{tf_d.ci_low:.1f}, {tf_d.ci_high:.1f}]) "
                    "is strictly below 18 deg, limiting linear-to-angular momentum transfer."
                ),
                actionable_recommendation="Pair front-leg bracing with dynamic chest-drive drills through release.",
            )
        )
    elif tf_lim_inc and tf_d is not None:
        _add_inconclusive("Limited Forward Trunk Flexion", tf_d, "< 18 deg")

    # 3. Delayed bowling arm at FFC (only at >=90 fps)
    da = metrics.get("delayed_arm_ffc_deg")
    da_strong, _ = _clears_upper(da, 128.0)
    da_early, da_early_inc = _clears_lower(da, 105.0)
    if da_strong and da is not None:
        strengths.append(
            PrioritizedCue(
                category="strength",
                title="Well-Delayed Bowling Arm at FFC",
                evidence_metrics=["delayed_arm_ffc_deg"],
                biomechanical_mechanism=(
                    f"Bowling arm remained {da.value:.1f} deg behind global vertical at FFC (interval "
                    f"[{da.ci_low:.1f}, {da.ci_high:.1f}]), supporting late proximal-to-distal acceleration."
                ),
                actionable_recommendation="Retain late arm acceleration rhythm under high-effort spells.",
            )
        )
    elif da_early and da is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Early Bowling-Arm Circumduction at FFC",
                evidence_metrics=["delayed_arm_ffc_deg"],
                biomechanical_mechanism=(
                    f"Bowling arm delay at FFC ({da.value:.1f} deg, interval [{da.ci_low:.1f}, {da.ci_high:.1f}]) "
                    "is strictly below 105 deg, suggesting arm acceleration begins ahead of front-foot block."
                ),
                actionable_recommendation="Practice gather-to-FFC separation drills keeping the bowling hand back until FFC.",
            )
        )
    elif da_early_inc and da is not None:
        _add_inconclusive("Early Bowling-Arm Circumduction", da, "< 105 deg")

    # 4. Shoulder Counter-Rotation (Portus 2001/2004 AIS mixed threshold >=30 deg; strict interval clearance!)
    scr = metrics.get("shoulder_counter_rotation_deg")
    scr_high, scr_inc = _clears_upper(scr, 30.0)
    if scr_high and scr is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Mixed-Action Shoulder Counter-Rotation (>=30 deg)",
                evidence_metrics=["shoulder_counter_rotation_deg"],
                biomechanical_mechanism=(
                    f"Shoulder counter-rotation ({scr.value:.1f} deg, interval [{scr.ci_low:.1f}, {scr.ci_high:.1f}]) "
                    "strictly clears the 30-degree AIS mixed-action threshold (Portus 2001/2004 retrospective; "
                    "note Alway 2021 prospective adult study did not find an independent SCR-LBSI association)."
                ),
                actionable_recommendation="Align hips and shoulders coherently at BFC to minimize abrupt upper-trunk recoil.",
            )
        )
    elif scr_inc and scr is not None:
        _add_inconclusive("Mixed-Action Shoulder Counter-Rotation", scr, ">= 30 deg")

    # 5. Global trunk lateral inclination at BR (honestly caveat adolescent vs adult literature!)
    lf_br = metrics.get("thorax_lat_flex_br_deg")
    lf_high, lf_high_inc = _clears_upper(lf_br, 48.0)
    if lf_high and lf_br is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="High Global Contralateral Trunk Inclination at Release",
                evidence_metrics=["thorax_lat_flex_br_deg"],
                biomechanical_mechanism=(
                    f"Global contralateral trunk inclination at BR ({lf_br.value:.1f} deg, interval "
                    f"[{lf_br.ci_low:.1f}, {lf_br.ci_high:.1f}]) exceeds 48 deg. Bayne (2016) linked >=50 deg "
                    "to low-back injury in adolescents (n=25), though Alway (2021) did not observe this direction in adults."
                ),
                actionable_recommendation="Do not force an artificial 0-degree vertical spine; assess trunk stability and workload progression.",
            )
        )
    elif lf_high_inc and lf_br is not None:
        _add_inconclusive("High Contralateral Trunk Inclination at BR", lf_br, ">= 48 deg")

    # 6. Pelvic tilt / drop & follow-through drift (only when metric passed translation/crop gates AND clears interval!)
    pd = metrics.get("pelvic_drop_ffc_deg")
    pd_high, pd_inc = _clears_abs_upper(pd, 14.0)
    if pd_high and pd is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Elevated Frontal-Plane Pelvic Tilt at FFC",
                evidence_metrics=["pelvic_drop_ffc_deg"],
                biomechanical_mechanism=(
                    f"Frontal pelvic tilt at FFC ({pd.value:.1f} deg, interval [{pd.ci_low:.1f}, {pd.ci_high:.1f}]) "
                    "strictly exceeds +-14 deg, indicating elevated frontal-plane lumbopelvic excursion."
                ),
                actionable_recommendation="Train eccentric single-leg gluteus medius and lumbopelvic control.",
            )
        )
    elif pd_inc and pd is not None:
        _add_inconclusive("Elevated Pelvic Tilt at FFC", pd, "|tilt| >= 14 deg")

    ft = metrics.get("follow_through_drift_deg")
    ft_high, ft_inc = _clears_abs_upper(ft, 18.0)
    if ft_high and ft is not None:
        limiters.append(
            PrioritizedCue(
                category="limiter",
                title="Excessive Lateral Follow-Through Drift",
                evidence_metrics=["follow_through_drift_deg"],
                biomechanical_mechanism=(
                    f"Calibrated post-release pelvis trajectory drifted {abs(ft.value):.1f} deg off pitch axis "
                    f"(interval [{ft.ci_low:.1f}, {ft.ci_high:.1f}])."
                ),
                actionable_recommendation="Cue driving the chest and back leg through the target corridor on the first two steps.",
            )
        )
    elif ft_inc and ft is not None:
        _add_inconclusive("Lateral Follow-Through Drift", ft, "|drift| >= 18 deg")

    # Capture quality upgrades
    if not events.is_complete_delivery:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Record Complete Unedited Delivery Through Follow-Through",
                evidence_metrics=["front_knee_br_deg", "thorax_lat_flex_br_deg"],
                biomechanical_mechanism="No valid overhead ball release was observed (clip was truncated, gather-only, or frozen).",
                actionable_recommendation="Record raw unedited footage from 3 steps before BFC through 3 steps after ball release.",
            )
        )
    if audit.timebase_unreliable:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Use Native Unedited Slow-Motion (Avoid Retimed/App Exports)",
                evidence_metrics=["runup_speed_mps", "peak_arm_angular_vel_dps"],
                biomechanical_mechanism=f"Clip timebase is {audit.timebase_status} ({audit.held_frame_ratio:.0%} repeated frames).",
                actionable_recommendation="Export original camera roll 120/240 fps files without third-party slow-mo re-encoding or freeze frames.",
            )
        )
    if audit.fps < 100.0:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Switch Smartphone to 120–240 fps Slow-Motion",
                evidence_metrics=["peak_arm_angular_vel_dps", "delayed_arm_ffc_deg"],
                biomechanical_mechanism=(
                    f"Effective unique rate is {audit.fps:.1f} fps (~{audit.est_arm_deg_per_frame:.0f} deg arm sweep/frame)."
                ),
                actionable_recommendation="Record outdoors in native 1080p 240 fps slow-motion.",
            )
        )
    if cal.is_intrinsics_calibrated and cal.is_panning_camera:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Lock Camera on a Static Tripod (Eliminate Handheld Pan)",
                evidence_metrics=["runup_speed_mps", "stride_length_pct_stature", "follow_through_drift_deg"],
                biomechanical_mechanism=(
                    "Pitch intrinsics were calibrated from visible wickets/creases, but handheld camera panning "
                    "prevents reliable world-frame pelvis translation, stride length, and follow-through drift."
                ),
                actionable_recommendation="Lock the tripod head during the delivery so background wickets stay stationary.",
            )
        )
    elif cal.is_panning_camera or not cal.is_metric_calibrated:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Use a Static Tripod Framing Both Wickets or Pitch Creases",
                evidence_metrics=["runup_speed_mps", "stride_length_pct_stature", "follow_through_drift_deg"],
                biomechanical_mechanism=(
                    "Calibrating camera focal length, gravity, and pitch axis from MCC creases/wickets removes "
                    "telephoto depth compression and unlocks world-grounded run-up speed, stride, and drift."
                ),
                actionable_recommendation="Mount on a static tripod framing visible creases/stumps and supply pitch landmarks.",
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
    if suppressed_coronal and abs(math.cos(math.radians(az_abs))) < 0.65:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Add a Behind-the-Bowler (Rear) Camera Angle",
                evidence_metrics=["thorax_lat_flex_br_deg", "pelvic_drop_ffc_deg"],
                biomechanical_mechanism="Side-on cameras cannot accurately resolve coronal lateral spine inclination or pelvic tilt.",
                actionable_recommendation="Place a tripod phone at waist height behind the umpire aligned with middle stump.",
            )
        )
    if suppressed_sagittal and abs(math.sin(math.radians(az_abs))) < 0.65:
        upgrades.append(
            PrioritizedCue(
                category="capture_upgrade",
                title="Add a Side-On (90 deg) Camera Angle",
                evidence_metrics=["front_knee_br_deg", "trunk_flexion_delta_deg", "delayed_arm_ffc_deg"],
                biomechanical_mechanism="Rear/front cameras cannot accurately resolve sagittal front-knee bracing or trunk flexion.",
                actionable_recommendation="Place a tripod phone 8-12 m perpendicular to the popping crease showing full body.",
            )
        )

    return strengths[:3], limiters[:4], inconclusive[:4], upgrades[:4]


def evaluate_repeatability(
    fact_sheets: List[BiomechanicalFactSheet],
) -> Optional[PillarScore]:
    """Compute multi-delivery inter-ball repeatability after subtracting mean within-ball measurement variance."""
    if len(fact_sheets) < 3:
        return None
    tracked_keys = [
        "front_knee_br_deg",
        "trunk_flexion_delta_deg",
        "thorax_lat_flex_br_deg",
        "stride_length_pct_stature",
        "release_height_m",
    ]
    used: List[str] = []
    for k in tracked_keys:
        entries = [
            fs.metrics[k]
            for fs in fact_sheets
            if k in fs.metrics and not fs.metrics[k].suppressed and fs.metrics[k].value is not None
        ]
        if len(entries) >= 3:
            vals = np.array([e.value for e in entries], dtype=np.float64)
            sigs = np.array([e.sigma_total or 0.0 for e in entries], dtype=np.float64)
            obs_var = float(np.var(vals, ddof=1))
            meas_var = float(np.mean(sigs**2))
            true_sd = math.sqrt(max(obs_var - meas_var, 0.0))
            used.append(f"{k}(raw_SD={math.sqrt(obs_var):.2f}, debiased_SD={true_sd:.2f})")
    if not used:
        return None
    return PillarScore(
        name="Multi-Delivery Inter-Ball Repeatability",
        score=None,
        ci_half_width=None,
        status_label="Descriptive Multi-Delivery Spread (Measurement-Variance Debiased)",
        contributing_metrics=used,
        missing_or_suppressed_metrics=[],
        summary=(
            f"Across {len(fact_sheets)} deliveries, measurement-debiased inter-delivery spread computed for "
            f"{len(used)} metrics ({'; '.join(used)})."
        ),
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
    interval_type: IntervalType = "analytical_uncertainty",
) -> BiomechanicalFactSheet:
    """Assemble the complete, strictly verified BiomechanicalFactSheet (schema v2.0.0)."""
    wk = workload if workload is not None else WorkloadContext()
    gated_action = _gate_action_class(action_class, metrics) if events.is_complete_delivery else BowlerActionClass.UNKNOWN
    cohort_comps = build_cohort_comparisons(metrics, wk)
    wk_guidance = build_workload_guidance(wk)

    kinetic = _retired_pillar(
        "Kinetic & Pace Efficiency",
        metrics,
        ["front_knee_br_deg", "front_knee_change_deg", "trunk_flexion_delta_deg", "delayed_arm_ffc_deg", "runup_speed_mps"],
    )
    durability = _retired_pillar(
        "Mechanical Load & Durability Screening",
        metrics,
        ["thorax_lat_flex_br_deg", "thorax_lat_flex_ffc_deg", "shoulder_counter_rotation_deg", "pelvic_drop_ffc_deg", "rear_hip_bfc_deg"],
    )
    quad = _retired_quadrant()
    strengths, limiters, inconclusive, upgrades = _extract_prioritized_cues(metrics, audit, cal, events)

    return BiomechanicalFactSheet(
        bowler_id=bowler_id,
        bowling_arm=bowling_arm,
        action_class=gated_action,
        pose_backend=pose_backend,
        interval_type=interval_type,
        audit=audit,
        calibration=cal,
        events=events,
        workload_context=wk,
        workload_guidance=wk_guidance,
        metrics=metrics,
        cohort_comparisons=cohort_comps,
        kinetic_efficiency=kinetic,
        lumbar_load_screening=durability,
        repeatability=None,
        quadrant=quad,
        strengths=strengths,
        limiters=limiters,
        inconclusive_observations=inconclusive,
        capture_upgrades=upgrades,
        metadata=metadata or {},
    )
