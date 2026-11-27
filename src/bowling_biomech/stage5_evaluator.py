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
import re
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .schemas import (
    ActionStyleSummary,
    BiomechanicalFactSheet,
    BodyCheckpoint,
    BodyZone,
    BowlerActionClass,
    BowlingArm,
    CannotSeeItem,
    CaptureGrade,
    CaptureQuality,
    CaptureViewKind,
    CheckpointId,
    CheckpointReference,
    CheckpointStatus,
    CheckpointStatusLabel,
    CoachingStory,
    CohortComparison,
    CompareCheckpointRow,
    CompareSideCell,
    CompareStory,
    CorrectiveDrill,
    DeliveryCardSummary,
    DeliveryEvents,
    DrillCard,
    HeadlineStat,
    IntervalType,
    KeyMoment,
    MetricEstimate,
    MomentId,
    ObservabilityGrade,
    PillarScore,
    PitchCalibration,
    PrioritizedCue,
    QuadrantPlacement,
    StoryMetricRow,
    StoryNotice,
    StoryNotices,
    StoryTakeaway,
    SupportingMetricItem,
    UnlockReason,
    VideoAudit,
    ViewClarity,
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


def build_corrective_drills(limiters: List[PrioritizedCue]) -> List[CorrectiveDrill]:
    """Map strictly verified non-straddling limiters to evidence-grounded coaching drills.

    Never prescribes drills from inconclusive (interval-straddling) or suppressed metrics,
    never produces clinical injury diagnoses, and preserves anatomical nuance.
    """
    drill_catalog: Dict[str, CorrectiveDrill] = {
        "front_knee_br_deg": CorrectiveDrill(
            id="drill_front_leg_brace",
            target_limiter_metric="front_knee_br_deg",
            title="Short-Approach Tall-Hip Front-Block Drill",
            cue_focus="Plant firmly on a braced front leg and resist knee collapse from FFC through release.",
            setup_steps=[
                "Start from a controlled 6-step approach at ~75% effort on a flat turf or rubber crease.",
                "Place a flat marker cone 1.1 m ahead of BFC to groove a repeatable stride length without over-striding.",
                "Cue 'tall front hip over a firm front heel' at FFC and hold balance for 1 second after release.",
                "Complement in the gym with eccentric split-squats and single-leg isometric braking holds.",
            ],
            literature_basis="Worthington, King, & Ranson (2013); Portus et al. (2004)",
            caution_note="Gradually build approach speed over 2-3 weeks; stop immediately if anterior knee pain occurs.",
        ),
        "front_knee_change_deg": CorrectiveDrill(
            id="drill_knee_extension_brace",
            target_limiter_metric="front_knee_change_deg",
            title="Eccentric-to-Braced Front-Leg Absorption Progression",
            cue_focus="Eliminate late front-knee buckling between front-foot strike and ball release.",
            setup_steps=[
                "Perform 2 sets of 6 step-in shadow deliveries emphasizing firm quad lock at FFC.",
                "Progress to target bowling from half run-up, checking knee angle stability on 240 fps side-on video.",
            ],
            literature_basis="Worthington, King, & Ranson (2013)",
            caution_note="Avoid sudden volume spikes while re-grooving ground-reaction braking mechanics.",
        ),
        "trunk_flexion_delta_deg": CorrectiveDrill(
            id="drill_trunk_flexion_drive",
            target_limiter_metric="trunk_flexion_delta_deg",
            title="Braced Lead-Leg Chest-Drive Medicine-Ball Throw",
            cue_focus="Drive the chest rapidly forward over a stabilized front leg between FFC and release.",
            setup_steps=[
                "Hold a light 2-3 kg medicine ball overhead in the bowling gather position.",
                "Step forward onto a braced lead leg and snap the trunk forward into a downward target throw.",
                "Transfer the same 'chest drives over front knee' timing into 6-ball half-run-up spells.",
            ],
            literature_basis="Worthington, King, & Ranson (2013)",
            caution_note="Keep medicine-ball weight light (2-3 kg) to prioritize speed and clean posture over load.",
        ),
        "delayed_arm_ffc_deg": CorrectiveDrill(
            id="drill_arm_delay_separation",
            target_limiter_metric="delayed_arm_ffc_deg",
            title="Gather-to-FFC 'Hip Leads Hand' Walk-Through",
            cue_focus="Keep the bowling hand behind the shoulder line at FFC before accelerating into release.",
            setup_steps=[
                "Start from a slow walk-through pausing momentarily at FFC with the bowling arm still below horizontal behind the hip.",
                "Feel the non-bowling arm pull down toward the front hip before the bowling arm whips through overhead.",
                "Film 3 reps side-on at 240 fps to confirm delayed-arm timing before scaling to full pace.",
            ],
            literature_basis="Worthington, King, & Ranson (2013)",
            caution_note="Maintain relaxed shoulders; forcing hyper-delay with shoulder tension reduces arm speed.",
        ),
        "shoulder_counter_rotation_deg": CorrectiveDrill(
            id="drill_scr_alignment",
            target_limiter_metric="shoulder_counter_rotation_deg",
            title="BFC Alignment Corridor & Coherent Hip-Shoulder Step-In",
            cue_focus="Keep back-foot alignment, pelvis, and shoulders in one consistent plane from BFC to FFC.",
            setup_steps=[
                "Lay two flat strips down the bowling crease forming a 0.6 m alignment corridor.",
                "Run in smoothly and land the back foot aligned with your natural gather (side-on or front-on) without rotating the chest away after BFC.",
                "Verify hip-shoulder coherence from a static rear-on (0 deg) tripod clip.",
            ],
            literature_basis="Portus et al. (2004); Alway et al. (2021) (note: SCR is a coaching alignment cue, not a standalone injury diagnosis)",
            caution_note="Make alignment changes gradually with a qualified coach; never force an unnatural action overnight.",
        ),
        "thorax_lat_flex_br_deg": CorrectiveDrill(
            id="drill_lateral_trunk_control",
            target_limiter_metric="thorax_lat_flex_br_deg",
            title="Side-Plank Hip-Abduction & Controlled Release-Plane Drill",
            cue_focus="Build lateral trunk & glute-medius capacity rather than forcing an artificial upright spine.",
            setup_steps=[
                "Perform 3 x 25 s side-planks with slow top-leg abduction and anti-lateral-flexion Pallof presses off-field.",
                "In nets, check that non-bowling arm pull stays close to the ribcage rather than flaring wide off-target.",
                "Cross-check weekly bowling volume against ECB / Cricket Australia workload ceilings.",
            ],
            literature_basis="Bayne et al. (2016) (adolescent cohort); Alway et al. (2021)",
            caution_note="Non-diagnostic screening only: fast bowlers naturally exhibit ~30-45 deg contralateral tilt — never force a 0 deg vertical spine. Manage weekly ball workload.",
        ),
        "pelvic_drop_ffc_deg": CorrectiveDrill(
            id="drill_pelvic_stability",
            target_limiter_metric="pelvic_drop_ffc_deg",
            title="Single-Leg Stance Hip-Lock & Step-Down Control",
            cue_focus="Stabilize the stance-side glute medius so the pelvis stays level through front-foot impact.",
            setup_steps=[
                "Perform slow controlled single-leg step-downs from a 15 cm box keeping both iliac crests level.",
                "Progress to hop-and-stick landings holding a level pelvis for 2 seconds.",
            ],
            literature_basis="Bayne et al. (2016); Glazier (2010)",
            caution_note="Mechanical stress screening only — consult a sports physiotherapist if hip or lower-back pain is present.",
        ),
        "follow_through_drift_deg": CorrectiveDrill(
            id="drill_follow_through_corridor",
            target_limiter_metric="follow_through_drift_deg",
            title="Two-Cone Target-Corridor Follow-Through Gate",
            cue_focus="Drive momentum straight down the pitch on the first two steps after ball release.",
            setup_steps=[
                "Place two flat cones 1.0 m apart, 2.5 m down the pitch on the bowler's delivery line (outside the danger area).",
                "Cue driving the back knee and chest cleanly through the gate on steps 1 and 2 post-release.",
            ],
            literature_basis="Glazier (2010)",
            caution_note="Keep cones flat and safely clear of landing zones to avoid trip hazards.",
        ),
    }

    drills: List[CorrectiveDrill] = []
    seen_ids: set = set()
    for cue in limiters:
        for mk in cue.evidence_metrics:
            d = drill_catalog.get(mk)
            if d is not None and d.id not in seen_ids:
                seen_ids.add(d.id)
                drills.append(d)
    return drills


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
    drills = build_corrective_drills(limiters)

    fs = BiomechanicalFactSheet(
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
        corrective_drills=drills,
        metadata=metadata or {},
    )
    return fs.model_copy(update={"coaching_story": build_coaching_story(fs)})


_CANONICAL_22_KEYS: Tuple[str, ...] = (
    "front_knee_br_deg",
    "front_knee_ffc_deg",
    "front_knee_change_deg",
    "trunk_flexion_delta_deg",
    "trunk_flexion_br_deg",
    "trunk_flexion_ffc_deg",
    "delayed_arm_ffc_deg",
    "thorax_lat_flex_br_deg",
    "thorax_lat_flex_ffc_deg",
    "pelvic_drop_ffc_deg",
    "shoulder_counter_rotation_deg",
    "shoulder_alignment_bfc_deg",
    "pelvis_alignment_bfc_deg",
    "hip_shoulder_sep_bfc_deg",
    "max_hip_shoulder_sep_deg",
    "stride_length_pct_stature",
    "runup_speed_mps",
    "follow_through_drift_deg",
    "rear_hip_bfc_deg",
    "non_bowling_arm_elev_ffc_deg",
    "peak_arm_angular_vel_dps",
    "release_height_m",
)

_PLAIN_METRIC_LABELS: Dict[str, str] = {
    "front_knee_br_deg": "Front-knee straightness at release",
    "front_knee_ffc_deg": "Front-knee angle at front-foot plant",
    "front_knee_change_deg": "Front-knee brace change (plant to release)",
    "trunk_flexion_delta_deg": "Forward chest whip (plant to release)",
    "trunk_flexion_br_deg": "Forward trunk lean at ball release",
    "trunk_flexion_ffc_deg": "Forward trunk lean at front-foot plant",
    "delayed_arm_ffc_deg": "Bowling-arm delay at front-foot plant",
    "thorax_lat_flex_br_deg": "Sideways trunk tilt at ball release",
    "thorax_lat_flex_ffc_deg": "Sideways trunk tilt at front-foot plant",
    "pelvic_drop_ffc_deg": "Hip levelness at front-foot plant",
    "shoulder_counter_rotation_deg": "Backward shoulder twist after back-foot land",
    "shoulder_alignment_bfc_deg": "Shoulder direction at back-foot land",
    "pelvis_alignment_bfc_deg": "Hip direction at back-foot land",
    "hip_shoulder_sep_bfc_deg": "Hip-to-shoulder twist at back-foot land",
    "max_hip_shoulder_sep_deg": "Peak hip-to-shoulder twist",
    "stride_length_pct_stature": "Delivery stride length",
    "runup_speed_mps": "Approach speed into the crease",
    "follow_through_drift_deg": "Follow-through line down the pitch",
    "rear_hip_bfc_deg": "Back-leg hip posture at back-foot land",
    "non_bowling_arm_elev_ffc_deg": "Front-arm height at front-foot plant",
    "peak_arm_angular_vel_dps": "Peak bowling-arm whip speed",
    "release_height_m": "Ball release height",
    "ball_release_speed_kph": "Estimated ball release speed",
}

_METRIC_GROUP_MAP: Dict[str, str] = {
    "front_knee_br_deg": "front_leg_brace",
    "front_knee_ffc_deg": "front_leg_brace",
    "front_knee_change_deg": "front_leg_brace",
    "trunk_flexion_delta_deg": "trunk_whip",
    "trunk_flexion_br_deg": "trunk_whip",
    "trunk_flexion_ffc_deg": "trunk_whip",
    "delayed_arm_ffc_deg": "trunk_whip",
    "thorax_lat_flex_br_deg": "side_bend",
    "thorax_lat_flex_ffc_deg": "side_bend",
    "pelvic_drop_ffc_deg": "side_bend",
    "shoulder_counter_rotation_deg": "shoulder_alignment",
    "shoulder_alignment_bfc_deg": "shoulder_alignment",
    "pelvis_alignment_bfc_deg": "shoulder_alignment",
    "hip_shoulder_sep_bfc_deg": "shoulder_alignment",
    "max_hip_shoulder_sep_deg": "shoulder_alignment",
    "stride_length_pct_stature": "runup_stride",
    "runup_speed_mps": "runup_stride",
    "follow_through_drift_deg": "runup_stride",
    "rear_hip_bfc_deg": "other",
    "non_bowling_arm_elev_ffc_deg": "other",
    "peak_arm_angular_vel_dps": "other",
    "release_height_m": "other",
    "ball_release_speed_kph": "other",
}


def _humanize_coaching_copy(text: str) -> str:
    """Replace biomechanics acronyms with calm, everyday cricket terms."""
    out = text
    replacements = [
        (r"\bBFC\s*->\s*FFC\b", "back-foot land to front-foot plant"),
        (r"\bFFC\s*->\s*BR\b", "front-foot plant to ball release"),
        (r"\bBFC\b", "back-foot landing"),
        (r"\bFFC\b", "front-foot plant"),
        (r"\bUAH\b", "arm-level position"),
        (r"\bBR\b", "ball release"),
        (r"\bSCR\b", "shoulder twist"),
    ]
    for pat, rep in replacements:
        out = re.sub(pat, rep, out)
    return out


def format_metric_display(m: Optional[MetricEstimate]) -> Tuple[Optional[str], Optional[str]]:
    """Format `(value_text, range_text)` cleanly for everyday users while matching numeric-verifier rounding."""
    if m is None or m.suppressed or m.value is None or m.ci_low is None or m.ci_high is None:
        return None, None
    u = m.unit
    if u == "deg":
        v, lo, hi = int(round(m.value)), int(round(m.ci_low)), int(round(m.ci_high))
        return f"{v}°", f"{lo}°–{hi}°"
    if u in ("kph", "km/h"):
        v, lo, hi = int(round(m.value)), int(round(m.ci_low)), int(round(m.ci_high))
        return f"{v} km/h", f"{lo}–{hi} km/h"
    if u == "% stature":
        v, lo, hi = int(round(m.value)), int(round(m.ci_low)), int(round(m.ci_high))
        return f"{v}% of height", f"{lo}%–{hi}%"
    if u == "m":
        return f"{m.value:.2f} m", f"{m.ci_low:.2f}–{m.ci_high:.2f} m"
    if u == "m/s":
        return f"{m.value:.1f} m/s", f"{m.ci_low:.1f}–{m.ci_high:.1f} m/s"
    if u == "deg/s":
        v = int(round(m.value, -1))
        lo = int(round(m.ci_low, -1))
        hi = int(round(m.ci_high, -1))
        return f"{v} °/s", f"{lo}–{hi} °/s"
    return f"{m.value:.1f} {u}", f"{m.ci_low:.1f}–{m.ci_high:.1f} {u}"


def _view_clarity_for(obs: ObservabilityGrade) -> Optional[ViewClarity]:
    if obs == ObservabilityGrade.HIGH:
        return "clear"
    if obs == ObservabilityGrade.MODERATE:
        return "fair"
    if obs == ObservabilityGrade.LOW:
        return "rough"
    return None


def _status_label_for(status: CheckpointStatus, unlock_reason: Optional[UnlockReason]) -> CheckpointStatusLabel:
    if status == "strong":
        return "Strong"
    if status == "on_track":
        return "On Track"
    if status == "watch":
        return "Borderline"
    if status == "focus":
        return "Needs Focus"
    if unlock_reason in ("angle", "frame_rate"):
        return "Needs Better Angle"
    return "Not Visible"


def _classify_unlock_reason(
    m: Optional[MetricEstimate],
    plane_hint: str,
    audit: VideoAudit,
    events: DeliveryEvents,
) -> Tuple[UnlockReason, str, str]:
    """Return `(unlock_reason, reason_plain, how_to_unlock)` with zero leaked numbers or acronyms."""
    if not events.is_complete_delivery:
        return (
            "incomplete_delivery",
            "Clip ended or froze before a clear overhead ball release.",
            "Film one continuous clip from run-up through two follow-through steps after release.",
        )
    raw = (m.suppression_reason or "").lower() if m is not None else ""
    if "timebase" in raw or "freeze" in raw or "repeated frames" in raw or (audit.timebase_unreliable and "fps" in raw):
        return (
            "timing_unreliable",
            "Edited slow-motion or repeated frames prevented accurate timing.",
            "Upload the original unedited camera-roll clip without app slow-mo or freeze effects.",
        )
    if any(w in raw for w in ("cropped", "out of frame", "missing", "visibility", "non-finite")):
        return (
            "out_of_frame",
            "Key body points stepped outside the camera frame.",
            "Step back slightly so head-to-toe and overhead reach stay inside the frame.",
        )
    if "fps" in raw or "high-velocity" in raw:
        return (
            "frame_rate",
            "Normal-speed video moves too fast between frames for this check.",
            "Switch your phone camera to slow-motion (120–240 fps) in bright light.",
        )
    if any(w in raw for w in ("world-grounded translation", "pan", "calibrated mcc", "uncalibrated fov")):
        return (
            "calibration",
            "Needs a steady tripod camera with visible pitch creases or stumps.",
            "Rest the phone on a steady tripod or fence framing the creases and stumps.",
        )
    if plane_hint == "sagittal":
        return (
            "angle",
            "Side-on knee and trunk movement cannot be seen clearly from behind or in front.",
            "Film from side-on (90 deg camera view) perpendicular to the popping crease.",
        )
    if plane_hint == "coronal":
        return (
            "angle",
            "Sideways spine and hip tilt cannot be seen clearly from a side-on angle.",
            "Film from behind the bowler (0 deg camera view) in line with the stumps.",
        )
    if plane_hint == "transverse":
        return (
            "angle",
            "Shoulder and hip twist were too uncertain from this camera angle and distance.",
            "Film from behind the bowler in crisp 120–240 fps slow-motion.",
        )
    return (
        "angle",
        "This camera angle could not measure this movement with a narrow enough range.",
        "Film from a steady side-on or behind-the-bowler tripod setup.",
    )


def _map_capture_view(cal: PitchCalibration) -> Tuple[CaptureViewKind, str]:
    vc = cal.view_class.value
    if vc == "dual_view_fused":
        return "dual_fused", "Dual-camera 3D"
    if vc in ("side_on_right", "side_on_left"):
        return "side_on", "Side-on"
    if vc == "rear_behind_bowler":
        return "rear_on", "Filmed from behind"
    if vc == "front_behind_batter":
        return "front_on", "Filmed from front"
    if vc in ("open_side_45", "broadcast_elevated"):
        return "oblique", "Angled view"
    return "unknown", "Camera angle unclear"


def _format_fps_label(fps: float, timebase_unreliable: bool) -> str:
    f_int = int(round(fps))
    if timebase_unreliable:
        return f"{f_int} fps (edited timing)"
    if fps >= 100.0:
        return f"{f_int} fps slow-mo"
    if fps >= 50.0:
        return f"{f_int} fps"
    return f"{f_int} fps (normal speed)"


def _format_cohort_reference(comp: Optional[CohortComparison]) -> Optional[CheckpointReference]:
    if comp is None:
        return None
    u = comp.unit
    if u == "deg":
        txt = f"Research group average: {int(round(comp.cohort_mean))}° (typical spread ±{int(round(comp.cohort_sd))}°)"
    elif u == "m/s":
        txt = f"Research group average: {comp.cohort_mean:.1f} m/s (typical spread ±{comp.cohort_sd:.1f} m/s)"
    else:
        txt = f"Research group average: {comp.cohort_mean:.1f} {u} (typical spread ±{comp.cohort_sd:.1f} {u})"
    return CheckpointReference(
        text=txt,
        source=comp.source,
        definition_match=comp.definition_match,
    )


def _build_action_style(action_class: BowlerActionClass) -> Optional[ActionStyleSummary]:
    if action_class == BowlerActionClass.UNKNOWN:
        return None
    mapping = {
        BowlerActionClass.SIDE_ON: (
            "Side-on action",
            "Back foot, hips, and shoulders enter the crease aligned sideways toward the target.",
        ),
        BowlerActionClass.FRONT_ON: (
            "Front-on action",
            "Hips and chest enter the crease facing open toward the batter in one matched plane.",
        ),
        BowlerActionClass.SEMI_OPEN: (
            "Semi-open action",
            "Hips and shoulders align on a balanced diagonal between side-on and front-on.",
        ),
        BowlerActionClass.SEMI_SIDE_ON: (
            "Semi-open action",
            "Hips and shoulders align on a balanced diagonal between side-on and front-on.",
        ),
        BowlerActionClass.SEMI_FRONT_ON: (
            "Semi-open action",
            "Hips and shoulders align on a balanced diagonal between side-on and front-on.",
        ),
        BowlerActionClass.MIXED: (
            "Mixed action",
            "Shoulders twist back away from the batter after back-foot landing while the hips stay more open — a coaching alignment pattern, not an injury diagnosis.",
        ),
    }
    lbl, desc = mapping.get(action_class, ("Unclassified", ""))
    if not desc:
        return None
    return ActionStyleSummary(code=action_class.value, label=lbl, plain_summary=desc)


def build_coaching_story(
    fs: BiomechanicalFactSheet,
    delivery_id: str = "",
    is_sample: bool = False,
    sample_kind: Optional[str] = None,
) -> CoachingStory:
    """Deterministically compile a jargon-free `CoachingStory` strictly from interval-verified FactSheet data."""
    metrics = fs.metrics
    ev = fs.events
    audit = fs.audit
    cal = fs.calibration
    is_right = fs.bowling_arm == BowlingArm.RIGHT

    # Resolve arm-aware SMPL-24 joint names for visual highlights and angle arcs
    f_hip = "lhip" if is_right else "rhip"
    f_kne = "lkne" if is_right else "rkne"
    f_ank = "lank" if is_right else "rank"
    f_toe = "ltoe" if is_right else "rtoe"
    r_ank = "rank" if is_right else "lank"
    r_toe = "rtoe" if is_right else "ltoe"
    b_sho = "rsho" if is_right else "lsho"
    b_elb = "relb" if is_right else "lelb"

    limiter_keys: Set[str] = {mk for c in fs.limiters for mk in c.evidence_metrics}
    watch_keys: Set[str] = {mk for c in fs.inconclusive_observations for mk in c.evidence_metrics}
    strength_keys: Set[str] = {mk for c in fs.strengths for mk in c.evidence_metrics}
    cohort_by_key: Dict[str, CohortComparison] = {c.metric_key: c for c in fs.cohort_comparisons}

    def _metric_status(mk: str) -> CheckpointStatus:
        m = metrics.get(mk)
        if (not ev.is_complete_delivery) or m is None or m.suppressed or m.value is None:
            return "not_visible"
        if mk in limiter_keys:
            return "focus"
        if mk in watch_keys:
            return "watch"
        if mk in strength_keys:
            return "strong"
        return "on_track"

    def _checkpoint_status(primary_k: str, cue_keys: Set[str]) -> CheckpointStatus:
        pm = metrics.get(primary_k)
        if (not ev.is_complete_delivery) or pm is None or pm.suppressed or pm.value is None:
            return "not_visible"
        active_cues = {
            k for k in cue_keys
            if k in metrics and (not metrics[k].suppressed) and metrics[k].value is not None
        }
        if active_cues & limiter_keys:
            return "focus"
        if active_cues & watch_keys:
            return "watch"
        if active_cues & strength_keys:
            return "strong"
        return "on_track"

    def _supporting_items(keys: List[str]) -> List[SupportingMetricItem]:
        items: List[SupportingMetricItem] = []
        for sk in keys:
            sm = metrics.get(sk)
            if sm is None:
                continue
            vt, rt = format_metric_display(sm)
            items.append(
                SupportingMetricItem(
                    metric_key=sk,
                    label_plain=_PLAIN_METRIC_LABELS.get(sk, sm.label),
                    value_text=vt,
                    range_text=rt,
                    suppressed=bool(sm.suppressed or sm.value is None or (not ev.is_complete_delivery)),
                )
            )
        return items

    bfc_idx = int(ev.bfc.frame_idx) if ev.is_complete_delivery else None
    ffc_idx = int(ev.ffc.frame_idx) if ev.is_complete_delivery else None
    br_idx = int(ev.br.frame_idx) if ev.is_complete_delivery else None

    # Resolve Checkpoint 5 primary key fallback (`stride_length_pct_stature` -> `runup_speed_mps` if stride suppressed & speed measured)
    stride_m = metrics.get("stride_length_pct_stature")
    speed_m = metrics.get("runup_speed_mps")
    if (
        ev.is_complete_delivery
        and (stride_m is None or stride_m.suppressed or stride_m.value is None)
        and speed_m is not None
        and (not speed_m.suppressed)
        and speed_m.value is not None
    ):
        cp5_primary = "runup_speed_mps"
        cp5_supporting = ["stride_length_pct_stature", "follow_through_drift_deg"]
    else:
        cp5_primary = "stride_length_pct_stature"
        cp5_supporting = ["runup_speed_mps", "follow_through_drift_deg"]

    cp_specs: List[Tuple[CheckpointId, BodyZone, str, Optional[MomentId], str, Optional[int], str, Set[str], str, List[str], Optional[Tuple[str, str, str]], List[str], str]] = [
        (
            "front_leg_brace",
            "knee",
            "Front-Leg Brace",
            "ball_release" if ev.is_complete_delivery else None,
            "At ball release",
            br_idx,
            "front_knee_br_deg",
            {"front_knee_br_deg", "front_knee_change_deg"},
            "sagittal",
            [f_hip, f_kne, f_ank],
            (f_hip, f_kne, f_ank),
            ["front_knee_ffc_deg", "front_knee_change_deg"],
            "A firm front leg acts as a brake so run-up momentum whips up into your chest and bowling arm.",
        ),
        (
            "trunk_whip",
            "trunk",
            "Trunk Whip",
            "ball_release" if ev.is_complete_delivery else None,
            "Front foot plants → ball release",
            br_idx,
            "trunk_flexion_delta_deg",
            {"trunk_flexion_delta_deg", "delayed_arm_ffc_deg"},
            "sagittal",
            ["pelv", "spi2", "neck", b_sho, b_elb],
            ("neck", "pelv", f_hip),
            ["trunk_flexion_br_deg", "delayed_arm_ffc_deg"],
            "Driving your chest forward over a planted front foot transfers momentum into release speed.",
        ),
        (
            "side_bend",
            "spine",
            "Side-Bend Load",
            "ball_release" if ev.is_complete_delivery else None,
            "At ball release",
            br_idx,
            "thorax_lat_flex_br_deg",
            {"thorax_lat_flex_br_deg", "pelvic_drop_ffc_deg"},
            "coronal",
            ["lhip", "rhip", "pelv", "spi2", "neck", "head"],
            ("neck", "pelv", f_hip),
            ["pelvic_drop_ffc_deg"],
            "Fast bowlers naturally tilt sideways to get a high release, while excessive tilt with high weekly bowling volume increases trunk load.",
        ),
        (
            "shoulder_alignment",
            "shoulders",
            "Shoulder Alignment",
            "back_foot_land" if ev.is_complete_delivery else None,
            "Back foot → front foot",
            bfc_idx,
            "shoulder_counter_rotation_deg",
            {"shoulder_counter_rotation_deg"},
            "transverse",
            ["lsho", "rsho", "lhip", "rhip"],
            None,
            ["shoulder_alignment_bfc_deg", "pelvis_alignment_bfc_deg", "hip_shoulder_sep_bfc_deg", "max_hip_shoulder_sep_deg"],
            "Keeping hips and shoulders moving in a matched plane avoids abrupt upper-body twist during the delivery stride.",
        ),
        (
            "runup_stride",
            "approach",
            "Run-Up & Stride",
            "front_foot_plant" if ev.is_complete_delivery else None,
            "At front foot plant",
            ffc_idx,
            cp5_primary,
            {"stride_length_pct_stature", "runup_speed_mps", "follow_through_drift_deg"},
            "metric_scale",
            [r_ank, f_ank, "pelv", r_toe, f_toe],
            None,
            cp5_supporting,
            "Smooth approach speed paired with a balanced stride lets your front foot plant without sinking or over-reaching.",
        ),
    ]

    checkpoints: List[BodyCheckpoint] = []
    cannot_see: List[CannotSeeItem] = []
    can_see: List[CheckpointId] = []

    for cid, zone, title, mom_id, mom_lbl, f_idx, p_key, cue_set, plane_hint, hl_joints, ang_joints, sup_keys, why_txt in cp_specs:
        pm = metrics.get(p_key)
        st = _checkpoint_status(p_key, cue_set)
        if st == "not_visible" or pm is None or pm.value is None or pm.ci_low is None or pm.ci_high is None:
            ureason, uplain, how_txt = _classify_unlock_reason(pm, plane_hint, audit, ev)
            s_lbl = _status_label_for("not_visible", ureason)
            cannot_see.append(CannotSeeItem(checkpoint_id=cid, reason=ureason, reason_plain=uplain))
            checkpoints.append(
                BodyCheckpoint(
                    id=cid,
                    body_zone=zone,
                    title=title,
                    moment_id=mom_id,
                    moment_label=mom_lbl,
                    frame_idx=f_idx,
                    status="not_visible",
                    status_label=s_lbl,
                    unlock_reason=ureason,
                    metric_key=p_key,
                    value=None,
                    ci_low=None,
                    ci_high=None,
                    unit=pm.unit if pm is not None else "deg",
                    view_clarity=None,
                    value_text=None,
                    range_text=None,
                    reference=None,
                    plain_meaning=f"Withheld on this clip: {uplain}",
                    why_it_matters=why_txt,
                    how_to_unlock=how_txt,
                    highlight_joints=hl_joints,
                    angle_joints=ang_joints,
                    supporting=_supporting_items(sup_keys),
                )
            )
            continue

        can_see.append(cid)
        vt, rt = format_metric_display(pm)
        s_lbl = _status_label_for(st, None)
        ref_obj = _format_cohort_reference(cohort_by_key.get(p_key))
        plain_m = _build_checkpoint_plain_meaning(cid, st, vt or "", rt or "")
        checkpoints.append(
            BodyCheckpoint(
                id=cid,
                body_zone=zone,
                title=title,
                moment_id=mom_id,
                moment_label=mom_lbl,
                frame_idx=f_idx,
                status=st,
                status_label=s_lbl,
                unlock_reason=None,
                metric_key=p_key,
                value=pm.value,
                ci_low=pm.ci_low,
                ci_high=pm.ci_high,
                unit=pm.unit,
                view_clarity=_view_clarity_for(pm.observability),
                value_text=vt,
                range_text=rt,
                reference=ref_obj,
                plain_meaning=plain_m,
                why_it_matters=why_txt,
                how_to_unlock=None,
                highlight_joints=hl_joints,
                angle_joints=ang_joints,
                supporting=_supporting_items(sup_keys),
            )
        )

    # Capture quality summary
    measured_22 = sum(
        1
        for k in _CANONICAL_22_KEYS
        if ev.is_complete_delivery and k in metrics and (not metrics[k].suppressed) and metrics[k].value is not None
    )
    view_kind, view_lbl = _map_capture_view(cal)
    fps_lbl = _format_fps_label(audit.fps, audit.timebase_unreliable)
    if not ev.is_complete_delivery:
        cap_grade: CaptureGrade = "unusable"
        cap_lbl = "Couldn't analyse"
        cap_sum = "No complete overhead bowling release was visible, so body measurements were safely withheld."
    elif audit.timebase_unreliable or measured_22 <= 5:
        cap_grade = "limited"
        cap_lbl = "Limited view"
        cap_sum = (
            f"{view_lbl} clip measured {measured_22} of 22 checks, while timing or framing limits hid the rest."
        )
    elif measured_22 >= 14 and audit.fps >= 100.0:
        cap_grade = "great"
        cap_lbl = "Great view"
        if cannot_see:
            cap_sum = f"Crisp {fps_lbl} {view_lbl.lower()} view measured {measured_22} of 22 checks clearly."
        else:
            cap_sum = f"Crisp {fps_lbl} {view_lbl.lower()} view unlocked all 5 body checkpoints."
    else:
        cap_grade = "partial"
        cap_lbl = "Partial view"
        seen_names = ", ".join(cp.title.lower() for cp in checkpoints if cp.status != "not_visible") or "limited checks"
        cap_sum = f"{view_lbl} view captured {seen_names}, while out-of-plane checks need another camera angle."

    next_tip: Optional[str] = None
    if fs.capture_upgrades:
        next_tip = _humanize_coaching_copy(fs.capture_upgrades[0].actionable_recommendation)

    capture = CaptureQuality(
        grade=cap_grade,
        label=cap_lbl,
        summary=cap_sum,
        view=view_kind,
        view_label=view_lbl,
        camera_azimuth_deg=round(float(cal.azimuth_deg), 1) if ev.is_complete_delivery else None,
        fps=round(float(audit.fps), 1),
        fps_label=fps_lbl,
        measured_count=measured_22,
        total_count=22,
        can_see=can_see,
        cannot_see=cannot_see,
        next_clip_tip=next_tip,
    )

    # Drills (mapped strictly 1:1 from interval-cleared `corrective_drills`)
    drills: List[DrillCard] = []
    cp_by_id: Dict[CheckpointId, BodyCheckpoint] = {cp.id: cp for cp in checkpoints}
    for dr in fs.corrective_drills[:2]:
        grp = _METRIC_GROUP_MAP.get(dr.target_limiter_metric, "other")
        for_cp: Optional[CheckpointId] = grp if grp != "other" else None  # type: ignore[assignment]
        cp_match = cp_by_id.get(for_cp) if for_cp else None
        why_txt = cp_match.plain_meaning if cp_match else "Targeted drill for a confirmed movement limiter."
        drills.append(
            DrillCard(
                id=dr.id,
                title=_humanize_coaching_copy(dr.title),
                for_checkpoint_id=for_cp,
                why_plain=why_txt,
                cue=_humanize_coaching_copy(dr.cue_focus),
                steps=[_humanize_coaching_copy(s) for s in dr.setup_steps],
                caution=_humanize_coaching_copy(dr.caution_note),
                source=dr.literature_basis,
                moment_id=cp_match.moment_id if cp_match else None,
                frame_idx=cp_match.frame_idx if cp_match else None,
            )
        )

    # Top Takeaway & Headline (strictly following locked priority: focus -> watch -> capture -> strong -> all_clear)
    headline, takeaway = _build_headline_and_takeaway(checkpoints, drills, capture, ev)

    # Key Moments (`[]` when incomplete delivery)
    key_moments: List[KeyMoment] = []
    if ev.is_complete_delivery:
        thumb_prefix = f"/api/deliveries/{delivery_id}/frame" if delivery_id else None
        key_moments.append(
            KeyMoment(
                id="back_foot_land",
                label="Back foot lands",
                frame_idx=int(ev.bfc.frame_idx),
                source_frame_idx=int(ev.bfc.source_frame_idx if ev.bfc.source_frame_idx is not None else ev.bfc.frame_idx),
                time_s=round(float(ev.bfc.timestamp_s), 3),
                timing_uncertainty_ms=round(float(ev.bfc.timing_uncertainty_ms), 1),
                plain_tip="Back foot lands — do hips and shoulders stay in one matched plane?",
                what_to_look_for="Tall posture over the back leg with hips and shoulders aligned smoothly.",
                checkpoint_ids=["shoulder_alignment"],
                thumbnail_url=f"{thumb_prefix}/{int(ev.bfc.frame_idx)}.jpg?w=480" if thumb_prefix else None,
            )
        )
        key_moments.append(
            KeyMoment(
                id="front_foot_plant",
                label="Front foot plants",
                frame_idx=int(ev.ffc.frame_idx),
                source_frame_idx=int(ev.ffc.source_frame_idx if ev.ffc.source_frame_idx is not None else ev.ffc.frame_idx),
                time_s=round(float(ev.ffc.timestamp_s), 3),
                timing_uncertainty_ms=round(float(ev.ffc.timing_uncertainty_ms), 1),
                plain_tip="Front foot plants — firm heel strike with bowling hand still delayed behind the hip.",
                what_to_look_for="Balanced stride length, level hips, and firm lead-leg landing.",
                checkpoint_ids=["runup_stride"],
                thumbnail_url=f"{thumb_prefix}/{int(ev.ffc.frame_idx)}.jpg?w=480" if thumb_prefix else None,
            )
        )
        if ev.uah is not None:
            key_moments.append(
                KeyMoment(
                    id="arm_horizontal",
                    label="Arm level",
                    frame_idx=int(ev.uah.frame_idx),
                    source_frame_idx=int(ev.uah.source_frame_idx if ev.uah.source_frame_idx is not None else ev.uah.frame_idx),
                    time_s=round(float(ev.uah.timestamp_s), 3),
                    timing_uncertainty_ms=round(float(ev.uah.timing_uncertainty_ms), 1),
                    plain_tip="Arm reaches shoulder height — chest begins whipping forward over the front knee.",
                    what_to_look_for="Front leg stays firm while the chest drives toward the batter.",
                    checkpoint_ids=["trunk_whip"],
                    thumbnail_url=f"{thumb_prefix}/{int(ev.uah.frame_idx)}.jpg?w=480" if thumb_prefix else None,
                )
            )
        key_moments.append(
            KeyMoment(
                id="ball_release",
                label="Ball release",
                frame_idx=int(ev.br.frame_idx),
                source_frame_idx=int(ev.br.source_frame_idx if ev.br.source_frame_idx is not None else ev.br.frame_idx),
                time_s=round(float(ev.br.timestamp_s), 3),
                timing_uncertainty_ms=round(float(ev.br.timing_uncertainty_ms), 1),
                plain_tip="Ball release — tall release over a braced front leg with controlled side-bend.",
                what_to_look_for="Straightened or firm front knee and controlled sideways trunk tilt.",
                checkpoint_ids=["front_leg_brace", "trunk_whip", "side_bend"],
                thumbnail_url=f"{thumb_prefix}/{int(ev.br.frame_idx)}.jpg?w=480" if thumb_prefix else None,
            )
        )

    # Headline stats (ball speed & release height ONLY when measured & non-suppressed)
    headline_stats: List[HeadlineStat] = []
    for hk, hlbl in [
        ("ball_release_speed_kph", "Release speed"),
        ("release_height_m", "Release height"),
    ]:
        hm = metrics.get(hk)
        if ev.is_complete_delivery and hm is not None and (not hm.suppressed) and hm.value is not None:
            hvt, hrt = format_metric_display(hm)
            hcl = _view_clarity_for(hm.observability) or "rough"
            if hvt and hrt:
                headline_stats.append(
                    HeadlineStat(
                        metric_key=hk,
                        label=hlbl,
                        value_text=hvt,
                        range_text=hrt,
                        view_clarity=hcl,
                    )
                )

    # All-22 Detail Rows (plus ball speed if present)
    detail_keys = list(_CANONICAL_22_KEYS)
    if "ball_release_speed_kph" in metrics:
        detail_keys.append("ball_release_speed_kph")
    detail_rows: List[StoryMetricRow] = []
    for dk in detail_keys:
        dm = metrics.get(dk)
        if dm is None:
            continue
        grp_name = _METRIC_GROUP_MAP.get(dk, "other")
        dst = _metric_status(dk)
        if dst == "not_visible":
            plane_h = "coronal" if grp_name == "side_bend" else ("transverse" if grp_name == "shoulder_alignment" else ("metric_scale" if grp_name == "runup_stride" else "sagittal"))
            ur, u_plain, _ = _classify_unlock_reason(dm, plane_h, audit, ev)
            dsl = _status_label_for("not_visible", ur)
            detail_rows.append(
                StoryMetricRow(
                    metric_key=dk,
                    group=grp_name,  # type: ignore[arg-type]
                    label_plain=_PLAIN_METRIC_LABELS.get(dk, dm.label),
                    value_text=None,
                    range_text=None,
                    view_clarity=None,
                    status="not_visible",
                    status_label=dsl,
                    suppressed=True,
                    suppression_plain=u_plain,
                    plain_note=u_plain,
                    definition_note=_humanize_coaching_copy(dm.definition_note),
                    literature_source=dm.literature_source,
                )
            )
        else:
            dvt, drt = format_metric_display(dm)
            dsl = _status_label_for(dst, None)
            note_map = {
                "strong": f"Measured at {dvt} (likely {drt}) — clear strength on this delivery.",
                "on_track": f"Measured at {dvt} (likely {drt}) — within normal fast-bowling range.",
                "watch": f"Measured at {dvt} (likely {drt}) — range crosses the coaching threshold, so treat as borderline.",
                "focus": f"Measured at {dvt} (likely {drt}) — entire range clears the coaching threshold.",
            }
            detail_rows.append(
                StoryMetricRow(
                    metric_key=dk,
                    group=grp_name,  # type: ignore[arg-type]
                    label_plain=_PLAIN_METRIC_LABELS.get(dk, dm.label),
                    value_text=dvt,
                    range_text=drt,
                    view_clarity=_view_clarity_for(dm.observability),
                    status=dst,
                    status_label=dsl,
                    suppressed=False,
                    suppression_plain=None,
                    plain_note=note_map.get(dst, f"Measured at {dvt} (likely {drt})."),
                    definition_note=_humanize_coaching_copy(dm.definition_note),
                    literature_source=dm.literature_source,
                )
            )

    # Permanent non-truncating notices + extra plain-English capture caveats
    extra_notices: List[StoryNotice] = []
    if audit.timebase_unreliable:
        extra_notices.append(
            StoryNotice(
                id="timing_caveat",
                title="Edited or repeated video frames detected",
                body="This clip contains repeated or retimed frames, so speed-sensitive checks were safely withheld. Export original camera-roll slow-motion for full timing accuracy.",
            )
        )
    if cal.is_panning_camera and ev.is_complete_delivery:
        extra_notices.append(
            StoryNotice(
                id="panning_caveat",
                title="Handheld camera movement detected",
                body="Because the camera panned during the run-up, ground-travel checks (approach speed, stride length, and follow-through line) were withheld rather than guessed.",
            )
        )

    notices = StoryNotices(
        icc=StoryNotice(
            id="icc",
            title="Elbow legality (ICC 15° rule) — never judged from video",
            body=(
                "This app permanently refuses to give a verdict on elbow straightening against the ICC 15° limit, "
                "because accredited testing requires high-speed (>=250 Hz) 3D laboratory motion capture that phone "
                "or multi-camera video cannot provide."
            ),
        ),
        medical=StoryNotice(
            id="medical",
            title="Not medical advice",
            body=(
                "These are movement measurements from a single delivery, not a clinical injury diagnosis. "
                "Injury risk also depends strongly on weekly bowling workload, age and skeletal growth, "
                "recovery, and physical conditioning. Speak to a qualified physiotherapist or doctor if pain is present."
            ),
        ),
        extra=extra_notices,
    )

    arm_lbl = "Right-arm" if is_right else "Left-arm"
    subhead = f"{arm_lbl} · {view_lbl} · {fps_lbl}"
    s_kind: Optional[str] = sample_kind if sample_kind in ("real", "synthetic") else None

    return CoachingStory(
        schema_version="coaching_story/1",
        delivery_id=delivery_id,
        headline=headline[:120],
        subhead=subhead,
        is_complete_delivery=bool(ev.is_complete_delivery),
        is_sample=bool(is_sample),
        sample_kind=s_kind,  # type: ignore[arg-type]
        action_style=_build_action_style(fs.action_class),
        capture=capture,
        top_takeaway=takeaway,
        checkpoints=checkpoints,
        key_moments=key_moments,
        headline_stats=headline_stats,
        drills=drills,
        detail_rows=detail_rows,
        notices=notices,
    )


def _build_checkpoint_plain_meaning(cid: CheckpointId, status: CheckpointStatus, vt: str, rt: str) -> str:
    if cid == "front_leg_brace":
        if status == "strong":
            return f"Your front knee stays firm at {vt} (likely {rt}) at ball release, giving a solid block to whip over."
        if status == "focus":
            return f"Your front knee bends to {vt} (likely {rt}) by ball release, softening the block as you let the ball go."
        if status == "watch":
            return f"Your front knee measured {vt} (likely {rt}) — the uncertainty range crosses the yielding line, so film in crisp slow-mo before changing technique."
        return f"Your front knee measured {vt} (likely {rt}) at ball release — steady braking support."
    if cid == "trunk_whip":
        if status == "strong":
            return f"Your chest whips forward by {vt} (likely {rt}) after front-foot plant, converting run-up momentum cleanly."
        if status == "focus":
            return f"Forward chest drive measured {vt} (likely {rt}) after front-foot plant, leaving extra whip on the table."
        if status == "watch":
            return f"Forward chest drive measured {vt} (likely {rt}) — the range crosses the coaching threshold, so confirm on slow-mo video."
        return f"Your chest drives forward by {vt} (likely {rt}) from front-foot plant to ball release."
    if cid == "side_bend":
        if status == "focus":
            return f"Sideways trunk tilt reached {vt} (likely {rt}) at release — build lateral trunk strength and monitor weekly ball counts rather than forcing a rigid upright spine."
        if status == "watch":
            return f"Sideways trunk tilt measured {vt} (likely {rt}) — borderline against adolescent screening norms, so track alongside weekly workload."
        return f"Sideways trunk tilt at release is {vt} (likely {rt}), within typical fast-bowling posture."
    if cid == "shoulder_alignment":
        if status == "focus":
            return f"Your shoulders twist back by {vt} (likely {rt}) after back-foot landing — keeping hips and shoulders aligned reduces abrupt trunk recoil."
        if status == "watch":
            return f"Shoulder twist measured {vt} (likely {rt}) — the range straddles the mixed-action line, so film at 120–240 fps to confirm before adjusting your action."
        if status == "strong":
            return f"Your shoulders twist back by only {vt} (likely {rt}), keeping hips and chest moving smoothly together."
        return f"Backward shoulder twist after back-foot landing measured {vt} (likely {rt})."
    # runup_stride
    return f"Measured at {vt} (likely {rt}) entering front-foot plant."


def _build_headline_and_takeaway(
    checkpoints: List[BodyCheckpoint],
    drills: List[DrillCard],
    capture: CaptureQuality,
    events: DeliveryEvents,
) -> Tuple[str, StoryTakeaway]:
    if not events.is_complete_delivery:
        return (
            "We couldn't find a complete bowling action in this clip.",
            StoryTakeaway(
                kind="capture",
                title="Film the full delivery through follow-through",
                plain_explanation=(
                    "This video ended, froze, or cut away before an overhead ball release could be confirmed, "
                    "so body measurements were safely withheld."
                ),
                what_to_try="Record one unedited clip starting three steps before the crease through follow-through.",
                checkpoint_id=None,
                moment_id=None,
                frame_idx=None,
                linked_drill_id=None,
            ),
        )

    focus_cps = [c for c in checkpoints if c.status == "focus"]
    watch_cps = [c for c in checkpoints if c.status == "watch"]
    strong_cps = [c for c in checkpoints if c.status == "strong"]
    on_track_cps = [c for c in checkpoints if c.status == "on_track"]
    drill_by_cp = {d.for_checkpoint_id: d for d in drills if d.for_checkpoint_id is not None}

    if focus_cps:
        fc = focus_cps[0]
        dr = drill_by_cp.get(fc.id) or (drills[0] if drills else None)
        if strong_cps:
            hl = f"Strong {strong_cps[0].title.lower()}, with {fc.title.lower()} as your main focus."
        else:
            hl = f"{fc.title} is your clearest area to work on from this ball."
        return (
            hl,
            StoryTakeaway(
                kind="focus",
                title=f"Focus on {fc.title.lower()}",
                plain_explanation=fc.plain_meaning,
                what_to_try=dr.cue if dr is not None else "Work gradually on controlled posture from a shorter run-up.",
                checkpoint_id=fc.id,
                moment_id=fc.moment_id,
                frame_idx=fc.frame_idx,
                linked_drill_id=dr.id if dr is not None else None,
            ),
        )

    if watch_cps:
        wc = watch_cps[0]
        good = strong_cps[0] if strong_cps else (on_track_cps[0] if on_track_cps else None)
        if good is not None:
            hl = f"{good.title} is steady — {wc.title.lower()} is borderline on this clip."
        else:
            hl = f"{wc.title} sits near the coaching line and needs a clearer clip."
        return (
            hl,
            StoryTakeaway(
                kind="watch",
                title=f"{wc.title} is borderline",
                plain_explanation=wc.plain_meaning,
                what_to_try=capture.next_clip_tip or "Film in 120–240 fps slow-motion from a steady tripod before changing your action.",
                checkpoint_id=wc.id,
                moment_id=wc.moment_id,
                frame_idx=wc.frame_idx,
                linked_drill_id=None,
            ),
        )

    angle_blocked = [item for item in capture.cannot_see if item.reason in ("angle", "frame_rate")]
    if (capture.measured_count / max(capture.total_count, 1) < 0.4) or (angle_blocked and not strong_cps):
        hl = (
            f"{on_track_cps[0].title} looks steady — add another camera angle to see the full picture."
            if on_track_cps
            else "This camera setup only captured part of your bowling action."
        )
        return (
            hl,
            StoryTakeaway(
                kind="capture",
                title="Unlock your remaining body checkpoints",
                plain_explanation=capture.summary,
                what_to_try=capture.next_clip_tip or "Film from side-on or behind on a steady tripod in 120–240 fps slow-motion.",
                checkpoint_id=angle_blocked[0].checkpoint_id if angle_blocked else None,
                moment_id=None,
                frame_idx=None,
                linked_drill_id=None,
            ),
        )

    if strong_cps:
        sc = strong_cps[0]
        if len(strong_cps) >= 2:
            hl = f"Strong {sc.title.lower()} and {strong_cps[1].title.lower()} through ball release."
        else:
            hl = f"Strong {sc.title.lower()} with clean mechanics across visible checks."
        return (
            hl,
            StoryTakeaway(
                kind="strength",
                title=f"{sc.title} is a clear strength",
                plain_explanation=sc.plain_meaning,
                what_to_try=capture.next_clip_tip or "Keep this repeatable rhythm across a full multi-ball spell.",
                checkpoint_id=sc.id,
                moment_id=sc.moment_id,
                frame_idx=sc.frame_idx,
                linked_drill_id=None,
            ),
        )

    return (
        "All visible body checkpoints are on track in this delivery.",
        StoryTakeaway(
            kind="all_clear",
            title="Steady posture across visible checks",
            plain_explanation="None of the visible body checkpoints crossed a coaching limiter threshold.",
            what_to_try=capture.next_clip_tip or "Compare another ball from your spell to check repeatability.",
            checkpoint_id=on_track_cps[0].id if on_track_cps else None,
            moment_id=on_track_cps[0].moment_id if on_track_cps else None,
            frame_idx=on_track_cps[0].frame_idx if on_track_cps else None,
            linked_drill_id=None,
        ),
    )


def hydrate_coaching_story(
    fs: BiomechanicalFactSheet,
    delivery_id: str,
    is_sample: bool = False,
    sample_kind: Optional[str] = None,
) -> CoachingStory:
    """Ensure `CoachingStory` carries runtime `delivery_id`, `thumbnail_url`s, and sample flags."""
    return build_coaching_story(
        fs,
        delivery_id=delivery_id,
        is_sample=is_sample,
        sample_kind=sample_kind,
    )


def build_compare_story(
    card_a: DeliveryCardSummary,
    fs_a: BiomechanicalFactSheet,
    story_a: CoachingStory,
    card_b: DeliveryCardSummary,
    fs_b: BiomechanicalFactSheet,
    story_b: CoachingStory,
) -> CompareStory:
    """Compare two deliveries checkpoint-by-checkpoint using strict non-overlapping uncertainty intervals."""
    sync_a = int(fs_a.events.ffc.frame_idx) if fs_a.events.is_complete_delivery else None
    sync_b = int(fs_b.events.ffc.frame_idx) if fs_b.events.is_complete_delivery else None

    cp_map_a = {c.id: c for c in story_a.checkpoints}
    cp_map_b = {c.id: c for c in story_b.checkpoints}
    rows: List[CompareCheckpointRow] = []
    comparable_count = 0
    clear_changes: List[str] = []

    for cid in ("front_leg_brace", "trunk_whip", "side_bend", "shoulder_alignment", "runup_stride"):
        ca = cp_map_a[cid]
        cb = cp_map_b[cid]
        cell_a = CompareSideCell(status=ca.status, status_label=ca.status_label, value_text=ca.value_text, range_text=ca.range_text)
        cell_b = CompareSideCell(status=cb.status, status_label=cb.status_label, value_text=cb.value_text, range_text=cb.range_text)

        if (
            ca.status == "not_visible"
            or cb.status == "not_visible"
            or ca.ci_low is None
            or ca.ci_high is None
            or cb.ci_low is None
            or cb.ci_high is None
            or ca.metric_key != cb.metric_key
        ):
            rows.append(
                CompareCheckpointRow(
                    checkpoint_id=cid,  # type: ignore[arg-type]
                    title=ca.title,
                    a=cell_a,
                    b=cell_b,
                    difference="not_comparable",
                    difference_plain="Can't compare (not visible in both clips)",
                )
            )
            continue

        comparable_count += 1
        if cb.ci_low > ca.ci_high:
            diff_kind = "clearly_higher"
            diff_txt = f"Clearly higher in second ball ({ca.value_text} → {cb.value_text})"
            clear_changes.append(f"{ca.title.lower()} increased ({ca.value_text} → {cb.value_text})")
        elif cb.ci_high < ca.ci_low:
            diff_kind = "clearly_lower"
            diff_txt = f"Clearly lower in second ball ({ca.value_text} → {cb.value_text})"
            clear_changes.append(f"{ca.title.lower()} decreased ({ca.value_text} → {cb.value_text})")
        else:
            diff_kind = "within_noise"
            diff_txt = "Within measurement noise (ranges overlap)"

        rows.append(
            CompareCheckpointRow(
                checkpoint_id=cid,  # type: ignore[arg-type]
                title=ca.title,
                a=cell_a,
                b=cell_b,
                difference=diff_kind,  # type: ignore[arg-type]
                difference_plain=diff_txt,
            )
        )

    caveats: List[str] = []
    if not fs_a.events.is_complete_delivery or not fs_b.events.is_complete_delivery:
        caveats.append("At least one clip did not contain a complete overhead release.")
    elif story_a.capture.view != story_b.capture.view:
        caveats.append(
            f"First clip was {story_a.capture.view_label.lower()} while second clip was {story_b.capture.view_label.lower()}, "
            f"so {comparable_count} of 5 checkpoints could be compared directly."
        )
    elif abs(story_a.capture.fps - story_b.capture.fps) >= 45.0:
        caveats.append(
            f"Frame rates differ ({story_a.capture.fps_label} vs {story_b.capture.fps_label}), widening uncertainty on the slower clip."
        )

    if comparable_count == 0:
        summary = "These two clips do not share any visible body checkpoints from matching camera angles."
    elif clear_changes:
        summary = "Clear shift beyond measurement ranges: " + "; ".join(clear_changes) + "."
    else:
        summary = f"Across {comparable_count} comparable checkpoints, differences stay within measurement ranges."

    return CompareStory(
        a=card_a,
        b=card_b,
        sync_moment="front_foot_plant",
        sync_frames={"a": sync_a, "b": sync_b},
        comparable=comparable_count > 0,
        summary=summary,
        caveat_plain=" ".join(caveats) if caveats else None,
        rows=rows,
    )
