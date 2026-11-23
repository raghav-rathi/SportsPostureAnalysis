"""Stage 6B: Strictly Measurement-Grounded Coaching Report Synthesizer.

Enforces the neuro-symbolic boundary: every numeric claim traces directly to a non-suppressed
MetricEstimate in BiomechanicalFactSheet, reports exact interval semantics (`repeatability_interval`
vs `analytical_uncertainty` — never claiming uncalibrated 95% CIs on real video), and preserves
all REFUSED guardrails (ICC 15-degree elbow legality) and medical non-diagnosis disclaimers.
"""

from __future__ import annotations

from typing import List

from .schemas import BiomechanicalFactSheet, CoachingReport, ObservabilityGrade

INTERVAL_LABEL = {
    "repeatability_interval": "Repeatability Interval",
    "analytical_uncertainty": "Analytical Uncertainty Interval",
    "conformal_calibrated_ci": "Conformal 95% CI",
}


def generate_grounded_report(fact_sheet: BiomechanicalFactSheet) -> CoachingReport:
    """Generate a deterministic, 100% measurement-grounded CoachingReport directly from the FactSheet."""
    cited: List[str] = []
    valid_metrics = [m for m in fact_sheet.metrics.values() if not m.suppressed and m.value is not None]
    refused_metrics = [m for m in fact_sheet.metrics.values() if m.observability == ObservabilityGrade.REFUSED]
    suppressed_other = [
        m for m in fact_sheet.metrics.values()
        if m.suppressed and m.observability != ObservabilityGrade.REFUSED
    ]

    for m in valid_metrics:
        cited.append(m.key)

    int_label = INTERVAL_LABEL.get(fact_sheet.interval_type, "Uncertainty Interval")
    fps_label = (
        f"{fact_sheet.audit.fps:.2f} nominal fps [UNVERIFIED TIMEBASE]"
        if fact_sheet.audit.timebase_unreliable
        else f"{fact_sheet.audit.fps:.2f} unique fps"
    )
    exec_parts = [
        f"**{fact_sheet.bowler_id}** ({fact_sheet.bowling_arm.value}-arm pace bowler, "
        f"action class: **{fact_sheet.action_class.value.replace('_', ' ')}**) was analyzed from "
        f"{fact_sheet.audit.width}x{fact_sheet.audit.height} @ **{fps_label}** "
        f"(timebase: `{fact_sheet.audit.timebase_status}`, view: `{fact_sheet.calibration.view_class.value}`, "
        f"azimuth `{fact_sheet.calibration.azimuth_deg:.1f} deg`, elevation `{fact_sheet.calibration.elevation_deg:.1f} deg`, "
        f"calibration: `{fact_sheet.calibration.method}`, intrinsics calibrated: `{fact_sheet.calibration.is_intrinsics_calibrated}`, "
        f"camera motion/pan: `{fact_sheet.calibration.is_panning_camera}`).",
        f"**Interval Semantics:** Reported ranges are **{int_label}s** (`{fact_sheet.interval_type}`). "
        f"Unvalidated 0-100 composite scores are intentionally retired in favor of individual verified metrics and cohort norms.",
    ]

    if fact_sheet.workload_guidance is not None:
        wg = fact_sheet.workload_guidance
        exec_parts.append(
            f"**Workload Guidance ({wg.age_band}):** {wg.status_note} *({wg.governing_body_source})*"
        )

    completeness = (
        f"Delivery stride duration (BFC->FFC) was **{fact_sheet.events.delivery_stride_duration_ms:.0f} ms** "
        f"(+- {fact_sheet.events.bfc.timing_uncertainty_ms:.1f}/{fact_sheet.events.ffc.timing_uncertainty_ms:.1f} ms) "
        f"and FFC->BR took **{fact_sheet.events.ffc_to_br_duration_ms:.0f} ms** "
        f"(+- {fact_sheet.events.br.timing_uncertainty_ms:.1f} ms)."
        if fact_sheet.events.is_complete_delivery
        else "**Incomplete delivery clip:** no verified high-speed overhead ball release passed sanity gates; all delivery metrics and cues were suppressed."
    )
    profile_summary = (
        f"{completeness} Out of {len(fact_sheet.metrics)} tracked variables, **{len(valid_metrics)}** "
        f"passed observability and confidence gates, **{len(suppressed_other)}** were withheld, and "
        f"**{len(refused_metrics)}** were permanently refused."
    )

    strengths_text = [
        f"**{c.title}:** {c.biomechanical_mechanism} *Next step:* {c.actionable_recommendation}"
        for c in fact_sheet.strengths
    ] or [
        "No strengths cleared interval-level confidence gates from this clip."
    ]

    improvements_text = [
        f"**{c.title}:** {c.biomechanical_mechanism} *Actionable focus:* {c.actionable_recommendation}"
        for c in fact_sheet.limiters
    ]
    for inc in fact_sheet.inconclusive_observations:
        improvements_text.append(f"**{inc.title}:** {inc.biomechanical_mechanism} *{inc.actionable_recommendation}*")
    if not improvements_text:
        improvements_text = [
            "No mechanical limiters cleared interval-level confidence gates among currently observable metrics."
        ]

    caveats_text = list(fact_sheet.audit.warnings) + list(fact_sheet.events.event_warnings)
    for up in fact_sheet.capture_upgrades:
        caveats_text.append(f"**Capture Upgrade — {up.title}:** {up.actionable_recommendation}")
    # ALWAYS include all REFUSED metrics without truncation
    for rm in refused_metrics:
        caveats_text.append(f"**Refused `{rm.label}`:** {rm.suppression_reason}")
    if suppressed_other:
        for sm in suppressed_other[:8]:
            caveats_text.append(f"Withheld `{sm.label}`: {sm.suppression_reason}")
        if len(suppressed_other) > 8:
            extra_names = ", ".join(m.label for m in suppressed_other[8:])
            caveats_text.append(f"Additional withheld metrics ({len(suppressed_other) - 8}): {extra_names}.")

    return CoachingReport(
        bowler_id=fact_sheet.bowler_id,
        executive_summary="\n\n".join(exec_parts),
        action_profile_summary=profile_summary,
        strengths_narrative=strengths_text,
        prioritized_improvements=improvements_text,
        capture_quality_caveats=caveats_text,
        disclaimers=list(fact_sheet.hard_guardrails),
        cited_metric_keys=cited,
        generator_mode="deterministic_template",
    )


def render_report_markdown(
    fact_sheet: BiomechanicalFactSheet,
    report: CoachingReport,
) -> str:
    """Format the verified FactSheet and grounded CoachingReport into a crisp Markdown document."""
    int_label = INTERVAL_LABEL.get(fact_sheet.interval_type, "Interval")
    lines = [
        f"# Fast-Bowling Biomechanical Analysis Report: {fact_sheet.bowler_id}",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Delivery Phase & Observability Overview",
        report.action_profile_summary,
        "",
        f"## Verified Biomechanical Measurements ({int_label})",
        "",
        f"| Metric | Value ({int_label}) | Observability | Literature Context & Definition Note |",
        "|---|---|---|---|",
    ]
    for m in fact_sheet.metrics.values():
        if m.suppressed or m.value is None:
            val_str = f"**[{m.observability.value.upper()}]**"
        else:
            val_str = f"**{m.value:.1f} {m.unit}** (`[{m.ci_low:.1f}, {m.ci_high:.1f}]`)"
        lines.append(f"| {m.label} | {val_str} | `{m.observability.value}` | {m.empirical_reference} |")

    if fact_sheet.cohort_comparisons:
        lines.extend([
            "",
            "## Definition-Matched Cohort Comparisons",
            "",
            "| Metric | Value | Cohort (Mean +- SD) | Z-Score | Approx %ile | Definition Match |",
            "|---|---|---|---|---|---|",
        ])
        for cc in fact_sheet.cohort_comparisons:
            lines.append(
                f"| {cc.label} | {cc.value:.1f} `[{cc.ci_low:.1f}, {cc.ci_high:.1f}]` {cc.unit} | "
                f"{cc.cohort_name}: {cc.cohort_mean:.1f} +- {cc.cohort_sd:.1f} | {cc.z_score:+.2f} | "
                f"{cc.percentile_approx:.0f}% | `{cc.definition_match}` |"
            )

    lines.extend(["", "## Interval-Cleared Strengths"])
    for s in report.strengths_narrative:
        lines.append(f"- {s}")

    lines.extend(["", "## Interval-Cleared Limiters & Inconclusive Observations"])
    for imp in report.prioritized_improvements:
        lines.append(f"- {imp}")

    lines.extend(["", "## Capture Quality Caveats & Suppressed Metrics"])
    for cav in report.capture_quality_caveats:
        lines.append(f"- {cav}")

    lines.extend(["", "## Scientific & Medical Guardrails"])
    for d in report.disclaimers:
        lines.append(f"> {d}")

    return "\n".join(lines) + "\n"
