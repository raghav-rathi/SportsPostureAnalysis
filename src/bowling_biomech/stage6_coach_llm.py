"""Stage 6B: Strictly Measurement-Grounded Multimodal Coaching Synthesizer.

Enforces the neuro-symbolic boundary: every numeric claim in the coaching narrative MUST
trace back to a non-suppressed MetricEstimate in the verified BiomechanicalFactSheet JSON,
and REFUSED guardrails (ICC 15-degree legality) are NEVER truncated or omitted.
"""

from __future__ import annotations

from typing import List

from .schemas import BiomechanicalFactSheet, CoachingReport, ObservabilityGrade

LLM_SYSTEM_PROMPT = """You are an accredited cricket fast-bowling biomechanist and performance coach.
You will receive a strictly verified JSON BiomechanicalFactSheet.
RULES (NON-NEGOTIABLE):
1. NEVER invent, guess, or alter any numeric angle, velocity, or score not explicitly present in the JSON.
2. NEVER quote a suppressed or REFUSED metric as a measurement. Explicitly explain why suppressed metrics were withheld and how to re-film to unlock them.
3. Always report 95% confidence intervals alongside key measurements.
4. Distinguish clearly between per-delivery mechanical loading and actual injury risk (which depends on workload, age/maturity, and physical capacity).
5. NEVER issue an ICC 15-degree elbow legality verdict from video.
"""


def build_llm_prompt(fact_sheet: BiomechanicalFactSheet) -> str:
    """Build a strict JSON-grounded user prompt ready for Gemini / OpenAI / local vLLM APIs."""
    return (
        "Analyze the following verified BiomechanicalFactSheet JSON and produce a clear, "
        "evidence-cited coaching evaluation covering strengths, prioritized limiters, "
        "capture-quality caveats, and actionable drills:\n\n"
        f"```json\n{fact_sheet.model_dump_json(indent=2)}\n```"
    )


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

    kin_p = fact_sheet.kinetic_efficiency
    lum_p = fact_sheet.lumbar_load_screening
    quad = fact_sheet.quadrant

    exec_parts = [
        f"**{fact_sheet.bowler_id}** ({fact_sheet.bowling_arm.value}-arm pace bowler, "
        f"classified as **{fact_sheet.action_class.value.replace('_', ' ')}**) was analyzed from "
        f"{fact_sheet.audit.width}x{fact_sheet.audit.height} @ **{fact_sheet.audit.fps:.1f} fps** "
        f"footage (view: `{fact_sheet.calibration.view_class.value}`, azimuth `{fact_sheet.calibration.azimuth_deg:.1f} deg`, "
        f"elevation `{fact_sheet.calibration.elevation_deg:.1f} deg`, calibration: `{fact_sheet.calibration.method}`, "
        f"panning: `{fact_sheet.calibration.is_panning_camera}`).",
        f"**Kinetic & Pace Efficiency:** {kin_p.summary}",
        f"**Mechanical Load & Durability:** {lum_p.summary}",
        f"**2D Quadrant Placement:** `{quad.quadrant_label}` — {quad.interpretation}",
    ]

    completeness = (
        f"Delivery stride duration (BFC->FFC) was **{fact_sheet.events.delivery_stride_duration_ms:.0f} ms** "
        f"and front-foot contact to ball release (FFC->BR) took **{fact_sheet.events.ffc_to_br_duration_ms:.0f} ms** "
        f"(+- {fact_sheet.events.br.timing_uncertainty_ms:.1f} ms frame quantization)."
        if fact_sheet.events.is_complete_delivery
        else "**Incomplete delivery clip:** overhead ball release was not observed before clip end; release-phase kinematics were suppressed."
    )
    profile_summary = (
        f"{completeness} Out of {len(fact_sheet.metrics)} tracked biomechanical variables, **{len(valid_metrics)}** "
        f"passed observability and confidence thresholds, **{len(suppressed_other)}** were withheld due to "
        f"camera-angle, crop, panning, or frame-rate constraints, and **{len(refused_metrics)}** were permanently refused."
    )

    strengths_text = [
        f"**{c.title}:** {c.biomechanical_mechanism} *Next step:* {c.actionable_recommendation}"
        for c in fact_sheet.strengths
    ] or [
        "No primary strengths could be confirmed with high confidence from this single camera angle/FPS; "
        "record a 120–240 fps static-tripod side-on + rear-on pair for a full kinetic profile."
    ]

    improvements_text = [
        f"**{c.title}:** {c.biomechanical_mechanism} *Actionable focus:* {c.actionable_recommendation}"
        for c in fact_sheet.limiters
    ] or [
        "No high-confidence mechanical limiters were flagged among the currently observable metrics."
    ]

    caveats_text = list(fact_sheet.audit.warnings) + list(fact_sheet.events.event_warnings)
    for up in fact_sheet.capture_upgrades:
        caveats_text.append(f"**Capture Upgrade — {up.title}:** {up.actionable_recommendation}")
    # ALWAYS include all REFUSED metrics (never truncated!)
    for rm in refused_metrics:
        caveats_text.append(f"**Refused `{rm.label}`:** {rm.suppression_reason}")
    # Summarize all suppressed metrics without silently dropping entries
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
    lines = [
        f"# Fast-Bowling Biomechanical Analysis Report: {fact_sheet.bowler_id}",
        "",
        "## Executive Summary",
        report.executive_summary,
        "",
        "## Delivery Phase & Observability Overview",
        report.action_profile_summary,
        "",
        "## Verified Biomechanical Measurements (with 95% Confidence Intervals)",
        "",
        "| Metric | Value (95% CI) | Observability | Empirical Literature Reference |",
        "|---|---|---|---|",
    ]
    for m in fact_sheet.metrics.values():
        if m.suppressed or m.value is None:
            val_str = f"**[{m.observability.value.upper()}]**"
        else:
            val_str = f"**{m.value:.1f} {m.unit}** (`[{m.ci_low:.1f}, {m.ci_high:.1f}]`)"
        lines.append(f"| {m.label} | {val_str} | `{m.observability.value}` | {m.empirical_reference} |")

    lines.extend(["", "## Key Strengths"])
    for s in report.strengths_narrative:
        lines.append(f"- {s}")

    lines.extend(["", "## Prioritized Mechanical Limiters & Coaching Focus"])
    for imp in report.prioritized_improvements:
        lines.append(f"- {imp}")

    lines.extend(["", "## Capture Quality Caveats & Suppressed Metrics"])
    for cav in report.capture_quality_caveats:
        lines.append(f"- {cav}")

    lines.extend(["", "## Scientific & Medical Guardrails"])
    for d in report.disclaimers:
        lines.append(f"> {d}")

    return "\n".join(lines) + "\n"
