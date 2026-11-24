"""Phase 2 (M2D): Deterministic Numeric-Verifier Coaching Copilot.

Answers coach/bowler questions grounded strictly in a verified `BiomechanicalFactSheet` (and
optional multi-delivery repeatability summary). Supports an optional Gemini API backend via
`GEMINI_API_KEY` plus a zero-dependency deterministic offline engine.

Hard Neuro-Symbolic Guardrail (`verify_coach_response`):
  - Strips Markdown formatting and splits sentences into independent clauses (preventing
    compound-sentence exemption poisoning).
  - Permanently blocks ANY ICC 15-degree elbow legality verdict ("legal", "illegal", "lawful",
    "chucking", "elbow straightens by N deg") unless the sentence explicitly states permanent refusal.
  - Blocks resurrected 0–100 composite scores ("95 / 100", "out of 100", "Overall ... Score").
  - Blocks ANY numeric claim on a suppressed metric via canonical per-metric synonym stems.
  - Verifies EVERY remaining numeric token against the non-suppressed FactSheet whitelist
    (no bare unitless metric-range constants like 6, 7, 8, 90, 95, 100, 120, 180 or image widths).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Dict, List, Optional, Set, Tuple

from .schemas import BiomechanicalFactSheet, ObservabilityGrade

_NUM_RE = re.compile(r"(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?(?![A-Za-z0-9_])")

# Only tiny structural counting constants are permitted bare & unitless;
# metric-range numbers (6, 7, 8, 90, 95, 100, 120, 180, etc.) require explicit contextual matching.
_ALLOWED_BARE_CONSTANTS = {0.0, 1.0, 2.0, 3.0, 4.0}

# Contextual technical constants stripped ONLY when accompanied by their exact technical unit/phrase
_CONTEXTUAL_CONSTANTS_RE = [
    re.compile(r"\((?:19|20)\d{2}(?:/\d{2,4})?\)"),                                   # citation years (2013), (2024/25)
    re.compile(r"(?m)^\s*\d+\.\s+"),                                                  # markdown numbered list indices
    re.compile(r"\b(?:25|30|50|60|90|100|120|240|250|1000)(?:\s*[-–—/]\s*(?:25|30|50|60|90|100|120|240|250|1000))?\s*(?:fps|hz)\b", re.I),    # camera/mocap sampling rates
    re.compile(r"\b(?:7|28)[\s-]*(?:day|days|d)\b", re.I),                            # rolling workload calendar windows
    re.compile(r"\b95\s*%\s*(?:ci|confidence|interval|repeatability)\b", re.I),       # 95% confidence interval
    re.compile(r"±\s*1\s*sd\b|\+/-\s*1\s*sd\b", re.I),                                # ±1 SD cohort shading
    re.compile(r"\b(?:0|45|90|180)\s*(?:deg(?:ree)?s?|°)\s*(?:side[\s-]*on|rear|front|open|camera|view|baseline)\b", re.I),
    re.compile(r"\b(?:side[\s-]*on|rear|front|open)\s*\(\s*~?(?:0|45|90|180)\s*(?:deg(?:ree)?s?|°)\s*\)", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?\s*m\s+(?:perpendicular|behind|back|away)\b", re.I),
    re.compile(r"\b(?:180|210|240|270)\s*(?:deg(?:ree)?s?|°)?\s*(?:side[\s-]*on|front[\s-]*on|mixed|straight)\b", re.I),
    re.compile(r"\b(?:side[\s-]*on|front[\s-]*on|mixed)\s*\(?\s*(?:180|210|240|270)\s*(?:deg(?:ree)?s?|°)?\s*\)?", re.I),
    re.compile(r"\bmcc\s+laws?\s+[678](?:\s*/\s*[678])*\b", re.I),                    # MCC Laws 6/7/8
    re.compile(r"\b9\s*:\s*16\b"),                                                    # 9:16 vertical aspect ratio
    re.compile(r"\b3[\s-]*dof\b|\b6[\s-]*dof\b|\b3d\b|\b2d\b", re.I),                 # DOF / 2D / 3D
    re.compile(r"\b0\s*[-–—]\s*100\b"),                                               # referencing retired "0-100" scale
]

# Sentences resurrecting retired 0-100 composite scores are permanently blocked
_COMPOSITE_SCORE_PATTERNS = [
    re.compile(r"\b\d+(?:\.\d+)?\s*/\s*100\b", re.I),
    re.compile(r"\b\d+(?:\.\d+)?\s+out\s+of\s+100\b", re.I),
    re.compile(r"\b(?:overall|composite|efficiency|kinetic|durability|biomechanical)\s+(?:[a-z]+\s+){0,2}score\s*[:=]?\s*\d", re.I),
]

# Explicit refusal / suppression vocabulary
_REFUSAL_WORDS_RE = re.compile(r"\b(?:permanently\s+)?refused\b", re.I)
_SUPPRESSION_WORDS_RE = re.compile(
    r"\b(?:suppressed|withheld|refused|unmeasurable|cannot be measured|not measured|gated|unavailable|missing|unreliable|omitted)\b",
    re.I,
)

# ICC elbow legality verdict detectors (evaluated on markdown-stripped sentences)
_ICC_SUBJECT_RE = re.compile(r"\b(?:elbow|arm|action|delivery|bowling|icc|chuck\w*|throw\w*)\b", re.I)
_ICC_VERDICT_RE = re.compile(
    r"\b(?:legal|illegal|lawful|unlawful|clean\s+action|chuck(?:ing|ed|s)?|throw(?:ing|n)?)\b"
    r"|\b(?:passes|fails|violates|exceeds|within|under|below|above)\s+(?:the\s+)?(?:icc\s+)?15[\s-]*(?:°|deg(?:ree)?s?)\b"
    r"|\b15[\s-]*(?:°|deg(?:ree)?s?)\s*(?:rule|limit|threshold|cap|elbow)\b",
    re.I,
)
_ICC_AFFIRMATIVE_VERDICT_RE = re.compile(
    r"\b(?:is|was|are|were|appears|looks|deemed|classified(?:\s+as)?|judged(?:\s+as)?|ruled)\s+(?:[a-z]+\s+){0,2}(?:legal|illegal|lawful|unlawful|clean)\b"
    r"|\belbow\s+(?:extension|flexion|straighten\w*|bend\w*)\s+(?:is|was|measured\s+at|at|of|by|reached|only)?\s*(?:[a-z]+\s+){0,2}\d",
    re.I,
)

# Canonical synonym stems per metric key (used to catch paraphrased suppressed-metric leaks)
_METRIC_SYNONYM_MAP: Dict[str, Tuple[str, ...]] = {
    "front_knee_ffc_deg": ("front knee at ffc", "front knee at front foot", "knee at ffc"),
    "front_knee_br_deg": ("front knee at br", "front knee at release", "front knee at ball release", "knee angle at release", "knee at br"),
    "front_knee_change_deg": ("front knee change", "knee flexion change", "knee collapse", "knee extension"),
    "runup_speed_mps": ("run-up speed", "runup speed", "approach speed", "run-up velocity", "runup velocity"),
    "stride_length_pct_stature": ("stride length", "delivery stride length"),
    "trunk_flexion_ffc_deg": ("trunk flexion at ffc", "forward flexion at ffc"),
    "trunk_flexion_br_deg": ("trunk flexion at br", "trunk flexion at release", "forward flexion at release"),
    "trunk_flexion_delta_deg": ("trunk flexion delta", "trunk flexion change", "trunk forward flexion"),
    "thorax_lat_flex_ffc_deg": ("lateral flexion at ffc", "side flexion at ffc", "contralateral tilt at ffc"),
    "thorax_lat_flex_br_deg": ("lateral flexion at br", "lateral flexion at release", "side flexion at release", "contralateral tilt at release"),
    "pelvis_alignment_bfc_deg": ("pelvis alignment", "pelvic alignment"),
    "shoulder_alignment_bfc_deg": ("shoulder alignment",),
    "shoulder_counter_rotation_deg": ("shoulder counter-rotation", "shoulder counter rotation", "counter-rotation", "counter rotation", "scr"),
    "hip_shoulder_sep_bfc_deg": ("hip-shoulder separation at bfc", "hip shoulder separation at bfc"),
    "max_hip_shoulder_sep_deg": ("max hip-shoulder separation", "peak hip-shoulder separation", "hip-shoulder separation"),
    "delayed_arm_ffc_deg": ("delayed arm", "bowling arm delay"),
    "rear_hip_bfc_deg": ("rear hip", "back hip"),
    "pelvic_drop_ffc_deg": ("pelvic drop", "hip drop"),
    "non_bowling_arm_elev_ffc_deg": ("non-bowling arm", "front arm elevation"),
    "peak_arm_angular_vel_dps": ("arm angular velocity", "peak angular velocity", "bowling arm speed"),
    "release_height_m": ("release height",),
    "follow_through_drift_deg": ("follow-through drift", "follow through drift"),
    "ball_release_speed_kph": ("ball release speed", "ball speed", "release speed", "bowling speed", "pace"),
}

# Broader family stems activated ONLY when ALL sibling metrics in that family are suppressed
_METRIC_FAMILY_STEMS: Dict[Tuple[str, ...], Tuple[str, ...]] = {
    ("front_knee_ffc_deg", "front_knee_br_deg", "front_knee_change_deg"): ("front knee", "knee angle"),
    ("trunk_flexion_ffc_deg", "trunk_flexion_br_deg", "trunk_flexion_delta_deg"): ("trunk flexion",),
    ("thorax_lat_flex_ffc_deg", "thorax_lat_flex_br_deg"): ("lateral flexion", "side flexion", "lateral bend", "contralateral tilt"),
}


@dataclass
class CoachCopilotResponse:
    """Verified coaching answer with explicit numeric-verifier audit trail."""

    answer_markdown: str
    cited_metric_keys: List[str]
    verification_passed: bool
    blocked_violations: List[str] = field(default_factory=list)
    generator_mode: str = "deterministic_grounded"


def _strip_markdown(text: str) -> str:
    """Remove bold/italic/code markdown markers so adverbs/phrases cannot evade regex checks."""
    s = re.sub(r"[`*_~]+", "", text)
    return s


def _build_allowed_numbers(fact_sheet: BiomechanicalFactSheet) -> Set[float]:
    allowed: Set[float] = set(_ALLOWED_BARE_CONSTANTS)
    a = fact_sheet.audit
    c = fact_sheet.calibration
    ev = fact_sheet.events
    # Exclude raw pixel dimensions (width, height) and arbitrary frame counts/indices from bare numeric whitelist
    # so e.g. `120` frames or `180` px never silently whitelists `"120 km/h"` or `"180 deg"`.
    for v in (
        a.fps, a.container_fps, a.duration_s, a.median_laplacian_var,
        a.held_frame_ratio * 100.0, a.longest_held_run_frames,
        c.azimuth_deg, c.elevation_deg, c.heading_sigma_deg, c.gravity_sigma_deg,
        ev.delivery_stride_duration_ms, ev.ffc_to_br_duration_ms,
        ev.bfc.timing_uncertainty_ms, ev.ffc.timing_uncertainty_ms, ev.br.timing_uncertainty_ms,
    ):
        if v is not None:
            allowed.add(float(v))
            allowed.add(abs(float(v)))

    for m in fact_sheet.metrics.values():
        if (not m.suppressed) and m.value is not None:
            for v in (m.value, m.ci_low, m.ci_high, m.sigma_total):
                if v is not None:
                    allowed.add(float(v))
                    allowed.add(abs(float(v)))

    for comp in fact_sheet.cohort_comparisons:
        for v in (comp.value, comp.ci_low, comp.ci_high, comp.cohort_mean, comp.cohort_sd, comp.z_score):
            allowed.add(float(v))
            allowed.add(abs(float(v)))

    if fact_sheet.workload_guidance is not None:
        wg = fact_sheet.workload_guidance
        for v in (wg.max_balls_per_spell, wg.max_balls_per_day, wg.max_balls_per_week, wg.current_bowls_per_week):
            if v is not None:
                allowed.add(float(v))

    # Allow exact benchmark numbers appearing verbatim inside REFUSED guardrail explanations (e.g., 22.71 deg OpenCap RMSE)
    for m in fact_sheet.metrics.values():
        if m.observability == ObservabilityGrade.REFUSED and m.suppression_reason:
            for tok in _NUM_RE.findall(m.suppression_reason):
                val = float(tok)
                if val not in (6.0, 7.0, 8.0, 15.0, 90.0, 95.0, 100.0, 120.0, 180.0):
                    allowed.add(val)
                    allowed.add(abs(val))

    return allowed


def _suppressed_stems_for_fact_sheet(fact_sheet: BiomechanicalFactSheet) -> List[Tuple[str, List[str]]]:
    """Return `(metric_key, [match_stems...])` for every suppressed or REFUSED metric."""
    sup_keys = {
        m.key
        for m in fact_sheet.metrics.values()
        if m.suppressed or m.observability == ObservabilityGrade.REFUSED
    }
    # Always treat optical ball speed as suppressed if absent or suppressed
    if "ball_release_speed_kph" not in fact_sheet.metrics:
        sup_keys.add("ball_release_speed_kph")

    out: List[Tuple[str, List[str]]] = []
    for key in sorted(sup_keys):
        stems: List[str] = []
        m = fact_sheet.metrics.get(key)
        if m is not None:
            stems.append(m.label.lower())
        short_key = re.sub(r"_(?:deg|mps|kph|dps|pct_stature|m)$", "", key).replace("_", " ")
        if len(short_key) >= 4:
            stems.append(short_key)
        stems.extend(_METRIC_SYNONYM_MAP.get(key, ()))
        out.append((key, list(dict.fromkeys(stems))))

    for fam_keys, fam_stems in _METRIC_FAMILY_STEMS.items():
        present = [k for k in fam_keys if k in fact_sheet.metrics]
        if present and all(k in sup_keys for k in present):
            out.append((present[0], list(fam_stems)))

    return out


def verify_coach_response(
    text: str,
    fact_sheet: BiomechanicalFactSheet,
    tolerance: float = 0.25,
) -> Tuple[bool, List[str]]:
    """Deterministically verify that a response contains ZERO hallucinated numbers, ICC verdicts,
    resurrected 0-100 composite scores, or suppressed-metric leaks."""
    violations: List[str] = []
    plain = _strip_markdown(text)

    # 1. Check permanent ICC 15-degree elbow legality refusal & retired 0-100 composite scores
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", plain) if s.strip()]
    icc_refusal_sentences: List[str] = []
    for sent in sentences:
        # 1a. Block resurrected composite /100 scores
        for cpat in _COMPOSITE_SCORE_PATTERNS:
            cm = cpat.search(sent)
            if cm:
                violations.append(f"Blocked retired composite score claim: '{cm.group(0)}' (0-100 composites are permanently retired).")

        # 1b. Block ICC legality verdicts or elbow-extension measurements
        if _ICC_AFFIRMATIVE_VERDICT_RE.search(sent):
            violations.append(f"Blocked ICC legality/elbow verdict: '{sent[:120]}' (permanently REFUSED on monocular/casual video).")
            continue
        if _ICC_SUBJECT_RE.search(sent) and _ICC_VERDICT_RE.search(sent):
            if _REFUSAL_WORDS_RE.search(sent) or re.search(r"\bno\s+elbow\s+extension\s+measurement\b|\bcannot\s+(?:issue|judge|determine)\b", sent, re.I):
                icc_refusal_sentences.append(sent)
            else:
                violations.append(f"Blocked ICC legality claim without explicit refusal: '{sent[:120]}'")

    # 2. Check suppressed metrics clause-by-clause (prevents compound-sentence exemption poisoning!)
    sup_stems = _suppressed_stems_for_fact_sheet(fact_sheet)
    for sent in sentences:
        clauses = [c.strip() for c in re.split(r"[,;:]|\b(?:but|although|though|while|whereas|however|yet|and)\b", sent, flags=re.I) if c.strip()]
        for clause in clauses:
            c_low = clause.lower()
            if _SUPPRESSION_WORDS_RE.search(c_low):
                continue
            if not _NUM_RE.search(clause):
                continue
            for key, stems in sup_stems:
                if any(st in c_low for st in stems):
                    violations.append(f"Leaked numeric claim for suppressed metric `{key}` in clause: '{clause[:120]}'")
                    break

    # 3. Verify every remaining numeric token appears in the non-suppressed whitelist
    allowed = _build_allowed_numbers(fact_sheet)
    cleaned = plain
    # Allow exact clip resolution string `WxH` matching audit dimensions
    cleaned = re.sub(rf"\b{fact_sheet.audit.width}\s*x\s*{fact_sheet.audit.height}\b", "RES", cleaned, flags=re.I)
    # Allow explicit `frame <idx>` references matching known event frames
    ev = fact_sheet.events
    valid_frames = {ev.bfc.frame_idx, ev.ffc.frame_idx, ev.br.frame_idx, ev.bfc.source_frame_idx, ev.ffc.source_frame_idx, ev.br.source_frame_idx}
    for fi in valid_frames:
        cleaned = re.sub(rf"\bframe\s+{int(fi)}\b", "FRAME", cleaned, flags=re.I)
    # Allow `15` ONLY inside verified ICC refusal sentences (never in affirmative verdict sentences)
    for ref_s in icc_refusal_sentences:
        ref_clean = re.sub(r"\b15\s*(?:°|[\s-]*deg(?:ree)?s?)\b", "ICC_LIMIT", ref_s, flags=re.I)
        cleaned = cleaned.replace(ref_s, ref_clean)
    # Strip exact verbatim FactSheet diagnostic prose (suppression reasons, workload note, capture upgrades)
    # ONLY in Step 3 (after Step 1 ICC/composite & Step 2 suppressed-clause checks already passed!)
    verbatim_prose: List[str] = []
    for m in fact_sheet.metrics.values():
        if m.suppression_reason:
            verbatim_prose.append(_strip_markdown(m.suppression_reason))
    if fact_sheet.workload_guidance is not None:
        if fact_sheet.workload_guidance.age_band:
            verbatim_prose.append(_strip_markdown(fact_sheet.workload_guidance.age_band))
        if fact_sheet.workload_guidance.status_note:
            verbatim_prose.append(_strip_markdown(fact_sheet.workload_guidance.status_note))
    for u in fact_sheet.capture_upgrades:
        if u.title:
            verbatim_prose.append(_strip_markdown(u.title))
        if u.actionable_recommendation:
            verbatim_prose.append(_strip_markdown(u.actionable_recommendation))
    for vp in sorted(set(verbatim_prose), key=len, reverse=True):
        if vp:
            cleaned = cleaned.replace(vp, " ")
    # Strip strictly matched technical contextual constants (fps/Hz, citation years, 95% CI, MCC Laws, etc.)
    for pat in _CONTEXTUAL_CONSTANTS_RE:
        cleaned = pat.sub(" ", cleaned)

    for tok in _NUM_RE.findall(cleaned):
        val = float(tok)
        if not any(abs(val - a) <= tolerance or abs(abs(val) - a) <= tolerance for a in allowed):
            violations.append(f"Unverified numeric value `{tok}` not grounded in non-suppressed FactSheet metrics.")

    return (len(violations) == 0), violations


def _build_deterministic_answer(question: str, fact_sheet: BiomechanicalFactSheet) -> Tuple[str, List[str]]:
    q = question.lower().strip()
    valid = {k: m for k, m in fact_sheet.metrics.items() if (not m.suppressed) and m.value is not None}
    suppressed = {k: m for k, m in fact_sheet.metrics.items() if m.suppressed}
    cited: List[str] = []

    # Handle ICC / chucking / elbow legality questions directly with the non-negotiable guardrail
    if any(w in q for w in ("icc", "chuck", "legal", "elbow", "bent arm", "15")):
        rm = fact_sheet.metrics["icc_elbow_legality_verdict"]
        return (
            f"**ICC 15-Degree Elbow Extension Legality is permanently REFUSED** on monocular or casual video. "
            f"{rm.suppression_reason} No elbow extension measurement or legality verdict can be issued from this clip.",
            [],
        )

    # Handle incomplete delivery clips cleanly
    if not fact_sheet.events.is_complete_delivery:
        return (
            f"All delivery metrics for **{fact_sheet.bowler_id}** were suppressed because this clip did not pass "
            f"overhead ball-release sanity gates (`{fact_sheet.events.br.detection_method}`). "
            f"Timebase status: `{fact_sheet.audit.timebase_status}` ({fact_sheet.audit.held_frame_ratio * 100:.1f}% repeated frames). "
            f"Record a complete, unedited delivery through follow-through to unlock measurements.",
            [],
        )

    # Handle injury / back / medical diagnosis questions with non-diagnosis guardrail
    if any(w in q for w in ("injur", "fracture", "back pain", "diagnos", "safe", "stress")):
        lines = [
            "**Medical Non-Diagnosis Guardrail:** Single-delivery video screening estimates mechanical stress indicators only and cannot diagnose or predict injury risk in isolation (injury depends on workload, skeletal maturity, and physical capacity)."
        ]
        for k in ("thorax_lat_flex_br_deg", "shoulder_counter_rotation_deg", "pelvic_drop_ffc_deg"):
            if k in valid:
                m = valid[k]
                cited.append(k)
                lines.append(f"- **{m.label}:** `{m.value}` (`[{m.ci_low}, {m.ci_high}]` {m.unit}, `{m.observability.value}`, `{m.interval_type}`).")
            elif k in suppressed:
                lines.append(f"- **{suppressed[k].label}:** Suppressed ({suppressed[k].suppression_reason})")
        if fact_sheet.workload_guidance is not None:
            wg = fact_sheet.workload_guidance
            lines.append(f"- **Workload Context ({wg.age_band}):** {wg.status_note}")
        return "\n".join(lines), cited

    # Handle "why suppressed / what is missing / camera" questions
    if any(w in q for w in ("suppress", "missing", "withheld", "camera", "improve capture", "upgrade", "fps", "knee", "speed", "run-up", "runup", "stride")):
        lines = [
            f"Capture audit for **{fact_sheet.bowler_id}**: `{fact_sheet.audit.width}x{fact_sheet.audit.height}` @ "
            f"`{fact_sheet.audit.fps:.1f}` fps (`{fact_sheet.audit.timebase_status}`), view `{fact_sheet.calibration.view_class.value}` "
            f"(azimuth `{fact_sheet.calibration.azimuth_deg:.1f}` deg, calibration `{fact_sheet.calibration.method}`, pan=`{fact_sheet.calibration.is_panning_camera}`)."
        ]
        for key_hint, m_key in [
            ("knee", "front_knee_br_deg"),
            ("run-up", "runup_speed_mps"),
            ("runup", "runup_speed_mps"),
            ("ball speed", "ball_release_speed_kph"),
            ("speed", "runup_speed_mps"),
            ("stride", "stride_length_pct_stature"),
        ]:
            if key_hint in q and m_key in fact_sheet.metrics:
                me = fact_sheet.metrics[m_key]
                if me.suppressed:
                    lines.append(f"- **{me.label} is suppressed:** {me.suppression_reason}")
                else:
                    cited.append(m_key)
                    lines.append(f"- **{me.label}:** `{me.value}` (`[{me.ci_low}, {me.ci_high}]` {me.unit}, `{me.observability.value}`).")
        if fact_sheet.capture_upgrades:
            lines.append("\n**Recommended Capture Upgrades:**")
            for u in fact_sheet.capture_upgrades:
                lines.append(f"- **{u.title}:** {u.actionable_recommendation}")
        return "\n".join(lines), cited

    # Default comprehensive grounded summary of verified measurements, limiters, strengths & inconclusive cues
    lines = [
        f"**Verified Grounded Summary for {fact_sheet.bowler_id}** ({fact_sheet.bowling_arm.value}-arm, action: `{fact_sheet.action_class.value}`, `{fact_sheet.interval_type}`):"
    ]
    for k, m in valid.items():
        cited.append(k)
        lines.append(f"- **{m.label}:** `{m.value}` (`[{m.ci_low}, {m.ci_high}]` {m.unit}, `{m.observability.value}`).")
    if fact_sheet.strengths:
        lines.append("\n**Verified Strengths:** " + "; ".join(c.title for c in fact_sheet.strengths) + ".")
    if fact_sheet.limiters:
        lines.append("**Verified Mechanical Limiters:** " + "; ".join(c.title for c in fact_sheet.limiters) + ".")
    if fact_sheet.inconclusive_observations:
        lines.append("**Inconclusive (Interval Straddles Cutoff):** " + "; ".join(c.title for c in fact_sheet.inconclusive_observations) + ".")
    lines.append("\n*Reminder: ICC 15-degree elbow legality is permanently REFUSED on monocular/casual video.*")
    return "\n".join(lines), cited


def ask_grounded_coach(
    question: str,
    fact_sheet: BiomechanicalFactSheet,
    use_external_llm: bool = False,
) -> CoachCopilotResponse:
    """Answer a coaching question with strict post-generation deterministic numeric verification."""
    det_text, det_cited = _build_deterministic_answer(question, fact_sheet)

    # Optional Gemini API integration strictly wrapped in verify_coach_response (env var GEMINI_API_KEY only!)
    api_key = os.environ.get("GEMINI_API_KEY", "").strip() if use_external_llm else ""
    if api_key:
        try:
            import httpx

            non_sup_summary = {
                k: {
                    "label": m.label,
                    "value": m.value,
                    "ci_low": m.ci_low,
                    "ci_high": m.ci_high,
                    "unit": m.unit,
                    "observability": m.observability.value,
                }
                for k, m in fact_sheet.metrics.items()
                if (not m.suppressed) and m.value is not None
            }
            sup_keys = [k for k, m in fact_sheet.metrics.items() if m.suppressed]
            prompt = (
                "You are a biomechanics assistant for cricket fast bowling. Answer strictly using ONLY the "
                "non-suppressed JSON numbers below. NEVER invent numbers, NEVER estimate suppressed metrics, "
                "NEVER output a 0-100 composite score, NEVER diagnose injury, and ALWAYS state that ICC 15-degree "
                "elbow legality is permanently REFUSED.\n\n"
                f"NON_SUPPRESSED_METRICS: {non_sup_summary}\n"
                f"SUPPRESSED_KEYS: {sup_keys}\n"
                f"QUESTION: {question}\n"
            )
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
            resp = httpx.post(url, json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=12.0)
            if resp.status_code == 200:
                data = resp.json()
                cand = data["candidates"][0]["content"]["parts"][0]["text"]
                ok, viols = verify_coach_response(cand, fact_sheet)
                if ok:
                    return CoachCopilotResponse(
                        answer_markdown=cand,
                        cited_metric_keys=[k for k in non_sup_summary if k in cand or non_sup_summary[k]["label"].lower() in cand.lower()],
                        verification_passed=True,
                        blocked_violations=[],
                        generator_mode="gemini_verified",
                    )
                # Verification failed -> fall back to deterministic grounded answer with audit trail!
                return CoachCopilotResponse(
                    answer_markdown=det_text,
                    cited_metric_keys=det_cited,
                    verification_passed=True,
                    blocked_violations=viols,
                    generator_mode="deterministic_fallback_after_verifier_block",
                )
        except Exception:
            pass

    ok_det, viols_det = verify_coach_response(det_text, fact_sheet)
    return CoachCopilotResponse(
        answer_markdown=det_text,
        cited_metric_keys=det_cited,
        verification_passed=ok_det,
        blocked_violations=viols_det,
        generator_mode="deterministic_grounded",
    )
