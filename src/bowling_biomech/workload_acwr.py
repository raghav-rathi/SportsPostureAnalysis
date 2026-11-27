"""Studio v5 (Module 9): Governing-Body Fast-Bowling Workload, ACWR & Wellness Engine.

Aligned 1:1 with `web/src/types.ts` (`WorkloadResponse`, `AcwrSummary`, `WorkloadDailyPoint`,
`DirectiveCheck[]`, `WorkloadAlert[]`, `WorkloadSnapshot`).
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from .schemas import WorkloadContext
from .stage5_evaluator import build_workload_guidance

WORKLOAD_MEDICAL_DISCLAIMER = (
    "Workload limits and load-change ratios track cumulative bowling volume across matches and nets — they do NOT "
    "predict or diagnose injury. ACWR is a descriptive load-progression heuristic (Hulin et al., 2014; critiqued by "
    "Lolli et al., 2019 & Impellizzeri et al., 2021). Stop bowling and consult a sports physiotherapist or doctor "
    "immediately if lower-back or joint pain occurs."
)

SORENESS_REFERRAL_NOTICE = (
    "Cricket Australia and ECB guidelines advise bowlers with low-back pain or elevated stiffness during/after "
    "bowling to stop fast bowling immediately and consult a sports physiotherapist or doctor. This app cannot assess injuries."
)


def _age_years_from_band(age_band: str) -> int:
    b = str(age_band or "adult").strip().lower()
    if "13" in b or "12" in b or "11" in b:
        return 13
    if "15" in b or "14" in b:
        return 15
    if "17" in b or "16" in b:
        return 17
    if "19" in b or "18" in b:
        return 19
    return 24


def compute_bowler_workload_analytics(
    bowler: Dict[str, Any],
    logs: List[Dict[str, Any]],
    ref_date: Optional[date] = None,
) -> Dict[str, Any]:
    """Compute 28-day daily workload series, ECB/CA `DirectiveCheck[]`, ACWR/EWMA, and `WorkloadAlert[]`."""
    today = ref_date or datetime.now(timezone.utc).date()
    age_band_raw = str(bowler.get("age_band", "adult")).strip().lower()
    age_yrs = _age_years_from_band(age_band_raw)

    horizon_days = 56
    start_day = today - timedelta(days=horizon_days - 1)
    daily_balls: Dict[date, int] = {start_day + timedelta(days=i): 0 for i in range(horizon_days)}
    daily_srpe: Dict[date, float] = {start_day + timedelta(days=i): 0.0 for i in range(horizon_days)}
    daily_rpe_max: Dict[date, Optional[int]] = {start_day + timedelta(days=i): None for i in range(horizon_days)}
    daily_sore_max: Dict[date, Optional[int]] = {start_day + timedelta(days=i): None for i in range(horizon_days)}
    daily_max_spell: Dict[date, int] = {start_day + timedelta(days=i): 0 for i in range(horizon_days)}
    daily_types: Dict[date, List[str]] = {start_day + timedelta(days=i): [] for i in range(horizon_days)}
    recent_soreness_alert = False

    for row in logs:
        try:
            d = date.fromisoformat(str(row.get("log_date", ""))[:10])
        except ValueError:
            continue
        balls = max(0, int(row.get("deliveries_count") or 0))
        rpe = int(row.get("rpe") if row.get("rpe") is not None else 6)
        spell_b = int(row.get("max_spell_balls") or 0)
        sore_raw = row.get("soreness_lumbar")
        sore = int(sore_raw) if sore_raw is not None else None
        pain_flag = bool(row.get("pain_when_bowling") or False)

        if d in daily_balls:
            daily_balls[d] += balls
            daily_srpe[d] += float(balls * rpe)
            daily_rpe_max[d] = max(daily_rpe_max[d] or 0, rpe) if balls > 0 else daily_rpe_max[d]
            if sore is not None:
                daily_sore_max[d] = max(daily_sore_max[d] or 0, sore)
            daily_max_spell[d] = max(daily_max_spell[d], spell_b)
            st_str = str(row.get("session_type") or "net")
            if st_str not in daily_types[d]:
                daily_types[d].append(st_str)

        delta = (today - d).days
        if 0 <= delta < 7:
            if pain_flag or (sore is not None and sore >= 4):
                recent_soreness_alert = True

    days_list = [start_day + timedelta(days=i) for i in range(horizon_days)]
    balls_vec = [daily_balls[d] for d in days_list]

    # Williams et al. (2017) EWMA: lambda_a = 2/8 = 0.25, lambda_c = 2/29
    lam_a = 2.0 / 8.0
    lam_c = 2.0 / 29.0
    init_mean = sum(balls_vec[:28]) / 28.0 if sum(balls_vec[:28]) > 0 else (sum(balls_vec) / float(horizon_days))
    ewma_a = init_mean
    ewma_c = init_mean
    ewma_a_hist: List[float] = []
    ewma_c_hist: List[float] = []
    for b_val in balls_vec:
        ewma_a = lam_a * float(b_val) + (1.0 - lam_a) * ewma_a
        ewma_c = lam_c * float(b_val) + (1.0 - lam_c) * ewma_c
        ewma_a_hist.append(ewma_a)
        ewma_c_hist.append(ewma_c)

    daily_series_28d: List[Dict[str, Any]] = []
    for idx in range(horizon_days - 28, horizon_days):
        d = days_list[idx]
        a_7d = sum(balls_vec[max(0, idx - 6) : idx + 1])
        c_28d_wk = sum(balls_vec[max(0, idx - 27) : idx + 1]) / 4.0
        prev_21 = balls_vec[max(0, idx - 27) : max(0, idx - 6)]
        c_uncoupled_wk = (sum(prev_21) / 3.0) if len(prev_21) == 21 else c_28d_wk

        ew_a_wk = 7.0 * ewma_a_hist[idx]
        ew_c_wk = 7.0 * ewma_c_hist[idx]

        r_coupled = round(a_7d / c_28d_wk, 2) if c_28d_wk >= 30.0 else None
        r_uncoupled = round(a_7d / c_uncoupled_wk, 2) if c_uncoupled_wk >= 30.0 else None
        r_ewma = round(ew_a_wk / ew_c_wk, 2) if ew_c_wk >= 30.0 else None

        daily_series_28d.append(
            {
                "date": d.isoformat(),
                "balls": balls_vec[idx],
                "overs": round(balls_vec[idx] / 6.0, 1),
                "acute_7d": a_7d,
                "chronic_coupled_wk": round(c_28d_wk, 1),
                "chronic_uncoupled_wk": round(c_uncoupled_wk, 1),
                "acwr_coupled": r_coupled,
                "acwr_uncoupled": r_uncoupled,
                "acwr_ewma": r_ewma,
                "rpe": daily_rpe_max[d],
                "soreness_lumbar": daily_sore_max[d],
                "session_types": daily_types[d],
            }
        )

    cur = daily_series_28d[-1]
    last_7_balls = int(cur["acute_7d"])
    last_28_balls = int(sum(balls_vec[-28:]))
    wg = build_workload_guidance(WorkloadContext(age_years=age_yrs, bowls_per_week=last_7_balls))

    last_7_vec = balls_vec[-7:]
    days_bowled_7d = sum(1 for x in last_7_vec if x > 0)
    max_days_per_wk = 3 if age_yrs <= 17 else 4
    max_consec_allowed = 1 if age_yrs <= 17 else 2

    max_consec_7d = 0
    run_c = 0
    for x in last_7_vec:
        if x > 0:
            run_c += 1
            max_consec_7d = max(max_consec_7d, run_c)
        else:
            run_c = 0

    max_day_7d = max(last_7_vec) if last_7_vec else 0
    max_spell_7d = max(daily_max_spell[days_list[i]] for i in range(horizon_days - 7, horizon_days))

    primary_ratio = cur["acwr_ewma"] if cur["acwr_ewma"] is not None else cur["acwr_coupled"]
    if last_28_balls == 0:
        zone = "no_data"
        zone_label = "No workload logged in past 28 days"
    elif cur["chronic_coupled_wk"] < 30.0 or primary_ratio is None:
        zone = "low_chronic_base"
        zone_label = "Building base (<30 balls/wk chronic)"
    elif primary_ratio < 0.80:
        zone = "underload"
        zone_label = "Lighter week / taper (<0.80)"
    elif primary_ratio <= 1.30:
        zone = "sweet_spot"
        zone_label = "Typical progression (0.80–1.30)"
    elif primary_ratio <= 1.50:
        zone = "caution_elevated"
        zone_label = "Elevated weekly ramp (1.31–1.50)"
    else:
        zone = "high_spike_alert"
        zone_label = "Sharp acute load spike (>1.50)"

    wk_cap = wg.max_balls_per_week if wg is not None else 240
    day_cap = wg.max_balls_per_day if wg is not None else None
    spell_cap = wg.max_balls_per_spell if wg is not None else None
    gov_src = wg.governing_body_source if wg is not None else "ECB (2024) & Cricket Australia (2026)"

    directive_checks: List[Dict[str, Any]] = [
        {
            "id": "weekly_balls",
            "label_plain": "Weekly volume (matches + nets)",
            "limit_text": f"≤ {wk_cap} balls / 7 days" if wk_cap is not None else "Manage gradual progression",
            "current_text": f"{last_7_balls} balls ({round(last_7_balls / 6.0, 1)} overs)",
            "ok": (last_7_balls <= wk_cap) if wk_cap is not None else True,
            "source": gov_src,
        },
        {
            "id": "bowling_days_per_week",
            "label_plain": "Bowling days in past 7 days",
            "limit_text": f"≤ {max_days_per_wk} days / week",
            "current_text": f"{days_bowled_7d} days",
            "ok": days_bowled_7d <= max_days_per_wk,
            "source": "Cricket Australia 1-3-5 / 2-4-6 & ECB Weekly Frequency Guidance",
        },
        {
            "id": "consecutive_days",
            "label_plain": "Consecutive bowling days",
            "limit_text": "No back-to-back days (≤ 1)" if max_consec_allowed == 1 else f"≤ {max_consec_allowed} consecutive days",
            "current_text": f"{max_consec_7d} consecutive",
            "ok": max_consec_7d <= max_consec_allowed,
            "source": "Cricket Australia Junior & Pathway Recovery Guidelines (2026)",
        },
    ]
    if day_cap is not None:
        directive_checks.append(
            {
                "id": "daily_balls",
                "label_plain": "Peak single-day balls (7d)",
                "limit_text": f"≤ {day_cap} balls / day",
                "current_text": f"{max_day_7d} balls",
                "ok": max_day_7d <= day_cap,
                "source": gov_src,
            }
        )
    if spell_cap is not None:
        directive_checks.append(
            {
                "id": "spell_balls",
                "label_plain": "Longest single spell (7d)",
                "limit_text": f"≤ {spell_cap} balls / spell",
                "current_text": f"{max_spell_7d} balls" if max_spell_7d > 0 else "Not logged",
                "ok": (max_spell_7d <= spell_cap) if max_spell_7d > 0 else None,
                "source": gov_src,
            }
        )

    alerts: List[Dict[str, Any]] = []
    if recent_soreness_alert:
        alerts.append(
            {
                "kind": "lumbar_stiffness_logged",
                "severity": "alert",
                "title": "Low-back pain or stiffness logged",
                "body_plain": SORENESS_REFERRAL_NOTICE,
                "source": "Cricket Australia & ECB Welfare Guidance",
            }
        )
    if wk_cap is not None and last_7_balls > wk_cap:
        alerts.append(
            {
                "kind": "weekly_cap_exceeded",
                "severity": "alert",
                "title": f"Weekly volume ({last_7_balls} balls) exceeds {wg.age_band if wg else 'guideline'} ceiling ({wk_cap} balls)",
                "body_plain": "Schedule recovery days across matches and nets before adding high-intensity spells.",
                "source": gov_src,
            }
        )
    if day_cap is not None and max_day_7d > day_cap:
        alerts.append(
            {
                "kind": "daily_cap_exceeded",
                "severity": "alert",
                "title": f"Single-day volume ({max_day_7d} balls) exceeded daily limit ({day_cap} balls)",
                "body_plain": "Age-band directives cap combined match and net balls in a single day.",
                "source": gov_src,
            }
        )
    if spell_cap is not None and max_spell_7d > spell_cap:
        alerts.append(
            {
                "kind": "spell_cap_exceeded",
                "severity": "caution",
                "title": f"Spell length ({max_spell_7d} balls) exceeded spell limit ({spell_cap} balls)",
                "body_plain": "Allow equivalent rest overs from the same end before bowling another spell.",
                "source": gov_src,
            }
        )
    if days_bowled_7d > max_days_per_wk:
        alerts.append(
            {
                "kind": "frequency_exceeded",
                "severity": "caution",
                "title": f"Bowled on {days_bowled_7d} of the past 7 days (guideline: ≤ {max_days_per_wk} days)",
                "body_plain": "Non-bowling recovery days between sessions allow bone and tendon remodeling.",
                "source": "Cricket Australia / ECB",
            }
        )
    if max_consec_7d > max_consec_allowed:
        alerts.append(
            {
                "kind": "consecutive_days_alert",
                "severity": "caution",
                "title": f"{max_consec_7d} consecutive bowling days logged (guideline: ≤ {max_consec_allowed})",
                "body_plain": "Alternate fast-bowling sessions with rest or non-bowling conditioning days.",
                "source": "Cricket Australia (2026)",
            }
        )
    if zone == "high_spike_alert" and primary_ratio is not None:
        alerts.append(
            {
                "kind": "high_spike_alert",
                "severity": "caution",
                "title": f"Sharp 7-day load jump ({primary_ratio:.2f}× chronic baseline)",
                "body_plain": "Hulin et al. (2014) observed elevated load exposure following >1.5× weekly volume spikes.",
                "source": "Hulin et al. (2014), BJSM",
            }
        )
    elif zone == "low_chronic_base" and last_7_balls > 0:
        alerts.append(
            {
                "kind": "low_chronic_base",
                "severity": "info",
                "title": "Building chronic bowling base (<30 balls/wk average)",
                "body_plain": "Ratio spikes are noisy at low chronic volume (Alway et al., 2019) — build weekly balls gradually by 15–20%.",
                "source": "Alway et al. (2019)",
            }
        )

    remaining_wk = max(0, (wk_cap or 240) - last_7_balls)
    if recent_soreness_alert:
        daily_loop = "Rest day advised: low-back pain or stiffness logged — consult a sports physiotherapist."
    elif max_consec_7d >= max_consec_allowed and last_7_vec[-1] > 0:
        daily_loop = f"Rest day recommended today after bowling yesterday ({last_7_balls}/{wk_cap} weekly balls used)."
    else:
        daily_loop = (
            f"{last_7_balls} of {wk_cap} weekly balls used across {days_bowled_7d} days "
            f"({remaining_wk} balls remaining within {wg.age_band if wg else 'Adult'} limit)."
        )

    return {
        "deliveries_last_7d": last_7_balls,
        "deliveries_last_28d": last_28_balls,
        "days_bowled_7d": days_bowled_7d,
        "max_days_per_week": max_days_per_wk,
        "guidance": wg.model_dump() if wg is not None else None,
        "acwr_summary": {
            "acute_7d": last_7_balls,
            "chronic_coupled_wk": cur["chronic_coupled_wk"],
            "chronic_uncoupled_wk": cur["chronic_uncoupled_wk"],
            "acwr_coupled": cur["acwr_coupled"],
            "acwr_uncoupled": cur["acwr_uncoupled"],
            "acwr_ewma": cur["acwr_ewma"],
            "zone": zone,
            "zone_label_plain": zone_label,
            "caveat_plain": (
                "ACWR is a contested heuristic (Lolli et al., 2019; Impellizzeri et al., 2021). "
                "It describes how fast your load changed; it does not predict injury."
            ),
            "sources": [
                "Hulin et al. (2014), BJSM",
                "Williams et al. (2017), BJSM (EWMA)",
                "Lolli et al. (2019), BJSM",
                "Impellizzeri et al. (2021), Sports Medicine",
                "Alway et al. (2019)",
            ],
        },
        "daily_series_28d": daily_series_28d,
        "directive_checks": directive_checks,
        "alerts": alerts,
        "daily_loop_summary": daily_loop,
        "medical_disclaimer": WORKLOAD_MEDICAL_DISCLAIMER,
    }
