import React, { useEffect, useMemo, useState } from "react";
import { api } from "../../api";
import type {
  AcwrSummary,
  DirectiveCheck,
  WorkloadAlert,
  WorkloadDailyPoint,
  WorkloadGuidance,
  WorkloadLog,
  WorkloadLogRequest,
} from "../../types";
import { SEG } from "../../theme";
import { CalmState, CardTitle, MONO, Tag, fmtNum, isoLocal, parseIsoLocal, shortDate } from "./ui";

/**
 * Workload & wellness panels (§5.4): limits FIRST, ratio SECOND, soreness referral notice.
 * Everything status-like is rendered verbatim from the backend; no risk %, ever.
 */

export const ACWR_CAVEAT =
  "ACWR is a contested load-change heuristic (Lolli et al. 2019; Impellizzeri et al. 2021) — it tracks rate of change, not injury prediction.";
export const CA_BACK_PAIN_NOTICE =
  "Cricket Australia advises young fast bowlers with low-back pain to stop bowling and see a sports physiotherapist or doctor. This app cannot assess injuries.";
export const MATCHES_AND_NETS = "Load counts every ball you bowl: matches and nets.";

// ── Daily loop (Whoop-style morning readiness strip) ─────────────────────────
export const DailyLoopCard: React.FC<{
  summary: string | null | undefined;
  last7: number | null;
  last28: number | null;
  ageBand?: string | null;
  acwrLabel?: string | null;
}> = ({ summary, last7, last28, ageBand, acwrLabel }) => (
  <div className="v-card p-4 sm:p-5">
    <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
      <div className="min-w-0 flex-1 space-y-1.5">
        <div className="flex flex-wrap items-center gap-2">
          <span className="inline-flex items-center gap-1.5 v-label text-[#171717]">
            <span className="w-2 h-2 rounded-full bg-[#297a3a] inline-block" aria-hidden />
            Daily loop · Morning readiness
          </span>
          {ageBand && <span className="v-chip">Age band {ageBand}</span>}
          {acwrLabel && <span className="v-chip text-[#4d4d4d]">{acwrLabel}</span>}
        </div>
        <p className="text-[15px] sm:text-[16px] font-[500] text-[#171717] leading-snug">
          {summary || "Log today's balls to start your daily summary. Rest days count too — they're part of the plan."}
        </p>
        <div className="text-[12px] text-[#666]">{MATCHES_AND_NETS}</div>
      </div>
      <div className="flex items-center gap-6 shrink-0 pt-2 lg:pt-0 border-t lg:border-t-0 lg:border-l border-[#ebebeb] lg:pl-6">
        <div>
          <div className="v-label">Last 7 days</div>
          <div className="v-hero-num text-[26px] sm:text-[28px] mt-1">
            {last7 ?? "—"}
            <span className="text-[12px] text-[#8f8f8f] font-normal ml-1">balls</span>
          </div>
        </div>
        <div>
          <div className="v-label">Last 28 days</div>
          <div className="v-hero-num text-[26px] sm:text-[28px] mt-1">
            {last28 ?? "—"}
            <span className="text-[12px] text-[#8f8f8f] font-normal ml-1">balls</span>
          </div>
        </div>
      </div>
    </div>
  </div>
);

// ── Quick logger ─────────────────────────────────────────────────────────────
const PRESETS = [18, 24, 30, 36, 48, 60];
type Remembered = { deliveries_count: number; session_type: "net" | "match"; rpe: number | null; max_spell_balls: number | null; log_date: string };
const lastKey = (bowlerId: string) => `spa.workload.last.${bowlerId}`;

function readRemembered(bowlerId: string): Remembered | null {
  try {
    const raw = window.localStorage.getItem(lastKey(bowlerId));
    if (!raw) return null;
    const p = JSON.parse(raw) as Partial<Remembered>;
    if (typeof p.deliveries_count !== "number") return null;
    return {
      deliveries_count: p.deliveries_count,
      session_type: p.session_type === "match" ? "match" : "net",
      rpe: typeof p.rpe === "number" ? p.rpe : null,
      max_spell_balls: typeof p.max_spell_balls === "number" ? p.max_spell_balls : null,
      log_date: typeof p.log_date === "string" ? p.log_date : "",
    };
  } catch {
    return null;
  }
}

const ChipButton: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode; label?: string; mono?: boolean }> = ({
  active,
  onClick,
  children,
  label,
  mono,
}) => (
  <button
    type="button"
    aria-pressed={active}
    aria-label={label}
    onClick={onClick}
    className={`v-chip ${active ? "v-chip-ink" : ""} ${mono ? "v-num" : ""} h-8 px-3 text-[13px]`}
  >
    {children}
  </button>
);

export const QuickLogger: React.FC<{ bowlerId: string; onLogged: () => void }> = ({ bowlerId, onLogged }) => {
  const [date, setDate] = useState(() => isoLocal(new Date()));
  const [sessionType, setSessionType] = useState<"net" | "match">("net");
  const [balls, setBalls] = useState<number | null>(null);
  const [custom, setCustom] = useState("");
  const [showCustom, setShowCustom] = useState(false);
  const [rpe, setRpe] = useState<number | null>(null);
  const [maxSpell, setMaxSpell] = useState("");
  const [soreness, setSoreness] = useState<number | null>(null);
  const [pain, setPain] = useState(false);
  const [more, setMore] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [ok, setOk] = useState<string | null>(null);
  const [remembered, setRemembered] = useState<Remembered | null>(null);

  useEffect(() => {
    const r = readRemembered(bowlerId);
    setRemembered(r);
    setDate(isoLocal(new Date()));
    setSoreness(null);
    setPain(false);
    setOk(null);
    setErr(null);
    if (r) {
      setSessionType(r.session_type);
      setRpe(r.rpe);
      setMaxSpell(r.max_spell_balls !== null ? String(r.max_spell_balls) : "");
      if (PRESETS.includes(r.deliveries_count)) {
        setBalls(r.deliveries_count);
        setShowCustom(false);
        setCustom("");
      } else {
        setBalls(r.deliveries_count);
        setShowCustom(true);
        setCustom(String(r.deliveries_count));
      }
    } else {
      setBalls(null);
      setRpe(null);
      setMaxSpell("");
      setShowCustom(false);
      setCustom("");
    }
  }, [bowlerId]);

  const submit = async () => {
    if (balls === null || !Number.isFinite(balls) || balls < 0) return;
    setBusy(true);
    setErr(null);
    setOk(null);
    const spell = maxSpell.trim() === "" ? null : Math.max(0, Math.round(Number(maxSpell)));
    const body: WorkloadLogRequest = {
      log_date: date || isoLocal(new Date()),
      deliveries_count: Math.round(balls),
      session_type: sessionType,
      notes: "",
      rpe,
      max_spell_balls: spell !== null && Number.isFinite(spell) ? spell : null,
      soreness_lumbar: soreness,
      pain_when_bowling: pain,
    };
    try {
      await api.logWorkload(bowlerId, body);
      const rem: Remembered = {
        deliveries_count: body.deliveries_count,
        session_type: sessionType,
        rpe,
        max_spell_balls: body.max_spell_balls ?? null,
        log_date: body.log_date,
      };
      try {
        window.localStorage.setItem(lastKey(bowlerId), JSON.stringify(rem));
      } catch {
        /* ignore */
      }
      setRemembered(rem);
      setOk(`Logged ${body.deliveries_count} balls (${sessionType === "match" ? "match" : "nets"}) for ${shortDate(body.log_date)}.`);
      setSoreness(null);
      setPain(false);
      onLogged();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="v-card p-4">
      <CardTitle
        label="Log a session"
        right={
          <input
            type="date"
            value={date}
            max={isoLocal(new Date())}
            onChange={(e) => setDate(e.target.value)}
            className="v-input v-num h-8 px-2 text-[12px]"
            aria-label="Session date"
          />
        }
      />
      <div className="v-seg mb-3" role="group" aria-label="Session type">
        <button type="button" aria-pressed={sessionType === "net"} onClick={() => setSessionType("net")}>
          Nets
        </button>
        <button type="button" aria-pressed={sessionType === "match"} onClick={() => setSessionType("match")}>
          Match
        </button>
      </div>
      <div className="v-label mb-1.5">Balls bowled</div>
      <div className="flex flex-wrap gap-1.5">
        {PRESETS.map((n) => (
          <ChipButton
            key={n}
            mono
            active={!showCustom && balls === n}
            onClick={() => {
              setShowCustom(false);
              setBalls(n);
            }}
          >
            {n}
          </ChipButton>
        ))}
        <ChipButton
          active={showCustom}
          onClick={() => {
            setShowCustom(true);
            const v = Number(custom);
            setBalls(custom.trim() !== "" && Number.isFinite(v) ? v : null);
          }}
        >
          Custom
        </ChipButton>
        {showCustom && (
          <input
            type="number"
            min={0}
            max={300}
            inputMode="numeric"
            value={custom}
            autoFocus
            onChange={(e) => {
              setCustom(e.target.value);
              const v = Number(e.target.value);
              setBalls(e.target.value.trim() !== "" && Number.isFinite(v) && v >= 0 ? v : null);
            }}
            className="v-input v-num h-8 w-20 px-2 text-[13px]"
            aria-label="Custom ball count"
            placeholder="balls"
          />
        )}
      </div>
      {balls !== null && <div className="v-num text-[11px] text-[#8f8f8f] mt-1">≈ {fmtNum(balls / 6, 1)} overs</div>}

      <button type="button" className="text-[12px] text-[#4d4d4d] underline underline-offset-2 mt-3" onClick={() => setMore((m) => !m)} aria-expanded={more}>
        {more ? "Hide effort & body check" : "Add effort & body check (optional)"}
      </button>

      {more && (
        <div className="mt-3 space-y-3">
          <div>
            <div className="v-label mb-1.5">Effort (RPE 1–10)</div>
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: 10 }, (_, i) => i + 1).map((v) => (
                <ChipButton key={v} mono active={rpe === v} onClick={() => setRpe(rpe === v ? null : v)} label={`RPE ${v}`}>
                  {v}
                </ChipButton>
              ))}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="v-label" htmlFor="max-spell">
              Longest spell (balls)
            </label>
            <input
              id="max-spell"
              type="number"
              min={0}
              max={120}
              inputMode="numeric"
              value={maxSpell}
              onChange={(e) => setMaxSpell(e.target.value)}
              className="v-input v-num h-8 w-20 px-2 text-[13px]"
              placeholder="e.g. 30"
            />
          </div>
          <div>
            <div className="v-label mb-1.5">Lower-back soreness (0 none – 10 worst)</div>
            <div className="flex flex-wrap gap-1">
              {Array.from({ length: 11 }, (_, i) => i).map((v) => (
                <ChipButton key={v} mono active={soreness === v} onClick={() => setSoreness(soreness === v ? null : v)} label={`Soreness ${v}`}>
                  {v}
                </ChipButton>
              ))}
            </div>
          </div>
          <div className="flex items-center justify-between gap-3">
            <span className="text-[13px] text-[#171717]">Pain when bowling?</span>
            <div className="v-seg" role="group" aria-label="Pain when bowling">
              <button type="button" aria-pressed={!pain} onClick={() => setPain(false)}>
                No
              </button>
              <button type="button" aria-pressed={pain} onClick={() => setPain(true)}>
                Yes
              </button>
            </div>
          </div>
          {(pain || (soreness ?? 0) >= 4) && (
            <div className="v-guardrail rounded-[6px] px-3 py-2" style={{ border: "1px dashed #8f8f8f" }}>
              {CA_BACK_PAIN_NOTICE}
            </div>
          )}
        </div>
      )}

      <button
        type="button"
        className="v-btn-primary w-full h-10 mt-4 text-[14px]"
        disabled={busy || balls === null}
        onClick={submit}
      >
        {busy ? "Saving…" : balls === null ? "Pick a ball count" : `Log ${Math.round(balls)} balls · ${sessionType === "match" ? "match" : "nets"}`}
      </button>
      {ok && (
        <div className="text-[12px] mt-2" style={{ color: "#297a3a" }} role="status">
          ✓ {ok}
        </div>
      )}
      {err && (
        <div className="text-[12px] text-[#171717] mt-2" role="alert">
          Couldn't save: {err}
        </div>
      )}
      <div className="text-[11px] text-[#8f8f8f] mt-2 leading-snug">
        {remembered
          ? `Last entry remembered: ${remembered.deliveries_count} balls · ${remembered.session_type === "match" ? "match" : "nets"}${
              remembered.log_date ? ` · ${shortDate(remembered.log_date)}` : ""
            }. `
          : ""}
        {MATCHES_AND_NETS}
      </div>
    </div>
  );
};

// ── Limits panel ─────────────────────────────────────────────────────────────
const CheckGlyph: React.FC<{ ok: boolean | null }> = ({ ok }) =>
  ok === true ? (
    <span style={{ color: "#297a3a" }} aria-label="Within limit">
      ✓
    </span>
  ) : ok === false ? (
    <span className="text-[#171717]" aria-label="Alert">
      ◆
    </span>
  ) : (
    <span className="text-[#8f8f8f]" aria-label="No data">
      —
    </span>
  );

const AlertRow: React.FC<{ a: WorkloadAlert }> = ({ a }) => {
  const style: React.CSSProperties =
    a.severity === "alert"
      ? { background: "#171717", color: "#fff" }
      : a.severity === "caution"
        ? { border: "1px dashed #8f8f8f" }
        : { boxShadow: "0 0 0 1px #ebebeb" };
  const glyph = a.severity === "alert" ? "◆" : a.severity === "caution" ? "◇" : "○";
  return (
    <li className="rounded-[6px] px-3 py-2" style={style}>
      <div className="text-[13px] font-medium">
        {glyph} {a.title}
      </div>
      <div className={`text-[12px] leading-snug mt-0.5 ${a.severity === "alert" ? "text-[#e5e5e5]" : "text-[#4d4d4d]"}`}>{a.body_plain}</div>
      {a.source && <div className={`text-[11px] mt-0.5 ${a.severity === "alert" ? "text-[#a8a8a8]" : "text-[#8f8f8f]"}`}>{a.source}</div>}
    </li>
  );
};

export const LimitsPanel: React.FC<{
  checks: DirectiveCheck[];
  guidance: WorkloadGuidance | null;
  alerts: WorkloadAlert[];
}> = ({ checks, guidance, alerts }) => {
  const sources = Array.from(new Set(checks.map((c) => c.source).filter(Boolean)));
  return (
    <div className="v-card p-4">
      <CardTitle
        label="Official limits · ECB (2024) & Cricket Australia (2026)"
        right={guidance?.age_band ? <span className="v-chip">{guidance.age_band}</span> : undefined}
      />
      {guidance && (
        <div className="grid grid-cols-3 gap-2 mb-3">
          {[
            { k: "Max spell", v: guidance.max_balls_per_spell },
            { k: "Max per day", v: guidance.max_balls_per_day },
            { k: "Max per week", v: guidance.max_balls_per_week },
          ].map((x) => (
            <div key={x.k} className="v-card-subtle px-2.5 py-2">
              <div className="v-label">{x.k}</div>
              <div className="v-num text-[18px] text-[#171717] mt-0.5">
                {x.v ?? "—"}
                {x.v !== null && <span className="text-[11px] text-[#8f8f8f] ml-1">balls</span>}
              </div>
              {x.v !== null && <div className="v-num text-[10px] text-[#8f8f8f]">{fmtNum(x.v / 6, 1)} overs</div>}
            </div>
          ))}
        </div>
      )}
      {checks.length === 0 ? (
        <CalmState title="No limit checks yet" body="Log a session to compare your load with the official guidelines." />
      ) : (
        <ul className="divide-y divide-[#ebebeb]">
          {checks.map((c) => (
            <li key={c.id} className="flex items-start gap-3 py-2">
              <span className="w-4 text-[15px] leading-5 text-center shrink-0">
                <CheckGlyph ok={c.ok} />
              </span>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline justify-between gap-x-3">
                  <span className="text-[13px] text-[#171717]">{c.label_plain}</span>
                  <span className="v-num text-[12px] text-[#4d4d4d]">{c.current_text}</span>
                </div>
                <div className="text-[11px] text-[#8f8f8f] leading-snug">
                  Limit: {c.limit_text}
                  {c.ok === null ? " · not enough data" : ""}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
      {alerts.length > 0 && (
        <ul className="space-y-2 mt-3" aria-label="Workload alerts">
          {alerts.map((a, i) => (
            <AlertRow key={`${a.kind}-${i}`} a={a} />
          ))}
        </ul>
      )}
      {guidance?.status_note && <div className="text-[12px] text-[#4d4d4d] mt-3 leading-snug">{guidance.status_note}</div>}
      <div className="text-[11px] text-[#8f8f8f] mt-2 leading-snug">
        Source: {guidance?.governing_body_source || sources.join(" · ") || "ECB (2024) Fast Bowling Directives / Cricket Australia (2026) junior guidelines"}.
        Guideline language only — rest equal to the spell length from the same end.
      </div>
    </div>
  );
};

// ── 28-day calendar ──────────────────────────────────────────────────────────
type DayCell = { date: string; balls: number; types: string[] };

function buildDays(series: WorkloadDailyPoint[] | undefined, logs: WorkloadLog[]): { end: Date; map: Map<string, DayCell> } {
  const map = new Map<string, DayCell>();
  if (series && series.length > 0) {
    series.forEach((p) => map.set(p.date.slice(0, 10), { date: p.date.slice(0, 10), balls: p.balls, types: p.session_types ?? [] }));
    const last = [...series].sort((a, b) => a.date.localeCompare(b.date))[series.length - 1];
    const today = new Date();
    const lastD = parseIsoLocal(last.date);
    return { end: lastD > today ? lastD : today, map };
  }
  logs.forEach((l) => {
    const d = l.log_date.slice(0, 10);
    const cur = map.get(d) ?? { date: d, balls: 0, types: [] };
    cur.balls += l.deliveries_count;
    if (l.session_type && !cur.types.includes(l.session_type)) cur.types.push(l.session_type);
    map.set(d, cur);
  });
  return { end: new Date(), map };
}

const HATCH = "repeating-linear-gradient(135deg, #d4d4d4 0 1.5px, transparent 1.5px 6px)";

export const WorkloadCalendar: React.FC<{ series?: WorkloadDailyPoint[]; logs: WorkloadLog[] }> = ({ series, logs }) => {
  const { end, map } = useMemo(() => buildDays(series, logs), [series, logs]);
  const endD = new Date(end.getFullYear(), end.getMonth(), end.getDate());
  const startWindow = new Date(endD);
  startWindow.setDate(endD.getDate() - 27);
  const gridStart = new Date(startWindow);
  gridStart.setDate(startWindow.getDate() - ((startWindow.getDay() + 6) % 7)); // back to Monday
  const weeks: Date[][] = [];
  const cursor = new Date(gridStart);
  while (cursor <= endD) {
    const wk: Date[] = [];
    for (let i = 0; i < 7; i++) {
      wk.push(new Date(cursor));
      cursor.setDate(cursor.getDate() + 1);
    }
    weeks.push(wk);
  }
  const todayIso = isoLocal(new Date());
  const anyData = Array.from(map.values()).some((c) => c.balls > 0);

  return (
    <div className="v-card p-4">
      <CardTitle label="Last 28 days" right={<span className="text-[11px] text-[#8f8f8f]">{MATCHES_AND_NETS}</span>} />
      <div className="grid gap-1" style={{ gridTemplateColumns: "repeat(7, minmax(0, 1fr)) minmax(36px, auto)" }}>
        {["M", "T", "W", "T", "F", "S", "S"].map((d, i) => (
          <div key={i} className="v-label text-center">
            {d}
          </div>
        ))}
        <div className="v-label text-right">Wk</div>
        {weeks.map((wk, wi) => {
          let total = 0;
          const cells = wk.map((d) => {
            const iso = isoLocal(d);
            const inWindow = d >= startWindow && d <= endD;
            const c = map.get(iso);
            const balls = inWindow && c ? c.balls : 0;
            if (inWindow) total += balls;
            const isMatch = !!c && c.types.some((t) => t === "match");
            const isToday = iso === todayIso;
            const style: React.CSSProperties = !inWindow
              ? { opacity: 0.25, border: "1px dotted #d4d4d4" }
              : balls > 0
                ? isMatch
                  ? { backgroundImage: HATCH, backgroundColor: "#fff", boxShadow: "inset 0 0 0 1px #171717" }
                  : { backgroundColor: "#ebebeb" }
                : { border: "1px dotted #c9c9c9", background: "#fff" };
            if (isToday) style.outline = "1.5px solid #171717";
            return (
              <div
                key={iso}
                className="aspect-square min-h-[34px] rounded-[4px] flex flex-col items-center justify-center"
                style={style}
                title={`${shortDate(iso)}: ${balls > 0 ? `${balls} balls${c && c.types.length ? ` (${c.types.join(", ")})` : ""}` : "rest / not logged"}`}
              >
                <span className="v-num text-[9px] text-[#8f8f8f] leading-none">{d.getDate()}</span>
                <span className={`v-num text-[13px] leading-tight ${balls > 0 ? "text-[#171717] font-medium" : "text-[#c9c9c9]"}`}>
                  {inWindow ? (balls > 0 ? balls : "·") : ""}
                </span>
              </div>
            );
          });
          return (
            <React.Fragment key={wi}>
              {cells}
              <div className="v-num text-[12px] text-[#4d4d4d] flex items-center justify-end">{total > 0 ? total : "—"}</div>
            </React.Fragment>
          );
        })}
      </div>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-3 text-[11px] text-[#666]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3.5 h-3.5 rounded-[3px]" style={{ background: "#ebebeb" }} /> nets
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3.5 h-3.5 rounded-[3px]" style={{ backgroundImage: HATCH, boxShadow: "inset 0 0 0 1px #171717" }} /> match
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block w-3.5 h-3.5 rounded-[3px]" style={{ border: "1px dotted #c9c9c9" }} /> rest / not logged
        </span>
        <span>number = balls</span>
      </div>
      {!anyData && <div className="text-[12px] text-[#8f8f8f] mt-2">No sessions logged in the last 28 days yet.</div>}
    </div>
  );
};

// ── Load-change (ACWR) chart — secondary ─────────────────────────────────────
type SeriesKey = "acwr_ewma" | "acwr_uncoupled" | "acwr_coupled";
const SERIES: Array<{ key: SeriesKey; label: string }> = [
  { key: "acwr_ewma", label: "EWMA" },
  { key: "acwr_uncoupled", label: "Rolling · uncoupled" },
  { key: "acwr_coupled", label: "Rolling · coupled" },
];

const CW = 600;
const CH = 210;
const CM = { l: 34, r: 12, t: 12, b: 24 };

export const LoadChangeChart: React.FC<{ series?: WorkloadDailyPoint[]; summary: AcwrSummary | null | undefined }> = ({
  series,
  summary,
}) => {
  const data = useMemo(() => [...(series ?? [])].sort((a, b) => a.date.localeCompare(b.date)), [series]);
  const available = SERIES.filter((s) => data.some((p) => p[s.key] != null) || (summary && summary[s.key] != null));
  const [sel, setSel] = useState<SeriesKey>("acwr_ewma");
  const active: SeriesKey = available.some((s) => s.key === sel) ? sel : available[0]?.key ?? "acwr_ewma";
  const lowBase = summary?.zone === "low_chronic_base";

  const vals = data.map((p) => p[active]).filter((v): v is number => v != null && Number.isFinite(v));
  const yMax = Math.max(2, ...vals.map((v) => v + 0.15));
  const pw = CW - CM.l - CM.r;
  const ph = CH - CM.t - CM.b;
  const n = data.length;
  const X = (i: number) => CM.l + (n <= 1 ? pw / 2 : (i / (n - 1)) * pw);
  const Y = (v: number) => CM.t + ph - (Math.min(v, yMax) / yMax) * ph;

  const paths: string[] = [];
  let cur = "";
  data.forEach((p, i) => {
    const v = p[active];
    if (v == null || !Number.isFinite(v)) {
      if (cur) paths.push(cur);
      cur = "";
      return;
    }
    cur += `${cur ? "L" : "M"}${X(i).toFixed(1)},${Y(v).toFixed(1)} `;
  });
  if (cur) paths.push(cur);

  const currentVal = summary ? summary[active] : null;
  const chronic = active === "acwr_coupled" ? summary?.chronic_coupled_wk : summary?.chronic_uncoupled_wk ?? summary?.chronic_coupled_wk;

  return (
    <div className="v-card p-4">
      <CardTitle
        label="Load change · acute : chronic (secondary)"
        right={
          available.length > 1 ? (
            <div className="v-seg" role="group" aria-label="Ratio method">
              {available.map((s) => (
                <button key={s.key} type="button" aria-pressed={active === s.key} onClick={() => setSel(s.key)}>
                  {s.label}
                </button>
              ))}
            </div>
          ) : undefined
        }
      />
      <p className="v-guardrail mb-3" style={{ color: "#171717" }}>
        {ACWR_CAVEAT}
      </p>
      {summary && (
        <div className="flex flex-wrap items-end gap-x-6 gap-y-2 mb-3">
          <div>
            <div className="v-label">Ratio now</div>
            <div className="v-hero-num text-[28px] mt-1">{lowBase ? "—" : fmtNum(currentVal, 2)}</div>
          </div>
          <div>
            <div className="v-label">Acute (7 d)</div>
            <div className="v-num text-[15px] mt-1">{fmtNum(summary.acute_7d, 0)} balls</div>
          </div>
          <div>
            <div className="v-label">Chronic (per week)</div>
            <div className="v-num text-[15px] mt-1">{chronic != null ? `${fmtNum(chronic, 0)} balls` : "—"}</div>
          </div>
          <div className="min-w-0">
            <div className="v-label">Reading</div>
            <div className="text-[13px] text-[#171717] mt-1">{summary.zone_label_plain}</div>
          </div>
        </div>
      )}
      {lowBase && (
        <div className="rounded-[6px] px-3 py-2 mb-3 text-[13px] text-[#171717]" style={{ border: "1px dashed #8f8f8f" }}>
          Chronic base under 30 balls/week — ratio not meaningful yet. Keep logging nets and matches; the ratio becomes readable once
          a steady weekly base exists.
        </div>
      )}
      {n === 0 ? (
        <CalmState title="No daily load series yet" body="The load-change line appears once you've logged a few sessions." />
      ) : (
        <div className={lowBase ? "v-ghosted" : ""}>
          <svg viewBox={`0 0 ${CW} ${CH}`} className="w-full h-auto" role="img" aria-label="Acute to chronic load ratio, last 28 days">
            <rect x={CM.l} y={Y(1.3)} width={pw} height={Y(0.8) - Y(1.3)} fill={SEG.pelvisTint} fillOpacity={0.7} />
            <text x={CM.l + 4} y={Y(1.3) + 11} fontSize={9} fill="#047857" fontFamily={MONO}>
              0.8–1.3 typical progression
            </text>
            {[0, 0.5, 1, 1.5, 2].filter((t) => t <= yMax).map((t) => (
              <g key={t}>
                <line x1={CM.l} x2={CM.l + pw} y1={Y(t)} y2={Y(t)} stroke="#000" strokeOpacity={t === 0 ? 0.25 : 0.06} />
                <text x={CM.l - 5} y={Y(t) + 3} fontSize={9} textAnchor="end" fill="#8f8f8f" fontFamily={MONO}>
                  {t.toFixed(1)}
                </text>
              </g>
            ))}
            <line x1={CM.l} x2={CM.l + pw} y1={Y(1.5)} y2={Y(1.5)} stroke={SEG.wrist} strokeDasharray="5 4" strokeWidth={1.2} />
            <text x={CM.l + pw - 4} y={Y(1.5) - 4} fontSize={9} textAnchor="end" fill={SEG.wrist} fontFamily={MONO}>
              1.5 sharp spike (Hulin et al. 2014)
            </text>
            {paths.map((d, i) => (
              <path key={i} d={d} fill="none" stroke={SEG.thorax} strokeWidth={1.8} strokeLinejoin="round" />
            ))}
            {data.map((p, i) => {
              const v = p[active];
              return v != null && Number.isFinite(v) ? (
                <circle key={p.date} cx={X(i)} cy={Y(v)} r={2.2} fill={SEG.thorax}>
                  <title>{`${shortDate(p.date)}: ratio ${fmtNum(v, 2)} · ${p.balls} balls that day`}</title>
                </circle>
              ) : null;
            })}
            {[0, Math.floor((n - 1) / 2), n - 1]
              .filter((v, i, a) => a.indexOf(v) === i)
              .map((i) => (
                <text key={i} x={X(i)} y={CH - 6} fontSize={9} textAnchor={i === 0 ? "start" : i === n - 1 ? "end" : "middle"} fill="#8f8f8f" fontFamily={MONO}>
                  {shortDate(data[i].date)}
                </text>
              ))}
          </svg>
        </div>
      )}
      {summary?.caveat_plain && <p className="v-guardrail mt-2">{summary.caveat_plain}</p>}
      {summary?.sources && summary.sources.length > 0 && (
        <div className="text-[11px] text-[#8f8f8f] mt-1 leading-snug">Sources: {summary.sources.join(" · ")}</div>
      )}
      <div className="text-[11px] text-[#8f8f8f] mt-1">No risk percentage is shown or computed — by design.</div>
    </div>
  );
};

// ── Soreness log ─────────────────────────────────────────────────────────────
const SW = 280;
const SH = 70;

export const SorenessLog: React.FC<{ logs: WorkloadLog[]; medicalDisclaimer?: string | null }> = ({ logs, medicalDisclaimer }) => {
  const pts = [...logs]
    .filter((l) => l.soreness_lumbar != null || l.pain_when_bowling)
    .sort((a, b) => a.log_date.localeCompare(b.log_date) || a.created_at.localeCompare(b.created_at))
    .slice(-20);
  const notice = logs.some((l) => l.pain_when_bowling === true || (l.soreness_lumbar ?? 0) >= 4);
  const latest = [...pts].reverse().find((l) => l.soreness_lumbar != null);
  const n = pts.length;
  const X = (i: number) => 10 + (n <= 1 ? (SW - 20) / 2 : (i / (n - 1)) * (SW - 20));
  const Y = (v: number) => 8 + (1 - v / 10) * (SH - 16);
  const line = pts
    .map((l, i) => (l.soreness_lumbar != null ? `${X(i).toFixed(1)},${Y(l.soreness_lumbar).toFixed(1)}` : null))
    .filter(Boolean)
    .join(" ");

  return (
    <div className="v-card p-4">
      <CardTitle
        label="Lower-back soreness (0–10)"
        right={latest ? <span className="v-num text-[13px] text-[#171717]">latest {latest.soreness_lumbar}/10</span> : undefined}
      />
      {n === 0 ? (
        <CalmState title="No body checks logged" body="Add soreness and 'pain when bowling?' in the logger's optional body check." />
      ) : (
        <svg viewBox={`0 0 ${SW} ${SH}`} className="w-full h-auto" role="img" aria-label="Lower-back soreness over recent logs">
          <line x1={10} x2={SW - 10} y1={Y(4)} y2={Y(4)} stroke="#8f8f8f" strokeDasharray="3 3" strokeWidth={0.8} />
          <text x={SW - 10} y={Y(4) - 3} fontSize={8} textAnchor="end" fill="#8f8f8f" fontFamily={MONO}>
            4 · notice threshold
          </text>
          <line x1={10} x2={SW - 10} y1={Y(0)} y2={Y(0)} stroke="#000" strokeOpacity={0.12} />
          {line && <polyline points={line} fill="none" stroke={SEG.com} strokeWidth={1.4} />}
          {pts.map((l, i) => (
            <g key={l.id}>
              <title>{`${shortDate(l.log_date)}: soreness ${l.soreness_lumbar ?? "—"}/10${l.pain_when_bowling ? " · pain when bowling" : ""}`}</title>
              {l.soreness_lumbar != null && <circle cx={X(i)} cy={Y(l.soreness_lumbar)} r={2.4} fill={SEG.com} />}
              {l.pain_when_bowling && (
                <text x={X(i)} y={8} fontSize={9} textAnchor="middle" fill="#171717">
                  ◆
                </text>
              )}
            </g>
          ))}
        </svg>
      )}
      {n > 0 && <div className="text-[11px] text-[#8f8f8f] mt-1">◆ = pain when bowling reported</div>}
      {notice && (
        <div className="v-guardrail rounded-[6px] px-3 py-2 mt-3 text-[#171717]" style={{ border: "1px solid #171717" }} role="note">
          {CA_BACK_PAIN_NOTICE}
        </div>
      )}
      {medicalDisclaimer && <p className="v-guardrail mt-2">{medicalDisclaimer}</p>}
    </div>
  );
};

// ── Log history ──────────────────────────────────────────────────────────────
export const LogHistory: React.FC<{ bowlerId: string; logs: WorkloadLog[]; onChanged: () => void }> = ({ bowlerId, logs, onChanged }) => {
  const [confirmId, setConfirmId] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [showAll, setShowAll] = useState(false);
  const sorted = [...logs].sort((a, b) => b.log_date.localeCompare(a.log_date) || b.created_at.localeCompare(a.created_at));
  const shown = showAll ? sorted : sorted.slice(0, 10);

  const del = async (id: string) => {
    setBusyId(id);
    setErr(null);
    try {
      await api.deleteWorkload(bowlerId, id);
      setConfirmId(null);
      onChanged();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <div className="v-card p-4">
      <CardTitle label={`Log history (${logs.length})`} />
      {sorted.length === 0 ? (
        <CalmState title="Nothing logged yet" body="Your nets and match sessions will appear here." />
      ) : (
        <ul className="divide-y divide-[#ebebeb]">
          {shown.map((l) => (
            <li key={l.id} className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 py-2">
              <div className="flex flex-wrap items-baseline gap-x-3 gap-y-0.5 min-w-0">
                <span className="v-num text-[12px] text-[#4d4d4d] w-[52px]">{shortDate(l.log_date)}</span>
                <span className="v-num text-[15px] text-[#171717]">{l.deliveries_count}</span>
                <span className="text-[12px] text-[#8f8f8f]">balls</span>
                <Tag variant={l.session_type === "match" ? "ink" : "plain"}>{l.session_type === "match" ? "Match" : l.session_type === "net" ? "Nets" : l.session_type}</Tag>
                <span className="v-num text-[11px] text-[#8f8f8f]">
                  {[
                    l.rpe != null ? `RPE ${l.rpe}` : null,
                    l.max_spell_balls != null ? `spell ${l.max_spell_balls}` : null,
                    l.soreness_lumbar != null ? `back ${l.soreness_lumbar}/10` : null,
                    l.pain_when_bowling ? "pain ◆" : null,
                  ]
                    .filter(Boolean)
                    .join(" · ")}
                </span>
              </div>
              {confirmId === l.id ? (
                <span className="flex gap-1.5">
                  <button type="button" className="v-btn-primary h-7 px-2.5 text-[12px]" disabled={busyId === l.id} onClick={() => del(l.id)}>
                    {busyId === l.id ? "Deleting…" : "Delete"}
                  </button>
                  <button type="button" className="v-btn-ghost h-7 px-2.5 text-[12px]" onClick={() => setConfirmId(null)}>
                    Keep
                  </button>
                </span>
              ) : (
                <button
                  type="button"
                  className="v-btn-ghost h-7 px-2.5 text-[12px]"
                  aria-label={`Delete log from ${shortDate(l.log_date)}`}
                  onClick={() => setConfirmId(l.id)}
                >
                  Remove
                </button>
              )}
            </li>
          ))}
        </ul>
      )}
      {sorted.length > 10 && (
        <button type="button" className="text-[12px] text-[#4d4d4d] underline underline-offset-2 mt-2" onClick={() => setShowAll((s) => !s)}>
          {showAll ? "Show recent only" : `Show all ${sorted.length}`}
        </button>
      )}
      {err && (
        <div className="text-[12px] text-[#171717] mt-2" role="alert">
          Couldn't delete: {err}
        </div>
      )}
    </div>
  );
};
