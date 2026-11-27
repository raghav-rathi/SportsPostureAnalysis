/**
 * Studio v5 "Paper Instrument" primitives (docs/research_app_landscape_v5.md §6.4).
 *
 * Chrome stays monochrome on paper; segment hues appear only inside rails / timelines / charts.
 * Status is encoded by glyph + words (never by segment hue). Every number wears its interval.
 *
 * Exports: RangeRail, ObsBadge, StatusPill, MetricTile, MetricRow, JogWheel, SpeedDetent, PhaseTimeline,
 *          UnlockStrip, GuardrailStrip, SectionLabel, + formatting helpers (fmtMetric, unitSuffix, autoRailDomain,
 *          buildPhases, frameAtTimeRelFfc, msPerFrame).
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { CheckpointStatus, DeliveryEvents, MetricEstimate, ObservabilityGrade } from "../../types";
import { GUARDRAIL_ICC, GUARDRAIL_MEDICAL, OBS_GLYPH, SEG, STATUS_GLYPH } from "../../theme";

// ─────────────────────────────────────────────────────────────────────────────
// Formatting helpers
// ─────────────────────────────────────────────────────────────────────────────

/** Display suffix for a backend unit string. */
export function unitSuffix(unit: string | null | undefined): string {
  const u = (unit || "").trim();
  if (!u) return "";
  if (u === "deg" || u === "°" || u === "degrees") return "°";
  if (u === "deg/s" || u === "dps" || u === "°/s") return "°/s";
  if (u === "%" || u === "pct") return "%";
  if (u === "% stature" || u === "pct_stature" || u === "%stature") return "% height";
  if (u === "ratio" || u === "x" || u === "×") return "×";
  if (u === "mps") return " m/s";
  return ` ${u}`;
}

function decimalsFor(unit: string | null | undefined, v: number): number {
  const u = (unit || "").trim();
  if (u === "m") return 2;
  if (u === "m/s" || u === "mps") return 1;
  if (u === "ratio" || u === "x" || u === "×" || u === "") return Math.abs(v) < 10 ? 2 : 1;
  if (u === "% stature" || u === "%") return Math.abs(v) < 10 ? 1 : 0;
  if (u === "cm") return Math.abs(v) < 10 ? 1 : 0;
  return 0; // deg, deg/s, ms, km/h
}

export function fmtNum(v: number | null | undefined, unit?: string | null): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  return v.toFixed(decimalsFor(unit, v));
}

/** `{ valueText: "171°", rangeText: "161–180°" }` from a MetricEstimate (null texts when suppressed). */
export function fmtMetric(m: MetricEstimate | null | undefined): { valueText: string | null; rangeText: string | null } {
  if (!m || m.suppressed || m.value === null || !Number.isFinite(m.value)) return { valueText: null, rangeText: null };
  const s = unitSuffix(m.unit);
  const valueText = `${fmtNum(m.value, m.unit)}${s}`;
  const rangeText =
    m.ci_low !== null && m.ci_high !== null && Number.isFinite(m.ci_low) && Number.isFinite(m.ci_high)
      ? `${fmtNum(m.ci_low, m.unit)}–${fmtNum(m.ci_high, m.unit)}${s}`
      : null;
  return { valueText, rangeText };
}

/** True when the metric carries a usable (non-suppressed) value. */
export function isMeasured(m: MetricEstimate | null | undefined): m is MetricEstimate & { value: number } {
  return !!m && !m.suppressed && m.value !== null && Number.isFinite(m.value) && m.observability !== "unmeasurable" && m.observability !== "refused";
}

const niceStep = (span: number) => {
  const raw = span / 4;
  const p = Math.pow(10, Math.floor(Math.log10(Math.max(raw, 1e-9))));
  const n = raw / p;
  return (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * p;
};

/** A calm rail domain that contains the value, its interval and an optional reference band.
 *  Joint angles default to the physical [0°, 180°] range (AGENTS.md §1.2.4). */
export function autoRailDomain(
  value: number | null,
  lo: number | null,
  hi: number | null,
  unit?: string | null,
  band?: [number, number] | null
): [number, number] {
  const u = (unit || "").trim();
  const pts = [value, lo, hi, band?.[0], band?.[1]].filter((x): x is number => x !== null && x !== undefined && Number.isFinite(x));
  if (u === "deg" || u === "°") {
    const mn = Math.min(0, ...pts);
    const mx = Math.max(180, ...pts);
    if (mn >= 0 && mx <= 180) return [0, 180];
    const step = niceStep(mx - mn);
    return [Math.floor(mn / step) * step, Math.ceil(mx / step) * step];
  }
  if (pts.length === 0) return [0, 1];
  let mn = Math.min(...pts);
  let mx = Math.max(...pts);
  if (mn >= 0) mn = 0;
  if (mx === mn) mx = mn + 1;
  const pad = (mx - mn) * 0.15;
  mx += pad;
  if (mn < 0) mn -= pad;
  const step = niceStep(mx - mn);
  return [Math.floor(mn / step) * step, Math.ceil(mx / step) * step];
}

/** Milliseconds per analysis frame derived from `time_rel_ffc_ms`. */
export function msPerFrame(timeRelFfcMs: number[], fallbackFps = 30): number {
  const n = timeRelFfcMs.length;
  if (n >= 2) {
    const span = timeRelFfcMs[n - 1] - timeRelFfcMs[0];
    if (Number.isFinite(span) && span > 0) return span / (n - 1);
  }
  return 1000 / fallbackFps;
}

/** Fractional frame index for a time relative to FFC (linear interpolation on `time_rel_ffc_ms`). */
export function frameAtTimeRelFfc(timeRelFfcMs: number[], tMs: number): number {
  const n = timeRelFfcMs.length;
  if (n === 0) return 0;
  if (tMs <= timeRelFfcMs[0]) return 0;
  if (tMs >= timeRelFfcMs[n - 1]) return n - 1;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (timeRelFfcMs[mid] <= tMs) lo = mid;
    else hi = mid;
  }
  const a = timeRelFfcMs[lo];
  const b = timeRelFfcMs[hi];
  return lo + (b > a ? (tMs - a) / (b - a) : 0);
}

export interface PhaseSpan {
  id: "runup" | "stride" | "braking" | "whip" | "follow";
  label: string;
  start: number;
  end: number;
  fill: string;
}

export interface EventTick {
  id: "bfc" | "ffc" | "uah" | "br" | "ft";
  short: string;
  label: string;
  frame: number;
  uncertaintyMs: number;
}

/** Phase spans + event ticks from DeliveryEvents (clamped to [0, numFrames-1]). */
export function buildPhases(events: DeliveryEvents, numFrames: number): { phases: PhaseSpan[]; ticks: EventTick[] } {
  const last = Math.max(0, numFrames - 1);
  const c = (f: number) => Math.max(0, Math.min(last, f));
  const bfc = c(events.bfc.frame_float ?? events.bfc.frame_idx);
  const ffc = c(events.ffc.frame_float ?? events.ffc.frame_idx);
  const br = c(events.br.frame_float ?? events.br.frame_idx);
  const uah = events.uah ? c(events.uah.frame_float ?? events.uah.frame_idx) : null;
  const ft = events.ft_peak ? c(events.ft_peak.frame_float ?? events.ft_peak.frame_idx) : null;
  const phases: PhaseSpan[] = [
    { id: "runup", label: "Run-up", start: 0, end: bfc, fill: "#ECECEC" },
    { id: "stride", label: "Stride", start: bfc, end: ffc, fill: SEG.pelvisTint },
    ...(uah !== null && uah > ffc && uah < br
      ? [
          { id: "braking" as const, label: "Brace", start: ffc, end: uah, fill: SEG.thoraxTint },
          { id: "whip" as const, label: "Whip", start: uah, end: br, fill: SEG.armTint },
        ]
      : [{ id: "braking" as const, label: "Brace & whip", start: ffc, end: br, fill: SEG.thoraxTint }]),
    { id: "follow", label: "Follow-through", start: br, end: last, fill: "#F1F1F1" },
  ];
  const ticks: EventTick[] = [
    { id: "bfc", short: "BFC", label: "Back foot lands", frame: bfc, uncertaintyMs: events.bfc.timing_uncertainty_ms },
    { id: "ffc", short: "FFC", label: "Front foot plants", frame: ffc, uncertaintyMs: events.ffc.timing_uncertainty_ms },
    ...(uah !== null && events.uah ? [{ id: "uah" as const, short: "UAH", label: "Arm level", frame: uah, uncertaintyMs: events.uah.timing_uncertainty_ms }] : []),
    { id: "br", short: "BR", label: "Ball release", frame: br, uncertaintyMs: events.br.timing_uncertainty_ms },
    ...(ft !== null && events.ft_peak ? [{ id: "ft" as const, short: "FT", label: "Follow-through", frame: ft, uncertaintyMs: events.ft_peak.timing_uncertainty_ms }] : []),
  ];
  return { phases: phases.filter((p) => p.end > p.start), ticks };
}

// ─────────────────────────────────────────────────────────────────────────────
// SectionLabel (TE-style mono eyebrow)
// ─────────────────────────────────────────────────────────────────────────────
export const SectionLabel: React.FC<{ children: React.ReactNode; right?: React.ReactNode; className?: string }> = ({
  children,
  right,
  className = "",
}) => (
  <div className={`flex items-center justify-between gap-2 ${className}`}>
    <span className="v-label">{children}</span>
    {right}
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// RangeRail
// ─────────────────────────────────────────────────────────────────────────────
export interface RangeRailProps {
  value: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  min: number;
  max: number;
  color?: string;
  cohortBand?: [number, number] | null;
  goalBand?: [number, number] | null;
  width?: number;
  /** Optional reference markers (e.g. research threshold lines). */
  markers?: Array<{ at: number; label?: string }>;
  className?: string;
}

export const RangeRail: React.FC<RangeRailProps> = ({
  value,
  ciLow,
  ciHigh,
  min,
  max,
  color = SEG.ink,
  cohortBand = null,
  goalBand = null,
  width,
  markers,
  className = "",
}) => {
  const W = 120;
  const H = 12;
  const span = max - min || 1;
  const x = (v: number) => Math.max(0, Math.min(W, ((v - min) / span) * W));
  const ok = (v: number | null): v is number => v !== null && Number.isFinite(v);
  const label =
    ok(value) && ok(ciLow) && ok(ciHigh)
      ? `Value ${value.toFixed(1)}, likely between ${ciLow.toFixed(1)} and ${ciHigh.toFixed(1)}`
      : ok(value)
      ? `Value ${value.toFixed(1)}`
      : "Not measured";
  return (
    <svg
      viewBox={`-4 0 ${W + 8} ${H}`}
      width={width}
      height={width ? (width * H) / (W + 8) : undefined}
      preserveAspectRatio="none"
      className={`block ${width ? "" : "w-full"} h-3 overflow-visible ${className}`}
      role="img"
      aria-label={label}
    >
      <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke="#D4D4D4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      <line x1={0} y1={H / 2 - 2.5} x2={0} y2={H / 2 + 2.5} stroke="#D4D4D4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      <line x1={W} y1={H / 2 - 2.5} x2={W} y2={H / 2 + 2.5} stroke="#D4D4D4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
      {cohortBand && ok(cohortBand[0]) && ok(cohortBand[1]) && (
        <rect x={x(cohortBand[0])} y={1} width={Math.max(0.5, x(cohortBand[1]) - x(cohortBand[0]))} height={H - 2} fill={SEG.ghostTint} rx={1}>
          <title>Published group range</title>
        </rect>
      )}
      {goalBand && ok(goalBand[0]) && ok(goalBand[1]) && (
        <rect
          x={x(goalBand[0])}
          y={0.5}
          width={Math.max(0.5, x(goalBand[1]) - x(goalBand[0]))}
          height={H - 1}
          fill="none"
          stroke={SEG.tele}
          strokeDasharray="2 1.5"
          strokeWidth={1}
          vectorEffect="non-scaling-stroke"
          rx={1}
        >
          <title>Your goal band</title>
        </rect>
      )}
      {(markers ?? []).map((m, i) => (
        <line key={i} x1={x(m.at)} x2={x(m.at)} y1={1} y2={H - 1} stroke="#8F8F8F" strokeDasharray="1.5 1.5" strokeWidth={1} vectorEffect="non-scaling-stroke">
          {m.label && <title>{m.label}</title>}
        </line>
      ))}
      {ok(ciLow) && ok(ciHigh) && (
        <rect x={x(ciLow)} y={H / 2 - 2} width={Math.max(1, x(ciHigh) - x(ciLow))} height={4} rx={2} fill={color} opacity={0.38} />
      )}
      {ok(value) && <circle cx={x(value)} cy={H / 2} r={3.2} fill={color} stroke="#FFFFFF" strokeWidth={1.2} />}
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// ObsBadge
// ─────────────────────────────────────────────────────────────────────────────
const OBS_TOOLTIP: Record<ObservabilityGrade, string> = {
  high: "Clearly visible from this camera angle — the likely range is tight.",
  moderate: "Reasonably visible — some uncertainty from camera angle, depth or timing is included in the range.",
  low: "Only roughly visible — camera angle, depth or frame rate widen the likely range a lot.",
  unmeasurable: "Not measurable from this clip (camera angle, frame rate or body out of frame).",
  refused: "Permanently refused: elbow legality needs an accredited laboratory 3D test at 250 Hz or faster.",
};
const OBS_TONE: Record<ObservabilityGrade, string> = {
  high: "text-[#171717]",
  moderate: "text-[#4d4d4d]",
  low: "text-[#8f8f8f]",
  unmeasurable: "text-[#a8a8a8]",
  refused: "text-[#4d4d4d]",
};

export const ObsBadge: React.FC<{ grade: ObservabilityGrade; detail?: string; compact?: boolean }> = ({ grade, detail, compact }) => {
  const g = OBS_GLYPH[grade] ?? OBS_GLYPH.unmeasurable;
  const tip = detail ? `${OBS_TOOLTIP[grade]} ${detail}` : OBS_TOOLTIP[grade];
  return (
    <span
      className={`inline-flex items-center gap-1 whitespace-nowrap ${OBS_TONE[grade]} ${compact ? "text-[11px]" : "text-[12px]"}`}
      title={tip}
      aria-label={`Visibility: ${g.word}. ${tip}`}
    >
      <span className="v-num tracking-[-0.08em] leading-none" aria-hidden>
        {g.bars}
      </span>
      {!compact && <span className="leading-none">{g.word}</span>}
    </span>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// StatusPill (glyph + words; never segment hues)
// ─────────────────────────────────────────────────────────────────────────────
export const STATUS_WORD: Record<CheckpointStatus, string> = {
  strong: "Strong",
  on_track: "On track",
  watch: "Watch",
  focus: "Focus",
  not_visible: "Not visible",
};

export const StatusPill: React.FC<{ status: CheckpointStatus; label?: string; className?: string }> = ({ status, label, className = "" }) => {
  const base = "inline-flex items-center gap-1 h-[20px] px-2 rounded-full text-[11.5px] font-[500] whitespace-nowrap leading-none";
  const style: Record<CheckpointStatus, string> = {
    strong: "bg-[#ffffff] text-[#297a3a] shadow-[0_0_0_1px_#cfe3d3]",
    on_track: "bg-[#ffffff] text-[#171717] shadow-[0_0_0_1px_#ebebeb]",
    watch: "bg-[#ffffff] text-[#171717] border border-dashed border-[#a8a8a8]",
    focus: "bg-[#171717] text-[#ffffff]",
    not_visible: "bg-transparent text-[#8f8f8f] border border-dashed border-[#d4d4d4]",
  };
  return (
    <span className={`${base} ${style[status]} ${className}`} title={label ?? STATUS_WORD[status]}>
      <span aria-hidden className="text-[10px]">
        {STATUS_GLYPH[status]}
      </span>
      {label ?? STATUS_WORD[status]}
    </span>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// MetricTile
// ─────────────────────────────────────────────────────────────────────────────
export interface MetricTileProps {
  label: string;
  valueText: string;
  rangeText?: string;
  unit?: string;
  range?: {
    value: number | null;
    ciLow: number | null;
    ciHigh: number | null;
    min: number;
    max: number;
    color?: string;
    cohortBand?: [number, number] | null;
  };
  observability: ObservabilityGrade;
  suppressed?: boolean;
  suppressionReason?: string | null;
  status?: CheckpointStatus;
  onSeeMoment?: () => void;
  accentColor?: string;
  hero?: boolean;
  active?: boolean;
  className?: string;
}

export const MetricTile: React.FC<MetricTileProps> = ({
  label,
  valueText,
  rangeText,
  unit,
  range,
  observability,
  suppressed,
  suppressionReason,
  status,
  onSeeMoment,
  accentColor,
  hero,
  active,
  className = "",
}) => {
  const interactive = !!onSeeMoment && !suppressed;
  const body = (
    <>
      <div className="flex items-start justify-between gap-2">
        <span className="v-label truncate" title={label}>
          {accentColor && <span className="inline-block w-1.5 h-1.5 rounded-full mr-1.5 align-middle" style={{ background: accentColor }} />}
          {label}
        </span>
        {status && !suppressed && <StatusPill status={status} />}
      </div>
      {suppressed ? (
        <div className="mt-2">
          <div className={`${hero ? "text-[28px]" : "text-[20px]"} v-hero-num text-[#c9c9c9]`}>—</div>
          <p className="mt-1.5 text-[12px] leading-snug text-[#8f8f8f]">{suppressionReason || "Not measurable on this clip."}</p>
          <div className="mt-1.5">
            <ObsBadge grade={observability === "refused" ? "refused" : "unmeasurable"} />
          </div>
        </div>
      ) : (
        <>
          <div className="mt-1.5 flex items-baseline gap-1.5">
            <span className={`v-hero-num text-[#171717] ${hero ? "text-[44px] md:text-[48px]" : "text-[26px]"}`}>{valueText}</span>
            {unit && <span className="text-[13px] text-[#8f8f8f]">{unit}</span>}
          </div>
          {rangeText && <div className="v-num text-[12.5px] text-[#8f8f8f] mt-1">likely {rangeText}</div>}
          {range && (
            <div className="mt-2">
              <RangeRail {...range} color={range.color ?? accentColor ?? SEG.ink} />
            </div>
          )}
          <div className="mt-2 flex items-center justify-between gap-2">
            <ObsBadge grade={observability} />
            {interactive && <span className="text-[12px] text-[#4d4d4d] underline underline-offset-2 decoration-[#d4d4d4]">See moment</span>}
          </div>
        </>
      )}
    </>
  );
  const cls = `v-panel block w-full text-left ${hero ? "p-4" : "p-3"} ${
    active ? "shadow-[0_0_0_1.5px_#171717]" : ""
  } ${suppressed ? "bg-[#fafafa]" : ""} ${className}`;
  return interactive ? (
    <button type="button" onClick={onSeeMoment} className={`${cls} hover:shadow-[0_0_0_1px_#a8a8a8] transition-shadow`}>
      {body}
    </button>
  ) : (
    <div className={cls}>{body}</div>
  );
};

/** Compact single-line metric row for deep cards (ghosted with its reason when suppressed — never omitted). */
export const MetricRow: React.FC<{
  label: string;
  metric: MetricEstimate | null | undefined;
  color?: string;
  domain?: [number, number];
  cohortBand?: [number, number] | null;
  markers?: Array<{ at: number; label?: string }>;
}> = ({ label, metric, color = SEG.ink, domain, cohortBand, markers }) => {
  const measured = isMeasured(metric);
  const { valueText, rangeText } = fmtMetric(metric);
  const [mn, mx] = domain ?? autoRailDomain(metric?.value ?? null, metric?.ci_low ?? null, metric?.ci_high ?? null, metric?.unit, cohortBand ?? null);
  return (
    <div className={`py-2 ${measured ? "" : "opacity-70"}`}>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-[12.5px] text-[#4d4d4d] min-w-0">{label}</span>
        {measured ? (
          <span className="flex items-baseline gap-1.5 shrink-0">
            <span className="v-num text-[13.5px] text-[#171717]">{valueText}</span>
            {rangeText && <span className="v-num text-[11px] text-[#8f8f8f]">{rangeText}</span>}
            <ObsBadge grade={metric!.observability} compact />
          </span>
        ) : (
          <span className="shrink-0">
            <ObsBadge grade={metric?.observability === "refused" ? "refused" : "unmeasurable"} compact />
          </span>
        )}
      </div>
      {measured ? (
        <div className="mt-1">
          <RangeRail value={metric!.value} ciLow={metric!.ci_low} ciHigh={metric!.ci_high} min={mn} max={mx} color={color} cohortBand={cohortBand} markers={markers} />
        </div>
      ) : (
        <p className="text-[11.5px] text-[#8f8f8f] leading-snug mt-0.5">{metric?.suppression_reason || "Not measurable on this clip."}</p>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// JogWheel (Teenage Engineering knurled dial)
// ─────────────────────────────────────────────────────────────────────────────
export interface JogWheelProps {
  frame: number;
  numFrames: number;
  playing: boolean;
  onTogglePlay: () => void;
  onStep: (delta: number) => void;
  onLongPressLoop?: () => void;
  size?: number;
  className?: string;
}

const DEG_PER_FRAME = 6;

export const JogWheel: React.FC<JogWheelProps> = ({ frame, numFrames, playing, onTogglePlay, onStep, onLongPressLoop, size, className = "" }) => {
  const ref = useRef<HTMLDivElement | null>(null);
  const drag = useRef<{ lastAngle: number; acc: number; lastT: number; vel: number } | null>(null);
  const inertia = useRef<number | null>(null);
  const press = useRef<{ timer: number | null; fired: boolean }>({ timer: null, fired: false });
  const [dragging, setDragging] = useState(false);
  const [isSmall, setIsSmall] = useState(() => typeof window !== "undefined" && window.innerWidth < 768);

  useEffect(() => {
    const on = () => setIsSmall(window.innerWidth < 768);
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  useEffect(
    () => () => {
      if (inertia.current) cancelAnimationFrame(inertia.current);
      if (press.current.timer) window.clearTimeout(press.current.timer);
    },
    []
  );

  const px = size ?? (isSmall ? 64 : 72);
  const angleOf = (e: { clientX: number; clientY: number }) => {
    const r = ref.current!.getBoundingClientRect();
    return (Math.atan2(e.clientY - (r.top + r.height / 2), e.clientX - (r.left + r.width / 2)) * 180) / Math.PI;
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if ((e.target as HTMLElement).closest("[data-jog-center]")) return;
    e.preventDefault();
    if (inertia.current) cancelAnimationFrame(inertia.current);
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    drag.current = { lastAngle: angleOf(e), acc: 0, lastT: performance.now(), vel: 0 };
    setDragging(true);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    const d = drag.current;
    if (!d) return;
    const a = angleOf(e);
    let delta = a - d.lastAngle;
    if (delta > 180) delta -= 360;
    if (delta < -180) delta += 360;
    const now = performance.now();
    const dt = Math.max(1, now - d.lastT);
    d.vel = 0.7 * d.vel + 0.3 * (delta / dt);
    d.lastT = now;
    d.lastAngle = a;
    d.acc += delta;
    const steps = Math.trunc(d.acc / DEG_PER_FRAME);
    if (steps !== 0) {
      d.acc -= steps * DEG_PER_FRAME;
      onStep(steps);
    }
  };
  const endDrag = () => {
    const d = drag.current;
    drag.current = null;
    setDragging(false);
    if (!d || Math.abs(d.vel) < 0.25) return;
    // Flick → inertial scrub, decaying.
    let vel = d.vel; // deg per ms
    let acc = 0;
    let last = performance.now();
    const tick = () => {
      const now = performance.now();
      const dt = now - last;
      last = now;
      acc += vel * dt;
      const steps = Math.trunc(acc / DEG_PER_FRAME);
      if (steps !== 0) {
        acc -= steps * DEG_PER_FRAME;
        onStep(steps);
      }
      vel *= Math.pow(0.992, dt);
      if (Math.abs(vel) > 0.05) inertia.current = requestAnimationFrame(tick);
      else inertia.current = null;
    };
    inertia.current = requestAnimationFrame(tick);
  };

  const centerDown = () => {
    press.current.fired = false;
    if (onLongPressLoop) {
      press.current.timer = window.setTimeout(() => {
        press.current.fired = true;
        onLongPressLoop();
      }, 550);
    }
  };
  const centerUp = () => {
    if (press.current.timer) window.clearTimeout(press.current.timer);
    press.current.timer = null;
    if (!press.current.fired) onTogglePlay();
  };

  const rot = (frame * DEG_PER_FRAME) % 360;
  const ticks = 36;
  const knurl = 48;
  const progress = numFrames > 1 ? frame / (numFrames - 1) : 0;
  const R = 50;

  const arrowCls =
    "v-btn-ghost w-7 h-7 flex items-center justify-center text-[13px] text-[#4d4d4d] select-none active:bg-[#ebebeb]";

  return (
    <div className={`inline-flex items-center gap-1.5 ${className}`}>
      <button type="button" className={arrowCls} onClick={() => onStep(-1)} aria-label="Previous frame" title="Previous frame (,)">
        ‹
      </button>
      <div
        ref={ref}
        role="slider"
        tabIndex={0}
        aria-label="Jog wheel: drag in a circle to scrub frames"
        aria-valuemin={0}
        aria-valuemax={Math.max(0, numFrames - 1)}
        aria-valuenow={frame}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight" || e.key === "ArrowUp") {
            e.preventDefault();
            e.stopPropagation();
            onStep(1);
          } else if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
            e.preventDefault();
            e.stopPropagation();
            onStep(-1);
          }
        }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        className={`relative shrink-0 rounded-full touch-none select-none ${dragging ? "cursor-grabbing" : "cursor-grab"}`}
        style={{ width: px, height: px }}
      >
        <svg viewBox="-60 -60 120 120" width={px} height={px} className="block">
          <defs>
            <radialGradient id="jw-face" cx="40%" cy="35%" r="75%">
              <stop offset="0%" stopColor="#FFFFFF" />
              <stop offset="100%" stopColor="#EDEDED" />
            </radialGradient>
          </defs>
          {/* progress arc */}
          <circle r={57} fill="none" stroke="#EBEBEB" strokeWidth={2} />
          <circle
            r={57}
            fill="none"
            stroke="#171717"
            strokeWidth={2}
            strokeDasharray={`${2 * Math.PI * 57 * progress} ${2 * Math.PI * 57}`}
            transform="rotate(-90)"
            strokeLinecap="round"
          />
          {/* fixed tick marks */}
          {Array.from({ length: ticks }).map((_, i) => {
            const a = (i / ticks) * Math.PI * 2;
            const long = i % 9 === 0;
            return (
              <line
                key={i}
                x1={Math.cos(a) * 53}
                y1={Math.sin(a) * 53}
                x2={Math.cos(a) * (long ? 49 : 51)}
                y2={Math.sin(a) * (long ? 49 : 51)}
                stroke={long ? "#8F8F8F" : "#C9C9C9"}
                strokeWidth={long ? 1.2 : 0.8}
              />
            );
          })}
          {/* rotating knurled ring */}
          <g transform={`rotate(${rot})`} style={{ transition: dragging ? "none" : "transform 80ms linear" }}>
            <circle r={R - 3} fill="url(#jw-face)" stroke="#D4D4D4" strokeWidth={1} />
            {Array.from({ length: knurl }).map((_, i) => {
              const a = (i / knurl) * Math.PI * 2;
              return (
                <line
                  key={i}
                  x1={Math.cos(a) * (R - 4)}
                  y1={Math.sin(a) * (R - 4)}
                  x2={Math.cos(a) * (R - 11)}
                  y2={Math.sin(a) * (R - 11)}
                  stroke="#BDBDBD"
                  strokeWidth={1.4}
                  strokeLinecap="round"
                />
              );
            })}
            <circle cx={0} cy={-(R - 16)} r={2.6} fill="#171717" />
          </g>
        </svg>
        <button
          type="button"
          data-jog-center
          onPointerDown={centerDown}
          onPointerUp={centerUp}
          onPointerLeave={() => {
            if (press.current.timer) window.clearTimeout(press.current.timer);
            press.current.timer = null;
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              e.stopPropagation();
              onTogglePlay();
            }
          }}
          aria-label={playing ? "Pause" : "Play"}
          title={onLongPressLoop ? `${playing ? "Pause" : "Play"} · long-press to loop this phase` : playing ? "Pause" : "Play"}
          className="absolute rounded-full bg-[#171717] text-[#ffffff] flex items-center justify-center hover:bg-[#333333] transition-colors"
          style={{ width: px * 0.38, height: px * 0.38, left: px * 0.31, top: px * 0.31 }}
        >
          {playing ? (
            <svg viewBox="0 0 10 10" width={px * 0.13} height={px * 0.13} aria-hidden>
              <rect x="1.5" y="1" width="2.4" height="8" fill="currentColor" />
              <rect x="6.1" y="1" width="2.4" height="8" fill="currentColor" />
            </svg>
          ) : (
            <svg viewBox="0 0 10 10" width={px * 0.13} height={px * 0.13} aria-hidden>
              <path d="M2.2 1 L9 5 L2.2 9 Z" fill="currentColor" />
            </svg>
          )}
        </button>
      </div>
      <button type="button" className={arrowCls} onClick={() => onStep(1)} aria-label="Next frame" title="Next frame (.)">
        ›
      </button>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// SpeedDetent
// ─────────────────────────────────────────────────────────────────────────────
export type SpeedRate = 0.1 | 0.25 | 0.5 | 1;
const RATES: SpeedRate[] = [0.1, 0.25, 0.5, 1];

export const SpeedDetent: React.FC<{ rate: SpeedRate; onChange: (r: SpeedRate) => void; captureFps?: number; showHint?: boolean }> = ({
  rate,
  onChange,
  captureFps,
  showHint = true,
}) => (
  <div className="inline-flex flex-col items-start gap-0.5">
    <div className="v-seg" role="group" aria-label="Playback speed">
      {RATES.map((r) => (
        <button key={r} type="button" aria-pressed={rate === r} onClick={() => onChange(r)} className="v-num">
          {r === 1 ? "1×" : `${r}×`}
        </button>
      ))}
    </div>
    {showHint && captureFps && captureFps > 0 ? (
      <span className="v-num text-[10.5px] text-[#8f8f8f] pl-1 whitespace-nowrap">
        {rate}× of {Math.round(captureFps)} fps → {Math.max(1, Math.round(captureFps * rate))} fps on screen
      </span>
    ) : null}
  </div>
);

// ─────────────────────────────────────────────────────────────────────────────
// PhaseTimeline
// ─────────────────────────────────────────────────────────────────────────────
export interface PhaseTimelineProps {
  numFrames: number;
  currentFrame: number;
  events: DeliveryEvents;
  timeRelFfcMs: number[];
  onSeek: (frame: number) => void;
  noteFrames?: number[];
  loopRange?: [number, number] | null;
  className?: string;
}

export const PhaseTimeline: React.FC<PhaseTimelineProps> = ({
  numFrames,
  currentFrame,
  events,
  timeRelFfcMs,
  onSeek,
  noteFrames = [],
  loopRange = null,
  className = "",
}) => {
  const trackRef = useRef<HTMLDivElement | null>(null);
  const scrubbing = useRef(false);
  const last = Math.max(1, numFrames - 1);
  const { phases, ticks } = useMemo(() => buildPhases(events, numFrames), [events, numFrames]);
  const mspf = msPerFrame(timeRelFfcMs);
  const pctNum = (f: number) => (Math.max(0, Math.min(last, f)) / last) * 100;
  const pct = (f: number) => `${pctNum(f)}%`;

  // Stagger event tick labels onto two rows whenever adjacent ticks sit within < 7.5% horizontally.
  const staggeredTicks = useMemo(() => {
    const sorted = [...ticks].sort((a, b) => a.frame - b.frame);
    let lastRow0 = -999;
    return sorted.map((t) => {
      const p = pctNum(t.frame);
      const row: 0 | 1 = p - lastRow0 < 7.5 ? 1 : 0;
      if (row === 0) lastRow0 = p;
      const align = p < 4 ? "translate-x-0" : p > 96 ? "-translate-x-full" : "-translate-x-1/2";
      return { ...t, p, row, align };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticks, last]);
  const hasStagger = staggeredTicks.some((t) => t.row === 1);

  const seekFromX = (clientX: number) => {
    const el = trackRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    const t = Math.max(0, Math.min(1, (clientX - r.left) / Math.max(1, r.width)));
    onSeek(Math.round(t * last));
  };

  return (
    <div className={`select-none ${className}`}>
      {/* phase names */}
      <div className="relative h-4 mb-0.5">
        {phases.map((p) => {
          const w = ((p.end - p.start) / last) * 100;
          return (
            <span
              key={p.id}
              className="absolute top-0 text-[10.5px] text-[#666666] truncate px-1"
              style={{ left: pct(p.start), width: `${w}%` }}
              title={p.label}
            >
              {w > 7 ? p.label : ""}
            </span>
          );
        })}
      </div>
      <div
        ref={trackRef}
        role="slider"
        aria-label="Delivery timeline"
        aria-valuemin={0}
        aria-valuemax={last}
        aria-valuenow={currentFrame}
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Home") onSeek(0);
          if (e.key === "End") onSeek(last);
        }}
        onPointerDown={(e) => {
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          scrubbing.current = true;
          seekFromX(e.clientX);
        }}
        onPointerMove={(e) => scrubbing.current && seekFromX(e.clientX)}
        onPointerUp={() => (scrubbing.current = false)}
        onPointerCancel={() => (scrubbing.current = false)}
        className="relative h-7 rounded-[5px] overflow-visible cursor-pointer touch-none"
        style={{ boxShadow: "inset 0 0 0 1px #EBEBEB" }}
      >
        {phases.map((p) => (
          <div
            key={p.id}
            className="absolute top-0 bottom-0"
            style={{ left: pct(p.start), width: `${((p.end - p.start) / last) * 100}%`, background: p.fill }}
          />
        ))}
        {loopRange && (
          <div
            className="absolute -top-0.5 -bottom-0.5 rounded-[3px] pointer-events-none"
            style={{
              left: pct(loopRange[0]),
              width: `${((loopRange[1] - loopRange[0]) / last) * 100}%`,
              boxShadow: "inset 0 0 0 1.5px #171717",
            }}
            title="Looping"
          />
        )}
        {/* event ticks + uncertainty whiskers */}
        {ticks.map((t) => {
          const halfFrames = (1.96 * (t.uncertaintyMs || 0)) / mspf;
          const a = Math.max(0, t.frame - halfFrames);
          const b = Math.min(last, t.frame + halfFrames);
          return (
            <React.Fragment key={t.id}>
              {halfFrames > 0.2 && (
                <div
                  className="absolute top-1/2 h-[6px] -translate-y-1/2 rounded-full bg-[#171717]/15 pointer-events-none"
                  style={{ left: pct(a), width: `${((b - a) / last) * 100}%` }}
                  title={`${t.label} timing ± ${Math.round(1.96 * t.uncertaintyMs)} ms`}
                />
              )}
              <div className="absolute top-0 bottom-0 w-px bg-[#171717] -translate-x-1/2 pointer-events-none" style={{ left: pct(t.frame) }} />
            </React.Fragment>
          );
        })}
        {/* note pins */}
        {noteFrames.map((f, i) => (
          <div
            key={`${f}-${i}`}
            className="absolute -top-1.5 w-2 h-2 rotate-45 -translate-x-1/2 pointer-events-none"
            style={{ left: pct(f), background: SEG.tele }}
            title="Note"
          />
        ))}
        {/* playhead */}
        <div className="absolute -top-1 -bottom-1 pointer-events-none" style={{ left: pct(currentFrame) }}>
          <div className="absolute top-0 bottom-0 w-[2px] -translate-x-1/2 bg-[#171717]" />
          <div className="absolute -top-[3px] w-[9px] h-[9px] -translate-x-1/2 rounded-full bg-[#171717] ring-2 ring-white" />
        </div>
      </div>
      {/* tick labels (auto-staggered when < 7.5% apart so FFC/UAH/BR never overlap) */}
      <div className={`relative mt-1 ${hasStagger ? "h-8" : "h-5"}`}>
        {staggeredTicks.map((t) => (
          <React.Fragment key={t.id}>
            {t.row === 1 && (
              <div
                className="absolute top-[-4px] h-[19px] w-px bg-[#d4d4d4] -translate-x-1/2 pointer-events-none"
                style={{ left: `${t.p}%` }}
              />
            )}
            <button
              type="button"
              onClick={() => onSeek(Math.round(t.frame))}
              className={`absolute ${t.align} v-num text-[10.5px] leading-none text-[#171717] hover:underline px-0.5 bg-[#ffffff]/80 rounded-[2px]`}
              style={{ left: `${t.p}%`, top: t.row === 1 ? 15 : 1 }}
              title={`${t.label}${t.uncertaintyMs ? ` (± ${Math.round(1.96 * t.uncertaintyMs)} ms)` : ""}`}
            >
              {t.short}
            </button>
          </React.Fragment>
        ))}
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// UnlockStrip
// ─────────────────────────────────────────────────────────────────────────────
export const UnlockStrip: React.FC<{ missingCount: number; hints: string[]; onOpenGuide?: () => void; className?: string }> = ({
  missingCount,
  hints,
  onOpenGuide,
  className = "",
}) => {
  const [open, setOpen] = useState(false);
  if (missingCount <= 0) return null;
  const uniq = Array.from(new Set(hints.filter(Boolean)));
  return (
    <div className={`rounded-[6px] border border-dashed border-[#d4d4d4] bg-[#fafafa] ${className}`}>
      <div className="flex items-center gap-2.5 px-3 py-2.5">
        <svg viewBox="0 0 20 20" width={18} height={18} className="shrink-0 text-[#4d4d4d]" aria-hidden>
          <rect x="2" y="5.5" width="16" height="11" rx="2" fill="none" stroke="currentColor" strokeWidth="1.4" />
          <path d="M7 5.5 L8.2 3.5 H11.8 L13 5.5" fill="none" stroke="currentColor" strokeWidth="1.4" />
          <circle cx="10" cy="11" r="3" fill="none" stroke="currentColor" strokeWidth="1.4" />
        </svg>
        <button type="button" onClick={() => setOpen((v) => !v)} className="flex-1 min-w-0 text-left" aria-expanded={open}>
          <span className="text-[13px] text-[#171717]">
            {missingCount} more check{missingCount === 1 ? "" : "s"} unlock with a better clip
          </span>
          {uniq[0] && !open && <span className="block text-[12px] text-[#8f8f8f] truncate">{uniq[0]}</span>}
        </button>
        {onOpenGuide && (
          <button type="button" onClick={onOpenGuide} className="v-btn-ghost h-7 px-2.5 text-[12px] shrink-0">
            How to film
          </button>
        )}
        <button type="button" onClick={() => setOpen((v) => !v)} className="text-[#8f8f8f] text-[12px] w-5 shrink-0" aria-label={open ? "Collapse" : "Expand"}>
          {open ? "−" : "+"}
        </button>
      </div>
      {open && uniq.length > 0 && (
        <ul className="px-3 pb-3 pl-[42px] space-y-1.5">
          {uniq.map((h, i) => (
            <li key={i} className="text-[12.5px] text-[#4d4d4d] leading-snug list-disc">
              {h}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// GuardrailStrip — ALWAYS the full, un-truncated ICC 15° + medical notices.
// ─────────────────────────────────────────────────────────────────────────────
export const GuardrailStrip: React.FC<{ compact?: boolean; className?: string }> = ({ compact, className = "" }) => (
  <aside
    className={`rounded-[6px] bg-[#ffffff] shadow-[0_0_0_1px_#ebebeb] ${compact ? "px-3 py-2" : "px-4 py-3"} ${
      compact ? "space-y-1.5" : "grid gap-3 md:grid-cols-2"
    } ${className}`}
    aria-label="Important notices"
  >
    <p className="v-guardrail flex items-start gap-2" style={{ minWidth: 0 }}>
      <span aria-hidden className="shrink-0 text-[12px] leading-[1.5]">
        🔒
      </span>
      <span>
        <span className="font-[500] text-[#171717]">Elbow legality: refused. </span>
        {GUARDRAIL_ICC}
      </span>
    </p>
    <p className="v-guardrail flex items-start gap-2" style={{ minWidth: 0 }}>
      <span aria-hidden className="shrink-0 text-[12px] leading-[1.5]">
        ✚
      </span>
      <span>
        <span className="font-[500] text-[#171717]">Not a medical diagnosis. </span>
        {GUARDRAIL_MEDICAL}
      </span>
    </p>
  </aside>
);
