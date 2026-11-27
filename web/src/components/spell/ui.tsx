import React, { useCallback, useEffect, useRef, useState } from "react";

/**
 * Small shared helpers for the Spell & Workload hub (Studio v5 "Paper Instrument").
 * Chrome stays monochrome; colour only appears inside SVG charts.
 */

export const MONO = "Geist Mono, ui-monospace, SFMono-Regular, Menlo, monospace";

/** Keyed async loader: resets data when `key` changes, keeps last data on `reload()`. Never throws. */
export function useKeyedAsync<T>(key: string | null, fn: () => Promise<T>) {
  const [state, setState] = useState<{ data: T | null; error: string | null; loading: boolean }>({
    data: null,
    error: null,
    loading: key !== null,
  });
  const [tick, setTick] = useState(0);
  const lastKey = useRef<string | null>(null);
  const fnRef = useRef(fn);
  fnRef.current = fn;

  useEffect(() => {
    const keyChanged = lastKey.current !== key;
    lastKey.current = key;
    if (key === null) {
      setState({ data: null, error: null, loading: false });
      return;
    }
    let cancelled = false;
    setState((s) => ({ data: keyChanged ? null : s.data, error: null, loading: true }));
    fnRef
      .current()
      .then((d) => {
        if (!cancelled) setState({ data: d, error: null, loading: false });
      })
      .catch((e: unknown) => {
        if (!cancelled) {
          const msg = e instanceof Error ? e.message : String(e);
          setState((s) => ({ data: keyChanged ? null : s.data, error: msg || "Request failed", loading: false }));
        }
      });
    return () => {
      cancelled = true;
    };
  }, [key, tick]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { ...state, reload };
}

export function fmtNum(v: number | null | undefined, digits?: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const d = digits ?? (Math.abs(v) >= 100 ? 0 : Math.abs(v) >= 10 ? 1 : 2);
  return v.toFixed(d);
}

export function fmtSigned(v: number | null | undefined, digits?: number): string {
  if (v === null || v === undefined || !Number.isFinite(v)) return "—";
  const s = fmtNum(v, digits);
  return v > 0 ? `+${s}` : s.replace("-", "−");
}

/** Plain display unit ("deg" → "°"). */
export function unitLabel(u: string | null | undefined): string {
  if (!u) return "";
  const k = u.trim().toLowerCase();
  if (k === "deg" || k === "degrees" || k === "°") return "°";
  if (k === "pct" || k === "percent" || k === "%") return "%";
  return ` ${u}`;
}

export function withUnit(v: number | null | undefined, u: string, digits?: number): string {
  const s = fmtNum(v, digits);
  return s === "—" ? s : `${s}${unitLabel(u)}`;
}

/** Local-date ISO "YYYY-MM-DD". */
export function isoLocal(d: Date): string {
  const m = `${d.getMonth() + 1}`.padStart(2, "0");
  const day = `${d.getDate()}`.padStart(2, "0");
  return `${d.getFullYear()}-${m}-${day}`;
}

export function parseIsoLocal(s: string): Date {
  const [y, m, d] = s.slice(0, 10).split("-").map((x) => Number(x));
  return new Date(y, (m || 1) - 1, d || 1);
}

const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
export function shortDate(s: string): string {
  const d = parseIsoLocal(s);
  return `${d.getDate()} ${MONTHS[d.getMonth()]}`;
}

/** Light → dark crimson ramp (SEG.wrist family), t ∈ [0, 1]. Chart-only colour. */
export function wristRamp(t: number): string {
  const stops = [
    [252, 165, 165], // #FCA5A5
    [220, 38, 38], // #DC2626 (SEG.wrist)
    [127, 29, 29], // #7F1D1D
  ];
  const x = Math.max(0, Math.min(1, t)) * 2;
  const i = Math.min(1, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  const c = a.map((av, k) => Math.round(av + (b[k] - av) * f));
  return `rgb(${c[0]}, ${c[1]}, ${c[2]})`;
}

export const SectionTitle: React.FC<{ eyebrow: string; title: string; right?: React.ReactNode; sub?: React.ReactNode }> = ({
  eyebrow,
  title,
  right,
  sub,
}) => (
  <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
    <div className="min-w-0">
      <div className="v-label">{eyebrow}</div>
      <h2 className="text-[18px] font-medium tracking-[-0.01em] text-[#171717] mt-0.5">{title}</h2>
      {sub && <div className="text-[13px] text-[#4d4d4d] mt-1 leading-snug">{sub}</div>}
    </div>
    {right}
  </div>
);

export const CardTitle: React.FC<{ label: string; right?: React.ReactNode }> = ({ label, right }) => (
  <div className="flex items-center justify-between gap-2 mb-3">
    <div className="v-label">{label}</div>
    {right}
  </div>
);

/** Calm per-section empty / error state. Never crashes the screen. */
export const CalmState: React.FC<{ title: string; body?: React.ReactNode; tone?: "empty" | "error" | "loading" }> = ({
  title,
  body,
  tone = "empty",
}) => (
  <div
    className="rounded-[6px] px-4 py-5 text-center"
    style={{ border: `1px ${tone === "error" ? "solid" : "dashed"} #d4d4d4`, background: "#fafafa" }}
    role={tone === "error" ? "status" : undefined}
  >
    <div className="text-[13px] font-medium text-[#171717]">
      {tone === "loading" ? "Loading…" : title}
    </div>
    {body && tone !== "loading" && <div className="text-[12px] text-[#666] mt-1 leading-snug">{body}</div>}
  </div>
);

/** Monochrome chip; `strong` uses the single chrome accent. */
export const Tag: React.FC<{
  children: React.ReactNode;
  variant?: "plain" | "ink" | "strong" | "dashed" | "muted";
  title?: string;
}> = ({ children, variant = "plain", title }) => {
  const style: React.CSSProperties =
    variant === "ink"
      ? { background: "#171717", color: "#fff", boxShadow: "0 0 0 1px #171717" }
      : variant === "strong"
        ? { color: "#297a3a" }
        : variant === "dashed"
          ? { boxShadow: "none", border: "1px dashed #8f8f8f" }
          : variant === "muted"
            ? { color: "#8f8f8f" }
            : {};
  return (
    <span className="v-chip whitespace-nowrap" style={style} title={title}>
      {children}
    </span>
  );
};
