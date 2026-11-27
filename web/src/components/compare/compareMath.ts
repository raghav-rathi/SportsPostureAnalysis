/**
 * Pure helpers for the Compare & Ghost Overlay Studio (docs/research_app_landscape_v5.md §8.3).
 * No statuses or differences are computed here — only time/frame alignment and default pairing.
 */
import type { BiomechanicalFactSheet, DeliveryCard, MetricEstimate, OverviewResponse, TelemetryV2 } from "../../types";

export type SyncEvent = "ffc" | "bfc" | "br";

export const SYNC_LABEL: Record<SyncEvent, string> = {
  ffc: "Front-foot plant (FFC)",
  bfc: "Back-foot landing (BFC)",
  br: "Ball release (BR)",
};

export const nFrames = (t: TelemetryV2) => Math.max(t.num_frames || t.joints2d_px?.length || 1, 1);
export const proxyFps = (t: TelemetryV2) => Math.max(t.proxy_fps || t.analysis_fps || 30, 1);
export const analysisFps = (t: TelemetryV2) => Math.max(t.analysis_fps || t.proxy_fps || 30, 1);

/** Time (ms) of every telemetry frame relative to FFC; synthesised from fps if missing. */
export function timesRelFfc(t: TelemetryV2, ffcFrame: number | null): number[] {
  const n = nFrames(t);
  const arr = t.time_rel_ffc_ms;
  if (arr && arr.length >= n) return arr.slice(0, n);
  const fps = analysisFps(t);
  const f0 = ffcFrame ?? 0;
  return Array.from({ length: n }, (_, i) => ((i - f0) * 1000) / fps);
}

export function syncFrameOf(fs: BiomechanicalFactSheet | null | undefined, ev: SyncEvent, n: number): number | null {
  const e = fs?.events;
  if (!e) return null;
  const pick = ev === "ffc" ? e.ffc : ev === "bfc" ? e.bfc : e.br;
  if (!pick || !Number.isFinite(pick.frame_idx)) return null;
  return Math.min(Math.max(Math.round(pick.frame_idx), 0), n - 1);
}

/** Nearest frame index whose time is closest to `tMs` (times monotonic). */
export function frameAtTime(times: number[], tMs: number): number {
  const n = times.length;
  if (n === 0) return 0;
  if (tMs <= times[0]) return 0;
  if (tMs >= times[n - 1]) return n - 1;
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (times[mid] <= tMs) lo = mid;
    else hi = mid;
  }
  return tMs - times[lo] <= times[hi] - tMs ? lo : hi;
}

/** Series value at time t (linear interpolation, nulls → null). */
export function valueAtTime(times: number[], series: Array<number | null> | undefined, tMs: number): number | null {
  if (!series || series.length === 0) return null;
  const f = frameAtTime(times, tMs);
  const v = series[f];
  return v === null || v === undefined || !Number.isFinite(v) ? null : v;
}

export function metricOf(fs: BiomechanicalFactSheet | null | undefined, key: string): MetricEstimate | null {
  if (!fs) return null;
  return fs.metrics?.[key] ?? fs.v5_metrics?.[key] ?? null;
}

const USABLE = (d: DeliveryCard) => d.capture_grade !== "unusable";

/**
 * Default pair: skip unusable clips; prefer the bowler with the most usable balls; within that bowler prefer
 * two balls with the same camera view (oldest → newest, e.g. Ball 1 fresh vs a later fatigued ball).
 */
export function defaultPair(ov: OverviewResponse | null): { a: string; b: string } | null {
  if (!ov) return null;
  const usable = ov.recent_deliveries.filter(USABLE);
  const pool = usable.length >= 2 ? usable : ov.recent_deliveries;
  if (pool.length < 2) return null;
  const byBowler = new Map<string, DeliveryCard[]>();
  pool.forEach((d) => byBowler.set(d.bowler_id, [...(byBowler.get(d.bowler_id) ?? []), d]));
  const groups = [...byBowler.values()].filter((g) => g.length >= 2).sort((x, y) => y.length - x.length);
  const candidates = groups.length ? groups : [pool];
  for (const g of candidates) {
    const sorted = [...g].sort((x, y) => (x.created_at || "").localeCompare(y.created_at || ""));
    const byView = new Map<string, DeliveryCard[]>();
    sorted.forEach((d) => byView.set(d.view_label, [...(byView.get(d.view_label) ?? []), d]));
    const sameView = [...byView.values()].filter((v) => v.length >= 2).sort((x, y) => y.length - x.length)[0];
    if (sameView) return { a: sameView[0].id, b: sameView[sameView.length - 1].id };
  }
  const g = [...candidates[0]].sort((x, y) => (x.created_at || "").localeCompare(y.created_at || ""));
  return { a: g[0].id, b: g[g.length - 1].id };
}

export function formatMs(t: number): string {
  const r = Math.round(t);
  return `${r > 0 ? "+" : r < 0 ? "−" : "±"}${Math.abs(r)} ms`;
}

export function formatShortDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

/** Aspect-preserving letterbox rect of the source video inside a canvas. */
export function videoRect(t: { width: number; height: number }, cw: number, ch: number) {
  const vw = Math.max(t.width || 1280, 1);
  const vh = Math.max(t.height || 720, 1);
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  return { ox: (cw - dw) * 0.5, oy: (ch - dh) * 0.5, dw, dh, scale };
}

/** Pelvis anchor (else centroid) + body size in source px, for skeleton-registered overlays. */
export function bodyFrameOf(t: TelemetryV2, f: number): { cx: number; cy: number; size: number } | null {
  const pts = t.joints2d_px?.[f];
  const inFr = t.joint_in_frame?.[f];
  if (!pts || pts.length < 16) return null;
  const ok = (i: number) => !!pts[i] && (!inFr || !!inFr[i]);
  const vis = pts.map((_, i) => i).filter(ok);
  if (vis.length < 4) return null;
  let cx: number;
  let cy: number;
  if (ok(0)) {
    cx = pts[0][0];
    cy = pts[0][1];
  } else {
    cx = vis.reduce((s, i) => s + pts[i][0], 0) / vis.length;
    cy = vis.reduce((s, i) => s + pts[i][1], 0) / vis.length;
  }
  let size = 0;
  if (ok(0) && ok(12)) size = Math.hypot(pts[12][0] - pts[0][0], pts[12][1] - pts[0][1]) * 3.2;
  if (size < 1e-3) {
    const ys = vis.map((i) => pts[i][1]);
    size = Math.max(...ys) - Math.min(...ys);
  }
  if (size < 1e-3) return null;
  return { cx, cy, size };
}
