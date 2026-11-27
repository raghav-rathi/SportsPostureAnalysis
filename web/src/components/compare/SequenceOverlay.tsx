import React, { useMemo, useRef } from "react";
import type { SequenceSeries } from "../../types";
import { SEQUENCE_LANES, SEG } from "../../theme";
import { formatMs } from "./compareMath";

/**
 * Overlaid kinematic-sequence lanes (A solid segment colour, B dashed slate), one lane per segment, on the shared
 * compare clock (ms relative to the sync moment). Each lane is normalised to the larger |peak| of A and B so shapes
 * and peak timing read at a glance; absolute peaks are printed. Click/drag anywhere to seek the compare clock.
 */

export interface SequenceTrack {
  /** Times (ms) of each sample relative to the sync moment. */
  times: number[];
  series: SequenceSeries | null | undefined;
  fps: number | null;
}

export interface SequenceOverlayProps {
  a: SequenceTrack;
  b: SequenceTrack | null;
  bLabel: string;
  tMs: number;
  range: [number, number];
  syncLabel: string;
  onSeek: (tMs: number) => void;
}

const W = 720;
const LANE_H = 46;
const PAD_L = 92;
const PAD_R = 14;
const TOP = 18;

function lanePath(times: number[], vals: Array<number | null> | undefined, range: [number, number], norm: number, y0: number) {
  if (!vals || !vals.length || norm <= 0) return "";
  const [t0, t1] = range;
  const sx = (t: number) => PAD_L + ((t - t0) / Math.max(t1 - t0, 1)) * (W - PAD_L - PAD_R);
  let d = "";
  let pen = false;
  for (let i = 0; i < Math.min(times.length, vals.length); i++) {
    const t = times[i];
    const v = vals[i];
    if (t < t0 || t > t1 || v === null || v === undefined || !Number.isFinite(v)) {
      pen = false;
      continue;
    }
    const x = sx(t);
    const y = y0 + LANE_H - 6 - (Math.abs(v) / norm) * (LANE_H - 12);
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  }
  return d;
}

function peakOf(times: number[], vals: Array<number | null> | undefined, range: [number, number]) {
  if (!vals) return null;
  let best: { t: number; v: number } | null = null;
  for (let i = 0; i < Math.min(times.length, vals.length); i++) {
    const v = vals[i];
    const t = times[i];
    if (v === null || v === undefined || !Number.isFinite(v) || t < range[0] || t > range[1]) continue;
    if (!best || Math.abs(v) > Math.abs(best.v)) best = { t, v };
  }
  return best;
}

const lowFps = (fps: number | null) => fps !== null && fps < 90;

export const SequenceOverlay: React.FC<SequenceOverlayProps> = ({ a, b, bLabel, tMs, range, syncLabel, onSeek }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [t0, t1] = range;
  const sx = (t: number) => PAD_L + ((t - t0) / Math.max(t1 - t0, 1)) * (W - PAD_L - PAD_R);
  const H = TOP + SEQUENCE_LANES.length * LANE_H + 22;

  const lanes = useMemo(
    () =>
      SEQUENCE_LANES.map((lane) => {
        const va = a.series?.[lane.seriesKey as keyof SequenceSeries];
        const vb = b?.series?.[lane.seriesKey as keyof SequenceSeries];
        const pa = peakOf(a.times, va, range);
        const pb = b ? peakOf(b.times, vb, range) : null;
        const norm = Math.max(Math.abs(pa?.v ?? 0), Math.abs(pb?.v ?? 0));
        return { lane, va, vb, pa, pb, norm };
      }),
    [a, b, range]
  );

  const anyData = lanes.some((l) => l.pa || l.pb);
  const fpsNote =
    lowFps(a.fps) || (b && lowFps(b.fps))
      ? `Peak timing needs 120–240 fps slow-motion — ${lowFps(a.fps) ? `A is ${Math.round(a.fps ?? 0)} fps` : ""}${
          lowFps(a.fps) && b && lowFps(b.fps) ? ", " : ""
        }${b && lowFps(b.fps) ? `B is ${Math.round(b.fps ?? 0)} fps` : ""}. Shapes are shown faded; do not read peak order.`
      : null;

  const seekFromEvent = (clientX: number) => {
    const svg = svgRef.current;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    const x = ((clientX - r.left) / r.width) * W;
    const t = t0 + ((x - PAD_L) / (W - PAD_L - PAD_R)) * (t1 - t0);
    onSeek(Math.min(t1, Math.max(t0, t)));
  };

  const ticks: number[] = [];
  const step = t1 - t0 > 800 ? 200 : 100;
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) ticks.push(t);

  return (
    <div className="v-panel p-3 sm:p-4 min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2 mb-2">
        <span className="v-label">Kinematic sequence · A vs B · aligned at {syncLabel}</span>
        <span className="flex items-center gap-3 text-[11px] text-[#4d4d4d]">
          <span className="inline-flex items-center gap-1">
            <svg width="18" height="6"><line x1="0" y1="3" x2="18" y2="3" stroke={SEG.ink} strokeWidth="2" /></svg>A
          </span>
          {b && (
            <span className="inline-flex items-center gap-1">
              <svg width="18" height="6">
                <line x1="0" y1="3" x2="18" y2="3" stroke={SEG.ghost} strokeWidth="2" strokeDasharray="4 3" />
              </svg>
              {bLabel}
            </span>
          )}
        </span>
      </div>
      <div className={!anyData || fpsNote ? "opacity-50" : ""}>
        <svg
          ref={svgRef}
          viewBox={`0 0 ${W} ${H}`}
          className="w-full h-auto select-none cursor-crosshair touch-none"
          onPointerDown={(e) => {
            (e.currentTarget as SVGSVGElement).setPointerCapture(e.pointerId);
            seekFromEvent(e.clientX);
          }}
          onPointerMove={(e) => {
            if (e.buttons & 1) seekFromEvent(e.clientX);
          }}
          role="img"
          aria-label="Overlaid kinematic sequence lanes"
        >
          {ticks.map((t) => (
            <g key={t}>
              <line x1={sx(t)} x2={sx(t)} y1={TOP - 4} y2={H - 18} stroke={t === 0 ? "#c9c9c9" : SEG.hairline} />
              <text x={sx(t)} y={H - 5} textAnchor="middle" fontSize="10" fill={SEG.graphite} fontFamily="Geist Mono, monospace">
                {t === 0 ? "sync" : t > 0 ? `+${t}` : `${t}`}
              </text>
            </g>
          ))}
          {lanes.map(({ lane, va, vb, pa, pb, norm }, i) => {
            const y0 = TOP + i * LANE_H;
            return (
              <g key={lane.segment}>
                <line x1={PAD_L} x2={W - PAD_R} y1={y0 + LANE_H - 6} y2={y0 + LANE_H - 6} stroke={SEG.hairline} />
                <text x={8} y={y0 + LANE_H / 2} fontSize="11" fill={lane.color} fontFamily="Geist, sans-serif" fontWeight={500}>
                  {lane.label}
                </text>
                <text x={8} y={y0 + LANE_H / 2 + 13} fontSize="9.5" fill={SEG.graphite} fontFamily="Geist Mono, monospace">
                  {pa ? `A ${Math.round(Math.abs(pa.v))}` : "A —"}
                  {b ? (pb ? ` · B ${Math.round(Math.abs(pb.v))}` : " · B —") : ""}
                </text>
                {b && (
                  <path
                    d={lanePath(b.times, vb, range, norm, y0)}
                    fill="none"
                    stroke={SEG.ghost}
                    strokeWidth={1.8}
                    strokeDasharray="5 4"
                    opacity={0.85}
                  />
                )}
                <path d={lanePath(a.times, va, range, norm, y0)} fill="none" stroke={lane.color} strokeWidth={2} />
                {pa && norm > 0 && (
                  <path
                    d={`M${sx(pa.t)},${y0 + 2} l4,4 l-4,4 l-4,-4 z`}
                    fill={lane.color}
                    stroke="#fff"
                    strokeWidth={1}
                  />
                )}
                {pb && norm > 0 && (
                  <path d={`M${sx(pb.t)},${y0 + 2} l4,4 l-4,4 l-4,-4 z`} fill="#fff" stroke={SEG.ghost} strokeWidth={1.4} />
                )}
              </g>
            );
          })}
          <line x1={sx(tMs)} x2={sx(tMs)} y1={TOP - 8} y2={H - 18} stroke={SEG.ink} strokeWidth={1.2} />
          <text x={Math.min(W - 40, Math.max(PAD_L + 20, sx(tMs)))} y={10} textAnchor="middle" fontSize="10" fill={SEG.ink} fontFamily="Geist Mono, monospace">
            {formatMs(tMs)}
          </text>
        </svg>
      </div>
      <p className="mt-2 text-[12px] leading-[1.5] text-[#4d4d4d]">
        {!anyData
          ? "Rotation-speed lanes are not available for these clips (they need the v5 sequence series and a clip of at least 90 fps)."
          : fpsNote ?? "Lanes are scaled to the larger peak of A and B (deg/s; wrist in its own units). Diamonds mark each clip's peak. Click to scrub both clips."}
      </p>
    </div>
  );
};

export default SequenceOverlay;
