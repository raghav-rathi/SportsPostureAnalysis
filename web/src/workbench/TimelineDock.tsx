/**
 * TimelineDock — the "one clock" dock under the Workbench stage (docs/research_app_landscape_v5.md §8.2).
 *
 * PhaseTimeline + JogWheel + SpeedDetent + loop controls + frame / ms-from-FFC readout, and two tabs:
 *   • Kinematic sequence (default): 4-lane proximal→distal waterfall from `telemetry.sequence_series`, peak diamonds
 *     with 95% timing whiskers (click → seek), order strip. Gated (< 90 fps or unmeasurable) → ghosted lanes with an
 *     honest message + How to film link. Never hidden.
 *   • Angle traces: multi-channel synced waveforms from `telemetry.kin_series` with event markers + hover readout.
 *     The bowling elbow trace is visual only — elbow legality is permanently refused.
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import type { BiomechanicalFactSheet, SequenceSegmentPeak, TelemetryV2 } from "../types";
import { SEG, SEQUENCE_LANES, GUARDRAIL_ICC } from "../theme";
import { usePlayhead } from "./PlayheadProvider";
import {
  JogWheel,
  PhaseTimeline,
  SpeedDetent,
  buildPhases,
  fmtMetric,
  frameAtTimeRelFfc,
  msPerFrame,
  type SpeedRate,
} from "../components/ui/PaperPrimitives";

export interface TimelineDockProps {
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  noteFrames?: number[];
  onOpenFilmGuide?: () => void;
  defaultTab?: "sequence" | "angles";
  className?: string;
}

type DockTab = "sequence" | "angles";

function useElementWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(0);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    setW(el.clientWidth);
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) setW(Math.round(e.contentRect.width));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

const finite = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

export const TimelineDock: React.FC<TimelineDockProps> = ({ telemetry, factSheet, noteFrames = [], onOpenFilmGuide, defaultTab = "sequence", className = "" }) => {
  const ph = usePlayhead();
  const [tab, setTab] = useState<DockTab>(defaultTab);
  const [zoomed, setZoomed] = useState(true);
  const events = factSheet.events;
  const numFrames = Math.max(1, ph.numFrames || telemetry.num_frames);
  const last = numFrames - 1;
  const tRel = telemetry.time_rel_ffc_ms ?? [];
  const mspf = msPerFrame(tRel, telemetry.analysis_fps || factSheet.audit.fps || 30);
  const captureFps = factSheet.audit.fps || telemetry.analysis_fps;
  const { phases, ticks } = useMemo(() => buildPhases(events, numFrames), [events, numFrames]);

  // Delivery window (BFC − 0.25 s → follow-through or BR + 0.3 s).
  const win: [number, number] = useMemo(() => {
    if (!zoomed || !events.is_complete_delivery) return [0, last];
    const pad0 = Math.round(250 / mspf);
    const pad1 = Math.round(300 / mspf);
    const a = Math.max(0, Math.floor(events.bfc.frame_idx - pad0));
    const endEv = events.ft_peak ? Math.max(events.ft_peak.frame_idx, events.br.frame_idx + pad1 / 2) : events.br.frame_idx + pad1;
    const b = Math.min(last, Math.ceil(endEv));
    return b - a >= 6 ? [a, b] : [0, last];
  }, [zoomed, events, last, mspf]);

  const frame = ph.frame;
  const tNow = finite(ph.timeRelFfcMs) ? ph.timeRelFfcMs : tRel[Math.round(frame)] ?? 0;

  const phaseAt = (f: number) => phases.find((p) => f >= p.start && f <= p.end) ?? null;
  const loopPhase = () => {
    const p = phaseAt(frame);
    if (!p) return;
    const r: [number, number] = [Math.floor(p.start), Math.ceil(p.end)];
    if (ph.loopRange && ph.loopRange[0] === r[0] && ph.loopRange[1] === r[1]) ph.setLoopRange(null);
    else ph.setLoopRange(r);
  };
  const actionRange: [number, number] | null = events.is_complete_delivery
    ? [Math.max(0, Math.floor(events.bfc.frame_idx - Math.round(120 / mspf))), Math.min(last, Math.ceil((events.ft_peak?.frame_idx ?? events.br.frame_idx + Math.round(250 / mspf))))]
    : null;
  const loopAction = () => {
    if (!actionRange) return;
    if (ph.loopRange && ph.loopRange[0] === actionRange[0] && ph.loopRange[1] === actionRange[1]) ph.setLoopRange(null);
    else ph.setLoopRange(actionRange);
  };
  const isLoopAction = !!(actionRange && ph.loopRange && ph.loopRange[0] === actionRange[0] && ph.loopRange[1] === actionRange[1]);
  const isLoopPhase = !!ph.loopRange && !isLoopAction;

  return (
    <section className={`v-panel p-3 md:p-4 ${className}`} aria-label="Timeline">
      <PhaseTimeline
        numFrames={numFrames}
        currentFrame={frame}
        events={events}
        timeRelFfcMs={tRel}
        onSeek={(f) => ph.seekToEventFrame(f, true)}
        noteFrames={noteFrames}
        loopRange={ph.loopRange}
      />
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-2">
        <JogWheel
          frame={Math.round(frame)}
          numFrames={numFrames}
          playing={ph.playing}
          onTogglePlay={() => ph.setPlaying((p) => !p)}
          onStep={(d) => {
            if (ph.playing) ph.setPlaying(false);
            ph.step(d);
          }}
          onLongPressLoop={loopPhase}
        />
        <SpeedDetent rate={ph.rate as SpeedRate} onChange={(r) => ph.setRate(r)} captureFps={captureFps} />
        <div className="v-seg" role="group" aria-label="Loop">
          <button type="button" aria-pressed={isLoopAction} disabled={!actionRange} onClick={loopAction} title="Loop the whole action (back foot → follow-through)">
            ⟲ Action
          </button>
          <button type="button" aria-pressed={isLoopPhase} onClick={loopPhase} title="Loop the phase under the playhead">
            ⟲ Phase
          </button>
        </div>
        <div className="ml-auto text-right leading-tight">
          <div className="v-num text-[13px] text-[#171717]">
            {tNow >= 0 ? "+" : "−"}
            {Math.abs(Math.round(tNow))} ms
            <span className="text-[#8f8f8f]"> from front-foot plant</span>
          </div>
          <div className="v-num text-[11px] text-[#8f8f8f]">
            frame {Math.round(frame) + 1} / {numFrames}
            {phaseAt(frame) ? ` · ${phaseAt(frame)!.label}` : ""}
          </div>
        </div>
      </div>

      <div className="mt-3 flex items-center justify-between gap-2 border-b border-[#ebebeb]">
        <nav className="flex items-center gap-1" aria-label="Timeline charts">
          {(
            [
              ["sequence", "Kinematic sequence"],
              ["angles", "Angle traces"],
            ] as Array<[DockTab, string]>
          ).map(([k, label]) => (
            <button
              key={k}
              type="button"
              onClick={() => setTab(k)}
              aria-current={tab === k ? "page" : undefined}
              className={`h-8 px-2.5 text-[13px] -mb-px border-b-2 whitespace-nowrap ${
                tab === k ? "border-[#171717] text-[#171717] font-[500]" : "border-transparent text-[#666666] hover:text-[#171717]"
              }`}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="v-seg mb-1" role="group" aria-label="Chart range">
          <button type="button" aria-pressed={zoomed} onClick={() => setZoomed(true)}>
            Delivery
          </button>
          <button type="button" aria-pressed={!zoomed} onClick={() => setZoomed(false)}>
            Whole clip
          </button>
        </div>
      </div>

      <div className="pt-3">
        {tab === "sequence" ? (
          <SequenceWaterfall
            telemetry={telemetry}
            factSheet={factSheet}
            win={win}
            frame={frame}
            ticks={ticks}
            mspf={mspf}
            captureFps={captureFps}
            onSeek={(f) => ph.seekToEventFrame(f, true)}
            onOpenFilmGuide={onOpenFilmGuide}
          />
        ) : (
          <AngleTraces telemetry={telemetry} win={win} frame={frame} ticks={ticks} onSeek={(f) => ph.seekToEventFrame(f, true)} />
        )}
      </div>
    </section>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Kinematic sequence waterfall
// ─────────────────────────────────────────────────────────────────────────────
const LANE_H = 34;
const LANE_GAP = 6;
const LABEL_W = 92;

const SequenceWaterfall: React.FC<{
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  win: [number, number];
  frame: number;
  ticks: ReturnType<typeof buildPhases>["ticks"];
  mspf: number;
  captureFps: number;
  onSeek: (f: number) => void;
  onOpenFilmGuide?: () => void;
}> = ({ telemetry, factSheet, win, frame, ticks, mspf, captureFps, onSeek, onOpenFilmGuide }) => {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const [hover, setHover] = useState<number | null>(null);
  const ks = factSheet.kinematic_sequence ?? null;
  const series = telemetry.sequence_series ?? null;
  const tRel = telemetry.time_rel_ffc_ms ?? [];
  const hasSeries = !!series && SEQUENCE_LANES.some((l) => (series[l.seriesKey] ?? []).some(finite));
  const lowFps = captureFps > 0 && captureFps < 90;

  const W = Math.max(280, width);
  const plotW = W - LABEL_W - 8;
  const headerPad = 24;
  const H = headerPad + SEQUENCE_LANES.length * (LANE_H + LANE_GAP);
  const [x0, x1] = win;
  const xOf = (f: number) => LABEL_W + ((f - x0) / Math.max(1, x1 - x0)) * plotW;
  const fOfX = (x: number) => x0 + ((x - LABEL_W) / plotW) * (x1 - x0);

  const lanes = useMemo(() => {
    return SEQUENCE_LANES.map((l, i) => {
      const arr = (series?.[l.seriesKey] ?? []) as Array<number | null>;
      let peakAbs = 0;
      for (let f = x0; f <= x1; f++) {
        const v = arr[f];
        if (finite(v)) peakAbs = Math.max(peakAbs, Math.abs(v));
      }
      const y0 = headerPad + i * (LANE_H + LANE_GAP);
      let d = "";
      let pen = false;
      if (peakAbs > 0) {
        for (let f = x0; f <= x1; f++) {
          const v = arr[f];
          if (!finite(v)) {
            pen = false;
            continue;
          }
          const y = y0 + LANE_H - (Math.abs(v) / peakAbs) * (LANE_H - 4);
          d += `${pen ? "L" : "M"}${xOf(f).toFixed(1)},${y.toFixed(1)}`;
          pen = true;
        }
      }
      const peak: SequenceSegmentPeak | undefined = ks?.segments.find((s) => s.segment === l.segment);
      const suppressed = !!peak && peak.peak_dps.suppressed;
      const live = !lowFps && peakAbs > 0 && !suppressed;
      const reason = lowFps
        ? `Needs 120–240 fps slow-motion (${Math.round(captureFps)} fps clip)`
        : peak?.peak_dps.suppression_reason ||
          (l.segment === "pelvis" || l.segment === "thorax"
            ? "Needs behind or two-camera view to measure axial twist"
            : "Not measurable from this camera angle");
      return { ...l, arr, peakAbs, y0, d, peak, live, reason };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [series, ks, x0, x1, plotW, lowFps, captureFps]);

  const liveCount = lanes.filter((l) => l.live).length;
  const allGated = lowFps || !ks || !hasSeries || liveCount === 0;
  const message = lowFps
    ? `Peak timing needs 120–240 fps slow-motion — this clip is ${Math.round(captureFps)} fps.`
    : !ks || !hasSeries
    ? "Kinematic sequence data isn't available for this delivery yet. Re-analyse it with the latest studio to see the four lanes."
    : ks.summary_plain || "Peak timing isn't measurable on this clip.";

  // Stagger SVG event tick headers whenever two ticks sit within 34 px horizontally.
  const svgTicks = useMemo(() => {
    const visible = ticks.filter((t) => t.frame >= x0 && t.frame <= x1).sort((a, b) => a.frame - b.frame);
    let lastRow0X = -999;
    return visible.map((t) => {
      const x = xOf(t.frame);
      const row = x - lastRow0X < 34 ? 1 : 0;
      if (row === 0) lastRow0X = x;
      return { ...t, x, y: row === 0 ? 9 : 19 };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticks, x0, x1, plotW]);

  // Order strip: chain of measured peaks sorted by time; "≈" when adjacent 95% timing CIs overlap.
  const order = useMemo(() => {
    if (!ks || lowFps) return [];
    const pts = ks.segments
      .filter((s) => finite(s.peak_time_rel_ffc_ms) && !s.peak_dps.suppressed)
      .map((s) => ({
        seg: s.segment,
        label: SEQUENCE_LANES.find((l) => l.segment === s.segment)?.label ?? s.label_plain,
        color: SEQUENCE_LANES.find((l) => l.segment === s.segment)?.color ?? SEG.ink,
        t: s.peak_time_rel_ffc_ms as number,
        sig: s.timing_sigma_ms ?? 0,
      }))
      .sort((a, b) => a.t - b.t);
    return pts.map((p, i) => {
      if (i === 0) return { ...p, sep: "" };
      const q = pts[i - 1];
      const close = Math.abs(p.t - q.t) <= 1.96 * Math.sqrt(p.sig * p.sig + q.sig * q.sig);
      return { ...p, sep: close ? "≈" : "→" };
    });
  }, [ks, lowFps]);

  const playX = xOf(Math.max(x0, Math.min(x1, frame)));
  const hoverF = hover !== null ? Math.round(hover) : null;

  return (
    <div>
      {/* Order strip */}
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 mb-2 min-h-[24px]">
        {allGated ? (
          <span className="v-label">Order · Hips → Chest → Arm → Wrist</span>
        ) : (
          <>
            {order.length > 0 && (
              <span className="flex flex-wrap items-center gap-1.5 text-[13px] text-[#171717]">
                {order.map((o) => (
                  <React.Fragment key={o.seg}>
                    {o.sep && (
                      <span className={`v-num ${o.sep === "≈" ? "text-[#8f8f8f]" : "text-[#4d4d4d]"}`} title={o.sep === "≈" ? "Too close to call: timing ranges overlap" : "Clearly later"}>
                        {o.sep}
                      </span>
                    )}
                    <span className="inline-flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full" style={{ background: o.color }} />
                      {o.label}
                    </span>
                  </React.Fragment>
                ))}
              </span>
            )}
            {ks && (
              <span className="v-chip">
                {ks.verdict !== "unmeasurable" ? ks.verdict_label_plain : `Partial sequence · ${liveCount}/4 lanes`}
              </span>
            )}
            {liveCount < 4 && onOpenFilmGuide && (
              <button type="button" onClick={onOpenFilmGuide} className="text-[11.5px] text-[#666666] underline underline-offset-2 hover:text-[#171717]">
                Add a second angle for all 4 lanes
              </button>
            )}
            {order.some((o) => o.sep === "≈") && <span className="text-[11.5px] text-[#8f8f8f]">≈ = too close to call at this frame rate</span>}
          </>
        )}
      </div>

      <div ref={wrapRef} className="relative w-full">
        <svg
          width={W}
          height={H}
          className={`block select-none ${allGated ? "" : "cursor-crosshair"}`}
          onMouseMove={(e) => {
            if (allGated) return;
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const x = e.clientX - r.left;
            setHover(x >= LABEL_W ? Math.max(x0, Math.min(x1, fOfX(x))) : null);
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            if (allGated) return;
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const x = e.clientX - r.left;
            if (x >= LABEL_W) onSeek(Math.round(Math.max(x0, Math.min(x1, fOfX(x)))));
          }}
          role="img"
          aria-label={allGated ? `Kinematic sequence unavailable. ${message}` : `Kinematic sequence: ${ks?.verdict_label_plain ?? ""}`}
        >
          {/* event verticals */}
          {svgTicks.map((t) => (
            <g key={t.id}>
              <line x1={t.x} x2={t.x} y1={headerPad - 3} y2={H} stroke="#C9C9C9" strokeDasharray="2 3" />
              <text x={t.x} y={t.y} textAnchor="middle" fontSize={9.5} fill="#666666" fontFamily="var(--font-geist-mono)">
                {t.short}
              </text>
            </g>
          ))}
          {lanes.map((l) => {
            const laneGated = allGated || !l.live;
            return (
              <g key={l.segment} opacity={allGated ? 0.42 : laneGated ? 0.62 : 1}>
                <rect x={LABEL_W} y={l.y0} width={plotW} height={LANE_H} fill={laneGated ? "#F4F4F5" : "#FAFAFA"} rx={3} />
                <line x1={LABEL_W} x2={LABEL_W + plotW} y1={l.y0 + LANE_H} y2={l.y0 + LANE_H} stroke="#E5E5E5" strokeDasharray={laneGated ? "3 3" : undefined} />
                <circle cx={10} cy={l.y0 + LANE_H / 2} r={4} fill={laneGated ? "#C9C9C9" : l.color} />
                <text x={20} y={l.y0 + LANE_H / 2 + 4} fontSize={12} fill={laneGated ? "#8F8F8F" : "#171717"} fontFamily="var(--font-geist-sans)">
                  {l.label}
                </text>
                {!laneGated && l.d && <path d={l.d} fill="none" stroke={l.color} strokeWidth={1.8} strokeLinejoin="round" />}
                {!allGated && laneGated && (
                  <text x={LABEL_W + 10} y={l.y0 + LANE_H / 2 + 4} fontSize={11} fill="#737373" fontFamily="var(--font-geist-sans)">
                    ⊘ {l.reason.length > 76 ? `${l.reason.slice(0, 75)}…` : l.reason}
                  </text>
                )}
                {!laneGated && l.peak && finite(l.peak.peak_time_rel_ffc_ms) && !l.peak.peak_dps.suppressed && (() => {
                  const pf = frameAtTimeRelFfc(tRel, l.peak.peak_time_rel_ffc_ms as number);
                  if (pf < x0 || pf > x1) return null;
                  const half = (1.96 * (l.peak.timing_sigma_ms ?? 0)) / mspf;
                  const cx = xOf(pf);
                  const cy = l.y0 + 6;
                  const pv = fmtMetric(l.peak.peak_dps);
                  return (
                    <g
                      className="cursor-pointer"
                      onClick={(e) => {
                        e.stopPropagation();
                        onSeek(Math.round(pf));
                      }}
                    >
                      <title>
                        {`${l.label} peak ${pv.valueText ?? ""}${pv.rangeText ? ` (likely ${pv.rangeText})` : ""} at ${Math.round(l.peak.peak_time_rel_ffc_ms as number)} ms from front-foot plant${
                          half > 0 ? `, ± ${Math.round(1.96 * (l.peak.timing_sigma_ms ?? 0))} ms` : ""
                        }. Click to jump.`}
                      </title>
                      {half > 0 && (
                        <line x1={xOf(Math.max(x0, pf - half))} x2={xOf(Math.min(x1, pf + half))} y1={cy} y2={cy} stroke={l.color} strokeWidth={2} opacity={0.55} strokeLinecap="round" />
                      )}
                      <rect x={cx - 5} y={cy - 5} width={10} height={10} transform={`rotate(45 ${cx} ${cy})`} fill={l.color} stroke="#FFFFFF" strokeWidth={1.5} />
                      <rect x={cx - 10} y={cy - 10} width={20} height={20} fill="transparent" />
                    </g>
                  );
                })()}
              </g>
            );
          })}
          {/* hover */}
          {!allGated && hoverF !== null && (
            <line x1={xOf(hoverF)} x2={xOf(hoverF)} y1={headerPad - 2} y2={H} stroke="#8F8F8F" strokeWidth={1} />
          )}
          {/* playhead */}
          <line x1={playX} x2={playX} y1={headerPad - 3} y2={H} stroke="#171717" strokeWidth={1.5} />
        </svg>

        {allGated && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none" style={{ paddingLeft: LABEL_W }}>
            <div className="pointer-events-auto max-w-[440px] mx-3 rounded-[6px] bg-[#ffffff] shadow-[0_0_0_1px_#ebebeb] px-3.5 py-2.5 text-center">
              <p className="text-[13px] text-[#171717] leading-snug">{message}</p>
              {onOpenFilmGuide && (
                <button type="button" onClick={onOpenFilmGuide} className="mt-1.5 text-[12.5px] underline underline-offset-2 text-[#171717]">
                  How to film
                </button>
              )}
            </div>
          </div>
        )}
      </div>

      {/* hover readout + lag metrics */}
      {!allGated && (
        <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-[#666666] min-h-[18px]">
          {hoverF !== null ? (
            <>
              <span className="v-num text-[#171717]">{Math.round(tRel[hoverF] ?? 0)} ms</span>
              {lanes.map((l) => {
                const v = l.arr[hoverF];
                return (
                  <span key={l.segment} className="inline-flex items-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ background: l.live ? l.color : "#c9c9c9" }} />
                    {l.label}{" "}
                    <span className="v-num text-[#171717]">{l.live && finite(v) ? `${Math.round(Math.abs(v))}°/s` : "⊘"}</span>
                  </span>
                );
              })}
            </>
          ) : (
            ks && (
              <>
                {(
                  [
                    ["Hips → chest", ks.pelvis_to_thorax_lag_ms],
                    ["Chest → arm", ks.thorax_to_arm_lag_ms],
                    ["Arm → wrist", ks.arm_to_wrist_lag_ms],
                  ] as const
                ).map(([lbl, m]) => {
                  const t = fmtMetric(m);
                  return (
                    <span key={lbl}>
                      {lbl}{" "}
                      {t.valueText ? (
                        <span className="v-num text-[#171717]">
                          {t.valueText}
                          {t.rangeText && <span className="text-[#8f8f8f]"> ({t.rangeText})</span>}
                        </span>
                      ) : (
                        <span className="text-[#a8a8a8]">⊘</span>
                      )}
                    </span>
                  );
                })}
                <span className="text-[#8f8f8f]">Each lane is scaled to its own peak · hover for °/s · click a ◆ to jump</span>
              </>
            )
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Angle traces
// ─────────────────────────────────────────────────────────────────────────────
interface Channel {
  key: string;
  label: string;
  color: string;
  dashed?: boolean;
  locked?: boolean;
}
const CHANNELS: Channel[] = [
  { key: "front_knee_deg", label: "Front knee", color: SEG.pelvis },
  { key: "thorax_lat_flex_deg", label: "Side-bend", color: SEG.thorax },
  { key: "hip_shoulder_sep_deg", label: "Hip–shoulder separation", color: SEG.com, dashed: true },
  { key: "trunk_flex_deg", label: "Trunk whip", color: SEG.ink },
  { key: "nb_arm_elev_deg", label: "Lead arm", color: SEG.lead },
  { key: "bowling_elbow_deg", label: "Bowling elbow", color: SEG.arm, dashed: true, locked: true },
];

const AngleTraces: React.FC<{
  telemetry: TelemetryV2;
  win: [number, number];
  frame: number;
  ticks: ReturnType<typeof buildPhases>["ticks"];
  onSeek: (f: number) => void;
}> = ({ telemetry, win, frame, ticks, onSeek }) => {
  const [wrapRef, width] = useElementWidth<HTMLDivElement>();
  const ks = telemetry.kin_series ?? {};
  const available = CHANNELS.filter((c) => (ks[c.key] ?? []).some(finite));
  const [sel, setSel] = useState<string[]>(() => available.filter((c) => c.key === "front_knee_deg" || c.key === "thorax_lat_flex_deg").map((c) => c.key));
  const [hover, setHover] = useState<number | null>(null);
  const tRel = telemetry.time_rel_ffc_ms ?? [];
  const active = available.filter((c) => sel.includes(c.key));
  const showLock = active.some((c) => c.locked);

  const W = Math.max(280, width);
  const H = 178;
  const L = 40;
  const R = 8;
  const T = 24;
  const B = 18;
  const [x0, x1] = win;
  const xOf = (f: number) => L + ((f - x0) / Math.max(1, x1 - x0)) * (W - L - R);
  const fOfX = (x: number) => x0 + ((x - L) / (W - L - R)) * (x1 - x0);

  const svgTicks = useMemo(() => {
    const visible = ticks.filter((t) => t.frame >= x0 && t.frame <= x1).sort((a, b) => a.frame - b.frame);
    let lastRow0X = -999;
    return visible.map((t) => {
      const x = xOf(t.frame);
      const row = x - lastRow0X < 34 ? 1 : 0;
      if (row === 0) lastRow0X = x;
      return { ...t, x, y: row === 0 ? 9 : 19 };
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ticks, x0, x1, W]);

  const [y0, y1] = useMemo(() => {
    let mn = Infinity;
    let mx = -Infinity;
    active.forEach((c) => {
      const arr = ks[c.key] ?? [];
      for (let f = x0; f <= x1; f++) {
        const v = arr[f];
        if (finite(v)) {
          mn = Math.min(mn, v);
          mx = Math.max(mx, v);
        }
      }
    });
    if (!Number.isFinite(mn)) return [0, 180];
    const pad = Math.max(5, (mx - mn) * 0.1);
    return [Math.floor((mn - pad) / 10) * 10, Math.ceil((mx + pad) / 10) * 10];
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sel, ks, x0, x1]);
  const yOf = (v: number) => T + (1 - (v - y0) / Math.max(1e-6, y1 - y0)) * (H - T - B);
  const yTicks = useMemo(() => {
    const span = y1 - y0;
    const step = span > 150 ? 45 : span > 80 ? 30 : span > 40 ? 15 : 10;
    const out: number[] = [];
    for (let v = Math.ceil(y0 / step) * step; v <= y1; v += step) out.push(v);
    return out;
  }, [y0, y1]);

  const pathFor = (key: string) => {
    const arr = ks[key] ?? [];
    let d = "";
    let pen = false;
    for (let f = x0; f <= x1; f++) {
      const v = arr[f];
      if (!finite(v)) {
        pen = false;
        continue;
      }
      d += `${pen ? "L" : "M"}${xOf(f).toFixed(1)},${yOf(v).toFixed(1)}`;
      pen = true;
    }
    return d;
  };

  const hoverF = hover !== null ? Math.round(hover) : null;
  const readF = hoverF ?? Math.round(Math.max(0, Math.min(tRel.length - 1, frame)));

  if (available.length === 0) {
    return <div className="v-ghosted rounded-[6px] bg-[#f4f4f5] h-[170px] flex items-center justify-center text-[13px]">Angle traces aren't available for this delivery.</div>;
  }

  return (
    <div>
      <div className="flex flex-wrap gap-1.5 mb-2" role="group" aria-label="Choose angle traces">
        {available.map((c) => {
          const on = sel.includes(c.key);
          return (
            <button
              key={c.key}
              type="button"
              aria-pressed={on}
              onClick={() => setSel((s) => (on ? s.filter((k) => k !== c.key) : [...s, c.key]))}
              className={`h-7 px-2.5 rounded-full text-[12px] inline-flex items-center gap-1.5 transition-colors ${
                on ? "bg-[#ffffff] text-[#171717] shadow-[0_0_0_1px_#171717]" : "bg-[#ffffff] text-[#666666] shadow-[0_0_0_1px_#ebebeb] hover:shadow-[0_0_0_1px_#c9c9c9]"
              }`}
              title={c.locked ? "Visual trace only — elbow legality (ICC 15°) is never judged here" : undefined}
            >
              <span className="w-3 h-[2px]" style={{ background: on ? c.color : "#c9c9c9" }} />
              {c.label}
              {c.locked && <span className="text-[10.5px] text-[#8f8f8f]">🔒 visual only</span>}
            </button>
          );
        })}
      </div>
      <div ref={wrapRef} className="w-full">
        <svg
          width={W}
          height={H}
          className="block select-none cursor-crosshair"
          onMouseMove={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const x = e.clientX - r.left;
            setHover(x >= L ? Math.max(x0, Math.min(x1, fOfX(x))) : null);
          }}
          onMouseLeave={() => setHover(null)}
          onClick={(e) => {
            const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const x = e.clientX - r.left;
            if (x >= L) onSeek(Math.round(Math.max(x0, Math.min(x1, fOfX(x)))));
          }}
          role="img"
          aria-label="Angle traces over the delivery"
        >
          <rect x={L} y={T} width={W - L - R} height={H - T - B} fill="#FAFAFA" rx={3} />
          {yTicks.map((v) => (
            <g key={v}>
              <line x1={L} x2={W - R} y1={yOf(v)} y2={yOf(v)} stroke="#EBEBEB" />
              <text x={L - 6} y={yOf(v) + 3.5} textAnchor="end" fontSize={10} fill="#8F8F8F" fontFamily="var(--font-geist-mono)">
                {v}°
              </text>
            </g>
          ))}
          {svgTicks.map((t) => (
            <g key={t.id}>
              <line x1={t.x} x2={t.x} y1={T - 3} y2={H - B} stroke="#C9C9C9" strokeDasharray="2 3" />
              <text x={t.x} y={t.y} textAnchor="middle" fontSize={9.5} fill="#666666" fontFamily="var(--font-geist-mono)">
                {t.short}
              </text>
            </g>
          ))}
          {active.map((c) => (
            <path key={c.key} d={pathFor(c.key)} fill="none" stroke={c.color} strokeWidth={1.8} strokeDasharray={c.dashed ? "5 3" : undefined} strokeLinejoin="round" />
          ))}
          {hoverF !== null && <line x1={xOf(hoverF)} x2={xOf(hoverF)} y1={T} y2={H - B} stroke="#8F8F8F" />}
          {frame >= x0 && frame <= x1 && <line x1={xOf(frame)} x2={xOf(frame)} y1={T - 4} y2={H - B} stroke="#171717" strokeWidth={1.5} />}
          <text x={W - R} y={H - 4} textAnchor="end" fontSize={10} fill="#8F8F8F" fontFamily="var(--font-geist-mono)">
            ms from front-foot plant: {Math.round(tRel[x0] ?? 0)} … {Math.round(tRel[x1] ?? 0)}
          </text>
        </svg>
      </div>
      <div className="mt-1.5 flex flex-wrap items-center gap-x-4 gap-y-1 text-[11.5px] text-[#666666] min-h-[18px]">
        <span className="v-num text-[#171717]">{Math.round(tRel[readF] ?? 0)} ms</span>
        {active.map((c) => {
          const v = (ks[c.key] ?? [])[readF];
          return (
            <span key={c.key} className="inline-flex items-center gap-1">
              <span className="w-3 h-[2px]" style={{ background: c.color }} />
              {c.label} <span className="v-num text-[#171717]">{finite(v) ? `${Math.round(v)}°` : "—"}</span>
            </span>
          );
        })}
        <span className="text-[#8f8f8f]">Raw per-frame traces (smoothed) · the headline numbers with ranges live in the inspector</span>
      </div>
      {showLock && (
        <p className="mt-2 v-guardrail rounded-[6px] bg-[#fafafa] shadow-[0_0_0_1px_#ebebeb] px-3 py-2">
          <span className="font-[500] text-[#171717]">🔒 Bowling elbow: visual trace only, not ICC legality. </span>
          {GUARDRAIL_ICC}
        </p>
      )}
    </div>
  );
};
