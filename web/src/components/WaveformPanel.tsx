import React, { useEffect, useMemo, useRef } from "react";
import uPlot from "uplot";
import { useStudioStore } from "../store";
import type { BiomechanicalFactSheet, TelemetryV2 } from "../types";

interface SingleChartProps {
  title: string;
  timeMs: number[];
  series: Array<{ label: string; values: number[]; color: string; suppressed?: boolean }>;
  bfcMs: number;
  ffcMs: number;
  brMs: number;
  isComplete: boolean;
  yRange?: [number, number];
}

const SingleUPlot: React.FC<SingleChartProps> = ({
  title,
  timeMs,
  series,
  bfcMs,
  ffcMs,
  brMs,
  isComplete,
  yRange,
}) => {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const plotRef = useRef<uPlot | null>(null);
  const cursorFrameRef = useRef<number>(useStudioStore.getState().currentFrame);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || timeMs.length < 2) return;

    const width = Math.max(el.clientWidth - 4, 240);
    const height = 160;
    const data: uPlot.AlignedData = [timeMs, ...series.map((s) => s.values)];

    const drawOverlayPlugin: uPlot.Plugin = {
      hooks: {
        draw: [
          (u) => {
            const { ctx } = u;
            const top = u.bbox.top;
            const bottom = top + u.bbox.height;
            ctx.save();
            // BFC, FFC, BR vertical phase lines
            if (isComplete) {
              const marks: Array<[number, string, string]> = [
                [bfcMs, "BFC", "#f59e0b"],
                [ffcMs, "FFC", "#38bdf8"],
                [brMs, "BR", "#22c55e"],
              ];
              for (const [xm, lbl, col] of marks) {
                const px = u.valToPos(xm, "x", true);
                if (px >= u.bbox.left && px <= u.bbox.left + u.bbox.width) {
                  ctx.strokeStyle = col;
                  ctx.setLineDash([4, 3]);
                  ctx.lineWidth = 1.2;
                  ctx.beginPath();
                  ctx.moveTo(px, top);
                  ctx.lineTo(px, bottom);
                  ctx.stroke();
                  ctx.setLineDash([]);
                  ctx.fillStyle = col;
                  ctx.font = "bold 9px monospace";
                  ctx.fillText(lbl, px + 3, top + 10);
                }
              }
            }
            // Live transient frame cursor line (updated at 60-120 Hz without React re-renders!)
            const fi = Math.min(Math.max(cursorFrameRef.current, 0), timeMs.length - 1);
            const curX = u.valToPos(timeMs[fi], "x", true);
            if (curX >= u.bbox.left && curX <= u.bbox.left + u.bbox.width) {
              ctx.strokeStyle = "#ffffff";
              ctx.lineWidth = 1.6;
              ctx.beginPath();
              ctx.moveTo(curX, top);
              ctx.lineTo(curX, bottom);
              ctx.stroke();
            }
            ctx.restore();
          },
        ],
      },
    };

    const opts: uPlot.Options = {
      width,
      height,
      padding: [8, 10, 6, 0],
      plugins: [drawOverlayPlugin],
      cursor: { show: true, drag: { x: false, y: false } },
      scales: {
        x: { time: false },
        y: yRange ? { auto: false, range: yRange } : { auto: true },
      },
      axes: [
        { stroke: "#94a3b8", grid: { stroke: "#1f2a3d", width: 1 }, font: "10px monospace", size: 28, gap: 4 },
        { stroke: "#94a3b8", grid: { stroke: "#1f2a3d", width: 1 }, font: "10px monospace", size: 40 },
      ],
      series: [
        { label: "ms" },
        ...series.map((s) => ({
          label: s.suppressed ? `${s.label} [SUP]` : s.label,
          stroke: s.suppressed ? `${s.color}44` : s.color,
          width: s.suppressed ? 1.2 : 2.1,
          dash: s.suppressed ? [5, 4] : undefined,
        })),
      ],
    };

    const u = new uPlot(opts, data, el);
    plotRef.current = u;

    // Click chart to jump master frame clock
    const onClick = (ev: MouseEvent) => {
      const rect = u.over.getBoundingClientRect();
      const xVal = u.posToVal(ev.clientX - rect.left, "x");
      let best = 0;
      let bestD = Infinity;
      for (let i = 0; i < timeMs.length; i++) {
        const d = Math.abs(timeMs[i] - xVal);
        if (d < bestD) { bestD = d; best = i; }
      }
      useStudioStore.getState().setIsPlaying(false);
      useStudioStore.getState().setCurrentFrame(best);
    };
    u.over.addEventListener("click", onClick);

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const nextW = Math.max(Math.floor(entry.contentRect.width) - 4, 220);
        if (Math.abs(u.width - nextW) >= 4) {
          u.setSize({ width: nextW, height });
        }
      }
    });
    ro.observe(el);

    const unsub = useStudioStore.subscribe(
      (s) => s.currentFrame,
      (frame) => {
        cursorFrameRef.current = frame;
        u.redraw(false);
      }
    );

    return () => {
      ro.disconnect();
      unsub();
      u.over.removeEventListener("click", onClick);
      u.destroy();
      plotRef.current = null;
    };
  }, [timeMs, series, bfcMs, ffcMs, brMs, isComplete, yRange]);

  return (
    <div className="bg-[#131926] border border-[#263248] rounded-xl p-2.5">
      <div className="text-[11px] font-semibold text-slate-200 mb-1 flex items-center justify-between">
        <span>{title}</span>
        {series.every((s) => s.suppressed) && (
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-amber-500/15 text-amber-300 border border-amber-500/30">
            SUPPRESSED (VIEW / CLIP)
          </span>
        )}
      </div>
      <div ref={containerRef} className="w-full" />
    </div>
  );
};

export const WaveformPanel: React.FC<{ telemetry: TelemetryV2; factSheet: BiomechanicalFactSheet }> = ({
  telemetry,
  factSheet,
}) => {
  const t = telemetry.time_rel_ffc_ms;
  const ks = telemetry.kin_series;
  const ev = factSheet.events;
  const fps = Math.max(factSheet.audit.fps, 1);
  const bfcMs = ((ev.bfc.frame_float - ev.ffc.frame_float) / fps) * 1000;
  const brMs = ((ev.br.frame_float - ev.ffc.frame_float) / fps) * 1000;

  const fkSup = factSheet.metrics["front_knee_br_deg"]?.suppressed ?? true;
  const tfSup = factSheet.metrics["trunk_flexion_delta_deg"]?.suppressed ?? true;
  const lfSup = factSheet.metrics["thorax_lat_flex_br_deg"]?.suppressed ?? true;
  const scrSup = factSheet.metrics["shoulder_counter_rotation_deg"]?.suppressed ?? true;

  const kneeSeries = useMemo(
    () => [{ label: "Front Knee", values: ks.front_knee_deg ?? [], color: "#38bdf8", suppressed: fkSup }],
    [ks.front_knee_deg, fkSup]
  );
  const trunkSeries = useMemo(
    () => [
      { label: "Sagittal Flex", values: ks.trunk_flex_deg ?? [], color: "#a855f7", suppressed: tfSup },
      { label: "Contralat. Tilt", values: ks.thorax_lat_flex_deg ?? [], color: "#fb7185", suppressed: lfSup },
    ],
    [ks.trunk_flex_deg, ks.thorax_lat_flex_deg, tfSup, lfSup]
  );
  const transSeries = useMemo(
    () => [
      { label: "Pelvis Yaw", values: ks.pelvis_yaw_deg ?? [], color: "#22d3ee", suppressed: scrSup },
      { label: "Shoulder Yaw", values: ks.shoulder_yaw_deg ?? [], color: "#f59e0b", suppressed: scrSup },
    ],
    [ks.pelvis_yaw_deg, ks.shoulder_yaw_deg, scrSup]
  );
  const transRange = useMemo<[number, number] | undefined>(() => (scrSup ? [95, 415] : undefined), [scrSup]);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
      <SingleUPlot
        title="Front Knee Internal Angle (180° = Straight)"
        timeMs={t}
        bfcMs={bfcMs}
        ffcMs={0}
        brMs={brMs}
        isComplete={ev.is_complete_delivery}
        series={kneeSeries}
      />
      <SingleUPlot
        title="Global Trunk Kinematics (Z-X Cardan)"
        timeMs={t}
        bfcMs={bfcMs}
        ffcMs={0}
        brMs={brMs}
        isComplete={ev.is_complete_delivery}
        series={trunkSeries}
      />
      <SingleUPlot
        title="Transverse Alignment (180° Side, 270° Front)"
        timeMs={t}
        bfcMs={bfcMs}
        ffcMs={0}
        brMs={brMs}
        isComplete={ev.is_complete_delivery}
        yRange={transRange}
        series={transSeries}
      />
    </div>
  );
};
