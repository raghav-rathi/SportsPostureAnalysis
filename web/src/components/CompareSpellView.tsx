import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Layers, Pause, Play, Plus } from "lucide-react";
import { api } from "../api";
import type { DeliveryDetailResponse, MetricEstimate, SessionSummary } from "../types";

interface CompareSpellViewProps {
  session: SessionSummary;
  onDualFusedCreated: (newDeliveryId: string) => Promise<void>;
}

const BONE_PAIRS_2D: Array<[number, number, "spine" | "lleg" | "rleg" | "larm" | "rarm"]> = [
  [0, 3, "spine"], [3, 6, "spine"], [6, 9, "spine"], [9, 12, "spine"], [12, 15, "spine"],
  [0, 1, "lleg"], [1, 4, "lleg"], [4, 7, "lleg"], [7, 10, "lleg"],
  [0, 2, "rleg"], [2, 5, "rleg"], [5, 8, "rleg"], [8, 11, "rleg"],
  [9, 13, "larm"], [13, 16, "larm"], [16, 18, "larm"], [18, 20, "larm"], [20, 22, "larm"],
  [9, 14, "rarm"], [14, 17, "rarm"], [17, 19, "rarm"], [19, 21, "rarm"], [21, 23, "rarm"],
];

function nearestIdxForRelMs(times: number[], targetMs: number): number {
  if (!times || times.length === 0) return 0;
  let best = 0;
  let bestD = Infinity;
  for (let i = 0; i < times.length; i++) {
    const d = Math.abs(times[i] - targetMs);
    if (d < bestD) {
      bestD = d;
      best = i;
    }
  }
  return best;
}

function metricSigma(m: MetricEstimate): number {
  if (m.sigma_total != null && m.sigma_total > 0) return m.sigma_total;
  if (m.ci_low != null && m.ci_high != null) return Math.max((m.ci_high - m.ci_low) / 3.92, 0.1);
  return 2.0;
}

// Aspect-preserving synced video + dual-cased monochrome 2D skeleton composite player (fixes P0-2!)
const SyncedCompositeVideo: React.FC<{ detail: DeliveryDetailResponse | null; relMs: number }> = ({
  detail,
  relMs,
}) => {
  const vidRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const seekingRef = useRef<boolean>(false);
  const pendingFrameRef = useRef<number | null>(null);

  const frameIdx = useMemo(() => {
    if (!detail) return 0;
    return nearestIdxForRelMs(detail.telemetry.time_rel_ffc_ms, relMs);
  }, [detail, relMs]);

  const drawFrame = useCallback(
    (fIdx: number) => {
      const canvas = canvasRef.current;
      const vid = vidRef.current;
      if (!canvas || !detail) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const cw = canvas.width;
      const ch = canvas.height;
      const vw = Math.max(detail.telemetry.width || 1280, 1);
      const vh = Math.max(detail.telemetry.height || 720, 1);
      const scale = Math.min(cw / vw, ch / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      const ox = (cw - dw) * 0.5;
      const oy = (ch - dh) * 0.5;

      ctx.fillStyle = "#f4f4f5";
      ctx.fillRect(0, 0, cw, ch);
      if (vid && vid.readyState >= 2) {
        ctx.drawImage(vid, ox, oy, dw, dh);
      }

      const n = Math.max(detail.telemetry.num_frames, 1);
      const f = Math.min(Math.max(fIdx, 0), n - 1);
      const pts = detail.telemetry.joints2d_px[f];
      const inFr = detail.telemetry.joint_in_frame[f];
      const isRight = detail.fact_sheet.bowling_arm === "right";

      if (pts && pts.length === 24) {
        ctx.save();
        ctx.lineCap = "round";
        for (const [a, b, grp] of BONE_PAIRS_2D) {
          if (inFr && (!inFr[a] || !inFr[b])) continue;
          const pa = pts[a];
          const pb = pts[b];
          if (!pa || !pb) continue;
          const isActive =
            (isRight && (grp === "rarm" || grp === "lleg")) || (!isRight && (grp === "larm" || grp === "rleg"));
          ctx.strokeStyle = "rgba(255,255,255,0.92)";
          ctx.lineWidth = 4.2;
          ctx.beginPath();
          ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
          ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
          ctx.stroke();

          ctx.strokeStyle = isActive ? "#297a3a" : "#171717";
          ctx.lineWidth = isActive ? 2.2 : 1.7;
          ctx.beginPath();
          ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
          ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
          ctx.stroke();
        }
        for (let i = 0; i < 24; i++) {
          if (inFr && !inFr[i]) continue;
          const p = pts[i];
          if (!p) continue;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = "#171717";
          ctx.lineWidth = 1.4;
          ctx.beginPath();
          ctx.arc(ox + p[0] * scale, oy + p[1] * scale, 2.8, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }
    },
    [detail]
  );

  const seekTo = useCallback(
    (fIdx: number) => {
      const vid = vidRef.current;
      if (!detail || !vid || vid.readyState < 1) {
        drawFrame(fIdx);
        return;
      }
      const proxyFps = Math.max(detail.telemetry.proxy_fps || 30, 1);
      const tSec = fIdx / proxyFps;
      if (Math.abs(vid.currentTime - tSec) <= 0.006) {
        drawFrame(fIdx);
        return;
      }
      if (seekingRef.current) {
        pendingFrameRef.current = fIdx;
        return;
      }
      seekingRef.current = true;
      pendingFrameRef.current = null;
      vid.currentTime = tSec;
    },
    [detail, drawFrame]
  );

  // Reset seek lock whenever source video URL changes (prevents aborted-seek lockup)
  useEffect(() => {
    seekingRef.current = false;
    pendingFrameRef.current = null;
  }, [detail?.delivery.video_proxy_url]);

  useEffect(() => {
    seekTo(frameIdx);
  }, [frameIdx, seekTo]);

  if (!detail) {
    return (
      <div className="aspect-video bg-[#f4f4f5] rounded-[6px] border border-[#ebebeb] flex items-center justify-center font-mono text-[11px] text-[#8f8f8f]">
        Select a delivery
      </div>
    );
  }

  return (
    <div className="aspect-video bg-[#f4f4f5] rounded-[6px] overflow-hidden border border-[#ebebeb] relative">
      <video
        ref={vidRef}
        src={detail.delivery.video_proxy_url}
        muted
        playsInline
        preload="auto"
        onLoadedData={() => seekTo(frameIdx)}
        onSeeked={() => {
          seekingRef.current = false;
          if (pendingFrameRef.current !== null) {
            const next = pendingFrameRef.current;
            pendingFrameRef.current = null;
            seekTo(next);
          } else {
            drawFrame(frameIdx);
          }
        }}
        onEmptied={() => {
          seekingRef.current = false;
          pendingFrameRef.current = null;
        }}
        onError={() => {
          seekingRef.current = false;
          pendingFrameRef.current = null;
        }}
        className="hidden"
      />
      <canvas ref={canvasRef} width={960} height={540} className="w-full h-full block" />
      <div className="absolute bottom-2 left-2 px-2 py-0.5 rounded-[4px] bg-[#ffffff]/90 border border-[#ebebeb] font-mono text-[10px] text-[#171717] v-tabular">
        f {frameIdx}/{Math.max(detail.telemetry.num_frames - 1, 0)} · {(detail.telemetry.time_rel_ffc_ms[frameIdx] ?? 0).toFixed(0)} ms
      </div>
    </div>
  );
};

export const CompareSpellView: React.FC<CompareSpellViewProps> = ({
  session,
  onDualFusedCreated,
}) => {
  const deliveries = session.deliveries ?? [];
  const [idA, setIdA] = useState<string>(deliveries[0]?.id ?? "");
  const [idB, setIdB] = useState<string>(deliveries[1]?.id ?? deliveries[0]?.id ?? "");
  const [detA, setDetA] = useState<DeliveryDetailResponse | null>(null);
  const [detB, setDetB] = useState<DeliveryDetailResponse | null>(null);

  const [relMs, setRelMs] = useState<number>(0);
  const [playing, setPlaying] = useState(false);

  const [fuseLabel, setFuseLabel] = useState("Dual-Phone 3D Fused Delivery");
  const [fuseBusy, setFuseBusy] = useState(false);
  const [fuseMsg, setFuseMsg] = useState<string | null>(null);

  const [compareData, setCompareData] = useState<any>(null);
  const [spagMetric, setSpagMetric] = useState<string>("front_knee_deg");
  const [workload, setWorkload] = useState<any>(null);
  const [logBalls, setLogBalls] = useState<string>("24");
  const [logType, setLogType] = useState<string>("net");
  const [logNote, setLogNote] = useState<string>("");
  const [logBusy, setLogBusy] = useState(false);

  useEffect(() => {
    const list = session.deliveries ?? [];
    setIdA(list[0]?.id ?? "");
    setIdB(list[1]?.id ?? list[0]?.id ?? "");
  }, [session.id, session.deliveries?.length]);

  useEffect(() => {
    if (idA) api.getDelivery(idA).then(setDetA).catch(() => setDetA(null));
  }, [idA]);

  useEffect(() => {
    if (idB) api.getDelivery(idB).then(setDetB).catch(() => setDetB(null));
  }, [idB]);

  useEffect(() => {
    api.compareSession(session.id).then(setCompareData).catch(() => setCompareData(null));
    api.getWorkload(session.bowler_id).then(setWorkload).catch(() => setWorkload(null));
  }, [session.id, session.bowler_id, deliveries.length]);

  const [minMs, maxMs] = useMemo(() => {
    const tA = detA?.telemetry?.time_rel_ffc_ms ?? [-260, 180];
    const tB = detB?.telemetry?.time_rel_ffc_ms ?? [-260, 180];
    const lo = Math.max(-320, Math.min(tA[0] ?? -260, tB[0] ?? -260));
    const hi = Math.min(260, Math.max(tA[tA.length - 1] ?? 180, tB[tB.length - 1] ?? 180));
    return [Math.round(lo), Math.round(Math.max(hi, lo + 80))];
  }, [detA, detB]);

  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    let raf = 0;
    const tick = (now: number) => {
      const dtMs = (now - last) * 0.22;
      last = now;
      setRelMs((prev) => {
        const next = prev + dtMs;
        return next > maxMs ? minMs : next;
      });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, minMs, maxMs]);

  const deltaRows = useMemo(() => {
    if (!detA || !detB) return [];
    const mA = detA.fact_sheet.metrics;
    const mB = detB.fact_sheet.metrics;
    const rows: Array<{
      key: string;
      label: string;
      unit: string;
      valA: number;
      valB: number;
      delta: number;
      thresh95: number;
      meaningful: boolean;
    }> = [];
    for (const k of Object.keys(mA)) {
      if (k === "icc_elbow_legality_verdict") continue;
      const a = mA[k];
      const b = mB[k];
      if (!a || !b || a.suppressed || b.suppressed || a.value == null || b.value == null) continue;
      const delta = b.value - a.value;
      const thresh95 = 1.96 * Math.hypot(metricSigma(a), metricSigma(b));
      rows.push({
        key: k,
        label: a.label,
        unit: a.unit,
        valA: a.value,
        valB: b.value,
        delta,
        thresh95,
        meaningful: Math.abs(delta) >= thresh95 && thresh95 > 0,
      });
    }
    return rows;
  }, [detA, detB]);

  const handleFuseDual = async () => {
    if (!idA || !idB || idA === idB) {
      setFuseMsg("Select two different deliveries (ideally Side-On 90° and Rear 0°) to fuse.");
      return;
    }
    setFuseBusy(true);
    setFuseMsg(null);
    try {
      const r = await api.fuseDual(session.id, idA, idB, fuseLabel.trim() || "Dual-Phone 3D Fused Delivery");
      setFuseMsg(
        `✓ Fused via ${r.alignment_method} (sync offset ${(r.sync.offset_b_minus_a_s * 1000).toFixed(1)} ms, baseline ${r.baseline_angle_deg.toFixed(0)}°).`
      );
      await onDualFusedCreated(r.delivery_id);
    } catch (err: any) {
      setFuseMsg(`Fusion failed: ${err?.message || err}`);
    } finally {
      setFuseBusy(false);
    }
  };

  const handleAddWorkload = async (e: React.FormEvent) => {
    e.preventDefault();
    const count = Number(logBalls);
    if (!Number.isFinite(count) || count < 0) return;
    setLogBusy(true);
    try {
      await api.addWorkload(session.bowler_id, {
        log_date: new Date().toISOString().slice(0, 10),
        deliveries_count: count,
        session_type: logType,
        notes: logNote.trim() || "Studio logged spell",
      });
      const updated = await api.getWorkload(session.bowler_id);
      setWorkload(updated);
      setLogNote("");
    } finally {
      setLogBusy(false);
    }
  };

  const spagObj = compareData?.spaghetti_curves?.[spagMetric];
  const spagSvg = useMemo(() => {
    if (!spagObj || !spagObj.time_ms?.length) return null;
    const W = 520;
    const H = 180;
    const padL = 38;
    const padR = 14;
    const padT = 14;
    const padB = 26;
    const times: number[] = spagObj.time_ms;
    const allVals: number[] = [];
    for (const b of spagObj.balls || []) {
      for (const v of b.values || []) if (typeof v === "number") allVals.push(v);
    }
    if (allVals.length < 2) return null;
    const yMin = Math.min(...allVals) - 4;
    const yMax = Math.max(...allVals) + 4;
    const xMin = times[0];
    const xMax = times[times.length - 1];
    const sx = (t: number) => padL + ((t - xMin) / Math.max(xMax - xMin, 1)) * (W - padL - padR);
    const sy = (v: number) => H - padB - ((v - yMin) / Math.max(yMax - yMin, 1)) * (H - padT - padB);

    const buildPath = (vals: Array<number | null>) => {
      let d = "";
      let penDown = false;
      for (let i = 0; i < vals.length; i++) {
        const v = vals[i];
        if (v == null) {
          penDown = false;
          continue;
        }
        d += `${penDown ? "L" : "M"}${sx(times[i]).toFixed(1)},${sy(v).toFixed(1)} `;
        penDown = true;
      }
      return d;
    };

    const upperPts: string[] = [];
    const lowerPts: string[] = [];
    for (let i = 0; i < times.length; i++) {
      const m = spagObj.mean[i];
      const sd = spagObj.sd[i];
      if (m != null && sd != null) {
        upperPts.push(`${sx(times[i]).toFixed(1)},${sy(m + sd).toFixed(1)}`);
        lowerPts.unshift(`${sx(times[i]).toFixed(1)},${sy(m - sd).toFixed(1)}`);
      }
    }
    const bandPoints = upperPts.length >= 2 ? [...upperPts, ...lowerPts].join(" ") : "";

    return (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
        <line x1={padL} y1={padT} x2={padL} y2={H - padB} stroke="#ebebeb" />
        <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#ebebeb" />
        <line x1={sx(0)} y1={padT} x2={sx(0)} y2={H - padB} stroke="#297a3a" strokeDasharray="4 3" />
        <text x={sx(0) + 4} y={padT + 9} fill="#297a3a" fontSize="9" fontFamily="monospace">
          FFC 0ms
        </text>
        {bandPoints && <polygon points={bandPoints} fill="rgba(41,122,58,0.10)" />}
        {(spagObj.balls || []).map((b: any, i: number) => (
          <path key={i} d={buildPath(b.values)} fill="none" stroke="#c9c9c9" strokeWidth="1.2" />
        ))}
        <path d={buildPath(spagObj.mean)} fill="none" stroke="#171717" strokeWidth="2.2" />
        <text x={padL - 4} y={padT + 6} textAnchor="end" fill="#8f8f8f" fontSize="9" fontFamily="monospace">
          {yMax.toFixed(0)}°
        </text>
        <text x={padL - 4} y={H - padB} textAnchor="end" fill="#8f8f8f" fontSize="9" fontFamily="monospace">
          {yMin.toFixed(0)}°
        </text>
        <text x={padL} y={H - 8} fill="#8f8f8f" fontSize="9" fontFamily="monospace">
          {xMin} ms
        </text>
        <text x={W - padR} y={H - 8} textAnchor="end" fill="#8f8f8f" fontSize="9" fontFamily="monospace">
          +{xMax} ms
        </text>
      </svg>
    );
  }, [spagObj]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3 pb-4 border-b border-[#ebebeb]">
        <div>
          <div className="v-eyebrow">▲ COMPARE_AND_SPELL · FFC-SYNCED REAL-TIME COMPARISON</div>
          <h1 className="text-[28px] font-[450] tracking-[-0.04em] text-[#171717] mt-0.5">
            Side-by-Side Delivery Comparison &amp; Spell Workload
          </h1>
          <p className="text-[13px] text-[#4d4d4d] mt-1">
            Compare any two deliveries aligned to Front-Foot Contact (even across mixed 25–240 fps clips), fuse orthogonal
            camera views into 3D, and track spell repeatability &amp; ECB/CA workload limits.
          </p>
        </div>
      </div>

      {/* 1. Side-by-Side FFC-Synced Split-Screen Composite Video Player */}
      <div className="v-card p-4 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="v-eyebrow text-[#171717]">DELIVERY A</span>
              {detA && (
                <span className="font-mono text-[10px] text-[#666666]">
                  {detA.fact_sheet.audit.fps.toFixed(0)} fps · {detA.fact_sheet.calibration.view_class}
                </span>
              )}
            </div>
            <select value={idA} onChange={(e) => setIdA(e.target.value)} className="v-input w-full h-8 px-2.5 text-[12px]">
              {deliveries.map((d) => (
                <option key={d.id} value={d.id}>{d.label}</option>
              ))}
            </select>
            <SyncedCompositeVideo detail={detA} relMs={relMs} />
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="v-eyebrow text-[#171717]">DELIVERY B</span>
              {detB && (
                <span className="font-mono text-[10px] text-[#666666]">
                  {detB.fact_sheet.audit.fps.toFixed(0)} fps · {detB.fact_sheet.calibration.view_class}
                </span>
              )}
            </div>
            <select value={idB} onChange={(e) => setIdB(e.target.value)} className="v-input w-full h-8 px-2.5 text-[12px]">
              {deliveries.map((d) => (
                <option key={d.id} value={d.id}>{d.label}</option>
              ))}
            </select>
            <SyncedCompositeVideo detail={detB} relMs={relMs} />
          </div>
        </div>

        {/* Unified FFC Real-Time Scrubber */}
        <div className="pt-2 border-t border-[#ebebeb] space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1.5">
              <button
                onClick={() => setPlaying(!playing)}
                className="v-btn-primary h-7 px-3 text-[12px] flex items-center gap-1.5"
              >
                {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {playing ? "Pause Sync" : "Play Synced"}
              </button>
              {[
                ["BFC (-190ms)", -190],
                ["FFC (0ms)", 0],
                ["BR (+108ms)", 108],
              ].map(([lbl, ms]) => (
                <button
                  key={lbl as string}
                  onClick={() => {
                    setPlaying(false);
                    setRelMs(ms as number);
                  }}
                  className="v-btn-ghost h-7 px-2 font-mono text-[10px]"
                >
                  {lbl}
                </button>
              ))}
            </div>
            <span className="font-mono text-[11px] text-[#171717] v-tabular">
              Synced Clock: {relMs >= 0 ? `+${relMs.toFixed(0)}` : relMs.toFixed(0)} ms rel FFC
            </span>
          </div>
          <input
            type="range"
            min={minMs}
            max={maxMs}
            step={2}
            value={Math.min(Math.max(relMs, minMs), maxMs)}
            onChange={(e) => {
              setPlaying(false);
              setRelMs(Number(e.target.value));
            }}
            className="w-full accent-[#171717] cursor-pointer h-1.5"
          />
        </div>
      </div>

      {/* 2. Noise-Gated Metric Delta Table + Dual-Phone 3D Fusion */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-8 v-card p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="v-eyebrow">INTERVAL_AWARE_DELTA_TABLE</div>
              <h2 className="text-[16px] font-[500] text-[#171717]">
                Noise-Gated Biomechanical Differences (B − A)
              </h2>
            </div>
            <span className="font-mono text-[10px] text-[#666666]">
              Threshold: |Δ| ≥ 1.96 · √(σ_A² + σ_B²)
            </span>
          </div>

          {deltaRows.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-[12px]">
                <thead>
                  <tr className="border-b border-[#ebebeb] font-mono text-[10px] uppercase tracking-[0.06em] text-[#666666]">
                    <th className="py-2 pr-2">Metric</th>
                    <th className="py-2 px-2 text-right">Delivery A</th>
                    <th className="py-2 px-2 text-right">Delivery B</th>
                    <th className="py-2 px-2 text-right">Δ (B − A)</th>
                    <th className="py-2 pl-2 text-right">Verdict</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#ebebeb] font-mono text-[11px] v-tabular">
                  {deltaRows.map((r) => (
                    <tr key={r.key}>
                      <td className="py-2 pr-2 font-sans text-[12px] text-[#171717]">{r.label}</td>
                      <td className="py-2 px-2 text-right text-[#4d4d4d]">{r.valA.toFixed(1)} {r.unit}</td>
                      <td className="py-2 px-2 text-right text-[#4d4d4d]">{r.valB.toFixed(1)} {r.unit}</td>
                      <td className={`py-2 px-2 text-right font-medium ${r.meaningful ? "text-[#171717]" : "text-[#8f8f8f]"}`}>
                        {r.delta >= 0 ? `+${r.delta.toFixed(1)}` : r.delta.toFixed(1)} {r.unit}
                      </td>
                      <td className="py-2 pl-2 text-right">
                        {r.meaningful ? (
                          <span className="px-2 py-0.5 rounded-[4px] bg-[#297a3a]/[0.08] text-[#297a3a] font-medium">
                            ✓ MEANINGFUL SHIFT
                          </span>
                        ) : (
                          <span className="text-[#8f8f8f]">Within noise (±{r.thresh95.toFixed(1)})</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="p-4 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] text-[#666666]">
              No overlapping non-suppressed metrics between Delivery A and Delivery B. If A is Side-On and B is Rear-On
              of the same ball, use <strong>Dual-Phone 3D Ray-Precision Fusion</strong> on the right to combine both planes!
            </div>
          )}
        </div>

        {/* Dual-Phone 3D Fusion Card */}
        <div className="lg:col-span-4 v-card p-4 space-y-3">
          <div className="flex items-center gap-2">
            <Layers className="w-4 h-4 text-[#000000]" />
            <div>
              <div className="v-eyebrow">DUAL_PHONE_3D_FUSION</div>
              <h2 className="text-[15px] font-[500] text-[#171717]">Fuse Delivery A + B into 3D</h2>
            </div>
          </div>
          <p className="text-[12px] text-[#4d4d4d]">
            Synchronizes unsynchronized Side-On + Rear views to &lt;2.5 ms via vertical wrist/ankle motion and fuses 3D
            rays to unlock sagittal, coronal, and transverse mechanics simultaneously.
          </p>
          <div className="space-y-1">
            <label className="v-eyebrow block">Fused Delivery Label</label>
            <input
              value={fuseLabel}
              onChange={(e) => setFuseLabel(e.target.value)}
              className="v-input w-full h-8 px-2.5 text-[12px]"
            />
          </div>
          <button
            onClick={handleFuseDual}
            disabled={fuseBusy || !idA || !idB || idA === idB}
            className="v-btn-primary w-full h-9 text-[12px]"
          >
            {fuseBusy ? "Synchronizing & Fusing 3D..." : "▲ Fuse Selected A + B into 3D"}
          </button>
          {fuseMsg && (
            <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] font-mono text-[11px] text-[#171717]">
              {fuseMsg}
            </div>
          )}
        </div>
      </div>

      {/* 3. Multi-Ball Spaghetti + Repeatability + Workload Log */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        <div className="lg:col-span-7 v-card p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div>
              <div className="v-eyebrow">SPELL_SPAGHETTI &amp; REPEATABILITY</div>
              <h2 className="text-[16px] font-[500] text-[#171717]">
                Multi-Ball Waveform Consistency (Mean ± 1 SD)
              </h2>
            </div>
            <select
              value={spagMetric}
              onChange={(e) => setSpagMetric(e.target.value)}
              className="v-input h-8 px-2 font-mono text-[11px]"
            >
              <option value="front_knee_deg">Front Knee (°)</option>
              <option value="thorax_lat_flex_deg">Thorax Side Bend (°)</option>
              <option value="trunk_flex_deg">Trunk Flexion (°)</option>
              <option value="shoulder_yaw_deg">Shoulder Alignment (°)</option>
            </select>
          </div>

          <div className="rounded-[6px] bg-[#fafafa] border border-[#ebebeb] p-2">
            {spagSvg || (
              <div className="py-10 text-center font-mono text-[11px] text-[#8f8f8f]">
                Record ≥2 complete deliveries in this session to render multi-ball envelopes.
              </div>
            )}
          </div>

          {compareData?.repeatability ? (
            <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] space-y-1">
              <div className="font-mono text-[11px] text-[#171717] font-medium">
                REPEATABILITY ({compareData.completed_deliveries_count} COMPLETE BALLS):{" "}
                <span className="text-[#297a3a]">{compareData.repeatability.status_label}</span>
              </div>
              <p className="text-[#4d4d4d]">{compareData.repeatability.summary}</p>
            </div>
          ) : (
            <div className="font-mono text-[11px] text-[#666666]">
              Pillar 3 Repeatability requires ≥3 complete deliveries in this session (currently{" "}
              {compareData?.completed_deliveries_count ?? 0}).
            </div>
          )}
        </div>

        <div className="lg:col-span-5 v-card p-4 space-y-3">
          <div>
            <div className="v-eyebrow">LONGITUDINAL_WORKLOAD · ECB 2024 / CA 2026</div>
            <h2 className="text-[16px] font-[500] text-[#171717]">{session.bowler_name}</h2>
          </div>

          {workload && (
            <div className="grid grid-cols-2 gap-2 font-mono">
              <div className="v-card-subtle p-2.5">
                <div className="text-[10px] text-[#8f8f8f]">ROLLING 7 DAYS</div>
                <div className="text-[20px] font-[450] text-[#171717] v-tabular">
                  {workload.deliveries_last_7d} <span className="text-[11px] text-[#666666]">balls</span>
                </div>
              </div>
              <div className="v-card-subtle p-2.5">
                <div className="text-[10px] text-[#8f8f8f]">ROLLING 28 DAYS</div>
                <div className="text-[20px] font-[450] text-[#171717] v-tabular">
                  {workload.deliveries_last_28d} <span className="text-[11px] text-[#666666]">balls</span>
                </div>
              </div>
            </div>
          )}

          {workload?.guidance && (
            <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] text-[#4d4d4d]">
              <div className="font-mono text-[10px] uppercase text-[#171717] mb-0.5">
                {workload.guidance.age_band} · {workload.guidance.governing_body_source}
              </div>
              {workload.guidance.status_note}
            </div>
          )}

          <form onSubmit={handleAddWorkload} className="space-y-2 pt-2 border-t border-[#ebebeb]">
            <div className="v-eyebrow">LOG TODAY&apos;S BOWLING VOLUME</div>
            <div className="flex gap-2">
              <input
                type="number"
                min={0}
                max={300}
                value={logBalls}
                onChange={(e) => setLogBalls(e.target.value)}
                className="v-input w-20 h-8 px-2 font-mono text-[12px]"
                placeholder="Balls"
              />
              <select
                value={logType}
                onChange={(e) => setLogType(e.target.value)}
                className="v-input h-8 px-2 text-[12px]"
              >
                <option value="net">Net</option>
                <option value="match">Match</option>
                <option value="drill">Drill</option>
              </select>
              <input
                value={logNote}
                onChange={(e) => setLogNote(e.target.value)}
                placeholder="Session note"
                className="v-input flex-1 h-8 px-2 text-[12px]"
              />
              <button type="submit" disabled={logBusy} className="v-btn-primary h-8 px-2.5 text-[12px] flex items-center gap-1">
                <Plus className="w-3.5 h-3.5" />
                Log
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};
