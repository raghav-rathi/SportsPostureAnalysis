import React, { useEffect, useState } from "react";
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Compass,
  Layers,
  MessageSquare,
  Send,
  ShieldAlert,
  Sliders,
  Sparkles,
  Video,
} from "lucide-react";
import { api } from "../api";
import { useStudioStore, type RightTab } from "../store";
import type { DeliveryDetailResponse, SessionSummary } from "../types";

const renderInlineMd = (text: string): React.ReactNode => {
  return text.split("\n").map((line, lIdx) => {
    const parts = line.split(/(\*\*[^*]+\*\*|`[^`]+`)/g);
    return (
      <div key={lIdx} className={line.startsWith("- ") ? "pl-2" : lIdx > 0 ? "mt-1" : ""}>
        {parts.map((part, pIdx) => {
          if (part.startsWith("**") && part.endsWith("**")) {
            return <strong key={pIdx} className="text-white font-semibold">{part.slice(2, -2)}</strong>;
          }
          if (part.startsWith("`") && part.endsWith("`")) {
            return (
              <code key={pIdx} className="px-1 py-0.5 rounded bg-[#0b0f17] text-sky-300 font-mono text-[10px]">
                {part.slice(1, -1)}
              </code>
            );
          }
          return <React.Fragment key={pIdx}>{part}</React.Fragment>;
        })}
      </div>
    );
  });
};

const CALIBRATION_LANDMARKS = [
  { key: "bowling_crease_off", label: "Bowling Crease — Off Return (0, 0, -1.32m)" },
  { key: "bowling_crease_leg", label: "Bowling Crease — Leg Return (0, 0, +1.32m)" },
  { key: "popping_crease_off", label: "Popping Crease — Off Return (1.22, 0, -1.32m)" },
  { key: "popping_crease_leg", label: "Popping Crease — Leg Return (1.22, 0, +1.32m)" },
  { key: "stump_middle_base", label: "Middle Stump — Base (0, 0, 0)" },
  { key: "stump_middle_top", label: "Middle Stump — Top (0, 0.711m, 0)" },
];

interface Props {
  detail: DeliveryDetailResponse;
  session: SessionSummary;
  recomputeMs: number | null;
  onApplyCalibration: () => void;
  onTogglePanTracking: (enabled: boolean) => void;
  onDualFusedCreated: (newDeliveryId: string) => void;
}

export const StudioInspector: React.FC<Props> = ({
  detail,
  session,
  recomputeMs,
  onApplyCalibration,
  onTogglePanTracking,
  onDualFusedCreated,
}) => {
  const activeTab = useStudioStore((s) => s.activeTab);
  const setActiveTab = useStudioStore((s) => s.setActiveTab);
  const calibrationMode = useStudioStore((s) => s.calibrationMode);
  const setCalibrationMode = useStudioStore((s) => s.setCalibrationMode);
  const activeLandmarkKey = useStudioStore((s) => s.activeLandmarkKey);
  const setActiveLandmarkKey = useStudioStore((s) => s.setActiveLandmarkKey);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const clearDraftLandmarks = useStudioStore((s) => s.clearDraftLandmarks);
  const previewReprojErrPx = useStudioStore((s) => s.previewReprojErrPx);
  const ghostDeliveryId = useStudioStore((s) => s.ghostDeliveryId);
  const setGhostDeliveryId = useStudioStore((s) => s.setGhostDeliveryId);

  const fs = detail.fact_sheet;
  const [showSuppressed, setShowSuppressed] = useState(false);

  // Multi-ball comparison & workload logging state
  const [compareData, setCompareData] = useState<any>(null);
  const [workloadData, setWorkloadData] = useState<any>(null);
  const [wlDate, setWlDate] = useState<string>(() => new Date().toISOString().slice(0, 10));
  const [wlBalls, setWlBalls] = useState<number>(24);
  const [wlType, setWlType] = useState<string>("net");
  const [wlBusy, setWlBusy] = useState(false);

  // Dual-phone fusion state
  const deliveries = session.deliveries ?? [];
  const [dualA, setDualA] = useState<string>(deliveries[0]?.id ?? detail.delivery.id);
  const [dualB, setDualB] = useState<string>(deliveries[1]?.id ?? detail.delivery.id);
  const [dualBusy, setDualBusy] = useState(false);
  const [dualRes, setDualRes] = useState<any>(null);
  const [dualErr, setDualErr] = useState<string | null>(null);

  // Coach Copilot state
  const [question, setQuestion] = useState("");
  const [coachBusy, setCoachBusy] = useState(false);
  const [chatHistory, setChatHistory] = useState<
    Array<{ q: string; answer: string; verified: boolean; mode: string; violations: string[] }>
  >([]);

  // Clear Coach chat history when switching deliveries so answers from Ball A never persist on Ball B!
  useEffect(() => {
    setChatHistory([]);
  }, [detail.delivery.id]);

  useEffect(() => {
    if (activeTab === "compare") {
      api.compareSession(session.id).then(setCompareData).catch(() => {});
      api.getWorkload(session.bowler_id).then(setWorkloadData).catch(() => {});
    }
  }, [activeTab, session.id, session.bowler_id, detail.delivery.id]);

  const handleLogWorkload = async () => {
    setWlBusy(true);
    try {
      await api.addWorkload(session.bowler_id, {
        log_date: wlDate,
        deliveries_count: Math.max(0, Number(wlBalls) || 0),
        session_type: wlType,
        notes: "Logged via Coaching Studio",
      });
      const updated = await api.getWorkload(session.bowler_id);
      setWorkloadData(updated);
    } finally {
      setWlBusy(false);
    }
  };

  useEffect(() => {
    if (deliveries.length >= 2) {
      setDualA(deliveries[0].id);
      setDualB(deliveries[1].id);
    }
  }, [session.id, deliveries.length]);

  const validMetrics = Object.values(fs.metrics).filter((m) => !m.suppressed && m.value !== null);
  const suppressedMetrics = Object.values(fs.metrics).filter((m) => m.suppressed && m.observability !== "refused");
  const refusedMetrics = Object.values(fs.metrics).filter((m) => m.observability === "refused");

  const handleAsk = async (qOverride?: string) => {
    const qText = (qOverride ?? question).trim();
    if (!qText) return;
    setCoachBusy(true);
    try {
      const r = await api.askCoach(detail.delivery.id, qText, false);
      setChatHistory((prev) => [
        ...prev,
        {
          q: qText,
          answer: r.answer_markdown,
          verified: r.verification_passed,
          mode: r.generator_mode,
          violations: r.blocked_violations,
        },
      ]);
      if (!qOverride) setQuestion("");
    } finally {
      setCoachBusy(false);
    }
  };

  const handleDualFuse = async () => {
    if (dualA === dualB) {
      setDualErr("Select two distinct camera views (e.g., Side-On + Rear-On) to fuse.");
      return;
    }
    setDualBusy(true);
    setDualErr(null);
    try {
      const r = await api.fuseDual(session.id, dualA, dualB, "Dual-Phone 3D Fused Delivery");
      setDualRes(r);
      onDualFusedCreated(r.delivery_id);
    } catch (e: any) {
      setDualErr(String(e.message || e));
    } finally {
      setDualBusy(false);
    }
  };

  const tabs: Array<[RightTab, string, React.ReactNode]> = [
    ["metrics", "Metrics", <Activity className="w-3.5 h-3.5" key="m" />],
    ["calibration", "4-Click Cal", <Compass className="w-3.5 h-3.5" key="c" />],
    ["compare", "Multi-Ball", <Layers className="w-3.5 h-3.5" key="p" />],
    ["dual", "Dual 3D", <Video className="w-3.5 h-3.5" key="d" />],
    ["coach", "Coach AI", <MessageSquare className="w-3.5 h-3.5" key="a" />],
  ];

  const obsBadge = (obs: string) => {
    if (obs === "high") return "bg-emerald-500/20 text-emerald-300 border-emerald-500/40";
    if (obs === "moderate") return "bg-sky-500/20 text-sky-300 border-sky-500/40";
    if (obs === "low") return "bg-amber-500/20 text-amber-300 border-amber-500/40";
    return "bg-rose-500/20 text-rose-300 border-rose-500/40";
  };

  return (
    <div className="flex flex-col bg-[#131926] border border-[#263248] rounded-xl overflow-hidden h-full max-h-[820px]">
      {/* Tab Strip */}
      <div className="grid grid-cols-5 border-b border-[#263248] bg-[#161e2e]">
        {tabs.map(([id, lbl, icon]) => (
          <button
            key={id}
            onClick={() => {
              setActiveTab(id);
              if (id !== "calibration" && calibrationMode) setCalibrationMode(false);
            }}
            className={`flex items-center justify-center gap-1 py-2.5 text-[11px] font-semibold transition border-b-2 ${
              activeTab === id
                ? "border-sky-400 text-sky-300 bg-[#1a2438]"
                : "border-transparent text-slate-400 hover:text-slate-200"
            }`}
          >
            {icon}
            <span className="truncate">{lbl}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 text-xs">
        {/* TAB 1: VERIFIED METRICS, COHORT NORMS & CUES */}
        {activeTab === "metrics" && (
          <>
            {/* Interval semantics & recompute latency */}
            <div className="p-2.5 rounded-lg bg-[#161f30] border border-[#263248] space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[11px] font-semibold text-sky-300">
                  {fs.interval_type === "repeatability_interval"
                    ? "Empirical Repeatability Intervals"
                    : "Analytical Uncertainty Intervals"}
                </span>
                {recomputeMs !== null && (
                  <span className="font-mono text-[10px] text-emerald-300 bg-emerald-500/15 px-1.5 py-0.5 rounded">
                    CPU HITL: {recomputeMs.toFixed(1)} ms
                  </span>
                )}
              </div>
              <p className="text-[11px] text-slate-400">
                Unvalidated 0–100 composite scores are retired. Action class:{" "}
                <strong className="text-slate-200 uppercase">{fs.action_class.replace("_", " ")}</strong> •{" "}
                <strong className="text-slate-200">{validMetrics.length}</strong> verified /{" "}
                <strong className="text-slate-400">{suppressedMetrics.length}</strong> withheld.
              </p>
              {detail.telemetry.needs_gpu_refit_for_K && (
                <div className="mt-1 p-1.5 rounded bg-amber-500/15 border border-amber-500/40 text-amber-200 text-[11px]">
                  Focal length shifted &gt;15% from cached NLF pass — re-run GPU inference with calibrated K for metric depth.
                </div>
              )}
            </div>

            {!fs.events.is_complete_delivery && (
              <div className="p-3 rounded-lg bg-amber-500/15 border border-amber-500/40 text-amber-200 space-y-1">
                <div className="font-bold flex items-center gap-1.5">
                  <AlertTriangle className="w-4 h-4" /> Incomplete Delivery — All Metrics Suppressed
                </div>
                <p className="text-[11px]">
                  No verified overhead ball release passed sanity gates (`{fs.events.br.detection_method}`).
                  Zero coaching cues or scores are issued on truncated or freeze-frame clips.
                </p>
              </div>
            )}

            {/* Cohort Z-Score Forest Plot Cards */}
            {fs.cohort_comparisons.length > 0 && (
              <div className="p-2.5 rounded-lg bg-[#111723] border border-[#263248] space-y-2">
                <div className="text-[11px] font-semibold text-slate-200 flex items-center justify-between">
                  <span>Primary-Source Cohort Z-Scores (±1 SD shaded)</span>
                  <span className="text-[10px] text-slate-400">Blue=exact • Amber=proxy</span>
                </div>
                {fs.cohort_comparisons.map((c) => {
                  const zLo = Math.max(-3.2, Math.min(3.2, (c.ci_low - c.cohort_mean) / Math.max(c.cohort_sd, 1e-6)));
                  const zHi = Math.max(-3.2, Math.min(3.2, (c.ci_high - c.cohort_mean) / Math.max(c.cohort_sd, 1e-6)));
                  const zPt = Math.max(-3.2, Math.min(3.2, c.z_score));
                  const toPct = (z: number) => ((z + 3.2) / 6.4) * 100;
                  const exact = c.definition_match === "exact";
                  return (
                    <div key={c.metric_key} className="space-y-1">
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="text-slate-200 font-medium">{c.metric_key.replace(/_deg|_mps/g, "")}</span>
                        <span className="font-mono text-slate-400">
                          z={c.z_score >= 0 ? "+" : ""}{c.z_score.toFixed(2)} ({exact ? "exact" : "proxy"})
                        </span>
                      </div>
                      <div className="relative h-3 rounded bg-[#1b2436] overflow-hidden">
                        {/* +-1 SD shaded band */}
                        <div
                          className="absolute inset-y-0 bg-emerald-500/20"
                          style={{ left: `${toPct(-1)}%`, width: `${toPct(1) - toPct(-1)}%` }}
                        />
                        <div className="absolute inset-y-0 w-px bg-slate-500/60" style={{ left: "50%" }} />
                        {/* Interval bar */}
                        <div
                          className={`absolute top-1 h-1 rounded ${exact ? "bg-sky-400" : "bg-amber-400"}`}
                          style={{ left: `${toPct(zLo)}%`, width: `${Math.max(2, toPct(zHi) - toPct(zLo))}%` }}
                        />
                        <div
                          className="absolute top-0.5 w-2 h-2 rounded-full bg-white border border-slate-900 -translate-x-1/2"
                          style={{ left: `${toPct(zPt)}%` }}
                        />
                      </div>
                      <div className="text-[10px] text-slate-500 truncate">
                        {c.cohort_name.toLowerCase().includes(c.source.split(" ")[0].toLowerCase().replace(/[^a-z0-9]/g, ""))
                          ? c.cohort_name
                          : `${c.cohort_name} (${c.source})`}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Verified Metric Cards */}
            <div className="space-y-1.5">
              {validMetrics.map((m) => (
                <div key={m.key} className="p-2.5 rounded-lg bg-[#161e2e] border border-[#263248]">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-slate-100">{m.label}</span>
                    <span className={`px-1.5 py-0.5 rounded border text-[10px] font-mono uppercase ${obsBadge(m.observability)}`}>
                      {m.observability}
                    </span>
                  </div>
                  <div className="flex items-baseline justify-between mt-1">
                    <span className="text-base font-bold font-mono text-sky-300">
                      {m.value} <span className="text-xs font-normal text-slate-400">{m.unit}</span>
                    </span>
                    <span className="font-mono text-[11px] text-slate-300">
                      [{m.ci_low}, {m.ci_high}] {m.unit}
                    </span>
                  </div>
                  <div className="text-[10px] text-slate-400 mt-1">{m.empirical_reference}</div>
                </div>
              ))}
            </div>

            {/* Optional ISB bml_movi_87 Rigid-Body Anatomy Card (Guardrail #3: CIs + observability or explicit suppression!) */}
            {detail.delivery.isb_summary && (
              <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30 space-y-1.5">
                <div className="font-semibold text-indigo-300 flex items-center justify-between">
                  <span>ISB Rigid-Body Anatomy (bml_movi_87)</span>
                  <span className="font-mono text-[10px]">
                    N_eff={detail.delivery.isb_summary.n_eff_pelvis} (ρ={detail.delivery.isb_summary.rho_assumed})
                  </span>
                </div>
                {detail.delivery.isb_summary.metrics ? (
                  <div className="space-y-1 font-mono text-[11px] text-slate-200">
                    {Object.values(detail.delivery.isb_summary.metrics).map((im) => (
                      <div key={im.key} className="bg-[#0f1522] px-2 py-1 rounded">
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[10px] text-slate-300 truncate">{im.label}</span>
                          <span className={`px-1 py-0.5 rounded border text-[9px] uppercase ${obsBadge(im.observability)}`}>
                            {im.suppressed ? "suppressed" : im.observability}
                          </span>
                        </div>
                        {im.suppressed || im.value === null ? (
                          <div className="text-[9px] text-slate-400 mt-0.5">{im.suppression_reason}</div>
                        ) : (
                          <div className="flex justify-between text-[11px] mt-0.5">
                            <span className="text-indigo-200 font-bold">{im.value}°</span>
                            <span className="text-slate-400">[{im.ci_low}, {im.ci_high}]°</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="text-[10px] text-slate-400">
                    Recompute delivery to propagate correlated-depth N_eff intervals for ISB anatomy.
                  </div>
                )}
              </div>
            )}

            {/* Verified Strengths / Limiters / Inconclusive Cues */}
            {(fs.strengths.length > 0 || fs.limiters.length > 0 || fs.inconclusive_observations.length > 0) && (
              <div className="space-y-1.5">
                {fs.strengths.map((c, i) => (
                  <div key={`s-${i}`} className="p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
                    <div className="font-semibold text-emerald-300 flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5" /> {c.title}
                    </div>
                    <p className="text-[11px] text-slate-300 mt-0.5">{c.biomechanical_mechanism}</p>
                  </div>
                ))}
                {fs.limiters.map((c, i) => (
                  <div key={`l-${i}`} className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/30">
                    <div className="font-semibold text-amber-300 flex items-center gap-1">
                      <AlertTriangle className="w-3.5 h-3.5" /> {c.title}
                    </div>
                    <p className="text-[11px] text-slate-300 mt-0.5">{c.actionable_recommendation}</p>
                  </div>
                ))}
                {fs.inconclusive_observations.map((c, i) => (
                  <div key={`inc-${i}`} className="p-2 rounded-lg bg-slate-800/60 border border-slate-700">
                    <div className="font-semibold text-slate-300">{c.title}</div>
                    <p className="text-[11px] text-slate-400 mt-0.5">{c.biomechanical_mechanism}</p>
                  </div>
                ))}
              </div>
            )}

            {/* Suppressed Metrics Collapsible Drawer */}
            <div className="border-t border-[#263248] pt-2">
              <button
                onClick={() => setShowSuppressed(!showSuppressed)}
                className="w-full text-left text-[11px] text-slate-400 hover:text-slate-200 font-semibold flex items-center justify-between"
              >
                <span>Withheld / Suppressed Metrics ({suppressedMetrics.length})</span>
                <span>{showSuppressed ? "Hide ▲" : "Show ▼"}</span>
              </button>
              {showSuppressed && (
                <div className="mt-2 space-y-1.5">
                  {suppressedMetrics.map((m) => (
                    <div key={m.key} className="p-2 rounded bg-[#0f1522] border border-[#1f2a3d] opacity-80">
                      <div className="font-semibold text-slate-300 text-[11px]">{m.label}</div>
                      <div className="text-[10px] text-slate-400 mt-0.5">{m.suppression_reason}</div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Permanent ICC 15-Degree Refusal Card */}
            {refusedMetrics.map((rm) => (
              <div key={rm.key} className="p-2.5 rounded-lg bg-rose-950/50 border border-rose-500/50 text-rose-200">
                <div className="font-bold flex items-center gap-1.5 text-[11px]">
                  <ShieldAlert className="w-4 h-4 text-rose-400 shrink-0" />
                  {rm.label}: PERMANENTLY REFUSED
                </div>
                <p className="text-[10px] text-rose-200/85 mt-1">{rm.suppression_reason}</p>
              </div>
            ))}
          </>
        )}

        {/* TAB 2: 4-CLICK MCC CREASE CALIBRATION & 3-DOF PAN TRACKING */}
        {activeTab === "calibration" && (
          <div className="space-y-3">
            <div className="p-2.5 rounded-lg bg-[#161f30] border border-[#263248] space-y-1">
              <div className="font-semibold text-slate-200">Current Pitch Calibration</div>
              <div className="font-mono text-[11px] text-slate-300">
                Method: <span className="text-sky-300">{fs.calibration.method}</span>
              </div>
              <div className="font-mono text-[11px] text-slate-400">
                Az: {fs.calibration.azimuth_deg.toFixed(1)}° • El: {fs.calibration.elevation_deg.toFixed(1)}° •
                Metric: {String(fs.calibration.is_metric_calibrated)} • Pan: {String(fs.calibration.is_panning_camera)}
              </div>
            </div>

            <div className="flex items-center gap-2">
              <button
                onClick={() => setCalibrationMode(!calibrationMode)}
                className={`flex-1 py-2 rounded-lg font-semibold text-xs transition ${
                  calibrationMode
                    ? "bg-amber-500 text-slate-950"
                    : "bg-sky-500/20 text-sky-300 border border-sky-500/40 hover:bg-sky-500/30"
                }`}
              >
                {calibrationMode ? "Exit 4-Click Placement Mode" : "Start 4-Click MCC Calibration (4× Loupe)"}
              </button>
              <button
                onClick={clearDraftLandmarks}
                className="px-2.5 py-2 rounded-lg bg-[#1a2233] text-slate-300 hover:bg-slate-700"
              >
                Reset
              </button>
            </div>

            <div className="space-y-1">
              <div className="text-[11px] font-semibold text-slate-300">
                Select landmark, then click on video canvas ({Object.keys(draftLandmarks).length}/4+ placed):
              </div>
              {CALIBRATION_LANDMARKS.map((lm) => {
                const pt = draftLandmarks[lm.key];
                const isActive = activeLandmarkKey === lm.key && calibrationMode;
                return (
                  <button
                    key={lm.key}
                    onClick={() => {
                      setCalibrationMode(true);
                      setActiveLandmarkKey(lm.key);
                    }}
                    className={`w-full text-left px-2.5 py-1.5 rounded border text-[11px] flex items-center justify-between transition ${
                      isActive
                        ? "bg-amber-500/20 border-amber-400 text-amber-200"
                        : pt
                        ? "bg-emerald-500/10 border-emerald-500/40 text-emerald-200"
                        : "bg-[#111723] border-[#263248] text-slate-300 hover:bg-[#1a2233]"
                    }`}
                  >
                    <span>{lm.label}</span>
                    <span className="font-mono text-[10px]">
                      {pt ? `(${pt[0].toFixed(0)}, ${pt[1].toFixed(0)})` : "click"}
                    </span>
                  </button>
                );
              })}
            </div>

            {previewReprojErrPx !== null && (
              <div className="p-2 rounded bg-emerald-500/15 border border-emerald-500/40 text-emerald-200 font-mono text-[11px]">
                Live PnP Preview Reprojection Error: {previewReprojErrPx.toFixed(2)} px
              </div>
            )}

            <button
              disabled={Object.keys(draftLandmarks).length < 4}
              onClick={onApplyCalibration}
              className="w-full py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 disabled:opacity-40 text-slate-950 font-bold"
            >
              Save Calibration & Recompute (&lt;35 ms CPU)
            </button>

            {/* 3-DOF Rotational Pan Tracking Toggle */}
            <div className="p-2.5 rounded-lg bg-[#111723] border border-[#263248] space-y-2">
              <div className="font-semibold text-slate-200 flex items-center gap-1.5">
                <Sliders className="w-3.5 h-3.5 text-sky-400" /> 3-DOF Rotational Background Pan Tracker
              </div>
              <p className="text-[11px] text-slate-400">
                Tracks bowler-masked background KLT features as pure SO(3) camera rotation anchored on MCC PnP.
                Unlocks run-up speed & stride only if median reprojection residual ≤ 2.5 px.
              </p>
              <button
                onClick={() => onTogglePanTracking(!detail.delivery.overrides?.enable_pan_tracking)}
                className="w-full py-1.5 rounded bg-[#1a2233] hover:bg-slate-700 border border-[#263248] text-slate-200 font-semibold"
              >
                {detail.delivery.overrides?.enable_pan_tracking
                  ? "Disable 3-DOF Pan Tracking"
                  : "Run Verified 3-DOF Pan Tracking"}
              </button>
              {detail.delivery.pan_summary && (
                <div className="text-[10px] font-mono text-slate-300 bg-[#0b0f17] p-2 rounded">
                  {detail.delivery.pan_summary.verification_reason}
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 3: MULTI-BALL REPEATABILITY, GHOST OVERLAY & WORKLOAD */}
        {activeTab === "compare" && (
          <div className="space-y-3">
            <div className="p-2.5 rounded-lg bg-[#161f30] border border-[#263248] space-y-2">
              <div className="font-semibold text-slate-200">3D Ghost Skeleton Overlay (FFC-Aligned)</div>
              <select
                value={ghostDeliveryId ?? ""}
                onChange={(e) => setGhostDeliveryId(e.target.value || null)}
                className="w-full bg-[#0b0f17] border border-[#263248] rounded px-2 py-1.5 text-slate-200 text-xs"
              >
                <option value="">None (Single 3D Skeleton)</option>
                {deliveries
                  .filter((d) => d.id !== detail.delivery.id)
                  .map((d) => (
                    <option key={d.id} value={d.id}>
                      Overlay Amber Ghost: {d.label}
                    </option>
                  ))}
              </select>
            </div>

            {compareData && (
              <div className="p-2.5 rounded-lg bg-[#111723] border border-[#263248] space-y-2">
                <div className="font-semibold text-slate-200 flex items-center justify-between">
                  <span>Spell Repeatability ({compareData.completed_deliveries_count} complete balls)</span>
                </div>
                {compareData.repeatability ? (
                  <div className="space-y-1 text-[11px]">
                    <div className="text-sky-300 font-semibold">{compareData.repeatability.status_label}</div>
                    <p className="text-slate-300">{compareData.repeatability.summary}</p>
                  </div>
                ) : (
                  <p className="text-[11px] text-slate-400">Record ≥2 complete deliveries in this session to score repeatability.</p>
                )}

                {compareData.spell_drift_points && (
                  <div className="space-y-1.5 pt-1 border-t border-[#263248]">
                    <div className="text-[11px] font-semibold text-slate-300">Within-Spell Metric Progression</div>
                    {Object.entries(compareData.spell_drift_points).map(([mk, pts]: [string, any]) =>
                      pts.length > 0 ? (
                        <div key={mk} className="bg-[#161e2e] p-2 rounded">
                          <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                            <span>{mk}</span>
                            {compareData.spell_drift_slope_per_ball?.[mk] !== null && (
                              <span>slope: {compareData.spell_drift_slope_per_ball[mk]}/ball</span>
                            )}
                          </div>
                          <div className="flex flex-wrap gap-1.5 mt-1 font-mono text-[11px]">
                            {pts.map((p: any) => (
                              <span key={p.ball_index} className="px-1.5 py-0.5 rounded bg-[#0b0f17] text-sky-300">
                                B{p.ball_index}: {p.value}
                              </span>
                            ))}
                          </div>
                        </div>
                      ) : null
                    )}
                  </div>
                )}
                {/* FFC-Aligned Multi-Ball Spaghetti Envelope (mean +- 1 SD) */}
                {compareData.spaghetti_curves?.front_knee_deg && (
                  <div className="pt-1.5 border-t border-[#263248] space-y-1">
                    <div className="text-[11px] font-semibold text-slate-300">
                      FFC-Aligned Front-Knee Spaghetti (Mean ± 1 SD)
                    </div>
                    {(() => {
                      const sp = compareData.spaghetti_curves.front_knee_deg;
                      const tArr: number[] = sp.time_ms ?? [];
                      const mu: Array<number | null> = sp.mean ?? [];
                      const sd: Array<number | null> = sp.sd ?? [];
                      const balls: Array<{ label: string; values: Array<number | null> }> = sp.balls ?? [];
                      const W = 320, H = 96;
                      const toX = (tm: number) => ((tm + 300) / 520) * W;
                      const toY = (v: number) => H - ((Math.min(Math.max(v, 70), 190) - 70) / 120) * H;
                      const polyPts = (arr: Array<number | null>) =>
                        arr
                          .map((v, i) => (v !== null && Number.isFinite(v) ? `${toX(tArr[i]).toFixed(1)},${toY(v).toFixed(1)}` : null))
                          .filter(Boolean)
                          .join(" ");
                      const bandUpper = mu.map((m, i) =>
                        m !== null && sd[i] !== null ? `${toX(tArr[i]).toFixed(1)},${toY(m + (sd[i] as number)).toFixed(1)}` : null
                      ).filter(Boolean);
                      const bandLower = mu.map((m, i) =>
                        m !== null && sd[i] !== null ? `${toX(tArr[i]).toFixed(1)},${toY(m - (sd[i] as number)).toFixed(1)}` : null
                      ).filter(Boolean).reverse();
                      return (
                        <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-24 bg-[#0b0f17] rounded border border-[#1f2a3d]">
                          <line x1={toX(0)} y1={0} x2={toX(0)} y2={H} stroke="#38bdf8" strokeDasharray="3 2" strokeWidth="1" opacity="0.6" />
                          {bandUpper.length > 1 && (
                            <polygon points={[...bandUpper, ...bandLower].join(" ")} fill="#38bdf8" opacity="0.18" />
                          )}
                          {balls.map((b, bi) => (
                            <polyline key={bi} fill="none" stroke="#94a3b8" strokeWidth="1" opacity="0.45" points={polyPts(b.values)} />
                          ))}
                          <polyline fill="none" stroke="#38bdf8" strokeWidth="2" points={polyPts(mu)} />
                        </svg>
                      );
                    })()}
                  </div>
                )}
              </div>
            )}

            {workloadData && (
              <div className="p-2.5 rounded-lg bg-[#111723] border border-[#263248] space-y-2">
                <div className="font-semibold text-slate-200 flex items-center justify-between">
                  <span>Longitudinal Workload Log</span>
                  <span className="font-mono text-[10px] text-sky-300">
                    {workloadData.deliveries_last_7d} / 7d • {workloadData.deliveries_last_28d} / 28d
                  </span>
                </div>
                {workloadData.guidance && (
                  <>
                    <p className="text-[11px] text-slate-300">{workloadData.guidance.status_note}</p>
                    <div className="text-[10px] text-slate-500">{workloadData.guidance.governing_body_source}</div>
                  </>
                )}
                <div className="grid grid-cols-3 gap-1.5 pt-1 border-t border-[#263248]">
                  <input
                    type="date"
                    value={wlDate}
                    onChange={(e) => setWlDate(e.target.value)}
                    className="bg-[#0b0f17] border border-[#263248] rounded px-1.5 py-1 text-[10px] text-slate-200"
                  />
                  <input
                    type="number"
                    min={0}
                    max={300}
                    value={wlBalls}
                    onChange={(e) => setWlBalls(Number(e.target.value))}
                    className="bg-[#0b0f17] border border-[#263248] rounded px-1.5 py-1 text-[10px] text-slate-200"
                    placeholder="Balls"
                  />
                  <select
                    value={wlType}
                    onChange={(e) => setWlType(e.target.value)}
                    className="bg-[#0b0f17] border border-[#263248] rounded px-1 py-1 text-[10px] text-slate-200"
                  >
                    <option value="net">Net</option>
                    <option value="match">Match</option>
                    <option value="drill">Drill</option>
                  </select>
                </div>
                <button
                  onClick={handleLogWorkload}
                  disabled={wlBusy}
                  className="w-full py-1.5 rounded bg-sky-500/20 hover:bg-sky-500/30 border border-sky-500/40 text-sky-300 font-semibold text-[11px]"
                >
                  {wlBusy ? "Saving Log..." : "Add Workload Entry"}
                </button>
              </div>
            )}
          </div>
        )}

        {/* TAB 4: UNSYNCHRONIZED DUAL-PHONE 3D RAY-PRECISION FUSION (M2C) */}
        {activeTab === "dual" && (
          <div className="space-y-3">
            <div className="p-2.5 rounded-lg bg-[#161f30] border border-[#263248] space-y-1">
              <div className="font-semibold text-sky-300">Academy Mode: Unsynchronized Dual-Phone 3D Fusion</div>
              <p className="text-[11px] text-slate-400">
                Sub-frame +Y_world vertical velocity sync (1000 Hz spline NCC) + per-joint anisotropic ray-precision
                3D fusion. Collinear cameras automatically keep unobservable planes suppressed; ICC elbow legality
                stays permanently REFUSED.
              </p>
            </div>

            <div className="space-y-2">
              <label className="block text-[11px] text-slate-300">
                Camera View A (e.g., Side-On 90°):
                <select
                  value={dualA}
                  onChange={(e) => setDualA(e.target.value)}
                  className="mt-1 w-full bg-[#0b0f17] border border-[#263248] rounded px-2 py-1.5 text-slate-200"
                >
                  {deliveries.map((d) => (
                    <option key={d.id} value={d.id}>{d.label}</option>
                  ))}
                </select>
              </label>

              <label className="block text-[11px] text-slate-300">
                Camera View B (e.g., Rear 0°):
                <select
                  value={dualB}
                  onChange={(e) => setDualB(e.target.value)}
                  className="mt-1 w-full bg-[#0b0f17] border border-[#263248] rounded px-2 py-1.5 text-slate-200"
                >
                  {deliveries.map((d) => (
                    <option key={d.id} value={d.id}>{d.label}</option>
                  ))}
                </select>
              </label>

              <button
                onClick={handleDualFuse}
                disabled={dualBusy || deliveries.length < 2}
                className="w-full py-2 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 font-bold"
              >
                {dualBusy ? "Synchronizing & Fusing 3D Rays..." : "Run Dual-Phone 3D Ray-Precision Fusion"}
              </button>

              {dualErr && (
                <div className="p-2 rounded bg-rose-950/60 border border-rose-500/40 text-rose-200 text-[11px]">
                  {dualErr}
                </div>
              )}

              {dualRes && (
                <div className="p-2.5 rounded-lg bg-emerald-500/10 border border-emerald-500/40 space-y-1 font-mono text-[11px] text-emerald-200">
                  <div className="font-bold">Unified 3D Skeleton Created!</div>
                  <div>Sync Offset: {(dualRes.sync.offset_b_minus_a_s * 1000).toFixed(2)} ms (NCC={dualRes.sync.sync_score})</div>
                  <div>Baseline Separation: {dualRes.baseline_angle_deg}° (|sin|={dualRes.baseline_sin})</div>
                  <div>Alignment: {dualRes.alignment_method}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 5: DETERMINISTIC NUMERIC-VERIFIER COACH COPILOT */}
        {activeTab === "coach" && (
          <div className="flex flex-col h-full space-y-2.5">
            <div className="p-2 rounded-lg bg-[#161f30] border border-[#263248] text-[11px] text-slate-300 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-sky-400 shrink-0" />
              <span>
                Every number is deterministically verified against non-suppressed FactSheet bounds. ICC legality claims
                and hallucinated metrics are hard-blocked.
              </span>
            </div>

            <div className="flex flex-wrap gap-1">
              {[
                "Summarize verified measurements",
                "Why was front knee suppressed?",
                "Is this bowling action ICC legal?",
                "What does this imply for lumbar load?",
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => handleAsk(q)}
                  className="px-2 py-1 rounded bg-[#1a2233] hover:bg-slate-700 text-[10px] text-sky-300 border border-[#263248]"
                >
                  {q}
                </button>
              ))}
            </div>

            <div className="flex-1 overflow-y-auto space-y-2 min-h-[180px]">
              {chatHistory.length === 0 && (
                <div className="text-slate-500 text-center py-8 text-[11px]">
                  Ask a question about this delivery or click a prompt chip above.
                </div>
              )}
              {chatHistory.map((item, i) => (
                <div key={i} className="space-y-1.5">
                  <div className="bg-sky-500/15 border border-sky-500/30 rounded-lg px-2.5 py-1.5 text-sky-100 font-medium">
                    {item.q}
                  </div>
                  <div className="bg-[#111723] border border-[#263248] rounded-lg px-2.5 py-2 space-y-1">
                    <div className="flex items-center justify-between text-[10px] font-mono">
                      <span className={item.verified ? "text-emerald-400" : "text-amber-400"}>
                        {item.verified ? "✓ NUMERIC VERIFIER PASSED" : "⚠ HALLUCINATION BLOCKED (FALLBACK)"}
                      </span>
                      <span className="text-slate-500">{item.mode}</span>
                    </div>
                    <div className="text-[11px] text-slate-200 leading-relaxed">
                      {renderInlineMd(item.answer)}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleAsk();
              }}
              className="flex gap-1.5 pt-1"
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Ask grounded biomechanics question..."
                className="flex-1 bg-[#0b0f17] border border-[#263248] rounded-lg px-2.5 py-1.5 text-xs text-slate-100 focus:outline-none focus:border-sky-400"
              />
              <button
                type="submit"
                disabled={coachBusy || !question.trim()}
                className="px-3 py-1.5 rounded-lg bg-sky-500 hover:bg-sky-400 disabled:opacity-40 text-slate-950 font-semibold"
              >
                <Send className="w-3.5 h-3.5" />
              </button>
            </form>
          </div>
        )}
      </div>
    </div>
  );
};
