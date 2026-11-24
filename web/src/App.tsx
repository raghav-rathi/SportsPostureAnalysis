import React, { useEffect, useState } from "react";
import { Cpu, Film, ShieldAlert, AlertTriangle, CheckCircle2, Upload, Scissors, X } from "lucide-react";
import { api } from "./api";
import { useStudioStore } from "./store";
import type { DeliveryDetailResponse, SessionSummary, TelemetryV2 } from "./types";
import { VideoCanvasPlayer } from "./components/VideoCanvasPlayer";
import { PitchViewport3D } from "./components/PitchViewport3D";
import { WaveformPanel } from "./components/WaveformPanel";
import { StudioInspector } from "./components/StudioInspector";

export function App() {
  const [health, setHealth] = useState<any>(null);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSession, setActiveSession] = useState<SessionSummary | null>(null);
  const [activeDeliveryId, setActiveDeliveryId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DeliveryDetailResponse | null>(null);
  const [ghostTelemetry, setGhostTelemetry] = useState<TelemetryV2 | null>(null);
  const [recomputeMs, setRecomputeMs] = useState<number | null>(null);
  const [reelBusy, setReelBusy] = useState(false);
  const [reelUrl, setReelUrl] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Upload & Net-Session Auto-Clipper drawer state
  const [showIngest, setShowIngest] = useState(false);
  const [ingestFile, setIngestFile] = useState<File | null>(null);
  const [ingestLabel, setIngestLabel] = useState("New Delivery");
  const [ingestStart, setIngestStart] = useState<string>("0");
  const [ingestEnd, setIngestEnd] = useState<string>("");
  const [ingestArm, setIngestArm] = useState<string>("");
  const [ingestBusy, setIngestBusy] = useState(false);
  const [scanCands, setScanCands] = useState<any[]>([]);
  const [ingestMsg, setIngestMsg] = useState<string | null>(null);

  const ghostDeliveryId = useStudioStore((s) => s.ghostDeliveryId);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const setDraftLandmark = useStudioStore((s) => s.setDraftLandmark);
  const clearDraftLandmarks = useStudioStore((s) => s.clearDraftLandmarks);
  const setPreviewCalibration = useStudioStore((s) => s.setPreviewCalibration);
  const setCalibrationMode = useStudioStore((s) => s.setCalibrationMode);
  const setCurrentFrame = useStudioStore((s) => s.setCurrentFrame);

  const loadSession = async (sid: string, selectDeliveryId?: string) => {
    const r = await api.getSession(sid);
    setActiveSession(r.session);
    const dels = r.session.deliveries ?? [];
    const target = selectDeliveryId ?? dels[0]?.id ?? null;
    if (target) {
      clearDraftLandmarks();
      setCalibrationMode(false);
      setActiveDeliveryId(target);
      const d = await api.getDelivery(target);
      setDetail(d);
      setReelUrl(null);
      setCurrentFrame(d.fact_sheet.events.is_complete_delivery ? d.fact_sheet.events.br.frame_idx : 0);
    }
  };

  useEffect(() => {
    (async () => {
      try {
        const [h, s] = await Promise.all([api.health(), api.listSessions()]);
        setHealth(h);
        setSessions(s.sessions);
        if (s.sessions.length > 0) {
          await loadSession(s.sessions[0].id);
        }
      } catch (e: any) {
        setErrorMsg(String(e.message || e));
      }
    })();
  }, []);

  // Load ghost delivery telemetry when selected for FFC-aligned ms-interpolated 3D comparison
  useEffect(() => {
    if (!ghostDeliveryId || !detail) {
      setGhostTelemetry(null);
      return;
    }
    api.getDelivery(ghostDeliveryId).then((g) => {
      setGhostTelemetry(g.telemetry);
    }).catch(() => setGhostTelemetry(null));
  }, [ghostDeliveryId, detail?.delivery.id]);

  const handleSelectDelivery = async (did: string) => {
    clearDraftLandmarks();
    setCalibrationMode(false);
    setActiveDeliveryId(did);
    setReelUrl(null);
    const d = await api.getDelivery(did);
    setDetail(d);
    setCurrentFrame(d.fact_sheet.events.is_complete_delivery ? d.fact_sheet.events.br.frame_idx : 0);
  };

  const handleRecompute = async (payload: Record<string, any>) => {
    if (!activeDeliveryId) return;
    const r = await api.recomputeDelivery(activeDeliveryId, payload);
    setRecomputeMs(r.recompute_ms);
    setDetail((prev) =>
      prev
        ? {
            ...prev,
            fact_sheet: r.fact_sheet,
            telemetry: r.telemetry,
            report_markdown: r.report_markdown,
            delivery: {
              ...prev.delivery,
              overrides: { ...prev.delivery.overrides, ...payload },
              isb_summary: r.isb_summary,
              pan_summary: r.pan_summary,
            },
          }
        : prev
    );
  };

  const handleLandmarkClick = async (key: string, pt: [number, number]) => {
    setDraftLandmark(key, pt);
    const nextMap = { ...draftLandmarks, [key]: pt };
    if (activeDeliveryId && Object.keys(nextMap).length >= 4) {
      try {
        const prev = await api.previewCalibration(activeDeliveryId, nextMap);
        setPreviewCalibration(prev.mcc_wireframe_2d, prev.reprojection_error_px);
      } catch {
        setPreviewCalibration(null, null);
      }
    }
  };

  const handleApplyCalibration = async () => {
    const anchorFrame = useStudioStore.getState().currentFrame;
    await handleRecompute({ mcc_landmarks_px: draftLandmarks, pan_anchor_frame: anchorFrame });
    setCalibrationMode(false);
    setPreviewCalibration(null, null);
  };

  const handleExportReel = async () => {
    if (!activeDeliveryId) return;
    setReelBusy(true);
    try {
      const r = await api.exportReel(activeDeliveryId);
      setReelUrl(r.reel_url);
    } finally {
      setReelBusy(false);
    }
  };

  const handleScanNetSession = async () => {
    if (!activeSession || !ingestFile) return;
    setIngestBusy(true);
    setIngestMsg(null);
    try {
      const res = await api.scanVideo(activeSession.id, ingestFile);
      setScanCands(res.candidates);
      setIngestMsg(`Detected ${res.candidates.length} candidate delivery burst(s). Click any card to populate start/end frames.`);
    } catch (e: any) {
      setIngestMsg(`Scan failed: ${e.message || e}`);
    } finally {
      setIngestBusy(false);
    }
  };

  const handleUploadDelivery = async () => {
    if (!activeSession || !ingestFile) return;
    setIngestBusy(true);
    setIngestMsg(null);
    try {
      const fd = new FormData();
      fd.append("video", ingestFile);
      fd.append("label", ingestLabel || "Uploaded Delivery");
      fd.append("start_frame", ingestStart || "0");
      if (ingestEnd.trim()) fd.append("end_frame", ingestEnd.trim());
      if (ingestArm) fd.append("bowling_arm", ingestArm);
      const res = await api.uploadDelivery(activeSession.id, fd);
      setShowIngest(false);
      await loadSession(activeSession.id, res.delivery_id);
    } catch (e: any) {
      setIngestMsg(`Upload failed: ${e.message || e}`);
    } finally {
      setIngestBusy(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0b0f17] text-slate-100 flex flex-col">
      {/* Studio Top Bar */}
      <header className="border-b border-[#263248] bg-[#101624] px-4 py-2.5 flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-sky-400 to-emerald-400 flex items-center justify-center text-slate-950 font-black text-sm">
            SP
          </div>
          <div>
            <div className="font-bold text-sm tracking-tight text-white">
              SportsPostureAnalysis <span className="text-sky-400 font-mono text-xs">Studio v2.0</span>
            </div>
            <div className="text-[11px] text-slate-400">
              Pitch-Calibrated, Uncertainty-Gated 3D Cricket Fast-Bowling Biomechanics
            </div>
          </div>
        </div>

        {/* Session & Delivery Selectors */}
        <div className="flex flex-wrap items-center gap-2 text-xs">
          {sessions.length > 0 && (
            <select
              value={activeSession?.id ?? ""}
              onChange={(e) => loadSession(e.target.value)}
              className="bg-[#161e2e] border border-[#263248] rounded-lg px-2.5 py-1.5 text-slate-200 max-w-[230px]"
            >
              {sessions.map((s) => (
                <option key={s.id} value={s.id}>{s.title}</option>
              ))}
            </select>
          )}
          {activeSession?.deliveries && (
            <select
              value={activeDeliveryId ?? ""}
              onChange={(e) => handleSelectDelivery(e.target.value)}
              className="bg-[#161e2e] border border-[#263248] rounded-lg px-2.5 py-1.5 text-sky-200 font-medium max-w-[290px]"
            >
              {activeSession.deliveries.map((d) => (
                <option key={d.id} value={d.id}>{d.label}</option>
              ))}
            </select>
          )}
          <button
            onClick={() => setShowIngest((v) => !v)}
            disabled={!activeSession}
            className="px-2.5 py-1.5 rounded-lg bg-sky-500/20 hover:bg-sky-500/30 border border-sky-500/40 text-sky-300 font-semibold flex items-center gap-1.5"
          >
            <Upload className="w-3.5 h-3.5" />
            Upload / Net Scan
          </button>
          <button
            onClick={handleExportReel}
            disabled={reelBusy || !activeDeliveryId}
            className="px-3 py-1.5 rounded-lg bg-emerald-500/20 hover:bg-emerald-500/30 border border-emerald-500/40 text-emerald-300 font-semibold flex items-center gap-1.5"
          >
            <Film className="w-3.5 h-3.5" />
            {reelBusy ? "Rendering 9:16..." : "Export 9:16 Reel"}
          </button>
          {reelUrl && (
            <a
              href={reelUrl}
              download
              className="px-2.5 py-1.5 rounded-lg bg-emerald-500 text-slate-950 font-bold text-xs"
            >
              Download MP4
            </a>
          )}
        </div>

        {/* Live System Status & Permanent ICC Guardrail Pill */}
        <div className="flex items-center gap-2 text-[11px] font-mono">
          <span className="px-2 py-1 rounded bg-[#161e2e] border border-[#263248] text-slate-300 flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-sky-400" />
            {health?.gpu_worker?.reachable ? "4090 Warm" : "Local CPU (<35ms)"}
          </span>
          <span className="px-2 py-1 rounded bg-rose-950/70 border border-rose-500/50 text-rose-200 font-bold flex items-center gap-1">
            <ShieldAlert className="w-3.5 h-3.5 text-rose-400" />
            ICC 15° Elbow: REFUSED
          </span>
        </div>
      </header>

      {/* Collapsible Upload & Net-Session Auto-Clipper Drawer */}
      {showIngest && activeSession && (
        <div className="bg-[#111827] border-b border-[#263248] px-4 py-3 text-xs space-y-2.5">
          <div className="flex items-center justify-between">
            <span className="font-semibold text-sky-300 flex items-center gap-1.5">
              <Scissors className="w-3.5 h-3.5" /> Video Upload & Multi-Ball Net-Session Auto-Clipper (M2A / M2B)
            </span>
            <button onClick={() => setShowIngest(false)} className="text-slate-400 hover:text-white">
              <X className="w-4 h-4" />
            </button>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <input
              type="file"
              accept="video/*"
              onChange={(e) => {
                setIngestFile(e.target.files?.[0] ?? null);
                setScanCands([]);
                setIngestMsg(null);
              }}
              className="text-[11px] text-slate-300 file:mr-2 file:px-2.5 file:py-1 file:rounded file:border-0 file:bg-[#1e293b] file:text-sky-300"
            />
            <input
              value={ingestLabel}
              onChange={(e) => setIngestLabel(e.target.value)}
              placeholder="Delivery label"
              className="bg-[#0b0f17] border border-[#263248] rounded px-2 py-1 text-slate-200 w-44"
            />
            <input
              value={ingestStart}
              onChange={(e) => setIngestStart(e.target.value)}
              placeholder="Start frame"
              className="bg-[#0b0f17] border border-[#263248] rounded px-2 py-1 text-slate-200 w-24 font-mono"
            />
            <input
              value={ingestEnd}
              onChange={(e) => setIngestEnd(e.target.value)}
              placeholder="End frame"
              className="bg-[#0b0f17] border border-[#263248] rounded px-2 py-1 text-slate-200 w-24 font-mono"
            />
            <select
              value={ingestArm}
              onChange={(e) => setIngestArm(e.target.value)}
              className="bg-[#0b0f17] border border-[#263248] rounded px-2 py-1 text-slate-200"
            >
              <option value="">Auto Arm</option>
              <option value="right">Right Arm</option>
              <option value="left">Left Arm</option>
            </select>
            <button
              onClick={handleScanNetSession}
              disabled={ingestBusy || !ingestFile}
              className="px-2.5 py-1 rounded bg-amber-500/20 hover:bg-amber-500/30 border border-amber-500/40 text-amber-300 font-semibold disabled:opacity-40"
            >
              {ingestBusy ? "Scanning..." : "Auto-Scan Net Clips"}
            </button>
            <button
              onClick={handleUploadDelivery}
              disabled={ingestBusy || !ingestFile}
              className="px-3 py-1 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold disabled:opacity-40"
            >
              {ingestBusy ? "Analyzing..." : "Upload & Run 3D GPU"}
            </button>
          </div>
          {ingestMsg && <div className="text-[11px] text-slate-300 font-mono">{ingestMsg}</div>}
          {scanCands.length > 0 && (
            <div className="flex flex-wrap gap-2">
              {scanCands.map((c) => (
                <button
                  key={c.candidate_id}
                  onClick={() => {
                    setIngestStart(String(c.start_frame));
                    setIngestEnd(String(c.end_frame));
                    setIngestLabel(`Net Delivery #${c.candidate_id}`);
                  }}
                  className="p-1.5 rounded bg-[#161e2e] border border-[#263248] hover:border-sky-400 text-left flex items-center gap-2"
                >
                  {c.thumbnail_b64 && (
                    <img src={`data:image/jpeg;base64,${c.thumbnail_b64}`} alt="" className="w-14 h-9 object-cover rounded" />
                  )}
                  <div className="font-mono text-[10px]">
                    <div className="text-sky-300 font-bold">Ball #{c.candidate_id}</div>
                    <div className="text-slate-400">f {c.start_frame}–{c.end_frame}</div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Capture & Timebase Audit Ribbon */}
      {detail && (
        <div className="px-4 py-1.5 bg-[#0e1420] border-b border-[#1e293b] flex flex-wrap items-center gap-3 text-[11px] font-mono">
          <span className={detail.fact_sheet.audit.timebase_unreliable ? "text-amber-300 font-bold flex items-center gap-1" : "text-emerald-300 flex items-center gap-1"}>
            {detail.fact_sheet.audit.timebase_unreliable ? <AlertTriangle className="w-3.5 h-3.5" /> : <CheckCircle2 className="w-3.5 h-3.5" />}
            {detail.fact_sheet.audit.fps.toFixed(1)}{" "}
            {detail.fact_sheet.audit.timebase_unreliable ? "nominal fps [UNVERIFIED TIMEBASE]" : "unique fps"} (
            {detail.fact_sheet.audit.timebase_status})
          </span>
          <span className="text-slate-400">
            View: <strong className="text-slate-200">{detail.fact_sheet.calibration.view_class}</strong> ({detail.fact_sheet.calibration.azimuth_deg.toFixed(1)}°)
          </span>
          <span className="text-slate-400">
            Cal: <strong className="text-sky-300">{detail.fact_sheet.calibration.method}</strong>
          </span>
          <span className="text-slate-400">
            ROI Blur: <strong className="text-slate-200">{detail.fact_sheet.audit.blur_grade}</strong> ({detail.fact_sheet.audit.median_laplacian_var.toFixed(0)})
          </span>
          <span className="text-slate-500 ml-auto hidden xl:inline">
            Non-diagnostic mechanical stress screening only • Not a clinical diagnosis
          </span>
        </div>
      )}

      {/* Main Studio Grid */}
      <main className="flex-1 p-4 max-w-[1760px] w-full mx-auto">
        {errorMsg && (
          <div className="p-3 rounded-lg bg-rose-950/60 border border-rose-500/40 text-rose-200 text-xs mb-3">
            API Connection Error: {errorMsg}
          </div>
        )}

        {detail && activeSession ? (
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-4 items-start">
            {/* Left + Center (8 cols): Video Scrubber + 3D Pitch Deck + Synchronized uPlot Waveforms */}
            <div className="xl:col-span-8 space-y-4">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <VideoCanvasPlayer
                  videoUrl={detail.delivery.video_proxy_url}
                  telemetry={detail.telemetry}
                  factSheet={detail.fact_sheet}
                  onNudgeEvent={(phase, frame) => handleRecompute({ event_overrides: { [phase]: frame } })}
                  onLandmarkClick={handleLandmarkClick}
                  onFpsChange={(fps) => handleRecompute({ capture_fps_override: fps })}
                />
                <PitchViewport3D
                  telemetry={detail.telemetry}
                  ghostTelemetry={ghostTelemetry}
                />
              </div>
              <WaveformPanel telemetry={detail.telemetry} factSheet={detail.fact_sheet} />
            </div>

            {/* Right Inspector Panel (4 cols): Metrics, 4-Click Calibration, Multi-Ball, Dual 3D, Coach Copilot */}
            <div className="xl:col-span-4">
              <StudioInspector
                detail={detail}
                session={activeSession}
                recomputeMs={recomputeMs}
                onApplyCalibration={handleApplyCalibration}
                onTogglePanTracking={(enabled) =>
                  handleRecompute({ enable_pan_tracking: enabled, pan_anchor_frame: useStudioStore.getState().currentFrame })
                }
                onDualFusedCreated={(newId) => loadSession(activeSession.id, newId)}
              />
            </div>
          </div>
        ) : (
          <div className="text-center py-24 text-slate-400 text-sm">Loading Coaching Studio session...</div>
        )}
      </main>
    </div>
  );
}
export default App;
