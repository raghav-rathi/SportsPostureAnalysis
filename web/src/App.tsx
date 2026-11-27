import React, { useEffect, useState } from "react";
import { Trash2, Upload, X } from "lucide-react";
import { api } from "./api";
import { useStudioStore } from "./store";
import type {
  BowlerSummary,
  CoachNote,
  DeliveryDetailResponse,
  HealthResponse,
  SessionSummary,
  TelestratorShape,
  TelemetryV2,
} from "./types";
import { TopNav } from "./components/TopNav";
import { UploadWizardModal } from "./components/UploadWizardModal";
import { VideoCanvasPlayer } from "./components/VideoCanvasPlayer";
import { PitchViewport3D } from "./components/PitchViewport3D";
import { WaveformPanel } from "./components/WaveformPanel";
import { CoachSummaryPanel } from "./components/CoachSummaryPanel";
import { LabTelemetryPanel } from "./components/LabTelemetryPanel";
import { CompareSpellView } from "./components/CompareSpellView";
import { FilmingGuideView } from "./components/FilmingGuideView";
import { CoachReportView } from "./components/CoachReportView";

export function App() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [bowlers, setBowlers] = useState<BowlerSummary[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [activeSession, setActiveSession] = useState<SessionSummary | null>(null);
  const [activeDeliveryId, setActiveDeliveryId] = useState<string | null>(null);
  const [detail, setDetail] = useState<DeliveryDetailResponse | null>(null);
  const [ghostTelemetry, setGhostTelemetry] = useState<TelemetryV2 | null>(null);
  const [phaseThumbs, setPhaseThumbs] = useState<Record<string, string>>({});
  const [recomputeMs, setRecomputeMs] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [statusBanner, setStatusBanner] = useState<string | null>(null);
  const [showUploadWizard, setShowUploadWizard] = useState(false);
  const [dismissDemoBanner, setDismissDemoBanner] = useState(false);

  const topView = useStudioStore((s) => s.topView);
  const analysisMode = useStudioStore((s) => s.analysisMode);
  const ghostDeliveryId = useStudioStore((s) => s.ghostDeliveryId);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);

  const setTopView = useStudioStore((s) => s.setTopView);
  const setAnalysisMode = useStudioStore((s) => s.setAnalysisMode);
  const setDraftLandmark = useStudioStore((s) => s.setDraftLandmark);
  const clearDraftLandmarks = useStudioStore((s) => s.clearDraftLandmarks);
  const setPreviewCalibration = useStudioStore((s) => s.setPreviewCalibration);
  const setCalibrationMode = useStudioStore((s) => s.setCalibrationMode);
  const setRetrackMode = useStudioStore((s) => s.setRetrackMode);
  const setCurrentFrame = useStudioStore((s) => s.setCurrentFrame);
  const setActiveTool = useStudioStore((s) => s.setActiveTool);
  const setGhostDeliveryId = useStudioStore((s) => s.setGhostDeliveryId);

  const loadDelivery = async (did: string) => {
    clearDraftLandmarks();
    setCalibrationMode(false);
    setRetrackMode(false);
    setActiveTool("cursor");
    setActiveDeliveryId(did);
    setPhaseThumbs({});
    const d = await api.getDelivery(did);
    setDetail(d);
    setCurrentFrame(d.fact_sheet.events.is_complete_delivery ? d.fact_sheet.events.br.frame_idx : 0);
  };

  const loadSession = async (sid: string, selectDeliveryId?: string) => {
    const r = await api.getSession(sid);
    setActiveSession(r.session);
    const dels = r.session.deliveries ?? [];
    const target = selectDeliveryId ?? dels[0]?.id ?? null;
    if (target) {
      await loadDelivery(target);
    } else {
      setActiveDeliveryId(null);
      setDetail(null);
    }
  };

  const refreshMetadata = async () => {
    const [h, b, s] = await Promise.all([api.health(), api.listBowlers(), api.listSessions()]);
    setHealth(h);
    setBowlers(b.bowlers);
    setSessions(s.sessions);
    return s.sessions;
  };

  useEffect(() => {
    (async () => {
      try {
        const sess = await refreshMetadata();
        if (sess.length > 0) {
          await loadSession(sess[0].id);
        }
      } catch (e: any) {
        setErrorMsg(String(e?.message || e));
      }
    })();
  }, []);

  useEffect(() => {
    if (!ghostDeliveryId || !detail) {
      setGhostTelemetry(null);
      return;
    }
    api
      .getDelivery(ghostDeliveryId)
      .then((g) => setGhostTelemetry(g.telemetry))
      .catch(() => setGhostTelemetry(null));
  }, [ghostDeliveryId, detail?.delivery.id]);

  const handleCreateSession = async (payload: {
    bowler_id?: string;
    bowler_name?: string;
    bowling_arm?: string;
    age_band?: string;
    title: string;
    notes?: string;
  }) => {
    const res = await api.createSession(payload);
    await refreshMetadata();
    await loadSession(res.session.id);
    setTopView("analysis");
  };

  const handleRecompute = async (payload: Record<string, any>) => {
    if (!activeDeliveryId) return;
    setErrorMsg(null);
    try {
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
    } catch (err: any) {
      setErrorMsg(err?.message || "Recompute failed.");
    }
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

  const handleRetrackBowler = async (anchorNorm: [number, number]) => {
    if (!activeDeliveryId) return;
    setStatusBanner("▲ Re-tracking 3D pose locked to selected bowler...");
    setErrorMsg(null);
    try {
      const r = await api.retrackBowler(activeDeliveryId, anchorNorm);
      setRecomputeMs(r.recompute_ms);
      setDetail((prev) =>
        prev
          ? {
              ...prev,
              fact_sheet: r.fact_sheet,
              telemetry: r.telemetry,
              report_markdown: r.report_markdown,
              delivery: { ...prev.delivery, isb_summary: r.isb_summary, pan_summary: r.pan_summary },
            }
          : prev
      );
      setStatusBanner("✓ Bowler tracking updated.");
      setTimeout(() => setStatusBanner(null), 2500);
    } catch (err: any) {
      setStatusBanner(null);
      setErrorMsg(err?.message || "Bowler re-tracking failed.");
    }
  };

  const handleSaveAnnotations = async (shapes: TelestratorShape[]) => {
    if (!activeDeliveryId) return;
    setDetail((prev) => (prev ? { ...prev, annotations: shapes } : prev));
    try {
      await api.saveAnnotations(activeDeliveryId, shapes);
    } catch (err: any) {
      setErrorMsg(err?.message || "Failed to save telestrator annotations.");
    }
  };

  const handleNotesChanged = (notes: CoachNote[]) => {
    setDetail((prev) => (prev ? { ...prev, notes } : prev));
  };

  const handleDeleteDelivery = async (did: string) => {
    if (!activeSession) return;
    if (ghostDeliveryId === did) setGhostDeliveryId(null);
    await api.deleteDelivery(did);
    await refreshMetadata();
    await loadSession(activeSession.id);
  };

  const isSampleSession = activeSession?.id === "ses_demo";
  const deliveries = activeSession?.deliveries ?? [];

  return (
    <div className="min-h-screen bg-[#fafafa] text-[#171717] flex flex-col">
      <TopNav
        health={health}
        bowlers={bowlers}
        sessions={sessions}
        activeSession={activeSession}
        recomputeMs={recomputeMs}
        onSelectSession={(sid) => loadSession(sid)}
        onCreateSession={handleCreateSession}
        onOpenUpload={() => setShowUploadWizard(true)}
      />

      {/* Sample Session Onboarding Strip (Dismissible) */}
      {isSampleSession && !dismissDemoBanner && (
        <div className="no-print bg-[#ffffff] border-b border-[#ebebeb]">
          <div className="max-w-[1440px] mx-auto px-4 md:px-6 py-2 flex flex-wrap items-center justify-between gap-2 text-[12px]">
            <div className="flex flex-wrap items-center gap-2 text-[#4d4d4d]">
              <span className="font-mono text-[10px] uppercase tracking-[0.06em] px-1.5 py-0.5 rounded-[4px] bg-[#171717] text-[#ffffff]">
                SAMPLE SESSION
              </span>
              <span>
                Exploring pre-loaded 240 fps calibrated deliveries &amp; real Dale Steyn / Glenn McGrath clips. Click{" "}
                <button
                  onClick={() => setShowUploadWizard(true)}
                  className="text-[#297a3a] font-medium hover:underline"
                >
                  Upload Video
                </button>{" "}
                anytime to analyze your own bowling, or read the{" "}
                <button
                  onClick={() => setTopView("guide")}
                  className="text-[#171717] font-medium underline underline-offset-2"
                >
                  30-sec Filming Guide
                </button>
                .
              </span>
            </div>
            <button
              onClick={() => setDismissDemoBanner(true)}
              className="text-[#8f8f8f] hover:text-[#171717] p-0.5"
              aria-label="Dismiss sample banner"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      )}

      {/* Horizontal Delivery Switcher Strip (in Analysis & Coach Report) */}
      {(topView === "analysis" || topView === "report") && activeSession && (
        <div className="no-print border-b border-[#ebebeb] bg-[#fafafa]">
          <div className="max-w-[1440px] mx-auto px-4 md:px-6 py-2 flex flex-col lg:flex-row lg:items-center lg:justify-between gap-2.5">
            <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5 flex-1 min-w-0">
              <span className="v-eyebrow mr-1 shrink-0">DELIVERIES</span>
              {deliveries.map((d) => {
                const active = d.id === activeDeliveryId;
                const badge =
                  d.pipeline_mode === "real_clip_fixture"
                    ? "REAL"
                    : d.pipeline_mode === "dual_3d_ray_fusion"
                    ? "3D FUSED"
                    : d.pipeline_mode === "single_view_studio"
                    ? "DEMO"
                    : "UPLOAD";
                return (
                  <div
                    key={d.id}
                    className={`group flex items-center rounded-[6px] border text-[12px] shrink-0 transition ${
                      active
                        ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                        : "bg-[#ffffff] text-[#4d4d4d] border-[#ebebeb] hover:border-[#171717] hover:text-[#171717]"
                    }`}
                  >
                    <button
                      onClick={() => loadDelivery(d.id)}
                      className="px-2.5 h-7 flex items-center gap-1.5 whitespace-nowrap"
                    >
                      <span
                        className={`font-mono text-[9px] uppercase px-1 py-0.5 rounded-[3px] ${
                          active ? "bg-[#ffffff]/15 text-[#ffffff]" : "bg-[#f4f4f5] text-[#666666]"
                        }`}
                      >
                        {badge}
                      </span>
                      <span className="max-w-[200px] truncate">{d.label}</span>
                    </button>
                    {!d.id.startsWith("del_demo_") && !d.id.startsWith("del_real_") && (
                      <button
                        onClick={() => handleDeleteDelivery(d.id)}
                        className={`pr-2 pl-0.5 h-7 flex items-center opacity-60 hover:opacity-100 ${
                          active ? "text-[#ffffff]" : "text-[#666666]"
                        }`}
                        title="Delete delivery"
                      >
                        <Trash2 className="w-3 h-3" />
                      </button>
                    )}
                  </div>
                );
              })}
              <button
                onClick={() => setShowUploadWizard(true)}
                className="v-btn-ghost h-7 px-2.5 text-[11px] font-mono flex items-center gap-1 shrink-0"
              >
                <Upload className="w-3 h-3" />
                Add Clip
              </button>
            </div>

            {/* Coach Summary vs Lab Telemetry Segmented Mode Switch (Analysis View Only) */}
            {topView === "analysis" && (
              <div className="flex items-center rounded-[6px] bg-[#f4f4f5] p-0.5 border border-[#ebebeb] shrink-0">
                <button
                  onClick={() => setAnalysisMode("coach")}
                  className={`px-3 h-7 rounded-[5px] text-[12px] font-medium transition ${
                    analysisMode === "coach" ? "bg-[#ffffff] text-[#171717]" : "text-[#666666] hover:text-[#171717]"
                  }`}
                >
                  Coach Summary
                </button>
                <button
                  onClick={() => setAnalysisMode("lab")}
                  className={`px-3 h-7 rounded-[5px] text-[12px] font-medium transition ${
                    analysisMode === "lab" ? "bg-[#ffffff] text-[#171717]" : "text-[#666666] hover:text-[#171717]"
                  }`}
                >
                  Lab Telemetry
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-[1440px] w-full mx-auto px-4 md:px-6 py-5 space-y-4">
        {errorMsg && (
          <div className="no-print p-3 rounded-[6px] bg-[#ffffff] border border-[#171717] flex items-center justify-between gap-2 font-mono text-[12px] text-[#171717]">
            <span>✕ {errorMsg}</span>
            <button onClick={() => setErrorMsg(null)} className="text-[#666666] hover:text-[#171717]">
              <X className="w-4 h-4" />
            </button>
          </div>
        )}

        {statusBanner && (
          <div className="no-print p-2.5 rounded-[6px] bg-[#171717] text-[#ffffff] font-mono text-[12px]">
            {statusBanner}
          </div>
        )}

        {/* VIEW 1: ANALYSIS (Coach Summary / Lab Telemetry) */}
        {topView === "analysis" && (
          detail && activeSession ? (
            analysisMode === "coach" ? (
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
                <div className="xl:col-span-7">
                  <VideoCanvasPlayer
                    videoUrl={detail.delivery.video_proxy_url}
                    telemetry={detail.telemetry}
                    factSheet={detail.fact_sheet}
                    annotations={detail.annotations || []}
                    onSaveAnnotations={handleSaveAnnotations}
                    onNudgeEvent={(phase, frame) => handleRecompute({ event_overrides: { [phase]: frame } })}
                    onLandmarkClick={handleLandmarkClick}
                    onRetrackBowler={handleRetrackBowler}
                    onFpsChange={(fps) => handleRecompute({ capture_fps_override: fps })}
                    onPhaseThumbsCaptured={setPhaseThumbs}
                  />
                </div>
                <div className="xl:col-span-5">
                  <CoachSummaryPanel
                    detail={detail}
                    onNotesChanged={handleNotesChanged}
                    onOpenFilmingGuide={() => setTopView("guide")}
                  />
                </div>
              </div>
            ) : (
              <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
                <div className="xl:col-span-8 space-y-4">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 items-start">
                    <VideoCanvasPlayer
                      videoUrl={detail.delivery.video_proxy_url}
                      telemetry={detail.telemetry}
                      factSheet={detail.fact_sheet}
                      annotations={detail.annotations || []}
                      onSaveAnnotations={handleSaveAnnotations}
                      onNudgeEvent={(phase, frame) => handleRecompute({ event_overrides: { [phase]: frame } })}
                      onLandmarkClick={handleLandmarkClick}
                      onRetrackBowler={handleRetrackBowler}
                      onFpsChange={(fps) => handleRecompute({ capture_fps_override: fps })}
                      onPhaseThumbsCaptured={setPhaseThumbs}
                      compact
                    />
                    <PitchViewport3D
                      telemetry={detail.telemetry}
                      ghostTelemetry={ghostTelemetry}
                      bowlingArm={detail.fact_sheet.bowling_arm}
                    />
                  </div>
                  <WaveformPanel telemetry={detail.telemetry} factSheet={detail.fact_sheet} />
                </div>
                <div className="xl:col-span-4">
                  <LabTelemetryPanel
                    detail={detail}
                    session={activeSession}
                    recomputeMs={recomputeMs}
                    onApplyCalibration={handleApplyCalibration}
                    onTogglePanTracking={(enabled) =>
                      handleRecompute({ enable_pan_tracking: enabled, pan_anchor_frame: useStudioStore.getState().currentFrame })
                    }
                    onOpenCompare={() => setTopView("compare")}
                  />
                </div>
              </div>
            )
          ) : (
            <div className="v-card p-12 text-center space-y-3 max-w-lg mx-auto my-8">
              <div className="v-eyebrow">▲ EMPTY_SESSION</div>
              <h2 className="text-[22px] font-[450] tracking-[-0.04em] text-[#171717]">
                No Deliveries in This Session Yet
              </h2>
              <p className="text-[13px] text-[#4d4d4d]">
                Upload a single delivery or a multi-ball net video to extract pitch-calibrated 3D biomechanics.
              </p>
              <div className="flex justify-center gap-2 pt-2">
                <button onClick={() => setShowUploadWizard(true)} className="v-btn-primary h-9 px-4 text-[13px]">
                  ▲ Upload Video
                </button>
                <button onClick={() => setTopView("guide")} className="v-btn-ghost h-9 px-4 text-[13px]">
                  Read Filming Guide
                </button>
              </div>
            </div>
          )
        )}

        {/* VIEW 2: COMPARE & SPELL */}
        {topView === "compare" && activeSession && (
          <CompareSpellView
            session={activeSession}
            onDualFusedCreated={async (newId) => {
              await refreshMetadata();
              await loadSession(activeSession.id, newId);
              setTopView("analysis");
            }}
          />
        )}

        {/* VIEW 3: FILMING GUIDE */}
        {topView === "guide" && <FilmingGuideView onOpenUpload={() => setShowUploadWizard(true)} />}

        {/* VIEW 4: COACH REPORT */}
        {topView === "report" && detail && activeSession && (
          <CoachReportView detail={detail} session={activeSession} phaseThumbs={phaseThumbs} />
        )}
      </main>

      {/* Guided 3-Step Upload & Net-Session Auto-Clipper Modal */}
      {showUploadWizard && activeSession && (
        <UploadWizardModal
          session={activeSession}
          health={health}
          onClose={() => setShowUploadWizard(false)}
          onImported={async (newId) => {
            await refreshMetadata();
            await loadSession(activeSession.id, newId);
            setTopView("analysis");
          }}
        />
      )}
    </div>
  );
}

export default App;
