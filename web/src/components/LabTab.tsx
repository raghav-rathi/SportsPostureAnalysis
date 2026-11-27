import React, { useEffect, useState } from "react";
import { api } from "../api";
import { useStudioStore } from "../store";
import type { DeliveryDetailResponse, SessionSummary, TelemetryV2 } from "../types";
import { VideoCanvasPlayer } from "./VideoCanvasPlayer";
import { PitchViewport3D } from "./PitchViewport3D";
import { WaveformPanel } from "./WaveformPanel";
import { LabTelemetryPanel } from "./LabTelemetryPanel";
import { CoachSummaryPanel } from "./CoachSummaryPanel";
import { CoachReportView } from "./CoachReportView";
import { CompareSpellView } from "./CompareSpellView";
import { navigate } from "../router";

/**
 * Pro Lab tab — hosts every v3 power tool unchanged (3D pitch, waveforms, telestrator,
 * 4-click calibration, pan tracking, ISB markers, event nudge, bowler re-track, fps override).
 * Only visible when the user opts in via the Pro Lab switch.
 */
export const LabTab: React.FC<{
  detail: DeliveryDetailResponse;
  session: SessionSummary | null;
  onDetailChange: (updater: (prev: DeliveryDetailResponse) => DeliveryDetailResponse) => void;
  onOpenCompare: () => void;
}> = ({ detail, session, onDetailChange, onOpenCompare }) => {
  const [recomputeMs, setRecomputeMs] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [busyMsg, setBusyMsg] = useState<string | null>(null);
  const [ghostTelemetry, setGhostTelemetry] = useState<TelemetryV2 | null>(null);
  const [phaseThumbs, setPhaseThumbs] = useState<Record<string, string>>({});
  const [lower, setLower] = useState<"none" | "coach" | "report" | "spell">("none");

  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const ghostDeliveryId = useStudioStore((s) => s.ghostDeliveryId);
  const setDraftLandmark = useStudioStore((s) => s.setDraftLandmark);
  const setPreviewCalibration = useStudioStore((s) => s.setPreviewCalibration);
  const setCalibrationMode = useStudioStore((s) => s.setCalibrationMode);

  const deliveryId = detail.delivery.id;

  useEffect(() => {
    if (!ghostDeliveryId) {
      setGhostTelemetry(null);
      return;
    }
    api
      .getDelivery(ghostDeliveryId)
      .then((g) => setGhostTelemetry(g.telemetry))
      .catch(() => setGhostTelemetry(null));
  }, [ghostDeliveryId, deliveryId]);

  const applyResult = (r: any, payload?: Record<string, any>) => {
    setRecomputeMs(r.recompute_ms ?? null);
    onDetailChange((prev) => ({
      ...prev,
      fact_sheet: r.fact_sheet ?? prev.fact_sheet,
      telemetry: r.telemetry ?? prev.telemetry,
      report_markdown: r.report_markdown ?? prev.report_markdown,
      story: r.story ?? prev.story,
      delivery: {
        ...prev.delivery,
        overrides: payload ? { ...prev.delivery.overrides, ...payload } : prev.delivery.overrides,
        isb_summary: r.isb_summary ?? prev.delivery.isb_summary,
        pan_summary: r.pan_summary ?? prev.delivery.pan_summary,
      },
    }));
  };

  const handleRecompute = async (payload: Record<string, any>) => {
    setErrorMsg(null);
    try {
      const r = await api.recomputeDelivery(deliveryId, payload);
      applyResult(r, payload);
    } catch (err: any) {
      setErrorMsg(err?.message || "Recompute failed.");
    }
  };

  const handleLandmarkClick = async (key: string, pt: [number, number]) => {
    setDraftLandmark(key, pt);
    const nextMap = { ...draftLandmarks, [key]: pt };
    if (Object.keys(nextMap).length >= 4) {
      try {
        const prev = await api.previewCalibration(deliveryId, nextMap);
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

  const handleRetrack = async (anchorNorm: [number, number]) => {
    setBusyMsg("Re-tracking the selected bowler…");
    setErrorMsg(null);
    try {
      const r = await api.retrackBowler(deliveryId, anchorNorm);
      applyResult(r);
      setBusyMsg("Bowler tracking updated.");
      setTimeout(() => setBusyMsg(null), 2500);
    } catch (err: any) {
      setBusyMsg(null);
      setErrorMsg(err?.message || "Bowler re-tracking failed.");
    }
  };

  const handleSaveAnnotations = async (shapes: any[]) => {
    onDetailChange((prev) => ({ ...prev, annotations: shapes }));
    try {
      await api.saveAnnotations(deliveryId, shapes);
    } catch (err: any) {
      setErrorMsg(err?.message || "Failed to save drawings.");
    }
  };

  return (
    <div className="space-y-4">
      <div className="v-card px-4 py-3 text-[12.5px] text-[#4d4d4d] leading-snug">
        <span className="font-[500] text-[#171717]">Pro Lab</span> shows the raw lab tools: 3D pitch view, joint-angle graphs,
        drawing tools, pitch calibration and event fine-tuning. Changes here re-run the analysis and update your Story.
      </div>
      {errorMsg && (
        <div className="v-card px-4 py-2.5 text-[13px] text-[#171717] border border-[#171717] flex justify-between gap-2">
          <span>{errorMsg}</span>
          <button onClick={() => setErrorMsg(null)} className="text-[#666666]">Dismiss</button>
        </div>
      )}
      {busyMsg && <div className="rounded-[6px] bg-[#171717] text-[#ffffff] px-4 py-2 text-[13px]">{busyMsg}</div>}

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
              onRetrackBowler={handleRetrack}
              onFpsChange={(fps) => handleRecompute({ capture_fps_override: fps })}
              onPhaseThumbsCaptured={setPhaseThumbs}
              compact
            />
            <PitchViewport3D telemetry={detail.telemetry} ghostTelemetry={ghostTelemetry} bowlingArm={detail.fact_sheet.bowling_arm} />
          </div>
          <WaveformPanel telemetry={detail.telemetry} factSheet={detail.fact_sheet} />
        </div>
        <div className="xl:col-span-4">
          {session ? (
            <LabTelemetryPanel
              detail={detail}
              session={session}
              recomputeMs={recomputeMs}
              onApplyCalibration={handleApplyCalibration}
              onTogglePanTracking={(enabled) =>
                handleRecompute({ enable_pan_tracking: enabled, pan_anchor_frame: useStudioStore.getState().currentFrame })
              }
              onOpenCompare={onOpenCompare}
            />
          ) : (
            <div className="v-card p-4 text-[13px] text-[#666666]">Loading lab panel…</div>
          )}
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          onClick={() => setLower((v) => (v === "coach" ? "none" : "coach"))}
          className={`h-8 px-3 text-[13px] rounded-[6px] border ${lower === "coach" ? "bg-[#171717] text-[#ffffff] border-[#171717]" : "bg-[#ffffff] border-[#ebebeb]"}`}
        >
          All metrics, cohort bands & coach copilot
        </button>
        <button
          onClick={() => setLower((v) => (v === "report" ? "none" : "report"))}
          className={`h-8 px-3 text-[13px] rounded-[6px] border ${lower === "report" ? "bg-[#171717] text-[#ffffff] border-[#171717]" : "bg-[#ffffff] border-[#ebebeb]"}`}
        >
          Full technical report
        </button>
        <button
          onClick={() => setLower((v) => (v === "spell" ? "none" : "spell"))}
          className={`h-8 px-3 text-[13px] rounded-[6px] border ${lower === "spell" ? "bg-[#171717] text-[#ffffff] border-[#171717]" : "bg-[#ffffff] border-[#ebebeb]"}`}
        >
          Session spell & dual-camera 3D fusion
        </button>
      </div>
      {lower === "coach" && (
        <CoachSummaryPanel
          detail={detail}
          onNotesChanged={(notes) => onDetailChange((prev) => ({ ...prev, notes }))}
          onOpenFilmingGuide={() => undefined}
        />
      )}
      {lower === "report" && session && <CoachReportView detail={detail} session={session} phaseThumbs={phaseThumbs} />}
      {lower === "spell" && session && (
        <CompareSpellView
          session={session}
          onDualFusedCreated={async (newId) => navigate({ name: "delivery", id: newId, tab: "story" })}
        />
      )}
    </div>
  );
};
