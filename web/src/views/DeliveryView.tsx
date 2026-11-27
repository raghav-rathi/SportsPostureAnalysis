/**
 * DeliveryView — the flagship Interactive Delivery Workbench (docs/research_app_landscape_v5.md §8.2).
 *
 * Header (back · inline rename · arm/view/fps pills · Compare · Share ▾) → sub-tabs:
 *   Workbench (default): ToolRail · Stage [Split | Video | 3D | Strobe] + layer bar + 3D ghost picker ·
 *                        TimelineDock (one clock) · InspectorPanel
 *   Practice:            drills linked to this delivery
 *   Details:             every canonical + v5 metric with CI and σ breakdown, events, calibration, pan tracker,
 *                        ISB anatomy, video audit, research export (never called "Pro")
 * GuardrailStrip is always visible at the bottom and never truncated.
 */
import React, { Suspense, lazy, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { navigate, type DeliveryTab } from "../router";
import type {
  Archetype,
  CheckpointId,
  CoachNote,
  CoachingStory,
  DeliveryDetailResponse,
  EventTimestamp,
  MetricEstimate,
  SessionSummary,
  TelestratorShape,
  TelestratorTool,
} from "../types";
import { SEG } from "../theme";
import { PlayheadProvider, usePlayhead } from "../workbench/PlayheadProvider";
import { VideoStage, type LayerVisibility } from "../workbench/Stage/VideoStage";
import type { VirtualCameraPreset } from "../workbench/Stage/PitchStage3D";
import { ToolRail } from "../workbench/ToolRail";
import { TimelineDock } from "../workbench/TimelineDock";
import { InspectorPanel, type InspectorFocus } from "../workbench/InspectorPanel";
import { CoachCardView } from "./CoachCardView";
import { GuardrailStrip, ObsBadge, SectionLabel, StatusPill, fmtMetric, fmtNum, isMeasured, unitSuffix } from "../components/ui/PaperPrimitives";

const PitchStage3D = lazy(() => import("../workbench/Stage/PitchStage3D").then((m) => ({ default: m.PitchStage3D })));
const StrobeStage = lazy(() => import("../workbench/Stage/StrobeStage").then((m) => ({ default: m.StrobeStage })));

export interface DeliveryViewProps {
  id: string;
  tab: DeliveryTab;
  /** Optional deep-link frame (`#/d/:id?f=`). */
  frame?: number | null;
  onOpenFilmGuide: () => void;
}

type SubTab = "workbench" | "practice" | "details";
const toSub = (t: DeliveryTab): SubTab => (t === "practice" ? "practice" : t === "details" || t === "lab" ? "details" : "workbench");
const toRouteTab = (s: SubTab): DeliveryTab => (s === "workbench" ? "story" : s);

const Loading: React.FC<{ label: string; className?: string }> = ({ label, className = "" }) => (
  <div className={`flex items-center justify-center gap-2 text-[13px] text-[#666666] ${className}`}>
    <span className="inline-block w-3.5 h-3.5 rounded-full border-2 border-[#d4d4d4] border-t-[#171717] animate-spin" />
    {label}
  </div>
);

export const DeliveryView: React.FC<DeliveryViewProps> = ({ id, tab, frame = null, onOpenFilmGuide }) => {
  const [detail, setDetail] = useState<DeliveryDetailResponse | null>(null);
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    api
      .getDelivery(id)
      .then(async (d) => {
        if (cancelled) return;
        setDetail(d);
        try {
          const s = await api.getSession(d.delivery.session_id);
          if (!cancelled) setSession(s.session);
        } catch {
          /* optional */
        }
      })
      .catch((e) => !cancelled && setError(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  if (error) {
    return (
      <div className="max-w-[560px] mx-auto py-16 text-center">
        <h1 className="text-[20px] font-[500] text-[#171717]">We couldn't open this delivery</h1>
        <p className="mt-2 text-[13.5px] text-[#4d4d4d]">It may have been deleted, or the studio server isn't running. ({error})</p>
        <button onClick={() => navigate({ name: "home" })} className="v-btn-primary h-9 px-4 text-[13px] mt-4">
          Back to home
        </button>
      </div>
    );
  }
  if (!detail) return <Loading label="Opening delivery…" className="py-16" />;

  const story = detail.story ?? null;
  if (tab === "card") return <CoachCardView detail={detail} story={story} session={session} />;

  const sub = toSub(tab);
  const notes = detail.notes ?? [];
  const setNotes = (n: CoachNote[]) => setDetail((p) => (p ? { ...p, notes: n } : p));

  return (
    <div className="space-y-4">
      <WorkbenchHeader
        detail={detail}
        story={story}
        session={session}
        sub={sub}
        onRenamed={(label) => setDetail((p) => (p ? { ...p, delivery: { ...p.delivery, label } } : p))}
      />
      {sub === "workbench" ? (
        <WorkbenchShell detail={detail} story={story} session={session} initialFrame={frame} notes={notes} onNotesChange={setNotes} onOpenFilmGuide={onOpenFilmGuide} />
      ) : sub === "practice" ? (
        <PracticeTab detail={detail} story={story} session={session} onOpenFilmGuide={onOpenFilmGuide} />
      ) : (
        <DetailsTab detail={detail} story={story} />
      )}
      <GuardrailStrip />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Header
// ─────────────────────────────────────────────────────────────────────────────
const WorkbenchHeader: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
  sub: SubTab;
  onRenamed: (label: string) => void;
}> = ({ detail, story, session, sub, onRenamed }) => {
  const id = detail.delivery.id;
  const fs = detail.fact_sheet;
  const builtIn = id.startsWith("del_demo_") || id.startsWith("del_real_");
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(detail.delivery.label);
  const [shareOpen, setShareOpen] = useState(false);
  const [moreOpen, setMoreOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [flash, setFlash] = useState<string | null>(null);
  useEffect(() => setLabel(detail.delivery.label), [detail.delivery.label]);
  useEffect(() => {
    if (!flash) return;
    const t = window.setTimeout(() => setFlash(null), 2600);
    return () => window.clearTimeout(t);
  }, [flash]);

  const save = async () => {
    const next = label.trim();
    setEditing(false);
    if (!next || next === detail.delivery.label) return;
    onRenamed(next);
    try {
      await api.patchDelivery(id, { label: next });
    } catch {
      setFlash("Renamed here; the server didn't save it.");
    }
  };

  const reel = async () => {
    setShareOpen(false);
    setBusy("reel");
    try {
      const r = await api.exportReel(id);
      window.open(r.reel_url, "_blank", "noopener");
    } catch {
      setFlash("Couldn't build the 9:16 reel right now.");
    } finally {
      setBusy(null);
    }
  };
  const copySummary = async () => {
    setShareOpen(false);
    try {
      await navigator.clipboard.writeText(detail.report_markdown || story?.headline || "");
      setFlash("Coach summary copied.");
    } catch {
      setFlash("Clipboard isn't available in this browser.");
    }
  };
  const del = async () => {
    setMoreOpen(false);
    if (!window.confirm("Delete this delivery? This can't be undone.")) return;
    try {
      await api.deleteDelivery(id);
      navigate({ name: "deliveries", bowler: session?.bowler_id ?? null });
    } catch {
      setFlash("Couldn't delete this delivery.");
    }
  };

  const viewLabel = story?.capture.view_label ?? fs.calibration.view_class.replace(/_/g, " ");
  const pills = [
    fs.bowling_arm === "left" ? "Left-arm" : "Right-arm",
    viewLabel,
    `${Math.round(fs.audit.fps)} fps`,
    ...(fs.audit.timebase_unreliable ? ["Timing unverified"] : []),
  ];
  const tabs: Array<[SubTab, string]> = [
    ["workbench", "Workbench"],
    ["practice", `Practice${story?.drills.length ? ` · ${story.drills.length}` : ""}`],
    ["details", "Details"],
  ];
  const menuItem = "w-full text-left px-3 py-2 text-[13px] hover:bg-[#fafafa] flex items-center justify-between gap-3";

  return (
    <header className="space-y-3">
      <button onClick={() => navigate({ name: "deliveries", bowler: session?.bowler_id ?? null })} className="text-[13px] text-[#666666] hover:text-[#171717]">
        ← Deliveries
      </button>
      <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-3">
        <div className="min-w-0">
          {editing ? (
            <div className="flex items-center gap-2">
              <input
                autoFocus
                value={label}
                onChange={(e) => setLabel(e.target.value)}
                onKeyDown={(e) => {
                  e.stopPropagation();
                  if (e.key === "Enter") save();
                  if (e.key === "Escape") setEditing(false);
                }}
                onBlur={save}
                className="v-input h-9 px-2.5 text-[18px] w-full max-w-[440px]"
                aria-label="Delivery name"
              />
            </div>
          ) : (
            <h1 className="text-[22px] md:text-[26px] font-[500] tracking-[-0.03em] text-[#171717] flex items-center gap-2 min-w-0">
              <span className="truncate">{detail.delivery.label}</span>
              {!builtIn && (
                <button onClick={() => setEditing(true)} className="text-[13px] text-[#a8a8a8] hover:text-[#171717] shrink-0" aria-label="Rename delivery" title="Rename">
                  ✎
                </button>
              )}
            </h1>
          )}
          <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
            {session?.bowler_name && <span className="text-[13px] text-[#171717] mr-1">{session.bowler_name}</span>}
            {pills.map((p) => (
              <span key={p} className="v-chip">
                {p}
              </span>
            ))}
            {story && (
              <span className="v-chip" title={story.capture.summary}>
                {story.capture.label} · {story.capture.measured_count}/{story.capture.total_count}
              </span>
            )}
            {story?.is_sample && <span className="v-chip">{story.sample_kind === "real" ? "Sample · real video" : "Sample · demo"}</span>}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {flash && <span className="text-[12px] text-[#4d4d4d]">{flash}</span>}
          <button onClick={() => navigate({ name: "compare", a: id, b: null })} className="v-btn-ghost h-8 px-3 text-[13px]">
            Compare
          </button>
          <div className="relative">
            <button onClick={() => setShareOpen((v) => !v)} className="v-btn-primary h-8 px-3 text-[13px]" aria-expanded={shareOpen} aria-haspopup="menu">
              {busy === "reel" ? "Building reel…" : "Share ▾"}
            </button>
            {shareOpen && (
              <div className="absolute right-0 mt-1 z-30 v-card py-1 w-60" role="menu" onMouseLeave={() => setShareOpen(false)}>
                <button role="menuitem" className={menuItem} onClick={() => navigate({ name: "card", id })}>
                  Breakdown card <span className="v-num text-[11px] text-[#8f8f8f]">PNG</span>
                </button>
                <button role="menuitem" className={menuItem} onClick={reel}>
                  Download 9:16 truthful reel <span className="v-num text-[11px] text-[#8f8f8f]">MP4</span>
                </button>
                <button role="menuitem" className={menuItem} onClick={copySummary}>
                  Copy coach summary <span className="v-num text-[11px] text-[#8f8f8f]">MD</span>
                </button>
                <div className="my-1 border-t border-[#f0f0f0]" />
                <a role="menuitem" className={menuItem} href={api.exportUrl(id, "csv")} target="_blank" rel="noopener noreferrer" onClick={() => setShareOpen(false)}>
                  Export CSV <span className="v-num text-[11px] text-[#8f8f8f]">research</span>
                </a>
                <a role="menuitem" className={menuItem} href={api.exportUrl(id, "json")} target="_blank" rel="noopener noreferrer" onClick={() => setShareOpen(false)}>
                  Export JSON <span className="v-num text-[11px] text-[#8f8f8f]">research</span>
                </a>
              </div>
            )}
          </div>
          {!builtIn && (
            <div className="relative">
              <button onClick={() => setMoreOpen((v) => !v)} className="v-btn-ghost h-8 w-8 text-[14px]" aria-label="More actions">
                ⋯
              </button>
              {moreOpen && (
                <div className="absolute right-0 mt-1 z-30 v-card py-1 w-44" onMouseLeave={() => setMoreOpen(false)}>
                  <button className={menuItem} onClick={() => (setMoreOpen(false), setEditing(true))}>
                    Rename
                  </button>
                  <button className={menuItem} onClick={del}>
                    Delete delivery
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <nav className="flex items-center gap-1 border-b border-[#ebebeb] overflow-x-auto" aria-label="Delivery sections">
        {tabs.map(([k, l]) => (
          <button
            key={k}
            onClick={() => navigate({ name: "delivery", id, tab: toRouteTab(k) })}
            aria-current={sub === k ? "page" : undefined}
            className={`h-9 px-3 text-[14px] whitespace-nowrap -mb-px border-b-2 transition-colors ${
              sub === k ? "border-[#171717] text-[#171717] font-[500]" : "border-transparent text-[#666666] hover:text-[#171717]"
            }`}
          >
            {l}
          </button>
        ))}
      </nav>
    </header>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Workbench
// ─────────────────────────────────────────────────────────────────────────────
const WorkbenchShell: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
  initialFrame: number | null;
  notes: CoachNote[];
  onNotesChange: (n: CoachNote[]) => void;
  onOpenFilmGuide: () => void;
}> = (props) => {
  const { detail } = props;
  const tel = detail.telemetry;
  const ev = detail.fact_sheet.events;
  const numFrames = Math.max(1, tel.num_frames || tel.time_rel_ffc_ms?.length || 1);
  const fps = tel.analysis_fps || detail.fact_sheet.audit.fps || tel.proxy_fps || 30;
  const start =
    props.initialFrame !== null && props.initialFrame !== undefined
      ? Math.min(numFrames - 1, Math.max(0, props.initialFrame))
      : ev.is_complete_delivery
      ? Math.min(numFrames - 1, Math.max(0, ev.br.frame_idx))
      : 0;
  return (
    <PlayheadProvider numFrames={numFrames} fps={fps} timeRelFfcMs={tel.time_rel_ffc_ms ?? []} initialFrame={start}>
      <WorkbenchBody {...props} />
    </PlayheadProvider>
  );
};

type StageMode = "split" | "video" | "3d" | "strobe";
type GhostSel = { kind: "none" } | { kind: "archetype"; id: string } | { kind: "delivery"; id: string };

function useViewportWidth(): number {
  const [w, setW] = useState<number>(() => (typeof window !== "undefined" ? window.innerWidth : 1280));
  useEffect(() => {
    const onResize = () => setW(window.innerWidth);
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
  return w;
}

const WorkbenchBody: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
  notes: CoachNote[];
  onNotesChange: (n: CoachNote[]) => void;
  onOpenFilmGuide: () => void;
}> = ({ detail, story, session, notes, onNotesChange, onOpenFilmGuide }) => {
  const ph = usePlayhead();
  const id = detail.delivery.id;
  const fs = detail.fact_sheet;
  const tel = detail.telemetry;
  const arm = fs.bowling_arm;

  // ── Stage & layers (reactive to viewport resizes 1440 ↔ 375 without reload) ──
  const vw = useViewportWidth();
  const canSplit = vw >= 768;
  const [userMode, setUserMode] = useState<StageMode | null>(null);
  const mode: StageMode = userMode === null ? (vw >= 1280 ? "split" : "video") : !canSplit && userMode === "split" ? "video" : userMode;
  const setMode = (next: StageMode) => setUserMode(next);
  const [preset, setPreset] = useState<VirtualCameraPreset>("match");
  const comAvailable = !!tel.com_2d_px && tel.com_2d_px.length > 0 && !(fs.com_braking && !isMeasured(fs.com_braking.com_ffc_speed_mps) && !isMeasured(fs.com_braking.com_vertical_sink_pct_stature));
  const lowObs = story ? story.capture.grade === "limited" || story.capture.grade === "unusable" : false;
  const [showSigma, setShowSigma] = useState(lowObs);
  const [layers, setLayers] = useState<LayerVisibility>({
    skeleton: true,
    segments: { pelvis: true, thorax: true, arm: true, lead: true },
    trails: { wrist: true, head: false, com: false },
    angleChips: true,
    telestration: true,
  });

  // ── Focus (highlight joints + caption) ──
  const [focus, setFocus] = useState<InspectorFocus | null>(null);
  const focusFrameRef = useRef<number | null>(null);
  useEffect(() => {
    if (focus?.frame !== null && focus?.frame !== undefined) focusFrameRef.current = focus.frame;
  }, [focus]);
  useEffect(() => {
    // Caption clears once the playhead moves away from the focused moment.
    if (focusFrameRef.current !== null && Math.abs(ph.frame - focusFrameRef.current) > 2) {
      focusFrameRef.current = null;
      setFocus((f) => (f ? { ...f, caption: null } : f));
    }
  }, [ph.frame]);

  // ── Telestration ──
  const [tool, setTool] = useState<TelestratorTool>("cursor");
  const [shapes, setShapes] = useState<TelestratorShape[]>(detail.annotations ?? []);
  const [history, setHistory] = useState<TelestratorShape[][]>([]);
  const [pinAll, setPinAll] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const saveTimer = useRef<number | null>(null);
  const pushShapes = useCallback(
    (next: TelestratorShape[]) => {
      setHistory((h) => [...h.slice(-49), shapes]);
      setShapes(next);
      if (saveTimer.current) window.clearTimeout(saveTimer.current);
      setSaveState("saving");
      saveTimer.current = window.setTimeout(() => {
        api
          .saveAnnotations(id, next)
          .then(() => setSaveState("saved"))
          .catch(() => setSaveState("error"));
      }, 700);
    },
    [id, shapes]
  );
  useEffect(() => () => void (saveTimer.current && window.clearTimeout(saveTimer.current)), []);
  const undo = () => {
    setHistory((h) => {
      if (!h.length) return h;
      const prev = h[h.length - 1];
      setShapes(prev);
      api.saveAnnotations(id, prev).catch(() => setSaveState("error"));
      return h.slice(0, -1);
    });
  };
  const clearFrame = () => {
    const f = Math.round(ph.frame);
    pushShapes(shapes.filter((s) => s.pinned_all_frames || s.frame !== f));
  };

  // ── Ghost picker ──
  const [archetypes, setArchetypes] = useState<Archetype[]>([]);
  const [ghostSel, setGhostSel] = useState<GhostSel>({ kind: "none" });
  const [ghostData, setGhostData] = useState<{ joints: number[][][]; t: number[]; label: string; arm?: "right" | "left"; disclaimer?: string; source?: string } | null>(null);
  const [ghostBusy, setGhostBusy] = useState(false);
  const [mirror, setMirror] = useState(false);
  useEffect(() => {
    let off = false;
    api
      .archetypes(arm)
      .then((r) => !off && setArchetypes(r.archetypes ?? []))
      .catch(() => {
        /* archetype library optional */
      });
    return () => {
      off = true;
    };
  }, [arm]);
  useEffect(() => {
    let off = false;
    if (ghostSel.kind === "none") {
      setGhostData(null);
      return;
    }
    if (ghostSel.kind === "archetype") {
      const a = archetypes.find((x) => x.id === ghostSel.id);
      setGhostData(a ? { joints: a.joints3d_world_m, t: a.time_rel_ffc_ms, label: `${a.title} (literature reference)`, arm: a.bowling_arm, disclaimer: a.disclaimer, source: a.literature_source } : null);
      return;
    }
    setGhostBusy(true);
    api
      .getDelivery(ghostSel.id)
      .then((d) => {
        if (off) return;
        setGhostData({ joints: d.telemetry.joints3d_world_m, t: d.telemetry.time_rel_ffc_ms, label: d.delivery.label, arm: d.fact_sheet.bowling_arm });
      })
      .catch(() => !off && setGhostData(null))
      .finally(() => !off && setGhostBusy(false));
    return () => {
      off = true;
    };
  }, [ghostSel, archetypes]);
  useEffect(() => {
    if (ghostData?.arm) setMirror(ghostData.arm !== arm);
  }, [ghostData, arm]);
  const otherBalls = (session?.deliveries ?? []).filter((d) => d.id !== id && d.status !== "failed");

  const eventFrames = {
    bfc: fs.events.bfc.frame_idx,
    ffc: fs.events.ffc.frame_idx,
    uah: fs.events.uah?.frame_idx ?? null,
    br: fs.events.br.frame_idx,
    ft: fs.events.ft_peak?.frame_idx ?? null,
  };
  const videoUrl = detail.delivery.video_proxy_url || api.videoUrl(id);
  const highlight = focus?.joints ?? [];
  const caption = focus?.caption ?? null;

  const video = (
    <VideoStage
      deliveryId={id}
      videoUrl={videoUrl}
      telemetry={tel}
      factSheet={fs}
      bowlingArm={arm}
      activeTool={tool}
      layers={layers}
      shapes={shapes}
      onShapesChange={pushShapes}
      pinNewShapes={pinAll}
      zoom={zoom}
      onZoomChange={setZoom}
      highlightJoints={highlight}
      activeCaption={caption}
    />
  );
  const pitch = (
    <Suspense fallback={<Loading label="Loading 3D pitch…" className="h-full v-stage" />}>
      <PitchStage3D
        telemetry={tel}
        bowlingArm={arm}
        ghostJoints3d={ghostData?.joints}
        ghostTimeRelFfcMs={ghostData?.t}
        ghostLabel={ghostData?.label}
        mirrorGhost={mirror}
        preset={preset}
        onPresetChange={setPreset}
        defaultPreset="match"
        showUncertainty={showSigma}
        showTrails={layers.trails}
        eventFrames={eventFrames}
        heightClass="h-full"
      />
    </Suspense>
  );

  const toggle = (on: boolean, label: string, onClick: () => void, opts: { disabled?: boolean; title?: string; swatch?: string } = {}) => (
    <button
      type="button"
      aria-pressed={on}
      disabled={opts.disabled}
      onClick={onClick}
      title={opts.title}
      className={`h-7 px-2.5 rounded-full text-[12px] inline-flex items-center gap-1.5 whitespace-nowrap transition-shadow disabled:opacity-40 disabled:cursor-not-allowed ${
        on ? "bg-[#ffffff] text-[#171717] shadow-[0_0_0_1px_#171717]" : "bg-[#ffffff] text-[#666666] shadow-[0_0_0_1px_#ebebeb] hover:shadow-[0_0_0_1px_#c9c9c9]"
      }`}
    >
      {opts.swatch && <span className="w-2 h-2 rounded-full" style={{ background: on ? opts.swatch : "#d4d4d4" }} />}
      {label}
    </button>
  );

  const ghostValue = ghostSel.kind === "none" ? "" : `${ghostSel.kind}:${ghostSel.id}`;
  const ghostIsArchetype = ghostSel.kind === "archetype";
  const show3d = mode === "split" || mode === "3d";

  const stageHeightCls =
    mode === "strobe"
      ? ""
      : mode === "video"
      ? "h-auto md:h-[54vh] md:min-h-[380px] md:max-h-[580px]"
      : mode === "split"
      ? "h-[54vh] min-h-[380px] max-h-[580px]"
      : "h-[360px] sm:h-[420px] md:h-[54vh] md:min-h-[380px] md:max-h-[580px]";

  return (
    <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_380px] gap-4 items-start">
      <div className="min-w-0 space-y-3">
        <div className="min-w-0 space-y-2">
          {/* Stage mode switcher */}
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="v-seg" role="group" aria-label="Stage mode">
              {(
                [
                  ["split", "Split video + 3D"],
                  ["video", "Video"],
                  ["3d", "3D pitch"],
                  ["strobe", "Strobe"],
                ] as Array<[StageMode, string]>
              ).map(([k, l]) => (
                <button key={k} type="button" aria-pressed={mode === k} onClick={() => setMode(k)} className={k === "split" ? "hidden md:inline-block" : undefined}>
                  {l}
                </button>
              ))}
            </div>
            <span className="text-[11.5px] text-[#8f8f8f]">
              {saveState === "saving" ? "Saving drawings…" : saveState === "saved" ? "Drawings saved" : saveState === "error" ? "Drawings not saved" : "Space play · , . step · J K L shuttle"}
            </span>
          </div>

          {/* Compact horizontal tool strip directly above video/split stage (zero dead vertical whitespace) */}
          {(mode === "video" || mode === "split") && (
            <div className="min-w-0">
              <ToolRail
                activeTool={tool}
                onSelectTool={setTool}
                pinAllFrames={pinAll}
                onTogglePinAll={() => setPinAll((v) => !v)}
                canUndo={history.length > 0}
                onUndo={undo}
                onClearFrame={clearFrame}
                zoom={zoom}
                onZoomChange={setZoom}
                shapeCount={shapes.length}
              />
            </div>
          )}

          {/* Stage */}
          <div className={`${stageHeightCls} min-w-0`}>
            {mode === "split" ? (
              <div className="grid grid-cols-2 gap-2 h-full min-w-0">
                <div className="min-w-0 h-full">{video}</div>
                <div className="min-w-0 h-full">{pitch}</div>
              </div>
            ) : mode === "video" ? (
              <div className="h-full min-w-0">{video}</div>
            ) : mode === "3d" ? (
              <div className="h-full min-w-0">{pitch}</div>
            ) : (
              <Suspense fallback={<Loading label="Loading strobe…" className="h-[320px] v-stage" />}>
                <StrobeStage deliveryId={id} telemetry={tel} factSheet={fs} bowlingArm={arm} stroboscopicUrl={story?.stroboscopic_url ?? null} />
              </Suspense>
            )}
          </div>

          {/* Layer bar + ghost picker */}
          {mode !== "strobe" && (
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="v-label mr-1">Layers</span>
              {toggle(layers.skeleton, "Skeleton", () => setLayers((l) => ({ ...l, skeleton: !l.skeleton })))}
              {layers.skeleton && (
                <div className="inline-flex items-center gap-1 px-1.5 h-7 rounded-full bg-[#fafafa] shadow-[0_0_0_1px_#ebebeb]" role="group" aria-label="Skeleton segments">
                  {(
                    [
                      ["pelvis", "Hips", SEG.pelvis],
                      ["thorax", "Trunk", SEG.thorax],
                      ["arm", "Arm", SEG.arm],
                      ["lead", "Lead", SEG.lead],
                    ] as const
                  ).map(([k, lbl, col]) => {
                    const on = layers.segments[k];
                    return (
                      <button
                        key={k}
                        type="button"
                        aria-pressed={on}
                        onClick={() => setLayers((l) => ({ ...l, segments: { ...l.segments, [k]: !l.segments[k] } }))}
                        title={`Toggle ${lbl.toLowerCase()} segment`}
                        className={`inline-flex items-center gap-1 px-1.5 h-5 rounded-full text-[11px] transition-opacity ${on ? "text-[#171717] bg-[#ffffff] shadow-[0_0_0_1px_#e5e5e5]" : "text-[#a8a8a8] opacity-60"}`}
                      >
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: col }} />
                        {lbl}
                      </button>
                    );
                  })}
                </div>
              )}
              {toggle(layers.trails.wrist, "Wrist trail", () => setLayers((l) => ({ ...l, trails: { ...l.trails, wrist: !l.trails.wrist } })), { swatch: SEG.wrist })}
              {toggle(layers.trails.head, "Head trail", () => setLayers((l) => ({ ...l, trails: { ...l.trails, head: !l.trails.head } })), { swatch: SEG.head })}
              {toggle(layers.trails.com, "Body-centre trail", () => setLayers((l) => ({ ...l, trails: { ...l.trails, com: !l.trails.com } })), {
                swatch: SEG.com,
                disabled: !comAvailable,
                title: comAvailable ? "Centre-of-mass path" : "Centre of mass isn't measurable on this clip",
              })}
              {toggle(layers.angleChips, "Angles", () => setLayers((l) => ({ ...l, angleChips: !l.angleChips })))}
              {toggle(layers.telestration, "Drawings", () => setLayers((l) => ({ ...l, telestration: !l.telestration })), { swatch: SEG.tele })}
              {show3d && toggle(showSigma, "3D uncertainty", () => setShowSigma((v) => !v), { title: "Translucent spheres sized to each joint's position uncertainty" })}
              <span className="mx-1 h-4 w-px bg-[#ebebeb]" />
              <label className="inline-flex items-center gap-1.5 text-[12px] text-[#4d4d4d]">
                <span className="v-label">Ghost</span>
                <select
                  value={ghostValue}
                  onChange={(e) => {
                    const v = e.target.value;
                    if (!v) return setGhostSel({ kind: "none" });
                    const [kind, ...rest] = v.split(":");
                    setGhostSel({ kind: kind as "archetype" | "delivery", id: rest.join(":") });
                    if (mode === "video") setMode(canSplit ? "split" : "3d");
                  }}
                  className="v-input h-7 px-2 text-[12px] max-w-[220px]"
                  aria-label="3D ghost"
                >
                  <option value="">None</option>
                  {archetypes.length > 0 && (
                    <optgroup label="Literature reference models (synthetic)">
                      {archetypes.map((a) => (
                        <option key={a.id} value={`archetype:${a.id}`}>
                          {a.title}
                        </option>
                      ))}
                    </optgroup>
                  )}
                  {otherBalls.length > 0 && (
                    <optgroup label="Your other balls (personal best)">
                      {otherBalls.map((d) => (
                        <option key={d.id} value={`delivery:${d.id}`}>
                          {d.label}
                        </option>
                      ))}
                    </optgroup>
                  )}
                </select>
              </label>
              {ghostData && toggle(mirror, "Mirror ghost ⇄", () => setMirror((v) => !v), { title: "Mirror the ghost left↔right (for opposite-arm comparisons)" })}
              {ghostBusy && <span className="text-[11.5px] text-[#8f8f8f]">loading ghost…</span>}
            </div>
          )}
          {ghostData && mode !== "strobe" && (
            <p className="text-[11.5px] text-[#4d4d4d] leading-snug rounded-[6px] bg-[#ffffff] shadow-[0_0_0_1px_#ebebeb] px-3 py-2">
              <span className="inline-block w-2 h-2 rounded-full mr-1.5 align-middle" style={{ background: SEG.ghost, opacity: 0.6 }} />
              Ghost: <span className="text-[#171717]">{ghostData.label}</span>
              {mirror ? " · mirrored" : ""} · aligned at front-foot plant{mode === "video" ? " · shows in the 3D view" : ""}.
              {ghostIsArchetype && (
                <>
                  {" "}
                  <span className="text-[#171717]">Synthetic archetype — not a real player.</span> {ghostData.disclaimer}
                  {ghostData.source ? ` Source: ${ghostData.source}.` : ""}
                </>
              )}
            </p>
          )}
        </div>

        <TimelineDock telemetry={tel} factSheet={fs} noteFrames={notes.map((n) => n.frame_idx).filter((f): f is number => f !== null)} onOpenFilmGuide={onOpenFilmGuide} />
      </div>

      <InspectorPanel
        detail={detail}
        story={story}
        notes={notes}
        onNotesChange={onNotesChange}
        onFocus={setFocus}
        activeCheckpointId={(focus?.checkpointId as CheckpointId | null | undefined) ?? null}
        onOpenFilmGuide={onOpenFilmGuide}
        className="xl:sticky xl:top-[72px] xl:max-h-[calc(100vh-88px)] xl:overflow-y-auto xl:pr-1"
      />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Practice tab
// ─────────────────────────────────────────────────────────────────────────────
const PracticeTab: React.FC<{ detail: DeliveryDetailResponse; story: CoachingStory | null; session: SessionSummary | null; onOpenFilmGuide: () => void }> = ({
  detail,
  story,
  session,
  onOpenFilmGuide,
}) => {
  const id = detail.delivery.id;
  const drills = story?.drills ?? [];
  const titleOf = (cid: CheckpointId | null) => (cid ? story?.checkpoints.find((c) => c.id === cid)?.title ?? cid : null);
  return (
    <div className="max-w-[820px] space-y-4">
      {drills.length > 0 ? (
        <>
          <p className="text-[14px] text-[#4d4d4d] leading-relaxed">
            {drills.length === 1 ? "One drill" : `${drills.length} drills`} picked from what this clip clearly measured. Do them, then film another ball to see whether it
            changes.
          </p>
          {drills.map((d, i) => {
            const cp = d.for_checkpoint_id ? story?.checkpoints.find((c) => c.id === d.for_checkpoint_id) : null;
            return (
              <article key={d.id} className="v-panel p-4 md:p-5">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <div className="v-label">
                      Drill {i + 1}
                      {titleOf(d.for_checkpoint_id) ? ` · for ${titleOf(d.for_checkpoint_id)}` : ""}
                    </div>
                    <h3 className="mt-1 text-[16px] font-[500] text-[#171717]">{d.title}</h3>
                  </div>
                  {cp && cp.value_text && (
                    <span className="text-right">
                      <span className="v-num text-[14px] text-[#171717]">{cp.value_text}</span>
                      {cp.range_text && <span className="block v-num text-[11.5px] text-[#8f8f8f]">likely {cp.range_text}</span>}
                    </span>
                  )}
                </div>
                <p className="mt-2 text-[13.5px] text-[#4d4d4d] leading-[1.55]">{d.why_plain}</p>
                <p className="mt-2 text-[13.5px] text-[#171717]">
                  <span className="text-[#8f8f8f]">Cue: </span>“{d.cue}”
                </p>
                {d.steps.length > 0 && (
                  <ol className="mt-2.5 space-y-1 list-decimal pl-5">
                    {d.steps.map((s, k) => (
                      <li key={k} className="text-[13px] text-[#171717] leading-snug">
                        {s}
                      </li>
                    ))}
                  </ol>
                )}
                {d.caution && <p className="mt-2.5 text-[12px] text-[#666666] leading-snug">⚠ {d.caution}</p>}
                <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                  {d.source && <span className="text-[11px] text-[#8f8f8f]">{d.source}</span>}
                  {d.frame_idx !== null && (
                    <button onClick={() => navigate({ name: "delivery", id, tab: "story", frame: d.frame_idx })} className="v-btn-ghost h-8 px-3 text-[12.5px]">
                      See the moment in the workbench
                    </button>
                  )}
                </div>
              </article>
            );
          })}
        </>
      ) : (
        <div className="v-panel p-5">
          <h2 className="text-[17px] font-[500] text-[#171717]">Nothing in this clip clearly needs fixing</h2>
          <p className="mt-2 text-[13.5px] text-[#4d4d4d] leading-relaxed">
            Drills are suggested only when a measurement's whole likely range crosses a research line. This clip didn't show one, so no generic drills are invented.
          </p>
          {story?.capture.next_clip_tip && (
            <p className="mt-2 text-[13.5px] text-[#171717]">
              {story.capture.next_clip_tip}{" "}
              <button onClick={onOpenFilmGuide} className="underline underline-offset-2">
                How to film
              </button>
            </p>
          )}
        </div>
      )}
      <div className="flex flex-wrap gap-2">
        <button onClick={() => navigate({ name: "practice", bowler: session?.bowler_id ?? null, delivery: id })} className="v-btn-primary h-9 px-4 text-[13px]">
          Open Practice & Drill Lab →
        </button>
        <button onClick={() => navigate({ name: "upload", bowler: session?.bowler_id ?? null })} className="v-btn-ghost h-9 px-4 text-[13px]">
          Film another ball
        </button>
        <button onClick={() => navigate({ name: "compare", a: id, b: null })} className="v-btn-ghost h-9 px-4 text-[13px]">
          Compare with another ball
        </button>
      </div>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Details tab (science layer)
// ─────────────────────────────────────────────────────────────────────────────
const sig = (v: number | null | undefined, unit: string) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : `${fmtNum(v, unit === "deg" ? "m" : unit)}`);

const MetricsTable: React.FC<{ title: string; metrics: MetricEstimate[]; filter: string }> = ({ title, metrics, filter }) => {
  const [open, setOpen] = useState<string | null>(null);
  const f = filter.trim().toLowerCase();
  const rows = metrics.filter((m) => !f || m.label.toLowerCase().includes(f) || m.key.toLowerCase().includes(f));
  const measured = metrics.filter(isMeasured).length;
  return (
    <section className="v-panel">
      <div className="px-4 pt-3.5 pb-2 flex items-center justify-between gap-2">
        <h3 className="text-[14px] font-[500] text-[#171717]">{title}</h3>
        <span className="v-num text-[11.5px] text-[#8f8f8f]">
          {measured}/{metrics.length} measured
        </span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[920px] text-[12.5px]">
          <thead>
            <tr className="text-left border-y border-[#f0f0f0]">
              {["Measurement", "Value", "95% range", "Visibility", "σ total", "σ view", "σ timing", "σ pose model", "σ calibration", "Interval"].map((h) => (
                <th key={h} className="px-3 py-2 v-label font-normal whitespace-nowrap">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((m) => {
              const t = fmtMetric(m);
              const ok = isMeasured(m);
              const u = m.unit;
              const isOpen = open === m.key;
              return (
                <React.Fragment key={m.key}>
                  <tr className={`border-b border-[#f5f5f5] cursor-pointer hover:bg-[#fafafa] ${ok ? "" : "text-[#8f8f8f]"}`} onClick={() => setOpen(isOpen ? null : m.key)}>
                    <td className="px-3 py-2">
                      <div className="text-[#171717]">{m.label}</div>
                      <div className="v-num text-[10.5px] text-[#a8a8a8]">{m.key}</div>
                    </td>
                    <td className="px-3 py-2 v-num text-[#171717] whitespace-nowrap">{ok ? t.valueText : "—"}</td>
                    <td className="px-3 py-2 v-num whitespace-nowrap">{ok ? t.rangeText ?? "—" : <span className="text-[11.5px]">{m.suppression_reason || "not measurable"}</span>}</td>
                    <td className="px-3 py-2">
                      <ObsBadge grade={m.observability} />
                    </td>
                    <td className="px-3 py-2 v-num">{ok ? sig(m.sigma_total, u) : "—"}</td>
                    <td className="px-3 py-2 v-num">{ok ? sig(m.sigma_projective, u) : "—"}</td>
                    <td className="px-3 py-2 v-num">{ok ? sig(m.sigma_temporal, u) : "—"}</td>
                    <td className="px-3 py-2 v-num">{ok ? sig(m.sigma_epistemic, u) : "—"}</td>
                    <td className="px-3 py-2 v-num">{ok ? sig(m.sigma_calibration, u) : "—"}</td>
                    <td className="px-3 py-2 text-[11px] whitespace-nowrap">{m.interval_type}</td>
                  </tr>
                  {isOpen && (
                    <tr className="bg-[#fafafa] border-b border-[#f0f0f0]">
                      <td colSpan={10} className="px-3 py-2.5 text-[12px] text-[#4d4d4d] leading-snug space-y-1">
                        {m.definition_note && <p>{m.definition_note}</p>}
                        {m.empirical_reference && <p className="text-[#666666]">Reference: {m.empirical_reference}</p>}
                        {m.literature_source && <p className="text-[#8f8f8f]">{m.literature_source}</p>}
                        {m.key === "icc_elbow_legality_verdict" && <p className="text-[#171717]">🔒 Permanently refused on phone video.</p>}
                        <p className="v-num text-[11px] text-[#8f8f8f]">σ unit: {unitSuffix(u).trim() || u || "—"}</p>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              );
            })}
            {rows.length === 0 && (
              <tr>
                <td colSpan={10} className="px-3 py-4 text-[12.5px] text-[#8f8f8f]">
                  No measurements match.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </section>
  );
};

const KV: React.FC<{ rows: Array<[string, React.ReactNode]> }> = ({ rows }) => (
  <dl className="grid grid-cols-[minmax(0,1fr)_auto] gap-x-4 gap-y-1.5 text-[12.5px]">
    {rows.map(([k, v]) => (
      <React.Fragment key={k}>
        <dt className="text-[#666666]">{k}</dt>
        <dd className="v-num text-[#171717] text-right break-all">{v}</dd>
      </React.Fragment>
    ))}
  </dl>
);
const yes = (b: boolean | null | undefined) => (b ? "yes" : "no");
const n2 = (v: number | null | undefined, d = 2) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : v.toFixed(d));

const DetailsTab: React.FC<{ detail: DeliveryDetailResponse; story: CoachingStory | null }> = ({ detail, story }) => {
  const fs = detail.fact_sheet;
  const id = detail.delivery.id;
  const [filter, setFilter] = useState("");
  const canonical = useMemo(() => Object.values(fs.metrics ?? {}), [fs.metrics]);
  const v5 = useMemo(() => Object.values(fs.v5_metrics ?? {}), [fs.v5_metrics]);
  const isb = detail.delivery.isb_summary ?? detail.telemetry.isb_anatomy ?? null;
  const pan = detail.delivery.pan_summary;
  const cal = fs.calibration;
  const au = fs.audit;
  const ev = fs.events;
  const evRows: Array<[string, EventTimestamp | null]> = [
    ["Back foot lands (BFC)", ev.bfc],
    ["Front foot plants (FFC)", ev.ffc],
    ["Arm level (UAH)", ev.uah],
    ["Ball release (BR)", ev.br],
    ["Follow-through peak", ev.ft_peak],
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <p className="text-[13px] text-[#4d4d4d] max-w-[760px] leading-relaxed">
          Every measurement for this delivery with its 95% range and uncertainty breakdown (σ from camera view, event timing, pose model and calibration). Tap a row for its
          definition and source. Unmeasurable items stay listed with the reason.
        </p>
        <div className="flex flex-wrap gap-2">
          <input value={filter} onChange={(e) => setFilter(e.target.value)} onKeyDown={(e) => e.stopPropagation()} placeholder="Filter measurements…" className="v-input h-8 px-2.5 text-[12.5px] w-[200px]" />
          <a href={api.exportUrl(id, "csv")} target="_blank" rel="noopener noreferrer" className="v-btn-ghost h-8 px-3 text-[12.5px] inline-flex items-center">
            Export CSV
          </a>
          <a href={api.exportUrl(id, "json")} target="_blank" rel="noopener noreferrer" className="v-btn-ghost h-8 px-3 text-[12.5px] inline-flex items-center">
            Export JSON
          </a>
        </div>
      </div>
      <p className="text-[11.5px] text-[#8f8f8f]">
        Research export note: pose estimates come from NLF ({fs.pose_backend}); its pretrained weights are licensed CC-BY-NC-4.0 (non-commercial).
      </p>

      <MetricsTable title={`Core measurements (${canonical.length})`} metrics={canonical} filter={filter} />
      {v5.length > 0 ? (
        <MetricsTable title={`Expanded analyses (${v5.length})`} metrics={v5} filter={filter} />
      ) : (
        <div className="v-panel px-4 py-3 text-[12.5px] text-[#8f8f8f]">Expanded analyses (sequence, momentum, head, lead arm, foot plant, tempo) aren't in this delivery's data yet.</div>
      )}

      {fs.cohort_comparisons?.length > 0 && (
        <section className="v-panel p-4">
          <SectionLabel className="mb-2">Published group comparisons (definition-matched)</SectionLabel>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <thead>
                <tr className="text-left border-b border-[#f0f0f0]">
                  {["Measurement", "You", "Group", "Group mean ± SD", "Definition match", "Source"].map((h) => (
                    <th key={h} className="py-1.5 pr-3 v-label font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {fs.cohort_comparisons.map((c) => (
                  <tr key={c.metric_key + c.cohort_name} className="border-b border-[#f5f5f5]">
                    <td className="py-1.5 pr-3">{c.label}</td>
                    <td className="py-1.5 pr-3 v-num">
                      {fmtNum(c.value, c.unit)} ({fmtNum(c.ci_low, c.unit)}–{fmtNum(c.ci_high, c.unit)})
                    </td>
                    <td className="py-1.5 pr-3">{c.cohort_name}</td>
                    <td className="py-1.5 pr-3 v-num">
                      {fmtNum(c.cohort_mean, c.unit)} ± {fmtNum(c.cohort_sd, c.unit)}
                    </td>
                    <td className="py-1.5 pr-3 text-[11.5px]">{c.definition_match}</td>
                    <td className="py-1.5 pr-3 text-[11px] text-[#8f8f8f]">{c.source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <section className="v-panel p-4">
          <SectionLabel className="mb-2" right={<span className="text-[11px] text-[#8f8f8f]">{ev.is_complete_delivery ? "complete delivery" : "incomplete delivery"}</span>}>
            Event timing
          </SectionLabel>
          <div className="overflow-x-auto">
            <table className="w-full text-[12.5px] min-w-[460px]">
              <thead>
                <tr className="text-left border-b border-[#f0f0f0]">
                  {["Event", "Frame", "Source frame", "Time", "± ms", "Confidence", "Method"].map((h) => (
                    <th key={h} className="py-1.5 pr-2 v-label font-normal">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {evRows.map(([l, e]) => (
                  <tr key={l} className="border-b border-[#f5f5f5]">
                    <td className="py-1.5 pr-2">{l}</td>
                    <td className="py-1.5 pr-2 v-num">{e ? n2(e.frame_float, 1) : "—"}</td>
                    <td className="py-1.5 pr-2 v-num">{e?.source_frame_idx ?? "—"}</td>
                    <td className="py-1.5 pr-2 v-num">{e ? `${n2(e.timestamp_s, 3)} s` : "—"}</td>
                    <td className="py-1.5 pr-2 v-num">{e ? n2(e.timing_uncertainty_ms, 1) : "—"}</td>
                    <td className="py-1.5 pr-2 v-num">{e ? n2(e.confidence, 2) : "—"}</td>
                    <td className="py-1.5 pr-2 text-[11px] text-[#666666]">{e?.detection_method ?? "not detected"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-2.5">
            <KV
              rows={[
                ["Stride (BFC → FFC)", `${n2(ev.delivery_stride_duration_ms, 0)} ms`],
                ["FFC → BR", `${n2(ev.ffc_to_br_duration_ms, 0)} ms`],
              ]}
            />
          </div>
          {ev.event_warnings.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {ev.event_warnings.map((w, i) => (
                <li key={i} className="text-[11.5px] text-[#666666]">
                  • {w}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="v-panel p-4">
          <SectionLabel className="mb-2">Camera calibration</SectionLabel>
          <KV
            rows={[
              ["Method", cal.method],
              ["View", cal.view_class],
              ["Azimuth / elevation", `${n2(cal.azimuth_deg, 1)}° / ${n2(cal.elevation_deg, 1)}°`],
              ["Reprojection error", cal.reprojection_error_px !== null ? `${n2(cal.reprojection_error_px, 2)} px` : "—"],
              ["Metric scale calibrated", yes(cal.is_metric_calibrated)],
              ["Lens calibrated", yes(cal.is_intrinsics_calibrated)],
              ["Panning camera", yes(cal.is_panning_camera)],
              ["Heading σ / gravity σ", `${n2(cal.heading_sigma_deg, 2)}° / ${n2(cal.gravity_sigma_deg, 2)}°`],
              ["Gravity source", cal.gravity_source],
              ["Camera position (m)", cal.camera_position_world_m.map((v) => n2(v, 2)).join(", ")],
            ]}
          />
          {cal.calibration_notes.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {cal.calibration_notes.map((w, i) => (
                <li key={i} className="text-[11.5px] text-[#666666]">
                  • {w}
                </li>
              ))}
            </ul>
          )}
          <div className="mt-3 pt-3 border-t border-[#f0f0f0]">
            <SectionLabel className="mb-2">Pan tracker</SectionLabel>
            {pan ? (
              <KV
                rows={[
                  ["Verified", yes(pan.is_verified)],
                  ["Median / p90 residual", `${n2(pan.median_residual_px, 3)} / ${n2(pan.p90_residual_px, 3)} px`],
                  ["Max pan", `${n2(pan.max_pan_deg, 2)}°`],
                  ["Pan σ", `${n2(pan.sigma_pan_deg, 3)}°`],
                  ["Reason", pan.verification_reason || "—"],
                ]}
              />
            ) : (
              <p className="text-[12px] text-[#8f8f8f]">No pan tracking for this clip.</p>
            )}
          </div>
        </section>

        <section className="v-panel p-4">
          <SectionLabel className="mb-2">Video audit</SectionLabel>
          <KV
            rows={[
              ["Resolution", `${au.width} × ${au.height}`],
              ["Capture fps / container fps", `${n2(au.fps, 1)} / ${au.container_fps !== null ? n2(au.container_fps, 1) : "—"}`],
              ["Frames (analysed / raw)", `${au.num_frames} / ${au.raw_num_frames}`],
              ["Duration", `${n2(au.duration_s, 2)} s`],
              ["Frame-rate grade", au.fps_grade],
              ["Arm travel per frame", `${n2(au.est_arm_deg_per_frame, 1)}°`],
              ["Blur (bowler region)", `${au.blur_grade} · ${n2(au.median_laplacian_var, 0)} (${au.blur_roi_method})`],
              ["Timebase", `${au.timebase_status}${au.timebase_unreliable ? " (unverified)" : ""}`],
              ["Held frames", `${n2(au.held_frame_ratio * 100, 1)}% · longest ${au.longest_held_run_frames}`],
            ]}
          />
          {au.warnings.length > 0 && (
            <ul className="mt-2 space-y-0.5">
              {au.warnings.map((w, i) => (
                <li key={i} className="text-[11.5px] text-[#666666]">
                  • {w}
                </li>
              ))}
            </ul>
          )}
        </section>

        <section className="v-panel p-4">
          <SectionLabel className="mb-2">ISB 87-marker anatomy</SectionLabel>
          {isb ? (
            <>
              <KV
                rows={[
                  ["Effective markers · pelvis", n2(isb.n_eff_pelvis, 3)],
                  ["Effective markers · thorax", n2(isb.n_eff_thorax, 3)],
                  ["Effective markers · foot", n2(isb.n_eff_foot ?? null, 3)],
                  ["Assumed marker correlation ρ", n2(isb.rho_assumed, 2)],
                  ["Front-foot strike", isb.front_foot_strike_mode || "—"],
                ]}
              />
              {Object.keys(isb.scalars ?? {}).length > 0 && (
                <div className="mt-3 pt-3 border-t border-[#f0f0f0]">
                  <KV rows={Object.entries(isb.scalars).map(([k, v]) => [k, n2(v, 2)] as [string, React.ReactNode])} />
                </div>
              )}
              {isb.metrics && Object.keys(isb.metrics).length > 0 && (
                <div className="mt-3 pt-3 border-t border-[#f0f0f0] space-y-1">
                  {Object.values(isb.metrics).map((m) => {
                    const t = fmtMetric(m);
                    return (
                      <div key={m.key} className="flex items-baseline justify-between gap-2 text-[12.5px]">
                        <span className="text-[#4d4d4d]">{m.label}</span>
                        <span className="flex items-baseline gap-1.5">
                          <span className="v-num text-[#171717]">{t.valueText ?? "—"}</span>
                          {t.rangeText && <span className="v-num text-[11px] text-[#8f8f8f]">{t.rangeText}</span>}
                          <ObsBadge grade={m.observability} compact />
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </>
          ) : (
            <p className="text-[12px] text-[#8f8f8f]">The 87-marker anatomy model wasn't run for this delivery.</p>
          )}
        </section>
      </div>

      {story && story.detail_rows.length > 0 && (
        <section className="v-panel p-4">
          <SectionLabel className="mb-2">Plain-English status per measurement</SectionLabel>
          <ul className="divide-y divide-[#f5f5f5]">
            {[...story.detail_rows, ...(story.v5_detail_rows ?? [])].map((r) => (
              <li key={r.metric_key} className="py-1.5 flex items-center justify-between gap-2 text-[12.5px]">
                <span className="text-[#171717] min-w-0 truncate">{r.label_plain}</span>
                <span className="flex items-center gap-2 shrink-0">
                  <span className="v-num text-[#171717]">{r.value_text ?? "—"}</span>
                  <StatusPill status={r.status} label={r.status_label} />
                </span>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(fs.hard_guardrails?.length ?? 0) > 0 && (
        <section className="v-panel p-4">
          <SectionLabel className="mb-2">Hard guardrails applied</SectionLabel>
          <ul className="space-y-1">
            {fs.hard_guardrails.map((g, i) => (
              <li key={i} className="v-guardrail">
                • {g}
              </li>
            ))}
          </ul>
        </section>
      )}

      {detail.report_markdown && (
        <details className="v-panel p-4">
          <summary className="cursor-pointer text-[13px] text-[#171717]">Full grounded report (Markdown)</summary>
          <pre className="mt-3 text-[11.5px] leading-[1.55] text-[#4d4d4d] whitespace-pre-wrap break-words font-[var(--font-geist-mono)]">{detail.report_markdown}</pre>
        </details>
      )}
    </div>
  );
};

export default DeliveryView;
