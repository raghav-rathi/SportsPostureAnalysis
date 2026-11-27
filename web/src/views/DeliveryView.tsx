import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Check, GitCompare, MoreHorizontal, Pencil, Send, Share2, Trash2, X } from "lucide-react";
import { api } from "../api";
import { navigate, setProLab, useProLab, type DeliveryTab } from "../router";
import { useStudioStore } from "../store";
import type {
  BodyCheckpoint,
  CheckpointId,
  CoachNote,
  CoachingStory,
  DeliveryDetailResponse,
  KeyMoment,
  MomentId,
  SessionSummary,
  StoryMetricRow,
} from "../types";
import { StoryPlayer, type StoryCaption, type StoryPlayerHandle } from "../components/story/StoryPlayer";
import { MomentsScrubber } from "../components/story/MomentsScrubber";
import { BodyMap } from "../components/story/BodyMap";
import { CameraCoachPill, CheckpointList, DrillCardView, TakeawayCard } from "../components/story/StoryCards";
import {
  CHECKPOINT_TITLES,
  CLARITY_LABEL,
  EmptyState,
  NoticeFooter,
  SectionTitle,
  Spinner,
  StatusBadge,
  ValueWithRange,
} from "../components/story/ui";
const LabTab = lazy(() => import("../components/LabTab").then((m) => ({ default: m.LabTab })));
import { CoachCardView } from "./CoachCardView";

interface DeliveryViewProps {
  id: string;
  tab: DeliveryTab;
  onOpenFilmGuide: () => void;
}

export const DeliveryView: React.FC<DeliveryViewProps> = ({ id, tab, onOpenFilmGuide }) => {
  const [detail, setDetail] = useState<DeliveryDetailResponse | null>(null);
  const [session, setSession] = useState<SessionSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const proLab = useProLab();

  useEffect(() => {
    let cancelled = false;
    setDetail(null);
    setError(null);
    // Reset lab-only transient state when switching deliveries.
    const st = useStudioStore.getState();
    st.clearDraftLandmarks();
    st.setCalibrationMode(false);
    st.setRetrackMode(false);
    st.setActiveTool("cursor");
    st.setGhostDeliveryId(null);
    api
      .getDelivery(id)
      .then(async (d) => {
        if (cancelled) return;
        setDetail(d);
        useStudioStore.getState().setCurrentFrame(d.fact_sheet.events.is_complete_delivery ? d.fact_sheet.events.br.frame_idx : 0);
        try {
          const s = await api.getSession(d.delivery.session_id);
          if (!cancelled) setSession(s.session);
        } catch {
          /* session strip is optional */
        }
      })
      .catch((e) => !cancelled && setError(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, [id]);

  // If Pro Lab gets switched off while on the lab tab, fall back to the story.
  useEffect(() => {
    if (tab === "lab" && !proLab) navigate({ name: "delivery", id, tab: "story" });
  }, [tab, proLab, id]);

  if (error) {
    return (
      <EmptyState
        title="We couldn't open this delivery"
        body={<span>It may have been deleted, or the studio server isn't running. ({error})</span>}
        action={
          <button onClick={() => navigate({ name: "home" })} className="v-btn-primary h-9 px-4 text-[13px]">
            Back to home
          </button>
        }
      />
    );
  }
  if (!detail) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
        <Spinner /> Opening delivery…
      </div>
    );
  }

  const story = detail.story ?? null;

  if (tab === "card") {
    return <CoachCardView detail={detail} story={story} session={session} />;
  }

  const isBuiltIn = id.startsWith("del_demo_") || id.startsWith("del_real_");

  return (
    <div className="space-y-5">
      <DeliveryHeader
        detail={detail}
        story={story}
        session={session}
        tab={tab}
        proLab={proLab}
        canEdit={!isBuiltIn}
        onRenamed={(label) => setDetail((p) => (p ? { ...p, delivery: { ...p.delivery, label } } : p))}
      />

      {tab === "lab" && proLab ? (
        <Suspense
          fallback={
            <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
              <Spinner /> Loading lab tools…
            </div>
          }
        >
          <LabTab
            detail={detail}
            session={session}
            onDetailChange={(fn) => setDetail((p) => (p ? fn(p) : p))}
            onOpenCompare={() => navigate({ name: "compare", a: id, b: null })}
          />
        </Suspense>
      ) : !story ? (
        <EmptyState
          title="Your coaching story is still being prepared"
          body="This delivery was analysed with an older version of the studio. Refresh in a moment, or switch on Pro Lab to see the raw measurements."
          action={
            <button onClick={() => window.location.reload()} className="v-btn-primary h-9 px-4 text-[13px]">
              Refresh
            </button>
          }
        />
      ) : tab === "practice" ? (
        <PracticeTab story={story} id={id} bowlerId={session?.bowler_id ?? null} onOpenFilmGuide={onOpenFilmGuide} />
      ) : tab === "details" ? (
        <DetailsTab story={story} detail={detail} onNotesChanged={(notes) => setDetail((p) => (p ? { ...p, notes } : p))} />
      ) : (
        <StoryTab detail={detail} story={story} session={session} onOpenFilmGuide={onOpenFilmGuide} />
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Header: back, title (rename), tabs, Pro Lab switch, actions
// ─────────────────────────────────────────────────────────────────────────────
const DeliveryHeader: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
  tab: DeliveryTab;
  proLab: boolean;
  canEdit: boolean;
  onRenamed: (label: string) => void;
}> = ({ detail, story, session, tab, proLab, canEdit, onRenamed }) => {
  const id = detail.delivery.id;
  const [editing, setEditing] = useState(false);
  const [label, setLabel] = useState(detail.delivery.label);
  const [menu, setMenu] = useState(false);
  useEffect(() => setLabel(detail.delivery.label), [detail.delivery.label]);

  const saveLabel = async () => {
    const next = label.trim();
    setEditing(false);
    if (!next || next === detail.delivery.label) return;
    onRenamed(next);
    try {
      await api.patchDelivery(id, { label: next });
    } catch {
      /* keep optimistic label; backend may be older */
    }
  };

  const del = async () => {
    if (!window.confirm("Delete this delivery? This can't be undone.")) return;
    await api.deleteDelivery(id);
    navigate({ name: "deliveries", bowler: null });
  };

  const tabs: Array<{ key: DeliveryTab; label: string }> = [
    { key: "story", label: "Story" },
    { key: "practice", label: `Practice${story && story.drills.length ? ` · ${story.drills.length}` : ""}` },
    { key: "details", label: "All measurements" },
    ...(proLab ? [{ key: "lab" as DeliveryTab, label: "Lab" }] : []),
  ];

  return (
    <header className="space-y-3">
      <button
        onClick={() => navigate({ name: "deliveries", bowler: null })}
        className="inline-flex items-center gap-1 text-[13px] text-[#666666] hover:text-[#171717]"
      >
        <ArrowLeft className="w-3.5 h-3.5" /> Deliveries
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
                  if (e.key === "Enter") saveLabel();
                  if (e.key === "Escape") setEditing(false);
                }}
                className="v-input h-9 px-2.5 text-[18px] w-full max-w-[420px]"
                aria-label="Delivery name"
              />
              <button onClick={saveLabel} className="v-btn-primary h-9 w-9 flex items-center justify-center" aria-label="Save name">
                <Check className="w-4 h-4" />
              </button>
              <button onClick={() => setEditing(false)} className="v-btn-ghost h-9 w-9 flex items-center justify-center" aria-label="Cancel">
                <X className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <h1 className="text-[24px] md:text-[28px] font-[500] tracking-[-0.03em] text-[#171717] flex items-center gap-2 min-w-0">
              <span className="truncate">{detail.delivery.label}</span>
              {canEdit && (
                <button onClick={() => setEditing(true)} className="text-[#a8a8a8] hover:text-[#171717] shrink-0" aria-label="Rename delivery">
                  <Pencil className="w-4 h-4" />
                </button>
              )}
            </h1>
          )}
          <div className="text-[13px] text-[#666666] mt-1 flex flex-wrap items-center gap-x-2">
            {session?.bowler_name && <span className="text-[#171717]">{session.bowler_name}</span>}
            {story?.subhead && <span>{session?.bowler_name ? "· " : ""}{story.subhead}</span>}
            {story?.is_sample && (
              <span className="h-5 px-2 rounded-full border border-[#ebebeb] bg-[#ffffff] text-[11px] inline-flex items-center">
                {story.sample_kind === "real" ? "Sample · real video" : "Sample · demo"}
              </span>
            )}
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <label className="flex items-center gap-2 h-8 px-2.5 rounded-[6px] border border-[#ebebeb] bg-[#ffffff] text-[13px] cursor-pointer select-none" title="Show raw lab tools">
            <span className="text-[#4d4d4d]">Pro Lab</span>
            <button
              role="switch"
              aria-checked={proLab}
              onClick={() => {
                const next = !proLab;
                setProLab(next);
                if (next) navigate({ name: "delivery", id, tab: "lab" });
              }}
              className={`relative w-8 h-[18px] rounded-full transition-colors ${proLab ? "bg-[#171717]" : "bg-[#d4d4d8]"}`}
            >
              <span className={`absolute top-[2px] w-[14px] h-[14px] rounded-full bg-[#ffffff] transition-all ${proLab ? "left-[16px]" : "left-[2px]"}`} />
            </button>
          </label>
          <button onClick={() => navigate({ name: "compare", a: id, b: null })} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <GitCompare className="w-3.5 h-3.5" /> Compare
          </button>
          <button onClick={() => navigate({ name: "delivery", id, tab: "card" })} className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Share2 className="w-3.5 h-3.5" /> Share card
          </button>
          {canEdit && (
            <div className="relative">
              <button onClick={() => setMenu((v) => !v)} className="v-btn-ghost h-8 w-8 flex items-center justify-center" aria-label="More actions">
                <MoreHorizontal className="w-4 h-4" />
              </button>
              {menu && (
                <div className="absolute right-0 mt-1 z-20 v-card py-1 w-44" onMouseLeave={() => setMenu(false)}>
                  <button onClick={() => { setMenu(false); setEditing(true); }} className="w-full text-left px-3 py-2 text-[13px] hover:bg-[#fafafa] flex items-center gap-2">
                    <Pencil className="w-3.5 h-3.5" /> Rename
                  </button>
                  <button onClick={del} className="w-full text-left px-3 py-2 text-[13px] hover:bg-[#fafafa] flex items-center gap-2">
                    <Trash2 className="w-3.5 h-3.5" /> Delete delivery
                  </button>
                </div>
              )}
            </div>
          )}
        </div>
      </div>
      <nav className="flex items-center gap-1 border-b border-[#ebebeb] overflow-x-auto" aria-label="Delivery sections">
        {tabs.map((t) => {
          const active = t.key === tab;
          return (
            <button
              key={t.key}
              onClick={() => navigate({ name: "delivery", id, tab: t.key })}
              className={`h-9 px-3 text-[14px] whitespace-nowrap -mb-px border-b-2 transition-colors ${
                active ? "border-[#171717] text-[#171717] font-[500]" : "border-transparent text-[#666666] hover:text-[#171717]"
              }`}
              aria-current={active ? "page" : undefined}
            >
              {t.label}
            </button>
          );
        })}
      </nav>
    </header>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Story tab
// ─────────────────────────────────────────────────────────────────────────────
const StoryTab: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory;
  session: SessionSummary | null;
  onOpenFilmGuide: () => void;
}> = ({ detail, story, session, onOpenFilmGuide }) => {
  const id = detail.delivery.id;
  const playerRef = useRef<StoryPlayerHandle | null>(null);
  const cpById = useMemo(() => {
    const m = new Map<CheckpointId, BodyCheckpoint>();
    story.checkpoints.forEach((c) => m.set(c.id, c));
    return m;
  }, [story.checkpoints]);
  const release = story.key_moments.find((m) => m.id === "ball_release") ?? story.key_moments[story.key_moments.length - 1] ?? null;

  const initialHighlight = (): CheckpointId | null => {
    const t = story.top_takeaway.checkpoint_id;
    if (t && cpById.get(t)?.status !== "not_visible") return t;
    const first = release?.checkpoint_ids.find((cid) => cpById.get(cid)?.status !== "not_visible");
    return first ?? null;
  };

  const [selectedId, setSelectedId] = useState<CheckpointId | null>(null);
  const [highlightId, setHighlightId] = useState<CheckpointId | null>(initialHighlight);
  const [activeMoment, setActiveMoment] = useState<MomentId | null>(release?.id ?? null);
  const [caption, setCaption] = useState<StoryCaption | null>(release ? { title: release.label, body: release.plain_tip } : null);
  const captionFrameRef = useRef<number | null>(release?.frame_idx ?? null);

  // Auto-intro once per browser session per delivery.
  const [introStop] = useState<number | null>(() => {
    if (!release) return null;
    const key = `spa.intro.${id}`;
    try {
      if (window.sessionStorage.getItem(key)) return null;
      window.sessionStorage.setItem(key, "1");
    } catch {
      /* ignore */
    }
    return release.frame_idx;
  });

  const seekWithCaption = (frame: number | null, cap: StoryCaption | null, moment: MomentId | null) => {
    if (frame === null) return;
    captionFrameRef.current = frame;
    setCaption(cap);
    setActiveMoment(moment);
    playerRef.current?.seekToFrame(frame);
  };

  const selectCheckpoint = (cid: CheckpointId) => {
    if (cid === selectedId) {
      setSelectedId(null);
      return;
    }
    setSelectedId(cid);
    const cp = cpById.get(cid);
    if (!cp) return;
    if (cp.status !== "not_visible") setHighlightId(cid);
    if (cp.frame_idx !== null) {
      seekWithCaption(cp.frame_idx, { title: `${cp.title} · ${cp.moment_label}`, body: cp.plain_meaning }, cp.moment_id);
    }
  };

  const seeCheckpoint = (cp: BodyCheckpoint) => {
    if (cp.status !== "not_visible") setHighlightId(cp.id);
    seekWithCaption(cp.frame_idx, { title: `${cp.title} · ${cp.moment_label}`, body: cp.plain_meaning }, cp.moment_id);
    window.scrollTo({ top: 0, behavior: "smooth" });
  };

  const selectMoment = (m: KeyMoment) => {
    const firstVisible = m.checkpoint_ids.find((cid) => cpById.get(cid)?.status !== "not_visible");
    setHighlightId(firstVisible ?? null);
    seekWithCaption(m.frame_idx, { title: m.label, body: m.plain_tip }, m.id);
  };

  const onSettledFrame = (f: number) => {
    if (captionFrameRef.current !== null && Math.abs(f - captionFrameRef.current) > 1) {
      captionFrameRef.current = null;
      setCaption(null);
      setActiveMoment(null);
    }
  };

  const takeaway = story.top_takeaway;
  const takeawayCp = takeaway.checkpoint_id ? cpById.get(takeaway.checkpoint_id) ?? null : null;
  const otherBalls = (session?.deliveries ?? []).filter((d) => d.id !== id);

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
        {/* LEFT: video + moments */}
        <div className="lg:col-span-7 min-w-0 space-y-4 lg:sticky lg:top-[72px]">
          <StoryPlayer
            ref={playerRef}
            videoUrl={detail.delivery.video_proxy_url}
            telemetry={detail.telemetry}
            keyMoments={story.key_moments}
            activeCheckpoint={highlightId ? cpById.get(highlightId) ?? null : null}
            activeMomentId={activeMoment}
            caption={caption}
            introStopFrame={introStop}
            onMomentClick={selectMoment}
            onSettledFrame={onSettledFrame}
          />
          <MomentsScrubber
            moments={story.key_moments}
            activeId={activeMoment}
            thumbFor={(m) => m.thumbnail_url ?? api.frameUrl(id, m.frame_idx)}
            onSelect={selectMoment}
          />
        </div>

        {/* RIGHT: understanding */}
        <div className="lg:col-span-5 min-w-0 space-y-4">
          <h2 className="text-[21px] md:text-[23px] leading-[1.3] font-[450] tracking-[-0.025em] text-[#171717]">{story.headline}</h2>
          <CameraCoachPill capture={story.capture} titleFor={(cid) => cpById.get(cid)?.title ?? CHECKPOINT_TITLES[cid]} onFilmingTips={onOpenFilmGuide} />
          <TakeawayCard
            takeaway={takeaway}
            onSeeMoment={() => {
              if (takeawayCp) seeCheckpoint(takeawayCp);
              else seekWithCaption(takeaway.frame_idx, { title: takeaway.title, body: takeaway.what_to_try }, takeaway.moment_id);
            }}
            onPractise={() => navigate({ name: "delivery", id, tab: "practice" })}
            onFilmingTips={onOpenFilmGuide}
          />

          {story.headline_stats.length > 0 && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              {story.headline_stats.map((s) => (
                <div key={s.metric_key} className="v-card px-4 py-3">
                  <div className="text-[12px] text-[#666666]">{s.label}</div>
                  <div className="mt-1">
                    <ValueWithRange value={s.value_text} range={s.range_text} size="md" inline />
                  </div>
                </div>
              ))}
            </div>
          )}

          <section className="v-card p-4 md:p-5">
            <SectionTitle hint="Tap a zone or a row">Body checkpoints</SectionTitle>
            <div className="grid grid-cols-1 sm:grid-cols-[minmax(0,240px)_1fr] lg:grid-cols-1 gap-2 items-start">
              <BodyMap checkpoints={story.checkpoints} selectedId={selectedId ?? highlightId} bowlingArm={detail.fact_sheet.bowling_arm} onSelect={selectCheckpoint} />
              <CheckpointList
                checkpoints={story.checkpoints}
                selectedId={selectedId}
                onSelect={selectCheckpoint}
                onSeeMoment={seeCheckpoint}
                onFilmingTips={onOpenFilmGuide}
              />
            </div>
            {story.action_style && (
              <div className="mt-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] px-3 py-2.5">
                <div className="text-[12px] text-[#666666]">Action style</div>
                <div className="text-[14px] text-[#171717] font-[500] mt-0.5">{story.action_style.label}</div>
                <div className="text-[12.5px] text-[#4d4d4d] mt-0.5 leading-snug">{story.action_style.plain_summary}</div>
              </div>
            )}
          </section>

          <div className="flex flex-wrap gap-2">
            {story.drills.length > 0 && (
              <button onClick={() => navigate({ name: "delivery", id, tab: "practice" })} className="v-btn-primary h-9 px-4 text-[13px]">
                Practise: {story.drills.length} drill{story.drills.length > 1 ? "s" : ""} for you →
              </button>
            )}
            <button onClick={() => navigate({ name: "delivery", id, tab: "details" })} className="v-btn-ghost h-9 px-4 text-[13px]">
              See all measurements
            </button>
          </div>
        </div>
      </div>

      {otherBalls.length > 0 && (
        <section>
          <SectionTitle>Other balls from this session</SectionTitle>
          <div className="flex gap-2 overflow-x-auto pb-1">
            {otherBalls.map((d) => (
              <button
                key={d.id}
                onClick={() => navigate({ name: "delivery", id: d.id, tab: "story" })}
                className="shrink-0 v-card px-3 py-2 text-[13px] text-[#171717] hover:bg-[#fafafa] max-w-[240px] truncate text-left"
              >
                {d.label}
              </button>
            ))}
          </div>
        </section>
      )}

      <NoticeFooter notices={story.notices} />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Practice tab
// ─────────────────────────────────────────────────────────────────────────────
const PracticeTab: React.FC<{
  story: CoachingStory;
  id: string;
  bowlerId: string | null;
  onOpenFilmGuide: () => void;
}> = ({ story, id, bowlerId, onOpenFilmGuide }) => {
  const titleOf = (cid: CheckpointId | null) => (cid ? story.checkpoints.find((c) => c.id === cid)?.title ?? CHECKPOINT_TITLES[cid] : null);
  const watch = story.checkpoints.filter((c) => c.status === "watch");
  return (
    <div className="max-w-[760px] min-w-0 space-y-5">
      {story.drills.length > 0 ? (
        <>
          <p className="text-[14px] text-[#4d4d4d] leading-relaxed">
            {story.drills.length === 1 ? "One drill" : `${story.drills.length} drills`} picked from what this clip clearly showed. Do them, then
            film another ball to see if it changes.
          </p>
          {story.drills.map((d, i) => (
            <DrillCardView
              key={d.id}
              drill={d}
              index={i}
              forTitle={titleOf(d.for_checkpoint_id)}
              onSeeMoment={d.frame_idx !== null ? () => navigate({ name: "delivery", id, tab: "story" }) : undefined}
            />
          ))}
        </>
      ) : (
        <div className="v-card p-6">
          <h2 className="text-[18px] font-[500] tracking-[-0.02em] text-[#171717]">Nothing in this clip clearly needs fixing</h2>
          <p className="mt-2 text-[14px] text-[#4d4d4d] leading-relaxed">
            We only suggest drills when a measurement clearly crosses a research threshold — even after allowing for measurement error. This clip
            didn't show one, so we won't invent generic drills.
          </p>
          {story.capture.next_clip_tip && (
            <p className="mt-3 text-[14px] text-[#171717]">
              {story.capture.next_clip_tip}{" "}
              <button onClick={onOpenFilmGuide} className="underline underline-offset-2">How to film</button>
            </p>
          )}
        </div>
      )}

      {watch.length > 0 && (
        <section className="v-card p-5">
          <SectionTitle>Keep an eye on</SectionTitle>
          <ul className="space-y-3">
            {watch.map((c) => (
              <li key={c.id} className="flex items-start justify-between gap-3">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="text-[14px] font-[500] text-[#171717]">{c.title}</span>
                    <StatusBadge status={c.status} label={c.status_label} compact />
                  </div>
                  <p className="text-[13px] text-[#4d4d4d] mt-1 leading-snug">
                    The likely range crosses the research line, so we can't call it from this clip. Film again to get a clearer read.
                  </p>
                </div>
                <ValueWithRange value={c.value_text} range={c.range_text} />
              </li>
            ))}
          </ul>
        </section>
      )}

      <div className="flex flex-wrap gap-2">
        <button onClick={() => navigate({ name: "new", bowler: bowlerId })} className="v-btn-primary h-9 px-4 text-[13px]">
          Film another ball →
        </button>
        <button onClick={() => navigate({ name: "compare", a: id, b: null })} className="v-btn-ghost h-9 px-4 text-[13px]">
          Compare with another ball
        </button>
      </div>
      <NoticeFooter notices={story.notices} />
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Details tab: all measurements in plain language + ask + notes
// ─────────────────────────────────────────────────────────────────────────────
const DetailsTab: React.FC<{
  story: CoachingStory;
  detail: DeliveryDetailResponse;
  onNotesChanged: (notes: CoachNote[]) => void;
}> = ({ story, detail, onNotesChanged }) => {
  const groups = useMemo(() => {
    const order: Array<CheckpointId | "other"> = [...story.checkpoints.map((c) => c.id), "other"];
    const by = new Map<string, StoryMetricRow[]>();
    (story.detail_rows ?? []).forEach((r) => {
      const k = r.group ?? "other";
      if (!by.has(k)) by.set(k, []);
      by.get(k)!.push(r);
    });
    return order
      .filter((k) => by.has(k))
      .map((k) => ({
        key: k,
        title: k === "other" ? "Other measurements" : story.checkpoints.find((c) => c.id === k)?.title ?? CHECKPOINT_TITLES[k as CheckpointId],
        rows: by.get(k)!,
      }));
  }, [story]);

  return (
    <div className="grid grid-cols-1 xl:grid-cols-12 gap-5 items-start">
      <div className="xl:col-span-8 min-w-0 space-y-4">
        <p className="text-[13.5px] text-[#4d4d4d] leading-relaxed">
          Every measurement from this clip, in plain language. The small grey range is where the true value very likely sits — when a part of
          the body wasn't clearly visible, we say so instead of guessing.
        </p>
        {groups.length === 0 && <div className="v-card p-5 text-[13px] text-[#666666]">Detailed measurements aren't available for this delivery yet.</div>}
        {groups.map((g) => (
          <section key={g.key} className="v-card">
            <div className="px-4 md:px-5 pt-4 pb-2 text-[14px] font-[500] text-[#171717]">{g.title}</div>
            <ul className="divide-y divide-[#ebebeb]">
              {g.rows.map((r) => (
                <li key={r.metric_key} className="px-4 md:px-5 py-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[14px] text-[#171717]">{r.label_plain}</span>
                        <StatusBadge status={r.status} label={r.status_label} compact />
                      </div>
                      <p className="text-[12.5px] text-[#666666] mt-0.5 leading-snug">
                        {r.suppressed ? r.suppression_plain || "Not clearly visible in this clip." : r.plain_note}
                      </p>
                    </div>
                    <div className="text-right shrink-0">
                      {r.suppressed ? (
                        <span className="text-[12px] text-[#a8a8a8]">Not visible</span>
                      ) : (
                        <>
                          <ValueWithRange value={r.value_text} range={r.range_text} size="sm" />
                          {r.view_clarity && <div className="text-[11px] text-[#8f8f8f] mt-0.5">{CLARITY_LABEL[r.view_clarity]}</div>}
                        </>
                      )}
                    </div>
                  </div>
                  {(r.definition_note || r.literature_source) && (
                    <details className="mt-1.5 group">
                      <summary className="text-[12px] text-[#8f8f8f] cursor-pointer hover:text-[#171717] list-none">How this is measured</summary>
                      <p className="mt-1 text-[12px] text-[#4d4d4d] leading-snug">{r.definition_note}</p>
                      {r.literature_source && <p className="mt-1 text-[11px] text-[#8f8f8f]">{r.literature_source}</p>}
                    </details>
                  )}
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
      <div className="xl:col-span-4 min-w-0 space-y-4">
        <AskCoach deliveryId={detail.delivery.id} />
        <NotesPanel deliveryId={detail.delivery.id} notes={detail.notes || []} onChanged={onNotesChanged} />
      </div>
      <div className="xl:col-span-12">
        <NoticeFooter notices={story.notices} />
      </div>
    </div>
  );
};

const SUGGESTED_QUESTIONS = ["What should I work on first?", "What did this camera angle miss?", "Which drill should I do?"];

const AskCoach: React.FC<{ deliveryId: string }> = ({ deliveryId }) => {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [answer, setAnswer] = useState<{ text: string; ok: boolean } | null>(null);
  const ask = async (question: string) => {
    if (!question.trim()) return;
    setBusy(true);
    setAnswer(null);
    try {
      const r = await api.askCoach(deliveryId, question.trim());
      setAnswer({
        text: r.verification_passed ? r.answer_markdown : "I can't answer that reliably from this clip's measurements.",
        ok: r.verification_passed,
      });
    } catch (e: any) {
      setAnswer({ text: "The coach is unavailable right now.", ok: false });
    } finally {
      setBusy(false);
    }
  };
  return (
    <section className="v-card p-4">
      <div className="text-[14px] font-[500] text-[#171717]">Ask about this ball</div>
      <p className="text-[12px] text-[#8f8f8f] mt-0.5 leading-snug">Answers only use this clip's measurements, and every number is checked.</p>
      <div className="mt-3 flex flex-wrap gap-1.5">
        {SUGGESTED_QUESTIONS.map((s) => (
          <button key={s} onClick={() => { setQ(s); ask(s); }} className="h-7 px-2.5 rounded-full border border-[#ebebeb] bg-[#ffffff] text-[12px] text-[#4d4d4d] hover:border-[#c9c9c9]">
            {s}
          </button>
        ))}
      </div>
      <form
        className="mt-3 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Type a question…" className="v-input h-9 px-2.5 text-[13px] flex-1 min-w-0" />
        <button type="submit" disabled={busy || !q.trim()} className="v-btn-primary h-9 w-9 flex items-center justify-center" aria-label="Ask">
          {busy ? <Spinner className="border-t-[#ffffff]" /> : <Send className="w-3.5 h-3.5" />}
        </button>
      </form>
      {answer && (
        <div className={`mt-3 rounded-[6px] border px-3 py-2.5 text-[13px] leading-[1.55] whitespace-pre-wrap ${answer.ok ? "border-[#ebebeb] bg-[#fafafa] text-[#171717]" : "border-dashed border-[#c9c9c9] text-[#4d4d4d]"}`}>
          {answer.text.replace(/\*\*/g, "").replace(/`/g, "")}
        </div>
      )}
    </section>
  );
};

const NotesPanel: React.FC<{ deliveryId: string; notes: CoachNote[]; onChanged: (n: CoachNote[]) => void }> = ({ deliveryId, notes, onChanged }) => {
  const [body, setBody] = useState("");
  const add = async () => {
    if (!body.trim()) return;
    try {
      const r = await api.addNote(deliveryId, { tag: "general", body: body.trim(), frame_idx: null, timestamp_s: null });
      onChanged([r.note, ...notes]);
      setBody("");
    } catch {
      /* ignore */
    }
  };
  const remove = async (nid: string) => {
    try {
      await api.deleteNote(deliveryId, nid);
      onChanged(notes.filter((n) => n.id !== nid));
    } catch {
      /* ignore */
    }
  };
  return (
    <section className="v-card p-4">
      <div className="text-[14px] font-[500] text-[#171717]">Coach notes</div>
      <textarea
        value={body}
        onChange={(e) => setBody(e.target.value)}
        placeholder="Add a note for this ball…"
        rows={3}
        className="v-input mt-3 w-full px-2.5 py-2 text-[13px] resize-none"
      />
      <div className="flex justify-end mt-2">
        <button onClick={add} disabled={!body.trim()} className="v-btn-primary h-8 px-3 text-[13px]">
          Save note
        </button>
      </div>
      {notes.length > 0 && (
        <ul className="mt-3 divide-y divide-[#ebebeb] border-t border-[#ebebeb]">
          {notes.map((n) => (
            <li key={n.id} className="py-2.5 flex items-start justify-between gap-2">
              <div>
                <p className="text-[13px] text-[#171717] leading-snug whitespace-pre-wrap">{n.body}</p>
                <p className="text-[11px] text-[#8f8f8f] mt-0.5">{new Date(n.created_at).toLocaleString()}</p>
              </div>
              <button onClick={() => remove(n.id)} className="text-[#a8a8a8] hover:text-[#171717]" aria-label="Delete note">
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};
