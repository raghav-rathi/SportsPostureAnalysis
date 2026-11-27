/**
 * InspectorPanel — right-hand inspector (desktop) / tabbed drawer (mobile) for the Delivery Workbench
 * (docs/research_app_landscape_v5.md §8.2 "Inspector").
 *
 * Takeaway + camera-quality pill · top-3 measured hero tiles · sculpted BodyMap · 7 deep cards (Action style,
 * Tempo, Head, Lead arm, COM & braking, Foot plant & crease, Release point) · UnlockStrip · grounded Coach
 * Copilot · frame-stamped notes (text + optional browser voice clip).
 *
 * Rules: every number carries its interval + observability glyph; unmeasured items are ghosted with their reason
 * (never removed); status uses glyph + words (StatusPill), never segment hues; nothing is computed client-side
 * that the backend did not measure.
 */
import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { navigate } from "../router";
import type {
  BodyCheckpoint,
  BodyZone,
  CheckpointId,
  CheckpointStatus,
  CoachNote,
  CoachingStory,
  DeliveryDetailResponse,
  MetricEstimate,
  ObservabilityGrade,
  TempoPhaseSegment,
  V5InsightCard,
  V5InsightId,
  ViewClarity,
} from "../types";
import { J, SEG, type Smpl24Joint } from "../theme";
import { usePlayhead } from "./PlayheadProvider";
import {
  MetricRow,
  MetricTile,
  ObsBadge,
  RangeRail,
  SectionLabel,
  StatusPill,
  UnlockStrip,
  autoRailDomain,
  fmtMetric,
  isMeasured,
} from "../components/ui/PaperPrimitives";

export interface InspectorFocus {
  frame: number | null;
  joints: number[];
  caption: string | null;
  checkpointId?: CheckpointId | null;
}

export interface InspectorPanelProps {
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  notes: CoachNote[];
  onNotesChange: (notes: CoachNote[]) => void;
  onFocus: (f: InspectorFocus) => void;
  activeCheckpointId?: CheckpointId | null;
  onOpenFilmGuide?: () => void;
  className?: string;
}

type MobileTab = "story" | "deep" | "coach" | "notes";

const STATUS_RANK: Record<CheckpointStatus, number> = { focus: 0, watch: 1, strong: 2, on_track: 3, not_visible: 9 };
const ZONE_COLOR: Record<BodyZone, string> = {
  knee: SEG.pelvis,
  approach: SEG.pelvis,
  trunk: SEG.thorax,
  spine: SEG.thorax,
  shoulders: SEG.thorax,
};
const CLARITY_TO_OBS: Record<Exclude<ViewClarity, null>, ObservabilityGrade> = { clear: "high", fair: "moderate", rough: "low" };

export function jointIndices(names: string[] | null | undefined): number[] {
  return (names ?? []).map((n) => J[n as Smpl24Joint]).filter((i): i is number => typeof i === "number");
}

export const InspectorPanel: React.FC<InspectorPanelProps> = ({
  detail,
  story,
  notes,
  onNotesChange,
  onFocus,
  activeCheckpointId = null,
  onOpenFilmGuide,
  className = "",
}) => {
  const ph = usePlayhead();
  const [mTab, setMTab] = useState<MobileTab>("story");
  const fs = detail.fact_sheet;
  const metricOf = (key: string): MetricEstimate | undefined => fs.metrics?.[key] ?? fs.v5_metrics?.[key];
  const cohortBand = (key: string): [number, number] | null => {
    const c = fs.cohort_comparisons?.find((x) => x.metric_key === key);
    return c ? [c.cohort_mean - c.cohort_sd, c.cohort_mean + c.cohort_sd] : null;
  };

  const focus = (f: InspectorFocus) => {
    if (f.frame !== null && f.frame !== undefined) ph.seekToEventFrame(f.frame, true);
    onFocus(f);
  };
  const focusCheckpoint = (cp: BodyCheckpoint) =>
    focus({
      frame: cp.frame_idx,
      joints: cp.status === "not_visible" ? [] : jointIndices(cp.highlight_joints),
      caption: `${cp.title} · ${cp.moment_label}${cp.value_text ? ` · ${cp.value_text}` : ""}`,
      checkpointId: cp.id,
    });

  const insights = useMemo(() => {
    const m = new Map<V5InsightId, V5InsightCard>();
    (story?.v5_insights ?? []).forEach((c) => m.set(c.id, c));
    return m;
  }, [story]);

  // ── Hero tiles: top 3 measured checkpoints by story priority ──
  const heroes = useMemo(() => {
    if (!story) return [] as BodyCheckpoint[];
    return story.checkpoints
      .filter((c) => c.status !== "not_visible" && c.value !== null)
      .sort((a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status])
      .slice(0, 3);
  }, [story]);

  // ── Unlock hints ──
  const unlock = useMemo(() => {
    const hints: string[] = [];
    let count = 0;
    if (story) {
      story.capture.cannot_see.forEach((c) => {
        count += 1;
        const cp = story.checkpoints.find((x) => x.id === c.checkpoint_id);
        hints.push(`${cp?.title ?? c.checkpoint_id}: ${cp?.how_to_unlock || c.reason_plain}`);
      });
      (story.v5_insights ?? []).forEach((c) => {
        if (c.status === "not_visible") {
          count += 1;
          if (c.unlock_tip) hints.push(`${c.title}: ${c.unlock_tip}`);
        }
      });
      if (story.capture.next_clip_tip) hints.unshift(story.capture.next_clip_tip);
    }
    return { count, hints };
  }, [story]);

  const sectionVis = (t: MobileTab) => (mTab === t ? "" : "hidden lg:block");

  return (
    <aside className={`min-w-0 ${className}`} aria-label="Inspector">
      {/* Mobile drawer tabs */}
      <nav className="lg:hidden sticky top-0 z-10 bg-[#fafafa] pb-2 -mx-1 px-1" aria-label="Inspector sections">
        <div className="v-seg w-full">
          {(
            [
              ["story", "Story"],
              ["deep", "Deep dive"],
              ["coach", "Coach"],
              ["notes", `Notes${notes.length ? ` · ${notes.length}` : ""}`],
            ] as Array<[MobileTab, string]>
          ).map(([k, l]) => (
            <button key={k} type="button" aria-pressed={mTab === k} onClick={() => setMTab(k)} className="flex-1">
              {l}
            </button>
          ))}
        </div>
      </nav>

      <div className="space-y-4">
        {/* ── STORY ── */}
        <div className={`space-y-4 ${sectionVis("story")}`}>
          {story ? (
            <>
              <div>
                <h2 className="text-[20px] leading-[1.3] font-[500] tracking-[-0.02em] text-[#171717]">{story.headline}</h2>
                {story.subhead && <p className="text-[13px] text-[#666666] mt-1">{story.subhead}</p>}
              </div>
              <CameraQualityPill story={story} onOpenFilmGuide={onOpenFilmGuide} />
              <TakeawayBlock
                story={story}
                onSee={() => {
                  const t = story.top_takeaway;
                  const cp = t.checkpoint_id ? story.checkpoints.find((c) => c.id === t.checkpoint_id) : null;
                  if (cp && cp.status !== "not_visible") focusCheckpoint(cp);
                  else focus({ frame: t.frame_idx, joints: [], caption: t.title, checkpointId: t.checkpoint_id });
                }}
                onPractise={() => navigate({ name: "delivery", id: detail.delivery.id, tab: "practice" })}
              />
            </>
          ) : (
            <div className="v-panel p-4 text-[13px] text-[#4d4d4d]">
              The plain-English story for this delivery is still being prepared. All measurements are in the Details tab.
            </div>
          )}

          {/* Hero tiles */}
          <div>
            <SectionLabel className="mb-2">Measured on this clip</SectionLabel>
            {heroes.length > 0 ? (
              <div className="grid grid-cols-1 sm:grid-cols-3 lg:grid-cols-1 xl:grid-cols-1 gap-2.5">
                {heroes.map((cp, i) => {
                  const m = metricOf(cp.metric_key);
                  const band = cohortBand(cp.metric_key);
                  const [mn, mx] = autoRailDomain(cp.value, cp.ci_low, cp.ci_high, cp.unit, band);
                  const obs: ObservabilityGrade = m?.observability ?? (cp.view_clarity ? CLARITY_TO_OBS[cp.view_clarity] : "moderate");
                  return (
                    <MetricTile
                      key={cp.id}
                      hero={i === 0}
                      label={cp.title}
                      valueText={cp.value_text ?? fmtMetric(m).valueText ?? "—"}
                      rangeText={cp.range_text ?? fmtMetric(m).rangeText ?? undefined}
                      range={{ value: cp.value, ciLow: cp.ci_low, ciHigh: cp.ci_high, min: mn, max: mx, color: ZONE_COLOR[cp.body_zone], cohortBand: band }}
                      observability={obs}
                      status={cp.status}
                      accentColor={ZONE_COLOR[cp.body_zone]}
                      active={activeCheckpointId === cp.id}
                      onSeeMoment={cp.frame_idx !== null ? () => focusCheckpoint(cp) : undefined}
                    />
                  );
                })}
              </div>
            ) : (
              <div className="rounded-[6px] border border-dashed border-[#d4d4d4] bg-[#fafafa] px-3 py-3 text-[13px] text-[#666666]">
                Nothing on this clip was clear enough to headline. The unlock tips below show what a better clip would add.
              </div>
            )}
            {heroes.some((h) => cohortBand(h.metric_key)) && (
              <p className="text-[11px] text-[#8f8f8f] mt-1.5">Light slate band on a rail = published group range (definition-matched study).</p>
            )}
          </div>

          {/* Body map */}
          {story && (
            <section className="v-panel p-3">
              <SectionLabel className="mb-1" right={<span className="text-[11px] text-[#8f8f8f]">Tap a zone</span>}>
                Body map
              </SectionLabel>
              <SculptedBodyMap
                checkpoints={story.checkpoints}
                selectedId={activeCheckpointId}
                bowlingArm={fs.bowling_arm}
                onSelect={(id) => {
                  const cp = story.checkpoints.find((c) => c.id === id);
                  if (cp) focusCheckpoint(cp);
                }}
              />
              <ul className="mt-1 divide-y divide-[#f0f0f0]">
                {story.checkpoints.map((cp) => (
                  <li key={cp.id}>
                    <button
                      type="button"
                      onClick={() => focusCheckpoint(cp)}
                      className={`w-full flex items-center justify-between gap-2 py-2 text-left ${activeCheckpointId === cp.id ? "bg-[#fafafa]" : ""}`}
                    >
                      <span className="text-[13px] text-[#171717] min-w-0 truncate">{cp.title}</span>
                      <span className="flex items-center gap-2 shrink-0">
                        {cp.value_text && <span className="v-num text-[12.5px] text-[#171717]">{cp.value_text}</span>}
                        <StatusPill status={cp.status} label={cp.status_label} />
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}

          <UnlockStrip missingCount={unlock.count} hints={unlock.hints} onOpenGuide={onOpenFilmGuide} />
        </div>

        {/* ── DEEP CARDS ── */}
        <div className={`space-y-2 ${sectionVis("deep")}`}>
          <SectionLabel className="mb-1">Deep dive</SectionLabel>
          <DeepCards detail={detail} story={story} insights={insights} metricOf={metricOf} onSeek={(f, caption, joints) => focus({ frame: f, joints: joints ?? [], caption })} />
        </div>

        {/* ── COACH ── */}
        <div className={sectionVis("coach")}>
          <CoachCopilot detail={detail} story={story} metricOf={metricOf} />
        </div>

        {/* ── NOTES ── */}
        <div className={sectionVis("notes")}>
          <FrameNotes detail={detail} notes={notes} onNotesChange={onNotesChange} />
        </div>
      </div>
    </aside>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Takeaway & camera quality
// ─────────────────────────────────────────────────────────────────────────────
const TakeawayBlock: React.FC<{ story: CoachingStory; onSee: () => void; onPractise: () => void }> = ({ story, onSee, onPractise }) => {
  const t = story.top_takeaway;
  const kindWord: Record<typeof t.kind, string> = {
    focus: "Focus",
    watch: "Keep an eye on",
    strength: "Strength",
    capture: "Better clip needed",
    all_clear: "All clear",
  };
  return (
    <section className="v-panel p-4">
      <SectionLabel>{kindWord[t.kind]}</SectionLabel>
      <h3 className="mt-1 text-[16px] font-[500] tracking-[-0.01em] text-[#171717] leading-snug">{t.title}</h3>
      <p className="mt-1.5 text-[13px] text-[#4d4d4d] leading-[1.55]">{t.plain_explanation}</p>
      {t.what_to_try && (
        <p className="mt-2 text-[13px] text-[#171717] leading-[1.55]">
          <span className="text-[#8f8f8f]">Try: </span>
          {t.what_to_try}
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-2">
        {(t.frame_idx !== null || t.checkpoint_id) && (
          <button type="button" onClick={onSee} className="v-btn-primary h-8 px-3 text-[12.5px]">
            See the moment
          </button>
        )}
        {t.linked_drill_id && (
          <button type="button" onClick={onPractise} className="v-btn-ghost h-8 px-3 text-[12.5px]">
            Practise this →
          </button>
        )}
      </div>
    </section>
  );
};

const GRADE_GLYPH: Record<string, string> = { great: "●", partial: "◐", limited: "◔", unusable: "○" };

const CameraQualityPill: React.FC<{ story: CoachingStory; onOpenFilmGuide?: () => void }> = ({ story, onOpenFilmGuide }) => {
  const [open, setOpen] = useState(false);
  const c = story.capture;
  return (
    <div className="v-panel">
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="w-full px-3 py-2.5 text-left space-y-1">
        <div className="flex items-center justify-between gap-2">
          <div className="flex flex-wrap items-center gap-x-1.5 gap-y-0.5 min-w-0">
            <span aria-hidden className="text-[13px] shrink-0">
              {GRADE_GLYPH[c.grade] ?? "◐"}
            </span>
            <span className="text-[13px] text-[#171717] font-[500]">{c.label}</span>
            <span className="v-num text-[12px] text-[#8f8f8f] whitespace-nowrap">
              {c.measured_count}/{c.total_count} measured
            </span>
          </div>
          <span className="text-[12px] text-[#666666] shrink-0 whitespace-nowrap">{open ? "Hide −" : "Camera coach +"}</span>
        </div>
        <div className="text-[12px] text-[#666666] leading-snug break-words">
          {c.view_label} · {c.fps_label}
        </div>
      </button>
      {open && (
        <div className="px-3 pb-3 border-t border-[#f0f0f0] pt-3 space-y-3">
          <div className="flex items-start gap-3">
            <PitchCameraDiagram view={c.view} />
            <p className="text-[12.5px] text-[#4d4d4d] leading-[1.5]">{c.summary}</p>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <div className="v-label mb-1">Could see</div>
              {c.can_see.length === 0 ? (
                <p className="text-[12px] text-[#8f8f8f]">Nothing clearly.</p>
              ) : (
                <ul className="space-y-0.5">
                  {c.can_see.map((id) => (
                    <li key={id} className="text-[12.5px] text-[#171717]">
                      ● {story.checkpoints.find((x) => x.id === id)?.title ?? id}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <div className="v-label mb-1">Couldn't see</div>
              {c.cannot_see.length === 0 ? (
                <p className="text-[12px] text-[#8f8f8f]">Everything visible.</p>
              ) : (
                <ul className="space-y-1">
                  {c.cannot_see.map((x) => (
                    <li key={x.checkpoint_id} className="text-[12.5px] text-[#4d4d4d] leading-snug">
                      ○ {story.checkpoints.find((y) => y.id === x.checkpoint_id)?.title ?? x.checkpoint_id}
                      <span className="block text-[11.5px] text-[#8f8f8f]">{x.reason_plain}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          {c.next_clip_tip && <p className="text-[12.5px] text-[#171717]">{c.next_clip_tip}</p>}
          {onOpenFilmGuide && (
            <button type="button" onClick={onOpenFilmGuide} className="v-btn-ghost h-7 px-2.5 text-[12px]">
              How to film
            </button>
          )}
        </div>
      )}
    </div>
  );
};

/** Top-down pitch: bowler runs left → right. Current camera (ink) and ideal spots (dashed). */
const PitchCameraDiagram: React.FC<{ view: CoachingStory["capture"]["view"] }> = ({ view }) => {
  const cams: Array<[number, number]> =
    view === "side_on"
      ? [[72, 62]]
      : view === "rear_on"
      ? [[8, 32]]
      : view === "front_on"
      ? [[164, 32]]
      : view === "oblique"
      ? [[22, 58]]
      : view === "dual_fused"
      ? [[72, 62], [8, 32]]
      : [];
  const Cam: React.FC<{ x: number; y: number; ghost?: boolean }> = ({ x, y, ghost }) => (
    <g transform={`translate(${x - 7},${y - 5})`}>
      <rect width="14" height="10" rx="2" fill={ghost ? "none" : "#171717"} stroke={ghost ? "#a8a8a8" : "none"} strokeDasharray={ghost ? "2 1.5" : undefined} />
      <circle cx="7" cy="5" r="2.4" fill={ghost ? "none" : "#ffffff"} stroke={ghost ? "#a8a8a8" : "none"} />
    </g>
  );
  return (
    <svg viewBox="0 0 172 70" className="w-[150px] h-[62px] shrink-0" role="img" aria-label="Where the camera was, from above">
      <rect x="24" y="22" width="124" height="20" rx="2" fill={SEG.grass} stroke="#d4d4d4" />
      <line x1="36" y1="22" x2="36" y2="42" stroke="#a8a8a8" />
      <line x1="136" y1="22" x2="136" y2="42" stroke="#a8a8a8" />
      <circle cx="44" cy="32" r="3.5" fill={SEG.pelvis} />
      <path d="M49 32 L64 32" stroke="#171717" strokeWidth="1" strokeDasharray="2 2" />
      <text x="44" y="14" fontSize="7.5" textAnchor="middle" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
        bowler
      </text>
      <Cam x={72} y={62} ghost />
      <Cam x={8} y={32} ghost />
      {cams.map(([x, y], i) => (
        <Cam key={i} x={x} y={y} />
      ))}
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Sculpted body map
// ─────────────────────────────────────────────────────────────────────────────
const ZONE_POS: Record<BodyZone, { x: number; y: number; lx: number; ly: number; anchor: "start" | "end" }> = {
  shoulders: { x: 112, y: 62, lx: 182, ly: 50, anchor: "start" },
  trunk: { x: 112, y: 96, lx: 182, ly: 98, anchor: "start" },
  spine: { x: 90, y: 104, lx: 20, ly: 96, anchor: "end" },
  knee: { x: 131, y: 178, lx: 182, ly: 172, anchor: "start" },
  approach: { x: 62, y: 236, lx: 20, ly: 222, anchor: "end" },
};

const SculptedBodyMap: React.FC<{
  checkpoints: BodyCheckpoint[];
  selectedId: CheckpointId | null;
  bowlingArm: "right" | "left";
  onSelect: (id: CheckpointId) => void;
}> = ({ checkpoints, selectedId, bowlingArm, onSelect }) => {
  const mirror = bowlingArm === "left";
  const sel = checkpoints.find((c) => c.id === selectedId) ?? null;
  const lit = (zones: BodyZone[]) => !sel || zones.includes(sel.body_zone);
  const op = (zones: BodyZone[]) => (lit(zones) ? 1 : 0.35);
  // capsule = solid outline under a tint fill (reads as a sculpted limb, not a stick)
  const Capsule: React.FC<{ pts: string; color: string; tint: string; w: number; zones: BodyZone[] }> = ({ pts, color, tint, w, zones }) => (
    <g opacity={op(zones)}>
      <polyline points={pts} fill="none" stroke={color} strokeWidth={w + 2.5} strokeLinecap="round" strokeLinejoin="round" />
      <polyline points={pts} fill="none" stroke={tint} strokeWidth={w} strokeLinecap="round" strokeLinejoin="round" />
    </g>
  );
  const fx = (x: number) => (mirror ? 200 - x : x);
  return (
    <svg viewBox="-80 0 360 262" className="w-full h-auto max-h-[280px]" role="group" aria-label="Body map. Tap a zone to jump to that checkpoint.">
      <defs>
        <radialGradient id="bm-shadow" cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#000" stopOpacity="0.12" />
          <stop offset="100%" stopColor="#000" stopOpacity="0" />
        </radialGradient>
      </defs>
      <line x1="-50" y1="244" x2="250" y2="244" stroke="#ebebeb" strokeWidth="1.5" />
      <ellipse cx={fx(100)} cy="245" rx="80" ry="5" fill="url(#bm-shadow)" />
      <g transform={mirror ? "translate(200,0) scale(-1,1)" : undefined}>
        {/* back leg + lead arm (far side) */}
        <Capsule pts="92,128 72,180 44,226 28,236" color={SEG.pelvis} tint="#E8F5EF" w={12} zones={["approach", "knee"]} />
        <Capsule pts="100,66 80,92 70,116" color={SEG.lead} tint="#EEF2FC" w={9} zones={["shoulders"]} />
        {/* pelvis block */}
        <g opacity={op(["spine", "trunk", "approach"])}>
          <ellipse cx="98" cy="128" rx="17" ry="11" fill={SEG.pelvisTint} stroke={SEG.pelvis} strokeWidth="1.6" transform="rotate(-12 98 128)" />
        </g>
        {/* ribcage / thorax */}
        <g opacity={op(["trunk", "spine", "shoulders"])}>
          <path d="M92 58 Q112 48 127 60 Q124 92 113 120 Q99 124 88 116 Q84 86 92 58 Z" fill={SEG.thoraxTint} stroke={SEG.thorax} strokeWidth="1.6" strokeLinejoin="round" />
          <path d="M96 74 Q108 70 120 76 M94 88 Q107 84 118 90 M94 102 Q105 98 115 104" fill="none" stroke={SEG.thorax} strokeOpacity="0.25" strokeWidth="1" />
        </g>
        {/* front leg */}
        <Capsule pts="104,130 131,178 150,236 168,238" color={SEG.pelvis} tint={SEG.pelvisTint} w={13} zones={["knee", "approach"]} />
        {/* bowling arm overhead + hand */}
        <Capsule pts="120,62 128,32 136,10" color={SEG.arm} tint={SEG.armTint} w={10} zones={["shoulders"]} />
        <g opacity={op(["shoulders"])}>
          <circle cx="137.5" cy="6" r="5" fill={SEG.wristTint} stroke={SEG.wrist} strokeWidth="1.6" />
        </g>
        {/* head */}
        <g opacity={op(["spine", "trunk", "shoulders"])}>
          <ellipse cx="112" cy="36" rx="11.5" ry="13" fill="#ffffff" stroke={SEG.head} strokeWidth="1.6" />
          <path d="M119 34 L124 36 L119 39" fill="none" stroke={SEG.head} strokeWidth="1.2" />
        </g>
        {/* approach arrow */}
        <path d="M-20 254 L56 254" stroke="#c9c9c9" strokeWidth="1.5" strokeDasharray="4 3" />
        <path d="M52 250 L58 254 L52 258" stroke="#c9c9c9" strokeWidth="1.5" fill="none" />
      </g>

      {checkpoints.map((cp) => {
        const z = ZONE_POS[cp.body_zone];
        if (!z) return null;
        const x = fx(z.x);
        const lx = mirror ? 200 - z.lx : z.lx;
        const anchor = mirror ? (z.anchor === "start" ? "end" : "start") : z.anchor;
        const selected = cp.id === selectedId;
        const nv = cp.status === "not_visible";
        const glyph = cp.status === "focus" ? "◆" : cp.status === "watch" ? "◇" : nv ? "○" : "●";
        const glyphFill = cp.status === "strong" ? SEG.positive : nv ? "#a8a8a8" : "#171717";
        return (
          <g
            key={cp.id}
            role="button"
            tabIndex={0}
            aria-label={`${cp.title}: ${cp.status_label}`}
            aria-pressed={selected}
            className="cursor-pointer focus:outline-none"
            onClick={() => onSelect(cp.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(cp.id);
              }
            }}
          >
            <line x1={x} y1={z.y} x2={lx + (anchor === "start" ? -4 : 4)} y2={z.ly - 4} stroke={selected ? "#171717" : "#d4d4d8"} strokeWidth="1" />
            <circle cx={x} cy={z.y} r={selected ? 12 : 10} fill={cp.status === "focus" ? "#171717" : "#ffffff"} stroke={selected ? "#171717" : nv ? "#c9c9c9" : "#a8a8a8"} strokeWidth={selected ? 2 : 1.2} strokeDasharray={nv ? "2 2" : undefined} />
            <text x={x} y={z.y + 4} textAnchor="middle" fontSize="11" fill={cp.status === "focus" ? "#ffffff" : glyphFill} style={{ pointerEvents: "none" }}>
              {glyph}
            </text>
            <text x={lx} y={z.ly - 6} textAnchor={anchor} fontSize="11.5" fontWeight={selected ? 500 : 400} fill="#171717" fontFamily="var(--font-geist-sans)">
              {cp.title}
            </text>
            <text x={lx} y={z.ly + 8} textAnchor={anchor} fontSize="10.5" fill={cp.status === "strong" ? SEG.positive : "#666666"} fontFamily="var(--font-geist-sans)">
              {cp.value_text ? `${cp.value_text} · ${cp.status_label}` : cp.status_label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Deep cards
// ─────────────────────────────────────────────────────────────────────────────
const DeepCard: React.FC<{
  title: string;
  status?: CheckpointStatus | null;
  headline?: string | null;
  headlineRange?: string | null;
  available: boolean;
  unavailableReason?: string | null;
  defaultOpen?: boolean;
  children?: React.ReactNode;
}> = ({ title, status, headline, headlineRange, available, unavailableReason, defaultOpen, children }) => {
  const [open, setOpen] = useState(!!defaultOpen);
  return (
    <section className={`v-panel ${available ? "" : "bg-[#fafafa]"}`}>
      <button type="button" onClick={() => setOpen((v) => !v)} aria-expanded={open} className="w-full flex items-center gap-2 px-3 py-2.5 text-left">
        <span className={`text-[13.5px] font-[500] flex-1 min-w-0 truncate ${available ? "text-[#171717]" : "text-[#8f8f8f]"}`}>{title}</span>
        {available && headline && (
          <span className="v-num text-[12.5px] text-[#171717] shrink-0">
            {headline}
            {headlineRange && <span className="text-[#8f8f8f]"> {headlineRange}</span>}
          </span>
        )}
        {status ? <StatusPill status={status} /> : !available ? <ObsBadge grade="unmeasurable" compact /> : null}
        <span className="text-[12px] text-[#a8a8a8] w-3 text-right">{open ? "−" : "+"}</span>
      </button>
      {open && (
        <div className="px-3 pb-3 border-t border-[#f0f0f0]">
          {!available ? (
            <p className="pt-2.5 text-[12.5px] text-[#666666] leading-snug">{unavailableReason || "Not measurable on this clip."}</p>
          ) : (
            children
          )}
        </div>
      )}
    </section>
  );
};

const Sentence: React.FC<{ children: React.ReactNode }> = ({ children }) => <p className="pt-2.5 pb-1 text-[12.5px] text-[#4d4d4d] leading-[1.5]">{children}</p>;

const DeepCards: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  insights: Map<V5InsightId, V5InsightCard>;
  metricOf: (k: string) => MetricEstimate | undefined;
  onSeek: (frame: number, caption: string, joints?: number[]) => void;
}> = ({ detail, story, insights, metricOf, onSeek }) => {
  const fs = detail.fact_sheet;
  const ks = fs.kinematic_sequence ?? null;
  const tempo = fs.tempo_rhythm ?? null;
  const head = fs.head_stability ?? null;
  const lead = fs.lead_arm ?? null;
  const com = fs.com_braking ?? null;
  const crease = fs.crease_tunnel ?? null;
  const ins = (id: V5InsightId) => insights.get(id) ?? null;
  const anyMeasured = (...ms: Array<MetricEstimate | null | undefined>) => ms.some((m) => isMeasured(m));
  const v5Missing = "This analysis isn't in this delivery's data yet — re-analyse it with the latest studio.";
  const insightReason = (id: V5InsightId, fallback: string) => ins(id)?.unlock_tip || ins(id)?.plain_summary || fallback;
  const scr = metricOf("shoulder_counter_rotation_deg");
  const ev = fs.events;

  const seekBtn = (frame: number | null | undefined, caption: string, joints?: string[]) =>
    frame !== null && frame !== undefined ? (
      <button type="button" onClick={() => onSeek(frame, caption, jointIndices(joints))} className="text-[12px] underline underline-offset-2 text-[#171717]">
        See the moment
      </button>
    ) : null;

  const b = fs.bowling_arm === "right" ? "r" : "l";
  const o = fs.bowling_arm === "right" ? "l" : "r";

  return (
    <div className="space-y-2">
      {/* 1 Action style */}
      <DeepCard
        title="Action style"
        headline={story?.action_style?.label ?? null}
        available={!!story?.action_style || isMeasured(scr)}
        unavailableReason={scr?.suppression_reason || "Shoulder alignment at back-foot landing wasn't measurable from this angle."}
        defaultOpen
      >
        <Sentence>{story?.action_style?.plain_summary || "Action type is read from how your hips and shoulders line up at back-foot landing (Portus 2004)."}</Sentence>
        <MetricRow
          label="Shoulder counter-rotation"
          metric={scr}
          color={SEG.thorax}
          domain={[0, 90]}
          markers={[{ at: 30, label: "Mixed-action research line: 30° (Portus 2004) — only called when the whole range sits above it" }]}
        />
        <MetricRow label="Shoulder alignment at back-foot landing" metric={metricOf("shoulder_alignment_bfc_deg")} color={SEG.thorax} />
        <MetricRow label="Hip alignment at back-foot landing" metric={metricOf("pelvis_alignment_bfc_deg")} color={SEG.pelvis} />
        <MetricRow label="Hip–shoulder separation at back-foot landing" metric={metricOf("hip_shoulder_sep_bfc_deg")} color={SEG.thorax} />
        <div className="pt-1">{seekBtn(ev.bfc.frame_idx, "Back foot lands · action alignment", ["lsho", "rsho", "lhip", "rhip"])}</div>
      </DeepCard>

      {/* 2 Tempo */}
      <DeepCard
        title="Delivery tempo"
        status={ins("tempo_rhythm")?.status ?? null}
        headline={ins("tempo_rhythm")?.headline_value_text ?? fmtMetric(tempo?.tempo_gather_to_whip_ratio).valueText}
        headlineRange={ins("tempo_rhythm")?.headline_range_text ?? null}
        available={!!tempo && (anyMeasured(tempo.tempo_stride_duration_ms, tempo.tempo_braking_to_br_ms) || tempo.waterfall_phases.length > 0)}
        unavailableReason={tempo ? insightReason("tempo_rhythm", tempo.summary_plain) : v5Missing}
      >
        {tempo && (
          <>
            <Sentence>{tempo.summary_plain}</Sentence>
            <TempoWaterfall phases={tempo.waterfall_phases} />
            <MetricRow label="Stride (back foot → front foot)" metric={tempo.tempo_stride_duration_ms} color={SEG.pelvis} />
            <MetricRow label="Brace to release (front foot → release)" metric={tempo.tempo_braking_to_br_ms} color={SEG.thorax} />
            <MetricRow label="Arm whip (arm level → release)" metric={tempo.tempo_uah_to_br_whip_ms} color={SEG.arm} />
            <MetricRow label="Gather-to-whip ratio" metric={tempo.tempo_gather_to_whip_ratio} />
          </>
        )}
      </DeepCard>

      {/* 3 Head */}
      <DeepCard
        title="Head stability"
        status={ins("head_stability")?.status ?? null}
        headline={ins("head_stability")?.headline_value_text ?? fmtMetric(head?.head_fall_away_ffc_to_br_deg).valueText}
        headlineRange={ins("head_stability")?.headline_range_text ?? null}
        available={!!head && anyMeasured(head.head_coronal_angle_br_deg, head.head_fall_away_ffc_to_br_deg, head.head_lateral_offset_from_ankle_br_cm, head.head_lateral_offset_from_ankle_br_pct_stature, head.head_vertical_drop_ffc_to_br_pct_stature)}
        unavailableReason={head ? insightReason("head_stability", head.summary_plain) : v5Missing}
      >
        {head && (
          <>
            <Sentence>{head.summary_plain}</Sentence>
            <HeadInset m={isMeasured(head.head_lateral_offset_from_ankle_br_cm) ? head.head_lateral_offset_from_ankle_br_cm : head.head_lateral_offset_from_ankle_br_pct_stature} />
            <MetricRow label="Head tilt at release" metric={head.head_coronal_angle_br_deg} color={SEG.ink} />
            <MetricRow label="Head fall-away (front foot → release)" metric={head.head_fall_away_ffc_to_br_deg} color={SEG.ink} />
            <MetricRow label="Head outside front ankle at release" metric={head.head_lateral_offset_from_ankle_br_cm} color={SEG.ink} />
            <MetricRow label="…as % of height" metric={head.head_lateral_offset_from_ankle_br_pct_stature} color={SEG.ink} />
            <MetricRow label="Head drop (front foot → release)" metric={head.head_vertical_drop_ffc_to_br_pct_stature} color={SEG.ink} />
            <div className="pt-1">{seekBtn(ev.br.frame_idx, "Ball release · head over front foot", ["head", "neck", `${o}ank`])}</div>
          </>
        )}
      </DeepCard>

      {/* 4 Lead arm */}
      <DeepCard
        title="Lead arm pull-down"
        status={ins("lead_arm")?.status ?? null}
        headline={ins("lead_arm")?.headline_value_text ?? fmtMetric(lead?.lead_arm_gather_peak_elev_deg).valueText}
        headlineRange={ins("lead_arm")?.headline_range_text ?? null}
        available={!!lead && anyMeasured(lead.lead_arm_gather_peak_elev_deg, lead.lead_arm_early_drop_pre_ffc_deg, lead.lead_arm_pulldown_peak_vel_dps, lead.lead_elbow_ribcage_flare_br_pct_stature)}
        unavailableReason={lead ? insightReason("lead_arm", lead.summary_plain) : v5Missing}
      >
        {lead && (
          <>
            <Sentence>{lead.summary_plain}</Sentence>
            <MetricRow label="Highest lead-arm reach in the gather" metric={lead.lead_arm_gather_peak_elev_deg} color={SEG.lead} domain={[-60, 90]} />
            <MetricRow label="Early drop before front foot" metric={lead.lead_arm_early_drop_pre_ffc_deg} color={SEG.lead} />
            <MetricRow label="Pull-down speed" metric={lead.lead_arm_pulldown_peak_vel_dps} color={SEG.lead} />
            <MetricRow label="Lead elbow away from ribcage at release" metric={lead.lead_elbow_ribcage_flare_br_pct_stature} color={SEG.lead} domain={[0, 45]} />
            <div className="pt-1">{seekBtn(ev.ffc.frame_idx, "Front foot plants · lead arm", [`${o}sho`, `${o}elb`, `${o}wri`])}</div>
          </>
        )}
      </DeepCard>

      {/* 5 COM & braking */}
      <DeepCard
        title="Body momentum & braking"
        status={ins("com_braking")?.status ?? null}
        headline={ins("com_braking")?.headline_value_text ?? fmtMetric(com?.front_leg_plant_angle_ffc_deg).valueText}
        headlineRange={ins("com_braking")?.headline_range_text ?? null}
        available={!!com && anyMeasured(com.com_ffc_speed_mps, com.com_br_speed_mps, com.com_braking_delta_vx_mps, com.com_braking_efficiency_pct, com.com_vertical_sink_pct_stature, com.front_leg_plant_angle_ffc_deg)}
        unavailableReason={com ? insightReason("com_braking", com.summary_plain) : v5Missing}
      >
        {com && (
          <>
            <Sentence>{com.summary_plain}</Sentence>
            <SpeedPair a={com.com_ffc_speed_mps} b={com.com_br_speed_mps} />
            <MetricRow label="Front-leg plant angle" metric={com.front_leg_plant_angle_ffc_deg} color={SEG.pelvis} />
            <MetricRow label="Body slows by (front foot → release)" metric={com.com_braking_delta_vx_mps} color={SEG.com} />
            <MetricRow label="Share of speed braked" metric={com.com_braking_efficiency_pct} color={SEG.com} />
            <MetricRow label="Body sink after front foot" metric={com.com_vertical_sink_pct_stature} color={SEG.com} />
            {com.model_reference && <p className="text-[11px] text-[#8f8f8f] pt-1">Body-mass model: {com.model_reference}</p>}
          </>
        )}
      </DeepCard>

      {/* 6 Foot plant & crease */}
      <DeepCard
        title="Foot plant & crease"
        status={ins("crease_tunnel")?.status ?? null}
        headline={crease && crease.corridor_verdict !== "unmeasurable" ? crease.corridor_label_plain : ins("crease_tunnel")?.headline_value_text ?? null}
        available={!!crease && (crease.corridor_verdict !== "unmeasurable" || anyMeasured(crease.foot_plant_corridor_offset_cm, crease.stride_alignment_angle_deg, crease.front_heel_crease_margin_cm))}
        unavailableReason={crease ? insightReason("crease_tunnel", crease.summary_plain) : v5Missing}
      >
        {crease && (
          <>
            <Sentence>{crease.summary_plain}</Sentence>
            <FootPlantInset offset={crease.foot_plant_corridor_offset_cm} crease={crease.front_heel_crease_margin_cm} label={crease.corridor_label_plain} />
            <MetricRow label="Front foot across from back foot (+ open, − crossed)" metric={crease.foot_plant_corridor_offset_cm} color={SEG.pelvis} />
            <MetricRow label="Stride direction vs pitch line" metric={crease.stride_alignment_angle_deg} color={SEG.pelvis} />
            <MetricRow label="Heel behind popping crease" metric={crease.front_heel_crease_margin_cm} color={SEG.pelvis} />
            {isMeasured(crease.front_heel_crease_margin_cm) && (
              <p className="text-[11px] text-[#8f8f8f] pt-1">Practice crease margin only — not an umpire's no-ball call.</p>
            )}
            <div className="pt-1">{seekBtn(ev.ffc.frame_idx, "Front foot plants · foot corridor", [`${b}ank`, `${o}ank`, `${b}toe`, `${o}toe`])}</div>
          </>
        )}
      </DeepCard>

      {/* 7 Release point */}
      <DeepCard
        title="Release point"
        headline={fmtMetric(metricOf("release_height_m")).valueText ?? fmtMetric(crease?.release_height_pct_stature).valueText}
        headlineRange={fmtMetric(metricOf("release_height_m")).rangeText}
        available={anyMeasured(metricOf("release_height_m"), crease?.release_height_pct_stature, crease?.release_lateral_from_ankle_m, crease?.release_lateral_from_stump_m)}
        unavailableReason={metricOf("release_height_m")?.suppression_reason || crease?.release_height_pct_stature?.suppression_reason || "Release point wasn't measurable on this clip."}
      >
        <Sentence>Where the ball left your hand, against real stump height (MCC Law 8: 71.1 cm).</Sentence>
        <ReleaseInset height={metricOf("release_height_m")} lateral={crease?.release_lateral_from_stump_m ?? crease?.release_lateral_from_ankle_m ?? null} lateralFromStump={!!crease && isMeasured(crease.release_lateral_from_stump_m)} />
        <MetricRow label="Release height" metric={metricOf("release_height_m")} color={SEG.wrist} />
        {crease && (
          <>
            <MetricRow label="Release height as % of your height" metric={crease.release_height_pct_stature} color={SEG.wrist} />
            <MetricRow label="Release out from front ankle" metric={crease.release_lateral_from_ankle_m} color={SEG.wrist} />
            <MetricRow label="Release out from middle stump" metric={crease.release_lateral_from_stump_m} color={SEG.wrist} />
            <MetricRow label="Release in front of popping crease" metric={crease.release_extension_past_crease_m} color={SEG.wrist} />
          </>
        )}
        <div className="pt-1">{seekBtn(ev.br.frame_idx, "Ball release", [`${b}wri`, `${b}han`, `${b}elb`])}</div>
      </DeepCard>

      {/* Sequence summary pointer (the waterfall itself lives in the timeline dock) */}
      {ks && (
        <p className="text-[11.5px] text-[#8f8f8f] px-1">
          Kinematic sequence: {ks.verdict_label_plain}. The four-lane chart is in the timeline below the video.
        </p>
      )}
    </div>
  );
};

/** GolfFix-style proportional ms waterfall with published reference bands (backend-supplied). */
const TempoWaterfall: React.FC<{ phases: TempoPhaseSegment[] }> = ({ phases }) => {
  const shown = phases.filter((p) => p.id !== "braking_ffc_br");
  if (shown.length === 0) return null;
  const t0 = Math.min(...shown.map((p) => Math.min(p.start_rel_ffc_ms, p.reference_range_ms ? p.start_rel_ffc_ms : Infinity)));
  const t1 = Math.max(
    ...shown.map((p) => Math.max(p.end_rel_ffc_ms, p.reference_range_ms ? p.start_rel_ffc_ms + p.reference_range_ms[1] : -Infinity))
  );
  const span = Math.max(1, t1 - t0);
  const pct = (t: number) => `${((t - t0) / span) * 100}%`;
  const color: Record<string, string> = {
    stride_bfc_ffc: SEG.pelvis,
    early_braking_ffc_uah: SEG.thorax,
    whip_uah_br: SEG.arm,
    follow_through: SEG.graphite,
  };
  return (
    <div className="py-2 space-y-1.5">
      {shown.map((p) => (
        <div key={p.id} className="grid grid-cols-[96px_1fr_52px] items-center gap-2">
          <span className="text-[11.5px] text-[#4d4d4d] truncate" title={p.label_plain}>
            {p.label_plain}
          </span>
          <div className="relative h-3.5 rounded-[3px] bg-[#fafafa] shadow-[inset_0_0_0_1px_#f0f0f0]">
            {p.reference_range_ms && (
              <div
                className="absolute top-0 bottom-0 rounded-[3px]"
                style={{
                  left: pct(p.start_rel_ffc_ms + p.reference_range_ms[0]),
                  width: `${((p.reference_range_ms[1] - p.reference_range_ms[0]) / span) * 100}%`,
                  background: SEG.ghostTint,
                }}
                title={`Published range ${p.reference_range_ms[0]}–${p.reference_range_ms[1]} ms`}
              />
            )}
            <div
              className="absolute top-[3px] bottom-[3px] rounded-[2px]"
              style={{ left: pct(p.start_rel_ffc_ms), width: `${(Math.max(0, p.duration_ms) / span) * 100}%`, background: color[p.id] ?? SEG.ink }}
            />
            {p.uncertainty_ms > 0 && (
              <div
                className="absolute top-1/2 h-px -translate-y-1/2 bg-[#171717]"
                style={{ left: pct(p.end_rel_ffc_ms - 1.96 * p.uncertainty_ms), width: `${((2 * 1.96 * p.uncertainty_ms) / span) * 100}%` }}
                title={`± ${Math.round(1.96 * p.uncertainty_ms)} ms`}
              />
            )}
          </div>
          <span className="v-num text-[11.5px] text-[#171717] text-right">{Math.round(p.duration_ms)} ms</span>
        </div>
      ))}
      <p className="text-[10.5px] text-[#8f8f8f]">Slate band = published range (e.g. Felton 2019 stride and brace timings); line = ± timing range.</p>
    </div>
  );
};

/** Coronal (from behind) inset: head offset outside the front-ankle plumb line, arm-agnostic labels. */
const HeadInset: React.FC<{ m: MetricEstimate | null | undefined }> = ({ m }) => {
  if (!isMeasured(m)) return null;
  const isCm = (m.unit || "").includes("cm");
  const lim = isCm ? 40 : 25;
  const x = (v: number) => 100 + (Math.max(-lim, Math.min(lim, v)) / lim) * 80;
  return (
    <svg viewBox="0 0 200 92" className="w-full h-[84px] my-1" role="img" aria-label="Head position relative to the front ankle at release, seen from behind">
      <line x1="10" y1="82" x2="190" y2="82" stroke="#ebebeb" />
      <line x1="100" y1="8" x2="100" y2="82" stroke="#a8a8a8" strokeDasharray="3 3" />
      <text x="100" y="90" fontSize="8" textAnchor="middle" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
        front ankle
      </text>
      {m.ci_low !== null && m.ci_high !== null && <rect x={x(m.ci_low)} y="20" width={Math.max(1, x(m.ci_high) - x(m.ci_low))} height="6" rx="3" fill={SEG.ink} opacity="0.15" />}
      <circle cx={x(m.value!)} cy="23" r="9" fill="#ffffff" stroke={SEG.head} strokeWidth="1.6" />
      <line x1={x(m.value!)} y1="32" x2="100" y2="80" stroke={SEG.thorax} strokeWidth="2" />
      <text x="190" y="12" fontSize="8" textAnchor="end" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
        falls away (non-bowling side) →
      </text>
      <text x="10" y="12" fontSize="8" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
        ← inside the front foot
      </text>
    </svg>
  );
};

const SpeedPair: React.FC<{ a: MetricEstimate; b: MetricEstimate }> = ({ a, b }) => {
  if (!isMeasured(a) && !isMeasured(b)) return null;
  const mx = Math.max(a.ci_high ?? a.value ?? 0, b.ci_high ?? b.value ?? 0, 1);
  const Bar: React.FC<{ label: string; m: MetricEstimate }> = ({ label, m }) => (
    <div className="grid grid-cols-[96px_1fr] items-center gap-2">
      <span className="text-[11.5px] text-[#4d4d4d]">{label}</span>
      {isMeasured(m) ? (
        <div className="flex items-center gap-2">
          <div className="flex-1">
            <RangeRail value={m.value} ciLow={m.ci_low} ciHigh={m.ci_high} min={0} max={mx * 1.1} color={SEG.com} />
          </div>
          <span className="v-num text-[11.5px] text-[#171717] w-[64px] text-right">{fmtMetric(m).valueText}</span>
        </div>
      ) : (
        <span className="text-[11.5px] text-[#a8a8a8]">⊘ {m.suppression_reason || "not measurable"}</span>
      )}
    </div>
  );
  return (
    <div className="py-2 space-y-1.5">
      <Bar label="Speed at front foot" m={a} />
      <Bar label="Speed at release" m={b} />
    </div>
  );
};

/** Top-down foot-plant corridor: back foot at the bottom, front foot offset left (open) / right (crossed). */
const FootPlantInset: React.FC<{ offset: MetricEstimate; crease: MetricEstimate; label: string }> = ({ offset, crease, label }) => {
  const measured = isMeasured(offset);
  const lim = 30;
  const x = (cm: number) => 100 - (Math.max(-lim, Math.min(lim, cm)) / lim) * 70; // + open → left
  const Foot: React.FC<{ cx: number; cy: number; fill: string; stroke: string }> = ({ cx, cy, fill, stroke }) => (
    <ellipse cx={cx} cy={cy} rx="6" ry="11" fill={fill} stroke={stroke} strokeWidth="1.4" />
  );
  return (
    <svg viewBox="0 0 200 128" className="w-full h-[118px] my-1" role="img" aria-label={`Foot plant from above: ${label}`}>
      <rect x="0" y="0" width="200" height="128" rx="4" fill={SEG.grass} />
      {/* train-track guide lines */}
      <line x1="84" y1="6" x2="84" y2="122" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="4 4" />
      <line x1="116" y1="6" x2="116" y2="122" stroke="#ffffff" strokeWidth="1.5" strokeDasharray="4 4" />
      {isMeasured(crease) && (
        <>
          <line x1="10" y1="26" x2="190" y2="26" stroke="#171717" strokeWidth="1.5" />
          <text x="188" y="22" fontSize="7.5" textAnchor="end" fill="#4d4d4d" fontFamily="var(--font-geist-sans)">
            popping crease
          </text>
        </>
      )}
      <Foot cx={100} cy={102} fill="#ffffff" stroke={SEG.pelvis} />
      <text x="100" y="124" fontSize="7.5" textAnchor="middle" fill="#4d4d4d" fontFamily="var(--font-geist-sans)">
        back foot
      </text>
      {measured ? (
        <>
          {offset.ci_low !== null && offset.ci_high !== null && (
            <rect x={x(offset.ci_high)} y="38" width={Math.max(1, x(offset.ci_low) - x(offset.ci_high))} height="22" rx="4" fill={SEG.pelvis} opacity="0.16" />
          )}
          <Foot cx={x(offset.value!)} cy={49} fill={SEG.pelvisTint} stroke={SEG.pelvis} />
          <path d={`M100 90 L${x(offset.value!)} 62`} stroke={SEG.pelvis} strokeWidth="1" strokeDasharray="2 2" />
          <text x={x(offset.value!)} y="74" fontSize="8" textAnchor="middle" fill="#171717" fontFamily="var(--font-geist-mono)">
            {fmtMetric(offset).valueText}
          </text>
        </>
      ) : (
        <text x="100" y="52" fontSize="8.5" textAnchor="middle" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
          front-foot position not measurable
        </text>
      )}
      <text x="6" y="12" fontSize="7.5" fill="#4d4d4d" fontFamily="var(--font-geist-sans)">
        ← open (train tracks)
      </text>
      <text x="194" y="12" fontSize="7.5" textAnchor="end" fill="#4d4d4d" fontFamily="var(--font-geist-sans)">
        crossed over →
      </text>
    </svg>
  );
};

/** Release point vs stumps drawn to scale (MCC Law 8: 0.711 m tall, 0.2286 m wide). */
const ReleaseInset: React.FC<{ height: MetricEstimate | undefined; lateral: MetricEstimate | null; lateralFromStump: boolean }> = ({ height, lateral, lateralFromStump }) => {
  if (!isMeasured(height)) return null;
  const H = 150;
  const top = 2.6; // metres shown
  const y = (m: number) => H - 14 - (m / top) * (H - 24);
  const latOk = isMeasured(lateral);
  const xm = (m: number) => 100 + (Math.max(-1, Math.min(1, m)) / 1) * 80; // + bowling-arm side → right
  const rx = latOk ? xm(lateral!.value!) : 100;
  return (
    <svg viewBox={`0 0 200 ${H}`} className="w-full h-[140px] my-1" role="img" aria-label="Release point compared with stump height">
      <line x1="10" y1={H - 14} x2="190" y2={H - 14} stroke="#d4d4d4" />
      {[0.5, 1, 1.5, 2, 2.5].map((m) => (
        <g key={m}>
          <line x1="14" x2="186" y1={y(m)} y2={y(m)} stroke="#f0f0f0" />
          <text x="12" y={y(m) + 3} fontSize="7" textAnchor="end" fill="#a8a8a8" fontFamily="var(--font-geist-mono)">
            {m}
          </text>
        </g>
      ))}
      {/* stumps */}
      {[-0.096, 0, 0.096].map((dz) => (
        <rect key={dz} x={100 + (dz / 1) * 80 - 1.2} y={y(0.711)} width="2.4" height={y(0) - y(0.711)} fill="#171717" />
      ))}
      <rect x={100 - 0.12 * 80} y={y(0.711) - 1.6} width={0.24 * 80} height="1.6" fill="#171717" />
      {/* release interval + dot */}
      {height.ci_low !== null && height.ci_high !== null && (
        <rect x={rx - 3} y={y(height.ci_high)} width="6" height={Math.max(1, y(height.ci_low) - y(height.ci_high))} rx="3" fill={SEG.wrist} opacity="0.2" />
      )}
      {latOk && lateral!.ci_low !== null && lateral!.ci_high !== null && (
        <rect x={xm(lateral!.ci_low!)} y={y(height.value!) - 3} width={Math.max(1, xm(lateral!.ci_high!) - xm(lateral!.ci_low!))} height="6" rx="3" fill={SEG.wrist} opacity="0.2" />
      )}
      <circle cx={rx} cy={y(height.value!)} r="5" fill={SEG.wrist} stroke="#ffffff" strokeWidth="1.5" />
      <text x={rx + 8} y={y(height.value!) + 3} fontSize="8.5" fill="#171717" fontFamily="var(--font-geist-mono)">
        {fmtMetric(height).valueText}
      </text>
      <text x="190" y="10" fontSize="7.5" textAnchor="end" fill="#8f8f8f" fontFamily="var(--font-geist-sans)">
        {latOk ? (lateralFromStump ? "out from middle stump, bowling-arm side →" : "out from front ankle, bowling-arm side →") : "sideways position not measured"}
      </text>
    </svg>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Grounded Coach Copilot
// ─────────────────────────────────────────────────────────────────────────────
interface CoachAnswer {
  q: string;
  text: string;
  ok: boolean;
  cited: string[];
  mode: string;
}

const CoachCopilot: React.FC<{ detail: DeliveryDetailResponse; story: CoachingStory | null; metricOf: (k: string) => MetricEstimate | undefined }> = ({ detail, story, metricOf }) => {
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [answers, setAnswers] = useState<CoachAnswer[]>([]);
  const id = detail.delivery.id;

  // Prompt chips are generated ONLY from non-suppressed measurements (and drills grounded in them).
  const chips = useMemo(() => {
    const out: string[] = [];
    (story?.checkpoints ?? [])
      .filter((c) => c.status !== "not_visible" && c.value_text && isMeasured(metricOf(c.metric_key)))
      .sort((a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status])
      .slice(0, 3)
      .forEach((c) => out.push(`What does my ${c.title.toLowerCase()} of ${c.value_text} mean?`));
    (story?.v5_insights ?? [])
      .filter((c) => c.status !== "not_visible" && c.headline_value_text)
      .slice(0, 2)
      .forEach((c) => out.push(`Explain my ${c.title.toLowerCase()} result`));
    (story?.drills ?? []).slice(0, 1).forEach((d) => out.push(`How do I do the "${d.title}" drill?`));
    if (out.length === 0) {
      Object.values(detail.fact_sheet.metrics ?? {})
        .filter((m) => isMeasured(m) && m.key !== "icc_elbow_legality_verdict")
        .slice(0, 3)
        .forEach((m) => out.push(`What does my ${m.label.toLowerCase()} of ${fmtMetric(m).valueText} mean?`));
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [story, detail]);

  const ask = async (question: string) => {
    const qq = question.trim();
    if (!qq || busy) return;
    setBusy(true);
    try {
      const r = await api.askCoach(id, qq);
      setAnswers((a) => [
        {
          q: qq,
          text: r.verification_passed ? r.answer_markdown : "I can't answer that reliably from this clip's measurements, so I won't guess.",
          ok: r.verification_passed,
          cited: r.verification_passed ? r.cited_metric_keys : [],
          mode: r.generator_mode,
        },
        ...a,
      ]);
      setQ("");
    } catch {
      setAnswers((a) => [{ q: qq, text: "The coach is unavailable right now.", ok: false, cited: [], mode: "offline" }, ...a]);
    } finally {
      setBusy(false);
    }
  };

  const labelOf = (k: string) => metricOf(k)?.label ?? k;

  return (
    <section className="v-panel p-3.5">
      <SectionLabel right={<span className="text-[11px] text-[#8f8f8f]">every number is checked</span>}>Coach copilot</SectionLabel>
      <p className="text-[12px] text-[#666666] mt-1 leading-snug">Answers use only this clip's measured facts. Anything not measured is refused, not guessed.</p>
      {chips.length > 0 && (
        <div className="mt-2.5 flex flex-wrap gap-1.5">
          {chips.map((c) => (
            <button key={c} type="button" onClick={() => ask(c)} disabled={busy} className="v-chip hover:shadow-[0_0_0_1px_#c9c9c9] text-left h-auto py-1 leading-snug">
              {c}
            </button>
          ))}
        </div>
      )}
      <form
        className="mt-2.5 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Ask about this ball…" className="v-input h-8 px-2.5 text-[13px] flex-1 min-w-0" aria-label="Question for the coach" />
        <button type="submit" disabled={busy || !q.trim()} className="v-btn-primary h-8 px-3 text-[12.5px]">
          {busy ? "…" : "Ask"}
        </button>
      </form>
      {answers.length > 0 && (
        <ul className="mt-3 space-y-2.5">
          {answers.map((a, i) => (
            <li key={i} className={`rounded-[6px] px-3 py-2.5 ${a.ok ? "bg-[#fafafa] shadow-[0_0_0_1px_#ebebeb]" : "border border-dashed border-[#c9c9c9]"}`}>
              <div className="text-[11.5px] text-[#8f8f8f] mb-1">{a.q}</div>
              <div className="text-[13px] text-[#171717] leading-[1.55] whitespace-pre-wrap">{a.text.replace(/\*\*/g, "").replace(/`/g, "")}</div>
              <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                {a.ok ? (
                  <span className="text-[11px] text-[#297a3a]">✓ Verified against measured facts</span>
                ) : (
                  <span className="text-[11px] text-[#8f8f8f]">⊘ Not answered (would need an unmeasured fact)</span>
                )}
                {a.cited.slice(0, 4).map((k) => (
                  <span key={k} className="h-[18px] px-1.5 rounded-full bg-[#ffffff] shadow-[0_0_0_1px_#ebebeb] text-[10.5px] text-[#4d4d4d] inline-flex items-center">
                    {labelOf(k)}
                  </span>
                ))}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Frame-stamped notes (text + optional browser voice clip)
// ─────────────────────────────────────────────────────────────────────────────
/** Voice clips stay in this browser tab only (object URLs keyed by note id) — nothing is uploaded. */
const VOICE_CLIPS = new Map<string, string>();

const FrameNotes: React.FC<{ detail: DeliveryDetailResponse; notes: CoachNote[]; onNotesChange: (n: CoachNote[]) => void }> = ({ detail, notes, onNotesChange }) => {
  const ph = usePlayhead();
  const id = detail.delivery.id;
  const tel = detail.telemetry;
  const [body, setBody] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [recording, setRecording] = useState(false);
  const [recSecs, setRecSecs] = useState(0);
  const recRef = useRef<{ rec: MediaRecorder; chunks: Blob[]; stream: MediaStream; t0: number; timer: number } | null>(null);
  const canRecord = typeof window !== "undefined" && typeof (window as any).MediaRecorder !== "undefined" && !!navigator.mediaDevices?.getUserMedia;
  const frame = Math.round(ph.frame);

  useEffect(
    () => () => {
      const r = recRef.current;
      if (r) {
        window.clearInterval(r.timer);
        r.stream.getTracks().forEach((t) => t.stop());
      }
    },
    []
  );

  const tsFor = (f: number) => {
    const pts = tel.frame_pts_ms?.[f];
    if (typeof pts === "number" && Number.isFinite(pts)) return pts / 1000;
    const fps = tel.analysis_fps || detail.fact_sheet.audit.fps || 30;
    return f / fps;
  };
  const relMs = (f: number | null) => (f === null ? null : tel.time_rel_ffc_ms?.[f] ?? null);

  const addText = async () => {
    const text = body.trim();
    if (!text) return;
    setErr(null);
    try {
      const r = await api.addNote(id, { frame_idx: frame, timestamp_s: tsFor(frame), tag: "text", body: text });
      onNotesChange([r.note, ...notes]);
      setBody("");
    } catch (e: any) {
      setErr(String(e?.message || "Couldn't save the note."));
    }
  };

  const startRec = async () => {
    setErr(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const rec = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      const f = frame;
      rec.onstop = async () => {
        const r = recRef.current;
        stream.getTracks().forEach((t) => t.stop());
        const secs = r ? Math.max(1, Math.round((performance.now() - r.t0) / 1000)) : 1;
        if (r) window.clearInterval(r.timer);
        recRef.current = null;
        setRecording(false);
        const url = URL.createObjectURL(new Blob(chunks, { type: rec.mimeType || "audio/webm" }));
        try {
          const res = await api.addNote(id, { frame_idx: f, timestamp_s: tsFor(f), tag: "voice", body: `🎙 Voice note · ${secs} s (playable in this browser tab)` });
          VOICE_CLIPS.set(res.note.id, url);
          onNotesChange([res.note, ...notes]);
        } catch (e: any) {
          setErr(String(e?.message || "Couldn't save the voice note."));
        }
      };
      const t0 = performance.now();
      const timer = window.setInterval(() => setRecSecs(Math.round((performance.now() - t0) / 1000)), 250);
      recRef.current = { rec, chunks, stream, t0, timer };
      rec.start();
      setRecSecs(0);
      setRecording(true);
    } catch {
      setErr("Microphone access was blocked, so a voice note can't be recorded.");
    }
  };
  const stopRec = () => recRef.current?.rec.stop();

  const remove = async (nid: string) => {
    try {
      await api.deleteNote(id, nid);
      onNotesChange(notes.filter((n) => n.id !== nid));
      const u = VOICE_CLIPS.get(nid);
      if (u) URL.revokeObjectURL(u);
      VOICE_CLIPS.delete(nid);
    } catch {
      /* ignore */
    }
  };

  const sorted = [...notes].sort((a, b) => (a.frame_idx ?? 1e9) - (b.frame_idx ?? 1e9));
  const curRel = relMs(frame);

  return (
    <section className="v-panel p-3.5">
      <SectionLabel right={<span className="v-num text-[11px] text-[#8f8f8f]">@ {curRel !== null ? `${curRel >= 0 ? "+" : "−"}${Math.abs(Math.round(curRel))} ms` : `frame ${frame + 1}`}</span>}>
        Notes at this moment
      </SectionLabel>
      <textarea
        value={body}
        onChange={(e) => setBody(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) addText();
          e.stopPropagation();
        }}
        rows={2}
        placeholder="Add a note pinned to this frame…"
        className="v-input mt-2 w-full px-2.5 py-2 text-[13px] resize-none"
        aria-label="Note text"
      />
      <div className="mt-2 flex items-center justify-between gap-2">
        {canRecord ? (
          <button type="button" onClick={recording ? stopRec : startRec} className={`h-8 px-3 text-[12.5px] ${recording ? "v-btn-primary" : "v-btn-ghost"}`} aria-pressed={recording}>
            {recording ? `■ Stop · ${recSecs}s` : "🎙 Voice note"}
          </button>
        ) : (
          <span className="text-[11px] text-[#a8a8a8]">Voice notes need a browser with recording support</span>
        )}
        <button type="button" onClick={addText} disabled={!body.trim()} className="v-btn-primary h-8 px-3 text-[12.5px]">
          Pin note
        </button>
      </div>
      {err && <p className="mt-2 text-[12px] text-[#4d4d4d]">{err}</p>}
      {sorted.length > 0 && (
        <ul className="mt-3 divide-y divide-[#f0f0f0] border-t border-[#f0f0f0]">
          {sorted.map((n) => {
            const r = relMs(n.frame_idx);
            const audio = VOICE_CLIPS.get(n.id);
            return (
              <li key={n.id} className="py-2 flex items-start gap-2">
                <button
                  type="button"
                  disabled={n.frame_idx === null}
                  onClick={() => n.frame_idx !== null && ph.seekToEventFrame(n.frame_idx, true)}
                  className="v-num text-[11px] text-[#171717] shrink-0 w-[62px] text-left hover:underline disabled:text-[#a8a8a8] disabled:no-underline"
                  title="Jump to this moment"
                >
                  <span className="inline-block w-1.5 h-1.5 rotate-45 mr-1 align-middle" style={{ background: SEG.tele }} />
                  {r !== null ? `${r >= 0 ? "+" : "−"}${Math.abs(Math.round(r))}ms` : n.frame_idx !== null ? `f${n.frame_idx + 1}` : "—"}
                </button>
                <div className="flex-1 min-w-0">
                  <p className="text-[13px] text-[#171717] leading-snug whitespace-pre-wrap break-words">{n.body}</p>
                  {audio && <audio src={audio} controls className="mt-1 h-7 w-full" />}
                </div>
                <button type="button" onClick={() => remove(n.id)} className="text-[#a8a8a8] hover:text-[#171717] text-[12px] shrink-0" aria-label="Delete note">
                  ✕
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
};
