import React, { useEffect, useMemo, useState } from "react";
import { ArrowUpRight, Camera, Check, ClipboardList } from "lucide-react";
import { api } from "../api";
import { navigate, setCurrentBowlerId, useCurrentBowlerId } from "../router";
import type {
  BodyCheckpoint,
  CheckpointId,
  CheckpointStatus,
  DeliveryCard,
  DeliveryDetailResponse,
  MomentId,
  OverviewResponse,
} from "../types";
import { GuardrailStrip, StatusPill } from "../components/ui/PaperPrimitives";
import { FormSchematic, type PracticePhase, type SchematicSegment } from "../components/practice/FormSchematic";

/**
 * Personalised Practice & Drill Lab (docs/research_app_landscape_v5.md §8.5).
 * Drills come from the backend (`story.drills` + `fact_sheet.corrective_drills`); each card cites its triggering
 * measurement (value + interval) or says it is a general drill. Capture/filming advice is never rendered as a drill.
 * Progression and the daily checklist are user-ticked and persisted in localStorage (never driven by a composite).
 */

type PhaseFilter = "all" | PracticePhase;

const PHASES: Array<{ id: PhaseFilter; label: string }> = [
  { id: "all", label: "All" },
  { id: "gather", label: "Gather" },
  { id: "back_foot_land", label: "Back-foot land" },
  { id: "front_foot_plant", label: "Front-foot plant" },
  { id: "release", label: "Release" },
  { id: "follow_through", label: "Follow-through" },
];

const CHECKPOINT_PHASE: Record<CheckpointId, PracticePhase> = {
  runup_stride: "gather",
  shoulder_alignment: "back_foot_land",
  front_leg_brace: "front_foot_plant",
  trunk_whip: "release",
  side_bend: "release",
};
const CHECKPOINT_SEGMENT: Record<CheckpointId, SchematicSegment> = {
  runup_stride: "pelvis",
  shoulder_alignment: "thorax",
  front_leg_brace: "pelvis",
  trunk_whip: "thorax",
  side_bend: "thorax",
};
const MOMENT_PHASE: Record<MomentId, PracticePhase> = {
  back_foot_land: "back_foot_land",
  front_foot_plant: "front_foot_plant",
  arm_horizontal: "release",
  ball_release: "release",
};

const LEVELS = [
  { n: 1, label: "Walk-through", dose: "3 × 6 slow walk-throughs" },
  { n: 2, label: "Jog-in, 50%", dose: "3 × 6 at about half pace" },
  { n: 3, label: "Short run, 75%", dose: "2 × 6 off a short run" },
  { n: 4, label: "Match intent", dose: "1 × 6 full run — counts toward bowling load" },
];

/** Anything that is filming/capture advice is not a drill (v4 contract bug fix). */
const CAPTURE_RE = /\b(film|filming|camera|fps|frame rate|slow[- ]?mo(tion)?|record(ing)?|tripod|phone|side-on clip|re-?film)\b/i;

interface UnifiedDrill {
  id: string;
  title: string;
  cue: string;
  why: string;
  steps: string[];
  caution: string;
  source: string;
  phase: PracticePhase | null;
  segment: SchematicSegment;
  checkpointId: CheckpointId | null;
  metricKey: string | null;
  frameIdx: number | null;
  general: boolean;
}

const GENERAL_DRILLS: UnifiedDrill[] = [
  {
    id: "gen_gather_rhythm",
    title: "Gather rhythm walk-throughs",
    cue: "A tall, quiet gather: eyes level, front arm reaching up towards the target.",
    why: "Builds a repeatable entry into the delivery stride.",
    steps: ["Walk three steps into a small gather jump.", "Land and hold for two seconds.", "Repeat, keeping the head still."],
    caution: "Stop if any drill causes back pain.",
    source: "General coaching practice",
    phase: "gather",
    segment: "lead",
    checkpointId: null,
    metricKey: null,
    frameIdx: null,
    general: true,
  },
  {
    id: "gen_bfc_freeze",
    title: "Back-foot landing freeze",
    cue: "Land on the back foot and hold: hips and shoulders pointing the same way.",
    why: "Lets you feel hip and shoulder alignment at back-foot landing.",
    steps: ["From a gather, land on the back foot.", "Freeze for two seconds and check your alignment.", "Step through and bowl gently."],
    caution: "Stop if any drill causes back pain.",
    source: "General coaching practice",
    phase: "back_foot_land",
    segment: "thorax",
    checkpointId: null,
    metricKey: null,
    frameIdx: null,
    general: true,
  },
  {
    id: "gen_ffc_brace",
    title: "Front-leg brace step-ins",
    cue: "Plant and push the ground away; stand tall over a firm front leg.",
    why: "Practises a firm front-leg plant that lets the trunk rotate over it.",
    steps: ["Start in the delivery stride position.", "Step onto the front leg and brace.", "Rotate the trunk over the braced leg without bowling."],
    caution: "Stop if any drill causes back or knee pain.",
    source: "General coaching practice",
    phase: "front_foot_plant",
    segment: "pelvis",
    checkpointId: null,
    metricKey: null,
    frameIdx: null,
    general: true,
  },
  {
    id: "gen_release_tall",
    title: "Tall-release line drill",
    cue: "Let the bowling hand pass high over the head, close to vertical.",
    why: "Grooves a high release slot.",
    steps: ["Stand side-on beside a straight line on the ground.", "Swing the bowling arm through slowly over the line.", "Progress to a gentle bowl."],
    caution: "Stop if any drill causes shoulder or back pain.",
    source: "General coaching practice",
    phase: "release",
    segment: "arm",
    checkpointId: null,
    metricKey: null,
    frameIdx: null,
    general: true,
  },
  {
    id: "gen_follow_run",
    title: "Run-through follow-through",
    cue: "Let the arm sweep across and keep running straight for five steps.",
    why: "Spreads braking over several steps instead of stopping abruptly.",
    steps: ["Bowl at gentle pace.", "Keep moving straight down the pitch for five steps.", "Peel away from the pitch at the end."],
    caution: "Stop if any drill causes back pain.",
    source: "General coaching practice",
    phase: "follow_through",
    segment: "pelvis",
    checkpointId: null,
    metricKey: null,
    frameIdx: null,
    general: true,
  },
];

// ── localStorage helpers ──
const LADDER_KEY = "spa.practice.ladder.v1";
const todayKey = () => new Date().toISOString().slice(0, 10);
const CHECK_KEY = () => `spa.practice.checklist.${todayKey()}`;

function readJson<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
function writeJson(key: string, v: unknown) {
  try {
    window.localStorage.setItem(key, JSON.stringify(v));
  } catch {
    /* ignore */
  }
}

const STATUS_RANK: Record<CheckpointStatus, number> = { focus: 0, watch: 1, strong: 3, on_track: 2, not_visible: 4 };

function pickFocusDelivery(cards: DeliveryCard[]): DeliveryCard | null {
  if (!cards.length) return null;
  const sorted = [...cards].sort((x, y) => (y.created_at || "").localeCompare(x.created_at || ""));
  const usable = sorted.filter((d) => d.capture_grade !== "unusable");
  const pool = usable.length ? usable : sorted;
  const has = (s: CheckpointStatus) => pool.find((d) => Object.values(d.checkpoint_statuses || {}).includes(s));
  return has("focus") ?? has("watch") ?? pool[0];
}

function pickFocusCheckpoint(cps: BodyCheckpoint[] | undefined): BodyCheckpoint | null {
  if (!cps?.length) return null;
  const c = [...cps].filter((x) => x.status === "focus" || x.status === "watch").sort((x, y) => STATUS_RANK[x.status] - STATUS_RANK[y.status]);
  return c[0] ?? null;
}

export const PracticeView: React.FC<{ bowlerId: string | null; deliveryId?: string | null }> = ({ bowlerId, deliveryId }) => {
  const storedBowler = useCurrentBowlerId();
  const [ov, setOv] = useState<OverviewResponse | null>(null);
  const [ovError, setOvError] = useState<string | null>(null);
  const [detail, setDetail] = useState<DeliveryDetailResponse | null>(null);
  const [detailError, setDetailError] = useState<string | null>(null);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [phase, setPhase] = useState<PhaseFilter>("all");
  const [ladder, setLadder] = useState<Record<string, number>>(() => readJson(LADDER_KEY, {}));
  const [checks, setChecks] = useState<Record<string, boolean>>(() => readJson(CHECK_KEY(), {}));

  useEffect(() => {
    let alive = true;
    api
      .overview()
      .then((o) => alive && setOv(o))
      .catch((e: Error) => alive && setOvError(e.message || "Could not load"));
    return () => {
      alive = false;
    };
  }, []);

  const bowler = useMemo(() => {
    if (!ov) return null;
    const want = bowlerId ?? storedBowler;
    return (
      ov.bowlers.find((b) => b.id === want) ??
      [...ov.bowlers].sort((x, y) => y.delivery_count - x.delivery_count)[0] ??
      null
    );
  }, [ov, bowlerId, storedBowler]);

  const balls = useMemo(
    () =>
      (ov?.recent_deliveries ?? [])
        .filter((d) => !bowler || d.bowler_id === bowler.id)
        .sort((x, y) => (y.created_at || "").localeCompare(x.created_at || "")),
    [ov, bowler]
  );

  const focusId = deliveryId ?? pickFocusDelivery(balls)?.id ?? null;

  useEffect(() => {
    if (!focusId) {
      setDetail(null);
      return;
    }
    let alive = true;
    setLoadingDetail(true);
    setDetailError(null);
    api
      .getDelivery(focusId)
      .then((d) => alive && setDetail(d))
      .catch((e: Error) => alive && setDetailError(e.message || "Could not load this ball"))
      .finally(() => alive && setLoadingDetail(false));
    return () => {
      alive = false;
    };
  }, [focusId]);

  const story = detail?.story ?? null;
  const fs = detail?.fact_sheet ?? null;
  const focusCp = pickFocusCheckpoint(story?.checkpoints);
  const ballLabel = detail?.delivery?.label ?? balls.find((b) => b.id === focusId)?.label ?? "this ball";

  // ── unified, capture-free drill list ──
  const drills = useMemo<UnifiedDrill[]>(() => {
    const out: UnifiedDrill[] = [];
    const seen = new Set<string>();
    const cps = story?.checkpoints ?? [];
    (story?.drills ?? []).forEach((d) => {
      if (CAPTURE_RE.test(`${d.title} ${d.cue}`)) return;
      seen.add(d.id);
      const cp = d.for_checkpoint_id ? cps.find((c) => c.id === d.for_checkpoint_id) : undefined;
      const phaseFrom: PracticePhase | null = d.moment_id
        ? MOMENT_PHASE[d.moment_id]
        : d.for_checkpoint_id
        ? CHECKPOINT_PHASE[d.for_checkpoint_id]
        : /follow/i.test(d.title + d.cue)
        ? "follow_through"
        : null;
      out.push({
        id: d.id,
        title: d.title,
        cue: d.cue,
        why: d.why_plain,
        steps: d.steps ?? [],
        caution: d.caution,
        source: d.source,
        phase: phaseFrom,
        segment: d.for_checkpoint_id ? CHECKPOINT_SEGMENT[d.for_checkpoint_id] : "arm",
        checkpointId: d.for_checkpoint_id,
        metricKey: cp?.metric_key ?? null,
        frameIdx: d.frame_idx ?? cp?.frame_idx ?? null,
        general: !d.for_checkpoint_id,
      });
    });
    (fs?.corrective_drills ?? []).forEach((d) => {
      if (seen.has(d.id) || CAPTURE_RE.test(`${d.title} ${d.cue_focus}`)) return;
      seen.add(d.id);
      const cp = cps.find((c) => c.metric_key === d.target_limiter_metric);
      out.push({
        id: d.id,
        title: d.title,
        cue: d.cue_focus,
        why: cp ? cp.why_it_matters : "",
        steps: d.setup_steps ?? [],
        caution: d.caution_note,
        source: d.literature_basis,
        phase: cp ? CHECKPOINT_PHASE[cp.id] : null,
        segment: cp ? CHECKPOINT_SEGMENT[cp.id] : "thorax",
        checkpointId: cp?.id ?? null,
        metricKey: d.target_limiter_metric || null,
        frameIdx: cp?.frame_idx ?? null,
        general: false,
      });
    });
    GENERAL_DRILLS.forEach((g) => {
      if (!out.some((d) => d.phase === g.phase && !d.general)) out.push(g);
    });
    // focus-linked drills first
    return out.sort((x, y) => {
      const fx = focusCp && x.checkpointId === focusCp.id ? 0 : x.general ? 2 : 1;
      const fy = focusCp && y.checkpointId === focusCp.id ? 0 : y.general ? 2 : 1;
      return fx - fy;
    });
  }, [story, fs, focusCp]);

  const visible = drills.filter((d) => phase === "all" || d.phase === phase);
  const targetDrill = drills.find((d) => !d.general) ?? drills[0] ?? null;

  const setLevel = (id: string, n: number) => {
    setLadder((prev) => {
      const cur = prev[id] ?? 0;
      const next = { ...prev, [id]: cur === n ? n - 1 : n };
      writeJson(LADDER_KEY, next);
      return next;
    });
  };
  const toggleCheck = (id: string) => {
    setChecks((prev) => {
      const next = { ...prev, [id]: !prev[id] };
      writeJson(CHECK_KEY(), next);
      return next;
    });
  };

  const evidenceFor = (d: UnifiedDrill): { text: string; frame: number | null } | null => {
    const cp = d.checkpointId ? story?.checkpoints.find((c) => c.id === d.checkpointId) : undefined;
    if (cp && cp.value_text) {
      return { text: `${ballLabel} · ${cp.title}: ${cp.value_text}${cp.range_text ? ` (${cp.range_text})` : ""} · ${cp.status_label}`, frame: cp.frame_idx };
    }
    const m = d.metricKey ? fs?.metrics?.[d.metricKey] ?? fs?.v5_metrics?.[d.metricKey] : undefined;
    if (m && !m.suppressed && m.value !== null) {
      const u = m.unit === "deg" ? "°" : m.unit ? ` ${m.unit}` : "";
      const ci = m.ci_low !== null && m.ci_high !== null ? ` [${m.ci_low.toFixed(0)}–${m.ci_high.toFixed(0)}${u.trim()}]` : "";
      return { text: `${ballLabel} · ${m.label}: ${m.value.toFixed(0)}${u}${ci}`, frame: d.frameIdx };
    }
    return null;
  };

  const openWorkbench = (frame: number | null) =>
    focusId && navigate({ name: "delivery", id: focusId, tab: "story", frame: frame ?? null });

  const checklist = [
    { id: "warmup", label: "Warm-up (10 min: jog, dynamic mobility, build-up throws)" },
    { id: "drill", label: targetDrill ? `Target drill: ${targetDrill.title}` : "Target drill" },
    { id: "film", label: "Film 2 check balls", action: () => navigate({ name: "upload", bowler: bowler?.id ?? null, mode: "live" }), cta: "Open camera" },
    { id: "load", label: "Log bowling load", action: () => navigate({ name: "spell", bowler: bowler?.id ?? null }), cta: "Open log" },
    { id: "lumbar", label: "Log lumbar check (back soreness 0–10)", action: () => navigate({ name: "spell", bowler: bowler?.id ?? null }), cta: "Open log" },
  ];
  const doneCount = checklist.filter((c) => checks[c.id]).length;

  // ── render ──
  if (ovError) {
    return <div className="max-w-3xl mx-auto v-panel p-6 text-[14px] text-[#4d4d4d]">Could not load practice data ({ovError}).</div>;
  }

  return (
    <div className="space-y-4 min-w-0">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[24px] font-[500] tracking-[-0.02em] text-[#171717]">Practice</h1>
          <p className="text-[13px] text-[#8f8f8f]">Drills linked to what was measured on your balls.</p>
        </div>
        {ov && ov.bowlers.length > 1 && (
          <div className="flex flex-wrap gap-1.5">
            {ov.bowlers.map((b) => (
              <button
                key={b.id}
                className={`v-chip ${bowler?.id === b.id ? "v-chip-ink" : ""}`}
                onClick={() => {
                  setCurrentBowlerId(b.id);
                  navigate({ name: "practice", bowler: b.id });
                }}
              >
                {b.name}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Focus banner ── */}
      <div className="v-panel p-4 sm:p-5">
        {!ov || loadingDetail ? (
          <p className="text-[13px] text-[#8f8f8f]">Finding this week's focus…</p>
        ) : !focusId ? (
          <div className="text-[14px] text-[#4d4d4d]">
            No analysed balls yet for {bowler?.name ?? "this bowler"}. The general drills below are a safe place to start.
          </div>
        ) : detailError ? (
          <p className="text-[13px] text-[#4d4d4d]">Could not load {ballLabel}: {detailError}</p>
        ) : focusCp ? (
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <span className="v-label">Focus this week · from {ballLabel}</span>
              <div className="mt-1.5 flex flex-wrap items-baseline gap-x-3 gap-y-1">
                <span className="text-[20px] font-[500] tracking-[-0.01em] text-[#171717]">{focusCp.title}</span>
                <StatusPill status={focusCp.status} label={focusCp.status_label} />
              </div>
              <div className="mt-1 flex flex-wrap items-baseline gap-2">
                <span className="v-hero-num text-[32px] text-[#171717]">{focusCp.value_text ?? "—"}</span>
                {focusCp.range_text && <span className="v-num text-[13px] text-[#8f8f8f]">{focusCp.range_text}</span>}
                {focusCp.moment_label && (
                  <span className="text-[12px] text-[#8f8f8f]">
                    · {/^at\s+/i.test(focusCp.moment_label) ? focusCp.moment_label : `At ${focusCp.moment_label}`}
                  </span>
                )}
              </div>
              <p className="mt-2 max-w-[640px] text-[13px] leading-[1.55] text-[#4d4d4d]">{focusCp.plain_meaning}</p>
            </div>
            <button
              className="v-btn-primary h-9 px-3.5 text-[13px] inline-flex items-center gap-1.5 shrink-0"
              onClick={() => openWorkbench(focusCp.frame_idx)}
            >
              See it on {ballLabel} <ArrowUpRight className="w-3.5 h-3.5" />
            </button>
          </div>
        ) : (
          <div className="flex flex-wrap items-center justify-between gap-3">
            <div>
              <span className="v-label">Focus this week · from {ballLabel}</span>
              <p className="mt-1 text-[14px] text-[#171717]">
                Nothing measured on this ball needs a focus. Keep grooving with the drills below.
              </p>
            </div>
            <button className="v-btn-ghost h-9 px-3 text-[13px]" onClick={() => openWorkbench(null)}>
              Open {ballLabel}
            </button>
          </div>
        )}
        {balls.length > 1 && (
          <div className="mt-4 flex flex-wrap items-center gap-1.5">
            <span className="text-[12px] text-[#8f8f8f] mr-1">Ball:</span>
            {balls.slice(0, 10).map((d) => (
              <button
                key={d.id}
                className={`v-chip ${d.id === focusId ? "v-chip-ink" : ""}`}
                onClick={() => navigate({ name: "practice", bowler: bowler?.id ?? null, delivery: d.id })}
                title={d.top_takeaway_title}
              >
                {d.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* ── Phase filter ── */}
      <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Filter drills by phase">
        {PHASES.map((p) => {
          const count = p.id === "all" ? drills.length : drills.filter((d) => d.phase === p.id).length;
          return (
            <button
              key={p.id}
              role="tab"
              aria-selected={phase === p.id}
              className={`v-chip h-8 px-3 ${phase === p.id ? "v-chip-ink" : ""}`}
              onClick={() => setPhase(p.id)}
            >
              {p.label}
              <span className={`v-num text-[11px] ${phase === p.id ? "text-white/70" : "text-[#8f8f8f]"}`}>{count}</span>
            </button>
          );
        })}
      </div>

      <div className="grid gap-4 lg:grid-cols-[1fr_320px] items-start">
        {/* ── Drill cards ── */}
        <div className="grid gap-3 md:grid-cols-2 min-w-0">
          {visible.length === 0 && (
            <div className="v-panel p-5 text-[13px] text-[#8f8f8f] md:col-span-2">No drills for this phase.</div>
          )}
          {visible.map((d) => {
            const ev = evidenceFor(d);
            const done = ladder[d.id] ?? 0;
            const current = Math.min(4, done + 1);
            return (
              <article key={d.id} className="v-panel p-4 flex flex-col gap-3 min-w-0">
                <div className="flex gap-3">
                  <div className="shrink-0 rounded-md bg-[#fafafa] v-hairline p-1">
                    <FormSchematic phase={d.phase ?? "all"} segment={d.segment} size={96} />
                  </div>
                  <div className="min-w-0">
                    <span className="v-label">{PHASES.find((p) => p.id === d.phase)?.label ?? "Whole action"}</span>
                    <h3 className="mt-0.5 text-[15px] font-[500] leading-[1.3] text-[#171717]">{d.title}</h3>
                    <p className="mt-1 text-[13px] leading-[1.5] text-[#171717]">“{d.cue}”</p>
                  </div>
                </div>

                <div className="text-[12px] leading-[1.5]">
                  {ev ? (
                    <button
                      className="text-left text-[#171717] underline decoration-[#c9c9c9] underline-offset-2 hover:decoration-[#171717]"
                      onClick={() => openWorkbench(ev.frame)}
                    >
                      Why: {ev.text} →
                    </button>
                  ) : (
                    <span className="text-[#8f8f8f]">General technique drill (not triggered by a measurement)</span>
                  )}
                  {d.why && <p className="mt-1 text-[#4d4d4d]">{d.why}</p>}
                </div>

                {d.steps.length > 0 && (
                  <ol className="list-decimal pl-5 text-[12.5px] leading-[1.55] text-[#4d4d4d] space-y-0.5">
                    {d.steps.map((s, i) => (
                      <li key={i}>{s}</li>
                    ))}
                  </ol>
                )}

                <div>
                  <div className="flex items-baseline justify-between">
                    <span className="v-label">Progression</span>
                    <span className="text-[11px] text-[#8f8f8f]">Dose: {LEVELS[current - 1].dose}</span>
                  </div>
                  <div className="mt-1.5 grid grid-cols-4 gap-1">
                    {LEVELS.map((l) => {
                      const ticked = done >= l.n;
                      const isCur = !ticked && l.n === current;
                      return (
                        <button
                          key={l.n}
                          onClick={() => setLevel(d.id, l.n)}
                          aria-pressed={ticked}
                          className={`rounded-md px-1.5 py-1.5 text-left text-[11px] leading-[1.25] transition-colors ${
                            ticked
                              ? "bg-[#171717] text-white"
                              : isCur
                              ? "bg-white text-[#171717] shadow-[0_0_0_1px_#171717]"
                              : "bg-[#fafafa] text-[#8f8f8f] shadow-[0_0_0_1px_#ebebeb]"
                          }`}
                          title={l.dose}
                        >
                          <span className="v-num block text-[10px] opacity-80">
                            L{l.n} {ticked ? "✓" : isCur ? "●" : "○"}
                          </span>
                          {l.label}
                        </button>
                      );
                    })}
                  </div>
                  <p className="mt-1.5 text-[11px] leading-[1.45] text-[#8f8f8f]">
                    Tick a level when it feels easy. Drill reps are separate from bowling load unless you bowl at full intensity (L4) — then log them.
                  </p>
                </div>

                <div className="mt-auto pt-2 border-t border-[#ebebeb] text-[11px] leading-[1.45] text-[#8f8f8f]">
                  {d.caution && <p className="text-[#4d4d4d]">{d.caution}</p>}
                  {d.source && <p className="mt-0.5">Source: {d.source}</p>}
                </div>
              </article>
            );
          })}
        </div>

        {/* ── Today's checklist ── */}
        <aside className="v-panel p-4 lg:sticky lg:top-4">
          <div className="flex items-center justify-between">
            <span className="v-label inline-flex items-center gap-1.5">
              <ClipboardList className="w-3.5 h-3.5" /> Today's session
            </span>
            <span className="v-num text-[12px] text-[#4d4d4d]">
              {doneCount}/{checklist.length}
            </span>
          </div>
          <div className="mt-2 h-1 rounded-full bg-[#ebebeb] overflow-hidden">
            <div className="h-full bg-[#297a3a] transition-[width] duration-200" style={{ width: `${(doneCount / checklist.length) * 100}%` }} />
          </div>
          <ul className="mt-3 space-y-2">
            {checklist.map((c) => (
              <li key={c.id} className="flex items-start gap-2.5">
                <button
                  onClick={() => toggleCheck(c.id)}
                  aria-pressed={!!checks[c.id]}
                  aria-label={`Mark "${c.label}" ${checks[c.id] ? "not done" : "done"}`}
                  className={`mt-0.5 w-5 h-5 shrink-0 rounded flex items-center justify-center ${
                    checks[c.id] ? "bg-[#297a3a]" : "bg-white shadow-[0_0_0_1px_#c9c9c9]"
                  }`}
                >
                  {checks[c.id] && <Check className="w-3.5 h-3.5 text-white" />}
                </button>
                <div className="min-w-0 flex-1">
                  <span className={`text-[13px] leading-[1.45] ${checks[c.id] ? "text-[#8f8f8f] line-through" : "text-[#171717]"}`}>
                    {c.label}
                  </span>
                  {c.action && (
                    <button
                      className="ml-2 text-[12px] text-[#171717] underline decoration-[#c9c9c9] underline-offset-2 inline-flex items-center gap-1"
                      onClick={c.action}
                    >
                      {c.id === "film" && <Camera className="w-3 h-3" />}
                      {c.cta}
                    </button>
                  )}
                </div>
              </li>
            ))}
          </ul>
          <p className="mt-3 text-[11px] leading-[1.45] text-[#8f8f8f]">
            Check balls are re-measured by the full analysis pipeline; progress is judged only when a measured interval moves, never by a score.
          </p>
        </aside>
      </div>

      <GuardrailStrip />
    </div>
  );
};

export default PracticeView;
