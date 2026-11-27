import React, { useEffect, useMemo, useState } from "react";
import { ArrowDown, ArrowLeftRight, ArrowRight, ArrowUp, Info } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type {
  BodyCheckpoint,
  CheckpointStatus,
  CompareSide,
  CompareStory,
  DeliveryCard,
  DeliveryDetailResponse,
  MomentId,
  OverviewResponse,
} from "../types";
import {
  Card,
  CheckpointDots,
  EmptyState,
  NoticeFooter,
  SectionTitle,
  Spinner,
  StatusBadge,
  ValueWithRange,
  formatDate,
} from "../components/story/ui";
import { DualComparePlayer, type DualMode, type DualMoment, type DualSide } from "../components/story/DualComparePlayer";

const MOMENT_ORDER: MomentId[] = ["back_foot_land", "front_foot_plant", "arm_horizontal", "ball_release"];
const MOMENT_FALLBACK_LABEL: Record<MomentId, string> = {
  back_foot_land: "Back foot lands",
  front_foot_plant: "Front foot plants",
  arm_horizontal: "Arm level",
  ball_release: "Ball release",
};

type Diff = CompareStory["rows"][number]["difference"];

/**
 * Default pair (older = A, newer = B). Skips unusable clips, then prefers the newest pair from the
 * same bowler with the same camera view, then the same bowler, then anything.
 */
function defaultPair(list: DeliveryCard[], keepA: string | null, keepB: string | null): { a: string; b: string } | null {
  const sorted = [...list].sort((x, y) => (y.created_at || "").localeCompare(x.created_at || ""));
  const byId = (id: string | null) => (id ? sorted.find((d) => d.id === id) ?? null : null);
  const usable = sorted.filter((d) => d.capture_grade !== "unusable");
  const tiers: Array<(x: DeliveryCard, y: DeliveryCard) => boolean> = [
    (x, y) => x.bowler_id === y.bowler_id && x.view_label === y.view_label,
    (x, y) => x.bowler_id === y.bowler_id,
    () => true,
  ];
  const order = (x: DeliveryCard, y: DeliveryCard) =>
    (x.created_at || "") <= (y.created_at || "") ? { a: x.id, b: y.id } : { a: y.id, b: x.id };

  const fixed = byId(keepB) ?? byId(keepA);
  if (fixed) {
    const pool = usable.filter((d) => d.id !== fixed.id);
    const fallback = sorted.filter((d) => d.id !== fixed.id);
    for (const tier of tiers) {
      const other = pool.find((d) => tier(fixed, d));
      if (other) return keepB && byId(keepB) ? { a: other.id, b: fixed.id } : { a: fixed.id, b: other.id };
    }
    if (!fallback.length) return null;
    return keepB && byId(keepB) ? { a: fallback[0].id, b: fixed.id } : { a: fixed.id, b: fallback[0].id };
  }

  const rank = (d: DeliveryCard) => (d.capture_grade === "great" ? 3 : d.capture_grade === "partial" ? 2 : d.capture_grade === "limited" ? 1 : 0);
  for (const pool of [usable, sorted]) {
    for (const tier of tiers) {
      // Best qualifying pair: clearest worst-view first (more checkpoints comparable), then newest.
      let best: { x: DeliveryCard; y: DeliveryCard; q: number; i: number } | null = null;
      for (let i = 0; i < pool.length; i++) {
        for (let j = i + 1; j < pool.length; j++) {
          if (!tier(pool[i], pool[j])) continue;
          const q = Math.min(rank(pool[i]), rank(pool[j]));
          if (!best || q > best.q) best = { x: pool[i], y: pool[j], q, i };
        }
      }
      if (best) return order(best.x, best.y);
    }
  }
  return null;
}

const ChangePill: React.FC<{ diff: Diff; text: string }> = ({ diff, text }) => {
  const cls =
    diff === "clearly_higher" || diff === "clearly_lower"
      ? "text-[#171717] border-[#171717] font-[500]"
      : diff === "within_noise"
      ? "text-[#8f8f8f] border-[#ebebeb]"
      : "text-[#8f8f8f] border-dashed border-[#c9c9c9] bg-[#fafafa]";
  return (
    <span className={`inline-flex items-center gap-1 min-h-6 px-2.5 py-0.5 rounded-full border text-[12px] leading-snug ${cls}`}>
      {diff === "clearly_higher" && <ArrowUp className="w-3 h-3 shrink-0" aria-hidden />}
      {diff === "clearly_lower" && <ArrowDown className="w-3 h-3 shrink-0" aria-hidden />}
      {text}
    </span>
  );
};

const SideCell: React.FC<{ tag: string; side: CompareSide; checkpoint: BodyCheckpoint | undefined }> = ({ tag, side, checkpoint }) => {
  // Use the story's own label/unlock reason only when it agrees with the backend compare status.
  const sameStatus = checkpoint && checkpoint.status === side.status;
  return (
    <div className="flex items-center justify-between md:justify-start md:flex-col md:items-start gap-2 md:gap-1.5 min-w-0">
      <span className="md:hidden text-[12px] text-[#8f8f8f] w-12 shrink-0">{tag}</span>
      <StatusBadge
        status={side.status as CheckpointStatus}
        label={sameStatus ? checkpoint!.status_label : null}
        unlockReason={sameStatus ? checkpoint!.unlock_reason : null}
        compact
      />
      <span className="ml-auto md:ml-0">
        <ValueWithRange value={side.value_text} range={side.range_text} size="sm" inline />
      </span>
    </div>
  );
};

const PickerCard: React.FC<{
  tag: string;
  selectedId: string | null;
  otherId: string | null;
  options: DeliveryCard[];
  onPick: (id: string) => void;
}> = ({ tag, selectedId, otherId, options, onPick }) => {
  const sel = options.find((d) => d.id === selectedId) ?? null;
  const groups = useMemo(() => {
    const m = new Map<string, { name: string; items: DeliveryCard[] }>();
    options.forEach((d) => {
      const g = m.get(d.bowler_id) ?? { name: d.bowler_name, items: [] };
      g.items.push(d);
      m.set(d.bowler_id, g);
    });
    return [...m.values()];
  }, [options]);
  return (
    <Card className="p-3 md:p-4 flex gap-3 min-w-0">
      <div className="w-[72px] h-[72px] md:w-[88px] md:h-[88px] rounded-[6px] bg-[#f4f4f5] overflow-hidden shrink-0">
        {sel?.thumbnail_url && <img src={sel.thumbnail_url} alt="" className="w-full h-full object-cover" loading="lazy" />}
      </div>
      <div className="flex-1 min-w-0 flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2">
          <span className="text-[12px] text-[#8f8f8f]">{tag}</span>
          {sel && (
            <button
              type="button"
              className="text-[12px] text-[#4d4d4d] hover:text-[#171717] inline-flex items-center gap-1"
              onClick={() => navigate({ name: "delivery", id: sel.id, tab: "story" })}
            >
              Open story <ArrowRight className="w-3 h-3" />
            </button>
          )}
        </div>
        <select
          className="v-input h-8 px-2 text-[13px] w-full min-w-0"
          value={selectedId ?? ""}
          onChange={(e) => e.target.value && onPick(e.target.value)}
          aria-label={`Choose ${tag}`}
        >
          {!sel && <option value="">Choose a ball…</option>}
          {groups.map((g) => (
            <optgroup key={g.name} label={g.name}>
              {g.items.map((d) => (
                <option key={d.id} value={d.id} disabled={d.id === otherId}>
                  {d.label || "Delivery"} · {formatDate(d.created_at)}
                </option>
              ))}
            </optgroup>
          ))}
        </select>
        {sel && (
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1 text-[12px] text-[#4d4d4d] min-w-0">
            <span className="truncate">{sel.bowler_name}</span>
            <span className="text-[#c9c9c9]">·</span>
            <span>{sel.view_label}</span>
            <span className="ml-auto">
              <CheckpointDots statuses={sel.checkpoint_statuses} />
            </span>
          </div>
        )}
      </div>
    </Card>
  );
};

export const CompareView: React.FC<{ a: string | null; b: string | null }> = ({ a, b }) => {
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [overviewErr, setOverviewErr] = useState<string | null>(null);
  const [cmp, setCmp] = useState<CompareStory | null>(null);
  const [cmpErr, setCmpErr] = useState<string | null>(null);
  const [cmpLoading, setCmpLoading] = useState(false);
  const [detail, setDetail] = useState<{ a: DeliveryDetailResponse | null; b: DeliveryDetailResponse | null }>({ a: null, b: null });
  const [detailErr, setDetailErr] = useState<{ a: string | null; b: string | null }>({ a: null, b: null });
  const [mode, setMode] = useState<DualMode>("side");

  useEffect(() => {
    let cancelled = false;
    api
      .overview()
      .then((o) => !cancelled && setOverview(o))
      .catch((e) => !cancelled && setOverviewErr(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const options = overview?.recent_deliveries ?? [];

  // Fill in missing picks with the latest two balls by the same bowler.
  useEffect(() => {
    if (!overview || (a && b && a !== b)) return;
    const pair = defaultPair(options, a, a === b ? null : b);
    if (pair) navigate({ name: "compare", a: pair.a, b: pair.b });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [overview, a, b]);

  const ready = !!a && !!b && a !== b;

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setCmp(null);
    setCmpErr(null);
    setCmpLoading(true);
    api
      .compare(a!, b!)
      .then((c) => !cancelled && setCmp(c))
      .catch((e) => !cancelled && setCmpErr(String(e?.message || e)))
      .finally(() => !cancelled && setCmpLoading(false));
    return () => {
      cancelled = true;
    };
  }, [a, b, ready]);

  useEffect(() => {
    if (!ready) return;
    let cancelled = false;
    setDetail({ a: null, b: null });
    setDetailErr({ a: null, b: null });
    (["a", "b"] as const).forEach((k) => {
      const id = k === "a" ? a! : b!;
      api
        .getDelivery(id)
        .then((d) => !cancelled && setDetail((p) => ({ ...p, [k]: d })))
        .catch((e) => !cancelled && setDetailErr((p) => ({ ...p, [k]: String(e?.message || e) })));
    });
    return () => {
      cancelled = true;
    };
  }, [a, b, ready]);

  const pick = (which: "a" | "b", id: string) => {
    navigate({ name: "compare", a: which === "a" ? id : a, b: which === "b" ? id : b });
  };

  const cardA = cmp?.a ?? options.find((d) => d.id === a) ?? null;
  const cardB = cmp?.b ?? options.find((d) => d.id === b) ?? null;

  const moments: DualMoment[] = useMemo(() => {
    const ka = detail.a?.story?.key_moments ?? [];
    const kb = detail.b?.story?.key_moments ?? [];
    return MOMENT_ORDER.filter((id) => ka.some((m) => m.id === id) || kb.some((m) => m.id === id)).map((id) => {
      const ma = ka.find((m) => m.id === id);
      const mb = kb.find((m) => m.id === id);
      return { id, label: ma?.label || mb?.label || MOMENT_FALLBACK_LABEL[id], a: ma?.frame_idx ?? null, b: mb?.frame_idx ?? null };
    });
  }, [detail.a, detail.b]);

  const syncFrames = useMemo(() => {
    const ffp = moments.find((m) => m.id === "front_foot_plant");
    return {
      a: cmp?.sync_frames?.a ?? ffp?.a ?? null,
      b: cmp?.sync_frames?.b ?? ffp?.b ?? null,
    };
  }, [cmp, moments]);

  const toSide = (tag: string, d: DeliveryDetailResponse | null, card: DeliveryCard | null): DualSide | null => {
    if (!d || !d.telemetry || !d.delivery?.video_proxy_url) return null;
    return {
      id: d.delivery.id,
      tag,
      caption: card ? formatDate(card.created_at) : undefined,
      videoUrl: d.delivery.video_proxy_url,
      telemetry: d.telemetry,
      bowlingArm: d.fact_sheet?.bowling_arm ?? "right",
    };
  };
  const sideA = useMemo(() => toSide("Ball A", detail.a, cardA), [detail.a, cardA?.created_at]); // eslint-disable-line react-hooks/exhaustive-deps
  const sideB = useMemo(() => toSide("Ball B", detail.b, cardB), [detail.b, cardB?.created_at]); // eslint-disable-line react-hooks/exhaustive-deps

  const notices = detail.a?.story?.notices ?? detail.b?.story?.notices ?? null;
  const viewsDiffer = !!cardA && !!cardB && cardA.view_label !== cardB.view_label;

  // ── Page states ────────────────────────────────────────────────────────────
  if (overviewErr && !ready) {
    return (
      <div className="max-w-[1200px] mx-auto px-4 md:px-6 py-8">
        <EmptyState title="Couldn't load your deliveries" body={overviewErr} action={<button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => window.location.reload()}>Try again</button>} />
      </div>
    );
  }
  if (overview && options.length < 2 && !ready) {
    return (
      <div className="max-w-[1200px] mx-auto px-4 md:px-6 py-8">
        <EmptyState
          title="You need two balls to compare"
          body="Analyse at least two deliveries and they'll line up here side by side, matched at the moment the front foot plants."
          action={
            <>
              <button className="v-btn-primary h-9 px-4 text-[13px]" onClick={() => navigate({ name: "new", bowler: null })}>
                New analysis
              </button>
              <button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => navigate({ name: "home" })}>
                Back to home
              </button>
            </>
          }
        />
      </div>
    );
  }

  return (
    <div className="max-w-[1200px] mx-auto px-4 md:px-6 py-6 md:py-8 flex flex-col gap-5">
      <header className="flex flex-col gap-1">
        <h1 className="text-[24px] md:text-[28px] font-[500] tracking-[-0.03em] text-[#171717]">Compare 2 balls</h1>
        <p className="text-[13px] text-[#4d4d4d]">Both videos are lined up at the moment the front foot plants. Changes only count when the ranges don't overlap.</p>
      </header>

      {/* Pickers */}
      <div className="grid gap-3 md:grid-cols-[1fr_auto_1fr] items-stretch">
        <PickerCard tag="Ball A" selectedId={a} otherId={b} options={options} onPick={(id) => pick("a", id)} />
        <div className="flex items-center justify-center">
          <button
            type="button"
            className="v-btn-ghost h-8 w-8 inline-flex items-center justify-center"
            title="Swap Ball A and Ball B"
            aria-label="Swap Ball A and Ball B"
            disabled={!ready}
            onClick={() => navigate({ name: "compare", a: b, b: a })}
          >
            <ArrowLeftRight className="w-3.5 h-3.5" />
          </button>
        </div>
        <PickerCard tag="Ball B" selectedId={b} otherId={a} options={options} onPick={(id) => pick("b", id)} />
      </div>

      {!overview && !overviewErr && !ready && (
        <div className="flex items-center gap-2 text-[13px] text-[#4d4d4d]">
          <Spinner /> Loading your deliveries…
        </div>
      )}

      {ready && (
        <>
          {/* Player */}
          <Card className="p-3 md:p-4">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <SectionTitle className="!mb-0">Watch them together</SectionTitle>
              <div className="inline-flex p-0.5 rounded-[6px] bg-[#f4f4f5]" role="tablist" aria-label="View mode">
                {(
                  [
                    ["side", "Side by side"],
                    ["overlay", "Overlay"],
                  ] as Array<[DualMode, string]>
                ).map(([m, label]) => (
                  <button
                    key={m}
                    type="button"
                    role="tab"
                    aria-selected={mode === m}
                    onClick={() => setMode(m)}
                    className={`h-7 px-3 rounded-[5px] text-[12.5px] transition-colors duration-150 ${
                      mode === m ? "bg-white text-[#171717] font-[500] shadow-[0_0_0_1px_#ebebeb]" : "text-[#4d4d4d] hover:text-[#171717]"
                    }`}
                  >
                    {label}
                  </button>
                ))}
              </div>
            </div>
            {mode === "overlay" && (
              <p className="text-[12px] text-[#4d4d4d] mb-2">
                Ball B's body outline is drawn as a dashed line over Ball A's video, matched in position and size at the moment the front foot plants.
                {viewsDiffer && " The two balls were filmed from different angles, so the outlines won't line up closely."}
              </p>
            )}
            {sideA && sideB ? (
              <DualComparePlayer a={sideA} b={sideB} syncFrames={syncFrames} moments={moments} mode={mode} speed={0.5} />
            ) : detailErr.a || detailErr.b ? (
              <div className="py-10 text-center text-[13px] text-[#4d4d4d]">
                The videos couldn't be loaded right now. The comparison below still applies.
              </div>
            ) : (
              <div className="py-10 flex items-center justify-center gap-2 text-[13px] text-[#4d4d4d]">
                <Spinner /> Loading videos…
              </div>
            )}
          </Card>

          {/* Summary + table */}
          <Card className="p-4 md:p-5">
            <SectionTitle hint={cmp && !cmp.comparable ? "Limited comparison" : undefined}>What changed</SectionTitle>
            {cmpLoading && (
              <div className="flex items-center gap-2 text-[13px] text-[#4d4d4d] py-4">
                <Spinner /> Comparing the two balls…
              </div>
            )}
            {cmpErr && !cmpLoading && (
              <div className="text-[13px] text-[#4d4d4d] py-2">
                The side-by-side measurements aren't available right now. Try again in a moment.
                <span className="block mt-1 text-[12px] text-[#8f8f8f] break-words">{cmpErr}</span>
              </div>
            )}
            {cmp && (
              <>
                <p className="text-[15px] leading-relaxed text-[#171717]">{cmp.summary}</p>
                {cmp.caveat_plain && (
                  <p className="mt-2 flex gap-2 items-start text-[12.5px] leading-[1.55] text-[#4d4d4d]">
                    <Info className="w-4 h-4 mt-[1px] text-[#666666] shrink-0" />
                    <span>{cmp.caveat_plain}</span>
                  </p>
                )}

                <div className="mt-4 border-t border-[#ebebeb]">
                  <div className="hidden md:grid grid-cols-[1.1fr_1fr_1fr_1.2fr] gap-4 py-2 text-[12px] text-[#8f8f8f] border-b border-[#ebebeb]">
                    <span>Checkpoint</span>
                    <span>Ball A</span>
                    <span>Ball B</span>
                    <span>Change</span>
                  </div>
                  {cmp.rows.map((row) => {
                    const cpA = detail.a?.story?.checkpoints.find((c) => c.id === row.checkpoint_id);
                    const cpB = detail.b?.story?.checkpoints.find((c) => c.id === row.checkpoint_id);
                    return (
                      <div
                        key={row.checkpoint_id}
                        className="grid gap-2 md:gap-4 md:grid-cols-[1.1fr_1fr_1fr_1.2fr] py-3 border-b border-[#ebebeb] last:border-b-0 md:items-center"
                      >
                        <div className="text-[14px] font-[500] text-[#171717]">{row.title}</div>
                        <SideCell tag="Ball A" side={row.a} checkpoint={cpA} />
                        <SideCell tag="Ball B" side={row.b} checkpoint={cpB} />
                        <div className="flex items-center gap-2">
                          <span className="md:hidden text-[12px] text-[#8f8f8f] w-12 shrink-0">Change</span>
                          <ChangePill diff={row.difference} text={row.difference_plain} />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </>
            )}
          </Card>

          <div className="flex flex-wrap gap-2">
            {cardB && (
              <button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => navigate({ name: "progress", bowler: cardB.bowler_id })}>
                See {cardB.bowler_name}'s progress
              </button>
            )}
          </div>
        </>
      )}

      <NoticeFooter notices={notices} />
    </div>
  );
};

export default CompareView;
