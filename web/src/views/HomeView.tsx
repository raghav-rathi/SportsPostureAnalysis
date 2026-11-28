import React, { Suspense, lazy, useEffect, useMemo, useRef, useState } from "react";
import { ArrowRight, Camera, Dumbbell, GitCompareArrows, ListChecks, Scissors, Upload } from "lucide-react";
import { api } from "../api";
import { navigate, useCurrentBowlerId } from "../router";
import type { DeliveryDetailResponse, OverviewResponse, SpellAnalyticsResponse, WorkloadResponse } from "../types";
import { EmptyState, Spinner } from "../components/story/ui";
import { DeliveryCardTile, useDeliveryDetail, useOverview, type Playhead } from "../components/home/shared";
import { HeroVideo } from "../components/home/HeroVideo";
import { MiniBullseye } from "../components/home/MiniBullseye";
import type { VirtualCameraPreset } from "../workbench/Stage/PitchStage3D";
import { SEG, SEQUENCE_LANES } from "../theme";
import { GuardrailStrip } from "../components/ui/PaperPrimitives";

// Re-exported for legacy imports (DeliveriesView v4 and others).
export { useOverview, DeliveryCardTile };

const PitchStage3D = lazy(() => import("../workbench/Stage/PitchStage3D"));

/** Preferred hero sources, in order. Only those present in /api/overview samples are offered. */
const HERO_SOURCES: Array<{ id: string; label: string; sub: string; defaultCam: VirtualCameraPreset }> = [
  { id: "del_real_steyn", label: "Dale Steyn", sub: "Real net video · behind", defaultCam: "behind" },
  { id: "del_real_indoor", label: "Indoor nets", sub: "Real slow-mo · front-on", defaultCam: "behind" },
  { id: "del_demo_dual_240", label: "Dual-Phone 240 fps", sub: "Full 3D sequence · 4 lanes", defaultCam: "side" },
  { id: "del_demo_side_240", label: "Side-on 240 fps", sub: "Studio demo · calibrated", defaultCam: "side" },
];

function defaultCamForHero(id: string | null, detail?: DeliveryDetailResponse | null): VirtualCameraPreset {
  const known = HERO_SOURCES.find((s) => s.id === id);
  if (known) return known.defaultCam;
  const view = detail?.fact_sheet?.calibration?.view_class;
  if (view === "rear_on" || view === "front_on") return "behind";
  return "side";
}

function useSnapshot(overview: OverviewResponse | null, bowlerId: string | null) {
  const [workload, setWorkload] = useState<WorkloadResponse | null>(null);
  const [spell, setSpell] = useState<SpellAnalyticsResponse | null>(null);
  useEffect(() => {
    if (!bowlerId) return;
    let alive = true;
    api
      .workload(bowlerId)
      .then((w) => alive && setWorkload(w))
      .catch(() => undefined);
    api
      .spellAnalytics(bowlerId, overview?.spell_snapshot?.latest_session_id ?? null)
      .then((s) => alive && setSpell(s))
      .catch(() => undefined);
    return () => {
      alive = false;
    };
  }, [bowlerId, overview?.spell_snapshot?.latest_session_id]);
  return { workload, spell };
}

function daysBowledLast7(w: WorkloadResponse | null): number | null {
  if (!w) return null;
  if (w.daily_series_28d?.length) return w.daily_series_28d.slice(-7).filter((d) => d.balls > 0).length;
  const now = new Date();
  const cutoff = new Date(now.getFullYear(), now.getMonth(), now.getDate() - 6).toISOString().slice(0, 10);
  return new Set(w.logs.filter((l) => l.log_date >= cutoff && l.deliveries_count > 0).map((l) => l.log_date)).size;
}

const SnapshotTile: React.FC<{ label: string; value: React.ReactNode; sub: React.ReactNode; muted?: boolean }> = ({
  label,
  value,
  sub,
  muted,
}) => (
  <div className="v-panel p-4 flex flex-col gap-2 min-h-[132px]">
    <span className="v-label">{label}</span>
    <span className={`v-hero-num text-[34px] ${muted ? "text-[#c9c9c9]" : "text-[#171717]"}`}>{value}</span>
    <span className="text-[12.5px] text-[#666666] leading-snug mt-auto">{sub}</span>
  </div>
);

const ActionButton: React.FC<{ icon: React.ReactNode; label: string; onClick: () => void }> = ({ icon, label, onClick }) => (
  <button
    onClick={onClick}
    className="v-panel h-12 px-4 flex items-center gap-2.5 text-[13.5px] font-[500] text-[#171717] hover:shadow-[0_0_0_1px_rgba(0,0,0,0.2)] transition-shadow whitespace-nowrap"
  >
    <span className="text-[#4d4d4d]">{icon}</span>
    {label}
  </button>
);

/** Small honest sequence preview: real lanes when the backend measured them, ghosted otherwise. */
const SequencePreview: React.FC<{ detail: DeliveryDetailResponse | null }> = ({ detail }) => {
  const ss = detail?.telemetry.sequence_series;
  const t = detail?.telemetry.time_rel_ffc_ms ?? [];
  const lanes = SEQUENCE_LANES.map((l) => ({ ...l, data: (ss as Record<string, Array<number | null>> | undefined)?.[l.seriesKey] }));
  const measured = lanes.some((l) => l.data && l.data.some((v) => v !== null && Number.isFinite(v as number)));
  const W = 260;
  const H = 96;
  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className={`w-full ${measured ? "" : "v-ghosted"}`} style={{ height: H }}>
        <rect width={W} height={H} rx={6} fill="#fafafa" />
        {(measured ? lanes : lanes.map((l, i) => ({ ...l, data: Array.from({ length: 40 }, (_, k) => Math.exp(-(((k - 14 - i * 5) / 4) ** 2))) }))).map(
          (l, li) => {
            const d = (l.data ?? []) as Array<number | null>;
            const vals = d.filter((v): v is number => v !== null && Number.isFinite(v));
            const max = Math.max(...vals.map(Math.abs), 1e-6);
            const n = d.length;
            const path = d
              .map((v, k) => (v === null ? null : `${(k / Math.max(n - 1, 1)) * W},${H - 8 - (Math.abs(v) / max) * (H - 20) * 0.92}`))
              .filter(Boolean)
              .join(" L ");
            return path ? <path key={li} d={`M ${path}`} fill="none" stroke={l.color} strokeWidth={1.8} /> : null;
          }
        )}
        {measured && t.length > 0 && (
          <line
            x1={(t.findIndex((v) => v >= 0) / Math.max(t.length - 1, 1)) * W}
            x2={(t.findIndex((v) => v >= 0) / Math.max(t.length - 1, 1)) * W}
            y1={4}
            y2={H - 4}
            stroke="#171717"
            strokeDasharray="2 3"
            strokeOpacity={0.5}
          />
        )}
      </svg>
      {!measured && (
        <div className="absolute inset-0 flex items-center justify-center text-center px-6 text-[12px] text-[#4d4d4d] leading-snug">
          Peak timing needs 120–240 fps slow-mo. This clip can't time it, so nothing is guessed.
        </div>
      )}
    </div>
  );
};

export const HomeView: React.FC<{ onOpenFilmGuide: () => void }> = ({ onOpenFilmGuide }) => {
  const { data, error } = useOverview();
  const storedBowler = useCurrentBowlerId();
  const bowlerId = storedBowler ?? data?.bowlers?.[0]?.id ?? null;

  const sources = useMemo(() => {
    const ids = new Set((data?.samples ?? []).map((s) => s.delivery_id));
    (data?.recent_deliveries ?? []).forEach((d) => ids.add(d.id));
    return HERO_SOURCES.filter((s) => ids.has(s.id));
  }, [data]);
  const [heroId, setHeroId] = useState<string | null>(null);
  const activeHeroId = heroId ?? sources[0]?.id ?? data?.samples?.[0]?.delivery_id ?? null;
  const { data: hero, error: heroError } = useDeliveryDetail(activeHeroId);
  const heroReady = hero && hero.delivery.id === activeHeroId ? hero : null;
  const playhead = useRef<Playhead>({ frame: 0 });
  const [heroFrame, setHeroFrame] = useState(0);
  const [cam, setCam] = useState<VirtualCameraPreset>(() => defaultCamForHero(activeHeroId));
  const [userPickedCam, setUserPickedCam] = useState(false);

  useEffect(() => {
    setUserPickedCam(false);
    setCam(defaultCamForHero(activeHeroId));
  }, [activeHeroId]);

  useEffect(() => {
    if (heroReady && !userPickedCam) {
      setCam(defaultCamForHero(heroReady.delivery.id, heroReady));
    }
  }, [heroReady, userPickedCam]);

  const pickHomeCam = (p: VirtualCameraPreset) => {
    setUserPickedCam(true);
    setCam(p);
  };

  const { workload, spell } = useSnapshot(data, bowlerId);

  if (error) {
    return (
      <EmptyState
        title="The studio server isn't responding"
        body={
          <span>
            Start it with <code className="font-mono text-[12px]">bowling-biomech serve-studio</code>, then refresh this page.
          </span>
        }
        action={
          <button onClick={() => window.location.reload()} className="v-btn-primary h-9 px-4 text-[13px]">
            Refresh
          </button>
        }
      />
    );
  }
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
        <Spinner /> Loading the studio…
      </div>
    );
  }

  const ev = heroReady?.fact_sheet.events;
  const brFrame = ev?.br?.frame_idx ?? null;

  // Snapshot numbers (backend snapshot first, then live endpoints; never invented).
  const ws = data.workload_snapshot ?? null;
  const ss = data.spell_snapshot ?? null;
  const balls7 = ws?.balls_7d ?? workload?.deliveries_last_7d ?? null;
  const days7 = ws?.days_bowled_7d ?? daysBowledLast7(workload);
  const maxDays = ws?.max_days_per_week ?? null;
  const spreadCm =
    ss?.release_spread_cm ??
    (spell?.release_tunnel?.debiased_radial_sd_m != null ? spell.release_tunnel.debiased_radial_sd_m * 100 : null);
  const completed = ss?.completed_count ?? spell?.completed_count ?? data.bowlers.find((b) => b.id === bowlerId)?.delivery_count ?? 0;
  const zoneLabel = ws?.acwr_zone_label_plain ?? workload?.acwr_summary?.zone_label_plain ?? null;
  const loopSentence = ws?.daily_loop_sentence ?? workload?.daily_loop_summary ?? null;
  const userCards = data.recent_deliveries;

  return (
    <div className="space-y-8 pb-20 md:pb-6">
      {/* ── HERO ─────────────────────────────────────────────────────────── */}
      <section className="space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-end lg:justify-between gap-4">
          <div className="max-w-[720px]">
            <span className="v-label">Fast-bowling biomechanics · from one phone</span>
            <h1 className="text-[34px] md:text-[44px] leading-[1.05] font-[500] tracking-[-0.035em] text-[#171717] mt-2">
              See what your bowling action is really doing.
            </h1>
            <p className="text-[15px] text-[#4d4d4d] mt-3 leading-relaxed">
              Your video, a 3D body you can spin, and the moments that matter — every number shown with its honest range.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <button
              onClick={() => navigate({ name: "upload", bowler: bowlerId })}
              className="v-btn-primary h-11 px-5 text-[14px] flex items-center gap-2"
            >
              <Upload className="w-4 h-4" /> Analyse my bowling
            </button>
            {activeHeroId && (
              <button
                onClick={() => navigate({ name: "delivery", id: activeHeroId, tab: "story", frame: brFrame })}
                className="v-btn-ghost h-11 px-5 text-[14px] flex items-center gap-2"
              >
                Open this breakdown <ArrowRight className="w-4 h-4" />
              </button>
            )}
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {sources.map((s) => (
            <button
              key={s.id}
              onClick={() => setHeroId(s.id)}
              aria-pressed={s.id === activeHeroId}
              className={`h-9 px-3 rounded-full text-[13px] flex items-center gap-2 transition-colors ${
                s.id === activeHeroId ? "bg-[#171717] text-white" : "bg-white text-[#4d4d4d] v-hairline hover:text-[#171717]"
              }`}
            >
              <span className="font-[500]">{s.label}</span>
              <span className={s.id === activeHeroId ? "text-white/60" : "text-[#8f8f8f]"}>{s.sub}</span>
            </button>
          ))}
        </div>

        <div className="grid lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] gap-3">
          <div className="v-panel p-2.5 h-[420px] md:h-[480px] flex flex-col">
            {heroReady ? (
              <HeroVideo key={heroReady.delivery.id} detail={heroReady} playhead={playhead} onFrameChange={setHeroFrame} className="flex-1" />
            ) : (
              <div className="flex-1 v-stage flex items-center justify-center text-[13px] text-[#666666] gap-2">
                {heroError ? "This sample couldn't load." : <><Spinner /> Loading real footage…</>}
              </div>
            )}
          </div>
          <div className="v-panel p-2.5 h-[360px] md:h-[480px] flex flex-col">
            <div className="flex items-center justify-between px-1 pb-2">
              <span className="v-label">3D body · synced to the video</span>
              <div className="v-seg" role="group" aria-label="3D camera">
                {([
                  ["side", "Side"],
                  ["behind", "Behind"],
                  ["match", "Match"],
                  ["top", "Top"],
                ] as Array<[VirtualCameraPreset, string]>).map(([p, lbl]) => (
                  <button key={p} aria-pressed={cam === p} onClick={() => pickHomeCam(p)}>
                    {lbl}
                  </button>
                ))}
              </div>
            </div>
            <div className="flex-1 rounded-[6px] overflow-hidden v-stage relative">
              {heroReady ? (
                <Suspense
                  fallback={
                    <div className="absolute inset-0 flex items-center justify-center text-[13px] text-[#666666] gap-2">
                      <Spinner /> Building the 3D body…
                    </div>
                  }
                >
                  <PitchStage3D
                    key={heroReady.delivery.id}
                    telemetry={heroReady.telemetry}
                    bowlingArm={heroReady.fact_sheet.bowling_arm}
                    frame={heroFrame}
                    preset={cam}
                    onPresetChange={pickHomeCam}
                    showTrails={{ wrist: true, head: false, com: false }}
                    eventFrames={{
                      bfc: ev?.bfc?.frame_idx ?? null,
                      ffc: ev?.ffc?.frame_idx ?? null,
                      uah: ev?.uah?.frame_idx ?? null,
                      br: ev?.br?.frame_idx ?? null,
                      ft: ev?.ft_peak?.frame_idx ?? null,
                    }}
                    idleOrbit={!userPickedCam}
                    compact
                    heightClass="h-full"
                  />
                </Suspense>
              ) : null}
              <div className="absolute bottom-2 left-2 flex flex-wrap gap-1 pointer-events-none">
                {[
                  ["Hips & legs", SEG.pelvis],
                  ["Trunk", SEG.thorax],
                  ["Bowling arm", SEG.arm],
                  ["Wrist", SEG.wrist],
                ].map(([l, c]) => (
                  <span key={l} className="v-chip !h-5 !text-[10.5px] !bg-white/90">
                    <span className="w-2 h-2 rounded-full" style={{ background: c }} />
                    {l}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
        {heroReady?.story && (
          <button
            onClick={() => navigate({ name: "delivery", id: heroReady.delivery.id, tab: "story", frame: brFrame })}
            className="w-full v-panel px-4 py-3 flex items-center justify-between gap-3 text-left hover:shadow-[0_0_0_1px_rgba(0,0,0,0.2)] transition-shadow"
          >
            <span className="min-w-0">
              <span className="v-label block">What this clip shows</span>
              <span className="text-[15px] font-[500] text-[#171717] block truncate">{heroReady.story.headline}</span>
            </span>
            <span className="shrink-0 text-[13px] text-[#4d4d4d] flex items-center gap-1">
              {heroReady.story.capture.measured_count}/{heroReady.story.capture.total_count} checks measured <ArrowRight className="w-4 h-4" />
            </span>
          </button>
        )}
      </section>

      {/* ── SESSION SNAPSHOT ─────────────────────────────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-end justify-between gap-3">
          <div>
            <span className="v-label">Your spell</span>
            <h2 className="text-[20px] font-[500] tracking-[-0.02em] text-[#171717]">
              {userCards.some((c) => c.kind === "upload") ? "Last session" : "Sample session"}
            </h2>
          </div>
          <button
            onClick={() => navigate({ name: "spell", bowler: bowlerId })}
            className="text-[13px] text-[#4d4d4d] hover:text-[#171717] flex items-center gap-1"
          >
            Spell & workload <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="grid sm:grid-cols-2 xl:grid-cols-[1fr_1fr_1fr_1fr_1.3fr] gap-3">
          <SnapshotTile label="Balls analysed" value={<span className="v-num">{completed}</span>} sub="Each one has its own 3D breakdown." />
          <SnapshotTile
            label="Release spread"
            muted={spreadCm === null}
            value={spreadCm === null ? "—" : <span className="v-num">{spreadCm.toFixed(0)} cm</span>}
            sub={
              spreadCm === null
                ? "Needs 2+ measured balls in one spell."
                : "How tightly your release points cluster, after removing measurement noise."
            }
          />
          <SnapshotTile
            label="This week"
            muted={balls7 === null}
            value={balls7 === null ? "—" : <span className="v-num">{balls7}</span>}
            sub={
              balls7 === null
                ? "Log today's balls to start your workload."
                : `balls · ${days7 ?? "—"} bowling day${days7 === 1 ? "" : "s"}${maxDays ? ` of ${maxDays} allowed` : ""}`
            }
          />
          <SnapshotTile
            label="Load change"
            muted={!zoneLabel}
            value={<span className="text-[20px] leading-tight block">{zoneLabel ?? "Not enough history"}</span>}
            sub="Describes how fast your load changed — not an injury prediction."
          />
          <div className="v-panel p-3 flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="v-label">Release bullseye</span>
              <span className="text-[11px] text-[#8f8f8f]">batter's view</span>
            </div>
            <MiniBullseye
              tunnel={spell?.release_tunnel}
              height={120}
              onPick={(id, f) => navigate({ name: "delivery", id, tab: "story", frame: f ?? null })}
            />
          </div>
        </div>
        {loopSentence && (
          <div className="v-panel px-4 py-3 text-[13.5px] text-[#171717] flex items-center gap-2">
            <ListChecks className="w-4 h-4 text-[#297a3a] shrink-0" /> {loopSentence}
          </div>
        )}
      </section>

      {/* ── INSTANT ACTIONS ──────────────────────────────────────────────── */}
      <section className="flex gap-2 overflow-x-auto pb-1 -mx-1 px-1">
        <ActionButton icon={<Upload className="w-4 h-4" />} label="Analyse a clip" onClick={() => navigate({ name: "upload", bowler: bowlerId, mode: "clip" })} />
        <ActionButton icon={<Scissors className="w-4 h-4" />} label="Clip a net session" onClick={() => navigate({ name: "upload", bowler: bowlerId, mode: "net" })} />
        <ActionButton icon={<GitCompareArrows className="w-4 h-4" />} label="Compare last 2 balls" onClick={() => navigate({ name: "compare", a: null, b: null })} />
        <ActionButton icon={<ListChecks className="w-4 h-4" />} label="Log today's load" onClick={() => navigate({ name: "spell", bowler: bowlerId })} />
        <ActionButton icon={<Dumbbell className="w-4 h-4" />} label="Today's drill" onClick={() => navigate({ name: "practice", bowler: bowlerId })} />
        <ActionButton icon={<Camera className="w-4 h-4" />} label="How to film" onClick={onOpenFilmGuide} />
      </section>

      {/* ── INSIDE EVERY BREAKDOWN ───────────────────────────────────────── */}
      {heroReady && (
        <section className="space-y-3">
          <div>
            <span className="v-label">Inside every breakdown</span>
            <h2 className="text-[20px] font-[500] tracking-[-0.02em] text-[#171717]">Instruments, not just words</h2>
          </div>
          <div className="grid md:grid-cols-3 gap-3">
            <div className="v-panel p-3 flex flex-col gap-2">
              <span className="v-label">Movement sequence</span>
              <div className="rounded-[6px] overflow-hidden bg-[#fafafa] aspect-[16/7] flex">
                <StrobeOrMoments detail={heroReady} />
              </div>
              <p className="text-[12.5px] text-[#666666] leading-snug">Key positions laid side by side, so you can see the whole action in one picture.</p>
            </div>
            <div className="v-panel p-3 flex flex-col gap-2">
              <span className="v-label">Hips → chest → arm → wrist</span>
              <SequencePreview detail={heroReady} />
              <p className="text-[12.5px] text-[#666666] leading-snug">Which body part speeds up first. Shown only when the clip is fast enough to time it.</p>
            </div>
            <div className="v-panel p-3 flex flex-col gap-2">
              <span className="v-label">Release tunnel</span>
              <MiniBullseye tunnel={spell?.release_tunnel} height={96} />
              <p className="text-[12.5px] text-[#666666] leading-snug">Where the ball leaves your hand, ball after ball, with each point's error ellipse.</p>
            </div>
          </div>
        </section>
      )}

      {/* ── BALL STRIP ───────────────────────────────────────────────────── */}
      <section className="space-y-3">
        <div className="flex items-end justify-between">
          <div>
            <span className="v-label">Library</span>
            <h2 className="text-[20px] font-[500] tracking-[-0.02em] text-[#171717]">Recent balls</h2>
          </div>
          <button onClick={() => navigate({ name: "deliveries", bowler: null })} className="text-[13px] text-[#4d4d4d] hover:text-[#171717] flex items-center gap-1">
            All deliveries <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
        <div className="flex gap-3 overflow-x-auto pb-2 -mx-1 px-1 snap-x">
          {userCards.map((c) => (
            <div key={c.id} className="w-[280px] shrink-0 snap-start flex">
              <DeliveryCardTile card={c} showBowler={false} />
            </div>
          ))}
        </div>
      </section>

      {/* ── GUARDRAILS (never truncated) ─────────────────────────────────── */}
      <footer className="border-t border-[#ebebeb] pt-4">
        <GuardrailStrip />
      </footer>
    </div>
  );
};

/** Server stroboscopic composite when available; otherwise the four real key-moment frames. */
const StrobeOrMoments: React.FC<{ detail: DeliveryDetailResponse }> = ({ detail }) => {
  const [failed, setFailed] = useState(false);
  const strobe = detail.story?.stroboscopic_url;
  if (strobe && !failed) {
    const src = strobe.includes("?") ? strobe : `${strobe}?w=960&skeleton=false&trails=true`;
    return <img src={src} alt="Stroboscopic sequence" className="w-full h-full object-cover" onError={() => setFailed(true)} />;
  }
  const moments = detail.story?.key_moments ?? [];
  if (moments.length === 0) return <div className="flex-1 v-stage" />;
  return (
    <div className="grid w-full" style={{ gridTemplateColumns: `repeat(${moments.length}, minmax(0, 1fr))` }}>
      {moments.map((m) => (
        <div key={m.id} className="relative overflow-hidden border-r border-white last:border-r-0">
          <img src={m.thumbnail_url ?? api.frameUrl(detail.delivery.id, m.frame_idx, 320)} alt={m.label} className="w-full h-full object-cover" />
          <span className="absolute bottom-1 left-1 v-chip !h-4 !text-[9.5px] !px-1.5 !bg-white/90">{m.label}</span>
        </div>
      ))}
    </div>
  );
};

export default HomeView;
