import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeftRight, ExternalLink } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type { Archetype, CompareStory, DeliveryCard, DeliveryDetailResponse, OverviewResponse } from "../types";
import { SEG } from "../theme";
import { PitchStage3D } from "../workbench/Stage/PitchStage3D";
import { GuardrailStrip, JogWheel, SpeedDetent } from "../components/ui/PaperPrimitives";
import { CompareCanvas } from "../components/compare/CompareCanvas";
import { SequenceOverlay, type SequenceTrack } from "../components/compare/SequenceOverlay";
import { ArchetypeEnvelopeMatrix, DeltaMatrix } from "../components/compare/DeltaMatrix";
import {
  SYNC_LABEL,
  type SyncEvent,
  analysisFps,
  defaultPair,
  formatMs,
  formatShortDate,
  frameAtTime,
  nFrames,
  syncFrameOf,
  timesRelFfc,
} from "../components/compare/compareMath";

/**
 * Compare & Ghost Overlay Studio (docs/research_app_landscape_v5.md §8.3).
 * Side-by-side (one FFC-locked clock) · Blend overlay (SimulCam-style opacity + nudge) · Dual 3D ghost,
 * overlaid kinematic-sequence lanes and the backend's noise-gated delta matrix. Statuses/differences are never
 * computed client-side; "Within noise" is a calm, first-class state.
 */

type StageMode = "side" | "blend" | "ghost3d";
type Rate = 0.1 | 0.25 | 0.5 | 1;

interface Side {
  detail: DeliveryDetailResponse;
  times: number[]; // ms rel FFC
  syncFrame: number;
  relSync: number[]; // ms rel sync event
  n: number;
}

function buildSide(detail: DeliveryDetailResponse, ev: SyncEvent): Side {
  const t = detail.telemetry;
  const n = nFrames(t);
  const ffc = syncFrameOf(detail.fact_sheet, "ffc", n);
  const times = timesRelFfc(t, ffc);
  const syncFrame = syncFrameOf(detail.fact_sheet, ev, n) ?? ffc ?? 0;
  const t0 = times[syncFrame] ?? 0;
  return { detail, times, syncFrame, relSync: times.map((x) => x - t0), n };
}

function useDelivery(id: string | null) {
  const [state, setState] = useState<{ data: DeliveryDetailResponse | null; loading: boolean; error: string | null }>({
    data: null,
    loading: false,
    error: null,
  });
  useEffect(() => {
    if (!id) {
      setState({ data: null, loading: false, error: null });
      return;
    }
    let alive = true;
    setState({ data: null, loading: true, error: null });
    api
      .getDelivery(id)
      .then((d) => alive && setState({ data: d, loading: false, error: null }))
      .catch((e: Error) => alive && setState({ data: null, loading: false, error: e.message || "Could not load" }));
    return () => {
      alive = false;
    };
  }, [id]);
  return state;
}

const deliveryOptionLabel = (d: DeliveryCard) =>
  `${d.label} · ${d.bowler_name} · ${formatShortDate(d.created_at)} · ${d.view_label}${d.capture_grade === "unusable" ? " (unusable)" : ""}`;

const isPanUnverified = (d: DeliveryDetailResponse) =>
  !!d.fact_sheet?.calibration?.is_panning_camera && !d.delivery?.pan_summary?.is_verified;

export const CompareView: React.FC<{ a: string | null; b: string | null; refId?: string | null }> = ({ a, b, refId }) => {
  const [ov, setOv] = useState<OverviewResponse | null>(null);
  const [ovError, setOvError] = useState<string | null>(null);
  const [syncEvent, setSyncEvent] = useState<SyncEvent>("ffc");
  const [mode, setMode] = useState<StageMode>("side");
  const [mirrorB, setMirrorB] = useState(false);
  const [opacity, setOpacity] = useState(0.5);
  const [nudge, setNudge] = useState<[number, number]>([0, 0]);
  const [tMs, setTMs] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState<Rate>(0.25);
  const [archetypes, setArchetypes] = useState<Archetype[] | null>(null);
  const [archError, setArchError] = useState<string | null>(null);
  const [compare, setCompare] = useState<{ data: CompareStory | null; loading: boolean; error: string | null }>({
    data: null,
    loading: false,
    error: null,
  });

  const bIsArchetype = !!refId;

  // ── overview + default pairing ──
  useEffect(() => {
    let alive = true;
    api
      .overview()
      .then((o) => alive && setOv(o))
      .catch((e: Error) => alive && setOvError(e.message || "Could not load your deliveries"));
    return () => {
      alive = false;
    };
  }, []);

  useEffect(() => {
    if (!ov) return;
    if (a && (b || refId)) return;
    const pair = defaultPair(ov);
    if (!a) {
      if (pair) navigate({ name: "compare", a: pair.a, b: refId ? null : pair.b, ref: refId ?? null });
      return;
    }
    // A chosen (e.g. from the Workbench "Compare" button): pick a partner, preferring the default pairing.
    const partner =
      (pair && pair.b !== a ? pair.b : pair && pair.a !== a ? pair.a : null) ??
      ov.recent_deliveries.find((d) => d.id !== a && d.capture_grade !== "unusable")?.id ??
      ov.recent_deliveries.find((d) => d.id !== a)?.id ??
      null;
    if (partner) navigate({ name: "compare", a, b: partner, ref: null });
  }, [ov, a, b, refId]);

  const A = useDelivery(a);
  const B = useDelivery(bIsArchetype ? null : b);
  const armA: "right" | "left" = A.data?.fact_sheet?.bowling_arm ?? "right";

  // ── archetypes (literature reference ghosts) ──
  useEffect(() => {
    if (!bIsArchetype) return;
    let alive = true;
    setArchError(null);
    api
      .archetypes(armA)
      .then((r) => alive && setArchetypes(r.archetypes ?? []))
      .catch((e: Error) => alive && setArchError(e.message || "Reference archetypes unavailable"));
    return () => {
      alive = false;
    };
  }, [bIsArchetype, armA]); // eslint-disable-line react-hooks/exhaustive-deps

  const archetype = useMemo(
    () => (bIsArchetype ? archetypes?.find((x) => x.id === refId) ?? null : null),
    [bIsArchetype, archetypes, refId]
  );

  // ── backend compare story (only for two real deliveries) ──
  useEffect(() => {
    if (!a || !b || bIsArchetype) {
      setCompare({ data: null, loading: false, error: null });
      return;
    }
    let alive = true;
    setCompare({ data: null, loading: true, error: null });
    api
      .compare(a, b)
      .then((c) => alive && setCompare({ data: c, loading: false, error: null }))
      .catch((e: Error) => alive && setCompare({ data: null, loading: false, error: e.message || "Could not compare" }));
    return () => {
      alive = false;
    };
  }, [a, b, bIsArchetype]);

  // Archetypes have no video → only the 3D ghost stage applies.
  useEffect(() => {
    if (bIsArchetype) {
      setMode("ghost3d");
      setSyncEvent("ffc");
    }
  }, [bIsArchetype]);

  const sideA = useMemo(() => (A.data ? buildSide(A.data, syncEvent) : null), [A.data, syncEvent]);
  const sideB = useMemo(() => (B.data && !bIsArchetype ? buildSide(B.data, syncEvent) : null), [B.data, syncEvent, bIsArchetype]);
  const armB: "right" | "left" = sideB?.detail.fact_sheet.bowling_arm ?? (archetype?.bowling_arm as "right" | "left" | undefined) ?? armA;

  // ── shared compare clock range (intersection of both clips, ms rel sync) ──
  const range = useMemo<[number, number]>(() => {
    if (!sideA) return [-400, 300];
    let lo = sideA.relSync[0];
    let hi = sideA.relSync[sideA.relSync.length - 1];
    const bTimes = sideB?.relSync ?? archetype?.time_rel_ffc_ms ?? null;
    if (bTimes && bTimes.length > 1) {
      lo = Math.max(lo, bTimes[0]);
      hi = Math.min(hi, bTimes[bTimes.length - 1]);
    }
    if (!(hi > lo)) return [sideA.relSync[0], sideA.relSync[sideA.relSync.length - 1]];
    return [lo, hi];
  }, [sideA, sideB, archetype]);

  useEffect(() => {
    setTMs((t) => Math.min(range[1], Math.max(range[0], t)));
  }, [range]);

  // ── playback (capture-time ms advance at `rate`) ──
  const rafRef = useRef<number | null>(null);
  useEffect(() => {
    if (!playing) return;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = now - last;
      last = now;
      setTMs((t) => {
        const nt = t + dt * rate;
        return nt > range[1] ? range[0] : nt;
      });
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, rate, range]);

  const fpsA = sideA ? analysisFps(sideA.detail.telemetry) : 30;
  const frameA = sideA ? frameAtTime(sideA.relSync, tMs) : 0;
  const frameB = sideB ? frameAtTime(sideB.relSync, tMs) : 0;
  const step = useCallback(
    (delta: number) => {
      setPlaying(false);
      setTMs((t) => Math.min(range[1], Math.max(range[0], t + (delta * 1000) / fpsA)));
    },
    [fpsA, range]
  );

  const setPick = (which: "a" | "b", id: string) => {
    if (which === "a") navigate({ name: "compare", a: id, b: bIsArchetype ? null : b, ref: refId ?? null });
    else navigate({ name: "compare", a, b: id, ref: null });
  };
  const setArchetypePick = (id: string) => navigate({ name: "compare", a, b: null, ref: id });

  const switchBSource = (src: "delivery" | "archetype") => {
    if (src === "archetype") {
      if (archetypes?.length) setArchetypePick(String(archetypes[0].id));
      else {
        api
          .archetypes(armA)
          .then((r) => {
            setArchetypes(r.archetypes ?? []);
            if (r.archetypes?.length) setArchetypePick(String(r.archetypes[0].id));
            else setArchError("No reference archetypes are available yet.");
          })
          .catch((e: Error) => setArchError(e.message || "Reference archetypes unavailable"));
      }
    } else {
      const pair = defaultPair(ov);
      const other = ov?.recent_deliveries.find((d) => d.id !== a)?.id ?? null;
      navigate({ name: "compare", a, b: pair && pair.b !== a ? pair.b : other, ref: null });
    }
  };

  // ── derived notices ──
  const viewA = A.data?.story?.capture?.view ?? A.data?.fact_sheet?.calibration?.view_class ?? null;
  const viewB = B.data?.story?.capture?.view ?? B.data?.fact_sheet?.calibration?.view_class ?? null;
  const viewsDiffer = !!(viewA && viewB && viewA !== viewB);
  const panIssue = (A.data && isPanUnverified(A.data)) || (B.data && isPanUnverified(B.data));
  const blendVideoOk = !viewsDiffer && !panIssue;
  const armsDiffer = armA !== armB;

  // ── 3D ghost inputs (aligned at the sync moment in A's FFC-relative time) ──
  const ghost = useMemo(() => {
    if (!sideA) return null;
    const aSyncRelFfc = sideA.times[sideA.syncFrame] ?? 0;
    if (sideB) {
      const bSyncRelFfc = sideB.times[sideB.syncFrame] ?? 0;
      return {
        joints: sideB.detail.telemetry.joints3d_world_m,
        times: sideB.times.map((t) => t - bSyncRelFfc + aSyncRelFfc),
        label: `B · ${sideB.detail.delivery.label}`,
      };
    }
    if (archetype) {
      return { joints: archetype.joints3d_world_m, times: archetype.time_rel_ffc_ms, label: `${archetype.title} (synthetic)` };
    }
    return null;
  }, [sideA, sideB, archetype]);

  const seqA: SequenceTrack | null = sideA
    ? { times: sideA.relSync, series: sideA.detail.telemetry.sequence_series, fps: sideA.detail.fact_sheet.audit?.fps ?? fpsA }
    : null;
  const seqB: SequenceTrack | null = sideB
    ? {
        times: sideB.relSync,
        series: sideB.detail.telemetry.sequence_series,
        fps: sideB.detail.fact_sheet.audit?.fps ?? analysisFps(sideB.detail.telemetry),
      }
    : archetype
    ? { times: archetype.time_rel_ffc_ms, series: archetype.sequence_series, fps: archetype.fps ?? null }
    : null;

  const eventChips = useMemo(() => {
    if (!sideA) return [];
    const evs: SyncEvent[] = ["bfc", "ffc", "br"];
    return evs
      .map((ev) => {
        const f = syncFrameOf(sideA.detail.fact_sheet, ev, sideA.n);
        return f === null ? null : { ev, t: sideA.relSync[f] };
      })
      .filter((x): x is { ev: SyncEvent; t: number } => !!x && x.t >= range[0] && x.t <= range[1]);
  }, [sideA, range]);

  const deliveries = ov?.recent_deliveries ?? [];
  const bProvenance = bIsArchetype
    ? "Synthetic archetype — not a real player"
    : B.data
    ? deliveries.find((d) => d.id === b)?.kind === "sample_real"
      ? "Analysed pro clip — inherits its own camera limits"
      : "Your analysed clip"
    : null;

  // ── render ──
  if (ovError && !a) {
    return (
      <div className="max-w-3xl mx-auto v-panel p-6 text-[14px] text-[#4d4d4d]">
        Could not load your deliveries ({ovError}). Make sure the Studio server is running.
      </div>
    );
  }
  if (ov && deliveries.length < 2 && !a) {
    return (
      <div className="max-w-3xl mx-auto space-y-4">
        <h1 className="text-[24px] font-[500] tracking-[-0.02em]">Compare</h1>
        <div className="v-panel p-6 text-[14px] text-[#4d4d4d]">
          You need at least two analysed balls to compare. Analyse another delivery, or compare one ball against a literature
          reference archetype once you have one.
          <div className="mt-4">
            <button className="v-btn-primary h-9 px-4 text-[13px]" onClick={() => navigate({ name: "upload", bowler: null })}>
              New analysis
            </button>
          </div>
        </div>
        <GuardrailStrip />
      </div>
    );
  }

  return (
    <div className="space-y-4 min-w-0">
      {/* ── Header & controls bar ── */}
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-[24px] font-[500] tracking-[-0.02em] text-[#171717]">Compare</h1>
          <p className="text-[13px] text-[#8f8f8f]">Two balls on one clock, locked at a key moment.</p>
        </div>
      </div>

      <div className="v-panel p-3 sm:p-4 grid grid-cols-1 gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] items-end min-w-0">
        <label className="block min-w-0">
          <div className="min-h-[30px] flex items-center">
            <span className="v-label">Delivery A · solid</span>
          </div>
          <select
            className="v-input mt-1 w-full min-w-0 h-10 px-2.5 text-[13px] truncate"
            value={a ?? ""}
            onChange={(e) => setPick("a", e.target.value)}
          >
            {!a && <option value="">Choose a ball…</option>}
            {deliveries.map((d) => (
              <option key={d.id} value={d.id}>
                {deliveryOptionLabel(d)}
              </option>
            ))}
          </select>
        </label>

        <div className="min-w-0">
          <div className="min-h-[30px] flex flex-wrap items-center justify-between gap-2">
            <span className="v-label">B · dashed slate</span>
            <div className="v-seg">
              <button aria-pressed={!bIsArchetype} onClick={() => switchBSource("delivery")}>
                Delivery
              </button>
              <button aria-pressed={bIsArchetype} onClick={() => switchBSource("archetype")}>
                Literature archetype
              </button>
            </div>
          </div>
          {bIsArchetype ? (
            <select
              className="v-input mt-1 w-full min-w-0 h-10 px-2.5 text-[13px] truncate"
              value={refId ?? ""}
              onChange={(e) => setArchetypePick(e.target.value)}
            >
              {!archetypes && <option value={refId ?? ""}>Loading archetypes…</option>}
              {archetypes?.map((x) => (
                <option key={x.id} value={x.id}>
                  {x.title} · {x.action_class}
                </option>
              ))}
            </select>
          ) : (
            <select
              className="v-input mt-1 w-full min-w-0 h-10 px-2.5 text-[13px] truncate"
              value={b ?? ""}
              onChange={(e) => setPick("b", e.target.value)}
            >
              {!b && <option value="">Choose a ball…</option>}
              {deliveries
                .filter((d) => d.id !== a)
                .map((d) => (
                  <option key={d.id} value={d.id}>
                    {deliveryOptionLabel(d)}
                  </option>
                ))}
            </select>
          )}
        </div>

        <div className="flex flex-wrap items-end gap-2 min-w-0">
          <div>
            <span className="v-label block">Sync at</span>
            <div className="v-seg mt-1 h-10 items-center">
              {(["bfc", "ffc", "br"] as SyncEvent[]).map((ev) => (
                <button
                  key={ev}
                  aria-pressed={syncEvent === ev}
                  disabled={bIsArchetype && ev !== "ffc"}
                  title={SYNC_LABEL[ev]}
                  onClick={() => setSyncEvent(ev)}
                  className="disabled:opacity-40"
                >
                  {ev.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div className="max-w-full">
            <span className="v-label block">Stage</span>
            <div className="v-seg mt-1 h-10 items-center max-w-full overflow-x-auto">
              <button aria-pressed={mode === "side"} disabled={bIsArchetype} onClick={() => setMode("side")} className="disabled:opacity-40">
                Side-by-side
              </button>
              <button aria-pressed={mode === "blend"} disabled={bIsArchetype} onClick={() => setMode("blend")} className="disabled:opacity-40">
                Blend
              </button>
              <button aria-pressed={mode === "ghost3d"} onClick={() => setMode("ghost3d")}>
                Dual 3D ghost
              </button>
            </div>
          </div>
          <button
            className={`h-10 px-3 text-[13px] inline-flex items-center gap-1.5 ${mirrorB ? "v-btn-primary" : "v-btn-ghost"}`}
            aria-pressed={mirrorB}
            onClick={() => setMirrorB((m) => !m)}
            title="Flip B left↔right to compare a left-arm and a right-arm action"
          >
            <ArrowLeftRight className="w-3.5 h-3.5" /> Mirror B {mirrorB ? "· Mirrored" : ""}
          </button>
        </div>
      </div>

      {/* ── provenance & notices ── */}
      <div className="flex flex-wrap items-center gap-2 text-[12px]">
        {bProvenance && <span className={`v-chip ${bIsArchetype ? "v-chip-ink" : ""}`}>B: {bProvenance}</span>}
        {mirrorB && <span className="v-chip">B is mirrored</span>}
        {armsDiffer && !mirrorB && (
          <span className="v-chip">A bowls {armA}-arm, B bowls {armB}-arm — try Mirror B</span>
        )}
        {archetype && <span className="text-[#4d4d4d] leading-[1.5]">{archetype.disclaimer}</span>}
        {archError && <span className="text-[#4d4d4d]">Reference archetypes: {archError}</span>}
      </div>

      {(A.error || B.error) && (
        <div className="v-panel p-3 text-[13px] text-[#4d4d4d]">
          {A.error && <div>Ball A could not be loaded: {A.error}</div>}
          {B.error && <div>Ball B could not be loaded: {B.error}</div>}
        </div>
      )}

      {/* ── stage ── */}
      {!sideA ? (
        <div className="v-stage h-[360px] flex items-center justify-center text-[13px] text-[#8f8f8f]">
          {A.loading || B.loading || !ov ? "Loading both balls…" : "Pick Delivery A to start."}
        </div>
      ) : (
        <div className="space-y-3">
          {mode === "side" && (
            <div className="grid gap-3 md:grid-cols-2">
              <CompareCanvas
                primary={{
                  videoUrl: sideA.detail.delivery.video_proxy_url || api.videoUrl(sideA.detail.delivery.id),
                  telemetry: sideA.detail.telemetry,
                  arm: armA,
                  frame: frameA,
                }}
                tag={`A · ${sideA.detail.delivery.label}`}
                caption={sideA.detail.story?.capture?.view_label ?? undefined}
              />
              {sideB ? (
                <CompareCanvas
                  primary={{
                    videoUrl: sideB.detail.delivery.video_proxy_url || api.videoUrl(sideB.detail.delivery.id),
                    telemetry: sideB.detail.telemetry,
                    arm: armB,
                    frame: frameB,
                  }}
                  tag={`B · ${sideB.detail.delivery.label}`}
                  caption={sideB.detail.story?.capture?.view_label ?? undefined}
                  mirrorPrimary={mirrorB}
                  primaryGhostStyle
                />
              ) : (
                <div className="v-stage h-[320px] md:h-[440px] flex items-center justify-center text-[13px] text-[#8f8f8f]">
                  {B.loading ? "Loading B…" : "Pick Delivery B."}
                </div>
              )}
            </div>
          )}

          {mode === "blend" && (
            <div className="grid gap-3 lg:grid-cols-[1fr_260px]">
              <CompareCanvas
                primary={{
                  videoUrl: sideA.detail.delivery.video_proxy_url || api.videoUrl(sideA.detail.delivery.id),
                  telemetry: sideA.detail.telemetry,
                  arm: armA,
                  frame: frameA,
                }}
                tag={`A solid · B dashed${mirrorB ? " (mirrored)" : ""}`}
                heightClass="h-[340px] md:h-[520px]"
                overlay={
                  sideB
                    ? {
                        videoUrl: sideB.detail.delivery.video_proxy_url || api.videoUrl(sideB.detail.delivery.id),
                        telemetry: sideB.detail.telemetry,
                        arm: armB,
                        frame: frameB,
                        syncFramePrimary: sideA.syncFrame,
                        syncFrameOverlay: sideB.syncFrame,
                        opacity,
                        nudge,
                        mirror: mirrorB,
                        showVideo: blendVideoOk,
                      }
                    : null
                }
              />
              <div className="v-panel p-3 space-y-4">
                <div>
                  <div className="flex items-baseline justify-between">
                    <span className="v-label">B opacity</span>
                    <span className="v-num text-[12px]">{Math.round(opacity * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min={0}
                    max={100}
                    value={Math.round(opacity * 100)}
                    onChange={(e) => setOpacity(Number(e.target.value) / 100)}
                    className="w-full accent-[#171717]"
                    disabled={!blendVideoOk}
                    aria-label="Opacity of delivery B"
                  />
                </div>
                <div>
                  <div className="flex items-baseline justify-between">
                    <span className="v-label">Micro-align B</span>
                    <span className="v-num text-[11px] text-[#8f8f8f]">
                      x {nudge[0] > 0 ? "+" : ""}
                      {nudge[0]} · y {nudge[1] > 0 ? "+" : ""}
                      {nudge[1]} px
                    </span>
                  </div>
                  <div className="mt-2 grid grid-cols-3 gap-1 w-[120px]">
                    <span />
                    <button className="v-btn-ghost h-8 text-[13px]" onClick={() => setNudge(([x, y]) => [x, y - 2])} aria-label="Nudge B up">
                      ↑
                    </button>
                    <span />
                    <button className="v-btn-ghost h-8 text-[13px]" onClick={() => setNudge(([x, y]) => [x - 2, y])} aria-label="Nudge B left">
                      ←
                    </button>
                    <button className="v-btn-ghost h-8 text-[11px]" onClick={() => setNudge([0, 0])} aria-label="Reset alignment">
                      0
                    </button>
                    <button className="v-btn-ghost h-8 text-[13px]" onClick={() => setNudge(([x, y]) => [x + 2, y])} aria-label="Nudge B right">
                      →
                    </button>
                    <span />
                    <button className="v-btn-ghost h-8 text-[13px]" onClick={() => setNudge(([x, y]) => [x, y + 2])} aria-label="Nudge B down">
                      ↓
                    </button>
                    <span />
                  </div>
                </div>
                <p className="text-[12px] leading-[1.5] text-[#4d4d4d]">
                  B is registered to A by the skeleton at {SYNC_LABEL[syncEvent].toLowerCase()} (hip position and body size).
                  {!blendVideoOk && (
                    <>
                      {" "}
                      <span className="text-[#171717]">
                        {viewsDiffer ? "The two clips were filmed from different camera views" : "At least one camera was moving without a verified pan track"}
                        , so B's video is hidden and only its skeleton outline is overlaid.
                      </span>
                    </>
                  )}
                </p>
              </div>
            </div>
          )}

          {mode === "ghost3d" && (
            <div className="space-y-2">
              <PitchStage3D
                telemetry={sideA.detail.telemetry}
                bowlingArm={armA}
                frame={frameA}
                ghostJoints3d={ghost?.joints}
                ghostTimeRelFfcMs={ghost?.times}
                ghostLabel={ghost?.label}
                mirrorGhost={mirrorB}
                defaultPreset="side"
                eventFrames={{
                  bfc: sideA.detail.fact_sheet.events?.bfc?.frame_idx ?? null,
                  ffc: sideA.detail.fact_sheet.events?.ffc?.frame_idx ?? null,
                  uah: sideA.detail.fact_sheet.events?.uah?.frame_idx ?? null,
                  br: sideA.detail.fact_sheet.events?.br?.frame_idx ?? null,
                  ft: sideA.detail.fact_sheet.events?.ft_peak?.frame_idx ?? null,
                }}
                heightClass="h-[380px] md:h-[520px]"
              />
              <p className="text-[12px] leading-[1.5] text-[#4d4d4d]">
                A is the coloured mannequin; B is the slate ghost, timed to A at {SYNC_LABEL[syncEvent].toLowerCase()} and placed
                at A's pelvis position.
                The ghost body is never rescaled to fake a match
                {archetype ? ` (archetype stature ${archetype.stature_m.toFixed(2)} m)` : ""}.
                {!ghost && " Pick B to show the ghost."}
              </p>
            </div>
          )}

          {/* ── shared transport ── */}
          <div className="v-panel p-3 flex flex-wrap items-center gap-3 min-w-0 overflow-hidden">
            <JogWheel
              frame={frameA}
              numFrames={sideA.n}
              playing={playing}
              onTogglePlay={() => setPlaying((p) => !p)}
              onStep={step}
            />
            <SpeedDetent rate={rate} onChange={setRate} captureFps={fpsA} />
            <div className="flex-1 min-w-[220px] max-w-full">
              <input
                type="range"
                min={Math.floor(range[0])}
                max={Math.ceil(range[1])}
                step={1}
                value={Math.round(tMs)}
                onChange={(e) => {
                  setPlaying(false);
                  setTMs(Number(e.target.value));
                }}
                className="w-full accent-[#171717]"
                aria-label={`Time relative to ${SYNC_LABEL[syncEvent]}`}
              />
              <div className="flex flex-wrap items-center justify-between gap-2 mt-1">
                <span className="v-num text-[12px] text-[#171717]">
                  {formatMs(tMs)} rel {syncEvent.toUpperCase()}
                </span>
                <span className="flex gap-1">
                  {eventChips.map((c) => (
                    <button
                      key={c.ev}
                      className="v-chip text-[11px] hover:bg-[#f4f4f5]"
                      onClick={() => {
                        setPlaying(false);
                        setTMs(c.t);
                      }}
                      title={`Jump to ${SYNC_LABEL[c.ev]}`}
                    >
                      {c.ev.toUpperCase()}
                    </button>
                  ))}
                </span>
              </div>
            </div>
          </div>

          {/* ── sequence overlay + delta matrix ── */}
          <div className="grid gap-3 xl:grid-cols-2">
            {seqA && (
              <SequenceOverlay
                a={seqA}
                b={seqB}
                bLabel={archetype ? "Archetype (synthetic)" : "B"}
                tMs={tMs}
                range={range}
                syncLabel={SYNC_LABEL[syncEvent].toLowerCase()}
                onSeek={(t) => {
                  setPlaying(false);
                  setTMs(t);
                }}
              />
            )}
            {bIsArchetype ? (
              archetype ? (
                <ArchetypeEnvelopeMatrix factSheet={sideA.detail.fact_sheet} archetype={archetype} />
              ) : (
                <div className="v-panel p-4 text-[13px] text-[#8f8f8f]">Loading the reference archetype…</div>
              )
            ) : (
              <DeltaMatrix story={compare.data} loading={compare.loading} error={compare.error} />
            )}
          </div>

          <div className="flex flex-wrap gap-2">
            <button
              className="v-btn-ghost h-9 px-3 text-[13px] inline-flex items-center gap-1.5"
              onClick={() => a && navigate({ name: "delivery", id: a, tab: "story", frame: frameA })}
            >
              <ExternalLink className="w-3.5 h-3.5" /> Open A in Workbench
            </button>
            {sideB && (
              <button
                className="v-btn-ghost h-9 px-3 text-[13px] inline-flex items-center gap-1.5"
                onClick={() => b && navigate({ name: "delivery", id: b, tab: "story", frame: frameB })}
              >
                <ExternalLink className="w-3.5 h-3.5" /> Open B in Workbench
              </button>
            )}
          </div>
        </div>
      )}

      <div className="text-[11px] text-[#8f8f8f] flex items-center gap-2">
        <span className="inline-block w-3 h-[2px]" style={{ background: SEG.pelvis }} />
        <span className="inline-block w-3 h-[2px]" style={{ background: SEG.thorax }} />
        <span className="inline-block w-3 h-[2px]" style={{ background: SEG.arm }} />
        <span className="inline-block w-3 h-[2px]" style={{ background: SEG.wrist }} />
        A segments · <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: SEG.ghost }} /> B / reference
      </div>

      <GuardrailStrip />
    </div>
  );
};

export default CompareView;
