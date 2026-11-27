import React, { useEffect, useRef, useState } from "react";
import { ArrowRight } from "lucide-react";
import { api } from "../../api";
import type { JobStatus, OverviewBowler } from "../../types";
import { navigate } from "../../router";
import {
  DropZone,
  FailureCard,
  FpsPicker,
  NeedBowler,
  PanelTitle,
  PIPELINE_STEPS,
  PipelineProgress,
  Spinner,
  explainFailure,
  fpsValue,
  jobStepWindow,
  pollJob,
  useElapsed,
  type Failure,
  type FpsChoice,
} from "./shared";

type Side = "a" | "b";
type Phase = "idle" | "uploading" | "processing" | "fusing" | "done" | "failed";

interface SideState {
  job: JobStatus | null;
  hasJob: boolean;
  lo: number;
  hi: number;
  deliveryId: string | null;
}
const EMPTY_SIDE: SideState = { job: null, hasJob: false, lo: -1, hi: -1, deliveryId: null };

const LABELS: Record<Side, string> = { a: "Side-on (phone A)", b: "Behind (phone B)" };

interface FuseSummary {
  deliveryId: string | null;
  offsetS: number | null;
  syncScore: number | null;
  overlapS: number | null;
  angleDeg: number | null;
  baselineSin: number | null;
}

function num(v: unknown): number | null {
  const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
  return Number.isFinite(n) ? n : null;
}

function summarise(res: any): FuseSummary {
  const sync = res?.sync ?? {};
  return {
    deliveryId: res?.delivery_id ?? res?.fused_delivery_id ?? res?.delivery?.id ?? null,
    offsetS: num(sync.offset_b_minus_a_s),
    syncScore: num(sync.sync_score),
    overlapS: num(sync.overlap_duration_s),
    angleDeg: num(res?.baseline_angle_deg),
    baselineSin: num(res?.baseline_sin),
  };
}

/** Two-phone pairing: side-on + behind → both analysed → time-matched 3D fusion. */
export const DualPhonePairing: React.FC<{ bowler: OverviewBowler | null; canAnalyse: boolean }> = ({ bowler, canAnalyse }) => {
  const [files, setFiles] = useState<Record<Side, File | null>>({ a: null, b: null });
  const [fps, setFps] = useState<FpsChoice>("auto");
  const [phase, setPhase] = useState<Phase>("idle");
  const [sides, setSides] = useState<Record<Side, SideState>>({ a: EMPTY_SIDE, b: EMPTY_SIDE });
  const [failure, setFailure] = useState<Failure | null>(null);
  const [summary, setSummary] = useState<FuseSummary | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const runRef = useRef(0);

  useEffect(
    () => () => {
      runRef.current = -1;
    },
    []
  );

  const busy = phase === "uploading" || phase === "processing" || phase === "fusing";
  const elapsed = useElapsed(startedAt, busy);

  const patchSide = (s: Side, p: Partial<SideState> | ((prev: SideState) => Partial<SideState>)) =>
    setSides((prev) => ({ ...prev, [s]: { ...prev[s], ...(typeof p === "function" ? p(prev[s]) : p) } }));

  const start = async () => {
    if (!bowler || !files.a || !files.b) return;
    const runId = ++runRef.current;
    const alive = () => runRef.current === runId;
    setPhase("uploading");
    setFailure(null);
    setSummary(null);
    setSides({ a: EMPTY_SIDE, b: EMPTY_SIDE });
    setStartedAt(Date.now());
    const fpsNum = fpsValue(fps);

    const upload = (s: Side) => {
      const fd = new FormData();
      fd.append("video", files[s] as File);
      fd.append("label", LABELS[s]);
      fd.append("start_frame", "0");
      fd.append("bowling_arm", bowler.bowling_arm);
      if (fpsNum) fd.append("capture_fps_override", String(fpsNum));
      return api.bowlerUploadDelivery(bowler.id, fd);
    };

    try {
      const [ra, rb] = await Promise.all([upload("a"), upload("b")]);
      if (!alive()) return;
      setPhase("processing");

      const follow = async (s: Side, res: { delivery_id: string; job_id?: string | null }): Promise<string> => {
        patchSide(s, { deliveryId: res.delivery_id || null });
        if (!res.job_id) {
          patchSide(s, { lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length });
          if (!res.delivery_id) throw new Error("upload returned no delivery");
          return res.delivery_id;
        }
        patchSide(s, { hasJob: true, lo: 0, hi: 0 });
        const out = await pollJob(
          res.job_id,
          () => !alive(),
          (j) => {
            const w = jobStepWindow(j);
            patchSide(s, (prev) => {
              const lo = Math.max(prev.lo, w.lo);
              return { job: j, lo, hi: Math.max(lo, w.lo >= prev.lo ? w.hi : prev.hi) };
            });
          },
          res.delivery_id || null
        );
        if (out.kind === "failed") {
          const err = new Error(out.failure.reason) as Error & { failure?: Failure; side?: Side };
          err.failure = out.failure;
          err.side = s;
          throw err;
        }
        if (out.kind === "cancelled") throw new Error("cancelled");
        const id = out.deliveryId || res.delivery_id;
        patchSide(s, { lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length, deliveryId: id });
        return id;
      };

      const [ida, idb] = await Promise.all([follow("a", ra), follow("b", rb)]);
      if (!alive()) return;

      let sessionId = ra.session_id || rb.session_id || null;
      if (!sessionId) {
        try {
          sessionId = (await api.getDelivery(ida)).delivery.session_id;
        } catch {
          sessionId = null;
        }
      }
      if (!sessionId) throw new Error("Both videos were analysed, but we couldn't find the session to combine them in.");

      setPhase("fusing");
      const fused = await api.fuseDual(sessionId, ida, idb, "Two-phone fused");
      if (!alive()) return;
      const sum = summarise(fused);
      setSummary(sum);
      setPhase("done");
    } catch (err: any) {
      if (!alive() || err?.message === "cancelled") return;
      const f: Failure = err?.failure ?? explainFailure(err?.message);
      const which = err?.side ? (err.side === "a" ? "Side-on video: " : "Behind video: ") : "";
      setFailure({ reason: `${which}${f.reason}`, fix: f.fix });
      setPhase("failed");
    }
  };

  const syncWord = (s: number | null) => (s === null ? null : s >= 0.8 ? "strong" : s >= 0.5 ? "fair" : "weak");
  const collinear = summary?.baselineSin !== null && summary?.baselineSin !== undefined && summary.baselineSin < 0.35;

  return (
    <div className="space-y-4">
      <PanelTitle eyebrow="Two phones" title="Film the same ball from two angles at once">
        Side-on shows front-knee bend, trunk forward lean and stride. Behind shows trunk side-bend and hip–shoulder alignment. Together, we line the two
        videos up to a fraction of a frame and combine them into one 3D view, so side-on, behind and twisting movements are all measured at the same
        moment.
      </PanelTitle>

      <div className="grid gap-3 sm:grid-cols-2">
        {(["a", "b"] as Side[]).map((s) => (
          <div key={s} className="space-y-2 min-w-0">
            <div className="flex items-baseline justify-between gap-2">
              <span className="text-[13px] font-[500] text-[#171717]">{LABELS[s]}</span>
              <span className="text-[12px] text-[#8f8f8f]">{s === "a" ? "square of the crease, 8–12 m" : "in line with both sets of stumps"}</span>
            </div>
            <DropZone
              compact
              file={files[s]}
              disabled={busy}
              onFile={(f) => {
                setFiles((p) => ({ ...p, [s]: f }));
                if (phase === "done" || phase === "failed") setPhase("idle");
              }}
              title={s === "a" ? "Choose the side-on video" : "Choose the behind video"}
              hint="Start filming before the run-up on both phones"
            />
            {phase !== "idle" && (
              <PipelineProgress
                lo={sides[s].lo}
                hi={sides[s].hi}
                indeterminate={phase === "uploading" || (phase === "processing" && !sides[s].hasJob && sides[s].lo < PIPELINE_STEPS.length)}
                elapsedS={elapsed}
                title={s === "a" ? "Side-on" : "Behind"}
                message={phase === "uploading" ? "Uploading…" : sides[s].job?.message_plain}
                failed={phase === "failed" && sides[s].lo < PIPELINE_STEPS.length}
              />
            )}
          </div>
        ))}
      </div>

      {phase === "idle" && (
        <div className="v-panel p-4 flex flex-col md:flex-row md:items-end gap-4">
          <div className="flex-1">
            <FpsPicker value={fps} onChange={setFps} />
          </div>
          <div className="flex flex-col gap-1.5 md:items-end">
            {!bowler && <NeedBowler />}
            <button
              type="button"
              disabled={!files.a || !files.b || !bowler || !canAnalyse}
              onClick={start}
              className="v-btn-primary h-10 px-5 text-[14px] w-full md:w-auto"
            >
              Analyse and combine
            </button>
          </div>
        </div>
      )}

      {phase === "fusing" && (
        <div className="v-panel p-4 flex items-center gap-3 text-[13px] text-[#4d4d4d]" aria-live="polite">
          <Spinner />
          Both videos analysed. Lining them up in time and combining them into 3D…
        </div>
      )}

      {failure && <FailureCard failure={failure} onRetry={start} />}

      {phase === "done" && summary && (
        <div className="v-panel p-4 space-y-3">
          <div className="v-label">Combined 3D delivery ready</div>
          <ul className="space-y-1.5 text-[13px] leading-[1.5] text-[#171717]">
            {summary.offsetS !== null && (
              <li>
                Videos lined up in time: phone B started{" "}
                <span className="v-num">{Math.abs(summary.offsetS).toFixed(2)} s</span> {summary.offsetS >= 0 ? "after" : "before"} phone A
                {syncWord(summary.syncScore) && (
                  <>
                    {" "}
                    · match <span className="font-[500]">{syncWord(summary.syncScore)}</span>
                  </>
                )}
                .
              </li>
            )}
            {summary.overlapS !== null && (
              <li>
                Both videos show the same moments for <span className="v-num">{summary.overlapS.toFixed(1)} s</span>.
              </li>
            )}
            {summary.angleDeg !== null && (
              <li>
                The phones were about <span className="v-num">{Math.round(summary.angleDeg)}°</span> apart.
              </li>
            )}
          </ul>
          {syncWord(summary.syncScore) === "weak" && (
            <p className="text-[12.5px] leading-[1.5] text-[#4d4d4d] border-l-2 border-[#171717] pl-3">
              The time match is weak. Check both videos show the same ball from start to finish; the combined ranges are wider to reflect this.
            </p>
          )}
          {collinear && (
            <p className="text-[12.5px] leading-[1.5] text-[#4d4d4d] border-l-2 border-[#171717] pl-3">
              The two phones were looking from almost the same direction, so the extra 3D checks stay switched off for this ball. Next time, place them
              roughly a right angle apart: one side-on, one behind.
            </p>
          )}
          <div className="flex flex-col sm:flex-row gap-2 pt-1">
            {summary.deliveryId && (
              <button
                type="button"
                onClick={() => navigate({ name: "delivery", id: summary.deliveryId as string, tab: "story" })}
                className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center justify-center gap-1.5"
              >
                Open the combined 3D delivery <ArrowRight className="w-3.5 h-3.5" />
              </button>
            )}
            {(["a", "b"] as Side[]).map(
              (s) =>
                sides[s].deliveryId && (
                  <button
                    key={s}
                    type="button"
                    onClick={() => navigate({ name: "delivery", id: sides[s].deliveryId as string, tab: "story" })}
                    className="v-btn-ghost h-9 px-3 text-[13px]"
                  >
                    {s === "a" ? "Side-on only" : "Behind only"}
                  </button>
                )
            )}
          </div>
        </div>
      )}
    </div>
  );
};

export default DualPhonePairing;
