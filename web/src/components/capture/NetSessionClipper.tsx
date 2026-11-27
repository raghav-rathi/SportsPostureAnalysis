import React, { useEffect, useMemo, useRef, useState } from "react";
import { Check, ImageOff } from "lucide-react";
import { api } from "../../api";
import type { JobStatus, OverviewBowler, ScanCandidate } from "../../types";
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
  formatClock,
  fpsValue,
  jobStepWindow,
  pollJob,
  useElapsed,
  type Failure,
  type FpsChoice,
} from "./shared";

type Phase = "idle" | "scanning" | "picking" | "importing" | "processing" | "done" | "failed";

/** Net-session auto-clipper: long video → detected deliveries → pick → batch analyse. */
export const NetSessionClipper: React.FC<{ bowler: OverviewBowler | null; canAnalyse: boolean }> = ({ bowler, canAnalyse }) => {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<Phase>("idle");
  const [scanId, setScanId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ScanCandidate[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [fps, setFps] = useState<FpsChoice>("auto");
  const [baseLabel, setBaseLabel] = useState("Net session");
  const [job, setJob] = useState<JobStatus | null>(null);
  const [win, setWin] = useState<{ lo: number; hi: number }>({ lo: -1, hi: -1 });
  const [hasJob, setHasJob] = useState(false);
  const [deliveries, setDeliveries] = useState<Array<{ delivery_id: string; label: string }>>([]);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [startedAt, setStartedAt] = useState<number | null>(null);
  const runRef = useRef(0);

  useEffect(
    () => () => {
      runRef.current = -1;
    },
    []
  );

  const busy = phase === "scanning" || phase === "importing" || phase === "processing";
  const elapsed = useElapsed(startedAt, busy);
  const maxScore = useMemo(() => Math.max(1e-6, ...candidates.map((c) => c.motion_score || 0)), [candidates]);
  const chosen = useMemo(() => candidates.filter((c) => selected.has(c.candidate_id)), [candidates, selected]);

  const reset = () => {
    runRef.current += 1;
    setPhase("idle");
    setScanId(null);
    setCandidates([]);
    setSelected(new Set());
    setJob(null);
    setWin({ lo: -1, hi: -1 });
    setDeliveries([]);
    setFailure(null);
  };

  const scan = async () => {
    if (!file || !bowler) return;
    const runId = ++runRef.current;
    setPhase("scanning");
    setFailure(null);
    setStartedAt(Date.now());
    try {
      const res = await api.bowlerScanVideo(bowler.id, file);
      if (runRef.current !== runId) return;
      const cands = res.candidates ?? [];
      setScanId(res.scan_id);
      setSessionId(res.session_id ?? null);
      setCandidates(cands);
      setSelected(new Set(cands.map((c) => c.candidate_id)));
      if (cands.length === 0) {
        setFailure({
          reason: "We couldn't find any deliveries in this video.",
          fix: "Make sure the bowler is in view for each ball, and that the camera is still. You can also trim one ball and use Upload clip.",
        });
        setPhase("failed");
      } else setPhase("picking");
    } catch (err: any) {
      if (runRef.current !== runId) return;
      setFailure(explainFailure(err?.message));
      setPhase("failed");
    }
  };

  const toggle = (id: number) =>
    setSelected((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });

  const analyse = async () => {
    if (!bowler || !scanId || chosen.length === 0) return;
    const runId = ++runRef.current;
    const alive = () => runRef.current === runId;
    setPhase("importing");
    setFailure(null);
    setStartedAt(Date.now());
    setWin({ lo: -1, hi: -1 });
    const base = baseLabel.trim() || "Net session";
    try {
      const res = await api.bowlerImportFromScan(bowler.id, {
        scan_id: scanId,
        clips: chosen.map((c, i) => ({ start_frame: c.start_frame, end_frame: c.end_frame, label: `${base} · Ball ${i + 1}` })),
        bowling_arm: bowler.bowling_arm,
        capture_fps_override: fpsValue(fps),
      });
      if (!alive()) return;
      setDeliveries(res.deliveries ?? []);
      if (res.session_id) setSessionId(res.session_id);
      const jobId = res.job_id ?? null;
      if (!jobId) {
        setWin({ lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length });
        setPhase("done");
        return;
      }
      setHasJob(true);
      setPhase("processing");
      setWin({ lo: 0, hi: 0 });
      const out = await pollJob(
        jobId,
        () => !alive(),
        (j) => {
          setJob(j);
          const w = jobStepWindow(j);
          setWin((s) => {
            const lo = Math.max(s.lo, w.lo);
            return { lo, hi: Math.max(lo, w.lo >= s.lo ? w.hi : s.hi) };
          });
        },
        res.deliveries?.[0]?.delivery_id ?? null
      );
      if (!alive()) return;
      if (out.kind === "done") {
        setWin({ lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length });
        setPhase("done");
      } else if (out.kind === "failed") {
        setFailure(out.failure);
        setPhase("failed");
      }
    } catch (err: any) {
      if (!alive()) return;
      setFailure(explainFailure(err?.message));
      setPhase("failed");
    }
  };

  const showGrid = candidates.length > 0 && (phase === "picking" || phase === "importing" || phase === "processing" || (phase === "failed" && deliveries.length === 0));

  return (
    <div className="space-y-4">
      <PanelTitle eyebrow="Net session" title="Find every ball in a long video">
        Upload a whole net session. We scan it, show you each delivery we found, and you pick which ones to analyse.
      </PanelTitle>

      {(phase === "idle" || phase === "scanning" || (phase === "failed" && candidates.length === 0)) && (
        <>
          <DropZone
            file={file}
            onFile={(f) => {
              reset();
              setFile(f);
            }}
            title="Drop your net session video here, or tap to choose"
            hint="Keep the camera still for the whole session. Long videos are fine."
            disabled={phase === "scanning"}
          />
          {phase === "scanning" ? (
            <div className="v-panel p-4 flex items-center gap-3 text-[13px] text-[#4d4d4d]" aria-live="polite">
              <Spinner />
              <span className="flex-1">Looking through your video for each delivery…</span>
              <span className="v-num text-[12px] text-[#8f8f8f]">{formatClock(elapsed)}</span>
            </div>
          ) : (
            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:justify-end">
              {!bowler && <NeedBowler />}
              <button type="button" disabled={!file || !bowler} onClick={scan} className="v-btn-primary h-10 px-5 text-[14px] w-full sm:w-auto">
                Find deliveries
              </button>
            </div>
          )}
        </>
      )}

      {failure && <FailureCard failure={failure} onRetry={candidates.length > 0 && scanId ? analyse : scan} />}

      {showGrid && (
        <div className="space-y-3">
          <div className="flex flex-wrap items-center gap-2">
            <div className="text-[13.5px] text-[#171717] mr-auto">
              Found <span className="v-num font-[500]">{candidates.length}</span> {candidates.length === 1 ? "delivery" : "deliveries"} ·{" "}
              <span className="v-num font-[500]">{chosen.length}</span> selected
            </div>
            <button
              type="button"
              disabled={busy}
              onClick={() => setSelected(new Set(candidates.map((c) => c.candidate_id)))}
              className="v-btn-ghost h-8 px-3 text-[12.5px]"
            >
              Select all
            </button>
            <button type="button" disabled={busy} onClick={() => setSelected(new Set())} className="v-btn-ghost h-8 px-3 text-[12.5px]">
              Select none
            </button>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
            {candidates.map((c, i) => {
              const on = selected.has(c.candidate_id);
              const score = Math.max(0, Math.min(1, (c.motion_score || 0) / maxScore));
              return (
                <button
                  key={c.candidate_id}
                  type="button"
                  aria-pressed={on}
                  disabled={busy}
                  onClick={() => toggle(c.candidate_id)}
                  className={`text-left rounded-[6px] bg-white overflow-hidden border transition-colors duration-150 ${
                    on ? "border-[#171717]" : "border-[#ebebeb] hover:border-[#c9c9c9]"
                  } ${on ? "" : "opacity-70"}`}
                >
                  <div className="relative aspect-video bg-[#f4f4f5]">
                    {c.thumbnail_b64 ? (
                      <img src={`data:image/jpeg;base64,${c.thumbnail_b64}`} alt={`Delivery ${i + 1}`} className="absolute inset-0 w-full h-full object-cover" />
                    ) : (
                      <div className="absolute inset-0 flex items-center justify-center text-[#8f8f8f]">
                        <ImageOff className="w-4 h-4" />
                      </div>
                    )}
                    <span
                      className={`absolute top-1.5 right-1.5 w-5 h-5 rounded-full flex items-center justify-center ${
                        on ? "bg-[#171717] text-white" : "bg-white/90 border border-[#c9c9c9]"
                      }`}
                    >
                      {on && <Check className="w-3 h-3" />}
                    </span>
                  </div>
                  <div className="p-2">
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="text-[12.5px] font-[500] text-[#171717]">Ball {i + 1}</span>
                      <span className="v-num text-[11.5px] text-[#4d4d4d]">
                        {formatClock(c.start_s)}–{formatClock(c.end_s)}
                      </span>
                    </div>
                    <div className="mt-1.5 flex items-center gap-1.5">
                      <span className="text-[10.5px] text-[#8f8f8f] shrink-0">Movement</span>
                      <span className="flex-1 h-1 rounded-full bg-[#ebebeb] overflow-hidden">
                        <span className="block h-full bg-[#171717]" style={{ width: `${Math.round(score * 100)}%` }} />
                      </span>
                    </div>
                  </div>
                </button>
              );
            })}
          </div>

          {phase === "picking" && (
            <div className="v-panel p-4 grid gap-4 md:grid-cols-[1fr_1fr_auto] md:items-end">
              <div>
                <label htmlFor="net-label" className="block text-[13px] font-[500] text-[#171717] mb-1.5">
                  Name
                </label>
                <input
                  id="net-label"
                  value={baseLabel}
                  maxLength={60}
                  onChange={(e) => setBaseLabel(e.target.value)}
                  className="v-input w-full h-9 px-3 text-[13.5px]"
                />
                <p className="mt-1.5 text-[12px] text-[#8f8f8f]">Each ball is saved as “{baseLabel.trim() || "Net session"} · Ball 1”, “· Ball 2”…</p>
              </div>
              <FpsPicker value={fps} onChange={setFps} />
              <button
                type="button"
                disabled={chosen.length === 0 || !canAnalyse || !bowler}
                onClick={analyse}
                className="v-btn-primary h-10 px-5 text-[14px] w-full md:w-auto"
              >
                Analyse <span className="v-num">{chosen.length}</span> {chosen.length === 1 ? "ball" : "balls"}
              </button>
            </div>
          )}
        </div>
      )}

      {(phase === "importing" || phase === "processing" || phase === "done" || (phase === "failed" && deliveries.length > 0)) && (
        <PipelineProgress
          lo={win.lo}
          hi={win.hi}
          indeterminate={phase === "importing" || (phase === "processing" && !hasJob)}
          elapsedS={elapsed}
          title={`Analysing ${chosen.length || deliveries.length} ${(chosen.length || deliveries.length) === 1 ? "ball" : "balls"}`}
          message={phase === "importing" ? "Cutting out the balls you picked…" : job?.message_plain}
          failed={phase === "failed"}
        />
      )}

      {(phase === "done" || (phase === "failed" && deliveries.length > 0)) && (
        <div className="v-panel p-4">
          <div className="v-label mb-2">{phase === "done" ? "Your deliveries" : "Saved so far"}</div>
          <ul className="divide-y divide-[#ebebeb]">
            {deliveries.map((d) => (
              <li key={d.delivery_id} className="py-2 flex items-center justify-between gap-3">
                <span className="text-[13px] text-[#171717] truncate min-w-0">{d.label}</span>
                <button
                  type="button"
                  onClick={() => navigate({ name: "delivery", id: d.delivery_id, tab: "story" })}
                  className="v-btn-ghost h-7 px-2.5 text-[12px] shrink-0"
                >
                  Open
                </button>
              </li>
            ))}
          </ul>
          <div className="mt-3 flex flex-col sm:flex-row gap-2">
            <button
              type="button"
              onClick={() => navigate({ name: "deliveries", bowler: bowler?.id ?? null })}
              className="v-btn-primary h-9 px-4 text-[13px]"
            >
              See all deliveries
            </button>
            <button
              type="button"
              onClick={() => navigate({ name: "spell", bowler: bowler?.id ?? null, session: sessionId })}
              className="v-btn-ghost h-9 px-4 text-[13px]"
            >
              View this spell
            </button>
            <button
              type="button"
              onClick={() => {
                reset();
                setFile(null);
              }}
              className="v-btn-ghost h-9 px-4 text-[13px] sm:ml-auto"
            >
              Scan another video
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default NetSessionClipper;
