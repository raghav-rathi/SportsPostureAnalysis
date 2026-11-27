import React, { useCallback, useEffect, useRef, useState } from "react";
import { Check, Upload, X } from "lucide-react";
import { api } from "../../api";
import type { JobStatus, OverviewBowler } from "../../types";

/**
 * Shared capture helpers for the v5 New Analysis studio (docs/research_app_landscape_v5.md §8.6):
 * plain-language failure mapping, monotonic 7-step pipeline progress, job polling, drop zone.
 * Plain words only in primary copy.
 */

export interface Failure {
  reason: string;
  fix: string;
}

export const sleep = (ms: number) => new Promise<void>((r) => window.setTimeout(r, ms));

export function formatClock(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Looks like a server/developer message rather than something a coach should read. */
export function looksTechnical(msg: string): boolean {
  return /[_{}<>[\]]|http|traceback|exception|errno|\bnull\b|undefined|\.py\b|stack|json|sigma|nlf|smpl/i.test(msg) || msg.length > 220;
}

/** Turn any error into a plain reason + a practical fix. */
export function explainFailure(raw: string | null | undefined, trustedPlain = false): Failure {
  const msg = (raw || "").trim();
  const lower = msg.toLowerCase();
  let reason = "Something went wrong while analysing this video.";
  let fix = "Check the filming tips, then try again.";

  if (/collinear|baseline|same direction/.test(lower)) {
    reason = "The two phones were filming from almost the same direction.";
    fix = "Put one phone side-on and the other straight behind the bowler, roughly a right angle apart.";
  } else if (/overlap|same delivery|sync/.test(lower)) {
    reason = "We couldn't line the two videos up in time.";
    fix = "Start both phones before the run-up and keep filming until after the follow-through, on the same ball.";
  } else if (/mismatched bowling arm/.test(lower)) {
    reason = "The two videos don't seem to show the same bowling arm.";
    fix = "Make sure both phones filmed the same bowler on the same ball.";
  } else if (/feet|foot|ankle|out of frame|crop|cut off|not fully visible|in.frame/.test(lower)) {
    reason = "We couldn't see the whole body all the way through the delivery.";
    fix = "Film a little wider so the bowler stays in view from the run-up to the follow-through.";
  } else if (/no (bowler|person|people)|bowler not|couldn.t find the bowler|no track/.test(lower)) {
    reason = "We couldn't find the bowler in this video.";
    fix = "Make sure the bowler is clearly visible and not too far from the camera.";
  } else if (/incomplete|release|follow.through|delivery not/.test(lower)) {
    reason = "This clip doesn't seem to show a full delivery.";
    fix = "Make sure the clip includes the run-up, the ball leaving the hand, and the follow-through.";
  } else if (/fps|frame rate|framerate|timebase|slow.?mo/.test(lower)) {
    reason = "We couldn't work out the video's speed reliably.";
    fix = "If you filmed in slow-motion, choose the speed you used (120 or 240) and try again.";
  } else if (/pose|gpu|engine|worker|model|nlf/.test(lower)) {
    reason = "The analysis engine isn't connected right now, so new videos can't be analysed.";
    fix = "You can explore the sample deliveries meanwhile, and try again later.";
  } else if (/too large|413|file size|payload/.test(lower)) {
    reason = "This video is too big to upload.";
    fix = "Trim it to just the ball you want analysed and try again.";
  } else if (/not found|404|failed to fetch|network|timeout|502|503|504/.test(lower)) {
    reason = "We couldn't reach the analysis service.";
    fix = "Check your connection and try again in a moment.";
  } else if (/format|codec|decode|unsupported|corrupt|could not open/.test(lower)) {
    reason = "We couldn't read this video file.";
    fix = "Try an MP4 or MOV straight from your phone's camera.";
  } else if (msg && (trustedPlain || !looksTechnical(msg))) {
    reason = msg;
  }
  return { reason, fix };
}

// ── 7-step pipeline progress ────────────────────────────────────────────────
export const PIPELINE_STEPS = [
  { id: "audit", label: "Audit", plain: "Checking video quality and speed" },
  { id: "calibrate", label: "Calibrate", plain: "Measuring the pitch from the stumps" },
  { id: "pose", label: "3D pose", plain: "Tracking the bowler in 3D" },
  { id: "events", label: "Events", plain: "Finding foot landings and release" },
  { id: "kinematics", label: "Kinematics", plain: "Measuring body positions" },
  { id: "uncertainty", label: "Uncertainty", plain: "Working out how sure we are" },
  { id: "story", label: "Story", plain: "Writing your story" },
] as const;

/** Active step window for a job snapshot: steps < lo are done, lo..hi are in progress. Done → lo = 7. */
export function jobStepWindow(job: JobStatus | null): { lo: number; hi: number } {
  if (!job) return { lo: -1, hi: -1 };
  if (job.status === "done") return { lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length };
  const p = Number.isFinite(job.progress) ? job.progress : 0;
  switch (job.stage) {
    case "finding_bowler":
      return p < 0.3 ? { lo: 0, hi: 1 } : { lo: 2, hi: 2 };
    case "spotting_moments":
      return { lo: 3, hi: 3 };
    case "measuring":
      return { lo: 4, hi: 5 };
    case "writing_story":
      return { lo: 6, hi: 6 };
    default:
      if (job.status === "queued") return { lo: 0, hi: 0 };
      if (p >= 0.95) return { lo: 6, hi: 6 };
      if (p >= 0.8) return { lo: 4, hi: 5 };
      if (p >= 0.6) return { lo: 3, hi: 3 };
      if (p >= 0.3) return { lo: 2, hi: 2 };
      return { lo: 0, hi: 1 };
  }
}

export type PollOutcome = { kind: "done"; deliveryId: string | null } | { kind: "failed"; failure: Failure } | { kind: "cancelled" };

/** Poll `/api/jobs/{id}` every ~1 s until done / failed / cancelled. */
export async function pollJob(
  jobId: string,
  isCancelled: () => boolean,
  onUpdate: (j: JobStatus) => void,
  fallbackDeliveryId: string | null = null
): Promise<PollOutcome> {
  let errors = 0;
  while (!isCancelled()) {
    await sleep(1000);
    if (isCancelled()) return { kind: "cancelled" };
    try {
      const j = await api.job(jobId);
      if (isCancelled()) return { kind: "cancelled" };
      errors = 0;
      onUpdate(j);
      if (j.status === "done") return { kind: "done", deliveryId: j.delivery_id || fallbackDeliveryId };
      if (j.status === "failed")
        return { kind: "failed", failure: explainFailure(j.error_plain || j.message_plain, Boolean(j.error_plain)) };
    } catch (err: any) {
      errors += 1;
      if (errors >= 3) {
        if (fallbackDeliveryId) return { kind: "done", deliveryId: fallbackDeliveryId };
        return { kind: "failed", failure: explainFailure(err?.message) };
      }
    }
  }
  return { kind: "cancelled" };
}

/** Seconds since `start` (ms epoch), ticking while `running`. */
export function useElapsed(start: number | null, running: boolean): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const t = window.setInterval(() => setNow(Date.now()), 500);
    return () => window.clearInterval(t);
  }, [running]);
  if (start === null) return 0;
  return Math.max(0, ((running ? now : Math.max(now, start)) - start) / 1000);
}

// ── Single-clip analysis run (upload → poll → open the story) ─────────────────
export interface RunState {
  phase: "idle" | "uploading" | "processing" | "done" | "failed";
  job: JobStatus | null;
  hasJob: boolean;
  lo: number; // monotonic step window
  hi: number;
  startedAt: number | null;
  failure: Failure | null;
  deliveryId: string | null;
}

const IDLE: RunState = { phase: "idle", job: null, hasJob: false, lo: -1, hi: -1, startedAt: null, failure: null, deliveryId: null };

export interface UploadOptions {
  label: string;
  fps: number | null;
  autoOpen?: boolean;
}

export function useAnalysisRun(bowler: OverviewBowler | null, onDone?: (deliveryId: string) => void) {
  const [state, setState] = useState<RunState>(IDLE);
  const runRef = useRef(0);

  useEffect(
    () => () => {
      runRef.current = -1;
    },
    []
  );

  const start = useCallback(
    async (file: File, opts: UploadOptions) => {
      if (!bowler) return;
      const runId = ++runRef.current;
      const alive = () => runRef.current === runId;
      setState({ ...IDLE, phase: "uploading", startedAt: Date.now() });
      try {
        const fd = new FormData();
        fd.append("video", file);
        fd.append("label", opts.label.trim() || "New delivery");
        fd.append("start_frame", "0");
        if (bowler.bowling_arm) fd.append("bowling_arm", bowler.bowling_arm);
        if (opts.fps) fd.append("capture_fps_override", String(opts.fps));
        const res = await api.bowlerUploadDelivery(bowler.id, fd);
        if (!alive()) return;
        const jobId = res.job_id ?? null;
        const fallback = res.delivery_id || null;
        const finish = (id: string | null) => {
          if (!alive()) return;
          if (!id) {
            setState((s) => ({ ...s, phase: "failed", failure: explainFailure(null) }));
            return;
          }
          setState((s) => ({ ...s, phase: "done", lo: PIPELINE_STEPS.length, hi: PIPELINE_STEPS.length, deliveryId: id }));
          if (onDone) window.setTimeout(() => alive() && onDone(id), 700);
        };
        if (!jobId) {
          finish(fallback);
          return;
        }
        setState((s) => ({ ...s, phase: "processing", hasJob: true, lo: 0, hi: 0 }));
        const out = await pollJob(
          jobId,
          () => !alive(),
          (j) => {
            const w = jobStepWindow(j);
            setState((s) => {
              const lo = Math.max(s.lo, w.lo);
              return { ...s, job: j, lo, hi: Math.max(lo, w.lo >= s.lo ? w.hi : s.hi) };
            });
          },
          fallback
        );
        if (out.kind === "done") finish(out.deliveryId);
        else if (out.kind === "failed" && alive()) setState((s) => ({ ...s, phase: "failed", failure: out.failure }));
      } catch (err: any) {
        if (alive()) setState((s) => ({ ...s, phase: "failed", failure: explainFailure(err?.message) }));
      }
    },
    [bowler, onDone]
  );

  const reset = useCallback(() => {
    runRef.current += 1;
    setState(IDLE);
  }, []);

  return { state, start, reset };
}

// ── UI pieces ──────────────────────────────────────────────────────────────
export const Spinner: React.FC<{ light?: boolean }> = ({ light }) => (
  <span
    aria-hidden
    className={`inline-block w-3.5 h-3.5 rounded-full border-2 animate-spin ${
      light ? "border-white/30 border-t-white" : "border-[#ebebeb] border-t-[#171717]"
    }`}
  />
);

export const PanelTitle: React.FC<{ eyebrow?: string; title: string; children?: React.ReactNode }> = ({ eyebrow, title, children }) => (
  <div className="mb-4">
    {eyebrow && <div className="v-label mb-1">{eyebrow}</div>}
    <h2 className="text-[17px] font-[500] tracking-[-0.01em] text-[#171717]">{title}</h2>
    {children && <p className="mt-1 text-[13px] leading-[1.55] text-[#4d4d4d]">{children}</p>}
  </div>
);

export const FailureCard: React.FC<{ failure: Failure; onRetry?: () => void; retryLabel?: string }> = ({
  failure,
  onRetry,
  retryLabel = "Try again",
}) => (
  <div className="rounded-[6px] border border-dashed border-[#c9c9c9] bg-white p-4" role="alert">
    <div className="flex items-start gap-2.5">
      <span className="mt-0.5 w-5 h-5 rounded-full bg-[#171717] text-white flex items-center justify-center shrink-0">
        <X className="w-3 h-3" />
      </span>
      <div className="min-w-0">
        <div className="text-[14px] font-[500] text-[#171717]">{failure.reason}</div>
        <div className="mt-1 text-[13px] leading-[1.55] text-[#4d4d4d]">
          <span className="font-[500] text-[#171717]">How to fix it: </span>
          {failure.fix}
        </div>
        {onRetry && (
          <button type="button" onClick={onRetry} className="v-btn-ghost mt-3 h-8 px-3 text-[13px]">
            {retryLabel}
          </button>
        )}
      </div>
    </div>
  </div>
);

/** Staged pipeline progress: Audit → Calibrate → 3D pose → Events → Kinematics → Uncertainty → Story. */
export const PipelineProgress: React.FC<{
  lo: number;
  hi: number;
  indeterminate: boolean;
  elapsedS: number;
  message?: string | null;
  failed?: boolean;
  title?: string;
}> = ({ lo, hi, indeterminate, elapsedS, message, failed, title = "Analysing" }) => {
  const done = lo >= PIPELINE_STEPS.length;
  const pct = done ? 100 : lo < 0 ? 0 : Math.round((lo / PIPELINE_STEPS.length) * 100);
  return (
    <div className="v-panel p-4" aria-live="polite">
      <div className="flex items-baseline justify-between gap-3">
        <div className="v-label">{done ? "Done" : failed ? "Stopped" : title}</div>
        <div className="v-num text-[12px] text-[#8f8f8f]">{formatClock(elapsedS)} elapsed</div>
      </div>
      <div className="mt-2.5 h-1 rounded-full bg-[#ebebeb] overflow-hidden">
        {indeterminate && !failed && !done ? (
          <div className="h-full w-1/3 bg-[#171717] rounded-full animate-pulse" />
        ) : (
          <div
            className={`h-full rounded-full transition-[width] duration-200 ${done ? "bg-[#297a3a]" : "bg-[#171717]"}`}
            style={{ width: `${Math.max(failed ? 0 : 3, pct)}%` }}
          />
        )}
      </div>
      <ol className="mt-3 grid grid-cols-1 sm:grid-cols-7 gap-1.5 sm:gap-1">
        {PIPELINE_STEPS.map((s, i) => {
          const isDone = !indeterminate && i < lo;
          const active = !failed && !indeterminate && i >= lo && i <= hi && !done;
          const failedHere = failed && (i === Math.max(0, lo) || (lo < 0 && i === 0));
          return (
            <li
              key={s.id}
              className={`flex sm:flex-col items-center sm:items-start gap-2 sm:gap-1 rounded-[5px] px-2 py-1.5 ${
                active ? "bg-[#f4f4f5]" : ""
              }`}
            >
              <span
                className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 text-[10px] ${
                  isDone
                    ? "bg-[#297a3a] text-white"
                    : failedHere
                    ? "bg-[#171717] text-white"
                    : active
                    ? "border-2 border-[#171717] animate-pulse"
                    : "border border-[#c9c9c9]"
                }`}
              >
                {isDone ? <Check className="w-2.5 h-2.5" /> : failedHere ? <X className="w-2.5 h-2.5" /> : null}
              </span>
              <span className="min-w-0">
                <span className={`block text-[12px] ${isDone || active ? "text-[#171717] font-[500]" : "text-[#8f8f8f]"}`}>
                  {s.label}
                </span>
                <span className="block sm:hidden text-[12px] text-[#8f8f8f]">{s.plain}</span>
              </span>
            </li>
          );
        })}
      </ol>
      <p className="mt-3 text-[12.5px] text-[#4d4d4d] leading-[1.5]">
        {done
          ? "Analysis complete."
          : message && !looksTechnical(message)
          ? message
          : indeterminate
          ? "Uploading and analysing. Live step updates aren't available for this run, so this may take a minute or two."
          : lo >= 0 && lo < PIPELINE_STEPS.length
          ? `${PIPELINE_STEPS[lo].plain}…`
          : "Uploading your video…"}
        {!done && !failed && <span className="text-[#8f8f8f]"> Usually 1–3 minutes on the fast analysis computer.</span>}
      </p>
    </div>
  );
};

/** Drag-and-drop + file picker. */
export const DropZone: React.FC<{
  file: File | null;
  onFile: (f: File) => void;
  title?: string;
  hint?: string;
  compact?: boolean;
  disabled?: boolean;
  id?: string;
}> = ({ file, onFile, title = "Drop your video here, or tap to choose one", hint = "A video straight from your phone works best", compact, disabled, id }) => {
  const ref = useRef<HTMLInputElement | null>(null);
  const [over, setOver] = useState(false);
  const open = () => !disabled && ref.current?.click();
  return (
    <div
      id={id}
      role="button"
      tabIndex={disabled ? -1 : 0}
      aria-disabled={disabled}
      onClick={open}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          open();
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        if (!disabled) setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files?.[0];
        if (f && !disabled) onFile(f);
      }}
      className={`rounded-[6px] border border-dashed bg-white text-center transition-colors duration-150 ${
        compact ? "p-4" : "p-6 sm:p-8"
      } ${disabled ? "opacity-50 cursor-not-allowed border-[#c9c9c9]" : over ? "border-[#171717] cursor-pointer" : "border-[#c9c9c9] hover:border-[#171717] cursor-pointer"}`}
    >
      <input
        ref={ref}
        type="file"
        accept="video/*"
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
      <span className="mx-auto mb-2.5 w-9 h-9 rounded-full bg-[#fafafa] border border-[#ebebeb] flex items-center justify-center">
        {file ? <Check className="w-4 h-4 text-[#297a3a]" /> : <Upload className="w-4 h-4 text-[#171717]" />}
      </span>
      {file ? (
        <>
          <div className="text-[13.5px] font-[500] text-[#171717] break-all">{file.name}</div>
          <div className="mt-1 text-[12px] text-[#8f8f8f]">
            <span className="v-num">{formatSize(file.size)}</span> · Tap to choose a different video
          </div>
        </>
      ) : (
        <>
          <div className="text-[13.5px] font-[500] text-[#171717]">{title}</div>
          <div className="mt-1 text-[12px] text-[#8f8f8f]">{hint}</div>
        </>
      )}
    </div>
  );
};

export type FpsChoice = "auto" | "120" | "240";
export const FPS_CHOICES: Array<{ value: FpsChoice; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "120", label: "120 fps" },
  { value: "240", label: "240 fps" },
];
export const fpsValue = (c: FpsChoice): number | null => (c === "auto" ? null : Number(c));

export const FpsPicker: React.FC<{ value: FpsChoice; onChange: (v: FpsChoice) => void }> = ({ value, onChange }) => (
  <div>
    <div className="text-[13px] font-[500] text-[#171717] mb-1.5">Filming speed</div>
    <div className="v-seg" role="group" aria-label="Filming speed">
      {FPS_CHOICES.map((c) => (
        <button key={c.value} type="button" aria-pressed={value === c.value} onClick={() => onChange(c.value)}>
          <span className={c.value === "auto" ? "" : "v-num"}>{c.label}</span>
        </button>
      ))}
    </div>
    <p className="mt-1.5 text-[12px] text-[#8f8f8f]">Pick the slow-motion speed you filmed at if you know it. Auto works for most phones.</p>
  </div>
);

export const NeedBowler: React.FC = () => (
  <p className="text-[12.5px] text-[#8f8f8f]">Choose who's bowling at the top of the page first.</p>
);
