import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  ChevronDown,
  Film,
  Home,
  Plus,
  RotateCcw,
  Upload,
  UserRound,
} from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type { JobStatus, OverviewBowler, OverviewResponse, ScanCandidate } from "../types";
import { Card, SectionTitle, Spinner } from "../components/story/ui";
import { HowToFilmContent } from "../components/story/HowToFilmSheet";

/**
 * Guided New Analysis flow (`#/new?bowler=`), Studio v4 blueprint §2.9.
 * Full-page stepper: Who's bowling → How to film → Upload → Processing → Story tab.
 * Plain language only — no acronyms, no frame numbers, no developer terms.
 */

type Step = 1 | 2 | 3 | 4;
type Arm = "right" | "left";

interface ChosenBowler {
  id: string;
  name: string;
  bowling_arm: Arm | null;
  age_band: string | null;
}

interface Failure {
  reason: string;
  fix: string;
}

const AGE_BANDS = ["U13", "U15", "U17", "U19", "Adult"];
const SKIP_FILM_KEY = "spa.skipFilmGuide";

const STEP_LABELS: Record<Step, string> = {
  1: "Who's bowling?",
  2: "How to film",
  3: "Upload",
  4: "Analysing",
};

const STAGES: Array<{ id: string; label: string }> = [
  { id: "finding_bowler", label: "Finding the bowler" },
  { id: "spotting_moments", label: "Spotting key moments" },
  { id: "measuring", label: "Measuring body positions" },
  { id: "writing_story", label: "Writing your story" },
];

const FPS_OPTIONS: Array<{ value: string; label: string }> = [
  { value: "", label: "Work it out automatically" },
  { value: "240", label: "240 fps slow-motion" },
  { value: "120", label: "120 fps slow-motion" },
  { value: "60", label: "60 fps" },
  { value: "30", label: "30 fps (normal speed)" },
];

function readSkipFilm(): boolean {
  try {
    return window.localStorage.getItem(SKIP_FILM_KEY) === "1";
  } catch {
    return false;
  }
}

function writeSkipFilm(v: boolean) {
  try {
    window.localStorage.setItem(SKIP_FILM_KEY, v ? "1" : "0");
  } catch {
    /* ignore */
  }
}

const sleep = (ms: number) => new Promise((r) => window.setTimeout(r, ms));

function formatClock(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "0:00";
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

function formatSize(bytes: number): string {
  const mb = bytes / (1024 * 1024);
  return mb >= 1 ? `${mb.toFixed(1)} MB` : `${Math.max(1, Math.round(bytes / 1024))} KB`;
}

/** Looks like a server/developer message rather than something a coach should read. */
function looksTechnical(msg: string): boolean {
  return /[_{}<>[\]]|http|traceback|exception|errno|\bnull\b|undefined|\.py\b|stack|json/i.test(msg) || msg.length > 220;
}

/** Turn any error into a plain reason + a practical fix. */
function explainFailure(raw: string | null | undefined, trustedPlain = false): Failure {
  const msg = (raw || "").trim();
  const lower = msg.toLowerCase();
  let reason = "Something went wrong while analysing this video.";
  let fix = "Check the filming tips, then try again.";

  if (/feet|foot|ankle|out of frame|crop|cut off|not fully visible|in.frame/.test(lower)) {
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
    fix = "If you filmed in slow-motion, open Advanced and choose the speed you used.";
  } else if (/pose|gpu|engine|worker|model|nlf/.test(lower)) {
    reason = "The pose engine isn't connected right now, so new videos can't be analysed.";
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

/** Minimal bowler figure seen from behind, with the bowling arm raised. */
const ArmFigure: React.FC<{ arm: Arm; active: boolean }> = ({ arm, active }) => {
  const ink = active ? "#171717" : "#8f8f8f";
  const raised = active ? "#297a3a" : "#8f8f8f";
  const flip = arm === "left";
  return (
    <svg viewBox="0 0 48 56" className="w-12 h-14" aria-hidden>
      <g transform={flip ? "translate(48 0) scale(-1 1)" : undefined}>
        <circle cx={24} cy={10} r={5} fill="none" stroke={ink} strokeWidth={2} />
        <line x1={24} y1={15} x2={24} y2={34} stroke={ink} strokeWidth={2} strokeLinecap="round" />
        {/* lowered arm */}
        <line x1={24} y1={20} x2={14} y2={30} stroke={ink} strokeWidth={2} strokeLinecap="round" />
        {/* bowling arm, raised */}
        <line x1={24} y1={20} x2={36} y2={4} stroke={raised} strokeWidth={2.4} strokeLinecap="round" />
        <circle cx={37.5} cy={2.8} r={2.2} fill={raised} />
        <line x1={24} y1={34} x2={16} y2={52} stroke={ink} strokeWidth={2} strokeLinecap="round" />
        <line x1={24} y1={34} x2={32} y2={52} stroke={ink} strokeWidth={2} strokeLinecap="round" />
      </g>
    </svg>
  );
};

const StepHeader: React.FC<{ step: Step }> = ({ step }) => (
  <div className="mb-5">
    <div className="flex items-center gap-1.5" aria-hidden>
      {([1, 2, 3, 4] as Step[]).map((s) => (
        <div
          key={s}
          className={`h-1 flex-1 rounded-full transition-colors duration-150 ${
            s < step ? "bg-[#297a3a]" : s === step ? "bg-[#171717]" : "bg-[#ebebeb]"
          }`}
        />
      ))}
    </div>
    <div className="mt-2.5 flex items-baseline justify-between gap-3">
      <div className="text-[12px] text-[#8f8f8f]">
        Step <span className="font-mono v-tabular">{step}</span> of <span className="font-mono v-tabular">4</span>
      </div>
      <ol className="hidden sm:flex items-center gap-3 text-[12px]">
        {([1, 2, 3, 4] as Step[]).map((s) => (
          <li
            key={s}
            className={`flex items-center gap-1 ${
              s === step ? "text-[#171717] font-[500]" : s < step ? "text-[#297a3a]" : "text-[#8f8f8f]"
            }`}
          >
            {s < step && <Check className="w-3 h-3" />}
            {STEP_LABELS[s]}
          </li>
        ))}
      </ol>
    </div>
  </div>
);

export const UploadFlowView: React.FC<{ initialBowlerId: string | null }> = ({ initialBowlerId }) => {
  const [step, setStep] = useState<Step>(1);

  // ── Overview (bowlers + engine health) ──────────────────────────────────────
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [overviewLoading, setOverviewLoading] = useState(true);
  const [overviewFailed, setOverviewFailed] = useState(false);

  // ── Step 1: bowler ─────────────────────────────────────────────────────────
  const [bowlerMode, setBowlerMode] = useState<"pick" | "new">("pick");
  const [chosen, setChosen] = useState<ChosenBowler | null>(null);
  const [newName, setNewName] = useState("");
  const [newArm, setNewArm] = useState<Arm | null>(null);
  const [newAge, setNewAge] = useState<string>("");
  const [creating, setCreating] = useState(false);
  const [createError, setCreateError] = useState<string | null>(null);

  // ── Step 2: film guide ─────────────────────────────────────────────────────
  const [skipFilm, setSkipFilm] = useState<boolean>(readSkipFilm);

  // ── Step 3: upload ─────────────────────────────────────────────────────────
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [label, setLabel] = useState("");
  const [isNetSession, setIsNetSession] = useState(false);
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [fpsOverride, setFpsOverride] = useState("");
  const [scanning, setScanning] = useState(false);
  const [scanId, setScanId] = useState<string | null>(null);
  const [candidates, setCandidates] = useState<ScanCandidate[] | null>(null);
  const [selected, setSelected] = useState<number[]>([]);
  const [scanFailure, setScanFailure] = useState<Failure | null>(null);

  // ── Step 4: processing ─────────────────────────────────────────────────────
  const [job, setJob] = useState<JobStatus | null>(null);
  const [processing, setProcessing] = useState(false);
  const [finished, setFinished] = useState(false);
  const [failure, setFailure] = useState<Failure | null>(null);
  const [batchCount, setBatchCount] = useState(1);
  const runIdRef = useRef(0);
  const autoAdvancedRef = useRef(false);

  useEffect(() => {
    return () => {
      runIdRef.current = -1; // stop any polling loop after unmount
    };
  }, []);

  useEffect(() => {
    let alive = true;
    api
      .overview()
      .then((o) => {
        if (!alive) return;
        setOverview(o);
        if (!o.bowlers || o.bowlers.length === 0) setBowlerMode("new");
      })
      .catch(() => {
        if (!alive) return;
        setOverviewFailed(true);
        setBowlerMode("new");
      })
      .finally(() => alive && setOverviewLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const bowlers: OverviewBowler[] = overview?.bowlers ?? [];
  const canAnalyse = overview ? overview.health?.can_analyse_new_video !== false : true;

  // Pre-select the bowler passed in the URL and jump straight ahead.
  useEffect(() => {
    if (autoAdvancedRef.current || overviewLoading || !initialBowlerId) return;
    const b = bowlers.find((x) => x.id === initialBowlerId);
    if (!b) return;
    autoAdvancedRef.current = true;
    setBowlerMode("pick");
    setChosen({ id: b.id, name: b.name, bowling_arm: b.bowling_arm, age_band: b.age_band });
    setStep(readSkipFilm() ? 3 : 2);
  }, [overviewLoading, initialBowlerId, bowlers]);

  const goAfterBowler = () => setStep(skipFilm ? 3 : 2);

  const handleBowlerContinue = async () => {
    setCreateError(null);
    if (bowlerMode === "pick") {
      if (chosen) goAfterBowler();
      return;
    }
    const name = newName.trim();
    if (!name || !newArm || !newAge) return;
    setCreating(true);
    try {
      const res = await api.createBowler({ name, bowling_arm: newArm, age_band: newAge });
      const b = res.bowler;
      setChosen({ id: b.id, name: b.name, bowling_arm: b.bowling_arm ?? newArm, age_band: b.age_band ?? newAge });
      goAfterBowler();
    } catch (err: any) {
      const f = explainFailure(err?.message);
      setCreateError(`We couldn't save this bowler. ${f.fix}`);
    } finally {
      setCreating(false);
    }
  };

  // ── File handling ──────────────────────────────────────────────────────────
  const resetScan = () => {
    setScanId(null);
    setCandidates(null);
    setSelected([]);
    setScanFailure(null);
  };

  const handleFile = (f: File | null) => {
    if (!f) return;
    setFile(f);
    resetScan();
    const stem = f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim().slice(0, 48);
    setLabel(stem || "New delivery");
  };

  const handleScan = async () => {
    if (!file || !chosen) return;
    setScanning(true);
    resetScan();
    try {
      const res = await api.bowlerScanVideo(chosen.id, file);
      setScanId(res.scan_id);
      setCandidates(res.candidates ?? []);
      setSelected((res.candidates ?? []).map((c) => c.candidate_id));
    } catch (err: any) {
      setScanFailure(explainFailure(err?.message));
    } finally {
      setScanning(false);
    }
  };

  const toggleCandidate = (cid: number) =>
    setSelected((prev) => (prev.includes(cid) ? prev.filter((x) => x !== cid) : [...prev, cid]));

  // ── Analysis run + job polling ─────────────────────────────────────────────
  const finish = useCallback((deliveryId: string, runId: number) => {
    if (runIdRef.current !== runId) return;
    setFinished(true);
    setProcessing(false);
    window.setTimeout(() => {
      if (runIdRef.current !== runId) return;
      navigate({ name: "delivery", id: deliveryId, tab: "story" });
    }, 600);
  }, []);

  const fail = useCallback((f: Failure, runId: number) => {
    if (runIdRef.current !== runId) return;
    setFailure(f);
    setProcessing(false);
  }, []);

  const pollJob = useCallback(
    async (jobId: string, fallbackId: string | null, runId: number) => {
      let errors = 0;
      while (runIdRef.current === runId) {
        await sleep(1000);
        if (runIdRef.current !== runId) return;
        try {
          const j = await api.job(jobId);
          if (runIdRef.current !== runId) return;
          errors = 0;
          setJob(j);
          if (j.status === "done") {
            const id = j.delivery_id || fallbackId;
            if (id) finish(id, runId);
            else fail(explainFailure(null), runId);
            return;
          }
          if (j.status === "failed") {
            fail(explainFailure(j.error_plain || j.message_plain, Boolean(j.error_plain)), runId);
            return;
          }
        } catch (err: any) {
          errors += 1;
          if (errors >= 3) {
            // Progress updates unavailable — if we already know the delivery, open it.
            if (fallbackId) finish(fallbackId, runId);
            else fail(explainFailure(err?.message), runId);
            return;
          }
        }
      }
    },
    [finish, fail]
  );

  const chosenCandidates = useMemo(
    () => (candidates ?? []).filter((c) => selected.includes(c.candidate_id)),
    [candidates, selected]
  );
  const useScan = isNetSession && Boolean(scanId) && chosenCandidates.length > 0;

  const runAnalysis = async () => {
    if (!file || !chosen) return;
    const runId = ++runIdRef.current;
    setStep(4);
    setJob(null);
    setFailure(null);
    setFinished(false);
    setProcessing(true);
    const baseLabel = label.trim() || "New delivery";
    const fps = fpsOverride ? Number(fpsOverride) : null;
    try {
      let jobId: string | null = null;
      let deliveryId: string | null = null;
      if (useScan && scanId) {
        setBatchCount(chosenCandidates.length);
        const res = await api.bowlerImportFromScan(chosen.id, {
          scan_id: scanId,
          clips: chosenCandidates.map((c, i) => ({
            start_frame: c.start_frame,
            end_frame: c.end_frame,
            label: chosenCandidates.length > 1 ? `${baseLabel} · Ball ${i + 1}` : baseLabel,
          })),
          bowling_arm: chosen.bowling_arm,
          capture_fps_override: fps,
        });
        jobId = res.job_id ?? null;
        deliveryId = res.deliveries?.[0]?.delivery_id ?? null;
      } else {
        setBatchCount(1);
        const fd = new FormData();
        fd.append("video", file);
        fd.append("label", baseLabel);
        fd.append("start_frame", "0");
        if (chosen.bowling_arm) fd.append("bowling_arm", chosen.bowling_arm);
        if (fps) fd.append("capture_fps_override", String(fps));
        const res = await api.bowlerUploadDelivery(chosen.id, fd);
        jobId = res.job_id ?? null;
        deliveryId = res.delivery_id || null;
      }
      if (runIdRef.current !== runId) return;
      if (jobId) {
        await pollJob(jobId, deliveryId, runId);
      } else if (deliveryId) {
        finish(deliveryId, runId);
      } else {
        fail(explainFailure(null), runId);
      }
    } catch (err: any) {
      fail(explainFailure(err?.message), runId);
    }
  };

  const cancelRun = () => {
    runIdRef.current += 1;
    setProcessing(false);
    setFailure(null);
    setJob(null);
    setStep(3);
  };

  // ── Derived for stage list ─────────────────────────────────────────────────
  const stageIndex = useMemo(() => {
    if (finished) return STAGES.length;
    if (!job) return -1;
    if (job.status === "done") return STAGES.length;
    const i = STAGES.findIndex((s) => s.id === job.stage);
    return i >= 0 ? i : job.status === "queued" ? 0 : -1;
  }, [job, finished]);
  const indeterminate = processing && stageIndex < 0;

  const newBowlerValid = newName.trim().length > 0 && newArm !== null && newAge !== "";
  const step1Ready = bowlerMode === "pick" ? Boolean(chosen) : newBowlerValid;

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="min-h-full bg-[#fafafa]">
      <div className="max-w-[880px] mx-auto px-4 sm:px-6 py-5 sm:py-8">
        <div className="flex items-center justify-between gap-3 mb-4">
          <button
            type="button"
            onClick={() => navigate({ name: "home" })}
            className="v-btn-ghost h-8 px-2.5 text-[13px] inline-flex items-center gap-1.5"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            Home
          </button>
          {chosen && step > 1 && (
            <div className="text-[12.5px] text-[#4d4d4d] truncate min-w-0">
              Analysing <span className="text-[#171717] font-[500]">{chosen.name}</span>
            </div>
          )}
        </div>

        <h1 className="text-[24px] sm:text-[28px] font-[500] tracking-[-0.03em] text-[#171717] mb-4">New analysis</h1>
        <StepHeader step={step} />

        {/* ── STEP 1 — Who's bowling? ─────────────────────────────────────── */}
        {step === 1 && (
          <div className="space-y-4">
            <SectionTitle>Who's bowling?</SectionTitle>

            {overviewLoading ? (
              <Card className="p-6 flex items-center gap-3 text-[13px] text-[#4d4d4d]">
                <Spinner /> Loading your bowlers…
              </Card>
            ) : (
              <>
                {bowlers.length > 0 && (
                  <div className="grid gap-2 sm:grid-cols-2">
                    {bowlers.map((b) => {
                      const active = bowlerMode === "pick" && chosen?.id === b.id;
                      return (
                        <button
                          key={b.id}
                          type="button"
                          aria-pressed={active}
                          onClick={() => {
                            setBowlerMode("pick");
                            setChosen({ id: b.id, name: b.name, bowling_arm: b.bowling_arm, age_band: b.age_band });
                          }}
                          className={`rounded-[8px] border bg-[#ffffff] p-3.5 text-left flex items-center gap-3 transition-colors duration-150 ${
                            active ? "border-[#171717]" : "border-[#ebebeb] hover:border-[#c9c9c9]"
                          }`}
                        >
                          <span
                            className={`w-9 h-9 rounded-full flex items-center justify-center shrink-0 ${
                              active ? "bg-[#171717] text-[#ffffff]" : "bg-[#fafafa] border border-[#ebebeb] text-[#4d4d4d]"
                            }`}
                          >
                            {active ? <Check className="w-4 h-4" /> : <UserRound className="w-4 h-4" />}
                          </span>
                          <span className="min-w-0 flex-1">
                            <span className="block text-[14px] font-[500] text-[#171717] truncate">{b.name}</span>
                            <span className="block text-[12px] text-[#8f8f8f]">
                              {b.bowling_arm === "left" ? "Left-arm" : "Right-arm"}
                              {b.age_band ? ` · ${b.age_band}` : ""} ·{" "}
                              <span className="font-mono v-tabular">{b.delivery_count}</span>{" "}
                              {b.delivery_count === 1 ? "ball" : "balls"} analysed
                            </span>
                          </span>
                        </button>
                      );
                    })}
                    <button
                      type="button"
                      aria-pressed={bowlerMode === "new"}
                      onClick={() => {
                        setBowlerMode("new");
                        setChosen(null);
                      }}
                      className={`rounded-[8px] border bg-[#ffffff] p-3.5 text-left flex items-center gap-3 transition-colors duration-150 ${
                        bowlerMode === "new" ? "border-[#171717]" : "border-dashed border-[#c9c9c9] hover:border-[#171717]"
                      }`}
                    >
                      <span className="w-9 h-9 rounded-full flex items-center justify-center shrink-0 bg-[#fafafa] border border-[#ebebeb] text-[#171717]">
                        <Plus className="w-4 h-4" />
                      </span>
                      <span className="text-[14px] font-[500] text-[#171717]">New bowler</span>
                    </button>
                  </div>
                )}

                {bowlerMode === "new" && (
                  <Card className="p-4 sm:p-5 space-y-5">
                    {bowlers.length === 0 && (
                      <p className="text-[13px] text-[#4d4d4d] leading-[1.55]">
                        {overviewFailed
                          ? "Add the bowler you're filming. We'll keep their deliveries together so you can track progress."
                          : "Let's start with the bowler you're filming. We'll keep their deliveries together so you can track progress."}
                      </p>
                    )}
                    <div className="space-y-1.5">
                      <label htmlFor="new-bowler-name" className="block text-[13px] font-[500] text-[#171717]">
                        Name
                      </label>
                      <input
                        id="new-bowler-name"
                        value={newName}
                        onChange={(e) => setNewName(e.target.value)}
                        placeholder="e.g. Sam Patel"
                        maxLength={80}
                        autoComplete="off"
                        className="v-input w-full h-10 px-3 text-[14px]"
                      />
                    </div>

                    <div className="space-y-1.5">
                      <div className="text-[13px] font-[500] text-[#171717]">Bowling arm</div>
                      <div className="grid grid-cols-2 gap-2">
                        {(["right", "left"] as Arm[]).map((a) => {
                          const active = newArm === a;
                          return (
                            <button
                              key={a}
                              type="button"
                              aria-pressed={active}
                              onClick={() => setNewArm(a)}
                              className={`rounded-[8px] border p-3 sm:p-4 flex flex-col items-center gap-2 transition-colors duration-150 ${
                                active ? "border-[#171717] bg-[#ffffff]" : "border-[#ebebeb] bg-[#fafafa] hover:border-[#c9c9c9]"
                              }`}
                            >
                              <ArmFigure arm={a} active={active} />
                              <span className={`text-[14px] ${active ? "font-[500] text-[#171717]" : "text-[#4d4d4d]"}`}>
                                {a === "right" ? "Right-arm" : "Left-arm"}
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </div>

                    <div className="space-y-1.5">
                      <label htmlFor="new-bowler-age" className="block text-[13px] font-[500] text-[#171717]">
                        Age group
                      </label>
                      <select
                        id="new-bowler-age"
                        value={newAge}
                        onChange={(e) => setNewAge(e.target.value)}
                        className="v-input w-full h-10 px-3 text-[14px]"
                      >
                        <option value="" disabled>
                          Choose an age group
                        </option>
                        {AGE_BANDS.map((b) => (
                          <option key={b} value={b}>
                            {b === "Adult" ? "Adult" : `Under ${b.slice(1)}`}
                          </option>
                        ))}
                      </select>
                      <p className="text-[12px] text-[#8f8f8f]">Used to suggest safe weekly bowling workloads for this age.</p>
                    </div>
                  </Card>
                )}

                {createError && <p className="text-[13px] text-[#171717]">{createError}</p>}

                <div className="flex justify-end pt-1">
                  <button
                    type="button"
                    disabled={!step1Ready || creating}
                    onClick={handleBowlerContinue}
                    className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center gap-2 w-full sm:w-auto justify-center"
                  >
                    {creating ? <Spinner className="!border-[#ffffff]/30 !border-t-[#ffffff]" /> : null}
                    {creating ? "Saving…" : "Continue"}
                    {!creating && <ArrowRight className="w-4 h-4" />}
                  </button>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── STEP 2 — How to film ─────────────────────────────────────────── */}
        {step === 2 && (
          <div className="space-y-4">
            <SectionTitle hint="Takes 30 seconds">How to film</SectionTitle>
            <HowToFilmContent />
            <label className="flex items-center gap-2.5 text-[13px] text-[#4d4d4d] cursor-pointer select-none">
              <input
                type="checkbox"
                checked={skipFilm}
                onChange={(e) => {
                  setSkipFilm(e.target.checked);
                  writeSkipFilm(e.target.checked);
                }}
                className="w-4 h-4 accent-[#171717]"
              />
              Don't show this again
            </label>
            <div className="flex flex-col-reverse sm:flex-row sm:items-center sm:justify-between gap-2 pt-1">
              <button
                type="button"
                onClick={() => setStep(1)}
                className="v-btn-ghost h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
              <div className="flex flex-col-reverse sm:flex-row gap-2">
                <button type="button" onClick={() => setStep(3)} className="v-btn-ghost h-10 px-4 text-[14px]">
                  Skip
                </button>
                <button
                  type="button"
                  onClick={() => setStep(3)}
                  className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
                >
                  Got it, upload a video
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ── STEP 3 — Upload ──────────────────────────────────────────────── */}
        {step === 3 && !canAnalyse && (
          <div className="space-y-4">
            <SectionTitle>Upload a video</SectionTitle>
            <Card className="p-5 sm:p-6">
              <div className="flex items-start gap-3">
                <span className="w-9 h-9 rounded-full bg-[#fafafa] border border-[#ebebeb] flex items-center justify-center shrink-0">
                  <Film className="w-4 h-4 text-[#4d4d4d]" />
                </span>
                <div className="min-w-0">
                  <h3 className="text-[15px] font-[500] text-[#171717]">New videos can't be analysed just yet</h3>
                  <p className="mt-1.5 text-[13px] leading-[1.6] text-[#4d4d4d]">
                    Analysing new videos needs the pose engine, which isn't connected right now. You can explore the sample
                    deliveries meanwhile.
                  </p>
                </div>
              </div>
              <div className="mt-5 flex flex-col-reverse sm:flex-row sm:justify-between gap-2">
                <button
                  type="button"
                  onClick={() => setStep(1)}
                  className="v-btn-ghost h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
                >
                  <ArrowLeft className="w-4 h-4" />
                  Back
                </button>
                <button
                  type="button"
                  onClick={() => navigate({ name: "home" })}
                  className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
                >
                  <Home className="w-4 h-4" />
                  Explore sample deliveries
                </button>
              </div>
            </Card>
          </div>
        )}

        {step === 3 && canAnalyse && (
          <div className="space-y-4">
            <SectionTitle hint={chosen ? `For ${chosen.name}` : undefined}>Upload a video</SectionTitle>

            {/* Drop zone */}
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragOver(true);
              }}
              onDragLeave={() => setDragOver(false)}
              onDrop={(e) => {
                e.preventDefault();
                setDragOver(false);
                handleFile(e.dataTransfer.files?.[0] ?? null);
              }}
              className={`rounded-[8px] border border-dashed p-6 sm:p-8 text-center cursor-pointer transition-colors duration-150 ${
                dragOver ? "border-[#171717] bg-[#ffffff]" : "border-[#c9c9c9] bg-[#ffffff] hover:border-[#171717]"
              }`}
            >
              <input
                ref={fileRef}
                type="file"
                accept="video/*"
                className="hidden"
                onChange={(e) => {
                  handleFile(e.target.files?.[0] ?? null);
                  e.target.value = "";
                }}
              />
              <span className="mx-auto mb-3 w-10 h-10 rounded-full bg-[#fafafa] border border-[#ebebeb] flex items-center justify-center">
                {file ? <Check className="w-4 h-4 text-[#297a3a]" /> : <Upload className="w-4 h-4 text-[#171717]" />}
              </span>
              {file ? (
                <>
                  <div className="text-[14px] font-[500] text-[#171717] break-all">{file.name}</div>
                  <div className="mt-1 text-[12px] text-[#8f8f8f]">
                    <span className="font-mono v-tabular">{formatSize(file.size)}</span> · Tap to choose a different video
                  </div>
                </>
              ) : (
                <>
                  <div className="text-[14px] font-[500] text-[#171717]">Drop your video here, or tap to choose one</div>
                  <div className="mt-1 text-[12px] text-[#8f8f8f]">A video straight from your phone works best</div>
                </>
              )}
            </div>

            {file && (
              <Card className="p-4 space-y-4">
                <div className="space-y-1.5">
                  <label htmlFor="delivery-name" className="block text-[13px] font-[500] text-[#171717]">
                    Name this video <span className="font-[400] text-[#8f8f8f]">(optional)</span>
                  </label>
                  <input
                    id="delivery-name"
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    maxLength={80}
                    className="v-input w-full h-10 px-3 text-[14px]"
                  />
                </div>

                <label className="flex items-start gap-2.5 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={isNetSession}
                    onChange={(e) => {
                      setIsNetSession(e.target.checked);
                      resetScan();
                    }}
                    className="mt-[3px] w-4 h-4 accent-[#171717]"
                  />
                  <span>
                    <span className="block text-[13.5px] text-[#171717]">This video has several balls (net session)</span>
                    <span className="block text-[12px] text-[#8f8f8f]">
                      We'll find each ball for you, then you choose which ones to analyse.
                    </span>
                  </span>
                </label>

                {/* Advanced */}
                <div className="border-t border-[#ebebeb] pt-3">
                  <button
                    type="button"
                    onClick={() => setShowAdvanced((v) => !v)}
                    aria-expanded={showAdvanced}
                    className="inline-flex items-center gap-1.5 text-[13px] text-[#4d4d4d] hover:text-[#171717]"
                  >
                    <ChevronDown className={`w-3.5 h-3.5 transition-transform duration-150 ${showAdvanced ? "rotate-180" : ""}`} />
                    Advanced
                  </button>
                  {showAdvanced && (
                    <div className="mt-3 space-y-1.5">
                      <label htmlFor="fps-override" className="block text-[13px] font-[500] text-[#171717]">
                        Filmed in slow-motion?
                      </label>
                      <select
                        id="fps-override"
                        value={fpsOverride}
                        onChange={(e) => setFpsOverride(e.target.value)}
                        className="v-input w-full sm:w-72 h-10 px-3 text-[14px]"
                      >
                        {FPS_OPTIONS.map((o) => (
                          <option key={o.value} value={o.value}>
                            {o.label}
                          </option>
                        ))}
                      </select>
                      <p className="text-[12px] text-[#8f8f8f] leading-[1.5]">
                        Some phones save slow-motion videos at normal speed. If yours does, pick the speed you filmed at so
                        timings come out right.
                      </p>
                    </div>
                  )}
                </div>
              </Card>
            )}

            {/* Net-session ball picker */}
            {file && isNetSession && (
              <Card className="p-4 space-y-3">
                {scanning ? (
                  <div className="flex items-center gap-3 text-[13px] text-[#4d4d4d]">
                    <Spinner />
                    Looking for each ball in your video… this can take a minute for long sessions.
                  </div>
                ) : candidates === null ? (
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                    <p className="text-[13px] text-[#4d4d4d]">
                      {scanFailure ? (
                        <>
                          <span className="block text-[#171717]">{scanFailure.reason}</span>
                          <span className="block mt-0.5">{scanFailure.fix}</span>
                        </>
                      ) : (
                        "First, let's find the balls in this video."
                      )}
                    </p>
                    <button
                      type="button"
                      onClick={handleScan}
                      className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2 shrink-0"
                    >
                      {scanFailure ? <RotateCcw className="w-4 h-4" /> : <Film className="w-4 h-4" />}
                      {scanFailure ? "Try again" : "Find the balls"}
                    </button>
                  </div>
                ) : candidates.length === 0 ? (
                  <div className="space-y-3">
                    <p className="text-[13px] text-[#4d4d4d] leading-[1.55]">
                      We couldn't find separate balls in this video. You can still analyse it as a single delivery.
                    </p>
                    <button
                      type="button"
                      onClick={() => {
                        setIsNetSession(false);
                        resetScan();
                      }}
                      className="v-btn-ghost h-9 px-3.5 text-[13px]"
                    >
                      Analyse as one ball
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="flex flex-wrap items-baseline justify-between gap-2">
                      <h3 className="text-[14px] font-[500] text-[#171717]">
                        We found <span className="font-mono v-tabular">{candidates.length}</span>{" "}
                        {candidates.length === 1 ? "ball" : "balls"}, pick the ones to analyse
                      </h3>
                      <div className="flex items-center gap-3 text-[12.5px]">
                        <button
                          type="button"
                          onClick={() => setSelected(candidates.map((c) => c.candidate_id))}
                          className="text-[#4d4d4d] hover:text-[#171717] underline-offset-2 hover:underline"
                        >
                          Select all
                        </button>
                        <button
                          type="button"
                          onClick={() => setSelected([])}
                          className="text-[#4d4d4d] hover:text-[#171717] underline-offset-2 hover:underline"
                        >
                          Clear
                        </button>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-2">
                      {candidates.map((c, i) => {
                        const on = selected.includes(c.candidate_id);
                        return (
                          <button
                            key={c.candidate_id}
                            type="button"
                            aria-pressed={on}
                            onClick={() => toggleCandidate(c.candidate_id)}
                            className={`rounded-[8px] border overflow-hidden text-left transition-colors duration-150 ${
                              on ? "border-[#171717] bg-[#ffffff]" : "border-[#ebebeb] bg-[#fafafa] hover:border-[#c9c9c9]"
                            }`}
                          >
                            <div className="relative aspect-video bg-[#f4f4f5]">
                              {c.thumbnail_b64 ? (
                                <img
                                  src={`data:image/jpeg;base64,${c.thumbnail_b64}`}
                                  alt={`Ball ${i + 1}`}
                                  className={`w-full h-full object-cover ${on ? "" : "opacity-60"}`}
                                />
                              ) : (
                                <div className="w-full h-full flex items-center justify-center">
                                  <Film className="w-5 h-5 text-[#a8a8a8]" />
                                </div>
                              )}
                              <span
                                className={`absolute top-1.5 right-1.5 w-5 h-5 rounded-full flex items-center justify-center border ${
                                  on ? "bg-[#171717] border-[#171717] text-[#ffffff]" : "bg-[#ffffff] border-[#c9c9c9]"
                                }`}
                              >
                                {on && <Check className="w-3 h-3" />}
                              </span>
                            </div>
                            <div className="px-2.5 py-2 flex items-baseline justify-between gap-2">
                              <span className="text-[13px] text-[#171717]">Ball {i + 1}</span>
                              <span className="font-mono v-tabular text-[11px] text-[#8f8f8f]">
                                {formatClock(c.start_s)}
                              </span>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </>
                )}
              </Card>
            )}

            <div className="flex flex-col-reverse sm:flex-row sm:items-center sm:justify-between gap-2 pt-1">
              <button
                type="button"
                onClick={() => setStep(skipFilm ? 1 : 2)}
                className="v-btn-ghost h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
              >
                <ArrowLeft className="w-4 h-4" />
                Back
              </button>
              <button
                type="button"
                disabled={!file || !chosen || scanning || (isNetSession && !useScan)}
                onClick={runAnalysis}
                className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
              >
                {isNetSession && useScan
                  ? chosenCandidates.length === 1
                    ? "Analyse 1 ball"
                    : `Analyse ${chosenCandidates.length} balls`
                  : "Analyse this ball"}
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        )}

        {/* ── STEP 4 — Processing ──────────────────────────────────────────── */}
        {step === 4 && (
          <div className="space-y-4">
            <SectionTitle>
              {failure
                ? "We couldn't finish this one"
                : finished
                ? "All done — opening your story"
                : batchCount > 1
                ? `Analysing ${batchCount} balls`
                : "Analysing your delivery"}
            </SectionTitle>

            <Card className="p-4 sm:p-5">
              {/* Progress bar: determinate with job progress, otherwise a calm indeterminate pulse. */}
              {!failure && (
                <div className="h-1 rounded-full bg-[#ebebeb] overflow-hidden mb-5" aria-hidden>
                  {finished ? (
                    <div className="h-full w-full bg-[#297a3a]" />
                  ) : job && job.progress > 0 ? (
                    <div
                      className="h-full bg-[#171717] transition-[width] duration-150 ease-out"
                      style={{ width: `${Math.min(100, Math.max(4, job.progress * 100))}%` }}
                    />
                  ) : (
                    <div className="h-full w-1/3 bg-[#171717] animate-pulse rounded-full" />
                  )}
                </div>
              )}

              <ol className="space-y-3" aria-live="polite">
                {STAGES.map((s, i) => {
                  const done = stageIndex > i;
                  const active = !failure && !indeterminate && stageIndex === i;
                  const failedHere = Boolean(failure) && (stageIndex === i || (stageIndex < 0 && i === 0));
                  return (
                    <li key={s.id} className="flex items-center gap-3">
                      <span className="w-5 h-5 flex items-center justify-center shrink-0">
                        {done ? (
                          <span className="w-5 h-5 rounded-full bg-[#297a3a] flex items-center justify-center">
                            <Check className="w-3 h-3 text-[#ffffff]" />
                          </span>
                        ) : active ? (
                          <Spinner />
                        ) : failedHere ? (
                          <span className="w-5 h-5 rounded-full border border-[#171717] flex items-center justify-center text-[11px] text-[#171717]">
                            !
                          </span>
                        ) : indeterminate ? (
                          <span className="w-2 h-2 rounded-full bg-[#a8a8a8] animate-pulse" />
                        ) : (
                          <span className="w-2 h-2 rounded-full bg-[#ebebeb]" />
                        )}
                      </span>
                      <span
                        className={`text-[14px] ${
                          done ? "text-[#4d4d4d]" : active || failedHere ? "text-[#171717] font-[500]" : "text-[#8f8f8f]"
                        }`}
                      >
                        {s.label}
                      </span>
                    </li>
                  );
                })}
              </ol>

              {!failure && !finished && (
                <p className="mt-5 text-[12.5px] text-[#8f8f8f] leading-[1.55]">
                  {job?.message_plain && !looksTechnical(job.message_plain)
                    ? job.message_plain
                    : "This usually takes under a minute. You can keep this page open while we work."}
                </p>
              )}
            </Card>

            {failure && (
              <Card className="p-4 sm:p-5 space-y-4">
                <div>
                  <p className="text-[14px] font-[500] text-[#171717] leading-[1.5]">{failure.reason}</p>
                  <p className="mt-1.5 text-[13px] text-[#4d4d4d] leading-[1.55]">
                    <span className="text-[#171717]">How to fix it:</span> {failure.fix}
                  </p>
                </div>
                <div className="flex flex-col-reverse sm:flex-row sm:justify-end gap-2">
                  <button type="button" onClick={cancelRun} className="v-btn-ghost h-10 px-4 text-[14px]">
                    Choose a different video
                  </button>
                  <button
                    type="button"
                    onClick={runAnalysis}
                    className="v-btn-primary h-10 px-4 text-[14px] inline-flex items-center justify-center gap-2"
                  >
                    <RotateCcw className="w-4 h-4" />
                    Try again
                  </button>
                </div>
              </Card>
            )}

            {processing && !failure && (
              <div className="flex justify-center">
                <button type="button" onClick={cancelRun} className="v-btn-ghost h-9 px-3.5 text-[13px]">
                  Back to upload
                </button>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadFlowView;
