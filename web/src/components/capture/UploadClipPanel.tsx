import React, { useCallback, useState } from "react";
import type { OverviewBowler } from "../../types";
import { navigate } from "../../router";
import {
  DropZone,
  FailureCard,
  FpsPicker,
  NeedBowler,
  PanelTitle,
  PipelineProgress,
  fpsValue,
  useAnalysisRun,
  useElapsed,
  type FpsChoice,
} from "./shared";

type ClipAngle = "side" | "behind" | "front" | "auto";
const ANGLES: Array<{ value: ClipAngle; label: string; hint: string }> = [
  { value: "side", label: "Side-on", hint: "Side-on clip" },
  { value: "behind", label: "Behind", hint: "Behind clip" },
  { value: "front", label: "Front", hint: "Front clip" },
  { value: "auto", label: "Auto", hint: "" },
];

/** Upload one clip → staged pipeline progress → open the story. */
export const UploadClipPanel: React.FC<{ bowler: OverviewBowler | null; canAnalyse: boolean }> = ({ bowler, canAnalyse }) => {
  const [file, setFile] = useState<File | null>(null);
  const [angle, setAngle] = useState<ClipAngle>("side");
  const [fps, setFps] = useState<FpsChoice>("auto");
  const [name, setName] = useState("");

  const onDone = useCallback((id: string) => navigate({ name: "delivery", id, tab: "story" }), []);
  const run = useAnalysisRun(bowler, onDone);
  const active = run.state.phase === "uploading" || run.state.phase === "processing";
  const elapsed = useElapsed(run.state.startedAt, active);
  const running = run.state.phase !== "idle";

  const handleFile = (f: File) => {
    setFile(f);
    run.reset();
    const stem = f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").trim().slice(0, 48);
    setName(stem);
  };

  const submit = () => {
    if (!file) return;
    const hint = ANGLES.find((a) => a.value === angle)?.hint ?? "";
    const base = name.trim();
    const label = hint ? (base ? `${base} · ${hint}` : hint) : base || "New delivery";
    void run.start(file, { label, fps: fpsValue(fps) });
  };

  return (
    <div className="space-y-4">
      <PanelTitle eyebrow="Upload clip" title="Analyse one delivery from your phone">
        Best results: your phone's slow-motion camera (120–240 fps), whole body in view, at least one set of stumps visible.
      </PanelTitle>

      <div className="grid gap-4 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
        <DropZone file={file} onFile={handleFile} disabled={active} />
        <div className="v-panel p-4 space-y-4">
          <div>
            <div className="text-[13px] font-[500] text-[#171717] mb-1.5">Where was the camera?</div>
            <div className="v-seg flex-wrap" role="group" aria-label="Camera angle">
              {ANGLES.map((a) => (
                <button key={a.value} type="button" aria-pressed={angle === a.value} onClick={() => setAngle(a.value)}>
                  {a.label}
                </button>
              ))}
            </div>
            <p className="mt-1.5 text-[12px] text-[#8f8f8f]">We still work the angle out from the stumps; this just labels the clip.</p>
          </div>
          <FpsPicker value={fps} onChange={setFps} />
          <div>
            <label htmlFor="clip-name" className="block text-[13px] font-[500] text-[#171717] mb-1.5">
              Name <span className="font-[400] text-[#8f8f8f]">(optional)</span>
            </label>
            <input
              id="clip-name"
              value={name}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
              placeholder="e.g. Nets, ball 3"
              className="v-input w-full h-9 px-3 text-[13.5px]"
            />
          </div>
        </div>
      </div>

      {!running && (
        <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:justify-end">
          {!bowler && <NeedBowler />}
          <button
            type="button"
            disabled={!file || !bowler || !canAnalyse}
            onClick={submit}
            className="v-btn-primary h-10 px-5 text-[14px] w-full sm:w-auto"
          >
            Analyse this clip
          </button>
        </div>
      )}

      {running && (
        <>
          <PipelineProgress
            lo={run.state.lo}
            hi={run.state.hi}
            indeterminate={run.state.phase === "uploading" || (run.state.phase === "processing" && !run.state.hasJob)}
            elapsedS={elapsed}
            message={run.state.phase === "uploading" ? "Uploading your video…" : run.state.job?.message_plain}
            failed={run.state.phase === "failed"}
          />
          {run.state.failure && <FailureCard failure={run.state.failure} onRetry={submit} />}
          {run.state.phase === "done" && run.state.deliveryId && (
            <p className="text-[13px] text-[#4d4d4d]">Opening your story…</p>
          )}
        </>
      )}
    </div>
  );
};

export default UploadClipPanel;
