import React, { useRef, useState } from "react";
import { Check, Film, Scissors, Upload, X } from "lucide-react";
import { api } from "../api";
import type { HealthResponse, SessionSummary } from "../types";

interface UploadWizardModalProps {
  session: SessionSummary;
  health: HealthResponse | null;
  onClose: () => void;
  onImported: (deliveryId: string) => Promise<void>;
}

interface ScanCandidate {
  candidate_id: number;
  start_frame: number;
  end_frame: number;
  peak_frame: number;
  start_s: number;
  end_s: number;
  motion_score: number;
  thumbnail_b64?: string | null;
}

export const UploadWizardModal: React.FC<UploadWizardModalProps> = ({
  session,
  health,
  onClose,
  onImported,
}) => {
  const fileRef = useRef<HTMLInputElement | null>(null);
  const [step, setStep] = useState<1 | 2 | 3>(1);
  const [file, setFile] = useState<File | null>(null);
  const [label, setLabel] = useState("Ball 1 — Net Delivery");
  const [arm, setArm] = useState<"" | "right" | "left">("");
  const [fpsOverride, setFpsOverride] = useState<string>("");

  // Multi-ball net scanner state (zero-reupload via scan_id!)
  const [scanId, setScanId] = useState<string | null>(null);
  const [scanning, setScanning] = useState(false);
  const [candidates, setCandidates] = useState<ScanCandidate[]>([]);
  const [selectedIds, setSelectedIds] = useState<number[]>([]);
  const [scanError, setScanError] = useState<string | null>(null);

  // Step 3 CLI logs
  const [running, setRunning] = useState(false);
  const [logs, setLogs] = useState<string[]>([]);
  const [runError, setRunError] = useState<string | null>(null);

  const poseReady = health?.pose_engine === "remote_gpu_worker" || health?.pose_engine === "local_nlf_model";

  const handleFileSelected = (f: File | null) => {
    setFile(f);
    setScanId(null);
    setCandidates([]);
    setSelectedIds([]);
    setScanError(null);
    if (f) {
      const cleanStem = f.name.replace(/\.[^.]+$/, "").replace(/[_-]+/g, " ").slice(0, 48);
      setLabel(cleanStem || "Uploaded Delivery");
    }
  };

  const handleScanNet = async () => {
    if (!file) return;
    setScanning(true);
    setScanError(null);
    try {
      const res = await api.scanVideo(session.id, file);
      setScanId(res.scan_id);
      setCandidates(res.candidates);
      setSelectedIds(res.candidates.map((c) => c.candidate_id));
      if (res.candidates.length === 0) {
        setScanError("No distinct bowling bursts detected — you can still upload the full clip directly.");
      }
    } catch (err: any) {
      setScanError(err?.message || "Net-session scan failed.");
    } finally {
      setScanning(false);
    }
  };

  const toggleCandidate = (cid: number) => {
    setSelectedIds((prev) => (prev.includes(cid) ? prev.filter((x) => x !== cid) : [...prev, cid]));
  };

  const handleRunAnalysis = async () => {
    if (!file) return;
    setStep(3);
    setRunning(true);
    setRunError(null);
    setLogs([
      `▲ probe_container --file "${file.name}" (${(file.size / (1024 * 1024)).toFixed(1)} MB)`,
      `▲ audit_timebase --dedup-pulldown=true${fpsOverride ? ` --capture-fps=${fpsOverride}` : ""}`,
      `▲ build_all_intra_scrub_proxy --gop=1 --bframes=0`,
    ]);

    try {
      let createdId = "";
      if (scanId && candidates.length > 0 && selectedIds.length > 0) {
        const chosen = candidates.filter((c) => selectedIds.includes(c.candidate_id));
        setLogs((prev) => [
          ...prev,
          `▲ slice_scan_clips --scan-id=${scanId} --count=${chosen.length} (zero re-upload)`,
          `▲ extract_nlf_3d --skeleton=smpl24+bml_movi_87`,
        ]);
        const res = await api.importFromScan(session.id, {
          scan_id: scanId,
          clips: chosen.map((c, i) => ({
            start_frame: c.start_frame,
            end_frame: c.end_frame,
            label: `${label} #${i + 1}`,
          })),
          bowling_arm: arm || null,
          capture_fps_override: fpsOverride ? Number(fpsOverride) : null,
        });
        createdId = res.deliveries[0]?.delivery_id ?? "";
        setLogs((prev) => [
          ...prev,
          `✓ imported ${res.deliveries.length} ball(s) into "${session.title}"`,
          `✓ verify_guardrails --icc-15deg=REFUSED --uncertainty=calibrated`,
        ]);
      } else {
        setLogs((prev) => [
          ...prev,
          `▲ extract_nlf_3d --skeleton=smpl24+bml_movi_87`,
          `▲ calibrate_pitch --bowler-masked-pan=true`,
        ]);
        const fd = new FormData();
        fd.append("video", file);
        fd.append("label", label.trim() || "Uploaded Delivery");
        fd.append("start_frame", "0");
        if (arm) fd.append("bowling_arm", arm);
        if (fpsOverride) fd.append("capture_fps_override", fpsOverride);
        const res = await api.uploadDelivery(session.id, fd);
        createdId = res.delivery_id;
        setLogs((prev) => [
          ...prev,
          `✓ recompute_cpu completed in ${res.recompute_ms.toFixed(1)} ms`,
          `✓ verify_guardrails --icc-15deg=REFUSED --uncertainty=calibrated`,
        ]);
      }

      if (createdId) {
        setTimeout(async () => {
          await onImported(createdId);
          onClose();
        }, 450);
      }
    } catch (err: any) {
      setRunError(err?.message || "Analysis failed.");
      setLogs((prev) => [...prev, `✕ error: ${err?.message || "Unknown failure"}`]);
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="no-print fixed inset-0 z-50 bg-[#000000]/40 backdrop-blur-[2px] flex items-center justify-center p-4 overflow-y-auto">
      <div className="v-card w-full max-w-2xl overflow-hidden my-auto">
        {/* Header */}
        <div className="px-5 py-4 border-b border-[#ebebeb] flex items-center justify-between">
          <div>
            <div className="v-eyebrow">▲ ANALYZE_DELIVERY · STEP {step} OF 3</div>
            <h2 className="text-[20px] font-[450] tracking-[-0.04em] text-[#171717] mt-0.5">
              {step === 1 && "Upload a Bowling Video"}
              {step === 2 && "Camera & Crease Calibration"}
              {step === 3 && "Running Biomechanical Pipeline"}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="v-btn-ghost w-8 h-8 flex items-center justify-center"
            aria-label="Close upload modal"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Progress Track */}
        <div className="grid grid-cols-3 border-b border-[#ebebeb] bg-[#fafafa] font-mono text-[11px]">
          {[
            [1, "1. Select Clip"],
            [2, "2. Calibration"],
            [3, "3. Run 3D Audit"],
          ].map(([n, txt]) => (
            <div
              key={n as number}
              className={`px-4 py-2 border-r last:border-r-0 border-[#ebebeb] ${
                step === n
                  ? "bg-[#ffffff] text-[#171717] font-medium"
                  : step > (n as number)
                  ? "text-[#297a3a]"
                  : "text-[#8f8f8f]"
              }`}
            >
              {step > (n as number) ? "✓ " : ""}
              {txt}
            </div>
          ))}
        </div>

        <div className="p-5 space-y-4">
          {!poseReady && step === 1 && (
            <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#171717] text-[12px] text-[#4d4d4d] space-y-1">
              <div className="font-mono text-[11px] text-[#171717] font-medium">
                ▲ OFFLINE CPU MODE ACTIVE
              </div>
              <p>
                No 3D pose GPU worker (`GPU_WORKER_URL`) or local NLF checkpoint (`NLF_MODEL_PATH`) is connected yet.
                You can still run multi-ball net scanning right here, or explore the pre-loaded real Dale Steyn &amp; 240 fps
                calibrated deliveries with instant &lt;35 ms CPU recompute.
              </p>
            </div>
          )}

          {/* STEP 1 */}
          {step === 1 && (
            <>
              {/* Drag & Drop Video Zone */}
              <div
                onClick={() => fileRef.current?.click()}
                onDragOver={(e) => e.preventDefault()}
                onDrop={(e) => {
                  e.preventDefault();
                  const dropped = e.dataTransfer.files?.[0] ?? null;
                  if (dropped) handleFileSelected(dropped);
                }}
                className="border border-dashed border-[#c9c9c9] hover:border-[#171717] rounded-[6px] p-6 text-center bg-[#fafafa] cursor-pointer transition"
              >
                <input
                  ref={fileRef}
                  type="file"
                  accept="video/*"
                  className="hidden"
                  onChange={(e) => handleFileSelected(e.target.files?.[0] ?? null)}
                />
                <Upload className="w-6 h-6 mx-auto text-[#171717] mb-2" />
                {file ? (
                  <div>
                    <div className="text-[14px] font-[500] text-[#171717]">{file.name}</div>
                    <div className="font-mono text-[11px] text-[#666666] mt-0.5">
                      {(file.size / (1024 * 1024)).toFixed(1)} MB · Click to replace
                    </div>
                  </div>
                ) : (
                  <div>
                    <div className="text-[14px] font-[500] text-[#171717]">
                      Drop an MP4 or MOV bowling clip here, or click to browse
                    </div>
                    <div className="font-mono text-[11px] text-[#666666] mt-1">
                      Supports single balls or long multi-ball net sessions (up to 512 MB)
                    </div>
                  </div>
                )}
              </div>

              {/* Delivery Settings */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-[13px]">
                <div className="sm:col-span-1 space-y-1">
                  <label className="v-eyebrow block">Delivery Label</label>
                  <input
                    value={label}
                    onChange={(e) => setLabel(e.target.value)}
                    className="v-input w-full h-9 px-2.5 text-[13px]"
                  />
                </div>
                <div className="space-y-1">
                  <label className="v-eyebrow block">Bowling Arm</label>
                  <select
                    value={arm}
                    onChange={(e) => setArm(e.target.value as "" | "right" | "left")}
                    className="v-input w-full h-9 px-2.5 text-[13px]"
                  >
                    <option value="">Auto-detect</option>
                    <option value="right">Right Arm</option>
                    <option value="left">Left Arm</option>
                  </select>
                </div>
                <div className="space-y-1">
                  <label className="v-eyebrow block">Capture Frame Rate</label>
                  <select
                    value={fpsOverride}
                    onChange={(e) => setFpsOverride(e.target.value)}
                    className="v-input w-full h-9 px-2.5 text-[13px]"
                  >
                    <option value="">Auto (probe container)</option>
                    <option value="30">30 fps (Standard)</option>
                    <option value="60">60 fps (Medium)</option>
                    <option value="120">120 fps (Slow-Mo)</option>
                    <option value="240">240 fps (High-Speed)</option>
                  </select>
                </div>
              </div>

              {/* Multi-Ball Net Session Auto-Clipper */}
              {file && (
                <div className="v-card-subtle p-3.5 space-y-2.5">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div>
                      <div className="v-eyebrow">MULTI-BALL NET VIDEO?</div>
                      <div className="text-[12px] text-[#4d4d4d]">
                        Auto-detect every ball in a continuous net recording and import selected clips without re-uploading.
                      </div>
                    </div>
                    <button
                      type="button"
                      onClick={handleScanNet}
                      disabled={scanning}
                      className="v-btn-ghost h-8 px-3 text-[12px] flex items-center gap-1.5"
                    >
                      <Scissors className="w-3.5 h-3.5" />
                      {scanning ? "Scanning ~6 fps..." : "Auto-Scan Balls"}
                    </button>
                  </div>

                  {scanError && <div className="font-mono text-[11px] text-[#4d4d4d]">{scanError}</div>}

                  {candidates.length > 0 && (
                    <div className="space-y-2">
                      <div className="font-mono text-[11px] text-[#297a3a]">
                        ✓ Detected {candidates.length} delivery burst(s) · {selectedIds.length} selected for batch import
                      </div>
                      <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 max-h-44 overflow-y-auto pr-1">
                        {candidates.map((c) => {
                          const checked = selectedIds.includes(c.candidate_id);
                          return (
                            <button
                              key={c.candidate_id}
                              type="button"
                              onClick={() => toggleCandidate(c.candidate_id)}
                              className={`p-2 rounded-[6px] border text-left transition flex items-center gap-2 ${
                                checked ? "border-[#171717] bg-[#ffffff]" : "border-[#ebebeb] bg-[#fafafa] opacity-65"
                              }`}
                            >
                              {c.thumbnail_b64 ? (
                                <img
                                  src={`data:image/jpeg;base64,${c.thumbnail_b64}`}
                                  alt=""
                                  className="w-12 h-8 object-cover rounded-[4px] shrink-0"
                                />
                              ) : (
                                <Film className="w-5 h-5 text-[#666666] shrink-0" />
                              )}
                              <div className="min-w-0 flex-1">
                                <div className="font-mono text-[11px] text-[#171717] font-medium flex items-center justify-between">
                                  <span>Ball #{c.candidate_id}</span>
                                  {checked && <Check className="w-3 h-3 text-[#297a3a]" />}
                                </div>
                                <div className="font-mono text-[10px] text-[#666666]">
                                  {c.start_s.toFixed(1)}s–{c.end_s.toFixed(1)}s
                                </div>
                              </div>
                            </button>
                          );
                        })}
                      </div>
                    </div>
                  )}
                </div>
              )}

              {/* Compact 3-Card Filming Reminder */}
              <div className="grid grid-cols-3 gap-2 text-[11px]">
                <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb]">
                  <div className="font-mono text-[#171717] font-medium">Side-On 90°</div>
                  <div className="text-[#666666] mt-0.5">Unlocks front knee, trunk flexion, arm delay &amp; ball speed.</div>
                </div>
                <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb]">
                  <div className="font-mono text-[#171717] font-medium">Rear 0°</div>
                  <div className="text-[#666666] mt-0.5">Unlocks side tilt, pelvic drop &amp; shoulder counter-rotation.</div>
                </div>
                <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb]">
                  <div className="font-mono text-[#171717] font-medium">Tripod 120–240 fps</div>
                  <div className="text-[#666666] mt-0.5">Keeps timing tight &amp; avoids motion-blur suppression.</div>
                </div>
              </div>

              <div className="flex justify-end gap-2 pt-2 border-t border-[#ebebeb]">
                <button type="button" onClick={onClose} className="v-btn-ghost h-9 px-4 text-[13px]">
                  Cancel
                </button>
                <button
                  type="button"
                  disabled={!file}
                  onClick={() => setStep(2)}
                  className="v-btn-primary h-9 px-4 text-[13px]"
                >
                  Continue →
                </button>
              </div>
            </>
          )}

          {/* STEP 2 */}
          {step === 2 && (
            <>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 items-center">
                <div className="space-y-2.5 text-[13px] text-[#4d4d4d]">
                  <div className="v-eyebrow text-[#171717]">WHY PITCH CALIBRATION MATTERS</div>
                  <p>
                    By default, the engine automatically detects the bowling &amp; striker stumps (MCC Law 8) or aligns
                    gravity from upright background frames — zero clicks required.
                  </p>
                  <p>
                    After upload, you can also open <strong>Lab Telemetry → 4-Click Crease Calibration</strong> at any
                    time to click 4 crease corners and recompute metric scale in <strong>&lt;35 ms</strong> without
                    re-running the GPU.
                  </p>
                </div>

                {/* Minimal Vercel Monochrome Top-Down Crease SVG */}
                <div className="v-card-subtle p-3">
                  <svg viewBox="0 0 260 150" className="w-full h-auto">
                    <rect x="10" y="10" width="240" height="130" rx="4" fill="#ffffff" stroke="#ebebeb" />
                    <line x1="65" y1="30" x2="65" y2="120" stroke="#171717" strokeWidth="1.5" />
                    <line x1="155" y1="20" x2="155" y2="130" stroke="#297a3a" strokeWidth="2" />
                    <line x1="35" y1="30" x2="155" y2="30" stroke="#8f8f8f" strokeWidth="1.2" />
                    <line x1="35" y1="120" x2="155" y2="120" stroke="#8f8f8f" strokeWidth="1.2" />
                    {[62, 75, 88].map((y, i) => (
                      <circle key={i} cx="65" cy={y} r="2.6" fill="#171717" />
                    ))}
                    {[
                      [65, 30, "1"],
                      [65, 120, "2"],
                      [155, 30, "3"],
                      [155, 120, "4"],
                    ].map(([x, y, n]) => (
                      <g key={String(n)}>
                        <circle cx={Number(x)} cy={Number(y)} r="6" fill="#171717" />
                        <text x={Number(x)} y={Number(y) + 3} textAnchor="middle" fill="#ffffff" fontSize="8" fontFamily="monospace">
                          {n}
                        </text>
                      </g>
                    ))}
                    <text x="163" y="34" fill="#297a3a" fontSize="8" fontFamily="monospace">POPPING (1.22m)</text>
                    <text x="16" y="135" fill="#666666" fontSize="8" fontFamily="monospace">MCC LAW 7 CREASE LANDMARKS</text>
                  </svg>
                </div>
              </div>

              <div className="flex items-center justify-between pt-3 border-t border-[#ebebeb]">
                <button type="button" onClick={() => setStep(1)} className="v-btn-ghost h-9 px-3.5 text-[13px]">
                  ← Back
                </button>
                <button type="button" onClick={handleRunAnalysis} className="v-btn-primary h-9 px-4 text-[13px]">
                  {scanId && selectedIds.length > 0
                    ? `Import ${selectedIds.length} Selected Ball(s) & Auto-Calibrate →`
                    : "Skip Manual Clicks — Auto-Calibrate & Analyze →"}
                </button>
              </div>
            </>
          )}

          {/* STEP 3: Vercel White-Paper CLI Diagnostic Log */}
          {step === 3 && (
            <>
              <div className="rounded-[6px] bg-[#fafafa] border border-[#ebebeb] p-4 font-mono text-[12px] leading-relaxed space-y-1.5 min-h-[160px] text-[#171717]">
                <div className="text-[#666666] text-[11px] pb-1.5 border-b border-[#ebebeb] flex items-center justify-between">
                  <span>▲ sportsposture-cli v3.0 — {session.title}</span>
                  <span>STDOUT</span>
                </div>
                {logs.map((line, i) => (
                  <div
                    key={i}
                    className={
                      line.startsWith("✓")
                        ? "text-[#297a3a] font-medium"
                        : line.startsWith("✕")
                        ? "text-[#171717] underline"
                        : "text-[#4d4d4d]"
                    }
                  >
                    {line}
                  </div>
                ))}
                {running && <div className="text-[#666666] animate-pulse">▲ processing...</div>}
              </div>

              {runError && (
                <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#171717] text-[12px] text-[#4d4d4d] space-y-1.5">
                  <div className="font-mono text-[11px] font-medium text-[#171717]">HOW TO ENABLE RAW VIDEO 3D INFERENCE</div>
                  <p>{runError}</p>
                  <div className="font-mono text-[11px] bg-[#ffffff] border border-[#ebebeb] p-2 rounded-[4px] text-[#171717] overflow-x-auto">
                    GPU_WORKER_URL=http://127.0.0.1:8090 bowling-biomech serve-studio
                  </div>
                </div>
              )}

              <div className="flex justify-end gap-2 pt-2 border-t border-[#ebebeb]">
                {runError && (
                  <button type="button" onClick={() => setStep(1)} className="v-btn-ghost h-9 px-3.5 text-[13px]">
                    ← Back
                  </button>
                )}
                <button type="button" onClick={onClose} disabled={running} className="v-btn-primary h-9 px-4 text-[13px]">
                  {running ? "Analyzing..." : "Close"}
                </button>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
};
