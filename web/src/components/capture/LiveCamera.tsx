import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Camera, CameraOff, Circle, RotateCcw, Square, Upload } from "lucide-react";
import type { OverviewBowler } from "../../types";
import { navigate } from "../../router";
import { FailureCard, NeedBowler, PanelTitle, PipelineProgress, Spinner, formatClock, useAnalysisRun, useElapsed, type Failure } from "./shared";

/**
 * Live camera framing assistant (§8.6). Framing only: this screen NEVER shows biomechanical numbers or angles.
 * Measurements come only from the calibrated pipeline after upload.
 */

type Framing = "side" | "behind";
type CamPhase = "off" | "starting" | "live" | "error";
type RecPhase = "idle" | "countdown" | "recording" | "preview";
type LevelMode = "checking" | "needs-permission" | "sensor" | "unavailable";

const MAX_REC_S = 8;
const LEVEL_OK_DEG = 1.5;

const CHECKS: Array<{ id: string; label: (f: Framing) => string }> = [
  { id: "body", label: () => "Whole body in frame, head to feet, all the way through the follow-through" },
  { id: "level", label: () => "Phone level and held still (tripod or wedge it)" },
  {
    id: "distance",
    label: (f) => (f === "side" ? "About 8–12 m away, square of the bowling crease" : "Straight behind the run-up, in line with both sets of stumps"),
  },
  { id: "light", label: () => "Good light: bowler not in deep shadow or against the sun" },
  { id: "stumps", label: () => "Stumps visible (all three), so we can measure the pitch" },
];

function pickMime(): string {
  if (typeof MediaRecorder === "undefined") return "";
  const opts = ["video/mp4;codecs=avc1", "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return opts.find((m) => (MediaRecorder.isTypeSupported ? MediaRecorder.isTypeSupported(m) : false)) ?? "";
}

function cameraFailure(err: any): Failure {
  const name = String(err?.name || "");
  if (name === "NotAllowedError" || name === "SecurityError")
    return {
      reason: "Camera permission was blocked.",
      fix: "Allow camera access for this site in your browser settings, then tap Start camera again. Or film with your phone's camera app and use Upload clip.",
    };
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError")
    return { reason: "We couldn't find a camera on this device.", fix: "Film with your phone's camera app and use Upload clip instead." };
  if (name === "NotReadableError" || name === "TrackStartError")
    return { reason: "The camera is being used by another app.", fix: "Close other apps or tabs using the camera, then try again." };
  return { reason: "The camera couldn't be started.", fix: "Try again, or film with your phone's camera app and use Upload clip." };
}

/** Screen-relative roll (deg) from DeviceOrientation Euler angles; null when the phone is lying flat. */
function screenRoll(e: DeviceOrientationEvent): number | null {
  if (e.beta == null || e.gamma == null) return null;
  const b = (e.beta * Math.PI) / 180;
  const g = (e.gamma * Math.PI) / 180;
  // World "up" expressed in device axes (W3C Z-X'-Y'' convention).
  const ux = -Math.sin(g) * Math.cos(b);
  const uy = Math.sin(b);
  const angle = ((typeof screen !== "undefined" && screen.orientation?.angle) || 0) * (Math.PI / 180);
  const up = ux * Math.sin(angle) + uy * Math.cos(angle);
  const right = ux * Math.cos(angle) - uy * Math.sin(angle);
  if (Math.hypot(up, right) < 0.35) return null; // lying flat: roll not meaningful
  return (Math.atan2(right, up) * 180) / Math.PI;
}

// ── Framing overlay ────────────────────────────────────────────────────────
const FramingOverlay: React.FC<{ framing: Framing; vw: number; vh: number; roll: number | null; showThirds: boolean; showGhost: boolean; showStumps: boolean }> = ({
  framing,
  vw,
  vh,
  roll,
  showThirds,
  showGhost,
  showStumps,
}) => {
  const box = framing === "side" ? { x: vw * 0.14, y: vh * 0.08, w: vw * 0.72, h: vh * 0.84 } : { x: vw * 0.3, y: vh * 0.06, w: vw * 0.4, h: vh * 0.88 };
  const level = roll !== null && Math.abs(roll) <= LEVEL_OK_DEG;
  const halo = { paintOrder: "stroke" as const, stroke: "rgba(0,0,0,0.55)", strokeWidth: 2.5 };
  const fs = Math.max(3.4, vw / 42);
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox={`0 0 ${vw} ${vh}`} preserveAspectRatio="none" aria-hidden>
      {showThirds && (
        <g stroke="rgba(255,255,255,0.35)" strokeWidth={1} vectorEffect="non-scaling-stroke">
          <line x1={vw / 3} y1={0} x2={vw / 3} y2={vh} vectorEffect="non-scaling-stroke" />
          <line x1={(2 * vw) / 3} y1={0} x2={(2 * vw) / 3} y2={vh} vectorEffect="non-scaling-stroke" />
          <line x1={0} y1={vh / 3} x2={vw} y2={vh / 3} vectorEffect="non-scaling-stroke" />
          <line x1={0} y1={(2 * vh) / 3} x2={vw} y2={(2 * vh) / 3} vectorEffect="non-scaling-stroke" />
        </g>
      )}
      {/* horizon from the level sensor */}
      {roll !== null && (
        <g transform={`rotate(${-roll} ${vw / 2} ${vh / 2})`}>
          <line
            x1={vw * 0.2}
            y1={vh / 2}
            x2={vw * 0.8}
            y2={vh / 2}
            stroke={level ? "#297a3a" : "#ffffff"}
            strokeWidth={level ? 2 : 1.5}
            strokeDasharray={level ? undefined : "6 4"}
            vectorEffect="non-scaling-stroke"
          />
        </g>
      )}
      {showGhost && (
        <>
          <rect
            x={box.x}
            y={box.y}
            width={box.w}
            height={box.h}
            fill="none"
            stroke="rgba(255,255,255,0.9)"
            strokeWidth={1.5}
            strokeDasharray="8 5"
            rx={2}
            vectorEffect="non-scaling-stroke"
          />
          <svg x={box.x} y={box.y} width={box.w} height={box.h} viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
            <g fill="none" stroke="rgba(226,232,240,0.85)" strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" vectorEffect="non-scaling-stroke">
              {framing === "side" ? (
                <>
                  <circle cx={48} cy={9} r={4.2} vectorEffect="non-scaling-stroke" />
                  <path d="M48 14 L45 42" vectorEffect="non-scaling-stroke" />
                  <path d="M47 17 L50 9 L52 1" vectorEffect="non-scaling-stroke" />
                  <path d="M47 18 L56 24 L63 22" vectorEffect="non-scaling-stroke" />
                  <path d="M45 42 L58 68 L66 96" vectorEffect="non-scaling-stroke" />
                  <path d="M45 42 L37 69 L25 93" vectorEffect="non-scaling-stroke" />
                </>
              ) : (
                <>
                  <circle cx={50} cy={10} r={4.6} vectorEffect="non-scaling-stroke" />
                  <path d="M50 15 L50 46" vectorEffect="non-scaling-stroke" />
                  <path d="M42 20 L58 20" vectorEffect="non-scaling-stroke" />
                  <path d="M58 20 L61 9 L62 1" vectorEffect="non-scaling-stroke" />
                  <path d="M42 20 L36 33 L38 42" vectorEffect="non-scaling-stroke" />
                  <path d="M45 46 L55 46" vectorEffect="non-scaling-stroke" />
                  <path d="M45 46 L41 72 L40 97" vectorEffect="non-scaling-stroke" />
                  <path d="M55 46 L59 72 L61 97" vectorEffect="non-scaling-stroke" />
                </>
              )}
            </g>
          </svg>
          <text x={box.x + 2} y={box.y - 2} fontSize={fs} fill="#ffffff" style={halo}>
            Fit the bowler inside the box
          </text>
        </>
      )}
      {showStumps &&
        (framing === "side" ? (
          <g>
            <line x1={box.x + box.w * 0.3} y1={box.y + box.h * 0.8} x2={box.x + box.w * 0.3} y2={box.y + box.h} stroke="#ffffff" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            <line x1={box.x + box.w * 0.18} y1={box.y + box.h} x2={box.x + box.w * 0.82} y2={box.y + box.h} stroke="rgba(255,255,255,0.7)" strokeDasharray="4 4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
            <text x={box.x + box.w * 0.3 + 2} y={box.y + box.h * 0.8} fontSize={fs * 0.9} fill="#ffffff" style={halo}>
              Stumps here
            </text>
          </g>
        ) : (
          <g>
            <line x1={vw / 2} y1={vh * 0.4} x2={vw / 2} y2={vh} stroke="rgba(255,255,255,0.75)" strokeDasharray="4 4" strokeWidth={1} vectorEffect="non-scaling-stroke" />
            {[-1, 0, 1].map((k) => (
              <line key={`n${k}`} x1={vw / 2 + k * vw * 0.025} y1={vh * 0.86} x2={vw / 2 + k * vw * 0.025} y2={vh * 0.98} stroke="#ffffff" strokeWidth={2} vectorEffect="non-scaling-stroke" />
            ))}
            {[-1, 0, 1].map((k) => (
              <line key={`f${k}`} x1={vw / 2 + k * vw * 0.006} y1={vh * 0.42} x2={vw / 2 + k * vw * 0.006} y2={vh * 0.45} stroke="#ffffff" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
            ))}
            <text x={vw / 2 + 3} y={vh * 0.41} fontSize={fs * 0.9} fill="#ffffff" style={halo}>
              Line up both sets of stumps
            </text>
          </g>
        ))}
    </svg>
  );
};

// ── Spirit level readout ────────────────────────────────────────────────────
const SpiritLevel: React.FC<{ roll: number | null; mode: LevelMode; onRequest: () => void; sim: number; onSim: (v: number) => void }> = ({
  roll,
  mode,
  onRequest,
  sim,
  onSim,
}) => {
  const r = roll ?? 0;
  const ok = roll !== null && Math.abs(r) <= LEVEL_OK_DEG;
  const bubble = Math.max(-1, Math.min(1, r / 10));
  return (
    <div className="rounded-[6px] border border-[#ebebeb] bg-white p-3">
      <div className="flex items-baseline justify-between gap-2">
        <span className="v-label">Phone level</span>
        <span className={`v-num text-[13px] ${ok ? "text-[#297a3a]" : "text-[#171717]"}`}>
          {roll === null ? "—" : `${r > 0 ? "+" : ""}${r.toFixed(1)}°`}
        </span>
      </div>
      <div className="relative mt-2 h-5 rounded-full bg-[#f4f4f5] border border-[#ebebeb] overflow-hidden">
        <div className="absolute inset-y-0 left-1/2 w-px bg-[#c9c9c9]" />
        <div className="absolute inset-y-0 left-[calc(50%-14px)] w-[28px] border-x border-[#c9c9c9]" />
        {roll !== null && (
          <div
            className={`absolute top-[3px] w-3 h-3 rounded-full transition-[left] duration-150 ${ok ? "bg-[#297a3a]" : "bg-[#171717]"}`}
            style={{ left: `calc(${50 + bubble * 45}% - 6px)` }}
          />
        )}
      </div>
      <p className="mt-2 text-[12px] leading-[1.45] text-[#4d4d4d]">
        {mode === "sensor" &&
          (roll === null ? "Hold the phone upright to read the level." : ok ? "Level. Keep it still." : r > 0 ? "Tilted: lower the right side." : "Tilted: lower the left side.")}
        {mode === "checking" && "Looking for the level sensor…"}
        {mode === "needs-permission" && (
          <>
            This phone needs permission to use its level sensor.{" "}
            <button type="button" onClick={onRequest} className="underline underline-offset-2 text-[#171717] font-[500]">
              Turn on level
            </button>
          </>
        )}
        {mode === "unavailable" && "Level sensor not available — place the phone on a flat surface or tripod."}
      </p>
      {mode === "unavailable" && (
        <label className="mt-2 block">
          <span className="text-[12px] text-[#8f8f8f]">Simulate tilt (preview only)</span>
          <input
            type="range"
            min={-10}
            max={10}
            step={0.5}
            value={sim}
            onChange={(e) => onSim(Number(e.target.value))}
            className="w-full accent-[#171717]"
            aria-label="Simulate phone tilt"
          />
        </label>
      )}
    </div>
  );
};

// ── Live camera panel ──────────────────────────────────────────────────────
export const LiveCamera: React.FC<{ bowler: OverviewBowler | null; canAnalyse: boolean; onUseUpload: () => void }> = ({ bowler, canAnalyse, onUseUpload }) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const stopTimerRef = useRef<number | null>(null);
  const previewUrlRef = useRef<string | null>(null);

  const [cam, setCam] = useState<CamPhase>("off");
  const [camError, setCamError] = useState<Failure | null>(null);
  const [settings, setSettings] = useState<{ fps: number | null; maxFps: number | null; w: number; h: number }>({ fps: null, maxFps: null, w: 16, h: 9 });
  const [framing, setFraming] = useState<Framing>("side");
  const [showThirds, setShowThirds] = useState(true);
  const [showGhost, setShowGhost] = useState(true);
  const [showStumps, setShowStumps] = useState(true);
  const [checks, setChecks] = useState<Record<string, boolean>>({});

  const [rec, setRec] = useState<RecPhase>("idle");
  const [count, setCount] = useState(3);
  const [recStart, setRecStart] = useState<number | null>(null);
  const [preview, setPreview] = useState<{ url: string; blob: Blob; mime: string } | null>(null);
  const [recError, setRecError] = useState<Failure | null>(null);

  const [levelMode, setLevelMode] = useState<LevelMode>("checking");
  const [sensorRoll, setSensorRoll] = useState<number | null>(null);
  const [simRoll, setSimRoll] = useState(0);
  const roll = levelMode === "sensor" ? sensorRoll : levelMode === "unavailable" ? simRoll : null;

  const onDone = useCallback((id: string) => navigate({ name: "delivery", id, tab: "story" }), []);
  const run = useAnalysisRun(bowler, onDone);
  const runElapsed = useElapsed(run.state.startedAt, run.state.phase === "uploading" || run.state.phase === "processing");
  const recElapsed = useElapsed(recStart, rec === "recording");

  // ── camera lifecycle ──
  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    if (videoRef.current) videoRef.current.srcObject = null;
    setCam("off");
  }, []);

  useEffect(
    () => () => {
      if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
      try {
        if (recorderRef.current && recorderRef.current.state !== "inactive") recorderRef.current.stop();
      } catch {
        /* ignore */
      }
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    },
    []
  );

  const startCamera = async () => {
    setCamError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setCamError({
        reason: window.isSecureContext ? "This browser can't open the camera." : "The camera only works over a secure (https) connection.",
        fix: "Film with your phone's camera app and use Upload clip instead.",
      });
      setCam("error");
      return;
    }
    setCam("starting");
    try {
      let stream: MediaStream;
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: { ideal: "environment" }, frameRate: { ideal: 120 }, width: { ideal: 1920 }, height: { ideal: 1080 } },
          audio: false,
        });
      } catch (e: any) {
        if (e?.name !== "OverconstrainedError") throw e;
        stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      }
      streamRef.current = stream;
      const track = stream.getVideoTracks()[0];
      const s = track?.getSettings ? track.getSettings() : {};
      let maxFps: number | null = null;
      try {
        const caps: any = track && typeof track.getCapabilities === "function" ? track.getCapabilities() : null;
        maxFps = caps?.frameRate?.max ? Number(caps.frameRate.max) : null;
      } catch {
        maxFps = null;
      }
      setSettings({ fps: s.frameRate ? Number(s.frameRate) : null, maxFps, w: s.width || 16, h: s.height || 9 });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play().catch(() => undefined);
      }
      setCam("live");
    } catch (err: any) {
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
      setCamError(cameraFailure(err));
      setCam("error");
    }
  };

  // ── level sensor ──
  const attachOrientation = useCallback(() => {
    let got = false;
    const handler = (e: DeviceOrientationEvent) => {
      if (e.beta == null || e.gamma == null) return;
      got = true;
      setLevelMode("sensor");
      setSensorRoll(screenRoll(e));
    };
    window.addEventListener("deviceorientation", handler);
    const t = window.setTimeout(() => {
      if (!got) setLevelMode("unavailable");
    }, 1500);
    return () => {
      window.removeEventListener("deviceorientation", handler);
      window.clearTimeout(t);
    };
  }, []);

  const detachRef = useRef<(() => void) | null>(null);
  useEffect(() => {
    if (typeof window === "undefined" || typeof (window as any).DeviceOrientationEvent === "undefined") {
      setLevelMode("unavailable");
      return;
    }
    const DOE: any = (window as any).DeviceOrientationEvent;
    if (typeof DOE.requestPermission === "function") {
      setLevelMode("needs-permission");
    } else {
      detachRef.current = attachOrientation();
    }
    return () => {
      detachRef.current?.();
      detachRef.current = null;
    };
  }, [attachOrientation]);

  const requestLevel = async () => {
    const DOE: any = (window as any).DeviceOrientationEvent;
    try {
      const res = await DOE.requestPermission();
      if (res === "granted") {
        setLevelMode("checking");
        detachRef.current?.();
        detachRef.current = attachOrientation();
      } else setLevelMode("unavailable");
    } catch {
      setLevelMode("unavailable");
    }
  };

  // ── recording ──
  const mime = useMemo(pickMime, []);
  const canRecord = typeof MediaRecorder !== "undefined";

  const beginRecording = useCallback(() => {
    const stream = streamRef.current;
    if (!stream) {
      setRec("idle");
      return;
    }
    try {
      chunksRef.current = [];
      const r = mime ? new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 }) : new MediaRecorder(stream);
      recorderRef.current = r;
      r.ondataavailable = (ev) => {
        if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
      };
      r.onstop = () => {
        if (stopTimerRef.current) window.clearTimeout(stopTimerRef.current);
        stopTimerRef.current = null;
        const type = r.mimeType || mime || "video/webm";
        const blob = new Blob(chunksRef.current, { type });
        if (blob.size === 0) {
          setRecError({ reason: "Nothing was recorded.", fix: "Try again, or film with your phone's camera app and use Upload clip." });
          setRec("idle");
          return;
        }
        if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
        const url = URL.createObjectURL(blob);
        previewUrlRef.current = url;
        setPreview({ url, blob, mime: type });
        setRec("preview");
      };
      r.start(250);
      setRecStart(Date.now());
      setRec("recording");
      stopTimerRef.current = window.setTimeout(() => {
        if (r.state !== "inactive") r.stop();
      }, MAX_REC_S * 1000);
    } catch {
      setRecError({ reason: "This browser couldn't record the camera.", fix: "Film with your phone's camera app and use Upload clip instead." });
      setRec("idle");
    }
  }, [mime]);

  useEffect(() => {
    if (rec !== "countdown") return;
    if (count <= 0) {
      beginRecording();
      return;
    }
    const t = window.setTimeout(() => setCount((c) => c - 1), 1000);
    return () => window.clearTimeout(t);
  }, [rec, count, beginRecording]);

  const startCountdown = () => {
    setRecError(null);
    setCount(3);
    setRec("countdown");
  };
  const stopRecording = () => {
    const r = recorderRef.current;
    if (r && r.state !== "inactive") r.stop();
  };
  const retake = () => {
    if (previewUrlRef.current) URL.revokeObjectURL(previewUrlRef.current);
    previewUrlRef.current = null;
    setPreview(null);
    setRec("idle");
    run.reset();
    if (!streamRef.current) void startCamera();
  };
  const analyse = () => {
    if (!preview || !bowler) return;
    const ext = preview.mime.includes("mp4") ? "mp4" : "webm";
    const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    const file = new File([preview.blob], `live-camera-${stamp}.${ext}`, { type: preview.mime });
    stopCamera();
    void run.start(file, { label: `Live camera · ${framing === "side" ? "Side-on" : "Behind"} clip`, fps: null });
  };

  const tickedCount = CHECKS.filter((c) => checks[c.id]).length;
  const aspect = settings.w > 0 && settings.h > 0 ? settings.w / settings.h : 16 / 9;
  const vw = 160;
  const vh = vw / aspect;
  const lowFps = settings.fps !== null && settings.fps < 100;
  const running = run.state.phase !== "idle";

  return (
    <div className="space-y-4">
      <PanelTitle eyebrow="Live camera" title="Frame the shot, then record a short clip">
        This screen only helps you set up the camera. It never shows angles or measurements; those come from the full analysis after you record.
      </PanelTitle>

      <div className="grid gap-4 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        {/* Stage */}
        <div className="space-y-3 min-w-0">
          <div
            className="relative mx-auto w-full overflow-hidden rounded-[6px] bg-[#171717]"
            style={{ aspectRatio: `${aspect}`, maxWidth: `calc(68vh * ${aspect})` }}
          >
            <video
              ref={videoRef}
              playsInline
              muted
              autoPlay
              onLoadedMetadata={(e) => {
                const v = e.currentTarget;
                if (v.videoWidth > 0 && v.videoHeight > 0) setSettings((s) => ({ ...s, w: v.videoWidth, h: v.videoHeight }));
              }}
              className={`absolute inset-0 w-full h-full object-cover ${cam === "live" && rec !== "preview" ? "" : "invisible"}`}
            />
            {rec === "preview" && preview && (
              <video src={preview.url} controls playsInline loop autoPlay muted className="absolute inset-0 w-full h-full object-contain bg-[#171717]" />
            )}
            {cam === "live" && rec !== "preview" && (
              <FramingOverlay framing={framing} vw={vw} vh={vh} roll={roll} showThirds={showThirds} showGhost={showGhost} showStumps={showStumps} />
            )}
            {cam !== "live" && rec !== "preview" && (
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center bg-[#f6f6f5]">
                {cam === "starting" ? (
                  <>
                    <Spinner />
                    <span className="text-[13px] text-[#4d4d4d]">Opening the camera…</span>
                  </>
                ) : (
                  <>
                    <span className="w-10 h-10 rounded-full bg-white border border-[#ebebeb] flex items-center justify-center">
                      {cam === "error" ? <CameraOff className="w-4 h-4 text-[#4d4d4d]" /> : <Camera className="w-4 h-4 text-[#171717]" />}
                    </span>
                    <span className="text-[13px] text-[#4d4d4d] max-w-[320px]">
                      {running ? "Camera closed while your clip is analysed." : "Use the back camera, propped up or on a tripod, so the shot stays still."}
                    </span>
                    {!running && (
                      <button type="button" onClick={startCamera} className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-2">
                        <Camera className="w-4 h-4" /> {cam === "error" ? "Try again" : "Start camera"}
                      </button>
                    )}
                  </>
                )}
              </div>
            )}
            {rec === "countdown" && count > 0 && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                <span className="v-hero-num text-white text-[96px]">{count}</span>
              </div>
            )}
            {rec === "recording" && (
              <div className="absolute top-2 left-2 v-chip v-chip-ink">
                <span className="w-2 h-2 rounded-full bg-[#dc2626] animate-pulse" aria-hidden />
                <span className="v-num">
                  REC {recElapsed.toFixed(1)} / {MAX_REC_S}s
                </span>
              </div>
            )}
          </div>

          {camError && <FailureCard failure={camError} />}
          {recError && <FailureCard failure={recError} />}

          {/* Controls */}
          {cam === "live" && rec !== "preview" && (
            <div className="flex flex-wrap items-center gap-2">
              <div className="v-seg" role="group" aria-label="Framing guide">
                <button type="button" aria-pressed={framing === "side"} onClick={() => setFraming("side")}>
                  Side-on
                </button>
                <button type="button" aria-pressed={framing === "behind"} onClick={() => setFraming("behind")}>
                  Behind
                </button>
              </div>
              {[
                { on: showGhost, set: setShowGhost, label: "Target box" },
                { on: showStumps, set: setShowStumps, label: "Stumps guide" },
                { on: showThirds, set: setShowThirds, label: "Grid" },
              ].map((t) => (
                <button
                  key={t.label}
                  type="button"
                  aria-pressed={t.on}
                  onClick={() => t.set(!t.on)}
                  className={`v-chip ${t.on ? "v-chip-ink" : ""}`}
                >
                  {t.label}
                </button>
              ))}
              <div className="flex-1" />
              {rec === "recording" ? (
                <button type="button" onClick={stopRecording} className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-2">
                  <Square className="w-3.5 h-3.5" /> Stop
                </button>
              ) : (
                <button
                  type="button"
                  onClick={startCountdown}
                  disabled={!canRecord || rec === "countdown"}
                  className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-2"
                >
                  <Circle className="w-3.5 h-3.5 fill-current" /> {rec === "countdown" ? "Get ready…" : "Record (3-2-1)"}
                </button>
              )}
              <button type="button" onClick={stopCamera} className="v-btn-ghost h-9 px-3 text-[13px]">
                Close camera
              </button>
            </div>
          )}
          {cam === "live" && !canRecord && (
            <p className="text-[12.5px] text-[#4d4d4d]">This browser can show the camera but can't record it. Film with your phone's camera app and use Upload clip.</p>
          )}

          {rec === "preview" && preview && !running && (
            <div className="v-panel p-3 flex flex-wrap items-center gap-2">
              <span className="text-[13px] text-[#4d4d4d] mr-auto">Check the whole delivery is in the clip, then analyse it.</span>
              <button type="button" onClick={retake} className="v-btn-ghost h-9 px-3 text-[13px] inline-flex items-center gap-1.5">
                <RotateCcw className="w-3.5 h-3.5" /> Retake
              </button>
              <button
                type="button"
                onClick={analyse}
                disabled={!bowler || !canAnalyse}
                className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-1.5"
              >
                Analyse this clip
              </button>
              {!bowler && (
                <div className="w-full">
                  <NeedBowler />
                </div>
              )}
            </div>
          )}

          {running && (
            <>
              <PipelineProgress
                lo={run.state.lo}
                hi={run.state.hi}
                indeterminate={run.state.phase === "uploading" || (run.state.phase === "processing" && !run.state.hasJob)}
                elapsedS={runElapsed}
                message={run.state.phase === "uploading" ? "Uploading your clip…" : run.state.job?.message_plain}
                failed={run.state.phase === "failed"}
              />
              {run.state.failure && <FailureCard failure={run.state.failure} onRetry={retake} retryLabel="Record again" />}
            </>
          )}
        </div>

        {/* Side column */}
        <div className="space-y-3 min-w-0">
          <div className="rounded-[6px] border border-dashed border-[#c9c9c9] bg-white p-3">
            <div className="flex items-baseline justify-between gap-2">
              <span className="v-label">Recording speed</span>
              <span className="v-num text-[13px] text-[#171717]">{settings.fps ? `${Math.round(settings.fps)} fps` : "—"}</span>
            </div>
            <p className="mt-1.5 text-[12.5px] leading-[1.5] text-[#4d4d4d]">
              {settings.fps
                ? `This browser is filming at ${Math.round(settings.fps)} fps${settings.maxFps ? ` (camera allows up to ${Math.round(settings.maxFps)} fps here)` : ""}. `
                : ""}
              Browser recording is usually 30–60 fps. {lowFps || !settings.fps ? "That's fine for framing, but for" : "For"} full speed and precise timing of
              foot landing and release, film with your phone's own slow-motion camera (120–240 fps) and use Upload clip.
            </p>
            <button type="button" onClick={onUseUpload} className="v-btn-ghost mt-2.5 h-8 px-3 text-[12.5px] inline-flex items-center gap-1.5">
              <Upload className="w-3.5 h-3.5" /> Use a slow-motion video instead
            </button>
          </div>

          <SpiritLevel roll={roll} mode={levelMode} onRequest={requestLevel} sim={simRoll} onSim={setSimRoll} />

          <div className="rounded-[6px] border border-[#ebebeb] bg-white p-3">
            <div className="flex items-baseline justify-between gap-2 mb-2">
              <span className="v-label">Before you record</span>
              <span className={`v-num text-[12px] ${tickedCount === CHECKS.length ? "text-[#297a3a]" : "text-[#8f8f8f]"}`}>
                {tickedCount}/{CHECKS.length}
              </span>
            </div>
            <ul className="space-y-1.5">
              {CHECKS.map((c) => {
                const on = Boolean(checks[c.id]);
                const sensorHint =
                  c.id === "level" && roll !== null ? (Math.abs(roll) <= LEVEL_OK_DEG ? "Sensor says level" : "Sensor says tilted") : null;
                return (
                  <li key={c.id}>
                    <label className="flex items-start gap-2.5 cursor-pointer select-none">
                      <input
                        type="checkbox"
                        checked={on}
                        onChange={(e) => setChecks((p) => ({ ...p, [c.id]: e.target.checked }))}
                        className="mt-[3px] w-4 h-4 accent-[#297a3a] shrink-0"
                      />
                      <span className={`text-[13px] leading-[1.45] ${on ? "text-[#171717]" : "text-[#4d4d4d]"}`}>
                        {c.label(framing)}
                        {sensorHint && <span className="block text-[12px] text-[#8f8f8f]">{sensorHint}</span>}
                      </span>
                    </label>
                  </li>
                );
              })}
            </ul>
            {rec === "recording" && (
              <p className="mt-2 text-[12px] text-[#8f8f8f]">
                Recording stops on its own after <span className="v-num">{formatClock(MAX_REC_S)}</span>.
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default LiveCamera;
