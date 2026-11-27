import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Camera,
  Circle,
  Crosshair,
  Minus,
  MousePointer2,
  Pause,
  Play,
  SkipBack,
  SkipForward,
  Trash2,
  Triangle,
  Undo2,
  UserCheck,
} from "lucide-react";
import { useStudioStore } from "../store";
import type {
  BiomechanicalFactSheet,
  TelestratorShape,
  TelestratorTool,
  TelemetryV2,
} from "../types";

const BONE_PAIRS_2D: Array<[number, number, "spine" | "lleg" | "rleg" | "larm" | "rarm"]> = [
  [0, 3, "spine"], [3, 6, "spine"], [6, 9, "spine"], [9, 12, "spine"], [12, 15, "spine"],
  [0, 1, "lleg"], [1, 4, "lleg"], [4, 7, "lleg"], [7, 10, "lleg"],
  [0, 2, "rleg"], [2, 5, "rleg"], [5, 8, "rleg"], [8, 11, "rleg"],
  [9, 13, "larm"], [13, 16, "larm"], [16, 18, "larm"], [18, 20, "larm"], [20, 22, "larm"],
  [9, 14, "rarm"], [14, 17, "rarm"], [17, 19, "rarm"], [19, 21, "rarm"], [21, 23, "rarm"],
];

const MCC_EDGES: Array<[string, string]> = [
  ["bowling_crease_off", "bowling_crease_leg"],
  ["popping_crease_off", "popping_crease_leg"],
  ["bowling_crease_off", "popping_crease_off"],
  ["bowling_crease_leg", "popping_crease_leg"],
  ["stump_middle_base", "stump_middle_top"],
];

const CALIBRATION_ORDER = [
  "bowling_crease_off",
  "bowling_crease_leg",
  "popping_crease_off",
  "popping_crease_leg",
  "stump_middle_base",
  "stump_middle_top",
];

function compute2DAngleDeg(a: [number, number], b: [number, number], c: [number, number]): number {
  const v1x = a[0] - b[0];
  const v1y = a[1] - b[1];
  const v2x = c[0] - b[0];
  const v2y = c[1] - b[1];
  const d1 = Math.hypot(v1x, v1y);
  const d2 = Math.hypot(v2x, v2y);
  if (d1 < 1e-4 || d2 < 1e-4) return 0;
  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (d1 * d2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

interface VideoCanvasPlayerProps {
  videoUrl: string;
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  annotations: TelestratorShape[];
  onSaveAnnotations: (shapes: TelestratorShape[]) => void;
  onNudgeEvent: (phase: "bfc_frame" | "ffc_frame" | "uah_frame" | "br_frame", frame: number) => void;
  onLandmarkClick: (key: string, pt: [number, number]) => void;
  onRetrackBowler?: (anchorNorm: [number, number]) => void;
  onFpsChange: (fps: number) => void;
  onPhaseThumbsCaptured?: (thumbs: Record<string, string>) => void;
  compact?: boolean;
}

export const VideoCanvasPlayer: React.FC<VideoCanvasPlayerProps> = ({
  videoUrl,
  telemetry,
  factSheet,
  annotations,
  onSaveAnnotations,
  onNudgeEvent,
  onLandmarkClick,
  onRetrackBowler,
  onFpsChange,
  onPhaseThumbsCaptured,
  compact = false,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const loupeRef = useRef<HTMLCanvasElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const seekingRef = useRef<boolean>(false);
  const pendingSeekFrameRef = useRef<number | null>(null);
  const drawnFrameRef = useRef<number>(useStudioStore.getState().currentFrame);
  const hoverPxRef = useRef<[number, number] | null>(null);
  const dragStartRef = useRef<[number, number] | null>(null);
  const draftPtsRef = useRef<Array<[number, number]>>([]);
  const [draftCount, setDraftCount] = useState(0);
  const [phaseThumbs, setPhaseThumbs] = useState<Record<string, string>>({});

  const currentFrame = useStudioStore((s) => s.currentFrame);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const playbackRate = useStudioStore((s) => s.playbackRate);
  const showSkeleton = useStudioStore((s) => s.showSkeleton);
  const showWireframe = useStudioStore((s) => s.showWireframe);
  const calibrationMode = useStudioStore((s) => s.calibrationMode);
  const retrackMode = useStudioStore((s) => s.retrackMode);
  const activeLandmarkKey = useStudioStore((s) => s.activeLandmarkKey);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const previewWireframe = useStudioStore((s) => s.previewWireframe);
  const activeTool = useStudioStore((s) => s.activeTool);

  const setCurrentFrame = useStudioStore((s) => s.setCurrentFrame);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);
  const setPlaybackRate = useStudioStore((s) => s.setPlaybackRate);
  const setShowSkeleton = useStudioStore((s) => s.setShowSkeleton);
  const setShowWireframe = useStudioStore((s) => s.setShowWireframe);
  const setActiveLandmarkKey = useStudioStore((s) => s.setActiveLandmarkKey);
  const setRetrackMode = useStudioStore((s) => s.setRetrackMode);
  const setActiveTool = useStudioStore((s) => s.setActiveTool);

  const nFrames = Math.max(telemetry.num_frames, 1);
  const proxyFps = Math.max(telemetry.proxy_fps || 30.0, 1.0);
  const isRightArm = factSheet.bowling_arm === "right";

  // Aspect-preserving letterbox/pillarbox mapping from raw audit pixels to canvas pixels (eliminates portrait stretch!)
  const getVideoRect = useCallback(
    (cw: number, ch: number) => {
      const vw = Math.max(telemetry.width || 1280, 1);
      const vh = Math.max(telemetry.height || 720, 1);
      const scale = Math.min(cw / vw, ch / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      const ox = (cw - dw) * 0.5;
      const oy = (ch - dh) * 0.5;
      return { ox, oy, dw, dh, scale, vw, vh };
    },
    [telemetry.width, telemetry.height]
  );

  // Extract 4-Phase Contact Sheet thumbnails preserving exact aspect ratio (never crops portrait feet/release hand!)
  const evKey = `${videoUrl}|${factSheet.events.bfc.frame_idx}|${factSheet.events.ffc.frame_idx}|${factSheet.events.uah?.frame_idx ?? "n"}|${factSheet.events.br.frame_idx}`;
  useEffect(() => {
    let cancelled = false;
    setPhaseThumbs({});
    const ev = factSheet.events;
    const targets: Array<[string, number]> = [
      ["bfc", ev.bfc.frame_idx],
      ["ffc", ev.ffc.frame_idx],
      ...(ev.uah ? ([["uah", ev.uah.frame_idx]] as Array<[string, number]>) : []),
      ["br", ev.br.frame_idx],
    ];
    const v = document.createElement("video");
    v.src = videoUrl;
    v.muted = true;
    v.playsInline = true;
    v.crossOrigin = "anonymous";
    v.preload = "auto";

    const captureAll = async () => {
      try {
        await new Promise<void>((resolve, reject) => {
          v.onloadeddata = () => resolve();
          v.onerror = () => reject(new Error("thumb video error"));
        });
        const tw = 240;
        const th = 135;
        const off = document.createElement("canvas");
        off.width = tw;
        off.height = th;
        const ctx = off.getContext("2d");
        if (!ctx) return;

        const { ox, oy, dw, dh, scale } = getVideoRect(tw, th);
        const out: Record<string, string> = {};
        for (const [k, fIdx] of targets) {
          if (cancelled) return;
          const tSec = Math.min(Math.max(fIdx, 0), nFrames - 1) / proxyFps;
          await new Promise<void>((resolve) => {
            const onSeeked = () => {
              v.removeEventListener("seeked", onSeeked);
              resolve();
            };
            v.addEventListener("seeked", onSeeked);
            v.currentTime = tSec + 0.001;
          });
          ctx.fillStyle = "#f4f4f5";
          ctx.fillRect(0, 0, tw, th);
          ctx.drawImage(v, ox, oy, dw, dh);
          const clampedIdx = Math.min(fIdx, nFrames - 1);
          const pts = telemetry.joints2d_px[clampedIdx];
          const inFr = telemetry.joint_in_frame[clampedIdx];
          if (pts && pts.length === 24) {
            for (const [a, b, grp] of BONE_PAIRS_2D) {
              if (inFr && (!inFr[a] || !inFr[b])) continue;
              const pa = pts[a];
              const pb = pts[b];
              if (!pa || !pb) continue;
              const isActive =
                (isRightArm && (grp === "rarm" || grp === "lleg")) ||
                (!isRightArm && (grp === "larm" || grp === "rleg"));
              ctx.strokeStyle = "#ffffff";
              ctx.lineWidth = 2.4;
              ctx.beginPath();
              ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
              ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
              ctx.stroke();
              ctx.strokeStyle = isActive ? "#297a3a" : "#171717";
              ctx.lineWidth = 1.25;
              ctx.beginPath();
              ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
              ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
              ctx.stroke();
            }
          }
          out[k] = off.toDataURL("image/jpeg", 0.8);
        }
        if (!cancelled) {
          setPhaseThumbs(out);
          onPhaseThumbsCaptured?.(out);
        }
      } catch {
        // Ignore offscreen decode error
      } finally {
        v.src = "";
      }
    };
    captureAll();
    return () => {
      cancelled = true;
      v.src = "";
    };
  }, [evKey, getVideoRect]);

  const drawShape = useCallback(
    (
      ctx: CanvasRenderingContext2D,
      shape: { type: Exclude<TelestratorTool, "cursor">; points: Array<[number, number]> },
      ox: number,
      oy: number,
      scale: number,
      isDraft = false
    ) => {
      const pts = shape.points.map(([x, y]) => [ox + x * scale, oy + y * scale] as [number, number]);
      if (pts.length < 2) return;
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      if (isDraft) ctx.setLineDash([5, 4]);

      const strokeDual = (drawPath: () => void, accentColor: string) => {
        ctx.strokeStyle = "rgba(255,255,255,0.92)";
        ctx.lineWidth = 4.4;
        drawPath();
        ctx.stroke();
        ctx.strokeStyle = accentColor;
        ctx.lineWidth = 2.0;
        drawPath();
        ctx.stroke();
      };

      if (shape.type === "line") {
        strokeDual(() => {
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
        }, "#171717");
      } else if (shape.type === "plumb") {
        strokeDual(() => {
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[0][0], pts[1][1]);
        }, "#297a3a");
        ctx.setLineDash([]);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#171717";
        ctx.lineWidth = 1;
        const bx = pts[0][0] + 6;
        const by = (pts[0][1] + pts[1][1]) * 0.5;
        ctx.fillRect(bx, by - 9, 54, 16);
        ctx.strokeRect(bx, by - 9, 54, 16);
        ctx.fillStyle = "#171717";
        ctx.font = '500 9px "Geist Mono", monospace';
        ctx.fillText("PLUMB 90°", bx + 4, by + 2);
      } else if (shape.type === "circle") {
        const r = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
        strokeDual(() => {
          ctx.beginPath();
          ctx.arc(pts[0][0], pts[0][1], r, 0, Math.PI * 2);
        }, "#171717");
      } else if (shape.type === "angle") {
        strokeDual(() => {
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          ctx.lineTo(pts[1][0], pts[1][1]);
          if (pts.length >= 3) ctx.lineTo(pts[2][0], pts[2][1]);
        }, "#297a3a");
        if (pts.length >= 3) {
          const deg = compute2DAngleDeg(shape.points[0], shape.points[1], shape.points[2]);
          ctx.setLineDash([]);
          const lbl = `${deg.toFixed(1)}° [2D]`;
          const tx = pts[1][0] + 8;
          const ty = pts[1][1] - 8;
          ctx.fillStyle = "#171717";
          ctx.fillRect(tx - 3, ty - 11, lbl.length * 6.3 + 8, 16);
          ctx.fillStyle = "#ffffff";
          ctx.font = '500 10px "Geist Mono", monospace';
          ctx.fillText(lbl, tx + 1, ty);
        }
      }

      ctx.setLineDash([]);
      for (const [px, py] of pts) {
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#171717";
        ctx.lineWidth = 1.6;
        ctx.beginPath();
        ctx.arc(px, py, 3.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    },
    []
  );

  // Synchronously composite offscreen <video> + overlays onto single <canvas> (zero seek tearing, zero portrait distortion!)
  const drawComposite = useCallback(
    (frameIdx: number) => {
      const canvas = canvasRef.current;
      const vid = videoRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      const cw = canvas.width;
      const ch = canvas.height;
      const { ox, oy, dw, dh, scale } = getVideoRect(cw, ch);

      // 0. Paper backdrop + aspect-correct video frame
      ctx.fillStyle = "#f4f4f5";
      ctx.fillRect(0, 0, cw, ch);
      if (vid && vid.readyState >= 2) {
        ctx.drawImage(vid, ox, oy, dw, dh);
      }

      // 1. Projected MCC Pitch Wireframe
      const wf = previewWireframe ?? telemetry.mcc_wireframe_2d;
      if (showWireframe && wf && Object.keys(wf).length >= 2) {
        ctx.save();
        for (const [a, b] of MCC_EDGES) {
          if (wf[a] && wf[b]) {
            const isPop = a.includes("popping") && b.includes("popping");
            ctx.strokeStyle = "rgba(255,255,255,0.85)";
            ctx.lineWidth = 3.8;
            ctx.beginPath();
            ctx.moveTo(ox + wf[a][0] * scale, oy + wf[a][1] * scale);
            ctx.lineTo(ox + wf[b][0] * scale, oy + wf[b][1] * scale);
            ctx.stroke();
            ctx.strokeStyle = isPop ? "#297a3a" : "#171717";
            ctx.lineWidth = isPop ? 2.0 : 1.5;
            ctx.setLineDash(previewWireframe ? [6, 3] : []);
            ctx.beginPath();
            ctx.moveTo(ox + wf[a][0] * scale, oy + wf[a][1] * scale);
            ctx.lineTo(ox + wf[b][0] * scale, oy + wf[b][1] * scale);
            ctx.stroke();
          }
        }
        ctx.restore();
      }

      // 2. Multi-person track boxes in Retrack mode
      if (retrackMode && telemetry.all_tracks && telemetry.all_tracks.length > 0) {
        ctx.save();
        for (const tr of telemetry.all_tracks) {
          const [x1, y1, x2, y2] = tr.box_xyxy;
          ctx.strokeStyle = tr.is_selected ? "#297a3a" : "#171717";
          ctx.lineWidth = tr.is_selected ? 2.5 : 1.5;
          ctx.setLineDash(tr.is_selected ? [] : [5, 4]);
          ctx.strokeRect(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
          ctx.setLineDash([]);
          ctx.fillStyle = tr.is_selected ? "#297a3a" : "#171717";
          ctx.fillRect(ox + x1 * scale, Math.max(0, oy + y1 * scale - 16), 88, 16);
          ctx.fillStyle = "#ffffff";
          ctx.font = '500 9px "Geist Mono", monospace';
          ctx.fillText(
            tr.is_selected ? `TRACK #${tr.track_id} ✓` : `TRACK #${tr.track_id}`,
            ox + x1 * scale + 4,
            Math.max(11, oy + y1 * scale - 5)
          );
        }
        ctx.restore();
      }

      // 3. Dual-Cased Editorial 2D Skeleton
      const f = Math.min(Math.max(frameIdx, 0), nFrames - 1);
      drawnFrameRef.current = f;
      const pts = telemetry.joints2d_px[f];
      const inFr = telemetry.joint_in_frame[f];
      if (showSkeleton && pts && pts.length === 24) {
        ctx.save();
        ctx.lineCap = "round";
        for (const [a, b, grp] of BONE_PAIRS_2D) {
          if (inFr && (!inFr[a] || !inFr[b])) continue;
          const pa = pts[a];
          const pb = pts[b];
          if (!pa || !pb) continue;
          const isActive =
            (isRightArm && (grp === "rarm" || grp === "lleg")) ||
            (!isRightArm && (grp === "larm" || grp === "rleg"));

          ctx.strokeStyle = "rgba(255,255,255,0.92)";
          ctx.lineWidth = 4.6;
          ctx.beginPath();
          ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
          ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
          ctx.stroke();

          ctx.strokeStyle = isActive ? "#297a3a" : "#171717";
          ctx.lineWidth = isActive ? 2.4 : 1.9;
          ctx.beginPath();
          ctx.moveTo(ox + pa[0] * scale, oy + pa[1] * scale);
          ctx.lineTo(ox + pb[0] * scale, oy + pb[1] * scale);
          ctx.stroke();
        }
        for (let i = 0; i < 24; i++) {
          if (inFr && !inFr[i]) continue;
          const p = pts[i];
          if (!p) continue;
          ctx.fillStyle = "#ffffff";
          ctx.strokeStyle = "#171717";
          ctx.lineWidth = 1.6;
          ctx.beginPath();
          ctx.arc(ox + p[0] * scale, oy + p[1] * scale, 3.2, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        ctx.restore();
      }

      // 4. Exact-Frame Telestrator Annotations (`sh.frame === f`, zero cross-frame smear at 25 fps!)
      const frameShapes = annotations.filter((sh) => sh.frame === f);
      for (const sh of frameShapes) {
        drawShape(ctx, sh, ox, oy, scale, false);
      }
      if (activeTool !== "cursor" && draftPtsRef.current.length > 0 && hoverPxRef.current) {
        const previewPts: Array<[number, number]> = [...draftPtsRef.current, hoverPxRef.current];
        drawShape(ctx, { type: activeTool, points: previewPts }, ox, oy, scale, true);
      }

      // 5. Draft 4-Click MCC Calibration Markers
      if (calibrationMode) {
        ctx.save();
        Object.entries(draftLandmarks).forEach(([k, [x, y]]) => {
          const px = ox + x * scale;
          const py = oy + y * scale;
          ctx.fillStyle = "#297a3a";
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(px, py, 5, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
          ctx.fillStyle = "#171717";
          ctx.fillRect(px + 7, py - 10, k.length * 5.8 + 8, 15);
          ctx.fillStyle = "#ffffff";
          ctx.font = '500 9px "Geist Mono", monospace';
          ctx.fillText(k, px + 11, py + 1);
        });
        ctx.restore();
      }

      // 6. 4x Calibration Pixel Loupe
      const loupe = loupeRef.current;
      if (loupe && calibrationMode && hoverPxRef.current && vid && vid.readyState >= 2) {
        const lctx = loupe.getContext("2d");
        if (lctx) {
          const [hx, hy] = hoverPxRef.current;
          const vScaleX = vid.videoWidth / Math.max(telemetry.width, 1);
          const vScaleY = vid.videoHeight / Math.max(telemetry.height, 1);
          const srcW = 36;
          const srcH = 36;
          lctx.clearRect(0, 0, loupe.width, loupe.height);
          lctx.imageSmoothingEnabled = false;
          lctx.drawImage(
            vid,
            Math.max(0, hx * vScaleX - srcW / 2),
            Math.max(0, hy * vScaleY - srcH / 2),
            srcW,
            srcH,
            0,
            0,
            loupe.width,
            loupe.height
          );
          lctx.strokeStyle = "#297a3a";
          lctx.lineWidth = 1.4;
          lctx.beginPath();
          lctx.moveTo(loupe.width / 2, 0);
          lctx.lineTo(loupe.width / 2, loupe.height);
          lctx.moveTo(0, loupe.height / 2);
          lctx.lineTo(loupe.width, loupe.height / 2);
          lctx.stroke();
        }
      }
    },
    [
      getVideoRect,
      telemetry,
      nFrames,
      showSkeleton,
      showWireframe,
      calibrationMode,
      retrackMode,
      draftLandmarks,
      previewWireframe,
      annotations,
      activeTool,
      isRightArm,
      drawShape,
    ]
  );

  // Coalesced zero-tear video seeker when scrubbing / paused
  const requestSeekToFrame = useCallback(
    (frameIdx: number) => {
      const vid = videoRef.current;
      const clamped = Math.min(Math.max(frameIdx, 0), nFrames - 1);
      if (!vid || vid.readyState < 1) {
        drawComposite(clamped);
        return;
      }
      const targetSec = clamped / proxyFps;
      if (Math.abs(vid.currentTime - targetSec) <= 0.004) {
        drawComposite(clamped);
        return;
      }
      if (seekingRef.current) {
        pendingSeekFrameRef.current = clamped;
        return;
      }
      seekingRef.current = true;
      pendingSeekFrameRef.current = null;
      vid.currentTime = targetSec;
    },
    [nFrames, proxyFps, drawComposite]
  );

  const handleVideoSeeked = useCallback(() => {
    seekingRef.current = false;
    const vid = videoRef.current;
    const pending = pendingSeekFrameRef.current;
    if (pending !== null) {
      pendingSeekFrameRef.current = null;
      requestSeekToFrame(pending);
      return;
    }
    const f = vid ? Math.min(Math.round(vid.currentTime * proxyFps), nFrames - 1) : useStudioStore.getState().currentFrame;
    drawComposite(f);
  }, [proxyFps, nFrames, drawComposite, requestSeekToFrame]);

  // Reset coalesced seek lock whenever videoUrl changes (HTML5 load algorithm aborts seeks without firing onSeeked)
  useEffect(() => {
    seekingRef.current = false;
    pendingSeekFrameRef.current = null;
  }, [videoUrl]);

  useEffect(() => {
    if (!isPlaying) {
      requestSeekToFrame(currentFrame);
    }
  }, [currentFrame, isPlaying, requestSeekToFrame]);

  // Re-composite immediately when toggles/annotations/drafts change while paused
  useEffect(() => {
    if (!isPlaying) {
      drawComposite(currentFrame);
    }
  }, [showSkeleton, showWireframe, calibrationMode, retrackMode, draftLandmarks, previewWireframe, annotations, activeTool, drawComposite, currentFrame, isPlaying]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.playbackRate = playbackRate;
    if (isPlaying) {
      vid.play().catch(() => setIsPlaying(false));
    } else {
      vid.pause();
    }
  }, [isPlaying, playbackRate, setIsPlaying]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || !isPlaying) return;
    let active = true;
    const step = () => {
      if (!active || !vid) return;
      const f = Math.min(Math.round(vid.currentTime * proxyFps), nFrames - 1);
      useStudioStore.getState().setCurrentFrame(f);
      drawComposite(f);
      if (f >= nFrames - 1) {
        vid.currentTime = 0;
      }
      rafRef.current = requestAnimationFrame(step);
    };
    rafRef.current = requestAnimationFrame(step);
    return () => {
      active = false;
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    };
  }, [isPlaying, proxyFps, nFrames, drawComposite]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) {
        return;
      }
      if (e.code === "Space") {
        e.preventDefault();
        setIsPlaying(!useStudioStore.getState().isPlaying);
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        setIsPlaying(false);
        setCurrentFrame(Math.min(useStudioStore.getState().currentFrame + 1, nFrames - 1));
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        setIsPlaying(false);
        setCurrentFrame(Math.max(useStudioStore.getState().currentFrame - 1, 0));
      } else if (e.code === "Escape") {
        draftPtsRef.current = [];
        dragStartRef.current = null;
        setDraftCount(0);
        setActiveTool("cursor");
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [nFrames, setCurrentFrame, setIsPlaying, setActiveTool]);

  // Exact aspect-preserving pixel coordinate inversion from mouse event -> raw audit pixels (P0-1)
  const eventToRawPx = (e: React.MouseEvent<HTMLCanvasElement>): [number, number] | null => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const cx = ((e.clientX - rect.left) / Math.max(rect.width, 1)) * canvas.width;
    const cy = ((e.clientY - rect.top) / Math.max(rect.height, 1)) * canvas.height;
    const { ox, oy, scale, vw, vh } = getVideoRect(canvas.width, canvas.height);
    const xRaw = Math.max(0, Math.min(vw, (cx - ox) / Math.max(scale, 1e-6)));
    const yRaw = Math.max(0, Math.min(vh, (cy - oy) / Math.max(scale, 1e-6)));
    return [Number(xRaw.toFixed(1)), Number(yRaw.toFixed(1))];
  };

  const handleCanvasMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = eventToRawPx(e);
    if (!pt) return;

    if (calibrationMode) {
      onLandmarkClick(activeLandmarkKey, pt);
      const idx = CALIBRATION_ORDER.indexOf(activeLandmarkKey);
      if (idx >= 0 && idx + 1 < CALIBRATION_ORDER.length) {
        setActiveLandmarkKey(CALIBRATION_ORDER[idx + 1]);
      }
      return;
    }

    if (retrackMode && onRetrackBowler) {
      const ax = Math.max(0, Math.min(1, pt[0] / Math.max(telemetry.width, 1)));
      const ay = Math.max(0, Math.min(1, pt[1] / Math.max(telemetry.height, 1)));
      setRetrackMode(false);
      onRetrackBowler([Number(ax.toFixed(4)), Number(ay.toFixed(4))]);
      return;
    }

    if (activeTool === "cursor") return;
    setIsPlaying(false);

    if (activeTool === "angle") {
      const next = [...draftPtsRef.current, pt];
      if (next.length >= 3) {
        const shape: TelestratorShape = {
          id: `ann_${Date.now()}`,
          frame: currentFrame,
          type: "angle",
          points: next.slice(0, 3),
        };
        draftPtsRef.current = [];
        setDraftCount(0);
        onSaveAnnotations([...annotations, shape]);
      } else {
        draftPtsRef.current = next;
        setDraftCount(next.length);
      }
      drawComposite(currentFrame);
      return;
    }

    dragStartRef.current = pt;
    draftPtsRef.current = [pt];
    setDraftCount(1);
  };

  const handleCanvasMouseUp = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (activeTool === "cursor" || activeTool === "angle" || !dragStartRef.current) return;
    const endPt = eventToRawPx(e);
    const startPt = dragStartRef.current;
    dragStartRef.current = null;
    draftPtsRef.current = [];
    setDraftCount(0);
    if (!endPt) return;
    if (Math.hypot(endPt[0] - startPt[0], endPt[1] - startPt[1]) < 8) {
      drawComposite(currentFrame);
      return;
    }
    const shape: TelestratorShape = {
      id: `ann_${Date.now()}`,
      frame: currentFrame,
      type: activeTool,
      points: [startPt, endPt],
    };
    onSaveAnnotations([...annotations, shape]);
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pt = eventToRawPx(e);
    if (!pt) return;
    hoverPxRef.current = pt;
    if (calibrationMode || activeTool !== "cursor") {
      drawComposite(currentFrame);
    }
  };

  const handleSnapshotPng = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawComposite(currentFrame);
    const a = document.createElement("a");
    a.href = canvas.toDataURL("image/png");
    a.download = `sportsposture_frame_${currentFrame}.png`;
    a.click();
  };

  const handleUndoShape = () => {
    if (annotations.length === 0) return;
    onSaveAnnotations(annotations.slice(0, -1));
  };

  const handleClearFrameShapes = () => {
    const remaining = annotations.filter((sh) => sh.frame !== currentFrame);
    onSaveAnnotations(remaining);
  };

  const ev = factSheet.events;
  const tRelMs = telemetry.time_rel_ffc_ms[Math.min(currentFrame, nFrames - 1)] ?? 0;
  const frameShapeCount = annotations.filter((sh) => sh.frame === currentFrame).length;
  const hasMultipleTracks =
    (telemetry.all_tracks?.length ?? 0) > 1 &&
    !factSheet.audit.video_path.startsWith("demo:") &&
    !factSheet.audit.video_path.startsWith("fixture:");

  const phases = useMemo(
    () => [
      { id: "bfc", label: "BFC", sub: "Back Foot", field: "bfc_frame" as const, frame: ev.bfc.frame_idx, ts: ev.bfc.timestamp_s },
      { id: "ffc", label: "FFC", sub: "Front Foot", field: "ffc_frame" as const, frame: ev.ffc.frame_idx, ts: ev.ffc.timestamp_s },
      ...(ev.uah
        ? [{ id: "uah", label: "UAH", sub: "Arm Horiz.", field: "uah_frame" as const, frame: ev.uah.frame_idx, ts: ev.uah.timestamp_s }]
        : []),
      { id: "br", label: "BR", sub: "Release", field: "br_frame" as const, frame: ev.br.frame_idx, ts: ev.br.timestamp_s },
    ],
    [ev]
  );

  const tools: Array<{ id: TelestratorTool; label: string; icon: React.ReactNode; hint: string }> = [
    { id: "cursor", label: "Select", icon: <MousePointer2 className="w-3.5 h-3.5" />, hint: "Default scrub cursor" },
    { id: "line", label: "Line", icon: <Minus className="w-3.5 h-3.5" />, hint: "Click and drag to draw a 2D reference line" },
    { id: "plumb", label: "Plumb", icon: <Crosshair className="w-3.5 h-3.5" />, hint: "Click and drag to drop a vertical plumb line" },
    { id: "angle", label: "2D Angle", icon: <Triangle className="w-3.5 h-3.5" />, hint: "Click 3 points (end -> vertex -> end) for a 2D camera-plane angle" },
    { id: "circle", label: "Circle", icon: <Circle className="w-3.5 h-3.5" />, hint: "Click center and drag to highlight a joint or release zone" },
  ];

  return (
    <div className="space-y-3">
      {/* Main Video + Composite Canvas + Telestrator Card */}
      <div className="v-card overflow-hidden flex flex-col">
        {/* Top Toolbar */}
        <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[#ebebeb] bg-[#ffffff]">
          <div className="flex items-center gap-2 font-mono text-[11px] text-[#4d4d4d]">
            <span className="text-[#000000]">▲</span>
            <span className="text-[#171717] font-medium v-tabular">
              f {currentFrame}/{nFrames - 1}
            </span>
            <span className="text-[#c9c9c9]">·</span>
            <span className="v-tabular">
              {tRelMs >= 0 ? `+${tRelMs.toFixed(0)}` : tRelMs.toFixed(0)} ms rel FFC
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-1.5">
            <button
              onClick={() => setShowSkeleton(!showSkeleton)}
              className={`px-2 py-1 rounded-[5px] font-mono text-[11px] border transition ${
                showSkeleton
                  ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                  : "bg-[#ffffff] text-[#666666] border-[#ebebeb]"
              }`}
            >
              Skeleton
            </button>
            <button
              onClick={() => setShowWireframe(!showWireframe)}
              className={`px-2 py-1 rounded-[5px] font-mono text-[11px] border transition ${
                showWireframe
                  ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                  : "bg-[#ffffff] text-[#666666] border-[#ebebeb]"
              }`}
            >
              Crease
            </button>
            {hasMultipleTracks && onRetrackBowler && (
              <button
                onClick={() => setRetrackMode(!retrackMode)}
                className={`px-2 py-1 rounded-[5px] font-mono text-[11px] border flex items-center gap-1 transition ${
                  retrackMode
                    ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                    : "bg-[#ffffff] text-[#4d4d4d] border-[#ebebeb] hover:border-[#171717]"
                }`}
                title="Multiple people detected — click the actual bowler to re-lock 3D tracking"
              >
                {retrackMode ? <span className="text-[#297a3a]">●</span> : <UserCheck className="w-3 h-3" />}
                Wrong person?
              </button>
            )}
            <div className="flex items-center gap-0.5 pl-1 border-l border-[#ebebeb]">
              {[30, 60, 120, 240].map((fpsOpt) => {
                const isActive = Math.abs(factSheet.audit.fps - fpsOpt) < 2;
                return (
                  <button
                    key={fpsOpt}
                    onClick={() => onFpsChange(fpsOpt)}
                    className={`px-1.5 py-1 rounded-[4px] font-mono text-[10px] transition ${
                      isActive
                        ? "bg-[#171717] text-[#ffffff]"
                        : "text-[#666666] hover:bg-[#f4f4f5] hover:text-[#171717]"
                    }`}
                    title={`Override true capture frame rate to ${fpsOpt} fps (instant <35 ms CPU recompute)`}
                  >
                    {isActive && <span className="text-[#297a3a] mr-0.5">●</span>}
                    {fpsOpt}
                  </button>
                );
              })}
              <span className="font-mono text-[10px] text-[#8f8f8f] pl-0.5">fps</span>
            </div>
          </div>
        </div>

        {/* Hidden Offscreen Video + Single Synchronous Composite Canvas */}
        <div className="relative bg-[#f4f4f5] aspect-video w-full overflow-hidden select-none">
          <video
            ref={videoRef}
            src={videoUrl}
            muted
            playsInline
            preload="auto"
            onLoadedData={() => requestSeekToFrame(useStudioStore.getState().currentFrame)}
            onSeeked={handleVideoSeeked}
            onEmptied={() => {
              seekingRef.current = false;
              pendingSeekFrameRef.current = null;
            }}
            onError={() => {
              seekingRef.current = false;
              pendingSeekFrameRef.current = null;
            }}
            className="hidden"
          />
          <canvas
            ref={canvasRef}
            width={1280}
            height={720}
            onMouseDown={handleCanvasMouseDown}
            onMouseUp={handleCanvasMouseUp}
            onMouseMove={handleCanvasMove}
            onMouseLeave={() => {
              hoverPxRef.current = null;
              if (calibrationMode) drawComposite(currentFrame);
            }}
            className={`w-full h-full block ${
              calibrationMode || retrackMode || activeTool !== "cursor" ? "cursor-crosshair" : "cursor-default"
            }`}
          />

          {/* Interactive Mode Banner Overlay */}
          {calibrationMode && (
            <div className="absolute top-3 left-3 px-2.5 py-1.5 rounded-[6px] bg-[#171717] text-[#ffffff] font-mono text-[11px] flex items-center gap-2">
              <span className="text-[#297a3a]">●</span>
              Click pitch landmark: <strong>{activeLandmarkKey}</strong>
            </div>
          )}
          {retrackMode && (
            <div className="absolute top-3 left-3 px-2.5 py-1.5 rounded-[6px] bg-[#171717] text-[#ffffff] font-mono text-[11px] flex items-center gap-1.5">
              <span className="text-[#297a3a]">●</span>
              Click directly on the bowler&apos;s torso to re-lock 3D tracking
            </div>
          )}
          {activeTool !== "cursor" && !calibrationMode && !retrackMode && (
            <div className="absolute top-3 left-3 px-2.5 py-1 rounded-[6px] bg-[#ffffff]/95 border border-[#171717] font-mono text-[10px] text-[#171717]">
              TELESTRATOR: {activeTool.toUpperCase()}
              {activeTool === "angle"
                ? ` (click ${3 - draftCount} pt${3 - draftCount === 1 ? "" : "s"} — 2D camera-plane only)`
                : " (click & drag)"}
            </div>
          )}

          {/* 4x Calibration Loupe */}
          {calibrationMode && (
            <div className="absolute top-3 right-3 rounded-[6px] overflow-hidden border border-[#171717] bg-[#ffffff]">
              <canvas ref={loupeRef} width={116} height={116} />
              <div className="text-[9px] font-mono text-center bg-[#171717] text-[#ffffff] py-0.5">
                4× LOUPE
              </div>
            </div>
          )}
        </div>

        {/* Telestrator Drawing Strip */}
        <div className="px-3 py-2 border-t border-[#ebebeb] bg-[#fafafa] flex flex-wrap items-center justify-between gap-1.5">
          <div className="flex flex-wrap items-center gap-1">
            {!compact && <span className="v-eyebrow mr-1">DRAW</span>}
            {tools.map((t) => {
              const active = activeTool === t.id;
              return (
                <button
                  key={t.id}
                  onClick={() => {
                    draftPtsRef.current = [];
                    dragStartRef.current = null;
                    setDraftCount(0);
                    setActiveTool(t.id);
                  }}
                  title={t.hint}
                  className={`h-7 ${compact ? "px-2" : "px-2"} rounded-[5px] text-[11px] font-mono flex items-center gap-1 border transition ${
                    active
                      ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                      : "bg-[#ffffff] text-[#4d4d4d] border-[#ebebeb] hover:border-[#171717] hover:text-[#171717]"
                  }`}
                >
                  {t.icon}
                  {!compact && <span className="hidden sm:inline">{t.label}</span>}
                </button>
              );
            })}
          </div>
          <div className="flex items-center gap-1">
            <button
              onClick={handleUndoShape}
              disabled={annotations.length === 0}
              className="v-btn-ghost h-7 px-2 font-mono text-[11px] flex items-center gap-1"
              title="Undo last drawing"
            >
              <Undo2 className="w-3 h-3" />
              {!compact && <span>Undo</span>}
            </button>
            <button
              onClick={handleClearFrameShapes}
              disabled={frameShapeCount === 0}
              className="v-btn-ghost h-7 px-2 font-mono text-[11px] flex items-center gap-1"
              title="Clear drawings on this frame"
            >
              <Trash2 className="w-3 h-3" />
              <span>{compact ? `(${frameShapeCount})` : `Clear (${frameShapeCount})`}</span>
            </button>
            <button
              onClick={handleSnapshotPng}
              className="v-btn-ghost h-7 px-2 font-mono text-[11px] flex items-center gap-1"
              title="Download annotated frame as PNG"
            >
              <Camera className="w-3 h-3" />
              <span>PNG</span>
            </button>
          </div>
        </div>

        {/* Scrubber & Playback Transport */}
        <div className="px-3 py-2.5 border-t border-[#ebebeb] bg-[#ffffff] space-y-2">
          <input
            type="range"
            aria-label="Frame scrubber"
            min={0}
            max={Math.max(nFrames - 1, 1)}
            value={Math.min(currentFrame, Math.max(nFrames - 1, 1))}
            onChange={(e) => {
              setIsPlaying(false);
              setCurrentFrame(Number(e.target.value));
            }}
            className="w-full accent-[#171717] cursor-pointer h-1.5"
          />

          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-1">
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setCurrentFrame(Math.max(0, currentFrame - 1));
                }}
                className="v-btn-ghost w-7 h-7 flex items-center justify-center"
                title="Previous frame (←)"
              >
                <SkipBack className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setIsPlaying(!isPlaying)}
                className="v-btn-primary h-7 px-3 text-[12px] flex items-center gap-1.5"
                title="Play / Pause (Space)"
              >
                {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
                {isPlaying ? "Pause" : "Play"}
              </button>
              <button
                onClick={() => {
                  setIsPlaying(false);
                  setCurrentFrame(Math.min(nFrames - 1, currentFrame + 1));
                }}
                className="v-btn-ghost w-7 h-7 flex items-center justify-center"
                title="Next frame (→)"
              >
                <SkipForward className="w-3.5 h-3.5" />
              </button>
              <div className="flex items-center gap-0.5 ml-1">
                {[0.1, 0.25, 0.5, 1.0].map((r) => (
                  <button
                    key={r}
                    onClick={() => setPlaybackRate(r)}
                    className={`px-1.5 h-7 rounded-[5px] font-mono text-[11px] transition ${
                      playbackRate === r
                        ? "bg-[#171717] text-[#ffffff]"
                        : "text-[#666666] hover:bg-[#f4f4f5]"
                    }`}
                  >
                    {r}×
                  </button>
                ))}
              </div>
            </div>

            <div className="font-mono text-[11px] text-[#666666] v-tabular">
              Stride {ev.delivery_stride_duration_ms.toFixed(0)} ms · FFC→BR {ev.ffc_to_br_duration_ms.toFixed(0)} ms
            </div>
          </div>
        </div>
      </div>

      {/* Feature #2: 4-Phase Visual Contact Sheet Strip (BFC -> FFC -> UAH -> BR) */}
      <div className="v-card p-3">
        <div className="flex flex-wrap items-center justify-between gap-1 mb-2">
          <div className="v-eyebrow">▲ PHASE_CONTACT_SHEET · Click or nudge ±1f</div>
          <span className="font-mono text-[10px] text-[#297a3a]">
            {ev.is_complete_delivery ? "✓ CAUSAL SEQUENCE VERIFIED" : "INCOMPLETE GATE"}
          </span>
        </div>
        <div
          className={`grid gap-2 ${
            compact
              ? "grid-cols-2"
              : phases.length === 4
              ? "grid-cols-2 sm:grid-cols-4"
              : "grid-cols-3"
          }`}
        >
          {phases.map((p) => {
            const isNear = Math.abs(currentFrame - p.frame) <= 1;
            return (
              <div
                key={p.id}
                className={`rounded-[6px] overflow-hidden border transition ${
                  isNear ? "border-[#171717] bg-[#fafafa]" : "border-[#ebebeb] bg-[#ffffff]"
                }`}
              >
                <button
                  onClick={() => {
                    setIsPlaying(false);
                    setCurrentFrame(p.frame);
                  }}
                  className="block w-full text-left"
                >
                  <div className="aspect-video bg-[#f4f4f5] relative overflow-hidden">
                    {phaseThumbs[p.id] ? (
                      <img src={phaseThumbs[p.id]} alt={p.sub} className="w-full h-full object-contain bg-[#f4f4f5]" />
                    ) : (
                      <div className="w-full h-full flex items-center justify-center font-mono text-[10px] text-[#8f8f8f]">
                        f {p.frame}
                      </div>
                    )}
                    <span className="absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded-[4px] font-mono text-[10px] font-medium bg-[#171717] text-[#ffffff]">
                      {p.id === "br" && <span className="text-[#297a3a] mr-1">●</span>}
                      {p.label}
                    </span>
                  </div>
                </button>
                <div className="px-2 py-1.5 flex items-center justify-between gap-1">
                  <div className="min-w-0">
                    <div className="text-[11px] font-[500] text-[#171717] truncate">{p.sub}</div>
                    <div className="font-mono text-[10px] text-[#666666] v-tabular whitespace-nowrap">
                      f {p.frame} · {p.ts.toFixed(2)}s
                    </div>
                  </div>
                  <div className="flex items-center gap-0.5 shrink-0">
                    <button
                      onClick={() => onNudgeEvent(p.field, Math.max(0, p.frame - 1))}
                      className="v-btn-ghost w-5 h-5 font-mono text-[10px] flex items-center justify-center"
                      title={`Move ${p.label} -1 frame`}
                    >
                      -
                    </button>
                    <button
                      onClick={() => onNudgeEvent(p.field, Math.min(nFrames - 1, p.frame + 1))}
                      className="v-btn-ghost w-5 h-5 font-mono text-[10px] flex items-center justify-center"
                      title={`Move ${p.label} +1 frame`}
                    >
                      +
                    </button>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
