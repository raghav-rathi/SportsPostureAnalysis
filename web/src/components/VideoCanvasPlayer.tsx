import React, { useEffect, useRef, useState } from "react";
import { Pause, Play, SkipBack, SkipForward, Crosshair, Eye, EyeOff, Grid } from "lucide-react";
import { useStudioStore } from "../store";
import type { BiomechanicalFactSheet, TelemetryV2 } from "../types";

const SKELETON_EDGES: Array<[number, number, string]> = [
  [0, 3, "#f078ff"], [3, 6, "#f078ff"], [6, 9, "#f078ff"], [9, 12, "#f078ff"], [12, 15, "#f078ff"],
  [0, 1, "#38bdf8"], [1, 4, "#38bdf8"], [4, 7, "#38bdf8"], [7, 10, "#38bdf8"],
  [0, 2, "#fb923c"], [2, 5, "#fb923c"], [5, 8, "#fb923c"], [8, 11, "#fb923c"],
  [9, 13, "#60a5fa"], [13, 16, "#60a5fa"], [16, 18, "#60a5fa"], [18, 20, "#60a5fa"], [20, 22, "#60a5fa"],
  [9, 14, "#4ade80"], [14, 17, "#4ade80"], [17, 19, "#4ade80"], [19, 21, "#4ade80"], [21, 23, "#4ade80"],
];

const WIREFRAME_SEGMENTS: Array<[string, string, string]> = [
  ["bowling_crease_off", "bowling_crease_leg", "#38bdf8"],
  ["popping_crease_off", "popping_crease_leg", "#22c55e"],
  ["bowling_crease_off", "popping_crease_off", "#94a3b8"],
  ["bowling_crease_leg", "popping_crease_leg", "#94a3b8"],
  ["stump_off_base", "stump_off_top", "#facc15"],
  ["stump_middle_base", "stump_middle_top", "#facc15"],
  ["stump_leg_base", "stump_leg_top", "#facc15"],
];

interface Props {
  videoUrl: string;
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  onNudgeEvent: (phase: "bfc_frame" | "ffc_frame" | "uah_frame" | "br_frame", frame: number) => void;
  onLandmarkClick: (key: string, pt: [number, number]) => void;
  onFpsChange: (fps: number) => void;
}

export const VideoCanvasPlayer: React.FC<Props> = ({
  videoUrl,
  telemetry,
  factSheet,
  onNudgeEvent,
  onLandmarkClick,
  onFpsChange,
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const hoverPxRef = useRef<[number, number] | null>(null);
  const pendingSeekRef = useRef<number | null>(null);
  const seekingRef = useRef<boolean>(false);

  const currentFrame = useStudioStore((s) => s.currentFrame);
  const isPlaying = useStudioStore((s) => s.isPlaying);
  const playbackRate = useStudioStore((s) => s.playbackRate);
  const showSkeleton = useStudioStore((s) => s.showSkeleton);
  const showWireframe = useStudioStore((s) => s.showWireframe);
  const calibrationMode = useStudioStore((s) => s.calibrationMode);
  const activeLandmarkKey = useStudioStore((s) => s.activeLandmarkKey);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const previewWireframe = useStudioStore((s) => s.previewWireframe);

  const setCurrentFrame = useStudioStore((s) => s.setCurrentFrame);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);
  const setPlaybackRate = useStudioStore((s) => s.setPlaybackRate);
  const setShowSkeleton = useStudioStore((s) => s.setShowSkeleton);
  const setShowWireframe = useStudioStore((s) => s.setShowWireframe);

  const [videoReady, setVideoReady] = useState(false);

  const nFrames = Math.max(telemetry.num_frames, 1);
  const proxyFps = Math.max(telemetry.proxy_fps || 30, 1);

  const getVideoRect = (cw: number, ch: number) => {
    const vw = Math.max(telemetry.width, 1);
    const vh = Math.max(telemetry.height, 1);
    const scale = Math.min(cw / vw, ch / vh);
    const drawW = vw * scale;
    const drawH = vh * scale;
    return { ox: (cw - drawW) * 0.5, oy: (ch - drawH) * 0.5, drawW, drawH, scale };
  };

  const clientToCanvas = (e: React.MouseEvent<HTMLCanvasElement>, canvas: HTMLCanvasElement): [number, number] | null => {
    const rect = canvas.getBoundingClientRect();
    const cssScale = Math.min(rect.width / canvas.width, rect.height / canvas.height);
    if (cssScale <= 0) return null;
    const padX = (rect.width - canvas.width * cssScale) * 0.5;
    const padY = (rect.height - canvas.height * cssScale) * 0.5;
    const x = (e.clientX - rect.left - padX) / cssScale;
    const y = (e.clientY - rect.top - padY) / cssScale;
    if (x < 0 || y < 0 || x > canvas.width || y > canvas.height) return null;
    return [x, y];
  };

  // Single-canvas synchronous draw: guarantees 0.0-frame pixel-to-skeleton tear & aspect-preserving letterbox!
  const paintComposite = (frameIdx: number) => {
    const canvas = canvasRef.current;
    const vid = videoRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const cw = canvas.width;
    const ch = canvas.height;
    ctx.fillStyle = "#0b0f17";
    ctx.fillRect(0, 0, cw, ch);

    const vr = getVideoRect(cw, ch);
    const toX = (vx: number) => vr.ox + vx * vr.scale;
    const toY = (vy: number) => vr.oy + vy * vr.scale;

    const fClamped = Math.max(0, Math.min(frameIdx, nFrames - 1));
    if (vid && vid.readyState >= 2) {
      ctx.drawImage(vid, vr.ox, vr.oy, vr.drawW, vr.drawH);
    }

    const st = useStudioStore.getState();

    // 1. Live MCC 3D pitch wireframe (either live preview or saved calibration)
    const wf = st.previewWireframe ?? telemetry.mcc_wireframe_2d;
    if (st.showWireframe && wf && Object.keys(wf).length > 0) {
      ctx.save();
      ctx.lineWidth = 2;
      for (const [a, b, col] of WIREFRAME_SEGMENTS) {
        if (wf[a] && wf[b]) {
          ctx.strokeStyle = col;
          ctx.beginPath();
          ctx.moveTo(toX(wf[a][0]), toY(wf[a][1]));
          ctx.lineTo(toX(wf[b][0]), toY(wf[b][1]));
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // 2. Pixel-locked 2D SMPL-24 skeleton
    const j2 = telemetry.joints2d_px[fClamped];
    const jIn = telemetry.joint_in_frame[fClamped];
    if (st.showSkeleton && j2) {
      ctx.save();
      ctx.lineCap = "round";
      for (const [a, b, col] of SKELETON_EDGES) {
        const pa = j2[a];
        const pb = j2[b];
        if (!pa || !pb || !Number.isFinite(pa[0]) || !Number.isFinite(pb[0])) continue;
        ctx.strokeStyle = col;
        ctx.globalAlpha = jIn && jIn[a] && jIn[b] ? 0.95 : 0.35;
        ctx.lineWidth = 2.6;
        ctx.beginPath();
        ctx.moveTo(toX(pa[0]), toY(pa[1]));
        ctx.lineTo(toX(pb[0]), toY(pb[1]));
        ctx.stroke();
      }
      ctx.globalAlpha = 1.0;
      for (let i = 0; i < j2.length; i++) {
        const p = j2[i];
        if (!p || !Number.isFinite(p[0])) continue;
        ctx.fillStyle = jIn && !jIn[i] ? "#f59e0b" : "#ffffff";
        ctx.beginPath();
        ctx.arc(toX(p[0]), toY(p[1]), 3.2, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    // 3. Draft 4-Click MCC calibration pins + 4x Corner Pixel Loupe
    if (st.calibrationMode) {
      ctx.save();
      for (const [k, pt] of Object.entries(st.draftLandmarks)) {
        const px = toX(pt[0]);
        const py = toY(pt[1]);
        ctx.strokeStyle = "#facc15";
        ctx.fillStyle = "#0b0f17";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(px, py, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        ctx.fillStyle = "#fde047";
        ctx.font = "11px monospace";
        ctx.fillText(k.replace("_crease_", " "), px + 9, py + 4);
      }
      const hov = hoverPxRef.current;
      if (hov && vid && vid.readyState >= 2) {
        const lSize = 136;
        const lPad = 14;
        const lx = cw - lSize - lPad;
        const ly = lPad;
        const zoom = 4;
        const vxNorm = (hov[0] - vr.ox) / Math.max(vr.drawW, 1);
        const vyNorm = (hov[1] - vr.oy) / Math.max(vr.drawH, 1);
        const srcW = (lSize / zoom) * (vid.videoWidth / Math.max(vr.drawW, 1));
        const srcH = (lSize / zoom) * (vid.videoHeight / Math.max(vr.drawH, 1));
        const srcX = vxNorm * vid.videoWidth - srcW / 2;
        const srcY = vyNorm * vid.videoHeight - srcH / 2;
        ctx.strokeStyle = "#38bdf8";
        ctx.lineWidth = 2;
        ctx.fillStyle = "#0b0f17";
        ctx.fillRect(lx, ly, lSize, lSize);
        ctx.drawImage(vid, srcX, srcY, srcW, srcH, lx, ly, lSize, lSize);
        ctx.strokeRect(lx, ly, lSize, lSize);
        // Crosshair inside loupe
        ctx.strokeStyle = "#fb7185";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(lx + lSize / 2, ly);
        ctx.lineTo(lx + lSize / 2, ly + lSize);
        ctx.moveTo(lx, ly + lSize / 2);
        ctx.lineTo(lx + lSize, ly + lSize / 2);
        ctx.stroke();
        ctx.fillStyle = "rgba(11,15,23,0.85)";
        ctx.fillRect(lx, ly + lSize - 18, lSize, 18);
        ctx.fillStyle = "#38bdf8";
        ctx.font = "10px monospace";
        ctx.fillText(`4x LOUPE: ${st.activeLandmarkKey.slice(0, 14)}`, lx + 5, ly + lSize - 5);
      }
      ctx.restore();
    }

    // 4. Compact phase HUD overlay (top-left)
    const ev = factSheet.events;
    const tMs = telemetry.time_rel_ffc_ms[fClamped] ?? 0;
    const phaseBadges: string[] = [];
    if (ev.is_complete_delivery) {
      if (Math.abs(fClamped - ev.bfc.frame_idx) <= 0) phaseBadges.push("BFC");
      if (Math.abs(fClamped - ev.ffc.frame_idx) <= 0) phaseBadges.push("FFC");
      if (ev.uah && Math.abs(fClamped - ev.uah.frame_idx) <= 0) phaseBadges.push("UAH");
      if (Math.abs(fClamped - ev.br.frame_idx) <= 0) phaseBadges.push("BALL RELEASE");
    }
    ctx.save();
    ctx.fillStyle = "rgba(11, 15, 23, 0.78)";
    ctx.fillRect(12, 12, 290, 44);
    ctx.fillStyle = "#e2e8f0";
    ctx.font = "bold 12px monospace";
    ctx.fillText(
      `FRAME ${fClamped}/${nFrames - 1} (src ${telemetry.source_frame_indices[fClamped] ?? fClamped})`,
      22,
      30
    );
    ctx.fillStyle = "#38bdf8";
    ctx.font = "11px monospace";
    ctx.fillText(`t_FFC: ${tMs >= 0 ? "+" : ""}${tMs.toFixed(1)} ms${phaseBadges.length ? "  [" + phaseBadges.join("/") + "]" : ""}`, 22, 47);
    ctx.restore();
  };

  // Coalesced frame-accurate seek when paused
  const seekToFrame = (targetIdx: number) => {
    const vid = videoRef.current;
    if (!vid || vid.readyState < 1) {
      paintComposite(targetIdx);
      return;
    }
    const targetSec = Math.min(Math.max(0, targetIdx) / proxyFps + 0.0005, Math.max(vid.duration - 0.001, 0));
    if (seekingRef.current) {
      pendingSeekRef.current = targetIdx;
      return;
    }
    if (Math.abs(vid.currentTime - targetSec) < 0.002) {
      paintComposite(targetIdx);
      return;
    }
    seekingRef.current = true;
    vid.currentTime = targetSec;
  };

  // Reset seek queue cleanly when switching delivery videos
  useEffect(() => {
    seekingRef.current = false;
    pendingSeekRef.current = null;
    setVideoReady(false);
  }, [videoUrl]);

  // Subscribe transiently to currentFrame changes without tearing
  useEffect(() => {
    const unsub = useStudioStore.subscribe(
      (s) => s.currentFrame,
      (frame) => {
        if (!useStudioStore.getState().isPlaying) {
          seekToFrame(frame);
        }
      }
    );
    return unsub;
  }, [telemetry, videoReady]);

  // Repaint (and sync video time if ready) on overlay toggle / draft landmark / preview wireframe / videoReady changes
  useEffect(() => {
    const cur = useStudioStore.getState().currentFrame;
    if (videoReady && !useStudioStore.getState().isPlaying) {
      seekToFrame(cur);
    } else {
      paintComposite(cur);
    }
  }, [showSkeleton, showWireframe, calibrationMode, draftLandmarks, previewWireframe, videoReady, factSheet]);

  // Playback loop using requestAnimationFrame synced to hidden video clock
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    vid.playbackRate = playbackRate;
    if (!isPlaying) {
      vid.pause();
      seekToFrame(useStudioStore.getState().currentFrame);
      return;
    }
    let rafId = 0;
    vid.play().catch(() => setIsPlaying(false));
    const tick = () => {
      if (!vid.paused && !vid.ended) {
        const idx = Math.min(nFrames - 1, Math.max(0, Math.floor(vid.currentTime * proxyFps + 0.05)));
        if (idx !== useStudioStore.getState().currentFrame) {
          setCurrentFrame(idx);
        }
        paintComposite(idx);
        rafId = requestAnimationFrame(tick);
      } else {
        setIsPlaying(false);
      }
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying, playbackRate, nFrames, proxyFps]);

  const handleVideoSeeked = () => {
    seekingRef.current = false;
    const vid = videoRef.current;
    const target = pendingSeekRef.current;
    if (target !== null) {
      pendingSeekRef.current = null;
      seekToFrame(target);
    } else if (vid) {
      paintComposite(useStudioStore.getState().currentFrame);
    }
  };

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!calibrationMode || !canvasRef.current) return;
    const pt = clientToCanvas(e, canvasRef.current);
    if (!pt) return;
    const vr = getVideoRect(canvasRef.current.width, canvasRef.current.height);
    const vx = Math.max(0, Math.min(telemetry.width, (pt[0] - vr.ox) / vr.scale));
    const vy = Math.max(0, Math.min(telemetry.height, (pt[1] - vr.oy) / vr.scale));
    onLandmarkClick(activeLandmarkKey, [Math.round(vx * 10) / 10, Math.round(vy * 10) / 10]);
  };

  const handleCanvasMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!calibrationMode || !canvasRef.current) {
      hoverPxRef.current = null;
      return;
    }
    hoverPxRef.current = clientToCanvas(e, canvasRef.current);
    paintComposite(useStudioStore.getState().currentFrame);
  };

  const ev = factSheet.events;
  const pct = (idx: number) => `${(Math.min(Math.max(idx, 0), nFrames - 1) / Math.max(nFrames - 1, 1)) * 100}%`;

  return (
    <div className="flex flex-col bg-[#131926] border border-[#263248] rounded-xl overflow-hidden">
      {/* Hidden offscreen video source (All-Intra -g 1 proxy) */}
      <video
        ref={videoRef}
        src={videoUrl}
        muted
        playsInline
        preload="auto"
        loop
        className="hidden"
        onLoadedData={() => {
          setVideoReady(true);
          seekToFrame(useStudioStore.getState().currentFrame);
        }}
        onSeeked={handleVideoSeeked}
      />

      {/* Top viewport toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-1.5 px-3 py-2 border-b border-[#263248] bg-[#161e2e] text-xs">
        <div className="flex items-center gap-2 shrink-0">
          <span className="font-semibold text-slate-200 whitespace-nowrap">Zero-Tear Scrubber</span>
          {calibrationMode && (
            <span className="px-2 py-0.5 rounded bg-amber-500/20 text-amber-300 border border-amber-500/40 font-mono text-[10px]">
              4-Click Cal ({activeLandmarkKey})
            </span>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1">
          {/* 3-Tier FPS Resolver Pill */}
          <span className="text-slate-400 text-[11px]">FPS:</span>
          {[25, 30, 60, 120, 240].map((f) => {
            const active = Math.abs(factSheet.audit.fps - f) < 2.0;
            return (
              <button
                key={f}
                onClick={() => onFpsChange(f)}
                className={`px-1.5 py-0.5 rounded font-mono text-[10px] transition ${
                  active ? "bg-sky-500 text-slate-950 font-bold" : "bg-[#1e293b] text-slate-300 hover:bg-slate-700"
                }`}
                title="Override capture FPS (<35 ms CPU recompute)"
              >
                {f}
              </button>
            );
          })}
          <button
            onClick={() => setShowSkeleton(!showSkeleton)}
            className={`px-1.5 py-0.5 rounded flex items-center gap-1 text-[11px] ${showSkeleton ? "bg-sky-500/20 text-sky-300" : "bg-[#1e293b] text-slate-400"}`}
          >
            {showSkeleton ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />} Pose
          </button>
          <button
            onClick={() => setShowWireframe(!showWireframe)}
            className={`px-1.5 py-0.5 rounded flex items-center gap-1 text-[11px] ${showWireframe ? "bg-emerald-500/20 text-emerald-300" : "bg-[#1e293b] text-slate-400"}`}
          >
            <Grid className="w-3 h-3" /> Pitch
          </button>
        </div>
      </div>

      {/* Composite Canvas */}
      <div className="relative bg-[#0b0f17] flex items-center justify-center aspect-video max-h-[360px]">
        <canvas
          ref={canvasRef}
          width={960}
          height={540}
          onClick={handleCanvasClick}
          onMouseMove={handleCanvasMove}
          onMouseLeave={() => {
            hoverPxRef.current = null;
            if (calibrationMode) paintComposite(useStudioStore.getState().currentFrame);
          }}
          className={`w-full h-full object-contain ${calibrationMode ? "cursor-crosshair" : "cursor-default"}`}
        />
      </div>

      {/* Causal Event Timeline & Playback Controls */}
      <div className="px-3 py-2.5 bg-[#111723] border-t border-[#263248] space-y-2">
        {/* Scrubber Track with BFC / FFC / UAH / BR Pins */}
        <div className="relative pt-4 pb-1">
          {ev.is_complete_delivery && (
            <div className="absolute inset-x-0 top-0 h-4 pointer-events-none text-[10px] font-mono">
              <span style={{ left: pct(ev.bfc.frame_idx) }} className="absolute -translate-x-1/2 text-amber-400 font-bold">BFC</span>
              <span style={{ left: pct(ev.ffc.frame_idx) }} className="absolute -translate-x-1/2 text-sky-400 font-bold">FFC</span>
              <span style={{ left: pct(ev.br.frame_idx) }} className="absolute -translate-x-1/2 text-emerald-400 font-bold">BR</span>
            </div>
          )}
          <input
            type="range"
            min={0}
            max={Math.max(nFrames - 1, 1)}
            value={currentFrame}
            onChange={(e) => {
              setIsPlaying(false);
              setCurrentFrame(Number(e.target.value));
            }}
            className="w-full accent-sky-400 cursor-pointer h-1.5 bg-slate-700 rounded-lg"
          />
        </div>

        {/* Transport & Quick Event Jump / Set Buttons */}
        <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-1.5">
            <button
              onClick={() => { setIsPlaying(false); setCurrentFrame(Math.max(0, currentFrame - 1)); }}
              className="p-1.5 rounded bg-[#1a2233] hover:bg-slate-700 text-slate-200"
              title="Previous Frame (-1)"
            >
              <SkipBack className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-3 py-1.5 rounded bg-sky-500 hover:bg-sky-400 text-slate-950 font-semibold flex items-center gap-1"
            >
              {isPlaying ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              onClick={() => { setIsPlaying(false); setCurrentFrame(Math.min(nFrames - 1, currentFrame + 1)); }}
              className="p-1.5 rounded bg-[#1a2233] hover:bg-slate-700 text-slate-200"
              title="Next Frame (+1)"
            >
              <SkipForward className="w-3.5 h-3.5" />
            </button>
            <div className="flex items-center gap-1 ml-1">
              {[0.1, 0.25, 0.5, 1.0].map((r) => (
                <button
                  key={r}
                  onClick={() => setPlaybackRate(r)}
                  className={`px-1.5 py-0.5 rounded font-mono text-[11px] ${
                    playbackRate === r ? "bg-slate-200 text-slate-950 font-bold" : "bg-[#1a2233] text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {r}x
                </button>
              ))}
            </div>
          </div>

          {/* Phase Jump + Human Causal Override Buttons */}
          <div className="flex items-center gap-1.5">
            {([
              ["BFC", "bfc_frame", ev.bfc.frame_idx, "text-amber-300 border-amber-500/40"],
              ["FFC", "ffc_frame", ev.ffc.frame_idx, "text-sky-300 border-sky-500/40"],
              ["BR", "br_frame", ev.br.frame_idx, "text-emerald-300 border-emerald-500/40"],
            ] as const).map(([lbl, field, fIdx, cls]) => (
              <div key={lbl} className={`flex items-center rounded border ${cls} bg-[#161e2e] overflow-hidden`}>
                <button
                  onClick={() => { setIsPlaying(false); setCurrentFrame(fIdx); }}
                  className="px-2 py-0.5 hover:bg-white/5 font-mono text-[11px]"
                  title={`Jump to ${lbl} (frame ${fIdx})`}
                >
                  {lbl}:{fIdx}
                </button>
                <button
                  onClick={() => onNudgeEvent(field, currentFrame)}
                  className="px-1.5 py-0.5 border-l border-white/10 hover:bg-white/10 text-[10px]"
                  title={`Set ${lbl} to current frame ${currentFrame} (causal clamped)`}
                >
                  <Crosshair className="w-3 h-3" />
                </button>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};
