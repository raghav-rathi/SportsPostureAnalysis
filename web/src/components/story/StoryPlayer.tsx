import React, { forwardRef, useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { Eye, EyeOff, Pause, Play, Repeat } from "lucide-react";
import type { BodyCheckpoint, KeyMoment, MomentId, TelemetryV2 } from "../../types";

/** SMPL-24 bone list (index pairs), shared with the v3 lab player. */
const BONES: Array<[number, number]> = [
  [0, 3], [3, 6], [6, 9], [9, 12], [12, 15],
  [0, 1], [1, 4], [4, 7], [7, 10],
  [0, 2], [2, 5], [5, 8], [8, 11],
  [9, 13], [13, 16], [16, 18], [18, 20], [20, 22],
  [9, 14], [14, 17], [17, 19], [19, 21], [21, 23],
];

export interface StoryPlayerHandle {
  seekToFrame: (frame: number) => void;
  play: () => void;
  pause: () => void;
}

export interface StoryCaption {
  title: string;
  body: string;
}

interface StoryPlayerProps {
  videoUrl: string;
  telemetry: TelemetryV2;
  keyMoments: KeyMoment[];
  activeCheckpoint: BodyCheckpoint | null;
  activeMomentId: MomentId | null;
  caption: StoryCaption | null;
  /** Auto-play the action once on first load and stop on this frame (e.g. ball release). */
  introStopFrame?: number | null;
  onMomentClick?: (m: KeyMoment) => void;
  /** Called when playback pauses or a seek lands (not on every animation frame). */
  onSettledFrame?: (frame: number) => void;
}

function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

export const StoryPlayer = forwardRef<StoryPlayerHandle, StoryPlayerProps>(function StoryPlayer(
  { videoUrl, telemetry, keyMoments, activeCheckpoint, activeMomentId, caption, introStopFrame, onMomentClick, onSettledFrame },
  ref
) {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  const frameRef = useRef(0);
  const pendingSeekRef = useRef<number | null>(null);
  const introRef = useRef<boolean>(false);

  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [ready, setReady] = useState(false);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [loop, setLoop] = useState(true);

  const nFrames = Math.max(telemetry.num_frames, 1);
  const proxyFps = Math.max(telemetry.proxy_fps || 30, 1);
  const analysisFps = Math.max(telemetry.analysis_fps || proxyFps, 1);
  // If the proxy already plays slower than real life (slow-mo capture), default to 1×; else 0.5×.
  const defaultRate = analysisFps / proxyFps >= 2 ? 1 : 0.5;
  const [rate, setRate] = useState(defaultRate);

  const vw = Math.max(telemetry.width || 1280, 1);
  const vh = Math.max(telemetry.height || 720, 1);
  const boxAspect = vw >= vh ? vw / vh : 4 / 5;

  const jointIndex = useMemo(() => {
    const m = new Map<string, number>();
    (telemetry.joint_names || []).forEach((n, i) => m.set(n, i));
    return m;
  }, [telemetry.joint_names]);

  const loopRange = useMemo(() => {
    if (keyMoments.length === 0) return [0, nFrames - 1] as const;
    const fs = keyMoments.map((m) => m.frame_idx);
    const first = Math.min(...fs);
    const last = Math.max(...fs);
    const margin = Math.max(4, Math.round((last - first) * 0.45));
    return [clamp(first - margin, 0, nFrames - 1), clamp(last + margin, 0, nFrames - 1)] as const;
  }, [keyMoments, nFrames]);

  // Latest-props ref so the imperative draw loop never reads stale values.
  const live = useRef({ activeCheckpoint, showSkeleton, loop, loopRange, caption });
  live.current = { activeCheckpoint, showSkeleton, loop, loopRange, caption };

  const timeLabel = useCallback(
    (f: number) => {
      const pts = telemetry.frame_pts_ms;
      if (pts && pts.length > f && pts.length > 0) return ((pts[f] - pts[0]) / 1000).toFixed(2);
      return (f / analysisFps).toFixed(2);
    },
    [telemetry.frame_pts_ms, analysisFps]
  );

  const draw = useCallback(
    (fIn: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = canvas.width;
      const ch = canvas.height;
      const scale = Math.min(cw / vw, ch / vh);
      const dw = vw * scale;
      const dh = vh * scale;
      const ox = (cw - dw) / 2;
      const oy = (ch - dh) / 2;

      ctx.fillStyle = "#f4f4f5";
      ctx.fillRect(0, 0, cw, ch);
      const vid = videoRef.current;
      if (vid && vid.readyState >= 2) ctx.drawImage(vid, ox, oy, dw, dh);

      const f = clamp(Math.round(fIn), 0, nFrames - 1);
      const pts = telemetry.joints2d_px?.[f];
      const inFr = telemetry.joint_in_frame?.[f];
      const { activeCheckpoint: cp, showSkeleton: sk } = live.current;
      if (!pts || pts.length < 24) return;
      const P = (i: number): [number, number] => [ox + pts[i][0] * scale, oy + pts[i][1] * scale];
      const visible = (i: number) => !inFr || inFr[i];

      const hi = new Set<number>();
      (cp?.highlight_joints || []).forEach((n) => {
        const i = jointIndex.get(n);
        if (i !== undefined) hi.add(i);
      });

      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      if (sk) {
        for (const [a, b] of BONES) {
          if (!visible(a) || !visible(b)) continue;
          const isHi = hi.has(a) && hi.has(b);
          if (isHi) continue;
          const [ax, ay] = P(a);
          const [bx, by] = P(b);
          ctx.strokeStyle = "rgba(255,255,255,0.85)";
          ctx.lineWidth = 3.6 * dpr;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
          ctx.strokeStyle = "rgba(23,23,23,0.7)";
          ctx.lineWidth = 1.5 * dpr;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(bx, by);
          ctx.stroke();
        }
      }
      // Highlighted checkpoint segments (drawn even when skeleton is hidden).
      for (const [a, b] of BONES) {
        if (!(hi.has(a) && hi.has(b)) || !visible(a) || !visible(b)) continue;
        const [ax, ay] = P(a);
        const [bx, by] = P(b);
        ctx.strokeStyle = "rgba(255,255,255,0.95)";
        ctx.lineWidth = 6.5 * dpr;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.strokeStyle = "#297a3a";
        ctx.lineWidth = 3.2 * dpr;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      hi.forEach((i) => {
        if (!visible(i)) return;
        const [x, y] = P(i);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = "#297a3a";
        ctx.lineWidth = 2 * dpr;
        ctx.beginPath();
        ctx.arc(x, y, 3.6 * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      });

      // Angle arc + measured value — only on the exact measured moment (±1 frame), never extrapolated.
      if (cp && cp.angle_joints && cp.value_text && cp.frame_idx !== null && Math.abs(f - cp.frame_idx) <= 1) {
        const [na, nv, nb] = cp.angle_joints;
        const ia = jointIndex.get(na);
        const iv = jointIndex.get(nv);
        const ib = jointIndex.get(nb);
        if (ia !== undefined && iv !== undefined && ib !== undefined && visible(ia) && visible(iv) && visible(ib)) {
          const [ax, ay] = P(ia);
          const [vx, vy] = P(iv);
          const [bx, by] = P(ib);
          const a1 = Math.atan2(ay - vy, ax - vx);
          const a2 = Math.atan2(by - vy, bx - vx);
          let d = a2 - a1;
          while (d > Math.PI) d -= Math.PI * 2;
          while (d < -Math.PI) d += Math.PI * 2;
          const r = 22 * dpr;
          ctx.strokeStyle = "#297a3a";
          ctx.lineWidth = 2 * dpr;
          ctx.setLineDash([4 * dpr, 3 * dpr]);
          ctx.beginPath();
          ctx.arc(vx, vy, r, a1, a1 + d, d < 0);
          ctx.stroke();
          ctx.setLineDash([]);
          // Label pill on the outside bisector.
          const mid = a1 + d / 2;
          const lx = vx - Math.cos(mid) * (r + 26 * dpr);
          const ly = vy - Math.sin(mid) * (r + 14 * dpr);
          const label = cp.value_text;
          ctx.font = `500 ${12 * dpr}px "Geist Mono", ui-monospace, monospace`;
          const tw = ctx.measureText(label).width;
          const padX = 7 * dpr;
          const bh = 20 * dpr;
          const bx0 = clamp(lx - tw / 2 - padX, 2, cw - tw - padX * 2 - 2);
          const by0 = clamp(ly - bh / 2, 2, ch - bh - 2);
          ctx.fillStyle = "#171717";
          ctx.beginPath();
          const rr = 5 * dpr;
          const bw = tw + padX * 2;
          ctx.moveTo(bx0 + rr, by0);
          ctx.arcTo(bx0 + bw, by0, bx0 + bw, by0 + bh, rr);
          ctx.arcTo(bx0 + bw, by0 + bh, bx0, by0 + bh, rr);
          ctx.arcTo(bx0, by0 + bh, bx0, by0, rr);
          ctx.arcTo(bx0, by0, bx0 + bw, by0, rr);
          ctx.fill();
          ctx.fillStyle = "#ffffff";
          ctx.textBaseline = "middle";
          ctx.fillText(label, bx0 + padX, by0 + bh / 2 + 0.5);
        }
      }
      ctx.restore();
    },
    [jointIndex, nFrames, telemetry.joints2d_px, telemetry.joint_in_frame, vw, vh]
  );

  // Resize canvas to its box (crisp on retina).
  useEffect(() => {
    const box = boxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.max(1, Math.round(box.clientWidth * dpr));
      canvas.height = Math.max(1, Math.round(box.clientHeight * dpr));
      draw(frameRef.current);
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, [draw]);

  // Redraw on overlay changes while paused.
  useEffect(() => {
    if (!playing) draw(frameRef.current);
  }, [activeCheckpoint, showSkeleton, draw, playing]);

  const seekToFrame = useCallback(
    (fIn: number) => {
      const f = clamp(Math.round(fIn), 0, nFrames - 1);
      frameRef.current = f;
      setFrame(f);
      const vid = videoRef.current;
      if (!vid || vid.readyState < 1) {
        pendingSeekRef.current = f;
        draw(f);
        return;
      }
      vid.currentTime = f / proxyFps + 0.001;
    },
    [nFrames, proxyFps, draw]
  );

  const stopLoop = () => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  };

  const tick = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const { loop: lp, loopRange: lr } = live.current;
    let f = clamp(Math.round(vid.currentTime * proxyFps), 0, nFrames - 1);
    if (introRef.current && introStopFrame != null && f >= introStopFrame) {
      introRef.current = false;
      vid.pause();
      seekToFrame(introStopFrame);
      return;
    }
    if (lp && !introRef.current && f >= lr[1]) {
      vid.currentTime = lr[0] / proxyFps + 0.001;
      f = lr[0];
    }
    if (f !== frameRef.current) {
      frameRef.current = f;
      setFrame(f);
    }
    draw(f);
    rafRef.current = requestAnimationFrame(tick);
  }, [draw, nFrames, proxyFps, introStopFrame, seekToFrame]);

  const play = useCallback(() => {
    const vid = videoRef.current;
    if (!vid) return;
    const { loop: lp, loopRange: lr } = live.current;
    if (lp && (frameRef.current >= lr[1] || frameRef.current < lr[0])) {
      vid.currentTime = lr[0] / proxyFps + 0.001;
    } else if (frameRef.current >= nFrames - 1) {
      vid.currentTime = 0;
    }
    vid.playbackRate = rate;
    void vid.play().catch(() => setPlaying(false));
  }, [nFrames, proxyFps, rate]);

  const pause = useCallback(() => {
    introRef.current = false;
    videoRef.current?.pause();
  }, []);

  useImperativeHandle(ref, () => ({ seekToFrame: (f) => { pause(); seekToFrame(f); }, play, pause }), [seekToFrame, play, pause]);

  useEffect(() => {
    if (videoRef.current) videoRef.current.playbackRate = rate;
  }, [rate]);

  // Reset when the clip changes.
  useEffect(() => {
    setReady(false);
    setPlaying(false);
    stopLoop();
    frameRef.current = introStopFrame ?? 0;
    setFrame(frameRef.current);
    introRef.current = false;
    return stopLoop;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [videoUrl]);

  const onLoaded = () => {
    setReady(true);
    const vid = videoRef.current;
    if (!vid) return;
    vid.playbackRate = rate;
    const reduceMotion = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
    if (introStopFrame != null && !reduceMotion) {
      introRef.current = true;
      vid.currentTime = loopRange[0] / proxyFps + 0.001;
      void vid.play().catch(() => {
        introRef.current = false;
        seekToFrame(introStopFrame);
      });
    } else {
      const target = pendingSeekRef.current ?? introStopFrame ?? frameRef.current;
      pendingSeekRef.current = null;
      seekToFrame(target);
    }
  };

  const onSeeked = () => {
    const vid = videoRef.current;
    if (!vid) return;
    const f = clamp(Math.round(vid.currentTime * proxyFps), 0, nFrames - 1);
    if (pendingSeekRef.current !== null) {
      const p = pendingSeekRef.current;
      pendingSeekRef.current = null;
      seekToFrame(p);
      return;
    }
    frameRef.current = f;
    setFrame(f);
    draw(f);
    if (vid.paused) onSettledFrame?.(f);
  };

  const momentAtFrame = keyMoments.find((m) => Math.abs(m.frame_idx - frame) <= 0);
  const showCaption = !playing && caption;
  const captionBody = (() => {
    if (!caption) return "";
    const b = caption.body.trim();
    const t = caption.title.trim().toLowerCase();
    if (b.toLowerCase().startsWith(t)) {
      return b.slice(t.length).replace(/^[\s—–:\-]+/, "");
    }
    return b;
  })();
  const pct = (f: number) => (nFrames <= 1 ? 0 : (f / (nFrames - 1)) * 100);

  return (
    <div className="v-card overflow-hidden">
      <div
        ref={boxRef}
        className="relative w-full bg-[#f4f4f5] select-none cursor-pointer"
        style={{ aspectRatio: `${boxAspect}`, maxHeight: "min(68vh, 680px)" }}
        onClick={() => (playing ? pause() : play())}
        role="button"
        aria-label={playing ? "Pause" : "Play"}
      >
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        <video
          ref={videoRef}
          src={videoUrl}
          muted
          playsInline
          preload="auto"
          crossOrigin="anonymous"
          className="hidden"
          onLoadedData={onLoaded}
          onSeeked={onSeeked}
          onPlay={() => {
            setPlaying(true);
            stopLoop();
            rafRef.current = requestAnimationFrame(tick);
          }}
          onPause={() => {
            setPlaying(false);
            stopLoop();
            onSettledFrame?.(frameRef.current);
          }}
        />
        {!ready && (
          <div className="absolute inset-0 flex items-center justify-center text-[13px] text-[#666666]">Loading video…</div>
        )}
        {!playing && ready && !showCaption && (
          <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
            <span className="w-12 h-12 rounded-full bg-[#ffffff]/90 border border-[#ebebeb] flex items-center justify-center">
              <Play className="w-5 h-5 text-[#171717] ml-0.5" />
            </span>
          </div>
        )}
        {momentAtFrame && !playing && (
          <div className="absolute top-3 left-3 pointer-events-none">
            <span className="inline-flex items-center h-6 px-2.5 rounded-full bg-[#171717] text-[#ffffff] text-[12px] font-[500]">
              {momentAtFrame.label}
            </span>
          </div>
        )}
        {showCaption && (
          <div className="absolute left-3 right-3 bottom-3 pointer-events-none flex">
            <div className="max-w-[560px] rounded-[8px] bg-[#ffffff]/95 border border-[#ebebeb] px-3 py-2">
              <div className="text-[13px] font-[500] text-[#171717]">{caption!.title}</div>
              <div className="text-[12.5px] text-[#4d4d4d] leading-snug mt-0.5">{captionBody.charAt(0).toUpperCase() + captionBody.slice(1)}</div>
            </div>
          </div>
        )}
      </div>

      {/* Controls */}
      <div className="px-3 pt-3 pb-2.5 border-t border-[#ebebeb] space-y-2">
        <div className="relative h-7 mt-2">
          {/* Moment ticks */}
          {keyMoments.map((m, i) => (
            <button
              key={m.id}
              onClick={() => onMomentClick?.(m)}
              className="absolute -translate-x-1/2 group"
              style={{
                left: `${pct(m.frame_idx)}%`,
                // Stagger vertically when a neighbour sits within 2.5% so dots never overlap.
                top:
                  i > 0 && Math.abs(pct(m.frame_idx) - pct(keyMoments[i - 1].frame_idx)) < 2.5 && i % 2 === 1
                    ? "-9px"
                    : "0px",
              }}
              title={m.label}
              aria-label={`Jump to ${m.label}`}
            >
              <span
                className={`block w-2.5 h-2.5 rounded-full border ${
                  activeMomentId === m.id ? "bg-[#297a3a] border-[#297a3a]" : "bg-[#ffffff] border-[#171717]"
                }`}
              />
            </button>
          ))}
          <input
            type="range"
            min={0}
            max={nFrames - 1}
            step={1}
            value={frame}
            onChange={(e) => {
              pause();
              seekToFrame(Number(e.target.value));
            }}
            className="absolute left-0 right-0 bottom-0 w-full accent-[#171717] h-1.5 cursor-pointer"
            aria-label="Scrub through the delivery"
          />
        </div>
        <div className="flex flex-wrap items-center gap-2 justify-between">
          <div className="flex items-center gap-2">
            <button onClick={() => (playing ? pause() : play())} className="v-btn-primary h-8 w-8 flex items-center justify-center" aria-label={playing ? "Pause" : "Play"}>
              {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-0.5" />}
            </button>
            <span className="font-mono text-[12px] text-[#666666] v-tabular">
              {timeLabel(frame)} s <span className="text-[#a8a8a8]">/ {timeLabel(nFrames - 1)} s</span>
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <div className="flex items-center rounded-[6px] bg-[#f4f4f5] p-0.5 border border-[#ebebeb]" role="group" aria-label="Playback speed">
              {[0.25, 0.5, 1].map((r) => (
                <button
                  key={r}
                  onClick={() => setRate(r)}
                  className={`h-6 px-2 rounded-[5px] font-mono text-[11px] ${rate === r ? "bg-[#ffffff] text-[#171717]" : "text-[#666666]"}`}
                >
                  {r}×
                </button>
              ))}
            </div>
            <button
              onClick={() => setLoop((v) => !v)}
              className={`h-7 px-2 rounded-[6px] text-[12px] flex items-center gap-1 border whitespace-nowrap ${
                loop ? "border-[#171717] text-[#171717] bg-[#ffffff]" : "border-[#ebebeb] text-[#666666] bg-[#ffffff]"
              }`}
              title="Loop just the bowling action"
              aria-pressed={loop}
            >
              <Repeat className="w-3.5 h-3.5" /> Loop action
            </button>
            <button
              onClick={() => setShowSkeleton((v) => !v)}
              className={`h-7 px-2 rounded-[6px] text-[12px] flex items-center gap-1 border whitespace-nowrap ${
                showSkeleton ? "border-[#171717] text-[#171717] bg-[#ffffff]" : "border-[#ebebeb] text-[#666666] bg-[#ffffff]"
              }`}
              aria-pressed={showSkeleton}
            >
              {showSkeleton ? <Eye className="w-3.5 h-3.5" /> : <EyeOff className="w-3.5 h-3.5" />} Skeleton
            </button>
          </div>
        </div>
      </div>
    </div>
  );
});
