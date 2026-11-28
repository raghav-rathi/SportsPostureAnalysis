import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import { api } from "../../api";
import type { DeliveryDetailResponse } from "../../types";
import { J, SEG, SEG_COLOR, SMPL24_BONES, boneSegment, jointSegment } from "../../theme";
import type { Playhead } from "./shared";

function clamp(v: number, lo: number, hi: number) {
  return Math.min(Math.max(v, lo), hi);
}

interface EventMark {
  key: "bfc" | "ffc" | "br";
  label: string;
  frame: number;
}

/**
 * Command Center hero video: real clip + segment-coloured skeleton (white halo), wrist-whip trail (crimson),
 * head path (ink, dashed), event chips and measured-value chips that pop at their exact moment.
 * Drives a shared `playhead` that the 3D mannequin reads every animation frame.
 */
export const HeroVideo: React.FC<{
  detail: DeliveryDetailResponse;
  playhead: React.MutableRefObject<Playhead>;
  onFrameChange?: (frame: number) => void;
  className?: string;
}> = ({ detail, playhead, onFrameChange, className = "" }) => {
  const t = detail.telemetry;
  const story = detail.story ?? null;
  const arm = detail.fact_sheet.bowling_arm;
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const posterImgRef = useRef<HTMLImageElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const boxRef = useRef<HTMLDivElement | null>(null);
  const lastDecodedFrameRef = useRef<number>(0);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [ready, setReady] = useState(false);

  const nFrames = Math.max(t.num_frames, 1);
  const proxyFps = Math.max(t.proxy_fps || 30, 1);
  const analysisFps = Math.max(t.analysis_fps || proxyFps, 1);
  const rate = analysisFps / proxyFps >= 2 ? 0.75 : 0.3;
  const vw = Math.max(t.width || 1280, 1);
  const vh = Math.max(t.height || 720, 1);

  const events = useMemo<EventMark[]>(() => {
    const e = detail.fact_sheet.events;
    const out: EventMark[] = [];
    if (e?.bfc) out.push({ key: "bfc", label: "Back foot lands", frame: e.bfc.frame_idx });
    if (e?.ffc) out.push({ key: "ffc", label: "Front foot plants", frame: e.ffc.frame_idx });
    if (e?.br) out.push({ key: "br", label: "Ball release", frame: e.br.frame_idx });
    return out.filter((m) => m.frame >= 0 && m.frame < nFrames);
  }, [detail.fact_sheet.events, nFrames]);

  const loopRange = useMemo<[number, number]>(() => {
    if (events.length === 0) return [0, nFrames - 1];
    const fs = events.map((m) => m.frame);
    const lo = Math.min(...fs);
    const hi = Math.max(...fs);
    const margin = Math.max(6, Math.round((hi - lo) * 0.9));
    return [clamp(lo - margin, 0, nFrames - 1), clamp(hi + margin, 0, nFrames - 1)];
  }, [events, nFrames]);

  /** Measured checkpoint chips, shown only within ±2 frames of their measured moment. */
  const chips = useMemo(
    () =>
      (story?.checkpoints ?? [])
        .filter((c) => c.status !== "not_visible" && c.value_text && c.frame_idx !== null)
        .map((c) => ({ title: c.title, value: c.value_text as string, range: c.range_text, frame: c.frame_idx as number })),
    [story]
  );

  const bowlWri = arm === "right" ? J.rwri : J.lwri;

  const draw = useCallback(
    (fIn: number) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const dpr = window.devicePixelRatio || 1;
      const cw = canvas.width;
      const ch = canvas.height;
      const scale = Math.min(cw / vw, ch / vh); // contain — never crop feet, creases, or high release hand
      const dw = vw * scale;
      const dh = vh * scale;
      const ox = (cw - dw) / 2;
      const oy = (ch - dh) / 2;
      ctx.fillStyle = "#f4f4f5";
      ctx.fillRect(0, 0, cw, ch);
      const vid = videoRef.current;
      if (vid && vid.readyState >= 2) {
        ctx.drawImage(vid, ox, oy, dw, dh);
      } else if (posterImgRef.current && posterImgRef.current.complete && posterImgRef.current.naturalWidth > 0) {
        ctx.drawImage(posterImgRef.current, ox, oy, dw, dh);
      }

      const f = clamp(Math.round(fIn), 0, nFrames - 1);
      const pts = t.joints2d_px?.[f];
      const inFr = t.joint_in_frame?.[f];
      if (!pts || pts.length < 24) return;
      const P = (p: number[]): [number, number] => [ox + p[0] * scale, oy + p[1] * scale];
      const vis = (i: number) => !inFr || inFr[i];
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";

      // Trails (last ~0.45 s of action): wrist whip crimson, head path ink dashed.
      const tail = Math.max(4, Math.round(analysisFps * 0.45));
      const trail = (ji: number, color: string, width: number, dashed: boolean) => {
        const pts2: Array<[number, number]> = [];
        for (let k = Math.max(0, f - tail); k <= f; k++) {
          const p = t.joints2d_px?.[k]?.[ji];
          const okv = !t.joint_in_frame?.[k] || t.joint_in_frame[k][ji];
          if (p && okv) pts2.push(P(p));
        }
        if (pts2.length < 2) return;
        ctx.setLineDash(dashed ? [4 * dpr, 4 * dpr] : []);
        for (let s = 1; s < pts2.length; s++) {
          const a = s / pts2.length;
          ctx.strokeStyle = color;
          ctx.globalAlpha = 0.15 + 0.85 * a;
          ctx.lineWidth = width * dpr * (0.4 + 0.6 * a);
          ctx.beginPath();
          ctx.moveTo(pts2[s - 1][0], pts2[s - 1][1]);
          ctx.lineTo(pts2[s][0], pts2[s][1]);
          ctx.stroke();
        }
        ctx.globalAlpha = 1;
        ctx.setLineDash([]);
      };
      trail(J.head, SEG.head, 2, true);
      trail(bowlWri, SEG.wrist, 4, false);

      // Skeleton: white halo then segment colour.
      for (const [a, b] of SMPL24_BONES) {
        if (!vis(a) || !vis(b)) continue;
        const [ax, ay] = P(pts[a]);
        const [bx, by] = P(pts[b]);
        ctx.strokeStyle = "rgba(255,255,255,0.92)";
        ctx.lineWidth = 6 * dpr;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
        ctx.strokeStyle = SEG_COLOR[boneSegment(a, b, arm)];
        ctx.lineWidth = 3.2 * dpr;
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
        ctx.stroke();
      }
      for (let i = 0; i < 24; i++) {
        if (!vis(i) || i === J.lhan || i === J.rhan || i === J.ltoe || i === J.rtoe) continue;
        const [x, y] = P(pts[i]);
        ctx.fillStyle = "#ffffff";
        ctx.strokeStyle = SEG_COLOR[jointSegment(i, arm)];
        ctx.lineWidth = 1.8 * dpr;
        ctx.beginPath();
        ctx.arc(x, y, (i === bowlWri ? 4.4 : 3) * dpr, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
      }
      ctx.restore();
    },
    [t, vw, vh, nFrames, analysisFps, arm, bowlWri]
  );

  // Resize canvas to its box (DPR aware).
  useEffect(() => {
    const box = boxRef.current;
    const canvas = canvasRef.current;
    if (!box || !canvas) return;
    const ro = new ResizeObserver(() => {
      const dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(box.clientWidth * dpr);
      canvas.height = Math.round(box.clientHeight * dpr);
      draw(playhead.current.frame);
    });
    ro.observe(box);
    return () => ro.disconnect();
  }, [draw, playhead]);

  // Preload instant still-frame poster at BR / FFC so canvas is crisp before <video> buffers.
  useEffect(() => {
    const posterFrame = detail.fact_sheet.events?.br?.frame_idx ?? loopRange[0];
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      posterImgRef.current = img;
      if (!videoRef.current || videoRef.current.readyState < 2) {
        lastDecodedFrameRef.current = posterFrame;
        playhead.current.frame = posterFrame;
        setFrame(posterFrame);
        onFrameChange?.(posterFrame);
        draw(posterFrame);
      }
    };
    img.src = api.frameUrl(detail.delivery.id, posterFrame, 960);
    return () => {
      posterImgRef.current = null;
    };
  }, [detail.delivery.id, detail.fact_sheet.events?.br?.frame_idx, loopRange, draw, playhead, onFrameChange]);

  // Playback loop over [BFC − margin, BR + margin].
  useEffect(() => {
    const vid = videoRef.current;
    if (!vid) return;
    let raf = 0;
    let lastUi = -1;
    const tick = () => {
      if (vid.readyState >= 2) {
        let f: number;
        if (vid.seeking) {
          // Hold the last decoded frame until seeked fires so video pixels & skeleton stay 100% locked across loop wraps.
          f = lastDecodedFrameRef.current;
        } else {
          f = clamp(Math.floor(vid.currentTime * proxyFps + 1e-4), 0, nFrames - 1);
          lastDecodedFrameRef.current = f;
          if (!vid.paused && (f >= loopRange[1] || vid.ended)) {
            vid.currentTime = (loopRange[0] + 0.25) / proxyFps;
            if (vid.ended) void vid.play().catch(() => undefined);
          }
        }
        playhead.current.frame = f;
        draw(f);
        if (f !== lastUi) {
          lastUi = f;
          setFrame(f);
          onFrameChange?.(f);
        }
      } else {
        draw(playhead.current.frame);
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [draw, loopRange, nFrames, proxyFps, playhead, onFrameChange]);

  useEffect(() => {
    const vid = videoRef.current;
    if (!vid || !ready) return;
    vid.playbackRate = rate;
    if (playing) void vid.play().catch(() => setPlaying(false));
    else vid.pause();
  }, [playing, ready, rate]);

  const onLoaded = () => {
    const vid = videoRef.current;
    if (!vid) return;
    lastDecodedFrameRef.current = loopRange[0];
    vid.currentTime = (loopRange[0] + 0.25) / proxyFps;
    setReady(true);
  };

  const onSeeked = () => {
    const vid = videoRef.current;
    if (!vid || vid.readyState < 2) return;
    const f = clamp(Math.floor(vid.currentTime * proxyFps + 1e-4), 0, nFrames - 1);
    lastDecodedFrameRef.current = f;
    playhead.current.frame = f;
    setFrame(f);
    onFrameChange?.(f);
    draw(f);
  };

  const seek = (f: number) => {
    const vid = videoRef.current;
    if (!vid) return;
    setPlaying(false);
    vid.currentTime = (clamp(f, 0, nFrames - 1) + 0.25) / proxyFps;
  };

  const activeEvent = events.find((e) => Math.abs(frame - e.frame) <= Math.max(2, Math.round(analysisFps * 0.06)));
  const activeChips = chips.filter((c) => Math.abs(frame - c.frame) <= 2).slice(0, 2);
  const span = Math.max(loopRange[1] - loopRange[0], 1);
  const pctNum = (f: number) => clamp(((f - loopRange[0]) / span) * 100, 0, 100);
  const pct = (f: number) => `${pctNum(f)}%`;
  const ffcEv = events.find((e) => e.key === "ffc");
  const brEv = events.find((e) => e.key === "br");
  const splitPair = Boolean(ffcEv && brEv && pctNum(brEv.frame) - pctNum(ffcEv.frame) < 14);

  return (
    <div className={`relative flex flex-col ${className}`}>
      <div ref={boxRef} className="relative flex-1 min-h-[240px] rounded-[6px] overflow-hidden bg-[#f4f4f5]">
        <video
          ref={videoRef}
          src={detail.delivery.video_proxy_url}
          muted
          playsInline
          preload="auto"
          crossOrigin="anonymous"
          onLoadedData={onLoaded}
          onSeeked={onSeeked}
          className="absolute w-px h-px opacity-0 pointer-events-none"
        />
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" />
        <div className="absolute top-3 left-3 flex flex-col items-start gap-1.5">
          <span className="v-chip !bg-white/95">
            <span className="w-1.5 h-1.5 rounded-full bg-[#dc2626] animate-pulse" />
            {(detail.story?.sample_kind ?? "real") === "real" ? "Real video" : "Studio demo"} ·{" "}
            <span className="v-num">{Math.round(analysisFps)} fps</span>
          </span>
          {activeEvent && <span className="v-chip v-chip-ink">{activeEvent.label}</span>}
        </div>
        <div className="absolute bottom-3 left-3 right-3 flex flex-wrap gap-1.5 justify-start">
          {activeChips.map((c) => (
            <span key={c.title} className="v-chip !h-auto !py-1 !bg-white/95 flex-col !items-start !gap-0 !rounded-[6px]">
              <span className="v-label !text-[9.5px]">{c.title}</span>
              <span className="text-[15px] font-[500] v-num leading-tight">
                {c.value}
                {c.range && <span className="text-[11px] text-[#8f8f8f] ml-1.5 font-[400]">{c.range}</span>}
              </span>
            </span>
          ))}
        </div>
      </div>
      {/* Phase timeline — symmetrical single-baseline event markers */}
      <div className="mt-2.5 flex items-start gap-2.5">
        <button
          onClick={() => setPlaying((p) => !p)}
          className="w-7 h-7 rounded-full v-hairline bg-white flex items-center justify-center shrink-0"
          aria-label={playing ? "Pause" : "Play"}
        >
          {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5 ml-[1px]" />}
        </button>
        <div className="relative flex-1 h-9">
          <div className="absolute top-[12px] inset-x-0 h-[3px] rounded-full bg-[#ebebeb]" />
          <div className="absolute top-[12px] left-0 h-[3px] rounded-full bg-[#171717]" style={{ width: pct(frame) }} />
          {events.map((e) => {
            const labelAlign =
              splitPair && e.key === "ffc"
                ? "-translate-x-[calc(100%+3px)] text-right"
                : splitPair && e.key === "br"
                ? "translate-x-[3px] text-left"
                : "-translate-x-1/2 text-center";
            return (
              <button
                key={e.key}
                onClick={() => seek(e.frame)}
                className="absolute top-0 left-0 group focus:outline-none"
                style={{ left: pct(e.frame) }}
                title={`Jump to ${e.label.toLowerCase()}`}
              >
                <span
                  className={`block -translate-x-1/2 w-2.5 h-2.5 mt-[8.5px] rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.16)] ${
                    e.key === "br" ? "bg-[#dc2626]" : "bg-[#171717]"
                  }`}
                />
                <span
                  className={`block mt-1 v-label !text-[9.5px] whitespace-nowrap leading-none ${labelAlign} ${
                    e.key === "br" ? "!text-[#dc2626]" : ""
                  }`}
                >
                  {e.key === "bfc" ? "Back foot" : e.key === "ffc" ? "Front foot" : "Release"}
                </span>
              </button>
            );
          })}
        </div>
        <span className="v-label w-[84px] text-right pt-[7px] shrink-0">
          {events.find((e) => e.key === "ffc") && t.time_rel_ffc_ms?.[frame] !== undefined
            ? `${t.time_rel_ffc_ms[frame] >= 0 ? "+" : ""}${Math.round(t.time_rel_ffc_ms[frame])} ms`
            : `frame ${frame}`}
        </span>
      </div>
    </div>
  );
};
