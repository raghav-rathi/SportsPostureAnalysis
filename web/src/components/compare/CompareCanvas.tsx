import React, { useCallback, useEffect, useRef, useState } from "react";
import type { TelemetryV2 } from "../../types";
import { SEG, SEG_COLOR, SMPL24_BONES, boneSegment } from "../../theme";
import { bodyFrameOf, proxyFps, videoRect } from "./compareMath";

/**
 * One compare stage canvas: a hidden <video> composited onto a <canvas> with a segment-coloured skeleton.
 * Optional `overlay` superimposes a second delivery (SimulCam-style blend): its video at `opacity`, registered to the
 * primary by a fixed similarity transform computed at the two sync frames (pelvis anchor + body size), plus an X/Y nudge,
 * and its skeleton drawn as a slate dashed ghost. Frame → video time = (frame + 0.5) / proxy_fps.
 */

export interface CanvasLayer {
  videoUrl: string | null;
  telemetry: TelemetryV2;
  arm: "right" | "left";
  frame: number;
}

export interface OverlayLayer extends CanvasLayer {
  syncFramePrimary: number;
  syncFrameOverlay: number;
  opacity: number; // 0..1, applies to overlay video only
  nudge: [number, number]; // canvas px
  mirror: boolean;
  showVideo: boolean;
}

export interface CompareCanvasProps {
  primary: CanvasLayer;
  tag: string;
  caption?: string;
  /** Draw the primary mirrored (used for "Mirror B" in side-by-side). */
  mirrorPrimary?: boolean;
  /** Dashed slate skeleton for the primary (B in side-by-side). */
  primaryGhostStyle?: boolean;
  overlay?: OverlayLayer | null;
  heightClass?: string;
}

type MapFn = (p: number[]) => [number, number];

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  t: TelemetryV2,
  f: number,
  arm: "right" | "left",
  map: MapFn,
  ghost: boolean
) {
  const pts = t.joints2d_px?.[f];
  const inFr = t.joint_in_frame?.[f];
  if (!pts || pts.length < 24) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const [ia, ib] of SMPL24_BONES) {
    if (inFr && (!inFr[ia] || !inFr[ib])) continue;
    const pa = pts[ia];
    const pb = pts[ib];
    if (!pa || !pb) continue;
    const [ax, ay] = map(pa);
    const [bx, by] = map(pb);
    ctx.setLineDash(ghost ? [6, 4] : []);
    ctx.strokeStyle = "rgba(255,255,255,0.9)";
    ctx.lineWidth = ghost ? 4 : 5;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.strokeStyle = ghost ? SEG.ghost : SEG_COLOR[boneSegment(ia, ib, arm)];
    ctx.lineWidth = ghost ? 2 : 2.6;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }
  ctx.setLineDash([]);
  if (!ghost) {
    for (let i = 0; i < 24; i++) {
      if (inFr && !inFr[i]) continue;
      const p = pts[i];
      if (!p) continue;
      const [x, y] = map(p);
      ctx.fillStyle = "#ffffff";
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = SEG.ink;
      ctx.beginPath();
      ctx.arc(x, y, 1.6, 0, Math.PI * 2);
      ctx.fill();
    }
  }
  ctx.restore();
}

function useSeek(ref: React.RefObject<HTMLVideoElement>, frame: number, t: TelemetryV2 | null, onReady: () => void) {
  useEffect(() => {
    const v = ref.current;
    if (!v || !t) return;
    const target = (frame + 0.5) / proxyFps(t);
    if (v.readyState >= 1 && Math.abs(v.currentTime - target) > 0.5 / proxyFps(t)) {
      try {
        v.currentTime = target;
      } catch {
        /* ignore */
      }
    } else {
      onReady();
    }
  }, [ref, frame, t, onReady]);
}

export const CompareCanvas: React.FC<CompareCanvasProps> = ({
  primary,
  tag,
  caption,
  mirrorPrimary,
  primaryGhostStyle,
  overlay,
  heightClass = "h-[320px] md:h-[440px]",
}) => {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const vA = useRef<HTMLVideoElement>(null);
  const vB = useRef<HTMLVideoElement>(null);
  const [size, setSize] = useState({ w: 640, h: 360 });
  const [videoError, setVideoError] = useState(false);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      setSize({ w: Math.max(50, Math.round(r.width)), h: Math.max(50, Math.round(r.height)) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const draw = useCallback(() => {
    const c = canvasRef.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = size.w;
    const ch = size.h;
    if (c.width !== Math.round(cw * dpr) || c.height !== Math.round(ch * dpr)) {
      c.width = Math.round(cw * dpr);
      c.height = Math.round(ch * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = "#f6f6f5";
    ctx.fillRect(0, 0, cw, ch);

    const tA = primary.telemetry;
    const rA = videoRect(tA, cw, ch);
    const mapA: MapFn = (p) => [rA.ox + p[0] * rA.scale, rA.oy + p[1] * rA.scale];

    // ── primary video + skeleton (optionally mirrored) ──
    ctx.save();
    if (mirrorPrimary) {
      ctx.translate(cw, 0);
      ctx.scale(-1, 1);
    }
    const va = vA.current;
    if (va && va.readyState >= 2) {
      ctx.drawImage(va, rA.ox, rA.oy, rA.dw, rA.dh);
    } else {
      ctx.fillStyle = "#efefee";
      ctx.fillRect(rA.ox, rA.oy, rA.dw, rA.dh);
    }

    // ── overlay (blend) ──
    let mapB: MapFn | null = null;
    if (overlay) {
      const tB = overlay.telemetry;
      const bfA = bodyFrameOf(tA, overlay.syncFramePrimary);
      const bfB = bodyFrameOf(tB, overlay.syncFrameOverlay);
      const mir = overlay.mirror ? -1 : 1;
      const [nx, ny] = overlay.nudge;
      let ax: number;
      let ay: number;
      let s: number;
      let bcx: number;
      let bcy: number;
      if (bfA && bfB) {
        ax = rA.ox + bfA.cx * rA.scale;
        ay = rA.oy + bfA.cy * rA.scale;
        s = (bfA.size * rA.scale) / bfB.size;
        bcx = bfB.cx;
        bcy = bfB.cy;
      } else {
        const rB = videoRect(tB, cw, ch);
        s = rB.scale;
        bcx = (tB.width || 1280) / 2;
        bcy = (tB.height || 720) / 2;
        ax = cw / 2;
        ay = ch / 2;
      }
      const m: MapFn = (p) => [ax + nx + mir * (p[0] - bcx) * s, ay + ny + (p[1] - bcy) * s];
      mapB = m;
      const vb = vB.current;
      if (overlay.showVideo && vb && vb.readyState >= 2 && overlay.opacity > 0.001) {
        ctx.save();
        ctx.globalAlpha = Math.min(1, Math.max(0, overlay.opacity));
        ctx.translate(ax + nx, ay + ny);
        ctx.scale(mir * s, s);
        ctx.translate(-bcx, -bcy);
        ctx.drawImage(vb, 0, 0, tB.width || vb.videoWidth, tB.height || vb.videoHeight);
        ctx.restore();
      }
    }

    drawSkeleton(ctx, tA, primary.frame, primary.arm, mapA, !!primaryGhostStyle);
    if (overlay && mapB) drawSkeleton(ctx, overlay.telemetry, overlay.frame, overlay.arm, mapB, true);
    ctx.restore();

    // ── tag (never mirrored) ──
    ctx.font = "500 12px Geist, Inter, sans-serif";
    const label = tag + (mirrorPrimary ? " · Mirrored" : "");
    const tw = ctx.measureText(label).width;
    ctx.fillStyle = "rgba(255,255,255,0.92)";
    ctx.beginPath();
    ctx.roundRect(10, 10, tw + 16, 22, 11);
    ctx.fill();
    ctx.fillStyle = SEG.ink;
    ctx.fillText(label, 18, 25);
  }, [size, primary, overlay, mirrorPrimary, primaryGhostStyle, tag]);

  useSeek(vA, primary.frame, primary.telemetry, draw);
  useSeek(vB, overlay?.frame ?? 0, overlay ? overlay.telemetry : null, draw);
  useEffect(() => {
    draw();
  }, [draw]);

  return (
    <div className="min-w-0">
      <div ref={wrapRef} className={`relative w-full ${heightClass} v-stage overflow-hidden`}>
        <canvas ref={canvasRef} className="absolute inset-0 w-full h-full" style={{ width: "100%", height: "100%" }} />
        {primary.videoUrl && (
          <video
            ref={vA}
            src={primary.videoUrl}
            muted
            playsInline
            preload="auto"
            crossOrigin="anonymous"
            className="hidden"
            onLoadedData={draw}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              v.currentTime = (primary.frame + 0.5) / proxyFps(primary.telemetry);
            }}
            onSeeked={draw}
            onError={() => setVideoError(true)}
          />
        )}
        {overlay?.videoUrl && overlay.showVideo && (
          <video
            ref={vB}
            src={overlay.videoUrl}
            muted
            playsInline
            preload="auto"
            crossOrigin="anonymous"
            className="hidden"
            onLoadedData={draw}
            onLoadedMetadata={(e) => {
              const v = e.currentTarget;
              v.currentTime = (overlay.frame + 0.5) / proxyFps(overlay.telemetry);
            }}
            onSeeked={draw}
          />
        )}
        {(videoError || !primary.videoUrl) && (
          <div className="absolute bottom-2 left-2 right-2 text-[12px] text-[#4d4d4d] bg-white/90 rounded px-2 py-1">
            Video not available for this ball — showing the measured skeleton only.
          </div>
        )}
      </div>
      {caption && <p className="mt-1.5 text-[12px] text-[#8f8f8f] truncate">{caption}</p>}
    </div>
  );
};

export default CompareCanvas;
