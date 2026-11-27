import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Pause, Play } from "lucide-react";
import type { MomentId, TelemetryV2 } from "../../types";

/**
 * Synced dual canvas player for Compare 2 Balls (Studio v4).
 *
 * - Each ball: hidden <video> composited onto a <canvas> + subtle 2D skeleton from
 *   `telemetry.joints2d_px`, respecting `joint_in_frame` (aspect-preserving letterbox,
 *   same math as `VideoCanvasPlayer`).
 * - One shared clock in "action seconds" relative to a line-up moment (default:
 *   Front foot plants via `syncFrames`). Each side maps clock → analysis index with its
 *   own `analysis_fps`, then index → video time with `frame_idx / proxy_fps`.
 * - "overlay" mode draws Ball B's skeleton as a dashed ghost over Ball A's video, aligned
 *   with a fixed similarity transform computed at the sync frames.
 * - Never shows frame numbers: only time relative to the line-up moment.
 */

export interface DualSide {
  id: string;
  /** Short tag drawn on the canvas, e.g. "Ball A". */
  tag: string;
  /** Secondary text, e.g. "27 Nov · Side-on". */
  caption?: string;
  videoUrl: string;
  telemetry: TelemetryV2;
  bowlingArm: "right" | "left";
}

export interface DualMoment {
  id: MomentId;
  label: string;
  a: number | null;
  b: number | null;
}

export type DualMode = "side" | "overlay";

interface Props {
  a: DualSide;
  b: DualSide;
  syncFrames: { a: number | null; b: number | null };
  syncLabel?: string;
  moments: DualMoment[];
  mode: DualMode;
  /** Action speed (0.5 = half speed). */
  speed?: number;
}

type Key = "a" | "b";

const BONE_PAIRS_2D: Array<[number, number, "spine" | "lleg" | "rleg" | "larm" | "rarm"]> = [
  [0, 3, "spine"], [3, 6, "spine"], [6, 9, "spine"], [9, 12, "spine"], [12, 15, "spine"],
  [0, 1, "lleg"], [1, 4, "lleg"], [4, 7, "lleg"], [7, 10, "lleg"],
  [0, 2, "rleg"], [2, 5, "rleg"], [5, 8, "rleg"], [8, 11, "rleg"],
  [9, 13, "larm"], [13, 16, "larm"], [16, 18, "larm"], [18, 20, "larm"], [20, 22, "larm"],
  [9, 14, "rarm"], [14, 17, "rarm"], [17, 19, "rarm"], [19, 21, "rarm"], [21, 23, "rarm"],
];

function videoRect(t: TelemetryV2, cw: number, ch: number) {
  const vw = Math.max(t.width || 1280, 1);
  const vh = Math.max(t.height || 720, 1);
  const scale = Math.min(cw / vw, ch / vh);
  const dw = vw * scale;
  const dh = vh * scale;
  return { ox: (cw - dw) * 0.5, oy: (ch - dh) * 0.5, dw, dh, scale };
}

const nFramesOf = (t: TelemetryV2) => Math.max(t.num_frames || t.joints2d_px?.length || 1, 1);
const proxyFpsOf = (t: TelemetryV2) => Math.max(t.proxy_fps || 30, 1);
const analysisFpsOf = (t: TelemetryV2) => Math.max(t.analysis_fps || t.proxy_fps || 30, 1);
const clampFrame = (t: TelemetryV2, f: number) => Math.min(Math.max(Math.round(f), 0), nFramesOf(t) - 1);

/** Anchor (pelvis, else centroid) + body size (torso, else in-frame bbox height) for ghost alignment. */
function bodyFrameOf(t: TelemetryV2, f: number): { cx: number; cy: number; size: number } | null {
  const pts = t.joints2d_px?.[f];
  const inFr = t.joint_in_frame?.[f];
  if (!pts || pts.length < 16) return null;
  const ok = (i: number) => !!pts[i] && (!inFr || !!inFr[i]);
  const vis = pts.map((_, i) => i).filter(ok);
  if (vis.length < 4) return null;
  let cx: number;
  let cy: number;
  if (ok(0)) {
    cx = pts[0][0];
    cy = pts[0][1];
  } else {
    cx = vis.reduce((s, i) => s + pts[i][0], 0) / vis.length;
    cy = vis.reduce((s, i) => s + pts[i][1], 0) / vis.length;
  }
  let size = 0;
  if (ok(0) && ok(12)) size = Math.hypot(pts[12][0] - pts[0][0], pts[12][1] - pts[0][1]) * 3.2;
  if (size < 1e-3) {
    const ys = vis.map((i) => pts[i][1]);
    size = Math.max(...ys) - Math.min(...ys);
  }
  if (size < 1e-3) return null;
  return { cx, cy, size };
}

function drawSkeleton(
  ctx: CanvasRenderingContext2D,
  t: TelemetryV2,
  f: number,
  map: (p: number[]) => [number, number],
  isRightArm: boolean,
  ghost: boolean
) {
  const pts = t.joints2d_px?.[f];
  const inFr = t.joint_in_frame?.[f];
  if (!pts || pts.length !== 24) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.globalAlpha = ghost ? 0.85 : 0.8;
  for (const [ia, ib, grp] of BONE_PAIRS_2D) {
    if (inFr && (!inFr[ia] || !inFr[ib])) continue;
    const pa = pts[ia];
    const pb = pts[ib];
    if (!pa || !pb) continue;
    const [ax, ay] = map(pa);
    const [bx, by] = map(pb);
    const isActive =
      (isRightArm && (grp === "rarm" || grp === "lleg")) || (!isRightArm && (grp === "larm" || grp === "rleg"));
    ctx.setLineDash(ghost ? [5, 4] : []);
    ctx.strokeStyle = "rgba(255,255,255,0.92)";
    ctx.lineWidth = ghost ? 3.4 : 3.6;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
    ctx.strokeStyle = ghost ? "#4d4d4d" : isActive ? "#297a3a" : "#171717";
    ctx.lineWidth = ghost ? 1.6 : 1.6;
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
      ctx.strokeStyle = "#171717";
      ctx.lineWidth = 1.2;
      ctx.beginPath();
      ctx.arc(x, y, 2.4, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    }
  }
  ctx.restore();
}

function formatRel(t: number, anchorLabel: string): string {
  if (Math.abs(t) < 0.005) return `At ${anchorLabel.toLowerCase()}`;
  return `${Math.abs(t).toFixed(2)} s ${t < 0 ? "before" : "after"} ${anchorLabel.toLowerCase()}`;
}

export const DualComparePlayer: React.FC<Props> = ({
  a,
  b,
  syncFrames,
  syncLabel = "Front foot plants",
  moments,
  mode,
  speed = 0.5,
}) => {
  const vidRef = { a: useRef<HTMLVideoElement | null>(null), b: useRef<HTMLVideoElement | null>(null) };
  const canRef = { a: useRef<HTMLCanvasElement | null>(null), b: useRef<HTMLCanvasElement | null>(null) };
  const wrapRef = useRef<HTMLDivElement | null>(null);
  const sides = useMemo(() => ({ a, b }), [a, b]);

  // Line-up anchor (analysis indices). Defaults to backend sync frames (Front foot plants).
  const defaultAnchor = useMemo(
    () => ({
      a: syncFrames.a ?? 0,
      b: syncFrames.b ?? 0,
      label: syncFrames.a != null && syncFrames.b != null ? syncLabel : "the start of each clip",
      momentId: (syncFrames.a != null && syncFrames.b != null ? "front_foot_plant" : null) as MomentId | null,
    }),
    [syncFrames.a, syncFrames.b, syncLabel]
  );
  const [anchor, setAnchor] = useState(defaultAnchor);
  useEffect(() => setAnchor(defaultAnchor), [defaultAnchor, a.id, b.id]);

  const range = useMemo(() => {
    const lo = (k: Key) => (0 - anchor[k]) / analysisFpsOf(sides[k].telemetry);
    const hi = (k: Key) => (nFramesOf(sides[k].telemetry) - 1 - anchor[k]) / analysisFpsOf(sides[k].telemetry);
    const tMin = Math.min(lo("a"), lo("b"));
    const tMax = Math.max(hi("a"), hi("b"));
    return { tMin, tMax: tMax > tMin ? tMax : tMin + 0.001 };
  }, [anchor, sides]);

  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const tRef = useRef(0);
  const playingRef = useRef(false);
  const settled = useRef<{ a: number; b: number }>({ a: 0, b: 0 });
  const seekBusy = useRef<{ a: boolean; b: boolean }>({ a: false, b: false });
  const seekPending = useRef<{ a: number | null; b: number | null }>({ a: null, b: null });
  const [ready, setReady] = useState<{ a: boolean; b: boolean }>({ a: false, b: false });
  const [failed, setFailed] = useState<{ a: boolean; b: boolean }>({ a: false, b: false });

  // Ghost similarity transform (B px → A px), fixed at the sync frames.
  const ghostXf = useMemo(() => {
    const fa = clampFrame(a.telemetry, syncFrames.a ?? anchor.a);
    const fb = clampFrame(b.telemetry, syncFrames.b ?? anchor.b);
    const A = bodyFrameOf(a.telemetry, fa);
    const B = bodyFrameOf(b.telemetry, fb);
    if (A && B) {
      const s = A.size / B.size;
      return (p: number[]): [number, number] => [A.cx + s * (p[0] - B.cx), A.cy + s * (p[1] - B.cy)];
    }
    const s = Math.max(a.telemetry.height || 720, 1) / Math.max(b.telemetry.height || 720, 1);
    return (p: number[]): [number, number] => [p[0] * s, p[1] * s];
  }, [a.telemetry, b.telemetry, syncFrames.a, syncFrames.b, anchor.a, anchor.b]);

  const frameAt = useCallback(
    (k: Key, tt: number) => {
      const tel = sides[k].telemetry;
      return clampFrame(tel, anchor[k] + tt * analysisFpsOf(tel));
    },
    [anchor, sides]
  );

  const frameFromVideo = useCallback(
    (k: Key) => {
      const v = vidRef[k].current;
      const tel = sides[k].telemetry;
      if (!v) return settled.current[k];
      return clampFrame(tel, v.currentTime * proxyFpsOf(tel));
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sides]
  );

  const drawCanvas = useCallback(
    (k: Key) => {
      const canvas = canRef[k].current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      const side = sides[k];
      const tel = side.telemetry;
      const cw = canvas.width;
      const ch = canvas.height;
      const { ox, oy, dw, dh, scale } = videoRect(tel, cw, ch);
      ctx.fillStyle = "#f4f4f5";
      ctx.fillRect(0, 0, cw, ch);
      const v = vidRef[k].current;
      if (v && v.readyState >= 2) ctx.drawImage(v, ox, oy, dw, dh);
      const f = playingRef.current ? frameFromVideo(k) : settled.current[k];
      const mapSelf = (p: number[]): [number, number] => [ox + p[0] * scale, oy + p[1] * scale];
      if (mode === "overlay" && k === "a") {
        const fb = playingRef.current ? frameFromVideo("b") : settled.current.b;
        drawSkeleton(ctx, b.telemetry, fb, (p) => mapSelf(ghostXf(p)), b.bowlingArm === "right", true);
      }
      drawSkeleton(ctx, tel, f, mapSelf, side.bowlingArm === "right", false);
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sides, mode, ghostXf, frameFromVideo, b]
  );

  const drawAll = useCallback(() => {
    drawCanvas("a");
    if (mode === "side") drawCanvas("b");
  }, [drawCanvas, mode]);

  // Coalesced seek (paused): draw only once the picture has landed, so skeleton never tears.
  const seekSide = useCallback(
    (k: Key, frame: number) => {
      const v = vidRef[k].current;
      const tel = sides[k].telemetry;
      const f = clampFrame(tel, frame);
      if (!v || v.readyState < 1) {
        settled.current[k] = f;
        drawAll();
        return;
      }
      const target = f / proxyFpsOf(tel);
      if (Math.abs(v.currentTime - target) <= 0.004) {
        settled.current[k] = f;
        drawAll();
        return;
      }
      if (seekBusy.current[k]) {
        seekPending.current[k] = f;
        return;
      }
      seekBusy.current[k] = true;
      seekPending.current[k] = null;
      v.currentTime = target + 0.0005;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [sides, drawAll]
  );

  const onSeeked = useCallback(
    (k: Key) => {
      seekBusy.current[k] = false;
      const pending = seekPending.current[k];
      if (pending !== null) {
        seekPending.current[k] = null;
        seekSide(k, pending);
        return;
      }
      if (!playingRef.current) {
        settled.current[k] = frameFromVideo(k);
        drawAll();
      }
    },
    [seekSide, frameFromVideo, drawAll]
  );

  const seekBoth = useCallback(
    (tt: number) => {
      tRef.current = tt;
      setT(tt);
      seekSide("a", frameAt("a", tt));
      seekSide("b", frameAt("b", tt));
    },
    [seekSide, frameAt]
  );

  // Reset when clips change.
  useEffect(() => {
    setPlaying(false);
    playingRef.current = false;
    seekBusy.current = { a: false, b: false };
    seekPending.current = { a: null, b: null };
    setReady({ a: false, b: false });
    setFailed({ a: false, b: false });
  }, [a.videoUrl, b.videoUrl]);

  // Re-seek to the clock whenever the anchor changes (or on first load).
  useEffect(() => {
    if (!playingRef.current) seekBoth(Math.min(Math.max(tRef.current, range.tMin), range.tMax));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [anchor, ready.a, ready.b]);

  // Canvas sizing (devicePixelRatio aware) — sizing the bitmap never changes layout.
  useEffect(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    let raf = 0;
    const resize = () => {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      (["a", "b"] as Key[]).forEach((k) => {
        const c = canRef[k].current;
        if (!c) return;
        const w = Math.max(1, Math.round(c.clientWidth * dpr));
        const h = Math.max(1, Math.round(c.clientHeight * dpr));
        if (c.width !== w || c.height !== h) {
          c.width = w;
          c.height = h;
        }
      });
      drawAll();
    };
    const ro = new ResizeObserver(() => {
      cancelAnimationFrame(raf);
      raf = requestAnimationFrame(resize);
    });
    ro.observe(wrap);
    resize();
    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, drawAll]);

  // Playback loop: master clock in action-seconds; each video plays at its own rate and is drift-corrected.
  useEffect(() => {
    playingRef.current = playing;
    const keys: Key[] = ["a", "b"];
    if (!playing) {
      keys.forEach((k) => vidRef[k].current?.pause());
      seekBoth(tRef.current);
      return;
    }
    let raf = 0;
    let wallStart = performance.now();
    let tStart = tRef.current >= range.tMax - 0.01 ? range.tMin : tRef.current;
    let lastUi = 0;
    const step = (now: number) => {
      let tt = tStart + ((now - wallStart) / 1000) * speed;
      if (tt > range.tMax) {
        tt = range.tMin;
        tStart = range.tMin;
        wallStart = now;
      }
      tRef.current = tt;
      for (const k of keys) {
        const v = vidRef[k].current;
        if (!v || v.readyState < 1) continue;
        const tel = sides[k].telemetry;
        const raw = anchor[k] + tt * analysisFpsOf(tel);
        const f = clampFrame(tel, raw);
        const target = f / proxyFpsOf(tel);
        const inside = raw >= 0 && raw <= nFramesOf(tel) - 1;
        const rate = Math.min(Math.max((speed * analysisFpsOf(tel)) / proxyFpsOf(tel), 0.0625), 16);
        if (Math.abs(v.playbackRate - rate) > 1e-3) v.playbackRate = rate;
        if (!inside) {
          if (!v.paused) v.pause();
          if (Math.abs(v.currentTime - target) > 0.02 && !v.seeking) v.currentTime = target;
        } else if (v.paused) {
          v.currentTime = target;
          v.play().catch(() => undefined);
        } else if (Math.abs(v.currentTime - target) > 0.12 && !v.seeking) {
          v.currentTime = target;
        }
      }
      drawAll();
      if (now - lastUi > 80) {
        lastUi = now;
        setT(tt);
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => {
      cancelAnimationFrame(raf);
      keys.forEach((k) => vidRef[k].current?.pause());
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, range.tMin, range.tMax, speed, anchor, sides, drawAll]);

  const lineUpAt = (m: DualMoment) => {
    if (m.a == null || m.b == null) return;
    setPlaying(false);
    playingRef.current = false;
    tRef.current = 0;
    setAnchor({ a: m.a, b: m.b, label: m.label, momentId: m.id });
  };

  const renderCanvas = (k: Key, side: DualSide, ghostTag?: string) => {
    // Fixed, identical box for every clip (portrait clips letterbox inside it, never stretch the layout).
    return (
      <div className="relative w-full rounded-[6px] overflow-hidden bg-[#f4f4f5]" style={{ aspectRatio: "16 / 10", maxHeight: "56vh" }}>
        <canvas ref={canRef[k]} className="absolute inset-0 w-full h-full block" aria-label={`${side.tag} video`} />
        <div className="absolute left-2 top-2 flex flex-wrap gap-1.5 pointer-events-none">
          <span className="inline-flex items-center h-6 px-2 rounded-full bg-white/90 text-[12px] font-[500] text-[#171717] shadow-[0_0_0_1px_#ebebeb]">
            {side.tag}
            {side.caption && <span className="ml-1.5 font-[400] text-[#4d4d4d]">{side.caption}</span>}
          </span>
          {ghostTag && (
            <span className="inline-flex items-center gap-1.5 h-6 px-2 rounded-full bg-white/90 text-[12px] text-[#4d4d4d] shadow-[0_0_0_1px_#ebebeb]">
              <svg width="16" height="4" aria-hidden>
                <line x1="0" y1="2" x2="16" y2="2" stroke="#4d4d4d" strokeWidth="1.6" strokeDasharray="4 3" />
              </svg>
              {ghostTag}
            </span>
          )}
        </div>
        {!ready[k] && !failed[k] && (
          <div className="absolute inset-0 flex items-center justify-center text-[12px] text-[#8f8f8f]">Loading video…</div>
        )}
        {failed[k] && (
          <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[12px] text-[#4d4d4d]">
            This video couldn't be loaded. The measurements below still apply.
          </div>
        )}
      </div>
    );
  };

  const hiddenVideo = (k: Key) => (
    <video
      key={`${k}-${sides[k].videoUrl}`}
      ref={vidRef[k]}
      src={sides[k].videoUrl}
      muted
      playsInline
      preload="auto"
      className="absolute w-px h-px opacity-0 pointer-events-none"
      aria-hidden
      onLoadedData={() => {
        setReady((r) => ({ ...r, [k]: true }));
        if (!playingRef.current) seekSide(k, frameAt(k, tRef.current));
      }}
      onSeeked={() => onSeeked(k)}
      onError={() => setFailed((f) => ({ ...f, [k]: true }))}
    />
  );

  const span = range.tMax - range.tMin;

  return (
    <div className="flex flex-col gap-3">
      <div className="relative" ref={wrapRef}>
        {hiddenVideo("a")}
        {hiddenVideo("b")}
        {mode === "side" ? (
          <div className="grid grid-cols-2 gap-2 items-start">
            {renderCanvas("a", a)}
            {renderCanvas("b", b)}
          </div>
        ) : (
          <div className="mx-auto w-full max-w-[720px]">{renderCanvas("a", a, `${b.tag} outline`)}</div>
        )}
      </div>

      <div className="flex items-center gap-3">
        <button
          type="button"
          onClick={() => setPlaying((p) => !p)}
          className="v-btn-primary h-8 px-3 inline-flex items-center gap-1.5 text-[13px] shrink-0"
          aria-label={playing ? "Pause both" : "Play both at half speed"}
        >
          {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
          {playing ? "Pause" : "Play"}
          <span className="font-mono text-[11px] opacity-70">{speed}×</span>
        </button>
        <input
          type="range"
          min={0}
          max={1000}
          step={1}
          value={span > 0 ? Math.round(((t - range.tMin) / span) * 1000) : 0}
          onChange={(e) => {
            setPlaying(false);
            playingRef.current = false;
            seekBoth(range.tMin + (Number(e.target.value) / 1000) * span);
          }}
          className="flex-1 min-w-0 accent-[#171717]"
          aria-label="Scrub both videos"
        />
      </div>
      <div className="flex flex-wrap items-center justify-between gap-2 text-[12px] text-[#4d4d4d]">
        <span>
          Lined up at <span className="text-[#171717] font-[500]">{anchor.label}</span>
        </span>
        {Math.abs(t) >= 0.005 && <span className="v-tabular text-[#8f8f8f]">{formatRel(t, anchor.label)}</span>}
      </div>

      {moments.length > 0 && (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Key moments">
          {moments.map((m) => {
            const usable = m.a != null && m.b != null;
            const active = anchor.momentId === m.id && Math.abs(t) < 0.005;
            return (
              <button
                key={m.id}
                type="button"
                disabled={!usable}
                onClick={() => lineUpAt(m)}
                title={usable ? `Show both balls at: ${m.label}` : "Only one of the two clips shows this moment"}
                className={`h-8 px-3 rounded-full text-[12.5px] transition-colors duration-150 ${
                  active
                    ? "bg-[#171717] text-white"
                    : usable
                    ? "bg-white text-[#171717] shadow-[0_0_0_1px_#ebebeb] hover:shadow-[0_0_0_1px_#c9c9c9]"
                    : "bg-[#fafafa] text-[#a8a8a8] shadow-[0_0_0_1px_#ebebeb] border-dashed cursor-not-allowed"
                }`}
              >
                {m.label}
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default DualComparePlayer;
