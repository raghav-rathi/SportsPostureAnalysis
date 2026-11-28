/**
 * VideoStage: pro video canvas for the Delivery Workbench (docs/research_app_landscape_v5.md §8.2).
 *
 * - The <video> is decoded offscreen and composited with every overlay onto ONE canvas (zero seek tearing, exact
 *   aspect-preserving letterbox, zoom/pan 1×–4× with a corner minimap when zoomed > 1.2×).
 * - Synced to the shared playhead: paused/scrubbing → coalesced exact seeks; playing forward at a supported speed →
 *   native playback at (fps × rate / proxy_fps) with drift correction; reverse shuttle → per-frame seeks.
 * - Segment-coloured skeleton with a 2 px white halo (reads on white kit AND dark backgrounds), per-segment toggles.
 * - Tap-a-joint live angle (Select tool): drawn 2D vs 3D pose angle with a propagated interval, plus the measured
 *   fact-sheet value when one exists. Bowling elbow ALWAYS carries "Visual trace only — not ICC legality".
 * - 2D trails: fading wrist whip (0.6 s tail), dashed head path, COM crosshair path (only when COM passes its gate).
 * - Magenta telestration (deliberately outside the science palette): select, line, angle (snaps to joints within 22 px →
 *   "drawn 2D" + "3D ± CI"), plumb (true vertical from the calibrated world frame), circle, arrow, freehand, ruler
 *   (px, or metres ± σ when calibrated), text, spotlight.
 */
import React, { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import type { BiomechanicalFactSheet, MetricEstimate, TelemetryV2, TelestratorShape, TelestratorTool } from "../../types";
import { J, OBS_GLYPH, SEG, SEG_COLOR, SMPL24_BONES, SMPL24_JOINTS, boneSegment, jointSegment, type SegmentKey } from "../../theme";
import { api } from "../../api";
import { usePlayhead } from "../PlayheadProvider";

export interface LayerVisibility {
  skeleton: boolean;
  segments: { pelvis: boolean; thorax: boolean; arm: boolean; lead: boolean };
  trails: { wrist: boolean; head: boolean; com: boolean };
  angleChips: boolean;
  telestration: boolean;
}

export const DEFAULT_LAYERS: LayerVisibility = {
  skeleton: true,
  segments: { pelvis: true, thorax: true, arm: true, lead: true },
  trails: { wrist: true, head: true, com: false },
  angleChips: true,
  telestration: true,
};

export interface VideoStageProps {
  deliveryId: string;
  videoUrl: string;
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  bowlingArm: "right" | "left";
  activeTool: TelestratorTool;
  layers: LayerVisibility;
  shapes: TelestratorShape[];
  onShapesChange: (next: TelestratorShape[]) => void;
  pinNewShapes: boolean;
  zoom: number;
  onZoomChange: (z: number) => void;
  highlightJoints?: number[];
  activeCaption?: string | null;
}

type P2 = [number, number];
const TELE = SEG.tele;
const SNAP_PX = 22;
const PICK_PX = 18;
const ZMIN = 1;
const ZMAX = 4;
const FONT = '500 11px "Geist Mono", ui-monospace, Menlo, monospace';
const FONT_SM = '500 10px "Geist Mono", ui-monospace, Menlo, monospace';

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const isP2 = (p: unknown): p is P2 => Array.isArray(p) && p.length >= 2 && Number.isFinite(p[0]) && Number.isFinite(p[1]);
const fin3 = (p: number[] | undefined | null): p is number[] => !!p && p.length >= 3 && p.every((v) => Number.isFinite(v));

function angle2d(a: P2, b: P2, c: P2): number {
  const ux = a[0] - b[0], uy = a[1] - b[1], vx = c[0] - b[0], vy = c[1] - b[1];
  const d1 = Math.hypot(ux, uy), d2 = Math.hypot(vx, vy);
  if (d1 < 1e-6 || d2 < 1e-6) return NaN;
  return (Math.acos(clamp((ux * vx + uy * vy) / (d1 * d2), -1, 1)) * 180) / Math.PI;
}

/** 3D angle at B with a conservative first-order propagation of per-joint σ (metres, treated per-axis). */
function angle3dWithCi(A: number[], B: number[], C: number[], sA: number, sB: number, sC: number) {
  const u = [A[0] - B[0], A[1] - B[1], A[2] - B[2]];
  const v = [C[0] - B[0], C[1] - B[1], C[2] - B[2]];
  const nu = Math.hypot(u[0], u[1], u[2]), nv = Math.hypot(v[0], v[1], v[2]);
  if (nu < 1e-4 || nv < 1e-4) return null;
  const cos = clamp((u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) / (nu * nv), -1, 1);
  const deg = (Math.acos(cos) * 180) / Math.PI;
  const varRad = (sA / nu) ** 2 + (sC / nv) ** 2 + sB ** 2 * (1 / nu + 1 / nv) ** 2;
  const sigDeg = (Math.sqrt(Math.max(0, varRad)) * 180) / Math.PI;
  const half = 1.96 * (Number.isFinite(sigDeg) ? sigDeg : 0);
  // Physical bounds for an internal joint angle: [0°, 180°].
  return { deg, lo: clamp(deg - half, 0, 180), hi: clamp(deg + half, 0, 180), sigDeg };
}

/** Solve a 4×4 linear system (Gaussian elimination, partial pivoting). */
function solve4(A: number[][], b: number[]): number[] | null {
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < 4; c++) {
    let piv = c;
    for (let r = c + 1; r < 4; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-9) return null;
    [M[c], M[piv]] = [M[piv], M[c]];
    for (let r = 0; r < 4; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      for (let k = c; k < 5; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[4] / M[i][i]);
}

interface AffineFit {
  u: number[]; // u = u0*X + u1*Y + u2*Z + u3
  v: number[];
  rmsPx: number;
}

/** Per-frame weak-perspective (affine) projection world → image fitted to the 24 joints. */
function fitAffine(J3: number[][] | undefined, J2: number[][] | undefined, inFr?: boolean[]): AffineFit | null {
  if (!J3 || !J2) return null;
  const AtA = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]];
  const Atu = [0, 0, 0, 0];
  const Atv = [0, 0, 0, 0];
  const rows: Array<[number[], number, number]> = [];
  for (let i = 0; i < Math.min(J3.length, J2.length); i++) {
    if (inFr && inFr[i] === false) continue;
    const w = J3[i], p = J2[i];
    if (!fin3(w) || !isP2(p)) continue;
    const r = [w[0], w[1], w[2], 1];
    rows.push([r, p[0], p[1]]);
    for (let a = 0; a < 4; a++) {
      Atu[a] += r[a] * p[0];
      Atv[a] += r[a] * p[1];
      for (let b = 0; b < 4; b++) AtA[a][b] += r[a] * r[b];
    }
  }
  if (rows.length < 6) return null;
  const u = solve4(AtA, Atu);
  const v = solve4(AtA, Atv);
  if (!u || !v) return null;
  let ss = 0;
  for (const [r, pu, pv] of rows) {
    const eu = u[0] * r[0] + u[1] * r[1] + u[2] * r[2] + u[3] - pu;
    const ev = v[0] * r[0] + v[1] * r[1] + v[2] * r[2] + v[3] - pv;
    ss += eu * eu + ev * ev;
  }
  return { u, v, rmsPx: Math.sqrt(ss / rows.length) };
}

// ── Joint angle catalogue (tap-a-joint) ─────────────────────────────────────
interface AngleDef {
  tri: [number, number, number];
  label: string;
  isBowlingElbow?: boolean;
  kinKey?: string;
  metrics?: Array<{ key: string; event: "bfc" | "ffc" | "br"; eventLabel: string }>;
}

function angleDefFor(idx: number, arm: "right" | "left"): AngleDef | null {
  const b = arm === "right" ? "r" : "l"; // bowling side
  const f = arm === "right" ? "l" : "r"; // front (lead) leg side for a fast bowler
  const name = SMPL24_JOINTS[idx];
  const side = name[0] as "l" | "r";
  const jj = (n: string) => J[n as keyof typeof J];
  const legWord = side === f ? "Front" : "Back";
  const armWord = side === b ? "Bowling" : "Lead";
  const base = name.slice(1);
  const resolved = base === "toe" ? "ank" : base === "han" ? "wri" : base === "cla" ? "sho" : base;
  const s = side;
  switch (resolved) {
    case "kne":
      return {
        tri: [jj(`${s}hip`), jj(`${s}kne`), jj(`${s}ank`)],
        label: `${legWord} knee`,
        kinKey: side === f ? "front_knee_deg" : undefined,
        metrics:
          side === f
            ? [
                { key: "front_knee_ffc_deg", event: "ffc", eventLabel: "front-foot plant" },
                { key: "front_knee_br_deg", event: "br", eventLabel: "ball release" },
              ]
            : undefined,
      };
    case "ank":
      return { tri: [jj(`${s}kne`), jj(`${s}ank`), jj(`${s}toe`)], label: `${legWord} ankle` };
    case "hip":
      return {
        tri: [jj(`${s}sho`), jj(`${s}hip`), jj(`${s}kne`)],
        label: `${legWord} hip`,
        kinKey: side !== f ? "rear_hip_deg" : undefined,
        metrics: side !== f ? [{ key: "rear_hip_bfc_deg", event: "bfc", eventLabel: "back-foot landing" }] : undefined,
      };
    case "elb":
      return {
        tri: [jj(`${s}sho`), jj(`${s}elb`), jj(`${s}wri`)],
        label: `${armWord} elbow`,
        isBowlingElbow: side === b,
      };
    case "sho":
      return { tri: [jj(`${s}elb`), jj(`${s}sho`), jj(`${s}hip`)], label: `${armWord} shoulder (arm-to-trunk)` };
    case "wri":
      return { tri: [jj(`${s}elb`), jj(`${s}wri`), jj(`${s}han`)], label: `${armWord} wrist` };
    default:
      return null;
  }
}

const JOINT_PLAIN: Record<string, string> = {
  pelv: "Pelvis", spi1: "Lower spine", spi2: "Mid spine", spi3: "Upper spine", neck: "Neck", head: "Head",
};

function layerKeyForSeg(seg: SegmentKey): keyof LayerVisibility["segments"] {
  if (seg === "pelvis") return "pelvis";
  if (seg === "arm" || seg === "wrist") return "arm";
  if (seg === "lead") return "lead";
  return "thorax";
}

function fmtDeg(v: number | null | undefined, d = 0) {
  return v == null || !Number.isFinite(v) ? "—" : `${v.toFixed(d)}°`;
}

function comGatePasses(fs: BiomechanicalFactSheet): boolean {
  const cb = fs.com_braking;
  if (!cb) return false;
  return [cb.com_ffc_speed_mps, cb.com_br_speed_mps, cb.com_braking_delta_vx_mps, cb.com_vertical_sink_pct_stature].some(
    (m) => m && !m.suppressed && m.observability !== "unmeasurable"
  );
}

interface Trail2D {
  frames: number[];
  pts: Array<P2 | null>;
}

function buildTrail2D(t: TelemetryV2, key: "bowling_wrist" | "head" | "com", jointIdx: number | null): Trail2D | null {
  const pre = t.trails?.["2d_px"]?.[key];
  const fi = t.trails?.frame_indices;
  if (pre && fi && pre.length === fi.length && pre.length > 1) {
    return { frames: fi.slice(), pts: pre.map((p) => (isP2(p) ? [p[0], p[1]] : null)) };
  }
  if (key === "com") {
    const c = t.com_2d_px;
    if (!c || c.length < 2) return null;
    return { frames: c.map((_, i) => i), pts: c.map((p) => (isP2(p) ? [p[0], p[1]] : null)) };
  }
  if (jointIdx == null) return null;
  const frames: number[] = [];
  const pts: Array<P2 | null> = [];
  for (let i = 0; i < t.joints2d_px.length; i++) {
    const p = t.joints2d_px[i]?.[jointIdx];
    const ok = isP2(p) && t.joint_in_frame?.[i]?.[jointIdx] !== false;
    frames.push(i);
    pts.push(ok ? [p[0], p[1]] : null);
  }
  return { frames, pts };
}

function newId() {
  return `ann_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

// ─────────────────────────────────────────────────────────────────────────────

export const VideoStage: React.FC<VideoStageProps> = ({
  deliveryId,
  videoUrl,
  telemetry,
  factSheet,
  bowlingArm,
  activeTool,
  layers,
  shapes,
  onShapesChange,
  pinNewShapes,
  zoom,
  onZoomChange,
  highlightJoints,
  activeCaption,
}) => {
  const ph = usePlayhead();
  const { frame, playing, rate, fps, setPlaying } = ph;
  const direction = ph.direction ?? 1;

  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [size, setSize] = useState({ w: 0, h: 0, dpr: 1 });
  const [videoState, setVideoState] = useState<"loading" | "ready" | "error">("loading");
  const [selectedJoint, setSelectedJoint] = useState<number | null>(null);
  const [textPrompt, setTextPrompt] = useState<{ pt: P2; value: string } | null>(null);
  const [draftTick, setDraftTick] = useState(0);

  const panRef = useRef<P2>([0, 0]);
  const pendingPanRef = useRef<P2 | null>(null);
  const prevZoomRef = useRef(zoom);
  const hoverRef = useRef<P2 | null>(null); // video px
  const draftRef = useRef<{ tool: TelestratorTool; pts: P2[]; snaps: Array<number | null> } | null>(null);
  const dragRef = useRef<{ kind: "pan" | "draw" | "tap"; startScreen: P2; startPan: P2; moved: boolean } | null>(null);
  const pointersRef = useRef<Map<number, P2>>(new Map());
  const pinchRef = useRef<{ d0: number; z0: number } | null>(null);
  const seekingRef = useRef(false);
  const pendingSeekRef = useRef<number | null>(null);
  const imgCacheRef = useRef<Map<number, HTMLImageElement>>(new Map());
  const lastImgRef = useRef<{ img: HTMLImageElement; frame: number } | null>(null);

  const n = Math.max(1, telemetry.num_frames || telemetry.joints2d_px.length || 1);
  const f = clamp(Math.round(frame), 0, n - 1);
  const frameRef = useRef(f);
  frameRef.current = f;
  const lastDecodedVidFrameRef = useRef(f);
  const lastSyncedVidFrameRef = useRef(-1);
  const prevPlayheadFrameRef = useRef(f);
  const nativeOkRef = useRef(false);
  const phSetFrameRef = useRef(ph.setFrame);
  phSetFrameRef.current = ph.setFrame;
  const loopRangeRef = useRef(ph.loopRange);
  loopRangeRef.current = ph.loopRange;

  const vw = Math.max(1, telemetry.width || 1280);
  const vh = Math.max(1, telemetry.height || 720);
  const proxyFps = Math.max(1, telemetry.proxy_fps || 30);
  const analysisFps = Math.max(1, telemetry.analysis_fps || fps || 30);
  const isMetric = !!factSheet.calibration?.is_metric_calibrated;

  const videoTimeOf = useCallback(
    (fi: number) => {
      const pts = telemetry.frame_pts_ms?.[fi];
      const t = Number.isFinite(pts) ? (pts as number) / 1000 : fi / proxyFps;
      return t + 0.25 / proxyFps; // land inside the frame, not on its boundary
    },
    [telemetry.frame_pts_ms, proxyFps]
  );

  const frameOfVideoTime = useCallback(
    (timeSec: number): number => {
      const pts = telemetry.frame_pts_ms;
      if (pts && pts.length === n) {
        const tMs = timeSec * 1000 + 0.1;
        if (tMs <= pts[0]) return 0;
        if (tMs >= pts[n - 1]) return n - 1;
        let lo = 0;
        let hi = n - 1;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if (pts[mid] <= tMs) lo = mid;
          else hi = mid;
        }
        return lo;
      }
      return clamp(Math.floor(timeSec * proxyFps + 1e-4), 0, n - 1);
    },
    [telemetry.frame_pts_ms, n, proxyFps]
  );

  // ── Geometry: video px ↔ screen (CSS px) ─────────────────────────────────
  const geo = useCallback(() => {
    const W = size.w, H = size.h;
    const s0 = Math.min(W / vw, H / vh) || 1;
    const ox = (W - vw * s0) / 2;
    const oy = (H - vh * s0) / 2;
    const [px, py] = panRef.current;
    const z = zoom;
    const toS = (x: number, y: number): P2 => [W / 2 + px + (ox + x * s0 - W / 2) * z, H / 2 + py + (oy + y * s0 - H / 2) * z];
    const toV = (sx: number, sy: number): P2 => [((sx - W / 2 - px) / z + W / 2 - ox) / s0, ((sy - H / 2 - py) / z + H / 2 - oy) / s0];
    return { W, H, s0, ox, oy, z, toS, toV, scale: s0 * z };
  }, [size.w, size.h, vw, vh, zoom]);

  const clampPan = useCallback(
    (p: P2, z: number): P2 => {
      const mx = Math.max(0, (size.w * z - size.w) / 2);
      const my = Math.max(0, (size.h * z - size.h) / 2);
      return [clamp(p[0], -mx, mx), clamp(p[1], -my, my)];
    },
    [size.w, size.h]
  );

  // Zoom changes (from the rail or wheel): keep the pan consistent.
  useEffect(() => {
    const prev = prevZoomRef.current || 1;
    if (pendingPanRef.current) {
      panRef.current = clampPan(pendingPanRef.current, zoom);
      pendingPanRef.current = null;
    } else {
      panRef.current = clampPan([(panRef.current[0] * zoom) / prev, (panRef.current[1] * zoom) / prev], zoom);
    }
    prevZoomRef.current = zoom;
  }, [zoom, clampPan]);

  // ── Resize ───────────────────────────────────────────────────────────────
  useLayoutEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const r = el.getBoundingClientRect();
      setSize({ w: Math.max(1, Math.round(r.width)), h: Math.max(1, Math.round(r.height)), dpr: Math.min(2.5, window.devicePixelRatio || 1) });
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // ── Derived data ─────────────────────────────────────────────────────────
  const wristIdx = bowlingArm === "right" ? J.rwri : J.lwri;
  const wristTrail = useMemo(() => buildTrail2D(telemetry, "bowling_wrist", wristIdx), [telemetry, wristIdx]);
  const headTrail = useMemo(() => buildTrail2D(telemetry, "head", J.head), [telemetry]);
  const comOk = useMemo(() => !!(telemetry.com_2d_px || telemetry.trails?.["2d_px"]?.com) && comGatePasses(factSheet), [telemetry, factSheet]);
  const comTrail = useMemo(() => (comOk ? buildTrail2D(telemetry, "com", null) : null), [telemetry, comOk]);
  const hl = useMemo(() => new Set(highlightJoints ?? []), [highlightJoints]);

  const affineCache = useRef<Map<number, AffineFit | null>>(new Map());
  useEffect(() => affineCache.current.clear(), [telemetry]);
  const affineAt = useCallback(
    (fi: number) => {
      if (affineCache.current.has(fi)) return affineCache.current.get(fi) ?? null;
      const a = fitAffine(telemetry.joints3d_world_m[fi], telemetry.joints2d_px[fi], telemetry.joint_in_frame?.[fi]);
      affineCache.current.set(fi, a);
      return a;
    },
    [telemetry]
  );

  /** Image-space "up" direction for true vertical (calibrated world +Y projected), with provenance. */
  const verticalAt = useCallback(
    (fi: number): { dir: P2; calibrated: boolean; source: string } => {
      const a = affineAt(fi);
      if (a) {
        const dx = a.u[1], dy = a.v[1];
        const L = Math.hypot(dx, dy);
        if (L > 1e-6) return { dir: [dx / L, dy / L], calibrated: true, source: "calibrated gravity" };
      }
      const wf = telemetry.mcc_wireframe_2d;
      const b = wf?.stump_middle_base, t = wf?.stump_middle_top;
      if (isP2(b) && isP2(t)) {
        const dx = t[0] - b[0], dy = t[1] - b[1];
        const L = Math.hypot(dx, dy);
        if (L > 1) return { dir: [dx / L, dy / L], calibrated: true, source: "calibrated stump axis" };
      }
      return { dir: [0, -1], calibrated: false, source: "image vertical (not calibrated)" };
    },
    [affineAt, telemetry.mcc_wireframe_2d]
  );

  const jointPosAt = useCallback(
    (fi: number, j: number): P2 | null => {
      const p = telemetry.joints2d_px[fi]?.[j];
      return isP2(p) ? [p[0], p[1]] : null;
    },
    [telemetry]
  );

  const nearestJoint = useCallback(
    (pt: P2, maxScreenPx: number): number | null => {
      const g = geo();
      const pts = telemetry.joints2d_px[f];
      const inFr = telemetry.joint_in_frame?.[f];
      if (!pts) return null;
      const [sx, sy] = g.toS(pt[0], pt[1]);
      let best: number | null = null;
      let bestD = maxScreenPx;
      for (let i = 0; i < pts.length; i++) {
        if (inFr && inFr[i] === false) continue;
        const p = pts[i];
        if (!isP2(p)) continue;
        const [qx, qy] = g.toS(p[0], p[1]);
        const d = Math.hypot(qx - sx, qy - sy);
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      return best;
    },
    [geo, telemetry, f]
  );

  const angle3dAt = useCallback(
    (fi: number, tri: [number, number, number]) => {
      const P = telemetry.joints3d_world_m[fi];
      const S = telemetry.joint_uncertainties_m?.[fi];
      if (!P) return null;
      const [a, b, c] = tri;
      if (!fin3(P[a]) || !fin3(P[b]) || !fin3(P[c])) return null;
      return angle3dWithCi(P[a], P[b], P[c], S?.[a] ?? 0, S?.[b] ?? 0, S?.[c] ?? 0);
    },
    [telemetry]
  );

  const shapePoints = useCallback(
    (sh: TelestratorShape, fi: number): P2[] => {
      // Pinned shapes snapped to joints follow the body live.
      if (sh.pinned_all_frames && sh.snap_joints && sh.snap_joints.length === sh.points.length) {
        const live = sh.snap_joints.map((j, k) => (j >= 0 ? jointPosAt(fi, j) : null) ?? (sh.points[k] as P2));
        return live;
      }
      return sh.points as P2[];
    },
    [jointPosAt]
  );

  const rulerLabel = useCallback(
    (pts: P2[], snaps: number[] | undefined, fi: number): string => {
      const px = Math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]);
      const pxTxt = `${px.toFixed(0)} px`;
      if (!isMetric) return `${pxTxt} · metres need a calibrated clip`;
      if (snaps && snaps.length === 2 && snaps[0] >= 0 && snaps[1] >= 0) {
        const P = telemetry.joints3d_world_m[fi];
        const S = telemetry.joint_uncertainties_m?.[fi];
        const a = P?.[snaps[0]], b = P?.[snaps[1]];
        if (fin3(a) && fin3(b)) {
          const d = Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
          const sig = Math.hypot(S?.[snaps[0]] ?? 0, S?.[snaps[1]] ?? 0);
          return `${d.toFixed(2)} m ± ${(1.96 * sig).toFixed(2)} · 3D joint-to-joint`;
        }
      }
      const A = affineAt(fi);
      if (A) {
        // Ground plane (Y = 0): [u - u3, v - v3] = [[u0, u2], [v0, v2]] · [X, Z]
        const a = A.u[0], b = A.u[2], c = A.v[0], d = A.v[2];
        const det = a * d - b * c;
        if (Math.abs(det) > 1e-6) {
          const inv = (p: P2): P2 => {
            const du = p[0] - A.u[3], dv = p[1] - A.v[3];
            return [(d * du - b * dv) / det, (-c * du + a * dv) / det];
          };
          const g0 = inv(pts[0]), g1 = inv(pts[1]);
          const m = Math.hypot(g1[0] - g0[0], g1[1] - g0[1]);
          const pxPerM = Math.sqrt(Math.abs(det));
          const sig = Math.hypot(Math.SQRT2 * (A.rmsPx / pxPerM), 0.05 * m); // fit residual ⊕ 5 % weak-perspective scale
          return `≈ ${m.toFixed(2)} m ± ${(1.96 * sig).toFixed(2)} · on pitch plane`;
        }
      }
      return pxTxt;
    },
    [isMetric, telemetry, affineAt]
  );

  // ── Frame image fallback (video loading / failed / exact cached seek frame) ──
  const drawRef = useRef<() => void>(() => {});
  const frameImage = useCallback(
    (fi: number): { img: HTMLImageElement; frame: number } | null => {
      const cache = imgCacheRef.current;
      let img = cache.get(fi);
      if (!img) {
        img = new Image();
        img.decoding = "async";
        img.onload = () => {
          lastImgRef.current = { img: img!, frame: fi };
          drawRef.current();
        };
        img.src = api.frameUrl(deliveryId, fi, 960);
        cache.set(fi, img);
        if (cache.size > 160) {
          const k = cache.keys().next().value;
          if (k !== undefined) cache.delete(k);
        }
      }
      if (img.complete && img.naturalWidth > 0) {
        const pair = { img, frame: fi };
        lastImgRef.current = pair;
        return pair;
      }
      return lastImgRef.current;
    },
    [deliveryId]
  );
  useEffect(() => {
    imgCacheRef.current.clear();
    lastImgRef.current = null;
  }, [deliveryId]);

  // ─────────────────────────────────────────────────────────────────────────
  // Draw
  // ─────────────────────────────────────────────────────────────────────────
  const draw = () => {
    const cv = canvasRef.current;
    if (!cv || size.w < 2) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const g = geo();
    const { W, H } = g;
    ctx.setTransform(size.dpr, 0, 0, size.dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = "#f4f4f5";
    ctx.fillRect(0, 0, W, H);

    // 1. Frame — lock displayed video/JPEG pixels and 2D skeleton/trails to the exact same frame `rf`.
    const [x0, y0] = g.toS(0, 0);
    const dw = vw * g.scale, dh = vh * g.scale;
    const vid = videoRef.current;
    let src: CanvasImageSource | null = null;
    let rf = f;

    if (vid && videoState === "ready" && vid.readyState >= 2) {
      if (vid.seeking || seekingRef.current) {
        // Never draw post-seek skeleton over pre-seek video pixels: use an already-cached exact JPEG or hold matched pair.
        const exact = imgCacheRef.current.get(f);
        if (exact && exact.complete && exact.naturalWidth > 0) {
          src = exact;
          rf = f;
        } else {
          src = vid;
          rf = lastDecodedVidFrameRef.current;
        }
      } else {
        const vidFrame = frameOfVideoTime(vid.currentTime);
        lastDecodedVidFrameRef.current = vidFrame;
        if (nativeOkRef.current) {
          src = vid;
          rf = vidFrame;
          // Keep shared Playhead (3D pitch, timeline, waterfall) locked gently to the decoded video frame.
          const lr = loopRangeRef.current;
          const lo = lr ? lr[0] : 0;
          const hi = lr ? lr[1] : n - 1;
          if (vidFrame !== lastSyncedVidFrameRef.current) {
            lastSyncedVidFrameRef.current = vidFrame;
            if (vidFrame >= lo && vidFrame <= hi && vidFrame !== frameRef.current) {
              phSetFrameRef.current(vidFrame);
            }
          }
        } else if (vidFrame === f) {
          src = vid;
          rf = f;
        } else {
          const exact = imgCacheRef.current.get(f);
          if (exact && exact.complete && exact.naturalWidth > 0) {
            src = exact;
            rf = f;
          } else {
            src = vid;
            rf = vidFrame;
          }
        }
      }
    } else {
      const fb = frameImage(f);
      if (fb) {
        src = fb.img;
        rf = fb.frame;
      }
    }

    if (src) {
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      try {
        ctx.drawImage(src, x0, y0, dw, dh);
      } catch {
        /* decode race — skip this paint */
      }
    }

    const pts = telemetry.joints2d_px[rf];
    const inFr = telemetry.joint_in_frame?.[rf];
    const ok = (i: number) => !!pts && isP2(pts[i]) && !(inFr && inFr[i] === false);
    const S = (i: number) => g.toS(pts![i][0], pts![i][1]);

    // 2. Trails (under the skeleton)
    if (layers.trails.wrist && wristTrail) drawWristTrail(ctx, g.toS, wristTrail, rf, Math.max(3, Math.round(0.6 * analysisFps)));
    if (layers.trails.head && headTrail) drawHeadTrail(ctx, g.toS, headTrail, rf);
    if (layers.trails.com && comTrail) drawComTrail(ctx, g.toS, comTrail, rf);

    // 3. Skeleton (segment colour + 2 px white halo)
    if (layers.skeleton && pts) {
      ctx.save();
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      const wBone = clamp(2.2 + g.scale * 0.6, 2.4, 4.2);
      const visibleBones = SMPL24_BONES.filter(([a, b]) => ok(a) && ok(b) && layers.segments[layerKeyForSeg(boneSegment(a, b, bowlingArm))]);
      ctx.strokeStyle = "rgba(255,255,255,0.95)";
      ctx.lineWidth = wBone + 4;
      ctx.beginPath();
      for (const [a, b] of visibleBones) {
        const pa = S(a), pb = S(b);
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
      }
      ctx.stroke();
      for (const [a, b] of visibleBones) {
        const pa = S(a), pb = S(b);
        ctx.strokeStyle = SEG_COLOR[boneSegment(a, b, bowlingArm)];
        ctx.lineWidth = wBone;
        ctx.beginPath();
        ctx.moveTo(pa[0], pa[1]);
        ctx.lineTo(pb[0], pb[1]);
        ctx.stroke();
      }
      for (let i = 0; i < 24; i++) {
        if (!ok(i)) continue;
        const seg = jointSegment(i, bowlingArm);
        if (!layers.segments[layerKeyForSeg(seg)]) continue;
        const [x, y] = S(i);
        const r = i === selectedJoint ? 6 : hl.has(i) ? 5.5 : 3.6;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = SEG_COLOR[seg];
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#ffffff";
        ctx.stroke();
        if (hl.has(i) || i === selectedJoint) {
          ctx.beginPath();
          ctx.arc(x, y, r + 5, 0, Math.PI * 2);
          ctx.lineWidth = 4;
          ctx.strokeStyle = "rgba(255,255,255,0.9)";
          ctx.stroke();
          ctx.lineWidth = 1.8;
          ctx.strokeStyle = SEG.ink;
          ctx.stroke();
        }
      }
      ctx.restore();
    }

    // 4. Selected-joint angle arc
    if (selectedJoint != null && layers.angleChips && pts) {
      const def = angleDefFor(selectedJoint, bowlingArm);
      if (def && def.tri.every(ok)) {
        const [A, B, C] = def.tri.map(S) as [P2, P2, P2];
        drawAngleArc(ctx, A, B, C, SEG_COLOR[jointSegment(def.tri[1], bowlingArm)], true);
      }
    }

    // 5. Spotlights (dim everything outside the circle)
    if (layers.telestration) {
      const spots = shapes.filter((s) => s.type === "spotlight" && (s.frame === rf || s.pinned_all_frames) && s.points.length >= 2);
      const draft = draftRef.current;
      const draftSpot = draft && draft.tool === "spotlight" && draft.pts.length >= 1 && hoverRef.current ? [draft.pts[0], hoverRef.current] : null;
      const circles: Array<[P2, number]> = [];
      for (const s of spots) {
        const p = shapePoints(s, rf);
        const c = g.toS(p[0][0], p[0][1]);
        const e = g.toS(p[1][0], p[1][1]);
        circles.push([c, Math.hypot(e[0] - c[0], e[1] - c[1])]);
      }
      if (draftSpot) {
        const c = g.toS(draftSpot[0][0], draftSpot[0][1]);
        const e = g.toS(draftSpot[1][0], draftSpot[1][1]);
        circles.push([c, Math.hypot(e[0] - c[0], e[1] - c[1])]);
      }
      if (circles.length) {
        ctx.save();
        ctx.beginPath();
        ctx.rect(0, 0, W, H);
        for (const [c, r] of circles) {
          ctx.moveTo(c[0] + r, c[1]);
          ctx.arc(c[0], c[1], r, 0, Math.PI * 2, true);
        }
        ctx.fillStyle = "rgba(10,10,10,0.58)";
        ctx.fill("evenodd");
        ctx.restore();
      }
    }

    // 6. Telestration shapes + draft
    if (layers.telestration) {
      for (const sh of shapes) {
        if (sh.frame !== rf && !sh.pinned_all_frames) continue;
        drawShape(ctx, sh.type, shapePoints(sh, rf), sh, false);
      }
      const d = draftRef.current;
      if (d && d.pts.length > 0) {
        const hv = hoverRef.current;
        const pv: P2[] = d.tool === "freehand" ? d.pts : hv ? [...d.pts, snapPoint(hv, d.tool).pt] : d.pts;
        const t = d.tool === "cursor" ? null : d.tool;
        if (t) drawShape(ctx, t, pv, { snap_joints: d.snaps.map((s) => s ?? -1) }, true);
      }
    }

    // 7. Minimap
    if (zoom > 1.2 && src) {
      const mw = Math.min(150, W * 0.28);
      const mh = (mw * vh) / vw;
      const mx = W - mw - 10, my = 10;
      ctx.save();
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(mx - 3, my - 3, mw + 6, mh + 6);
      try {
        ctx.drawImage(src, mx, my, mw, mh);
      } catch {
        /* ignore */
      }
      const a = g.toV(0, 0), b = g.toV(W, H);
      const rx = mx + (clamp(a[0], 0, vw) / vw) * mw;
      const ry = my + (clamp(a[1], 0, vh) / vh) * mh;
      const rw = ((clamp(b[0], 0, vw) - clamp(a[0], 0, vw)) / vw) * mw;
      const rh = ((clamp(b[1], 0, vh) - clamp(a[1], 0, vh)) / vh) * mh;
      ctx.fillStyle = "rgba(0,0,0,0.28)";
      ctx.beginPath();
      ctx.rect(mx, my, mw, mh);
      ctx.rect(rx, ry, rw, rh);
      ctx.fill("evenodd");
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.strokeRect(rx, ry, rw, rh);
      ctx.strokeStyle = "rgba(0,0,0,0.12)";
      ctx.lineWidth = 1;
      ctx.strokeRect(mx - 3, my - 3, mw + 6, mh + 6);
      ctx.restore();
    }

    // ── local drawing helpers (close over g / f) ──
    function snapPoint(p: P2, tool: TelestratorTool): { pt: P2; joint: number | null } {
      if (tool !== "angle" && tool !== "ruler") return { pt: p, joint: null };
      const j = nearestJoint(p, SNAP_PX);
      if (j == null) return { pt: p, joint: null };
      const q = jointPosAt(f, j);
      return q ? { pt: q, joint: j } : { pt: p, joint: null };
    }

    function haloStroke(path: () => void, color = TELE, width = 2.5, dashed = false) {
      ctx!.save();
      ctx!.lineCap = "round";
      ctx!.lineJoin = "round";
      ctx!.setLineDash([]);
      ctx!.strokeStyle = "rgba(255,255,255,0.95)";
      ctx!.lineWidth = width + 4;
      ctx!.beginPath();
      path();
      ctx!.stroke();
      if (dashed) ctx!.setLineDash([7, 5]);
      ctx!.strokeStyle = color;
      ctx!.lineWidth = width;
      ctx!.beginPath();
      path();
      ctx!.stroke();
      ctx!.restore();
    }

    function chip(x: number, y: number, lines: string[], opts: { border?: string; ink?: string; bg?: string } = {}) {
      ctx!.save();
      ctx!.font = FONT;
      const padX = 6, lh = 15;
      const w = Math.max(...lines.map((l) => ctx!.measureText(l).width)) + padX * 2;
      const h = lines.length * lh + 6;
      const bx = clamp(x, 4, W - w - 4);
      const by = clamp(y - h, 4, H - h - 4);
      ctx!.fillStyle = opts.bg ?? "rgba(255,255,255,0.96)";
      roundRect(ctx!, bx, by, w, h, 5);
      ctx!.fill();
      ctx!.lineWidth = 1.25;
      ctx!.strokeStyle = opts.border ?? TELE;
      ctx!.stroke();
      ctx!.fillStyle = opts.ink ?? SEG.ink;
      ctx!.textBaseline = "middle";
      lines.forEach((l, i) => ctx!.fillText(l, bx + padX, by + 3 + lh * i + lh / 2));
      ctx!.restore();
    }

    function handle(p: P2) {
      ctx!.beginPath();
      ctx!.arc(p[0], p[1], 3.6, 0, Math.PI * 2);
      ctx!.fillStyle = "#ffffff";
      ctx!.fill();
      ctx!.lineWidth = 1.8;
      ctx!.strokeStyle = TELE;
      ctx!.stroke();
    }

    function drawShape(
      c: CanvasRenderingContext2D,
      type: Exclude<TelestratorTool, "cursor">,
      ptsV: P2[],
      meta: { label?: string; snap_joints?: number[] },
      isDraft: boolean
    ) {
      if (!ptsV.length) return;
      const P = ptsV.map((p) => g.toS(p[0], p[1]));
      const dash = isDraft;
      switch (type) {
        case "line": {
          if (P.length < 2) break;
          haloStroke(() => {
            c.moveTo(P[0][0], P[0][1]);
            c.lineTo(P[1][0], P[1][1]);
          }, TELE, 2.5, dash);
          let ang = (Math.atan2(-(ptsV[1][1] - ptsV[0][1]), ptsV[1][0] - ptsV[0][0]) * 180) / Math.PI;
          if (ang > 90) ang -= 180;
          if (ang < -90) ang += 180;
          haloStroke(() => {
            c.moveTo(P[0][0], P[0][1]);
            c.lineTo(P[0][0] + 36 * Math.sign(P[1][0] - P[0][0] || 1), P[0][1]);
          }, TELE, 1, true);
          P.forEach(handle);
          chip((P[0][0] + P[1][0]) / 2 + 8, (P[0][1] + P[1][1]) / 2 - 6, [`${Math.abs(ang).toFixed(1)}° to horizontal`]);
          break;
        }
        case "arrow": {
          if (P.length < 2) break;
          const [a, b] = P;
          const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
          const ux = (b[0] - a[0]) / L, uy = (b[1] - a[1]) / L;
          const hs = Math.min(16, L * 0.4);
          haloStroke(() => {
            c.moveTo(a[0], a[1]);
            c.lineTo(b[0], b[1]);
            c.moveTo(b[0] - ux * hs - uy * hs * 0.55, b[1] - uy * hs + ux * hs * 0.55);
            c.lineTo(b[0], b[1]);
            c.lineTo(b[0] - ux * hs + uy * hs * 0.55, b[1] - uy * hs - ux * hs * 0.55);
          }, TELE, 2.8, dash);
          break;
        }
        case "circle": {
          if (P.length < 2) break;
          const r = Math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]);
          haloStroke(() => c.arc(P[0][0], P[0][1], r, 0, Math.PI * 2), TELE, 2.5, dash);
          break;
        }
        case "spotlight": {
          if (P.length < 2) break;
          const r = Math.hypot(P[1][0] - P[0][0], P[1][1] - P[0][1]);
          haloStroke(() => c.arc(P[0][0], P[0][1], r, 0, Math.PI * 2), TELE, 1.5, true);
          break;
        }
        case "freehand": {
          if (P.length < 2) break;
          haloStroke(() => {
            c.moveTo(P[0][0], P[0][1]);
            for (let i = 1; i < P.length - 1; i++) {
              const mx = (P[i][0] + P[i + 1][0]) / 2, my = (P[i][1] + P[i + 1][1]) / 2;
              c.quadraticCurveTo(P[i][0], P[i][1], mx, my);
            }
            c.lineTo(P[P.length - 1][0], P[P.length - 1][1]);
          }, TELE, 2.6, false);
          break;
        }
        case "text": {
          const p = P[0];
          handle(p);
          chip(p[0] + 8, p[1] - 4, [meta.label || (isDraft ? "Type a note…" : "")]);
          break;
        }
        case "ruler": {
          if (P.length < 2) break;
          const [a, b] = P;
          const L = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
          const nx = -(b[1] - a[1]) / L, ny = (b[0] - a[0]) / L;
          haloStroke(() => {
            c.moveTo(a[0], a[1]);
            c.lineTo(b[0], b[1]);
            for (const q of [a, b]) {
              c.moveTo(q[0] - nx * 7, q[1] - ny * 7);
              c.lineTo(q[0] + nx * 7, q[1] + ny * 7);
            }
          }, TELE, 2.2, dash);
          const label = rulerLabel(ptsV.slice(0, 2), meta.snap_joints, rf);
          chip((a[0] + b[0]) / 2 + 10, (a[1] + b[1]) / 2 - 8, [label]);
          break;
        }
        case "plumb": {
          const vert = verticalAt(rf);
          const p0 = P[0];
          // Image-space direction is the same in screen space (uniform zoom); extend across the frame.
          const ext = Math.max(W, H) * 1.5;
          haloStroke(() => {
            c.moveTo(p0[0] - vert.dir[0] * ext, p0[1] - vert.dir[1] * ext);
            c.lineTo(p0[0] + vert.dir[0] * ext, p0[1] + vert.dir[1] * ext);
          }, TELE, 1.8, !vert.calibrated);
          handle(p0);
          const lines: string[] = [];
          if (P.length >= 2 && Math.hypot(P[1][0] - p0[0], P[1][1] - p0[1]) > 8) {
            const dx = ptsV[1][0] - ptsV[0][0], dy = ptsV[1][1] - ptsV[0][1];
            const L = Math.hypot(dx, dy) || 1;
            const cosv = Math.abs((dx * vert.dir[0] + dy * vert.dir[1]) / L);
            const lean = (Math.acos(clamp(cosv, -1, 1)) * 180) / Math.PI;
            haloStroke(() => {
              c.moveTo(p0[0], p0[1]);
              c.lineTo(P[1][0], P[1][1]);
            }, TELE, 2.5, dash);
            handle(P[1]);
            lines.push(`Lean ${lean.toFixed(1)}° from ${vert.calibrated ? "true vertical" : "image vertical"} (2D)`);
          } else {
            lines.push(vert.calibrated ? "True vertical" : "Image vertical");
          }
          const gs = factSheet.calibration?.gravity_sigma_deg;
          lines.push(vert.calibrated ? `${vert.source}${Number.isFinite(gs) ? ` · ±${(gs as number).toFixed(1)}°` : ""}` : vert.source);
          chip(p0[0] + 10, p0[1] - 10, lines);
          break;
        }
        case "angle": {
          if (P.length < 2) {
            handle(P[0]);
            break;
          }
          haloStroke(() => {
            c.moveTo(P[0][0], P[0][1]);
            c.lineTo(P[1][0], P[1][1]);
            if (P.length >= 3) c.lineTo(P[2][0], P[2][1]);
          }, TELE, 2.5, dash);
          P.forEach(handle);
          if (P.length >= 3) {
            drawAngleArc(c, P[0], P[1], P[2], TELE, false);
            const d2 = angle2d(ptsV[0], ptsV[1], ptsV[2]);
            const lines = [`Drawn (2D) ${fmtDeg(d2, 1)}`];
            const sj = meta.snap_joints;
            if (sj && sj.length === 3 && sj.every((j) => j >= 0) && new Set(sj).size === 3) {
              const a3 = angle3dAt(rf, [sj[0], sj[1], sj[2]]);
              if (a3) lines.push(`Measured (3D) ${fmtDeg(a3.deg)}  likely ${a3.lo.toFixed(0)}–${a3.hi.toFixed(0)}°`);
              const bw = bowlingArm === "right" ? ["rsho", "relb", "rwri"] : ["lsho", "lelb", "lwri"];
              if (SMPL24_JOINTS[sj[1]] === bw[1]) lines.push("Visual trace only — not ICC legality");
            } else {
              lines.push("Snap all 3 points to joints for the 3D value");
            }
            chip(P[1][0] + 12, P[1][1] - 10, lines);
          }
          break;
        }
      }
    }
  };
  drawRef.current = draw;

  // Redraw after every render when not playing (frame, layers, shapes, size, selection…).
  useEffect(() => {
    if (!playing) drawRef.current();
  });

  // ── Video sync ───────────────────────────────────────────────────────────
  const requestSeek = useCallback(
    (fi: number) => {
      const v = videoRef.current;
      if (!v || videoState !== "ready" || v.readyState < 1) {
        drawRef.current();
        return;
      }
      const t = videoTimeOf(fi);
      const curF = frameOfVideoTime(v.currentTime);
      if (!seekingRef.current && !v.seeking && curF === fi && Math.abs(v.currentTime - t) < 0.45 / proxyFps) {
        lastDecodedVidFrameRef.current = curF;
        drawRef.current();
        return;
      }
      if (seekingRef.current || v.seeking) {
        pendingSeekRef.current = fi;
        return;
      }
      seekingRef.current = true;
      pendingSeekRef.current = null;
      try {
        v.currentTime = t;
      } catch {
        seekingRef.current = false;
      }
    },
    [videoState, videoTimeOf, frameOfVideoTime, proxyFps]
  );

  const onSeeked = useCallback(() => {
    seekingRef.current = false;
    const v = videoRef.current;
    if (v && v.readyState >= 2) {
      lastDecodedVidFrameRef.current = frameOfVideoTime(v.currentTime);
    }
    const p = pendingSeekRef.current;
    if (p != null) {
      pendingSeekRef.current = null;
      requestSeek(p);
      return;
    }
    drawRef.current();
  }, [requestSeek, frameOfVideoTime]);

  useEffect(() => {
    seekingRef.current = false;
    pendingSeekRef.current = null;
    lastSyncedVidFrameRef.current = -1;
    setVideoState("loading");
  }, [videoUrl]);

  const vRate = (fps * rate) / proxyFps;
  const nativeOk = playing && direction > 0 && vRate >= 0.0625 && vRate <= 16 && videoState === "ready";
  nativeOkRef.current = nativeOk;

  // Native playback on/off
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (nativeOk) {
      v.playbackRate = vRate;
      lastSyncedVidFrameRef.current = -1;
      const target = videoTimeOf(frameRef.current);
      if (frameOfVideoTime(v.currentTime) !== frameRef.current && Math.abs(v.currentTime - target) > 0.45 / proxyFps) {
        seekingRef.current = true;
        try {
          v.currentTime = target;
        } catch {
          seekingRef.current = false;
        }
      }
      v.play().catch(() => {
        /* autoplay blocked: fall back to per-frame seeks */
      });
    } else {
      if (!v.paused) v.pause();
      // Snap video cleanly to the paused playhead frame if needed.
      if (videoState === "ready") requestSeek(frameRef.current);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [nativeOk, vRate]);

  // Every playhead change: seek (paused / reverse) or handle discontinuous jumps / loop wraps (native).
  useEffect(() => {
    const prev = prevPlayheadFrameRef.current;
    prevPlayheadFrameRef.current = f;
    const v = videoRef.current;
    if (nativeOk && v) {
      const target = videoTimeOf(f);
      const jumped = f < prev || Math.abs(f - prev) > 2;
      if (jumped || (!seekingRef.current && !v.seeking && Math.abs(v.currentTime - target) > 3 / proxyFps)) {
        requestSeek(f);
      }
      if (v.ended) {
        v.play().catch(() => undefined);
      }
    } else {
      requestSeek(f);
    }
  }, [f, nativeOk, requestSeek, videoTimeOf, proxyFps]);

  // Continuous compositing while playing (the decoded video advances between playhead ticks).
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const loop = () => {
      drawRef.current();
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // ── Wheel zoom (pinch on trackpads sends ctrlKey) ─────────────────────────
  const zoomAround = useCallback(
    (sx: number, sy: number, z1: number) => {
      const g = geo();
      const z = clamp(z1, ZMIN, ZMAX);
      const [vx, vy] = g.toV(sx, sy);
      const panX = sx - g.W / 2 - (g.ox + vx * g.s0 - g.W / 2) * z;
      const panY = sy - g.H / 2 - (g.oy + vy * g.s0 - g.H / 2) * z;
      pendingPanRef.current = [panX, panY];
      if (Math.abs(z - zoom) < 1e-3) {
        panRef.current = clampPan([panX, panY], z);
        pendingPanRef.current = null;
        drawRef.current();
      } else onZoomChange(Math.round(z * 100) / 100);
    },
    [geo, zoom, onZoomChange, clampPan]
  );

  useEffect(() => {
    const el = canvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (!(e.ctrlKey || e.metaKey || e.altKey || zoom > 1.001)) return;
      e.preventDefault();
      const r = el.getBoundingClientRect();
      zoomAround(e.clientX - r.left, e.clientY - r.top, zoom * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)));
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [zoom, zoomAround]);

  // ── Pointer interaction ───────────────────────────────────────────────────
  const evtScreen = (e: React.PointerEvent): P2 => {
    const r = canvasRef.current!.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };
  const toVideoClamped = (s: P2): P2 => {
    const [x, y] = geo().toV(s[0], s[1]);
    return [Number(clamp(x, 0, vw).toFixed(1)), Number(clamp(y, 0, vh).toFixed(1))];
  };

  const commit = (sh: Omit<TelestratorShape, "id" | "frame">) => {
    const full: TelestratorShape = { id: newId(), frame: f, ...sh };
    if (pinNewShapes) full.pinned_all_frames = true;
    onShapesChange([...shapes, full]);
  };

  const snapFor = (p: P2, tool: TelestratorTool): { pt: P2; joint: number | null } => {
    if (tool !== "angle" && tool !== "ruler") return { pt: p, joint: null };
    const j = nearestJoint(p, SNAP_PX);
    if (j == null) return { pt: p, joint: null };
    const q = jointPosAt(f, j);
    return q ? { pt: q, joint: j } : { pt: p, joint: null };
  };

  const minimapHit = (s: P2): P2 | null => {
    if (zoom <= 1.2) return null;
    const W = size.w;
    const mw = Math.min(150, W * 0.28);
    const mh = (mw * vh) / vw;
    const mx = W - mw - 10, my = 10;
    if (s[0] < mx || s[0] > mx + mw || s[1] < my || s[1] > my + mh) return null;
    return [((s[0] - mx) / mw) * vw, ((s[1] - my) / mh) * vh];
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (textPrompt) return;
    const s = evtScreen(e);
    pointersRef.current.set(e.pointerId, s);
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (pointersRef.current.size === 2) {
      const [a, b] = [...pointersRef.current.values()];
      pinchRef.current = { d0: Math.hypot(a[0] - b[0], a[1] - b[1]) || 1, z0: zoom };
      draftRef.current = null;
      dragRef.current = null;
      return;
    }
    const mm = minimapHit(s);
    if (mm) {
      const g = geo();
      const panX = -(g.ox + mm[0] * g.s0 - g.W / 2) * zoom;
      const panY = -(g.oy + mm[1] * g.s0 - g.H / 2) * zoom;
      panRef.current = clampPan([panX, panY], zoom);
      drawRef.current();
      return;
    }
    const v = toVideoClamped(s);
    if (activeTool === "cursor") {
      dragRef.current = { kind: zoom > 1.001 ? "pan" : "tap", startScreen: s, startPan: [...panRef.current] as P2, moved: false };
      return;
    }
    if (!layers.telestration) return;
    setPlaying(false);
    if (activeTool === "text") {
      setTextPrompt({ pt: v, value: "" });
      return;
    }
    if (activeTool === "angle") {
      const sn = snapFor(v, "angle");
      const d = draftRef.current && draftRef.current.tool === "angle" ? draftRef.current : { tool: "angle" as TelestratorTool, pts: [], snaps: [] };
      d.pts = [...d.pts, sn.pt];
      d.snaps = [...d.snaps, sn.joint];
      if (d.pts.length >= 3) {
        const snaps = d.snaps.map((j) => (j == null ? -1 : j));
        commit({ type: "angle", points: d.pts.slice(0, 3), ...(snaps.some((j) => j >= 0) ? { snap_joints: snaps.slice(0, 3) } : {}) });
        draftRef.current = null;
      } else draftRef.current = d;
      setDraftTick((t) => t + 1);
      drawRef.current();
      return;
    }
    const sn = snapFor(v, activeTool);
    draftRef.current = { tool: activeTool, pts: [sn.pt], snaps: [sn.joint] };
    dragRef.current = { kind: "draw", startScreen: s, startPan: [...panRef.current] as P2, moved: false };
    hoverRef.current = v;
    drawRef.current();
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const s = evtScreen(e);
    if (pointersRef.current.has(e.pointerId)) pointersRef.current.set(e.pointerId, s);
    if (pinchRef.current && pointersRef.current.size === 2) {
      const [a, b] = [...pointersRef.current.values()];
      const d = Math.hypot(a[0] - b[0], a[1] - b[1]) || 1;
      zoomAround((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, pinchRef.current.z0 * (d / pinchRef.current.d0));
      return;
    }
    const v = toVideoClamped(s);
    hoverRef.current = v;
    const dr = dragRef.current;
    if (dr) {
      if (Math.hypot(s[0] - dr.startScreen[0], s[1] - dr.startScreen[1]) > 4) dr.moved = true;
      if (dr.kind === "pan" && dr.moved) {
        panRef.current = clampPan([dr.startPan[0] + s[0] - dr.startScreen[0], dr.startPan[1] + s[1] - dr.startScreen[1]], zoom);
        drawRef.current();
        return;
      }
      if (dr.kind === "draw" && draftRef.current?.tool === "freehand") {
        const pts = draftRef.current.pts;
        const last = pts[pts.length - 1];
        if (Math.hypot(v[0] - last[0], v[1] - last[1]) * geo().scale > 2.5) pts.push(v);
      }
    }
    if (draftRef.current) drawRef.current();
  };

  const onPointerUp = (e: React.PointerEvent<HTMLCanvasElement>) => {
    pointersRef.current.delete(e.pointerId);
    if (pinchRef.current) {
      if (pointersRef.current.size < 2) pinchRef.current = null;
      return;
    }
    const dr = dragRef.current;
    dragRef.current = null;
    const s = evtScreen(e);
    const v = toVideoClamped(s);
    if (!dr) return;
    if (dr.kind === "tap" || (dr.kind === "pan" && !dr.moved)) {
      if (activeTool === "cursor") {
        const j = nearestJoint(v, PICK_PX);
        setSelectedJoint((prev) => (j == null ? null : prev === j ? null : j));
      }
      return;
    }
    if (dr.kind !== "draw") return;
    const d = draftRef.current;
    draftRef.current = null;
    if (!d) return;
    const tool = d.tool;
    if (tool === "freehand") {
      if (d.pts.length >= 2) commit({ type: "freehand", points: d.pts });
      drawRef.current();
      return;
    }
    const end = snapFor(v, tool);
    const start = d.pts[0];
    const sS = geo().toS(start[0], start[1]);
    const eS = geo().toS(end.pt[0], end.pt[1]);
    const lenScreen = Math.hypot(eS[0] - sS[0], eS[1] - sS[1]);
    if (tool === "plumb") {
      commit({ type: "plumb", points: lenScreen > 8 ? [start, end.pt] : [start, start] });
    } else if (lenScreen >= 6 && tool !== "cursor" && tool !== "angle" && tool !== "text") {
      const snaps = [d.snaps[0] ?? -1, end.joint ?? -1];
      commit({ type: tool, points: [start, end.pt], ...(tool === "ruler" && snaps.some((j) => j >= 0) ? { snap_joints: snaps } : {}) });
    }
    drawRef.current();
  };

  const onPointerLeave = () => {
    hoverRef.current = null;
    if (!dragRef.current && draftRef.current?.tool !== "angle") drawRef.current();
  };

  // Escape cancels drafts / selection.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      draftRef.current = null;
      dragRef.current = null;
      setTextPrompt(null);
      setSelectedJoint(null);
      setDraftTick((t) => t + 1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  // Tool change clears half-finished drafts.
  useEffect(() => {
    draftRef.current = null;
    setDraftTick((t) => t + 1);
  }, [activeTool]);

  // ── Selected-joint HUD chip (DOM) ─────────────────────────────────────────
  const chipData = useMemo(() => {
    if (selectedJoint == null || !layers.angleChips) return null;
    const name = SMPL24_JOINTS[selectedJoint];
    const def = angleDefFor(selectedJoint, bowlingArm);
    const p = jointPosAt(f, selectedJoint);
    if (!p) return null;
    if (!def) return { p, title: JOINT_PLAIN[name] ?? name, rows: [] as Array<[string, string]>, note: "No single joint angle here — tap a knee, hip, elbow or shoulder.", measured: null as null | { m: MetricEstimate; label: string; frame: number }, elbow: false };
    const P2s = def.tri.map((j) => jointPosAt(f, j));
    const d2 = P2s.every(Boolean) ? angle2d(P2s[0]!, P2s[1]!, P2s[2]!) : NaN;
    const a3 = angle3dAt(f, def.tri);
    const kin = def.kinKey ? telemetry.kin_series?.[def.kinKey]?.[f] : undefined;
    const rows: Array<[string, string]> = [["Drawn (2D)", fmtDeg(d2)]];
    if (a3) rows.push(["3D pose", `${fmtDeg(Number.isFinite(kin as number) ? (kin as number) : a3.deg)}  likely ${a3.lo.toFixed(0)}–${a3.hi.toFixed(0)}°`]);
    // Measured fact-sheet value (with CI) at the nearest relevant event.
    let measured: { m: MetricEstimate; label: string; frame: number } | null = null;
    if (def.metrics) {
      let bestD = Infinity;
      for (const mm of def.metrics) {
        const m = factSheet.metrics?.[mm.key];
        const ev = factSheet.events?.[mm.event];
        if (!m || m.suppressed || m.value == null || !ev) continue;
        const d = Math.abs(ev.frame_idx - f);
        if (d < bestD) {
          bestD = d;
          measured = { m, label: mm.eventLabel, frame: ev.frame_idx };
        }
      }
    }
    return { p, title: def.label, rows, note: null as string | null, measured, elbow: !!def.isBowlingElbow };
  }, [selectedJoint, layers.angleChips, bowlingArm, jointPosAt, f, angle3dAt, telemetry.kin_series, factSheet]);

  const chipScreen = chipData ? geo().toS(chipData.p[0], chipData.p[1]) : null;
  const promptScreen = textPrompt ? geo().toS(textPrompt.pt[0], textPrompt.pt[1]) : null;
  void draftTick;

  const cursor =
    activeTool === "cursor" ? (zoom > 1.001 ? "grab" : "default") : activeTool === "text" ? "text" : layers.telestration ? "crosshair" : "not-allowed";

  const toolHint: Partial<Record<TelestratorTool, string>> = {
    line: "Drag a line · shows its angle to horizontal",
    angle: "Tap 3 points · snaps to joints within 22 px for the 3D value",
    plumb: "Tap for true vertical · drag to read lean",
    circle: "Drag from the centre",
    arrow: "Drag a direction cue",
    freehand: "Draw freely",
    ruler: isMetric ? "Drag · snap both ends to joints for 3D metres" : "Drag · this clip is not metric-calibrated, so pixels only",
    text: "Tap to place a note",
    spotlight: "Drag a circle to spotlight",
  };

  return (
    <div
      ref={wrapRef}
      style={{ ["--video-ar" as string]: `${vw} / ${vh}` }}
      className="relative w-full [aspect-ratio:var(--video-ar)] max-h-[58vh] md:[aspect-ratio:auto] md:max-h-none md:h-full md:min-h-[300px] overflow-hidden rounded-[6px] bg-[#f4f4f5] select-none"
    >
      <video
        ref={videoRef}
        src={videoUrl}
        muted
        playsInline
        preload="auto"
        aria-hidden
        tabIndex={-1}
        style={{ position: "absolute", width: 1, height: 1, opacity: 0, pointerEvents: "none", left: 0, top: 0 }}
        onLoadedData={() => {
          setVideoState("ready");
          const v = videoRef.current;
          if (v && v.readyState >= 1) {
            const target = videoTimeOf(frameRef.current);
            if (frameOfVideoTime(v.currentTime) !== frameRef.current) {
              seekingRef.current = true;
              try {
                v.currentTime = target;
              } catch {
                seekingRef.current = false;
              }
            } else {
              lastDecodedVidFrameRef.current = frameRef.current;
            }
          }
          requestAnimationFrame(() => drawRef.current());
        }}
        onError={() => setVideoState("error")}
        onSeeked={onSeeked}
        onEnded={() => {
          /* the playhead owns looping */
        }}
      />
      <canvas
        ref={canvasRef}
        width={Math.max(1, Math.round(size.w * size.dpr))}
        height={Math.max(1, Math.round(size.h * size.dpr))}
        style={{ width: size.w || "100%", height: size.h || "100%", cursor, touchAction: "none", display: "block" }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onPointerLeave={onPointerLeave}
        aria-label="Delivery video with skeleton overlay"
        role="img"
      />

      {/* Tap-a-joint HUD chip */}
      {chipData && chipScreen && (
        <div
          className="absolute z-10 w-[min(280px,calc(100%-16px))] rounded-[6px] bg-white/97 shadow-[0_0_0_1px_rgba(0,0,0,0.1),0_6px_18px_-8px_rgba(0,0,0,0.25)] px-2.5 py-2"
          style={{
            left: clamp(chipScreen[0] + 16, 8, Math.max(8, size.w - 290)),
            top: clamp(chipScreen[1] - 20, 8, Math.max(8, size.h - 170)),
          }}
        >
          <div className="flex items-center justify-between gap-2">
            <span className="v-label !text-[#171717]">{chipData.title}</span>
            <button type="button" className="font-mono text-[11px] text-[#8f8f8f] hover:text-[#171717]" onClick={() => setSelectedJoint(null)} aria-label="Close">
              ✕
            </button>
          </div>
          {chipData.rows.map(([k, v]) => (
            <div key={k} className="mt-1 flex items-baseline justify-between gap-3 font-mono text-[11.5px] tabular-nums">
              <span className="text-[#8f8f8f]">{k}</span>
              <span className="text-[#171717]">{v}</span>
            </div>
          ))}
          {chipData.measured && (
            <div className="mt-1.5 pt-1.5 border-t border-[#ebebeb]">
              <div className="flex items-baseline justify-between gap-2 font-mono text-[11.5px] tabular-nums">
                <span className="text-[#4d4d4d]">Measured · {chipData.measured.label}</span>
                <span className="text-[#171717]">
                  {fmtDeg(chipData.measured.m.value)}
                  {chipData.measured.m.ci_low != null && chipData.measured.m.ci_high != null
                    ? ` [${chipData.measured.m.ci_low.toFixed(0)}–${chipData.measured.m.ci_high.toFixed(0)}]`
                    : ""}{" "}
                  <span className="text-[#8f8f8f]" title={OBS_GLYPH[chipData.measured.m.observability]?.word}>
                    {OBS_GLYPH[chipData.measured.m.observability]?.bars}
                  </span>
                </span>
              </div>
              {chipData.measured.frame !== f && (
                <button
                  type="button"
                  onClick={() => ph.seekToEventFrame(chipData.measured!.frame)}
                  className="mt-1 font-mono text-[10.5px] text-[#171717] underline underline-offset-2 decoration-[#c9c9c9] hover:decoration-[#171717]"
                >
                  See it at {chipData.measured.label} →
                </button>
              )}
            </div>
          )}
          {chipData.note && <p className="mt-1 text-[11.5px] leading-snug text-[#4d4d4d]">{chipData.note}</p>}
          {!chipData.note && (
            <p className="mt-1.5 text-[10.5px] leading-snug text-[#8f8f8f]">
              2D is the on-screen angle; 3D is the pose model's angle on this frame with a propagated uncertainty range.
            </p>
          )}
          {chipData.elbow && (
            <p className="mt-1.5 px-1.5 py-1 rounded-[4px] bg-[#f4f4f5] text-[11px] leading-snug text-[#171717] v-guardrail">
              🔒 Visual trace only — not ICC legality. Elbow legality (ICC 15° rule) needs an accredited laboratory 3D test at 250 Hz or faster.
            </p>
          )}
        </div>
      )}

      {/* Inline text prompt */}
      {textPrompt && promptScreen && (
        <form
          className="absolute z-20 flex items-center gap-1 rounded-[6px] bg-white p-1 shadow-[0_0_0_1px_#C026D3,0_6px_18px_-8px_rgba(0,0,0,0.3)]"
          style={{ left: clamp(promptScreen[0] + 8, 8, Math.max(8, size.w - 250)), top: clamp(promptScreen[1] - 18, 8, Math.max(8, size.h - 44)) }}
          onSubmit={(e) => {
            e.preventDefault();
            const val = textPrompt.value.trim();
            if (val) commit({ type: "text", points: [textPrompt.pt], label: val.slice(0, 140) });
            setTextPrompt(null);
          }}
        >
          <input
            autoFocus
            value={textPrompt.value}
            onChange={(e) => setTextPrompt((t) => (t ? { ...t, value: e.target.value } : t))}
            onKeyDown={(e) => {
              if (e.key === "Escape") {
                e.stopPropagation();
                setTextPrompt(null);
              }
            }}
            placeholder="Note on this frame…"
            className="v-input h-7 w-[180px] px-2 text-[12px]"
            maxLength={140}
          />
          <button type="submit" className="v-btn-primary h-7 px-2 text-[11px]">
            Add
          </button>
        </form>
      )}

      {/* Status strip: tool hint / zoom badge */}
      <div className="pointer-events-none absolute left-2 top-2 flex flex-wrap items-center gap-1.5">
        {activeTool !== "cursor" && layers.telestration && toolHint[activeTool] && (
          <span className="inline-flex items-center gap-1.5 h-[22px] px-2 rounded-full bg-white/95 shadow-[0_0_0_1px_#ebebeb] font-mono text-[10.5px] text-[#171717]">
            <span className="w-1.5 h-1.5 rounded-full" style={{ background: TELE }} />
            {toolHint[activeTool]}
          </span>
        )}
        {zoom > 1.001 && (
          <span className="inline-flex items-center h-[22px] px-2 rounded-full bg-[#171717] font-mono text-[10.5px] text-white tabular-nums">
            {zoom.toFixed(1)}× · drag to pan
          </span>
        )}
        {videoState === "error" && (
          <span className="inline-flex items-center h-[22px] px-2 rounded-full bg-white/95 shadow-[0_0_0_1px_#ebebeb] font-mono text-[10.5px] text-[#4d4d4d]">
            Video unavailable · showing still frames
          </span>
        )}
      </div>

      {activeCaption && (
        <div className="pointer-events-none absolute inset-x-0 bottom-3 flex justify-center px-3">
          <span className="max-w-[min(640px,100%)] rounded-[6px] bg-white/95 px-3 py-1.5 text-center text-[13px] leading-snug text-[#171717] shadow-[0_0_0_1px_rgba(0,0,0,0.08)]">
            {activeCaption}
          </span>
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Canvas helpers
// ─────────────────────────────────────────────────────────────────────────────

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawAngleArc(ctx: CanvasRenderingContext2D, A: P2, B: P2, C: P2, color: string, dashed: boolean) {
  const a1 = Math.atan2(A[1] - B[1], A[0] - B[0]);
  const a2 = Math.atan2(C[1] - B[1], C[0] - B[0]);
  let d = a2 - a1;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  const r = clamp(Math.min(Math.hypot(A[0] - B[0], A[1] - B[1]), Math.hypot(C[0] - B[0], C[1] - B[1])) * 0.35, 14, 34);
  ctx.save();
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.arc(B[0], B[1], r, a1, a1 + d, d < 0);
  ctx.strokeStyle = "rgba(255,255,255,0.95)";
  ctx.lineWidth = 5.5;
  ctx.stroke();
  ctx.beginPath();
  ctx.arc(B[0], B[1], r, a1, a1 + d, d < 0);
  if (dashed) ctx.setLineDash([4, 3]);
  ctx.strokeStyle = color;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

type ToS = (x: number, y: number) => P2;

function drawWristTrail(ctx: CanvasRenderingContext2D, toS: ToS, tr: Trail2D, f: number, tail: number) {
  const segs: Array<[P2, P2, number]> = [];
  for (let k = 1; k < tr.frames.length; k++) {
    const fa = tr.frames[k - 1], fb = tr.frames[k];
    if (fb > f || fb < f - tail) continue;
    const a = tr.pts[k - 1], b = tr.pts[k];
    if (!a || !b || fb - fa > 3) continue;
    const age = (f - fb) / Math.max(1, tail); // 0 = newest
    segs.push([toS(a[0], a[1]), toS(b[0], b[1]), 1 - age]);
  }
  if (!segs.length) return;
  ctx.save();
  ctx.lineCap = "round";
  for (const [a, b, t] of segs) {
    ctx.globalAlpha = 0.85 * t;
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1.5 + 3.2 * t + 4;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  for (const [a, b, t] of segs) {
    ctx.globalAlpha = Math.pow(t, 1.1);
    ctx.strokeStyle = SEG.wrist;
    ctx.lineWidth = 1.5 + 3.2 * t;
    ctx.beginPath();
    ctx.moveTo(a[0], a[1]);
    ctx.lineTo(b[0], b[1]);
    ctx.stroke();
  }
  ctx.restore();
}

function polyUpTo(toS: ToS, tr: Trail2D, f: number): P2[][] {
  const runs: P2[][] = [];
  let cur: P2[] = [];
  for (let k = 0; k < tr.frames.length; k++) {
    if (tr.frames[k] > f) break;
    const p = tr.pts[k];
    if (!p || (k > 0 && tr.frames[k] - tr.frames[k - 1] > 3)) {
      if (cur.length > 1) runs.push(cur);
      cur = [];
      if (!p) continue;
    }
    cur.push(toS(p[0], p[1]));
  }
  if (cur.length > 1) runs.push(cur);
  return runs;
}

function drawHeadTrail(ctx: CanvasRenderingContext2D, toS: ToS, tr: Trail2D, f: number) {
  const runs = polyUpTo(toS, tr, f);
  if (!runs.length) return;
  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  for (const pass of [0, 1]) {
    ctx.setLineDash(pass === 0 ? [] : [6, 5]);
    ctx.strokeStyle = pass === 0 ? "rgba(255,255,255,0.9)" : SEG.head;
    ctx.lineWidth = pass === 0 ? 5 : 1.8;
    for (const run of runs) {
      ctx.beginPath();
      ctx.moveTo(run[0][0], run[0][1]);
      for (const p of run.slice(1)) ctx.lineTo(p[0], p[1]);
      ctx.stroke();
    }
  }
  ctx.restore();
}

function drawComTrail(ctx: CanvasRenderingContext2D, toS: ToS, tr: Trail2D, f: number) {
  const runs = polyUpTo(toS, tr, f);
  ctx.save();
  ctx.lineCap = "round";
  for (const run of runs) {
    ctx.strokeStyle = "rgba(255,255,255,0.85)";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(run[0][0], run[0][1]);
    for (const p of run.slice(1)) ctx.lineTo(p[0], p[1]);
    ctx.stroke();
    ctx.strokeStyle = SEG.com;
    ctx.lineWidth = 1.4;
    ctx.stroke();
  }
  // Crosshair at the current COM
  let cur: P2 | null = null;
  for (let k = 0; k < tr.frames.length; k++) {
    if (tr.frames[k] === f && tr.pts[k]) cur = tr.pts[k];
  }
  if (cur) {
    const [x, y] = toS(cur[0], cur[1]);
    for (const pass of [0, 1]) {
      ctx.strokeStyle = pass === 0 ? "rgba(255,255,255,0.95)" : SEG.com;
      ctx.lineWidth = pass === 0 ? 4.5 : 1.6;
      ctx.beginPath();
      ctx.moveTo(x - 9, y);
      ctx.lineTo(x + 9, y);
      ctx.moveTo(x, y - 9);
      ctx.lineTo(x, y + 9);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.font = FONT_SM;
    ctx.fillStyle = SEG.com;
    ctx.fillText("COM", x + 9, y - 8);
  }
  ctx.restore();
}

export default VideoStage;
