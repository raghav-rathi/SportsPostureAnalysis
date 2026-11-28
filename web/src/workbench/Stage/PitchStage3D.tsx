/**
 * PitchStage3D: studio-lit, sculpted 3D mannequin on an MCC-accurate pitch (docs/research_app_landscape_v5.md §6.6, §8.2).
 *
 * - Sculpted (NOT stick) mannequin: tapered limb capsules, ellipsoid ribcage / abdomen / pelvis blocks, head with a visor
 *   orientation cue and joint spheres, tinted by segment (theme.ts SEG) under a soft studio key + fill + hemisphere rig.
 * - MCC scene: paper ground, pale turf strip (#e7efe9), Law 7 creases (popping = #297a3a; bowling & return = #171717),
 *   true 3D Law 8 stumps (0.7112 m, outer stumps centred ±0.096 m) + bails at both ends, real shadow + soft contact discs,
 *   faint reference grid in Top-down only.
 * - Six virtual cameras with a 600 ms cubic ease (OrbitControls always live): Match (the calibrated phone position, with a
 *   distance-matched field of view), Side-on, Behind, Front, Top-down, Arm close-up.
 * - Translucent 1-sigma joint uncertainty spheres, 3D trajectory ribbons (wrist / head / COM) revealed up to the playhead.
 * - FFC-time-interpolated slate ghost (pelvis-aligned at FFC, optional mirror across the pitch centre-line) and a 6-pose
 *   strobe (BFC → FT) with opacity rising toward ball release.
 *
 * World frame (src/bowling_biomech/constants.py): +X down the pitch toward the striker, +Y up, +Z bowler's right.
 * `joints3d_world_m` is ground-zeroed by the backend; `camera_position_world_m` is not (we subtract `y_ground_offset_m`).
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree, type ThreeEvent } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import * as THREE from "three";
import type { TelemetryV2 } from "../../types";
import { GHOST_OPACITY, J, SEG, SEG_COLOR, SMPL24_BONES, SMPL24_JOINTS, boneSegment, jointSegment } from "../../theme";
import { usePlayhead } from "../PlayheadProvider";

export type VirtualCameraPreset = "match" | "side" | "behind" | "front" | "top" | "arm";

export interface PitchStage3DProps {
  telemetry: TelemetryV2;
  bowlingArm?: "right" | "left";
  frame?: number; // falls back to usePlayhead() if omitted
  ghostJoints3d?: number[][][]; // optional (T,24,3) ghost skeleton
  ghostTimeRelFfcMs?: number[];
  ghostLabel?: string;
  mirrorGhost?: boolean;
  defaultPreset?: VirtualCameraPreset;
  showUncertainty?: boolean;
  showTrails?: { wrist?: boolean; head?: boolean; com?: boolean };
  strobeMode?: boolean; // 6 semi-transparent sculpted mannequins from BFC→FT
  idleOrbit?: boolean; // slow calm idle orbit for Home hero ONLY
  compact?: boolean; // hides top camera bar
  heightClass?: string;
  onSelectJoint?: (jointIdx: number) => void;
  // ── Extra optional props (additive; safe to omit) ──
  /** Event frames (telemetry frame indices) — used for strobe spacing, Arm close-up target and FFC alignment. */
  eventFrames?: { bfc?: number | null; ffc?: number | null; uah?: number | null; br?: number | null; ft?: number | null };
  /** Explicit strobe frames (overrides the BFC→FT spacing). */
  strobeFrames?: number[];
  /** Controlled preset (if provided, the internal bar follows it). */
  preset?: VirtualCameraPreset;
  onPresetChange?: (p: VirtualCameraPreset) => void;
}

type Pose = number[][]; // (24, 3)
type PoseFn = () => Pose | null;

export const CAMERA_PRESET_LABELS: Array<[VirtualCameraPreset, string]> = [
  ["match", "Match"],
  ["side", "Side-on"],
  ["behind", "Behind"],
  ["front", "Front"],
  ["top", "Top"],
  ["arm", "Arm"],
];

// ── MCC geometry (Laws 6–8) ─────────────────────────────────────────────────
const PITCH_LEN = 20.12;
const POPPING = 1.22;
const RETURN_HALF = 1.32;
const POPPING_HALF = 1.83;
const PITCH_HALF_WIDTH = 1.525;
const STUMP_H = 0.7112;
const STUMP_Z = 0.096;
const STUMP_R = 0.018;
const BAIL_LEN = 0.1095;
const BAIL_R = 0.0085;

const DEFAULT_FOV = 38;
const TRANSITION_MS = 600;

// ── Mannequin proportions (metres) ──────────────────────────────────────────
/** Bone radii keyed by the CHILD joint name: [radius at parent end, radius at child end]. */
const BONE_RADII: Partial<Record<string, [number, number]>> = {
  lhip: [0.055, 0.06], rhip: [0.055, 0.06],
  spi1: [0.07, 0.075], spi2: [0.075, 0.08], spi3: [0.08, 0.08],
  lkne: [0.074, 0.052], rkne: [0.074, 0.052],
  lank: [0.052, 0.036], rank: [0.052, 0.036],
  ltoe: [0.038, 0.026], rtoe: [0.038, 0.026],
  neck: [0.05, 0.046],
  lcla: [0.042, 0.046], rcla: [0.042, 0.046],
  head: [0.046, 0.05],
  lsho: [0.046, 0.052], rsho: [0.046, 0.052],
  lelb: [0.05, 0.04], relb: [0.05, 0.04],
  lwri: [0.04, 0.03], rwri: [0.04, 0.03],
  lhan: [0.03, 0.024], rhan: [0.03, 0.024],
};
const JOINT_RADII: Partial<Record<string, number>> = {
  pelv: 0.05, lhip: 0.062, rhip: 0.062, lkne: 0.054, rkne: 0.054, lank: 0.04, rank: 0.04, ltoe: 0.03, rtoe: 0.03,
  spi1: 0.06, spi2: 0.06, spi3: 0.06, neck: 0.048, lcla: 0.045, rcla: 0.045, head: 0.0,
  lsho: 0.058, rsho: 0.058, lelb: 0.044, relb: 0.044, lwri: 0.033, rwri: 0.033, lhan: 0.027, rhan: 0.027,
};
/** Internal SMPL torso connectors & joint spheres already enclosed by the sculpted pelvis/abdomen/ribcage ellipsoids. */
const HIDDEN_TORSO_BONE_CHILDREN = new Set(["lhip", "rhip", "spi1", "spi2", "spi3", "lcla", "rcla", "lsho", "rsho"]);
const HIDDEN_TORSO_JOINTS = new Set(["pelv", "spi1", "spi2", "spi3", "lcla", "rcla"]);

// ── Small math helpers (module-level scratch objects; render loop is single-threaded) ──
const UP = new THREE.Vector3(0, 1, 0);
const _a = new THREE.Vector3();
const _b = new THREE.Vector3();
const _d = new THREE.Vector3();
const _x = new THREE.Vector3();
const _y = new THREE.Vector3();
const _z = new THREE.Vector3();
const _c = new THREE.Vector3();
const _m = new THREE.Matrix4();
const _q = new THREE.Quaternion();
const SHEEN_WHITE = new THREE.Color("#ffffff");
const EMISSIVE_TELE = new THREE.Color(SEG.tele);
const EMISSIVE_NONE = new THREE.Color("#000000");

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const finite3 = (p: number[] | undefined | null): p is number[] =>
  !!p && p.length >= 3 && Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]);

function placeBone(mesh: THREE.Object3D, a: number[], b: number[]) {
  if (!finite3(a) || !finite3(b)) {
    mesh.visible = false;
    return;
  }
  _a.set(a[0], a[1], a[2]);
  _b.set(b[0], b[1], b[2]);
  _d.subVectors(_b, _a);
  const len = _d.length();
  if (len < 1e-4) {
    mesh.visible = false;
    return;
  }
  mesh.visible = true;
  mesh.position.copy(_a).add(_b).multiplyScalar(0.5);
  mesh.scale.set(1, len, 1);
  _q.setFromUnitVectors(UP, _d.divideScalar(len));
  mesh.quaternion.copy(_q);
}

/** Orient `obj` so local +Y = up, local +X = right (orthonormalised), local +Z = right × up (= body backward). */
function orientBasis(obj: THREE.Object3D, up: THREE.Vector3, rightHint: THREE.Vector3) {
  _y.copy(up);
  if (_y.lengthSq() < 1e-10) _y.set(0, 1, 0);
  _y.normalize();
  _x.copy(rightHint).addScaledVector(_y, -rightHint.dot(_y));
  if (_x.lengthSq() < 1e-8) {
    _x.set(1, 0, 0).addScaledVector(_y, -_y.x);
    if (_x.lengthSq() < 1e-8) _x.set(0, 0, 1);
  }
  _x.normalize();
  _z.crossVectors(_x, _y).normalize();
  _m.makeBasis(_x, _y, _z);
  obj.quaternion.setFromRotationMatrix(_m);
}

function nearestFfcIndex(times: number[] | undefined, n: number): number {
  if (!times || times.length === 0) return Math.floor(n / 2);
  let best = 0;
  let bestAbs = Infinity;
  for (let i = 0; i < times.length; i++) {
    const a = Math.abs(times[i]);
    if (Number.isFinite(a) && a < bestAbs) {
      bestAbs = a;
      best = i;
    }
  }
  return best;
}

function mixHex(hex: string, withHex: string, t: number): string {
  const c = new THREE.Color(hex).lerp(new THREE.Color(withHex), t);
  return `#${c.getHexString()}`;
}

function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const on = () => setReduced(mq.matches);
    on();
    mq.addEventListener?.("change", on);
    return () => mq.removeEventListener?.("change", on);
  }, []);
  return reduced;
}

/** Soft radial texture for contact-shadow discs (created per mount so WebGL unmount disposal never poisons a later Canvas). */
function createContactDiscTexture(): THREE.CanvasTexture | null {
  if (typeof document === "undefined") return null;
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const g = c.getContext("2d");
  if (!g) return null;
  const grd = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grd.addColorStop(0, "rgba(23,23,23,0.55)");
  grd.addColorStop(0.45, "rgba(23,23,23,0.28)");
  grd.addColorStop(1, "rgba(23,23,23,0)");
  g.fillStyle = grd;
  g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/** Local WebGL / R3F error boundary so route transitions or context loss never throw unhandled errors. */
class CanvasErrorBoundary extends React.Component<
  { resetKey: string; children: React.ReactNode },
  { hasError: boolean }
> {
  constructor(props: { resetKey: string; children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }
  static getDerivedStateFromError() {
    return { hasError: true };
  }
  componentDidCatch() {
    // Swallowed locally so SPA navigation stays spotless; user can reload the 3D view with one tap.
  }
  componentDidUpdate(prevProps: { resetKey: string }) {
    if (prevProps.resetKey !== this.props.resetKey && this.state.hasError) {
      this.setState({ hasError: false });
    }
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="v-stage flex flex-col items-center justify-center gap-2 h-full w-full p-4 text-center">
          <span className="v-label">3D pitch paused</span>
          <button
            type="button"
            onClick={() => this.setState({ hasError: false })}
            className="v-btn-ghost h-7 px-2.5 text-[11.5px]"
          >
            Reload 3D view
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Sculpted mannequin
// ─────────────────────────────────────────────────────────────────────────────

interface MannequinProps {
  getPose: PoseFn;
  arm: "right" | "left";
  /** Single colour for every part (ghost). Omit for segment colours. */
  tint?: string;
  opacity?: number;
  castShadow?: boolean;
  interactive?: boolean;
  selectedJoint?: number | null;
  highlightJoints?: number[];
  onJointClick?: (idx: number) => void;
  onJointDoubleClick?: (idx: number, pos: THREE.Vector3) => void;
}

const noRaycast = () => undefined;

const Mannequin: React.FC<MannequinProps> = ({
  getPose,
  arm,
  tint,
  opacity = 1,
  castShadow = true,
  interactive = false,
  selectedJoint = null,
  highlightJoints,
  onJointClick,
  onJointDoubleClick,
}) => {
  const groupRef = useRef<THREE.Group>(null);
  const boneRefs = useRef<Array<THREE.Mesh | null>>([]);
  const jointRefs = useRef<Array<THREE.Mesh | null>>([]);
  const ribRef = useRef<THREE.Mesh>(null);
  const abdRef = useRef<THREE.Mesh>(null);
  const pelvRef = useRef<THREE.Mesh>(null);
  const headRef = useRef<THREE.Group>(null);
  const lastPoseRef = useRef<Pose | null>(null);
  const lastSelRef = useRef<string>("");

  useEffect(() => {
    return () => {
      if (typeof document !== "undefined" && document.body.style.cursor === "pointer") {
        document.body.style.cursor = "";
      }
    };
  }, []);

  const translucent = opacity < 0.999;
  const bones = useMemo(
    () =>
      SMPL24_BONES.filter(([, c]) => !HIDDEN_TORSO_BONE_CHILDREN.has(SMPL24_JOINTS[c])).map(([p, c]) => {
        const name = SMPL24_JOINTS[c];
        const r = BONE_RADII[name] ?? [0.04, 0.04];
        const seg = boneSegment(p, c, arm);
        return { p, c, rParent: r[0], rChild: r[1], color: tint ?? SEG_COLOR[seg] };
      }),
    [arm, tint]
  );
  const joints = useMemo(
    () =>
      SMPL24_JOINTS.map((name, i) => ({
        i,
        r: HIDDEN_TORSO_JOINTS.has(name) ? 0 : JOINT_RADII[name] ?? 0.04,
        color: tint ?? SEG_COLOR[jointSegment(i, arm)],
      })),
    [arm, tint]
  );

  const matProps = (color: string) => ({
    color,
    roughness: 0.46,
    metalness: 0.02,
    clearcoat: tint ? 0 : 0.35,
    clearcoatRoughness: 0.45,
    sheen: tint ? 0 : 0.25,
    sheenColor: SHEEN_WHITE,
    transparent: translucent,
    opacity,
    depthWrite: !translucent || opacity > 0.6,
  });

  const thoraxCol = tint ?? SEG.thorax;
  const abdomenCol = tint ?? mixHex(SEG.thorax, "#ffffff", 0.28);
  const pelvisCol = tint ?? SEG.pelvis;
  const headCol = tint ?? "#2b2b2b";
  const visorCol = tint ? mixHex(tint, "#000000", 0.3) : "#0b0b0b";
  const hl = useMemo(() => new Set(highlightJoints ?? []), [highlightJoints]);

  useFrame(() => {
    const P = getPose();
    const g = groupRef.current;
    if (!g) return;
    if (!P || P.length < 24) {
      g.visible = false;
      return;
    }
    g.visible = true;
    const selKey = `${selectedJoint}|${[...hl].join(",")}`;
    if (P === lastPoseRef.current && selKey === lastSelRef.current) return;
    lastPoseRef.current = P;
    lastSelRef.current = selKey;

    for (let b = 0; b < bones.length; b++) {
      const m = boneRefs.current[b];
      if (m) placeBone(m, P[bones[b].p], P[bones[b].c]);
    }
    for (let i = 0; i < 24; i++) {
      const m = jointRefs.current[i];
      if (!m) continue;
      const p = P[i];
      if (!finite3(p) || joints[i].r <= 0) {
        m.visible = false;
        continue;
      }
      m.visible = true;
      m.position.set(p[0], p[1], p[2]);
      const boost = i === selectedJoint ? 1.75 : hl.has(i) ? 1.45 : 1;
      m.scale.setScalar(joints[i].r * boost);
    }

    const pel = P[J.pelv], lh = P[J.lhip], rh = P[J.rhip], s1 = P[J.spi1], s2 = P[J.spi2];
    const nk = P[J.neck], hd = P[J.head], ls = P[J.lsho], rs = P[J.rsho];
    if ([pel, lh, rh, s1, s2, nk, hd, ls, rs].every(finite3)) {
      // Pelvis block
      if (pelvRef.current) {
        const hipW = Math.hypot(rh[0] - lh[0], rh[1] - lh[1], rh[2] - lh[2]);
        _c.set((pel[0] + lh[0] + rh[0]) / 3, (pel[1] + lh[1] + rh[1]) / 3 + 0.015, (pel[2] + lh[2] + rh[2]) / 3);
        pelvRef.current.position.copy(_c);
        orientBasis(pelvRef.current, _a.set(s1[0] - pel[0], s1[1] - pel[1], s1[2] - pel[2]), _b.set(rh[0] - lh[0], rh[1] - lh[1], rh[2] - lh[2]));
        pelvRef.current.scale.set(clamp(hipW * 0.5 + 0.075, 0.12, 0.24), 0.105, 0.11);
      }
      // Abdomen
      if (abdRef.current) {
        const len = Math.hypot(s2[0] - pel[0], s2[1] - pel[1], s2[2] - pel[2]);
        abdRef.current.position.set(pel[0] + (s2[0] - pel[0]) * 0.58, pel[1] + (s2[1] - pel[1]) * 0.58, pel[2] + (s2[2] - pel[2]) * 0.58);
        orientBasis(abdRef.current, _a.set(s2[0] - pel[0], s2[1] - pel[1], s2[2] - pel[2]), _b.set(rh[0] - lh[0], rh[1] - lh[1], rh[2] - lh[2]));
        abdRef.current.scale.set(0.118, clamp(len * 0.56, 0.08, 0.2), 0.092);
      }
      // Ribcage
      if (ribRef.current) {
        const shW = Math.hypot(rs[0] - ls[0], rs[1] - ls[1], rs[2] - ls[2]);
        const len = Math.hypot(nk[0] - s2[0], nk[1] - s2[1], nk[2] - s2[2]);
        ribRef.current.position.set(s2[0] + (nk[0] - s2[0]) * 0.4, s2[1] + (nk[1] - s2[1]) * 0.4, s2[2] + (nk[2] - s2[2]) * 0.4);
        orientBasis(ribRef.current, _a.set(nk[0] - s1[0], nk[1] - s1[1], nk[2] - s1[2]), _b.set(rs[0] - ls[0], rs[1] - ls[1], rs[2] - ls[2]));
        ribRef.current.scale.set(clamp(shW * 0.4, 0.11, 0.2), clamp(len * 0.62, 0.12, 0.26), 0.112);
      }
      // Head + visor (visor faces the thorax-forward direction: forward = up × right; blend upper-spine for 25 fps stability)
      if (headRef.current) {
        _a.set(
          0.65 * (hd[0] - nk[0]) + 0.35 * (nk[0] - s2[0]),
          0.65 * (hd[1] - nk[1]) + 0.35 * (nk[1] - s2[1]),
          0.65 * (hd[2] - nk[2]) + 0.35 * (nk[2] - s2[2])
        );
        if (_a.lengthSq() < 1e-10) _a.set(0, 1, 0);
        _a.normalize();
        headRef.current.position.set(hd[0] + _a.x * 0.055, hd[1] + _a.y * 0.055, hd[2] + _a.z * 0.055);
        orientBasis(headRef.current, _a, _b.set(rs[0] - ls[0], rs[1] - ls[1], rs[2] - ls[2]));
      }
    }
  });

  const handleClick = useCallback(
    (i: number) => (e: ThreeEvent<MouseEvent>) => {
      if (!interactive) return;
      e.stopPropagation();
      if (e.delta > 4) return; // it was an orbit drag
      onJointClick?.(i);
    },
    [interactive, onJointClick]
  );
  const handleDbl = useCallback(
    (i: number) => (e: ThreeEvent<MouseEvent>) => {
      if (!interactive) return;
      e.stopPropagation();
      const m = jointRefs.current[i];
      if (m) onJointDoubleClick?.(i, m.position.clone());
    },
    [interactive, onJointDoubleClick]
  );

  return (
    <group ref={groupRef}>
      {bones.map((b, k) => (
        <mesh
          key={`b${k}`}
          ref={(el) => {
            boneRefs.current[k] = el;
          }}
          castShadow={castShadow}
          raycast={noRaycast}
        >
          <cylinderGeometry args={[b.rChild, b.rParent, 1, 18, 1]} />
          <meshPhysicalMaterial {...matProps(b.color)} />
        </mesh>
      ))}
      {joints.filter((j) => j.r > 0).map((j) => (
        <mesh
          key={`j${j.i}`}
          ref={(el) => {
            jointRefs.current[j.i] = el;
          }}
          castShadow={castShadow}
          {...(interactive
            ? {
                onClick: handleClick(j.i),
                onDoubleClick: handleDbl(j.i),
                onPointerOver: (e: ThreeEvent<PointerEvent>) => {
                  e.stopPropagation();
                  document.body.style.cursor = "pointer";
                },
                onPointerOut: () => {
                  document.body.style.cursor = "";
                },
              }
            : { raycast: noRaycast })}
        >
          <sphereGeometry args={[1, 20, 16]} />
          <meshPhysicalMaterial
            {...matProps(j.i === selectedJoint && !tint ? "#ffffff" : j.color)}
            emissive={j.i === selectedJoint && !tint ? EMISSIVE_TELE : EMISSIVE_NONE}
            emissiveIntensity={j.i === selectedJoint && !tint ? 0.35 : 0}
          />
        </mesh>
      ))}
      <mesh ref={pelvRef} castShadow={castShadow} raycast={noRaycast}>
        <sphereGeometry args={[1, 28, 20]} />
        <meshPhysicalMaterial {...matProps(pelvisCol)} />
      </mesh>
      <mesh ref={abdRef} castShadow={castShadow} raycast={noRaycast}>
        <sphereGeometry args={[1, 28, 20]} />
        <meshPhysicalMaterial {...matProps(abdomenCol)} />
      </mesh>
      <mesh ref={ribRef} castShadow={castShadow} raycast={noRaycast}>
        <sphereGeometry args={[1, 32, 24]} />
        <meshPhysicalMaterial {...matProps(thoraxCol)} />
      </mesh>
      <group ref={headRef}>
        <mesh castShadow={castShadow} scale={[0.092, 0.112, 0.102]} raycast={noRaycast}>
          <sphereGeometry args={[1, 28, 20]} />
          <meshPhysicalMaterial {...matProps(headCol)} />
        </mesh>
        {/* Visor: curved band on the face side (local −Z = forward). theta=0 is +Z, so centre the arc on π. */}
        <mesh position={[0, 0.008, 0]} raycast={noRaycast}>
          <cylinderGeometry args={[0.104, 0.104, 0.036, 24, 1, true, Math.PI - 0.85, 1.7]} />
          <meshPhysicalMaterial {...matProps(visorCol)} side={THREE.DoubleSide} roughness={0.2} clearcoat={0.8} />
        </mesh>
        <mesh position={[0, 0.0, -0.104]} raycast={noRaycast}>
          <boxGeometry args={[0.03, 0.012, 0.012]} />
          <meshBasicMaterial color={tint ? visorCol : SEG.wrist} transparent={translucent} opacity={opacity} />
        </mesh>
      </group>
    </group>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Uncertainty spheres
// ─────────────────────────────────────────────────────────────────────────────

const SigmaSpheres: React.FC<{ getPose: PoseFn; getSigma: () => number[] | null; arm: "right" | "left" }> = ({ getPose, getSigma, arm }) => {
  const refs = useRef<Array<THREE.Mesh | null>>([]);
  const colors = useMemo(() => SMPL24_JOINTS.map((_, i) => SEG_COLOR[jointSegment(i, arm)]), [arm]);
  useFrame(() => {
    const P = getPose();
    const S = getSigma();
    for (let i = 0; i < 24; i++) {
      const m = refs.current[i];
      if (!m) continue;
      const p = P?.[i];
      const s = S?.[i];
      if (!finite3(p) || s == null || !Number.isFinite(s) || s <= 0) {
        m.visible = false;
        continue;
      }
      m.visible = true;
      m.position.set(p[0], p[1], p[2]);
      m.scale.setScalar(clamp(s, 0.012, 0.35));
    }
  });
  return (
    <group>
      {colors.map((c, i) => (
        <mesh key={i} ref={(el) => { refs.current[i] = el; }} raycast={noRaycast} renderOrder={2}>
          <sphereGeometry args={[1, 18, 14]} />
          <meshBasicMaterial color={c} transparent opacity={0.13} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 3D trajectory ribbons
// ─────────────────────────────────────────────────────────────────────────────

/** Catmull-Rom curve parameterised by point index (not arc length) so drawRange reveal maps exactly to frames. */
class IndexCurve extends THREE.Curve<THREE.Vector3> {
  private c: THREE.CatmullRomCurve3;
  constructor(points: THREE.Vector3[]) {
    super();
    this.c = new THREE.CatmullRomCurve3(points, false, "centripetal", 0.2);
  }
  getPoint(t: number, optionalTarget: THREE.Vector3 = new THREE.Vector3()): THREE.Vector3 {
    return this.c.getPoint(t, optionalTarget);
  }
  getPointAt(u: number, optionalTarget?: THREE.Vector3): THREE.Vector3 {
    return this.getPoint(u, optionalTarget);
  }
  getTangentAt(u: number, optionalTarget?: THREE.Vector3): THREE.Vector3 {
    return this.getTangent(u, optionalTarget);
  }
}

interface TrailSeries {
  frames: number[];
  points: THREE.Vector3[];
}

const SUB = 6;
const RADIAL = 8;

const TrailRibbon: React.FC<{ series: TrailSeries; color: string; radius: number; getFrame: () => number; reveal?: boolean; dashed?: boolean }> = ({
  series,
  color,
  radius,
  getFrame,
  reveal = true,
}) => {
  const geom = useMemo(() => {
    if (series.points.length < 2) return null;
    const curve = new IndexCurve(series.points);
    return new THREE.TubeGeometry(curve, (series.points.length - 1) * SUB, radius, RADIAL, false);
  }, [series, radius]);
  useEffect(
    () => () => {
      try {
        geom?.dispose();
      } catch {
        /* already disposed */
      }
    },
    [geom]
  );
  const revealRef = useRef<THREE.Mesh>(null);
  const lastRef = useRef<number>(-1);

  useFrame(() => {
    if (!geom || !revealRef.current) return;
    const f = getFrame();
    if (f === lastRef.current) return;
    lastRef.current = f;
    const fr = series.frames;
    const n = fr.length;
    let u: number;
    if (!reveal) u = 1;
    else if (f <= fr[0]) u = 0;
    else if (f >= fr[n - 1]) u = 1;
    else {
      let k = 0;
      while (k < n - 2 && fr[k + 1] <= f) k++;
      const span = fr[k + 1] - fr[k] || 1;
      u = (k + clamp((f - fr[k]) / span, 0, 1)) / (n - 1);
    }
    const segs = (n - 1) * SUB;
    revealRef.current.geometry.setDrawRange(0, Math.floor(u * segs) * RADIAL * 6);
  });

  if (!geom) return null;
  return (
    <group>
      <mesh geometry={geom} dispose={null} raycast={noRaycast} renderOrder={1}>
        <meshBasicMaterial color={color} transparent opacity={0.14} depthWrite={false} />
      </mesh>
      <RevealMesh geom={geom} color={color} meshRef={revealRef} />
    </group>
  );
};

/** Second mesh needs its own geometry instance (drawRange is per-geometry). */
const RevealMesh: React.FC<{ geom: THREE.BufferGeometry; color: string; meshRef: React.RefObject<THREE.Mesh> }> = ({ geom, color, meshRef }) => {
  const g2 = useMemo(() => geom.clone(), [geom]);
  useEffect(
    () => () => {
      try {
        g2.dispose();
      } catch {
        /* already disposed */
      }
    },
    [g2]
  );
  return (
    <mesh ref={meshRef} geometry={g2} dispose={null} raycast={noRaycast} renderOrder={1}>
      <meshStandardMaterial color={color} roughness={0.35} emissive={color} emissiveIntensity={0.25} />
    </mesh>
  );
};

function buildTrail(telemetry: TelemetryV2, key: "bowling_wrist" | "head" | "com", jointIdx: number | null): TrailSeries | null {
  const out: TrailSeries = { frames: [], points: [] };
  const t3 = telemetry.trails?.["3d_m"]?.[key];
  const fi = telemetry.trails?.frame_indices;
  if (t3 && fi && t3.length === fi.length && t3.length >= 2) {
    // Precomputed trails: COM path is ground-zeroed already; joint trails are assumed to match joints3d_world_m.
    for (let i = 0; i < t3.length; i++) {
      const p = t3[i];
      if (p && finite3(p)) {
        out.frames.push(fi[i]);
        out.points.push(new THREE.Vector3(p[0], p[1], p[2]));
      }
    }
  } else if (key === "com") {
    const com = telemetry.com_world_m;
    if (!com) return null;
    for (let i = 0; i < com.length; i++) {
      if (finite3(com[i])) {
        out.frames.push(i);
        out.points.push(new THREE.Vector3(com[i][0], com[i][1], com[i][2]));
      }
    }
  } else if (jointIdx != null) {
    const J3 = telemetry.joints3d_world_m;
    for (let i = 0; i < J3.length; i++) {
      const p = J3[i]?.[jointIdx];
      if (finite3(p)) {
        out.frames.push(i);
        out.points.push(new THREE.Vector3(p[0], p[1], p[2]));
      }
    }
  }
  // Drop obvious teleports (>1.5 m in one frame) by truncating to the longest continuous run.
  if (out.points.length < 2) return null;
  let bestS = 0, bestE = 0, s = 0;
  for (let i = 1; i <= out.points.length; i++) {
    const broken = i === out.points.length || out.points[i].distanceTo(out.points[i - 1]) > 1.5;
    if (broken) {
      if (i - 1 - s > bestE - bestS) {
        bestS = s;
        bestE = i - 1;
      }
      s = i;
    }
  }
  if (bestE - bestS < 1) return null;
  return { frames: out.frames.slice(bestS, bestE + 1), points: out.points.slice(bestS, bestE + 1) };
}

// ─────────────────────────────────────────────────────────────────────────────
// Pitch scene
// ─────────────────────────────────────────────────────────────────────────────

const Strip: React.FC<{ x0: number; x1: number; z0: number; z1: number; color: string; y?: number }> = ({ x0, x1, z0, z1, color, y = 0.002 }) => (
  <mesh position={[(x0 + x1) / 2, y, (z0 + z1) / 2]} rotation={[-Math.PI / 2, 0, 0]} raycast={noRaycast}>
    <planeGeometry args={[Math.max(Math.abs(x1 - x0), 0.001), Math.max(Math.abs(z1 - z0), 0.001)]} />
    <meshBasicMaterial color={color} />
  </mesh>
);

const CREASE_W = 0.05;
const Creases: React.FC<{ end: "bowler" | "striker" }> = ({ end }) => {
  const s = end === "bowler" ? 1 : -1;
  const xb = end === "bowler" ? 0 : PITCH_LEN; // bowling crease
  const xp = xb + s * POPPING; // popping crease
  const xr = xb - s * POPPING; // back of return crease
  const h = CREASE_W / 2;
  return (
    <group>
      <Strip x0={xb - h} x1={xb + h} z0={-RETURN_HALF} z1={RETURN_HALF} color={SEG.ink} y={0.003} />
      <Strip x0={xp - h} x1={xp + h} z0={-POPPING_HALF} z1={POPPING_HALF} color={SEG.positive} y={0.0035} />
      <Strip x0={Math.min(xr, xp)} x1={Math.max(xr, xp)} z0={-RETURN_HALF - h} z1={-RETURN_HALF + h} color={SEG.ink} y={0.003} />
      <Strip x0={Math.min(xr, xp)} x1={Math.max(xr, xp)} z0={RETURN_HALF - h} z1={RETURN_HALF + h} color={SEG.ink} y={0.003} />
    </group>
  );
};

const Wicket: React.FC<{ x: number }> = ({ x }) => (
  <group position={[x, 0, 0]}>
    {[-STUMP_Z, 0, STUMP_Z].map((z) => (
      <mesh key={z} position={[0, STUMP_H / 2, z]} castShadow raycast={noRaycast}>
        <cylinderGeometry args={[STUMP_R, STUMP_R * 1.04, STUMP_H, 18]} />
        <meshPhysicalMaterial color="#c79a52" roughness={0.52} clearcoat={0.25} />
      </mesh>
    ))}
    {[-STUMP_Z / 2, STUMP_Z / 2].map((z) => (
      <mesh key={`bail${z}`} position={[0, STUMP_H + BAIL_R * 0.9, z]} rotation={[Math.PI / 2, 0, 0]} castShadow raycast={noRaycast}>
        <capsuleGeometry args={[BAIL_R, BAIL_LEN - 2 * BAIL_R, 4, 10]} />
        <meshPhysicalMaterial color="#ae7d38" roughness={0.48} clearcoat={0.25} />
      </mesh>
    ))}
  </group>
);

const PitchScene: React.FC<{ showGrid: boolean; focus: THREE.Vector3; creaseCalibrated: boolean }> = ({
  showGrid,
  focus,
  creaseCalibrated,
}) => (
  <group>
    {/* Paper ground (unlit so it stays true paper) + shadow catcher */}
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[creaseCalibrated ? 10 : focus.x, -0.002, creaseCalibrated ? 0 : focus.z]} raycast={noRaycast}>
      <planeGeometry args={[160, 120]} />
      <meshBasicMaterial color="#f7f7f5" />
    </mesh>
    {creaseCalibrated ? (
      <>
        <Strip x0={-4.5} x1={PITCH_LEN + 4.5} z0={-PITCH_HALF_WIDTH} z1={PITCH_HALF_WIDTH} color="#e7efe9" y={0.0008} />
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[focus.x, 0.004, focus.z]} receiveShadow raycast={noRaycast}>
          <planeGeometry args={[30, 30]} />
          <shadowMaterial transparent opacity={0.14} />
        </mesh>
        <Creases end="bowler" />
        <Creases end="striker" />
        <Wicket x={0} />
        <Wicket x={PITCH_LEN} />
        {showGrid && <gridHelper args={[48, 96, "#cfd6d1", "#e6e9e6"]} position={[PITCH_LEN / 2, 0.005, 0]} />}
      </>
    ) : (
      <>
        <Strip
          x0={focus.x - 5.5}
          x1={focus.x + 5.5}
          z0={focus.z - PITCH_HALF_WIDTH}
          z1={focus.z + PITCH_HALF_WIDTH}
          color="#e7efe9"
          y={0.0008}
        />
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[focus.x, 0.004, focus.z]} receiveShadow raycast={noRaycast}>
          <planeGeometry args={[30, 30]} />
          <shadowMaterial transparent opacity={0.14} />
        </mesh>
        <gridHelper
          args={showGrid ? [24, 24, "#cfd6d1", "#e4e8e5"] : [16, 16, "#d5ddd8", "#e8ece9"]}
          position={[focus.x, 0.005, focus.z]}
        />
      </>
    )}
  </group>
);

const ContactDiscs: React.FC<{ getPose: PoseFn }> = ({ getPose }) => {
  const tex = useMemo(() => createContactDiscTexture(), []);
  useEffect(
    () => () => {
      try {
        tex?.dispose();
      } catch {
        /* already disposed */
      }
    },
    [tex]
  );
  const lRef = useRef<THREE.Mesh>(null);
  const rRef = useRef<THREE.Mesh>(null);
  const bRef = useRef<THREE.Mesh>(null);
  useFrame(() => {
    if (!tex) return;
    const P = getPose();
    const put = (m: THREE.Mesh | null, a: number[] | undefined, b: number[] | undefined, size: number, maxOp: number) => {
      if (!m) return;
      if (!finite3(a) || !finite3(b)) {
        m.visible = false;
        return;
      }
      const h = Math.max(0, Math.min(a[1], b[1]));
      m.visible = true;
      m.position.set((a[0] + b[0]) / 2, 0.006, (a[2] + b[2]) / 2);
      const s = size * (1 + h * 0.8);
      m.scale.set(s, s * 0.72, 1);
      const mat = m.material as THREE.MeshBasicMaterial | undefined;
      if (mat && typeof mat.opacity === "number") {
        mat.opacity = maxOp * clamp(1 - h / 0.4, 0, 1);
      }
    };
    if (!P) return;
    put(lRef.current, P[J.lank], P[J.ltoe], 0.34, 0.55);
    put(rRef.current, P[J.rank], P[J.rtoe], 0.34, 0.55);
    put(bRef.current, P[J.lank], P[J.rank], 0.95, 0.35);
  });
  if (!tex) return null;
  return (
    <group>
      {[lRef, rRef, bRef].map((r, i) => (
        <mesh key={i} ref={r} rotation={[-Math.PI / 2, 0, 0]} raycast={noRaycast} renderOrder={0}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial map={tex} transparent depthWrite={false} opacity={0.5} />
        </mesh>
      ))}
    </group>
  );
};

const StudioLights: React.FC<{ focus: THREE.Vector3 }> = ({ focus }) => {
  const keyRef = useRef<THREE.DirectionalLight>(null);
  const { scene } = useThree();
  useEffect(() => {
    const l = keyRef.current;
    if (!l || !l.target || !scene) return;
    l.target.position.copy(focus);
    scene.add(l.target);
    l.target.updateMatrixWorld();
    return () => {
      try {
        if (scene && l?.target) scene.remove(l.target);
      } catch {
        /* scene already torn down */
      }
    };
  }, [focus, scene]);
  return (
    <>
      <hemisphereLight args={["#ffffff", "#dfe7e1", 0.95]} />
      <ambientLight intensity={0.18} />
      <directionalLight
        ref={keyRef}
        position={[focus.x - 2.5, 7.5, focus.z + 4.5]}
        intensity={1.55}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
        shadow-bias={-0.0004}
        shadow-normalBias={0.02}
        shadow-camera-left={-5}
        shadow-camera-right={5}
        shadow-camera-top={5}
        shadow-camera-bottom={-5}
        shadow-camera-near={0.5}
        shadow-camera-far={25}
      />
      <directionalLight position={[focus.x + 6, 3.5, focus.z - 5]} intensity={0.55} color="#eef2ff" />
      <directionalLight position={[focus.x - 4, 2.5, focus.z - 6]} intensity={0.35} color="#fff7ed" />
    </>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Camera rig (600 ms cubic ease between presets; OrbitControls stay live)
// ─────────────────────────────────────────────────────────────────────────────

interface CamGoal {
  pos: THREE.Vector3;
  target: THREE.Vector3;
  fov: number;
}

const easeInOutCubic = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

const CameraRig: React.FC<{ goal: CamGoal; goalKey: string; reduced: boolean; cancelRef: React.MutableRefObject<number> }> = ({
  goal,
  goalKey,
  reduced,
  cancelRef,
}) => {
  const anim = useRef<{ t0: number; from: CamGoal; to: CamGoal } | null>(null);
  const appliedKey = useRef<string>("");
  const lastCancel = useRef<number>(cancelRef.current);
  const first = useRef(true);

  useFrame((state) => {
    const controls = state.controls as { target?: THREE.Vector3; update?: () => void } | null;
    if (!controls?.target) return;
    const camera = state.camera as THREE.PerspectiveCamera;

    if (cancelRef.current !== lastCancel.current) {
      lastCancel.current = cancelRef.current;
      anim.current = null;
    }

    if (appliedKey.current !== goalKey) {
      appliedKey.current = goalKey;
      const fromTarget = controls.target.clone();
      if (first.current || reduced) {
        first.current = false;
        camera.position.copy(goal.pos);
        if (Math.abs(camera.fov - goal.fov) > 1e-3) {
          camera.fov = goal.fov;
          camera.updateProjectionMatrix();
        }
        controls.target.copy(goal.target);
        controls.update?.();
        anim.current = null;
        return;
      } else {
        anim.current = { t0: performance.now(), from: { pos: camera.position.clone(), target: fromTarget, fov: camera.fov }, to: goal };
      }
    }
    const a = anim.current;
    if (!a) return;
    const t = clamp((performance.now() - a.t0) / TRANSITION_MS, 0, 1);
    const e = easeInOutCubic(t);
    camera.position.lerpVectors(a.from.pos, a.to.pos, e);
    const fov = a.from.fov + (a.to.fov - a.from.fov) * e;
    if (Math.abs(camera.fov - fov) > 1e-3) {
      camera.fov = fov;
      camera.updateProjectionMatrix();
    }
    _c.lerpVectors(a.from.target, a.to.target, e);
    controls.target.copy(_c);
    controls.update?.();
    if (t >= 1) anim.current = null;
  });
  return null;
};

// ─────────────────────────────────────────────────────────────────────────────
// Main component
// ─────────────────────────────────────────────────────────────────────────────

export const PitchStage3D: React.FC<PitchStage3DProps> = ({
  telemetry,
  bowlingArm = "right",
  frame: frameProp,
  ghostJoints3d,
  ghostTimeRelFfcMs,
  ghostLabel,
  mirrorGhost = false,
  defaultPreset = "match",
  showUncertainty = false,
  showTrails,
  strobeMode = false,
  idleOrbit = false,
  compact = false,
  heightClass,
  onSelectJoint,
  eventFrames,
  strobeFrames,
  preset: presetProp,
  onPresetChange,
}) => {
  const playhead = usePlayhead();
  const joints3d = Array.isArray(telemetry?.joints3d_world_m) ? telemetry.joints3d_world_m : [];
  const n = joints3d.length;
  const rawFrame = frameProp ?? playhead.frame;
  const frame = n > 0 ? clamp(Math.round(Number.isFinite(rawFrame) ? rawFrame : 0), 0, n - 1) : 0;
  const frameRef = useRef(frame);
  frameRef.current = frame;
  const reduced = usePrefersReducedMotion();

  const [presetState, setPresetState] = useState<VirtualCameraPreset>(presetProp ?? defaultPreset);
  const [userPickedCamera, setUserPickedCamera] = useState(false);
  const cancelAnimRef = useRef(0);
  const preset = presetProp ?? presetState;
  const setPreset = (p: VirtualCameraPreset) => {
    setUserPickedCamera(true);
    setPresetState(p);
    onPresetChange?.(p);
    setNonce((x) => x + 1);
  };
  const [nonce, setNonce] = useState(0);
  const [sigmaOn, setSigmaOn] = useState<boolean>(showUncertainty);
  useEffect(() => setSigmaOn(showUncertainty), [showUncertainty]);
  const [trailsOn, setTrailsOn] = useState({ wrist: showTrails?.wrist ?? true, head: showTrails?.head ?? false, com: showTrails?.com ?? false });
  useEffect(() => {
    if (showTrails) setTrailsOn({ wrist: showTrails.wrist ?? true, head: showTrails.head ?? false, com: showTrails.com ?? false });
  }, [showTrails?.wrist, showTrails?.head, showTrails?.com]); // eslint-disable-line react-hooks/exhaustive-deps
  const [selectedJoint, setSelectedJoint] = useState<number | null>(null);
  const [focusOverride, setFocusOverride] = useState<THREE.Vector3 | null>(null);

  const times = Array.isArray(telemetry?.time_rel_ffc_ms) ? telemetry.time_rel_ffc_ms : [];
  const ffcIdx = useMemo(() => {
    const e = eventFrames?.ffc;
    if (e != null && Number.isFinite(e)) return clamp(Math.round(e), 0, Math.max(0, n - 1));
    return clamp(nearestFfcIndex(times, n), 0, Math.max(0, n - 1));
  }, [eventFrames?.ffc, times, n]);
  const brIdx = useMemo(() => {
    const e = eventFrames?.br;
    if (e != null && Number.isFinite(e)) return clamp(Math.round(e), 0, Math.max(0, n - 1));
    // Felton (2019) prior: BR ≈ FFC + ~108 ms.
    let best = ffcIdx;
    let bestAbs = Infinity;
    for (let i = 0; i < times.length; i++) {
      const a = Math.abs(times[i] - 108);
      if (a < bestAbs) {
        bestAbs = a;
        best = i;
      }
    }
    return clamp(best, 0, Math.max(0, n - 1));
  }, [eventFrames?.br, times, ffcIdx, n]);

  const poseAt = useCallback(
    (f: number): Pose | null => (n > 0 ? (joints3d[clamp(f, 0, n - 1)] as Pose) ?? null : null),
    [joints3d, n]
  );
  const getLivePose = useCallback<PoseFn>(() => poseAt(frameRef.current), [poseAt]);
  const getSigma = useCallback(
    () => (Array.isArray(telemetry?.joint_uncertainties_m) ? telemetry.joint_uncertainties_m[frameRef.current] ?? null : null),
    [telemetry]
  );
  const getFrame = useCallback(() => frameRef.current, []);

  // ── Focus point (bowler's pelvis at FFC) ─────────────────────────────────
  const focus = useMemo(() => {
    const p = poseAt(ffcIdx)?.[J.pelv];
    if (finite3(p)) return new THREE.Vector3(p[0], 0.95, p[2]);
    return new THREE.Vector3(0.8, 0.95, 0);
  }, [poseAt, ffcIdx]);

  const camWorld = useMemo(() => {
    const c = telemetry?.camera_position_world_m;
    if (!finite3(c)) return null;
    const v = new THREE.Vector3(c[0], c[1] - (telemetry.y_ground_offset_m || 0), c[2]);
    const d = v.distanceTo(focus);
    if (!(d > 1.5 && d < 90)) return null;
    return v;
  }, [telemetry, focus]);

  const s = bowlingArm === "right" ? 1 : -1;
  const goal = useMemo<CamGoal>(() => {
    const f = focusOverride ?? focus;
    switch (preset) {
      case "side":
        return { pos: new THREE.Vector3(f.x, f.y + 0.35, f.z + s * 6.4), target: f.clone(), fov: DEFAULT_FOV };
      case "behind":
        return { pos: new THREE.Vector3(f.x - 6.6, f.y + 0.9, f.z), target: f.clone(), fov: DEFAULT_FOV };
      case "front":
        return { pos: new THREE.Vector3(f.x + 7.0, f.y + 0.55, f.z), target: f.clone(), fov: DEFAULT_FOV };
      case "top":
        return { pos: new THREE.Vector3(f.x, 9.5, f.z + 0.02), target: new THREE.Vector3(f.x, 0, f.z), fov: DEFAULT_FOV };
      case "arm": {
        const sh = poseAt(brIdx)?.[bowlingArm === "right" ? J.rsho : J.lsho];
        const t = finite3(sh) ? new THREE.Vector3(sh[0], sh[1] + 0.12, sh[2]) : new THREE.Vector3(f.x, 1.55, f.z);
        return { pos: new THREE.Vector3(t.x + 1.25, t.y + 0.2, t.z + s * 1.9), target: t, fov: 34 };
      }
      case "match":
      default: {
        if (!camWorld) return { pos: new THREE.Vector3(f.x, f.y + 0.35, f.z + s * 6.4), target: f.clone(), fov: DEFAULT_FOV };
        const d = camWorld.distanceTo(f);
        // Distance-matched field of view: ~3.6 m of scene height at the bowler, like the real phone framing.
        const fov = clamp((2 * Math.atan(1.8 / d) * 180) / Math.PI, 6, 50);
        return { pos: camWorld.clone(), target: f.clone(), fov };
      }
    }
  }, [preset, focus, focusOverride, camWorld, poseAt, brIdx, bowlingArm, s]);
  const goalKey = `${preset}|${nonce}|${goal.pos.toArray().map((v) => v.toFixed(2)).join(",")}|${goal.target.toArray().map((v) => v.toFixed(2)).join(",")}`;

  // ── Trails ───────────────────────────────────────────────────────────────
  const wristIdx = bowlingArm === "right" ? J.rwri : J.lwri;
  const wristTrail = useMemo(() => (n > 1 ? buildTrail(telemetry, "bowling_wrist", wristIdx) : null), [telemetry, wristIdx, n]);
  const headTrail = useMemo(() => (n > 1 ? buildTrail(telemetry, "head", J.head) : null), [telemetry, n]);
  const comTrail = useMemo(() => (n > 1 ? buildTrail(telemetry, "com", null) : null), [telemetry, n]);

  // ── Ghost ────────────────────────────────────────────────────────────────
  const ghost = useMemo(() => {
    if (!ghostJoints3d || ghostJoints3d.length === 0 || n === 0) return null;
    const G = mirrorGhost ? ghostJoints3d.map((fr) => fr.map((p) => [p[0], p[1], -p[2]])) : ghostJoints3d;
    const gTimes = ghostTimeRelFfcMs && ghostTimeRelFfcMs.length === G.length ? ghostTimeRelFfcMs : null;
    const gFfc = gTimes ? nearestFfcIndex(gTimes, G.length) : clamp(ffcIdx, 0, G.length - 1);
    const ourP = poseAt(ffcIdx)?.[J.pelv];
    const gP = G[gFfc]?.[J.pelv];
    const dx = finite3(ourP) && finite3(gP) ? ourP[0] - gP[0] : 0;
    const dz = finite3(ourP) && finite3(gP) ? ourP[2] - gP[2] : 0;
    return { G, gTimes, dx, dz };
  }, [ghostJoints3d, ghostTimeRelFfcMs, mirrorGhost, n, ffcIdx, poseAt]);

  const ghostCache = useRef<{ f: number; pose: Pose | null }>({ f: -1, pose: null });
  useEffect(() => {
    ghostCache.current = { f: -1, pose: null };
  }, [ghost]);
  const getGhostPose = useCallback<PoseFn>(() => {
    if (!ghost) return null;
    const f = frameRef.current;
    if (ghostCache.current.f === f) return ghostCache.current.pose;
    const { G, gTimes, dx, dz } = ghost;
    let i0 = 0, i1 = 0, alpha = 0;
    if (gTimes) {
      const t = times[f] ?? 0;
      const m = gTimes.length;
      if (t <= gTimes[0]) i0 = i1 = 0;
      else if (t >= gTimes[m - 1]) i0 = i1 = m - 1;
      else {
        let lo = 0, hi = m - 1;
        while (lo + 1 < hi) {
          const mid = (lo + hi) >> 1;
          if (gTimes[mid] <= t) lo = mid;
          else hi = mid;
        }
        i0 = lo;
        i1 = hi;
        const span = gTimes[i1] - gTimes[i0];
        alpha = span > 1e-6 ? clamp((t - gTimes[i0]) / span, 0, 1) : 0;
      }
    } else {
      i0 = i1 = clamp(f, 0, G.length - 1);
    }
    const A = G[i0], B = G[i1];
    let pose: Pose | null = null;
    if (A && B && A.length >= 24 && B.length >= 24) {
      pose = new Array(24);
      for (let j = 0; j < 24; j++) {
        const a = A[j], b = B[j];
        if (!finite3(a) || !finite3(b)) {
          pose[j] = [NaN, NaN, NaN];
          continue;
        }
        pose[j] = [a[0] + (b[0] - a[0]) * alpha + dx, a[1] + (b[1] - a[1]) * alpha, a[2] + (b[2] - a[2]) * alpha + dz];
      }
    }
    ghostCache.current = { f, pose };
    return pose;
  }, [ghost, times]);

  // ── Strobe frames (6 poses BFC → FT) ─────────────────────────────────────
  const strobe = useMemo(() => {
    if (!strobeMode || n === 0) return [] as Array<{ f: number; opacity: number }>;
    let frames: number[];
    if (strobeFrames && strobeFrames.length > 0) {
      frames = strobeFrames.map((f) => clamp(Math.round(f), 0, n - 1));
    } else {
      const fpsA = telemetry.analysis_fps || 30;
      let lo: number | null = eventFrames?.bfc ?? null;
      let hi: number | null = eventFrames?.ft ?? null;
      if (lo == null) {
        const fi = telemetry.trails?.frame_indices;
        lo = fi && fi.length ? fi[0] + Math.round(0.15 * fpsA) : ffcIdx - Math.round(0.2 * fpsA);
      }
      if (hi == null) hi = Math.min(n - 1, brIdx + Math.max(2, Math.round(0.18 * fpsA)));
      lo = clamp(Math.round(lo), 0, n - 1);
      hi = clamp(Math.round(hi), lo, n - 1);
      frames = Array.from({ length: 6 }, (_, k) => Math.round(lo! + ((hi! - lo!) * k) / 5));
    }
    frames = [...new Set(frames)].sort((a, b) => a - b);
    const maxD = Math.max(1, ...frames.map((f) => Math.abs(f - brIdx)));
    return frames.map((f) => ({ f, opacity: clamp(0.2 + 0.72 * (1 - Math.abs(f - brIdx) / maxD), 0.18, 0.92) }));
  }, [strobeMode, strobeFrames, eventFrames?.bfc, eventFrames?.ft, n, telemetry, ffcIdx, brIdx]);
  const strobeBest = strobe.length ? strobe.reduce((a, b) => (b.opacity > a.opacity ? b : a)).f : -1;

  const handleJointClick = useCallback(
    (i: number) => {
      setSelectedJoint((prev) => (prev === i ? null : i));
      onSelectJoint?.(i);
    },
    [onSelectJoint]
  );
  const handleJointDbl = useCallback((_i: number, pos: THREE.Vector3) => {
    setFocusOverride(pos.clone());
    setNonce((x) => x + 1);
  }, []);

  if (!telemetry || n === 0) {
    return (
      <div className={`v-stage flex items-center justify-center ${heightClass ?? "h-[420px]"} w-full`}>
        <span className="v-label">3D pose not available for this clip</span>
      </div>
    );
  }

  const hasCom = !!comTrail;
  const camMarker = camWorld && preset !== "match" && !compact ? camWorld : null;
  const creaseCalibrated =
    telemetry.is_crease_calibrated ??
    (telemetry.pitch_display_mode === "calibrated_mcc"
      ? true
      : telemetry.pitch_display_mode === "bowler_centered"
      ? false
      : !!telemetry.mcc_wireframe_2d && Object.keys(telemetry.mcc_wireframe_2d).length > 0);
  const shouldAutoRotate = idleOrbit && !reduced && !userPickedCamera && preset !== "top" && preset !== "match";

  return (
    <div className={`relative w-full overflow-hidden rounded-[6px] bg-[#fafaf9] shadow-[0_0_0_1px_rgba(0,0,0,0.08)] flex flex-col ${heightClass ?? "h-[420px]"}`}>
      {!compact && (
        <div className="flex flex-wrap items-center justify-between gap-2 px-2.5 py-2 border-b border-[#ebebeb] bg-white">
          <div className="v-seg" role="group" aria-label="Virtual camera">
            {CAMERA_PRESET_LABELS.map(([p, lbl]) => (
              <button
                key={p}
                type="button"
                aria-pressed={preset === p}
                onClick={() => {
                  setFocusOverride(null);
                  setPreset(p);
                }}
                title={p === "match" ? (camWorld ? "The real phone position from calibration" : "Calibrated phone position unavailable — showing side-on") : undefined}
              >
                {lbl}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1.5">
            <ToggleChip on={sigmaOn} onClick={() => setSigmaOn((v) => !v)} title="1-sigma 3D joint uncertainty spheres">
              σ
            </ToggleChip>
            <ToggleChip on={trailsOn.wrist} dot={SEG.wrist} onClick={() => setTrailsOn((t) => ({ ...t, wrist: !t.wrist }))} title="Bowling-wrist path">
              Wrist
            </ToggleChip>
            <ToggleChip on={trailsOn.head} dot={SEG.head} onClick={() => setTrailsOn((t) => ({ ...t, head: !t.head }))} title="Head path">
              Head
            </ToggleChip>
            <ToggleChip
              on={trailsOn.com && hasCom}
              dot={SEG.com}
              disabled={!hasCom}
              onClick={() => setTrailsOn((t) => ({ ...t, com: !t.com }))}
              title={hasCom ? "Centre-of-mass path" : "Centre-of-mass path is not measurable on this clip"}
            >
              COM
            </ToggleChip>
          </div>
        </div>
      )}

      <div className="relative flex-1 min-h-0">
        <CanvasErrorBoundary resetKey={`${n}|${strobeMode}|${bowlingArm}`}>
          <Canvas
            shadows="soft"
            dpr={[1, 2]}
            camera={{ fov: goal.fov, near: 0.05, far: 400, position: goal.pos.toArray() as [number, number, number] }}
            gl={{ antialias: true, preserveDrawingBuffer: true }}
            onPointerMissed={() => setSelectedJoint(null)}
            onCreated={({ gl, camera }) => {
              gl.toneMapping = THREE.ACESFilmicToneMapping;
              gl.toneMappingExposure = 1.05;
              camera.lookAt(goal.target);
              gl.domElement?.addEventListener?.("webglcontextlost", (ev) => ev.preventDefault(), false);
            }}
          >
            <color attach="background" args={["#fafaf9"]} />
            <StudioLights focus={focus} />
            <PitchScene showGrid={preset === "top"} focus={focus} creaseCalibrated={creaseCalibrated} />

            {!strobeMode && (
              <>
                <Mannequin
                  getPose={getLivePose}
                  arm={bowlingArm}
                  interactive
                  selectedJoint={selectedJoint}
                  onJointClick={handleJointClick}
                  onJointDoubleClick={handleJointDbl}
                />
                <ContactDiscs getPose={getLivePose} />
                {sigmaOn && <SigmaSpheres getPose={getLivePose} getSigma={getSigma} arm={bowlingArm} />}
              </>
            )}

            {strobeMode &&
              strobe.map(({ f, opacity }) => (
                <group key={`s${f}`}>
                  <Mannequin getPose={() => poseAt(f)} arm={bowlingArm} opacity={f === strobeBest ? 1 : opacity} castShadow={f === strobeBest} />
                  <ContactDiscs getPose={() => poseAt(f)} />
                </group>
              ))}

            {ghost && !strobeMode && <Mannequin getPose={getGhostPose} arm={bowlingArm} tint={SEG.ghost} opacity={GHOST_OPACITY} castShadow={false} />}

            {trailsOn.wrist && wristTrail && <TrailRibbon series={wristTrail} color={SEG.wrist} radius={0.009} getFrame={getFrame} reveal={!strobeMode} />}
            {trailsOn.head && headTrail && <TrailRibbon series={headTrail} color={SEG.head} radius={0.006} getFrame={getFrame} reveal={!strobeMode} />}
            {trailsOn.com && comTrail && <TrailRibbon series={comTrail} color={SEG.com} radius={0.008} getFrame={getFrame} reveal={!strobeMode} />}

            {camMarker && (
              <group position={camMarker.toArray() as [number, number, number]}>
                <mesh raycast={noRaycast}>
                  <boxGeometry args={[0.16, 0.1, 0.2]} />
                  <meshStandardMaterial color={SEG.ink} />
                </mesh>
              </group>
            )}

            <OrbitControls
              makeDefault
              enableDamping
              dampingFactor={0.08}
              minDistance={0.6}
              maxDistance={80}
              maxPolarAngle={Math.PI / 2 - 0.02}
              autoRotate={shouldAutoRotate}
              autoRotateSpeed={0.35}
              onStart={() => {
                setUserPickedCamera(true);
                cancelAnimRef.current += 1;
              }}
            />
            <CameraRig goal={goal} goalKey={goalKey} reduced={reduced} cancelRef={cancelAnimRef} />
          </Canvas>
        </CanvasErrorBoundary>

        {!creaseCalibrated && (
          <div className="pointer-events-none absolute top-2 left-2 max-w-[calc(100%-1rem)] px-2 py-0.5 rounded-[5px] bg-white/90 shadow-[0_0_0_1px_#ebebeb] font-mono text-[10px] text-[#666666]">
            Crease &amp; stumps hidden · clip not crease-calibrated (ground aligned to bowler)
          </div>
        )}

        {/* Legend (DOM, crisp at any DPR) */}
        {!compact && (
          <div className="pointer-events-none absolute left-2 bottom-2 flex flex-wrap items-center gap-x-2.5 gap-y-1 max-w-[calc(100%-1rem)] px-2 py-1 rounded-[5px] bg-white/90 shadow-[0_0_0_1px_#ebebeb] font-mono text-[10px] text-[#4d4d4d]">
            <LegendDot c={SEG.pelvis} label="Hips & legs" />
            <LegendDot c={SEG.thorax} label="Trunk" />
            <LegendDot c={SEG.arm} label="Bowling arm" />
            <LegendDot c={SEG.wrist} label="Wrist" />
            <LegendDot c={SEG.lead} label="Lead arm" />
            {sigmaOn && !strobeMode && <span>· σ spheres = 1-sigma joint uncertainty</span>}
            {strobeMode && <span>· {strobe.length} poses back-foot → follow-through</span>}
            {ghost && !strobeMode && (
              <span className="inline-flex items-center gap-1">
                <span className="inline-block w-2 h-2 rounded-full" style={{ background: SEG.ghost, opacity: 0.6 }} />
                {ghostLabel ?? "Reference"}
                {mirrorGhost ? " (mirrored)" : ""} · aligned at front-foot plant
              </span>
            )}
          </div>
        )}
        {!compact && preset === "match" && !camWorld && (
          <div className="pointer-events-none absolute right-2 bottom-2 px-2 py-1 rounded-[5px] bg-white/90 shadow-[0_0_0_1px_#ebebeb] font-mono text-[10px] text-[#8f8f8f]">
            Phone position not calibrated · showing side-on
          </div>
        )}
      </div>
    </div>
  );
};

const ToggleChip: React.FC<{ on: boolean; onClick: () => void; title: string; children: React.ReactNode; dot?: string; disabled?: boolean }> = ({
  on,
  onClick,
  title,
  children,
  dot,
  disabled,
}) => (
  <button
    type="button"
    aria-pressed={on}
    title={title}
    disabled={disabled}
    onClick={onClick}
    className={`inline-flex items-center gap-1 h-[26px] px-2 rounded-[5px] font-mono text-[11px] transition-colors duration-150 disabled:opacity-40 disabled:cursor-not-allowed ${
      on ? "bg-[#171717] text-white" : "bg-[#f4f4f5] text-[#4d4d4d] hover:text-[#171717]"
    }`}
  >
    {dot && <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: dot, boxShadow: on ? "0 0 0 1px #fff" : undefined }} />}
    {children}
  </button>
);

const LegendDot: React.FC<{ c: string; label: string }> = ({ c, label }) => (
  <span className="inline-flex items-center gap-1">
    <span className="inline-block w-2 h-2 rounded-full" style={{ background: c }} />
    {label}
  </span>
);

export default PitchStage3D;
