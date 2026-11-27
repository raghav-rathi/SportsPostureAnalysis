import React, { useEffect, useRef, useState } from "react";
import { SEG } from "../../theme";

/**
 * Animated biomechanical form schematic (side-on, travelling right) for a drill card.
 * Sculpted capsule limbs in paper greys; the drill's target segment is drawn in its SEG colour.
 * Loops a short keyframe path for the drill's phase over ~4 s; static under prefers-reduced-motion.
 */

export type PracticePhase = "gather" | "back_foot_land" | "front_foot_plant" | "release" | "follow_through";
export type SchematicSegment = "pelvis" | "thorax" | "arm" | "wrist" | "lead";

type JointKey = "head" | "neck" | "pelvis" | "bKnee" | "bAnkle" | "fKnee" | "fAnkle" | "bElb" | "bWri" | "lElb" | "lWri";
type Pose = Record<JointKey, [number, number]>;

const POSES: Pose[] = [
  // 0 gather
  { head: [96, 36], neck: [98, 52], pelvis: [100, 98], bKnee: [92, 128], bAnkle: [88, 160], fKnee: [116, 122], fAnkle: [108, 146], bElb: [90, 74], bWri: [96, 92], lElb: [112, 40], lWri: [118, 26] },
  // 1 back-foot landing
  { head: [88, 40], neck: [92, 56], pelvis: [100, 100], bKnee: [102, 130], bAnkle: [100, 164], fKnee: [122, 118], fAnkle: [124, 142], bElb: [74, 74], bWri: [62, 92], lElb: [110, 38], lWri: [122, 24] },
  // 2 front-foot plant
  { head: [104, 42], neck: [104, 58], pelvis: [104, 102], bKnee: [82, 128], bAnkle: [66, 160], fKnee: [132, 130], fAnkle: [152, 164], bElb: [80, 48], bWri: [62, 40], lElb: [124, 52], lWri: [136, 62] },
  // 3 release
  { head: [122, 46], neck: [116, 60], pelvis: [110, 102], bKnee: [96, 130], bAnkle: [84, 158], fKnee: [136, 132], fAnkle: [152, 164], bElb: [114, 34], bWri: [118, 10], lElb: [110, 82], lWri: [100, 96] },
  // 4 follow-through
  { head: [146, 72], neck: [138, 82], pelvis: [122, 110], bKnee: [132, 134], bAnkle: [118, 160], fKnee: [146, 138], fAnkle: [152, 164], bElb: [144, 108], bWri: [140, 130], lElb: [116, 98], lWri: [110, 116] },
];

const PHASE_PATH: Record<PracticePhase | "all", number[]> = {
  gather: [0, 1],
  back_foot_land: [0, 1, 2],
  front_foot_plant: [1, 2, 3],
  release: [2, 3, 4],
  follow_through: [3, 4],
  all: [0, 1, 2, 3, 4],
};

const KEY_POSE: Record<PracticePhase | "all", number> = {
  gather: 0,
  back_foot_land: 1,
  front_foot_plant: 2,
  release: 3,
  follow_through: 4,
  all: 3,
};

const ease = (x: number) => 0.5 - 0.5 * Math.cos(Math.PI * x);

function lerpPose(a: Pose, b: Pose, t: number): Pose {
  const out = {} as Pose;
  (Object.keys(a) as JointKey[]).forEach((k) => {
    out[k] = [a[k][0] + (b[k][0] - a[k][0]) * t, a[k][1] + (b[k][1] - a[k][1]) * t];
  });
  return out;
}

function poseAt(path: number[], u: number): Pose {
  // ping-pong over the path, holding briefly at the key pose
  const legs = path.length - 1;
  if (legs <= 0) return POSES[path[0]];
  const pp = u < 0.5 ? u * 2 : 2 - u * 2;
  const x = pp * legs;
  const i = Math.min(legs - 1, Math.floor(x));
  return lerpPose(POSES[path[i]], POSES[path[i + 1]], ease(x - i));
}

function usePrefersReducedMotion() {
  const [r, setR] = useState(false);
  useEffect(() => {
    const m = window.matchMedia?.("(prefers-reduced-motion: reduce)");
    if (!m) return;
    setR(m.matches);
    const h = () => setR(m.matches);
    m.addEventListener?.("change", h);
    return () => m.removeEventListener?.("change", h);
  }, []);
  return r;
}

const Limb: React.FC<{ a: [number, number]; b: [number, number]; color?: string; w?: number }> = ({ a, b, color, w = 9 }) => (
  <g>
    <line x1={a[0]} y1={a[1]} x2={b[0]} y2={b[1]} stroke={color ? color : "#cfcfcf"} strokeWidth={w + 2.5} strokeLinecap="round" />
    <line
      x1={a[0]}
      y1={a[1]}
      x2={b[0]}
      y2={b[1]}
      stroke={color ? "rgba(255,255,255,0.28)" : "#ececec"}
      strokeWidth={Math.max(2, w - 4)}
      strokeLinecap="round"
    />
  </g>
);

export const FormSchematic: React.FC<{
  phase: PracticePhase | "all";
  segment: SchematicSegment;
  size?: number;
  playing?: boolean;
}> = ({ phase, segment, size = 168, playing = true }) => {
  const reduced = usePrefersReducedMotion();
  const [u, setU] = useState(0.5);
  const raf = useRef<number | null>(null);
  const path = PHASE_PATH[phase];

  useEffect(() => {
    if (reduced || !playing) {
      setU(0.5);
      return;
    }
    const start = performance.now();
    const tick = (now: number) => {
      setU(((now - start) % 4000) / 4000);
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => {
      if (raf.current) cancelAnimationFrame(raf.current);
    };
  }, [reduced, playing]);

  const p = reduced || !playing ? POSES[KEY_POSE[phase]] : poseAt(path, u);
  const col = (s: SchematicSegment) => (s === segment ? SEG[s] : undefined);
  const shoulder: [number, number] = [p.neck[0], p.neck[1] + 4];

  return (
    <svg viewBox="40 0 140 176" width={size} height={size * (176 / 140)} role="img" aria-label="Animated drill form schematic">
      <line x1={44} x2={176} y1={167} y2={167} stroke={SEG.hairline} strokeWidth={2} />
      <ellipse cx={(p.bAnkle[0] + p.fAnkle[0]) / 2} cy={168} rx={26} ry={3} fill="rgba(0,0,0,0.06)" />
      {/* lead arm (behind) */}
      <Limb a={shoulder} b={p.lElb} color={col("lead")} w={7} />
      <Limb a={p.lElb} b={p.lWri} color={col("lead")} w={6} />
      {/* legs */}
      <Limb a={p.pelvis} b={p.bKnee} color={col("pelvis")} w={10} />
      <Limb a={p.bKnee} b={p.bAnkle} color={col("pelvis")} w={8} />
      <Limb a={p.pelvis} b={p.fKnee} color={col("pelvis")} w={10} />
      <Limb a={p.fKnee} b={p.fAnkle} color={col("pelvis")} w={8} />
      {/* trunk */}
      <Limb a={p.pelvis} b={p.neck} color={col("thorax")} w={15} />
      {/* bowling arm (front) */}
      <Limb a={shoulder} b={p.bElb} color={col("arm")} w={7.5} />
      <Limb a={p.bElb} b={p.bWri} color={col("arm")} w={6.5} />
      <circle cx={p.bWri[0]} cy={p.bWri[1]} r={segment === "wrist" ? 5.5 : 4} fill={segment === "wrist" ? SEG.wrist : "#cfcfcf"} />
      {/* head */}
      <circle cx={p.head[0]} cy={p.head[1]} r={9} fill="#e4e4e4" stroke="#cfcfcf" strokeWidth={2} />
      <circle cx={p.pelvis[0]} cy={p.pelvis[1]} r={6} fill={segment === "pelvis" ? SEG.pelvis : "#d9d9d9"} />
    </svg>
  );
};

export default FormSchematic;
