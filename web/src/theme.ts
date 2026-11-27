/**
 * Studio v5 "Paper Instrument" theme constants for canvas / three.js / uPlot.
 * Mirrors the CSS variables in index.css. Segment colours are for INSIDE canvases, 3D and charts only.
 * (docs/research_app_landscape_v5.md §6.2)
 */
import type { CheckpointStatus, ObservabilityGrade } from "./types";

export const SEG = {
  pelvis: "#059669",
  pelvisTint: "#D1FAE5",
  thorax: "#2451D6",
  thoraxTint: "#DBE4FB",
  arm: "#D97706",
  armTint: "#FEF3C7",
  wrist: "#DC2626",
  wristTint: "#FEE2E2",
  lead: "#7C9BEA",
  head: "#171717",
  com: "#525252",
  ghost: "#64748B",
  ghostTint: "#E2E8F0",
  tele: "#C026D3",
  grass: "#EEF3EA",
  ink: "#171717",
  graphite: "#8F8F8F",
  hairline: "#EBEBEB",
  positive: "#297A3A",
} as const;

export const GHOST_OPACITY = 0.45;

/** Canonical SMPL-24 joint order (src/bowling_biomech/constants.py SMPL24_JOINTS). */
export const SMPL24_JOINTS = [
  "pelv", "lhip", "rhip", "spi1", "lkne", "rkne", "spi2", "lank",
  "rank", "spi3", "ltoe", "rtoe", "neck", "lcla", "rcla", "head",
  "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhan", "rhan",
] as const;
export type Smpl24Joint = (typeof SMPL24_JOINTS)[number];
export const J: Record<Smpl24Joint, number> = Object.fromEntries(SMPL24_JOINTS.map((n, i) => [n, i])) as Record<
  Smpl24Joint,
  number
>;

export type SegmentKey = "pelvis" | "thorax" | "arm" | "wrist" | "lead" | "head";

/** SMPL-24 kinematic tree as [parent, child] index pairs. */
export const SMPL24_BONES: Array<[number, number]> = [
  [J.pelv, J.lhip], [J.pelv, J.rhip], [J.pelv, J.spi1],
  [J.lhip, J.lkne], [J.rhip, J.rkne], [J.spi1, J.spi2],
  [J.lkne, J.lank], [J.rkne, J.rank], [J.spi2, J.spi3],
  [J.lank, J.ltoe], [J.rank, J.rtoe], [J.spi3, J.neck],
  [J.spi3, J.lcla], [J.spi3, J.rcla], [J.neck, J.head],
  [J.lcla, J.lsho], [J.rcla, J.rsho], [J.lsho, J.lelb],
  [J.rsho, J.relb], [J.lelb, J.lwri], [J.relb, J.rwri],
  [J.lwri, J.lhan], [J.rwri, J.rhan],
];

/** Segment for a bone (colour by child joint), arm-aware: bowling upper arm + forearm = amber, bowling hand = crimson,
 *  non-bowling arm = cobalt-300, legs = emerald, head link = ink, everything else (spine, clavicles) = cobalt. */
export function boneSegment(_parent: number, child: number, arm: "right" | "left"): SegmentKey {
  const b = arm === "right" ? "r" : "l";
  const name = SMPL24_JOINTS[child];
  if (["lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"].includes(name)) return "pelvis";
  if (name === "head") return "head";
  if (name === `${b}elb` || name === `${b}wri`) return "arm";
  if (name === `${b}han`) return "wrist";
  if (name.endsWith("elb") || name.endsWith("wri") || name.endsWith("han")) return "lead";
  return "thorax";
}

/** Segment for a joint marker. */
export function jointSegment(idx: number, arm: "right" | "left"): SegmentKey {
  const b = arm === "right" ? "r" : "l";
  const n = SMPL24_JOINTS[idx];
  if (["pelv", "lhip", "rhip", "lkne", "rkne", "lank", "rank", "ltoe", "rtoe"].includes(n)) return "pelvis";
  if (n === "head") return "head";
  if (n === `${b}wri` || n === `${b}han`) return "wrist";
  if (n === `${b}elb`) return "arm";
  if (n.endsWith("elb") || n.endsWith("wri") || n.endsWith("han")) return "lead";
  return "thorax";
}

export const SEG_COLOR: Record<SegmentKey, string> = {
  pelvis: SEG.pelvis,
  thorax: SEG.thorax,
  arm: SEG.arm,
  wrist: SEG.wrist,
  lead: SEG.lead,
  head: SEG.head,
};

/** Kinematic-sequence lane colours (backend `SequenceSegment` ids + sequence_series keys). */
export const SEQUENCE_LANES = [
  { segment: "pelvis", seriesKey: "pelvis_omega_dps", label: "Hips", color: SEG.pelvis },
  { segment: "thorax", seriesKey: "thorax_omega_dps", label: "Chest", color: SEG.thorax },
  { segment: "bowling_arm", seriesKey: "arm_omega_dps", label: "Bowling arm", color: SEG.arm },
  { segment: "wrist_whip", seriesKey: "wrist_whip_dps", label: "Wrist whip", color: SEG.wrist },
] as const;

/** Status → glyph (status is never encoded with segment hues). */
export const STATUS_GLYPH: Record<CheckpointStatus, string> = {
  strong: "●",
  on_track: "●",
  watch: "◇",
  focus: "◆",
  not_visible: "○",
};

/** Observability → bar glyph + plain word. */
export const OBS_GLYPH: Record<ObservabilityGrade, { bars: string; word: string }> = {
  high: { bars: "▮▮▮", word: "Clear" },
  moderate: { bars: "▮▮▯", word: "Fair" },
  low: { bars: "▮▯▯", word: "Rough" },
  unmeasurable: { bars: "⊘", word: "Not measurable" },
  refused: { bars: "🔒", word: "Refused" },
};

/** Never-truncated guardrail copy (AGENTS.md §1.2). Render in full wherever analysis is shown or exported. */
export const GUARDRAIL_ICC =
  "Elbow legality (ICC 15° rule) is never judged here: it needs an accredited laboratory 3D test at 250 Hz or faster. Phone video cannot decide it, so this check is permanently refused.";
export const GUARDRAIL_MEDICAL =
  "Load and body-stress notes describe one delivery's mechanics and your logged workload only. They are not a medical diagnosis. If you have back pain when bowling, stop and see a sports physiotherapist or doctor.";
