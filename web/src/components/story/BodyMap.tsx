import React from "react";
import type { BodyCheckpoint, BodyZone, CheckpointId } from "../../types";
import { StatusDot } from "./ui";

/**
 * Clickable bowler body map — a calm side-view mannequin in the delivery stride
 * with 5 hotspot zones. Mirrored for left-arm bowlers.
 */
const ZONE_POS: Record<BodyZone, { x: number; y: number; lx: number; ly: number; anchor: "start" | "end" }> = {
  shoulders: { x: 112, y: 62, lx: 178, ly: 52, anchor: "start" },
  trunk: { x: 116, y: 96, lx: 178, ly: 100, anchor: "start" },
  spine: { x: 88, y: 108, lx: 22, ly: 100, anchor: "end" },
  knee: { x: 131, y: 178, lx: 178, ly: 172, anchor: "start" },
  approach: { x: 70, y: 252, lx: 22, ly: 226, anchor: "end" },
};

export const BodyMap: React.FC<{
  checkpoints: BodyCheckpoint[];
  selectedId: CheckpointId | null;
  bowlingArm: "right" | "left";
  onSelect: (id: CheckpointId) => void;
}> = ({ checkpoints, selectedId, bowlingArm, onSelect }) => {
  const mirror = bowlingArm === "left";
  const limb = { stroke: "#e4e4e7", strokeWidth: 13, strokeLinecap: "round" as const, fill: "none" };
  const limbBack = { ...limb, stroke: "#efeff1" };
  // Labels must never be mirrored, so only the figure group is flipped.
  const fx = (x: number) => (mirror ? 200 - x : x);

  return (
    <svg viewBox="-84 0 368 268" className="w-full h-auto max-h-[300px]" role="group" aria-label="Body map. Tap a zone to see that checkpoint.">
      {/* Ground */}
      <line x1="-40" y1="244" x2="240" y2="244" stroke="#ebebeb" strokeWidth="1.5" />
      <g transform={mirror ? "translate(200,0) scale(-1,1)" : undefined}>
        {/* Back leg & back arm (lighter) */}
        <polyline points="90,128 70,180 42,226 28,236" {...limbBack} />
        <polyline points="100,64 122,92 112,114" {...limbBack} strokeWidth={10} />
        {/* Torso */}
        <path d="M95 56 Q110 50 124 58 L110 132 Q96 136 84 128 Z" fill="#e4e4e7" stroke="#e4e4e7" strokeWidth="6" strokeLinejoin="round" />
        {/* Front leg */}
        <polyline points="104,130 131,178 150,238 166,240" {...limb} />
        {/* Bowling arm (overhead) */}
        <polyline points="118,60 126,30 134,6" {...limb} strokeWidth={10} />
        {/* Head */}
        <circle cx="114" cy="36" r="13" fill="#e4e4e7" />
        {/* Approach arrow */}
        <path d="M-10 254 L60 254" stroke="#c9c9c9" strokeWidth="1.5" strokeDasharray="4 3" />
        <path d="M56 250 L62 254 L56 258" stroke="#c9c9c9" strokeWidth="1.5" fill="none" />
      </g>

      {checkpoints.map((cp) => {
        const z = ZONE_POS[cp.body_zone];
        if (!z) return null;
        const x = fx(z.x);
        const lx = mirror ? 200 - z.lx : z.lx;
        const anchor = mirror ? (z.anchor === "start" ? "end" : "start") : z.anchor;
        const selected = cp.id === selectedId;
        const ring = selected ? (cp.status === "strong" ? "#297a3a" : "#171717") : "#c9c9c9";
        return (
          <g
            key={cp.id}
            role="button"
            tabIndex={0}
            aria-label={`${cp.title}: ${cp.status_label}`}
            aria-pressed={selected}
            className="cursor-pointer focus:outline-none"
            onClick={() => onSelect(cp.id)}
            onKeyDown={(e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onSelect(cp.id);
              }
            }}
          >
            <line x1={x} y1={z.y} x2={lx + (anchor === "start" ? -4 : 4)} y2={z.ly - 4} stroke={selected ? ring : "#d4d4d8"} strokeWidth="1" />
            <circle cx={x} cy={z.y} r={selected ? 13 : 11} fill="#ffffff" stroke={ring} strokeWidth={selected ? 2 : 1.2} />
            <foreignObject x={x - 6} y={z.y - 6} width="12" height="12" style={{ pointerEvents: "none" }}>
              <StatusDot status={cp.status} size={12} />
            </foreignObject>
            <text x={lx} y={z.ly - 6} textAnchor={anchor} fontSize="11" fontWeight={selected ? 500 : 400} fill="#171717" fontFamily="Geist, Inter, sans-serif">
              {cp.title}
            </text>
            <text x={lx} y={z.ly + 8} textAnchor={anchor} fontSize="10" fill={cp.status === "strong" ? "#297a3a" : "#666666"} fontFamily="Geist, Inter, sans-serif">
              {cp.status_label}
            </text>
          </g>
        );
      })}
    </svg>
  );
};
