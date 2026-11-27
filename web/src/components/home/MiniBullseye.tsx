import React from "react";
import type { ReleaseTunnel } from "../../types";
import { SEG } from "../../theme";

/**
 * Release-point bullseye (batter's view): stumps drawn to MCC Law 8 scale, one crimson dot per ball
 * with its 95% ellipse, colour ramp light→dark by ball order. Falls back to a ghosted instrument with
 * the reason when there is no calibrated spell yet — never hidden, never invented.
 */
export const MiniBullseye: React.FC<{
  tunnel: ReleaseTunnel | null | undefined;
  height?: number;
  onPick?: (deliveryId: string, frameIdx: number | null | undefined) => void;
  emptyText?: string;
}> = ({ tunnel, height = 150, onPick, emptyText = "Release points appear after 2+ measured balls in a spell." }) => {
  const W = 220;
  const H = height;
  const pts = tunnel?.points ?? [];
  const hasData = pts.length >= 1;
  // Frame: lateral ±0.9 m around the centroid lateral (or 0), height 0..2.6 m.
  const cx = tunnel?.centroid?.lateral_m ?? 0;
  const latSpan = 0.9;
  const hMin = 0;
  const hMax = Math.max(2.6, ...pts.map((p) => p.height_m + 0.3));
  const X = (lat: number) => ((lat - (cx - latSpan)) / (2 * latSpan)) * W;
  const Y = (h: number) => H - ((h - hMin) / (hMax - hMin)) * H;
  const sx = W / (2 * latSpan);
  const sy = H / (hMax - hMin);
  const stumpsW = (tunnel?.stump_dims_m?.outer_width ?? 0.2286) * sx;
  const stumpsH = (tunnel?.stump_dims_m?.height ?? 0.7112) * sy;
  const stumpX = tunnel?.mode === "world_stump_m" ? X(0) : null;

  return (
    <div className="relative">
      <svg viewBox={`0 0 ${W} ${H}`} className={`w-full ${hasData ? "" : "v-ghosted"}`} style={{ height }} role="img" aria-label="Release point bullseye">
        <rect x={0} y={0} width={W} height={H} rx={6} fill={SEG.grass} />
        {[0.5, 1, 1.5, 2, 2.5].map((h) => (
          <g key={h}>
            <line x1={0} x2={W} y1={Y(h)} y2={Y(h)} stroke="#000" strokeOpacity={0.06} />
            <text x={4} y={Y(h) - 2} fontSize={8} fill="#8f8f8f" fontFamily="Geist Mono, monospace">
              {h.toFixed(1)} m
            </text>
          </g>
        ))}
        <line x1={0} x2={W} y1={Y(0)} y2={Y(0)} stroke="#171717" strokeOpacity={0.5} />
        {stumpX !== null && (
          <g>
            {[-1, 0, 1].map((k) => (
              <rect
                key={k}
                x={stumpX + k * (stumpsW / 2.4) - 1.2}
                y={Y(0) - stumpsH}
                width={2.4}
                height={stumpsH}
                fill="#ffffff"
                stroke="#171717"
                strokeWidth={0.6}
              />
            ))}
          </g>
        )}
        {tunnel?.ellipse_95 && tunnel.centroid && (
          <ellipse
            cx={X(tunnel.centroid.lateral_m)}
            cy={Y(tunnel.centroid.height_m)}
            rx={tunnel.ellipse_95.semi_major_m * sx}
            ry={tunnel.ellipse_95.semi_minor_m * sy}
            transform={`rotate(${-tunnel.ellipse_95.angle_deg} ${X(tunnel.centroid.lateral_m)} ${Y(tunnel.centroid.height_m)})`}
            fill={SEG.wrist}
            fillOpacity={0.06}
            stroke={SEG.wrist}
            strokeOpacity={0.5}
            strokeDasharray="3 3"
          />
        )}
        {pts.map((p, i) => {
          const a = pts.length > 1 ? 0.35 + 0.65 * (i / (pts.length - 1)) : 1;
          return (
            <g key={p.delivery_id} style={{ cursor: onPick ? "pointer" : "default" }} onClick={() => onPick?.(p.delivery_id, p.frame_idx)}>
              {p.ellipse_95 && (
                <ellipse
                  cx={X(p.lateral_m)}
                  cy={Y(p.height_m)}
                  rx={Math.max(p.ellipse_95.semi_major_m * sx, 1.5)}
                  ry={Math.max(p.ellipse_95.semi_minor_m * sy, 1.5)}
                  transform={`rotate(${-p.ellipse_95.angle_deg} ${X(p.lateral_m)} ${Y(p.height_m)})`}
                  fill="none"
                  stroke={SEG.wrist}
                  strokeOpacity={0.35 * a}
                />
              )}
              <circle cx={X(p.lateral_m)} cy={Y(p.height_m)} r={3.6} fill={SEG.wrist} fillOpacity={a} stroke="#fff" strokeWidth={1} />
              <title>
                {p.label}: height {p.height_m.toFixed(2)} m, lateral {p.lateral_m >= 0 ? "+" : ""}
                {p.lateral_m.toFixed(2)} m
              </title>
            </g>
          );
        })}
      </svg>
      {!hasData && (
        <div className="absolute inset-0 flex items-center justify-center p-4 text-center text-[12px] text-[#4d4d4d] leading-snug">
          {emptyText}
        </div>
      )}
    </div>
  );
};
