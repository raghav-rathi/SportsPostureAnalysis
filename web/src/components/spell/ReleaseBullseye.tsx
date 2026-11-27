import React, { useMemo, useState } from "react";
import type { ReleaseTunnel, ReleaseTunnelPoint } from "../../types";
import { SEG } from "../../theme";
import { navigate } from "../../router";
import { ObsBadge } from "../ui/PaperPrimitives";
import { CalmState, MONO, fmtNum, fmtSigned, wristRamp } from "./ui";

/**
 * TrackMan-style release-point bullseye (batter's view looking back at the bowler's end).
 * Uniform metres-per-pixel so MCC Law 8 stumps, per-ball ellipses and the tunnel ellipse are true to scale.
 * world_stump_m → origin = middle stump (calibrated). body_corridor_m → origin = planted front ankle (no stumps drawn).
 */

const W = 600;
const H = 440;
const M = { l: 46, r: 14, t: 14, b: 44 };
const PW = W - M.l - M.r;
const PH = H - M.t - M.b;

const LAW8 = { height: 0.7112, outer_width: 0.2286, center_half_spacing: 0.096 };
const STUMP_DIAM = 0.0365; // Law 8: ≤ 1.50 in, ≥ 1.38 in
const BAIL_RISE = 0.0127; // bails project ≤ 0.5 in above the stumps

export const MODE_LABEL: Record<ReleaseTunnel["mode"], string> = {
  world_stump_m: "From middle stump (calibrated)",
  body_corridor_m: "Relative to front ankle (not stump-calibrated)",
};

function gridStep(span: number): number {
  if (span <= 1.4) return 0.25;
  if (span <= 3.5) return 0.5;
  return 1;
}

export const ReleaseBullseye: React.FC<{ tunnel: ReleaseTunnel; bowlingArm?: "right" | "left" | null }> = ({
  tunnel,
  bowlingArm,
}) => {
  const [hover, setHover] = useState<ReleaseTunnelPoint | null>(null);
  const pts = useMemo(() => [...(tunnel.points ?? [])].sort((a, b) => a.order - b.order), [tunnel.points]);
  const world = tunnel.mode === "world_stump_m";
  const dims = tunnel.stump_dims_m ?? LAW8;

  const geo = useMemo(() => {
    let latMin = world ? -0.45 : -0.6;
    let latMax = world ? 0.45 : 0.6;
    let hMax = 2.6;
    const take = (lat: number, h: number, r: number) => {
      latMin = Math.min(latMin, lat - r - 0.15);
      latMax = Math.max(latMax, lat + r + 0.15);
      hMax = Math.max(hMax, h + r + 0.2);
    };
    pts.forEach((p) => take(p.lateral_m, p.height_m, p.ellipse_95?.semi_major_m ?? 0));
    if (tunnel.centroid) take(tunnel.centroid.lateral_m, tunnel.centroid.height_m, tunnel.ellipse_95?.semi_major_m ?? 0);
    const s = Math.min(PW / (latMax - latMin), PH / hMax);
    const latMid = (latMin + latMax) / 2;
    const latHalf = PW / s / 2;
    const lo = latMid - latHalf;
    const hi = latMid + latHalf;
    const top = PH / s;
    const X = (lat: number) => M.l + (lat - lo) * s;
    const Y = (h: number) => M.t + PH - h * s;
    return { s, lo, hi, top, X, Y, step: gridStep(Math.max(hi - lo, top)) };
  }, [pts, tunnel.centroid, tunnel.ellipse_95, world]);

  const { s, lo, hi, top, X, Y, step } = geo;
  const latTicks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) latTicks.push(Number(v.toFixed(3)));
  const hTicks: number[] = [];
  for (let v = step; v <= top + 1e-9; v += step) hTicks.push(Number(v.toFixed(3)));

  const n = pts.length;
  const colorFor = (i: number) => wristRamp(n > 1 ? 0.12 + (0.88 * i) / (n - 1) : 0.6);
  const open = (p: ReleaseTunnelPoint) => navigate({ name: "delivery", id: p.delivery_id, tab: "story" });

  const debiasedCm = tunnel.debiased_radial_sd_m !== null ? tunnel.debiased_radial_sd_m * 100 : null;
  const rawCm = tunnel.raw_radial_sd_m !== null ? tunnel.raw_radial_sd_m * 100 : null;
  const plusSide = bowlingArm ? `${bowlingArm}-arm side` : "bowler's arm side";

  return (
    <div>
      <div className="flex flex-wrap items-end justify-between gap-3 mb-2">
        <div>
          <div className="v-label">Release spread (debiased)</div>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="v-hero-num text-[40px] text-[#171717]">{debiasedCm !== null ? fmtNum(debiasedCm, 1) : "—"}</span>
            <span className="text-[14px] text-[#4d4d4d]">cm</span>
          </div>
          <div className="v-num text-[12px] text-[#8f8f8f] mt-1">
            raw {rawCm !== null ? `${fmtNum(rawCm, 1)} cm` : "—"} · measurement noise removed · {n} ball{n === 1 ? "" : "s"}
          </div>
        </div>
        <span className="v-chip" title={tunnel.mode_label_plain ?? undefined}>
          {MODE_LABEL[tunnel.mode] ?? tunnel.mode_label_plain ?? tunnel.mode}
        </span>
      </div>

      {n === 0 ? (
        <CalmState
          title="No release points to plot yet"
          body="Release points appear here once balls in this spell have a measured release."
        />
      ) : (
        <div className="relative">
          <svg
            viewBox={`0 0 ${W} ${H}`}
            className="w-full h-auto select-none"
            role="img"
            aria-label="Release point bullseye, batter's view"
            onMouseLeave={() => setHover(null)}
          >
            <rect x={M.l} y={M.t} width={PW} height={PH} fill={SEG.grass} rx={4} />
            {/* metre gridlines */}
            {hTicks.map((h) => (
              <g key={`h${h}`}>
                <line x1={M.l} x2={M.l + PW} y1={Y(h)} y2={Y(h)} stroke="#000" strokeOpacity={0.07} />
                <text x={M.l - 6} y={Y(h) + 3} fontSize={10} textAnchor="end" fill="#8f8f8f" fontFamily={MONO}>
                  {h.toFixed(step < 0.5 ? 2 : 1)}
                </text>
              </g>
            ))}
            {latTicks.map((v) => (
              <g key={`l${v}`}>
                <line
                  x1={X(v)}
                  x2={X(v)}
                  y1={M.t}
                  y2={M.t + PH}
                  stroke="#000"
                  strokeOpacity={Math.abs(v) < 1e-6 ? 0.16 : 0.07}
                  strokeDasharray={Math.abs(v) < 1e-6 ? "2 3" : undefined}
                />
                <text x={X(v)} y={M.t + PH + 14} fontSize={10} textAnchor="middle" fill="#8f8f8f" fontFamily={MONO}>
                  {Math.abs(v) < 1e-6 ? "0" : fmtSigned(v, step < 0.5 ? 2 : 1)}
                </text>
              </g>
            ))}
            <text x={12} y={M.t + PH / 2} fontSize={10} fill="#666" fontFamily={MONO} transform={`rotate(-90 12 ${M.t + PH / 2})`} textAnchor="middle">
              RELEASE HEIGHT (m)
            </text>
            <text x={M.l} y={H - 8} fontSize={10} fill="#666" fontFamily={MONO}>
              ← − non-arm side
            </text>
            <text x={M.l + PW} y={H - 8} fontSize={10} fill="#666" fontFamily={MONO} textAnchor="end">
              + {plusSide} →
            </text>
            <text x={M.l + PW / 2} y={H - 8} fontSize={10} fill="#666" fontFamily={MONO} textAnchor="middle">
              LATERAL (m)
            </text>

            {/* ground / popping-crease line */}
            <line x1={M.l} x2={M.l + PW} y1={Y(0)} y2={Y(0)} stroke="#171717" strokeWidth={1.4} />
            <text x={M.l + PW - 4} y={Y(0) - 5} fontSize={9} textAnchor="end" fill="#4d4d4d" fontFamily={MONO}>
              {world ? "popping crease · ground" : "ground"}
            </text>

            {world ? (
              <g aria-label="Stumps to MCC Law 8 scale">
                {[-1, 0, 1].map((k) => {
                  const c = k * dims.center_half_spacing;
                  return (
                    <rect
                      key={k}
                      x={X(c - STUMP_DIAM / 2)}
                      y={Y(dims.height)}
                      width={Math.max(STUMP_DIAM * s, 1.5)}
                      height={dims.height * s}
                      fill="#ffffff"
                      stroke="#171717"
                      strokeWidth={0.8}
                    />
                  );
                })}
                {[-1, 1].map((k) => {
                  const x0 = k < 0 ? -dims.outer_width / 2 : 0;
                  return (
                    <rect
                      key={`bail${k}`}
                      x={X(x0) + (k < 0 ? 0 : 1)}
                      y={Y(dims.height + BAIL_RISE)}
                      width={Math.max((dims.outer_width / 2) * s - 1, 2)}
                      height={Math.max(BAIL_RISE * s, 1.5)}
                      fill="#ffffff"
                      stroke="#171717"
                      strokeWidth={0.8}
                    />
                  );
                })}
                <text x={X(dims.outer_width / 2) + 6} y={Y(dims.height) + 10} fontSize={9} fill="#4d4d4d" fontFamily={MONO}>
                  stumps · Law 8 ({dims.height.toFixed(3)} × {dims.outer_width.toFixed(4)} m)
                </text>
              </g>
            ) : (
              <g aria-label="Front ankle reference">
                <line x1={X(0)} x2={X(0)} y1={Y(0)} y2={M.t} stroke="#171717" strokeOpacity={0.6} strokeDasharray="5 4" />
                <circle cx={X(0)} cy={Y(0)} r={4} fill="#fff" stroke="#171717" />
                <text x={X(0) + 6} y={M.t + 12} fontSize={9} fill="#4d4d4d" fontFamily={MONO}>
                  planted front ankle (reference, not stumps)
                </text>
              </g>
            )}

            {/* tunnel ellipse + centroid */}
            {tunnel.centroid && tunnel.ellipse_95 && (
              <ellipse
                cx={X(tunnel.centroid.lateral_m)}
                cy={Y(tunnel.centroid.height_m)}
                rx={Math.max(tunnel.ellipse_95.semi_major_m * s, 2)}
                ry={Math.max(tunnel.ellipse_95.semi_minor_m * s, 2)}
                transform={`rotate(${-tunnel.ellipse_95.angle_deg} ${X(tunnel.centroid.lateral_m)} ${Y(tunnel.centroid.height_m)})`}
                fill={SEG.wrist}
                fillOpacity={0.05}
                stroke={SEG.wrist}
                strokeOpacity={0.7}
                strokeDasharray="4 3"
              />
            )}

            {/* per-ball ellipses first, then dots on top */}
            {pts.map((p, i) =>
              p.ellipse_95 ? (
                <ellipse
                  key={`e-${p.delivery_id}`}
                  cx={X(p.lateral_m)}
                  cy={Y(p.height_m)}
                  rx={Math.max(p.ellipse_95.semi_major_m * s, 1.5)}
                  ry={Math.max(p.ellipse_95.semi_minor_m * s, 1.5)}
                  transform={`rotate(${-p.ellipse_95.angle_deg} ${X(p.lateral_m)} ${Y(p.height_m)})`}
                  fill="none"
                  stroke={colorFor(i)}
                  strokeOpacity={hover && hover.delivery_id === p.delivery_id ? 0.9 : 0.4}
                  strokeWidth={1}
                />
              ) : null
            )}
            {tunnel.centroid && (
              <g pointerEvents="none">
                <circle cx={X(tunnel.centroid.lateral_m)} cy={Y(tunnel.centroid.height_m)} r={8} fill="none" stroke="#171717" strokeWidth={1.2} />
                <line
                  x1={X(tunnel.centroid.lateral_m) - 12}
                  x2={X(tunnel.centroid.lateral_m) + 12}
                  y1={Y(tunnel.centroid.height_m)}
                  y2={Y(tunnel.centroid.height_m)}
                  stroke="#171717"
                  strokeWidth={0.8}
                />
                <line
                  x1={X(tunnel.centroid.lateral_m)}
                  x2={X(tunnel.centroid.lateral_m)}
                  y1={Y(tunnel.centroid.height_m) - 12}
                  y2={Y(tunnel.centroid.height_m) + 12}
                  stroke="#171717"
                  strokeWidth={0.8}
                />
              </g>
            )}
            {pts.map((p, i) => {
              const active = hover?.delivery_id === p.delivery_id;
              return (
                <g
                  key={p.delivery_id}
                  role="button"
                  tabIndex={0}
                  aria-label={`${p.label}: open this ball`}
                  style={{ cursor: "pointer" }}
                  onMouseEnter={() => setHover(p)}
                  onFocus={() => setHover(p)}
                  onBlur={() => setHover(null)}
                  onClick={() => open(p)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      open(p);
                    }
                  }}
                >
                  <circle cx={X(p.lateral_m)} cy={Y(p.height_m)} r={12} fill="transparent" />
                  <circle
                    cx={X(p.lateral_m)}
                    cy={Y(p.height_m)}
                    r={active ? 6.5 : 5}
                    fill={colorFor(i)}
                    stroke="#ffffff"
                    strokeWidth={1.5}
                  />
                </g>
              );
            })}
          </svg>

          {hover && (
            <div
              className="pointer-events-none absolute z-10 v-panel px-3 py-2 text-[12px] leading-snug"
              style={{
                left: `${(X(hover.lateral_m) / W) * 100}%`,
                top: `${(Y(hover.height_m) / H) * 100}%`,
                transform: "translate(-50%, calc(-100% - 14px))",
                minWidth: 170,
              }}
            >
              <div className="font-medium text-[#171717]">{hover.label}</div>
              <div className="v-num text-[#4d4d4d] mt-0.5">
                height {fmtNum(hover.height_m, 2)} m
                {hover.sigma_height_m != null ? ` ± ${fmtNum(hover.sigma_height_m * 100, 1)} cm` : ""}
              </div>
              <div className="v-num text-[#4d4d4d]">
                lateral {fmtSigned(hover.lateral_m, 2)} m
                {hover.sigma_lateral_m != null ? ` ± ${fmtNum(hover.sigma_lateral_m * 100, 1)} cm` : ""}
              </div>
              {hover.observability && (
                <div className="mt-1">
                  <ObsBadge grade={hover.observability} compact />
                </div>
              )}
              <div className="text-[11px] text-[#8f8f8f] mt-1">Click to open this ball</div>
            </div>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] text-[#666]">
        <span className="inline-flex items-center gap-1.5">
          <svg width={54} height={10} aria-hidden>
            {[0, 1, 2, 3, 4].map((k) => (
              <circle key={k} cx={5 + k * 11} cy={5} r={4} fill={wristRamp(0.12 + k * 0.22)} />
            ))}
          </svg>
          ball order: first → last
        </span>
        <span className="inline-flex items-center gap-1.5">
          <svg width={22} height={12} aria-hidden>
            <ellipse cx={11} cy={6} rx={9} ry={4.5} fill="none" stroke={SEG.wrist} strokeDasharray="3 2" />
          </svg>
          95% region of the spell
        </span>
        <span>thin rings = each ball's 95% measurement region</span>
        <span>⊕ = spell centre</span>
      </div>

      {(tunnel.excluded?.length ?? 0) > 0 && (
        <div className="mt-4">
          <div className="v-label mb-1.5">Height only · not calibrated ({tunnel.excluded!.length})</div>
          <ul className="divide-y divide-[#ebebeb] rounded-[6px]" style={{ boxShadow: "0 0 0 1px #ebebeb" }}>
            {tunnel.excluded!.map((x) => (
              <li key={x.delivery_id} className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-0.5 px-3 py-2">
                <button
                  type="button"
                  className="text-[13px] font-medium text-[#171717] underline decoration-[#d4d4d4] underline-offset-2 hover:decoration-[#171717]"
                  onClick={() => navigate({ name: "delivery", id: x.delivery_id, tab: "story" })}
                >
                  {x.label}
                </button>
                <span className="text-[12px] text-[#666] leading-snug">{x.reason_plain}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default ReleaseBullseye;
