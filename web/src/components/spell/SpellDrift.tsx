import React from "react";
import type { FatigueDriftRow, RepeatabilityGrade, RepeatabilityRow } from "../../types";
import { SEG } from "../../theme";
import { navigate } from "../../router";
import { CalmState, MONO, Tag, fmtNum, fmtSigned, unitLabel, withUnit } from "./ui";

/**
 * Spell fatigue drift (S2) + repeatability. Statuses are rendered verbatim from the backend:
 * a trend line is drawn ONLY when the backend says `verified_drift`.
 */

const DW = 300;
const DH = 118;
const DM = { l: 40, r: 10, t: 10, b: 22 };

const DriftBadge: React.FC<{ row: FatigueDriftRow }> = ({ row }) => {
  if (row.status === "verified_drift") {
    return (
      <Tag variant="ink" title={row.status_plain}>
        ◆ Verified drift {row.spell_shift !== null ? `${fmtSigned(row.spell_shift)}${unitLabel(row.unit)}` : ""}
      </Tag>
    );
  }
  if (row.status === "insufficient_data") {
    return (
      <Tag variant="muted" title={row.status_plain}>
        ○ Not enough balls
      </Tag>
    );
  }
  return <Tag title={row.status_plain}>● Stable / within noise</Tag>;
};

export const DriftChart: React.FC<{ row: FatigueDriftRow; labels: Record<string, string> }> = ({ row, labels }) => {
  const pts = [...row.points].sort((a, b) => a.order - b.order);
  const valid = pts.filter((p) => p.value !== null && Number.isFinite(p.value));
  const ys: number[] = [];
  valid.forEach((p) => {
    ys.push(p.value as number);
    if (p.ci_low !== null) ys.push(p.ci_low);
    if (p.ci_high !== null) ys.push(p.ci_high);
  });
  const orders = pts.map((p) => p.order);
  const oMin = orders.length ? Math.min(...orders) : 1;
  const oMax = orders.length ? Math.max(...orders) : 1;
  let yMin = ys.length ? Math.min(...ys) : 0;
  let yMax = ys.length ? Math.max(...ys) : 1;
  if (yMax - yMin < 1e-9) {
    yMin -= 1;
    yMax += 1;
  }
  const pad = (yMax - yMin) * 0.1;
  yMin -= pad;
  yMax += pad;
  const pw = DW - DM.l - DM.r;
  const ph = DH - DM.t - DM.b;
  const X = (o: number) => DM.l + (oMax === oMin ? pw / 2 : ((o - oMin) / (oMax - oMin)) * pw);
  const Y = (v: number) => DM.t + ph - ((v - yMin) / (yMax - yMin)) * ph;

  const verified = row.status === "verified_drift" && row.slope_per_ball !== null && valid.length >= 2;
  let trend: { x1: number; y1: number; x2: number; y2: number } | null = null;
  if (verified) {
    // Backend slope, drawn through the centre of the plotted balls (display anchor only).
    const xbar = valid.reduce((a, p) => a + p.order, 0) / valid.length;
    const ybar = valid.reduce((a, p) => a + (p.value as number), 0) / valid.length;
    const slope = row.slope_per_ball as number;
    trend = { x1: X(oMin), y1: Y(ybar + slope * (oMin - xbar)), x2: X(oMax), y2: Y(ybar + slope * (oMax - xbar)) };
  }

  return (
    <div className="v-panel p-3">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[13px] font-medium text-[#171717] leading-snug">{row.label_plain}</div>
          <div className="v-num text-[11px] text-[#8f8f8f] mt-0.5">
            {row.n_balls} ball{row.n_balls === 1 ? "" : "s"}
            {row.slope_per_ball !== null && row.status === "verified_drift"
              ? ` · ${fmtSigned(row.slope_per_ball)}${unitLabel(row.unit)} per ball`
              : ""}
          </div>
        </div>
        <DriftBadge row={row} />
      </div>
      {valid.length === 0 ? (
        <div className="mt-2 text-[12px] text-[#8f8f8f]">No measured values for this metric in the spell.</div>
      ) : (
        <svg viewBox={`0 0 ${DW} ${DH}`} className="w-full h-auto mt-2" role="img" aria-label={`${row.label_plain} by ball order`}>
          {[yMin + pad, (yMin + yMax) / 2, yMax - pad].map((v, i) => (
            <g key={i}>
              <line x1={DM.l} x2={DW - DM.r} y1={Y(v)} y2={Y(v)} stroke="#000" strokeOpacity={0.06} />
              <text x={DM.l - 5} y={Y(v) + 3} fontSize={9} textAnchor="end" fill="#8f8f8f" fontFamily={MONO}>
                {fmtNum(v)}
              </text>
            </g>
          ))}
          {trend && (
            <line {...trend} stroke={SEG.thorax} strokeWidth={1.2} strokeDasharray="5 3" strokeOpacity={0.85} />
          )}
          {pts.map((p) => {
            const label = labels[p.delivery_id] ?? `Ball ${p.order}`;
            const x = X(p.order);
            return (
              <g
                key={`${p.delivery_id}-${p.order}`}
                style={{ cursor: "pointer" }}
                onClick={() => navigate({ name: "delivery", id: p.delivery_id, tab: "story" })}
              >
                <title>
                  {`${label}: ${withUnit(p.value, row.unit)}${
                    p.ci_low !== null && p.ci_high !== null ? ` [${fmtNum(p.ci_low)}–${fmtNum(p.ci_high)}]` : ""
                  }`}
                </title>
                <rect x={x - 8} y={DM.t} width={16} height={ph} fill="transparent" />
                {p.value !== null && p.ci_low !== null && p.ci_high !== null && (
                  <>
                    <line x1={x} x2={x} y1={Y(p.ci_low)} y2={Y(p.ci_high)} stroke={SEG.thorax} strokeOpacity={0.55} strokeWidth={1.4} />
                    <line x1={x - 3} x2={x + 3} y1={Y(p.ci_low)} y2={Y(p.ci_low)} stroke={SEG.thorax} strokeOpacity={0.55} />
                    <line x1={x - 3} x2={x + 3} y1={Y(p.ci_high)} y2={Y(p.ci_high)} stroke={SEG.thorax} strokeOpacity={0.55} />
                  </>
                )}
                {p.value !== null ? (
                  <circle cx={x} cy={Y(p.value)} r={3.4} fill={SEG.thorax} stroke="#fff" strokeWidth={1} />
                ) : (
                  <text x={x} y={DM.t + ph - 2} fontSize={9} textAnchor="middle" fill="#c9c9c9" fontFamily={MONO}>
                    ⊘
                  </text>
                )}
                <text x={x} y={DH - 6} fontSize={9} textAnchor="middle" fill="#8f8f8f" fontFamily={MONO}>
                  {p.order}
                </text>
              </g>
            );
          })}
        </svg>
      )}
      <div className="text-[12px] text-[#4d4d4d] leading-snug mt-1">{row.status_plain}</div>
    </div>
  );
};

export const SpellDrift: React.FC<{ rows: FatigueDriftRow[]; labels: Record<string, string> }> = ({ rows, labels }) => {
  if (rows.length === 0) {
    return <CalmState title="No drift rows yet" body="Ball-by-ball drift appears once a spell has several measured balls." />;
  }
  return (
    <div>
      <div className="grid gap-3 sm:grid-cols-2">
        {rows.map((r) => (
          <DriftChart key={r.metric_key} row={r} labels={labels} />
        ))}
      </div>
      <p className="text-[12px] text-[#666] mt-2 leading-snug">
        Dots are each ball's value with its likely range, by ball order. A trend is only flagged when the backend finds the
        first and last balls clearly separated beyond measurement noise; otherwise no trend line is drawn.
      </p>
    </div>
  );
};

const GRADE_WORD: Record<RepeatabilityGrade, string> = {
  tight: "Tight",
  moderate: "Moderate",
  variable: "Variable",
  measurement_limited: "Measurement-limited",
};

const GradePill: React.FC<{ grade: RepeatabilityGrade }> = ({ grade }) => {
  const variant = grade === "tight" ? "strong" : grade === "variable" ? "dashed" : grade === "measurement_limited" ? "muted" : "plain";
  const glyph = grade === "tight" ? "●" : grade === "moderate" ? "●" : grade === "variable" ? "◇" : "○";
  return (
    <Tag variant={variant}>
      {glyph} {GRADE_WORD[grade] ?? grade}
    </Tag>
  );
};

export const RepeatabilityTable: React.FC<{ rows: RepeatabilityRow[] }> = ({ rows }) => {
  if (rows.length === 0) {
    return <CalmState title="No repeatability rows yet" body="Needs at least two measured balls of the same metric." />;
  }
  return (
    <div className="overflow-x-auto rounded-[6px]" style={{ boxShadow: "0 0 0 1px #ebebeb" }}>
      <table className="w-full text-[13px] min-w-[520px]">
        <thead>
          <tr className="text-left">
            {["Metric", "Balls", "Mean", "Debiased SD", "CV", "Consistency"].map((h, i) => (
              <th key={h} className={`v-label font-normal px-3 py-2 ${i > 0 && i < 5 ? "text-right" : ""}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.metric_key} className="border-t border-[#ebebeb]">
              <td className="px-3 py-2 text-[#171717]">{r.label_plain}</td>
              <td className="px-3 py-2 text-right v-num">{r.n_balls}</td>
              <td className="px-3 py-2 text-right v-num">{withUnit(r.mean, r.unit)}</td>
              <td className="px-3 py-2 text-right v-num" title={r.raw_sd !== null ? `raw SD ${withUnit(r.raw_sd, r.unit)}` : undefined}>
                {withUnit(r.debiased_sd, r.unit)}
                {r.raw_sd !== null && <span className="text-[#8f8f8f] text-[11px] block">raw {withUnit(r.raw_sd, r.unit)}</span>}
              </td>
              <td className="px-3 py-2 text-right v-num">{r.debiased_cv_pct !== null ? `${fmtNum(r.debiased_cv_pct, 1)}%` : "—"}</td>
              <td className="px-3 py-2">
                <GradePill grade={r.grade} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="px-3 py-2 text-[11px] text-[#8f8f8f] border-t border-[#ebebeb] leading-snug">
        Debiased SD removes each ball's own measurement noise from the spread. "Measurement-limited" means the camera noise is
        larger than the ball-to-ball variation, so consistency can't be judged from this footage.
      </div>
    </div>
  );
};
