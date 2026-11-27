import React, { useState } from "react";
import { ChevronDown } from "lucide-react";
import { SEG } from "../../theme";

/**
 * Camera Angle Coach: top-down pitch with phone placements + what each angle unlocks (§8.6).
 * Scale: 10 px per metre. Bowler's end on the left. Pitch 20.12 m × 3.05 m (MCC Law 6).
 */
type Angle = "side" | "behind" | "front" | "dual";

const PX_M = 10;
const BOWL_X = 70; // bowler's stumps
const BAT_X = BOWL_X + 20.12 * PX_M; // batter's stumps
const CY = 52; // pitch centre line
const HALF_W = (3.05 / 2) * PX_M;
const POP = 1.22 * PX_M; // popping crease in front of the stumps
const RET = 1.32 * PX_M; // return creases either side of the middle stump

const PHONES: Record<Exclude<Angle, "dual">, { x: number; y: number; label: string; sub: string }> = {
  side: { x: BOWL_X + POP, y: CY + 10 * PX_M, label: "Side-on", sub: "square of the crease, 8–12 m" },
  behind: { x: BOWL_X - 6.5 * PX_M, y: CY, label: "Behind", sub: "in line with both sets of stumps" },
  front: { x: BAT_X + 2.2 * PX_M, y: CY, label: "Front", sub: "behind the net, safely" },
};

const UNLOCKS: Record<Angle, { title: string; items: string[]; note?: string }> = {
  side: {
    title: "Side-on unlocks",
    items: [
      "Front-knee bend when the front foot lands",
      "Trunk forward lean through release",
      "Stride length and run-up rhythm",
      "Bowling-arm speed and release timing",
      "Ball speed (only measured from side-on)",
    ],
  },
  behind: {
    title: "Behind unlocks",
    items: [
      "Trunk side-bend at release",
      "Hip and shoulder alignment, and how much the shoulders twist back",
      "Where the ball is released across the body",
      "Head position over the front foot",
    ],
  },
  front: {
    title: "Front unlocks",
    items: ["Trunk side-bend and release position, when you can't film from behind"],
    note: "Behind is usually clearer for alignment. Only film from the front from behind a net.",
  },
  dual: {
    title: "Two phones together unlock",
    items: [
      "Everything from side-on and behind at the same moment",
      "A true 3D view of the action, with the two videos lined up to a fraction of a frame",
      "Tighter ranges on every check",
    ],
    note: "Place the phones roughly a right angle apart: one side-on, one behind.",
  },
};

export const CameraAngleCoach: React.FC<{ defaultOpen?: boolean }> = ({ defaultOpen = false }) => {
  const [open, setOpen] = useState(defaultOpen);
  const [angle, setAngle] = useState<Angle>("side");
  const active = (k: Exclude<Angle, "dual">) => angle === k || (angle === "dual" && (k === "side" || k === "behind"));
  const u = UNLOCKS[angle];

  return (
    <section className="v-panel">
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center justify-between gap-3 px-4 py-3 text-left"
      >
        <span className="min-w-0">
          <span className="block v-label">Camera angle coach</span>
          <span className="block text-[13.5px] text-[#171717] font-[500] mt-0.5">Where to put the phone, and what each angle shows</span>
        </span>
        <ChevronDown className={`w-4 h-4 shrink-0 text-[#4d4d4d] transition-transform duration-200 ${open ? "rotate-180" : ""}`} />
      </button>
      {open && (
        <div className="border-t border-[#ebebeb] p-4 grid gap-4 md:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)]">
          <div>
            <div className="v-seg mb-3 flex-wrap" role="group" aria-label="Camera angle">
              {(["side", "behind", "front", "dual"] as Angle[]).map((a) => (
                <button key={a} type="button" aria-pressed={angle === a} onClick={() => setAngle(a)}>
                  {a === "side" ? "Side-on" : a === "behind" ? "Behind" : a === "front" ? "Front" : "Two phones"}
                </button>
              ))}
            </div>
            <div className="v-stage p-2">
              <svg viewBox="-22 0 352 178" className="w-full h-auto" role="img" aria-label="Top-down pitch showing where to place the phone">
                {/* grass + pitch strip */}
                <rect x={BOWL_X - 2.4 * PX_M} y={CY - HALF_W} width={(BAT_X - BOWL_X) + 4.8 * PX_M} height={HALF_W * 2} fill={SEG.grass} stroke={SEG.hairline} />
                {[BOWL_X, BAT_X].map((sx, i) => {
                  const dir = i === 0 ? 1 : -1;
                  const pop = sx + dir * POP;
                  return (
                    <g key={sx} stroke={SEG.ink} strokeWidth={1}>
                      {/* bowling crease */}
                      <line x1={sx} y1={CY - RET - 3} x2={sx} y2={CY + RET + 3} />
                      {/* popping crease */}
                      <line x1={pop} y1={CY - HALF_W} x2={pop} y2={CY + HALF_W} />
                      {/* return creases */}
                      <line x1={pop} y1={CY - RET} x2={sx - dir * 2.4 * PX_M} y2={CY - RET} />
                      <line x1={pop} y1={CY + RET} x2={sx - dir * 2.4 * PX_M} y2={CY + RET} />
                      {/* stumps (3) */}
                      {[-1, 0, 1].map((k) => (
                        <circle key={k} cx={sx} cy={CY + k * 1.6} r={1.1} fill={SEG.ink} stroke="none" />
                      ))}
                    </g>
                  );
                })}
                <text x={BOWL_X} y={CY - HALF_W - 5} textAnchor="middle" fontSize={7} fill={SEG.graphite} fontFamily="var(--font-geist-mono)">
                  BOWLER'S END
                </text>
                <text x={BAT_X} y={CY - HALF_W - 5} textAnchor="middle" fontSize={7} fill={SEG.graphite} fontFamily="var(--font-geist-mono)">
                  BATTER
                </text>
                {/* sight lines */}
                {active("side") && (
                  <line x1={PHONES.side.x} y1={PHONES.side.y - 5} x2={PHONES.side.x} y2={CY + HALF_W} stroke={SEG.positive} strokeDasharray="3 3" strokeWidth={1} />
                )}
                {active("behind") && (
                  <line x1={PHONES.behind.x + 5} y1={CY} x2={BAT_X} y2={CY} stroke={SEG.positive} strokeDasharray="3 3" strokeWidth={1} opacity={0.8} />
                )}
                {active("front") && (
                  <line x1={PHONES.front.x - 5} y1={CY} x2={BOWL_X} y2={CY} stroke={SEG.positive} strokeDasharray="3 3" strokeWidth={1} opacity={0.8} />
                )}
                {/* distance tick for side-on */}
                <g fontFamily="var(--font-geist-mono)" fontSize={7} fill={SEG.graphite}>
                  <text x={PHONES.side.x + 6} y={(PHONES.side.y + CY + HALF_W) / 2 + 2}>8–12 m</text>
                </g>
                {/* phones */}
                {(Object.keys(PHONES) as Array<Exclude<Angle, "dual">>).map((k) => {
                  const p = PHONES[k];
                  const on = active(k);
                  const labelBelow = k === "side";
                  return (
                    <g key={k} style={{ cursor: "pointer" }} onClick={() => setAngle(k)}>
                      <rect
                        x={p.x - 3.5}
                        y={p.y - 6}
                        width={7}
                        height={12}
                        rx={1.5}
                        fill={on ? SEG.positive : "#ffffff"}
                        stroke={on ? SEG.positive : SEG.graphite}
                        strokeWidth={1}
                      />
                      <text
                        x={p.x}
                        y={labelBelow ? p.y + 16 : p.y + 20}
                        textAnchor="middle"
                        fontSize={8}
                        fontWeight={500}
                        fill={on ? SEG.ink : SEG.graphite}
                      >
                        {p.label}
                      </text>
                      {k !== "side" && (
                        <text x={p.x} y={p.y + 29} textAnchor="middle" fontSize={6.5} fill={SEG.graphite}>
                          {k === "behind" ? "in line" : "behind net"}
                        </text>
                      )}
                    </g>
                  );
                })}
              </svg>
            </div>
            <p className="mt-2 text-[12px] text-[#8f8f8f] leading-[1.5]">
              {angle === "dual"
                ? `${PHONES.side.label}: ${PHONES.side.sub}. ${PHONES.behind.label}: ${PHONES.behind.sub}.`
                : `${PHONES[angle].label}: ${PHONES[angle].sub}. Phone at about hip height, held still and level.`}
            </p>
          </div>
          <div>
            <div className="v-label mb-2">{u.title}</div>
            <ul className="space-y-1.5">
              {u.items.map((it) => (
                <li key={it} className="flex gap-2 text-[13px] leading-[1.5] text-[#171717]">
                  <span className="text-[#297a3a] shrink-0" aria-hidden>
                    ●
                  </span>
                  {it}
                </li>
              ))}
            </ul>
            {u.note && <p className="mt-3 text-[12.5px] leading-[1.5] text-[#4d4d4d]">{u.note}</p>}
            <div className="mt-4 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] p-3 text-[12.5px] leading-[1.55] text-[#4d4d4d]">
              <span className="font-[500] text-[#171717]">Every angle: </span>
              film in slow-motion (120–240 fps) if your phone has it, keep the whole body in view from run-up to follow-through, and
              keep at least one set of stumps visible so we can measure the pitch.
            </div>
          </div>
        </div>
      )}
    </section>
  );
};

export default CameraAngleCoach;
