import React, { useEffect, useRef } from "react";
import { Check, X } from "lucide-react";

/**
 * "How to film" guidance — reusable body (`HowToFilmContent`) used inline in the
 * guided New Analysis flow, plus a right-side sheet wrapper (`HowToFilmSheet`).
 * Plain coaching language only: no acronyms, no frame numbers, no jargon.
 */

type Placement = "side" | "behind" | "both";

const ACCENT = "#297a3a";

/** Small camera glyph: body + lens, rotated to face `angle` (deg, 0 = pointing right). */
const CameraMark: React.FC<{ x: number; y: number; angle: number }> = ({ x, y, angle }) => (
  <g transform={`translate(${x} ${y}) rotate(${angle})`}>
    <rect x={-7} y={-5} width={11} height={10} rx={2} fill={ACCENT} />
    <path d="M 4 -3 L 8 -5 L 8 5 L 4 3 Z" fill={ACCENT} />
    <circle cx={-1.5} cy={0} r={2.2} fill="#ffffff" />
  </g>
);

/** Top-down pitch diagram with the camera position(s) for each placement. */
const PitchDiagram: React.FC<{ placement: Placement }> = ({ placement }) => {
  // Pitch strip runs left → right; bowler runs in from the left towards the batter's end.
  const bowlerX = 58;
  const bowlerY = 55;
  const showSide = placement === "side" || placement === "both";
  const showBehind = placement === "behind" || placement === "both";
  return (
    <svg viewBox="0 0 200 110" className="w-full h-auto" role="img" aria-label="Top-down view of the pitch showing where to put the camera">
      <rect x={0.5} y={0.5} width={199} height={109} rx={6} fill="#fafafa" stroke="#ebebeb" />
      {/* Fields of view (drawn first so the pitch sits on top) */}
      {showSide && <path d={`M ${bowlerX} 94 L ${bowlerX - 22} 40 L ${bowlerX + 22} 40 Z`} fill={ACCENT} opacity={0.08} />}
      {showBehind && <path d={`M 14 ${bowlerY} L 74 ${bowlerY - 18} L 74 ${bowlerY + 18} Z`} fill={ACCENT} opacity={0.08} />}
      {/* Pitch strip */}
      <rect x={40} y={45} width={150} height={20} rx={1.5} fill="#ffffff" stroke="#c9c9c9" />
      {/* Creases */}
      <line x1={54} y1={41} x2={54} y2={69} stroke="#8f8f8f" strokeWidth={1} />
      <line x1={176} y1={41} x2={176} y2={69} stroke="#8f8f8f" strokeWidth={1} />
      {/* Stumps */}
      {[52, 55, 58].map((y) => (
        <circle key={`a${y}`} cx={47} cy={y} r={1} fill="#171717" />
      ))}
      {[52, 55, 58].map((y) => (
        <circle key={`b${y}`} cx={183} cy={y} r={1} fill="#171717" />
      ))}
      {/* Run-up path */}
      <line x1={22} y1={bowlerY - 14} x2={bowlerX - 6} y2={bowlerY - 2} stroke="#a8a8a8" strokeWidth={1} strokeDasharray="2.5 2.5" />
      {/* Bowler */}
      <circle cx={bowlerX} cy={bowlerY} r={4} fill="#171717" />
      {/* Cameras */}
      {showSide && <CameraMark x={bowlerX} y={97} angle={-90} />}
      {showBehind && <CameraMark x={11} y={bowlerY} angle={0} />}
      {/* Labels */}
      <text x={bowlerX + 7} y={bowlerY - 6} fontSize={7} fill="#4d4d4d" fontFamily="inherit">
        Bowler
      </text>
      <text x={186} y={36} fontSize={7} fill="#8f8f8f" textAnchor="end" fontFamily="inherit">
        Batter's end
      </text>
      {showSide && (
        <text x={bowlerX + 12} y={100} fontSize={7} fill={ACCENT} fontFamily="inherit">
          Phone, side-on
        </text>
      )}
      {showBehind && (
        <text x={6} y={bowlerY + 20} fontSize={7} fill={ACCENT} fontFamily="inherit">
          Phone, behind
        </text>
      )}
    </svg>
  );
};

const PLACEMENTS: Array<{ id: Placement; title: string; where: string; unlocks: string[] }> = [
  {
    id: "side",
    title: "Side-on",
    where: "Stand the phone level with where the bowler lets go of the ball, about 10–15 steps away, facing across the pitch.",
    unlocks: ["Front-Leg Brace", "Trunk Whip", "Run-Up & Stride"],
  },
  {
    id: "behind",
    title: "From behind",
    where: "Put the phone straight behind the run-up, in line with the stumps, so you look down the pitch over the bowler.",
    unlocks: ["Side-Bend Load", "Shoulder Alignment"],
  },
  {
    id: "both",
    title: "Both phones",
    where: "Film the same ball side-on and from behind at once. Combining the two views builds a 3D picture of the action.",
    unlocks: ["Everything — all body checkpoints in 3D"],
  },
];

const CHECKLIST = [
  "Whole body in frame, from the run-up to the follow-through",
  "Phone steady — on a tripod or wedged against a fence",
  "Slow-motion (120 or 240 fps) if your phone has it",
  "Good light — avoid filming into the sun",
];

export const HowToFilmContent: React.FC = () => (
  <div className="space-y-5">
    <div>
      <h3 className="text-[15px] font-[500] tracking-[-0.01em] text-[#171717]">Where to put the phone</h3>
      <p className="mt-1 text-[13px] leading-[1.55] text-[#4d4d4d]">
        Each camera angle shows different parts of the action. Pick the one that fits what you want to check.
      </p>
    </div>
    <div className="grid gap-3 sm:grid-cols-3">
      {PLACEMENTS.map((p) => (
        <div key={p.id} className="v-card p-3.5 flex flex-col gap-3">
          <PitchDiagram placement={p.id} />
          <div>
            <div className="text-[14px] font-[500] text-[#171717]">{p.title}</div>
            <p className="mt-1 text-[12.5px] leading-[1.5] text-[#4d4d4d]">{p.where}</p>
          </div>
          <div className="mt-auto pt-2 border-t border-[#ebebeb]">
            <div className="text-[12px] text-[#8f8f8f] mb-1.5">Lets you see</div>
            <ul className="space-y-1">
              {p.unlocks.map((u) => (
                <li key={u} className="flex items-start gap-1.5 text-[12.5px] text-[#171717]">
                  <Check className="w-3.5 h-3.5 mt-[2px] shrink-0 text-[#297a3a]" />
                  <span>{u}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      ))}
    </div>
    <div className="v-card p-4">
      <h3 className="text-[14px] font-[500] text-[#171717] mb-2.5">Before you press record</h3>
      <ul className="space-y-2">
        {CHECKLIST.map((item) => (
          <li key={item} className="flex items-start gap-2.5 text-[13px] leading-[1.5] text-[#4d4d4d]">
            <span className="mt-[2px] w-4 h-4 rounded-full border border-[#297a3a]/40 bg-[#297a3a]/[0.06] flex items-center justify-center shrink-0">
              <Check className="w-2.5 h-2.5 text-[#297a3a]" />
            </span>
            <span>{item}</span>
          </li>
        ))}
      </ul>
    </div>
  </div>
);

export const HowToFilmSheet: React.FC<{ open: boolean; onClose: () => void }> = ({ open, onClose }) => {
  const closeRef = useRef<HTMLButtonElement | null>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    closeRef.current?.focus();
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = prevOverflow;
    };
  }, [open, onClose]);

  if (!open) return null;

  return (
    <div className="no-print fixed inset-0 z-50">
      <div className="absolute inset-0 bg-[#171717]/20" onClick={onClose} aria-hidden />
      <aside
        role="dialog"
        aria-modal="true"
        aria-labelledby="how-to-film-title"
        className="absolute right-0 top-0 h-full w-full max-w-[560px] bg-[#fafafa] border-l border-[#ebebeb] overflow-y-auto"
      >
        <div className="sticky top-0 z-10 bg-[#ffffff] border-b border-[#ebebeb] px-4 sm:px-5 h-14 flex items-center justify-between">
          <h2 id="how-to-film-title" className="text-[16px] font-[500] tracking-[-0.01em] text-[#171717]">
            How to film
          </h2>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="v-btn-ghost w-8 h-8 flex items-center justify-center"
            aria-label="Close filming guide"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4 sm:p-5">
          <HowToFilmContent />
        </div>
      </aside>
    </div>
  );
};

export default HowToFilmSheet;
