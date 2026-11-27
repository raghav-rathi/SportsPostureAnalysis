import React, { useState } from "react";
import { Camera, ChevronDown, Crosshair, PlayCircle } from "lucide-react";
import type { BodyCheckpoint, CaptureQuality, CheckpointId, DrillCard, Takeaway } from "../../types";
import { CaptureGlyph, CLARITY_LABEL, StatusBadge, StatusDot, ValueWithRange } from "./ui";

// ─────────────────────────────────────────────────────────────────────────────
// Your #1 Takeaway hero card
// ─────────────────────────────────────────────────────────────────────────────
const TAKEAWAY_EYEBROW: Record<Takeaway["kind"], string> = {
  focus: "Your #1 thing to work on",
  watch: "Keep an eye on",
  strength: "Your standout strength",
  capture: "Get a clearer read next time",
  all_clear: "Your #1 takeaway",
};

export const TakeawayCard: React.FC<{
  takeaway: Takeaway;
  onSeeMoment?: () => void;
  onPractise?: () => void;
  onFilmingTips?: () => void;
}> = ({ takeaway, onSeeMoment, onPractise, onFilmingTips }) => (
  <section className="v-card p-5" aria-label="Your number one takeaway">
    <div className="flex items-center gap-2 text-[12px] text-[#666666]">
      <span className={`w-1.5 h-1.5 rounded-full ${takeaway.kind === "strength" ? "bg-[#297a3a]" : "bg-[#171717]"}`} />
      {TAKEAWAY_EYEBROW[takeaway.kind]}
    </div>
    <h2 className="mt-2 text-[19px] leading-[1.3] font-[500] tracking-[-0.02em] text-[#171717]">{takeaway.title}</h2>
    <p className="mt-2 text-[14px] leading-[1.55] text-[#4d4d4d]">{takeaway.plain_explanation}</p>
    {takeaway.what_to_try && (
      <div className="mt-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] px-3 py-2.5">
        <div className="text-[12px] text-[#666666]">Try this</div>
        <div className="text-[14px] text-[#171717] mt-0.5 leading-snug">{takeaway.what_to_try}</div>
      </div>
    )}
    <div className="mt-4 flex flex-wrap gap-2">
      {takeaway.frame_idx !== null && onSeeMoment && (
        <button onClick={onSeeMoment} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
          <PlayCircle className="w-3.5 h-3.5" /> See the moment
        </button>
      )}
      {takeaway.linked_drill_id && onPractise && (
        <button onClick={onPractise} className="v-btn-primary h-8 px-3 text-[13px]">
          Practise this →
        </button>
      )}
      {takeaway.kind === "capture" && onFilmingTips && (
        <button onClick={onFilmingTips} className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5">
          <Camera className="w-3.5 h-3.5" /> How to film
        </button>
      )}
    </div>
  </section>
);

// ─────────────────────────────────────────────────────────────────────────────
// Camera Angle Coach pill (expands)
// ─────────────────────────────────────────────────────────────────────────────
const MiniPitch: React.FC<{ view: CaptureQuality["view"] }> = ({ view }) => {
  // Top-down: bowler runs left → right toward batter. Camera icon(s) placed by view class.
  const cams: Array<[number, number]> =
    view === "side_on"
      ? [[70, 54]]
      : view === "rear_on"
      ? [[8, 28]]
      : view === "front_on"
      ? [[152, 28]]
      : view === "oblique"
      ? [[22, 52]]
      : view === "dual_fused"
      ? [[70, 54], [8, 28]]
      : [];
  return (
    <svg viewBox="0 0 160 60" className="w-[150px] h-[56px] shrink-0" aria-hidden>
      <rect x="22" y="18" width="116" height="20" rx="2" fill="#fafafa" stroke="#d4d4d8" />
      <line x1="34" y1="18" x2="34" y2="38" stroke="#c9c9c9" />
      <line x1="126" y1="18" x2="126" y2="38" stroke="#c9c9c9" />
      <circle cx="40" cy="28" r="3.5" fill="#171717" />
      <path d="M44 28 L58 28" stroke="#171717" strokeWidth="1" strokeDasharray="2 2" />
      <text x="40" y="12" fontSize="7" textAnchor="middle" fill="#8f8f8f" fontFamily="Geist, sans-serif">bowler</text>
      {cams.map(([x, y], i) => (
        <g key={i} transform={`translate(${x - 6},${y - 5})`}>
          <rect width="12" height="9" rx="2" fill="#297a3a" />
          <circle cx="6" cy="4.5" r="2.2" fill="#ffffff" />
        </g>
      ))}
    </svg>
  );
};

export const CameraCoachPill: React.FC<{
  capture: CaptureQuality;
  titleFor: (id: CheckpointId) => string;
  onFilmingTips?: () => void;
}> = ({ capture, titleFor, onFilmingTips }) => {
  const [open, setOpen] = useState(false);
  return (
    <div className="v-card overflow-hidden">
      <button
        onClick={() => setOpen((v) => !v)}
        className="w-full flex items-center gap-2.5 px-4 py-3 text-left hover:bg-[#fafafa]"
        aria-expanded={open}
      >
        <CaptureGlyph grade={capture.grade} size={14} />
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-baseline gap-x-2">
            <span className="text-[13.5px] font-[500] text-[#171717] whitespace-nowrap">{capture.label}</span>
            <span className="text-[12.5px] text-[#666666] whitespace-nowrap">
              Saw <span className="font-mono v-tabular">{capture.measured_count}</span> of{" "}
              <span className="font-mono v-tabular">{capture.total_count}</span> checks
            </span>
          </span>
          <span className="block text-[12px] text-[#8f8f8f] mt-0.5">
            {capture.view_label} · {capture.fps_label} · Camera angle coach
          </span>
        </span>
        <ChevronDown className={`ml-auto w-4 h-4 text-[#8f8f8f] shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {!open && <p className="px-4 pb-3 -mt-1 text-[12.5px] text-[#4d4d4d] leading-snug">{capture.summary}</p>}
      {open && (
        <div className="px-4 pb-4 space-y-3">
          <div className="flex gap-3 items-start">
            <MiniPitch view={capture.view} />
            <p className="text-[13px] text-[#4d4d4d] leading-[1.5]">{capture.summary}</p>
          </div>
          <div className="grid sm:grid-cols-2 gap-3">
            <div>
              <div className="text-[12px] text-[#666666] mb-1.5">This angle could see</div>
              {capture.can_see.length === 0 ? (
                <div className="text-[13px] text-[#8f8f8f]">No checkpoints clearly.</div>
              ) : (
                <ul className="space-y-1">
                  {capture.can_see.map((id) => (
                    <li key={id} className="text-[13px] text-[#171717] flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-[#297a3a]" /> {titleFor(id)}
                    </li>
                  ))}
                </ul>
              )}
            </div>
            <div>
              <div className="text-[12px] text-[#666666] mb-1.5">Couldn't see</div>
              {capture.cannot_see.length === 0 ? (
                <div className="text-[13px] text-[#8f8f8f]">Nothing — every checkpoint was visible.</div>
              ) : (
                <ul className="space-y-1.5">
                  {capture.cannot_see.map((c) => (
                    <li key={c.checkpoint_id} className="text-[13px] text-[#171717] leading-snug">
                      <span className="flex items-center gap-1.5">
                        <StatusDot status="not_visible" size={9} /> {titleFor(c.checkpoint_id)}
                      </span>
                      <span className="block pl-[15px] text-[12px] text-[#666666]">{c.reason_plain}</span>
                    </li>
                  ))}
                </ul>
              )}
            </div>
          </div>
          {capture.next_clip_tip && (
            <div className="rounded-[6px] bg-[#fafafa] border border-[#ebebeb] px-3 py-2.5 flex flex-wrap items-center gap-2 justify-between">
              <span className="text-[13px] text-[#171717] flex items-center gap-1.5">
                <Camera className="w-3.5 h-3.5 text-[#297a3a]" /> {capture.next_clip_tip}
              </span>
              {onFilmingTips && (
                <button onClick={onFilmingTips} className="text-[12.5px] text-[#171717] underline underline-offset-2">
                  How to film
                </button>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// Body checkpoint list (expandable rows)
// ─────────────────────────────────────────────────────────────────────────────
export const CheckpointList: React.FC<{
  checkpoints: BodyCheckpoint[];
  selectedId: CheckpointId | null;
  onSelect: (id: CheckpointId) => void;
  onSeeMoment: (cp: BodyCheckpoint) => void;
  onFilmingTips?: () => void;
}> = ({ checkpoints, selectedId, onSelect, onSeeMoment, onFilmingTips }) => (
  <ul className="divide-y divide-[#ebebeb] border-t border-[#ebebeb]">
    {checkpoints.map((cp) => {
      const open = cp.id === selectedId;
      return (
        <li key={cp.id}>
          <button
            onClick={() => onSelect(cp.id)}
            className={`w-full flex items-center gap-3 py-3 px-1 text-left ${open ? "" : "hover:bg-[#fafafa]"}`}
            aria-expanded={open}
          >
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <span className="text-[14px] font-[500] text-[#171717]">{cp.title}</span>
                <StatusBadge status={cp.status} label={cp.status_label} unlockReason={cp.unlock_reason} compact />
              </div>
              <div className="text-[12px] text-[#8f8f8f] mt-0.5">{cp.moment_label}</div>
            </div>
            {cp.status !== "not_visible" ? (
              <ValueWithRange value={cp.value_text} range={cp.range_text} />
            ) : (
              <span className="text-[12px] text-[#a8a8a8]">—</span>
            )}
            <ChevronDown className={`w-4 h-4 text-[#a8a8a8] shrink-0 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
          {open && (
            <div className="pb-4 px-1 space-y-3">
              <p className="text-[13.5px] leading-[1.55] text-[#171717]">{cp.plain_meaning}</p>
              {cp.status === "not_visible" ? (
                cp.how_to_unlock && (
                  <div className="rounded-[6px] bg-[#fafafa] border border-dashed border-[#c9c9c9] px-3 py-2.5 text-[13px] text-[#171717] flex items-start gap-2">
                    <Camera className="w-3.5 h-3.5 mt-0.5 shrink-0 text-[#666666]" />
                    <span>
                      {cp.how_to_unlock}{" "}
                      {onFilmingTips && (
                        <button onClick={onFilmingTips} className="underline underline-offset-2">
                          How to film
                        </button>
                      )}
                    </span>
                  </div>
                )
              ) : (
                <>
                  <div>
                    <div className="text-[12px] text-[#666666]">Why it matters</div>
                    <p className="text-[13px] leading-[1.55] text-[#4d4d4d] mt-0.5">{cp.why_it_matters}</p>
                  </div>
                  {cp.reference && (
                    <div className="rounded-[6px] bg-[#fafafa] border border-[#ebebeb] px-3 py-2">
                      <div className="text-[12px] text-[#666666]">Research comparison</div>
                      <p className="text-[13px] text-[#171717] mt-0.5 leading-snug">{cp.reference.text}</p>
                      <p className="text-[11px] text-[#8f8f8f] mt-1">{cp.reference.source}</p>
                    </div>
                  )}
                  <div className="text-[12px] text-[#8f8f8f] leading-snug">
                    {cp.view_clarity ? `${CLARITY_LABEL[cp.view_clarity]} · ` : ""}
                    The true value is very likely inside the range shown.
                  </div>
                </>
              )}
              {cp.supporting.length > 0 && (
                <div>
                  <div className="text-[12px] text-[#666666] mb-1">Related measurements</div>
                  <ul className="space-y-1">
                    {cp.supporting.map((s) => (
                      <li key={s.metric_key} className="flex items-center justify-between gap-3 text-[13px]">
                        <span className="text-[#4d4d4d]">{s.label_plain}</span>
                        {s.suppressed || !s.value_text ? (
                          <span className="text-[12px] text-[#a8a8a8]">Not visible</span>
                        ) : (
                          <ValueWithRange value={s.value_text} range={s.range_text} size="sm" inline />
                        )}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {cp.frame_idx !== null && (
                <button onClick={() => onSeeMoment(cp)} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
                  <Crosshair className="w-3.5 h-3.5" /> Show me on the video
                </button>
              )}
            </div>
          )}
        </li>
      );
    })}
  </ul>
);

// ─────────────────────────────────────────────────────────────────────────────
// Drill card (Practice tab)
// ─────────────────────────────────────────────────────────────────────────────
export const DrillCardView: React.FC<{
  drill: DrillCard;
  index: number;
  forTitle: string | null;
  onSeeMoment?: () => void;
}> = ({ drill, index, forTitle, onSeeMoment }) => {
  const storageKey = `spa.drill.${drill.id}`;
  const [done, setDone] = useState<boolean[]>(() => {
    try {
      const raw = window.localStorage.getItem(storageKey);
      return raw ? JSON.parse(raw) : drill.steps.map(() => false);
    } catch {
      return drill.steps.map(() => false);
    }
  });
  const toggle = (i: number) => {
    const next = drill.steps.map((_, j) => (j === i ? !done[j] : !!done[j]));
    setDone(next);
    try {
      window.localStorage.setItem(storageKey, JSON.stringify(next));
    } catch {
      /* ignore */
    }
  };
  return (
    <article className="v-card p-5 md:p-6">
      <div className="flex flex-wrap items-center gap-2 text-[12px] text-[#666666]">
        <span className="font-mono">Drill {index + 1}</span>
        {forTitle && <span className="h-5 px-2 rounded-full border border-[#ebebeb] bg-[#fafafa] inline-flex items-center">For: {forTitle}</span>}
      </div>
      <h3 className="mt-2 text-[18px] font-[500] tracking-[-0.02em] text-[#171717]">{drill.title}</h3>
      <p className="mt-1.5 text-[14px] leading-[1.55] text-[#4d4d4d]">{drill.why_plain}</p>
      {drill.cue && (
        <div className="mt-4 border-l-2 border-[#297a3a] pl-3">
          <div className="text-[12px] text-[#666666]">Say this to yourself</div>
          <div className="text-[17px] text-[#171717] leading-snug mt-0.5">“{drill.cue}”</div>
        </div>
      )}
      <ol className="mt-4 space-y-2">
        {drill.steps.map((s, i) => (
          <li key={i}>
            <label className="flex items-start gap-2.5 cursor-pointer">
              <input type="checkbox" checked={!!done[i]} onChange={() => toggle(i)} className="mt-[3px] accent-[#297a3a] w-4 h-4" />
              <span className={`text-[14px] leading-[1.5] ${done[i] ? "text-[#8f8f8f] line-through" : "text-[#171717]"}`}>
                <span className="font-mono text-[12px] text-[#8f8f8f] mr-1.5">{i + 1}.</span>
                {s}
              </span>
            </label>
          </li>
        ))}
      </ol>
      {drill.caution && (
        <p className="mt-4 text-[12.5px] leading-[1.5] text-[#4d4d4d] rounded-[6px] bg-[#fafafa] border border-[#ebebeb] px-3 py-2">
          <span className="font-[500] text-[#171717]">Take care: </span>
          {drill.caution}
        </p>
      )}
      <div className="mt-4 flex flex-wrap items-center justify-between gap-2">
        <span className="text-[11px] text-[#8f8f8f]">{drill.source}</span>
        {onSeeMoment && drill.frame_idx !== null && (
          <button onClick={onSeeMoment} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <PlayCircle className="w-3.5 h-3.5" /> See your moment
          </button>
        )}
      </div>
    </article>
  );
};
