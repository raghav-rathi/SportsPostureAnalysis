import React from "react";
import { Camera, Info } from "lucide-react";
import type { CaptureGrade, CheckpointId, CheckpointStatus, StoryNotice, ViewClarity } from "../../types";

/** Fixed checkpoint order + fallback titles (backend `title` always wins when present). */
export const CHECKPOINT_ORDER: CheckpointId[] = [
  "front_leg_brace",
  "trunk_whip",
  "side_bend",
  "shoulder_alignment",
  "runup_stride",
];

export const CHECKPOINT_TITLES: Record<CheckpointId, string> = {
  front_leg_brace: "Front-Leg Brace",
  trunk_whip: "Trunk Whip",
  side_bend: "Side-Bend Load",
  shoulder_alignment: "Shoulder Alignment",
  runup_stride: "Run-Up & Stride",
};

export const STATUS_LABELS: Record<CheckpointStatus, string> = {
  strong: "Strong",
  on_track: "On Track",
  watch: "Borderline",
  focus: "Needs Focus",
  not_visible: "Not Visible",
};

/** Status dot: shape + fill carry meaning, never colour alone (text label always accompanies). */
export const StatusDot: React.FC<{ status: CheckpointStatus; size?: number; className?: string }> = ({
  status,
  size = 10,
  className = "",
}) => {
  const s = size;
  const r = s / 2 - 1;
  const c = s / 2;
  return (
    <svg width={s} height={s} viewBox={`0 0 ${s} ${s}`} className={`shrink-0 ${className}`} aria-hidden>
      {status === "strong" && <circle cx={c} cy={c} r={r} fill="#297a3a" />}
      {status === "focus" && <circle cx={c} cy={c} r={r} fill="#171717" />}
      {status === "on_track" && <circle cx={c} cy={c} r={r - 0.4} fill="#ffffff" stroke="#171717" strokeWidth={1.4} />}
      {status === "watch" && (
        <>
          <circle cx={c} cy={c} r={r - 0.4} fill="#ffffff" stroke="#171717" strokeWidth={1.4} />
          <path d={`M ${c} ${c - r + 0.4} A ${r - 0.4} ${r - 0.4} 0 0 1 ${c} ${c + r - 0.4} Z`} fill="#171717" />
        </>
      )}
      {status === "not_visible" && (
        <circle cx={c} cy={c} r={r - 0.4} fill="none" stroke="#a8a8a8" strokeWidth={1.3} strokeDasharray="2 1.6" />
      )}
    </svg>
  );
};

export const StatusBadge: React.FC<{
  status: CheckpointStatus;
  label?: string | null;
  unlockReason?: string | null;
  compact?: boolean;
}> = ({ status, label, unlockReason, compact }) => {
  const text = label || STATUS_LABELS[status];
  const tone =
    status === "strong"
      ? "text-[#297a3a] border-[#297a3a]/30 bg-[#297a3a]/[0.06]"
      : status === "focus"
      ? "text-[#171717] border-[#171717] bg-[#ffffff] font-[500]"
      : status === "not_visible"
      ? "text-[#8f8f8f] border-dashed border-[#c9c9c9] bg-[#fafafa]"
      : "text-[#171717] border-[#ebebeb] bg-[#ffffff]";
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full border ${tone} ${
        compact ? "h-5 px-2 text-[11px]" : "h-6 px-2.5 text-[12px]"
      } whitespace-nowrap`}
    >
      {status === "not_visible" && (unlockReason === "angle" || unlockReason === "frame_rate") ? (
        <Camera className="w-3 h-3" />
      ) : (
        <StatusDot status={status} size={compact ? 8 : 9} />
      )}
      {text}
    </span>
  );
};

export const CAPTURE_GRADE_LABEL: Record<CaptureGrade, string> = {
  great: "Great view",
  partial: "Partial view",
  limited: "Limited view",
  unusable: "Couldn't analyse",
};

/** Capture grade glyph: full / half / quarter / empty circle. */
export const CaptureGlyph: React.FC<{ grade: CaptureGrade; size?: number }> = ({ grade, size = 12 }) => {
  const c = size / 2;
  const r = c - 1;
  const frac = grade === "great" ? 1 : grade === "partial" ? 0.5 : grade === "limited" ? 0.25 : 0;
  const end = -Math.PI / 2 + frac * Math.PI * 2;
  const large = frac > 0.5 ? 1 : 0;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden className="shrink-0">
      <circle cx={c} cy={c} r={r - 0.4} fill="#ffffff" stroke={grade === "great" ? "#297a3a" : "#171717"} strokeWidth={1.3} />
      {frac >= 1 ? (
        <circle cx={c} cy={c} r={r - 0.4} fill="#297a3a" />
      ) : frac > 0 ? (
        <path
          d={`M ${c} ${c} L ${c} ${c - r + 0.4} A ${r - 0.4} ${r - 0.4} 0 ${large} 1 ${c + (r - 0.4) * Math.cos(end)} ${
            c + (r - 0.4) * Math.sin(end)
          } Z`}
          fill="#171717"
        />
      ) : null}
    </svg>
  );
};

export const CLARITY_LABEL: Record<Exclude<ViewClarity, null>, string> = {
  clear: "Clear view",
  fair: "Fair view",
  rough: "Rough view",
};

/** Value + subtle range, e.g. `171°` · likely 161°–180°. */
export const ValueWithRange: React.FC<{
  value: string | null | undefined;
  range: string | null | undefined;
  size?: "sm" | "md" | "lg";
  inline?: boolean;
}> = ({ value, range, size = "md", inline = false }) => {
  if (!value) return <span className="text-[13px] text-[#a8a8a8]">—</span>;
  const vCls = size === "lg" ? "text-[28px] tracking-[-0.03em]" : size === "md" ? "text-[17px]" : "text-[13px]";
  return (
    <span className={`inline-flex ${inline ? "items-baseline gap-2" : "flex-col items-end leading-tight"}`}>
      <span className={`font-mono font-[500] v-tabular text-[#171717] whitespace-nowrap ${vCls}`}>{value}</span>
      {range && (
        <span className="text-[11px] text-[#8f8f8f] whitespace-nowrap">
          likely <span className="font-mono v-tabular">{range}</span>
        </span>
      )}
    </span>
  );
};

/** Full-text, never-truncated guardrail notices (ICC + medical + extra caveats). */
export const NoticeFooter: React.FC<{
  notices: { icc: StoryNotice; medical: StoryNotice; extra?: StoryNotice[] } | null | undefined;
  className?: string;
}> = ({ notices, className = "" }) => {
  const list: StoryNotice[] = notices
    ? [notices.icc, notices.medical, ...(notices.extra ?? [])].filter(Boolean)
    : [FALLBACK_ICC, FALLBACK_MEDICAL];
  return (
    <section className={`v-card p-4 md:p-5 ${className}`} aria-label="About these results">
      <div className="text-[13px] font-[500] text-[#171717] mb-3">About these results</div>
      <div className="grid gap-3 md:grid-cols-2">
        {list.map((n) => (
          <div key={n.id} className="flex gap-2.5 items-start">
            <Info className="w-4 h-4 mt-[1px] text-[#666666] shrink-0" />
            <p className="text-[12.5px] leading-[1.55] text-[#4d4d4d] whitespace-normal break-words">
              <span className="font-[500] text-[#171717]">{n.title}</span> {n.body}
            </p>
          </div>
        ))}
      </div>
    </section>
  );
};

export const FALLBACK_ICC: StoryNotice = {
  id: "icc",
  title: "Elbow legality (ICC 15° limit): not assessed — ever.",
  body:
    "Judging elbow straightening against the ICC 15° limit requires accredited laboratory testing with high-speed (≥250 Hz) 3D motion capture. Phone or casual multi-camera video cannot measure it reliably, so this app permanently refuses to give a verdict.",
};

export const FALLBACK_MEDICAL: StoryNotice = {
  id: "medical",
  title: "Not medical advice.",
  body:
    "These are movement measurements from a single delivery, not an injury diagnosis. Injury risk also depends on workload, age and growth, and physical conditioning. If you have pain or concerns, speak to a qualified physiotherapist or doctor.",
};

export const Card: React.FC<React.HTMLAttributes<HTMLDivElement>> = ({ className = "", ...rest }) => (
  <div className={`v-card ${className}`} {...rest} />
);

export const SectionTitle: React.FC<{ children: React.ReactNode; hint?: React.ReactNode; className?: string }> = ({
  children,
  hint,
  className = "",
}) => (
  <div className={`flex items-baseline justify-between gap-3 mb-3 ${className}`}>
    <h2 className="text-[15px] font-[500] tracking-[-0.01em] text-[#171717]">{children}</h2>
    {hint && <div className="text-[12px] text-[#8f8f8f]">{hint}</div>}
  </div>
);

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
}

/** Tiny 5-dot strip used on delivery cards / progress. */
export const CheckpointDots: React.FC<{
  statuses: Partial<Record<CheckpointId, CheckpointStatus>>;
  size?: number;
}> = ({ statuses, size = 9 }) => (
  <div className="flex items-center gap-1" aria-label="Body checkpoint statuses">
    {CHECKPOINT_ORDER.map((id) => {
      const st = statuses[id] ?? "not_visible";
      return (
        <span key={id} title={`${CHECKPOINT_TITLES[id]}: ${STATUS_LABELS[st]}`}>
          <StatusDot status={st} size={size} />
        </span>
      );
    })}
  </div>
);

export const Spinner: React.FC<{ className?: string }> = ({ className = "" }) => (
  <span
    className={`inline-block w-4 h-4 rounded-full border-[1.5px] border-[#ebebeb] border-t-[#171717] animate-spin ${className}`}
    aria-hidden
  />
);

export const EmptyState: React.FC<{ title: string; body?: React.ReactNode; action?: React.ReactNode }> = ({
  title,
  body,
  action,
}) => (
  <div className="v-card p-10 text-center max-w-lg mx-auto">
    <h2 className="text-[18px] font-[500] tracking-[-0.02em] text-[#171717]">{title}</h2>
    {body && <div className="mt-2 text-[13px] text-[#4d4d4d] leading-relaxed">{body}</div>}
    {action && <div className="mt-5 flex justify-center gap-2">{action}</div>}
  </div>
);
