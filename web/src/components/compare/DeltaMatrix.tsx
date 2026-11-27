import React from "react";
import type { Archetype, BiomechanicalFactSheet, CompareDifference, CompareSide, CompareStory } from "../../types";
import { StatusPill } from "../ui/PaperPrimitives";
import { metricOf } from "./compareMath";

/**
 * Noise-gated delta matrix: the 5 core CompareStory rows + v5 `extended_rows`, rendered verbatim from the backend.
 * "Within noise" is a calm first-class state (S4: |Δ| > 1.96·√(σa²+σb²) else within noise). Wording stays descriptive.
 */

const DIFF_STYLE: Record<CompareDifference, string> = {
  clearly_higher: "bg-[#171717] text-white border border-[#171717]",
  clearly_lower: "bg-[#171717] text-white border border-[#171717]",
  within_noise: "bg-white text-[#666666] border border-[#ebebeb]",
  not_comparable: "bg-[#fafafa] text-[#8f8f8f] border border-dashed border-[#c9c9c9]",
};

const DiffPill: React.FC<{ diff: CompareDifference; text: string }> = ({ diff, text }) => (
  <span className={`inline-block rounded-md px-2.5 py-1 text-[12px] font-[500] leading-snug max-w-full break-words ${DIFF_STYLE[diff]}`}>
    {diff === "clearly_higher" ? "▲ " : diff === "clearly_lower" ? "▼ " : diff === "within_noise" ? "≈ " : ""}
    {text}
  </span>
);

const SideCell: React.FC<{ side: CompareSide; showStatus?: boolean }> = ({ side, showStatus }) => (
  <div className="min-w-0 pr-1">
    {showStatus && side.status && (
      <div className="mb-1">
        <StatusPill status={side.status} label={side.status_label} className="max-w-full" />
      </div>
    )}
    <div className="v-num text-[14px] text-[#171717]">{side.value_text ?? "—"}</div>
    {side.range_text && <div className="v-num text-[11px] text-[#8f8f8f]">{side.range_text}</div>}
  </div>
);

const GROUP_LABEL: Record<string, string> = {
  kinematic_sequence: "Sequence",
  com_braking: "Braking",
  head_stability: "Head",
  lead_arm: "Lead arm",
  crease_tunnel: "Release & crease",
  tempo_rhythm: "Tempo",
  other: "Other",
};

export const DeltaMatrix: React.FC<{ story: CompareStory | null; loading?: boolean; error?: string | null }> = ({
  story,
  loading,
  error,
}) => {
  if (loading) return <div className="v-panel p-4 text-[13px] text-[#8f8f8f]">Comparing the two balls…</div>;
  if (error || !story)
    return (
      <div className="v-panel p-4 text-[13px] text-[#4d4d4d]">
        {error ? `The comparison could not be loaded: ${error}` : "Pick two balls to see the delta matrix."}
      </div>
    );
  const ext = story.extended_rows ?? [];
  return (
    <div className="v-panel p-3 sm:p-4 min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="v-label">Delta matrix · noise-gated</span>
        <span className="text-[11px] text-[#8f8f8f]">B read relative to A</span>
      </div>
      <p className="mt-2 text-[13px] leading-[1.55] text-[#171717]">{story.summary}</p>
      {story.caveat_plain && <p className="mt-1 text-[12px] leading-[1.5] text-[#4d4d4d]">{story.caveat_plain}</p>}

      <div className="mt-3 hidden md:grid grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.35fr)] gap-x-5 px-1 v-label">
        <span>Checkpoint</span>
        <span>A</span>
        <span>B</span>
        <span>Change</span>
      </div>
      <ul className="mt-1 divide-y divide-[#ebebeb]">
        {story.rows.map((r) => (
          <li
            key={r.checkpoint_id}
            className="grid grid-cols-2 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.35fr)] gap-x-4 md:gap-x-5 gap-y-2.5 py-3 px-1 items-start"
          >
            <span className="col-span-2 md:col-span-1 text-[13px] font-[500] text-[#171717]">{r.title}</span>
            <SideCell side={r.a} showStatus />
            <SideCell side={r.b} showStatus />
            <span className="col-span-2 md:col-span-1 min-w-0">
              <DiffPill diff={r.difference} text={r.difference_plain} />
            </span>
          </li>
        ))}
      </ul>

      {ext.length > 0 && (
        <>
          <div className="mt-4 v-label">Extended v5 checks</div>
          <ul className="mt-1 divide-y divide-[#ebebeb]">
            {ext.map((r) => (
              <li
                key={r.metric_key}
                className="grid grid-cols-2 md:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)_minmax(0,1fr)_minmax(0,1.35fr)] gap-x-4 md:gap-x-5 gap-y-2.5 py-3 px-1 items-start"
              >
                <span className="col-span-2 md:col-span-1 min-w-0">
                  <span className="text-[13px] font-[500] text-[#171717]">{r.title}</span>
                  <span className="ml-1.5 text-[11px] text-[#8f8f8f]">{GROUP_LABEL[r.group] ?? ""}</span>
                </span>
                <SideCell side={r.a} />
                <SideCell side={r.b} />
                <span className="col-span-2 md:col-span-1 flex flex-wrap items-center gap-1.5 min-w-0">
                  <DiffPill diff={r.difference} text={r.difference_plain} />
                  {r.delta_text && r.difference !== "within_noise" && r.difference !== "not_comparable" && (
                    <span className="v-num text-[12px] text-[#4d4d4d]">{r.delta_text}</span>
                  )}
                </span>
              </li>
            ))}
          </ul>
        </>
      )}
    </div>
  );
};

/** A vs literature archetype envelope (descriptive overlap only; the envelope is a published range, never a target). */
export const ArchetypeEnvelopeMatrix: React.FC<{ factSheet: BiomechanicalFactSheet | null; archetype: Archetype }> = ({
  factSheet,
  archetype,
}) => {
  const env = Object.entries(archetype.metric_envelopes ?? {});
  return (
    <div className="v-panel p-3 sm:p-4 min-w-0">
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <span className="v-label">A vs literature envelope · {archetype.title}</span>
        <span className="v-chip text-[11px]">Synthetic archetype — not a real player</span>
      </div>
      <p className="mt-2 text-[12px] leading-[1.5] text-[#4d4d4d]">
        Each envelope is a plausible published range ({archetype.literature_source}). It is not a target or a score.
        "Overlaps" means A's measured interval touches the range.
      </p>
      {env.length === 0 ? (
        <p className="mt-3 text-[13px] text-[#8f8f8f]">This archetype ships no metric envelopes.</p>
      ) : (
        <ul className="mt-2 divide-y divide-[#ebebeb]">
          {env.map(([key, e]) => {
            const m = metricOf(factSheet, key);
            const usable = m && !m.suppressed && m.value !== null && m.ci_low !== null && m.ci_high !== null;
            let rel = "Not measured on A";
            if (usable) {
              rel = m!.ci_high! < e.low ? "A below range" : m!.ci_low! > e.high ? "A above range" : "Overlaps range";
            }
            const unit = e.unit ?? m?.unit ?? "";
            return (
              <li key={key} className="grid grid-cols-2 md:grid-cols-[1.4fr_1fr_1fr_1fr] gap-x-3 gap-y-1 py-2 px-1">
                <span className="col-span-2 md:col-span-1 text-[13px] text-[#171717]">{m?.label ?? key.replace(/_/g, " ")}</span>
                <span className="v-num text-[13px]">
                  {usable ? `${m!.value!.toFixed(1)}${unit} [${m!.ci_low!.toFixed(1)}–${m!.ci_high!.toFixed(1)}]` : m?.suppressed ? "Not measurable" : "—"}
                </span>
                <span className="v-num text-[12px] text-[#64748b]">
                  {e.low}–{e.high}
                  {unit}
                </span>
                <span className={`text-[12px] ${rel === "Overlaps range" ? "text-[#8f8f8f]" : "text-[#171717]"}`}>{rel}</span>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default DeltaMatrix;
