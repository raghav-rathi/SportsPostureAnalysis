import React, { useMemo, useState } from "react";
import { ChevronDown, ChevronRight, MessageSquare, Plus, Send, Trash2 } from "lucide-react";
import { api } from "../api";
import { useStudioStore } from "../store";
import type { CoachNote, CohortComparison, DeliveryDetailResponse, MetricEstimate } from "../types";

interface CoachSummaryPanelProps {
  detail: DeliveryDetailResponse;
  onNotesChanged: (notes: CoachNote[]) => void;
  onOpenFilmingGuide: () => void;
}

const PLAIN_EXPLANATIONS: Record<string, string> = {
  front_knee_br_deg: "Higher angle (closer to 180°) means a firmer front-leg brace transferring approach momentum into ball speed.",
  front_knee_ffc_deg: "Knee angle at initial front-foot plant before absorbing landing forces.",
  front_knee_change_deg: "Positive means extending through release; large negative values indicate knee collapse.",
  trunk_flexion_delta_deg: "Forward torso whip from front-foot plant to release — a primary pace driver (Worthington 2013).",
  thorax_lat_flex_br_deg: "Contralateral side-bend away from the bowling arm at release; extreme values raise unilateral lumbar load.",
  shoulder_counter_rotation_deg: "How far shoulders rotate back toward side-on after back-foot contact; ≥30° classifies as a mixed action.",
  pelvis_alignment_bfc_deg: "Pelvis angle at back-foot plant (180° = side-on, 270° = front-on).",
  shoulder_alignment_bfc_deg: "Shoulder line angle at back-foot plant (180° = side-on, 270° = front-on).",
  hip_shoulder_sep_bfc_deg: "Twist separation between hips and shoulders at back-foot contact.",
  max_hip_shoulder_sep_deg: "Peak elastic coil between pelvis and thorax through delivery.",
  delayed_arm_ffc_deg: "How far behind vertical the bowling arm stays at front-foot plant (delayed arm adds whip).",
  pelvic_drop_ffc_deg: "Contralateral hip drop at front-foot impact; lower drop indicates stronger frontal-plane stability.",
  runup_speed_mps: "Pelvis forward speed approaching back-foot contact (requires locked camera + crease scale).",
  stride_length_pct_stature: "Back-foot to front-foot delivery stride as a percentage of bowler height.",
  release_height_m: "Vertical height of the bowling wrist above ground at release.",
  ball_release_speed_kph: "Optical ballistic speed tracked right after wrist release (requires ≥120 fps calibrated side-on view).",
};

const METRIC_GROUPS: Array<{ title: string; eyebrow: string; keys: string[] }> = [
  {
    title: "Front-Leg Brace & Momentum Transfer",
    eyebrow: "LOWER_BODY",
    keys: ["front_knee_br_deg", "front_knee_ffc_deg", "front_knee_change_deg", "rear_hip_bfc_deg"],
  },
  {
    title: "Trunk Drive & Side-Bend Load",
    eyebrow: "TRUNK_SPINE",
    keys: ["trunk_flexion_delta_deg", "thorax_lat_flex_br_deg", "trunk_flexion_br_deg", "pelvic_drop_ffc_deg"],
  },
  {
    title: "Alignment & Counter-Rotation",
    eyebrow: "TRANSVERSE_ACTION",
    keys: [
      "shoulder_counter_rotation_deg",
      "pelvis_alignment_bfc_deg",
      "shoulder_alignment_bfc_deg",
      "hip_shoulder_sep_bfc_deg",
      "max_hip_shoulder_sep_deg",
    ],
  },
  {
    title: "Arm Whip, Speed & Stride",
    eyebrow: "PACE_OUTPUT",
    keys: [
      "ball_release_speed_kph",
      "delayed_arm_ffc_deg",
      "peak_arm_angular_vel_dps",
      "runup_speed_mps",
      "stride_length_pct_stature",
      "release_height_m",
      "follow_through_drift_deg",
    ],
  },
];

const CohortBandBar: React.FC<{ comp: CohortComparison }> = ({ comp }) => {
  const minVal = comp.cohort_mean - 2.5 * comp.cohort_sd;
  const maxVal = comp.cohort_mean + 2.5 * comp.cohort_sd;
  const span = Math.max(maxVal - minVal, 1e-4);
  const toPct = (v: number) => Math.max(2, Math.min(98, ((v - minVal) / span) * 100));
  const bandLo = toPct(comp.cohort_mean - comp.cohort_sd);
  const bandHi = toPct(comp.cohort_mean + comp.cohort_sd);
  const ciLo = toPct(comp.ci_low);
  const ciHi = toPct(comp.ci_high);
  const pt = toPct(comp.value);

  return (
    <div className="mt-2 space-y-1">
      <div className="relative h-4 w-full rounded-[3px] bg-[#fafafa] border border-[#ebebeb] overflow-hidden">
        {/* Cohort ±1 SD band in subtle Terminal Green tint */}
        <div
          className="absolute top-0 bottom-0 bg-[#297a3a]/[0.10] border-x border-[#297a3a]/30"
          style={{ left: `${bandLo}%`, width: `${Math.max(bandHi - bandLo, 2)}%` }}
          title={`Cohort ±1 SD: ${(comp.cohort_mean - comp.cohort_sd).toFixed(1)} to ${(comp.cohort_mean + comp.cohort_sd).toFixed(1)} ${comp.unit}`}
        />
        {/* Cohort mean hairline */}
        <div className="absolute top-0 bottom-0 w-px bg-[#297a3a]/50" style={{ left: `${toPct(comp.cohort_mean)}%` }} />
        {/* Bowler 95% CI whisker */}
        <div
          className="absolute top-1/2 -translate-y-1/2 h-[2px] bg-[#171717]"
          style={{ left: `${ciLo}%`, width: `${Math.max(ciHi - ciLo, 1.5)}%` }}
        />
        {/* Bowler value point */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-2.5 h-2.5 rounded-full bg-[#171717] border border-[#ffffff]"
          style={{ left: `${pt}%` }}
        />
      </div>
      <div className="flex items-center justify-between font-mono text-[10px] text-[#666666]">
        <span>
          Cohort: {comp.cohort_mean.toFixed(1)} ± {comp.cohort_sd.toFixed(1)} {comp.unit} ({comp.source})
        </span>
        <span className="v-tabular">z = {comp.z_score >= 0 ? `+${comp.z_score.toFixed(2)}` : comp.z_score.toFixed(2)}</span>
      </div>
    </div>
  );
};

const renderInlineMarkdown = (text: string): React.ReactNode => {
  const lines = text.split("\n");
  return lines.map((line, li) => {
    const tokens = line.split(/(\*\*[^*]+\*\*|`[^`]+`|\*[^*]+\*)/g);
    return (
      <div key={li} className={line.trim() === "" ? "h-2" : ""}>
        {tokens.map((tok, ti) => {
          if (tok.startsWith("**") && tok.endsWith("**") && tok.length > 4) {
            return (
              <strong key={ti} className="font-[500] text-[#171717]">
                {tok.slice(2, -2)}
              </strong>
            );
          }
          if (tok.startsWith("`") && tok.endsWith("`") && tok.length > 2) {
            return (
              <code key={ti} className="font-mono text-[11px] px-1 py-0.5 rounded-[3px] bg-[#ffffff] border border-[#ebebeb] text-[#171717]">
                {tok.slice(1, -1)}
              </code>
            );
          }
          if (tok.startsWith("*") && tok.endsWith("*") && tok.length > 2) {
            return (
              <em key={ti} className="italic text-[#666666]">
                {tok.slice(1, -1)}
              </em>
            );
          }
          return <React.Fragment key={ti}>{tok}</React.Fragment>;
        })}
      </div>
    );
  });
};

export const CoachSummaryPanel: React.FC<CoachSummaryPanelProps> = ({
  detail,
  onNotesChanged,
  onOpenFilmingGuide,
}) => {
  const fs = detail.fact_sheet;
  const currentFrame = useStudioStore((s) => s.currentFrame);
  const setCurrentFrame = useStudioStore((s) => s.setCurrentFrame);
  const setIsPlaying = useStudioStore((s) => s.setIsPlaying);

  const [showWithheld, setShowWithheld] = useState(false);

  // Coach Notes state
  const [noteBody, setNoteBody] = useState("");
  const [noteTag, setNoteTag] = useState("cue");
  const [savingNote, setSavingNote] = useState(false);

  // Coach Copilot state
  const [coachQ, setCoachQ] = useState("");
  const [coachBusy, setCoachBusy] = useState(false);
  const [coachReply, setCoachReply] = useState<{
    answer_markdown: string;
    verification_passed: boolean;
    generator_mode: string;
  } | null>(null);

  const cohortByKey = useMemo(() => {
    const m: Record<string, CohortComparison> = {};
    for (const c of fs.cohort_comparisons) m[c.metric_key] = c;
    return m;
  }, [fs.cohort_comparisons]);

  const { verifiedList, withheldList } = useMemo(() => {
    const v: MetricEstimate[] = [];
    const w: MetricEstimate[] = [];
    for (const m of Object.values(fs.metrics)) {
      if (m.key === "icc_elbow_legality_verdict") continue;
      if (!m.suppressed && m.value !== null) v.push(m);
      else w.push(m);
    }
    return { verifiedList: v, withheldList: w };
  }, [fs.metrics]);

  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!noteBody.trim()) return;
    setSavingNote(true);
    try {
      const ts = Number((currentFrame / Math.max(fs.audit.fps, 1)).toFixed(3));
      const res = await api.addNote(detail.delivery.id, {
        frame_idx: currentFrame,
        timestamp_s: ts,
        tag: noteTag,
        body: noteBody.trim(),
      });
      onNotesChanged([res.note, ...(detail.notes || [])]);
      setNoteBody("");
    } finally {
      setSavingNote(false);
    }
  };

  const handleDeleteNote = async (noteId: string) => {
    await api.deleteNote(detail.delivery.id, noteId);
    onNotesChanged((detail.notes || []).filter((n) => n.id !== noteId));
  };

  const handleAskCoach = async (prompt: string) => {
    const q = prompt.trim();
    if (!q) return;
    setCoachQ(q);
    setCoachBusy(true);
    try {
      const ans = await api.askCoach(detail.delivery.id, q);
      setCoachReply(ans);
    } finally {
      setCoachBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      {/* 1. Executive Takeaway Card */}
      <div className="v-card p-4 space-y-3">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <div className="v-eyebrow">▲ COACH_SUMMARY</div>
            <h2 className="text-[20px] font-[450] tracking-[-0.04em] text-[#171717] mt-0.5">
              {!fs.events.is_complete_delivery
                ? "Incomplete Delivery Clip — Metrics Withheld"
                : fs.action_class === "unknown"
                ? `${fs.bowling_arm === "right" ? "Right-Arm" : "Left-Arm"} Delivery — Action Class Unverified`
                : `${fs.bowling_arm === "right" ? "Right-Arm" : "Left-Arm"} ${fs.action_class.replace("_", "-").toUpperCase()} Action`}
            </h2>
          </div>
          <div className="flex items-center gap-1.5 font-mono text-[11px]">
            <span className="px-2 py-0.5 rounded-[4px] bg-[#297a3a]/[0.08] text-[#297a3a] border border-[#297a3a]/25">
              ✓ {verifiedList.length} verified
            </span>
            <span className="px-2 py-0.5 rounded-[4px] bg-[#fafafa] text-[#666666] border border-[#ebebeb]">
              {withheldList.length} withheld
            </span>
          </div>
        </div>

        {/* Plain-English Takeaway */}
        <p className="text-[13px] leading-relaxed text-[#4d4d4d]">
          {!fs.events.is_complete_delivery ? (
            <>
              This clip did not pass overhead ball-release sanity checks (e.g., freeze-frames or cut before release).
              No biomechanical angles or coaching cues were fabricated. Upload an unedited full delivery to unlock measurements.
            </>
          ) : (
            <>
              Captured from <strong className="text-[#171717]">{fs.calibration.view_class.replace(/_/g, " ")}</strong> (
              {fs.calibration.azimuth_deg.toFixed(0)}°) at{" "}
              <strong className="text-[#171717]">{fs.audit.fps.toFixed(0)} fps</strong>.{" "}
              {fs.strengths.length > 0 && `Key strength: ${fs.strengths[0].title}. `}
              {fs.limiters.length > 0
                ? `Primary focus: ${fs.limiters[0].title}.`
                : fs.inconclusive_observations.length > 0
                ? `${fs.inconclusive_observations.length} observation(s) sit on our Watchlist because the 95% interval straddles the coaching threshold.`
                : "All measurable mechanics sit within expected cohort ranges."}
            </>
          )}
        </p>

        {/* Quick Strengths & Limiters Strip */}
        {(fs.strengths.length > 0 || fs.limiters.length > 0) && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-1">
            {fs.strengths.map((s, i) => (
              <div key={`str-${i}`} className="p-2.5 rounded-[6px] bg-[#297a3a]/[0.05] border border-[#297a3a]/25">
                <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#297a3a]">✓ STRENGTH</div>
                <div className="text-[13px] font-[500] text-[#171717] mt-0.5">{s.title}</div>
                <div className="text-[12px] text-[#4d4d4d] mt-0.5">{s.biomechanical_mechanism}</div>
              </div>
            ))}
            {fs.limiters.map((l, i) => (
              <div key={`lim-${i}`} className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#171717]">
                <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#171717]">▲ VERIFIED LIMITER</div>
                <div className="text-[13px] font-[500] text-[#171717] mt-0.5">{l.title}</div>
                <div className="text-[12px] text-[#4d4d4d] mt-0.5">{l.actionable_recommendation}</div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* 2. Evidence-Grounded Corrective Drills & Watchlist (Feature #4) */}
      {(fs.corrective_drills.length > 0 || fs.inconclusive_observations.length > 0) && (
        <div className="v-card p-4 space-y-3">
          <div>
            <div className="v-eyebrow">▲ TRAINING_PLAN · INTERVAL-GATED</div>
            <h3 className="text-[16px] font-[500] text-[#171717] mt-0.5">
              Corrective Drills &amp; Watchlist
            </h3>
          </div>

          {fs.corrective_drills.map((dr) => (
            <div key={dr.id} className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="text-[13px] font-[500] text-[#171717]">{dr.title}</span>
                <span className="font-mono text-[10px] px-1.5 py-0.5 rounded-[4px] bg-[#297a3a]/[0.08] text-[#297a3a]">
                  ✓ PRESCRIBED
                </span>
              </div>
              <div className="text-[12px] text-[#171717]">
                <strong>Cue:</strong> {dr.cue_focus}
              </div>
              {dr.setup_steps.length > 0 && (
                <ul className="list-disc pl-4 text-[12px] text-[#4d4d4d] space-y-0.5">
                  {dr.setup_steps.map((st, idx) => (
                    <li key={idx}>{st}</li>
                  ))}
                </ul>
              )}
              <div className="font-mono text-[10px] text-[#666666] pt-1 border-t border-[#ebebeb]">
                Basis: {dr.literature_basis} · Caution: {dr.caution_note}
              </div>
            </div>
          ))}

          {fs.inconclusive_observations.length > 0 && (
            <div className="space-y-1.5">
              <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#666666]">
                WATCHLIST — INTERVAL STRADDLES THRESHOLD (CONFIRM WITH HIGHER FPS / BETTER ANGLE FIRST)
              </div>
              {fs.inconclusive_observations.map((obs, idx) => (
                <div key={idx} className="p-2.5 rounded-[6px] border border-dashed border-[#c9c9c9] bg-[#ffffff]">
                  <div className="text-[13px] font-[500] text-[#171717]">{obs.title}</div>
                  <div className="text-[12px] text-[#666666] mt-0.5">{obs.actionable_recommendation}</div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* 3. Verified Measurements Grouped in Plain English */}
      {verifiedList.length > 0 && (
        <div className="space-y-3">
          {METRIC_GROUPS.map((grp) => {
            const items = grp.keys
              .map((k) => fs.metrics[k])
              .filter((m): m is MetricEstimate => Boolean(m && !m.suppressed && m.value !== null));
            if (items.length === 0) return null;
            return (
              <div key={grp.eyebrow} className="v-card p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div>
                    <div className="v-eyebrow">{grp.eyebrow}</div>
                    <h3 className="text-[15px] font-[500] text-[#171717]">{grp.title}</h3>
                  </div>
                  <span className="font-mono text-[10px] text-[#8f8f8f]">95% {fs.interval_type.replace(/_/g, " ")}</span>
                </div>
                <div className="divide-y divide-[#ebebeb]">
                  {items.map((m) => (
                    <div key={m.key} className="py-2.5 first:pt-0 last:pb-0">
                      <div className="flex items-baseline justify-between gap-2">
                        <div className="text-[13px] font-[500] text-[#171717]">{m.label}</div>
                        <div className="text-right shrink-0">
                          <span className="font-mono text-[16px] font-[500] text-[#171717] v-tabular">
                            {m.value?.toFixed(1)}
                          </span>
                          <span className="font-mono text-[11px] text-[#666666] ml-1">{m.unit}</span>
                          {m.ci_low != null && m.ci_high != null && (
                            <span className="font-mono text-[11px] text-[#8f8f8f] ml-2 v-tabular">
                              [{m.ci_low.toFixed(1)}, {m.ci_high.toFixed(1)}]
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="flex items-center justify-between gap-2 mt-0.5">
                        <p className="text-[12px] text-[#666666]">
                          {PLAIN_EXPLANATIONS[m.key] || m.definition_note}
                        </p>
                        <span
                          className={`font-mono text-[10px] uppercase px-1.5 py-0.5 rounded-[4px] shrink-0 ${
                            m.observability === "high"
                              ? "text-[#297a3a] bg-[#297a3a]/[0.08]"
                              : "text-[#666666] bg-[#f4f4f5]"
                          }`}
                        >
                          {m.observability}
                        </span>
                      </div>
                      {cohortByKey[m.key] && <CohortBandBar comp={cohortByKey[m.key]} />}
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* 4. Timestamped Coach Notes (Feature #5) */}
      <div className="v-card p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div>
            <div className="v-eyebrow">▲ COACH_NOTES</div>
            <h3 className="text-[15px] font-[500] text-[#171717]">Timestamped Frame Notes</h3>
          </div>
          <span className="font-mono text-[11px] text-[#666666]">Pinned to f {currentFrame}</span>
        </div>

        <form onSubmit={handleAddNote} className="flex flex-wrap gap-2">
          <select
            value={noteTag}
            onChange={(e) => setNoteTag(e.target.value)}
            className="v-input h-8 px-2 font-mono text-[11px]"
          >
            <option value="cue">cue</option>
            <option value="good">good</option>
            <option value="fix">fix</option>
            <option value="question">question</option>
          </select>
          <input
            value={noteBody}
            onChange={(e) => setNoteBody(e.target.value)}
            placeholder={`Add note at frame ${currentFrame}...`}
            className="v-input flex-1 min-w-[160px] h-8 px-2.5 text-[12px]"
          />
          <button type="submit" disabled={savingNote || !noteBody.trim()} className="v-btn-primary h-8 px-3 text-[12px] flex items-center gap-1">
            <Plus className="w-3.5 h-3.5" />
            Save
          </button>
        </form>

        {(detail.notes?.length ?? 0) > 0 ? (
          <div className="space-y-1.5">
            {detail.notes.map((n) => (
              <div
                key={n.id}
                className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] flex items-start justify-between gap-2 text-[12px]"
              >
                <div className="min-w-0 space-y-0.5">
                  <div className="flex items-center gap-1.5 font-mono text-[10px]">
                    <span className="px-1.5 py-0.5 rounded-[4px] bg-[#171717] text-[#ffffff] uppercase">{n.tag}</span>
                    {n.frame_idx != null && (
                      <button
                        onClick={() => {
                          setIsPlaying(false);
                          setCurrentFrame(n.frame_idx!);
                        }}
                        className="text-[#297a3a] hover:underline"
                      >
                        Jump to f {n.frame_idx} ({(n.timestamp_s ?? 0).toFixed(2)}s)
                      </button>
                    )}
                  </div>
                  <p className="text-[#171717]">{n.body}</p>
                </div>
                <button
                  onClick={() => handleDeleteNote(n.id)}
                  className="text-[#8f8f8f] hover:text-[#171717] p-1"
                  title="Delete note"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-[12px] text-[#8f8f8f]">No notes yet — scrub to a key frame and pin a coaching cue.</div>
        )}
      </div>

      {/* 5. Verifier-Gated Coach AI Copilot */}
      <div className="v-card p-4 space-y-3">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-4 h-4 text-[#000000]" />
            <div>
              <div className="v-eyebrow">GROUNDED_COPILOT</div>
              <h3 className="text-[15px] font-[500] text-[#171717]">Ask About This Delivery</h3>
            </div>
          </div>
          {coachReply && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded-[4px] bg-[#297a3a]/[0.08] text-[#297a3a]">
              {coachReply.verification_passed ? "✓ VERIFIER PASSED" : "BLOCKED"}
            </span>
          )}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {[
            "Summarize verified measurements",
            "What drills should I practice?",
            "Why were metrics withheld?",
            "What does this mean for lumbar load?",
            "Is this action ICC legal?",
          ].map((q) => (
            <button
              key={q}
              onClick={() => handleAskCoach(q)}
              className="v-btn-ghost px-2.5 py-1 text-[11px] font-mono text-[#4d4d4d]"
            >
              {q}
            </button>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleAskCoach(coachQ);
          }}
          className="flex gap-2"
        >
          <input
            value={coachQ}
            onChange={(e) => setCoachQ(e.target.value)}
            placeholder="Ask a biomechanics question grounded strictly in verified metrics..."
            className="v-input flex-1 h-8 px-2.5 text-[12px]"
          />
          <button type="submit" disabled={coachBusy || !coachQ.trim()} className="v-btn-primary h-8 px-3 text-[12px] flex items-center gap-1">
            <Send className="w-3.5 h-3.5" />
            Ask
          </button>
        </form>

        {coachReply && (
          <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] text-[#4d4d4d] leading-relaxed space-y-0.5">
            {renderInlineMarkdown(coachReply.answer_markdown)}
          </div>
        )}
      </div>

      {/* 6. Plain-English Collapsible Accordion: Why Metrics Were Withheld */}
      <div className="v-card overflow-hidden">
        <button
          onClick={() => setShowWithheld(!showWithheld)}
          className="w-full px-4 py-3 flex items-center justify-between text-left hover:bg-[#fafafa] transition"
        >
          <div>
            <div className="v-eyebrow">TRANSPARENCY_AUDIT</div>
            <div className="text-[13px] font-[500] text-[#171717]">
              Why {withheldList.length} Metric{withheldList.length === 1 ? "" : "s"} Were Withheld (And How to Unlock Them)
            </div>
          </div>
          {showWithheld ? <ChevronDown className="w-4 h-4 text-[#666666]" /> : <ChevronRight className="w-4 h-4 text-[#666666]" />}
        </button>

        {showWithheld && (
          <div className="px-4 pb-4 pt-2 border-t border-[#ebebeb] space-y-3">
            {fs.capture_upgrades.length > 0 && (
              <div className="space-y-1.5">
                <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#297a3a]">
                  RECOMMENDED FILMING UPGRADES
                </div>
                {fs.capture_upgrades.map((u, i) => (
                  <div key={i} className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px]">
                    <div className="font-[500] text-[#171717]">{u.title}</div>
                    <div className="text-[#4d4d4d] mt-0.5">{u.actionable_recommendation}</div>
                  </div>
                ))}
                <button onClick={onOpenFilmingGuide} className="v-btn-ghost h-8 px-3 text-[12px] font-mono">
                  Open 30-Second Visual Filming Guide →
                </button>
              </div>
            )}

            <div className="divide-y divide-[#ebebeb] text-[12px]">
              {withheldList.map((m) => (
                <div key={m.key} className="py-2 flex flex-col gap-0.5">
                  <div className="font-[500] text-[#171717]">{m.label}</div>
                  <div className="text-[#666666]">{m.suppression_reason}</div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* 7. Non-Negotiable Scientific & Medical Guardrail Footer */}
      <div className="v-card-subtle p-3.5 text-[11px] text-[#4d4d4d] space-y-1.5">
        <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#171717]">
          ▲ PERMANENT SCIENTIFIC GUARDRAILS
        </div>
        <p>
          <strong className="text-[#171717]">ICC 15° Elbow Legality — PERMANENTLY REFUSED:</strong>{" "}
          {fs.metrics["icc_elbow_legality_verdict"]?.suppression_reason}
        </p>
        <p>
          <strong className="text-[#171717]">Medical Non-Diagnosis:</strong> Single-delivery video biomechanics
          estimates mechanical stress only and cannot diagnose or predict injury risk in isolation (depends on workload,
          age/skeletal maturity, and physical capacity).
        </p>
      </div>
    </div>
  );
};
