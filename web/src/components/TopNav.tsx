import React, { useState } from "react";
import { Plus, Upload, X } from "lucide-react";
import { useStudioStore, type TopView } from "../store";
import type { BowlerSummary, HealthResponse, SessionSummary } from "../types";

interface TopNavProps {
  health: HealthResponse | null;
  bowlers: BowlerSummary[];
  sessions: SessionSummary[];
  activeSession: SessionSummary | null;
  recomputeMs: number | null;
  onSelectSession: (sid: string) => void;
  onCreateSession: (payload: {
    bowler_id?: string;
    bowler_name?: string;
    bowling_arm?: string;
    age_band?: string;
    title: string;
    notes?: string;
  }) => Promise<void>;
  onOpenUpload: () => void;
}

const NAV_ITEMS: Array<{ id: TopView; label: string }> = [
  { id: "analysis", label: "Analysis" },
  { id: "compare", label: "Compare & Spell" },
  { id: "guide", label: "Filming Guide" },
  { id: "report", label: "Coach Report" },
];

export const TopNav: React.FC<TopNavProps> = ({
  health,
  bowlers,
  sessions,
  activeSession,
  recomputeMs,
  onSelectSession,
  onCreateSession,
  onOpenUpload,
}) => {
  const topView = useStudioStore((s) => s.topView);
  const setTopView = useStudioStore((s) => s.setTopView);

  const [showNewSession, setShowNewSession] = useState(false);
  const [bowlerMode, setBowlerMode] = useState<"existing" | "new">(bowlers.length > 0 ? "existing" : "new");
  const [selectedBowlerId, setSelectedBowlerId] = useState<string>(bowlers[0]?.id ?? "");
  const [newBowlerName, setNewBowlerName] = useState("");
  const [newArm, setNewArm] = useState<"right" | "left">("right");
  const [newAgeBand, setNewAgeBand] = useState("adult");
  const [sessionTitle, setSessionTitle] = useState("");
  const [sessionNotes, setSessionNotes] = useState("");
  const [creating, setCreating] = useState(false);
  const [createErr, setCreateErr] = useState<string | null>(null);

  const handleOpenNew = () => {
    setBowlerMode(bowlers.length > 0 ? "existing" : "new");
    setSelectedBowlerId(activeSession?.bowler_id ?? bowlers[0]?.id ?? "");
    setSessionTitle(`Net Session — ${new Date().toISOString().slice(0, 10)}`);
    setSessionNotes("");
    setCreateErr(null);
    setShowNewSession(true);
  };

  const handleSubmitSession = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionTitle.trim()) return;
    setCreating(true);
    setCreateErr(null);
    try {
      if (bowlerMode === "existing" && selectedBowlerId) {
        await onCreateSession({
          bowler_id: selectedBowlerId,
          title: sessionTitle.trim(),
          notes: sessionNotes.trim(),
        });
      } else {
        if (!newBowlerName.trim()) {
          setCreateErr("Enter a bowler name.");
          setCreating(false);
          return;
        }
        await onCreateSession({
          bowler_name: newBowlerName.trim(),
          bowling_arm: newArm,
          age_band: newAgeBand,
          title: sessionTitle.trim(),
          notes: sessionNotes.trim(),
        });
      }
      setShowNewSession(false);
      setNewBowlerName("");
    } catch (err: any) {
      setCreateErr(err?.message || "Could not create session.");
    } finally {
      setCreating(false);
    }
  };

  const gpuWarm = Boolean(health?.gpu_worker?.reachable);
  const localNlf = Boolean(health?.local_nlf_configured);

  return (
    <>
      <header className="no-print sticky top-0 z-40 h-16 bg-[#fafafa]/90 backdrop-blur-[20px] border-b border-[#ebebeb]">
        <div className="max-w-[1440px] h-full mx-auto px-4 md:px-6 flex items-center justify-between gap-3">
          {/* Left: ▲ Wordmark + Session Switcher */}
          <div className="flex items-center gap-3 min-w-0">
            <button
              onClick={() => setTopView("analysis")}
              className="flex items-center gap-2 text-left shrink-0 group"
              title="SportsPosture Coaching Studio"
            >
              <span className="text-[#000000] text-base leading-none select-none" aria-hidden="true">
                ▲
              </span>
              <span className="text-[15px] font-[500] tracking-[-0.03em] text-[#171717]">
                SportsPosture
              </span>
              <span className="hidden sm:inline font-mono text-[11px] text-[#666666] px-1.5 py-0.5 rounded-[4px] bg-[#f4f4f5]">
                v3
              </span>
            </button>

            <span className="text-[#c9c9c9] hidden md:inline select-none">/</span>

            <div className="hidden md:flex items-center gap-1.5 min-w-0">
              <select
                aria-label="Select coaching session"
                value={activeSession?.id ?? ""}
                onChange={(e) => onSelectSession(e.target.value)}
                className="v-input h-8 px-2.5 text-[13px] text-[#171717] max-w-[230px] truncate cursor-pointer"
              >
                {sessions.map((s) => (
                  <option key={s.id} value={s.id}>
                    {s.bowler_name} — {s.title}
                  </option>
                ))}
              </select>
              <button
                onClick={handleOpenNew}
                className="v-btn-ghost h-8 px-2 text-[12px] flex items-center gap-1 shrink-0"
                title="Create a new session (reusing an existing bowler or adding a new bowler)"
              >
                <Plus className="w-3.5 h-3.5 text-[#000000]" />
                <span className="hidden xl:inline">New</span>
              </button>
            </div>
          </div>

          {/* Center: 4 Plain-English Top-Level Views */}
          <nav aria-label="Primary studio navigation" className="flex items-center gap-1">
            {NAV_ITEMS.map((item) => {
              const active = topView === item.id;
              return (
                <button
                  key={item.id}
                  onClick={() => setTopView(item.id)}
                  className={`px-3 h-8 rounded-[6px] text-[13px] transition whitespace-nowrap ${
                    active
                      ? "bg-[#171717] text-[#ffffff] font-[500]"
                      : "text-[#4d4d4d] hover:text-[#171717] hover:bg-[#f4f4f5]"
                  }`}
                >
                  {item.label}
                </button>
              );
            })}
          </nav>

          {/* Right: Engine Pill, Permanent ICC Refusal Pill, Primary Upload CTA */}
          <div className="flex items-center gap-2 shrink-0">
            <span
              className="hidden lg:inline-flex items-center gap-1.5 px-2.5 h-7 rounded-full font-mono text-[11px] bg-[#ffffff] border border-[#ebebeb] text-[#4d4d4d]"
              title={
                gpuWarm
                  ? `Connected to Warm GPU Worker (${health?.gpu_worker?.gpu?.device_name ?? "CUDA"})`
                  : localNlf
                  ? "Local NLF TorchScript model configured for raw video inference"
                  : "Offline Local CPU Recompute Mode (<35 ms HITL); connect GPU_WORKER_URL or NLF_MODEL_PATH to infer raw videos"
              }
            >
              {gpuWarm ? (
                <>
                  <span className="text-[#297a3a] font-medium">✓</span>
                  <span className="text-[#171717]">GPU Warm</span>
                </>
              ) : localNlf ? (
                <>
                  <span className="text-[#297a3a] font-medium">✓</span>
                  <span className="text-[#171717]">Local NLF</span>
                </>
              ) : (
                <>
                  <span className="text-[#666666]">●</span>
                  <span>CPU · {recomputeMs != null ? `${recomputeMs.toFixed(0)}ms` : "<35ms"}</span>
                </>
              )}
            </span>

            <span
              className="hidden xl:inline-flex items-center px-2.5 h-7 rounded-full font-mono text-[11px] tracking-[0.03em] bg-[#171717] text-[#ffffff]"
              title="ICC 15-degree elbow extension legality is permanently REFUSED on monocular or casual video. Accredited >=250 Hz 3D optical lab testing is required."
            >
              ICC 15°: REFUSED
            </span>

            <button
              onClick={onOpenUpload}
              className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>Upload Video</span>
            </button>
          </div>
        </div>
      </header>

      {/* New Session / Bowler Modal */}
      {showNewSession && (
        <div className="no-print fixed inset-0 z-50 bg-[#000000]/35 backdrop-blur-[2px] flex items-center justify-center p-4">
          <div className="v-card w-full max-w-md p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="v-eyebrow">▲ SESSION_CREATE</div>
                <h2 className="text-[18px] font-[450] tracking-[-0.03em] text-[#171717] mt-0.5">
                  Start a Coaching Session
                </h2>
              </div>
              <button
                onClick={() => setShowNewSession(false)}
                className="v-btn-ghost w-8 h-8 flex items-center justify-center"
                aria-label="Close modal"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <form onSubmit={handleSubmitSession} className="space-y-3 text-[13px]">
              <div className="flex rounded-[6px] bg-[#f4f4f5] p-0.5">
                <button
                  type="button"
                  onClick={() => setBowlerMode("existing")}
                  disabled={bowlers.length === 0}
                  className={`flex-1 py-1.5 rounded-[5px] text-[12px] font-medium transition ${
                    bowlerMode === "existing" ? "bg-[#ffffff] text-[#171717]" : "text-[#666666]"
                  }`}
                >
                  Existing Bowler
                </button>
                <button
                  type="button"
                  onClick={() => setBowlerMode("new")}
                  className={`flex-1 py-1.5 rounded-[5px] text-[12px] font-medium transition ${
                    bowlerMode === "new" ? "bg-[#ffffff] text-[#171717]" : "text-[#666666]"
                  }`}
                >
                  New Bowler
                </button>
              </div>

              {bowlerMode === "existing" ? (
                <div className="space-y-1">
                  <label className="v-eyebrow block">Select Bowler (preserves 7d / 28d workload)</label>
                  <select
                    value={selectedBowlerId}
                    onChange={(e) => setSelectedBowlerId(e.target.value)}
                    className="v-input w-full h-9 px-2.5 text-[13px]"
                  >
                    {bowlers.map((b) => (
                      <option key={b.id} value={b.id}>
                        {b.name} ({b.bowling_arm}-arm · {b.age_band.toUpperCase()})
                      </option>
                    ))}
                  </select>
                </div>
              ) : (
                <div className="space-y-2.5">
                  <div className="space-y-1">
                    <label className="v-eyebrow block">Bowler Name</label>
                    <input
                      value={newBowlerName}
                      onChange={(e) => setNewBowlerName(e.target.value)}
                      placeholder="e.g. Arjun Sharma"
                      className="v-input w-full h-9 px-2.5 text-[13px]"
                    />
                  </div>
                  <div className="grid grid-cols-2 gap-2.5">
                    <div className="space-y-1">
                      <label className="v-eyebrow block">Bowling Arm</label>
                      <select
                        value={newArm}
                        onChange={(e) => setNewArm(e.target.value as "right" | "left")}
                        className="v-input w-full h-9 px-2.5 text-[13px]"
                      >
                        <option value="right">Right Arm</option>
                        <option value="left">Left Arm</option>
                      </select>
                    </div>
                    <div className="space-y-1">
                      <label className="v-eyebrow block">Age Band (ECB/CA)</label>
                      <select
                        value={newAgeBand}
                        onChange={(e) => setNewAgeBand(e.target.value)}
                        className="v-input w-full h-9 px-2.5 text-[13px]"
                      >
                        <option value="u13">Under-13</option>
                        <option value="u15">Under-15</option>
                        <option value="u17">Under-17</option>
                        <option value="u19">Under-19</option>
                        <option value="adult">Adult (19+)</option>
                      </select>
                    </div>
                  </div>
                </div>
              )}

              <div className="space-y-1">
                <label className="v-eyebrow block">Session Title</label>
                <input
                  value={sessionTitle}
                  onChange={(e) => setSessionTitle(e.target.value)}
                  placeholder="Morning Net Spell — Side + Rear"
                  className="v-input w-full h-9 px-2.5 text-[13px]"
                />
              </div>

              <div className="space-y-1">
                <label className="v-eyebrow block">Session Notes (Optional)</label>
                <input
                  value={sessionNotes}
                  onChange={(e) => setSessionNotes(e.target.value)}
                  placeholder="e.g. Outdoor turf net, 240 fps tripod"
                  className="v-input w-full h-9 px-2.5 text-[13px]"
                />
              </div>

              {createErr && (
                <div className="p-2.5 rounded-[6px] bg-[#f4f4f5] border border-[#171717] font-mono text-[11px] text-[#171717]">
                  {createErr}
                </div>
              )}

              <div className="flex justify-end gap-2 pt-1">
                <button type="button" onClick={() => setShowNewSession(false)} className="v-btn-ghost h-9 px-3.5 text-[13px]">
                  Cancel
                </button>
                <button type="submit" disabled={creating} className="v-btn-primary h-9 px-4 text-[13px]">
                  {creating ? "Creating..." : "Create Session"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
};
