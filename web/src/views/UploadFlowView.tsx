import React, { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, Camera, Columns2, Plus, Scissors, Upload } from "lucide-react";
import { api } from "../api";
import { navigate, setCurrentBowlerId, useCurrentBowlerId, type UploadMode } from "../router";
import type { OverviewBowler, OverviewResponse } from "../types";
import { GuardrailStrip } from "../components/ui/PaperPrimitives";
import { LiveCamera } from "../components/capture/LiveCamera";
import { UploadClipPanel } from "../components/capture/UploadClipPanel";
import { NetSessionClipper } from "../components/capture/NetSessionClipper";
import { DualPhonePairing } from "../components/capture/DualPhonePairing";
import { CameraAngleCoach } from "../components/capture/CameraAngleCoach";
import { Spinner, explainFailure, looksTechnical } from "../components/capture/shared";

/**
 * Studio v5 · Live Camera & Guided Upload Studio (`#/upload?bowler=&mode=`), docs/research_app_landscape_v5.md §8.6.
 * Four capture modes share one bowler selector. Live camera is framing-only: no live measurements, ever.
 */

type Arm = "right" | "left";
const AGE_BANDS = ["U13", "U15", "U17", "U19", "Adult"];

const MODES: Array<{ id: UploadMode; label: string; short: string; Icon: React.FC<{ className?: string }> }> = [
  { id: "live", label: "Live camera", short: "Live", Icon: Camera },
  { id: "clip", label: "Upload clip", short: "Upload", Icon: Upload },
  { id: "net", label: "Net session", short: "Net session", Icon: Scissors },
  { id: "dual", label: "Two phones", short: "Two phones", Icon: Columns2 },
];

// ── Bowler selector ────────────────────────────────────────────────────────
const BowlerSelector: React.FC<{
  bowlers: OverviewBowler[];
  loading: boolean;
  selectedId: string | null;
  onSelect: (id: string) => void;
  onCreated: (b: OverviewBowler) => void;
}> = ({ bowlers, loading, selectedId, onSelect, onCreated }) => {
  const [adding, setAdding] = useState(false);
  const [name, setName] = useState("");
  const [arm, setArm] = useState<Arm>("right");
  const [age, setAge] = useState("Adult");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!loading && bowlers.length === 0) setAdding(true);
  }, [loading, bowlers.length]);

  const save = async () => {
    const n = name.trim();
    if (!n) return;
    setSaving(true);
    setError(null);
    try {
      const res = await api.createBowler({ name: n, bowling_arm: arm, age_band: age });
      const b = res.bowler;
      onCreated({
        id: b.id,
        name: b.name,
        bowling_arm: b.bowling_arm ?? arm,
        age_band: b.age_band ?? age,
        delivery_count: 0,
        last_delivery_at: null,
      });
      setName("");
      setAdding(false);
    } catch (err: any) {
      setError(`We couldn't save this bowler. ${explainFailure(err?.message).fix}`);
    } finally {
      setSaving(false);
    }
  };

  const selected = bowlers.find((b) => b.id === selectedId) ?? null;

  return (
    <div className="v-panel p-3 sm:p-4">
      <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3">
        <label htmlFor="bowler-select" className="v-label shrink-0">
          Who's bowling?
        </label>
        {loading ? (
          <span className="inline-flex items-center gap-2 text-[13px] text-[#4d4d4d]">
            <Spinner /> Loading bowlers…
          </span>
        ) : (
          <div className="flex flex-1 items-center gap-2 min-w-0">
            {bowlers.length > 0 && (
              <select
                id="bowler-select"
                value={selectedId ?? ""}
                onChange={(e) => e.target.value && onSelect(e.target.value)}
                className="v-input h-9 px-2.5 text-[13.5px] min-w-0 flex-1 sm:flex-none sm:w-[260px]"
              >
                {!selected && (
                  <option value="" disabled>
                    Choose a bowler
                  </option>
                )}
                {bowlers.map((b) => (
                  <option key={b.id} value={b.id}>
                    {b.name} · {b.bowling_arm === "left" ? "Left-arm" : "Right-arm"}
                    {b.age_band ? ` · ${b.age_band}` : ""}
                  </option>
                ))}
              </select>
            )}
            {!adding && (
              <button type="button" onClick={() => setAdding(true)} className="v-btn-ghost h-9 px-3 text-[13px] inline-flex items-center gap-1.5 shrink-0">
                <Plus className="w-3.5 h-3.5" /> Add bowler
              </button>
            )}
            {selected && !adding && (
              <span className="hidden md:inline text-[12px] text-[#8f8f8f] ml-auto">
                <span className="v-num">{selected.delivery_count}</span> {selected.delivery_count === 1 ? "ball" : "balls"} analysed
              </span>
            )}
          </div>
        )}
      </div>

      {adding && (
        <div className="mt-3 pt-3 border-t border-[#ebebeb] grid gap-3 sm:grid-cols-[minmax(0,1.4fr)_auto_auto_auto] sm:items-end">
          <div>
            <label htmlFor="nb-name" className="block text-[12.5px] font-[500] text-[#171717] mb-1">
              Name
            </label>
            <input
              id="nb-name"
              value={name}
              maxLength={80}
              autoComplete="off"
              placeholder="e.g. Sam Patel"
              onChange={(e) => setName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && void save()}
              className="v-input w-full h-9 px-3 text-[13.5px]"
            />
          </div>
          <div>
            <div className="text-[12.5px] font-[500] text-[#171717] mb-1">Bowling arm</div>
            <div className="v-seg" role="group" aria-label="Bowling arm">
              {(["right", "left"] as Arm[]).map((a) => (
                <button key={a} type="button" aria-pressed={arm === a} onClick={() => setArm(a)}>
                  {a === "right" ? "Right-arm" : "Left-arm"}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label htmlFor="nb-age" className="block text-[12.5px] font-[500] text-[#171717] mb-1">
              Age group
            </label>
            <select id="nb-age" value={age} onChange={(e) => setAge(e.target.value)} className="v-input h-9 px-2.5 text-[13.5px] w-full">
              {AGE_BANDS.map((b) => (
                <option key={b} value={b}>
                  {b === "Adult" ? "Adult" : `Under ${b.slice(1)}`}
                </option>
              ))}
            </select>
          </div>
          <div className="flex gap-2">
            {bowlers.length > 0 && (
              <button type="button" onClick={() => setAdding(false)} className="v-btn-ghost h-9 px-3 text-[13px]">
                Cancel
              </button>
            )}
            <button type="button" disabled={!name.trim() || saving} onClick={save} className="v-btn-primary h-9 px-4 text-[13px] flex-1 sm:flex-none">
              {saving ? "Saving…" : "Save bowler"}
            </button>
          </div>
          <p className="sm:col-span-4 text-[12px] text-[#8f8f8f]">Age group is used to show the official weekly bowling workload guidance for that age.</p>
          {error && <p className="sm:col-span-4 text-[13px] text-[#171717]">{error}</p>}
        </div>
      )}
    </div>
  );
};

// ── Screen ─────────────────────────────────────────────────────────────────
export const UploadFlowView: React.FC<{ initialBowlerId: string | null; mode?: UploadMode }> = ({ initialBowlerId, mode = "clip" }) => {
  const currentBowlerId = useCurrentBowlerId();
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [extraBowlers, setExtraBowlers] = useState<OverviewBowler[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(initialBowlerId ?? currentBowlerId);

  useEffect(() => {
    let alive = true;
    api
      .overview()
      .then((o) => alive && setOverview(o))
      .catch(() => undefined)
      .finally(() => alive && setLoading(false));
    return () => {
      alive = false;
    };
  }, []);

  const bowlers = useMemo(() => {
    const base = overview?.bowlers ?? [];
    const seen = new Set(base.map((b) => b.id));
    return [...base, ...extraBowlers.filter((b) => !seen.has(b.id))];
  }, [overview, extraBowlers]);

  // Resolve the default bowler once the list is known.
  useEffect(() => {
    if (loading || bowlers.length === 0) return;
    if (selectedId && bowlers.some((b) => b.id === selectedId)) return;
    const fallback = (initialBowlerId && bowlers.find((b) => b.id === initialBowlerId)) || (currentBowlerId && bowlers.find((b) => b.id === currentBowlerId)) || bowlers[0];
    setSelectedId(fallback.id);
  }, [loading, bowlers, selectedId, initialBowlerId, currentBowlerId]);

  const selectBowler = useCallback((id: string) => {
    setSelectedId(id);
    setCurrentBowlerId(id);
  }, []);

  const bowler = bowlers.find((b) => b.id === selectedId) ?? null;
  const canAnalyse = overview ? overview.health?.can_analyse_new_video !== false : true;
  const reason = overview?.health?.reason_plain;

  const goMode = (m: UploadMode) => navigate({ name: "upload", bowler: bowler?.id ?? initialBowlerId ?? null, mode: m });

  return (
    <div className="min-h-full bg-[#fafafa]">
      <div className="max-w-[1120px] mx-auto px-4 sm:px-6 py-5 sm:py-8 space-y-4">
        <div className="flex items-center justify-between gap-3">
          <button type="button" onClick={() => navigate({ name: "home" })} className="v-btn-ghost h-8 px-2.5 text-[13px] inline-flex items-center gap-1.5">
            <ArrowLeft className="w-3.5 h-3.5" /> Home
          </button>
          {overview?.health?.engine_label && (
            <span className="v-chip text-[11.5px]">
              <span className={`w-1.5 h-1.5 rounded-full ${canAnalyse ? "bg-[#297a3a]" : "bg-[#c9c9c9]"}`} aria-hidden />
              {overview.health.engine_label}
            </span>
          )}
        </div>

        <div>
          <h1 className="text-[24px] sm:text-[28px] font-[500] tracking-[-0.03em] text-[#171717]">New analysis</h1>
          <p className="mt-1 text-[13.5px] text-[#4d4d4d]">Film or upload a delivery. Every measurement comes with an honest range.</p>
        </div>

        {/* Mode tabs */}
        <div role="tablist" aria-label="How do you want to add a delivery?" className="grid grid-cols-4 gap-1 p-1 rounded-[8px] bg-[#f4f4f5] border border-[#ebebeb]">
          {MODES.map(({ id, label, short, Icon }) => {
            const on = mode === id;
            return (
              <button
                key={id}
                type="button"
                role="tab"
                aria-selected={on}
                onClick={() => goMode(id)}
                className={`h-10 rounded-[6px] px-1.5 sm:px-3 inline-flex items-center justify-center gap-1.5 text-[12px] sm:text-[13.5px] font-[500] transition-colors duration-150 ${
                  on ? "bg-white text-[#171717] shadow-[0_0_0_1px_rgba(0,0,0,0.08)]" : "text-[#4d4d4d] hover:text-[#171717]"
                }`}
              >
                <Icon className="w-3.5 h-3.5 shrink-0 hidden sm:block" />
                <span className="sm:hidden truncate">{short}</span>
                <span className="hidden sm:inline">{label}</span>
              </button>
            );
          })}
        </div>

        <BowlerSelector
          bowlers={bowlers}
          loading={loading}
          selectedId={selectedId}
          onSelect={selectBowler}
          onCreated={(b) => {
            setExtraBowlers((p) => [...p, b]);
            selectBowler(b.id);
          }}
        />

        {!canAnalyse && (
          <div className="rounded-[6px] border border-dashed border-[#c9c9c9] bg-white p-4" role="status">
            <div className="text-[14px] font-[500] text-[#171717]">New videos can't be analysed right now</div>
            <p className="mt-1 text-[13px] leading-[1.55] text-[#4d4d4d]">
              {reason && !looksTechnical(reason) ? reason : "The analysis engine isn't connected at the moment."} You can still set up and practise framing with
              Live camera, and explore the sample deliveries meanwhile.
            </p>
            <button type="button" onClick={() => navigate({ name: "home" })} className="v-btn-ghost mt-3 h-8 px-3 text-[13px]">
              Explore sample deliveries
            </button>
          </div>
        )}

        <section role="tabpanel" aria-label={MODES.find((m) => m.id === mode)?.label} className="v-card p-4 sm:p-5">
          {mode === "live" && <LiveCamera bowler={bowler} canAnalyse={canAnalyse} onUseUpload={() => goMode("clip")} />}
          {mode === "clip" && <UploadClipPanel bowler={bowler} canAnalyse={canAnalyse} />}
          {mode === "net" && <NetSessionClipper bowler={bowler} canAnalyse={canAnalyse} />}
          {mode === "dual" && <DualPhonePairing bowler={bowler} canAnalyse={canAnalyse} />}
        </section>

        <CameraAngleCoach defaultOpen={mode === "dual"} />

        <GuardrailStrip />
      </div>
    </div>
  );
};

export default UploadFlowView;
