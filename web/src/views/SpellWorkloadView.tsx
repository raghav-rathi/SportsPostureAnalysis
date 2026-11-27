import React, { useCallback, useMemo } from "react";
import { api } from "../api";
import { navigate, setCurrentBowlerId, useCurrentBowlerId } from "../router";
import type { OverviewBowler } from "../types";
import { GuardrailStrip } from "../components/ui/PaperPrimitives";
import { ReleaseBullseye } from "../components/spell/ReleaseBullseye";
import { RepeatabilityTable, SpellDrift } from "../components/spell/SpellDrift";
import {
  ACWR_CAVEAT,
  DailyLoopCard,
  LimitsPanel,
  LoadChangeChart,
  LogHistory,
  QuickLogger,
  SorenessLog,
  WorkloadCalendar,
} from "../components/spell/WorkloadPanels";
import { CalmState, SectionTitle, useKeyedAsync } from "../components/spell/ui";

/**
 * Spell, Release Tunnel & Workload Hub (docs/research_app_landscape_v5.md §8.4).
 * A. Release-point bullseye · B. Spell drift + repeatability · C. Workload & wellness (limits first, ratio second).
 * Each section loads independently and fails calmly. All statuses are rendered verbatim from the backend.
 */
export const SpellWorkloadView: React.FC<{ bowlerId: string | null; sessionId?: string | null }> = ({ bowlerId, sessionId = null }) => {
  const storedBowler = useCurrentBowlerId();
  const overview = useKeyedAsync("overview", () => api.overview());
  const bowlers: OverviewBowler[] = overview.data?.bowlers ?? [];

  const fallbackBowler = useMemo(() => {
    if (bowlers.length === 0) return null;
    return [...bowlers].sort((a, b) => b.delivery_count - a.delivery_count)[0].id;
  }, [bowlers]);
  const resolved = bowlerId ?? storedBowler ?? fallbackBowler;

  const spell = useKeyedAsync(resolved ? `spell:${resolved}:${sessionId ?? ""}` : null, () =>
    api.spellAnalytics(resolved as string, sessionId)
  );
  const workload = useKeyedAsync(resolved ? `workload:${resolved}` : null, () => api.workload(resolved as string));
  const progress = useKeyedAsync(resolved ? `progress:${resolved}` : null, () => api.progress(resolved as string));

  const bowlerInfo =
    bowlers.find((b) => b.id === resolved) ?? workload.data?.bowler ?? progress.data?.bowler ?? null;

  const labels = useMemo(() => {
    const m: Record<string, string> = {};
    progress.data?.deliveries.forEach((d) => (m[d.id] = d.label));
    spell.data?.release_tunnel?.points.forEach((p) => (m[p.delivery_id] = p.label));
    spell.data?.release_tunnel?.excluded?.forEach((x) => (m[x.delivery_id] = x.label));
    return m;
  }, [progress.data, spell.data]);

  const pickBowler = useCallback(
    (id: string) => {
      setCurrentBowlerId(id);
      navigate({ name: "spell", bowler: id });
    },
    []
  );

  const reloadWorkload = workload.reload;
  const wl = workload.data;
  const logs = wl?.logs ?? [];
  const guidance = wl?.guidance ?? progress.data?.workload ?? null;

  return (
    <div className="mx-auto w-full max-w-[1200px] px-4 sm:px-6 py-6 space-y-8" style={{ background: "#fafafa" }}>
      {/* Header + bowler switcher */}
      <header className="space-y-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div className="min-w-0">
            <div className="v-label">Spell &amp; workload</div>
            <h1 className="text-[26px] sm:text-[30px] font-medium tracking-[-0.02em] text-[#171717] leading-tight mt-1">
              {bowlerInfo ? bowlerInfo.name : "Spell & Workload"}
            </h1>
            {bowlerInfo && (
              <div className="text-[13px] text-[#666] mt-1">
                {bowlerInfo.bowling_arm === "left" ? "Left-arm" : "Right-arm"} · {bowlerInfo.age_band}
                {spell.data ? ` · ${spell.data.completed_count} analysed ball${spell.data.completed_count === 1 ? "" : "s"}` : ""}
              </div>
            )}
          </div>
          {sessionId && resolved && (
            <div className="flex items-center gap-2">
              <span className="v-chip v-num">Session {sessionId.slice(0, 10)}</span>
              <button type="button" className="v-btn-ghost h-7 px-2.5 text-[12px]" onClick={() => navigate({ name: "spell", bowler: resolved })}>
                All sessions
              </button>
            </div>
          )}
        </div>
        {bowlers.length > 1 && (
          <div className="flex flex-wrap gap-1.5" role="group" aria-label="Choose bowler">
            {bowlers.map((b) => (
              <button
                key={b.id}
                type="button"
                aria-pressed={b.id === resolved}
                className={`v-chip h-8 px-3 text-[13px] ${b.id === resolved ? "v-chip-ink" : ""}`}
                onClick={() => pickBowler(b.id)}
              >
                {b.name}
                <span className={`v-num text-[11px] ${b.id === resolved ? "text-[#c9c9c9]" : "text-[#8f8f8f]"}`}>{b.delivery_count}</span>
              </button>
            ))}
          </div>
        )}
      </header>

      {!resolved ? (
        overview.loading ? (
          <CalmState tone="loading" title="Loading" />
        ) : (
          <CalmState
            tone={overview.error ? "error" : "empty"}
            title={overview.error ? "Couldn't load bowlers" : "No bowler yet"}
            body={
              overview.error ? (
                overview.error
              ) : (
                <>
                  Add a bowler by analysing a first clip.{" "}
                  <button type="button" className="underline underline-offset-2" onClick={() => navigate({ name: "upload", bowler: null })}>
                    New analysis
                  </button>
                </>
              )
            }
          />
        )
      ) : (
        <>
          {/* A + B: spell analytics */}
          <section aria-labelledby="spell-heading" className="grid gap-4 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)]">
            <div className="v-card p-4 sm:p-5 min-w-0">
              <SectionTitle
                eyebrow="Release bullseye · batter's view"
                title="Where the ball leaves your hand"
                sub="One dot per ball with its 95% measurement ring. Tap a dot to open that ball."
              />
              {spell.loading && !spell.data ? (
                <CalmState tone="loading" title="Loading" />
              ) : spell.error ? (
                <CalmState title="Spell analytics not available yet" body={spell.error} tone="error" />
              ) : !spell.data?.release_tunnel ? (
                <CalmState
                  title="Not enough measured releases yet"
                  body="The bullseye appears once two or more balls in a spell have a measured release point."
                />
              ) : (
                <ReleaseBullseye tunnel={spell.data.release_tunnel} bowlingArm={bowlerInfo?.bowling_arm ?? null} />
              )}
            </div>

            <div className="v-card p-4 sm:p-5 min-w-0">
              <SectionTitle
                eyebrow="Spell drift · ball 1 → N"
                title="Does anything change as you tire?"
                sub="Only flagged when the change is bigger than measurement noise."
              />
              {spell.loading && !spell.data ? (
                <CalmState tone="loading" title="Loading" />
              ) : spell.error ? (
                <CalmState title="Spell drift not available yet" body={spell.error} tone="error" />
              ) : (
                <SpellDrift rows={spell.data?.fatigue_drift ?? []} labels={labels} />
              )}
            </div>
          </section>

          <section aria-label="Repeatability" className="v-card p-4 sm:p-5">
            <SectionTitle eyebrow="Repeatability" title="How consistent is each measure across balls?" />
            {spell.loading && !spell.data ? (
              <CalmState tone="loading" title="Loading" />
            ) : spell.error ? (
              <CalmState title="Repeatability not available yet" body={spell.error} tone="error" />
            ) : (
              <RepeatabilityTable rows={spell.data?.repeatability_table ?? []} />
            )}
          </section>

          {/* C: Workload & wellness */}
          <section aria-labelledby="workload-heading" className="space-y-4">
            <SectionTitle
              eyebrow="Workload & wellness"
              title="Your load against the official limits"
              sub="Limits first. The load-change ratio is a secondary, contested indicator."
            />
            {workload.error && (
              <CalmState
                tone="error"
                title="Workload history couldn't load"
                body={`${workload.error}. You can still log a session below.`}
              />
            )}
            <DailyLoopCard
              summary={wl?.daily_loop_summary}
              last7={wl ? wl.deliveries_last_7d : null}
              last28={wl ? wl.deliveries_last_28d : null}
              ageBand={guidance?.age_band ?? bowlerInfo?.age_band ?? null}
              acwrLabel={wl?.acwr_summary?.zone_label_plain ?? null}
            />
            <div className="grid gap-4 lg:grid-cols-2 items-start">
              <div className="space-y-4 min-w-0">
                {workload.loading && !wl ? (
                  <CalmState tone="loading" title="Loading" />
                ) : (
                  <LimitsPanel checks={wl?.directive_checks ?? []} guidance={guidance} alerts={wl?.alerts ?? []} />
                )}
              </div>
              <div className="space-y-4 min-w-0">
                <QuickLogger bowlerId={resolved} onLogged={reloadWorkload} />
                <SorenessLog logs={logs} medicalDisclaimer={wl?.medical_disclaimer} />
              </div>
            </div>
            <div className="grid gap-4 lg:grid-cols-2">
              <WorkloadCalendar series={wl?.daily_series_28d} logs={logs} />
              {wl ? (
                <LoadChangeChart series={wl.daily_series_28d} summary={wl.acwr_summary} />
              ) : (
                <div className="v-card p-4">
                  <div className="v-label mb-2">Load change · acute : chronic (secondary)</div>
                  <p className="v-guardrail" style={{ color: "#171717" }}>
                    {ACWR_CAVEAT}
                  </p>
                  <div className="mt-3">
                    <CalmState tone={workload.loading ? "loading" : "empty"} title="No load series yet" />
                  </div>
                </div>
              )}
            </div>
            <LogHistory bowlerId={resolved} logs={logs} onChanged={reloadWorkload} />
          </section>
        </>
      )}

      <GuardrailStrip />
    </div>
  );
};

export default SpellWorkloadView;
