import React, { useEffect, useMemo, useState } from "react";
import { ArrowRight, Info } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type { CheckpointId, CheckpointStatus, OverviewResponse, ProgressResponse } from "../types";
import {
  CHECKPOINT_ORDER,
  CHECKPOINT_TITLES,
  Card,
  EmptyState,
  SectionTitle,
  Spinner,
  STATUS_LABELS,
  StatusDot,
  ValueWithRange,
  formatDate,
} from "../components/story/ui";

type Series = ProgressResponse["series"][number];
type Point = Series["points"][number];
type DeliveryCol = ProgressResponse["deliveries"][number];

function shortDate(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return d.toLocaleDateString(undefined, { day: "numeric", month: "short" });
}

function todayIso(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`;
}

function describePoint(title: string, col: DeliveryCol | undefined, p: Point | undefined): string {
  const when = col ? `${col.label ? `${col.label} · ` : ""}${formatDate(col.created_at)}` : "";
  if (!p) return `${title} · ${when} · Not measured`;
  const val = p.value_text ? ` · ${p.value_text}${p.range_text ? ` (likely ${p.range_text})` : ""}` : "";
  return `${title} · ${when} · ${STATUS_LABELS[p.status]}${val}`;
}

const isMeasured = (p: Point | undefined): p is Point & { value: number } =>
  !!p && p.status !== "not_visible" && typeof p.value === "number" && Number.isFinite(p.value);

/** Small value chart: measured points with range whiskers only — no connecting/trend lines. */
const MiniChart: React.FC<{ series: Series; cols: DeliveryCol[] }> = ({ series, cols }) => {
  const byId = new Map(series.points.map((p) => [p.delivery_id, p]));
  const measured = cols
    .map((c, i) => ({ c, i, p: byId.get(c.id) }))
    .filter((x): x is { c: DeliveryCol; i: number; p: Point & { value: number } } => isMeasured(x.p));
  const latest = measured[measured.length - 1];
  const W = 280;
  const H = 96;
  const padX = 14;
  const padY = 10;

  let body: React.ReactNode;
  if (measured.length === 0) {
    body = (
      <div className="h-[96px] flex items-center justify-center text-[12px] text-[#8f8f8f] text-center px-4">
        Not measured in any of these clips yet.
      </div>
    );
  } else {
    const lows = measured.map(({ p }) => (typeof p.ci_low === "number" ? p.ci_low : p.value));
    const highs = measured.map(({ p }) => (typeof p.ci_high === "number" ? p.ci_high : p.value));
    let lo = Math.min(...lows);
    let hi = Math.max(...highs);
    if (hi - lo < 1e-6) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;
    const n = Math.max(cols.length, 1);
    const x = (i: number) => (n === 1 ? W / 2 : padX + (i * (W - 2 * padX)) / (n - 1));
    const y = (v: number) => padY + ((hi - v) / (hi - lo)) * (H - 2 * padY);
    body = (
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-[96px]" preserveAspectRatio="none" role="img" aria-label={`${series.title} measurements over time`}>
        <line x1={0} y1={H - 0.5} x2={W} y2={H - 0.5} stroke="#ebebeb" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        {cols.map((c, i) => (
          <line key={c.id} x1={x(i)} y1={padY} x2={x(i)} y2={H - padY} stroke="#f4f4f5" strokeWidth={1} vectorEffect="non-scaling-stroke" />
        ))}
        {measured.map(({ c, i, p }) => {
          const cx = x(i);
          const yl = y(typeof p.ci_low === "number" ? p.ci_low : p.value);
          const yh = y(typeof p.ci_high === "number" ? p.ci_high : p.value);
          const ink = p.status === "strong" ? "#297a3a" : "#171717";
          return (
            <g key={c.id} className="cursor-pointer" onClick={() => navigate({ name: "delivery", id: c.id, tab: "story" })}>
              <title>{describePoint(series.title, c, p)}</title>
              <line x1={cx} y1={yh} x2={cx} y2={yl} stroke="#a8a8a8" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
              <line x1={cx - 4} y1={yh} x2={cx + 4} y2={yh} stroke="#a8a8a8" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
              <line x1={cx - 4} y1={yl} x2={cx + 4} y2={yl} stroke="#a8a8a8" strokeWidth={1.2} vectorEffect="non-scaling-stroke" />
              <circle cx={cx} cy={y(p.value)} r={3.6} fill={p.status === "on_track" || p.status === "watch" ? "#ffffff" : ink} stroke={ink} strokeWidth={1.4} vectorEffect="non-scaling-stroke" />
              <rect x={cx - 10} y={0} width={20} height={H} fill="transparent" />
            </g>
          );
        })}
      </svg>
    );
  }

  return (
    <Card className="p-4 flex flex-col gap-2 min-w-0">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[14px] font-[500] text-[#171717]">{series.title || CHECKPOINT_TITLES[series.checkpoint_id]}</div>
          <div className="text-[12px] text-[#8f8f8f]">
            {measured.length} of {cols.length} balls measured
          </div>
        </div>
        {latest && (
          <div className="text-right shrink-0">
            <div className="text-[11px] text-[#8f8f8f]">Latest</div>
            <ValueWithRange value={latest.p.value_text} range={latest.p.range_text} size="sm" />
          </div>
        )}
      </div>
      {body}
      <div className="flex items-center gap-3 text-[11px] text-[#8f8f8f]">
        <span className="inline-flex items-center gap-1">
          <svg width="10" height="12" aria-hidden>
            <line x1="5" y1="1" x2="5" y2="11" stroke="#a8a8a8" />
            <line x1="2" y1="1" x2="8" y2="1" stroke="#a8a8a8" />
            <line x1="2" y1="11" x2="8" y2="11" stroke="#a8a8a8" />
          </svg>
          likely range
        </span>
        <span>Balls we couldn't see are left out.</span>
      </div>
    </Card>
  );
};

const WorkloadCard: React.FC<{ bowlerId: string; progress: ProgressResponse; onLogged: () => void }> = ({ bowlerId, progress, onLogged }) => {
  const w = progress.workload;
  const [date, setDate] = useState(todayIso());
  const [count, setCount] = useState("");
  const [kind, setKind] = useState("nets");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<{ ok: boolean; text: string } | null>(null);
  const ageBand = w?.age_band || ("age_band" in progress.bowler ? progress.bowler.age_band : "");

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    const n = Number(count);
    if (!Number.isFinite(n) || n <= 0 || !Number.isInteger(n)) {
      setMsg({ ok: false, text: "Enter how many balls were bowled as a whole number." });
      return;
    }
    setBusy(true);
    setMsg(null);
    try {
      await api.addWorkload(bowlerId, { log_date: date, deliveries_count: n, session_type: kind, notes });
      setCount("");
      setNotes("");
      setMsg({ ok: true, text: "Logged. Guidance below is updated." });
      onLogged();
    } catch (err: any) {
      setMsg({ ok: false, text: `Couldn't save that just now. ${err?.message ?? ""}`.trim() });
    } finally {
      setBusy(false);
    }
  };

  const limits: Array<[string, number | null]> = w
    ? [
        ["Per spell", w.max_balls_per_spell],
        ["Per day", w.max_balls_per_day],
        ["Per week", w.max_balls_per_week],
      ]
    : [];

  return (
    <Card className="p-4 md:p-5 flex flex-col gap-4">
      <SectionTitle hint={ageBand ? `Age group: ${ageBand}` : undefined}>Bowling workload</SectionTitle>
      {w ? (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
            {limits.map(([label, v]) => (
              <div key={label} className="v-card-subtle p-3">
                <div className="text-[11.5px] text-[#8f8f8f]">{label}</div>
                <div className="font-mono font-[500] v-tabular text-[17px] text-[#171717]">{v != null ? `${v} balls` : "—"}</div>
                <div className="text-[11px] text-[#8f8f8f]">suggested limit</div>
              </div>
            ))}
            <div className="v-card-subtle p-3">
              <div className="text-[11.5px] text-[#8f8f8f]">This week</div>
              <div className="font-mono font-[500] v-tabular text-[17px] text-[#171717]">
                {w.current_bowls_per_week != null ? `${w.current_bowls_per_week} balls` : "—"}
              </div>
              <div className="text-[11px] text-[#8f8f8f]">logged so far</div>
            </div>
          </div>
          {w.status_note && <p className="text-[13px] leading-relaxed text-[#171717]">{w.status_note}</p>}
          {w.governing_body_source && <p className="text-[11.5px] text-[#8f8f8f]">Source: {w.governing_body_source}</p>}
        </>
      ) : (
        <p className="text-[13px] text-[#4d4d4d]">
          Workload guidance (from the England &amp; Wales Cricket Board and Cricket Australia) appears here once the bowler's age group is set and some balls are logged.
        </p>
      )}
      <p className="flex gap-2 items-start text-[12px] leading-[1.5] text-[#4d4d4d]">
        <Info className="w-3.5 h-3.5 mt-[2px] text-[#666666] shrink-0" />
        <span>These are general guidelines from governing bodies, not medical advice. Growth, fitness and recent load matter too.</span>
      </p>

      <form onSubmit={submit} className="border-t border-[#ebebeb] pt-4 flex flex-col gap-3">
        <div className="text-[13px] font-[500] text-[#171717]">Log today's balls</div>
        <div className="grid grid-cols-2 sm:grid-cols-[1fr_1fr_1fr] gap-2">
          <label className="flex flex-col gap-1 text-[12px] text-[#4d4d4d]">
            Date
            <input type="date" className="v-input h-9 px-2 text-[13px]" value={date} max={todayIso()} onChange={(e) => setDate(e.target.value)} required />
          </label>
          <label className="flex flex-col gap-1 text-[12px] text-[#4d4d4d]">
            Balls bowled
            <input
              type="number"
              inputMode="numeric"
              min={1}
              step={1}
              className="v-input h-9 px-2 text-[13px] font-mono"
              placeholder="e.g. 36"
              value={count}
              onChange={(e) => setCount(e.target.value)}
              required
            />
          </label>
          <label className="flex flex-col gap-1 text-[12px] text-[#4d4d4d] col-span-2 sm:col-span-1">
            Type
            <select className="v-input h-9 px-2 text-[13px]" value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="nets">Nets</option>
              <option value="match">Match</option>
              <option value="training">Skills training</option>
            </select>
          </label>
        </div>
        <input className="v-input h-9 px-2 text-[13px]" placeholder="Notes (optional)" value={notes} onChange={(e) => setNotes(e.target.value)} />
        <div className="flex flex-wrap items-center gap-3">
          <button type="submit" className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-2" disabled={busy}>
            {busy && <Spinner className="!border-t-white !border-white/30" />}
            Save
          </button>
          {msg && <span className={`text-[12.5px] ${msg.ok ? "text-[#297a3a]" : "text-[#4d4d4d]"}`}>{msg.text}</span>}
        </div>
      </form>
    </Card>
  );
};

export const ProgressView: React.FC<{ bowlerId: string | null }> = ({ bowlerId }) => {
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [overviewErr, setOverviewErr] = useState<string | null>(null);
  const [progress, setProgress] = useState<ProgressResponse | null>(null);
  const [progressErr, setProgressErr] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);
  const [hover, setHover] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .overview()
      .then((o) => !cancelled && setOverview(o))
      .catch((e) => !cancelled && setOverviewErr(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, []);

  const bowlers = useMemo(() => [...(overview?.bowlers ?? [])].sort((x, y) => y.delivery_count - x.delivery_count), [overview]);

  // Default to the bowler with the most deliveries.
  useEffect(() => {
    if (!bowlerId && bowlers.length > 0) navigate({ name: "progress", bowler: bowlers[0].id });
  }, [bowlerId, bowlers]);

  // Clear the previous bowler's data on switch (but not on a same-bowler refresh after logging).
  useEffect(() => {
    setProgress(null);
    setHover(null);
  }, [bowlerId]);

  useEffect(() => {
    if (!bowlerId) return;
    let cancelled = false;
    setLoading(true);
    setProgressErr(null);
    api
      .progress(bowlerId)
      .then((p) => !cancelled && setProgress(p))
      .catch((e) => {
        if (cancelled) return;
        setProgress(null);
        setProgressErr(String(e?.message || e));
      })
      .finally(() => !cancelled && setLoading(false));
    return () => {
      cancelled = true;
    };
  }, [bowlerId, reloadKey]);

  const cols = useMemo(
    () => [...(progress?.deliveries ?? [])].sort((x, y) => (x.created_at || "").localeCompare(y.created_at || "")),
    [progress]
  );
  const seriesById = useMemo(() => {
    const m = new Map<CheckpointId, Series>();
    (progress?.series ?? []).forEach((s) => m.set(s.checkpoint_id, s));
    return m;
  }, [progress]);
  const rows: Series[] = CHECKPOINT_ORDER.map(
    (id) => seriesById.get(id) ?? { checkpoint_id: id, title: CHECKPOINT_TITLES[id], points: [] }
  );

  const bowlerName =
    progress?.bowler?.name ?? bowlers.find((b) => b.id === bowlerId)?.name ?? "";
  const bowlerMeta = progress?.bowler
    ? [progress.bowler.bowling_arm === "left" ? "Left-arm" : "Right-arm", progress.bowler.age_band].filter(Boolean).join(" · ")
    : "";

  const compareLast2 = () => {
    if (cols.length < 2) return;
    navigate({ name: "compare", a: cols[cols.length - 2].id, b: cols[cols.length - 1].id });
  };

  const current = progress;

  return (
    <div className="max-w-[1200px] mx-auto px-4 md:px-6 py-6 md:py-8 flex flex-col gap-5">
      <header className="flex flex-col gap-3">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="text-[24px] md:text-[28px] font-[500] tracking-[-0.03em] text-[#171717]">Progress</h1>
            {bowlerName && (
              <p className="text-[13px] text-[#4d4d4d]">
                {bowlerName}
                {bowlerMeta && <span className="text-[#8f8f8f]"> · {bowlerMeta}</span>}
              </p>
            )}
          </div>
          <button className="v-btn-primary h-9 px-4 text-[13px] inline-flex items-center gap-1.5" disabled={cols.length < 2} onClick={compareLast2}>
            Compare last 2 balls <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
        {bowlers.length > 0 && (
          <div className="flex gap-1.5 overflow-x-auto pb-1 -mx-1 px-1" role="tablist" aria-label="Bowler">
            {bowlers.map((b) => {
              const active = b.id === bowlerId;
              return (
                <button
                  key={b.id}
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => navigate({ name: "progress", bowler: b.id })}
                  className={`h-8 px-3 rounded-full text-[12.5px] whitespace-nowrap inline-flex items-center gap-1.5 transition-colors duration-150 ${
                    active ? "bg-[#171717] text-white" : "bg-white text-[#171717] shadow-[0_0_0_1px_#ebebeb] hover:shadow-[0_0_0_1px_#c9c9c9]"
                  }`}
                >
                  {b.name}
                  <span className={`font-mono text-[11px] ${active ? "text-white/70" : "text-[#8f8f8f]"}`}>{b.delivery_count}</span>
                </button>
              );
            })}
          </div>
        )}
      </header>

      {overviewErr && !bowlerId && (
        <EmptyState
          title="Couldn't load your bowlers"
          body={overviewErr}
          action={<button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => window.location.reload()}>Try again</button>}
        />
      )}
      {overview && bowlers.length === 0 && !bowlerId && (
        <EmptyState
          title="No bowlers yet"
          body="Analyse a delivery and the bowler's progress will build up here, ball by ball."
          action={<button className="v-btn-primary h-9 px-4 text-[13px]" onClick={() => navigate({ name: "new", bowler: null })}>New analysis</button>}
        />
      )}
      {(loading && !progress) || (!overview && !overviewErr && !bowlerId) ? (
        <div className="flex items-center gap-2 text-[13px] text-[#4d4d4d]">
          <Spinner /> Loading progress…
        </div>
      ) : null}
      {progressErr && !loading && (
        <EmptyState
          title="Progress isn't available right now"
          body={<span className="break-words">{progressErr}</span>}
          action={<button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => setReloadKey((k) => k + 1)}>Try again</button>}
        />
      )}

      {current && bowlerId && !progressErr && (
        <>
          {cols.length < 2 ? (
            <EmptyState
              title={cols.length === 0 ? "No balls analysed yet" : "One ball so far"}
              body="Progress needs at least two analysed balls. Film another delivery from the same angle and the checkpoints will line up here over time."
              action={
                <>
                  <button className="v-btn-primary h-9 px-4 text-[13px]" onClick={() => navigate({ name: "new", bowler: bowlerId })}>
                    Film another ball
                  </button>
                  {cols.length === 1 && (
                    <button className="v-btn-ghost h-9 px-4 text-[13px]" onClick={() => navigate({ name: "delivery", id: cols[0].id, tab: "story" })}>
                      Open that ball
                    </button>
                  )}
                </>
              }
            />
          ) : (
            <>
              {/* Dot matrix */}
              <Card className="p-4 md:p-5">
                <SectionTitle hint="Tap a dot to open that ball">Checkpoints ball by ball</SectionTitle>
                <div className="overflow-x-auto -mx-1 px-1">
                  <table className="border-separate border-spacing-0 text-left">
                    <thead>
                      <tr>
                        <th className="sticky left-0 z-10 bg-white min-w-[132px] md:min-w-[168px]" />
                        {cols.map((c) => (
                          <th key={c.id} className="px-1.5 pb-2 align-bottom font-[400]">
                            <button
                              type="button"
                              className="flex flex-col items-center gap-1 w-[64px] group"
                              onClick={() => navigate({ name: "delivery", id: c.id, tab: "story" })}
                              title={`${c.label || "Delivery"} · ${formatDate(c.created_at)}`}
                            >
                              <span className="w-[48px] h-[48px] rounded-[6px] bg-[#f4f4f5] overflow-hidden shadow-[0_0_0_1px_#ebebeb] group-hover:shadow-[0_0_0_1px_#171717]">
                                {c.thumbnail_url && <img src={c.thumbnail_url} alt="" className="w-full h-full object-cover" loading="lazy" />}
                              </span>
                              <span className="text-[11px] text-[#4d4d4d] whitespace-nowrap">{shortDate(c.created_at)}</span>
                              <span className="block max-w-[64px] truncate text-[11px] text-[#8f8f8f]" title={c.label || "Delivery"}>
                                {c.label || "Delivery"}
                              </span>
                            </button>
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((s) => {
                        const byId = new Map(s.points.map((p) => [p.delivery_id, p]));
                        return (
                          <tr key={s.checkpoint_id}>
                            <th scope="row" className="sticky left-0 z-10 bg-white pr-3 py-2 text-[13px] font-[400] text-[#171717] whitespace-nowrap border-t border-[#f4f4f5]">
                              {s.title || CHECKPOINT_TITLES[s.checkpoint_id]}
                            </th>
                            {cols.map((c) => {
                              const p = byId.get(c.id);
                              const status: CheckpointStatus = p?.status ?? "not_visible";
                              const desc = describePoint(s.title || CHECKPOINT_TITLES[s.checkpoint_id], c, p);
                              return (
                                <td key={c.id} className="px-1.5 py-1 text-center border-t border-[#f4f4f5]">
                                  <button
                                    type="button"
                                    className="w-[64px] h-9 inline-flex items-center justify-center rounded-[6px] hover:bg-[#f4f4f5] focus:bg-[#f4f4f5] outline-none focus-visible:shadow-[0_0_0_1px_#171717]"
                                    title={desc}
                                    aria-label={desc}
                                    onMouseEnter={() => setHover(desc)}
                                    onFocus={() => setHover(desc)}
                                    onMouseLeave={() => setHover(null)}
                                    onBlur={() => setHover(null)}
                                    onClick={() => navigate({ name: "delivery", id: c.id, tab: "story" })}
                                  >
                                    <StatusDot status={status} size={14} />
                                  </button>
                                </td>
                              );
                            })}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="mt-3 min-h-[20px] text-[12.5px] text-[#4d4d4d] font-mono v-tabular break-words" aria-live="polite">
                  {hover ?? <span className="font-sans text-[#8f8f8f]">Hover or focus a dot to see its value and likely range.</span>}
                </div>
                <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1.5 text-[11.5px] text-[#4d4d4d]">
                  {(["strong", "on_track", "watch", "focus", "not_visible"] as CheckpointStatus[]).map((st) => (
                    <span key={st} className="inline-flex items-center gap-1.5">
                      <StatusDot status={st} size={10} />
                      {STATUS_LABELS[st]}
                    </span>
                  ))}
                </div>
              </Card>

              {/* Per-checkpoint value charts */}
              <div>
                <SectionTitle hint="Each dot is one ball; the bar shows its likely range">Measured values</SectionTitle>
                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                  {rows.map((s) => (
                    <MiniChart key={s.checkpoint_id} series={s} cols={cols} />
                  ))}
                </div>
              </div>
            </>
          )}

          {progress?.summary_plain && (
            <Card className="p-4 md:p-5">
              <SectionTitle>In short</SectionTitle>
              <p className="text-[15px] leading-relaxed text-[#171717]">{progress.summary_plain}</p>
            </Card>
          )}

          {progress && <WorkloadCard bowlerId={bowlerId} progress={progress} onLogged={() => setReloadKey((k) => k + 1)} />}
        </>
      )}
    </div>
  );
};

export default ProgressView;
