import React, { useMemo, useState } from "react";
import { BarChart3, GitCompareArrows, Upload, X } from "lucide-react";
import { navigate } from "../router";
import type { CaptureGrade, DeliveryCard } from "../types";
import { EmptyState, Spinner } from "../components/story/ui";
import { DeliveryCardTile, KIND_LABEL, useOverview } from "../components/home/shared";

type KindFilter = "all" | "upload" | "sample_real" | "sample_demo" | "fused_3d";
type GradeFilter = "all" | "measured" | CaptureGrade;
type SortKey = "newest" | "oldest" | "best_view" | "name";

const GRADE_RANK: Record<CaptureGrade, number> = { great: 0, partial: 1, limited: 2, unusable: 3 };

const Pill: React.FC<{ active: boolean; onClick: () => void; children: React.ReactNode }> = ({ active, onClick, children }) => (
  <button
    onClick={onClick}
    aria-pressed={active}
    className={`h-8 px-3 rounded-full text-[13px] whitespace-nowrap transition-colors ${
      active ? "bg-[#171717] text-white" : "bg-white text-[#4d4d4d] v-hairline hover:text-[#171717]"
    }`}
  >
    {children}
  </button>
);

/** Ball library: SwingVision/FullTrack-style filterable grid; multi-select → Compare (2) or Spell view. */
export const DeliveriesView: React.FC<{ bowler: string | null }> = ({ bowler }) => {
  const { data, error } = useOverview();
  const [kind, setKind] = useState<KindFilter>("all");
  const [grade, setGrade] = useState<GradeFilter>("all");
  const [view, setView] = useState<string>("all");
  const [sort, setSort] = useState<SortKey>("newest");
  const [query, setQuery] = useState("");
  const [selected, setSelected] = useState<string[]>([]);

  const all = useMemo(() => {
    const cards = data?.recent_deliveries ?? [];
    return bowler ? cards.filter((c) => c.bowler_id === bowler) : cards;
  }, [data, bowler]);

  const views = useMemo(() => Array.from(new Set(all.map((c) => c.view_label).filter(Boolean))), [all]);
  const kinds = useMemo(() => Array.from(new Set(all.map((c) => c.kind))), [all]);

  const cards = useMemo(() => {
    const q = query.trim().toLowerCase();
    let out = all.filter((c: DeliveryCard) => {
      if (kind !== "all" && c.kind !== kind) return false;
      if (grade === "measured" && !(c.capture_grade === "great" || c.capture_grade === "partial")) return false;
      if (grade !== "all" && grade !== "measured" && c.capture_grade !== grade) return false;
      if (view !== "all" && c.view_label !== view) return false;
      if (q && !`${c.label} ${c.headline} ${c.bowler_name}`.toLowerCase().includes(q)) return false;
      return true;
    });
    out = [...out].sort((a, b) => {
      switch (sort) {
        case "oldest":
          return a.created_at.localeCompare(b.created_at);
        case "best_view":
          return GRADE_RANK[a.capture_grade] - GRADE_RANK[b.capture_grade] || b.created_at.localeCompare(a.created_at);
        case "name":
          return a.label.localeCompare(b.label);
        default:
          return b.created_at.localeCompare(a.created_at);
      }
    });
    return out;
  }, [all, kind, grade, view, sort, query]);

  const toggle = (id: string) =>
    setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id].slice(-6)));

  if (error) return <EmptyState title="Couldn't load your deliveries" body={error} />;
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
        <Spinner /> Loading deliveries…
      </div>
    );
  }

  const bowlers = data.bowlers.filter((b) => b.delivery_count > 0);
  const selectedCards = selected.map((id) => all.find((c) => c.id === id)).filter(Boolean) as DeliveryCard[];
  const sameBowler = selectedCards.length > 0 && selectedCards.every((c) => c.bowler_id === selectedCards[0].bowler_id);

  return (
    <div className="space-y-5 pb-28 md:pb-20">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
        <div>
          <span className="v-label">Library</span>
          <h1 className="text-[28px] font-[500] tracking-[-0.03em] text-[#171717]">Deliveries</h1>
          <p className="text-[13.5px] text-[#666666] mt-1">
            Every ball you've analysed. Open one for its breakdown, or tick two to compare.
          </p>
        </div>
        <button
          onClick={() => navigate({ name: "upload", bowler })}
          className="v-btn-primary h-9 px-4 text-[13px] flex items-center gap-1.5 self-start"
        >
          <Upload className="w-3.5 h-3.5" /> New analysis
        </button>
      </div>

      {/* Filters */}
      <div className="v-panel p-3 space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search balls…"
            className="v-input h-8 px-3 text-[13px] w-full sm:w-[220px]"
            aria-label="Search deliveries"
          />
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="v-label">Sort</span>
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)} className="v-input h-8 px-2 text-[13px]" aria-label="Sort">
              <option value="newest">Newest first</option>
              <option value="oldest">Oldest first</option>
              <option value="best_view">Clearest view first</option>
              <option value="name">Name</option>
            </select>
          </div>
        </div>
        <div className="flex gap-1.5 overflow-x-auto -mx-1 px-1 pb-0.5" role="group" aria-label="Filter">
          {bowlers.length > 1 && (
            <>
              <Pill active={!bowler} onClick={() => navigate({ name: "deliveries", bowler: null })}>
                Everyone
              </Pill>
              {bowlers.map((b) => (
                <Pill key={b.id} active={bowler === b.id} onClick={() => navigate({ name: "deliveries", bowler: b.id })}>
                  {b.name} <span className="v-num text-[11px] opacity-70 ml-0.5">{b.delivery_count}</span>
                </Pill>
              ))}
              <span className="w-px bg-[#ebebeb] mx-1 shrink-0" />
            </>
          )}
          <Pill active={grade === "all"} onClick={() => setGrade("all")}>
            Any view
          </Pill>
          <Pill active={grade === "measured"} onClick={() => setGrade("measured")}>
            Well measured
          </Pill>
          <Pill active={grade === "great"} onClick={() => setGrade("great")}>
            Great view only
          </Pill>
          <span className="w-px bg-[#ebebeb] mx-1 shrink-0" />
          {views.length > 1 && (
            <>
              <Pill active={view === "all"} onClick={() => setView("all")}>
                All angles
              </Pill>
              {views.map((v) => (
                <Pill key={v} active={view === v} onClick={() => setView(v)}>
                  {v}
                </Pill>
              ))}
              <span className="w-px bg-[#ebebeb] mx-1 shrink-0" />
            </>
          )}
          {kinds.length > 1 && (
            <>
              <Pill active={kind === "all"} onClick={() => setKind("all")}>
                All sources
              </Pill>
              {kinds.map((k) => (
                <Pill key={k} active={kind === k} onClick={() => setKind(k as KindFilter)}>
                  {KIND_LABEL[k]}
                </Pill>
              ))}
            </>
          )}
        </div>
      </div>

      {all.length === 0 ? (
        <EmptyState
          title="No deliveries yet"
          body="Film one ball from the side or from behind, then upload it here."
          action={
            <button onClick={() => navigate({ name: "upload", bowler })} className="v-btn-primary h-9 px-4 text-[13px]">
              Analyse my bowling
            </button>
          }
        />
      ) : cards.length === 0 ? (
        <EmptyState title="No balls match these filters" body="Clear a filter to see more." />
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {cards.map((c) => (
            <DeliveryCardTile key={c.id} card={c} showBowler={!bowler} selected={selected.includes(c.id)} onSelect={toggle} />
          ))}
        </div>
      )}

      {/* Selection tray */}
      {selected.length > 0 && (
        <div className="fixed left-1/2 -translate-x-1/2 bottom-[76px] md:bottom-6 z-30 v-panel px-3 py-2 flex items-center gap-2 shadow-[0_0_0_1px_rgba(0,0,0,0.12)]">
          <span className="text-[13px] text-[#171717] font-[500] v-num">{selected.length} selected</span>
          <button
            disabled={selected.length !== 2}
            onClick={() => navigate({ name: "compare", a: selected[0], b: selected[1] })}
            className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5"
            title={selected.length !== 2 ? "Pick exactly two balls to compare" : undefined}
          >
            <GitCompareArrows className="w-3.5 h-3.5" /> Compare
          </button>
          <button
            disabled={!sameBowler}
            onClick={() => navigate({ name: "spell", bowler: selectedCards[0]?.bowler_id ?? null, session: selectedCards[0]?.session_id ?? null })}
            className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5"
          >
            <BarChart3 className="w-3.5 h-3.5" /> Spell view
          </button>
          <button onClick={() => setSelected([])} className="w-8 h-8 flex items-center justify-center text-[#8f8f8f] hover:text-[#171717]" aria-label="Clear selection">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
};

export default DeliveriesView;
