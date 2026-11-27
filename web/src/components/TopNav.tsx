import React, { useEffect, useState } from "react";
import { BarChart3, ChevronDown, Dumbbell, GitCompareArrows, HelpCircle, Home, Library, Plus, Video } from "lucide-react";
import { api } from "../api";
import type { BowlerSummary } from "../types";
import { navigate, setCurrentBowlerId, useCurrentBowlerId, type Route, type RouteInput } from "../router";

type SectionKey = "home" | "deliveries" | "compare" | "spell" | "practice";

const ITEMS: Array<{ key: SectionKey; label: string; short: string; icon: React.FC<{ className?: string }> }> = [
  { key: "home", label: "Home", short: "Home", icon: Home },
  { key: "deliveries", label: "Deliveries", short: "Balls", icon: Library },
  { key: "compare", label: "Compare", short: "Compare", icon: GitCompareArrows },
  { key: "spell", label: "Spell & Workload", short: "Spell", icon: BarChart3 },
  { key: "practice", label: "Practice", short: "Practice", icon: Dumbbell },
];

function sectionOf(route: Route): SectionKey | null {
  switch (route.name) {
    case "home":
      return "home";
    case "deliveries":
    case "delivery":
    case "card":
      return "deliveries";
    case "compare":
      return "compare";
    case "spell":
      return "spell";
    case "practice":
      return "practice";
    default:
      return null;
  }
}

function targetFor(key: SectionKey, bowler: string | null): RouteInput {
  switch (key) {
    case "home":
      return { name: "home" };
    case "deliveries":
      return { name: "deliveries", bowler };
    case "compare":
      return { name: "compare", a: null, b: null };
    case "spell":
      return { name: "spell", bowler };
    case "practice":
      return { name: "practice", bowler };
  }
}

/** Bowler switcher: scopes Home / Deliveries / Spell / Practice. */
const BowlerSwitcher: React.FC = () => {
  const current = useCurrentBowlerId();
  const [bowlers, setBowlers] = useState<BowlerSummary[]>([]);
  const [open, setOpen] = useState(false);

  useEffect(() => {
    let alive = true;
    api
      .listBowlers()
      .then((r) => {
        if (!alive) return;
        setBowlers(r.bowlers || []);
        if (!current && r.bowlers?.[0]) setCurrentBowlerId(r.bowlers[0].id);
      })
      .catch(() => undefined);
    return () => {
      alive = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const active = bowlers.find((b) => b.id === current) ?? bowlers[0];
  if (!active) return null;

  return (
    <div className="relative hidden md:block">
      <button
        onClick={() => setOpen((o) => !o)}
        className="h-8 pl-2 pr-1.5 rounded-[6px] flex items-center gap-1.5 text-[13px] text-[#171717] hover:bg-white v-hairline"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="w-5 h-5 rounded-full bg-[#f4f4f5] text-[10px] font-[600] flex items-center justify-center">
          {active.name.slice(0, 1).toUpperCase()}
        </span>
        <span className="max-w-[140px] truncate">{active.name}</span>
        <span className="v-label !text-[10px]">{active.bowling_arm === "left" ? "LA" : "RA"}</span>
        <ChevronDown className="w-3.5 h-3.5 text-[#8f8f8f]" />
      </button>
      {open && (
        <div
          role="listbox"
          className="absolute left-0 top-10 z-40 min-w-[220px] v-panel p-1"
          onMouseLeave={() => setOpen(false)}
        >
          {bowlers.map((b) => (
            <button
              key={b.id}
              role="option"
              aria-selected={b.id === active.id}
              onClick={() => {
                setCurrentBowlerId(b.id);
                setOpen(false);
              }}
              className={`w-full text-left h-9 px-2 rounded-[5px] text-[13px] flex items-center justify-between hover:bg-[#f4f4f5] ${
                b.id === active.id ? "font-[500]" : "text-[#4d4d4d]"
              }`}
            >
              <span className="truncate">{b.name}</span>
              <span className="v-label !text-[10px]">
                {b.bowling_arm === "left" ? "Left-arm" : "Right-arm"} · {b.age_band}
              </span>
            </button>
          ))}
          <div className="h-px bg-[#ebebeb] my-1" />
          <button
            onClick={() => {
              setOpen(false);
              navigate({ name: "upload", bowler: null });
            }}
            className="w-full text-left h-9 px-2 rounded-[5px] text-[13px] text-[#4d4d4d] hover:bg-[#f4f4f5] flex items-center gap-1.5"
          >
            <Plus className="w-3.5 h-3.5" /> Add a bowler
          </button>
        </div>
      )}
    </div>
  );
};

/** Studio v5 navigation: 5 nouns + Record + New analysis (desktop), bottom tab bar + floating + (mobile). */
export const TopNav: React.FC<{ route: Route; onOpenFilmGuide: () => void }> = ({ route, onOpenFilmGuide }) => {
  const section = sectionOf(route);
  const bowler = useCurrentBowlerId();

  return (
    <>
      <header className="no-print sticky top-0 z-30 bg-[#fafafa]/90 backdrop-blur border-b border-[#ebebeb]">
        <div className="max-w-[1600px] mx-auto px-4 md:px-6 h-14 flex items-center gap-3 md:gap-4">
          <button
            onClick={() => navigate({ name: "home" })}
            className="flex items-center gap-2 shrink-0"
            aria-label="SportsPostureAnalysis home"
          >
            <span className="w-6 h-6 rounded-[6px] bg-[#171717] flex items-center justify-center">
              <span className="w-2 h-2 rounded-full bg-[#297a3a]" />
            </span>
            <span className="hidden lg:inline text-[14px] font-[500] tracking-[-0.01em] text-[#171717]">
              SportsPostureAnalysis
            </span>
          </button>
          <BowlerSwitcher />
          <nav className="hidden md:flex items-center gap-0.5" aria-label="Main">
            {ITEMS.map((it) => {
              const active = section === it.key;
              return (
                <button
                  key={it.key}
                  onClick={() => navigate(targetFor(it.key, bowler))}
                  className={`h-8 px-2.5 lg:px-3 rounded-[6px] text-[13.5px] transition-colors whitespace-nowrap ${
                    active
                      ? "bg-[#ffffff] text-[#171717] font-[500] border border-[#ebebeb]"
                      : "text-[#666666] hover:text-[#171717] border border-transparent"
                  }`}
                  aria-current={active ? "page" : undefined}
                >
                  {it.label}
                </button>
              );
            })}
          </nav>
          <div className="ml-auto flex items-center gap-2">
            <button
              onClick={onOpenFilmGuide}
              className="hidden lg:flex h-8 px-2.5 items-center gap-1.5 text-[13px] text-[#666666] hover:text-[#171717]"
            >
              <HelpCircle className="w-3.5 h-3.5" /> How to film
            </button>
            <button
              onClick={() => navigate({ name: "upload", bowler, mode: "live" })}
              className="v-btn-ghost h-8 px-2.5 text-[13px] flex items-center gap-1.5"
              aria-label="Record with live camera"
            >
              <span className="w-2 h-2 rounded-full bg-[#dc2626]" />
              <span className="hidden sm:inline">Record</span>
            </button>
            <button
              onClick={() => navigate({ name: "upload", bowler })}
              className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5"
            >
              <Plus className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">New analysis</span>
              <span className="sm:hidden">New</span>
            </button>
          </div>
        </div>
      </header>

      {/* Mobile bottom tab bar */}
      <nav
        className="no-print md:hidden fixed bottom-0 inset-x-0 z-30 bg-white/95 backdrop-blur border-t border-[#ebebeb] pb-[env(safe-area-inset-bottom)]"
        aria-label="Main (mobile)"
      >
        <div className="grid grid-cols-5 h-14">
          {ITEMS.map((it) => {
            const active = section === it.key;
            const Icon = it.icon;
            return (
              <button
                key={it.key}
                onClick={() => navigate(targetFor(it.key, bowler))}
                className={`flex flex-col items-center justify-center gap-0.5 text-[10.5px] ${
                  active ? "text-[#171717] font-[500]" : "text-[#8f8f8f]"
                }`}
                aria-current={active ? "page" : undefined}
              >
                <Icon className="w-[18px] h-[18px]" />
                {it.short}
              </button>
            );
          })}
        </div>
      </nav>
      <button
        onClick={() => navigate({ name: "upload", bowler, mode: "live" })}
        className="no-print md:hidden fixed right-4 bottom-[72px] z-30 w-12 h-12 rounded-full bg-[#171717] text-white flex items-center justify-center shadow-[0_0_0_1px_#171717]"
        aria-label="New analysis"
      >
        <Video className="w-5 h-5" />
      </button>
    </>
  );
};
