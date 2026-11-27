import React from "react";
import { HelpCircle, Plus } from "lucide-react";
import { navigate, type Route } from "../router";

/** Calm 3-item nav + one primary action. */
export const TopNav: React.FC<{ route: Route; onOpenFilmGuide: () => void }> = ({ route, onOpenFilmGuide }) => {
  const section =
    route.name === "home"
      ? "home"
      : route.name === "deliveries" || route.name === "delivery"
      ? "deliveries"
      : route.name === "progress" || route.name === "compare"
      ? "progress"
      : null;
  const items: Array<{ key: "home" | "deliveries" | "progress"; label: string; to: Route }> = [
    { key: "home", label: "Home", to: { name: "home" } },
    { key: "deliveries", label: "Deliveries", to: { name: "deliveries", bowler: null } },
    { key: "progress", label: "Progress", to: { name: "progress", bowler: null } },
  ];
  return (
    <header className="no-print sticky top-0 z-30 bg-[#fafafa]/90 backdrop-blur border-b border-[#ebebeb]">
      <div className="max-w-[1320px] mx-auto px-4 md:px-6 h-14 flex items-center gap-3 md:gap-6">
        <button onClick={() => navigate({ name: "home" })} className="flex items-center gap-2 shrink-0" aria-label="SportsPostureAnalysis home">
          <span className="w-6 h-6 rounded-[6px] bg-[#171717] flex items-center justify-center">
            <span className="w-2 h-2 rounded-full bg-[#297a3a]" />
          </span>
          <span className="hidden sm:inline text-[14px] font-[500] tracking-[-0.01em] text-[#171717]">SportsPostureAnalysis</span>
        </button>
        <nav className="flex items-center gap-0.5" aria-label="Main">
          {items.map((it) => {
            const active = section === it.key;
            return (
              <button
                key={it.key}
                onClick={() => navigate(it.to)}
                className={`h-8 px-2.5 md:px-3 rounded-[6px] text-[13.5px] transition-colors ${
                  active ? "bg-[#ffffff] text-[#171717] font-[500] border border-[#ebebeb]" : "text-[#666666] hover:text-[#171717] border border-transparent"
                }`}
                aria-current={active ? "page" : undefined}
              >
                {it.label}
              </button>
            );
          })}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <button onClick={onOpenFilmGuide} className="hidden md:flex h-8 px-2.5 items-center gap-1.5 text-[13px] text-[#666666] hover:text-[#171717]">
            <HelpCircle className="w-3.5 h-3.5" /> How to film
          </button>
          <button onClick={() => navigate({ name: "new", bowler: null })} className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Plus className="w-3.5 h-3.5" />
            <span className="hidden sm:inline">New analysis</span>
            <span className="sm:hidden">New</span>
          </button>
        </div>
      </div>
    </header>
  );
};
