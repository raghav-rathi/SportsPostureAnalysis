import React from "react";
import { ChevronRight } from "lucide-react";
import type { KeyMoment, MomentId } from "../../types";

/** Story-style chapter chips: Back foot lands → Front foot plants → Arm level → Ball release. */
export const MomentsScrubber: React.FC<{
  moments: KeyMoment[];
  activeId: MomentId | null;
  thumbFor: (m: KeyMoment) => string | null;
  onSelect: (m: KeyMoment) => void;
}> = ({ moments, activeId, thumbFor, onSelect }) => {
  if (moments.length === 0) {
    return (
      <div className="v-card px-4 py-3 text-[13px] text-[#666666]">
        We couldn't find the key moments of a full delivery in this clip, so there's nothing to step through yet.
      </div>
    );
  }
  return (
    <div>
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-[14px] font-[500] text-[#171717]">Key moments</h2>
        <span className="text-[12px] text-[#8f8f8f]">Tap to jump there</span>
      </div>
      <ol className="grid grid-cols-2 sm:grid-cols-4 gap-2">
        {moments.map((m, i) => {
          const active = m.id === activeId;
          const thumb = thumbFor(m);
          return (
            <li key={m.id} className="relative">
              <button
                onClick={() => onSelect(m)}
                className={`w-full text-left rounded-[8px] border bg-[#ffffff] overflow-hidden transition-colors ${
                  active ? "border-[#171717]" : "border-[#ebebeb] hover:border-[#c9c9c9]"
                }`}
                aria-current={active ? "step" : undefined}
              >
                <div className="aspect-[16/10] bg-[#f4f4f5] overflow-hidden">
                  {thumb ? (
                    <img src={thumb} alt="" className="w-full h-full object-cover" loading="lazy" onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />
                  ) : null}
                </div>
                <div className="px-2.5 py-2">
                  <div className="flex items-center gap-1.5">
                    <span
                      className={`w-4 h-4 rounded-full text-[10px] font-mono flex items-center justify-center shrink-0 ${
                        active ? "bg-[#297a3a] text-[#ffffff]" : "bg-[#f4f4f5] text-[#4d4d4d] border border-[#ebebeb]"
                      }`}
                    >
                      {i + 1}
                    </span>
                    <span className="text-[13px] font-[500] text-[#171717] truncate">{m.label}</span>
                  </div>
                  <div className="mt-1 text-[12px] text-[#666666] leading-snug line-clamp-2">{m.what_to_look_for}</div>
                </div>
              </button>
              {i < moments.length - 1 && (
                <ChevronRight className="hidden sm:block absolute -right-[11px] top-[30%] w-3.5 h-3.5 text-[#c9c9c9] z-10" aria-hidden />
              )}
            </li>
          );
        })}
      </ol>
    </div>
  );
};
