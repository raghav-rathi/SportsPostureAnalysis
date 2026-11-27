import React, { useEffect, useState } from "react";
import { api } from "../../api";
import { navigate } from "../../router";
import type { DeliveryCard, DeliveryDetailResponse, OverviewResponse } from "../../types";
import { CaptureGlyph, CAPTURE_GRADE_LABEL, CheckpointDots, formatDate } from "../story/ui";

/** Shared overview loader (Home, Deliveries). */
export function useOverview() {
  const [data, setData] = useState<OverviewResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let cancelled = false;
    api
      .overview()
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, []);
  return { data, error };
}

/** Loads one delivery detail; keeps the last good payload while switching. */
export function useDeliveryDetail(id: string | null) {
  const [data, setData] = useState<DeliveryDetailResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    if (!id) return;
    let cancelled = false;
    setError(null);
    api
      .getDelivery(id)
      .then((d) => !cancelled && setData(d))
      .catch((e) => !cancelled && setError(String(e?.message || e)));
    return () => {
      cancelled = true;
    };
  }, [id]);
  return { data: data && data.delivery?.id === id ? data : data, error };
}

/** Mutable playhead shared between a video canvas and a 3D view without React re-renders. */
export interface Playhead {
  frame: number;
}

export const KIND_LABEL: Record<DeliveryCard["kind"], string> = {
  upload: "Your clip",
  sample_real: "Sample · real video",
  sample_demo: "Sample · studio demo",
  fused_3d: "Dual-camera 3D",
};

export const DeliveryCardTile: React.FC<{
  card: DeliveryCard;
  showBowler?: boolean;
  selected?: boolean;
  onSelect?: (id: string) => void;
}> = ({ card, showBowler = true, selected = false, onSelect }) => (
  <div
    className={`v-panel overflow-hidden text-left transition-shadow group flex flex-col relative ${
      selected ? "!shadow-[0_0_0_2px_#171717]" : "hover:shadow-[0_0_0_1px_rgba(0,0,0,0.18)]"
    }`}
  >
    <button
      onClick={() => navigate({ name: "delivery", id: card.id, tab: "story" })}
      className="text-left flex flex-col flex-1"
      aria-label={`Open ${card.label}`}
    >
      <div className="aspect-[16/10] bg-[#f4f4f5] overflow-hidden relative">
        {card.thumbnail_url && (
          <img
            src={card.thumbnail_url}
            alt=""
            className="w-full h-full object-cover group-hover:scale-[1.02] transition-transform duration-300"
            loading="lazy"
            onError={(e) => ((e.target as HTMLImageElement).style.display = "none")}
          />
        )}
        <span className="absolute top-2 left-2 v-chip !h-5 !text-[11px] bg-white/95">{KIND_LABEL[card.kind]}</span>
      </div>
      <div className="p-3.5 flex-1 flex flex-col gap-1.5">
        <div className="flex items-center justify-between gap-2 text-[12px] text-[#8f8f8f]">
          <span className="truncate">
            {showBowler ? `${card.bowler_name} · ` : ""}
            {formatDate(card.created_at)}
          </span>
          <CheckpointDots statuses={card.checkpoint_statuses} />
        </div>
        <div className="text-[14px] font-[500] text-[#171717] truncate">{card.label}</div>
        <p className="text-[13px] text-[#4d4d4d] leading-snug line-clamp-2">{card.headline}</p>
        <div className="mt-auto pt-1 flex items-center gap-1.5 text-[12px] text-[#666666]">
          <CaptureGlyph grade={card.capture_grade} size={11} /> {CAPTURE_GRADE_LABEL[card.capture_grade]} · {card.view_label}
        </div>
      </div>
    </button>
    {onSelect && (
      <button
        onClick={() => onSelect(card.id)}
        className={`absolute top-2 right-2 w-6 h-6 rounded-full flex items-center justify-center text-[12px] ${
          selected ? "bg-[#171717] text-white" : "bg-white/95 text-[#8f8f8f] v-hairline"
        }`}
        aria-pressed={selected}
        aria-label={selected ? "Deselect" : "Select for compare"}
      >
        {selected ? "✓" : "+"}
      </button>
    )}
  </div>
);
