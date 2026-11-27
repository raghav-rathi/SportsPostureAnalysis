import React, { useMemo } from "react";
import { Upload } from "lucide-react";
import { navigate } from "../router";
import { EmptyState, Spinner } from "../components/story/ui";
import { DeliveryCardTile, useOverview } from "./HomeView";

export const DeliveriesView: React.FC<{ bowler: string | null }> = ({ bowler }) => {
  const { data, error } = useOverview();
  const cards = useMemo(() => {
    const all = data?.recent_deliveries ?? [];
    return bowler ? all.filter((c) => c.bowler_id === bowler) : all;
  }, [data, bowler]);

  if (error) return <EmptyState title="Couldn't load your deliveries" body={error} />;
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
        <Spinner /> Loading deliveries…
      </div>
    );
  }

  const bowlers = data.bowlers.filter((b) => b.delivery_count > 0);

  return (
    <div className="space-y-5">
      <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-3">
        <div>
          <h1 className="text-[26px] font-[500] tracking-[-0.03em] text-[#171717]">Deliveries</h1>
          <p className="text-[13.5px] text-[#666666] mt-1">Every ball you've analysed. Tap one to open its story.</p>
        </div>
        <button onClick={() => navigate({ name: "new", bowler })} className="v-btn-primary h-9 px-4 text-[13px] flex items-center gap-1.5 self-start">
          <Upload className="w-3.5 h-3.5" /> New analysis
        </button>
      </div>

      {bowlers.length > 1 && (
        <div className="flex flex-wrap gap-1.5" role="group" aria-label="Filter by bowler">
          <button
            onClick={() => navigate({ name: "deliveries", bowler: null })}
            className={`h-8 px-3 rounded-full text-[13px] border ${!bowler ? "bg-[#171717] text-[#ffffff] border-[#171717]" : "bg-[#ffffff] text-[#4d4d4d] border-[#ebebeb]"}`}
          >
            Everyone
          </button>
          {bowlers.map((b) => (
            <button
              key={b.id}
              onClick={() => navigate({ name: "deliveries", bowler: b.id })}
              className={`h-8 px-3 rounded-full text-[13px] border ${
                bowler === b.id ? "bg-[#171717] text-[#ffffff] border-[#171717]" : "bg-[#ffffff] text-[#4d4d4d] border-[#ebebeb]"
              }`}
            >
              {b.name} <span className="font-mono text-[11px] opacity-70 ml-0.5">{b.delivery_count}</span>
            </button>
          ))}
        </div>
      )}

      {cards.length === 0 ? (
        <EmptyState
          title="No deliveries yet"
          body="Film one ball from the side or from behind, then upload it here."
          action={
            <button onClick={() => navigate({ name: "new", bowler })} className="v-btn-primary h-9 px-4 text-[13px]">
              Analyse my bowling
            </button>
          }
        />
      ) : (
        <div className="grid sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {cards.map((c) => (
            <DeliveryCardTile key={c.id} card={c} showBowler={!bowler} />
          ))}
        </div>
      )}
    </div>
  );
};
