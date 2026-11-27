import React, { useEffect, useMemo, useState } from "react";
import { ArrowRight, Camera, GitCompare, PlayCircle, Upload } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type { DeliveryCard, OverviewResponse } from "../types";
import { CaptureGlyph, CAPTURE_GRADE_LABEL, CheckpointDots, EmptyState, formatDate, SectionTitle, Spinner } from "../components/story/ui";

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

export const DeliveryCardTile: React.FC<{ card: DeliveryCard; showBowler?: boolean }> = ({ card, showBowler = true }) => (
  <button
    onClick={() => navigate({ name: "delivery", id: card.id, tab: "story" })}
    className="v-card overflow-hidden text-left hover:bg-[#fafafa] transition-colors group flex flex-col"
  >
    <div className="aspect-[16/10] bg-[#f4f4f5] overflow-hidden relative">
      {card.thumbnail_url && (
        <img src={card.thumbnail_url} alt="" className="w-full h-full object-cover" loading="lazy" onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />
      )}
      {(card.kind === "sample_real" || card.kind === "sample_demo") && (
        <span className="absolute top-2 left-2 h-5 px-2 rounded-full bg-[#ffffff]/95 border border-[#ebebeb] text-[11px] text-[#171717] inline-flex items-center">
          {card.kind === "sample_real" ? "Sample · real video" : "Sample · demo"}
        </span>
      )}
      {card.kind === "fused_3d" && (
        <span className="absolute top-2 left-2 h-5 px-2 rounded-full bg-[#ffffff]/95 border border-[#ebebeb] text-[11px] text-[#171717] inline-flex items-center">
          Dual-camera 3D
        </span>
      )}
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
);

export const HomeView: React.FC<{ onOpenFilmGuide: () => void }> = ({ onOpenFilmGuide }) => {
  const { data, error } = useOverview();
  const userDeliveries = useMemo(
    () => (data?.recent_deliveries ?? []).filter((d) => d.kind === "upload" || d.kind === "fused_3d"),
    [data]
  );

  if (error) {
    return (
      <EmptyState
        title="The studio server isn't responding"
        body={<span>Start it with <code className="font-mono text-[12px]">bowling-biomech serve-studio</code>, then refresh this page.</span>}
        action={<button onClick={() => window.location.reload()} className="v-btn-primary h-9 px-4 text-[13px]">Refresh</button>}
      />
    );
  }
  if (!data) {
    return (
      <div className="flex items-center gap-2 text-[13px] text-[#666666] py-16 justify-center">
        <Spinner /> Loading…
      </div>
    );
  }

  const samples = data.samples.filter((s) => s.capture_grade !== "unusable").slice(0, 4);
  const firstSample = samples[0] ?? null;
  const latest = userDeliveries[0] ?? null;
  const showFirstRun = data.is_first_run || !latest;
  const sameBowlerPair = latest ? userDeliveries.filter((d) => d.bowler_id === latest.bowler_id).slice(0, 2) : [];

  return (
    <div className="space-y-10">
      {showFirstRun ? (
        <section className="pt-4 md:pt-10 max-w-[760px]">
          <h1 className="text-[34px] md:text-[48px] leading-[1.05] font-[500] tracking-[-0.045em] text-[#171717]">
            See what your bowling action is really doing.
          </h1>
          <p className="mt-4 text-[16px] md:text-[17px] leading-[1.55] text-[#4d4d4d] max-w-[620px]">
            Film one ball on your phone. Get five body checkpoints, one clear takeaway and a drill to try — with honest ranges, never guesses.
          </p>
          <div className="mt-7 flex flex-col sm:flex-row gap-2.5">
            <button onClick={() => navigate({ name: "new", bowler: null })} className="v-btn-primary h-11 px-5 text-[14px] flex items-center justify-center gap-2">
              <Upload className="w-4 h-4" /> Analyse my bowling
            </button>
            {firstSample && (
              <button
                onClick={() => navigate({ name: "delivery", id: firstSample.delivery_id, tab: "story" })}
                className="v-btn-ghost h-11 px-5 text-[14px] flex items-center justify-center gap-2"
              >
                <PlayCircle className="w-4 h-4" /> Try a sample: {firstSample.title}
              </button>
            )}
          </div>
          {!data.health.can_analyse_new_video && (
            <p className="mt-3 text-[12.5px] text-[#8f8f8f]">
              {data.health.reason_plain || "Analysing new videos needs the pose engine, which isn't connected right now — samples work fully."}
            </p>
          )}
          <ol className="mt-10 grid sm:grid-cols-3 gap-3">
            {[
              { n: 1, t: "Film from the side or behind", d: "Whole body in frame, phone steady. Slow-mo helps.", icon: <Camera className="w-4 h-4" /> },
              { n: 2, t: "Upload the clip", d: "One ball, or a whole net session — we'll find each delivery.", icon: <Upload className="w-4 h-4" /> },
              { n: 3, t: "Understand & practise", d: "Tap your key moments and body checkpoints, then try a drill.", icon: <PlayCircle className="w-4 h-4" /> },
            ].map((s) => (
              <li key={s.n} className="v-card p-4">
                <div className="flex items-center gap-2 text-[#171717]">
                  <span className="w-6 h-6 rounded-full bg-[#f4f4f5] border border-[#ebebeb] font-mono text-[11px] flex items-center justify-center">{s.n}</span>
                  {s.icon}
                </div>
                <div className="mt-3 text-[14px] font-[500] text-[#171717]">{s.t}</div>
                <div className="mt-1 text-[13px] text-[#666666] leading-snug">{s.d}</div>
              </li>
            ))}
          </ol>
          <button onClick={onOpenFilmGuide} className="mt-3 text-[13px] text-[#171717] underline underline-offset-2">
            See the 30-second filming guide
          </button>
        </section>
      ) : (
        <section className="pt-2 md:pt-6">
          <div className="text-[13px] text-[#666666]">Your latest ball</div>
          <div className="mt-2 v-card overflow-hidden grid md:grid-cols-[minmax(0,420px)_1fr]">
            <div className="aspect-[16/10] md:aspect-auto bg-[#f4f4f5] overflow-hidden">
              {latest!.thumbnail_url && <img src={latest!.thumbnail_url} alt="" className="w-full h-full object-cover" />}
            </div>
            <div className="p-5 md:p-7 flex flex-col gap-3">
              <div className="flex items-center justify-between gap-2 text-[12px] text-[#8f8f8f]">
                <span>
                  {latest!.bowler_name} · {formatDate(latest!.created_at)}
                </span>
                <CheckpointDots statuses={latest!.checkpoint_statuses} size={10} />
              </div>
              <h1 className="text-[24px] md:text-[28px] leading-[1.2] font-[500] tracking-[-0.03em] text-[#171717]">{latest!.headline}</h1>
              {latest!.top_takeaway_title && (
                <p className="text-[14px] text-[#4d4d4d]">
                  <span className="text-[#666666]">#1 takeaway: </span>
                  {latest!.top_takeaway_title}
                </p>
              )}
              <div className="mt-auto flex flex-wrap gap-2 pt-2">
                <button onClick={() => navigate({ name: "delivery", id: latest!.id, tab: "story" })} className="v-btn-primary h-9 px-4 text-[13px] flex items-center gap-1.5">
                  Continue <ArrowRight className="w-3.5 h-3.5" />
                </button>
                {sameBowlerPair.length === 2 && (
                  <button
                    onClick={() => navigate({ name: "compare", a: sameBowlerPair[1].id, b: sameBowlerPair[0].id })}
                    className="v-btn-ghost h-9 px-4 text-[13px] flex items-center gap-1.5"
                  >
                    <GitCompare className="w-3.5 h-3.5" /> Compare last 2 balls
                  </button>
                )}
                <button onClick={() => navigate({ name: "new", bowler: latest!.bowler_id })} className="v-btn-ghost h-9 px-4 text-[13px] flex items-center gap-1.5">
                  <Upload className="w-3.5 h-3.5" /> Film another ball
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {!showFirstRun && userDeliveries.length > 1 && (
        <section>
          <SectionTitle hint={<button onClick={() => navigate({ name: "deliveries", bowler: null })} className="underline underline-offset-2">See all</button>}>
            Recent deliveries
          </SectionTitle>
          <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-3">
            {userDeliveries.slice(1, 7).map((c) => (
              <DeliveryCardTile key={c.id} card={c} />
            ))}
          </div>
        </section>
      )}

      {samples.length > 0 && (
        <section>
          <SectionTitle hint="Explore how it works — no upload needed">Sample deliveries</SectionTitle>
          <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-3">
            {samples.map((s) => (
              <button
                key={s.delivery_id}
                onClick={() => navigate({ name: "delivery", id: s.delivery_id, tab: "story" })}
                className="v-card overflow-hidden text-left hover:bg-[#fafafa] transition-colors"
              >
                <div className="aspect-[16/10] bg-[#f4f4f5] overflow-hidden relative">
                  {s.thumbnail_url && <img src={s.thumbnail_url} alt="" className="w-full h-full object-cover" loading="lazy" onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} />}
                  <span className="absolute top-2 left-2 h-5 px-2 rounded-full bg-[#ffffff]/95 border border-[#ebebeb] text-[11px] text-[#171717] inline-flex items-center">
                    {s.kind === "real" ? "Real video" : "Demo"}
                  </span>
                </div>
                <div className="p-3.5">
                  <div className="text-[14px] font-[500] text-[#171717]">{s.title}</div>
                  <div className="text-[12.5px] text-[#666666] mt-0.5 leading-snug">{s.subtitle}</div>
                  <div className="mt-2 flex items-center gap-1.5 text-[12px] text-[#666666]">
                    <CaptureGlyph grade={s.capture_grade} size={11} /> {CAPTURE_GRADE_LABEL[s.capture_grade]}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </section>
      )}

      <section className="border-t border-[#ebebeb] pt-6 grid md:grid-cols-3 gap-4 text-[13px] text-[#4d4d4d] leading-relaxed">
        <div>
          <div className="text-[#171717] font-[500]">Honest by design</div>
          Every number shows the range it very likely sits in. If the camera couldn't see something, we say so instead of guessing.
        </div>
        <div>
          <div className="text-[#171717] font-[500]">No made-up scores</div>
          No 0–100 ratings. Statuses only appear when a measurement clearly clears a research threshold, even after allowing for error.
        </div>
        <div>
          <div className="text-[#171717] font-[500]">Not a legality or medical test</div>
          We never judge elbow legality (that needs an accredited lab) and never diagnose injuries.
        </div>
      </section>
      <footer className="text-[12px] text-[#a8a8a8] pb-4 flex items-center gap-2">
        <span className={`w-1.5 h-1.5 rounded-full ${data.health.can_analyse_new_video ? "bg-[#297a3a]" : "bg-[#c9c9c9]"}`} />
        {data.health.engine_label}
      </footer>
    </div>
  );
};
