import React, { useState } from "react";
import { ArrowLeft, Film, Link2, Printer } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type { CoachingStory, DeliveryDetailResponse, SessionSummary } from "../types";
import { CaptureGlyph, NoticeFooter, StatusBadge, ValueWithRange } from "../components/story/ui";

/** Printable / shareable Coach Card. Notices are always printed in full (card grows, text never clips). */
export const CoachCardView: React.FC<{
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
}> = ({ detail, story, session }) => {
  const id = detail.delivery.id;
  const [reelBusy, setReelBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [reelErr, setReelErr] = useState<string | null>(null);

  const exportReel = async () => {
    setReelBusy(true);
    setReelErr(null);
    try {
      const r = await api.exportReel(id);
      window.open(r.reel_url, "_blank");
    } catch (e: any) {
      setReelErr("Couldn't build the video reel right now.");
    } finally {
      setReelBusy(false);
    }
  };

  const copyLink = async () => {
    try {
      await navigator.clipboard.writeText(window.location.href.replace(/\/card$/, ""));
      setCopied(true);
      setTimeout(() => setCopied(false), 1800);
    } catch {
      /* ignore */
    }
  };

  const date = new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });

  return (
    <div className="space-y-4">
      <div className="no-print flex flex-wrap items-center justify-between gap-2">
        <button onClick={() => navigate({ name: "delivery", id, tab: "story" })} className="inline-flex items-center gap-1 text-[13px] text-[#666666] hover:text-[#171717]">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to story
        </button>
        <div className="flex flex-wrap gap-2">
          <button onClick={copyLink} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Link2 className="w-3.5 h-3.5" /> {copied ? "Link copied" : "Copy link"}
          </button>
          <button onClick={exportReel} disabled={reelBusy} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Film className="w-3.5 h-3.5" /> {reelBusy ? "Building reel…" : "Export video reel"}
          </button>
          <button onClick={() => window.print()} className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Printer className="w-3.5 h-3.5" /> Print / save PDF
          </button>
        </div>
      </div>
      {reelErr && <div className="no-print text-[13px] text-[#4d4d4d]">{reelErr}</div>}

      <article className="mx-auto max-w-[760px] bg-[#ffffff] border border-[#ebebeb] rounded-[10px] p-6 md:p-8 space-y-6 print:border-0 print:p-0">
        <header className="flex flex-col-reverse sm:flex-row sm:items-start sm:justify-between gap-3 sm:gap-4 border-b border-[#ebebeb] pb-4">
          <div>
            <div className="text-[12px] text-[#666666]">Coach card · {date}</div>
            <h1 className="mt-1 text-[24px] font-[500] tracking-[-0.03em] text-[#171717]">{detail.delivery.label}</h1>
            <div className="text-[13px] text-[#666666] mt-0.5">
              {session?.bowler_name ? `${session.bowler_name} · ` : ""}
              {story?.subhead}
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-[12px] text-[#171717] font-[500] shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-[#297a3a]" /> SportsPostureAnalysis
          </div>
        </header>

        {!story ? (
          <p className="text-[14px] text-[#4d4d4d]">This delivery's coaching story isn't available yet.</p>
        ) : (
          <>
            <p className="text-[19px] leading-[1.35] tracking-[-0.02em] text-[#171717]">{story.headline}</p>

            {story.key_moments.length > 0 && (
              <div className="grid grid-cols-4 gap-2">
                {story.key_moments.map((m) => (
                  <figure key={m.id} className="space-y-1">
                    <div className="aspect-[4/5] rounded-[6px] overflow-hidden bg-[#f4f4f5] border border-[#ebebeb]">
                      <img src={m.thumbnail_url ?? api.frameUrl(id, m.frame_idx, 360)} alt={m.label} className="w-full h-full object-cover" />
                    </div>
                    <figcaption className="text-[11px] text-[#4d4d4d] text-center">{m.label}</figcaption>
                  </figure>
                ))}
              </div>
            )}

            <section className="rounded-[8px] border border-[#ebebeb] p-4">
              <div className="text-[12px] text-[#666666]">#1 takeaway</div>
              <div className="text-[16px] font-[500] text-[#171717] mt-1">{story.top_takeaway.title}</div>
              <p className="text-[13.5px] text-[#4d4d4d] mt-1 leading-[1.55]">{story.top_takeaway.plain_explanation}</p>
              {story.top_takeaway.what_to_try && (
                <p className="text-[13.5px] text-[#171717] mt-2">
                  <span className="text-[#666666]">Try this: </span>
                  {story.top_takeaway.what_to_try}
                </p>
              )}
            </section>

            <section>
              <div className="text-[14px] font-[500] text-[#171717] mb-2">Body checkpoints</div>
              <ul className="divide-y divide-[#ebebeb] border-y border-[#ebebeb]">
                {story.checkpoints.map((c) => (
                  <li key={c.id} className="py-2.5 flex items-center justify-between gap-3 break-inside-avoid">
                    <div className="min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <span className="text-[14px] text-[#171717]">{c.title}</span>
                        <StatusBadge status={c.status} label={c.status_label} unlockReason={c.unlock_reason} compact />
                      </div>
                      <div className="text-[11.5px] text-[#8f8f8f]">{c.moment_label}</div>
                    </div>
                    {c.status !== "not_visible" ? <ValueWithRange value={c.value_text} range={c.range_text} size="sm" /> : <span className="text-[12px] text-[#a8a8a8]">—</span>}
                  </li>
                ))}
              </ul>
            </section>

            {story.drills.length > 0 && (
              <section className="space-y-3">
                <div className="text-[14px] font-[500] text-[#171717]">Practice plan</div>
                {story.drills.map((d, i) => (
                  <div key={d.id} className="rounded-[8px] border border-[#ebebeb] p-4 break-inside-avoid">
                    <div className="text-[14px] font-[500] text-[#171717]">
                      {i + 1}. {d.title}
                    </div>
                    {d.cue && <div className="text-[13.5px] text-[#171717] mt-1">“{d.cue}”</div>}
                    <ol className="mt-2 space-y-1 list-decimal pl-5 text-[13px] text-[#4d4d4d]">
                      {d.steps.map((s, j) => (
                        <li key={j}>{s}</li>
                      ))}
                    </ol>
                    {d.caution && <p className="mt-2 text-[12px] text-[#666666]">Take care: {d.caution}</p>}
                  </div>
                ))}
              </section>
            )}

            <div className="flex items-start gap-2 text-[12.5px] text-[#4d4d4d]">
              <CaptureGlyph grade={story.capture.grade} />
              <span>
                <span className="text-[#171717]">{story.capture.label}:</span> {story.capture.summary}
              </span>
            </div>
          </>
        )}

        <NoticeFooter notices={story?.notices} className="print:border print:border-[#d4d4d8]" />
      </article>
    </div>
  );
};
