import React, { useState } from "react";
import { Download, Film, Printer } from "lucide-react";
import { api } from "../api";
import type { DeliveryDetailResponse, MetricEstimate, SessionSummary } from "../types";

interface CoachReportViewProps {
  detail: DeliveryDetailResponse;
  session: SessionSummary;
  phaseThumbs: Record<string, string>;
}

export const CoachReportView: React.FC<CoachReportViewProps> = ({
  detail,
  session,
  phaseThumbs,
}) => {
  const fs = detail.fact_sheet;
  const [reelBusy, setReelBusy] = useState(false);
  const [reelUrl, setReelUrl] = useState<string | null>(null);
  const [reelErr, setReelErr] = useState<string | null>(null);

  const verifiedMetrics: MetricEstimate[] = Object.values(fs.metrics).filter(
    (m) => m.key !== "icc_elbow_legality_verdict" && !m.suppressed && m.value != null
  );
  const withheldCount = Object.values(fs.metrics).filter(
    (m) => m.key !== "icc_elbow_legality_verdict" && (m.suppressed || m.value == null)
  ).length;

  const handleExportReel = async () => {
    setReelBusy(true);
    setReelErr(null);
    try {
      const r = await api.exportReel(detail.delivery.id);
      setReelUrl(r.reel_url);
    } catch (err: any) {
      setReelErr(err?.message || "Reel export failed.");
    } finally {
      setReelBusy(false);
    }
  };

  const handleDownloadStandaloneHtml = () => {
    const el = document.getElementById("printable-coach-dossier");
    if (!el) return;
    const esc = (s: string) =>
      s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    const html = `<!doctype html>
<html lang="en"><head><meta charset="utf-8" /><title>SportsPosture Dossier — ${esc(detail.delivery.label)}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #171717; background: #fff; margin: 28px auto; max-width: 860px; line-height: 1.5; }
h1,h2,h3 { font-weight: 500; letter-spacing: -0.02em; }
table { width: 100%; border-collapse: collapse; font-size: 13px; margin: 12px 0; }
th, td { border-bottom: 1px solid #ebebeb; padding: 6px 8px; text-align: left; }
.mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; }
.box { border: 1px solid #d4d4d8; border-radius: 6px; padding: 12px; margin: 12px 0; background: #fafafa; }
.green { color: #297a3a; }
</style></head><body>${el.innerHTML}</body></html>`;
    const blob = new Blob([html], { type: "text/html;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${detail.delivery.id}_coach_dossier.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const phases = [
    ["bfc", "BFC · Back Foot Contact", fs.events.bfc.frame_idx, fs.events.bfc.timestamp_s],
    ["ffc", "FFC · Front Foot Contact", fs.events.ffc.frame_idx, fs.events.ffc.timestamp_s],
    ...(fs.events.uah ? [["uah", "UAH · Upper Arm Horiz.", fs.events.uah.frame_idx, fs.events.uah.timestamp_s]] : []),
    ["br", "BR · Ball Release", fs.events.br.frame_idx, fs.events.br.timestamp_s],
  ];

  return (
    <div className="space-y-4 max-w-[920px] mx-auto">
      {/* Export Actions Bar (Hidden on Print) */}
      <div className="no-print flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-[#ebebeb]">
        <div>
          <div className="v-eyebrow">▲ EXPORT_DOSSIER</div>
          <h1 className="text-[24px] font-[450] tracking-[-0.04em] text-[#171717]">
            Printable Coach Report &amp; 9:16 Video Reel
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <button onClick={() => window.print()} className="v-btn-primary h-8 px-3.5 text-[12px] flex items-center gap-1.5">
            <Printer className="w-3.5 h-3.5" />
            Print / Save PDF
          </button>
          <button onClick={handleDownloadStandaloneHtml} className="v-btn-ghost h-8 px-3 text-[12px] flex items-center gap-1.5">
            <Download className="w-3.5 h-3.5" />
            HTML Dossier
          </button>
          <button onClick={handleExportReel} disabled={reelBusy} className="v-btn-ghost h-8 px-3 text-[12px] flex items-center gap-1.5">
            <Film className="w-3.5 h-3.5" />
            {reelBusy ? "Rendering 9:16..." : "Export 9:16 Reel"}
          </button>
          {reelUrl && (
            <a href={reelUrl} download className="v-btn-primary h-8 px-3 text-[12px] flex items-center gap-1.5">
              <span className="text-[#297a3a]">✓</span>
              <Download className="w-3.5 h-3.5" />
              Download MP4
            </a>
          )}
        </div>
      </div>

      {reelErr && (
        <div className="no-print p-3 rounded-[6px] border border-[#171717] bg-[#fafafa] font-mono text-[11px]">
          {reelErr}
        </div>
      )}

      {/* Printable White-Paper Dossier */}
      <article id="printable-coach-dossier" className="v-card p-6 md:p-8 space-y-6 bg-[#ffffff]">
        {/* Masthead */}
        <header className="flex flex-wrap items-start justify-between gap-4 pb-4 border-b border-[#171717]">
          <div>
            <div className="font-mono text-[11px] uppercase tracking-[0.071em] text-[#666666]">
              ▲ SPORTSPOSTURE · BIOMECHANICAL COACHING DOSSIER
            </div>
            <h2 className="text-[26px] font-[450] tracking-[-0.04em] text-[#171717] mt-1">
              {detail.delivery.label}
            </h2>
            <div className="text-[13px] text-[#4d4d4d] mt-0.5">
              Athlete: <strong className="text-[#171717]">{session.bowler_name}</strong> ({fs.bowling_arm}-arm ·{" "}
              {fs.action_class.replace("_", " ").toUpperCase()} action) · Session: {session.title}
            </div>
          </div>
          <div className="text-right font-mono text-[11px] text-[#4d4d4d] space-y-0.5">
            <div>{new Date().toISOString().slice(0, 10)}</div>
            <div>{fs.audit.width}×{fs.audit.height} @ {fs.audit.fps.toFixed(1)} fps</div>
            <div>{fs.calibration.view_class} ({fs.calibration.azimuth_deg.toFixed(0)}°)</div>
            <div className="text-[#297a3a]">
              {verifiedMetrics.length} verified · {withheldCount} withheld
            </div>
          </div>
        </header>

        {/* 4-Phase Contact Sheet */}
        <section className="space-y-2">
          <div className="v-eyebrow">1. DELIVERY PHASE SEQUENCE (STRIDE {fs.events.delivery_stride_duration_ms.toFixed(0)} MS · FFC→BR {fs.events.ffc_to_br_duration_ms.toFixed(0)} MS)</div>
          <div className={`grid gap-2.5 ${phases.length === 4 ? "grid-cols-2 sm:grid-cols-4" : "grid-cols-3"}`}>
            {phases.map(([id, lbl, fr, ts]) => (
              <div key={String(id)} className="border border-[#ebebeb] rounded-[6px] overflow-hidden">
                {phaseThumbs[String(id)] ? (
                  <img src={phaseThumbs[String(id)]} alt={String(lbl)} className="w-full aspect-video object-contain bg-[#f4f4f5]" />
                ) : (
                  <div className="aspect-video bg-[#fafafa] flex items-center justify-center font-mono text-[11px] text-[#8f8f8f]">
                    Frame {String(fr)}
                  </div>
                )}
                <div className="p-2 bg-[#ffffff]">
                  <div className="font-mono text-[10px] font-medium text-[#171717]">{String(lbl)}</div>
                  <div className="font-mono text-[10px] text-[#666666]">
                    f {String(fr)} · {Number(ts).toFixed(2)} s
                  </div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Verified Measurements Table */}
        <section className="space-y-2">
          <div className="v-eyebrow">2. VERIFIED 3D MEASUREMENTS ({fs.interval_type.replace(/_/g, " ").toUpperCase()})</div>
          {verifiedMetrics.length > 0 ? (
            <table className="w-full text-left border-collapse text-[12px]">
              <thead>
                <tr className="border-b border-[#171717] font-mono text-[10px] uppercase tracking-[0.06em] text-[#666666]">
                  <th className="py-1.5 pr-2">Metric</th>
                  <th className="py-1.5 px-2 text-right">Value</th>
                  <th className="py-1.5 px-2 text-right">95% Interval</th>
                  <th className="py-1.5 px-2">Grade</th>
                  <th className="py-1.5 pl-2">Reference</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#ebebeb]">
                {verifiedMetrics.map((m) => (
                  <tr key={m.key}>
                    <td className="py-1.5 pr-2 font-[500] text-[#171717]">{m.label}</td>
                    <td className="py-1.5 px-2 text-right font-mono v-tabular">
                      {m.value?.toFixed(1)} {m.unit}
                    </td>
                    <td className="py-1.5 px-2 text-right font-mono text-[#666666] v-tabular">
                      {m.ci_low != null && m.ci_high != null ? `[${m.ci_low.toFixed(1)}, ${m.ci_high.toFixed(1)}]` : "—"}
                    </td>
                    <td className="py-1.5 px-2 font-mono text-[10px] uppercase text-[#297a3a]">{m.observability}</td>
                    <td className="py-1.5 pl-2 text-[11px] text-[#666666]">{m.literature_source}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="text-[12px] text-[#666666]">
              No metrics cleared reliability gates for this clip (timebase or overhead release check failed).
            </p>
          )}
        </section>

        {/* Drills & Watchlist */}
        {(fs.corrective_drills.length > 0 || fs.strengths.length > 0 || fs.inconclusive_observations.length > 0) && (
          <section className="space-y-2.5">
            <div className="v-eyebrow">3. COACHING PRIORITIES, CORRECTIVE DRILLS &amp; WATCHLIST</div>
            {fs.strengths.length > 0 && (
              <div className="text-[12px] text-[#171717]">
                <strong className="text-[#297a3a]">Strengths:</strong> {fs.strengths.map((s) => s.title).join(" · ")}
              </div>
            )}
            {fs.corrective_drills.map((dr) => (
              <div key={dr.id} className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] text-[12px] space-y-1">
                <div className="font-[500] text-[#171717]">{dr.title}</div>
                <div className="text-[#4d4d4d]"><strong>Cue:</strong> {dr.cue_focus}</div>
                <div className="text-[#4d4d4d]">{dr.setup_steps.join(" ")}</div>
                <div className="font-mono text-[10px] text-[#666666]">{dr.literature_basis} · {dr.caution_note}</div>
              </div>
            ))}
            {fs.inconclusive_observations.length > 0 && (
              <div className="text-[12px] text-[#4d4d4d]">
                <strong className="text-[#171717]">Watchlist (interval straddles cutoff — verify before coaching):</strong>{" "}
                {fs.inconclusive_observations.map((o) => o.title).join(" · ")}
              </div>
            )}
          </section>
        )}

        {/* Coach Notes */}
        {(detail.notes?.length ?? 0) > 0 && (
          <section className="space-y-1.5">
            <div className="v-eyebrow">4. COACH FRAME NOTES</div>
            <ul className="space-y-1 text-[12px] text-[#171717]">
              {detail.notes.map((n) => (
                <li key={n.id}>
                  <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 rounded-[4px] bg-[#f4f4f5] mr-1.5">
                    {n.tag} {n.frame_idx != null ? `f${n.frame_idx}` : ""}
                  </span>
                  {n.body}
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Mandatory Non-Truncating Scientific & Legal Guardrails */}
        <footer className="p-3.5 rounded-[6px] border border-[#171717] bg-[#fafafa] text-[11px] text-[#4d4d4d] space-y-1.5">
          <div className="font-mono text-[10px] uppercase tracking-[0.06em] text-[#171717]">
            ▲ MANDATORY SCIENTIFIC &amp; CLINICAL GUARDRAILS
          </div>
          <p>
            <strong className="text-[#171717]">ICC 15° Elbow Extension Legality — PERMANENTLY REFUSED:</strong>{" "}
            {fs.metrics["icc_elbow_legality_verdict"]?.suppression_reason}
          </p>
          <p>
            <strong className="text-[#171717]">Medical Non-Diagnosis Disclaimer:</strong> Mechanical load &amp; durability
            screening estimates single-delivery mechanical posture only and is NOT a clinical injury diagnosis. Injury risk
            depends heavily on bowling workload, age/skeletal maturity, and physical capacity.
          </p>
        </footer>
      </article>
    </div>
  );
};
