/**
 * CoachCardView — 1-page printable breakdown card + 1080×1350 PNG export (docs/research_app_landscape_v5.md §8.2, §9).
 *
 * Renders: hero frame with segment-colored skeleton overlay, top 3 measured metrics with CI ranges & RangeRails,
 * #1 takeaway, linked practice drills, and FULL un-truncated ICC 15° & medical notices both on screen/print and
 * inside the exported 1080×1350 PNG canvas.
 */
import React, { useMemo, useState } from "react";
import { ArrowLeft, Download, Film, Link2, Printer } from "lucide-react";
import { api } from "../api";
import { navigate } from "../router";
import type {
  BodyCheckpoint,
  BodyZone,
  CheckpointStatus,
  CoachingStory,
  DeliveryDetailResponse,
  MetricEstimate,
  ObservabilityGrade,
  SessionSummary,
  TelestratorShape,
  ViewClarity,
} from "../types";
import {
  GUARDRAIL_ICC,
  GUARDRAIL_MEDICAL,
  OBS_GLYPH,
  SEG,
  SEG_COLOR,
  SMPL24_BONES,
  boneSegment,
  jointSegment,
} from "../theme";
import {
  GuardrailStrip,
  MetricTile,
  SectionLabel,
  StatusPill,
  autoRailDomain,
  fmtMetric,
  isMeasured,
} from "../components/ui/PaperPrimitives";

export interface CoachCardViewProps {
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
}

const STATUS_RANK: Record<CheckpointStatus, number> = { focus: 0, watch: 1, strong: 2, on_track: 3, not_visible: 9 };
const ZONE_COLOR: Record<BodyZone, string> = {
  knee: SEG.pelvis,
  approach: SEG.pelvis,
  trunk: SEG.thorax,
  spine: SEG.thorax,
  shoulders: SEG.thorax,
};
const CLARITY_TO_OBS: Record<Exclude<ViewClarity, null>, ObservabilityGrade> = { clear: "high", fair: "moderate", rough: "low" };

interface CardMetric {
  label: string;
  valueText: string;
  rangeText: string;
  value: number | null;
  ciLow: number | null;
  ciHigh: number | null;
  min: number;
  max: number;
  color: string;
  observability: ObservabilityGrade;
  status?: CheckpointStatus;
  statusLabel?: string;
}

export const CoachCardView: React.FC<CoachCardViewProps> = ({ detail, story, session }) => {
  const id = detail.delivery.id;
  const fs = detail.fact_sheet;
  const tel = detail.telemetry;
  const [reelBusy, setReelBusy] = useState(false);
  const [pngBusy, setPngBusy] = useState(false);
  const [copied, setCopied] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const metricOf = (key: string): MetricEstimate | undefined => fs.metrics?.[key] ?? fs.v5_metrics?.[key];

  // Hero frame: prefer Ball Release (or takeaway frame), clamped to telemetry bounds.
  const heroFrameIdx = useMemo(() => {
    const maxF = Math.max(0, (tel.num_frames || 1) - 1);
    const candidate =
      story?.key_moments.find((m) => m.id === "ball_release")?.frame_idx ??
      (fs.events.is_complete_delivery ? fs.events.br.frame_idx : null) ??
      story?.top_takeaway.frame_idx ??
      0;
    return Math.max(0, Math.min(maxF, Math.round(candidate)));
  }, [story, fs.events, tel.num_frames]);

  // Top 3 measured metrics (checkpoints first by story priority, fallback to canonical measured metrics).
  const topMetrics = useMemo<CardMetric[]>(() => {
    const out: CardMetric[] = [];
    if (story) {
      const measuredCps = story.checkpoints
        .filter((c) => c.status !== "not_visible" && c.value !== null)
        .sort((a, b) => STATUS_RANK[a.status] - STATUS_RANK[b.status])
        .slice(0, 3);
      for (const cp of measuredCps) {
        const m = metricOf(cp.metric_key);
        const [mn, mx] = autoRailDomain(cp.value, cp.ci_low, cp.ci_high, cp.unit);
        const obs: ObservabilityGrade = m?.observability ?? (cp.view_clarity ? CLARITY_TO_OBS[cp.view_clarity] : "moderate");
        out.push({
          label: cp.title,
          valueText: cp.value_text ?? fmtMetric(m).valueText ?? "—",
          rangeText: cp.range_text ?? fmtMetric(m).rangeText ?? "",
          value: cp.value,
          ciLow: cp.ci_low,
          ciHigh: cp.ci_high,
          min: mn,
          max: mx,
          color: ZONE_COLOR[cp.body_zone],
          observability: obs,
          status: cp.status,
          statusLabel: cp.status_label,
        });
      }
    }
    if (out.length === 0) {
      Object.values(fs.metrics ?? {})
        .filter((m) => isMeasured(m) && m.key !== "icc_elbow_legality_verdict")
        .slice(0, 3)
        .forEach((m) => {
          const t = fmtMetric(m);
          const [mn, mx] = autoRailDomain(m.value, m.ci_low, m.ci_high, m.unit);
          out.push({
            label: m.label,
            valueText: t.valueText ?? "—",
            rangeText: t.rangeText ?? "",
            value: m.value,
            ciLow: m.ci_low,
            ciHigh: m.ci_high,
            min: mn,
            max: mx,
            color: SEG.thorax,
            observability: m.observability,
          });
        });
    }
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [story, fs.metrics, fs.v5_metrics]);

  const exportReel = async () => {
    setReelBusy(true);
    setErr(null);
    try {
      const r = await api.exportReel(id);
      window.open(r.reel_url, "_blank", "noopener");
    } catch {
      setErr("Couldn't build the 9:16 video reel right now.");
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
      setErr("Clipboard isn't available in this browser.");
    }
  };

  const downloadPng = async () => {
    setPngBusy(true);
    setErr(null);
    try {
      const dataUrl = await renderBreakdownCardPng({
        detail,
        story,
        session,
        heroFrameIdx,
        topMetrics,
      });
      const slug = (detail.delivery.label || id).toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "") || id;
      const a = document.createElement("a");
      a.href = dataUrl;
      a.download = `${slug}_breakdown_1080x1350.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (e: any) {
      setErr(`Couldn't export PNG: ${String(e?.message || e)}`);
    } finally {
      setPngBusy(false);
    }
  };

  const date = new Date().toLocaleDateString(undefined, { day: "numeric", month: "long", year: "numeric" });
  const heroUrl = api.frameUrl(id, heroFrameIdx, 960);

  return (
    <div className="space-y-4">
      <div className="no-print flex flex-wrap items-center justify-between gap-2">
        <button onClick={() => navigate({ name: "delivery", id, tab: "story" })} className="inline-flex items-center gap-1 text-[13px] text-[#666666] hover:text-[#171717]">
          <ArrowLeft className="w-3.5 h-3.5" /> Back to workbench
        </button>
        <div className="flex flex-wrap gap-2">
          <button onClick={copyLink} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Link2 className="w-3.5 h-3.5" /> {copied ? "Link copied" : "Copy link"}
          </button>
          <button onClick={exportReel} disabled={reelBusy} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Film className="w-3.5 h-3.5" /> {reelBusy ? "Building reel…" : "Export 9:16 reel"}
          </button>
          <button onClick={() => window.print()} className="v-btn-ghost h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Printer className="w-3.5 h-3.5" /> Print / PDF
          </button>
          <button onClick={downloadPng} disabled={pngBusy} className="v-btn-primary h-8 px-3 text-[13px] flex items-center gap-1.5">
            <Download className="w-3.5 h-3.5" /> {pngBusy ? "Rendering PNG…" : "Download PNG (1080×1350)"}
          </button>
        </div>
      </div>
      {err && <div className="no-print text-[13px] text-[#4d4d4d]">{err}</div>}

      <article className="mx-auto max-w-[780px] bg-[#ffffff] shadow-[0_0_0_1px_rgba(0,0,0,0.08)] rounded-[8px] p-6 md:p-8 space-y-6 print:shadow-none print:border-0 print:p-0">
        {/* Header */}
        <header className="flex flex-col-reverse sm:flex-row sm:items-start sm:justify-between gap-3 border-b border-[#ebebeb] pb-4">
          <div>
            <div className="v-label">Breakdown card · {date}</div>
            <h1 className="mt-1 text-[24px] font-[500] tracking-[-0.03em] text-[#171717]">{detail.delivery.label}</h1>
            <div className="text-[13px] text-[#666666] mt-0.5 flex flex-wrap items-center gap-x-2">
              {session?.bowler_name && <span className="text-[#171717]">{session.bowler_name}</span>}
              <span>
                {fs.bowling_arm === "left" ? "Left-arm" : "Right-arm"} · {story?.capture.view_label ?? fs.calibration.view_class.replace(/_/g, " ")} · {Math.round(fs.audit.fps)} fps
              </span>
            </div>
          </div>
          <div className="flex items-center gap-1.5 text-[12px] text-[#171717] font-[500] shrink-0">
            <span className="w-2.5 h-2.5 rounded-full bg-[#297a3a]" /> SportsPostureAnalysis
          </div>
        </header>

        {story && <p className="text-[19px] leading-[1.35] tracking-[-0.02em] text-[#171717] font-[500]">{story.headline}</p>}

        {/* Hero frame + key moments strip */}
        <div className="grid grid-cols-1 md:grid-cols-12 gap-3 items-start">
          <div className="md:col-span-7">
            <div className="relative aspect-[16/10] rounded-[6px] overflow-hidden bg-[#f4f4f5] shadow-[0_0_0_1px_#ebebeb]">
              <img src={heroUrl} alt="Key delivery moment" className="w-full h-full object-contain bg-[#f4f4f5]" />
              <div className="absolute bottom-2 left-2 rounded-full bg-[#ffffff]/90 px-2.5 py-0.5 text-[11px] v-num text-[#171717] shadow-[0_0_0_1px_rgba(0,0,0,0.08)]">
                {fs.events.is_complete_delivery ? "Ball release" : `Frame ${heroFrameIdx + 1}`}
              </div>
            </div>
          </div>
          <div className="md:col-span-5 space-y-2.5">
            {story && (
              <section className="rounded-[6px] bg-[#fafafa] shadow-[0_0_0_1px_#ebebeb] p-3.5">
                <SectionLabel>#1 Takeaway</SectionLabel>
                <div className="text-[15px] font-[500] text-[#171717] mt-1 leading-snug">{story.top_takeaway.title}</div>
                <p className="text-[12.5px] text-[#4d4d4d] mt-1 leading-[1.5]">{story.top_takeaway.plain_explanation}</p>
                {story.top_takeaway.what_to_try && (
                  <p className="text-[12.5px] text-[#171717] mt-2 leading-[1.45]">
                    <span className="text-[#8f8f8f]">Try: </span>
                    {story.top_takeaway.what_to_try}
                  </p>
                )}
              </section>
            )}
          </div>
        </div>

        {/* Top 3 measured hero tiles */}
        {topMetrics.length > 0 && (
          <section className="space-y-2">
            <SectionLabel>Top measured numbers (with 95% likely range)</SectionLabel>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {topMetrics.map((m) => (
                <MetricTile
                  key={m.label}
                  label={m.label}
                  valueText={m.valueText}
                  rangeText={m.rangeText || undefined}
                  range={{ value: m.value, ciLow: m.ciLow, ciHigh: m.ciHigh, min: m.min, max: m.max, color: m.color }}
                  observability={m.observability}
                  status={m.status}
                  accentColor={m.color}
                />
              ))}
            </div>
          </section>
        )}

        {/* Key moments 4-up */}
        {story && story.key_moments.length > 0 && (
          <section className="space-y-2">
            <SectionLabel>Key moments</SectionLabel>
            <div className="grid grid-cols-4 gap-2">
              {story.key_moments.map((m) => (
                <figure key={m.id} className="space-y-1">
                  <div className="aspect-[4/5] rounded-[6px] overflow-hidden bg-[#f4f4f5] shadow-[0_0_0_1px_#ebebeb]">
                    <img src={m.thumbnail_url ?? api.frameUrl(id, m.frame_idx, 360)} alt={m.label} className="w-full h-full object-cover" />
                  </div>
                  <figcaption className="text-[11px] text-[#4d4d4d] text-center">{m.label}</figcaption>
                </figure>
              ))}
            </div>
          </section>
        )}

        {/* Full checkpoints list */}
        {story && story.checkpoints.length > 0 && (
          <section className="space-y-2">
            <SectionLabel>All 5 body checkpoints</SectionLabel>
            <ul className="divide-y divide-[#ebebeb] border-y border-[#ebebeb]">
              {story.checkpoints.map((c: BodyCheckpoint) => (
                <li key={c.id} className="py-2 flex items-center justify-between gap-3 break-inside-avoid">
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="text-[13.5px] text-[#171717]">{c.title}</span>
                      <StatusPill status={c.status} label={c.status_label} />
                    </div>
                    <div className="text-[11.5px] text-[#8f8f8f]">{c.moment_label}</div>
                  </div>
                  {c.status !== "not_visible" ? (
                    <div className="text-right shrink-0">
                      <div className="v-num text-[14px] text-[#171717]">{c.value_text}</div>
                      {c.range_text && <div className="v-num text-[11px] text-[#8f8f8f]">likely {c.range_text}</div>}
                    </div>
                  ) : (
                    <span className="text-[12px] text-[#a8a8a8]">Not visible</span>
                  )}
                </li>
              ))}
            </ul>
          </section>
        )}

        {/* Practice drills */}
        {story && story.drills.length > 0 && (
          <section className="space-y-2.5">
            <SectionLabel>Practice plan</SectionLabel>
            {story.drills.map((d, i) => (
              <div key={d.id} className="rounded-[6px] shadow-[0_0_0_1px_#ebebeb] p-3.5 break-inside-avoid">
                <div className="text-[14px] font-[500] text-[#171717]">
                  {i + 1}. {d.title}
                </div>
                {d.cue && <div className="text-[13px] text-[#171717] mt-1">“{d.cue}”</div>}
                {d.steps.length > 0 && (
                  <ol className="mt-1.5 space-y-0.5 list-decimal pl-5 text-[12.5px] text-[#4d4d4d]">
                    {d.steps.map((s, j) => (
                      <li key={j}>{s}</li>
                    ))}
                  </ol>
                )}
                {d.caution && <p className="mt-1.5 text-[11.5px] text-[#666666]">⚠ {d.caution}</p>}
              </div>
            ))}
          </section>
        )}

        {/* Always-full, never-truncated guardrail strip */}
        <GuardrailStrip />
      </article>
    </div>
  );
};

// ─────────────────────────────────────────────────────────────────────────────
// 1080×1350 Offscreen Canvas PNG Renderer (Paper Instrument + full guardrails)
// ─────────────────────────────────────────────────────────────────────────────

function loadImg(url: string): Promise<HTMLImageElement | null> {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

function wrapLines(ctx: CanvasRenderingContext2D, text: string, maxWidth: number): string[] {
  const words = text.replace(/\s+/g, " ").trim().split(" ").filter(Boolean);
  const lines: string[] = [];
  let line = "";
  for (const w of words) {
    const test = line ? `${line} ${w}` : w;
    if (ctx.measureText(test).width > maxWidth && line) {
      lines.push(line);
      line = w;
    } else {
      line = test;
    }
  }
  if (line) lines.push(line);
  return lines;
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

async function renderBreakdownCardPng(opts: {
  detail: DeliveryDetailResponse;
  story: CoachingStory | null;
  session: SessionSummary | null;
  heroFrameIdx: number;
  topMetrics: CardMetric[];
}): Promise<string> {
  const { detail, story, session, heroFrameIdx, topMetrics } = opts;
  const W = 1080;
  const H = 1350;
  const PAD = 48;
  const CW = W - PAD * 2;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas 2D context unavailable");

  const sans = 'Geist, Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif';
  const mono = '"Geist Mono", ui-monospace, SFMono-Regular, Menlo, monospace';
  const fs = detail.fact_sheet;
  const tel = detail.telemetry;

  // Paper background
  ctx.fillStyle = "#FAFAFA";
  ctx.fillRect(0, 0, W, H);
  // Subtle outer frame
  ctx.fillStyle = "#FFFFFF";
  roundRect(ctx, 20, 20, W - 40, H - 40, 12);
  ctx.fill();
  ctx.strokeStyle = "#E5E5E5";
  ctx.lineWidth = 1.5;
  ctx.stroke();

  // Pre-measure FULL un-truncated guardrail footer height so it NEVER clips or truncates!
  ctx.font = `400 15.5px ${sans}`;
  const guardInnerW = CW - 36;
  const iccLines = wrapLines(ctx, `🔒 Elbow legality: refused. ${GUARDRAIL_ICC}`, guardInnerW);
  const medLines = wrapLines(ctx, `✚ Not a medical diagnosis. ${GUARDRAIL_MEDICAL}`, guardInnerW);
  const guardLineH = 22;
  const guardH = 22 + iccLines.length * guardLineH + 10 + medLines.length * guardLineH + 18;
  const guardY = H - 20 - 24 - guardH;

  // 1. Header
  let y = 52;
  ctx.fillStyle = SEG.positive;
  ctx.beginPath();
  ctx.arc(PAD + 7, y + 7, 6, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = "#171717";
  ctx.font = `500 16px ${sans}`;
  ctx.fillText("SportsPostureAnalysis", PAD + 22, y + 13);

  const dateStr = new Date().toLocaleDateString(undefined, { day: "numeric", month: "short", year: "numeric" });
  const metaRight = `${fs.bowling_arm === "left" ? "LEFT-ARM" : "RIGHT-ARM"} · ${(story?.capture.view_label ?? fs.calibration.view_class.replace(/_/g, " ")).toUpperCase()} · ${Math.round(fs.audit.fps)} FPS · ${dateStr.toUpperCase()}`;
  ctx.font = `400 13px ${mono}`;
  ctx.fillStyle = "#8F8F8F";
  const mrW = ctx.measureText(metaRight).width;
  ctx.fillText(metaRight, W - PAD - mrW, y + 12);

  y += 34;
  ctx.fillStyle = "#171717";
  ctx.font = `500 32px ${sans}`;
  const titleText = session?.bowler_name ? `${detail.delivery.label} · ${session.bowler_name}` : detail.delivery.label;
  ctx.fillText(titleText.slice(0, 52), PAD, y + 24);
  y += 42;

  // Divider
  ctx.strokeStyle = "#EBEBEB";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PAD, y);
  ctx.lineTo(W - PAD, y);
  ctx.stroke();
  y += 22;

  // 2. Compute remaining vertical budget above guardrail box so everything fits harmoniously
  const drills = (story?.drills ?? []).slice(0, 2);
  const tilesH = topMetrics.length > 0 ? 180 : 0;
  const takeawayEstH = story ? 158 : 0;
  const drillsEstH = drills.length > 0 ? 52 + drills.length * 64 : 0;
  const gaps = 20 + (tilesH ? 20 : 0) + (takeawayEstH ? 20 : 0) + (drillsEstH ? 20 : 0);
  const availForHero = guardY - y - tilesH - takeawayEstH - drillsEstH - gaps;
  const heroH = Math.max(280, Math.min(440, availForHero));

  // Hero frame container
  ctx.save();
  roundRect(ctx, PAD, y, CW, heroH, 10);
  ctx.fillStyle = "#F4F4F5";
  ctx.fill();
  ctx.clip();

  const img = await loadImg(api.frameUrl(detail.delivery.id, heroFrameIdx, 1080));
  if (img && img.naturalWidth > 0) {
    const scale = Math.min(CW / img.naturalWidth, heroH / img.naturalHeight);
    const dw = img.naturalWidth * scale;
    const dh = img.naturalHeight * scale;
    const dx = PAD + (CW - dw) / 2;
    const dy = y + (heroH - dh) / 2;
    ctx.drawImage(img, dx, dy, dw, dh);

    // Draw segment-colored 2D skeleton with 2px white halo
    const pts = tel.joints2d_px?.[heroFrameIdx];
    const inF = tel.joint_in_frame?.[heroFrameIdx];
    const srcW = tel.width || img.naturalWidth;
    const srcH = tel.height || img.naturalHeight;
    if (pts && pts.length >= 24 && srcW > 0 && srcH > 0) {
      const sx = dw / srcW;
      const sy = dh / srcH;
      const px = (i: number) => dx + pts[i][0] * sx;
      const py = (i: number) => dy + pts[i][1] * sy;
      const vis = (i: number) => pts[i] && Number.isFinite(pts[i][0]) && Number.isFinite(pts[i][1]) && (!inF || inF[i] !== false);
      ctx.lineCap = "round";
      for (const [a, b] of SMPL24_BONES) {
        if (!vis(a) || !vis(b)) continue;
        const seg = boneSegment(a, b, fs.bowling_arm);
        const col = SEG_COLOR[seg];
        ctx.strokeStyle = "rgba(255,255,255,0.92)";
        ctx.lineWidth = 7;
        ctx.beginPath();
        ctx.moveTo(px(a), py(a));
        ctx.lineTo(px(b), py(b));
        ctx.stroke();
        ctx.strokeStyle = col;
        ctx.lineWidth = 3.5;
        ctx.beginPath();
        ctx.moveTo(px(a), py(a));
        ctx.lineTo(px(b), py(b));
        ctx.stroke();
      }
      for (let i = 0; i < 24; i++) {
        if (!vis(i)) continue;
        const col = SEG_COLOR[jointSegment(i, fs.bowling_arm)];
        ctx.fillStyle = "#FFFFFF";
        ctx.beginPath();
        ctx.arc(px(i), py(i), 5.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = col;
        ctx.beginPath();
        ctx.arc(px(i), py(i), 3.6, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // Draw any user telestration shapes on this frame or pinned
    const frameShapes: TelestratorShape[] = (detail.annotations ?? []).filter((s) => s.pinned_all_frames || s.frame === heroFrameIdx);
    if (frameShapes.length > 0 && srcW > 0 && srcH > 0) {
      const sx = dw / srcW;
      const sy = dh / srcH;
      ctx.strokeStyle = SEG.tele;
      ctx.lineWidth = 3;
      for (const s of frameShapes) {
        if (s.points.length >= 2 && (s.type === "line" || s.type === "arrow" || s.type === "angle" || s.type === "freehand" || s.type === "ruler")) {
          ctx.beginPath();
          s.points.forEach(([x0, y0], idx) => {
            const cx = dx + x0 * sx;
            const cy = dy + y0 * sy;
            if (idx === 0) ctx.moveTo(cx, cy);
            else ctx.lineTo(cx, cy);
          });
          ctx.stroke();
        }
      }
    }
  } else {
    ctx.fillStyle = "#8F8F8F";
    ctx.font = `400 18px ${sans}`;
    ctx.fillText("Video frame unavailable for offline export", PAD + 32, y + heroH / 2);
  }

  // Hero moment pill
  const momentBadge = fs.events.is_complete_delivery ? "BALL RELEASE · 3D POSE OVERLAY" : `FRAME ${heroFrameIdx + 1}`;
  ctx.font = `500 13px ${mono}`;
  const mbW = ctx.measureText(momentBadge).width + 24;
  ctx.fillStyle = "rgba(255,255,255,0.92)";
  roundRect(ctx, PAD + 16, y + heroH - 44, mbW, 28, 14);
  ctx.fill();
  ctx.fillStyle = "#171717";
  ctx.fillText(momentBadge, PAD + 28, y + heroH - 25);
  ctx.restore();

  roundRect(ctx, PAD, y, CW, heroH, 10);
  ctx.strokeStyle = "#E5E5E5";
  ctx.lineWidth = 1;
  ctx.stroke();

  y += heroH + 22;

  // 3. Top 3 Metric Tiles
  if (topMetrics.length > 0) {
    const gap = 16;
    const cols = topMetrics.length;
    const tileW = (CW - gap * (cols - 1)) / cols;
    topMetrics.forEach((m, idx) => {
      const tx = PAD + idx * (tileW + gap);
      ctx.fillStyle = "#FFFFFF";
      roundRect(ctx, tx, y, tileW, tilesH, 8);
      ctx.fill();
      ctx.strokeStyle = "#E5E5E5";
      ctx.lineWidth = 1;
      ctx.stroke();

      // Label + segment dot
      ctx.fillStyle = m.color;
      ctx.beginPath();
      ctx.arc(tx + 22, y + 26, 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#8F8F8F";
      ctx.font = `500 13px ${mono}`;
      ctx.fillText(m.label.toUpperCase().slice(0, 24), tx + 34, y + 31);

      // Hero numeral
      ctx.fillStyle = "#171717";
      ctx.font = `500 44px ${sans}`;
      ctx.fillText(m.valueText, tx + 18, y + 86);

      // Interval
      ctx.fillStyle = "#666666";
      ctx.font = `400 15px ${mono}`;
      ctx.fillText(m.rangeText ? `likely ${m.rangeText}` : "interval shown on rail", tx + 18, y + 112);

      // RangeRail
      const rx = tx + 18;
      const rw = tileW - 36;
      const ry = y + 132;
      const span = m.max - m.min || 1;
      const pos = (v: number) => rx + Math.max(0, Math.min(1, (v - m.min) / span)) * rw;
      ctx.strokeStyle = "#D4D4D4";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(rx, ry);
      ctx.lineTo(rx + rw, ry);
      ctx.stroke();
      if (m.ciLow !== null && m.ciHigh !== null) {
        const x1 = pos(m.ciLow);
        const x2 = pos(m.ciHigh);
        ctx.fillStyle = m.color;
        ctx.globalAlpha = 0.35;
        roundRect(ctx, x1, ry - 4, Math.max(4, x2 - x1), 8, 4);
        ctx.fill();
        ctx.globalAlpha = 1;
      }
      if (m.value !== null) {
        ctx.fillStyle = m.color;
        ctx.beginPath();
        ctx.arc(pos(m.value), ry, 6, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = "#FFFFFF";
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Observability + status footer
      const obs = OBS_GLYPH[m.observability] ?? OBS_GLYPH.moderate;
      ctx.fillStyle = "#4D4D4D";
      ctx.font = `400 13.5px ${mono}`;
      ctx.fillText(`${obs.bars} ${obs.word}`, tx + 18, y + tilesH - 16);
      if (m.statusLabel) {
        const sw = ctx.measureText(m.statusLabel).width;
        ctx.fillStyle = m.status === "strong" ? SEG.positive : "#171717";
        ctx.fillText(m.statusLabel, tx + tileW - 18 - sw, y + tilesH - 16);
      }
    });
    y += tilesH + 20;
  }

  // 4. #1 Takeaway
  if (story) {
    const tkH = Math.min(takeawayEstH, Math.max(118, guardY - y - drillsEstH - (drillsEstH ? 24 : 16)));
    ctx.fillStyle = "#FAFAFA";
    roundRect(ctx, PAD, y, CW, tkH, 8);
    ctx.fill();
    ctx.strokeStyle = "#EBEBEB";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.fillStyle = "#8F8F8F";
    ctx.font = `500 12.5px ${mono}`;
    ctx.fillText("#1 TAKEAWAY", PAD + 20, y + 26);

    ctx.fillStyle = "#171717";
    ctx.font = `500 21px ${sans}`;
    ctx.fillText(story.top_takeaway.title.slice(0, 72), PAD + 20, y + 54);

    ctx.fillStyle = "#4D4D4D";
    ctx.font = `400 16px ${sans}`;
    const expl = wrapLines(ctx, story.top_takeaway.plain_explanation, CW - 40).slice(0, 2);
    expl.forEach((ln, i) => ctx.fillText(ln, PAD + 20, y + 82 + i * 22));
    if (story.top_takeaway.what_to_try && y + 82 + expl.length * 22 + 16 < y + tkH - 6) {
      ctx.fillStyle = "#171717";
      ctx.font = `500 15.5px ${sans}`;
      const tryLines = wrapLines(ctx, `Try: ${story.top_takeaway.what_to_try}`, CW - 40).slice(0, 1);
      ctx.fillText(tryLines[0], PAD + 20, y + 82 + expl.length * 22 + 10);
    }
    y += tkH + 18;
  }

  // 5. Drills (if any and space remains above guardrails)
  if (drills.length > 0 && guardY - y > 76) {
    ctx.fillStyle = "#8F8F8F";
    ctx.font = `500 12.5px ${mono}`;
    ctx.fillText("PRACTICE FOCUS", PAD, y + 14);
    y += 24;
    for (let i = 0; i < drills.length; i++) {
      if (guardY - y < 60) break;
      const d = drills[i];
      const dh = 54;
      ctx.fillStyle = "#FFFFFF";
      roundRect(ctx, PAD, y, CW, dh, 8);
      ctx.fill();
      ctx.strokeStyle = "#EBEBEB";
      ctx.stroke();

      ctx.fillStyle = "#171717";
      ctx.font = `500 16px ${sans}`;
      ctx.fillText(`${i + 1}. ${d.title}`, PAD + 16, y + 23);
      if (d.cue) {
        ctx.fillStyle = "#4D4D4D";
        ctx.font = `400 14.5px ${sans}`;
        const cueLine = wrapLines(ctx, `Cue: “${d.cue}”`, CW - 32)[0] ?? "";
        ctx.fillText(cueLine, PAD + 16, y + 44);
      }
      y += dh + 10;
    }
  }

  // 6. FULL, UN-TRUNCATED ICC 15° & MEDICAL GUARDRAILS (always rendered at bottom, never clipped)
  ctx.fillStyle = "#FFFFFF";
  roundRect(ctx, PAD, guardY, CW, guardH, 8);
  ctx.fill();
  ctx.strokeStyle = "#D4D4D4";
  ctx.lineWidth = 1.2;
  ctx.stroke();

  let gy = guardY + 26;
  ctx.font = `400 15.5px ${sans}`;
  ctx.fillStyle = "#333333";
  iccLines.forEach((ln) => {
    ctx.fillText(ln, PAD + 18, gy);
    gy += guardLineH;
  });
  gy += 10;
  medLines.forEach((ln) => {
    ctx.fillText(ln, PAD + 18, gy);
    gy += guardLineH;
  });

  return canvas.toDataURL("image/png");
}

export default CoachCardView;
