/**
 * StrobeStage: Stromotion & Kinogram stage for the Delivery Workbench (docs/research_app_landscape_v5.md §8.2).
 *
 * Three sub-modes on the same shared playhead:
 *  1. Photo Strobe  — backend view-adaptive multi-exposure JPEG (`api.stroboscopicUrl`) with `trails` & `skeleton`
 *                     toggles and an honest camera-motion / pan-verification note.
 *  2. 3D Strobe     — `<PitchStage3D strobeMode />` rendering 6 sculpted mannequins from BFC → FT along the MCC pitch.
 *  3. Kinogram Grid — Kinovea-style 2×4 sequence of frames from BFC → FT (`api.frameUrl`), preserving portrait aspect
 *                     ratio (never cropping feet or release hand), with phase labels, ms rel FFC, optional segment
 *                     skeleton overlay, and click-to-seek on the shared playhead.
 */
import React, { useEffect, useMemo, useState } from "react";
import { AlertCircle, Box, Grid2x2, Layers, RefreshCw } from "lucide-react";
import type { BiomechanicalFactSheet, TelemetryV2 } from "../../types";
import { SEG_COLOR, SMPL24_BONES, boneSegment } from "../../theme";
import { api } from "../../api";
import { usePlayhead } from "../PlayheadProvider";
import { PitchStage3D } from "./PitchStage3D";

export type StrobeSubMode = "photo" | "3d" | "kinogram";

export interface StrobeStageProps {
  deliveryId: string;
  telemetry: TelemetryV2;
  factSheet: BiomechanicalFactSheet;
  bowlingArm: "right" | "left";
  stroboscopicUrl?: string | null;
  /** Optional pan-tracker summary if passed by parent (additive; safe to omit). */
  panSummary?: {
    is_verified: boolean;
    median_residual_px: number;
    max_pan_deg: number;
    verification_reason: string;
  } | null;
}

interface KinogramCell {
  frameIdx: number;
  label: string;
  relMs: number;
  isKeyEvent: boolean;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function buildStrobeUrl(
  deliveryId: string,
  baseUrl: string | null | undefined,
  opts: { skeleton: boolean; trails: boolean }
): string {
  if (!baseUrl) {
    return api.stroboscopicUrl(deliveryId, { w: 1280, skeleton: opts.skeleton, trails: opts.trails });
  }
  const sep = baseUrl.includes("?") ? "&" : "?";
  const cleaned = baseUrl.replace(/([?&])(w|skeleton|trails)=[^&]*/g, "").replace(/[?&]+$/, "");
  const joiner = cleaned.includes("?") ? "&" : sep;
  return `${cleaned}${joiner}w=1280&skeleton=${opts.skeleton ? "true" : "false"}&trails=${opts.trails ? "true" : "false"}`;
}

function buildKinogramCells(telemetry: TelemetryV2, factSheet: BiomechanicalFactSheet): KinogramCell[] {
  const n = Math.max(1, telemetry.num_frames || telemetry.joints2d_px.length || 1);
  const fps = Math.max(1, telemetry.analysis_fps || factSheet.audit?.fps || 30);
  const ev = factSheet.events;
  const times = telemetry.time_rel_ffc_ms ?? [];

  const bfc = clamp(Math.round(ev?.bfc?.frame_idx ?? Math.floor(n * 0.2)), 0, n - 1);
  const ffc = clamp(Math.round(ev?.ffc?.frame_idx ?? Math.floor(n * 0.5)), 0, n - 1);
  const br = clamp(Math.round(ev?.br?.frame_idx ?? Math.floor(n * 0.72)), 0, n - 1);
  const uah =
    ev?.uah?.frame_idx != null
      ? clamp(Math.round(ev.uah.frame_idx), 0, n - 1)
      : clamp(Math.round(ffc + (br - ffc) * 0.55), 0, n - 1);
  const ft =
    ev?.ft_peak?.frame_idx != null
      ? clamp(Math.round(ev.ft_peak.frame_idx), 0, n - 1)
      : clamp(Math.round(br + Math.max(2, 0.16 * fps)), 0, n - 1);

  const preBfc = clamp(Math.round(bfc - Math.max(1, 0.08 * fps)), 0, bfc);
  const midStride = clamp(Math.round((bfc + ffc) / 2), bfc, ffc);
  const earlyFt = clamp(Math.round((br + ft) / 2), br, ft);

  const raw: Array<{ f: number; label: string; key: boolean }> = [
    { f: preBfc, label: "Gather", key: false },
    { f: bfc, label: "Back-foot land", key: true },
    { f: midStride, label: "Delivery stride", key: false },
    { f: ffc, label: "Front-foot plant", key: true },
    { f: uah, label: ev?.uah ? "Arm horizontal" : "Pre-release", key: !!ev?.uah },
    { f: br, label: "Ball release", key: true },
    { f: earlyFt, label: "Early follow-thru", key: false },
    { f: ft, label: "Follow-through", key: false },
  ];

  // If the clip is very short, preserve 8 chronological slots by spreading around [preBfc..ft] while keeping key events intact.
  const used = new Set<number>();
  const cells: KinogramCell[] = raw.map((item, idx) => {
    let f = item.f;
    if (used.has(f) && n >= 8) {
      const target = clamp(Math.round(preBfc + ((ft - preBfc) * idx) / 7), 0, n - 1);
      if (!used.has(target)) f = target;
    }
    used.add(f);
    const rel = times[f] ?? Math.round(((f - ffc) * 1000) / fps);
    return {
      frameIdx: f,
      label: item.label,
      relMs: Math.round(rel),
      isKeyEvent: item.key,
    };
  });

  return cells;
}

const KinogramSkeletonOverlay: React.FC<{
  telemetry: TelemetryV2;
  frameIdx: number;
  bowlingArm: "right" | "left";
}> = ({ telemetry, frameIdx, bowlingArm }) => {
  const pts = telemetry.joints2d_px?.[frameIdx];
  const inFr = telemetry.joint_in_frame?.[frameIdx];
  const vw = Math.max(1, telemetry.width || 1280);
  const vh = Math.max(1, telemetry.height || 720);
  if (!pts || pts.length < 24) return null;

  const strokeW = Math.max(2.2, Math.min(vw, vh) * 0.0055);
  const haloW = strokeW + Math.max(2.4, Math.min(vw, vh) * 0.005);

  const bones = SMPL24_BONES.filter(([a, b]) => {
    if (inFr && (inFr[a] === false || inFr[b] === false)) return false;
    const pa = pts[a];
    const pb = pts[b];
    return (
      Array.isArray(pa) &&
      Array.isArray(pb) &&
      Number.isFinite(pa[0]) &&
      Number.isFinite(pa[1]) &&
      Number.isFinite(pb[0]) &&
      Number.isFinite(pb[1])
    );
  });

  return (
    <svg
      viewBox={`0 0 ${vw} ${vh}`}
      preserveAspectRatio="xMidYMid meet"
      className="pointer-events-none absolute inset-0 h-full w-full"
      aria-hidden
    >
      <g strokeLinecap="round" strokeLinejoin="round">
        {bones.map(([a, b], i) => (
          <line
            key={`h-${i}`}
            x1={pts[a][0]}
            y1={pts[a][1]}
            x2={pts[b][0]}
            y2={pts[b][1]}
            stroke="rgba(255,255,255,0.92)"
            strokeWidth={haloW}
          />
        ))}
        {bones.map(([a, b], i) => (
          <line
            key={`b-${i}`}
            x1={pts[a][0]}
            y1={pts[a][1]}
            x2={pts[b][0]}
            y2={pts[b][1]}
            stroke={SEG_COLOR[boneSegment(a, b, bowlingArm)]}
            strokeWidth={strokeW}
          />
        ))}
      </g>
    </svg>
  );
};

export const StrobeStage: React.FC<StrobeStageProps> = ({
  deliveryId,
  telemetry,
  factSheet,
  bowlingArm,
  stroboscopicUrl,
  panSummary,
}) => {
  const playhead = usePlayhead();
  const isPanning = !!factSheet.calibration?.is_panning_camera;
  const panUnverified = isPanning && panSummary != null && !panSummary.is_verified;
  const photoExplicitlyDisabled = stroboscopicUrl === null || panUnverified;

  const [subMode, setSubMode] = useState<StrobeSubMode>(() => (photoExplicitlyDisabled ? "3d" : "photo"));
  const [showTrails, setShowTrails] = useState(true);
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [photoStatus, setPhotoStatus] = useState<"loading" | "ready" | "error">("loading");
  const [reloadNonce, setReloadNonce] = useState(0);

  const imgSrc = useMemo(() => {
    const base = buildStrobeUrl(deliveryId, stroboscopicUrl, { skeleton: showSkeleton, trails: showTrails });
    return reloadNonce > 0 ? `${base}&_r=${reloadNonce}` : base;
  }, [deliveryId, stroboscopicUrl, showSkeleton, showTrails, reloadNonce]);

  useEffect(() => {
    if (!photoExplicitlyDisabled) {
      setPhotoStatus("loading");
    }
  }, [imgSrc, photoExplicitlyDisabled]);

  const kinogramCells = useMemo(() => buildKinogramCells(telemetry, factSheet), [telemetry, factSheet]);

  const eventFrames = useMemo(
    () => ({
      bfc: factSheet.events?.bfc?.frame_idx ?? null,
      ffc: factSheet.events?.ffc?.frame_idx ?? null,
      uah: factSheet.events?.uah?.frame_idx ?? null,
      br: factSheet.events?.br?.frame_idx ?? null,
      ft: factSheet.events?.ft_peak?.frame_idx ?? null,
    }),
    [factSheet.events]
  );

  const viewClass = factSheet.calibration?.view_class ?? "unknown";
  const isCoronalView = viewClass.includes("rear") || viewClass.includes("front");
  const fps = Math.round(telemetry.analysis_fps || factSheet.audit?.fps || 30);

  return (
    <div className="flex flex-col w-full rounded-[6px] bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.08)] overflow-hidden">
      {/* Sub-mode bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-[#ebebeb] bg-[#fafafa]">
        <div className="v-seg" role="tablist" aria-label="Strobe view mode">
          <button
            type="button"
            role="tab"
            aria-selected={subMode === "photo"}
            aria-pressed={subMode === "photo"}
            onClick={() => setSubMode("photo")}
            className="inline-flex items-center gap-1.5"
          >
            <Layers size={13} />
            Photo Strobe
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={subMode === "3d"}
            aria-pressed={subMode === "3d"}
            onClick={() => setSubMode("3d")}
            className="inline-flex items-center gap-1.5"
          >
            <Box size={13} />
            3D Strobe
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={subMode === "kinogram"}
            aria-pressed={subMode === "kinogram"}
            onClick={() => setSubMode("kinogram")}
            className="inline-flex items-center gap-1.5"
          >
            <Grid2x2 size={13} />
            Kinogram Grid
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          {(subMode === "photo" || subMode === "kinogram") && (
            <button
              type="button"
              aria-pressed={showSkeleton}
              onClick={() => setShowSkeleton((v) => !v)}
              className={`h-[26px] px-2 rounded-[5px] font-mono text-[11px] transition-colors ${
                showSkeleton ? "bg-[#171717] text-white" : "bg-white text-[#4d4d4d] shadow-[0_0_0_1px_#ebebeb] hover:text-[#171717]"
              }`}
            >
              Skeleton
            </button>
          )}
          {subMode === "photo" && (
            <button
              type="button"
              aria-pressed={showTrails}
              onClick={() => setShowTrails((v) => !v)}
              className={`h-[26px] px-2 rounded-[5px] font-mono text-[11px] transition-colors ${
                showTrails ? "bg-[#171717] text-white" : "bg-white text-[#4d4d4d] shadow-[0_0_0_1px_#ebebeb] hover:text-[#171717]"
              }`}
            >
              Trails
            </button>
          )}
          <span className="font-mono text-[10.5px] text-[#8f8f8f] pl-1">
            {fps} fps · {viewClass.replace(/_/g, " ")}
          </span>
        </div>
      </div>

      {/* 1. Photo Strobe */}
      {subMode === "photo" && (
        <div className="flex flex-col bg-[#f6f6f5]">
          {photoExplicitlyDisabled || photoStatus === "error" ? (
            <div className="flex flex-col items-center justify-center gap-3 px-6 py-14 text-center">
              <AlertCircle size={22} className="text-[#4d4d4d]" />
              <div className="max-w-md space-y-1">
                <p className="text-[14px] font-medium text-[#171717]">
                  {panUnverified
                    ? "Photo strobe is disabled on this panning clip"
                    : "Photo strobe could not be rendered for this delivery"}
                </p>
                <p className="text-[12.5px] leading-relaxed text-[#4d4d4d]">
                  {panUnverified
                    ? panSummary?.verification_reason ||
                      "Camera rotation could not be verified within the 2.5 px homography threshold, so compositing multiple exposures onto one background would introduce false motion blur."
                    : "A static or pan-verified background is required for multi-exposure photo compositing. 3D Strobe and the 2×4 Kinogram Grid remain fully accurate."}
                </p>
              </div>
              <div className="flex items-center gap-2 pt-1">
                <button
                  type="button"
                  onClick={() => setSubMode("3d")}
                  className="v-btn-primary h-8 px-3 text-[12px]"
                >
                  Open 3D Strobe
                </button>
                <button
                  type="button"
                  onClick={() => setSubMode("kinogram")}
                  className="v-btn-ghost h-8 px-3 text-[12px]"
                >
                  Open Kinogram Grid
                </button>
                {!photoExplicitlyDisabled && (
                  <button
                    type="button"
                    onClick={() => setReloadNonce((n) => n + 1)}
                    className="v-btn-ghost h-8 px-2.5 text-[12px] inline-flex items-center gap-1"
                  >
                    <RefreshCw size={12} />
                    Retry
                  </button>
                )}
              </div>
            </div>
          ) : (
            <div className="relative flex items-center justify-center min-h-[360px] max-h-[540px] w-full overflow-hidden bg-[#f4f4f5]">
              {photoStatus === "loading" && (
                <div className="absolute inset-0 flex items-center justify-center bg-[#f4f4f5]/80">
                  <span className="v-label">Compositing multi-exposure strobe…</span>
                </div>
              )}
              <img
                src={imgSrc}
                alt="Multi-exposure stroboscopic bowling sequence"
                onLoad={() => setPhotoStatus("ready")}
                onError={() => setPhotoStatus("error")}
                className={`max-h-[520px] w-full object-contain transition-opacity duration-200 ${
                  photoStatus === "ready" ? "opacity-100" : "opacity-0"
                }`}
              />
            </div>
          )}

          {/* Honest camera-registration / pan-verification note */}
          <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-t border-[#ebebeb] bg-white font-mono text-[11px] text-[#4d4d4d]">
            <span>
              {isPanning
                ? panSummary?.is_verified
                  ? `Pan-registered strobe · verified 3-DOF rotation (median residual ${panSummary.median_residual_px.toFixed(2)} px, max pan ${panSummary.max_pan_deg.toFixed(1)}°)`
                  : "Panning clip detected · compositing uses verified rotational registration or falls back to a coronal strip when background registration is unreliable."
                : isCoronalView
                  ? "Coronal sequence strip · rear/front camera clips display spaced frames side-by-side because the bowler moves along the lens axis."
                  : "Static-camera multi-exposure composite · back-foot landing through follow-through."}
            </span>
            <span className="text-[#8f8f8f]">BFC → FT</span>
          </div>
        </div>
      )}

      {/* 2. 3D Strobe */}
      {subMode === "3d" && (
        <div className="p-2 bg-[#fafafa]">
          <PitchStage3D
            telemetry={telemetry}
            bowlingArm={bowlingArm}
            strobeMode
            defaultPreset="side"
            showTrails={{ wrist: true, head: false, com: false }}
            eventFrames={eventFrames}
            heightClass="h-[460px]"
          />
        </div>
      )}

      {/* 3. Kinogram Grid (2×4, click any frame to seek playhead) */}
      {subMode === "kinogram" && (
        <div className="p-3 bg-[#fafafa]">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2.5">
            {kinogramCells.map((cell, idx) => {
              const isCurrent = playhead.frame === cell.frameIdx;
              const sign = cell.relMs > 0 ? "+" : "";
              return (
                <button
                  key={`${cell.label}-${idx}`}
                  type="button"
                  onClick={() => playhead.seekToEventFrame(cell.frameIdx, true)}
                  className={`group flex flex-col overflow-hidden rounded-[6px] bg-white text-left transition-shadow ${
                    isCurrent
                      ? "shadow-[0_0_0_2px_#171717]"
                      : "shadow-[0_0_0_1px_#ebebeb] hover:shadow-[0_0_0_1px_#a8a8a8]"
                  }`}
                >
                  <div className="relative aspect-[4/3] w-full bg-[#f4f4f5] overflow-hidden">
                    <img
                      src={api.frameUrl(deliveryId, cell.frameIdx, 480)}
                      alt={`${cell.label} (${sign}${cell.relMs} ms relative to front-foot plant)`}
                      loading="lazy"
                      className="h-full w-full object-contain"
                    />
                    {showSkeleton && (
                      <KinogramSkeletonOverlay telemetry={telemetry} frameIdx={cell.frameIdx} bowlingArm={bowlingArm} />
                    )}
                    <span
                      className={`absolute left-1.5 top-1.5 rounded-[4px] px-1.5 py-0.5 font-mono text-[9.5px] leading-none tabular-nums ${
                        cell.isKeyEvent
                          ? "bg-[#171717] text-white"
                          : "bg-white/92 text-[#171717] shadow-[0_0_0_1px_rgba(0,0,0,0.08)]"
                      }`}
                    >
                      {sign}
                      {cell.relMs} ms
                    </span>
                    <span className="absolute right-1.5 top-1.5 rounded-[4px] bg-white/90 px-1 py-0.5 font-mono text-[9px] leading-none text-[#8f8f8f] tabular-nums">
                      #{idx + 1}
                    </span>
                  </div>
                  <div className="flex items-center justify-between gap-1 px-2 py-1.5 border-t border-[#ebebeb]">
                    <span className="truncate text-[11.5px] font-medium text-[#171717]">{cell.label}</span>
                    <span className="shrink-0 font-mono text-[10px] text-[#8f8f8f] tabular-nums">
                      f{cell.frameIdx}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>
          <p className="mt-2 font-mono text-[10.5px] text-[#8f8f8f]">
            Tap any frame to seek the playhead · times shown in ms relative to front-foot plant (FFC = 0 ms).
          </p>
        </div>
      )}
    </div>
  );
};

export default StrobeStage;
