import React, { useMemo } from "react";
import { Crosshair, RotateCcw } from "lucide-react";
import { useStudioStore } from "../store";
import type { DeliveryDetailResponse, SessionSummary } from "../types";

const LANDMARK_LABELS: Array<[string, string]> = [
  ["bowling_crease_off", "1. Bowling Crease (Off)"],
  ["bowling_crease_leg", "2. Bowling Crease (Leg)"],
  ["popping_crease_off", "3. Popping Crease (Off)"],
  ["popping_crease_leg", "4. Popping Crease (Leg)"],
  ["stump_middle_base", "5. Middle Stump Base"],
  ["stump_middle_top", "6. Middle Stump Top"],
];

const SHORT_COHORT_LABELS: Record<string, string> = {
  runup_speed_mps: "Run-Up Speed",
  front_knee_br_deg: "Front Knee (BR)",
  front_knee_ffc_deg: "Front Knee (FFC)",
  trunk_flexion_delta_deg: "Trunk Flex Δ",
  trunk_flexion_br_deg: "Trunk Flex (BR)",
  thorax_lat_flex_br_deg: "Thorax Side Tilt",
  rear_hip_bfc_deg: "Rear Hip (BFC)",
  shoulder_counter_rotation_deg: "Shoulder SCR",
  pelvis_alignment_bfc_deg: "Pelvis Align",
  delayed_arm_ffc_deg: "Delayed Arm",
  stride_length_pct_stature: "Stride Length",
};

interface LabTelemetryPanelProps {
  detail: DeliveryDetailResponse;
  session: SessionSummary;
  recomputeMs: number | null;
  onApplyCalibration: () => Promise<void>;
  onTogglePanTracking: (enabled: boolean) => Promise<void>;
  onOpenCompare: () => void;
}

export const LabTelemetryPanel: React.FC<LabTelemetryPanelProps> = ({
  detail,
  session,
  recomputeMs,
  onApplyCalibration,
  onTogglePanTracking,
  onOpenCompare,
}) => {
  const fs = detail.fact_sheet;
  const tel = detail.telemetry;
  const isb = detail.delivery.isb_summary ?? tel.isb_anatomy;
  const pan = detail.delivery.pan_summary;

  const calibrationMode = useStudioStore((s) => s.calibrationMode);
  const activeLandmarkKey = useStudioStore((s) => s.activeLandmarkKey);
  const draftLandmarks = useStudioStore((s) => s.draftLandmarks);
  const previewReprojErrPx = useStudioStore((s) => s.previewReprojErrPx);
  const ghostDeliveryId = useStudioStore((s) => s.ghostDeliveryId);

  const setCalibrationMode = useStudioStore((s) => s.setCalibrationMode);
  const setActiveLandmarkKey = useStudioStore((s) => s.setActiveLandmarkKey);
  const clearDraftLandmarks = useStudioStore((s) => s.clearDraftLandmarks);
  const setGhostDeliveryId = useStudioStore((s) => s.setGhostDeliveryId);

  const otherDeliveries = useMemo(
    () => (session.deliveries ?? []).filter((d) => d.id !== detail.delivery.id),
    [session.deliveries, detail.delivery.id]
  );

  const nDraft = Object.keys(draftLandmarks).length;

  return (
    <div className="space-y-4">
      {/* 1. 4-Click MCC Crease Calibration & 3-DOF Pan Tracker */}
      <div className="v-card p-4 space-y-3">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="v-eyebrow">▲ MCC_CALIBRATION &amp; 3-DOF_PAN</div>
            <h3 className="text-[15px] font-[500] text-[#171717] mt-0.5">
              Pitch Geometry &amp; Camera Tracking
            </h3>
          </div>
          {recomputeMs != null && (
            <span className="font-mono text-[10px] px-2 py-0.5 rounded-[4px] bg-[#297a3a]/[0.08] text-[#297a3a]">
              ✓ {recomputeMs.toFixed(1)} ms CPU
            </span>
          )}
        </div>

        {/* Current Calibration Metadata */}
        <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
          <div className="v-card-subtle p-2">
            <div className="text-[10px] text-[#8f8f8f]">METHOD</div>
            <div className="text-[#171717] truncate">{fs.calibration.method}</div>
          </div>
          <div className="v-card-subtle p-2">
            <div className="text-[10px] text-[#8f8f8f]">AZIMUTH / ELEV</div>
            <div className="text-[#171717] v-tabular">
              {fs.calibration.azimuth_deg.toFixed(1)}° / {fs.calibration.elevation_deg.toFixed(1)}°
            </div>
          </div>
          <div className="v-card-subtle p-2">
            <div className="text-[10px] text-[#8f8f8f]">METRIC SCALE</div>
            <div className={fs.calibration.is_metric_calibrated ? "text-[#297a3a]" : "text-[#666666]"}>
              {fs.calibration.is_metric_calibrated ? "✓ Calibrated" : "Self-Cal / Uncalibrated"}
            </div>
          </div>
          <div className="v-card-subtle p-2">
            <div className="text-[10px] text-[#8f8f8f]">CAMERA MOTION</div>
            <div className="text-[#171717]">
              {fs.calibration.is_panning_camera ? "Panning / Handheld" : "✓ Static Tripod"}
            </div>
          </div>
        </div>

        {/* 4-Click Calibration Controls */}
        <div className="pt-1 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setCalibrationMode(!calibrationMode)}
              className="v-btn-primary h-8 px-3 text-[12px] flex items-center gap-1.5"
            >
              {calibrationMode ? <span className="text-[#297a3a]">●</span> : <Crosshair className="w-3.5 h-3.5" />}
              {calibrationMode ? "Exit 4-Click Mode" : "Start 4-Click Crease Calibration"}
            </button>
            {nDraft > 0 && (
              <button onClick={clearDraftLandmarks} className="v-btn-ghost h-8 px-2.5 text-[12px] flex items-center gap-1">
                <RotateCcw className="w-3 h-3" />
                Reset ({nDraft})
              </button>
            )}
          </div>

          {calibrationMode && (
            <div className="p-3 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] space-y-2">
              <div className="font-mono text-[10px] text-[#4d4d4d]">
                Click ≥4 pitch corners directly on the video player using the 4× pixel loupe:
              </div>
              <div className="grid grid-cols-2 gap-1.5">
                {LANDMARK_LABELS.map(([k, lbl]) => {
                  const placed = Boolean(draftLandmarks[k]);
                  const active = activeLandmarkKey === k;
                  return (
                    <button
                      key={k}
                      onClick={() => setActiveLandmarkKey(k)}
                      className={`px-2 py-1.5 rounded-[5px] text-left font-mono text-[10px] border transition ${
                        active
                          ? "border-[#171717] bg-[#171717] text-[#ffffff]"
                          : placed
                          ? "border-[#297a3a] bg-[#297a3a]/[0.06] text-[#297a3a]"
                          : "border-[#ebebeb] bg-[#ffffff] text-[#4d4d4d]"
                      }`}
                    >
                      {placed ? "✓ " : ""}
                      {lbl}
                    </button>
                  );
                })}
              </div>
              <div className="flex items-center justify-between pt-1">
                <span className="font-mono text-[11px] text-[#4d4d4d]">
                  {previewReprojErrPx != null
                    ? `Live PnP error: ${previewReprojErrPx.toFixed(2)} px`
                    : `Placed ${nDraft}/4 minimum points`}
                </span>
                <button
                  onClick={onApplyCalibration}
                  disabled={nDraft < 4}
                  className="v-btn-primary h-7 px-3 text-[11px]"
                >
                  Apply &amp; Recompute
                </button>
              </div>
            </div>
          )}

          {tel.needs_gpu_refit_for_K && (
            <div className="p-2.5 rounded-[6px] border border-[#171717] bg-[#fafafa] font-mono text-[11px] text-[#171717]">
              ▲ Focal length shifted &gt;15% from initial pose pass; re-run GPU inference with this calibration for full metric scale.
            </div>
          )}
        </div>

        {/* 3-DOF Rotational Pan Tracker Toggle */}
        {(() => {
          const panEligible = !["single_view_studio", "real_clip_fixture", "dual_3d_ray_fusion"].includes(
            detail.delivery.pipeline_mode
          );
          return (
            <div className="pt-2 border-t border-[#ebebeb] space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div>
                  <div className="text-[12px] font-[500] text-[#171717]">3-DOF SO(3) Rotational Pan Tracker</div>
                  <div className="text-[11px] text-[#666666]">
                    {panEligible
                      ? "Recovers pure tripod pan/tilt on real footage with textured background."
                      : "Available on uploaded raw videos with real background texture (disabled for demo/fixture/dual proxies)."}
                  </div>
                </div>
                <button
                  disabled={!panEligible}
                  onClick={() => onTogglePanTracking(!detail.delivery.overrides?.enable_pan_tracking)}
                  className={`h-7 px-2.5 rounded-[5px] font-mono text-[11px] border transition shrink-0 ${
                    detail.delivery.overrides?.enable_pan_tracking
                      ? "bg-[#171717] text-[#ffffff] border-[#171717]"
                      : "v-btn-ghost"
                  }`}
                >
                  {detail.delivery.overrides?.enable_pan_tracking ? "Enabled" : "Run 3-DOF"}
                </button>
              </div>
              {pan && (
                <div className="p-2.5 rounded-[6px] bg-[#fafafa] border border-[#ebebeb] font-mono text-[11px] space-y-0.5">
                  <div className={pan.is_verified ? "text-[#297a3a] font-medium" : "text-[#4d4d4d]"}>
                    {pan.is_verified
                      ? `✓ VERIFIED · median ${pan.median_residual_px.toFixed(2)} px · max pan ${pan.max_pan_deg.toFixed(1)}°`
                      : "NOT VERIFIED / REJECTED"}
                  </div>
                  <div className="text-[#666666]">{pan.verification_reason}</div>
                </div>
              )}
            </div>
          );
        })()}
      </div>

      {/* 2. 3D Ghost Overlay Selector */}
      <div className="v-card p-4 space-y-2.5">
        <div className="flex items-center justify-between">
          <div>
            <div className="v-eyebrow">3D_GHOST_OVERLAY</div>
            <h3 className="text-[14px] font-[500] text-[#171717]">FFC-Aligned Skeleton Comparison</h3>
          </div>
          <button onClick={onOpenCompare} className="v-btn-ghost h-7 px-2.5 font-mono text-[11px] whitespace-nowrap shrink-0">
            Split Compare →
          </button>
        </div>
        <select
          value={ghostDeliveryId ?? ""}
          onChange={(e) => setGhostDeliveryId(e.target.value || null)}
          className="v-input w-full h-8 px-2.5 text-[12px]"
        >
          <option value="">No Ghost Overlay</option>
          {otherDeliveries.map((d) => (
            <option key={d.id} value={d.id}>
              Ghost: {d.label}
            </option>
          ))}
        </select>
      </div>

      {/* 3. ISB bml_movi_87 87-Marker Rigid-Body Anatomy */}
      <div className="v-card p-4 space-y-2.5">
        <div className="flex items-start justify-between gap-2">
          <div>
            <div className="v-eyebrow">▲ ISB_BML_MOVI_87_ANATOMY</div>
            <h3 className="text-[15px] font-[500] text-[#171717]">Pelvis, Thorax &amp; Foot Rigid Bodies</h3>
          </div>
          {isb && (
            <span className="font-mono text-[10px] text-[#666666]">
              ρ={isb.rho_assumed} · N_eff={isb.n_eff_pelvis.toFixed(2)}
            </span>
          )}
        </div>

        {isb && isb.metrics ? (
          <div className="divide-y divide-[#ebebeb] text-[12px]">
            {Object.values(isb.metrics).map((m) => (
              <div key={m.key} className="py-2 flex items-start justify-between gap-2">
                <div className="min-w-0">
                  <div className="font-[500] text-[#171717]">{m.label}</div>
                  {m.suppressed && m.suppression_reason && (
                    <div className="text-[11px] text-[#8f8f8f] mt-0.5">{m.suppression_reason}</div>
                  )}
                </div>
                {!m.suppressed && m.value != null ? (
                  <div className="text-right font-mono shrink-0 v-tabular">
                    <div className="text-[#171717] font-medium">
                      {m.value.toFixed(1)} {m.unit}
                    </div>
                    {m.ci_low != null && m.ci_high != null && (
                      <div className="text-[10px] text-[#666666]">
                        [{m.ci_low.toFixed(1)}, {m.ci_high.toFixed(1)}] · {m.observability}
                      </div>
                    )}
                  </div>
                ) : (
                  <span className="font-mono text-[10px] uppercase px-1.5 py-0.5 rounded-[4px] border border-dashed border-[#c9c9c9] text-[#8f8f8f] shrink-0">
                    Withheld
                  </span>
                )}
              </div>
            ))}
            <div className="pt-2 flex items-center justify-between font-mono text-[11px]">
              <span className="text-[#666666]">Front Foot Strike Mode</span>
              <span className="text-[#171717] uppercase">{isb.front_foot_strike_mode}</span>
            </div>
          </div>
        ) : (
          <div className="text-[12px] text-[#666666]">
            87-marker ISB anatomy is populated automatically on raw videos processed through the Warm GPU Worker or local NLF model.
          </div>
        )}
      </div>

      {/* 4. Monochrome Cohort Z-Score Forest Plot */}
      {fs.cohort_comparisons.length > 0 && (
        <div className="v-card p-4 space-y-2.5">
          <div>
            <div className="v-eyebrow">COHORT_Z_FOREST_PLOT</div>
            <h3 className="text-[15px] font-[500] text-[#171717]">Primary-Source Cohort Placement</h3>
          </div>
          <svg viewBox="0 0 360 180" className="w-full h-auto">
            {/* ±1 SD green band */}
            <rect x="155" y="12" width="90" height="148" fill="rgba(41, 122, 58, 0.08)" />
            <line x1="200" y1="12" x2="200" y2="160" stroke="#297a3a" strokeWidth="1" strokeDasharray="3 3" />
            <line x1="110" y1="160" x2="290" y2="160" stroke="#ebebeb" strokeWidth="1" />
            {[-2, -1, 0, 1, 2].map((z) => (
              <text key={z} x={200 + z * 45} y="174" textAnchor="middle" fill="#8f8f8f" fontSize="9" fontFamily="monospace">
                {z > 0 ? `+${z}σ` : `${z}σ`}
              </text>
            ))}
            {fs.cohort_comparisons.slice(0, 6).map((c, idx) => {
              const y = 26 + idx * 24;
              const clampX = (zv: number) => 200 + Math.max(-2.1, Math.min(2.1, zv)) * 45;
              const xPt = clampX(c.z_score);
              const xLo = clampX((c.ci_low - c.cohort_mean) / Math.max(c.cohort_sd, 1e-4));
              const xHi = clampX((c.ci_high - c.cohort_mean) / Math.max(c.cohort_sd, 1e-4));
              return (
                <g key={c.metric_key}>
                  <text x="4" y={y + 3} fill="#4d4d4d" fontSize="9.5" fontFamily="monospace">
                    {SHORT_COHORT_LABELS[c.metric_key] ?? c.label.slice(0, 17)}
                  </text>
                  <line x1={xLo} y1={y} x2={xHi} y2={y} stroke="#171717" strokeWidth="1.5" />
                  <circle cx={xPt} cy={y} r="3.5" fill="#171717" />
                </g>
              );
            })}
          </svg>
        </div>
      )}

      {/* 5. Complete Audit & Timing Diagnostics */}
      <div className="v-card p-4 space-y-2 font-mono text-[11px]">
        <div className="v-eyebrow">CAPTURE_AUDIT_DIAGNOSTICS</div>
        <div className="grid grid-cols-2 gap-y-1 text-[#4d4d4d]">
          <span>Resolution</span>
          <span className="text-right text-[#171717]">{fs.audit.width}×{fs.audit.height}</span>
          <span>Unique / Container FPS</span>
          <span className="text-right text-[#171717]">
            {fs.audit.fps.toFixed(2)} / {(fs.audit.container_fps ?? fs.audit.fps).toFixed(2)}
          </span>
          <span>Timebase Status</span>
          <span className="text-right text-[#171717]">{fs.audit.timebase_status}</span>
          <span>Held Frames</span>
          <span className="text-right text-[#171717]">{(fs.audit.held_frame_ratio * 100).toFixed(1)}% (max run {fs.audit.longest_held_run_frames})</span>
          <span>Bowler-ROI Blur</span>
          <span className="text-right text-[#171717]">{fs.audit.blur_grade} ({fs.audit.median_laplacian_var.toFixed(0)})</span>
          <span>Arm Rotation / Frame</span>
          <span className="text-right text-[#171717]">{fs.audit.est_arm_deg_per_frame.toFixed(1)}°</span>
        </div>
      </div>
    </div>
  );
};
