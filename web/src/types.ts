export type ObservabilityGrade = "high" | "moderate" | "low" | "unmeasurable" | "refused";

export interface MetricEstimate {
  key: string;
  label: string;
  unit: string;
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  interval_type: string;
  sigma_total: number | null;
  sigma_projective: number | null;
  sigma_temporal: number | null;
  sigma_epistemic: number | null;
  sigma_calibration: number | null;
  observability: ObservabilityGrade;
  suppressed: boolean;
  suppression_reason: string | null;
  definition_note: string;
  empirical_reference: string;
  literature_source: string;
}

export interface EventTimestamp {
  frame_float: number;
  frame_idx: number;
  source_frame_idx: number;
  timestamp_s: number;
  timing_uncertainty_ms: number;
  confidence: number;
  detection_method: string;
}

export interface DeliveryEvents {
  runup_start_frame: number;
  bfc: EventTimestamp;
  ffc: EventTimestamp;
  uah: EventTimestamp | null;
  br: EventTimestamp;
  ft_peak: EventTimestamp | null;
  delivery_stride_duration_ms: number;
  ffc_to_br_duration_ms: number;
  is_complete_delivery: boolean;
  event_warnings: string[];
}

export interface PrioritizedCue {
  category: string;
  title: string;
  evidence_metrics: string[];
  biomechanical_mechanism: string;
  actionable_recommendation: string;
}

export interface CorrectiveDrill {
  id: string;
  target_limiter_metric: string;
  title: string;
  cue_focus: string;
  setup_steps: string[];
  literature_basis: string;
  caution_note: string;
}

export interface CohortComparison {
  metric_key: string;
  label: string;
  value: number;
  ci_low: number;
  ci_high: number;
  unit: string;
  cohort_name: string;
  cohort_mean: number;
  cohort_sd: number;
  z_score: number;
  definition_match: string;
  source: string;
}

export interface WorkloadGuidance {
  age_band: string;
  max_balls_per_spell: number | null;
  max_balls_per_day: number | null;
  max_balls_per_week: number | null;
  current_bowls_per_week: number | null;
  status_note: string;
  governing_body_source: string;
}

export interface BiomechanicalFactSheet {
  schema_version: string;
  bowler_id: string;
  bowling_arm: "right" | "left";
  action_class: string;
  pose_backend: string;
  pipeline_mode?: string;
  interval_type: string;
  audit: {
    video_path: string;
    width: number;
    height: number;
    fps: number;
    container_fps: number | null;
    num_frames: number;
    raw_num_frames: number;
    duration_s: number;
    median_laplacian_var: number;
    blur_grade: string;
    blur_roi_method: string;
    fps_grade: string;
    est_arm_deg_per_frame: number;
    timebase_status: string;
    timebase_unreliable: boolean;
    held_frame_ratio: number;
    longest_held_run_frames: number;
    warnings: string[];
  };
  calibration: {
    method: string;
    azimuth_deg: number;
    elevation_deg: number;
    view_class: string;
    reprojection_error_px: number | null;
    is_metric_calibrated: boolean;
    is_intrinsics_calibrated: boolean;
    is_panning_camera: boolean;
    camera_position_world_m: [number, number, number];
    heading_sigma_deg: number;
    gravity_sigma_deg: number;
    gravity_source: string;
    calibration_notes: string[];
  };
  events: DeliveryEvents;
  metrics: Record<string, MetricEstimate>;
  cohort_comparisons: CohortComparison[];
  workload_guidance: WorkloadGuidance | null;
  strengths: PrioritizedCue[];
  limiters: PrioritizedCue[];
  inconclusive_observations: PrioritizedCue[];
  corrective_drills: CorrectiveDrill[];
  capture_upgrades: PrioritizedCue[];
  hard_guardrails: string[];
  // ── v5 additive (all optional so v2–v4 payloads still type-check) ──
  v5_metrics?: Record<string, MetricEstimate>;
  kinematic_sequence?: KinematicSequenceAnalysis | null;
  com_braking?: COMBrakingAnalysis | null;
  head_stability?: HeadStabilityAnalysis | null;
  lead_arm?: LeadArmAnalysis | null;
  crease_tunnel?: CreaseReleaseTunnelAnalysis | null;
  tempo_rhythm?: TempoRhythmAnalysis | null;
}

export interface TelemetryV2 {
  schema_version: string;
  joint_names: string[];
  num_frames: number;
  analysis_fps: number;
  proxy_fps: number;
  source_frame_indices: number[];
  frame_pts_ms: number[];
  time_rel_ffc_ms: number[];
  width: number;
  height: number;
  y_ground_offset_m: number;
  joints3d_world_m: number[][][];
  joints2d_px: number[][][];
  joint_uncertainties_m: number[][];
  joint_in_frame: boolean[][];
  kin_series: Record<string, number[]>;
  mcc_wireframe_2d: Record<string, [number, number]>;
  camera_position_world_m: [number, number, number];
  all_tracks: Array<{
    track_id: number;
    is_selected: boolean;
    n_frames: number;
    mid_frame: number;
    box_xyxy: [number, number, number, number];
    anchor_xy_norm: [number, number];
  }>;
  isb_anatomy: {
    scalars: Record<string, number>;
    metrics?: Record<string, MetricEstimate>;
    n_eff_pelvis: number;
    n_eff_thorax: number;
    n_eff_foot?: number;
    rho_assumed: number;
    front_foot_strike_mode: string;
  } | null;
  needs_gpu_refit_for_K: boolean;
  // ── v5 additive (telemetry_minor_version "2.1"; schema_version stays "2.0") ──
  telemetry_minor_version?: string;
  /** (T, 3) ground-zeroed 3D centre-of-mass path, metres. */
  com_world_m?: number[][] | null;
  /** (T, 2) projected COM pixels. */
  com_2d_px?: number[][] | null;
  trails?: TelemetryTrails | null;
  sequence_series?: SequenceSeries | null;
  camera_poses_per_frame?: any;
}

/** Precomputed multi-joint trails over [BFC − 0.15 s, BR + 0.18 s]. Arrays index-aligned with `frame_indices`. */
export interface TelemetryTrails {
  frame_indices: number[];
  wrist_speed_mps?: Array<number | null>;
  "2d_px": Partial<Record<TrailJoint, Array<[number, number] | null>>>;
  "3d_m": Partial<Record<TrailJoint, Array<[number, number, number] | null>>>;
}

export type TrailJoint =
  | "bowling_wrist"
  | "bowling_elbow"
  | "lead_wrist"
  | "head"
  | "com"
  | "front_ankle"
  | "rear_ankle";

/** Full-length (T) series aligned to `telemetry.time_rel_ffc_ms`; null where unmeasurable. */
export interface SequenceSeries {
  pelvis_omega_dps?: Array<number | null>;
  thorax_omega_dps?: Array<number | null>;
  arm_omega_dps?: Array<number | null>;
  wrist_whip_dps?: Array<number | null>;
  com_vx_mps?: Array<number | null>;
}

export interface BowlerSummary {
  id: string;
  name: string;
  bowling_arm: "right" | "left";
  age_band: string;
  created_at: string;
}

export interface SessionSummary {
  id: string;
  bowler_id: string;
  bowler_name: string;
  bowling_arm: "right" | "left";
  age_band: string;
  title: string;
  notes: string;
  delivery_count?: number;
  deliveries?: Array<{
    id: string;
    session_id: string;
    label: string;
    pipeline_mode: string;
    status: string;
    start_frame: number;
    end_frame: number | null;
  }>;
}

export interface CoachNote {
  id: string;
  delivery_id: string;
  frame_idx: number | null;
  timestamp_s: number | null;
  tag: string;
  body: string;
  created_at: string;
}

/** v5 telestration toolkit (v4 tools first; v5 adds arrow/freehand/ruler/text/spotlight). Non-drawing UI modes
 *  (zoom/pan) live in component state, not in this union. */
export type TelestratorTool =
  | "cursor"
  | "line"
  | "plumb"
  | "angle"
  | "circle"
  | "arrow"
  | "freehand"
  | "ruler"
  | "text"
  | "spotlight";

export interface TelestratorShape {
  id: string;
  frame: number;
  type: Exclude<TelestratorTool, "cursor">;
  /** Points in source-video pixel space (telemetry.width × telemetry.height). */
  points: Array<[number, number]>;
  label?: string;
  /** v5: draw on every frame instead of only `frame`. */
  pinned_all_frames?: boolean;
  /** v5: SMPL-24 joint indices the points are snapped to (angle tool → live 3D measured angle). */
  snap_joints?: number[];
}

export interface HealthResponse {
  status: string;
  version: string;
  mode: string;
  pose_engine: "remote_gpu_worker" | "local_nlf_model" | "cpu_recompute_only";
  local_nlf_configured: boolean;
  gpu_worker: {
    configured: boolean;
    reachable: boolean;
    status?: string;
    jit_warmed?: boolean;
    gpu?: { device_name?: string };
    error?: string;
  };
  mcc_landmark_keys: string[];
}

export interface DeliveryDetailResponse {
  delivery: {
    id: string;
    session_id: string;
    label: string;
    pipeline_mode: string;
    status: string;
    overrides: Record<string, any>;
    isb_summary: TelemetryV2["isb_anatomy"];
    pan_summary: {
      is_verified: boolean;
      median_residual_px: number;
      p90_residual_px: number;
      max_pan_deg: number;
      sigma_pan_deg: number;
      verification_reason: string;
    } | null;
    video_proxy_url: string;
  };
  fact_sheet: BiomechanicalFactSheet;
  telemetry: TelemetryV2;
  report_markdown: string;
  notes: CoachNote[];
  annotations: TelestratorShape[];
  mcc_landmarks_3d: Record<string, [number, number, number]>;
  story?: CoachingStory | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Studio v4 — CoachingStory contract (docs/studio_v4_ux_blueprint.md §3)
// ─────────────────────────────────────────────────────────────────────────────

export type CheckpointId = "front_leg_brace" | "trunk_whip" | "side_bend" | "shoulder_alignment" | "runup_stride";
export type BodyZone = "knee" | "trunk" | "spine" | "shoulders" | "approach";
export type CheckpointStatus = "strong" | "on_track" | "watch" | "focus" | "not_visible";
export type UnlockReason =
  | "angle"
  | "frame_rate"
  | "out_of_frame"
  | "incomplete_delivery"
  | "timing_unreliable"
  | "calibration";
export type MomentId = "back_foot_land" | "front_foot_plant" | "arm_horizontal" | "ball_release";
export type CaptureGrade = "great" | "partial" | "limited" | "unusable";
export type ViewClarity = "clear" | "fair" | "rough" | null;

export interface CaptureQuality {
  grade: CaptureGrade;
  label: string;
  summary: string;
  view: "side_on" | "rear_on" | "front_on" | "oblique" | "dual_fused" | "unknown";
  view_label: string;
  camera_azimuth_deg: number | null;
  fps: number;
  fps_label: string;
  measured_count: number;
  total_count: number;
  can_see: CheckpointId[];
  cannot_see: Array<{ checkpoint_id: CheckpointId; reason: UnlockReason; reason_plain: string }>;
  next_clip_tip: string | null;
}

export interface Takeaway {
  kind: "focus" | "watch" | "strength" | "capture" | "all_clear";
  title: string;
  plain_explanation: string;
  what_to_try: string;
  checkpoint_id: CheckpointId | null;
  moment_id: MomentId | null;
  frame_idx: number | null;
  linked_drill_id: string | null;
}

export interface StorySupportingMetric {
  metric_key: string;
  label_plain: string;
  value_text: string | null;
  range_text: string | null;
  suppressed: boolean;
}

export interface BodyCheckpoint {
  id: CheckpointId;
  body_zone: BodyZone;
  title: string;
  moment_id: MomentId | null;
  moment_label: string;
  frame_idx: number | null;
  status: CheckpointStatus;
  status_label: string;
  unlock_reason: UnlockReason | null;
  metric_key: string;
  value: number | null;
  ci_low: number | null;
  ci_high: number | null;
  unit: string;
  view_clarity: ViewClarity;
  value_text: string | null;
  range_text: string | null;
  reference: { text: string; source: string; definition_match: string } | null;
  plain_meaning: string;
  why_it_matters: string;
  how_to_unlock: string | null;
  highlight_joints: string[];
  angle_joints: [string, string, string] | null;
  supporting: StorySupportingMetric[];
}

export interface KeyMoment {
  id: MomentId;
  label: string;
  frame_idx: number;
  source_frame_idx: number;
  time_s: number;
  timing_uncertainty_ms: number;
  plain_tip: string;
  what_to_look_for: string;
  checkpoint_ids: CheckpointId[];
  thumbnail_url: string | null;
}

export interface HeadlineStat {
  metric_key: string;
  label: string;
  value_text: string;
  range_text: string;
  view_clarity: string | null;
}

export interface DrillCard {
  id: string;
  title: string;
  for_checkpoint_id: CheckpointId | null;
  why_plain: string;
  cue: string;
  steps: string[];
  caution: string;
  source: string;
  moment_id: MomentId | null;
  frame_idx: number | null;
}

export interface StoryNotice {
  id: string;
  title: string;
  body: string;
}

export interface StoryMetricRow {
  metric_key: string;
  group: CheckpointId | "other";
  label_plain: string;
  value_text: string | null;
  range_text: string | null;
  view_clarity: ViewClarity;
  status: CheckpointStatus;
  status_label: string;
  suppressed: boolean;
  suppression_plain?: string | null;
  plain_note: string;
  definition_note: string;
  literature_source: string;
}

export interface CoachingStory {
  schema_version: string;
  delivery_id: string;
  headline: string;
  subhead: string;
  is_complete_delivery: boolean;
  is_sample: boolean;
  sample_kind: "real" | "synthetic" | null;
  capture: CaptureQuality;
  top_takeaway: Takeaway;
  checkpoints: BodyCheckpoint[];
  key_moments: KeyMoment[];
  headline_stats: HeadlineStat[];
  drills: DrillCard[];
  notices: { icc: StoryNotice; medical: StoryNotice; extra: StoryNotice[] };
  detail_rows: StoryMetricRow[];
  action_style: { code: string; label: string; plain_summary: string } | null;
  // ── v5 additive ──
  v5_insights?: V5InsightCard[];
  v5_detail_rows?: StoryMetricRow[];
  /** e.g. "/api/deliveries/{id}/stroboscopic.jpg" (append ?w=&skeleton=&trails=). */
  stroboscopic_url?: string | null;
}

export interface DeliveryCard {
  id: string;
  session_id: string;
  bowler_id: string;
  bowler_name: string;
  label: string;
  created_at: string;
  thumbnail_url: string | null;
  kind: "upload" | "sample_real" | "sample_demo" | "fused_3d";
  headline: string;
  top_takeaway_title: string;
  capture_grade: CaptureGrade;
  view_label: string;
  checkpoint_statuses: Partial<Record<CheckpointId, CheckpointStatus>>;
}

export interface OverviewBowler {
  id: string;
  name: string;
  bowling_arm: "right" | "left";
  age_band: string;
  delivery_count: number;
  last_delivery_at: string | null;
}

export interface OverviewSample {
  delivery_id: string;
  title: string;
  subtitle: string;
  kind: "real" | "synthetic";
  thumbnail_url: string | null;
  capture_grade: CaptureGrade;
}

export interface OverviewResponse {
  is_first_run: boolean;
  health: {
    pose_engine: string;
    engine_label: string;
    can_analyse_new_video: boolean;
    reason_plain: string | null;
  };
  bowlers: OverviewBowler[];
  samples: OverviewSample[];
  recent_deliveries: DeliveryCard[];
  // ── v5 additive (Command Center Home) ──
  workload_snapshot?: WorkloadSnapshot | null;
  spell_snapshot?: SpellSnapshot | null;
}

export interface CompareSide {
  status: CheckpointStatus;
  status_label?: string;
  value_text: string | null;
  range_text: string | null;
}

export interface CompareStory {
  a: DeliveryCard;
  b: DeliveryCard;
  sync_moment: "front_foot_plant";
  sync_frames: { a: number | null; b: number | null };
  comparable: boolean;
  summary: string;
  caveat_plain: string | null;
  rows: Array<{
    checkpoint_id: CheckpointId;
    title: string;
    a: CompareSide;
    b: CompareSide;
    difference: "clearly_higher" | "clearly_lower" | "within_noise" | "not_comparable";
    difference_plain: string;
  }>;
  // ── v5 additive: noise-gated rows across v5 metrics (B read relative to A) ──
  extended_rows?: CompareExtendedRow[];
}

export interface ProgressResponse {
  bowler: OverviewBowler | BowlerSummary;
  deliveries: Array<{ id: string; label: string; created_at: string; thumbnail_url: string | null }>;
  series: Array<{
    checkpoint_id: CheckpointId;
    title: string;
    points: Array<{
      delivery_id: string;
      status: CheckpointStatus;
      value: number | null;
      ci_low: number | null;
      ci_high: number | null;
      value_text: string | null;
      range_text: string | null;
    }>;
  }>;
  workload: WorkloadGuidance | null;
  summary_plain: string;
}

export interface JobStatus {
  status: "queued" | "running" | "done" | "failed";
  stage: "finding_bowler" | "spotting_moments" | "measuring" | "writing_story" | string;
  progress: number;
  message_plain: string;
  delivery_id: string | null;
  error_plain: string | null;
}

export interface ScanCandidate {
  candidate_id: number;
  start_frame: number;
  end_frame: number;
  peak_frame: number;
  start_s: number;
  end_s: number;
  motion_score: number;
  thumbnail_b64?: string | null;
}

// ─────────────────────────────────────────────────────────────────────────────
// Studio v5 — locked contracts (docs/research_app_landscape_v5.md §10 +
// docs/research_biomechanics_v5.md §4–5). Every number is a MetricEstimate
// (value, ci_low, ci_high, observability) or suppressed with a reason.
// ─────────────────────────────────────────────────────────────────────────────

// ── Fact-sheet sub-analyses (Modules 1–6) ───────────────────────────────────
export type SequenceVerdict = "optimal_proximal_to_distal" | "simultaneous_rotational" | "out_of_order" | "unmeasurable";
export type SequenceSegment = "pelvis" | "thorax" | "bowling_arm" | "wrist_whip";

export interface SequenceSegmentPeak {
  segment: SequenceSegment;
  label_plain: string;
  peak_dps: MetricEstimate;
  peak_time_rel_ffc_ms: number | null;
  peak_time_rel_br_ms: number | null;
  timing_sigma_ms: number | null;
}

export interface KinematicSequenceAnalysis {
  verdict: SequenceVerdict;
  verdict_label_plain: string;
  summary_plain: string;
  segments: SequenceSegmentPeak[];
  pelvis_to_thorax_lag_ms: MetricEstimate;
  thorax_to_arm_lag_ms: MetricEstimate;
  arm_to_wrist_lag_ms: MetricEstimate;
  whip_gain_arm_over_thorax: MetricEstimate;
  pelvis_braking_decel_pct: MetricEstimate;
}

export interface COMBrakingAnalysis {
  model_reference: string;
  summary_plain: string;
  com_ffc_speed_mps: MetricEstimate;
  com_br_speed_mps: MetricEstimate;
  com_braking_delta_vx_mps: MetricEstimate;
  com_braking_efficiency_pct: MetricEstimate;
  com_vertical_sink_pct_stature: MetricEstimate;
  front_leg_plant_angle_ffc_deg: MetricEstimate;
}

export interface HeadStabilityAnalysis {
  summary_plain: string;
  head_coronal_angle_br_deg: MetricEstimate;
  head_fall_away_ffc_to_br_deg: MetricEstimate;
  head_lateral_offset_from_ankle_br_pct_stature: MetricEstimate;
  head_lateral_offset_from_ankle_br_cm: MetricEstimate;
  head_vertical_drop_ffc_to_br_pct_stature: MetricEstimate;
}

export interface LeadArmAnalysis {
  summary_plain: string;
  lead_arm_gather_peak_elev_deg: MetricEstimate;
  lead_arm_early_drop_pre_ffc_deg: MetricEstimate;
  lead_arm_pulldown_peak_vel_dps: MetricEstimate;
  lead_elbow_ribcage_flare_br_pct_stature: MetricEstimate;
}

export type CorridorVerdict = "train_tracks_open" | "tightrope_inline" | "crossed_over" | "unmeasurable";

export interface CreaseReleaseTunnelAnalysis {
  corridor_verdict: CorridorVerdict;
  corridor_label_plain: string;
  summary_plain: string;
  foot_plant_corridor_offset_cm: MetricEstimate;
  stride_alignment_angle_deg: MetricEstimate;
  front_heel_crease_margin_cm: MetricEstimate;
  release_height_pct_stature: MetricEstimate;
  release_lateral_from_ankle_m: MetricEstimate;
  release_lateral_from_stump_m: MetricEstimate;
  release_extension_past_crease_m: MetricEstimate;
}

export type TempoPhaseId = "stride_bfc_ffc" | "early_braking_ffc_uah" | "whip_uah_br" | "braking_ffc_br" | "follow_through";

export interface TempoPhaseSegment {
  id: TempoPhaseId;
  label_plain: string;
  start_rel_ffc_ms: number;
  end_rel_ffc_ms: number;
  duration_ms: number;
  uncertainty_ms: number;
  reference_range_ms: [number, number] | null;
}

export interface TempoRhythmAnalysis {
  summary_plain: string;
  tempo_stride_duration_ms: MetricEstimate;
  tempo_braking_to_br_ms: MetricEstimate;
  tempo_uah_to_br_whip_ms: MetricEstimate;
  tempo_gather_to_whip_ratio: MetricEstimate;
  waterfall_phases: TempoPhaseSegment[];
}

// ── Story additions ─────────────────────────────────────────────────────────
export type V5InsightId = "kinematic_sequence" | "com_braking" | "head_stability" | "lead_arm" | "crease_tunnel" | "tempo_rhythm";

export interface V5InsightCard {
  id: V5InsightId;
  title: string;
  badge_label: string;
  status: CheckpointStatus;
  headline_value_text: string | null;
  headline_range_text: string | null;
  plain_summary: string;
  unlock_tip: string | null;
  moment_id: MomentId | null;
  frame_idx: number | null;
}

// ── Compare additions ───────────────────────────────────────────────────────
export type CompareDifference = "clearly_higher" | "clearly_lower" | "within_noise" | "not_comparable";

/** Noise-gated v5 row: difference is "within_noise" unless |B−A| > 1.96·√(σa²+σb²). */
export interface CompareExtendedRow {
  metric_key: string;
  group: V5InsightId | CheckpointId | "other";
  title: string;
  unit: string;
  a: CompareSide & { value?: number | null; ci_low?: number | null; ci_high?: number | null };
  b: CompareSide & { value?: number | null; ci_low?: number | null; ci_high?: number | null };
  delta_value?: number | null;
  delta_text?: string | null;
  difference: CompareDifference;
  difference_plain: string;
}

// ── Overview additions (Command Center Home) ────────────────────────────────
export type AcwrZone = "low_chronic_base" | "underload" | "sweet_spot" | "caution_elevated" | "high_spike_alert" | "no_data";

export interface WorkloadSnapshot {
  bowler_id: string;
  balls_7d: number;
  days_bowled_7d: number;
  max_days_per_week?: number | null;
  acwr_zone: AcwrZone;
  acwr_zone_label_plain: string;
  acwr_value?: number | null;
  daily_loop_sentence: string;
}

export interface SpellSnapshot {
  bowler_id?: string;
  latest_session_id: string | null;
  completed_count: number;
  /** Measurement-debiased radial spread of release points, cm (null when < 2 comparable balls). */
  release_spread_cm: number | null;
  release_spread_ci_cm?: [number, number] | null;
  tunnel_mode?: ReleaseTunnelMode | null;
}

// ── Workload & wellness (Module 9) ──────────────────────────────────────────
export type WorkloadSessionType = "net" | "match" | "drill" | string;

export interface WorkloadLog {
  id: string;
  bowler_id: string;
  log_date: string; // YYYY-MM-DD
  deliveries_count: number;
  session_type: WorkloadSessionType;
  notes: string;
  created_at: string;
  rpe?: number | null;
  max_spell_balls?: number | null;
  soreness_lumbar?: number | null;
  fatigue_level?: number | null;
  pain_when_bowling?: boolean | null;
}

export interface AcwrSummary {
  acute_7d: number;
  chronic_coupled_wk: number | null;
  chronic_uncoupled_wk: number | null;
  acwr_coupled: number | null;
  acwr_uncoupled: number | null;
  acwr_ewma: number | null;
  zone: AcwrZone;
  zone_label_plain: string;
  caveat_plain?: string;
  sources?: string[];
}

export interface WorkloadDailyPoint {
  date: string; // YYYY-MM-DD
  balls: number;
  acute_7d?: number | null;
  chronic_coupled_wk?: number | null;
  acwr_coupled?: number | null;
  acwr_uncoupled?: number | null;
  acwr_ewma?: number | null;
  rpe?: number | null;
  soreness_lumbar?: number | null;
  session_types?: string[];
}

export interface DirectiveCheck {
  id: string;
  label_plain: string;
  limit_text: string;
  current_text: string;
  ok: boolean | null; // null = not enough data
  source: string;
}

export type WorkloadAlertKind =
  | "weekly_cap_exceeded"
  | "daily_cap_exceeded"
  | "spell_cap_exceeded"
  | "frequency_exceeded"
  | "consecutive_days_alert"
  | "lumbar_stiffness_logged"
  | "high_spike_alert"
  | "low_chronic_base"
  | string;

export interface WorkloadAlert {
  kind: WorkloadAlertKind;
  severity: "info" | "caution" | "alert";
  title: string;
  body_plain: string;
  source?: string | null;
}

export interface WorkloadResponse {
  bowler: BowlerSummary | OverviewBowler;
  logs: WorkloadLog[];
  deliveries_last_7d: number;
  deliveries_last_28d: number;
  guidance: WorkloadGuidance | null;
  // v5
  acwr_summary?: AcwrSummary | null;
  daily_series_28d?: WorkloadDailyPoint[];
  directive_checks?: DirectiveCheck[];
  alerts?: WorkloadAlert[];
  daily_loop_summary?: string | null;
  medical_disclaimer?: string | null;
}

export interface WorkloadLogRequest {
  log_date: string;
  deliveries_count: number;
  session_type: WorkloadSessionType;
  notes: string;
  rpe?: number | null;
  max_spell_balls?: number | null;
  soreness_lumbar?: number | null;
  fatigue_level?: number | null;
  pain_when_bowling?: boolean | null;
}

// ── Spell analytics (Module 10) ─────────────────────────────────────────────
export type ReleaseTunnelMode = "world_stump_m" | "body_corridor_m";

export interface Ellipse95 {
  semi_major_m: number;
  semi_minor_m: number;
  angle_deg: number;
}

export interface ReleaseTunnelPoint {
  delivery_id: string;
  label: string;
  order: number;
  /** Lateral coordinate (m): from middle stump (world_stump_m) or from planted front ankle (body_corridor_m), + = bowler's arm side. */
  lateral_m: number;
  height_m: number;
  sigma_lateral_m?: number | null;
  sigma_height_m?: number | null;
  ellipse_95?: Ellipse95 | null;
  frame_idx?: number | null;
  observability?: ObservabilityGrade;
}

export interface ReleaseTunnel {
  mode: ReleaseTunnelMode;
  mode_label_plain?: string;
  points: ReleaseTunnelPoint[];
  centroid: { lateral_m: number; height_m: number } | null;
  raw_radial_sd_m: number | null;
  debiased_radial_sd_m: number | null;
  ellipse_95: Ellipse95 | null;
  stump_dims_m: { height: number; outer_width: number; center_half_spacing: number };
  excluded?: Array<{ delivery_id: string; label: string; reason_plain: string }>;
}

export type RepeatabilityGrade = "tight" | "moderate" | "variable" | "measurement_limited";

export interface RepeatabilityRow {
  metric_key: string;
  label_plain: string;
  unit: string;
  n_balls: number;
  mean: number | null;
  raw_sd: number | null;
  mean_measurement_sigma: number | null;
  debiased_sd: number | null;
  debiased_cv_pct: number | null;
  grade: RepeatabilityGrade;
}

export interface FatigueDriftRow {
  metric_key: string;
  label_plain: string;
  unit: string;
  n_balls: number;
  slope_per_ball: number | null;
  slope_se: number | null;
  spell_shift: number | null;
  status: "verified_drift" | "within_measurement_noise" | "insufficient_data";
  status_plain: string;
  points: Array<{ delivery_id: string; order: number; value: number | null; ci_low: number | null; ci_high: number | null }>;
}

export interface SpellAnalyticsResponse {
  bowler_id: string;
  session_id: string | null;
  completed_count: number;
  release_tunnel: ReleaseTunnel | null;
  repeatability_table: RepeatabilityRow[];
  fatigue_drift: FatigueDriftRow[];
}

// ── Reference archetypes (Module 8) ─────────────────────────────────────────
export type ArchetypeId = "side_on_express" | "semi_open_all_round" | "front_on_power" | "high_arm_bounce";

export interface Archetype {
  id: ArchetypeId | string;
  title: string;
  action_class: string;
  literature_source: string;
  /** Glazier (2010) non-template disclaimer — must be shown wherever the ghost is drawn. */
  disclaimer: string;
  stature_m: number;
  fps?: number;
  bowling_arm?: "right" | "left";
  time_rel_ffc_ms: number[];
  joints3d_world_m: number[][][];
  com_world_m?: number[][];
  kin_series?: Record<string, number[]>;
  sequence_series?: SequenceSeries;
  /** Literature envelope per metric key: plausible published range (never a target score). */
  metric_envelopes?: Record<string, { low: number; high: number; unit?: string; source?: string }>;
}

export interface ArchetypesResponse {
  archetypes: Archetype[];
}

// ── Exports ─────────────────────────────────────────────────────────────────
export type ExportFormat = "csv" | "json";

// ── Delivery detail (v5 view of the same endpoint) ──────────────────────────
/** Convenience alias: `GET /api/deliveries/{id}` with v5 optional fields filled when the backend supports them. */
export type DeliveryDetailV5 = DeliveryDetailResponse;
