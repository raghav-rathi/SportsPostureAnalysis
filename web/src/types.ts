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

export type TelestratorTool = "cursor" | "line" | "plumb" | "angle" | "circle";

export interface TelestratorShape {
  id: string;
  frame: number;
  type: Exclude<TelestratorTool, "cursor">;
  points: Array<[number, number]>;
  label?: string;
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
