"""Strict Pydantic v2 data contracts for the Cricket Fast-Bowling Biomechanics Engine.

Enforces uncertainty quantification, timebase honesty, observability gating, and
definition-matched literature reporting structurally across every stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field


class BowlingArm(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class BowlerActionClass(str, Enum):
    SIDE_ON = "side_on"
    SEMI_OPEN = "semi_open"
    SEMI_SIDE_ON = "semi_side_on"  # retained for backward enum parsing compatibility
    SEMI_FRONT_ON = "semi_front_on"
    FRONT_ON = "front_on"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class CameraViewClass(str, Enum):
    SIDE_ON_RIGHT = "side_on_right"
    SIDE_ON_LEFT = "side_on_left"
    REAR_BEHIND_BOWLER = "rear_behind_bowler"
    FRONT_BEHIND_BATTER = "front_behind_batter"
    OPEN_SIDE_45 = "open_side_45"
    BROADCAST_ELEVATED = "broadcast_elevated"
    DUAL_VIEW_FUSED = "dual_view_fused"
    UNKNOWN = "unknown"


class ObservabilityGrade(str, Enum):
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    UNMEASURABLE = "unmeasurable"
    REFUSED = "refused"


IntervalType = Literal[
    "repeatability_interval",
    "analytical_uncertainty",
    "conformal_calibrated_ci",
]


class VideoAudit(BaseModel):
    video_path: str
    width: int
    height: int
    fps: float = Field(description="Effective unique-frame rate used for kinematics.")
    container_fps: Optional[float] = Field(default=None, description="Raw container frame rate before pulldown dedup.")
    num_frames: int = Field(description="Number of processed frames (after pulldown dedup if applied).")
    raw_num_frames: Optional[int] = Field(default=None, description="Raw frame count in source slice before dedup.")
    duration_s: float
    median_laplacian_var: float = Field(
        description="Median sharpness proxy on bowler/central ROI (higher = crisper)."
    )
    blur_grade: str = Field(description="crisp | moderate_blur | severe_blur")
    blur_roi_method: str = Field(
        default="center_crop",
        description="bowler_roi | center_crop | full_frame",
    )
    fps_grade: str = Field(description="high_speed_240 | slow_mo_120 | medium_60 | standard_30 | low_25")
    est_arm_deg_per_frame: float = Field(
        description="Estimated bowling arm sweep (deg) between unique frames at ~2300 deg/s."
    )
    est_ball_m_per_frame: float = Field(
        description="Estimated ball travel (m) between unique frames at 140 km/h (38.9 m/s)."
    )
    # Timebase integrity diagnostics (EXP-007 B2)
    timebase_status: str = Field(
        default="verified_native",
        description="verified_native | pulldown_deduped | retimed_unverified_override | edited_or_frozen",
    )
    timebase_unreliable: bool = Field(
        default=False,
        description="True when irregular slow-motion repeats, freeze frames, or unverified retiming invalidate timing/velocity.",
    )
    held_frame_ratio: float = 0.0
    longest_held_run_frames: int = 0
    regular_repeat_cadence: Optional[int] = None
    scene_cut_indices: List[int] = Field(default_factory=list)
    source_frame_indices: List[int] = Field(
        default_factory=list,
        description="0-based absolute frame indices in source video corresponding to processed frames 0..T-1.",
    )
    warnings: List[str] = Field(default_factory=list)


class PitchCalibration(BaseModel):
    method: str = Field(
        description="mcc_law7_pnp | two_wicket_geometry | motion_gravity_self_cal | fallback_uncalibrated"
    )
    camera_matrix_K: List[List[float]] = Field(description="3x3 intrinsic camera matrix (larger-side FOV convention).")
    rotation_matrix_R: List[List[float]] = Field(
        description="3x3 rotation matrix transforming Pitch World (+X down pitch, +Y up, +Z off-to-leg) to Camera frame."
    )
    translation_t: List[float] = Field(description="3-vector translation in meters.")
    camera_position_world_m: List[float] = Field(
        description="Camera optical center [X, Y, Z] in Pitch World meters."
    )
    azimuth_deg: float = Field(
        description="Camera horizontal azimuth relative to +X delivery axis (0=behind bowler, 90=side-on, 180=behind batter)."
    )
    elevation_deg: float = Field(
        description="Camera elevation above pitch plane in degrees (negative = looking upward from ground level)."
    )
    px_per_meter_at_crease: Optional[float] = None
    reprojection_error_px: Optional[float] = None
    view_class: CameraViewClass = CameraViewClass.UNKNOWN
    is_metric_calibrated: bool = False
    is_intrinsics_calibrated: bool = False
    is_panning_camera: bool = False
    heading_sigma_deg: float = Field(
        default=7.5,
        description="Estimated 1-sigma heading (+X pitch axis) uncertainty in degrees.",
    )
    gravity_sigma_deg: float = Field(
        default=6.0,
        description="Estimated 1-sigma gravity (+Y vertical) uncertainty in degrees.",
    )
    gravity_source: str = Field(
        default="uncalibrated_torso",
        description="mcc_pnp | two_wickets | standing_bg_bodies | upright_bowler_torso | fallback",
    )
    camera_motion_metrics: Dict[str, float] = Field(default_factory=dict)
    calibration_notes: List[str] = Field(default_factory=list)


class EventTimestamp(BaseModel):
    frame_float: float = Field(description="Sub-frame index (0-based in processed unique-frame timeline).")
    frame_idx: int = Field(description="Nearest integer frame index in processed timeline.")
    source_frame_idx: Optional[int] = Field(
        default=None,
        description="Nearest 0-based absolute frame index in the raw source video.",
    )
    timestamp_s: float
    timing_uncertainty_ms: float
    confidence: float = Field(ge=0.0, le=1.0)
    detection_method: str


class DeliveryEvents(BaseModel):
    runup_start_frame: int = 0
    bfc: EventTimestamp = Field(description="Back Foot Contact")
    ffc: EventTimestamp = Field(description="Front Foot Contact")
    uah: Optional[EventTimestamp] = Field(default=None, description="Upper Arm Horizontal")
    br: EventTimestamp = Field(description="Ball Release")
    ft_peak: Optional[EventTimestamp] = Field(default=None, description="Follow-through deceleration peak")
    delivery_stride_duration_ms: float
    ffc_to_br_duration_ms: float
    is_complete_delivery: bool = True
    event_warnings: List[str] = Field(default_factory=list)


class MetricEstimate(BaseModel):
    key: str
    label: str
    unit: str
    value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    interval_type: IntervalType = "analytical_uncertainty"
    sigma_total: Optional[float] = None
    sigma_projective: Optional[float] = None
    sigma_temporal: Optional[float] = None
    sigma_epistemic: Optional[float] = None
    sigma_calibration: Optional[float] = None
    observability: ObservabilityGrade
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    definition_note: str = Field(
        default="",
        description="Exact geometric definition computed by this engine and how it relates to cited studies.",
    )
    empirical_reference: str = Field(
        description="Primary-source cohort norm or biomechanical context."
    )
    literature_source: str


class CohortComparison(BaseModel):
    metric_key: str
    label: str
    value: float
    ci_low: float
    ci_high: float
    unit: str
    cohort_name: str
    cohort_mean: float
    cohort_sd: float
    z_score: float
    percentile_approx: float
    definition_match: Literal["exact", "proxy_global_frame", "proxy_trunk_vector"]
    definition_note: str
    source: str


class WorkloadGuidance(BaseModel):
    age_band: str
    governing_body_source: str
    max_balls_per_spell: Optional[int] = None
    max_balls_per_day: Optional[int] = None
    max_balls_per_week: Optional[int] = None
    current_bowls_per_week: Optional[int] = None
    status_note: str


class PillarScore(BaseModel):
    name: str
    score: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    ci_half_width: Optional[float] = None
    status_label: str
    contributing_metrics: List[str] = Field(default_factory=list)
    missing_or_suppressed_metrics: List[str] = Field(default_factory=list)
    summary: str


class QuadrantPlacement(BaseModel):
    kinetic_efficiency_x: Optional[float] = None
    kinetic_efficiency_ci: Optional[float] = None
    durability_safety_y: Optional[float] = None
    durability_safety_ci: Optional[float] = None
    quadrant_label: str
    interpretation: str


class PrioritizedCue(BaseModel):
    category: str = Field(description="strength | limiter | capture_upgrade | inconclusive")
    title: str
    evidence_metrics: List[str]
    biomechanical_mechanism: str
    actionable_recommendation: str


class CorrectiveDrill(BaseModel):
    """Evidence-grounded drill prescription mapped strictly from verified non-straddling limiters."""

    id: str
    target_limiter_metric: str
    title: str
    cue_focus: str
    setup_steps: List[str] = Field(default_factory=list)
    literature_basis: str
    caution_note: str


class WorkloadContext(BaseModel):
    age_years: Optional[int] = None
    sex: str = "male"
    bowls_per_week: Optional[int] = None
    recent_growth_spurt: Optional[bool] = None
    known_pace_kph: Optional[float] = None


CheckpointId = Literal[
    "front_leg_brace",
    "trunk_whip",
    "side_bend",
    "shoulder_alignment",
    "runup_stride",
]
BodyZone = Literal["knee", "trunk", "spine", "shoulders", "approach"]
CheckpointStatus = Literal["strong", "on_track", "watch", "focus", "not_visible"]
CheckpointStatusLabel = Literal[
    "Strong",
    "On Track",
    "Borderline",
    "Needs Focus",
    "Needs Better Angle",
    "Not Visible",
]
UnlockReason = Literal[
    "angle",
    "frame_rate",
    "out_of_frame",
    "incomplete_delivery",
    "timing_unreliable",
    "calibration",
]
MomentId = Literal[
    "back_foot_land",
    "front_foot_plant",
    "arm_horizontal",
    "ball_release",
]
CaptureGrade = Literal["great", "partial", "limited", "unusable"]
CaptureViewKind = Literal["side_on", "rear_on", "front_on", "oblique", "dual_fused", "unknown"]
ViewClarity = Literal["clear", "fair", "rough"]


class CannotSeeItem(BaseModel):
    checkpoint_id: CheckpointId
    reason: UnlockReason
    reason_plain: str


class CaptureQuality(BaseModel):
    grade: CaptureGrade
    label: str
    summary: str
    view: CaptureViewKind
    view_label: str
    camera_azimuth_deg: Optional[float] = None
    fps: float
    fps_label: str
    measured_count: int
    total_count: int = 22
    can_see: List[CheckpointId] = Field(default_factory=list)
    cannot_see: List[CannotSeeItem] = Field(default_factory=list)
    next_clip_tip: Optional[str] = None


class StoryTakeaway(BaseModel):
    kind: Literal["focus", "watch", "strength", "capture", "all_clear"]
    title: str
    plain_explanation: str
    what_to_try: str
    checkpoint_id: Optional[CheckpointId] = None
    moment_id: Optional[MomentId] = None
    frame_idx: Optional[int] = None
    linked_drill_id: Optional[str] = None


class CheckpointReference(BaseModel):
    text: str
    source: str
    definition_match: str


class SupportingMetricItem(BaseModel):
    metric_key: str
    label_plain: str
    value_text: Optional[str] = None
    range_text: Optional[str] = None
    suppressed: bool = False


class BodyCheckpoint(BaseModel):
    id: CheckpointId
    body_zone: BodyZone
    title: str
    moment_id: Optional[MomentId] = None
    moment_label: str
    frame_idx: Optional[int] = None
    status: CheckpointStatus
    status_label: CheckpointStatusLabel
    unlock_reason: Optional[UnlockReason] = None
    metric_key: str
    value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None
    unit: str
    view_clarity: Optional[ViewClarity] = None
    value_text: Optional[str] = None
    range_text: Optional[str] = None
    reference: Optional[CheckpointReference] = None
    plain_meaning: str
    why_it_matters: str
    how_to_unlock: Optional[str] = None
    highlight_joints: List[str] = Field(default_factory=list)
    angle_joints: Optional[Tuple[str, str, str]] = None
    supporting: List[SupportingMetricItem] = Field(default_factory=list)


class KeyMoment(BaseModel):
    id: MomentId
    label: str
    frame_idx: int
    source_frame_idx: int
    time_s: float
    timing_uncertainty_ms: float
    plain_tip: str
    what_to_look_for: str
    checkpoint_ids: List[CheckpointId] = Field(default_factory=list)
    thumbnail_url: Optional[str] = None


class HeadlineStat(BaseModel):
    metric_key: str
    label: str
    value_text: str
    range_text: str
    view_clarity: ViewClarity


class DrillCard(BaseModel):
    id: str
    title: str
    for_checkpoint_id: Optional[CheckpointId] = None
    why_plain: str
    cue: str
    steps: List[str] = Field(default_factory=list)
    caution: str
    source: str
    moment_id: Optional[MomentId] = None
    frame_idx: Optional[int] = None


class StoryMetricRow(BaseModel):
    metric_key: str
    group: Literal[
        "front_leg_brace",
        "trunk_whip",
        "side_bend",
        "shoulder_alignment",
        "runup_stride",
        "kinematic_sequence",
        "com_braking",
        "head_stability",
        "lead_arm",
        "crease_tunnel",
        "tempo_rhythm",
        "other",
    ]
    label_plain: str
    value_text: Optional[str] = None
    range_text: Optional[str] = None
    view_clarity: Optional[ViewClarity] = None
    status: CheckpointStatus
    status_label: CheckpointStatusLabel
    suppressed: bool
    suppression_plain: Optional[str] = None
    plain_note: str
    definition_note: str
    literature_source: str


SequenceVerdict = Literal[
    "optimal_proximal_to_distal",
    "simultaneous_rotational",
    "out_of_order",
    "unmeasurable",
]


class SequenceSegmentPeak(BaseModel):
    segment: Literal["pelvis", "thorax", "bowling_arm", "wrist_whip"]
    label_plain: str
    peak_dps: MetricEstimate
    peak_time_rel_ffc_ms: Optional[float] = None
    peak_time_rel_br_ms: Optional[float] = None
    timing_sigma_ms: Optional[float] = None


class KinematicSequenceAnalysis(BaseModel):
    verdict: SequenceVerdict
    verdict_label_plain: str
    summary_plain: str
    segments: List[SequenceSegmentPeak] = Field(default_factory=list)
    pelvis_to_thorax_lag_ms: MetricEstimate
    thorax_to_arm_lag_ms: MetricEstimate
    arm_to_wrist_lag_ms: MetricEstimate
    whip_gain_arm_over_thorax: MetricEstimate
    pelvis_braking_decel_pct: MetricEstimate


class COMBrakingAnalysis(BaseModel):
    model_reference: str = "de Leva (1996) 14-segment anthropometric COM (correlated-depth covariance)"
    summary_plain: str
    com_ffc_speed_mps: MetricEstimate
    com_br_speed_mps: MetricEstimate
    com_braking_delta_vx_mps: MetricEstimate
    com_braking_efficiency_pct: MetricEstimate
    com_vertical_sink_pct_stature: MetricEstimate
    front_leg_plant_angle_ffc_deg: MetricEstimate


class HeadStabilityAnalysis(BaseModel):
    summary_plain: str
    head_coronal_angle_br_deg: MetricEstimate
    head_fall_away_ffc_to_br_deg: MetricEstimate
    head_lateral_offset_from_ankle_br_pct_stature: MetricEstimate
    head_lateral_offset_from_ankle_br_cm: MetricEstimate
    head_vertical_drop_ffc_to_br_pct_stature: MetricEstimate


class LeadArmAnalysis(BaseModel):
    summary_plain: str
    lead_arm_gather_peak_elev_deg: MetricEstimate
    lead_arm_early_drop_pre_ffc_deg: MetricEstimate
    lead_arm_pulldown_peak_vel_dps: MetricEstimate
    lead_elbow_ribcage_flare_br_pct_stature: MetricEstimate


class CreaseReleaseTunnelAnalysis(BaseModel):
    corridor_verdict: Literal["train_tracks_open", "tightrope_inline", "crossed_over", "unmeasurable"]
    corridor_label_plain: str
    summary_plain: str
    foot_plant_corridor_offset_cm: MetricEstimate
    stride_alignment_angle_deg: MetricEstimate
    front_heel_crease_margin_cm: MetricEstimate
    release_height_pct_stature: MetricEstimate
    release_lateral_from_ankle_m: MetricEstimate
    release_lateral_from_stump_m: MetricEstimate
    release_extension_past_crease_m: MetricEstimate


class TempoPhaseSegment(BaseModel):
    id: Literal["stride_bfc_ffc", "early_braking_ffc_uah", "whip_uah_br", "braking_ffc_br", "follow_through"]
    label_plain: str
    start_rel_ffc_ms: float
    end_rel_ffc_ms: float
    duration_ms: float
    uncertainty_ms: float
    reference_range_ms: Optional[Tuple[float, float]] = None


class TempoRhythmAnalysis(BaseModel):
    summary_plain: str
    tempo_stride_duration_ms: MetricEstimate
    tempo_braking_to_br_ms: MetricEstimate
    tempo_uah_to_br_whip_ms: MetricEstimate
    tempo_gather_to_whip_ratio: MetricEstimate
    waterfall_phases: List[TempoPhaseSegment] = Field(default_factory=list)


class V5InsightCard(BaseModel):
    id: Literal[
        "kinematic_sequence",
        "com_braking",
        "head_stability",
        "lead_arm",
        "crease_tunnel",
        "tempo_rhythm",
    ]
    title: str
    badge_label: str
    status: CheckpointStatus
    headline_value_text: Optional[str] = None
    headline_range_text: Optional[str] = None
    plain_summary: str
    unlock_tip: Optional[str] = None
    moment_id: Optional[MomentId] = None
    frame_idx: Optional[int] = None


class ActionStyleSummary(BaseModel):
    code: str
    label: str
    plain_summary: str


class StoryNotice(BaseModel):
    id: str
    title: str
    body: str


class StoryNotices(BaseModel):
    icc: StoryNotice
    medical: StoryNotice
    extra: List[StoryNotice] = Field(default_factory=list)


class CoachingStory(BaseModel):
    """Deterministic, jargon-free consumer coaching layer derived strictly from verified FactSheet intervals."""

    schema_version: Literal["coaching_story/1"] = "coaching_story/1"
    delivery_id: str = ""
    headline: str
    subhead: str
    is_complete_delivery: bool
    is_sample: bool = False
    sample_kind: Optional[Literal["real", "synthetic"]] = None
    action_style: Optional[ActionStyleSummary] = None
    capture: CaptureQuality
    top_takeaway: StoryTakeaway
    checkpoints: List[BodyCheckpoint] = Field(default_factory=list)
    key_moments: List[KeyMoment] = Field(default_factory=list)
    headline_stats: List[HeadlineStat] = Field(default_factory=list)
    drills: List[DrillCard] = Field(default_factory=list)
    detail_rows: List[StoryMetricRow] = Field(default_factory=list)
    v5_insights: List[V5InsightCard] = Field(default_factory=list)
    v5_detail_rows: List[StoryMetricRow] = Field(default_factory=list)
    stroboscopic_url: Optional[str] = None
    notices: StoryNotices


class DeliveryCardSummary(BaseModel):
    id: str
    session_id: str
    bowler_id: str
    bowler_name: str
    label: str
    created_at: str
    thumbnail_url: Optional[str] = None
    kind: Literal["upload", "sample_real", "sample_demo", "fused_3d"]
    headline: str
    top_takeaway_title: str
    capture_grade: CaptureGrade
    view_label: str
    checkpoint_statuses: Dict[CheckpointId, CheckpointStatus]


class CompareSideCell(BaseModel):
    status: CheckpointStatus
    status_label: CheckpointStatusLabel
    value_text: Optional[str] = None
    range_text: Optional[str] = None
    value: Optional[float] = None
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class CompareCheckpointRow(BaseModel):
    checkpoint_id: CheckpointId
    title: str
    a: CompareSideCell
    b: CompareSideCell
    difference: Literal["clearly_higher", "clearly_lower", "within_noise", "not_comparable"]
    difference_plain: str


class CompareExtendedRow(BaseModel):
    metric_key: str
    group: str
    title: str
    a: CompareSideCell
    b: CompareSideCell
    delta_value: Optional[float] = None
    delta_ci_half: Optional[float] = None
    delta_text: Optional[str] = None
    unit: str = ""
    difference: Literal["clearly_higher", "clearly_lower", "within_noise", "not_comparable"]
    difference_plain: str


class CompareStory(BaseModel):
    a: DeliveryCardSummary
    b: DeliveryCardSummary
    sync_moment: Literal["front_foot_plant"] = "front_foot_plant"
    sync_frames: Dict[str, Optional[int]]
    comparable: bool
    summary: str
    caveat_plain: Optional[str] = None
    rows: List[CompareCheckpointRow] = Field(default_factory=list)
    extended_rows: List[CompareExtendedRow] = Field(default_factory=list)


class BiomechanicalFactSheet(BaseModel):
    """Strict neuro-symbolic contract between deterministic biomechanics engine and coaching narrator."""

    schema_version: str = "2.0.0"
    bowler_id: str
    bowling_arm: BowlingArm
    action_class: BowlerActionClass
    pose_backend: str
    interval_type: IntervalType = "analytical_uncertainty"
    audit: VideoAudit
    calibration: PitchCalibration
    events: DeliveryEvents
    workload_context: WorkloadContext = Field(default_factory=WorkloadContext)
    workload_guidance: Optional[WorkloadGuidance] = None
    metrics: Dict[str, MetricEstimate]
    v5_metrics: Dict[str, MetricEstimate] = Field(default_factory=dict)
    cohort_comparisons: List[CohortComparison] = Field(default_factory=list)
    # Retained for schema compatibility; composite 0-100 scores are retired (score=None) pending external validation.
    kinetic_efficiency: PillarScore
    lumbar_load_screening: PillarScore
    repeatability: Optional[PillarScore] = None
    quadrant: QuadrantPlacement
    strengths: List[PrioritizedCue] = Field(default_factory=list)
    limiters: List[PrioritizedCue] = Field(default_factory=list)
    inconclusive_observations: List[PrioritizedCue] = Field(default_factory=list)
    capture_upgrades: List[PrioritizedCue] = Field(default_factory=list)
    corrective_drills: List[CorrectiveDrill] = Field(default_factory=list)
    coaching_story: Optional[CoachingStory] = None
    kinematic_sequence: Optional[KinematicSequenceAnalysis] = None
    com_braking: Optional[COMBrakingAnalysis] = None
    head_stability: Optional[HeadStabilityAnalysis] = None
    lead_arm: Optional[LeadArmAnalysis] = None
    crease_tunnel: Optional[CreaseReleaseTunnelAnalysis] = None
    tempo_rhythm: Optional[TempoRhythmAnalysis] = None
    hard_guardrails: List[str] = Field(
        default_factory=lambda: [
            "ICC 15-degree elbow legality CANNOT be adjudicated from monocular or casual 2D/3D video; accredited >=250 Hz 3D optical lab testing is required.",
            "Biomechanical load screening estimates single-delivery mechanical posture only and is NOT a medical diagnosis; injury risk depends strongly on workload, age/skeletal maturity, and physical capacity.",
            "Intervals reported without held-out physical reference calibration represent repeatability or analytical sensitivity estimates rather than externally calibrated laboratory confidence intervals.",
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def all_metric_estimates(self) -> Dict[str, MetricEstimate]:
        """Return merged dictionary of canonical 23 metrics + extended Studio v5 MetricEstimates."""
        if not self.v5_metrics:
            return dict(self.metrics)
        return {**self.v5_metrics, **self.metrics}


class CoachingReport(BaseModel):
    bowler_id: str
    executive_summary: str
    action_profile_summary: str
    strengths_narrative: List[str]
    prioritized_improvements: List[str]
    capture_quality_caveats: List[str]
    disclaimers: List[str]
    cited_metric_keys: List[str]
    generator_mode: str = Field(description="deterministic_template | llm_grounded")
