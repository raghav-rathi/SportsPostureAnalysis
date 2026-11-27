"""Strict Pydantic v2 data contracts for the Cricket Fast-Bowling Biomechanics Engine.

Enforces uncertainty quantification, timebase honesty, observability gating, and
definition-matched literature reporting structurally across every stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

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
    hard_guardrails: List[str] = Field(
        default_factory=lambda: [
            "ICC 15-degree elbow legality CANNOT be adjudicated from monocular or casual 2D/3D video; accredited >=250 Hz 3D optical lab testing is required.",
            "Biomechanical load screening estimates single-delivery mechanical posture only and is NOT a medical diagnosis; injury risk depends strongly on workload, age/skeletal maturity, and physical capacity.",
            "Intervals reported without held-out physical reference calibration represent repeatability or analytical sensitivity estimates rather than externally calibrated laboratory confidence intervals.",
        ]
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)


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
