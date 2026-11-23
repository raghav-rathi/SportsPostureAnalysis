"""Strict Pydantic v2 data contracts for the Cricket Fast-Bowling Biomechanics Engine.

Enforces uncertainty quantification and observability gating structurally across every stage.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


class BowlingArm(str, Enum):
    RIGHT = "right"
    LEFT = "left"


class BowlerActionClass(str, Enum):
    SIDE_ON = "side_on"
    SEMI_SIDE_ON = "semi_side_on"
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


class VideoAudit(BaseModel):
    video_path: str
    width: int
    height: int
    fps: float
    num_frames: int
    duration_s: float
    median_laplacian_var: float = Field(
        description="Median sharpness proxy across frames (higher = crisper)."
    )
    blur_grade: str = Field(description="crisp | moderate_blur | severe_blur")
    fps_grade: str = Field(description="high_speed_240 | slow_mo_120 | medium_60 | standard_30 | low_25")
    est_arm_deg_per_frame: float = Field(
        description="Estimated bowling arm sweep (deg) between frames at peak 2300 deg/s."
    )
    est_ball_m_per_frame: float = Field(
        description="Estimated ball travel (m) between frames at 140 km/h (38.9 m/s)."
    )
    warnings: List[str] = Field(default_factory=list)


class PitchCalibration(BaseModel):
    method: str = Field(
        description="mcc_law7_pnp | motion_gravity_self_cal | fallback_uncalibrated"
    )
    camera_matrix_K: List[List[float]] = Field(description="3x3 intrinsic camera matrix.")
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
        description="Camera elevation above pitch plane in degrees (0=ground level, >15=elevated broadcast)."
    )
    px_per_meter_at_crease: Optional[float] = None
    reprojection_error_px: Optional[float] = None
    view_class: CameraViewClass = CameraViewClass.UNKNOWN
    is_metric_calibrated: bool = False
    is_panning_camera: bool = False
    calibration_notes: List[str] = Field(default_factory=list)


class EventTimestamp(BaseModel):
    frame_float: float = Field(description="Sub-frame index (0-based).")
    frame_idx: int = Field(description="Nearest integer frame index.")
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
    sigma_total: Optional[float] = None
    sigma_projective: Optional[float] = None
    sigma_temporal: Optional[float] = None
    sigma_epistemic: Optional[float] = None
    observability: ObservabilityGrade
    suppressed: bool = False
    suppression_reason: Optional[str] = None
    empirical_reference: str = Field(
        description="Peer-reviewed norm/range from cricket biomechanics literature."
    )
    literature_source: str


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
    category: str = Field(description="strength | limiter | capture_upgrade")
    title: str
    evidence_metrics: List[str]
    biomechanical_mechanism: str
    actionable_recommendation: str


class WorkloadContext(BaseModel):
    age_years: Optional[int] = None
    sex: str = "male"
    bowls_per_week: Optional[int] = None
    recent_growth_spurt: Optional[bool] = None
    known_pace_kph: Optional[float] = None


class BiomechanicalFactSheet(BaseModel):
    """Strict neuro-symbolic contract between deterministic biomechanics engine and LLM narrator."""

    schema_version: str = "1.0.0"
    bowler_id: str
    bowling_arm: BowlingArm
    action_class: BowlerActionClass
    pose_backend: str
    audit: VideoAudit
    calibration: PitchCalibration
    events: DeliveryEvents
    workload_context: WorkloadContext = Field(default_factory=WorkloadContext)
    metrics: Dict[str, MetricEstimate]
    kinetic_efficiency: PillarScore
    lumbar_load_screening: PillarScore
    repeatability: Optional[PillarScore] = None
    quadrant: QuadrantPlacement
    strengths: List[PrioritizedCue] = Field(default_factory=list)
    limiters: List[PrioritizedCue] = Field(default_factory=list)
    capture_upgrades: List[PrioritizedCue] = Field(default_factory=list)
    hard_guardrails: List[str] = Field(
        default_factory=lambda: [
            "ICC 15-degree elbow legality CANNOT be adjudicated from monocular or casual 2D/3D video; accredited >=250 Hz 3D optical lab testing is required.",
            "Mechanical load screening estimates stress per delivery only and is NOT a medical diagnosis; actual injury risk depends strongly on bowling workload, age/skeletal maturity, and physical capacity.",
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
