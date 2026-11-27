"""Studio v5 (Module 8): Peer-Reviewed Pro Archetype 3D Reference Library (`GET /api/archetypes`).

Synthesizes 4 literature-grounded 3D fast-bowling reference trajectories via `generate_synthetic_delivery`
parameterized strictly by published cohort envelopes (Portus et al., 2004; Worthington et al., 2013;
Felton et al., 2018/2019; King et al., 2016).

Scientific Guardrails (AGENTS.md §1.2 & Glazier, 2010 Dynamical Systems Theory):
  - Never impersonates or claims optical motion capture of a named real player.
  - Never computes a 0-100 Euclidean "similarity score" against a single "ideal" template.
  - Supports both `right` and `left` bowling arms via exact chirality generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .constants import JOINT_IDX, SMPL24_JOINTS
from .schemas import BowlerActionClass, BowlingArm, PitchCalibration, VideoAudit
from .stage1_pitch_calibration import calibrate_from_mcc_landmarks
from .stage2_event_spotting import detect_delivery_events
from .stage3_kinematics import extract_fast_bowling_kinematics
from .stage3_v5_metrics import compute_3d_com_trajectory, extract_and_quantify_v5_biomechanics
from .synthetic_ground_truth import (
    SyntheticDeliveryParams,
    generate_synthetic_delivery,
    make_virtual_camera,
    project_mcc_landmarks,
)

ARCHETYPE_DISCLAIMER = (
    "Synthetic archetype: literature-parameterized 3D reference model illustrating published biomechanical "
    "principles — NOT optical motion capture of a real player and NEVER a mandatory one-size-fits-all template "
    "(Glazier, 2010: elite fast bowlers self-organize around individual anthropometry and strength)."
)


@dataclass(frozen=True)
class ArchetypeSpec:
    id: str
    title: str
    subtitle: str
    action_class: BowlerActionClass
    literature_source: str
    coaching_notes: str
    params_kwargs: Dict[str, Any]
    metric_envelopes: Dict[str, Dict[str, Any]]


_ARCHETYPE_SPECS: Tuple[ArchetypeSpec, ...] = (
    ArchetypeSpec(
        id="side_on_express",
        title="Side-On Braced Express",
        subtitle="Braced front-knee extender · delayed arm · rapid chest whip",
        action_class=BowlerActionClass.SIDE_ON,
        literature_source="Portus et al. (2004), Sports Biomechanics; Worthington, King, & Ranson (2013), JAB",
        coaching_notes=(
            "Enters back-foot landing aligned <200° side-on with minimal counter-rotation, plants onto a firm "
            "extending front leg (162° -> 173°), and delays arm circumduction behind the hip at FFC."
        ),
        params_kwargs={
            "fps": 120.0,
            "duration_s": 1.05,
            "runup_speed_mps": 6.10,
            "bfc_time_s": 0.38,
            "ffc_time_s": 0.572,
            "br_time_s": 0.676,
            "stride_length_m": 1.52,
            "front_knee_ffc_deg": 162.0,
            "front_knee_br_deg": 173.0,
            "trunk_flexion_ffc_deg": 11.0,
            "trunk_flexion_br_deg": 46.0,
            "thorax_lat_flex_ffc_deg": 17.0,
            "thorax_lat_flex_br_deg": 36.0,
            "pelvis_yaw_bfc_deg": 192.0,
            "shoulder_yaw_bfc_deg": 188.0,
            "shoulder_counter_rot_deg": 10.0,
            "delayed_arm_ffc_deg": 152.0,
            "br_phi_deg": 6.0,
            "stature_m": 1.86,
        },
        metric_envelopes={
            "front_knee_br_deg": {"mean": 173.0, "range": [164.0, 179.0], "unit": "deg"},
            "trunk_flexion_delta_deg": {"mean": 35.0, "range": [28.0, 42.0], "unit": "deg"},
            "delayed_arm_ffc_deg": {"mean": 152.0, "range": [136.0, 165.0], "unit": "deg"},
            "shoulder_alignment_bfc_deg": {"mean": 188.0, "range": [175.0, 208.0], "unit": "deg"},
            "shoulder_counter_rotation_deg": {"mean": 10.0, "range": [4.0, 22.0], "unit": "deg"},
            "thorax_lat_flex_br_deg": {"mean": 36.0, "range": [28.0, 44.0], "unit": "deg"},
        },
    ),
    ArchetypeSpec(
        id="semi_open_all_round",
        title="Semi-Open Balanced Rhythm",
        subtitle="AIS 210°–240° diagonal alignment · smooth 1.8:1 gather-to-snap tempo",
        action_class=BowlerActionClass.SEMI_OPEN,
        literature_source="Portus et al. (2001/2004); Felton, Lister, Worthington, & King (2019), Sports Biomechanics",
        coaching_notes=(
            "Aligns pelvis and shoulders coherently in the 210°–240° semi-open corridor at back-foot landing, "
            "pairing a ~192 ms delivery stride with a ~106 ms front-foot-to-release snap."
        ),
        params_kwargs={
            "fps": 120.0,
            "duration_s": 1.05,
            "runup_speed_mps": 5.83,
            "bfc_time_s": 0.38,
            "ffc_time_s": 0.572,
            "br_time_s": 0.678,
            "stride_length_m": 1.46,
            "front_knee_ffc_deg": 160.0,
            "front_knee_br_deg": 168.0,
            "trunk_flexion_ffc_deg": 12.0,
            "trunk_flexion_br_deg": 44.0,
            "thorax_lat_flex_ffc_deg": 18.0,
            "thorax_lat_flex_br_deg": 35.0,
            "pelvis_yaw_bfc_deg": 224.0,
            "shoulder_yaw_bfc_deg": 220.0,
            "shoulder_counter_rot_deg": 14.0,
            "delayed_arm_ffc_deg": 145.0,
            "br_phi_deg": 7.0,
            "stature_m": 1.85,
        },
        metric_envelopes={
            "front_knee_br_deg": {"mean": 168.0, "range": [158.0, 176.0], "unit": "deg"},
            "trunk_flexion_delta_deg": {"mean": 32.0, "range": [25.0, 39.0], "unit": "deg"},
            "delayed_arm_ffc_deg": {"mean": 145.0, "range": [128.0, 158.0], "unit": "deg"},
            "shoulder_alignment_bfc_deg": {"mean": 220.0, "range": [210.0, 238.0], "unit": "deg"},
            "shoulder_counter_rotation_deg": {"mean": 14.0, "range": [6.0, 25.0], "unit": "deg"},
            "thorax_lat_flex_br_deg": {"mean": 35.0, "range": [28.0, 43.0], "unit": "deg"},
        },
    ),
    ArchetypeSpec(
        id="front_on_power",
        title="Front-On Rotational Power",
        subtitle="Open chest (>240° BFC) · axial hip-to-thorax torque · low counter-rotation",
        action_class=BowlerActionClass.FRONT_ON,
        literature_source="Portus et al. (2004); Felton, Lister, Worthington, & King (2018), J. Sports Sciences",
        coaching_notes=(
            "Enters the crease chest-open (>240° alignment) without twisting the shoulders back away from the batter, "
            "generating speed via coordinated axial torso uncoiling and flexor-extender lead-leg support."
        ),
        params_kwargs={
            "fps": 120.0,
            "duration_s": 1.05,
            "runup_speed_mps": 5.70,
            "bfc_time_s": 0.38,
            "ffc_time_s": 0.566,
            "br_time_s": 0.678,
            "stride_length_m": 1.42,
            "front_knee_ffc_deg": 156.0,
            "front_knee_br_deg": 164.0,
            "trunk_flexion_ffc_deg": 13.0,
            "trunk_flexion_br_deg": 43.0,
            "thorax_lat_flex_ffc_deg": 16.0,
            "thorax_lat_flex_br_deg": 34.0,
            "pelvis_yaw_bfc_deg": 256.0,
            "shoulder_yaw_bfc_deg": 252.0,
            "shoulder_counter_rot_deg": 12.0,
            "delayed_arm_ffc_deg": 138.0,
            "br_phi_deg": 8.0,
            "stature_m": 1.84,
        },
        metric_envelopes={
            "front_knee_br_deg": {"mean": 164.0, "range": [153.0, 174.0], "unit": "deg"},
            "trunk_flexion_delta_deg": {"mean": 30.0, "range": [23.0, 38.0], "unit": "deg"},
            "delayed_arm_ffc_deg": {"mean": 138.0, "range": [120.0, 152.0], "unit": "deg"},
            "shoulder_alignment_bfc_deg": {"mean": 252.0, "range": [242.0, 272.0], "unit": "deg"},
            "shoulder_counter_rotation_deg": {"mean": 12.0, "range": [4.0, 24.0], "unit": "deg"},
            "thorax_lat_flex_br_deg": {"mean": 34.0, "range": [26.0, 42.0], "unit": "deg"},
        },
    ),
    ArchetypeSpec(
        id="high_arm_bounce",
        title="Tall High-Release Bounce",
        subtitle="Controlled compact stride · upright front block · ~113% stature release height",
        action_class=BowlerActionClass.SEMI_OPEN,
        literature_source="Worthington et al. (2013); King, Worthington, & Ranson (2016), J. Sports Sciences",
        coaching_notes=(
            "Maximizes vertical release height (~113% of stature) and steep bounce via a compact ~74% stature stride, "
            "minimal vertical COM sink, stacked head over the front ankle, and tall ear-skimming release."
        ),
        params_kwargs={
            "fps": 120.0,
            "duration_s": 1.05,
            "runup_speed_mps": 5.55,
            "bfc_time_s": 0.38,
            "ffc_time_s": 0.568,
            "br_time_s": 0.672,
            "stride_length_m": 1.44,
            "front_knee_ffc_deg": 165.0,
            "front_knee_br_deg": 174.0,
            "trunk_flexion_ffc_deg": 8.0,
            "trunk_flexion_br_deg": 36.0,
            "thorax_lat_flex_ffc_deg": 15.0,
            "thorax_lat_flex_br_deg": 31.0,
            "pelvis_yaw_bfc_deg": 216.0,
            "shoulder_yaw_bfc_deg": 212.0,
            "shoulder_counter_rot_deg": 9.0,
            "delayed_arm_ffc_deg": 142.0,
            "br_phi_deg": 3.0,
            "stature_m": 1.95,
        },
        metric_envelopes={
            "front_knee_br_deg": {"mean": 174.0, "range": [165.0, 180.0], "unit": "deg"},
            "trunk_flexion_delta_deg": {"mean": 28.0, "range": [21.0, 35.0], "unit": "deg"},
            "release_height_pct_stature": {"mean": 113.0, "range": [107.0, 118.0], "unit": "% stature"},
            "shoulder_counter_rotation_deg": {"mean": 9.0, "range": [3.0, 20.0], "unit": "deg"},
            "thorax_lat_flex_br_deg": {"mean": 31.0, "range": [24.0, 38.0], "unit": "deg"},
        },
    ),
)

_ARCHETYPE_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def get_reference_archetypes(arm: BowlingArm = BowlingArm.RIGHT) -> List[Dict[str, Any]]:
    """Build (and memoize per bowling arm) the 4 peer-reviewed 3D reference archetypes."""
    cache_key = arm.value
    if cache_key in _ARCHETYPE_CACHE:
        return _ARCHETYPE_CACHE[cache_key]

    lm_keys = [
        "bowling_crease_off", "bowling_crease_leg", "popping_crease_off",
        "popping_crease_leg", "stump_middle_base", "stump_middle_top",
    ]
    K, R, t, _ = make_virtual_camera(azimuth_deg=55.0, elevation_deg=6.0, distance_m=10.5, fov_deg=48.0)
    cal: PitchCalibration = calibrate_from_mcc_landmarks(project_mcc_landmarks(K, R, t, lm_keys), 1280, 720, K=K)

    built: List[Dict[str, Any]] = []
    for spec in _ARCHETYPE_SPECS:
        params = SyntheticDeliveryParams(bowler_arm=arm, **spec.params_kwargs)
        jw, _ = generate_synthetic_delivery(params)
        n = jw.shape[0]
        fps = float(params.fps)
        ev = detect_delivery_events(jw, fps=fps, bowling_arm=arm)
        kin = extract_fast_bowling_kinematics(jw, ev, fps=fps, bowling_arm=arm, debias_near_straight=False)
        audit = VideoAudit(
            video_path=f"archetype:{spec.id}:{arm.value}",
            width=1280,
            height=720,
            fps=fps,
            container_fps=fps,
            num_frames=n,
            raw_num_frames=n,
            duration_s=round(n / fps, 3),
            median_laplacian_var=320.0,
            blur_grade="crisp",
            fps_grade="slow_mo_120",
            est_arm_deg_per_frame=round(2300.0 / fps, 2),
            est_ball_m_per_frame=round(38.89 / fps, 3),
        )
        v5_bundle = extract_and_quantify_v5_biomechanics(kin, audit, cal, ev)

        y_ground = float(np.percentile(jw[:, [JOINT_IDX["lank"], JOINT_IDX["rank"]], 1], 8))
        jw_render = jw.copy()
        jw_render[:, :, 1] -= y_ground
        com_render = compute_3d_com_trajectory(jw_render)
        t_rel_ffc_ms = [round((i - ev.ffc.frame_float) * 1000.0 / fps, 2) for i in range(n)]

        envelopes_ts = {
            mk: {
                "low": float(ev_d["range"][0]),
                "high": float(ev_d["range"][1]),
                "mean": float(ev_d["mean"]),
                "range": list(ev_d["range"]),
                "unit": str(ev_d["unit"]),
                "source": spec.literature_source,
            }
            for mk, ev_d in spec.metric_envelopes.items()
        }

        built.append(
            {
                "id": spec.id,
                "title": spec.title,
                "subtitle": spec.subtitle,
                "bowling_arm": arm.value,
                "action_class": spec.action_class.value,
                "is_synthetic_archetype": True,
                "provenance_badge": "Synthetic archetype: not a real player",
                "disclaimer": ARCHETYPE_DISCLAIMER,
                "literature_source": spec.literature_source,
                "coaching_notes": spec.coaching_notes,
                "stature_m": params.stature_m,
                "fps": fps,
                "num_frames": n,
                "joint_names": list(SMPL24_JOINTS),
                "events": {
                    "bfc_frame": ev.bfc.frame_idx,
                    "ffc_frame": ev.ffc.frame_idx,
                    "uah_frame": ev.uah.frame_idx if ev.uah is not None else None,
                    "br_frame": ev.br.frame_idx,
                    "bfc_rel_ffc_ms": round((ev.bfc.frame_float - ev.ffc.frame_float) * 1000.0 / fps, 1),
                    "br_rel_ffc_ms": round((ev.br.frame_float - ev.ffc.frame_float) * 1000.0 / fps, 1),
                },
                "time_rel_ffc_ms": t_rel_ffc_ms,
                "joints3d_world_m": np.round(jw_render, 4).tolist(),
                "com_world_m": np.round(com_render, 4).tolist(),
                "kin_series": {k: np.round(np.asarray(v, dtype=np.float64), 2).tolist() for k, v in kin.series.items()},
                "sequence_series": {
                    k: np.round(np.asarray(v, dtype=np.float64), 2).tolist()
                    for k, v in v5_bundle.sequence_series.items()
                },
                "metric_envelopes": envelopes_ts,
            }
        )

    _ARCHETYPE_CACHE[cache_key] = built
    return built
