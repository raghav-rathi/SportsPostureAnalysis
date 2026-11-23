"""Stage 3: Anatomically Grounded 3D Fast-Bowling Kinematics Extractor.

Fixes EXP-007:
  - B5: Local front-foot ground reference `[FFC, BR]` for `release_height_m` (eliminating 0-6% coverage
        under self-cal gravity tilt across the run-up) + explicit `ft_disp_x_m` travel diagnostic.
  - B10: Rician/Jensen out-of-plane debiasing for near-180 deg straight knees (`front_knee_*`),
         quadratic Savitzky-Golay derivative for `peak_arm_angular_vel_dps`, and 10 Hz physiological
         filtering on transverse pelvis/shoulder yaw to curb extremum uplift.
  - B18: Anchors `shoulder_yaw_deg` at BFC to the nearest 360-deg branch of `pelvis_yaw_deg` so
         lines straddling 90 deg NEVER produce `-354 deg` hip-shoulder separation.
  - B21 / B23: Drops dead `crunch_factor_proxy`; imports `JOINT_IDX` from `constants.py`.
  - Section 6: Updates `classify_action_portus` to published AIS cutoffs (`<210` side-on,
               `210-240` semi-open, `>240` front-on, `>=30` SCR or BFC separation = mixed).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.signal import savgol_filter

from .constants import JOINT_IDX
from .schemas import BowlerActionClass, BowlingArm, DeliveryEvents
from .stage2_event_spotting import smooth_trajectory


@dataclass
class KinematicsResult:
    scalars: Dict[str, float]
    local_rates_per_sec: Dict[str, float]
    metric_joint_uncertainties_m: Dict[str, Tuple[float, ...]]
    metric_joints_in_frame: Dict[str, bool]
    action_class: BowlerActionClass
    series: Dict[str, np.ndarray]
    estimated_stature_m: float
    # Full trajectory context passed to Stage 4 for exact Jacobian, timing, & calibration propagation
    joints_world_smoothed: Optional[np.ndarray] = None
    joint_uncertainties_array: Optional[np.ndarray] = None
    interp_gap_frames: Optional[np.ndarray] = None
    bowling_arm: BowlingArm = BowlingArm.RIGHT
    fps: float = 120.0
    events: Optional[DeliveryEvents] = None
    ft_disp_x_m: float = 0.0
    savgol_window_s: float = 0.05
    joint_uncertainties_m: Dict[str, float] = field(default_factory=dict)


def _interp_frame(arr: np.ndarray, frame_float: float) -> np.ndarray:
    """Linearly interpolate a (T, ...) array at fractional frame index."""
    n = arr.shape[0]
    f = float(np.clip(frame_float, 0.0, max(n - 1, 0)))
    i0 = int(math.floor(f))
    i1 = min(i0 + 1, n - 1)
    alpha = f - i0
    return (1.0 - alpha) * arr[i0] + alpha * arr[i1]


def _internal_joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Compute internal 3D angle ABC in degrees (180 = fully straight limb)."""
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos_ang = np.sum(v1 * v2, axis=-1) / np.maximum((n1 * n2).squeeze(-1), 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def _angle_between_vectors_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Compute 3D angle between vector sequences v1 and v2 (180 = aligned straight)."""
    n1 = np.linalg.norm(v1, axis=-1, keepdims=True)
    n2 = np.linalg.norm(v2, axis=-1, keepdims=True)
    cos_ang = np.sum(v1 * v2, axis=-1) / np.maximum((n1 * n2).squeeze(-1), 1e-8)
    return np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))


def _debias_near_straight_angle_deg(
    theta_meas_deg: float,
    pa: np.ndarray,
    pb: np.ndarray,
    pc: np.ndarray,
    cov_a: np.ndarray,
    cov_b: np.ndarray,
    cov_c: np.ndarray,
) -> float:
    """Rician/Jensen out-of-plane debiasing for 3-point internal angles near 180 deg (EXP-007 B10).

    In 3D, out-of-plane noise perpendicular to the flexion plane strictly adds in quadrature to the
    supplement phi = 180 - theta: E[phi_meas^2] ~= phi_true^2 + sigma_oop^2.
    """
    u = pa - pb
    w = pc - pb
    l1 = max(float(np.linalg.norm(u)), 0.15)
    l2 = max(float(np.linalg.norm(w)), 0.15)
    u_hat = u / l1
    w_hat = w / l2
    n_vec = np.cross(u_hat, w_hat)
    n_norm = float(np.linalg.norm(n_vec))
    if n_norm < 1e-4:
        # Almost collinear: pick arbitrary unit vector orthogonal to u_hat (camera depth direction dominates)
        ref = np.array([0.0, 0.0, 1.0]) if abs(u_hat[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        n_hat = np.cross(u_hat, ref)
        n_hat /= max(float(np.linalg.norm(n_hat)), 1e-6)
    else:
        n_hat = n_vec / n_norm

    g_a = -n_hat / l1
    g_b = (1.0 / l1 + 1.0 / l2) * n_hat
    g_c = -n_hat / l2
    var_rad = float(g_a @ cov_a @ g_a + g_b @ cov_b @ g_b + g_c @ cov_c @ g_c)
    sigma_oop_deg = min(math.degrees(math.sqrt(max(var_rad, 0.0))), 12.0)

    phi_meas = max(0.0, 180.0 - theta_meas_deg)
    # Conservative 0.85 factor removes the systematic negative bias without snapping true 165 deg knees to 180
    phi_corr = math.sqrt(max(phi_meas * phi_meas - 0.85 * sigma_oop_deg * sigma_oop_deg, 0.0))
    return float(np.clip(180.0 - phi_corr, 0.0, 180.0))


def _estimate_stature_m(j: np.ndarray) -> float:
    """Estimate bowler stature from median SMPL-24 segment lengths (including pelv->mid_hip gap)."""
    mid_hip = 0.5 * (j[:, JOINT_IDX["lhip"]] + j[:, JOINT_IDX["rhip"]])
    pelv_hip_gap = float(np.median(np.linalg.norm(j[:, JOINT_IDX["pelv"]] - mid_hip, axis=1)))
    thigh = float(np.median(np.linalg.norm(j[:, JOINT_IDX["lhip"]] - j[:, JOINT_IDX["lkne"]], axis=1)))
    shank = float(np.median(np.linalg.norm(j[:, JOINT_IDX["lkne"]] - j[:, JOINT_IDX["lank"]], axis=1)))
    torso = float(np.median(np.linalg.norm(j[:, JOINT_IDX["neck"]] - j[:, JOINT_IDX["pelv"]], axis=1)))
    neck_head = float(np.median(np.linalg.norm(j[:, JOINT_IDX["head"]] - j[:, JOINT_IDX["neck"]], axis=1)))
    raw = thigh + shank + pelv_hip_gap + torso + neck_head + 0.25
    return float(np.clip(raw, 1.40, 2.20))


def _transverse_alignment_deg(
    left_pt: np.ndarray,
    right_pt: np.ndarray,
    is_right_arm: bool,
    ref_idx: int = 0,
    anchor_target_deg: Optional[float] = None,
) -> np.ndarray:
    """Compute unwrapped Portus (2001/2004) / AIS transverse alignment in degrees relative to +X pitch axis.

    Right-handed Pitch World (+X down pitch, +Y up, +Z = bowler's right when facing +X):
      - Front-On (270 deg): chest faces +X => lr = right - left = (0, 0, +w)
      - Right-arm Side-On (180 deg): left shoulder points +X => lr = (-w, 0, 0)
      - Left-arm Side-On (180 deg): right shoulder points +X => lr = (+w, 0, 0)
    If `anchor_target_deg` is provided, anchors `deg[ref_idx]` to the nearest 360-deg branch around
    `anchor_target_deg` (fixes EXP-007 B18 `-354 deg` pelvis-shoulder wrap straddle!).
    """
    sign_arm = 1.0 if is_right_arm else -1.0
    lr = right_pt - left_pt
    rad_raw = np.arctan2(sign_arm * lr[:, 0], lr[:, 2])
    rad_unwrapped = np.unwrap(rad_raw)
    deg = 270.0 + np.degrees(rad_unwrapped)
    idx_safe = int(np.clip(ref_idx, 0, len(deg) - 1))
    if anchor_target_deg is not None:
        shift = 360.0 * round((float(deg[idx_safe]) - float(anchor_target_deg)) / 360.0)
    else:
        # Branch cut at [60, 420) deg (centered on 240 deg, midpoint of cricket [150, 330] actions)
        shift = 360.0 * math.floor((float(deg[idx_safe]) - 60.0) / 360.0)
    return deg - shift


def classify_action_portus(
    shoulder_bfc_deg: float,
    pelvis_bfc_deg: float,
    scr_deg: float,
) -> BowlerActionClass:
    """Published Portus et al. (2001/2004) / AIS fast-bowling action classification.

    Cutoffs from Portus et al. (2001) ISBS / AIS convention:
      - Mixed: SCR >= 30 deg OR |shoulder_bfc - pelvis_bfc| >= 30 deg (wrapped mod 360)
      - Side-On: shoulder_bfc < 210 deg
      - Semi-Open: 210 deg <= shoulder_bfc <= 240 deg
      - Front-On: shoulder_bfc > 240 deg
    """
    sep_bfc = abs(((pelvis_bfc_deg - shoulder_bfc_deg + 180.0) % 360.0) - 180.0)
    if scr_deg >= 30.0 or sep_bfc >= 30.0:
        return BowlerActionClass.MIXED
    if shoulder_bfc_deg < 210.0:
        return BowlerActionClass.SIDE_ON
    if shoulder_bfc_deg <= 240.0:
        return BowlerActionClass.SEMI_OPEN
    return BowlerActionClass.FRONT_ON


def _compute_world_cov(R_w2c: np.ndarray, unc_m: float, unc_scale: float = 1.0) -> np.ndarray:
    s_xy = 0.604 * unc_scale * unc_m
    s_z = 1.450 * unc_scale * unc_m
    cov_cam = np.diag([s_xy * s_xy, s_xy * s_xy, s_z * s_z])
    return R_w2c.T @ cov_cam @ R_w2c


def extract_fast_bowling_kinematics(
    joints_world: np.ndarray,
    events: DeliveryEvents,
    fps: float,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    joint_uncertainties_m: Optional[np.ndarray] = None,
    joint_in_frame: Optional[np.ndarray] = None,
    interp_gap_frames: Optional[np.ndarray] = None,
    camera_R_w2c: Optional[np.ndarray] = None,
    unc_scale_factor: float = 1.0,
) -> KinematicsResult:
    """Extract 3D fast-bowling kinematics with debiasing, local ground reference, & Jacobian context."""
    j = smooth_trajectory(joints_world, fps, cutoff_hz=min(16.0, 0.36 * fps))
    # Lower-bandwidth 10 Hz filtered copy strictly for pelvis/thorax transverse rotation (reduces extremum uplift)
    j_trans = smooth_trajectory(joints_world, fps, cutoff_hz=min(10.0, 0.30 * fps))
    n = j.shape[0]
    is_right = bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0

    f_hip_i = JOINT_IDX["lhip" if is_right else "rhip"]
    f_kne_i = JOINT_IDX["lkne" if is_right else "rkne"]
    f_ank_i = JOINT_IDX["lank" if is_right else "rank"]
    r_hip_i = JOINT_IDX["rhip" if is_right else "lhip"]
    r_kne_i = JOINT_IDX["rkne" if is_right else "lkne"]
    r_ank_i = JOINT_IDX["rank" if is_right else "lank"]
    b_sho_i = JOINT_IDX["rsho" if is_right else "lsho"]
    b_elb_i = JOINT_IDX["relb" if is_right else "lelb"]
    b_wri_i = JOINT_IDX["rwri" if is_right else "lwri"]
    nb_sho_i = JOINT_IDX["lsho" if is_right else "rsho"]
    nb_elb_i = JOINT_IDX["lelb" if is_right else "relb"]

    pelv = j[:, JOINT_IDX["pelv"]]
    lhip, rhip = j[:, JOINT_IDX["lhip"]], j[:, JOINT_IDX["rhip"]]
    neck = j[:, JOINT_IDX["neck"]]
    lsho, rsho = j[:, JOINT_IDX["lsho"]], j[:, JOINT_IDX["rsho"]]

    f_hip, f_kne, f_ank = j[:, f_hip_i], j[:, f_kne_i], j[:, f_ank_i]
    r_hip, r_kne, r_ank = j[:, r_hip_i], j[:, r_kne_i], j[:, r_ank_i]
    b_sho, b_elb, b_wri = j[:, b_sho_i], j[:, b_elb_i], j[:, b_wri_i]
    nb_sho, nb_elb = j[:, nb_sho_i], j[:, nb_elb_i]

    bfc_f = events.bfc.frame_float
    ffc_f = events.ffc.frame_float
    br_f = events.br.frame_float
    i_bfc = int(np.clip(round(bfc_f), 0, n - 1))
    i_ffc = int(np.clip(round(ffc_f), i_bfc, n - 1))
    i_br = int(np.clip(round(br_f), i_ffc, n - 1))

    # Continuous kinematic time-series
    front_knee_deg = _internal_joint_angle_deg(f_hip, f_kne, f_ank)
    # Note: trunk-to-thigh vector proxy (NOT 3D pelvis-to-thigh segment angle from Alway 2021)
    rear_hip_deg = _angle_between_vectors_deg(neck - pelv, r_kne - r_hip)
    bowling_elbow_deg = _internal_joint_angle_deg(b_sho, b_elb, b_wri)

    # Exact Z-X Cardan decomposition of global trunk inclination vector (neck - pelv) in Pitch World Frame
    spine_vec = neck - pelv
    trunk_flex_deg = np.degrees(np.arctan2(spine_vec[:, 0], np.maximum(spine_vec[:, 1], 1e-4)))
    thorax_lat_flex_deg = np.degrees(
        np.arctan2(-sign_arm * spine_vec[:, 2], np.hypot(spine_vec[:, 0], np.maximum(spine_vec[:, 1], 1e-4)))
    )

    # Continuously unwrapped transverse alignments anchored TOGETHER at BFC (fixes B18 `-354 deg` wrap!)
    pelvis_yaw_deg = _transverse_alignment_deg(
        j_trans[:, JOINT_IDX["lhip"]], j_trans[:, JOINT_IDX["rhip"]], is_right, ref_idx=i_bfc
    )
    shoulder_yaw_deg = _transverse_alignment_deg(
        j_trans[:, JOINT_IDX["lsho"]],
        j_trans[:, JOINT_IDX["rsho"]],
        is_right,
        ref_idx=i_bfc,
        anchor_target_deg=float(pelvis_yaw_deg[i_bfc]),
    )
    hip_shoulder_sep_deg = pelvis_yaw_deg - shoulder_yaw_deg

    # Ipsilateral pelvic drop (front hip lower than rear hip in +Y -> positive drop)
    hip_lr = f_hip - r_hip
    pelvic_drop_deg = np.degrees(
        np.arctan2(r_hip[:, 1] - f_hip[:, 1], np.maximum(np.hypot(hip_lr[:, 0], hip_lr[:, 2]), 1e-4))
    )

    # Continuously unwrapped bowling-arm circumduction angle anchored at BR in [-45, +90]
    b_arm_vec = b_elb - b_sho
    b_arm_phi_raw = np.degrees(np.arctan2(b_arm_vec[:, 0], b_arm_vec[:, 1]))
    b_arm_phi_unwrap = np.degrees(np.unwrap(np.radians(b_arm_phi_raw)))
    br_phi = float(_interp_frame(b_arm_phi_unwrap, br_f))
    br_shift = 360.0 * math.floor((br_phi + 45.0) / 360.0)
    b_arm_phi_anchored = b_arm_phi_unwrap - br_shift

    # Quadratic Savitzky-Golay derivative (~50 ms span) replaces noisy 2-point np.gradient (EXP-007 B10)
    sg_win = max(5, 2 * int(round(0.025 * fps)) + 1)
    if sg_win > n:
        sg_win = n if (n % 2 == 1) else max(3, n - 1)
    if sg_win >= 5 and n >= sg_win:
        b_arm_omega_dps = savgol_filter(b_arm_phi_anchored, window_length=sg_win, polyorder=2, deriv=1, delta=1.0 / max(fps, 1.0))
        savgol_window_s = float(sg_win - 1) / max(fps, 1.0)
    else:
        b_arm_omega_dps = np.gradient(b_arm_phi_anchored) * fps
        savgol_window_s = 2.0 / max(fps, 1.0)

    # Non-bowling arm elevation above horizontal (+90 = straight up, 0 = horizontal)
    nb_arm_vec = nb_elb - nb_sho
    nb_arm_elev_deg = np.degrees(
        np.arctan2(nb_arm_vec[:, 1], np.maximum(np.hypot(nb_arm_vec[:, 0], nb_arm_vec[:, 2]), 1e-4))
    )

    # Local derivatives for diagnostics
    d_front_knee = np.abs(np.gradient(front_knee_deg) * fps)
    d_trunk_flex = np.abs(np.gradient(trunk_flex_deg) * fps)
    d_lat_flex = np.abs(np.gradient(thorax_lat_flex_deg) * fps)
    d_pelvis_yaw = np.abs(np.gradient(pelvis_yaw_deg) * fps)
    d_shoulder_yaw = np.abs(np.gradient(shoulder_yaw_deg) * fps)
    d_sep = np.abs(np.gradient(hip_shoulder_sep_deg) * fps)
    d_arm = np.abs(b_arm_omega_dps)
    d_rear_hip = np.abs(np.gradient(rear_hip_deg) * fps)
    d_pelvic_drop = np.abs(np.gradient(pelvic_drop_deg) * fps)

    # Run-up speed: linear horizontal velocity across [BFC - 220 ms, BFC]
    r_start = max(0, int(math.floor(bfc_f - 0.22 * fps)))
    r_end = max(r_start + 1, min(n - 1, int(math.ceil(bfc_f))))
    if r_end > r_start:
        dt_run = (r_end - r_start) / max(fps, 1.0)
        runup_speed_mps = float((pelv[r_end, 0] - pelv[r_start, 0]) / max(dt_run, 1e-4))
    else:
        v_pelv_x = np.gradient(pelv[:, 0]) * fps
        runup_speed_mps = float(v_pelv_x[i_bfc])

    # Stride length: X_front_ankle(FFC) - X_rear_ankle(BFC)
    r_ank_bfc = _interp_frame(r_ank, bfc_f)
    f_ank_ffc = _interp_frame(f_ank, ffc_f)
    stride_length_m = float(f_ank_ffc[0] - r_ank_bfc[0])
    stature_m = _estimate_stature_m(j)
    stride_pct_stature = 100.0 * stride_length_m / max(stature_m, 1.0)

    # Key event scalar extractions (+ optional Rician out-of-plane debiasing on straight knees)
    fk_ffc = float(_interp_frame(front_knee_deg, ffc_f))
    fk_br = float(_interp_frame(front_knee_deg, br_f))
    unc_arr = (
        joint_uncertainties_m
        if (joint_uncertainties_m is not None and joint_uncertainties_m.shape[0] == n)
        else np.full((n, 24), 0.035, dtype=np.float64)
    )
    if camera_R_w2c is not None and joint_uncertainties_m is not None:
        R_use = np.asarray(camera_R_w2c, dtype=np.float64)
        fk_ffc = _debias_near_straight_angle_deg(
            fk_ffc,
            _interp_frame(f_hip, ffc_f),
            _interp_frame(f_kne, ffc_f),
            _interp_frame(f_ank, ffc_f),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_hip_i], ffc_f)), unc_scale_factor),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_kne_i], ffc_f)), unc_scale_factor),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_ank_i], ffc_f)), unc_scale_factor),
        )
        fk_br = _debias_near_straight_angle_deg(
            fk_br,
            _interp_frame(f_hip, br_f),
            _interp_frame(f_kne, br_f),
            _interp_frame(f_ank, br_f),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_hip_i], br_f)), unc_scale_factor),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_kne_i], br_f)), unc_scale_factor),
            _compute_world_cov(R_use, float(_interp_frame(unc_arr[:, f_ank_i], br_f)), unc_scale_factor),
        )
    fk_change = fk_br - fk_ffc

    tf_ffc = float(_interp_frame(trunk_flex_deg, ffc_f))
    tf_br = float(_interp_frame(trunk_flex_deg, br_f))
    tf_delta = tf_br - tf_ffc

    lf_ffc = float(_interp_frame(thorax_lat_flex_deg, ffc_f))
    lf_br = float(_interp_frame(thorax_lat_flex_deg, br_f))

    pelv_bfc = float(_interp_frame(pelvis_yaw_deg, bfc_f))
    sho_bfc = float(_interp_frame(shoulder_yaw_deg, bfc_f))
    sep_bfc = float(_interp_frame(hip_shoulder_sep_deg, bfc_f))

    min_sho_stride = float(np.min(shoulder_yaw_deg[i_bfc : i_ffc + 1]))
    scr_deg = max(0.0, sho_bfc - min_sho_stride)
    max_sep_deg = float(np.max(hip_shoulder_sep_deg[i_bfc : i_br + 1]))

    delayed_arm_ffc = float(np.clip(-_interp_frame(b_arm_phi_anchored, ffc_f), 0.0, 260.0))

    rear_hip_bfc = float(_interp_frame(rear_hip_deg, bfc_f))
    pelvic_drop_ffc = float(_interp_frame(pelvic_drop_deg, ffc_f))
    nb_arm_elev_ffc = float(_interp_frame(nb_arm_elev_deg, ffc_f))

    i_br_post = min(n - 1, int(math.ceil(br_f + 0.02 * fps)))
    peak_arm_omega = float(np.max(np.abs(b_arm_omega_dps[i_ffc : i_br_post + 1])))

    ft_end = min(n - 1, int(math.ceil(br_f + 0.25 * fps)))
    ft_disp_x = 0.0
    if ft_end > i_br + 1:
        ft_disp = pelv[ft_end] - pelv[i_br]
        ft_disp_x = float(ft_disp[0])
        ft_drift_deg = float(math.degrees(math.atan2(-sign_arm * ft_disp[2], max(ft_disp[0], 0.05))))
    else:
        ft_drift_deg = 0.0

    # Local front-foot ground reference during [FFC, BR] (fixes EXP-007 B5 / S6 0-6% release-height coverage!)
    local_f_ank_y = f_ank[i_ffc : i_br + 1, 1]
    ground_y_local = float(np.percentile(local_f_ank_y, 10)) if len(local_f_ank_y) > 0 else float(f_ank[i_br, 1])
    release_height_m = float(_interp_frame(b_wri[:, 1], br_f) - ground_y_local)

    action_class = classify_action_portus(sho_bfc, pelv_bfc, scr_deg)

    scalars = {
        "runup_speed_mps": runup_speed_mps,
        "stride_length_m": stride_length_m,
        "stride_length_pct_stature": stride_pct_stature,
        "front_knee_ffc_deg": fk_ffc,
        "front_knee_br_deg": fk_br,
        "front_knee_change_deg": fk_change,
        "trunk_flexion_ffc_deg": tf_ffc,
        "trunk_flexion_br_deg": tf_br,
        "trunk_flexion_delta_deg": tf_delta,
        "thorax_lat_flex_ffc_deg": lf_ffc,
        "thorax_lat_flex_br_deg": lf_br,
        "pelvis_alignment_bfc_deg": pelv_bfc,
        "shoulder_alignment_bfc_deg": sho_bfc,
        "shoulder_counter_rotation_deg": scr_deg,
        "hip_shoulder_sep_bfc_deg": sep_bfc,
        "max_hip_shoulder_sep_deg": max_sep_deg,
        "delayed_arm_ffc_deg": delayed_arm_ffc,
        "rear_hip_bfc_deg": rear_hip_bfc,
        "pelvic_drop_ffc_deg": pelvic_drop_ffc,
        "non_bowling_arm_elev_ffc_deg": nb_arm_elev_ffc,
        "peak_arm_angular_vel_dps": peak_arm_omega,
        "release_height_m": release_height_m,
        "follow_through_drift_deg": ft_drift_deg,
    }

    local_rates = {
        "runup_speed_mps": 4.0,
        "stride_length_m": max(abs(runup_speed_mps), 3.0),
        "stride_length_pct_stature": 200.0,
        "front_knee_ffc_deg": float(_interp_frame(d_front_knee, ffc_f)),
        "front_knee_br_deg": float(_interp_frame(d_front_knee, br_f)),
        "front_knee_change_deg": float(math.hypot(_interp_frame(d_front_knee, ffc_f), _interp_frame(d_front_knee, br_f))),
        "trunk_flexion_ffc_deg": float(_interp_frame(d_trunk_flex, ffc_f)),
        "trunk_flexion_br_deg": float(_interp_frame(d_trunk_flex, br_f)),
        "trunk_flexion_delta_deg": float(math.hypot(_interp_frame(d_trunk_flex, ffc_f), _interp_frame(d_trunk_flex, br_f))),
        "thorax_lat_flex_ffc_deg": float(_interp_frame(d_lat_flex, ffc_f)),
        "thorax_lat_flex_br_deg": float(_interp_frame(d_lat_flex, br_f)),
        "pelvis_alignment_bfc_deg": float(_interp_frame(d_pelvis_yaw, bfc_f)),
        "shoulder_alignment_bfc_deg": float(_interp_frame(d_shoulder_yaw, bfc_f)),
        "shoulder_counter_rotation_deg": float(np.max(d_shoulder_yaw[i_bfc : i_ffc + 1])),
        "hip_shoulder_sep_bfc_deg": float(_interp_frame(d_sep, bfc_f)),
        "max_hip_shoulder_sep_deg": float(np.max(d_sep[i_bfc : i_br + 1])),
        "delayed_arm_ffc_deg": float(_interp_frame(d_arm, ffc_f)),
        "rear_hip_bfc_deg": float(_interp_frame(d_rear_hip, bfc_f)),
        "pelvic_drop_ffc_deg": float(_interp_frame(d_pelvic_drop, ffc_f)),
        "non_bowling_arm_elev_ffc_deg": 280.0,
        "peak_arm_angular_vel_dps": 8000.0,
        "release_height_m": 2.5,
        "follow_through_drift_deg": 40.0,
    }

    pelv_i, neck_i = JOINT_IDX["pelv"], JOINT_IDX["neck"]
    lhip_i, rhip_i = JOINT_IDX["lhip"], JOINT_IDX["rhip"]
    lsho_i, rsho_i = JOINT_IDX["lsho"], JOINT_IDX["rsho"]

    req_spec: Dict[str, List[Tuple[int, List[int]]]] = {
        "runup_speed_mps": [(i_bfc, [pelv_i])],
        "stride_length_pct_stature": [(i_bfc, [r_ank_i]), (i_ffc, [f_ank_i])],
        "front_knee_ffc_deg": [(i_ffc, [f_hip_i, f_kne_i, f_ank_i])],
        "front_knee_br_deg": [(i_br, [f_hip_i, f_kne_i, f_ank_i])],
        "front_knee_change_deg": [(i_ffc, [f_hip_i, f_kne_i, f_ank_i]), (i_br, [f_hip_i, f_kne_i, f_ank_i])],
        "trunk_flexion_ffc_deg": [(i_ffc, [pelv_i, neck_i])],
        "trunk_flexion_br_deg": [(i_br, [pelv_i, neck_i])],
        "trunk_flexion_delta_deg": [(i_ffc, [pelv_i, neck_i]), (i_br, [pelv_i, neck_i])],
        "thorax_lat_flex_ffc_deg": [(i_ffc, [pelv_i, neck_i])],
        "thorax_lat_flex_br_deg": [(i_br, [pelv_i, neck_i])],
        "pelvis_alignment_bfc_deg": [(i_bfc, [lhip_i, rhip_i])],
        "shoulder_alignment_bfc_deg": [(i_bfc, [lsho_i, rsho_i])],
        "shoulder_counter_rotation_deg": [(i_bfc, [lsho_i, rsho_i]), (i_ffc, [lsho_i, rsho_i])],
        "hip_shoulder_sep_bfc_deg": [(i_bfc, [lhip_i, rhip_i, lsho_i, rsho_i])],
        "max_hip_shoulder_sep_deg": [(i_bfc, [lhip_i, rhip_i, lsho_i, rsho_i]), (i_ffc, [lhip_i, rhip_i, lsho_i, rsho_i])],
        "delayed_arm_ffc_deg": [(i_ffc, [b_sho_i, b_elb_i])],
        "rear_hip_bfc_deg": [(i_bfc, [neck_i, pelv_i, r_hip_i, r_kne_i])],
        "pelvic_drop_ffc_deg": [(i_ffc, [lhip_i, rhip_i])],
        "non_bowling_arm_elev_ffc_deg": [(i_ffc, [nb_sho_i, nb_elb_i])],
        "peak_arm_angular_vel_dps": [(i_br, [b_sho_i, b_elb_i, b_wri_i])],
        "release_height_m": [(i_ffc, [f_ank_i]), (i_br, [b_wri_i, f_ank_i])],
        "follow_through_drift_deg": [(i_br, [pelv_i]), (ft_end, [pelv_i])],
    }

    vis_arr = (
        joint_in_frame
        if (joint_in_frame is not None and joint_in_frame.shape[0] == n)
        else np.ones((n, 24), dtype=bool)
    )

    metric_joint_unc: Dict[str, Tuple[float, ...]] = {}
    metric_in_frame: Dict[str, bool] = {}

    for m_key, frame_joint_pairs in req_spec.items():
        all_vis = True
        uncs: List[float] = []
        for f_i, j_list in frame_joint_pairs:
            lo = max(0, f_i - 1)
            hi = min(n, f_i + 2)
            for ji in j_list:
                if not bool(vis_arr[f_i, ji]) and float(np.mean(vis_arr[lo:hi, ji])) < 0.65:
                    all_vis = False
                uncs.append(float(np.median(unc_arr[lo:hi, ji])))
        metric_in_frame[m_key] = all_vis
        metric_joint_unc[m_key] = tuple(uncs)

    series = {
        "front_knee_deg": front_knee_deg,
        "trunk_flex_deg": trunk_flex_deg,
        "thorax_lat_flex_deg": thorax_lat_flex_deg,
        "pelvis_yaw_deg": pelvis_yaw_deg,
        "shoulder_yaw_deg": shoulder_yaw_deg,
        "hip_shoulder_sep_deg": hip_shoulder_sep_deg,
        "pelvic_drop_deg": pelvic_drop_deg,
        "rear_hip_deg": rear_hip_deg,
        "nb_arm_elev_deg": nb_arm_elev_deg,
        "bowling_arm_phi_deg": b_arm_phi_anchored,
        "bowling_arm_omega_dps": b_arm_omega_dps,
        "bowling_elbow_deg": bowling_elbow_deg,
    }

    return KinematicsResult(
        scalars=scalars,
        local_rates_per_sec=local_rates,
        metric_joint_uncertainties_m=metric_joint_unc,
        metric_joints_in_frame=metric_in_frame,
        action_class=action_class,
        series=series,
        estimated_stature_m=stature_m,
        joints_world_smoothed=j,
        joint_uncertainties_array=unc_arr,
        interp_gap_frames=interp_gap_frames,
        bowling_arm=bowling_arm,
        fps=fps,
        events=events,
        ft_disp_x_m=ft_disp_x,
        savgol_window_s=savgol_window_s,
    )
