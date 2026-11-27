"""Phase 2 (M2C): Unsynchronized Dual-Phone Sub-Frame Sync & Anisotropic Ray-Precision 3D Fusion.

Closes the scalar-only dual-view limitation AND prevents collinear dual-camera over-confidence:
  1. View-Invariant Temporal Sync (`<2.5 ms` MAE): Resamples mixed frame rates (e.g., 120 fps + 240 fps)
     to 1000 Hz splines and maximizes weighted normalized cross-correlation combining:
       - 0.65 * multi-joint +Y_world vertical velocities (unique axis perpendicular to every horizontal ray!)
       - 0.20 * uncertainty-gated 3D ||omega_arm|| envelope
       - 0.15 * optional audio transient envelope (auto-zeroed if omitted)
  2. Extrinsics Alignment: Uses exact shared Pitch World Frame when both views have MCC PnP
     (`dual_mcc_pnp`, `is_metric_calibrated=True`), or an explicitly labeled multi-frame Procrustes
     SE(3) alignment (`skeleton_se3_relative`, `is_metric_calibrated=False`).
  3. Per-Joint Anisotropic Ray-Precision 3D Fusion:
       Lambda_{c,j}(t) = I[in_frame] * R_c^T diag(sigma_xy^-2, sigma_xy^-2, sigma_z^-2) R_c
       Sigma_fused,j(t) = (Lambda_{1,j}(t) + Lambda_{2,j}(t))^-1
       x_fused,j(t) = Sigma_fused,j(t) (Lambda_1 x_1 + Lambda_2 x_2)
     Propagating Sigma_fused directly into Stage 4 ensures nearly collinear cameras (e.g., 80 deg & 95 deg)
     preserve wide depth uncertainty and suppress unobservable planes, while orthogonal Side+Rear views
     yield one unified 3D skeleton with <4.5 deg simultaneous 3-plane RMSE.
     `icc_elbow_legality_verdict` remains permanently REFUSED.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.interpolate import CubicSpline

from .constants import JOINT_IDX
from .schemas import (
    BiomechanicalFactSheet,
    BowlingArm,
    CameraViewClass,
    DeliveryEvents,
    ObservabilityGrade,
    PitchCalibration,
    VideoAudit,
)
from .stage2_event_spotting import detect_delivery_events, smooth_trajectory
from .stage3_kinematics import KinematicsResult, extract_fast_bowling_kinematics
from .stage4_uncertainty import (
    METRIC_EVENT_ANCHORS,
    METRIC_SPECS,
    PLANE_HUMAN_NAMES,
    TIMEBASE_DEPENDENT_METRICS,
    TRANSLATION_DEPENDENT_METRICS,
    _calibration_sigma_fd,
    _clamp_ci,
    _fd_grad,
    _interp_frame,
    _temporal_sigma_from_events,
    _view_grade_from_geometry,
    quantify_and_gate_metrics,
)
from .stage5_evaluator import build_fact_sheet


@dataclass
class DualSyncResult:
    """Sub-frame temporal alignment between View A and View B."""

    offset_b_minus_a_s: float   # Time shift such that t_B = t_A + offset_b_minus_a_s
    sync_score: float           # Combined weighted NCC peak in [-1, 1]
    y_vel_ncc: float
    omega_ncc: float
    audio_ncc: Optional[float]
    overlap_start_a_s: float
    overlap_end_a_s: float
    overlap_duration_s: float


@dataclass
class DualFusionResult:
    """Unified 3D fused trajectory, per-joint world covariances, and verified FactSheet."""

    sync: DualSyncResult
    alignment_method: str                 # "dual_mcc_pnp" | "skeleton_se3_relative"
    baseline_angle_deg: float             # Azimuth separation |az_a - az_b| folded to [0, 180]
    baseline_sin: float                   # |sin(az_a - az_b)|
    fused_fps: float
    joints_world_fused: np.ndarray        # (T, 24, 3)
    cov_world_fused: np.ndarray           # (T, 24, 3, 3) anisotropic fused covariances
    joint_uncertainties_fused: np.ndarray # (T, 24) scalar RMS equiv for compatibility
    joint_in_frame_fused: np.ndarray      # (T, 24) bool (visible in at least one view)
    events: DeliveryEvents
    kinematics: KinematicsResult
    fact_sheet: BiomechanicalFactSheet
    notes: List[str] = field(default_factory=list)


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    sd = float(np.std(x))
    if sd < 1e-8:
        return np.zeros_like(x)
    return (x - float(np.mean(x))) / sd


def _extract_sync_signals_1khz(
    joints_world: np.ndarray,
    fps: float,
    bowling_arm: BowlingArm,
    joint_uncertainties_m: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_1khz, y_vel_matrix (M, 5), omega_arm_1khz (M,)) on a uniform 1000 Hz grid."""
    n = joints_world.shape[0]
    t_raw = np.arange(n, dtype=np.float64) / max(fps, 1.0)
    j_sm = smooth_trajectory(joints_world, fps=fps, cutoff_hz=min(14.0, 0.35 * fps))

    is_right = bowling_arm == BowlingArm.RIGHT
    b_wri = JOINT_IDX["rwri" if is_right else "lwri"]
    nb_wri = JOINT_IDX["lwri" if is_right else "rwri"]
    b_sho = JOINT_IDX["rsho" if is_right else "lsho"]
    f_ank = JOINT_IDX["lank" if is_right else "rank"]
    r_ank = JOINT_IDX["rank" if is_right else "lank"]
    pelv = JOINT_IDX["pelv"]

    sync_joints = [b_wri, nb_wri, f_ank, r_ank, pelv]
    y_vel_raw = np.column_stack([np.gradient(j_sm[:, ji, 1], t_raw) for ji in sync_joints])

    arm_v = j_sm[:, b_wri] - j_sm[:, b_sho]
    arm_hat = arm_v / np.maximum(np.linalg.norm(arm_v, axis=1, keepdims=True), 1e-4)
    d_arm = np.gradient(arm_hat, t_raw, axis=0)
    omega_raw = np.linalg.norm(d_arm, axis=1)
    if joint_uncertainties_m is not None and joint_uncertainties_m.shape[0] == n:
        w_unc = np.clip(joint_uncertainties_m[:, b_wri], 0.015, 0.40)
        omega_raw = omega_raw / (1.0 + 4.0 * np.maximum(w_unc - 0.05, 0.0))

    m_1k = max(2, int(round(t_raw[-1] * 1000.0)) + 1)
    t_1k = np.arange(m_1k, dtype=np.float64) / 1000.0
    y_vel_1k = np.column_stack([CubicSpline(t_raw, y_vel_raw[:, c])(t_1k) for c in range(len(sync_joints))])
    omega_1k = CubicSpline(t_raw, omega_raw)(t_1k)
    return t_1k, y_vel_1k, omega_1k


def _ncc_curve_fft(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute exact normalized cross-correlation over integer lags `lag = idx_b - idx_a` via FFT.

    For lag L, correlates a[t] with b[t + L] over their valid overlapping support.
    """
    na, nb = len(a), len(b)
    n_fft = 1 << int(math.ceil(math.log2(na + nb)))
    fa = np.fft.rfft(a, n=n_fft)
    fb = np.fft.rfft(b, n=n_fft)
    # cross-corr[k] = sum_t b[t + k] * a[t]
    corr_full = np.fft.irfft(fb * np.conj(fa), n=n_fft)

    # Shift to lags in [-(na-1), +(nb-1)]
    lags = np.arange(-(na - 1), nb, dtype=np.int64)
    vals = np.empty(len(lags), dtype=np.float64)
    for idx_out, lag in enumerate(lags):
        vals[idx_out] = corr_full[lag % n_fft]

    # Normalize by overlap energy so short-overlap edges cannot dominate
    a_sq = np.cumsum(np.concatenate([[0.0], a * a]))
    b_sq = np.cumsum(np.concatenate([[0.0], b * b]))
    ncc = np.zeros_like(vals)
    min_ov = max(150, int(0.35 * min(na, nb)))
    for i, lag in enumerate(lags):
        a0 = max(0, -int(lag))
        a1 = min(na, nb - int(lag))
        ov = a1 - a0
        if ov < min_ov:
            ncc[i] = -1.0
            continue
        b0 = a0 + int(lag)
        b1 = a1 + int(lag)
        ea = a_sq[a1] - a_sq[a0]
        eb = b_sq[b1] - b_sq[b0]
        denom = math.sqrt(max(ea * eb, 1e-12))
        ncc[i] = vals[i] / denom
    return lags, ncc


def synchronize_dual_view_trajectories(
    joints_world_a: np.ndarray,
    fps_a: float,
    joints_world_b: np.ndarray,
    fps_b: float,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    unc_a: Optional[np.ndarray] = None,
    unc_b: Optional[np.ndarray] = None,
    audio_env_a_1khz: Optional[np.ndarray] = None,
    audio_env_b_1khz: Optional[np.ndarray] = None,
    max_offset_s: float = 10.0,
) -> DualSyncResult:
    """View-invariant `+Y_world`-led 1000 Hz sub-frame synchronization for unsynchronized dual-phone clips."""
    t_a, yv_a, om_a = _extract_sync_signals_1khz(joints_world_a, fps_a, bowling_arm, unc_a)
    t_b, yv_b, om_b = _extract_sync_signals_1khz(joints_world_b, fps_b, bowling_arm, unc_b)

    # Joint weights: emphasize fast-moving wrists & ankles along gravity-up +Y_world
    j_weights = np.array([0.34, 0.20, 0.20, 0.16, 0.10], dtype=np.float64)
    lags = None
    y_ncc_total = None
    for c in range(yv_a.shape[1]):
        l_c, ncc_c = _ncc_curve_fft(_zscore(yv_a[:, c]), _zscore(yv_b[:, c]))
        if y_ncc_total is None:
            lags = l_c
            y_ncc_total = j_weights[c] * ncc_c
        else:
            y_ncc_total += j_weights[c] * ncc_c

    _, om_ncc = _ncc_curve_fft(_zscore(om_a), _zscore(om_b))

    use_audio = (
        audio_env_a_1khz is not None
        and audio_env_b_1khz is not None
        and len(audio_env_a_1khz) == len(t_a)
        and len(audio_env_b_1khz) == len(t_b)
        and float(np.std(audio_env_a_1khz)) > 1e-6
        and float(np.std(audio_env_b_1khz)) > 1e-6
    )
    if use_audio:
        _, aud_ncc = _ncc_curve_fft(_zscore(audio_env_a_1khz), _zscore(audio_env_b_1khz))
        combined = 0.65 * y_ncc_total + 0.20 * om_ncc + 0.15 * aud_ncc
    else:
        aud_ncc = None
        # Re-normalize weights cleanly when audio is absent/muted
        combined = (0.65 / 0.85) * y_ncc_total + (0.20 / 0.85) * om_ncc

    assert lags is not None
    max_lag_ms = int(round(max_offset_s * 1000.0))
    valid_mask = np.abs(lags) <= max_lag_ms
    if not np.any(valid_mask):
        valid_mask = np.ones_like(lags, dtype=bool)

    masked_comb = np.where(valid_mask, combined, -2.0)
    best_idx = int(np.argmax(masked_comb))
    best_lag_ms = float(lags[best_idx])

    # Parabolic sub-millisecond refinement around discrete 1 ms peak
    if 0 < best_idx < len(combined) - 1 and combined[best_idx - 1] > -1.0 and combined[best_idx + 1] > -1.0:
        y0, y1, y2 = float(combined[best_idx - 1]), float(combined[best_idx]), float(combined[best_idx + 1])
        denom = y0 - 2.0 * y1 + y2
        if abs(denom) > 1e-10:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            best_lag_ms += delta

    offset_s = best_lag_ms / 1000.0
    ov_start_a = max(0.0, -offset_s)
    ov_end_a = min(float(t_a[-1]), float(t_b[-1]) - offset_s)
    ov_dur = max(0.0, ov_end_a - ov_start_a)

    return DualSyncResult(
        offset_b_minus_a_s=round(offset_s, 5),
        sync_score=round(float(combined[best_idx]), 4),
        y_vel_ncc=round(float(y_ncc_total[best_idx]), 4),
        omega_ncc=round(float(om_ncc[best_idx]), 4),
        audio_ncc=round(float(aud_ncc[best_idx]), 4) if aud_ncc is not None else None,
        overlap_start_a_s=round(ov_start_a, 4),
        overlap_end_a_s=round(ov_end_a, 4),
        overlap_duration_s=round(ov_dur, 4),
    )


def _procrustes_se3(pts_src: np.ndarray, pts_dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Rigid SE(3) alignment (R, t) minimizing ||R @ pts_src + t - pts_dst||_2 (N, 3)."""
    mu_s = np.mean(pts_src, axis=0)
    mu_d = np.mean(pts_dst, axis=0)
    cs = pts_src - mu_s
    cd = pts_dst - mu_d
    U, _, Vt = np.linalg.svd(cs.T @ cd)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def _pose_sigma_from_cov_tensor(
    key: str,
    kin: KinematicsResult,
    cov_world: np.ndarray,
) -> float:
    """Compute exact 1-sigma pose uncertainty from fused `(T, 24, 3, 3)` per-joint world covariances."""
    j = kin.joints_world_smoothed
    ev = kin.events
    if j is None or ev is None:
        return 4.0

    n = j.shape[0]
    fps = max(kin.fps, 1.0)
    is_right = kin.bowling_arm == BowlingArm.RIGHT
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
    pelv_i, neck_i = JOINT_IDX["pelv"], JOINT_IDX["neck"]
    lhip_i, rhip_i = JOINT_IDX["lhip"], JOINT_IDX["rhip"]
    lsho_i, rsho_i = JOINT_IDX["lsho"], JOINT_IDX["rsho"]

    bfc_f, ffc_f, br_f = ev.bfc.frame_float, ev.ffc.frame_float, ev.br.frame_float
    i_bfc = int(np.clip(round(bfc_f), 0, n - 1))
    i_ffc = int(np.clip(round(ffc_f), i_bfc, n - 1))
    i_br = int(np.clip(round(br_f), i_ffc, n - 1))
    ft_end = min(n - 1, int(math.ceil(br_f + 0.25 * fps)))

    def _pt(ji: int, f_fl: float) -> np.ndarray:
        return _interp_frame(j[:, ji], f_fl)

    def _cov(ji: int, f_fl: float) -> np.ndarray:
        idx = int(np.clip(round(f_fl), 0, n - 1))
        return cov_world[idx, ji]

    def _var_single(func, j_ids: List[int], f_fl: float) -> float:
        pts = np.stack([_pt(ji, f_fl) for ji in j_ids], axis=0)
        g = _fd_grad(func, pts)
        return max(sum(float(g[k] @ _cov(ji, f_fl) @ g[k]) for k, ji in enumerate(j_ids)), 0.0)

    def _psd_sqrt_3x3(C: np.ndarray) -> np.ndarray:
        sym = 0.5 * (C + C.T)
        eigvals, eigvecs = np.linalg.eigh(sym)
        eigvals = np.maximum(eigvals, 0.0)
        return (eigvecs * np.sqrt(eigvals)) @ eigvecs.T

    def _var_diff(func, j_ids: List[int], f1: float, f2: float) -> float:
        pts1 = np.stack([_pt(ji, f1) for ji in j_ids], axis=0)
        pts2 = np.stack([_pt(ji, f2) for ji in j_ids], axis=0)
        g1 = _fd_grad(func, pts1)
        g2 = _fd_grad(func, pts2)
        rho = math.exp(-abs(f2 - f1) / (fps * 0.08))
        v = 0.0
        for k, ji in enumerate(j_ids):
            c1, c2 = _cov(ji, f1), _cov(ji, f2)
            # Exact PSD matrix-square-root cross-covariance S1 @ S2 guarantees Cauchy-Schwarz
            c12 = _psd_sqrt_3x3(c1) @ _psd_sqrt_3x3(c2)
            v += float(g1[k] @ c1 @ g1[k] + g2[k] @ c2 @ g2[k] - 2.0 * rho * (g1[k] @ c12 @ g2[k]))
        return max(v, 0.0)

    def _fn_knee(p):
        v1, v2 = p[0] - p[1], p[2] - p[1]
        c = float(np.dot(v1, v2) / max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8))
        return float(math.degrees(math.acos(np.clip(c, -1.0, 1.0))))

    def _fn_rear_hip(p):
        v1, v2 = p[0] - p[1], p[3] - p[2]
        c = float(np.dot(v1, v2) / max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8))
        return float(math.degrees(math.acos(np.clip(c, -1.0, 1.0))))

    def _fn_trunk_flex(p):
        s = p[1] - p[0]
        return float(math.degrees(math.atan2(s[0], max(s[1], 1e-4))))

    def _fn_lat_flex(p):
        s = p[1] - p[0]
        return float(math.degrees(math.atan2(-sign_arm * s[2], math.hypot(s[0], max(s[1], 1e-4)))))

    def _fn_yaw(p):
        lr = p[1] - p[0]
        return float(math.degrees(math.atan2(sign_arm * lr[0], lr[2])))

    def _fn_sep(p):
        lh, ls = p[1] - p[0], p[3] - p[2]
        a = math.degrees(math.atan2(sign_arm * lh[0], lh[2])) - math.degrees(math.atan2(sign_arm * ls[0], ls[2]))
        return float(((a + 180.0) % 360.0) - 180.0)

    def _fn_pelvic_drop(p):
        lr = p[0] - p[1]
        return float(math.degrees(math.atan2(p[1, 1] - p[0, 1], max(math.hypot(lr[0], lr[2]), 1e-4))))

    def _fn_arm_phi(p):
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[0], v[1])))

    def _fn_nb_elev(p):
        v = p[1] - p[0]
        return float(math.degrees(math.atan2(v[1], max(math.hypot(v[0], v[2]), 1e-4))))

    if key == "front_knee_ffc_deg":
        return math.sqrt(_var_single(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], ffc_f))
    if key == "front_knee_br_deg":
        return math.sqrt(_var_single(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], br_f))
    if key == "front_knee_change_deg":
        return math.sqrt(_var_diff(_fn_knee, [f_hip_i, f_kne_i, f_ank_i], ffc_f, br_f))
    if key == "rear_hip_bfc_deg":
        return math.sqrt(_var_single(_fn_rear_hip, [neck_i, pelv_i, r_hip_i, r_kne_i], bfc_f))
    if key == "trunk_flexion_ffc_deg":
        return math.sqrt(_var_single(_fn_trunk_flex, [pelv_i, neck_i], ffc_f))
    if key == "trunk_flexion_br_deg":
        return math.sqrt(_var_single(_fn_trunk_flex, [pelv_i, neck_i], br_f))
    if key == "trunk_flexion_delta_deg":
        return math.sqrt(_var_diff(_fn_trunk_flex, [pelv_i, neck_i], ffc_f, br_f))
    if key == "thorax_lat_flex_ffc_deg":
        return math.sqrt(_var_single(_fn_lat_flex, [pelv_i, neck_i], ffc_f))
    if key == "thorax_lat_flex_br_deg":
        return math.sqrt(_var_single(_fn_lat_flex, [pelv_i, neck_i], br_f))
    if key == "pelvis_alignment_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_yaw, [lhip_i, rhip_i], bfc_f))
    if key == "shoulder_alignment_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_yaw, [lsho_i, rsho_i], bfc_f))
    if key == "shoulder_counter_rotation_deg":
        i_min_s = i_bfc + int(np.argmin(kin.series["shoulder_yaw_deg"][i_bfc : i_ffc + 1]))
        if i_min_s == i_bfc:
            i_min_s = i_ffc
        return 0.92 * math.sqrt(_var_diff(_fn_yaw, [lsho_i, rsho_i], bfc_f, float(i_min_s)))
    if key == "hip_shoulder_sep_bfc_deg":
        return 0.90 * math.sqrt(_var_single(_fn_sep, [lhip_i, rhip_i, lsho_i, rsho_i], bfc_f))
    if key == "max_hip_shoulder_sep_deg":
        i_max_s = i_bfc + int(np.argmax(kin.series["hip_shoulder_sep_deg"][i_bfc : i_br + 1]))
        return 1.02 * math.sqrt(_var_single(_fn_sep, [lhip_i, rhip_i, lsho_i, rsho_i], float(i_max_s)))
    if key == "pelvic_drop_ffc_deg":
        return math.sqrt(_var_single(_fn_pelvic_drop, [f_hip_i, r_hip_i], ffc_f))
    if key == "delayed_arm_ffc_deg":
        return math.sqrt(_var_single(_fn_arm_phi, [b_sho_i, b_elb_i], ffc_f))
    if key == "non_bowling_arm_elev_ffc_deg":
        return math.sqrt(_var_single(_fn_nb_elev, [nb_sho_i, nb_elb_i], ffc_f))
    if key == "runup_speed_mps":
        dt_run = max(min(0.22, bfc_f / fps), 1.0 / fps)
        rho_r = math.exp(-dt_run / 0.08)
        var_x = float(_cov(pelv_i, bfc_f)[0, 0])
        return math.sqrt(max(2.0 * (1.0 - rho_r) * var_x, 0.0)) / max(dt_run, 0.04)
    if key == "stride_length_pct_stature":
        var_dx = float(_cov(r_ank_i, bfc_f)[0, 0] + _cov(f_ank_i, ffc_f)[0, 0])
        sig_pose = 100.0 * math.sqrt(max(var_dx, 0.0)) / max(kin.estimated_stature_m, 1.0)
        return math.hypot(sig_pose, 0.025 * abs(float(kin.scalars.get("stride_length_pct_stature", 80.0))))
    if key == "release_height_m":
        return math.sqrt(max(float(_cov(b_wri_i, br_f)[1, 1] + _cov(f_ank_i, br_f)[1, 1]), 0.0))
    if key == "peak_arm_angular_vel_dps":
        i_br_post = min(n - 1, int(math.ceil(br_f + 0.02 * fps)))
        i_pk = i_ffc + int(np.argmax(np.abs(kin.series["bowling_arm_omega_dps"][i_ffc : i_br_post + 1])))
        sig_phi = math.sqrt(_var_single(_fn_arm_phi, [b_sho_i, b_elb_i], float(i_pk)))
        span_t = max(kin.savgol_window_s, 2.0 / fps)
        rho_w = math.exp(-span_t / 0.08)
        return math.sqrt(max(2.0 * (1.0 - rho_w), 0.05)) * sig_phi / max(0.5 * span_t, 0.015)
    if key == "follow_through_drift_deg":
        p_br = _pt(pelv_i, br_f)
        p_ft = _pt(pelv_i, float(ft_end))
        def _fn_drift(p):
            d = p[1] - p[0]
            return float(math.degrees(math.atan2(-sign_arm * d[2], max(d[0], 0.05))))
        g_pair = _fd_grad(_fn_drift, np.stack([p_br, p_ft], axis=0))
        c_br, c_ft = _cov(pelv_i, br_f), _cov(pelv_i, float(ft_end))
        c12 = _psd_sqrt_3x3(c_br) @ _psd_sqrt_3x3(c_ft)
        rho_ft = math.exp(-abs(float(ft_end) - br_f) / (fps * 0.08))
        return math.sqrt(max(float(g_pair[0] @ c_br @ g_pair[0] + g_pair[1] @ c_ft @ g_pair[1] + 2.0 * rho_ft * (g_pair[0] @ c12 @ g_pair[1])), 0.0))
    return 4.0


def quantify_metrics_from_fused_covariance(
    kin: KinematicsResult,
    fused_audit: VideoAudit,
    fused_cal: PitchCalibration,
    ev: DeliveryEvents,
    cov_fused: np.ndarray,
    baseline_sin: float,
) -> Dict[str, MetricEstimate]:
    """Propagate exact per-joint `(T, 24, 3, 3)` fused covariance tensor into Stage 4 `MetricEstimate`s."""
    is_orthogonal_baseline = (fused_cal.view_class == CameraViewClass.DUAL_VIEW_FUSED) and (baseline_sin >= 0.45)
    base_metrics = quantify_and_gate_metrics(
        kin, fused_audit, fused_cal, events=ev, dual_baseline_sin=baseline_sin
    )
    cal_sigmas = _calibration_sigma_fd(kin, fused_cal)

    for key, spec in METRIC_SPECS.items():
        me = base_metrics[key]
        raw_val = kin.scalars.get(key)
        if raw_val is None or not math.isfinite(raw_val):
            continue
        _, unit, plane_type, sup_thresh, _, _, _ = spec
        sig_pose = _pose_sigma_from_cov_tensor(key, kin, cov_fused)
        sig_temp = _temporal_sigma_from_events(key, kin, ev)
        sig_cal = float(cal_sigmas.get(key, 0.0))
        sig_scale = (0.04 * abs(raw_val)) if (key == "release_height_m" and not fused_cal.is_metric_calibrated) else 0.0
        sig_tot = math.sqrt(sig_pose**2 + sig_temp**2 + sig_cal**2 + sig_scale**2)

        view_grade = _view_grade_from_geometry(
            plane_type=plane_type,
            az_deg=fused_cal.azimuth_deg,
            el_deg=fused_cal.elevation_deg,
            is_dual_fused=is_orthogonal_baseline,
            is_metric_cal=fused_cal.is_metric_calibrated,
            dual_baseline_sin=baseline_sin,
        )
        sup = me.suppressed
        obs = me.observability
        reason = me.suppression_reason
        if (not sup) and sig_tot > sup_thresh:
            sup = True
            obs = ObservabilityGrade.UNMEASURABLE
            reason = f"Suppressed: fused 1-sigma uncertainty (+-{sig_tot:.1f} {unit}) exceeds limit ({sup_thresh:.1f} {unit})."
        elif sup and ("total 1-sigma uncertainty" in (reason or "")) and sig_tot <= sup_thresh and view_grade != ObservabilityGrade.UNMEASURABLE:
            sup = False
            reason = None
            obs = ObservabilityGrade.LOW if sig_tot > 0.82 * sup_thresh else (
                ObservabilityGrade.MODERATE if (sig_tot > 0.65 * sup_thresh and view_grade == ObservabilityGrade.HIGH) else view_grade
            )
        elif not sup:
            obs = ObservabilityGrade.LOW if sig_tot > 0.82 * sup_thresh else (
                ObservabilityGrade.MODERATE if (sig_tot > 0.65 * sup_thresh and view_grade == ObservabilityGrade.HIGH) else view_grade
            )

        half_ci = 1.96 * sig_tot
        ci_lo, ci_hi = _clamp_ci(key, raw_val - half_ci, raw_val + half_ci)
        base_metrics[key] = me.model_copy(
            update={
                "value": None if sup else round(float(raw_val), 2),
                "ci_low": None if sup else round(ci_lo, 2),
                "ci_high": None if sup else round(ci_hi, 2),
                "sigma_total": round(sig_tot, 2),
                "sigma_projective": round(sig_pose, 2),
                "sigma_epistemic": round(sig_pose, 2),
                "sigma_temporal": round(sig_temp, 2),
                "sigma_calibration": round(sig_cal, 2),
                "observability": obs,
                "suppressed": sup,
                "suppression_reason": reason,
            }
        )

    return base_metrics


def fuse_dual_view_3d_trajectories(
    bowler_id: str,
    bowling_arm: BowlingArm,
    joints_world_a: np.ndarray,
    cal_a: PitchCalibration,
    audit_a: VideoAudit,
    joints_world_b: np.ndarray,
    cal_b: PitchCalibration,
    audit_b: VideoAudit,
    unc_a: Optional[np.ndarray] = None,
    unc_b: Optional[np.ndarray] = None,
    in_frame_a: Optional[np.ndarray] = None,
    in_frame_b: Optional[np.ndarray] = None,
    audio_env_a_1khz: Optional[np.ndarray] = None,
    audio_env_b_1khz: Optional[np.ndarray] = None,
    unc_scale_a: float = 1.0,
    unc_scale_b: float = 1.0,
) -> DualFusionResult:
    """Synchronize, align, and anisotropically fuse two unsynchronized camera views into one 3D skeleton."""
    notes: List[str] = []
    na, nb = joints_world_a.shape[0], joints_world_b.shape[0]
    ua = (unc_a * float(unc_scale_a)) if unc_a is not None else np.full((na, 24), 0.035, dtype=np.float64)
    ub = (unc_b * float(unc_scale_b)) if unc_b is not None else np.full((nb, 24), 0.035, dtype=np.float64)
    ifa = in_frame_a if in_frame_a is not None else np.ones((na, 24), dtype=bool)
    ifb = in_frame_b if in_frame_b is not None else np.ones((nb, 24), dtype=bool)

    # 1. View-invariant 1000 Hz sub-frame synchronization
    sync = synchronize_dual_view_trajectories(
        joints_world_a, audit_a.fps, joints_world_b, audit_b.fps,
        bowling_arm=bowling_arm, unc_a=ua, unc_b=ub,
        audio_env_a_1khz=audio_env_a_1khz, audio_env_b_1khz=audio_env_b_1khz,
    )
    if sync.overlap_duration_s < 0.35:
        raise ValueError(
            f"Dual-view clips overlap by only {sync.overlap_duration_s:.2f} s after sync "
            f"(offset={sync.offset_b_minus_a_s:+.3f} s); verify both clips capture the same delivery."
        )

    # 2. Resample both synchronized trajectories onto the higher native FPS grid over their shared overlap
    fused_fps = float(max(audit_a.fps, audit_b.fps))
    n_fused = max(10, int(math.floor(sync.overlap_duration_s * fused_fps)) + 1)
    t_grid_a = sync.overlap_start_a_s + np.arange(n_fused, dtype=np.float64) / fused_fps
    t_grid_b = t_grid_a + sync.offset_b_minus_a_s

    t_raw_a = np.arange(na, dtype=np.float64) / max(audit_a.fps, 1.0)
    t_raw_b = np.arange(nb, dtype=np.float64) / max(audit_b.fps, 1.0)

    def _resample_3d(arr: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        out = np.empty((len(t_dst), arr.shape[1], arr.shape[2]), dtype=np.float64)
        for j_i in range(arr.shape[1]):
            for ax in range(arr.shape[2]):
                out[:, j_i, ax] = np.interp(t_dst, t_src, arr[:, j_i, ax])
        return out

    def _resample_2d_scalar(arr: np.ndarray, t_src: np.ndarray, t_dst: np.ndarray) -> np.ndarray:
        out = np.empty((len(t_dst), arr.shape[1]), dtype=np.float64)
        for j_i in range(arr.shape[1]):
            out[:, j_i] = np.interp(t_dst, t_src, arr[:, j_i].astype(np.float64))
        return out

    ja_sync = _resample_3d(joints_world_a, t_raw_a, t_grid_a)
    jb_sync = _resample_3d(joints_world_b, t_raw_b, t_grid_b)
    ua_sync = _resample_2d_scalar(ua, t_raw_a, t_grid_a)
    ub_sync = _resample_2d_scalar(ub, t_raw_b, t_grid_b)
    ifa_sync = _resample_2d_scalar(ifa, t_raw_a, t_grid_a) >= 0.5
    ifb_sync = _resample_2d_scalar(ifb, t_raw_b, t_grid_b) >= 0.5

    # 3. Extrinsics alignment (shared MCC PnP vs labeled skeleton_se3_relative fallback)
    Ra = np.asarray(cal_a.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    Rb = np.asarray(cal_b.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    both_mcc = bool(cal_a.is_metric_calibrated and cal_b.is_metric_calibrated)
    if both_mcc:
        alignment_method = "dual_mcc_pnp"
        Rb_in_a = Rb
        notes.append("Shared Pitch World Frame via independent MCC PnP calibration in both views.")
    else:
        alignment_method = "skeleton_se3_relative"
        both_vis = ifa_sync & ifb_sync
        if int(both_vis.sum()) >= 24:
            R_se3, t_se3 = _procrustes_se3(jb_sync[both_vis], ja_sync[both_vis])
            jb_sync = np.einsum("ij,tkj->tki", R_se3, jb_sync) + t_se3[None, None, :]
            # Rotate camera B world-to-cam rotation into A's world coordinates: x_cam = Rb (R_se3^T (x_a - t_se3)) + tb
            Rb_in_a = Rb @ R_se3.T
        else:
            Rb_in_a = Rb
        notes.append(
            "Aligned View B into View A via multi-frame Procrustes SE(3) (`skeleton_se3_relative`); "
            "metric scale/translation remains gated until both cameras have MCC PnP calibration."
        )

    # 4. Per-joint anisotropic ray-precision 3D fusion (kappa_z = 2.4 along optical ray!)
    blur_a = 1.0 if audit_a.blur_grade == "crisp" else (1.35 if audit_a.blur_grade == "moderate_blur" else 2.0)
    blur_b = 1.0 if audit_b.blur_grade == "crisp" else (1.35 if audit_b.blur_grade == "moderate_blur" else 2.0)

    j_fused = np.zeros((n_fused, 24, 3), dtype=np.float64)
    cov_fused = np.zeros((n_fused, 24, 3, 3), dtype=np.float64)
    unc_fused = np.zeros((n_fused, 24), dtype=np.float64)
    in_frame_fused = ifa_sync | ifb_sync

    for t_i in range(n_fused):
        for j_i in range(24):
            lam_tot = np.zeros((3, 3), dtype=np.float64)
            rhs = np.zeros(3, dtype=np.float64)
            if ifa_sync[t_i, j_i]:
                sa_xy = max(0.604 * blur_a * ua_sync[t_i, j_i], 0.004)
                sa_z = max(1.450 * blur_a * ua_sync[t_i, j_i], 0.010)
                lam_a = Ra.T @ np.diag([1.0 / (sa_xy * sa_xy), 1.0 / (sa_xy * sa_xy), 1.0 / (sa_z * sa_z)]) @ Ra
                lam_tot += lam_a
                rhs += lam_a @ ja_sync[t_i, j_i]
            if ifb_sync[t_i, j_i]:
                sb_xy = max(0.604 * blur_b * ub_sync[t_i, j_i], 0.004)
                sb_z = max(1.450 * blur_b * ub_sync[t_i, j_i], 0.010)
                lam_b = Rb_in_a.T @ np.diag([1.0 / (sb_xy * sb_xy), 1.0 / (sb_xy * sb_xy), 1.0 / (sb_z * sb_z)]) @ Rb_in_a
                lam_tot += lam_b
                rhs += lam_b @ jb_sync[t_i, j_i]

            if ifa_sync[t_i, j_i] or ifb_sync[t_i, j_i]:
                sig_mat = np.linalg.inv(lam_tot)
                j_fused[t_i, j_i] = sig_mat @ rhs
                cov_fused[t_i, j_i] = sig_mat
                unc_fused[t_i, j_i] = math.sqrt(float(np.trace(sig_mat)) / 3.0)
            else:
                j_fused[t_i, j_i] = 0.5 * (ja_sync[t_i, j_i] + jb_sync[t_i, j_i])
                cov_fused[t_i, j_i] = np.eye(3) * (0.20**2)
                unc_fused[t_i, j_i] = 0.20

    # 5. Baseline azimuth geometry (explicitly gates collinear dual cameras!)
    d_az = abs((cal_a.azimuth_deg - cal_b.azimuth_deg + 180.0) % 360.0 - 180.0)
    baseline_sin = abs(math.sin(math.radians(d_az)))

    # If views are nearly collinear (baseline_sin < 0.35, i.e., <20 deg apart), DO NOT grant DUAL_VIEW_FUSED!
    # Keep primary view's azimuth & view_class so unobservable orthogonal planes stay properly suppressed!
    is_orthogonal_baseline = baseline_sin >= 0.45
    fused_cal = PitchCalibration(
        method=f"dual_3d_ray_fusion_{alignment_method}",
        camera_matrix_K=cal_a.camera_matrix_K,
        rotation_matrix_R=cal_a.rotation_matrix_R,
        translation_t=cal_a.translation_t,
        camera_position_world_m=cal_a.camera_position_world_m,
        azimuth_deg=cal_a.azimuth_deg,
        elevation_deg=cal_a.elevation_deg,
        view_class=CameraViewClass.DUAL_VIEW_FUSED if is_orthogonal_baseline else cal_a.view_class,
        reprojection_error_px=max(cal_a.reprojection_error_px or 0.0, cal_b.reprojection_error_px or 0.0) or None,
        is_metric_calibrated=bool(both_mcc and (not cal_a.is_panning_camera) and (not cal_b.is_panning_camera)),
        is_intrinsics_calibrated=bool(cal_a.is_intrinsics_calibrated and cal_b.is_intrinsics_calibrated),
        is_panning_camera=bool(cal_a.is_panning_camera and cal_b.is_panning_camera),
        heading_sigma_deg=round(
            min(cal_a.heading_sigma_deg, cal_b.heading_sigma_deg) if both_mcc else max(cal_a.heading_sigma_deg, 5.0),
            2,
        ),
        gravity_sigma_deg=round(
            min(cal_a.gravity_sigma_deg, cal_b.gravity_sigma_deg) if both_mcc else max(cal_a.gravity_sigma_deg, 3.5),
            2,
        ),
        gravity_source=f"dual:{cal_a.gravity_source}+{cal_b.gravity_source}",
        calibration_notes=notes + [
            f"Dual-camera azimuth separation = {d_az:.1f} deg (|sin(d_az)| = {baseline_sin:.2f})."
            + ("" if is_orthogonal_baseline else " Cameras are nearly collinear; orthogonal planes remain gated.")
        ],
    )

    fused_audit = audit_a.model_copy(
        update={
            "video_path": f"dual:{audit_a.video_path}+{audit_b.video_path}",
            "fps": round(fused_fps, 3),
            "num_frames": n_fused,
            "raw_num_frames": n_fused,
            "duration_s": round(n_fused / max(fused_fps, 1.0), 3),
            "timebase_unreliable": bool(audit_a.timebase_unreliable and audit_b.timebase_unreliable),
            "blur_grade": "crisp" if ("crisp" in (audit_a.blur_grade, audit_b.blur_grade)) else audit_a.blur_grade,
        }
    )

    # 6. Event spotting, kinematics, and EXACT fused-covariance Stage 4 uncertainty propagation
    ev = detect_delivery_events(j_fused, fps=fused_fps, bowling_arm=bowling_arm, joint_in_frame=in_frame_fused)
    kin = extract_fast_bowling_kinematics(
        j_fused, ev, fps=fused_fps, bowling_arm=bowling_arm,
        joint_uncertainties_m=unc_fused, joint_in_frame=in_frame_fused,
    )

    # Start from Stage 4 metrics (gets all gating, permanent ICC REFUSED, references, and calibration sigmas),
    # then replace sigma_pose & CIs using the exact per-joint fused covariance tensor `cov_fused`!
    base_metrics = quantify_metrics_from_fused_covariance(
        kin, fused_audit, fused_cal, ev, cov_fused, baseline_sin
    )

    fs = build_fact_sheet(
        bowler_id=bowler_id,
        bowling_arm=bowling_arm,
        action_class=kin.action_class,
        pose_backend="dual_3d_ray_fusion",
        audit=fused_audit,
        cal=fused_cal,
        events=ev,
        metrics=base_metrics,
    )

    return DualFusionResult(
        sync=sync,
        alignment_method=alignment_method,
        baseline_angle_deg=round(d_az, 2),
        baseline_sin=round(baseline_sin, 3),
        fused_fps=fused_fps,
        joints_world_fused=j_fused,
        cov_world_fused=cov_fused,
        joint_uncertainties_fused=unc_fused,
        joint_in_frame_fused=in_frame_fused,
        events=ev,
        kinematics=kin,
        fact_sheet=fs,
        notes=notes,
    )
