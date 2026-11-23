"""Stage 2: Multi-Cue Sub-Frame Temporal Event Spotting for Cricket Fast Bowling.

Detects canonical biomechanical phases from 3D Pitch World SMPL-24 trajectories with
strict causal ordering (BFC < FFC <= UAH < BR < FT), overhead arm-extension gating,
in-frame ankle visibility checks, and slow-motion / FPS-mismatch diagnostics.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from .schemas import BowlingArm, DeliveryEvents, EventTimestamp
from .synthetic_ground_truth import JOINT_IDX


def smooth_trajectory(joints: np.ndarray, fps: float, cutoff_hz: float = 15.0) -> np.ndarray:
    """Zero-lag 4th-order Butterworth biomechanical low-pass filter (Winter convention)."""
    n = joints.shape[0]
    if n < 15:
        return joints.copy()
    nyq = 0.5 * fps
    wn = min(cutoff_hz / max(nyq, 1e-3), 0.85)
    if wn <= 0.02 or wn >= 0.99:
        return joints.copy()
    b, a = butter(4, wn, btype="low")
    out = joints.copy()
    flat = out.reshape(n, -1)
    padlen = min(3 * max(len(a), len(b)), n - 1)
    for col in range(flat.shape[1]):
        if np.all(np.isfinite(flat[:, col])):
            flat[:, col] = filtfilt(b, a, flat[:, col], padlen=padlen)
    return flat.reshape(joints.shape)


def _parabolic_peak(series: np.ndarray, idx: int) -> float:
    """Quadratic sub-frame refinement around a local extremum."""
    if idx <= 0 or idx >= len(series) - 1:
        return float(idx)
    y0, y1, y2 = float(series[idx - 1]), float(series[idx]), float(series[idx + 1])
    denom = y0 - 2.0 * y1 + y2
    if abs(denom) < 1e-9:
        return float(idx)
    delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
    return float(idx) + delta


def _linear_zero_crossing(series: np.ndarray, idx_before: int) -> float:
    """Linear sub-frame interpolation where `series` crosses zero between idx_before and idx_before+1."""
    if idx_before < 0 or idx_before >= len(series) - 1:
        return float(max(0, min(idx_before, len(series) - 1)))
    y0, y1 = float(series[idx_before]), float(series[idx_before + 1])
    if abs(y1 - y0) < 1e-9:
        return float(idx_before)
    frac = float(np.clip(-y0 / (y1 - y0), 0.0, 1.0))
    return float(idx_before) + frac


def _make_event(
    frame_float: float,
    fps: float,
    n_frames: int,
    confidence: float,
    method: str,
) -> EventTimestamp:
    ff = float(np.clip(frame_float, 0.0, max(n_frames - 1, 0)))
    idx = int(np.clip(round(ff), 0, max(n_frames - 1, 0)))
    timing_unc_ms = 500.0 / max(fps, 1.0)
    return EventTimestamp(
        frame_float=round(ff, 3),
        frame_idx=idx,
        timestamp_s=round(ff / max(fps, 1.0), 4),
        timing_uncertainty_ms=round(timing_unc_ms, 2),
        confidence=round(float(np.clip(confidence, 0.05, 0.99)), 3),
        detection_method=method,
    )


def detect_delivery_events(
    joints_world: np.ndarray,
    fps: float,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    joint_in_frame: Optional[np.ndarray] = None,
) -> DeliveryEvents:
    """Detect BFC, FFC, UAH, BR, and FT sub-frame timestamps from Pitch World 3D joints (T, 24, 3).

    Enforces strict causal ordering BFC < FFC <= UAH < BR < FT and flags incomplete/truncated clips
    or slow-motion playback mismatches.
    """
    n = joints_world.shape[0]
    if n < 6:
        raise ValueError(f"Delivery clip too short for event spotting ({n} frames).")

    j = smooth_trajectory(joints_world, fps, cutoff_hz=min(18.0, 0.38 * fps))
    warnings: List[str] = []
    is_complete = True

    is_right = bowling_arm == BowlingArm.RIGHT
    b_sho_idx = JOINT_IDX["rsho" if is_right else "lsho"]
    b_elb_idx = JOINT_IDX["relb" if is_right else "lelb"]
    b_wri_idx = JOINT_IDX["rwri" if is_right else "lwri"]
    nb_wri_idx = JOINT_IDX["lwri" if is_right else "rwri"]
    f_ank_idx = JOINT_IDX["lank" if is_right else "rank"]
    r_ank_idx = JOINT_IDX["rank" if is_right else "lank"]

    b_sho = j[:, b_sho_idx]
    b_elb = j[:, b_elb_idx]
    b_wri = j[:, b_wri_idx]
    nb_wri = j[:, nb_wri_idx]
    pelv = j[:, JOINT_IDX["pelv"]]
    f_ank = j[:, f_ank_idx]
    r_ank = j[:, r_ank_idx]

    # 1. BALL RELEASE (BR):
    # Require overhead extended bowling arm (strictly rejecting flexed-elbow gather & sub-shoulder frames!)
    arm_vec = b_wri - b_sho
    arm_len = np.linalg.norm(arm_vec, axis=1)
    max_arm_len = max(float(np.percentile(arm_len, 95)), 0.35)
    hand_sep = np.linalg.norm(b_wri - nb_wri, axis=1)

    u_vec = b_sho - b_elb
    f_vec = b_wri - b_elb
    cos_elb = np.sum(u_vec * f_vec, axis=1) / np.maximum(
        np.linalg.norm(u_vec, axis=1) * np.linalg.norm(f_vec, axis=1), 1e-6
    )
    elbow_deg = np.degrees(np.arccos(np.clip(cos_elb, -1.0, 1.0)))

    arm_phi_deg = np.degrees(np.arctan2(arm_vec[:, 0], arm_vec[:, 1]))
    arm_phi_unwrap = np.degrees(np.unwrap(np.radians(arm_phi_deg)))
    arm_omega_dps = np.gradient(arm_phi_unwrap) * fps
    wrist_rel_y = b_wri[:, 1] - b_sho[:, 1]

    margin = max(1, int(0.04 * n))
    br_score = np.full(n, -1e9, dtype=np.float64)
    for t in range(margin, max(margin + 1, n - margin)):
        if wrist_rel_y[t] <= 0.14:
            continue
        ext_ratio = arm_len[t] / max_arm_len
        if elbow_deg[t] < 115.0 or ext_ratio < 0.72 or hand_sep[t] < 0.25:
            continue
        angle_prox = math.exp(-0.5 * ((arm_phi_deg[t] - 8.0) / 18.0) ** 2)
        vel_bonus = max(arm_omega_dps[t], 0.0) / 1500.0
        height_bonus = wrist_rel_y[t] / max(float(np.max(wrist_rel_y)), 0.3)
        ext_bonus = min(ext_ratio, 1.0)
        br_score[t] = 2.0 * angle_prox + 1.2 * min(vel_bonus, 2.0) + 1.5 * height_bonus + 1.0 * ext_bonus

    if float(np.max(br_score)) > -1e8:
        br_idx = int(np.argmax(br_score))
        if 1 <= br_idx < n - 1 and arm_omega_dps[br_idx] > 200.0:
            shifted = arm_phi_deg - 8.0
            if shifted[br_idx - 1] <= 0.0 <= shifted[br_idx]:
                br_ff = _linear_zero_crossing(shifted, br_idx - 1)
            elif shifted[br_idx] <= 0.0 <= shifted[br_idx + 1]:
                br_ff = _linear_zero_crossing(shifted, br_idx)
            else:
                br_ff = _parabolic_peak(br_score, br_idx)
        else:
            br_ff = _parabolic_peak(br_score, br_idx)
        br_conf = 0.88 if fps >= 100.0 else (0.74 if fps >= 48.0 else 0.58)
        br_method = "overhead_extended_circumduction_peak"
    else:
        # Relaxed fallback STILL requires overhead wrist (wrist_rel_y > 0.10) + extended arm!
        ext_mask = (wrist_rel_y > 0.10) & (elbow_deg >= 110.0) & (hand_sep >= 0.20)
        cand_y = np.where(ext_mask, wrist_rel_y, -1e9)
        if float(np.max(cand_y)) > -1e8:
            br_idx = int(np.argmax(cand_y))
            br_ff = _parabolic_peak(wrist_rel_y, br_idx)
            br_conf = 0.52
            br_method = "extended_overhead_wrist_fallback"
        else:
            # Clip does NOT contain an overhead ball release (e.g. truncated at FFC or gather-only)!
            is_complete = False
            br_idx = max(1, n - 1)
            br_ff = float(br_idx)
            br_conf = 0.15
            br_method = "incomplete_clip_no_overhead_release"
            warnings.append(
                "No overhead extended-arm ball release detected in this clip (possible truncated delivery "
                "or slow-motion clip ending before release); release-phase metrics are suppressed."
            )

    br_ev = _make_event(br_ff, fps, n, br_conf, br_method)

    # 2. FRONT FOOT CONTACT (FFC) — computed BEFORE UAH to enforce strict BFC < FFC <= UAH < BR!
    ffc_min = max(1, int(math.floor(br_ff - 0.24 * fps)))
    ffc_max = max(ffc_min + 1, min(int(math.floor(br_ff - max(1.0, 0.045 * fps))), n - 2))
    if ffc_max <= ffc_min:
        ffc_min = max(0, int(br_ff) - 3)
        ffc_max = max(ffc_min + 1, int(br_ff) - 1)

    f_y = f_ank[:, 1]
    f_vy = np.gradient(f_y) * fps
    f_ay = np.gradient(f_vy) * fps
    f_reach_x = f_ank[:, 0] - pelv[:, 0]

    # Check if front ankle is actually visible inside the camera frame during FFC search window
    f_ank_vis_ratio = 1.0
    if joint_in_frame is not None and joint_in_frame.shape[0] == n:
        f_ank_vis_ratio = float(np.mean(joint_in_frame[ffc_min : min(ffc_max + 2, n), f_ank_idx]))
        if f_ank_vis_ratio < 0.40:
            warnings.append(
                f"Front ankle cropped out of frame during FFC window (visibility={f_ank_vis_ratio:.0%}); "
                "FFC timestamp confidence reduced and lower-limb kinematics gated."
            )

    ground_y = float(np.percentile(f_y[ffc_min : min(int(br_ff) + 1, n)], 10))
    thresh_y = ground_y + 0.035
    candidates = []
    for t in range(ffc_min, min(ffc_max, n - 1)):
        if f_y[t] > thresh_y >= f_y[t + 1] and f_reach_x[t + 1] > 0.02:
            candidates.append(_linear_zero_crossing(f_y - thresh_y, t))

    if candidates:
        target_ffc = br_ff - 0.13 * fps
        ffc_ff = min(candidates, key=lambda c: abs(c - target_ffc))
        ffc_method = "front_foot_ground_touchdown"
        ffc_conf = 0.85 if f_ank_vis_ratio >= 0.40 else 0.35
    else:
        win = np.arange(ffc_min, min(ffc_max + 1, n))
        h_penalty = np.maximum(f_y[win] - ground_y, 0.0) / 0.10
        decel_bonus = np.maximum(f_ay[win], 0.0) / max(float(np.max(np.abs(f_ay[win]))), 1.0)
        reach_bonus = np.clip(f_reach_x[win] / 0.40, -1.0, 1.5)
        score = decel_bonus + 0.8 * reach_bonus - 1.5 * h_penalty
        best_idx = int(win[int(np.argmax(score))])
        ffc_ff = _parabolic_peak(f_ay, best_idx)
        ffc_method = "front_foot_deceleration_peak"
        ffc_conf = 0.68 if f_ank_vis_ratio >= 0.40 else 0.30

    # Enforce strict causality ffc < br
    if ffc_ff >= br_ff - 0.5:
        ffc_ff = max(0.0, br_ff - max(1.0, 0.08 * fps))
    ffc_ev = _make_event(ffc_ff, fps, n, ffc_conf, ffc_method)

    # 3. UPPER ARM HORIZONTAL (UAH) — strictly constrained to [FFC, BR)!
    uarm_vec = b_elb - b_sho
    uah_start = max(0, int(math.floor(ffc_ff)))
    uah_end = max(uah_start, min(n - 2, int(math.floor(br_ff - 0.25))))
    uah_ev: Optional[EventTimestamp] = None
    if is_complete and uah_end > uah_start:
        for t in range(uah_end, uah_start - 1, -1):
            if uarm_vec[t, 0] < 0.0 and uarm_vec[t, 1] <= 0.0 <= uarm_vec[t + 1, 1]:
                uah_ff = _linear_zero_crossing(uarm_vec[:, 1], t)
                if ffc_ff - 0.25 <= uah_ff < br_ff:
                    uah_ff = max(ffc_ff, uah_ff)
                    uah_ev = _make_event(
                        uah_ff,
                        fps,
                        n,
                        0.85 if fps >= 100.0 else 0.60,
                        "humerus_horizontal_crossing",
                    )
                    break
        if uah_ev is None:
            # Nearest-to-horizontal fallback strictly inside [FFC, BR)
            win_slice = slice(uah_start, min(uah_end + 1, n))
            sub_dy = np.abs(uarm_vec[win_slice, 1]) + np.where(uarm_vec[win_slice, 0] < 0.0, 0.0, 0.4)
            if len(sub_dy) > 0 and float(np.min(sub_dy)) < 0.20:
                best_rel = int(np.argmin(sub_dy))
                uah_ff = max(ffc_ff, min(float(uah_start + best_rel), br_ff - 0.25))
                uah_ev = _make_event(uah_ff, fps, n, 0.48, "humerus_horizontal_nearest_in_ffc_br")

    # 4. BACK FOOT CONTACT (BFC) — strictly constrained to [FFC - 450ms, FFC - 110ms]
    bfc_min = max(0, int(math.floor(ffc_ff - 0.45 * fps)))
    bfc_max = max(bfc_min + 1, min(int(math.floor(ffc_ff - max(1.0, 0.11 * fps))), n - 2))
    if bfc_max >= int(ffc_ff):
        bfc_max = max(0, int(ffc_ff) - 1)
        bfc_min = max(0, min(bfc_min, bfc_max))

    r_y = r_ank[:, 1]
    r_vy = np.gradient(r_y) * fps
    r_ay = np.gradient(r_vy) * fps

    r_ank_vis_ratio = 1.0
    if joint_in_frame is not None and joint_in_frame.shape[0] == n:
        r_ank_vis_ratio = float(np.mean(joint_in_frame[bfc_min : min(bfc_max + 2, n), r_ank_idx]))
        if r_ank_vis_ratio < 0.40:
            warnings.append(
                f"Rear ankle cropped out of frame during BFC window (visibility={r_ank_vis_ratio:.0%}); "
                "BFC timestamp confidence reduced."
            )

    win_b = np.arange(bfc_min, min(max(bfc_min + 1, bfc_max + 1), n))
    r_ground = float(np.percentile(r_y[win_b], 10))
    r_thresh = r_ground + 0.035
    bfc_candidates = []
    for t in range(bfc_min, min(bfc_max, n - 1)):
        if r_y[t] > r_thresh >= r_y[t + 1]:
            bfc_candidates.append(_linear_zero_crossing(r_y - r_thresh, t))

    if bfc_candidates:
        target_bfc = ffc_ff - 0.27 * fps
        bfc_ff = min(bfc_candidates, key=lambda c: abs(c - target_bfc))
        bfc_conf = 0.85 if r_ank_vis_ratio >= 0.40 else 0.35
        bfc_method = "rear_foot_ground_touchdown"
    else:
        b_height_score = -((r_y[win_b] - r_ground) / 0.06)
        b_vel_score = -np.abs(r_vy[win_b]) / max(float(np.max(np.abs(r_vy[win_b]))), 0.5)
        b_decel_score = np.maximum(r_ay[win_b], 0.0) / max(float(np.max(np.abs(r_ay[win_b]))), 1.0)
        b_total = 2.0 * b_height_score + 0.6 * b_vel_score + 0.5 * b_decel_score
        bfc_best = int(win_b[int(np.argmax(b_total))])
        bfc_ff = _parabolic_peak(-r_y, bfc_best)
        bfc_conf = (0.80 if (r_y[bfc_best] - ground_y) < 0.08 else 0.58) if r_ank_vis_ratio >= 0.40 else 0.30
        bfc_method = "rear_ankle_ground_minimum"

    if bfc_ff >= ffc_ff - 0.5:
        bfc_ff = max(0.0, ffc_ff - 1.0)
    bfc_ev = _make_event(bfc_ff, fps, n, bfc_conf, bfc_method)

    # 5. FOLLOW-THROUGH PEAK (FT)
    ft_start = min(n - 1, int(math.ceil(br_ff + max(1.0, 0.04 * fps))))
    ft_end = min(n - 1, int(math.ceil(br_ff + 0.25 * fps)))
    ft_ev: Optional[EventTimestamp] = None
    if is_complete and ft_end > ft_start:
        neck = j[:, JOINT_IDX["neck"]]
        trunk_fwd_x = neck[ft_start : ft_end + 1, 0] - pelv[ft_start : ft_end + 1, 0]
        ft_idx = ft_start + int(np.argmax(trunk_fwd_x))
        ft_ev = _make_event(float(ft_idx), fps, n, 0.72, "post_release_trunk_flexion_peak")

    stride_ms = max((ffc_ev.timestamp_s - bfc_ev.timestamp_s) * 1000.0, 1.0)
    ffc_br_ms = max((br_ev.timestamp_s - ffc_ev.timestamp_s) * 1000.0, 1.0)

    # Kinematic time-scale sanity check (detects slow-motion playback vs container FPS mismatch)
    if is_complete and (ffc_br_ms > 230.0 or stride_ms > 500.0):
        warnings.append(
            f"Atypical delivery duration (stride={stride_ms:.0f} ms, FFC->BR={ffc_br_ms:.0f} ms) "
            "suggests clip may be slow-motion playback; pass --capture-fps with the true recording rate."
        )
    elif is_complete and (ffc_br_ms < 45.0 or stride_ms < 85.0):
        warnings.append(
            f"Unusually compressed delivery duration (stride={stride_ms:.0f} ms, FFC->BR={ffc_br_ms:.0f} ms); "
            "verify --capture-fps is not overestimated."
        )

    return DeliveryEvents(
        runup_start_frame=max(0, bfc_ev.frame_idx - int(round(0.35 * fps))),
        bfc=bfc_ev,
        ffc=ffc_ev,
        uah=uah_ev,
        br=br_ev,
        ft_peak=ft_ev,
        delivery_stride_duration_ms=round(stride_ms, 1),
        ffc_to_br_duration_ms=round(ffc_br_ms, 1),
        is_complete_delivery=is_complete,
        event_warnings=warnings,
    )
