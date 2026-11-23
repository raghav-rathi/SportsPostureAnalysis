"""Stage 2B: Multi-Cue Sub-Frame Temporal Event Spotting for Cricket Fast Bowling.

Fixes EXP-007:
  - B1: Strict Ball Release sanity gates (coronal abduction reach `r_cor` supporting slingy bowlers,
        vertical-window OR 25fps bracketing crossing, pre-release backswing <= -70 deg, forward angular
        velocity >= 220 deg/s, non-frozen check, peak-omega pass selection eliminating 30fps second-pass
        traps, and removal of weak fallback). Truncated/frozen clips (McGrath, Ambrose) cleanly return
        `is_complete_delivery=False`.
  - B7: Realistic method- & visibility-dependent `timing_uncertainty_ms` (replacing hardcoded 500/fps),
        explicit `_boundary_clamped` method tagging & confidence drop when clamping fires, and source
        video frame mapping (`source_frame_idx`).
  - B23: Imports `JOINT_IDX` & Felton (2019) priors (`192 ms`, `108 ms`) from `constants.py`.
  - B27: True Winter (2009) dual-pass 2nd-order Butterworth with cutoff correction `C = 1.2465`.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
from scipy.signal import butter, filtfilt

from .constants import FELTON_2019_FFC_BR_MS, FELTON_2019_STRIDE_MS, JOINT_IDX
from .schemas import BowlingArm, DeliveryEvents, EventTimestamp

# Winter (2009) dual-pass cutoff correction factor for 2nd-order Butterworth passed through filtfilt:
# C = (2^(1/2) - 1)^(-1/4) ~= 1.24650 -> effective 4th-order zero-lag with exact -3 dB at cutoff_hz!
_WINTER_DUALPASS_CORR = 1.24650


def smooth_trajectory(joints: np.ndarray, fps: float, cutoff_hz: float = 15.0) -> np.ndarray:
    """Zero-lag 4th-order effective Butterworth filter (Winter 2nd-order dual-pass with cutoff correction)."""
    n = joints.shape[0]
    if n < 10:
        return joints.copy()
    nyq = 0.5 * max(fps, 1.0)
    wn = min((cutoff_hz * _WINTER_DUALPASS_CORR) / max(nyq, 1e-3), 0.90)
    if wn <= 0.02 or wn >= 0.99:
        return joints.copy()
    b, a = butter(2, wn, btype="low")
    out = joints.copy()
    flat = out.reshape(n, -1)
    padlen = min(3 * max(len(a), len(b)), n - 1)
    for col in range(flat.shape[1]):
        if np.all(np.isfinite(flat[:, col])):
            flat[:, col] = filtfilt(b, a, flat[:, col], padlen=padlen)
    return flat.reshape(joints.shape)


def _median_despike_3(arr: np.ndarray) -> np.ndarray:
    """3-frame sliding median to strip single-frame keypoint glitches without shrinking circular arcs."""
    n = arr.shape[0]
    if n < 3:
        return arr.copy()
    out = arr.copy()
    out[1:-1] = np.median(np.stack([arr[:-2], arr[1:-1], arr[2:]], axis=0), axis=0)
    return out


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
    timing_unc_ms: float,
    source_frame_indices: Optional[List[int]] = None,
) -> EventTimestamp:
    ff = float(np.clip(frame_float, 0.0, max(n_frames - 1, 0)))
    idx = int(np.clip(round(ff), 0, max(n_frames - 1, 0)))
    src_idx = (
        int(source_frame_indices[idx])
        if (source_frame_indices is not None and 0 <= idx < len(source_frame_indices))
        else idx
    )
    return EventTimestamp(
        frame_float=round(ff, 3),
        frame_idx=idx,
        source_frame_idx=src_idx,
        timestamp_s=round(ff / max(fps, 1.0), 4),
        timing_uncertainty_ms=round(float(timing_unc_ms), 2),
        confidence=round(float(np.clip(confidence, 0.0, 0.99)), 3),
        detection_method=method,
    )


def detect_delivery_events(
    joints_world: np.ndarray,
    fps: float,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    joint_in_frame: Optional[np.ndarray] = None,
    source_frame_indices: Optional[List[int]] = None,
    held_frame_mask: Optional[np.ndarray] = None,
) -> DeliveryEvents:
    """Detect BFC, FFC, UAH, BR, and FT with strict release sanity gates and Felton (2019) priors."""
    n = joints_world.shape[0]
    if n < 6:
        raise ValueError(f"Delivery clip too short for event spotting ({n} frames).")

    # Smooth trajectory for derivatives/foot-strike; use 3-frame median-despiked raw joints only at >=100 fps
    # so Butterworth filtering doesn't shrink circular arm chord length, while preserving narrow overhead peaks
    # at 25-60 fps where 38-90 deg/frame sweeps cause a 3-frame median to chop off the overhead release peak!
    j = smooth_trajectory(joints_world, fps, cutoff_hz=min(18.0, 0.36 * fps))
    j_raw = _median_despike_3(joints_world) if fps >= 100.0 else joints_world.copy()
    warnings: List[str] = []
    is_complete = True

    is_right = bowling_arm == BowlingArm.RIGHT
    sign_arm = 1.0 if is_right else -1.0
    b_sho_idx = JOINT_IDX["rsho" if is_right else "lsho"]
    b_elb_idx = JOINT_IDX["relb" if is_right else "lelb"]
    b_wri_idx = JOINT_IDX["rwri" if is_right else "lwri"]
    nb_wri_idx = JOINT_IDX["lwri" if is_right else "rwri"]
    f_ank_idx = JOINT_IDX["lank" if is_right else "rank"]
    r_ank_idx = JOINT_IDX["rank" if is_right else "lank"]

    b_sho, b_elb, b_wri = j[:, b_sho_idx], j[:, b_elb_idx], j[:, b_wri_idx]
    b_sho_r, b_elb_r, b_wri_r = j_raw[:, b_sho_idx], j_raw[:, b_elb_idx], j_raw[:, b_wri_idx]
    nb_wri_r = j_raw[:, nb_wri_idx]
    pelv = j[:, JOINT_IDX["pelv"]]
    f_ank = j[:, f_ank_idx]
    r_ank = j[:, r_ank_idx]

    # 1. BALL RELEASE (BR) WITH STRICT SANITY GATES (EXP-007 B1)
    # Coronal abduction reach supports both orthodox high-arm AND slingy round-arm fast bowlers,
    # while rejecting across-body gathers (sign_arm * a_z < 0) and sub-shoulder poses!
    arm_r = b_wri_r - b_sho_r
    arm_len_r = np.linalg.norm(arm_r, axis=1)
    hand_sep_r = np.linalg.norm(b_wri_r - nb_wri_r, axis=1)

    u_r = b_sho_r - b_elb_r
    f_r = b_wri_r - b_elb_r
    # Noise-robust anatomical arm-length reference via median segment sum (avoids 95th-pct depth-noise inflation)
    max_arm_len = max(
        float(np.median(np.linalg.norm(u_r, axis=1)) + np.median(np.linalg.norm(f_r, axis=1))),
        0.42,
    )
    cos_elb_r = np.sum(u_r * f_r, axis=1) / np.maximum(
        np.linalg.norm(u_r, axis=1) * np.linalg.norm(f_r, axis=1), 1e-6
    )
    elbow_deg_r = np.degrees(np.arccos(np.clip(cos_elb_r, -1.0, 1.0)))

    # Coronal reach & sagittal circumduction angle (using raw despiked arm for 25fps fidelity, smoothed for omega)
    outward_z_r = np.maximum(sign_arm * arm_r[:, 2], 0.0)
    up_y_r = np.maximum(arm_r[:, 1], 0.0)
    r_cor_r = np.hypot(up_y_r, outward_z_r)
    # Full 360-deg sagittal circumduction angle phi in [-180, 180]: 0 = overhead/coronal peak, -90 = behind, +90 = front
    phi_cor_r = np.degrees(np.arctan2(arm_r[:, 0], np.where(arm_r[:, 1] >= -0.05, np.maximum(r_cor_r, 1e-4), arm_r[:, 1])))

    arm_s = b_wri - b_sho
    phi_s_raw = np.degrees(np.arctan2(arm_s[:, 0], arm_s[:, 1]))
    phi_s_unwrap = np.degrees(np.unwrap(np.radians(phi_s_raw)))
    # Use raw despiked unwrap for 25-35 fps so heavy smoothing doesn't damp peak omega
    phi_use_unwrap = np.degrees(np.unwrap(np.radians(phi_cor_r))) if fps < 45.0 else phi_s_unwrap
    arm_omega_dps = np.gradient(phi_use_unwrap) * fps

    # Per-frame 3D wrist motion magnitude (rejects freeze-frame graphic stills!)
    wri_step_m = np.zeros(n, dtype=np.float64)
    wri_step_m[1:] = np.linalg.norm(np.diff(b_wri_r, axis=0), axis=1)
    # Neighborhood wrist motion across +-1 frame
    wri_local_motion = np.zeros(n, dtype=np.float64)
    for t in range(n):
        lo_m, hi_m = max(0, t - 1), min(n - 1, t + 1)
        wri_local_motion[t] = float(np.linalg.norm(b_wri_r[hi_m] - b_wri_r[lo_m]))

    def _is_frozen_around(t_idx: int) -> bool:
        if held_frame_mask is not None and len(held_frame_mask) == n:
            lo_h = max(0, t_idx - 1)
            hi_h = min(n, t_idx + 2)
            # If current AND adjacent frame are held repeats, it's inside a freeze run
            if int(np.sum(held_frame_mask[lo_h:hi_h])) >= 2:
                return True
        return wri_local_motion[t_idx] < 0.055

    def _has_pre_backswing(t_idx: int) -> bool:
        # True bowling circumduction passes behind the shoulder (phi <= -65 deg) within ~260 ms before BR!
        win_lo = max(0, int(math.floor(t_idx - max(3.0, 0.28 * fps))))
        if win_lo >= t_idx:
            return False
        # Check unwrapped drop behind current overhead position (handles slingy & orthodox equally)
        pre_drop = float(phi_use_unwrap[t_idx] - np.min(phi_use_unwrap[win_lo:t_idx]))
        min_phi = float(np.min(phi_cor_r[win_lo:t_idx]))
        return (min_phi <= -60.0) or (pre_drop >= 75.0)

    def _limb_extended_overhead(t_idx: int) -> bool:
        # Rejects McGrath (-46 deg, 0.68 arm_len) and Ambrose (+56 deg, 0.44 arm_len) while passing slingy bowlers
        if arm_r[t_idx, 1] < 0.06:
            return False
        if r_cor_r[t_idx] < max(0.36, 0.72 * max_arm_len):
            return False
        # For orthodox (non-slingy) arm paths, wrist itself must be genuinely overhead (>=0.32 m above shoulder)
        if outward_z_r[t_idx] < 0.34 and arm_r[t_idx, 1] < max(0.32, 0.70 * max_arm_len):
            return False
        if elbow_deg_r[t_idx] < 120.0 or hand_sep_r[t_idx] < 0.24:
            return False
        return True

    def _bracket_extended_overhead(t_idx: int, dphi_pair: float) -> bool:
        # Supports wide 25-30 fps straddles (>+-44 deg) using 3D limb extension + arc-restored midpoint coronal reach
        if _limb_extended_overhead(t_idx) or _limb_extended_overhead(t_idx + 1):
            return True
        ext_ok = any(
            arm_len_r[k] >= 0.75 * max_arm_len and elbow_deg_r[k] >= 120.0 and hand_sep_r[k] >= 0.24
            for k in (t_idx, t_idx + 1)
        )
        if not ext_ok:
            return False
        mid_y = max(0.5 * (float(arm_r[t_idx, 1]) + float(arm_r[t_idx + 1, 1])), 0.0)
        mid_z = max(0.5 * sign_arm * (float(arm_r[t_idx, 2]) + float(arm_r[t_idx + 1, 2])), 0.0)
        cos_half = max(math.cos(math.radians(0.5 * min(dphi_pair, 110.0))), 0.55)
        arc_r_cor = math.hypot(mid_y, mid_z) / cos_half
        return (mid_y >= 0.06) and (arc_r_cor >= max(0.36, 0.72 * max_arm_len))

    # Minimum frames needed before BR so FFC is observable (at least ~45 ms or 2 frames)
    min_br_idx = max(2, int(math.ceil(0.045 * fps)))
    valid_candidates: List[tuple] = []  # (peak_omega, score, br_ff, br_idx, is_crossing)

    # Check both integer frames inside [-28, +32] deg AND 25-30fps bracketing crossings across +8 deg
    for t in range(min_br_idx, n - 1):
        if _is_frozen_around(t):
            continue
        # Case A: Integer frame inside overhead release corridor [-28, +32] deg
        in_corridor = (-28.0 <= phi_cor_r[t] <= 32.0) and _limb_extended_overhead(t)
        # Case B: Fast 25–30 fps sweep straddling +8 deg between t and t+1
        dphi_pair = float(phi_use_unwrap[t + 1] - phi_use_unwrap[t])
        is_bracket = (
            (phi_cor_r[t] <= 8.0 <= phi_cor_r[t + 1])
            and (8.0 <= dphi_pair <= 135.0)
            and _bracket_extended_overhead(t, dphi_pair)
            and (not _is_frozen_around(t + 1))
        )
        if not (in_corridor or is_bracket):
            continue

        # Require genuine forward bowling circumduction speed + pre-release backswing
        local_omega = float(max(arm_omega_dps[t], arm_omega_dps[min(t + 1, n - 1)]))
        if local_omega < 220.0:
            continue
        if not _has_pre_backswing(t + (1 if is_bracket and not in_corridor else 0)):
            continue

        if is_bracket:
            br_ff_c = _linear_zero_crossing(phi_cor_r - 8.0, t)
            prox = 1.0
        else:
            br_ff_c = float(t)
            prox = math.exp(-0.5 * ((phi_cor_r[t] - 8.0) / 16.0) ** 2)

        # Rank valid passes primarily by peak forward circumduction speed (kills 30fps second-pass/celebration trap!)
        vel_norm = min(local_omega / 1200.0, 2.5)
        ext_norm = min(r_cor_r[t] / max_arm_len, 1.0)
        sc = 2.0 * vel_norm + 1.4 * prox + 0.8 * ext_norm
        valid_candidates.append((local_omega, sc, br_ff_c, t, is_bracket))

    if valid_candidates:
        # Group adjacent candidate frames into contiguous overhead passes and pick the highest-speed/score pass!
        valid_candidates.sort(key=lambda x: x[3])
        passes: List[List[tuple]] = [[valid_candidates[0]]]
        for item in valid_candidates[1:]:
            if item[3] - passes[-1][-1][3] <= max(3, int(round(0.06 * fps))):
                passes[-1].append(item)
            else:
                passes.append([item])

        # Select pass with highest peak angular velocity (breaks ties via composite score)
        best_pass = max(passes, key=lambda p_list: (max(x[0] for x in p_list), max(x[1] for x in p_list)))
        best_item = max(best_pass, key=lambda x: (1 if x[4] else 0, x[1]))
        _, _, br_ff, br_idx, is_cross = best_item

        if (not is_cross) and 1 <= br_idx < n - 1:
            shifted = phi_cor_r - 8.0
            if shifted[br_idx - 1] <= 0.0 <= shifted[br_idx]:
                br_ff = _linear_zero_crossing(shifted, br_idx - 1)
                is_cross = True
            elif shifted[br_idx] <= 0.0 <= shifted[br_idx + 1]:
                br_ff = _linear_zero_crossing(shifted, br_idx)
                is_cross = True
            else:
                # Parabolic peak on wrist overhead height
                br_ff = _parabolic_peak(r_cor_r, br_idx)

        br_unc_ms = math.hypot(2.5 if is_cross else 4.5, (350.0 if is_cross else 500.0) / max(fps, 1.0))
        br_conf = 0.88 if fps >= 100.0 else (0.76 if fps >= 48.0 else 0.62)
        br_method = "overhead_circumduction_crossing" if is_cross else "overhead_circumduction_peak"
    else:
        # Strict refusal: NO weak fallback! Clip lacks a valid overhead delivery release.
        is_complete = False
        br_idx = max(1, n - 1)
        br_ff = float(br_idx)
        br_conf = 0.0
        br_unc_ms = max(40.0, 1500.0 / max(fps, 1.0))
        br_method = "incomplete_clip_no_overhead_release"
        warnings.append(
            "Incomplete delivery: no valid high-speed overhead bowling-arm release passed sanity gates "
            "(requires extended arm near vertical with pre-release backswing and non-frozen motion). "
            "Delivery metrics and coaching cues are suppressed."
        )

    br_ev = _make_event(br_ff, fps, n, br_conf, br_method, br_unc_ms, source_frame_indices)

    # 2. FRONT FOOT CONTACT (FFC) — Felton (2019) prior = 108 ms before BR, window [BR - 210 ms, BR - 45 ms]
    ffc_min = max(1, int(math.floor(br_ff - 0.21 * fps)))
    ffc_max = max(ffc_min + 1, min(int(math.floor(br_ff - max(1.0, 0.045 * fps))), n - 2))
    if ffc_max <= ffc_min:
        ffc_min = max(0, int(br_ff) - 3)
        ffc_max = max(ffc_min + 1, int(br_ff) - 1)

    f_y = f_ank[:, 1]
    f_vy = np.gradient(f_y) * fps
    f_ay = np.gradient(f_vy) * fps
    f_reach_x = f_ank[:, 0] - pelv[:, 0]

    f_ank_vis_ratio = 1.0
    if joint_in_frame is not None and joint_in_frame.shape[0] == n:
        f_ank_vis_ratio = float(np.mean(joint_in_frame[ffc_min : min(ffc_max + 2, n), f_ank_idx]))
        if f_ank_vis_ratio < 0.40:
            warnings.append(
                f"Front ankle cropped out of frame during FFC window (visibility={f_ank_vis_ratio:.0%}); "
                "FFC timestamp confidence reduced and lower-limb kinematics gated."
            )

    ground_y = float(np.percentile(f_y[ffc_min : min(int(br_ff) + 1, n)], 10))
    thresh_hi = ground_y + 0.035
    thresh_lo = ground_y + 0.010
    candidates = []
    for t in range(ffc_min, min(ffc_max, n - 1)):
        if f_y[t] > thresh_hi >= f_y[t + 1] and f_reach_x[t + 1] > 0.02:
            c_hi = _linear_zero_crossing(f_y - thresh_hi, t)
            # Advance from 3.5 cm descent gate to true ground contact (~1.0 cm) so smooth continuous
            # foot landings aren't biased ~13 ms early (EXP-007 S2/S3)
            c_td = None
            t_search_end = min(ffc_max, t + max(2, int(round(0.035 * fps))))
            for t2 in range(t, min(t_search_end, n - 1)):
                if f_y[t2] > thresh_lo >= f_y[t2 + 1]:
                    c_td = _linear_zero_crossing(f_y - thresh_lo, t2)
                    break
            if c_td is None:
                v_down = max(-float(f_vy[t + 1]), 0.6)
                c_td = min(float(ffc_max), c_hi + min(0.025 / v_down, 0.020) * fps)
            candidates.append(c_td)

    ffc_prior_s = FELTON_2019_FFC_BR_MS / 1000.0
    if candidates:
        target_ffc = br_ff - ffc_prior_s * fps
        ffc_ff = min(candidates, key=lambda c: abs(c - target_ffc))
        ffc_method = "front_foot_ground_touchdown"
        ffc_conf = 0.85 if f_ank_vis_ratio >= 0.40 else 0.28
        ffc_unc_ms = math.hypot(6.5, 600.0 / max(fps, 1.0))
    else:
        win = np.arange(ffc_min, min(ffc_max + 1, n))
        h_penalty = np.maximum(f_y[win] - ground_y, 0.0) / 0.10
        decel_bonus = np.maximum(f_ay[win], 0.0) / max(float(np.max(np.abs(f_ay[win]))), 1.0)
        reach_bonus = np.clip(f_reach_x[win] / 0.40, -1.0, 1.5)
        score = decel_bonus + 0.8 * reach_bonus - 1.5 * h_penalty
        best_idx = int(win[int(np.argmax(score))])
        ffc_ff = _parabolic_peak(f_ay, best_idx)
        ffc_method = "front_foot_deceleration_peak"
        ffc_conf = 0.65 if f_ank_vis_ratio >= 0.40 else 0.24
        ffc_unc_ms = math.hypot(11.0, 750.0 / max(fps, 1.0))

    if f_ank_vis_ratio < 0.40:
        ffc_unc_ms *= 1.8

    # Honest boundary / causality clamp flagging (EXP-007 B7)
    ffc_clamped = False
    if ffc_ff >= br_ff - 0.5:
        ffc_ff = max(0.0, br_ff - max(1.0, 0.08 * fps))
        ffc_clamped = True
    elif ffc_ff <= ffc_min + 0.25 and not candidates:
        ffc_clamped = True

    if ffc_clamped:
        ffc_method += "_boundary_clamped"
        ffc_conf = min(ffc_conf, 0.24)
        ffc_unc_ms = max(ffc_unc_ms * 1.9, 20.0)
        warnings.append(
            f"FFC hit search-window or causality boundary (frame {ffc_ff:.1f}); "
            "FFC-dependent metrics carry degraded confidence."
        )
    if not is_complete:
        ffc_conf = min(ffc_conf, 0.20)
    ffc_ev = _make_event(ffc_ff, fps, n, ffc_conf, ffc_method, ffc_unc_ms, source_frame_indices)

    # 3. UPPER ARM HORIZONTAL (UAH) — strictly inside [FFC, BR)
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
                        math.hypot(4.0, 450.0 / max(fps, 1.0)),
                        source_frame_indices,
                    )
                    break
        if uah_ev is None:
            win_slice = slice(uah_start, min(uah_end + 1, n))
            sub_dy = np.abs(uarm_vec[win_slice, 1]) + np.where(uarm_vec[win_slice, 0] < 0.0, 0.0, 0.4)
            if len(sub_dy) > 0 and float(np.min(sub_dy)) < 0.20:
                best_rel = int(np.argmin(sub_dy))
                uah_ff = max(ffc_ff, min(float(uah_start + best_rel), br_ff - 0.25))
                uah_ev = _make_event(
                    uah_ff,
                    fps,
                    n,
                    0.48,
                    "humerus_horizontal_nearest_in_ffc_br",
                    math.hypot(7.0, 600.0 / max(fps, 1.0)),
                    source_frame_indices,
                )

    # 4. BACK FOOT CONTACT (BFC) — Felton (2019) prior = 192 ms before FFC, window [FFC - 360 ms, FFC - 90 ms]
    bfc_min = max(0, int(math.floor(ffc_ff - 0.36 * fps)))
    bfc_max = max(bfc_min + 1, min(int(math.floor(ffc_ff - max(1.0, 0.09 * fps))), n - 2))
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
    r_thresh_lo = r_ground + 0.010
    bfc_candidates = []
    for t in range(bfc_min, min(bfc_max, n - 1)):
        if r_y[t] > r_thresh >= r_y[t + 1]:
            # Require genuine aerial descent (peak >= 6.5 cm above ground within 160 ms prior) so ground-level noise doesn't trigger BFC
            pre_lo = max(0, t - max(2, int(round(0.16 * fps))))
            if float(np.max(r_y[pre_lo : t + 1])) < r_ground + 0.065:
                continue
            c_hi = _linear_zero_crossing(r_y - r_thresh, t)
            c_td = None
            t_search_end = min(bfc_max, t + max(2, int(round(0.035 * fps))))
            for t2 in range(t, min(t_search_end, n - 1)):
                if r_y[t2] > r_thresh_lo >= r_y[t2 + 1]:
                    c_td = _linear_zero_crossing(r_y - r_thresh_lo, t2)
                    break
            if c_td is None:
                v_down = max(-float(r_vy[t + 1]), 0.6)
                c_td = min(float(bfc_max), c_hi + min(0.025 / v_down, 0.020) * fps)
            bfc_candidates.append(c_td)

    stride_prior_s = FELTON_2019_STRIDE_MS / 1000.0
    if bfc_candidates:
        target_bfc = ffc_ff - stride_prior_s * fps
        bfc_ff = min(bfc_candidates, key=lambda c: abs(c - target_bfc))
        bfc_conf = 0.85 if r_ank_vis_ratio >= 0.40 else 0.28
        bfc_method = "rear_foot_ground_touchdown"
        bfc_unc_ms = math.hypot(10.5, 700.0 / max(fps, 1.0))
    else:
        b_height_score = -((r_y[win_b] - r_ground) / 0.06)
        b_vel_score = -np.abs(r_vy[win_b]) / max(float(np.max(np.abs(r_vy[win_b]))), 0.5)
        b_decel_score = np.maximum(r_ay[win_b], 0.0) / max(float(np.max(np.abs(r_ay[win_b]))), 1.0)
        b_total = 2.0 * b_height_score + 0.6 * b_vel_score + 0.5 * b_decel_score
        bfc_best = int(win_b[int(np.argmax(b_total))])
        bfc_ff = _parabolic_peak(-r_y, bfc_best)
        bfc_conf = (0.76 if (r_y[bfc_best] - ground_y) < 0.08 else 0.52) if r_ank_vis_ratio >= 0.40 else 0.24
        bfc_method = "rear_ankle_ground_minimum"
        bfc_unc_ms = math.hypot(14.5, 850.0 / max(fps, 1.0))

    if r_ank_vis_ratio < 0.40:
        bfc_unc_ms *= 1.8

    bfc_clamped = False
    if bfc_ff >= ffc_ff - 0.5:
        bfc_ff = max(0.0, ffc_ff - 1.0)
        bfc_clamped = True
    elif (bfc_ff <= bfc_min + 0.25 or bfc_ff >= bfc_max - 0.25) and not bfc_candidates:
        bfc_clamped = True

    if bfc_clamped:
        bfc_method += "_boundary_clamped"
        bfc_conf = min(bfc_conf, 0.24)
        bfc_unc_ms = max(bfc_unc_ms * 1.9, 25.0)
        warnings.append(
            f"BFC landed on search-window boundary (frame {bfc_ff:.1f}); "
            "BFC-dependent metrics carry degraded confidence."
        )
    if not is_complete:
        bfc_conf = min(bfc_conf, 0.20)
    bfc_ev = _make_event(bfc_ff, fps, n, bfc_conf, bfc_method, bfc_unc_ms, source_frame_indices)

    # 5. FOLLOW-THROUGH PEAK (FT)
    ft_start = min(n - 1, int(math.ceil(br_ff + max(1.0, 0.04 * fps))))
    ft_end = min(n - 1, int(math.ceil(br_ff + 0.25 * fps)))
    ft_ev: Optional[EventTimestamp] = None
    if is_complete and ft_end > ft_start:
        neck = j[:, JOINT_IDX["neck"]]
        trunk_fwd_x = neck[ft_start : ft_end + 1, 0] - pelv[ft_start : ft_end + 1, 0]
        ft_idx = ft_start + int(np.argmax(trunk_fwd_x))
        ft_ev = _make_event(
            float(ft_idx),
            fps,
            n,
            0.72,
            "post_release_trunk_flexion_peak",
            math.hypot(8.0, 600.0 / max(fps, 1.0)),
            source_frame_indices,
        )

    stride_ms = max((ffc_ev.timestamp_s - bfc_ev.timestamp_s) * 1000.0, 1.0)
    ffc_br_ms = max((br_ev.timestamp_s - ffc_ev.timestamp_s) * 1000.0, 1.0)

    # Tightened kinematic time-scale sanity check aligned with Felton et al. (2019): 192+-31 ms, 103+-11 ms
    if is_complete and (ffc_br_ms > 180.0 or stride_ms > 340.0):
        warnings.append(
            f"Atypical delivery duration (stride={stride_ms:.0f} ms, FFC->BR={ffc_br_ms:.0f} ms vs "
            "Felton 2019 means 192/103 ms) suggests clip may be slow-motion playback; pass --capture-fps."
        )
    elif is_complete and (ffc_br_ms < 50.0 or stride_ms < 85.0):
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
