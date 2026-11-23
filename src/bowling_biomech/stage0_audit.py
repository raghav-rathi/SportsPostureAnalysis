"""Stage 0: Untrusted-Video Timebase Integrity, ROI Sharpness, & 4-DOF Camera Motion Audit.

Fixes EXP-007 B2 (repeated-frame pulldown, slow-mo retiming, freeze frames, scene cuts, inverted caption blur)
and B14 (short-baseline 4-DOF similarity RANSAC detecting horizontal pan, vertical tilt, AND focal zoom
without Farneback aliasing on long clips).
"""

from __future__ import annotations

import json
import math
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import NOMINAL_FAST_ARM_DPS, NOMINAL_FAST_BALL_MPS
from .schemas import VideoAudit


def _probe_ffprobe(video_path: str) -> Tuple[int, int, float, int, float]:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        video_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode == 0 and proc.stdout.strip():
        data = json.loads(proc.stdout)
        vstreams = [s for s in data.get("streams", []) if s.get("codec_type") == "video"]
        if vstreams:
            vs = vstreams[0]
            w = int(vs.get("width", 0))
            h = int(vs.get("height", 0))
            fps_str = vs.get("avg_frame_rate") or vs.get("r_frame_rate") or "30/1"
            num, den = fps_str.split("/")
            fps = float(num) / max(float(den), 1.0)
            dur = float(vs.get("duration") or data.get("format", {}).get("duration") or 0.0)
            nb = int(vs.get("nb_frames") or round(dur * fps))
            if w > 0 and h > 0 and fps > 0:
                return w, h, fps, nb, dur

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    nb = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    dur = nb / max(fps, 1.0)
    return w, h, fps, nb, dur


def _fps_grade(fps: float) -> str:
    if fps >= 200.0:
        return "high_speed_240"
    if fps >= 100.0:
        return "slow_mo_120"
    if fps >= 48.0:
        return "medium_60"
    if fps >= 28.0:
        return "standard_30"
    return "low_25"


def analyze_frame_cadence(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    roi_box_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> Dict:
    """Scan consecutive frames for exact/near-exact repeats, pulldown cadence, freeze frames, and scene cuts."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    sf = max(0, int(start_frame))
    ef = min(total, int(end_frame)) if end_frame is not None else total
    if ef <= sf:
        ef = max(sf + 1, total)

    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    prev_small: Optional[np.ndarray] = None
    diffs: List[float] = []
    lap_vars: List[float] = []
    sample_stride = max(1, (ef - sf) // 24)

    for rel_i in range(ef - sf):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if roi_box_xyxy is not None:
            x0, y0, x1, y1 = roi_box_xyxy
            crop = frame[max(0, y0) : min(h, y1), max(0, x0) : min(w, x1)]
            if crop.size == 0:
                crop = frame
        else:
            # Central 60% crop avoids top/bottom letterbox bars, scoreboards, and burned-in captions
            y0, y1 = int(0.20 * h), int(0.80 * h)
            x0, x1 = int(0.15 * w), int(0.85 * w)
            crop = frame[y0:y1, x0:x1]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Downsample to width ~240 for fast, compression-noise-averaged frame-difference cadence check
        sc = 240.0 / max(gray.shape[1], 1)
        small = cv2.resize(
            gray,
            (max(16, int(round(gray.shape[1] * sc))), max(16, int(round(gray.shape[0] * sc)))),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)

        if prev_small is not None:
            diffs.append(float(np.mean(np.abs(small - prev_small))))
        prev_small = small

        if rel_i % sample_stride == 0:
            # Compute sharpness strictly on the non-black, non-caption interior crop
            norm_h = 360
            sc_l = norm_h / max(gray.shape[0], 1)
            g_l = cv2.resize(gray, (max(16, int(round(gray.shape[1] * sc_l))), norm_h))
            lap_vars.append(float(cv2.Laplacian(g_l, cv2.CV_64F).var()))
    cap.release()

    n_read = len(diffs) + (1 if prev_small is not None else 0)
    diffs_arr = np.asarray(diffs, dtype=np.float64)
    if diffs_arr.size == 0:
        return {
            "raw_num_frames": max(1, n_read),
            "held_mask": np.zeros(max(1, n_read), dtype=bool),
            "held_frame_ratio": 0.0,
            "longest_held_run_frames": 0,
            "regular_repeat_cadence": None,
            "scene_cut_indices": [],
            "median_laplacian_var": float(np.median(lap_vars)) if lap_vars else 0.0,
        }

    moving = diffs_arr[diffs_arr > 0.0]
    med_all = float(np.median(moving)) if moving.size else 0.0
    # Two-pass moving median so clips with >50% repeats still find a sensible motion scale
    p75_all = float(np.percentile(moving, 75)) if moving.size else 0.0
    ref_motion = max(med_all, 0.5 * p75_all)
    held_thresh = max(0.22 * ref_motion, 0.35)
    held_pairs = diffs_arr < held_thresh

    held_idx = np.flatnonzero(held_pairs)
    gaps = np.diff(held_idx)
    run = 0
    max_run = 0
    for is_h in held_pairs:
        run = run + 1 if is_h else 0
        if run > max_run:
            max_run = run

    cadence: Optional[int] = None
    if gaps.size >= 4:
        vals, counts = np.unique(gaps, return_counts=True)
        top_gap = int(vals[int(np.argmax(counts))])
        if top_gap >= 2 and (float(counts.max()) / float(gaps.size)) >= 0.70:
            cadence = top_gap

    # Hard scene cuts / graphic jump-cuts (e.g., creator inserting an AI graphic freeze frame)
    moving_diffs = diffs_arr[~held_pairs]
    med_mov = float(np.median(moving_diffs)) if moving_diffs.size else 10.0
    cut_thresh = max(4.2 * med_mov, 28.0)
    scene_cuts = [int(i + 1) for i in np.flatnonzero(diffs_arr > cut_thresh)]

    # Per-frame held mask aligned to frame 0..n_read-1 (frame i is a repeat of i-1)
    held_frame_mask = np.zeros(n_read, dtype=bool)
    held_frame_mask[1:] = held_pairs

    return {
        "raw_num_frames": n_read,
        "held_mask": held_frame_mask,
        "held_frame_ratio": round(float(np.mean(held_pairs)), 3),
        "longest_held_run_frames": int(max_run),
        "regular_repeat_cadence": cadence,
        "scene_cut_indices": scene_cuts,
        "median_laplacian_var": float(np.median(lap_vars)) if lap_vars else 0.0,
    }


def measure_bowler_roi_blur(
    video_path: str | Path,
    source_frame_indices: List[int],
    bowler_boxes_xyxy: np.ndarray,
    sample_count: int = 18,
) -> Tuple[float, str]:
    """Measure motion blur strictly inside the bowler's bounding box (rejects Reel captions/borders)."""
    n = min(len(source_frame_indices), len(bowler_boxes_xyxy))
    if n < 2:
        return 0.0, "moderate_blur"
    pick = np.unique(np.linspace(0, n - 1, min(n, sample_count), dtype=int))
    pick_src = {source_frame_indices[int(i)]: int(i) for i in pick}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0, "moderate_blur"
    sf = int(source_frame_indices[0])
    ef = int(source_frame_indices[-1]) + 1
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    lap_vars: List[float] = []
    for abs_f in range(sf, ef):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        rel_u = pick_src.get(abs_f)
        if rel_u is None:
            continue
        h, w = frame.shape[:2]
        bx0, by0, bx1, by1 = bowler_boxes_xyxy[rel_u]
        x0 = max(0, int(bx0))
        y0 = max(0, int(by0))
        x1 = min(w, int(bx1))
        y1 = min(h, int(by1))
        if (x1 - x0) < 24 or (y1 - y0) < 40:
            continue
        crop = frame[y0:y1, x0:x1]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Normalize bowler height to 320 px so sharpness is scale-comparable across 720p/1080p/crop sizes
        sc = 320.0 / max(gray.shape[0], 1)
        g_norm = cv2.resize(gray, (max(16, int(round(gray.shape[1] * sc))), 320))
        # Mask out extreme white burned-in subtitle pixels (>238) inside the crop
        sub_mask = g_norm < 238
        lap = cv2.Laplacian(g_norm, cv2.CV_64F)
        if int(sub_mask.sum()) > 100:
            lap_vars.append(float(np.var(lap[sub_mask])))
        else:
            lap_vars.append(float(lap.var()))
    cap.release()

    if not lap_vars:
        return 0.0, "moderate_blur"
    med_lap = float(np.median(lap_vars))
    if med_lap >= 140.0:
        grade = "crisp"
    elif med_lap >= 45.0:
        grade = "moderate_blur"
    else:
        grade = "severe_blur"
    return round(med_lap, 2), grade


def audit_video(
    video_path: str | Path,
    sample_count: int = 24,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    capture_fps_override: Optional[float] = None,
    dedup_regular_pulldown: bool = True,
) -> VideoAudit:
    """Audit resolution, timebase/pulldown integrity, freeze frames, cuts, and central-ROI sharpness."""
    path_str = str(video_path)
    w, h, container_fps, total_nb, _ = _probe_ffprobe(path_str)

    sf = max(0, int(start_frame))
    ef = min(total_nb, int(end_frame)) if end_frame is not None else total_nb
    if ef <= sf:
        ef = max(sf + 1, total_nb)

    cad = analyze_frame_cadence(path_str, start_frame=sf, end_frame=ef)
    raw_nb = int(cad["raw_num_frames"])
    held_mask: np.ndarray = cad["held_mask"]
    held_ratio = float(cad["held_frame_ratio"])
    longest_run = int(cad["longest_held_run_frames"])
    cadence: Optional[int] = cad["regular_repeat_cadence"]
    scene_cuts: List[int] = cad["scene_cut_indices"]
    med_lap = float(cad["median_laplacian_var"])

    raw_indices = [sf + i for i in range(raw_nb)]
    warnings: List[str] = []
    timebase_unreliable = False
    timebase_status = "verified_native"

    # Classify timebase integrity and deduplicate regular pulldown frames cleanly (EXP-007 B2)
    if longest_run >= 4 or held_ratio >= 0.35:
        timebase_status = "edited_or_frozen"
        timebase_unreliable = True
        source_indices = raw_indices
        eff_fps = float(capture_fps_override) if (capture_fps_override and capture_fps_override > 0) else container_fps
        warnings.append(
            f"Unreliable/edited timebase: {held_ratio:.0%} repeated frames (longest held run={longest_run} frames); "
            "clip contains slow-motion retiming or freeze frames. Time-dependent velocities/durations are gated."
        )
    elif cadence is not None and cadence >= 2 and held_ratio >= 0.08:
        # Regular pulldown cadence (e.g., 25 fps in 29.97 -> repeat every 6th; 24 in 30 -> repeat every 5th)
        if dedup_regular_pulldown:
            source_indices = [sf + i for i in range(raw_nb) if not bool(held_mask[i])]
            if len(source_indices) < 6:
                source_indices = raw_indices
        else:
            source_indices = raw_indices
        unique_factor = len(source_indices) / max(raw_nb, 1)
        if capture_fps_override is not None and capture_fps_override > 0:
            # Caller supplied a slow-mo capture rate on a clip that ALSO contains regular repeats (e.g., SloPro 1-in-5)
            eff_fps = float(capture_fps_override) * unique_factor
            timebase_status = "retimed_unverified_override"
            timebase_unreliable = True
            warnings.append(
                f"Container has regular 1-in-{cadence} repeated frames ({held_ratio:.0%} repeats) under manual "
                f"capture-FPS override ({capture_fps_override:.1f} -> {eff_fps:.1f} unique fps); "
                "true physical capture rate is unverified."
            )
        else:
            eff_fps = container_fps * unique_factor
            timebase_status = "pulldown_deduped"
            warnings.append(
                f"Regular 1-in-{cadence} pulldown detected ({held_ratio:.0%} repeated frames); "
                f"dropped {raw_nb - len(source_indices)} duplicate frames and adjusted FPS from "
                f"{container_fps:.2f} to {eff_fps:.2f} unique fps."
            )
    else:
        source_indices = raw_indices
        if capture_fps_override is not None and capture_fps_override > 0:
            eff_fps = float(capture_fps_override)
            timebase_status = "retimed_unverified_override"
            warnings.append(
                f"Using caller-specified true capture FPS={eff_fps:.1f} fps (slow-motion playback override)."
            )
        else:
            eff_fps = container_fps
            if held_ratio >= 0.12 or longest_run >= 2:
                timebase_status = "edited_or_frozen"
                timebase_unreliable = True
                warnings.append(
                    f"Irregular repeated frames detected ({held_ratio:.0%}, max run={longest_run}); "
                    "time-dependent velocities may be biased."
                )

    if scene_cuts:
        warnings.append(
            f"Detected {len(scene_cuts)} abrupt scene/graphic cut(s) at relative frame(s) {scene_cuts[:5]}; "
            "verify clip does not splice multiple shots or freeze-frame overlays."
        )

    proc_nb = len(source_indices)
    slice_dur = proc_nb / max(eff_fps, 1.0)

    if med_lap >= 140.0:
        blur_grade = "crisp"
    elif med_lap >= 45.0:
        blur_grade = "moderate_blur"
    else:
        blur_grade = "severe_blur"

    arm_deg_per_frame = NOMINAL_FAST_ARM_DPS / max(eff_fps, 1.0)
    ball_m_per_frame = NOMINAL_FAST_BALL_MPS / max(eff_fps, 1.0)

    if eff_fps < 90.0:
        warnings.append(
            f"Effective unique frame rate is {eff_fps:.1f} fps (~{arm_deg_per_frame:.1f} deg arm sweep/frame); "
            "release-instant kinematics carry elevated temporal quantization error."
        )
    if blur_grade == "severe_blur":
        warnings.append(
            f"Low interior sharpness detected (Laplacian var={med_lap:.1f}); motion blur or compression "
            "will widen joint-angle uncertainty intervals."
        )
    if min(w, h) < 480:
        warnings.append(f"Low video resolution ({w}x{h}) degrades distal limb keypoint localization.")

    return VideoAudit(
        video_path=path_str,
        width=w,
        height=h,
        fps=round(eff_fps, 3),
        container_fps=round(container_fps, 3),
        num_frames=proc_nb,
        raw_num_frames=raw_nb,
        duration_s=round(slice_dur, 3),
        median_laplacian_var=round(med_lap, 2),
        blur_grade=blur_grade,
        blur_roi_method="center_crop",
        fps_grade=_fps_grade(eff_fps),
        est_arm_deg_per_frame=round(arm_deg_per_frame, 2),
        est_ball_m_per_frame=round(ball_m_per_frame, 3),
        timebase_status=timebase_status,
        timebase_unreliable=timebase_unreliable,
        held_frame_ratio=held_ratio,
        longest_held_run_frames=longest_run,
        regular_repeat_cadence=cadence,
        scene_cut_indices=scene_cuts,
        source_frame_indices=source_indices,
        warnings=warnings,
    )


def detect_camera_motion(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    bowler_boxes_xyxy: Optional[np.ndarray] = None,
    source_frame_indices: Optional[List[int]] = None,
    fps_hint: float = 30.0,
) -> Tuple[bool, float, Dict[str, float]]:
    """Detect camera pan, vertical tilt, AND focal zoom robustly via short-baseline 4-DOF similarity RANSAC.

    Fixes EXP-007 B14:
      - Uses short-baseline consecutive/near-consecutive frame pairs `(idx, idx + dt_step)` across up to 28
        anchor points (preventing Farneback/KLT aliasing on long 300-frame panning clips!)
      - Fits 4-DOF similarity transform `(tx, ty, scale, rot)` via RANSAC on masked textured background features
        (catching horizontal pan, vertical tilt, AND zoom while staying False on static cameras with moving TVs).

    Returns:
        (is_moving_camera, total_motion_fraction, motion_metrics_dict)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0.0, {"pan_x_frac": 0.0, "tilt_y_frac": 0.0, "zoom_frac": 0.0}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_frame_indices and len(source_frame_indices) >= 4:
        frame_list = [int(i) for i in source_frame_indices]
    else:
        sf = max(0, int(start_frame))
        ef = min(total, int(end_frame)) if end_frame is not None else total
        frame_list = list(range(sf, ef))

    n_seq = len(frame_list)
    if n_seq < 4:
        cap.release()
        return False, 0.0, {"pan_x_frac": 0.0, "tilt_y_frac": 0.0, "zoom_frac": 0.0}

    # Short baseline dt_step ensures pixel displacement between pair frames stays small (<15 px) even on 300+ frame pans
    dt_step = max(1, min(3, int(round(max(fps_hint, 20.0) / 30.0))))
    n_anchors = min(28, max(4, (n_seq - dt_step) // max(dt_step, 1)))
    anchor_pos = np.unique(np.linspace(0, max(0, n_seq - 1 - dt_step), n_anchors, dtype=int))

    # Map needed absolute source frames for a single sequential read (avoiding GOP seek drift)
    needed_seq_pos = sorted(set(int(p) for p in anchor_pos) | set(int(p + dt_step) for p in anchor_pos))
    needed_abs = {frame_list[p]: p for p in needed_seq_pos}
    target_w = 360
    scale_x = target_w / max(w_orig, 1)
    target_h = max(24, int(round(h_orig * scale_x)))
    scale_y = target_h / max(h_orig, 1)

    seq_grays: Dict[int, np.ndarray] = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_list[needed_seq_pos[0]])
    for abs_f in range(frame_list[needed_seq_pos[0]], frame_list[needed_seq_pos[-1]] + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        seq_p = needed_abs.get(abs_f)
        if seq_p is not None:
            g = cv2.cvtColor(cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            seq_grays[seq_p] = g
    cap.release()

    if len(seq_grays) < 4:
        return False, 0.0, {"pan_x_frac": 0.0, "tilt_y_frac": 0.0, "zoom_frac": 0.0}

    # Build static letterbox + bright-watermark exclusion mask across sampled frames
    stack = np.stack([seq_grays[p] for p in sorted(seq_grays)], axis=0).astype(np.float32)
    med_img = np.median(stack, axis=0)
    std_img = np.std(stack, axis=0)
    non_letterbox = med_img > 18.0
    bright_static_overlay = (med_img > 230.0) & (std_img < 1.2)
    scene_med_std = float(np.median(std_img[non_letterbox])) if np.any(non_letterbox) else 0.0
    if scene_med_std > 6.0:
        valid_bg = non_letterbox & (std_img > 1.5) & (~bright_static_overlay)
    else:
        valid_bg = non_letterbox & (~bright_static_overlay)

    tx_rates: List[float] = []
    ty_rates: List[float] = []
    log_s_rates: List[float] = []
    cx, cy = 0.5 * target_w, 0.5 * target_h

    for p in anchor_pos:
        p0, p1 = int(p), int(p + dt_step)
        if p0 not in seq_grays or p1 not in seq_grays:
            continue
        g0, g1 = seq_grays[p0], seq_grays[p1]
        mask = valid_bg.copy()
        if bowler_boxes_xyxy is not None and p0 < len(bowler_boxes_xyxy):
            bx0, by0, bx1, by1 = bowler_boxes_xyxy[p0]
            x0 = max(0, int(bx0 * scale_x) - 20)
            y0 = max(0, int(by0 * scale_y) - 20)
            x1 = min(target_w, int(bx1 * scale_x) + 20)
            y1 = min(target_h, int(by1 * scale_y) + 20)
            mask[y0:y1, x0:x1] = False

        mask_u8 = (mask.astype(np.uint8) * 255)
        pts0 = cv2.goodFeaturesToTrack(
            g0, maxCorners=260, qualityLevel=0.015, minDistance=8, mask=mask_u8, blockSize=5
        )
        tx_step, ty_step, log_s_step = 0.0, 0.0, 0.0
        solved_klt = False

        if pts0 is not None and len(pts0) >= 12:
            pts1, st, _ = cv2.calcOpticalFlowPyrLK(
                g0, g1, pts0, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            # Forward-backward consistency filter rejects foreground/occlusion outliers
            pts0_back, st_b, _ = cv2.calcOpticalFlowPyrLK(
                g1, g0, pts1, None, winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
            )
            fb_err = np.linalg.norm((pts0 - pts0_back).reshape(-1, 2), axis=1)
            ok_m = (st.reshape(-1) == 1) & (st_b.reshape(-1) == 1) & (fb_err < 1.2)
            p_src = pts0.reshape(-1, 2)[ok_m]
            p_dst = pts1.reshape(-1, 2)[ok_m]
            if len(p_src) >= 8:
                M, inliers = cv2.estimateAffinePartial2D(
                    p_src, p_dst, method=cv2.RANSAC, ransacReprojThreshold=1.5, maxIters=1500
                )
                if M is not None and inliers is not None and int(inliers.sum()) >= 6:
                    # Evaluate pure translation at frame center (decoupled from scale around center!)
                    a, b, tx_raw = float(M[0, 0]), float(M[0, 1]), float(M[0, 2])
                    c, d, ty_raw = float(M[1, 0]), float(M[1, 1]), float(M[1, 2])
                    sc_step = float(np.clip(math.hypot(a, b), 0.75, 1.35))
                    dst_cx = a * cx + b * cy + tx_raw
                    dst_cy = c * cx + d * cy + ty_raw
                    tx_step = dst_cx - cx
                    ty_step = dst_cy - cy
                    log_s_step = math.log(sc_step)
                    solved_klt = True

        if not solved_klt:
            # Dense Farneback fallback on low-corner scenes
            flow = cv2.calcOpticalFlowFarneback(
                g0, g1, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            g0_f = g0.astype(np.float32)
            tex = np.hypot(cv2.Sobel(g0_f, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g0_f, cv2.CV_32F, 0, 1, ksize=3)) > 8.0
            m2 = mask & tex
            if int(m2.sum()) < 150:
                m2 = mask
            if int(m2.sum()) >= 50:
                tx_step = float(np.median(flow[..., 0][m2]))
                ty_step = float(np.median(flow[..., 1][m2]))
                log_s_step = 0.0

        # Normalize to per-unique-frame rate
        tx_rates.append(tx_step / dt_step)
        ty_rates.append(ty_step / dt_step)
        log_s_rates.append(log_s_step / dt_step)

    if not tx_rates:
        return False, 0.0, {"pan_x_frac": 0.0, "tilt_y_frac": 0.0, "zoom_frac": 0.0}

    # Integrate per-frame rates across the full clip span `(n_seq - 1)` frames
    span = float(max(n_seq - 1, 1))
    net_tx = float(np.mean(tx_rates)) * span
    net_ty = float(np.mean(ty_rates)) * span
    # Also track peak cumulative excursion (catches pan-then-pan-back!)
    seg_w = span / max(len(tx_rates), 1)
    cum_tx = np.cumsum(np.asarray(tx_rates) * seg_w)
    cum_ty = np.cumsum(np.asarray(ty_rates) * seg_w)
    cum_ls = np.cumsum(np.asarray(log_s_rates) * seg_w)

    exc_x = max(abs(net_tx), float(np.ptp(np.concatenate([[0.0], cum_tx]))))
    exc_y = max(abs(net_ty), float(np.ptp(np.concatenate([[0.0], cum_ty]))))
    pan_x_frac = exc_x / float(target_w)
    tilt_y_frac = exc_y / float(target_h)
    trans_frac = math.hypot(exc_x, exc_y) / float(max(target_w, target_h))

    net_zoom = abs(math.exp(float(np.mean(log_s_rates)) * span) - 1.0)
    exc_zoom = abs(math.exp(float(np.ptp(np.concatenate([[0.0], cum_ls])))) - 1.0)
    zoom_frac = max(net_zoom, exc_zoom)

    # Check coherent directional motion to reject tiny zero-mean jitter
    sign_x = 1.0 if net_tx >= 0.0 else -1.0
    sign_y = 1.0 if net_ty >= 0.0 else -1.0
    coh_x = sum(1 for r in tx_rates if (r * sign_x) >= 0.35)
    coh_y = sum(1 for r in ty_rates if (r * sign_y) >= 0.35)
    min_coh = max(3, int(0.40 * len(tx_rates)))

    is_moving = (
        trans_frac >= 0.065
        or pan_x_frac >= 0.075
        or tilt_y_frac >= 0.075
        or zoom_frac >= 0.06
        or (coh_x >= min_coh and pan_x_frac >= 0.045)
        or (coh_y >= min_coh and tilt_y_frac >= 0.045)
    )
    total_frac = round(max(trans_frac, pan_x_frac, tilt_y_frac, zoom_frac), 4)
    metrics = {
        "pan_x_frac": round(pan_x_frac, 4),
        "tilt_y_frac": round(tilt_y_frac, 4),
        "zoom_frac": round(zoom_frac, 4),
        "trans_frac": round(trans_frac, 4),
    }
    return bool(is_moving), total_frac, metrics


def detect_camera_panning(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    bowler_boxes_xyxy: Optional[np.ndarray] = None,
    source_frame_indices: Optional[List[int]] = None,
    fps_hint: float = 30.0,
) -> Tuple[bool, float]:
    """Backward-compatible 2-tuple wrapper around `detect_camera_motion` (pan, tilt, and zoom)."""
    is_moving, total_frac, _ = detect_camera_motion(
        video_path=video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        bowler_boxes_xyxy=bowler_boxes_xyxy,
        source_frame_indices=source_frame_indices,
        fps_hint=fps_hint,
    )
    return is_moving, total_frac
