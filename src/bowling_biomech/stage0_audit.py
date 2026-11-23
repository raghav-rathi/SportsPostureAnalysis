"""Stage 0: Video Capture Quality, Frame-Rate Physics, Sharpness, & Background Panning Audit."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .schemas import VideoAudit

PEAK_ARM_ANGULAR_VEL_DPS = 2300.0  # Elite mean ~40 rad/s (35-45 rad/s in literature)
NOMINAL_FAST_BALL_MPS = 38.89      # 140 km/h


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


def audit_video(
    video_path: str | Path,
    sample_count: int = 24,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
) -> VideoAudit:
    """Audit resolution, FPS physics, and Laplacian sharpness strictly on [start_frame, end_frame)."""
    path_str = str(video_path)
    w, h, fps, total_nb, _ = _probe_ffprobe(path_str)

    sf = max(0, int(start_frame))
    ef = min(total_nb, int(end_frame)) if end_frame is not None else total_nb
    if ef <= sf:
        ef = max(sf + 1, total_nb)
    slice_nb = max(1, ef - sf)
    slice_dur = slice_nb / max(fps, 1.0)

    cap = cv2.VideoCapture(path_str)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video file: {path_str}")

    indices = np.linspace(sf, max(ef - 1, sf), min(slice_nb, sample_count), dtype=int)
    lap_vars: List[float] = []
    for idx in np.unique(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Exclude pure black letterbox borders when computing sharpness
        non_black = gray > 16
        if np.any(non_black):
            ys, xs = np.where(non_black)
            gray = gray[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        scale = 720.0 / max(gray.shape[0], 1)
        if abs(scale - 1.0) > 0.05:
            gray = cv2.resize(gray, (max(1, int(gray.shape[1] * scale)), 720))
        lap_vars.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    cap.release()

    med_lap = float(np.median(lap_vars)) if lap_vars else 0.0
    if med_lap >= 140.0:
        blur_grade = "crisp"
    elif med_lap >= 45.0:
        blur_grade = "moderate_blur"
    else:
        blur_grade = "severe_blur"

    if fps >= 200.0:
        fps_grade = "high_speed_240"
    elif fps >= 100.0:
        fps_grade = "slow_mo_120"
    elif fps >= 48.0:
        fps_grade = "medium_60"
    elif fps >= 28.0:
        fps_grade = "standard_30"
    else:
        fps_grade = "low_25"

    arm_deg_per_frame = PEAK_ARM_ANGULAR_VEL_DPS / max(fps, 1.0)
    ball_m_per_frame = NOMINAL_FAST_BALL_MPS / max(fps, 1.0)

    warnings: List[str] = []
    if fps < 90.0:
        warnings.append(
            f"Frame rate is {fps:.1f} fps (~{arm_deg_per_frame:.1f} deg bowling-arm sweep per frame); "
            "release-instant kinematics carry elevated temporal quantization error. "
            "Record at 120-240 fps slow-motion outdoors for precision analysis."
        )
    if blur_grade == "severe_blur":
        warnings.append(
            f"Low frame sharpness detected (Laplacian var={med_lap:.1f}); motion blur or compression "
            "will widen joint-angle confidence intervals."
        )
    if min(w, h) < 480:
        warnings.append(f"Low video resolution ({w}x{h}) degrades distal limb keypoint localization.")

    return VideoAudit(
        video_path=path_str,
        width=w,
        height=h,
        fps=round(fps, 3),
        num_frames=slice_nb,
        duration_s=round(slice_dur, 3),
        median_laplacian_var=round(med_lap, 2),
        blur_grade=blur_grade,
        fps_grade=fps_grade,
        est_arm_deg_per_frame=round(arm_deg_per_frame, 2),
        est_ball_m_per_frame=round(ball_m_per_frame, 3),
        warnings=warnings,
    )


def detect_camera_panning(
    video_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    bowler_boxes_xyxy: Optional[np.ndarray] = None,
    sample_step: int = 2,
) -> Tuple[bool, float]:
    """Detect background camera panning robustly despite letterbox bars and static watermarks.

    Uses dense Farneback optical flow on non-letterbox, non-static-overlay, non-bowler regions.
    Returns:
        (is_panning_camera, cumulative_pan_fraction_of_width)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sf = max(0, int(start_frame))
    ef = min(total, int(end_frame)) if end_frame is not None else total
    if ef - sf < 6:
        cap.release()
        return False, 0.0

    # Sample up to ~28 evenly spaced frames across the delivery slice
    n_samples = min(28, max(6, (ef - sf) // max(sample_step, 1)))
    idxs = np.linspace(sf, ef - 1, n_samples, dtype=int)
    grays: List[np.ndarray] = []
    rel_idxs: List[int] = []
    target_w = 320
    scale_x = 1.0
    scale_y = 1.0

    for idx in np.unique(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        h, w = frame.shape[:2]
        scale_x = target_w / max(w, 1)
        target_h = max(16, int(round(h * scale_x)))
        scale_y = target_h / max(h, 1)
        g = cv2.cvtColor(cv2.resize(frame, (target_w, target_h)), cv2.COLOR_BGR2GRAY)
        grays.append(g)
        rel_idxs.append(int(idx) - sf)
    cap.release()

    if len(grays) < 4:
        return False, 0.0

    stack = np.stack(grays, axis=0).astype(np.float32)
    # Exclude black letterbox bars (median intensity <= 18) and burned-in bright static caption overlays.
    # Crucially, DO NOT drop low-std pixels on a static camera (where >90% of true background has std < 1.8!);
    # only exclude frozen digital overlays (std < 1.5) when the overall scene itself is changing (scene_med_std > 6.0).
    med_img = np.median(stack, axis=0)
    std_img = np.std(stack, axis=0)
    non_letterbox = med_img > 18.0
    bright_static_overlay = (med_img > 230.0) & (std_img < 1.2)
    scene_med_std = float(np.median(std_img[non_letterbox])) if np.any(non_letterbox) else 0.0
    if scene_med_std > 6.0:
        valid_bg = non_letterbox & (std_img > 1.5) & (~bright_static_overlay)
    else:
        valid_bg = non_letterbox & (~bright_static_overlay)

    dx_list: List[float] = []
    for k in range(len(grays) - 1):
        g0, g1 = grays[k], grays[k + 1]
        flow = cv2.calcOpticalFlowFarneback(
            g0, g1, None, pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        # Keep spatially textured pixels on the current frame so static textured backgrounds
        # (nets, seats, walls) anchor median flow at ~0 while fast-panning backgrounds stay crisp.
        g0_f = g0.astype(np.float32)
        gx = cv2.Sobel(g0_f, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g0_f, cv2.CV_32F, 0, 1, ksize=3)
        textured = np.hypot(gx, gy) > 8.0
        mask = valid_bg & textured
        # Exclude bowler bounding box (with generous padding) so athlete motion doesn't masquerade as pan
        if bowler_boxes_xyxy is not None and rel_idxs[k] < len(bowler_boxes_xyxy):
            bx0, by0, bx1, by1 = bowler_boxes_xyxy[rel_idxs[k]]
            x0 = max(0, int(bx0 * scale_x) - 18)
            y0 = max(0, int(by0 * scale_y) - 18)
            x1 = min(target_w, int(bx1 * scale_x) + 18)
            y1 = min(g0.shape[0], int(by1 * scale_y) + 18)
            mask[y0:y1, x0:x1] = False

        if int(mask.sum()) < 200:
            mask = valid_bg.copy()
            if bowler_boxes_xyxy is not None and rel_idxs[k] < len(bowler_boxes_xyxy):
                mask[y0:y1, x0:x1] = False
        if int(mask.sum()) < 50:
            mask = non_letterbox
        if int(mask.sum()) < 50:
            continue
        med_dx = float(np.median(flow[..., 0][mask]))
        dx_list.append(med_dx)

    if not dx_list:
        return False, 0.0

    # Net coherent horizontal background pan as a fraction of frame width
    net_sum = float(np.sum(dx_list))
    net_pan_px = abs(net_sum)
    pan_frac = net_pan_px / float(target_w)
    sign_dir = 1.0 if net_sum >= 0.0 else -1.0
    # Require coherent per-step flow in the dominant pan direction (>0.45 px/step across 320px width)
    coherent_steps = sum(1 for d in dx_list if (d * sign_dir) >= 0.45)
    is_panning = (pan_frac >= 0.09) or (coherent_steps >= max(3, int(0.45 * len(dx_list))) and pan_frac >= 0.05)
    return bool(is_panning), round(float(pan_frac), 4)
