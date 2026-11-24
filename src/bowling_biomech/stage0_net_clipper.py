"""Phase 2 (M2B): Hybrid Multi-Ball Net-Session Auto-Clipper.

Scans continuous multi-delivery net videos at a coarse low-FPS stride (~6 fps) using
downsampled frame-difference motion energy + upper-half circumduction burst detection
to isolate candidate `[start_frame, end_frame]` delivery slices with JPEG thumbnail previews
for 1-click user confirmation before GPU inference.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import math
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class DeliveryCandidateClip:
    """Candidate delivery segment detected in a longer net-session video."""

    candidate_id: int
    start_frame: int
    end_frame: int
    peak_frame: int
    start_s: float
    end_s: float
    duration_s: float
    confidence: float
    motion_score: float
    thumbnail_b64: Optional[str] = None


def detect_delivery_candidates_from_signal(
    motion_energy: np.ndarray,
    sample_frames: np.ndarray,
    fps: float,
    total_frames: int,
    pre_peak_s: float = 1.6,
    post_peak_s: float = 0.9,
    min_separation_s: float = 3.0,
    z_thresh: float = 1.4,
    max_candidates: int = 30,
) -> List[DeliveryCandidateClip]:
    """Locate delivery burst peaks on a sampled motion-energy trace via adaptive NMS."""
    if len(motion_energy) < 3 or fps <= 0.0:
        return []

    e = np.asarray(motion_energy, dtype=np.float64)
    # 3-sample smoothing to reject single-frame exposure flashes
    if len(e) >= 3:
        kernel = np.array([0.25, 0.50, 0.25], dtype=np.float64)
        e_sm = np.convolve(e, kernel, mode="same")
    else:
        e_sm = e.copy()

    med = float(np.median(e_sm))
    mad = float(np.median(np.abs(e_sm - med))) * 1.4826
    sd = max(mad, float(np.std(e_sm)) * 0.5, 1e-4)
    z = (e_sm - med) / sd

    # Find local maxima above z_thresh and minimum absolute motion floor
    abs_floor = max(med * 1.35, 1.5)
    peaks = []
    for i in range(1, len(e_sm) - 1):
        if e_sm[i] >= e_sm[i - 1] and e_sm[i] >= e_sm[i + 1] and z[i] >= z_thresh and e_sm[i] >= abs_floor:
            peaks.append((float(z[i]), int(i), float(e_sm[i])))

    # Fallback: if clip is short (<6 s) and has clear activity without multiple peaks, return single clip
    dur_total = total_frames / max(fps, 1.0)
    if not peaks and dur_total <= 6.5 and float(np.max(e_sm)) >= 1.5:
        pk_i = int(np.argmax(e_sm))
        peaks.append((max(float(z[pk_i]), 1.0), pk_i, float(e_sm[pk_i])))

    # Greedy NMS sorted by descending z-score
    peaks.sort(key=lambda x: x[0], reverse=True)
    min_sep_frames = int(round(min_separation_s * fps))
    selected = []
    for z_val, s_idx, score_val in peaks:
        f_pk = int(sample_frames[s_idx])
        if any(abs(f_pk - int(sample_frames[prev[1]])) < min_sep_frames for prev in selected):
            continue
        selected.append((z_val, s_idx, score_val))
        if len(selected) >= max_candidates:
            break

    selected.sort(key=lambda x: int(sample_frames[x[1]]))
    out: List[DeliveryCandidateClip] = []
    for c_id, (z_val, s_idx, score_val) in enumerate(selected, start=1):
        f_pk = int(sample_frames[s_idx])
        sf = max(0, int(round(f_pk - pre_peak_s * fps)))
        ef = min(total_frames, int(round(f_pk + post_peak_s * fps)))
        if ef - sf < max(6, int(0.4 * fps)):
            continue
        conf = float(np.clip(0.45 + 0.15 * min(z_val, 3.5), 0.45, 0.96))
        out.append(
            DeliveryCandidateClip(
                candidate_id=c_id,
                start_frame=sf,
                end_frame=ef,
                peak_frame=f_pk,
                start_s=round(sf / fps, 2),
                end_s=round(ef / fps, 2),
                duration_s=round((ef - sf) / fps, 2),
                confidence=round(conf, 3),
                motion_score=round(score_val, 2),
            )
        )
    return out


def scan_net_session_video(
    video_path: str | Path,
    scan_fps: float = 6.0,
    pre_peak_s: float = 1.6,
    post_peak_s: float = 0.9,
    min_separation_s: float = 3.0,
    include_thumbnails: bool = True,
) -> List[DeliveryCandidateClip]:
    """Scan a raw multi-delivery net video at ~6 fps and extract candidate delivery slices + thumbnails."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for net-session scanning: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    stride = max(1, int(round(fps / max(scan_fps, 1.0))))

    sample_frames: List[int] = []
    energies: List[float] = []
    prev_small: Optional[np.ndarray] = None
    f_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if f_idx % stride == 0:
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90), interpolation=cv2.INTER_AREA).astype(np.float32)
            if prev_small is not None:
                diff = np.abs(small - prev_small)
                # Emphasize active moving region (top 85% quantile focuses on bowler rather than quiet turf)
                upper_half = diff[:55, :]
                score = float(0.60 * np.percentile(diff, 92) + 0.40 * np.percentile(upper_half, 95))
                sample_frames.append(f_idx)
                energies.append(score)
            prev_small = small
        f_idx += 1

    total_frames = max(total_frames, f_idx)
    candidates = detect_delivery_candidates_from_signal(
        motion_energy=np.asarray(energies, dtype=np.float64),
        sample_frames=np.asarray(sample_frames, dtype=np.int64),
        fps=fps,
        total_frames=total_frames,
        pre_peak_s=pre_peak_s,
        post_peak_s=post_peak_s,
        min_separation_s=min_separation_s,
    )

    if include_thumbnails and candidates:
        for cand in candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cand.peak_frame)
            ok, thumb_img = cap.read()
            if ok and thumb_img is not None:
                th = cv2.resize(thumb_img, (240, 135), interpolation=cv2.INTER_AREA)
                ok_enc, buf = cv2.imencode(".jpg", th, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                if ok_enc:
                    cand.thumbnail_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

    cap.release()
    return candidates
