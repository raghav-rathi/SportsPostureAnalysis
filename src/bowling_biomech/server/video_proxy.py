"""Phase 2 (M2A): 3-Tier True-FPS Resolver & Zero-Tear All-Intra (-g 1 -bf 0) Scrub Proxy Builder."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np


@dataclass
class ProbedVideoMeta:
    """Metadata resolved from QuickTime/ffprobe container inspection."""

    width: int
    height: int
    container_fps: float
    quicktime_capture_fps: Optional[float]
    num_frames: int
    duration_s: float
    is_probable_slomo_container: bool
    probe_notes: List[str]


def probe_video_metadata(video_path: str | Path) -> ProbedVideoMeta:
    """Tier-1 metadata probe inspecting QuickTime slow-mo atoms (`com.apple.quicktime.video-framerate`) & ffprobe."""
    p = Path(video_path)
    notes: List[str] = []
    qt_fps: Optional[float] = None
    w, h, n_frames = 0, 0, 0
    fps_container = 30.0

    if shutil.which("ffprobe") and p.exists():
        try:
            cmd = [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", "-show_streams", str(p),
            ]
            out = subprocess.check_output(cmd, text=True, timeout=10)
            info = json.loads(out)
            fmt_tags = (info.get("format") or {}).get("tags") or {}
            for k, v in fmt_tags.items():
                if "video-framerate" in k.lower() or "capture_fps" in k.lower():
                    try:
                        qt_fps = float(v)
                        notes.append(f"Detected QuickTime capture rate tag {k}={qt_fps:.1f} fps.")
                    except ValueError:
                        pass
            for st in info.get("streams") or []:
                if st.get("codec_type") == "video":
                    w = int(st.get("width") or 0)
                    h = int(st.get("height") or 0)
                    rfr = str(st.get("avg_frame_rate") or st.get("r_frame_rate") or "30/1")
                    if "/" in rfr:
                        num, den = rfr.split("/", 1)
                        if float(den) > 0:
                            fps_container = float(num) / float(den)
                    if st.get("nb_frames") and str(st["nb_frames"]).isdigit():
                        n_frames = int(st["nb_frames"])
                    break
        except Exception:
            pass

    if w == 0 or h == 0 or n_frames == 0:
        cap = cv2.VideoCapture(str(p))
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or w)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or h)
            fps_cv = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if fps_cv > 1.0:
                fps_container = fps_cv
            n_frames = max(n_frames, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
            cap.release()

    probable_slomo = bool(qt_fps is not None and qt_fps > fps_container * 1.5)
    dur = n_frames / max(fps_container, 1.0)
    return ProbedVideoMeta(
        width=w,
        height=h,
        container_fps=round(fps_container, 3),
        quicktime_capture_fps=round(qt_fps, 2) if qt_fps is not None else None,
        num_frames=n_frames,
        duration_s=round(dur, 3),
        is_probable_slomo_container=probable_slomo,
        probe_notes=notes,
    )


def build_all_intra_scrub_proxy(
    video_path: str | Path,
    out_mp4: str | Path,
    source_frame_indices: Optional[List[int]] = None,
    proxy_fps: float = 30.0,
    max_width: int = 1280,
) -> Path:
    """Build a CFR All-Intra (`-g 1 -bf 0 -pix_fmt yuv420p +faststart`) proxy MP4 for 0.0-frame browser scrubbing.

    Decodes the exact `source_frame_indices` sequentially (preserving zero-drift pulldown dedup!)
    so frame `i` of the proxy matches telemetry index `i` identically (`t_i = i / proxy_fps`).
    """
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for scrub proxy: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    indices = [int(i) for i in source_frame_indices] if source_frame_indices else list(range(max(total, 1)))
    min_f, max_f = min(indices), max(indices)
    need = set(indices)
    if min_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)

    decoded: Dict[int, np.ndarray] = {}
    for f_idx in range(min_f, max_f + 1):
        ok, img = cap.read()
        if not ok or img is None:
            break
        if f_idx in need:
            decoded[f_idx] = img
    cap.release()

    frames = [decoded[i] for i in indices if i in decoded]
    if not frames:
        raise RuntimeError(f"No frames decoded for scrub proxy from {video_path}")

    h0, w0 = frames[0].shape[:2]
    if w0 > max_width:
        scale = max_width / float(w0)
        w_out = int(round(w0 * scale)) // 2 * 2
        h_out = int(round(h0 * scale)) // 2 * 2
    else:
        w_out = w0 // 2 * 2
        h_out = h0 // 2 * 2

    tmp_raw = out.with_suffix(".proxy_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), float(proxy_fps), (w_out, h_out))
    for fr in frames:
        if fr.shape[1] != w_out or fr.shape[0] != h_out:
            fr = cv2.resize(fr, (w_out, h_out), interpolation=cv2.INTER_AREA)
        vw.write(fr)
    vw.release()

    if shutil.which("ffmpeg"):
        # Prefer Apple Silicon VideoToolbox on macOS for <0.5 s All-Intra transcoding, with libx264 fallback
        encoders = (
            [
                ["-c:v", "h264_videotoolbox", "-q:v", "65", "-g", "1", "-bf", "0"],
                ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-g", "1", "-bf", "0"],
            ]
            if platform.system() == "Darwin"
            else [["-c:v", "libx264", "-preset", "ultrafast", "-crf", "20", "-g", "1", "-bf", "0"]]
        )
        for enc_args in encoders:
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_raw),
                *enc_args, "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
            ]
            res = subprocess.run(cmd, capture_output=True, check=False)
            if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
                tmp_raw.unlink(missing_ok=True)
                return out

    tmp_raw.replace(out)
    return out


def render_synthetic_proxy_video(
    joints2d_px: np.ndarray,
    width: int,
    height: int,
    out_mp4: str | Path,
    proxy_fps: float = 30.0,
    title: str = "Synthetic Pitch-Calibrated Delivery",
    mcc_wireframe_2d: Optional[Dict[str, List[float]]] = None,
) -> Path:
    """Generate a crisp All-Intra synthetic video proxy for offline/demo deliveries without raw .mp4 files."""
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    w_out = (width // 2) * 2
    h_out = (height // 2) * 2
    tmp_raw = out.with_suffix(".synth_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), float(proxy_fps), (w_out, h_out))

    n = len(joints2d_px)
    for i in range(n):
        img = np.full((h_out, w_out, 3), (18, 24, 33), dtype=np.uint8)
        # Subtle turf gradient band on lower half
        cv2.rectangle(img, (0, int(0.55 * h_out)), (w_out, h_out), (22, 36, 30), -1)
        # Draw projected MCC pitch lines if available
        if mcc_wireframe_2d:
            for a, b in [
                ("bowling_crease_off", "bowling_crease_leg"),
                ("popping_crease_off", "popping_crease_leg"),
                ("bowling_crease_off", "popping_crease_off"),
                ("bowling_crease_leg", "popping_crease_leg"),
                ("stump_middle_base", "stump_middle_top"),
            ]:
                if a in mcc_wireframe_2d and b in mcc_wireframe_2d:
                    p1 = (int(mcc_wireframe_2d[a][0]), int(mcc_wireframe_2d[a][1]))
                    p2 = (int(mcc_wireframe_2d[b][0]), int(mcc_wireframe_2d[b][1]))
                    col = (80, 210, 255) if "stump" in a else (160, 195, 175)
                    cv2.line(img, p1, p2, col, 2, cv2.LINE_AA)
        cv2.putText(img, f"{title} | Frame {i}/{n - 1}", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (160, 185, 215), 2, cv2.LINE_AA)
        vw.write(img)
    vw.release()

    if shutil.which("ffmpeg"):
        res = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_raw),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "22",
                "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
            ],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            tmp_raw.unlink(missing_ok=True)
            return out
    tmp_raw.replace(out)
    return out
