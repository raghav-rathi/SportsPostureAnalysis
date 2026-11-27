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


def slice_video_segment(
    video_path: str | Path,
    out_mp4: str | Path,
    start_frame: int,
    end_frame: Optional[int],
) -> Path:
    """Extract `[start_frame, end_frame]` into a compact MP4 locally (avoids double-uploading 300 MB net videos)."""
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source video for slicing: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if fps <= 1.0:
        fps = 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    sf = max(0, int(start_frame))
    ef = min(total - 1, int(end_frame)) if (end_frame is not None and total > 0) else max(total - 1, sf)
    if sf > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, sf)

    frames: List[np.ndarray] = []
    for _ in range(sf, ef + 1):
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded in slice [{sf}, {ef}] from {video_path}")

    h0, w0 = frames[0].shape[:2]
    tmp_raw = out.with_suffix(".slice_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w0, h0))
    for fr in frames:
        vw.write(fr)
    vw.release()

    if shutil.which("ffmpeg"):
        res = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_raw),
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
            ],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            tmp_raw.unlink(missing_ok=True)
            return out
    tmp_raw.replace(out)
    return out


def extract_proxy_frame_jpeg(
    proxy_mp4: str | Path,
    frame_idx: int,
    cache_dir: Optional[str | Path] = None,
    max_width: int = 480,
) -> bytes:
    """Extract and optionally disk-cache a crisp JPEG thumbnail from an All-Intra scrub proxy at `frame_idx`."""
    p = Path(proxy_mp4)
    if not p.exists():
        raise FileNotFoundError(f"Proxy video not found: {p}")

    safe_w = int(max(120, min(max_width, 1920)))
    idx_req = max(0, int(frame_idx))
    cache_path: Optional[Path] = None
    if cache_dir is not None:
        cdir = Path(cache_dir)
        cdir.mkdir(parents=True, exist_ok=True)
        cache_path = cdir / f"f_{idx_req}_w{safe_w}.jpg"
        if cache_path.exists() and cache_path.stat().st_size > 0:
            return cache_path.read_bytes()

    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open proxy video for frame extraction: {p}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    target = min(idx_req, max(total - 1, 0)) if total > 0 else idx_req
    if target > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ok, img = cap.read()
    if (not ok or img is None) and total > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, img = cap.read()
    cap.release()
    if not ok or img is None:
        raise RuntimeError(f"Could not decode frame {target} from {p}")

    h0, w0 = img.shape[:2]
    if w0 > safe_w:
        scale = safe_w / float(w0)
        img = cv2.resize(img, (safe_w, max(1, int(round(h0 * scale)))), interpolation=cv2.INTER_AREA)

    ok_enc, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 86])
    if not ok_enc:
        raise RuntimeError("Failed to encode JPEG thumbnail.")
    raw_bytes = buf.tobytes()
    if cache_path is not None:
        cache_path.write_bytes(raw_bytes)
    return raw_bytes


def _draw_capsule(
    img: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    r1: float,
    r2: float,
    fill_bgr: tuple[int, int, int],
    outline_bgr: tuple[int, int, int],
) -> None:
    """Draw a smooth tapered anatomical limb capsule between 2D joints `p1` and `p2`."""
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return
    v = p2 - p1
    length = float(np.hypot(v[0], v[1]))
    c1 = (int(round(p1[0])), int(round(p1[1])))
    c2 = (int(round(p2[0])), int(round(p2[1])))
    ir1 = max(2, int(round(r1)))
    ir2 = max(2, int(round(r2)))
    if length < 1.5:
        cv2.circle(img, c1, ir1, fill_bgr, -1, cv2.LINE_AA)
        cv2.circle(img, c1, ir1, outline_bgr, 1, cv2.LINE_AA)
        return
    nx, ny = -v[1] / length, v[0] / length
    poly = np.array(
        [
            [p1[0] + nx * r1, p1[1] + ny * r1],
            [p2[0] + nx * r2, p2[1] + ny * r2],
            [p2[0] - nx * r2, p2[1] - ny * r2],
            [p1[0] - nx * r1, p1[1] - ny * r1],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(img, poly, fill_bgr, cv2.LINE_AA)
    cv2.circle(img, c1, ir1, fill_bgr, -1, cv2.LINE_AA)
    cv2.circle(img, c2, ir2, fill_bgr, -1, cv2.LINE_AA)
    cv2.polylines(img, [poly], True, outline_bgr, 1, cv2.LINE_AA)


def render_synthetic_proxy_video(
    joints2d_px: np.ndarray,
    width: int,
    height: int,
    out_mp4: str | Path,
    proxy_fps: float = 30.0,
    title: str = "Synthetic Pitch-Calibrated Delivery",
    mcc_wireframe_2d: Optional[Dict[str, List[float]]] = None,
    joint_in_frame: Optional[np.ndarray] = None,
    bowling_arm: str = "right",
    br_frame_idx: Optional[int] = None,
) -> Path:
    """Generate a crisp All-Intra anatomical-silhouette studio proxy MP4 for offline/synthetic deliveries."""
    del title  # Never bake raw developer captions or frame counters into consumer video pixels
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    w_out = (width // 2) * 2
    h_out = (height // 2) * 2
    sx = w_out / max(float(width), 1.0)
    sy = h_out / max(float(height), 1.0)
    tmp_raw = out.with_suffix(".synth_tmp.mp4")
    vw = cv2.VideoWriter(str(tmp_raw), cv2.VideoWriter_fourcc(*"mp4v"), float(proxy_fps), (w_out, h_out))

    # SMPL-24 canonical indices
    PELV, LHIP, RHIP, LKNE, RKNE, LANK, RANK = 0, 1, 2, 4, 5, 7, 8
    LTOE, RTOE, NECK, HEAD = 10, 11, 12, 15
    LSHO, RSHO, LELB, RELB, LWRI, RWRI, LHAN, RHAN = 16, 17, 18, 19, 20, 21, 22, 23
    is_right = bowling_arm != "left"
    b_wri = RWRI if is_right else LWRI
    b_han = RHAN if is_right else LHAN

    n = len(joints2d_px)
    # Estimate ground horizon line from ankles/crease so the athlete stands grounded on the pitch
    ank_ys = joints2d_px[:, [LANK, RANK], 1].reshape(-1) * sy
    valid_ank = ank_ys[np.isfinite(ank_ys) & (ank_ys > 0) & (ank_ys < h_out)]
    base_horizon_y = int(np.clip(np.percentile(valid_ank, 20) - 70.0, 0.42 * h_out, 0.62 * h_out)) if len(valid_ank) else int(0.52 * h_out)

    # Precompute release ball velocity in 2D pixels if br_frame_idx is known
    ball_origin_px: Optional[np.ndarray] = None
    ball_vel_px: Optional[np.ndarray] = None
    if br_frame_idx is not None and 1 <= br_frame_idx < n:
        p_br = joints2d_px[br_frame_idx, b_han] * np.array([sx, sy])
        p_prev = joints2d_px[max(0, br_frame_idx - 2), b_han] * np.array([sx, sy])
        if np.all(np.isfinite(p_br)) and np.all(np.isfinite(p_prev)):
            ball_origin_px = p_br
            dv = (p_br - p_prev) / max(br_frame_idx - max(0, br_frame_idx - 2), 1)
            spd = max(float(np.hypot(dv[0], dv[1])), 12.0)
            dir_u = dv / max(float(np.hypot(dv[0], dv[1])), 1e-4)
            ball_vel_px = dir_u * max(spd * 1.45, 18.0)

    for i in range(n):
        # Vercel light-paper studio backdrop (#f6f6f7) + grounded turf plane (#ededef)
        img = np.full((h_out, w_out, 3), (247, 246, 246), dtype=np.uint8)
        cv2.rectangle(img, (0, base_horizon_y), (w_out, h_out), (239, 237, 237), -1)
        cv2.line(img, (0, base_horizon_y), (w_out, base_horizon_y), (222, 222, 226), 1, cv2.LINE_AA)

        # Subtle cut-turf pitch strip & MCC crease/stump geometry
        if mcc_wireframe_2d:
            quad_keys = ("bowling_crease_off", "bowling_crease_leg", "popping_crease_leg", "popping_crease_off")
            if all(k in mcc_wireframe_2d for k in quad_keys):
                b_off = np.array(mcc_wireframe_2d["bowling_crease_off"], dtype=np.float64) * [sx, sy]
                b_leg = np.array(mcc_wireframe_2d["bowling_crease_leg"], dtype=np.float64) * [sx, sy]
                p_leg = np.array(mcc_wireframe_2d["popping_crease_leg"], dtype=np.float64) * [sx, sy]
                p_off = np.array(mcc_wireframe_2d["popping_crease_off"], dtype=np.float64) * [sx, sy]
                fwd_off = p_off - b_off
                fwd_leg = p_leg - b_leg
                strip_poly = np.array(
                    [
                        b_off - 1.5 * fwd_off,
                        b_leg - 1.5 * fwd_leg,
                        p_leg + 5.0 * fwd_leg,
                        p_off + 5.0 * fwd_off,
                    ],
                    dtype=np.int32,
                )
                cv2.fillConvexPoly(img, strip_poly, (230, 230, 234), cv2.LINE_AA)

            for a, b in [
                ("bowling_crease_off", "bowling_crease_leg"),
                ("popping_crease_off", "popping_crease_leg"),
                ("bowling_crease_off", "popping_crease_off"),
                ("bowling_crease_leg", "popping_crease_leg"),
                ("return_crease_back_off", "bowling_crease_off"),
                ("return_crease_back_leg", "bowling_crease_leg"),
                ("stump_off_base", "stump_off_top"),
                ("stump_middle_base", "stump_middle_top"),
                ("stump_leg_base", "stump_leg_top"),
                ("stump_off_top", "stump_leg_top"),
            ]:
                if a in mcc_wireframe_2d and b in mcc_wireframe_2d:
                    p1 = (int(round(mcc_wireframe_2d[a][0] * sx)), int(round(mcc_wireframe_2d[a][1] * sy)))
                    p2 = (int(round(mcc_wireframe_2d[b][0] * sx)), int(round(mcc_wireframe_2d[b][1] * sy)))
                    col = (58, 122, 41) if "stump" in a else (120, 120, 126)
                    thick = 3 if "stump" in a else 2
                    cv2.line(img, p1, p2, col, thick, cv2.LINE_AA)

        pts = joints2d_px[i] * np.array([sx, sy])
        in_fr = joint_in_frame[i] if (joint_in_frame is not None and i < len(joint_in_frame)) else np.ones(len(pts), dtype=bool)
        if len(pts) >= 24:
            # Estimate apparent athlete scale in pixels for anatomically proportioned silhouette widths
            vis_y = pts[in_fr, 1] if np.any(in_fr) else pts[:, 1]
            vis_y = vis_y[np.isfinite(vis_y)]
            span_px = float(np.ptp(vis_y)) if len(vis_y) >= 2 else 220.0
            u = float(np.clip(span_px / 220.0, 0.55, 2.4))

            # Ground shadow ellipse beneath ankles/feet
            foot_pts = [pts[j] for j in (LANK, RANK, LTOE, RTOE) if in_fr[j] and np.all(np.isfinite(pts[j]))]
            if foot_pts:
                fx_m = int(round(float(np.mean([p[0] for p in foot_pts]))))
                fy_m = int(round(float(np.max([p[1] for p in foot_pts])) + 4 * u))
                sh_w = max(int(round(float(np.ptp([p[0] for p in foot_pts])) * 0.55 + 26 * u)), int(round(24 * u)))
                cv2.ellipse(img, (fx_m, fy_m), (sh_w, max(5, int(round(7 * u)))), 0, 0, 360, (214, 214, 218), -1, cv2.LINE_AA)

            def _ok(j_idx: int) -> bool:
                return bool(in_fr[j_idx]) and bool(np.all(np.isfinite(pts[j_idx])))

            far_fill, far_out = (182, 184, 192), (138, 140, 148)
            near_fill, near_out = (196, 198, 205), (126, 128, 136)
            torso_fill, torso_out = (188, 190, 198), (128, 130, 138)

            # Back/non-bowling limbs first
            nb_limbs = (
                [(RHIP, RKNE, 13, 10), (RKNE, RANK, 10, 7), (RANK, RTOE, 7, 5), (LSHO, LELB, 9, 7), (LELB, LWRI, 7, 5), (LWRI, LHAN, 5, 4)]
                if is_right
                else [(LHIP, LKNE, 13, 10), (LKNE, LANK, 10, 7), (LANK, LTOE, 7, 5), (RSHO, RELB, 9, 7), (RELB, RWRI, 7, 5), (RWRI, RHAN, 5, 4)]
            )
            for ja, jb, r1, r2 in nb_limbs:
                if _ok(ja) and _ok(jb):
                    _draw_capsule(img, pts[ja], pts[jb], r1 * u, r2 * u, far_fill, far_out)

            # Torso & pelvis polygon
            if all(_ok(j) for j in (LSHO, RSHO, RHIP, LHIP)):
                torso_poly = np.array([pts[LSHO], pts[RSHO], pts[RHIP], pts[LHIP]], dtype=np.int32)
                cv2.fillConvexPoly(img, torso_poly, torso_fill, cv2.LINE_AA)
                cv2.polylines(img, [torso_poly], True, torso_out, 1, cv2.LINE_AA)
            elif _ok(NECK) and _ok(PELV):
                _draw_capsule(img, pts[NECK], pts[PELV], 16 * u, 14 * u, torso_fill, torso_out)

            # Neck & head silhouette
            if _ok(NECK) and _ok(HEAD):
                _draw_capsule(img, pts[NECK], pts[HEAD], 6.5 * u, 7.5 * u, near_fill, near_out)
                hc = (int(round(pts[HEAD][0])), int(round(pts[HEAD][1])))
                hr = max(6, int(round(13.5 * u)))
                cv2.circle(img, hc, hr, near_fill, -1, cv2.LINE_AA)
                cv2.circle(img, hc, hr, near_out, 1, cv2.LINE_AA)

            # Lead leg & bowling arm on foreground layer
            fg_limbs = (
                [(LHIP, LKNE, 13.5, 10.5), (LKNE, LANK, 10.5, 7.5), (LANK, LTOE, 7.5, 5.5), (RSHO, RELB, 9.5, 7.5), (RELB, RWRI, 7.5, 5.5), (RWRI, RHAN, 5.5, 4.5)]
                if is_right
                else [(RHIP, RKNE, 13.5, 10.5), (RKNE, RANK, 10.5, 7.5), (RANK, RTOE, 7.5, 5.5), (LSHO, LELB, 9.5, 7.5), (LELB, LWRI, 7.5, 5.5), (LWRI, LHAN, 5.5, 4.5)]
            )
            for ja, jb, r1, r2 in fg_limbs:
                if _ok(ja) and _ok(jb):
                    _draw_capsule(img, pts[ja], pts[jb], r1 * u, r2 * u, near_fill, near_out)

            # Cricket ball in bowling hand or flying post-release
            ball_r = max(3, int(round(4.5 * u)))
            if br_frame_idx is not None and i > br_frame_idx and ball_origin_px is not None and ball_vel_px is not None:
                dt_f = float(i - br_frame_idx)
                bp = ball_origin_px + ball_vel_px * dt_f + np.array([0.0, 0.35 * (dt_f**2)])
                bx, by = int(round(bp[0])), int(round(bp[1]))
                if 0 <= bx < w_out and 0 <= by < h_out:
                    cv2.circle(img, (bx, by), ball_r, (42, 46, 158), -1, cv2.LINE_AA)
                    cv2.circle(img, (bx, by), ball_r, (23, 23, 23), 1, cv2.LINE_AA)
            elif _ok(b_han):
                bx, by = int(round(pts[b_han][0])), int(round(pts[b_han][1]))
                cv2.circle(img, (bx, by), ball_r, (42, 46, 158), -1, cv2.LINE_AA)
            elif _ok(b_wri):
                bx, by = int(round(pts[b_wri][0])), int(round(pts[b_wri][1]))
                cv2.circle(img, (bx, by), ball_r, (42, 46, 158), -1, cv2.LINE_AA)

        vw.write(img)
    vw.release()

    if shutil.which("ffmpeg"):
        encoders = (
            [
                ["-c:v", "h264_videotoolbox", "-q:v", "65", "-g", "1", "-bf", "0"],
                ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-g", "1", "-bf", "0"],
            ]
            if platform.system() == "Darwin"
            else [["-c:v", "libx264", "-preset", "ultrafast", "-crf", "22", "-g", "1", "-bf", "0"]]
        )
        for enc_args in encoders:
            res = subprocess.run(
                [
                    "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_raw),
                    *enc_args, "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out),
                ],
                capture_output=True,
                check=False,
            )
            if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
                tmp_raw.unlink(missing_ok=True)
                return out
    tmp_raw.replace(out)
    return out
