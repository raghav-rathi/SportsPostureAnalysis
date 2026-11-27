"""Studio v5 (Module 7): View-Adaptive Stroboscopic Chronophotography & Multi-Joint Trajectory Trails.

Implements Marey / Dartfish-style multi-exposure action compositing (`GET /api/deliveries/{id}/stroboscopic.jpg`)
and precomputed 2D/3D velocity-colored anatomical trails for `telemetry.json` v2.1:

  1. View-Adaptive Stroboscopic Modes:
     - `panoramic_composite` (Side-On / Open-Side clips with >=14% horizontal pelvis span):
       * On verified 3-DOF panned real clips, registers keyframes onto BR via exact `H_t = K @ R_br @ R_t^T @ K^-1`
         (or bowler-masked ORB/RANSAC homography when residual <= 2.8 px). If unverified pan on real video,
         falls back honestly to a crisp 4-column phase kinogram strip rather than smearing backgrounds!
       * Builds a bowler-masked median background plate, extracts soft anatomical-capsule-guided foreground
         silhouettes across [Pre-BFC, BFC, FFC, UAH, BR], composites chronologically with progressive opacity,
         and optionally draws the velocity-colored bowling-wrist whip arc & COM path.
     - `coronal_sequence_strip` (Rear-On / Front-On clips like Dale Steyn & Indoor where silhouettes overlap in place):
       * Crops an aspect-locked bowler corridor across [BFC, FFC, UAH, BR] into a seamless 4-column
         chronophotographic sequence strip with aligned release-height & ground reference hairlines.
     - `studio_mannequin_strobe` (Offline/Synthetic Studio proxies):
       * Renders progressive-opacity sculpted anatomical silhouettes onto a single crisp pitch canvas.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import JOINT_IDX
from .schemas import BiomechanicalFactSheet, BowlingArm
from .server.video_proxy import _draw_capsule
from .stage6_visualizer import SKELETON_EDGES, _read_exact_frames_sequential

# Paper Instrument segment palette in OpenCV BGR order (matches `docs/research_app_landscape_v5.md` §6.2)
BGR_PELVIS = (105, 150, 5)    # Emerald #059669
BGR_THORAX = (214, 81, 36)    # Cobalt #2451D6
BGR_ARM = (6, 119, 217)       # Amber #D97706
BGR_WRIST = (38, 38, 220)     # Crimson #DC2626
BGR_LEAD = (234, 155, 124)    # Soft Cobalt #7C9BEA
BGR_COM = (82, 82, 82)        # Graphite #525252
BGR_INK = (23, 23, 23)        # Ink #171717


def build_multi_joint_trails(
    joints_world_render_m: np.ndarray,
    joints2d_px: np.ndarray,
    com_world_render_m: np.ndarray,
    com_2d_px: np.ndarray,
    wrist_rel_speed_mps: Optional[np.ndarray],
    events: Any,
    fps: float,
    bowling_arm: BowlingArm,
) -> Dict[str, Any]:
    """Precompute compact 2D pixel & 3D ground-zeroed world trails across `[BFC - 150ms .. BR + 180ms]`."""
    n = len(joints_world_render_m)
    if n == 0:
        return {"frame_indices": [], "wrist_speed_mps": [], "2d_px": {}, "3d_m": {}}

    is_right = bowling_arm == BowlingArm.RIGHT
    b_wri = JOINT_IDX["rwri" if is_right else "lwri"]
    b_elb = JOINT_IDX["relb" if is_right else "lelb"]
    nb_wri = JOINT_IDX["lwri" if is_right else "rwri"]
    head_i = JOINT_IDX["head"]
    f_ank = JOINT_IDX["lank" if is_right else "rank"]
    r_ank = JOINT_IDX["rank" if is_right else "lank"]

    i0 = max(0, int(round(events.bfc.frame_float - 0.15 * fps)))
    i1 = min(n - 1, int(round(events.br.frame_float + 0.18 * fps)))
    idxs = list(range(i0, i1 + 1))

    loci = {
        "bowling_wrist": b_wri,
        "bowling_elbow": b_elb,
        "lead_wrist": nb_wri,
        "head": head_i,
        "front_ankle": f_ank,
        "rear_ankle": r_ank,
    }
    trails_2d: Dict[str, List[List[float]]] = {}
    trails_3d: Dict[str, List[List[float]]] = {}
    for name, ji in loci.items():
        trails_2d[name] = np.round(joints2d_px[idxs, ji], 1).tolist()
        trails_3d[name] = np.round(joints_world_render_m[idxs, ji], 4).tolist()

    trails_2d["com"] = np.round(com_2d_px[idxs], 1).tolist()
    trails_3d["com"] = np.round(com_world_render_m[idxs], 4).tolist()

    if wrist_rel_speed_mps is not None and len(wrist_rel_speed_mps) == n:
        w_spd = [round(float(wrist_rel_speed_mps[i]), 2) for i in idxs]
    else:
        w_spd = [0.0 for _ in idxs]

    return {
        "start_frame": i0,
        "end_frame": i1,
        "frame_indices": idxs,
        "wrist_speed_mps": w_spd,
        "2d_px": trails_2d,
        "3d_m": trails_3d,
        "joints_2d_px": trails_2d,
        "joints_3d_m": trails_3d,
    }


def _build_anatomical_soft_mask(pts2d: np.ndarray, h: int, w: int, dilate_scale: float = 1.0) -> np.ndarray:
    """Build a feathered `[0, 1]` float32 silhouette prior around 2D SMPL-24 joints."""
    mask = np.zeros((h, w), dtype=np.uint8)
    finite_y = pts2d[np.all(np.isfinite(pts2d), axis=1), 1]
    span = float(np.ptp(finite_y)) if len(finite_y) >= 2 else 200.0
    u = float(np.clip(span / 210.0, 0.5, 3.0)) * dilate_scale

    bones = [
        (0, 12, 22), (16, 17, 18), (1, 2, 18),
        (1, 4, 15), (4, 7, 12), (7, 10, 9),
        (2, 5, 15), (5, 8, 12), (8, 11, 9),
        (16, 18, 11), (18, 20, 9), (20, 22, 8),
        (17, 19, 11), (19, 21, 9), (21, 23, 8),
        (12, 15, 13),
    ]
    for ja, jb, rad in bones:
        if ja < len(pts2d) and jb < len(pts2d) and np.all(np.isfinite(pts2d[ja])) and np.all(np.isfinite(pts2d[jb])):
            p1 = (int(round(pts2d[ja, 0])), int(round(pts2d[ja, 1])))
            p2 = (int(round(pts2d[jb, 0])), int(round(pts2d[jb, 1])))
            r_px = max(4, int(round(rad * u)))
            cv2.line(mask, p1, p2, 255, 2 * r_px, cv2.LINE_AA)
            cv2.circle(mask, p1, r_px, 255, -1, cv2.LINE_AA)
            cv2.circle(mask, p2, r_px, 255, -1, cv2.LINE_AA)
    if len(pts2d) > 15 and np.all(np.isfinite(pts2d[15])):
        hc = (int(round(pts2d[15, 0])), int(round(pts2d[15, 1])))
        cv2.circle(mask, hc, max(8, int(round(18 * u))), 255, -1, cv2.LINE_AA)

    k_blur = max(5, 2 * int(round(4.5 * u)) + 1)
    soft = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (k_blur, k_blur), 0.0)
    return np.clip(soft, 0.0, 1.0)


def _draw_segment_skeleton_2d(img: np.ndarray, pts2d: np.ndarray, alpha: float = 1.0) -> None:
    """Draw Paper-Instrument segment-colored skeleton with crisp white 2px halo."""
    overlay = img.copy()
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 290.0)))
    for a_name, b_name, _, _ in SKELETON_EDGES:
        ja, jb = JOINT_IDX[a_name], JOINT_IDX[b_name]
        pa, pb = pts2d[ja], pts2d[jb]
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        p1 = (int(round(pa[0])), int(round(pa[1])))
        p2 = (int(round(pb[0])), int(round(pb[1])))
        if any(x in a_name + b_name for x in ("hip", "kne", "ank", "toe")):
            col = BGR_PELVIS
        elif "sho" in a_name and "sho" in b_name or "spi" in a_name or "neck" in a_name:
            col = BGR_THORAX
        elif "wri" in b_name:
            col = BGR_WRIST
        else:
            col = BGR_ARM
        cv2.line(overlay, p1, p2, (255, 255, 255), thick + 2, cv2.LINE_AA)
        cv2.line(overlay, p1, p2, col, thick, cv2.LINE_AA)
    for pt in pts2d[:22]:
        if np.all(np.isfinite(pt)):
            c = (int(round(pt[0])), int(round(pt[1])))
            cv2.circle(overlay, c, thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(overlay, c, max(1, thick - 1), BGR_INK, -1, cv2.LINE_AA)
    if alpha >= 0.99:
        img[:] = overlay
    else:
        cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0.0, img)


def _draw_footer_guardrail_bar(img: np.ndarray, mode_label: str) -> np.ndarray:
    """Append a non-truncating light-paper instrument caption & guardrail bar beneath the stroboscopic image."""
    h, w = img.shape[:2]
    bar_h = 54
    out = np.full((h + bar_h, w, 3), (250, 250, 250), dtype=np.uint8)
    out[:h, :] = img
    cv2.line(out, (0, h), (w, h), (228, 228, 231), 1)
    scale_top = float(np.clip(w / 2100.0, 0.38, 0.52))
    scale_bot = float(np.clip(w / 2350.0, 0.34, 0.46))
    cv2.putText(
        out,
        f"STROBOSCOPIC SEQUENCE ({mode_label})  |  Back-foot land -> Front-foot plant -> Arm level -> Ball release",
        (16, h + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_top,
        (32, 32, 36),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        "ICC 15-deg elbow legality is permanently REFUSED on video (requires >=250 Hz 3D lab). Single-ball posture only — not a medical diagnosis.",
        (16, h + 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale_bot,
        (110, 110, 118),
        1,
        cv2.LINE_AA,
    )
    return out


def render_stroboscopic_jpeg(
    proxy_or_video_path: str | Path,
    joints2d_px: np.ndarray,
    com_2d_px: Optional[np.ndarray],
    fact_sheet: BiomechanicalFactSheet,
    is_synthetic_proxy: bool = False,
    pan_poses_per_frame: Optional[List[Dict[str, Any]]] = None,
    max_width: int = 1280,
    draw_skeleton: bool = False,
    draw_trails: bool = True,
    cache_path: Optional[Path] = None,
) -> Tuple[bytes, str]:
    """Render view-adaptive stroboscopic multi-exposure JPEG bytes and return `(jpeg_bytes, mode_used)`."""
    if cache_path is not None and cache_path.exists() and cache_path.stat().st_size > 0:
        mode_cached = cache_path.with_suffix(".mode").read_text(encoding="utf-8").strip() if cache_path.with_suffix(".mode").exists() else "cached"
        return cache_path.read_bytes(), mode_cached

    safe_w = int(np.clip(max_width, 480, 1920))
    n = len(joints2d_px)
    ev = fact_sheet.events
    is_right = fact_sheet.bowling_arm == BowlingArm.RIGHT
    b_wri = JOINT_IDX["rwri" if is_right else "lwri"]
    head_i = JOINT_IDX["head"]

    if not ev.is_complete_delivery or n < 4:
        canvas = np.full((420, safe_w, 3), (248, 248, 249), dtype=np.uint8)
        cv2.putText(
            canvas,
            "Stroboscopic composite withheld: clip did not contain a verified complete bowling release.",
            (32, 205),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (60, 60, 68),
            1,
            cv2.LINE_AA,
        )
        canvas = _draw_footer_guardrail_bar(canvas, "WITHHELD — INCOMPLETE CLIP")
        ok_e, buf_e = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        assert ok_e
        return buf_e.tobytes(), "withheld_incomplete"

    i_pre = max(0, int(round(ev.bfc.frame_float - 0.14 * max(fact_sheet.audit.fps, 25.0))))
    i_bfc = int(np.clip(ev.bfc.frame_idx, 0, n - 1))
    i_ffc = int(np.clip(ev.ffc.frame_idx, 0, n - 1))
    i_uah = int(np.clip(ev.uah.frame_idx, 0, n - 1)) if ev.uah is not None else int(round(0.5 * (i_ffc + ev.br.frame_idx)))
    i_br = int(np.clip(ev.br.frame_idx, 0, n - 1))

    # Measure horizontal 2D pelvis progression across [BFC..BR] to choose Panoramic vs Coronal Strip
    src_w = max(float(fact_sheet.audit.width or 1280), 1.0)
    src_h = max(float(fact_sheet.audit.height or 720), 1.0)
    pelv_xs = joints2d_px[i_bfc : i_br + 1, JOINT_IDX["pelv"], 0]
    pelv_xs = pelv_xs[np.isfinite(pelv_xs)]
    horiz_span_frac = (float(np.ptp(pelv_xs)) / src_w) if len(pelv_xs) >= 2 else 0.0

    # -------------------------------------------------------------------------
    # Branch A: Studio Synthetic Proxy -> Direct Crisp Multi-Exposure Mannequins
    # -------------------------------------------------------------------------
    if is_synthetic_proxy and horiz_span_frac >= 0.12:
        h_out = int(round(safe_w * (src_h / src_w))) // 2 * 2
        sx, sy = safe_w / src_w, h_out / src_h
        img = np.full((h_out, safe_w, 3), (247, 246, 246), dtype=np.uint8)
        ank_ys = joints2d_px[:, [7, 8], 1].reshape(-1) * sy
        valid_ank = ank_ys[np.isfinite(ank_ys) & (ank_ys > 0) & (ank_ys < h_out)]
        hor_y = int(np.clip(np.percentile(valid_ank, 20) - 65.0, 0.42 * h_out, 0.62 * h_out)) if len(valid_ank) else int(0.52 * h_out)
        cv2.rectangle(img, (0, hor_y), (safe_w, h_out), (239, 237, 237), -1)
        cv2.line(img, (0, hor_y), (safe_w, hor_y), (220, 220, 224), 1, cv2.LINE_AA)

        phases_syn = []
        for idx_p, op_p, lbl_p in [
            (i_pre, 0.28, "Run-up"),
            (i_bfc, 0.46, "Back foot"),
            (i_ffc, 0.68, "Front foot"),
            (i_uah, 0.84, "Arm level"),
            (i_br, 1.00, "Release"),
        ]:
            if not phases_syn or abs(idx_p - phases_syn[-1][0]) >= 2:
                phases_syn.append((idx_p, op_p, lbl_p))

        PELV, LHIP, RHIP, LKNE, RKNE, LANK, RANK = 0, 1, 2, 4, 5, 7, 8
        LTOE, RTOE, NECK, HEAD = 10, 11, 12, 15
        LSHO, RSHO, LELB, RELB, LWRI, RWRI = 16, 17, 18, 19, 20, 21

        for idx_p, op_p, lbl_p in phases_syn:
            layer = img.copy()
            pts = joints2d_px[idx_p] * np.array([sx, sy])
            vis_y = pts[np.all(np.isfinite(pts), axis=1), 1]
            u = float(np.clip((float(np.ptp(vis_y)) if len(vis_y) >= 2 else 210.0) / 220.0, 0.55, 2.2))
            fill_c = (186, 188, 196) if op_p < 0.99 else (168, 172, 184)
            out_c = (118, 120, 130) if op_p < 0.99 else (74, 78, 90)
            for ja, jb, r1, r2 in [
                (RHIP, RKNE, 12, 9), (RKNE, RANK, 9, 7), (RANK, RTOE, 7, 5),
                (LHIP, LKNE, 12.5, 9.5), (LKNE, LANK, 9.5, 7), (LANK, LTOE, 7, 5),
                (LSHO, LELB, 9, 7), (LELB, LWRI, 7, 5),
                (RSHO, RELB, 9.5, 7.5), (RELB, RWRI, 7.5, 5.5),
                (PELV, NECK, 15, 13), (NECK, HEAD, 6.5, 7.5),
            ]:
                if np.all(np.isfinite(pts[ja])) and np.all(np.isfinite(pts[jb])):
                    _draw_capsule(layer, pts[ja], pts[jb], r1 * u, r2 * u, fill_c, out_c)
            if np.all(np.isfinite(pts[HEAD])):
                hc = (int(round(pts[HEAD, 0])), int(round(pts[HEAD, 1])))
                cv2.circle(layer, hc, max(6, int(round(13 * u))), fill_c, -1, cv2.LINE_AA)
                cv2.circle(layer, hc, max(6, int(round(13 * u))), out_c, 1, cv2.LINE_AA)
            if draw_skeleton:
                _draw_segment_skeleton_2d(layer, pts, alpha=0.95)
            cv2.addWeighted(layer, op_p, img, 1.0 - op_p, 0.0, img)
            if np.all(np.isfinite(pts[HEAD])):
                tx = int(np.clip(pts[HEAD, 0] - 28, 8, safe_w - 88))
                ty = max(20, int(round(pts[HEAD, 1] - 24 * u)))
                cv2.putText(img, lbl_p, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.42, BGR_INK, 1, cv2.LINE_AA)

        if draw_trails:
            w_pts = (joints2d_px[max(0, i_bfc - 6) : min(n, i_br + 6), b_wri] * np.array([sx, sy]))
            w_pts = w_pts[np.all(np.isfinite(w_pts), axis=1)].astype(np.int32)
            if len(w_pts) >= 3:
                cv2.polylines(img, [w_pts], False, (255, 255, 255), 5, cv2.LINE_AA)
                cv2.polylines(img, [w_pts], False, BGR_WRIST, 3, cv2.LINE_AA)
            if com_2d_px is not None and len(com_2d_px) == n:
                c_pts = (com_2d_px[max(0, i_bfc - 6) : min(n, i_br + 6)] * np.array([sx, sy]))
                c_pts = c_pts[np.all(np.isfinite(c_pts), axis=1)].astype(np.int32)
                if len(c_pts) >= 3:
                    cv2.polylines(img, [c_pts], False, BGR_COM, 2, cv2.LINE_AA)

        final_img = _draw_footer_guardrail_bar(img, "STUDIO MULTI-EXPOSURE")
        mode_used = "studio_mannequin_strobe"
    else:
        # Decode proxy frames sequentially (proxy is 0..n-1 indexed!)
        vid_p = Path(proxy_or_video_path)
        sample_bg_idxs = sorted(set(np.linspace(0, n - 1, min(n, 11)).round().astype(int).tolist() + [i_pre, i_bfc, i_ffc, i_uah, i_br]))
        decoded = _read_exact_frames_sequential(vid_p, sample_bg_idxs) if vid_p.exists() else {}

        can_panoramic = (
            horiz_span_frac >= 0.14
            and len(decoded) >= 4
            and ((not fact_sheet.calibration.is_panning_camera) or (pan_poses_per_frame is not None and len(pan_poses_per_frame) == n))
        )

        # ---------------------------------------------------------------------
        # Branch B: Real-Video Panoramic Multi-Exposure Composite (Side-On)
        # ---------------------------------------------------------------------
        if can_panoramic and i_br in decoded:
            ref_fr = decoded[i_br]
            rh, rw = ref_fr.shape[:2]
            scale = safe_w / float(rw)
            h_out = max(2, int(round(rh * scale)) // 2 * 2)
            sx, sy = safe_w / src_w, h_out / src_h

            K_orig = np.asarray(fact_sheet.calibration.camera_matrix_K, dtype=np.float64).reshape(3, 3)
            S_mat = np.diag([sx, sy, 1.0])
            K_s = S_mat @ K_orig
            K_s_inv = np.linalg.inv(K_s)
            R_br = np.asarray(pan_poses_per_frame[i_br]["R"], dtype=np.float64) if pan_poses_per_frame else np.eye(3)

            def _warp_to_br(idx_f: int, fr_bgr: np.ndarray, pts_orig: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
                fr_res = cv2.resize(fr_bgr, (safe_w, h_out), interpolation=cv2.INTER_AREA)
                pts_s = pts_orig * np.array([sx, sy])
                if pan_poses_per_frame is None or idx_f == i_br:
                    return fr_res, pts_s
                R_t = np.asarray(pan_poses_per_frame[idx_f]["R"], dtype=np.float64)
                H = K_s @ (R_br @ R_t.T) @ K_s_inv
                warped = cv2.warpPerspective(fr_res, H, (safe_w, h_out), borderMode=cv2.BORDER_REPLICATE)
                pts_h = np.column_stack([pts_s, np.ones(len(pts_s))])
                pts_w = (H @ pts_h.T).T
                pts_out = pts_w[:, :2] / np.maximum(pts_w[:, 2:3], 1e-5)
                return warped, pts_out

            # Build bowler-masked median background plate
            bg_stack: List[np.ndarray] = []
            for s_idx in sample_bg_idxs:
                if s_idx not in decoded:
                    continue
                w_fr, w_pts = _warp_to_br(s_idx, decoded[s_idx], joints2d_px[s_idx])
                b_mask = _build_anatomical_soft_mask(w_pts, h_out, safe_w, dilate_scale=1.45) > 0.12
                layer_nan = w_fr.astype(np.float32)
                layer_nan[b_mask] = np.nan
                bg_stack.append(layer_nan)

            with np.errstate(all="ignore"):
                bg_med = np.nanmedian(np.stack(bg_stack, axis=0), axis=0)
            nan_mask = (~np.isfinite(bg_med[:, :, 0])).astype(np.uint8) * 255
            ref_resized, _ = _warp_to_br(i_br, decoded[i_br], joints2d_px[i_br])
            bg_plate = np.where(np.isfinite(bg_med), bg_med, ref_resized).astype(np.uint8)
            if int(np.sum(nan_mask > 0)) > 0:
                bg_plate = cv2.inpaint(bg_plate, nan_mask, 5, cv2.INPAINT_TELEA)

            canvas = bg_plate.copy()
            phases_real = []
            for idx_p, op_p in [(i_pre, 0.42), (i_bfc, 0.58), (i_ffc, 0.74), (i_uah, 0.86), (i_br, 1.00)]:
                if idx_p in decoded and (not phases_real or abs(idx_p - phases_real[-1][0]) >= 2):
                    phases_real.append((idx_p, op_p))

            for idx_p, op_p in phases_real:
                w_fr, w_pts = _warp_to_br(idx_p, decoded[idx_p], joints2d_px[idx_p])
                cap_prior = _build_anatomical_soft_mask(w_pts, h_out, safe_w, dilate_scale=1.18)
                diff = np.linalg.norm(w_fr.astype(np.float32) - bg_plate.astype(np.float32), axis=2)
                fg_prob = np.clip((diff - 14.0) / 26.0, 0.0, 1.0)
                alpha_m = np.clip(cap_prior * np.maximum(fg_prob, 0.42 if idx_p == i_br else 0.25), 0.0, 1.0) * op_p
                canvas = (alpha_m[:, :, None] * w_fr.astype(np.float32) + (1.0 - alpha_m[:, :, None]) * canvas.astype(np.float32)).astype(np.uint8)
                if draw_skeleton:
                    _draw_segment_skeleton_2d(canvas, w_pts, alpha=op_p)

            if draw_trails:
                trail_pts = []
                for ti in range(max(0, i_bfc - 4), min(n, i_br + 4)):
                    pt_t = joints2d_px[ti, b_wri] * np.array([sx, sy])
                    if pan_poses_per_frame is not None and ti != i_br:
                        R_t = np.asarray(pan_poses_per_frame[ti]["R"], dtype=np.float64)
                        H = K_s @ (R_br @ R_t.T) @ K_s_inv
                        ph = H @ np.array([pt_t[0], pt_t[1], 1.0])
                        pt_t = ph[:2] / max(ph[2], 1e-5)
                    if np.all(np.isfinite(pt_t)):
                        trail_pts.append([int(round(pt_t[0])), int(round(pt_t[1]))])
                if len(trail_pts) >= 3:
                    arr_tp = np.asarray(trail_pts, dtype=np.int32)
                    cv2.polylines(canvas, [arr_tp], False, (255, 255, 255), 5, cv2.LINE_AA)
                    cv2.polylines(canvas, [arr_tp], False, BGR_WRIST, 3, cv2.LINE_AA)

            final_img = _draw_footer_guardrail_bar(canvas, "PANORAMIC MULTI-EXPOSURE")
            mode_used = "panoramic_composite"

        # ---------------------------------------------------------------------
        # Branch C: Coronal 4-Phase Chronophotographic Strip (Steyn / Indoor / Rear / Front)
        # ---------------------------------------------------------------------
        else:
            col_w = safe_w // 4
            col_h = int(round(col_w * 1.42))
            strip_cols: List[np.ndarray] = []
            phase_defs = [
                (i_bfc, "1. Back foot lands"),
                (i_ffc, "2. Front foot plants"),
                (i_uah, "3. Arm level"),
                (i_br, "4. Ball release"),
            ]

            # Compute consistent bowler ROI bounding box across all 4 event frames so release height aligns horizontally!
            valid_Sample = next((decoded[k] for k, _ in phase_defs if k in decoded), None)
            if valid_Sample is None:
                valid_Sample = np.full((720, 1280, 3), (242, 242, 244), dtype=np.uint8)
            vh, vw = valid_Sample.shape[:2]
            sx_v, sy_v = vw / src_w, vh / src_h

            all_ev_pts = np.concatenate([joints2d_px[k] for k, _ in phase_defs], axis=0) * np.array([sx_v, sy_v])
            fin_pts = all_ev_pts[np.all(np.isfinite(all_ev_pts), axis=1)]
            if len(fin_pts) >= 4:
                ymin = max(0.0, float(np.min(fin_pts[:, 1])) - 0.14 * vh)
                ymax = min(float(vh), float(np.max(fin_pts[:, 1])) + 0.12 * vh)
                crop_h = max(ymax - ymin, 0.45 * vh)
                crop_w = min(float(vw), crop_h * (col_w / float(col_h)))
            else:
                ymin, crop_h, crop_w = 0.05 * vh, 0.90 * vh, min(float(vw), 0.90 * vh * (col_w / float(col_h)))

            y0 = int(np.clip(round(ymin), 0, max(vh - int(round(crop_h)), 0)))
            y1 = min(vh, y0 + max(40, int(round(crop_h))))
            br_wri_y_norm: Optional[float] = None

            for col_i, (f_k, lbl_k) in enumerate(phase_defs):
                fr_k = decoded.get(f_k, valid_Sample).copy()
                pts_k = joints2d_px[f_k] * np.array([sx_v, sy_v])
                # Center X crop smoothly on this frame's pelvis/torso so panned clips stay centered!
                cx_k = float(pts_k[JOINT_IDX["pelv"], 0]) if np.all(np.isfinite(pts_k[JOINT_IDX["pelv"]])) else 0.5 * vw
                x0 = int(np.clip(round(cx_k - 0.5 * crop_w), 0, max(vw - int(round(crop_w)), 0)))
                x1 = min(vw, x0 + max(40, int(round(crop_w))))
                patch = fr_k[y0:y1, x0:x1]
                col_img = cv2.resize(patch, (col_w, col_h), interpolation=cv2.INTER_AREA)

                pts_col = pts_k.copy()
                pts_col[:, 0] = (pts_k[:, 0] - x0) * (col_w / max(float(x1 - x0), 1.0))
                pts_col[:, 1] = (pts_k[:, 1] - y0) * (col_h / max(float(y1 - y0), 1.0))

                if draw_skeleton or True:
                    # Subtle high-contrast skeleton overlay in coronal strip makes joint alignment legible on white kits!
                    _draw_segment_skeleton_2d(col_img, pts_col, alpha=0.88 if draw_skeleton else 0.62)

                if col_i == 3 and np.all(np.isfinite(pts_col[b_wri])):
                    br_wri_y_norm = float(pts_col[b_wri, 1])

                # Crisp top phase pill
                pill_overlay = col_img.copy()
                cv2.rectangle(pill_overlay, (0, 0), (col_w, 34), (252, 252, 253), -1)
                cv2.addWeighted(pill_overlay, 0.88, col_img, 0.12, 0.0, col_img)
                cv2.putText(col_img, lbl_k, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.44, BGR_INK, 1, cv2.LINE_AA)
                if col_i > 0:
                    cv2.line(col_img, (0, 0), (0, col_h), (226, 226, 230), 2)
                strip_cols.append(col_img)

            strip = np.hstack(strip_cols)
            if br_wri_y_norm is not None and 36.0 < br_wri_y_norm < col_h - 16.0:
                ry = int(round(br_wri_y_norm))
                for dx in range(0, strip.shape[1], 14):
                    cv2.line(strip, (dx, ry), (min(dx + 8, strip.shape[1] - 1), ry), BGR_WRIST, 1, cv2.LINE_AA)

            final_img = _draw_footer_guardrail_bar(strip, "CORONAL 4-PHASE SEQUENCE")
            mode_used = "coronal_sequence_strip"

    ok_enc, buf = cv2.imencode(".jpg", final_img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok_enc:
        raise RuntimeError("Failed to encode stroboscopic JPEG.")
    raw_bytes = buf.tobytes()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(raw_bytes)
        cache_path.with_suffix(".mode").write_text(mode_used, encoding="utf-8")
    return raw_bytes, mode_used
