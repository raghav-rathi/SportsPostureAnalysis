"""Phase 2 (M2B): 1-Frame MCC PnP + 3-DOF Rotational Background Pan Tracking.

Unlocks world-grounded run-up speed, stride length, and front-foot crease margin on tripod/handheld
panning cameras without grazing-angle 6-DOF planar homography blowups:
  - Anchors on a single-frame MCC PnP pose (R_0, t_0, K) at `anchor_frame`.
  - Tracks bowler-masked background KLT features forward & backward with FB-consistency gating.
  - Fits exact 3-DOF camera rotation dR_t in SO(3) (via calibrated bearing vectors + Kabsch RANSAC,
    guaranteeing det(dR_t) = +1 and fixed camera optical center C_0 = -R_0^T t_0).
  - Self-verifies via median reprojection residual `eps_pan_px <= 2.5 px`; automatically keeps
    ground translation suppressed if parallax, whip-pan blur, or weak texture breaches the gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class PanTrackingResult:
    """Per-frame 3-DOF rotational extrinsics anchored to 1-frame MCC PnP."""

    R_per_frame: np.ndarray  # (T, 3, 3) world-to-camera rotations
    t_per_frame: np.ndarray  # (T, 3) world-to-camera translations (fixed C_0 = -R_t.T @ t_t)
    anchor_frame: int
    median_residual_px: float
    p90_residual_px: float
    min_inlier_count: int
    mean_inlier_ratio: float
    max_pan_deg: float
    sigma_pan_deg: float
    is_verified: bool
    verification_reason: str
    per_frame_residual_px: List[float] = field(default_factory=list)


def _kabsch_rotation_3dof(u_src: np.ndarray, u_dst: np.ndarray) -> np.ndarray:
    """Solve exact SO(3) rotation dR mapping unit bearing vectors u_src -> u_dst (N, 3)."""
    H = u_src.T @ u_dst
    U, _, Vt = np.linalg.svd(H)
    dR = Vt.T @ U.T
    if np.linalg.det(dR) < 0.0:
        Vt[-1, :] *= -1.0
        dR = Vt.T @ U.T
    return dR


def _fit_rotation_ransac(
    pts_prev: np.ndarray,
    pts_curr: np.ndarray,
    K: np.ndarray,
    reproj_thresh_px: float = 2.0,
    n_iters: int = 80,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """RANSAC pure 3-DOF rotation dR in SO(3) minimizing pixel reprojection error K dR K^-1 p_prev ~ p_curr."""
    n = len(pts_prev)
    if n < 3:
        return np.eye(3, dtype=np.float64), np.zeros(n, dtype=bool), 99.0

    K_inv = np.linalg.inv(K)
    hom_prev = np.column_stack([pts_prev, np.ones(n, dtype=np.float64)])
    hom_curr = np.column_stack([pts_curr, np.ones(n, dtype=np.float64)])
    u_prev = (K_inv @ hom_prev.T).T
    u_prev /= np.maximum(np.linalg.norm(u_prev, axis=1, keepdims=True), 1e-9)
    u_curr = (K_inv @ hom_curr.T).T
    u_curr /= np.maximum(np.linalg.norm(u_curr, axis=1, keepdims=True), 1e-9)

    def _reproj_errs(dR: np.ndarray) -> np.ndarray:
        proj = (K @ (dR @ u_prev.T)).T
        z = np.maximum(proj[:, 2:3], 1e-6)
        uv = proj[:, :2] / z
        return np.linalg.norm(uv - pts_curr, axis=1)

    rng = np.random.default_rng(seed)
    best_dR = np.eye(3, dtype=np.float64)
    best_inliers = np.zeros(n, dtype=bool)
    best_count = -1
    best_med = 999.0

    # First evaluate identity (static frame shortcut)
    err_id = _reproj_errs(best_dR)
    inl_id = err_id <= reproj_thresh_px
    best_inliers = inl_id
    best_count = int(inl_id.sum())
    best_med = float(np.median(err_id[inl_id])) if best_count > 0 else float(np.median(err_id))

    for _ in range(n_iters):
        idx = rng.choice(n, size=3, replace=False)
        cand_R = _kabsch_rotation_3dof(u_prev[idx], u_curr[idx])
        errs = _reproj_errs(cand_R)
        inl = errs <= reproj_thresh_px
        cnt = int(inl.sum())
        if cnt > best_count or (cnt == best_count and cnt > 0 and float(np.median(errs[inl])) < best_med):
            best_count = cnt
            best_inliers = inl
            best_dR = cand_R
            best_med = float(np.median(errs[inl])) if cnt > 0 else float(np.median(errs))

    if best_count >= 3:
        best_dR = _kabsch_rotation_3dof(u_prev[best_inliers], u_curr[best_inliers])
        final_errs = _reproj_errs(best_dR)
        best_inliers = final_errs <= reproj_thresh_px
        med_res = float(np.median(final_errs[best_inliers])) if best_inliers.any() else float(np.median(final_errs))
    else:
        med_res = float(np.median(_reproj_errs(best_dR)))

    return best_dR, best_inliers, med_res


def _build_background_mask(
    h: int,
    w: int,
    boxes_xyxy: Optional[List[Tuple[float, float, float, float]]] = None,
    pad_frac: float = 0.14,
) -> np.ndarray:
    """Create uint8 mask (255 = valid background, 0 = excluded people/overlays/borders)."""
    mask = np.full((h, w), 255, dtype=np.uint8)
    # Exclude thin top/bottom letterbox & broadcast scorebug bands (6% top, 8% bottom)
    mask[: max(4, int(0.06 * h)), :] = 0
    mask[min(h - 4, int(0.92 * h)) :, :] = 0
    if boxes_xyxy:
        for b in boxes_xyxy:
            if b is None or len(b) < 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in b[:4]]
            if not all(math.isfinite(v) for v in (x0, y0, x1, y1)) or x1 <= x0 or y1 <= y0:
                continue
            px = pad_frac * (x1 - x0)
            py = pad_frac * (y1 - y0)
            ix0 = max(0, int(math.floor(x0 - px)))
            iy0 = max(0, int(math.floor(y0 - py)))
            ix1 = min(w, int(math.ceil(x1 + px)))
            iy1 = min(h, int(math.ceil(y1 + py)))
            mask[iy0:iy1, ix0:ix1] = 0
    return mask


def track_camera_rotation_3dof_frames(
    gray_frames: List[np.ndarray],
    K: np.ndarray,
    R_anchor: np.ndarray,
    t_anchor: np.ndarray,
    anchor_frame: int = 0,
    per_frame_exclude_boxes: Optional[List[List[Tuple[float, float, float, float]]]] = None,
    residual_gate_px: float = 2.5,
) -> PanTrackingResult:
    """Track per-frame 3-DOF camera rotation (R_t, t_t) forward & backward from `anchor_frame`.

    Preserves fixed optical center `C_0 = -R_anchor.T @ t_anchor` across all frames so
    `t_t = dR_t @ t_anchor` and `-R_t.T @ t_t == C_0` identically.
    """
    n = len(gray_frames)
    if n == 0:
        raise ValueError("Cannot run 3-DOF pan tracking on empty frame sequence.")

    h, w = gray_frames[0].shape[:2]
    K64 = np.asarray(K, dtype=np.float64).reshape(3, 3)
    R0 = np.asarray(R_anchor, dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(t_anchor, dtype=np.float64).reshape(3)
    anc = int(np.clip(anchor_frame, 0, n - 1))

    R_all = np.zeros((n, 3, 3), dtype=np.float64)
    t_all = np.zeros((n, 3), dtype=np.float64)
    R_all[anc] = R0
    t_all[anc] = t0

    res_list = np.zeros(n, dtype=np.float64)
    inlier_counts: List[int] = []
    inlier_ratios: List[float] = []

    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    def _step_pair(i_from: int, i_to: int) -> Tuple[np.ndarray, float, int, float]:
        g0 = gray_frames[i_from]
        g1 = gray_frames[i_to]
        boxes = per_frame_exclude_boxes[i_from] if (per_frame_exclude_boxes and i_from < len(per_frame_exclude_boxes)) else None
        mask = _build_background_mask(h, w, boxes)
        pts0 = cv2.goodFeaturesToTrack(g0, maxCorners=260, qualityLevel=0.015, minDistance=12, mask=mask)
        if pts0 is None or len(pts0) < 10:
            return np.eye(3, dtype=np.float64), 9.9, 0, 0.0

        pts1, st1, _ = cv2.calcOpticalFlowPyrLK(g0, g1, pts0, None, **lk_params)
        pts0_back, st2, _ = cv2.calcOpticalFlowPyrLK(g1, g0, pts1, None, **lk_params)
        if pts1 is None or pts0_back is None:
            return np.eye(3, dtype=np.float64), 9.9, 0, 0.0

        fb_err = np.linalg.norm((pts0 - pts0_back).reshape(-1, 2), axis=1)
        ok = (st1.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb_err <= 1.5)

        # Also filter points landing inside excluded boxes on destination frame
        p0 = pts0.reshape(-1, 2)[ok]
        p1 = pts1.reshape(-1, 2)[ok]
        if per_frame_exclude_boxes and i_to < len(per_frame_exclude_boxes):
            mask1 = _build_background_mask(h, w, per_frame_exclude_boxes[i_to])
            ix = np.clip(np.round(p1[:, 0]).astype(int), 0, w - 1)
            iy = np.clip(np.round(p1[:, 1]).astype(int), 0, h - 1)
            keep_bg = mask1[iy, ix] > 0
            p0 = p0[keep_bg]
            p1 = p1[keep_bg]

        if len(p0) < 8:
            return np.eye(3, dtype=np.float64), 9.9, int(len(p0)), 0.0

        dR, inliers, med_res = _fit_rotation_ransac(p0, p1, K64, reproj_thresh_px=2.2, seed=42 + i_from + i_to)
        inl_cnt = int(inliers.sum())
        inl_ratio = float(inl_cnt) / max(len(p0), 1)
        return dR, med_res, inl_cnt, inl_ratio

    # Forward pass: anc -> n - 1
    for i in range(anc, n - 1):
        dR, med_res, cnt, ratio = _step_pair(i, i + 1)
        R_all[i + 1] = dR @ R_all[i]
        t_all[i + 1] = dR @ t_all[i]
        res_list[i + 1] = med_res
        inlier_counts.append(cnt)
        inlier_ratios.append(ratio)

    # Backward pass: anc -> 0
    for i in range(anc, 0, -1):
        dR, med_res, cnt, ratio = _step_pair(i, i - 1)
        R_all[i - 1] = dR @ R_all[i]
        t_all[i - 1] = dR @ t_all[i]
        res_list[i - 1] = med_res
        inlier_counts.append(cnt)
        inlier_ratios.append(ratio)

    # Compute total rotational excursion from anchor & self-verification metrics
    pan_angles = []
    for i in range(n):
        dR_tot = R_all[i] @ R0.T
        tr = float(np.clip(0.5 * (np.trace(dR_tot) - 1.0), -1.0, 1.0))
        pan_angles.append(math.degrees(math.acos(tr)))
    max_pan = float(max(pan_angles)) if pan_angles else 0.0

    step_res = res_list[np.arange(n) != anc] if n > 1 else np.array([0.0])
    med_res = float(np.median(step_res))
    p90_res = float(np.percentile(step_res, 90))
    min_inl = int(min(inlier_counts)) if inlier_counts else 0
    mean_ratio = float(np.mean(inlier_ratios)) if inlier_ratios else 1.0

    # Convert pixel residual + cumulative random-walk drift sqrt(max_excursion_steps) into angular uncertainty (deg)
    f_px = float(max(K64[0, 0], K64[1, 1], 100.0))
    max_excursion_steps = max(anc, n - 1 - anc, 1)
    sigma_pan_deg = math.degrees(med_res / f_px) * math.sqrt(float(max_excursion_steps))

    is_verified = (med_res <= residual_gate_px) and (p90_res <= 1.8 * residual_gate_px) and (min_inl >= 8) and (mean_ratio >= 0.45)
    if is_verified:
        reason = (
            f"Verified 3-DOF rotational pan track (median residual={med_res:.2f} px <= {residual_gate_px:.1f} px, "
            f"min_inliers={min_inl}, max_pan={max_pan:.1f} deg, sigma_pan={sigma_pan_deg:.2f} deg)."
        )
    else:
        reason = (
            f"Rejected 3-DOF pan unlock (median residual={med_res:.2f} px, p90={p90_res:.2f} px, "
            f"min_inliers={min_inl}, mean_inlier_ratio={mean_ratio:.2f}); keeping world translation suppressed."
        )

    return PanTrackingResult(
        R_per_frame=R_all,
        t_per_frame=t_all,
        anchor_frame=anc,
        median_residual_px=round(med_res, 3),
        p90_residual_px=round(p90_res, 3),
        min_inlier_count=min_inl,
        mean_inlier_ratio=round(mean_ratio, 3),
        max_pan_deg=round(max_pan, 2),
        sigma_pan_deg=round(sigma_pan_deg, 3),
        is_verified=bool(is_verified),
        verification_reason=reason,
        per_frame_residual_px=[round(float(r), 3) for r in res_list],
    )


def track_camera_rotation_3dof_video(
    video_path: str | Path,
    source_frame_indices: List[int],
    K: np.ndarray,
    R_anchor: np.ndarray,
    t_anchor: np.ndarray,
    anchor_frame: int = 0,
    bowler_boxes_xyxy: Optional[np.ndarray] = None,
    residual_gate_px: float = 2.5,
) -> PanTrackingResult:
    """Decode `source_frame_indices` sequentially and run 3-DOF rotational background pan tracking."""
    if not source_frame_indices:
        raise ValueError("source_frame_indices must be non-empty.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for 3-DOF pan tracking: {video_path}")

    needed_set = set(int(i) for i in source_frame_indices)
    max_f = max(needed_set)
    min_f = min(needed_set)
    if min_f > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, min_f)

    decoded: Dict[int, np.ndarray] = {}
    for f_idx in range(min_f, max_f + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if f_idx in needed_set:
            decoded[f_idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cap.release()

    gray_frames = [decoded[int(i)] for i in source_frame_indices if int(i) in decoded]
    if len(gray_frames) != len(source_frame_indices):
        raise RuntimeError(f"Decoded {len(gray_frames)}/{len(source_frame_indices)} frames for 3-DOF pan tracking.")

    exclude_boxes: Optional[List[List[Tuple[float, float, float, float]]]] = None
    if bowler_boxes_xyxy is not None and len(bowler_boxes_xyxy) == len(gray_frames):
        exclude_boxes = [[tuple(float(x) for x in bowler_boxes_xyxy[i][:4])] for i in range(len(gray_frames))]

    return track_camera_rotation_3dof_frames(
        gray_frames=gray_frames,
        K=K,
        R_anchor=R_anchor,
        t_anchor=t_anchor,
        anchor_frame=anchor_frame,
        per_frame_exclude_boxes=exclude_boxes,
        residual_gate_px=residual_gate_px,
    )


def transform_cam_to_pitch_world_per_frame(
    joints_cam_m: np.ndarray,
    R_per_frame: np.ndarray,
    t_per_frame: np.ndarray,
) -> np.ndarray:
    """Transform `(T, J, 3)` camera-frame 3D joints into Pitch World Frame using per-frame `(R_t, t_t)`."""
    n = joints_cam_m.shape[0]
    out = np.empty_like(joints_cam_m, dtype=np.float64)
    for i in range(n):
        # X_world = R_t^T @ (X_cam - t_t)
        shifted = joints_cam_m[i] - t_per_frame[i][None, :]
        out[i] = shifted @ R_per_frame[i]
    return out
