"""Stage 2/3 Frontend: Frontier 3D Human Pose & Uncertainty Extraction via NLF (NeurIPS 2024).

Uses István Sárándi's Neural Localizer Fields (`nlf_l_multi.torchscript`) on CUDA GPU
to extract metric 3D SMPL-24 joints, 2D pixel projections, in-frame visibility masks,
native per-joint 3D uncertainties, and upright standing background gravity vectors
(e.g. umpires/non-strikers, strictly excluding squatting wicketkeepers and crouched batters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import BowlingArm
from .synthetic_ground_truth import JOINT_IDX

_NLF_MODEL_CACHE: Dict[Tuple[str, str], object] = {}


def _load_nlf_cached(model_path: str, device: str):
    import torch
    import torchvision  # Required before torch.jit.load to register torchvision::nms

    _ = torchvision.__version__
    key = (str(model_path), str(device))
    if key not in _NLF_MODEL_CACHE:
        _NLF_MODEL_CACHE[key] = torch.jit.load(str(model_path), map_location=device).eval()
    return _NLF_MODEL_CACHE[key]


@dataclass
class TrackedBowler3D:
    joints3d_cam_m: np.ndarray        # (T, 24, 3) meters in OpenCV camera coordinates
    joints2d_px: np.ndarray           # (T, 24, 2) pixels
    joint_uncertainties_m: np.ndarray # (T, 24) meters
    joint_in_frame: np.ndarray        # (T, 24) bool strictly inside image bounds + detected frame
    valid_mask: np.ndarray            # (T,) bool frames where bowler was detected directly
    bowler_boxes_xyxy: np.ndarray     # (T, 4) bounding boxes for background pan masking
    detected_bowling_arm: BowlingArm
    track_score: float
    num_tracks_seen: int
    background_upright_vectors: List[np.ndarray] = field(default_factory=list)
    backend_name: str = "nlf_l_multi_smpl24"


def _bbox_from_2d(pts2d: np.ndarray) -> np.ndarray:
    x0, y0 = np.min(pts2d, axis=0)
    x1, y1 = np.max(pts2d, axis=0)
    return np.array([x0, y0, x1, y1], dtype=np.float64)


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ix0 = max(box_a[0], box_b[0])
    iy0 = max(box_a[1], box_b[1])
    ix1 = min(box_a[2], box_b[2])
    iy1 = min(box_a[3], box_b[3])
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return float(inter / max(area_a + area_b - inter, 1e-6))


def _angle_3d_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    denom = max(float(np.linalg.norm(v1) * np.linalg.norm(v2)), 1e-8)
    return float(np.degrees(np.arccos(np.clip(float(np.dot(v1, v2)) / denom, -1.0, 1.0))))


def _associate_tracks(
    per_frame_dets: List[List[Dict[str, np.ndarray]]],
    img_diag: float,
    max_gap: int = 8,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    """3D-depth-aware IoU + pelvis-distance multi-person tracklet builder."""
    tracks: List[Dict[int, Dict[str, np.ndarray]]] = []
    last_seen: List[int] = []
    last_det: List[Dict[str, np.ndarray]] = []

    for f_idx, dets in enumerate(per_frame_dets):
        if not dets:
            continue
        active_tids = [tid for tid, ls in enumerate(last_seen) if f_idx - ls <= max_gap]
        used_dets = set()

        if active_tids:
            pairs = []
            for tid in active_tids:
                prev = last_det[tid]
                prev_box = _bbox_from_2d(prev["joints2d"])
                prev_pelv2d = prev["joints2d"][JOINT_IDX["pelv"]]
                prev_z = float(prev["joints3d"][JOINT_IDX["pelv"], 2])
                gap = max(1, f_idx - last_seen[tid])
                for d_idx, det in enumerate(dets):
                    cur_box = _bbox_from_2d(det["joints2d"])
                    cur_pelv2d = det["joints2d"][JOINT_IDX["pelv"]]
                    cur_z = float(det["joints3d"][JOINT_IDX["pelv"], 2])
                    box_iou = _iou(prev_box, cur_box)
                    pelv_dist = float(np.linalg.norm(cur_pelv2d - prev_pelv2d)) / max(img_diag, 1.0)
                    dz = abs(cur_z - prev_z)
                    # Reject depth swaps (e.g. bowler passing near umpire at different depth)
                    if dz > 2.5 * gap:
                        continue
                    if box_iou > 0.08 or pelv_dist < 0.14 * (1.0 + 0.25 * gap):
                        depth_pen = min(dz / 4.0, 0.8)
                        score = box_iou + max(0.0, 1.0 - pelv_dist * 5.0) - 0.45 * depth_pen
                        pairs.append((score, tid, d_idx))
            pairs.sort(key=lambda x: x[0], reverse=True)
            matched_tids = set()
            for _, tid, d_idx in pairs:
                if tid in matched_tids or d_idx in used_dets:
                    continue
                tracks[tid][f_idx] = dets[d_idx]
                last_seen[tid] = f_idx
                last_det[tid] = dets[d_idx]
                matched_tids.add(tid)
                used_dets.add(d_idx)

        for d_idx, det in enumerate(dets):
            if d_idx not in used_dets:
                tracks.append({f_idx: det})
                last_seen.append(f_idx)
                last_det.append(det)

    return tracks


def _select_bowler_track(
    tracks: List[Dict[int, Dict[str, np.ndarray]]],
    n_frames: int,
    img_w: int,
    img_h: int,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
) -> Tuple[int, float]:
    """Select the active fast bowler among multiple people (umpire, batter, non-striker)."""
    if not tracks:
        raise RuntimeError("No human poses detected in video clip.")

    diag = float(np.hypot(img_w, img_h))
    best_tid = 0
    best_score = -1e9

    for tid, frame_map in enumerate(tracks):
        f_indices = sorted(frame_map.keys())
        if len(f_indices) < max(4, int(0.15 * n_frames)):
            continue
        j2d = np.stack([frame_map[f]["joints2d"] for f in f_indices], axis=0)
        j3d = np.stack([frame_map[f]["joints3d"] for f in f_indices], axis=0)

        coverage = len(f_indices) / max(n_frames, 1)
        heights = np.max(j2d[:, :, 1], axis=1) - np.min(j2d[:, :, 1], axis=1)
        rel_height = float(np.median(heights)) / max(img_h, 1)

        pelv_2d = j2d[:, JOINT_IDX["pelv"]]
        pelv_path_2d = float(np.sum(np.linalg.norm(np.diff(pelv_2d, axis=0), axis=1))) / max(diag, 1.0)
        pelv_3d = j3d[:, JOINT_IDX["pelv"]]
        pelv_disp_3d = float(np.linalg.norm(pelv_3d[-1] - pelv_3d[0]))

        rwri_y = -(j2d[:, JOINT_IDX["rwri"], 1] - j2d[:, JOINT_IDX["rsho"], 1]) / np.maximum(heights, 10.0)
        lwri_y = -(j2d[:, JOINT_IDX["lwri"], 1] - j2d[:, JOINT_IDX["lsho"], 1]) / np.maximum(heights, 10.0)
        peak_overhead = float(max(np.max(rwri_y), np.max(lwri_y)))
        rwri_travel = float(np.sum(np.linalg.norm(np.diff(j2d[:, JOINT_IDX["rwri"]], axis=0), axis=1))) / max(diag, 1.0)
        lwri_travel = float(np.sum(np.linalg.norm(np.diff(j2d[:, JOINT_IDX["lwri"]], axis=0), axis=1))) / max(diag, 1.0)
        arm_activity = max(rwri_travel, lwri_travel)

        score = (
            1.8 * coverage
            + 2.2 * min(rel_height * 2.5, 1.5)
            + 2.5 * min(pelv_path_2d * 2.0, 2.0)
            + 1.0 * min(pelv_disp_3d / 2.0, 2.0)
            + 3.0 * max(0.0, peak_overhead)
            + 2.5 * min(arm_activity * 1.5, 2.5)
        )

        if anchor_xy_norm is not None:
            med_pelv_norm = np.median(pelv_2d, axis=0) / np.array([img_w, img_h])
            dist_anchor = float(np.linalg.norm(med_pelv_norm - np.array(anchor_xy_norm)))
            score += 4.0 * max(0.0, 1.0 - dist_anchor * 2.5)

        if score > best_score:
            best_score = score
            best_tid = tid

    if best_score < -1e8:
        best_tid = max(range(len(tracks)), key=lambda i: len(tracks[i]))
        best_score = float(len(tracks[best_tid]))

    return best_tid, float(best_score)


def _extract_standing_background_gravity(
    tracks: List[Dict[int, Dict[str, np.ndarray]]],
    bowler_tid: int,
    img_w: int,
    img_h: int,
) -> List[np.ndarray]:
    """Collect torso vertical vectors ONLY from genuinely standing stationary background people
    (e.g. umpires, standing coaches) while strictly rejecting squatting wicketkeepers & crouched batters."""
    upright_vecs: List[np.ndarray] = []
    margin = 12.0
    for tid, frame_map in enumerate(tracks):
        if tid == bowler_tid or len(frame_map) < 5:
            continue
        f_idxs = sorted(frame_map.keys())
        pelvs = np.stack([frame_map[f]["joints3d"][JOINT_IDX["pelv"]] for f in f_idxs], axis=0)
        # Must be relatively stationary (< 1.0 m travel)
        if float(np.linalg.norm(pelvs[-1] - pelvs[0])) > 1.0:
            continue
        for f in f_idxs[:: max(1, len(f_idxs) // 10)]:
            j3 = frame_map[f]["joints3d"]
            j2 = frame_map[f]["joints2d"]
            # Require core standing joints inside frame
            core_idx = [JOINT_IDX["pelv"], JOINT_IDX["neck"], JOINT_IDX["lkne"], JOINT_IDX["rkne"], JOINT_IDX["lank"], JOINT_IDX["rank"]]
            pts_c = j2[core_idx]
            if not np.all(
                (pts_c[:, 0] >= margin) & (pts_c[:, 0] <= img_w - margin)
                & (pts_c[:, 1] >= margin) & (pts_c[:, 1] <= img_h - margin)
            ):
                continue
            # Strictly verify standing posture: both knees > 155 deg AND hip-trunk > 155 deg!
            l_knee = _angle_3d_deg(j3[JOINT_IDX["lhip"]], j3[JOINT_IDX["lkne"]], j3[JOINT_IDX["lank"]])
            r_knee = _angle_3d_deg(j3[JOINT_IDX["rhip"]], j3[JOINT_IDX["rkne"]], j3[JOINT_IDX["rank"]])
            l_hip = _angle_3d_deg(j3[JOINT_IDX["neck"]], j3[JOINT_IDX["lhip"]], j3[JOINT_IDX["lkne"]])
            r_hip = _angle_3d_deg(j3[JOINT_IDX["neck"]], j3[JOINT_IDX["rhip"]], j3[JOINT_IDX["rkne"]])
            if l_knee > 155.0 and r_knee > 155.0 and l_hip > 155.0 and r_hip > 155.0:
                mid_ank = 0.5 * (j3[JOINT_IDX["lank"]] + j3[JOINT_IDX["rank"]])
                vec = j3[JOINT_IDX["head"]] - mid_ank
                nrm = float(np.linalg.norm(vec))
                if nrm > 1.0:
                    upright_vecs.append(vec / nrm)
    return upright_vecs


def _infer_bowling_arm(joints3d_cam_m: np.ndarray) -> BowlingArm:
    """Detect whether right or left arm is the bowling arm via peak straight overhead reach & shoulder-relative speed."""
    # Gravity-up approx is -Y_cam
    def _arm_score(sho_k: str, elb_k: str, wri_k: str) -> float:
        sho = joints3d_cam_m[:, JOINT_IDX[sho_k]]
        elb = joints3d_cam_m[:, JOINT_IDX[elb_k]]
        wri = joints3d_cam_m[:, JOINT_IDX[wri_k]]
        arm_vec = wri - sho
        above = np.maximum(-arm_vec[:, 1], 0.0)
        arm_len = np.linalg.norm(arm_vec, axis=1)
        # Straightness at elbow (bowling arm is straight overhead; front arm often bends during gather/pull)
        v1 = sho - elb
        v2 = wri - elb
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        cos_e = np.clip(np.sum(v1 * v2, axis=1) / np.maximum(n1 * n2, 1e-6), -1.0, 1.0)
        straight_w = np.clip((np.degrees(np.arccos(cos_e)) - 90.0) / 90.0, 0.2, 1.0)
        # Mean of top-2 frames so 25 fps 1-2 frame ball release isn't skipped by a 97th percentile!
        reach_series = above * arm_len * straight_w
        k_top = min(2, len(reach_series))
        peak_reach = float(np.mean(np.sort(reach_series)[-k_top:]))
        # Peak shoulder-relative wrist speed isolates bowling-arm circumduction from body run-up
        rel_vel = np.linalg.norm(np.diff(arm_vec, axis=0), axis=1)
        k_vel = min(3, len(rel_vel))
        peak_rel_vel = float(np.mean(np.sort(rel_vel)[-k_vel:])) if k_vel > 0 else 0.0
        # Full vertical circumduction range (bottom of downswing to top of release)
        vert_range = float(np.max(-arm_vec[:, 1]) - np.min(-arm_vec[:, 1]))
        return peak_reach * 4.0 + peak_rel_vel * 5.0 + vert_range * 1.5

    r_score = _arm_score("rsho", "relb", "rwri")
    l_score = _arm_score("lsho", "lelb", "lwri")
    return BowlingArm.RIGHT if r_score >= l_score else BowlingArm.LEFT


def _interpolate_missing_frames(arr: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate missing frames along axis 0, holding boundary values."""
    out = arr.copy()
    n = arr.shape[0]
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0 or len(valid_idx) == n:
        return out
    flat = out.reshape(n, -1)
    x_all = np.arange(n, dtype=np.float64)
    for col in range(flat.shape[1]):
        flat[:, col] = np.interp(x_all, valid_idx.astype(np.float64), flat[valid_idx, col])
    return flat.reshape(arr.shape)


def extract_bowler_3d_nlf(
    video_path: str | Path,
    model_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    batch_size: int = 16,
    num_aug: int = 1,
    fov_deg: float = 55.0,
    in_frame_margin_px: float = 12.0,
    bowling_arm_override: Optional[BowlingArm] = None,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    device: str = "cuda",
) -> TrackedBowler3D:
    """Run NLF (`nlf_l_multi.torchscript`) to extract metric 3D bowler joints, visibility, & uncertainties."""
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ef = min(total_frames, end_frame) if end_frame is not None else total_frames
    sf = max(0, start_frame)

    frames_rgb: List[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, sf)
    for _ in range(sf, ef):
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            break
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()

    if not frames_rgb:
        raise RuntimeError(f"No frames extracted from {video_path} [{sf}:{ef}].")

    model = _load_nlf_cached(str(model_path), device)
    per_frame_dets: List[List[Dict[str, np.ndarray]]] = []
    n_frames = len(frames_rgb)

    with torch.inference_mode():
        for b0 in range(0, n_frames, batch_size):
            batch_np = np.stack(frames_rgb[b0 : b0 + batch_size], axis=0)
            batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(device)
            out = model.detect_smpl_batched(
                batch_t,
                default_fov_degrees=float(fov_deg),
                num_aug=int(num_aug),
                detector_threshold=0.30,
                max_detections=12,
                suppress_implausible_poses=True,
            )
            j3d_list = out["joints3d"]
            j2d_list = out["joints2d"]
            unc_list = out["joint_uncertainties"]

            for i in range(len(j3d_list)):
                dets_i: List[Dict[str, np.ndarray]] = []
                n_det = int(j3d_list[i].shape[0])
                if n_det > 0:
                    j3d_np = j3d_list[i].detach().cpu().numpy().astype(np.float64)
                    j2d_np = j2d_list[i].detach().cpu().numpy().astype(np.float64)
                    unc_np = unc_list[i].detach().cpu().numpy().astype(np.float64)
                    # Infer mm vs meters reliably from anatomical torso length ||neck - pelv|| (~0.50 m vs ~500 mm)
                    torso_lens = np.linalg.norm(
                        j3d_np[:, JOINT_IDX["neck"]] - j3d_np[:, JOINT_IDX["pelv"]], axis=-1
                    )
                    if float(np.median(torso_lens)) > 2.0:
                        j3d_np = j3d_np / 1000.0
                    if float(np.median(unc_np)) > 2.0:
                        unc_np = unc_np / 1000.0
                    for p in range(n_det):
                        dets_i.append(
                            {
                                "joints3d": j3d_np[p],
                                "joints2d": j2d_np[p],
                                "unc": unc_np[p],
                            }
                        )
                per_frame_dets.append(dets_i)

    tracks = _associate_tracks(per_frame_dets, img_diag=float(np.hypot(img_w, img_h)))
    best_tid, best_score = _select_bowler_track(
        tracks, n_frames=n_frames, img_w=img_w, img_h=img_h, anchor_xy_norm=anchor_xy_norm
    )
    chosen = tracks[best_tid]
    bg_upright = _extract_standing_background_gravity(tracks, best_tid, img_w, img_h)

    j3d_full = np.zeros((n_frames, 24, 3), dtype=np.float64)
    j2d_full = np.zeros((n_frames, 24, 2), dtype=np.float64)
    unc_full = np.full((n_frames, 24), 0.12, dtype=np.float64)
    in_frame = np.zeros((n_frames, 24), dtype=bool)
    valid_mask = np.zeros(n_frames, dtype=bool)
    boxes = np.zeros((n_frames, 4), dtype=np.float64)

    for f_idx, det in chosen.items():
        j3d_full[f_idx] = det["joints3d"]
        j2d_full[f_idx] = det["joints2d"]
        unc_full[f_idx] = det["unc"]
        valid_mask[f_idx] = True
        boxes[f_idx] = _bbox_from_2d(det["joints2d"])
        xs = det["joints2d"][:, 0]
        ys = det["joints2d"][:, 1]
        in_frame[f_idx] = (
            np.isfinite(xs)
            & np.isfinite(ys)
            & (xs >= in_frame_margin_px)
            & (xs <= img_w - in_frame_margin_px)
            & (ys >= in_frame_margin_px)
            & (ys <= img_h - in_frame_margin_px)
        )

    j3d_full = _interpolate_missing_frames(j3d_full, valid_mask)
    j2d_full = _interpolate_missing_frames(j2d_full, valid_mask)
    unc_full = _interpolate_missing_frames(unc_full, valid_mask)
    boxes = _interpolate_missing_frames(boxes, valid_mask)

    detected_arm = (
        bowling_arm_override if bowling_arm_override is not None else _infer_bowling_arm(j3d_full)
    )

    return TrackedBowler3D(
        joints3d_cam_m=j3d_full,
        joints2d_px=j2d_full,
        joint_uncertainties_m=unc_full,
        joint_in_frame=in_frame,
        valid_mask=valid_mask,
        bowler_boxes_xyxy=boxes,
        detected_bowling_arm=detected_arm,
        track_score=best_score,
        num_tracks_seen=len(tracks),
        background_upright_vectors=bg_upright,
    )
