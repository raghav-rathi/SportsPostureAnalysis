"""Stage 2A: Metric 3D Multi-Person Pose & Uncertainty Extraction via Neural Localizer Fields (NLF).

Fixes EXP-007:
  - B3 / B16: Passes calibrated `intrinsic_matrix` `(B, 3, 3)` and `world_up_vector` `(3,)` directly into
              `model.detect_smpl_batched(...)` when pitch/two-wicket calibration is available!
  - B2: Decodes exact `source_frame_indices` sequentially (zero drift on pulldown-deduplicated clips).
  - B12: Inflates uncertainty proportionally to gap length on interpolated missing frames.
  - B20: Streams batches directly from `cv2.VideoCapture` instead of buffering 15 GB of raw RGB frames in RAM.
  - B23: Imports `JOINT_IDX` & `LR_JOINT_PAIRS` from `constants.py` (not `synthetic_ground_truth.py`).
  - B26: Computes and records SHA-256 checkpoint prefix on `TrackedBowler3D`.
  - Adds fast anchor-locked perturbation ensemble (`base`, `mirror`, `aug3`) for empirical repeatability intervals.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import JOINT_IDX, LR_JOINT_PAIRS
from .schemas import BowlingArm

_NLF_MODEL_CACHE: Dict[Tuple[str, str], object] = {}
_NLF_HASH_CACHE: Dict[str, str] = {}


@dataclass
class TrackedBowler3D:
    joints3d_cam_m: np.ndarray        # (T, 24, 3) meters in OpenCV camera frame
    joints2d_px: np.ndarray           # (T, 24, 2) pixels in source image
    joint_uncertainties_m: np.ndarray # (T, 24) NLF self-gating uncertainty (m-scaled, inflated on interp gaps)
    joint_in_frame: np.ndarray        # (T, 24) bool: True iff 2D joint lies inside image margins
    valid_mask: np.ndarray            # (T,) bool: True iff bowler detected on frame
    interp_gap_frames: np.ndarray     # (T,) int: distance in frames to nearest real detection (0 if valid)
    bowler_boxes_xyxy: np.ndarray     # (T, 4) [x1, y1, x2, y2] pixels
    detected_bowling_arm: BowlingArm
    track_score: float
    num_tracks_seen: int
    background_upright_vectors: List[np.ndarray] = field(default_factory=list)
    backend_name: str = "nlf_l_multi_torchscript"
    model_sha256: str = ""
    used_calibrated_intrinsics: bool = False


def _file_sha256_prefix(model_path: str, n_chars: int = 16) -> str:
    if model_path in _NLF_HASH_CACHE:
        return _NLF_HASH_CACHE[model_path]
    h = hashlib.sha256()
    with open(model_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    prefix = h.hexdigest()[:n_chars]
    _NLF_HASH_CACHE[model_path] = prefix
    return prefix


def _load_nlf_cached(model_path: str, device: str):
    import torch
    import torchvision  # noqa: F401 - required to register torchvision::nms before torch.jit.load

    key = (str(model_path), device)
    if key not in _NLF_MODEL_CACHE:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval()
        _NLF_MODEL_CACHE[key] = model
    return _NLF_MODEL_CACHE[key]


def _bbox_from_2d(j2d: np.ndarray) -> np.ndarray:
    finite = np.all(np.isfinite(j2d), axis=1)
    if not np.any(finite):
        return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float64)
    pts = j2d[finite]
    return np.array([pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max()], dtype=np.float64)


def _associate_tracks(
    per_frame_dets: List[List[Dict[str, np.ndarray]]],
    img_diag: float,
    max_gap: int = 10,
) -> List[Dict[int, Dict[str, np.ndarray]]]:
    """Depth-aware greedy multi-person tracker across video frames (2D pelvis + 3D pelvis proximity)."""
    tracks: List[Dict[int, Dict[str, np.ndarray]]] = []
    last_frame: List[int] = []
    last_pelv_2d: List[np.ndarray] = []
    last_pelv_3d: List[np.ndarray] = []

    dist_2d_thresh = max(0.14 * img_diag, 85.0)

    for f_idx, dets in enumerate(per_frame_dets):
        if not dets:
            continue
        unmatched_dets = list(range(len(dets)))
        candidates = []
        for t_id in range(len(tracks)):
            gap = f_idx - last_frame[t_id]
            if gap <= 0 or gap > max_gap:
                continue
            for d_i in unmatched_dets:
                p2 = dets[d_i]["joints2d"][JOINT_IDX["pelv"]]
                p3 = dets[d_i]["joints3d"][JOINT_IDX["pelv"]]
                d2 = float(np.linalg.norm(p2 - last_pelv_2d[t_id]))
                d3 = float(np.linalg.norm(p3 - last_pelv_3d[t_id]))
                # Allow slightly larger jump across multi-frame gaps
                if d2 <= dist_2d_thresh * (1.0 + 0.18 * (gap - 1)) and d3 <= 1.8 * (1.0 + 0.35 * (gap - 1)):
                    cost = (d2 / max(img_diag, 1.0)) + 0.25 * d3
                    candidates.append((cost, t_id, d_i))

        candidates.sort(key=lambda x: x[0])
        used_t = set()
        used_d = set()
        for _, t_id, d_i in candidates:
            if t_id in used_t or d_i in used_d:
                continue
            tracks[t_id][f_idx] = dets[d_i]
            last_frame[t_id] = f_idx
            last_pelv_2d[t_id] = dets[d_i]["joints2d"][JOINT_IDX["pelv"]]
            last_pelv_3d[t_id] = dets[d_i]["joints3d"][JOINT_IDX["pelv"]]
            used_t.add(t_id)
            used_d.add(d_i)

        for d_i in unmatched_dets:
            if d_i not in used_d:
                tracks.append({f_idx: dets[d_i]})
                last_frame.append(f_idx)
                last_pelv_2d.append(dets[d_i]["joints2d"][JOINT_IDX["pelv"]])
                last_pelv_3d.append(dets[d_i]["joints3d"][JOINT_IDX["pelv"]])

    return tracks


def _select_bowler_track(
    tracks: List[Dict[int, Dict[str, np.ndarray]]],
    n_frames: int,
    img_w: int,
    img_h: int,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    anchor_path_2d: Optional[np.ndarray] = None,
) -> Tuple[int, float]:
    """Score all tracked people and select the active fast bowler (filtering umpires/batters/keepers)."""
    if not tracks:
        raise RuntimeError("NLF detected zero human tracks in the video slice.")

    # When tracking a perturbation pass against a baseline path, lock strictly to the same person!
    if anchor_path_2d is not None and len(anchor_path_2d) == n_frames:
        best_i, best_d = 0, float("inf")
        min_len = max(4, int(0.15 * n_frames))
        for i, trk in enumerate(tracks):
            if len(trk) < min_len:
                continue
            ds = [float(np.linalg.norm(det["joints2d"][JOINT_IDX["pelv"]] - anchor_path_2d[f])) for f, det in trk.items()]
            cost = float(np.median(ds)) - 0.5 * len(ds)
            if cost < best_d:
                best_d = cost
                best_i = i
        return best_i, float(-best_d)

    best_idx = 0
    best_score = -1e9
    img_diag = max(float(np.hypot(img_w, img_h)), 1.0)

    for i, trk in enumerate(tracks):
        if len(trk) < max(4, int(0.15 * n_frames)):
            continue
        f_ids = sorted(trk.keys())
        j3 = np.stack([trk[f]["joints3d"] for f in f_ids], axis=0)
        j2 = np.stack([trk[f]["joints2d"] for f in f_ids], axis=0)

        cov = len(f_ids) / max(n_frames, 1)
        pelv3 = j3[:, JOINT_IDX["pelv"]]
        pelv_travel = float(np.linalg.norm(pelv3[-1] - pelv3[0]))
        pelv_path = float(np.sum(np.linalg.norm(np.diff(pelv3, axis=0), axis=1)))

        rwri_rel = j3[:, JOINT_IDX["rwri"]] - j3[:, JOINT_IDX["rsho"]]
        lwri_rel = j3[:, JOINT_IDX["lwri"]] - j3[:, JOINT_IDX["lsho"]]
        max_above_sho = float(max(np.max(-rwri_rel[:, 1]), np.max(-lwri_rel[:, 1])))
        wri_speed = float(
            max(
                np.percentile(np.linalg.norm(np.diff(j3[:, JOINT_IDX["rwri"]], axis=0), axis=1), 95),
                np.percentile(np.linalg.norm(np.diff(j3[:, JOINT_IDX["lwri"]], axis=0), axis=1), 95),
            )
        )

        box_heights = np.max(j2[:, :, 1], axis=1) - np.min(j2[:, :, 1], axis=1)
        med_h_frac = float(np.median(box_heights)) / max(img_h, 1)

        score = (
            2.2 * cov
            + 1.6 * min(pelv_travel, 4.0)
            + 0.8 * min(pelv_path, 6.0)
            + 3.5 * max(max_above_sho, 0.0)
            + 4.0 * min(wri_speed, 0.5)
            + 1.8 * min(med_h_frac, 0.6)
        )

        if anchor_xy_norm is not None:
            ax, ay = anchor_xy_norm[0] * img_w, anchor_xy_norm[1] * img_h
            med_p2 = np.median(j2[:, JOINT_IDX["pelv"]], axis=0)
            d_norm = float(np.hypot(med_p2[0] - ax, med_p2[1] - ay)) / img_diag
            score -= 10.0 * d_norm

        if score > best_score:
            best_score = score
            best_idx = i

    return best_idx, best_score


def _angle3_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    v1 = a - b
    v2 = c - b
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_a)))


def _extract_standing_background_gravity(
    tracks: List[Dict[int, Dict[str, np.ndarray]]],
    bowler_tid: int,
    img_w: int,
    img_h: int,
) -> List[np.ndarray]:
    """Extract upright torso vectors ONLY from verified standing-straight non-bowler people."""
    upright_vecs: List[np.ndarray] = []
    for tid, trk in enumerate(tracks):
        if tid == bowler_tid or len(trk) < 3:
            continue
        f_ids = sorted(trk.keys())
        pelv_seq = np.stack([trk[f]["joints3d"][JOINT_IDX["pelv"]] for f in f_ids], axis=0)
        if float(np.linalg.norm(pelv_seq[-1] - pelv_seq[0])) > 1.2:
            continue
        for f in f_ids:
            j3 = trk[f]["joints3d"]
            j2 = trk[f]["joints2d"]
            unc = trk[f]["unc"]
            req_i = [
                JOINT_IDX["pelv"], JOINT_IDX["neck"],
                JOINT_IDX["lhip"], JOINT_IDX["lkne"], JOINT_IDX["lank"],
                JOINT_IDX["rhip"], JOINT_IDX["rkne"], JOINT_IDX["rank"],
            ]
            pts2 = j2[req_i]
            if np.any(pts2[:, 0] < 10) or np.any(pts2[:, 0] > img_w - 10):
                continue
            if np.any(pts2[:, 1] < 10) or np.any(pts2[:, 1] > img_h - 10):
                continue
            if float(np.max(unc[req_i])) > 0.11:
                continue

            l_knee = _angle3_deg(j3[JOINT_IDX["lhip"]], j3[JOINT_IDX["lkne"]], j3[JOINT_IDX["lank"]])
            r_knee = _angle3_deg(j3[JOINT_IDX["rhip"]], j3[JOINT_IDX["rkne"]], j3[JOINT_IDX["rank"]])
            torso = j3[JOINT_IDX["neck"]] - j3[JOINT_IDX["pelv"]]
            l_thigh = j3[JOINT_IDX["lkne"]] - j3[JOINT_IDX["lhip"]]
            r_thigh = j3[JOINT_IDX["rkne"]] - j3[JOINT_IDX["rhip"]]
            l_hip_trunk = _angle3_deg(j3[JOINT_IDX["neck"]], j3[JOINT_IDX["pelv"]], j3[JOINT_IDX["pelv"]] + l_thigh)
            r_hip_trunk = _angle3_deg(j3[JOINT_IDX["neck"]], j3[JOINT_IDX["pelv"]], j3[JOINT_IDX["pelv"]] + r_thigh)

            if l_knee > 155.0 and r_knee > 155.0 and l_hip_trunk > 155.0 and r_hip_trunk > 155.0:
                mid_ank = 0.5 * (j3[JOINT_IDX["lank"]] + j3[JOINT_IDX["rank"]])
                vec = j3[JOINT_IDX["neck"]] - mid_ank
                nrm = float(np.linalg.norm(vec))
                if nrm > 1.0:
                    upright_vecs.append(vec / nrm)
    return upright_vecs


def _infer_bowling_arm(joints3d_cam_m: np.ndarray) -> BowlingArm:
    """Detect right vs left bowling arm via peak straight overhead reach, shoulder-relative speed, & arc."""
    def _arm_score(sho_k: str, elb_k: str, wri_k: str) -> float:
        sho = joints3d_cam_m[:, JOINT_IDX[sho_k]]
        elb = joints3d_cam_m[:, JOINT_IDX[elb_k]]
        wri = joints3d_cam_m[:, JOINT_IDX[wri_k]]
        arm_vec = wri - sho
        above = np.maximum(-arm_vec[:, 1], 0.0)
        arm_len = np.linalg.norm(arm_vec, axis=1)
        v1 = sho - elb
        v2 = wri - elb
        cos_e = np.clip(
            np.sum(v1 * v2, axis=1) / np.maximum(np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1), 1e-6),
            -1.0,
            1.0,
        )
        straight_w = np.clip((np.degrees(np.arccos(cos_e)) - 90.0) / 90.0, 0.2, 1.0)
        reach_series = above * arm_len * straight_w
        k_top = min(2, len(reach_series))
        peak_reach = float(np.mean(np.sort(reach_series)[-k_top:]))
        rel_vel = np.linalg.norm(np.diff(arm_vec, axis=0), axis=1)
        k_vel = min(3, len(rel_vel))
        peak_rel_vel = float(np.mean(np.sort(rel_vel)[-k_vel:])) if k_vel > 0 else 0.0
        vert_range = float(np.max(-arm_vec[:, 1]) - np.min(-arm_vec[:, 1]))
        return peak_reach * 4.0 + peak_rel_vel * 5.0 + vert_range * 1.5

    return BowlingArm.RIGHT if _arm_score("rsho", "relb", "rwri") >= _arm_score("lsho", "lelb", "lwri") else BowlingArm.LEFT


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


def _compute_interp_gap_frames(valid_mask: np.ndarray) -> np.ndarray:
    """Compute distance in frames to the nearest real detection for every frame (0 when valid)."""
    n = len(valid_mask)
    valid_idx = np.where(valid_mask)[0]
    if len(valid_idx) == 0:
        return np.full(n, n, dtype=np.int32)
    if len(valid_idx) == n:
        return np.zeros(n, dtype=np.int32)
    gaps = np.zeros(n, dtype=np.int32)
    for t in range(n):
        if not valid_mask[t]:
            gaps[t] = int(np.min(np.abs(valid_idx - t)))
    return gaps


def mirror_tracked_bowler(tracked: TrackedBowler3D, img_w: int) -> TrackedBowler3D:
    """Undo horizontal image flip: negate camera X, mirror pixel x, swap anatomical left/right joints."""
    j3 = tracked.joints3d_cam_m.copy()
    j2 = tracked.joints2d_px.copy()
    unc = tracked.joint_uncertainties_m.copy()
    in_f = tracked.joint_in_frame.copy()
    boxes = tracked.bowler_boxes_xyxy.copy()

    j3[..., 0] *= -1.0
    j2[..., 0] = (img_w - 1.0) - j2[..., 0]
    x0_new = (img_w - 1.0) - boxes[:, 2]
    x1_new = (img_w - 1.0) - boxes[:, 0]
    boxes[:, 0], boxes[:, 2] = x0_new, x1_new

    for l_i, r_i in LR_JOINT_PAIRS:
        j3[:, [l_i, r_i]] = j3[:, [r_i, l_i]]
        j2[:, [l_i, r_i]] = j2[:, [r_i, l_i]]
        unc[:, [l_i, r_i]] = unc[:, [r_i, l_i]]
        in_f[:, [l_i, r_i]] = in_f[:, [r_i, l_i]]

    bg = [v * np.array([-1.0, 1.0, 1.0]) for v in tracked.background_upright_vectors]
    arm = BowlingArm.LEFT if tracked.detected_bowling_arm == BowlingArm.RIGHT else BowlingArm.RIGHT

    return TrackedBowler3D(
        joints3d_cam_m=j3,
        joints2d_px=j2,
        joint_uncertainties_m=unc,
        joint_in_frame=in_f,
        valid_mask=tracked.valid_mask.copy(),
        interp_gap_frames=tracked.interp_gap_frames.copy(),
        bowler_boxes_xyxy=boxes,
        detected_bowling_arm=arm,
        track_score=tracked.track_score,
        num_tracks_seen=tracked.num_tracks_seen,
        background_upright_vectors=bg,
        backend_name=tracked.backend_name,
        model_sha256=tracked.model_sha256,
        used_calibrated_intrinsics=tracked.used_calibrated_intrinsics,
    )


def extract_bowler_3d_nlf(
    video_path: str | Path,
    model_path: str | Path,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    source_frame_indices: Optional[List[int]] = None,
    batch_size: int = 16,
    num_aug: int = 1,
    fov_deg: float = 55.0,
    intrinsic_matrix: Optional[np.ndarray] = None,
    world_up_vector: Optional[np.ndarray] = None,
    horizontal_flip: bool = False,
    in_frame_margin_px: float = 12.0,
    bowling_arm_override: Optional[BowlingArm] = None,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    anchor_path_2d: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> TrackedBowler3D:
    """Stream video frames through NLF (`nlf_l_multi.torchscript`) with optional calibrated K & world_up."""
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_frame_indices and len(source_frame_indices) > 0:
        target_indices = [int(i) for i in source_frame_indices]
    else:
        sf = max(0, int(start_frame))
        ef = min(total_frames, int(end_frame)) if end_frame is not None else total_frames
        target_indices = list(range(sf, ef))

    if not target_indices:
        cap.release()
        raise RuntimeError(f"No frames requested from {video_path}.")

    model_str = str(model_path)
    model_sha = _file_sha256_prefix(model_str)
    model = _load_nlf_cached(model_str, device)

    # Prepare calibrated intrinsic_matrix & world_up_vector tensors (mirroring cleanly if horizontal_flip=True!)
    K_t_base = None
    if intrinsic_matrix is not None:
        K_use = np.asarray(intrinsic_matrix, dtype=np.float32).copy()
        if horizontal_flip:
            K_use[0, 2] = float(img_w - 1) - K_use[0, 2]
        K_t_base = torch.from_numpy(K_use).to(device=device, dtype=torch.float32)

    up_t = None
    if world_up_vector is not None:
        up_use = np.asarray(world_up_vector, dtype=np.float32).reshape(3).copy()
        if horizontal_flip:
            up_use[0] *= -1.0
        up_t = torch.from_numpy(up_use).to(device=device, dtype=torch.float32)

    per_frame_dets: List[List[Dict[str, np.ndarray]]] = []
    batch_buf: List[np.ndarray] = []

    def _flush_batch() -> None:
        if not batch_buf:
            return
        b_len = len(batch_buf)
        batch_np = np.stack(batch_buf, axis=0)
        batch_t = torch.from_numpy(batch_np).permute(0, 3, 1, 2).contiguous().to(device)
        kwargs: Dict[str, object] = {
            "default_fov_degrees": float(fov_deg),
            "num_aug": int(num_aug),
            "detector_threshold": 0.30,
            "max_detections": 12,
            "suppress_implausible_poses": True,
        }
        if K_t_base is not None:
            kwargs["intrinsic_matrix"] = K_t_base.unsqueeze(0).expand(b_len, 3, 3).contiguous()
        if up_t is not None:
            kwargs["world_up_vector"] = up_t
        out = model.detect_smpl_batched(batch_t, **kwargs)

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
                torso_lens = np.linalg.norm(
                    j3d_np[:, JOINT_IDX["neck"]] - j3d_np[:, JOINT_IDX["pelv"]], axis=-1
                )
                if float(np.median(torso_lens)) > 2.0:
                    j3d_np = j3d_np / 1000.0
                if float(np.median(unc_np)) > 2.0:
                    unc_np = unc_np / 1000.0
                for p in range(n_det):
                    dets_i.append({"joints3d": j3d_np[p], "joints2d": j2d_np[p], "unc": unc_np[p]})
            per_frame_dets.append(dets_i)
        batch_buf.clear()

    # Stream sequentially through VideoCapture (avoids 15 GB RAM spikes & GOP seek bugs, EXP-007 B20)
    target_set = set(target_indices)
    first_f, last_f = target_indices[0], target_indices[-1]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_f)
    with torch.inference_mode():
        for abs_f in range(first_f, last_f + 1):
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                break
            if abs_f not in target_set:
                continue
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if horizontal_flip:
                rgb = np.ascontiguousarray(rgb[:, ::-1])
            batch_buf.append(rgb)
            if len(batch_buf) >= batch_size:
                _flush_batch()
        _flush_batch()
    cap.release()

    n_frames = len(per_frame_dets)
    if n_frames == 0:
        raise RuntimeError(f"No frames extracted from {video_path} [{first_f}:{last_f}].")

    anchor_path_use = anchor_path_2d
    if horizontal_flip and anchor_path_use is not None:
        anchor_path_use = anchor_path_use.copy()
        anchor_path_use[:, 0] = (img_w - 1.0) - anchor_path_use[:, 0]

    anchor_xy_use = anchor_xy_norm
    if horizontal_flip and anchor_xy_use is not None:
        anchor_xy_use = (1.0 - anchor_xy_use[0], anchor_xy_use[1])

    tracks = _associate_tracks(per_frame_dets, img_diag=float(np.hypot(img_w, img_h)))
    best_tid, best_score = _select_bowler_track(
        tracks,
        n_frames=n_frames,
        img_w=img_w,
        img_h=img_h,
        anchor_xy_norm=anchor_xy_use,
        anchor_path_2d=anchor_path_use,
    )
    chosen = tracks[best_tid]
    bg_upright = _extract_standing_background_gravity(tracks, best_tid, img_w, img_h)

    j3d_full = np.zeros((n_frames, 24, 3), dtype=np.float64)
    j2d_full = np.zeros((n_frames, 24, 2), dtype=np.float64)
    unc_full = np.full((n_frames, 24), 0.14, dtype=np.float64)
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

    interp_gaps = _compute_interp_gap_frames(valid_mask)
    j3d_full = _interpolate_missing_frames(j3d_full, valid_mask)
    j2d_full = _interpolate_missing_frames(j2d_full, valid_mask)
    unc_full = _interpolate_missing_frames(unc_full, valid_mask)
    boxes = _interpolate_missing_frames(boxes, valid_mask)
    # Inflate uncertainty on interpolated missing frames (EXP-007 B12)
    if np.any(interp_gaps > 0):
        gap_mult = (1.0 + 0.35 * interp_gaps.astype(np.float64))[:, None]
        unc_full = np.minimum(unc_full * gap_mult, 0.25)

    eff_override = bowling_arm_override
    if horizontal_flip and eff_override is not None:
        eff_override = BowlingArm.LEFT if eff_override == BowlingArm.RIGHT else BowlingArm.RIGHT

    detected_arm = eff_override if eff_override is not None else _infer_bowling_arm(j3d_full)

    tracked = TrackedBowler3D(
        joints3d_cam_m=j3d_full,
        joints2d_px=j2d_full,
        joint_uncertainties_m=unc_full,
        joint_in_frame=in_frame,
        valid_mask=valid_mask,
        interp_gap_frames=interp_gaps,
        bowler_boxes_xyxy=boxes,
        detected_bowling_arm=detected_arm,
        track_score=best_score,
        num_tracks_seen=len(tracks),
        background_upright_vectors=bg_upright,
        backend_name=f"nlf_l_multi_torchscript@{model_sha}",
        model_sha256=model_sha,
        used_calibrated_intrinsics=(intrinsic_matrix is not None),
    )
    if horizontal_flip:
        tracked = mirror_tracked_bowler(tracked, img_w=img_w)
    return tracked


def extract_bowler_3d_nlf_ensemble(
    video_path: str | Path,
    model_path: str | Path,
    source_frame_indices: List[int],
    batch_size: int = 16,
    fov_deg: float = 55.0,
    intrinsic_matrix: Optional[np.ndarray] = None,
    world_up_vector: Optional[np.ndarray] = None,
    bowling_arm_override: Optional[BowlingArm] = None,
    anchor_xy_norm: Optional[Tuple[float, float]] = None,
    device: str = "cuda",
) -> Tuple[TrackedBowler3D, List[TrackedBowler3D]]:
    """Run anchor-locked (base, horizontal_flip, num_aug=3) NLF passes for empirical repeatability intervals."""
    base = extract_bowler_3d_nlf(
        video_path=video_path,
        model_path=model_path,
        source_frame_indices=source_frame_indices,
        batch_size=batch_size,
        num_aug=1,
        fov_deg=fov_deg,
        intrinsic_matrix=intrinsic_matrix,
        world_up_vector=world_up_vector,
        horizontal_flip=False,
        bowling_arm_override=bowling_arm_override,
        anchor_xy_norm=anchor_xy_norm,
        device=device,
    )
    anchor_path = base.joints2d_px[:, JOINT_IDX["pelv"]]
    arm_locked = base.detected_bowling_arm

    flipped = extract_bowler_3d_nlf(
        video_path=video_path,
        model_path=model_path,
        source_frame_indices=source_frame_indices,
        batch_size=batch_size,
        num_aug=1,
        fov_deg=fov_deg,
        intrinsic_matrix=intrinsic_matrix,
        world_up_vector=world_up_vector,
        horizontal_flip=True,
        bowling_arm_override=arm_locked,
        anchor_path_2d=anchor_path,
        device=device,
    )
    aug3 = extract_bowler_3d_nlf(
        video_path=video_path,
        model_path=model_path,
        source_frame_indices=source_frame_indices,
        batch_size=batch_size,
        num_aug=3,
        fov_deg=fov_deg,
        intrinsic_matrix=intrinsic_matrix,
        world_up_vector=world_up_vector,
        horizontal_flip=False,
        bowling_arm_override=arm_locked,
        anchor_path_2d=anchor_path,
        device=device,
    )
    return base, [base, flipped, aug3]
