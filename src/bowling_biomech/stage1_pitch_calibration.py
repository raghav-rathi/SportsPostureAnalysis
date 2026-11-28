"""Stage 1: Cricket Pitch Geometry Calibration & Camera View Estimation.

Fixes EXP-007:
  - B3 / B16: Larger-side FOV intrinsic convention matching NLF (`max(width, height)`), plus
              closed-form Two-Wicket Geometry calibration (`auto_detect_two_wickets` with strict
              Law 8 stump-triplet ratio invariants + explicit two-wicket pixel coordinates) to supply
              calibrated `K` and `world_up_vector` BEFORE NLF inference!
  - B4: Never discards `is_panning_camera` on landmark/two-wicket calibration paths.
  - B6: Reports explicit `heading_sigma_deg`, `gravity_sigma_deg`, and `gravity_source`.
  - B15: Removes underground clamps that corrupted upward-looking ground-level phones and desynced
         `camera_position_world_m` from `(R, t)`.
  - B24: Regulation MCC Law 6 (20.12 m pitch), Law 7 (creases), and Law 8 (0.7112 m stumps,
         off/leg centers at +-0.0960 m).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .constants import (
    MCC_PITCH_LANDMARKS_M,
    MCC_PITCH_LENGTH_M,
    MCC_STUMP_HEIGHT_M,
)
from .schemas import CameraViewClass, PitchCalibration

# Backward-compatible export name pointing to corrected MCC Laws 6/7/8 landmarks
MCC_LAW7_LANDMARKS_M: Dict[str, Tuple[float, float, float]] = MCC_PITCH_LANDMARKS_M


def default_camera_matrix(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Estimate pinhole intrinsic matrix K applying `fov_deg` to the LARGER image side (matches NLF)."""
    focal = 0.5 * float(max(width, height)) / math.tan(math.radians(fov_deg) / 2.0)
    return np.array(
        [[focal, 0.0, width / 2.0], [0.0, focal, height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def classify_camera_view(azimuth_deg: float, elevation_deg: float) -> CameraViewClass:
    """Classify canonical camera viewing angle from azimuth (0=behind bowler, 90=side-on, 180=behind batter)."""
    az = abs(azimuth_deg) % 360.0
    if az > 180.0:
        az = 360.0 - az

    if elevation_deg >= 18.0 and (az <= 30.0 or az >= 150.0):
        return CameraViewClass.BROADCAST_ELEVATED
    if az <= 22.0:
        return CameraViewClass.REAR_BEHIND_BOWLER
    if az >= 158.0:
        return CameraViewClass.FRONT_BEHIND_BATTER
    if 68.0 <= az <= 112.0:
        return (
            CameraViewClass.SIDE_ON_RIGHT
            if (azimuth_deg % 360.0) <= 180.0
            else CameraViewClass.SIDE_ON_LEFT
        )
    if 22.0 < az < 68.0 or 112.0 < az < 158.0:
        return CameraViewClass.OPEN_SIDE_45
    return CameraViewClass.UNKNOWN


def _angles_from_camera_position(cam_pos: np.ndarray, look_target: np.ndarray) -> Tuple[float, float]:
    """Compute camera azimuth (deg from -X behind bowler) and elevation (deg above horizontal)."""
    v = cam_pos - look_target
    dx = -v[0]
    dz = v[2]
    dy = v[1]
    horiz_dist = max(float(math.hypot(dx, dz)), 1e-6)
    azimuth_deg = math.degrees(math.atan2(dz, dx))
    elevation_deg = math.degrees(math.atan2(dy, horiz_dist))
    return float(azimuth_deg), float(elevation_deg)


def world_up_from_calibration(cal: PitchCalibration) -> np.ndarray:
    """Return unit world-UP (+Y_world) expressed in OpenCV camera coordinates (+X right, +Y down, +Z forward)."""
    R = np.asarray(cal.rotation_matrix_R, dtype=np.float64)
    up = R[:, 1].copy()
    nrm = float(np.linalg.norm(up))
    if nrm < 1e-6:
        return np.array([0.0, -1.0, 0.0], dtype=np.float64)
    up /= nrm
    # Sanity orientation: in upright camera video, world-UP has negative OpenCV Y component
    if up[1] > 0.0:
        up = -up
    return up


def calibrate_from_mcc_landmarks(
    image_landmarks_px: Dict[str, Tuple[float, float]],
    width: int,
    height: int,
    K: Optional[np.ndarray] = None,
    fov_search_deg: Tuple[float, ...] = (8.0, 12.0, 18.0, 26.0, 38.0, 50.0, 62.0),
    is_panning_camera: bool = False,
) -> PitchCalibration:
    """Solve PnP camera extrinsics from >=4 named MCC Law 6/7/8 pitch landmarks.

    Handles both 3D (creases + stumps) and coplanar (Y=0 creases) configurations via
    `solvePnPGeneric(SOLVEPNP_IPPE)` with cheirality and above-ground checks.
    Preserves `is_panning_camera` (EXP-007 B4): single-frame landmarks calibrate lens intrinsics `K`,
    but world ground translation remains uncalibrated if the camera pans.
    """
    valid_keys = [k for k in image_landmarks_px if k in MCC_PITCH_LANDMARKS_M]
    if len(valid_keys) < 4:
        raise ValueError(
            f"At least 4 valid MCC pitch landmarks required for PnP; got {len(valid_keys)}."
        )

    obj_pts = np.array([MCC_PITCH_LANDMARKS_M[k] for k in valid_keys], dtype=np.float64)
    img_pts = np.array([image_landmarks_px[k] for k in valid_keys], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    candidate_Ks = [K] if K is not None else [default_camera_matrix(width, height, f) for f in fov_search_deg]

    best_err = float("inf")
    best_solution = None
    is_coplanar = bool(np.ptp(obj_pts[:, 1]) < 1e-3)

    for cand_K in candidate_Ks:
        raw_solutions: List[Tuple[np.ndarray, np.ndarray]] = []
        if is_coplanar:
            try:
                n_sol, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    obj_pts, img_pts, cand_K, dist_coeffs, flags=cv2.SOLVEPNP_IPPE
                )
                for s_i in range(n_sol):
                    raw_solutions.append((rvecs[s_i], tvecs[s_i]))
            except cv2.error:
                pass
        else:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, cand_K, dist_coeffs, flags=cv2.SOLVEPNP_EPNP
                )
                if ok:
                    raw_solutions.append((rvec, tvec))
            except cv2.error:
                pass
        if not raw_solutions:
            try:
                ok, rvec, tvec = cv2.solvePnP(
                    obj_pts, img_pts, cand_K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
                )
                if ok:
                    raw_solutions.append((rvec, tvec))
            except cv2.error:
                continue

        for rvec0, tvec0 in raw_solutions:
            try:
                rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, cand_K, dist_coeffs, rvec0, tvec0)
            except cv2.error:
                rvec, tvec = rvec0, tvec0
            R, _ = cv2.Rodrigues(rvec)
            pts_cam = (R @ obj_pts.T).T + tvec.reshape(1, 3)
            if np.any(pts_cam[:, 2] <= 0.2):
                continue
            cam_pos = (-R.T @ tvec).reshape(3)
            # Physical camera height must be strictly above pitch surface (>0.05 m; allows ground-level phones!)
            if cam_pos[1] < 0.05:
                continue
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, cand_K, dist_coeffs)
            err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - img_pts, axis=1)))
            if err < best_err:
                best_err = err
                best_solution = (cand_K, R, tvec.reshape(3), cam_pos)

    if best_solution is None:
        raise RuntimeError("PnP solver failed to find a valid above-ground camera pose.")

    best_K, R, tvec, cam_pos = best_solution
    crease_pelvis_ref = np.array([0.61, 1.0, 0.0], dtype=np.float64)
    az_deg, el_deg = _angles_from_camera_position(cam_pos, crease_pelvis_ref)

    p0, _ = cv2.projectPoints(
        np.array([[1.22, 0.0, 0.0], [1.22, 1.0, 0.0]], dtype=np.float64),
        cv2.Rodrigues(R)[0],
        tvec.reshape(3, 1),
        best_K,
        dist_coeffs,
    )
    px_per_m = float(np.linalg.norm(p0[1, 0] - p0[0, 0]))
    view_class = classify_camera_view(az_deg, el_deg)

    notes = [
        f"Calibrated via MCC Pitch PnP ({'coplanar IPPE' if is_coplanar else '3D EPNP+LM'}) "
        f"using {len(valid_keys)} landmarks (mean reproj err={best_err:.2f} px)."
    ]
    if best_err > 12.0:
        notes.append("Warning: high reprojection error (>12 px); check landmark accuracy or lens zoom.")
    if is_panning_camera:
        notes.append(
            "Camera motion/pan detected: single-frame MCC PnP supplies calibrated intrinsics K, "
            "but ground-referenced translation metrics remain gated."
        )

    return PitchCalibration(
        method="mcc_law7_pnp",
        camera_matrix_K=best_K.tolist(),
        rotation_matrix_R=R.tolist(),
        translation_t=tvec.tolist(),
        camera_position_world_m=cam_pos.tolist(),
        azimuth_deg=round(az_deg, 2),
        elevation_deg=round(el_deg, 2),
        px_per_meter_at_crease=round(px_per_m, 2),
        reprojection_error_px=round(best_err, 3),
        view_class=view_class,
        is_metric_calibrated=(not is_panning_camera) and (best_err <= 15.0),
        is_intrinsics_calibrated=(best_err <= 15.0),
        is_panning_camera=bool(is_panning_camera),
        heading_sigma_deg=1.5 if not is_panning_camera else 3.5,
        gravity_sigma_deg=1.0 if not is_panning_camera else 2.0,
        gravity_source="mcc_pnp",
        calibration_notes=notes,
    )


def solve_two_wicket_calibration(
    h_near_px: float,
    base_near_y_px: float,
    h_far_px: float,
    base_far_y_px: float,
    width: int,
    height: int,
    roll_deg: float = 0.0,
    near_x_px: Optional[float] = None,
    far_x_px: Optional[float] = None,
    is_panning_camera: bool = False,
    from_front: bool = False,
) -> Optional[PitchCalibration]:
    """Closed-form camera intrinsics & extrinsics from two visible regulation wickets 20.12 m apart.

    Uses MCC Law 6 (L = 20.12 m) and Law 8 (H = 0.7112 m):
      Z_near = L / (h_near / h_far - 1),   f = h_near * Z_near / H
    Recovers exact focal length, camera distance, camera height, downward pitch, and Pitch World R, t.
    """
    if h_near_px <= 15.0 or h_far_px <= 10.0:
        return None
    height_ratio = h_near_px / h_far_px
    if not (1.15 <= height_ratio <= 5.0):
        return None

    z_near = MCC_PITCH_LENGTH_M / (height_ratio - 1.0)
    focal = h_near_px * z_near / MCC_STUMP_HEIGHT_M
    fov_max_deg = 2.0 * math.degrees(math.atan(0.5 * float(max(width, height)) / max(focal, 1.0)))
    if not (3.5 <= fov_max_deg <= 80.0 and 4.0 <= z_near <= 95.0):
        return None

    ratio_z = (z_near + MCC_PITCH_LENGTH_M) / z_near
    horizon_y = (ratio_z * base_far_y_px - base_near_y_px) / max(ratio_z - 1.0, 1e-6)
    cam_h = (base_near_y_px - horizon_y) * z_near / max(focal, 1.0)
    # Positive pitch_down_rad = camera tilted downward (horizon lies above image center cy)
    cy = 0.5 * height
    cx = 0.5 * width
    pitch_down_rad = math.atan((cy - horizon_y) / max(focal, 1.0))
    pitch_down_deg = math.degrees(pitch_down_rad)
    if not (0.20 <= cam_h <= 22.0 and abs(pitch_down_deg) <= 30.0):
        return None

    # Optional horizontal yaw offset from wicket line
    yaw_rad = 0.0
    if near_x_px is not None and far_x_px is not None:
        vp_x = (ratio_z * far_x_px - near_x_px) / max(ratio_z - 1.0, 1e-6)
        yaw_rad = float(np.clip(math.atan((vp_x - cx) / max(focal, 1.0)), -math.radians(22.0), math.radians(22.0)))

    roll_rad = math.radians(roll_deg)
    # Build world (+X down pitch, +Y gravity-up, +Z bowler's right) -> OpenCV camera (+X right, +Y down, +Z forward)
    # For a camera behind the bowler looking along +X (yaw=0, pitch=0, roll=0):
    #   X_w -> +Z_c,  Y_w -> -Y_c,  Z_w -> +X_c
    # Rotate by downward pitch (Rx(-pitch_down)), yaw (Ry(-yaw)), and roll (Rz(roll))
    cp, sp = math.cos(pitch_down_rad), math.sin(pitch_down_rad)
    cr, sr = math.cos(roll_rad), math.sin(roll_rad)
    cyw, syw = math.cos(yaw_rad), math.sin(yaw_rad)

    R_roll = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    R_pitch = np.array([[1.0, 0.0, 0.0], [0.0, cp, sp], [0.0, -sp, cp]], dtype=np.float64)
    R_yaw = np.array([[cyw, 0.0, -syw], [0.0, 1.0, 0.0], [syw, 0.0, cyw]], dtype=np.float64)

    if not from_front:
        # Behind bowler: X_w -> +Z_c, Y_w -> -Y_c, Z_w -> +X_c
        R_base = np.array([[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    else:
        # Behind batter looking toward bowler (-X): X_w -> -Z_c, Y_w -> -Y_c, Z_w -> -X_c
        R_base = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64)

    R = R_roll @ R_pitch @ R_yaw @ R_base
    if near_x_px is not None:
        # Anchor near wicket base explicitly to its detected pixel ray so horizontal yaw never shifts
        # the bowler's wicket laterally off Z = 0, while keeping `cam_pos_world == -R.T @ t` exact (EXP-007 B15).
        tx_cam = (float(near_x_px) - cx) * z_near / max(focal, 1.0)
        ty_cam = (float(base_near_y_px) - cy) * z_near / max(focal, 1.0)
        p_near_cam = np.array([tx_cam, ty_cam, z_near], dtype=np.float64)
        if not from_front:
            tvec = p_near_cam
        else:
            tvec = (p_near_cam - R @ np.array([MCC_PITCH_LENGTH_M, 0.0, 0.0], dtype=np.float64)).reshape(3)
        cam_pos_world = (-R.T @ tvec).reshape(3)
    else:
        if not from_front:
            cam_pos_world = np.array([-z_near, cam_h, 0.0], dtype=np.float64)
        else:
            cam_pos_world = np.array([MCC_PITCH_LENGTH_M + z_near, cam_h, 0.0], dtype=np.float64)
        tvec = (-R @ cam_pos_world).reshape(3)

    K = np.array([[focal, 0.0, cx], [0.0, focal, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    crease_pelvis_ref = np.array([0.61, 1.0, 0.0], dtype=np.float64)
    az_deg, el_deg = _angles_from_camera_position(cam_pos_world, crease_pelvis_ref)
    view_class = classify_camera_view(az_deg, el_deg)
    px_per_m = focal / max(z_near, 1.0)

    notes = [
        f"Closed-form Two-Wicket calibration (MCC Laws 6 & 8): f={focal:.0f} px "
        f"(FOV_max={fov_max_deg:.1f} deg), Z_near={z_near:.1f} m, cam_height={cam_h:.2f} m, "
        f"pitch={-pitch_down_deg:.2f} deg, roll={roll_deg:.2f} deg."
    ]
    if is_panning_camera:
        notes.append(
            "Camera motion/pan detected: Two-Wicket focal length K and vertical gravity are calibrated "
            "(passed directly into NLF), while ground-referenced translation metrics remain gated."
        )

    return PitchCalibration(
        method="two_wicket_geometry",
        camera_matrix_K=K.tolist(),
        rotation_matrix_R=R.tolist(),
        translation_t=tvec.tolist(),
        camera_position_world_m=cam_pos_world.tolist(),
        azimuth_deg=round(az_deg, 2),
        elevation_deg=round(el_deg, 2),
        px_per_meter_at_crease=round(px_per_m, 2),
        reprojection_error_px=None,
        view_class=view_class,
        is_metric_calibrated=(not is_panning_camera),
        is_intrinsics_calibrated=True,
        is_panning_camera=bool(is_panning_camera),
        heading_sigma_deg=2.2 if not is_panning_camera else 3.8,
        gravity_sigma_deg=1.4,
        gravity_source="two_wickets",
        calibration_notes=notes,
    )


def auto_detect_two_wickets(
    video_path: str | Path,
    source_frame_indices: Optional[List[int]] = None,
    start_frame: int = 0,
    end_frame: Optional[int] = None,
    is_panning_camera: bool = False,
) -> Optional[PitchCalibration]:
    """Conservatively auto-detect two regulation training wickets (near + far) with strict Law 8 invariants.

    Guards against false positives on yellow shirts/banners via:
      1. Multi-frame stationarity across 7–9 sampled frames (IQR(h)/med(h) <= 0.08)
      2. Tall vertical post geometry (h/w in [4.5, 45], fill >= 0.40)
      3. Dimensionless MCC Law 8 stump-spacing ratio (0.192 / 0.7112 = 0.270 outer pair; 0.135 adjacent)
      4. Perspective consistency (h_near/h_far in [1.18, 4.50], y_base_near > y_base_far + 10 px)
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if source_frame_indices and len(source_frame_indices) >= 3:
        pool = source_frame_indices
    else:
        sf = max(0, int(start_frame))
        ef = min(total, int(end_frame)) if end_frame is not None else total
        pool = list(range(sf, max(sf + 1, ef)))

    # Sample up to 9 evenly spaced frames across the clip (focusing on late frames too when bowler clears stumps)
    pick_pos = np.unique(np.linspace(0, len(pool) - 1, min(9, len(pool)), dtype=int))
    pick_abs = [int(pool[int(p)]) for p in pick_pos]
    pick_set = set(pick_abs)

    cap.set(cv2.CAP_PROP_POS_FRAMES, pick_abs[0])
    per_frame_sols: List[Tuple[float, float, float, float, float, float]] = []

    for abs_f in range(pick_abs[0], pick_abs[-1] + 1):
        ok, img = cap.read()
        if not ok or img is None:
            break
        if abs_f not in pick_set:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # Saturated yellow/orange-yellow training stumps
        mask = cv2.inRange(hsv, (18, 120, 120), (36, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n_cc, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        posts = []
        for i in range(1, n_cc):
            x_c, y_c, bw, bh, area = (int(stats[i][k]) for k in range(5))
            if bh < 55 or bw < 2:
                continue
            aspect = bh / max(bw, 1)
            fill = area / max(bw * bh, 1)
            if 4.0 <= aspect <= 48.0 and fill >= 0.40 and area >= 260:
                posts.append((x_c, y_c, bw, bh))

        if len(posts) < 3:
            continue

        # Cluster posts into near vs far wickets by largest height jump
        posts.sort(key=lambda p: p[3])
        h_vals = np.array([p[3] for p in posts], dtype=np.float64)
        split = int(np.argmax(np.diff(h_vals))) + 1
        far_posts, near_posts = posts[:split], posts[split:]
        if not far_posts or not near_posts:
            continue

        def _valid_wicket_cluster(cl: List[Tuple[int, int, int, int]]) -> Tuple[bool, bool, float, float, float]:
            # Returns (is_valid_cluster, passes_law8_spacing, h_rep, y_base, x_center)
            h_rep = float(max(p[3] for p in cl))
            # Filter outliers inconsistent in base-y or height
            base_med = float(np.median([p[1] + p[3] for p in cl]))
            clean = [
                p for p in cl
                if abs((p[1] + p[3]) - base_med) <= max(6.0, 0.05 * h_rep)
                and (p[3] / h_rep) >= 0.80
            ]
            if not clean:
                return False, False, 0.0, 0.0, 0.0
            h_clean = float(max(p[3] for p in clean))
            y_base = float(np.mean([p[1] + p[3] for p in clean]))
            xs = sorted(p[0] + 0.5 * p[2] for p in clean)
            x_ctr = float(np.mean(xs))
            law8_ok = False
            if 2 <= len(clean) <= 3:
                span_ratio = (xs[-1] - xs[0]) / max(h_clean, 1.0)
                # MCC Law 8: outer off-to-leg spacing/height = 0.192/0.7112 = 0.270; adjacent = 0.135
                if (0.21 <= span_ratio <= 0.315) or (len(clean) == 2 and 0.105 <= span_ratio <= 0.165):
                    law8_ok = True
            return True, law8_ok, h_clean, y_base, x_ctr

        ok_n, law8_n, h_n, yb_n, xc_n = _valid_wicket_cluster(near_posts)
        ok_f, law8_f, h_f, yb_f, xc_f = _valid_wicket_cluster(far_posts)
        if not (ok_n and ok_f and (law8_n or law8_f)):
            continue
        if not (1.18 <= (h_n / max(h_f, 1.0)) <= 4.50 and yb_n > yb_f + 10.0):
            continue
        per_frame_sols.append((h_n, yb_n, h_f, yb_f, xc_n, xc_f))
    cap.release()

    # Require multi-frame stationarity across >=3 frames (rejects transient yellow jerseys/foreground flashes!)
    if len(per_frame_sols) < min(3, len(pick_abs)):
        return None

    arr = np.asarray(per_frame_sols, dtype=np.float64)
    hn_med = float(np.median(arr[:, 0]))
    hf_med = float(np.median(arr[:, 2]))
    iqr_n = float(np.percentile(arr[:, 0], 75) - np.percentile(arr[:, 0], 25))
    iqr_f = float(np.percentile(arr[:, 2], 75) - np.percentile(arr[:, 2], 25))
    if (iqr_n / max(hn_med, 1.0)) > 0.08 or (iqr_f / max(hf_med, 1.0)) > 0.08:
        return None

    return solve_two_wicket_calibration(
        h_near_px=hn_med,
        base_near_y_px=float(np.median(arr[:, 1])),
        h_far_px=hf_med,
        base_far_y_px=float(np.median(arr[:, 3])),
        width=w,
        height=h,
        roll_deg=0.0,
        near_x_px=float(np.median(arr[:, 4])),
        far_x_px=float(np.median(arr[:, 5])),
        is_panning_camera=is_panning_camera,
    )


def _estimate_gravity_up_in_cam(
    up_vectors_cam: np.ndarray,
    valid_mask: np.ndarray,
    background_upright_vectors: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, str, str, float]:
    """Estimate world gravity-up (+Y_world) in OpenCV camera frame without silent underground clamping.

    Returns:
        (unit_up_cam, description, gravity_source, gravity_sigma_deg)
    """
    cam_neg_y = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    if background_upright_vectors and len(background_upright_vectors) >= 3:
        arr = np.asarray(background_upright_vectors, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-6)
        arr = np.where(arr[:, 1:2] > 0, -arr, arr)
        g = np.median(arr, axis=0)
        g /= max(float(np.linalg.norm(g)), 1e-6)
        return (
            g,
            f"gravity from {len(background_upright_vectors)} standing upright background pose frames",
            "standing_bg_bodies",
            3.5,
        )

    valid_ups = up_vectors_cam[valid_mask]
    norms = np.linalg.norm(valid_ups, axis=1, keepdims=True)
    unit_ups = valid_ups / np.maximum(norms, 1e-6)
    unit_ups = np.where(unit_ups[:, 1:2] > 0, -unit_ups, unit_ups)

    n_pre = max(3, int(0.45 * len(unit_ups)))
    pre_ups = unit_ups[:n_pre]
    align = pre_ups @ cam_neg_y
    keep_k = max(2, len(pre_ups) // 2)
    top_idx = np.argsort(align)[-keep_k:]
    g = np.median(pre_ups[top_idx], axis=0)
    g /= max(float(np.linalg.norm(g)), 1e-6)
    return (
        g,
        "pre-delivery upright bowler torso gravity (uncalibrated body proxy; carries ~7.5 deg lean uncertainty)",
        "upright_bowler_torso",
        7.5,
    )


def calibrate_from_3d_motion_and_gravity(
    pelvis_traj_cam_m: np.ndarray,
    up_vectors_cam: np.ndarray,
    width: int,
    height: int,
    fov_deg: float = 55.0,
    joints3d_cam_m: Optional[np.ndarray] = None,
    joint_in_frame: Optional[np.ndarray] = None,
    joint_uncertainties_m: Optional[np.ndarray] = None,
    valid_mask: Optional[np.ndarray] = None,
    bowling_arm_is_right: bool = True,
    is_panning_camera: bool = False,
    background_upright_vectors: Optional[List[np.ndarray]] = None,
    K_override: Optional[np.ndarray] = None,
) -> PitchCalibration:
    """Zero-annotation Motion + Anatomy Gravity Self-Calibration from 3D poses.

    Always marks `is_metric_calibrated=False` and propagates honest `heading_sigma_deg` and
    `gravity_sigma_deg` into Stage 4. Keeps `camera_position_world_m = -R.T @ t` strictly consistent (B15).
    """
    finite_mask = np.all(np.isfinite(pelvis_traj_cam_m), axis=1) & np.all(np.isfinite(up_vectors_cam), axis=1)
    if valid_mask is not None and len(valid_mask) == len(finite_mask):
        valid = finite_mask & valid_mask
        if valid.sum() < 5:
            valid = finite_mask
    else:
        valid = finite_mask

    K = K_override.copy() if K_override is not None else default_camera_matrix(width, height, fov_deg)

    if valid.sum() < 5:
        R_id = np.eye(3, dtype=np.float64)
        t_fb = np.array([0.0, 0.0, 10.0], dtype=np.float64)
        return PitchCalibration(
            method="fallback_uncalibrated",
            camera_matrix_K=K.tolist(),
            rotation_matrix_R=R_id.tolist(),
            translation_t=t_fb.tolist(),
            camera_position_world_m=(-R_id.T @ t_fb).tolist(),
            azimuth_deg=90.0,
            elevation_deg=5.0,
            view_class=CameraViewClass.UNKNOWN,
            is_metric_calibrated=False,
            is_intrinsics_calibrated=(K_override is not None),
            is_panning_camera=is_panning_camera,
            heading_sigma_deg=15.0,
            gravity_sigma_deg=10.0,
            gravity_source="fallback",
            calibration_notes=["Insufficient 3D trajectory frames for motion-gravity self-calibration."],
        )

    pts = pelvis_traj_cam_m[valid]
    notes: List[str] = []

    y_world_in_cam, grav_note, grav_src, grav_sig = _estimate_gravity_up_in_cam(
        up_vectors_cam, valid, background_upright_vectors=background_upright_vectors
    )
    notes.append(f"Gravity axis: {grav_note}.")

    n = len(pts)
    k = max(1, n // 5)
    p_start = np.median(pts[:k], axis=0)
    p_end = np.median(pts[-k:], axis=0)
    disp = p_end - p_start
    disp_horiz = disp - np.dot(disp, y_world_in_cam) * y_world_in_cam
    horiz_norm = float(np.linalg.norm(disp_horiz))

    stride_axis_cam: Optional[np.ndarray] = None
    stride_spread_m = 0.0
    chest_forward_cam: Optional[np.ndarray] = None

    if joints3d_cam_m is not None and joints3d_cam_m.shape[0] == len(finite_mask) and joints3d_cam_m.shape[1] >= 18:
        f_idx = 7 if bowling_arm_is_right else 8
        r_idx = 8 if bowling_arm_is_right else 7

        lsho = joints3d_cam_m[valid, 16]
        rsho = joints3d_cam_m[valid, 17]
        lr_sh = rsho - lsho
        chest_vecs = np.cross(np.broadcast_to(y_world_in_cam, lr_sh.shape), lr_sh)
        chest_h = chest_vecs - np.outer(chest_vecs @ y_world_in_cam, y_world_in_cam)
        late_chest = np.median(chest_h[len(chest_h) // 2 :], axis=0)
        if float(np.linalg.norm(late_chest)) > 0.08:
            chest_forward_cam = late_chest / float(np.linalg.norm(late_chest))

        ankle_ok = valid.copy()
        if joint_in_frame is not None and joint_in_frame.shape[0] == len(finite_mask):
            ankle_ok &= joint_in_frame[:, f_idx] & joint_in_frame[:, r_idx]
        if joint_uncertainties_m is not None and joint_uncertainties_m.shape[0] == len(finite_mask):
            ankle_ok &= (joint_uncertainties_m[:, f_idx] < 0.14) & (joint_uncertainties_m[:, r_idx] < 0.14)

        if int(ankle_ok.sum()) >= 3:
            f_ank = joints3d_cam_m[ankle_ok, f_idx]
            r_ank = joints3d_cam_m[ankle_ok, r_idx]
            foot_diff = f_ank - r_ank
            foot_horiz = foot_diff - np.outer(foot_diff @ y_world_in_cam, y_world_in_cam)
            spreads = np.linalg.norm(foot_horiz, axis=1)
            top_k = max(1, min(5, len(spreads) // 8))
            top_idxs = np.argsort(spreads)[-top_k:]
            stride_spread_m = float(np.median(spreads[top_idxs]))
            if stride_spread_m >= 0.45:
                cand = np.median(foot_horiz[top_idxs], axis=0)
                cand_norm = float(np.linalg.norm(cand))
                if cand_norm > 0.25:
                    cand_unit = cand / cand_norm
                    if chest_forward_cam is not None and float(np.dot(cand_unit, chest_forward_cam)) < -0.25:
                        cand_unit = -cand_unit
                    stride_axis_cam = cand_unit

    disp_unit = (disp_horiz / horiz_norm) if horiz_norm > 1e-4 else None
    pan_inferred = bool(is_panning_camera)
    if (
        not pan_inferred
        and stride_axis_cam is not None
        and disp_unit is not None
        and horiz_norm < 2.2
        and abs(float(np.dot(disp_unit, stride_axis_cam))) < 0.55
    ):
        pan_inferred = True
        notes.append(
            f"Geometric pan override: pelvis camera displacement ({horiz_norm:.2f} m) is nearly orthogonal "
            f"to in-frame anatomical stride (spread={stride_spread_m:.2f} m); locking +X to stride axis."
        )

    heading_sig = 7.5
    if pan_inferred and stride_axis_cam is not None:
        x_world_in_cam = stride_axis_cam
        heading_sig = 9.0
        notes.append(
            f"Panning camera detected: calibrated +X delivery axis from in-frame anatomical stride "
            f"(spread={stride_spread_m:.2f} m)."
        )
    elif (not pan_inferred) and horiz_norm >= 0.90:
        x_world_in_cam = disp_horiz / horiz_norm
        if stride_axis_cam is not None and float(np.dot(x_world_in_cam, stride_axis_cam)) > 0.55:
            x_world_in_cam = 0.72 * x_world_in_cam + 0.28 * stride_axis_cam
            x_world_in_cam /= max(float(np.linalg.norm(x_world_in_cam)), 1e-6)
        heading_sig = 6.5
        notes.append(f"Static camera: calibrated +X delivery axis from {horiz_norm:.2f} m 3D pelvis travel.")
    elif stride_axis_cam is not None:
        x_world_in_cam = stride_axis_cam
        heading_sig = 8.5
        notes.append(
            f"Calibrated +X delivery axis from in-frame anatomical stride (spread={stride_spread_m:.2f} m; "
            f"pelvis travel={horiz_norm:.2f} m)."
        )
    elif horiz_norm >= 0.25:
        x_world_in_cam = disp_horiz / horiz_norm
        heading_sig = 9.5
        notes.append(f"Calibrated +X delivery axis from {horiz_norm:.2f} m 3D pelvis travel.")
    else:
        centered = pts - np.mean(pts, axis=0)
        centered_h = centered - np.outer(centered @ y_world_in_cam, y_world_in_cam)
        _, _, vt = np.linalg.svd(centered_h, full_matrices=False)
        x_world_in_cam = vt[0]
        ref_dir = chest_forward_cam if chest_forward_cam is not None else disp_horiz
        if float(np.linalg.norm(ref_dir)) > 1e-4 and float(np.dot(x_world_in_cam, ref_dir)) < 0.0:
            x_world_in_cam = -x_world_in_cam
        heading_sig = 12.0
        notes.append("Small pelvis displacement and occluded ankles: estimated +X via sign-disambiguated PCA.")

    is_panning_camera = pan_inferred

    # When gravity was inferred only from the running bowler's own torso (`upright_bowler_torso`),
    # a 10-18 deg forward sprint lean biases y_world_in_cam along +X_pitch. Because the cricket
    # pitch is flat (pelvis travel during early run-up is horizontal), removing the torso's
    # sagittal projection onto the early horizontal run-up vector cancels forward sprint lean.
    if grav_src == "upright_bowler_torso" and not is_panning_camera and n >= 10:
        end_early = max(4, int(0.42 * n))
        k_early = max(1, end_early // 4)
        disp_early = np.median(pts[end_early - k_early : end_early], axis=0) - np.median(pts[:k_early], axis=0)
        early_norm = float(np.linalg.norm(disp_early))
        if early_norm < 0.80:
            half_n = max(4, n // 2)
            k_h = max(1, half_n // 4)
            disp_early = np.median(pts[half_n - k_h : half_n], axis=0) - np.median(pts[:k_h], axis=0)
            early_norm = float(np.linalg.norm(disp_early))
        if early_norm >= 0.80:
            x_flat = disp_early / early_norm
            y_proj = y_world_in_cam - float(np.dot(y_world_in_cam, x_flat)) * x_flat
            y_proj_norm = float(np.linalg.norm(y_proj))
            if y_proj_norm > 0.70:
                y_cand = y_proj / y_proj_norm
                lean_corr_deg = math.degrees(
                    math.acos(float(np.clip(np.dot(y_cand, y_world_in_cam), -1.0, 1.0)))
                )
                if lean_corr_deg <= 22.0 and y_cand[1] < -0.5:
                    y_world_in_cam = y_cand
                    grav_src = "bowler_torso_deleaned"
                    grav_sig = 4.5
                    notes.append(
                        f"Sagittal sprint-lean correction: orthogonalized torso gravity against "
                        f"{early_norm:.2f} m early run-up plane ({lean_corr_deg:.1f} deg forward lean removed)."
                    )

    z_world_in_cam = np.cross(x_world_in_cam, y_world_in_cam)
    z_norm = float(np.linalg.norm(z_world_in_cam))
    if z_norm < 1e-6:
        z_world_in_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        z_world_in_cam = z_world_in_cam / z_norm
    x_world_in_cam = np.cross(y_world_in_cam, z_world_in_cam)
    x_world_in_cam /= max(float(np.linalg.norm(x_world_in_cam)), 1e-6)

    R = np.column_stack([x_world_in_cam, y_world_in_cam, z_world_in_cam])

    mid_pelvis_cam = np.median(pts, axis=0)
    origin_cam = mid_pelvis_cam - 0.95 * y_world_in_cam
    tvec = origin_cam
    # Strict consistency: cam_pos_world ALWAYS equals -R.T @ t (no silent desyncing clamp, EXP-007 B15)
    cam_pos_world = (-R.T @ tvec).reshape(3)

    if cam_pos_world[1] < 0.0:
        notes.append(
            f"Warning: body-estimated gravity placed camera height at {cam_pos_world[1]:.2f} m "
            "(likely pre-delivery sprint lean); supply pitch landmarks/wickets for calibrated gravity."
        )
        grav_sig = max(grav_sig, 9.0)

    az_deg, el_deg = _angles_from_camera_position(cam_pos_world, np.array([0.0, 0.95, 0.0], dtype=np.float64))
    dist_m = max(float(np.linalg.norm(mid_pelvis_cam)), 1.0)
    px_per_m = float(K[0, 0] / dist_m)
    view_class = classify_camera_view(az_deg, el_deg)

    return PitchCalibration(
        method="motion_gravity_self_cal",
        camera_matrix_K=K.tolist(),
        rotation_matrix_R=R.tolist(),
        translation_t=tvec.tolist(),
        camera_position_world_m=cam_pos_world.tolist(),
        azimuth_deg=round(az_deg, 2),
        elevation_deg=round(el_deg, 2),
        px_per_meter_at_crease=round(px_per_m, 2),
        reprojection_error_px=None,
        view_class=view_class,
        is_metric_calibrated=False,
        is_intrinsics_calibrated=(K_override is not None),
        is_panning_camera=is_panning_camera,
        heading_sigma_deg=heading_sig,
        gravity_sigma_deg=grav_sig,
        gravity_source=grav_src,
        calibration_notes=notes,
    )


def transform_cam_to_pitch_world(
    points_cam_m: np.ndarray,
    calibration: PitchCalibration,
) -> np.ndarray:
    """Transform (..., 3) points from OpenCV camera coordinates (m) to canonical Pitch World Frame (m)."""
    R = np.asarray(calibration.rotation_matrix_R, dtype=np.float64)
    t = np.asarray(calibration.translation_t, dtype=np.float64)
    pts = np.asarray(points_cam_m, dtype=np.float64)
    return (pts - t) @ R
