"""Stage 1: Cricket Pitch Geometry Self-Calibration & Camera View Estimation.

Exploits universal MCC Law 7 cricket pitch markings (1.22 m popping-to-bowling crease,
2.64 m return crease width, 0.711 m stump height, 20.12 m pitch length) via cheirality-checked
coplanar IPPE / EPNP solving to recover camera intrinsics/extrinsics (K, R, t), metric scale,
and camera azimuth/elevation.
Falls back cleanly to Panning-Safe, Upright-Gravity Self-Calibration when pitch markings are cropped.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .schemas import CameraViewClass, PitchCalibration

# Canonical Pitch World Frame (meters):
#   Origin (0, 0, 0): Base of middle stump on the bowler's bowling crease
#   +X: Down the pitch toward the striker's stumps
#   +Y: Vertical gravity-up
#   +Z: Across the pitch (+X x +Y = bowler's right when facing +X)
MCC_LAW7_LANDMARKS_M: Dict[str, Tuple[float, float, float]] = {
    "stump_middle_base": (0.0, 0.0, 0.0),
    "stump_middle_top": (0.0, 0.711, 0.0),
    "stump_off_base": (0.0, 0.0, -0.1143),
    "stump_off_top": (0.0, 0.711, -0.1143),
    "stump_leg_base": (0.0, 0.0, 0.1143),
    "stump_leg_top": (0.0, 0.711, 0.1143),
    "bowling_crease_off": (0.0, 0.0, -1.32),
    "bowling_crease_leg": (0.0, 0.0, 1.32),
    "popping_crease_off": (1.22, 0.0, -1.32),
    "popping_crease_leg": (1.22, 0.0, 1.32),
    "popping_crease_center": (1.22, 0.0, 0.0),
    "return_crease_back_off": (-1.22, 0.0, -1.32),
    "return_crease_back_leg": (-1.22, 0.0, 1.32),
    "far_popping_crease_off": (18.90, 0.0, -1.32),
    "far_popping_crease_leg": (18.90, 0.0, 1.32),
    "far_stump_middle_base": (20.12, 0.0, 0.0),
    "far_stump_middle_top": (20.12, 0.711, 0.0),
}


def default_camera_matrix(width: int, height: int, fov_deg: float = 55.0) -> np.ndarray:
    """Estimate pinhole intrinsic matrix K from image dimensions and horizontal FOV."""
    focal = 0.5 * width / math.tan(math.radians(fov_deg) / 2.0)
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


def calibrate_from_mcc_landmarks(
    image_landmarks_px: Dict[str, Tuple[float, float]],
    width: int,
    height: int,
    K: Optional[np.ndarray] = None,
    fov_search_deg: Tuple[float, ...] = (12.0, 20.0, 30.0, 42.0, 55.0, 68.0),
) -> PitchCalibration:
    """Solve PnP camera extrinsics from >=4 named MCC Law 7 pitch landmarks.

    Handles both general 3D (creases + stump tops) and coplanar (Y=0 creases only) configurations
    via solvePnPGeneric(SOLVEPNP_IPPE) with strict cheirality (Z_cam > 0) and above-ground checks.
    """
    valid_keys = [k for k in image_landmarks_px if k in MCC_LAW7_LANDMARKS_M]
    if len(valid_keys) < 4:
        raise ValueError(
            f"At least 4 valid MCC Law 7 pitch landmarks required for PnP; got {len(valid_keys)}."
        )

    obj_pts = np.array([MCC_LAW7_LANDMARKS_M[k] for k in valid_keys], dtype=np.float64)
    img_pts = np.array([image_landmarks_px[k] for k in valid_keys], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    candidate_Ks = [K] if K is not None else [default_camera_matrix(width, height, f) for f in fov_search_deg]

    best_err = float("inf")
    best_solution = None
    is_coplanar = bool(np.ptp(obj_pts[:, 1]) < 1e-3)

    for cand_K in candidate_Ks:
        raw_solutions: List[Tuple[np.ndarray, np.ndarray]] = []
        if is_coplanar:
            # IPPE works for any N >= 4 coplanar points and returns BOTH planar ambiguity branches!
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
            # Cheirality check: every landmark must lie strictly in front of camera (Z_cam > 0.2 m)
            pts_cam = (R @ obj_pts.T).T + tvec.reshape(1, 3)
            if np.any(pts_cam[:, 2] <= 0.2):
                continue
            cam_pos = (-R.T @ tvec).reshape(3)
            # Physical check: camera must be above the cricket pitch ground plane
            if cam_pos[1] < 0.15:
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
        f"Calibrated via MCC Law 7 PnP ({'coplanar IPPE' if is_coplanar else '3D EPNP+LM'}) "
        f"using {len(valid_keys)} pitch landmarks (mean reproj err={best_err:.2f} px)."
    ]
    if best_err > 12.0:
        notes.append("Warning: high reprojection error (>12 px); check landmark accuracy or lens zoom.")

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
        is_metric_calibrated=True,
        is_panning_camera=False,
        calibration_notes=notes,
    )


def _estimate_gravity_up_in_cam(
    up_vectors_cam: np.ndarray,
    valid_mask: np.ndarray,
    background_upright_vectors: Optional[List[np.ndarray]] = None,
) -> Tuple[np.ndarray, str]:
    """Estimate world gravity-up (+Y_world) in OpenCV camera frame without forward-lean corruption.

    Priority:
      1. Standing upright background tracks (e.g. umpires/non-strikers verified knee>155 & hip-trunk>155)
      2. Pre-delivery upright bowler frames (first 30% & most vertically aligned torso frames, excluding BFC..FT lean)
    """
    cam_neg_y = np.array([0.0, -1.0, 0.0], dtype=np.float64)

    if background_upright_vectors and len(background_upright_vectors) >= 3:
        arr = np.asarray(background_upright_vectors, dtype=np.float64)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.maximum(norms, 1e-6)
        arr = np.where(arr[:, 1:2] > 0, -arr, arr)
        g = np.median(arr, axis=0)
        g /= max(float(np.linalg.norm(g)), 1e-6)
        return g, f"gravity from {len(background_upright_vectors)} standing upright background pose frames"

    valid_ups = up_vectors_cam[valid_mask]
    norms = np.linalg.norm(valid_ups, axis=1, keepdims=True)
    unit_ups = valid_ups / np.maximum(norms, 1e-6)
    unit_ups = np.where(unit_ups[:, 1:2] > 0, -unit_ups, unit_ups)

    # Restrict to pre-delivery upright portion (first 40% of clip, then top 50% closest to -Y_cam)
    n_pre = max(3, int(0.45 * len(unit_ups)))
    pre_ups = unit_ups[:n_pre]
    # Score alignment with camera vertical -Y_cam (rejects mid-delivery 45-deg forward flexion)
    align = pre_ups @ cam_neg_y
    keep_k = max(2, len(pre_ups) // 2)
    top_idx = np.argsort(align)[-keep_k:]
    g = np.median(pre_ups[top_idx], axis=0)
    g /= max(float(np.linalg.norm(g)), 1e-6)

    # Guard against forward bowler running lean tilting gravity into +Z_cam (which places camera underground!):
    # In OpenCV camera coords (+Z into scene, +Y down), an elevated camera looking down has gravity_z <= 0;
    # gravity_z > 0.05 means looking UP from underground (physically impossible on a cricket pitch).
    if g[2] > 0.03:
        g[2] = 0.0
        g /= max(float(np.linalg.norm(g)), 1e-6)
        return g, "pre-delivery upright torso gravity (clamped underground pitch bias)"
    return g, "pre-delivery upright bowler torso gravity"


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
) -> PitchCalibration:
    """Zero-annotation Panning-Safe Motion + Anatomy Gravity Self-Calibration from metric 3D poses.

    Recovers canonical Pitch World Frame (+X along delivery travel direction, +Y gravity-up, +Z cross)
    and camera azimuth/elevation. Marks `is_metric_calibrated=False` (only MCC Law 7 PnP is truly
    metric-calibrated) and records `is_panning_camera` so Stage 4 suppresses panning velocity illusions.
    """
    finite_mask = np.all(np.isfinite(pelvis_traj_cam_m), axis=1) & np.all(np.isfinite(up_vectors_cam), axis=1)
    if valid_mask is not None and len(valid_mask) == len(finite_mask):
        valid = finite_mask & valid_mask
        if valid.sum() < 5:
            valid = finite_mask
    else:
        valid = finite_mask

    if valid.sum() < 5:
        K = default_camera_matrix(width, height, fov_deg)
        return PitchCalibration(
            method="fallback_uncalibrated",
            camera_matrix_K=K.tolist(),
            rotation_matrix_R=np.eye(3).tolist(),
            translation_t=[0.0, 0.0, 10.0],
            camera_position_world_m=[0.0, 1.5, 10.0],
            azimuth_deg=90.0,
            elevation_deg=5.0,
            view_class=CameraViewClass.UNKNOWN,
            is_metric_calibrated=False,
            is_panning_camera=is_panning_camera,
            calibration_notes=["Insufficient 3D trajectory frames for motion-gravity self-calibration."],
        )

    pts = pelvis_traj_cam_m[valid]
    notes: List[str] = []

    # 1. Gravity-up axis (+Y_world) in camera frame
    y_world_in_cam, grav_note = _estimate_gravity_up_in_cam(
        up_vectors_cam, valid, background_upright_vectors=background_upright_vectors
    )
    notes.append(f"Gravity axis: {grav_note}.")

    # 2. Global pelvis horizontal displacement vector in camera coordinates
    n = len(pts)
    k = max(1, n // 5)
    p_start = np.median(pts[:k], axis=0)
    p_end = np.median(pts[-k:], axis=0)
    disp = p_end - p_start
    disp_horiz = disp - np.dot(disp, y_world_in_cam) * y_world_in_cam
    horiz_norm = float(np.linalg.norm(disp_horiz))

    # 3. Panning-invariant anatomical delivery-stride vector (strictly gated on in-frame, low-uncertainty ankles!)
    stride_axis_cam: Optional[np.ndarray] = None
    stride_spread_m = 0.0
    chest_forward_cam: Optional[np.ndarray] = None

    if joints3d_cam_m is not None and joints3d_cam_m.shape[0] == len(finite_mask) and joints3d_cam_m.shape[1] >= 18:
        # SMPL-24: lank=7, rank=8, lsho=16, rsho=17
        f_idx = 7 if bowling_arm_is_right else 8
        r_idx = 8 if bowling_arm_is_right else 7

        # Estimate chest-forward normal near late delivery (rsho - lsho crossed with +Y_world?
        # Check right-hand rule: +X = cross(+Y_world, +Z_world), where front-on rsho-lsho points along +Z_world!)
        lsho = joints3d_cam_m[valid, 16]
        rsho = joints3d_cam_m[valid, 17]
        lr_sh = rsho - lsho
        # Cross(+Y_world, lr_sh) points forward out of the chest in Pitch World!
        chest_vecs = np.cross(np.broadcast_to(y_world_in_cam, lr_sh.shape), lr_sh)
        # Project horizontal & average over second half (near FFC/BR when chest turns toward batter)
        chest_h = chest_vecs - np.outer(chest_vecs @ y_world_in_cam, y_world_in_cam)
        late_chest = np.median(chest_h[len(chest_h) // 2 :], axis=0)
        if float(np.linalg.norm(late_chest)) > 0.08:
            chest_forward_cam = late_chest / float(np.linalg.norm(late_chest))

        # Gate ankle stride frames strictly by valid + in-frame + reasonable uncertainty
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
                    # Disambiguate sign against chest-forward normal (rejects follow-through rear-leg crossover!)
                    if chest_forward_cam is not None and float(np.dot(cand_unit, chest_forward_cam)) < -0.25:
                        cand_unit = -cand_unit
                    stride_axis_cam = cand_unit

    # 4. Select +X_world axis based on panning status and geometric consistency between
    #    pelvis translation and in-frame anatomical stride vector!
    disp_unit = (disp_horiz / horiz_norm) if horiz_norm > 1e-4 else None
    pan_inferred = bool(is_panning_camera)
    if (
        not pan_inferred
        and stride_axis_cam is not None
        and disp_unit is not None
        and horiz_norm < 2.2
        and abs(float(np.dot(disp_unit, stride_axis_cam))) < 0.55
    ):
        # Classic telephoto side-on pan: camera pan cancels lateral X translation, leaving ~1 m of
        # noisy monocular depth drift orthogonal to the true in-frame anatomical stride vector!
        pan_inferred = True
        notes.append(
            f"Geometric pan override: pelvis camera displacement ({horiz_norm:.2f} m) is nearly orthogonal "
            f"to in-frame anatomical stride (spread={stride_spread_m:.2f} m); locking +X to stride axis."
        )

    if pan_inferred and stride_axis_cam is not None:
        x_world_in_cam = stride_axis_cam
        notes.append(
            f"Panning camera detected: calibrated +X delivery axis from in-frame anatomical stride "
            f"(spread={stride_spread_m:.2f} m)."
        )
    elif (not pan_inferred) and horiz_norm >= 0.90:
        x_world_in_cam = disp_horiz / horiz_norm
        if stride_axis_cam is not None and float(np.dot(x_world_in_cam, stride_axis_cam)) > 0.55:
            x_world_in_cam = 0.72 * x_world_in_cam + 0.28 * stride_axis_cam
            x_world_in_cam /= max(float(np.linalg.norm(x_world_in_cam)), 1e-6)
        notes.append(
            f"Static camera: calibrated +X delivery axis from {horiz_norm:.2f} m 3D pelvis travel."
        )
    elif stride_axis_cam is not None:
        x_world_in_cam = stride_axis_cam
        notes.append(
            f"Calibrated +X delivery axis from in-frame anatomical stride (spread={stride_spread_m:.2f} m; "
            f"pelvis travel={horiz_norm:.2f} m)."
        )
    elif horiz_norm >= 0.25:
        x_world_in_cam = disp_horiz / horiz_norm
        notes.append(f"Calibrated +X delivery axis from {horiz_norm:.2f} m 3D pelvis travel.")
    else:
        centered = pts - np.mean(pts, axis=0)
        centered_h = centered - np.outer(centered @ y_world_in_cam, y_world_in_cam)
        _, _, vt = np.linalg.svd(centered_h, full_matrices=False)
        x_world_in_cam = vt[0]
        ref_dir = chest_forward_cam if chest_forward_cam is not None else disp_horiz
        if float(np.linalg.norm(ref_dir)) > 1e-4 and float(np.dot(x_world_in_cam, ref_dir)) < 0.0:
            x_world_in_cam = -x_world_in_cam
        notes.append("Small pelvis displacement and occluded ankles: estimated +X via sign-disambiguated PCA.")

    is_panning_camera = pan_inferred

    # 5. Right-handed canonical basis: X_world, Y_world, Z_world = X x Y
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
    cam_pos_world = (-R.T @ tvec).reshape(3)

    # Final physical elevation guard: if camera ended up underground (<0.20 m), clamp camera Y to ground level
    if cam_pos_world[1] < 0.20:
        cam_pos_world[1] = max(0.20, cam_pos_world[1])

    az_deg, el_deg = _angles_from_camera_position(cam_pos_world, np.array([0.0, 0.95, 0.0], dtype=np.float64))
    el_deg = max(el_deg, -1.0)
    K = default_camera_matrix(width, height, fov_deg)
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
        is_panning_camera=is_panning_camera,
        calibration_notes=notes,
    )


def transform_cam_to_pitch_world(
    points_cam_m: np.ndarray,
    calibration: PitchCalibration,
) -> np.ndarray:
    """Transform (..., 3) 3D points from camera metric frame into canonical Pitch World Frame."""
    R = np.asarray(calibration.rotation_matrix_R, dtype=np.float64)
    t = np.asarray(calibration.translation_t, dtype=np.float64)
    orig_shape = points_cam_m.shape
    pts = points_cam_m.reshape(-1, 3)
    world = (pts - t) @ R
    return world.reshape(orig_shape)
