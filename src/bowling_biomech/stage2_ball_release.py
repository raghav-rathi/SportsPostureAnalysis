"""Phase 2 (M2D): Calibrated Side-On Wrist-ROI Optical Ball Release Speed Estimator.

Zero-training ballistic physics module:
  - Hard-gated strictly to:
      1. `cal.is_metric_calibrated == True` and static camera (`not cal.is_panning_camera`)
      2. `audit.fps >= 115.0` (120–240 fps slow-motion) AND `not audit.timebase_unreliable`
      3. Complete overhead delivery (`events.is_complete_delivery == True`)
      4. Near-side-on view (`|sin(azimuth)| >= 0.70`) — automatically suppressed on rear/front-on or
         30 fps clips and whenever propagated azimuth/depth uncertainty exceeds the reliability cap.
  - Asymmetric down-pitch flight-corridor ROI projected from `wri_world_br` to `wri_world_br + 3.2 m (+X)`
    so fast deliveries (`130-155 km/h` at 120 fps) remain inside the crop for 8-12 frames regardless of
    left-to-right vs right-to-left bowling direction.
  - Masks the tracked bowling hand & forearm capsule on every frame so follow-through arm motion cannot
    contaminate RANSAC ball tracks.
  - Propagates three uncertainty sources in quadrature:
      1. OLS ballistic velocity slope SE (`sqrt(var_v)`)
      2. Monocular wrist release-plane depth uncertainty (`sigma_z_plane_m / cam_dist_m`)
      3. Cross-pitch azimuth parallax uncertainty (`|cot(az)| * tan(3.5 deg)` for off-stump/leg-stump line)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .constants import JOINT_IDX
from .schemas import BowlingArm, DeliveryEvents, MetricEstimate, ObservabilityGrade, PitchCalibration, VideoAudit

_GRAVITY_MPS2 = 9.81


@dataclass
class OpticalBallSpeedResult:
    """Result of calibrated wrist-ROI ballistic release speed estimation."""

    metric: MetricEstimate
    optical_br_frame: Optional[float] = None
    inlier_points_world_m: List[Tuple[float, float, float, float]] = field(default_factory=list)  # (t_s, x, y, z)
    ransac_inliers: int = 0
    fit_residual_m: Optional[float] = None


def fit_ballistic_speed_ransac(
    times_s: np.ndarray,
    pts_world_xy: np.ndarray,
    inlier_thresh_m: float = 0.08,
    min_inliers: int = 4,
    depth_rel_sigma: float = 0.025,
    azimuth_rel_sigma: float = 0.0,
) -> Tuple[Optional[float], Optional[float], int, Optional[float], np.ndarray]:
    """Fit gravity-compensated ballistic trajectory `(x(t), y(t))` via exhaustive-pair RANSAC.

    In Pitch World (+X down pitch, +Y up):
      x(t) = x0 + vx * dt
      y(t) + 0.5 * g * dt^2 = y0 + vy * dt
    Returns `(speed_kph, sigma_kph, n_inliers, rmse_m, inlier_mask)`.
    """
    n = len(times_s)
    if n < min_inliers or pts_world_xy.shape[0] != n:
        return None, None, 0, None, np.zeros(n, dtype=bool)

    t0 = float(times_s[0])
    dt = np.asarray(times_s, dtype=np.float64) - t0
    x = np.asarray(pts_world_xy[:, 0], dtype=np.float64)
    y_corr = np.asarray(pts_world_xy[:, 1], dtype=np.float64) + 0.5 * _GRAVITY_MPS2 * (dt**2)

    best_mask = np.zeros(n, dtype=bool)
    best_cnt = 0
    best_res = 999.0

    for i in range(n):
        for j in range(i + 1, n):
            ddt = dt[j] - dt[i]
            if ddt < 0.004:
                continue
            vx = (x[j] - x[i]) / ddt
            vy = (y_corr[j] - y_corr[i]) / ddt
            spd = math.hypot(vx, vy)
            # Physical cricket delivery speed bounds: 43 km/h (12 m/s) to 180 km/h (50 m/s)
            if vx < 10.0 or spd < 12.0 or spd > 50.0:
                continue
            x_pred = x[i] + vx * (dt - dt[i])
            y_pred = y_corr[i] + vy * (dt - dt[i])
            err = np.hypot(x - x_pred, y_corr - y_pred)
            inl = err <= inlier_thresh_m
            cnt = int(inl.sum())
            if cnt > best_cnt or (cnt == best_cnt and cnt > 0 and float(np.median(err[inl])) < best_res):
                best_cnt = cnt
                best_mask = inl
                best_res = float(np.median(err[inl])) if cnt > 0 else 999.0

    if best_cnt < min_inliers:
        return None, None, best_cnt, None, best_mask

    # Final OLS refit on inliers + analytical standard error of velocity slope
    dt_i = dt[best_mask]
    x_i = x[best_mask]
    y_i = y_corr[best_mask]
    dt_mean = float(np.mean(dt_i))
    s_tt = float(np.sum((dt_i - dt_mean) ** 2))
    if s_tt < 1e-7:
        return None, None, best_cnt, None, best_mask

    vx_hat = float(np.sum((dt_i - dt_mean) * (x_i - np.mean(x_i))) / s_tt)
    vy_hat = float(np.sum((dt_i - dt_mean) * (y_i - np.mean(y_i))) / s_tt)
    x0_hat = float(np.mean(x_i) - vx_hat * dt_mean)
    y0_hat = float(np.mean(y_i) - vy_hat * dt_mean)

    res_x = x_i - (x0_hat + vx_hat * dt_i)
    res_y = y_i - (y0_hat + vy_hat * dt_i)
    dof = max(best_cnt - 2, 1)
    s2_x = float(np.sum(res_x**2) / dof)
    s2_y = float(np.sum(res_y**2) / dof)
    rmse_m = math.sqrt(0.5 * (float(np.mean(res_x**2)) + float(np.mean(res_y**2))))

    speed_mps = math.hypot(vx_hat, vy_hat)
    if speed_mps < 12.0 or speed_mps > 50.0:
        return None, None, best_cnt, rmse_m, best_mask

    # Combine OLS slope SE + monocular depth-plane uncertainty + cross-pitch azimuth parallax uncertainty
    var_vx = s2_x / s_tt
    var_vy = s2_y / s_tt
    var_v = ((vx_hat**2) * var_vx + (vy_hat**2) * var_vy) / max(speed_mps**2, 1e-6)
    rel_sys = math.hypot(max(float(depth_rel_sigma), 0.02), max(float(azimuth_rel_sigma), 0.0))
    sig_mps = math.hypot(math.sqrt(max(var_v, 0.0)), rel_sys * speed_mps)
    return speed_mps * 3.6, sig_mps * 3.6, best_cnt, rmse_m, best_mask


def _ray_to_release_plane_world(
    uv: Tuple[float, float],
    K_inv: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    z_plane_world: float,
) -> Optional[np.ndarray]:
    """Intersect pixel ray `uv` with the vertical sagittal delivery plane `Z_world = z_plane_world`."""
    ray_cam = K_inv @ np.array([uv[0], uv[1], 1.0], dtype=np.float64)
    ray_world = R.T @ ray_cam
    cam_pos = (-R.T @ t).reshape(3)
    if abs(ray_world[2]) < 1e-6:
        return None
    lam = (z_plane_world - cam_pos[2]) / ray_world[2]
    if lam <= 0.1:
        return None
    return cam_pos + lam * ray_world


def _project_world_point(pt_w: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray) -> Optional[Tuple[float, float]]:
    p_c = R @ np.asarray(pt_w, dtype=np.float64).reshape(3) + t.reshape(3)
    if p_c[2] <= 0.2:
        return None
    uvw = K @ p_c
    return float(uvw[0] / uvw[2]), float(uvw[1] / uvw[2])


def _suppressed_speed_estimate(reason: str) -> OpticalBallSpeedResult:
    return OpticalBallSpeedResult(
        metric=MetricEstimate(
            key="ball_release_speed_kph",
            label="Optical Ball Release Speed (Wrist-ROI)",
            unit="km/h",
            value=None,
            ci_low=None,
            ci_high=None,
            interval_type="analytical_uncertainty",
            observability=ObservabilityGrade.UNMEASURABLE,
            suppressed=True,
            suppression_reason=reason,
            definition_note=(
                "Gravity-compensated ballistic speed fitted to post-release wrist-ROI ball centroids "
                "projected onto the calibrated sagittal delivery plane."
            ),
            empirical_reference="Elite male fast bowlers: ~130-150 km/h (36-42 m/s).",
            literature_source="Worthington et al. (2013)",
        )
    )


def estimate_ball_release_speed_from_world_points(
    times_s: np.ndarray,
    pts_world_xy: np.ndarray,
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    wrist_depth_sigma_m: Optional[float] = None,
) -> OpticalBallSpeedResult:
    """Gate and fit optical ballistic speed directly from calibrated 2D/3D release-plane ball tracks."""
    # Non-negotiable hard gates (M2D Gate 2D)
    if not events.is_complete_delivery:
        return _suppressed_speed_estimate("Suppressed: clip did not pass overhead ball release sanity gates.")
    if audit.timebase_unreliable:
        return _suppressed_speed_estimate(
            f"Suppressed: unreliable video timebase ({audit.timebase_status}); ball speed requires verified >=120 fps."
        )
    if audit.fps < 115.0:
        return _suppressed_speed_estimate(
            f"Suppressed: effective frame rate ({audit.fps:.1f} fps) is below the 120 fps minimum for optical ball tracking."
        )
    if not cal.is_metric_calibrated or cal.is_panning_camera:
        return _suppressed_speed_estimate(
            "Suppressed: optical release speed requires a static camera with calibrated MCC pitch/wicket extrinsics."
        )
    az_rad = math.radians(cal.azimuth_deg)
    sin_az = abs(math.sin(az_rad))
    cos_az = abs(math.cos(az_rad))
    if sin_az < 0.70:
        return _suppressed_speed_estimate(
            f"Suppressed: camera azimuth ({cal.azimuth_deg:.1f} deg, |sin(az)|={sin_az:.2f} < 0.70) is too rear/front-on "
            "for parallax-based sagittal ball speed estimation; use a side-on (60-120 deg) tripod camera."
        )

    # Propagate monocular release-plane depth error (sigma_z / cam_dist) and cross-pitch angle bias |cot(az)| * tan(3.5 deg)
    cam_pos = np.asarray(cal.camera_position_world_m, dtype=np.float64)
    cam_dist = max(float(np.linalg.norm(cam_pos)), 3.0)
    # Default 0.22 m wrist depth sigma on monocular view unless exact wrist uncertainty is provided
    sig_z = float(wrist_depth_sigma_m) if wrist_depth_sigma_m is not None else 0.22
    depth_rel_sigma = max(0.02, sig_z / cam_dist)
    cot_az = cos_az / max(sin_az, 1e-4)
    azimuth_rel_sigma = cot_az * math.tan(math.radians(3.5))

    speed_kph, sig_kph, n_inl, rmse_m, mask = fit_ballistic_speed_ransac(
        times_s,
        pts_world_xy,
        depth_rel_sigma=depth_rel_sigma,
        azimuth_rel_sigma=azimuth_rel_sigma,
    )
    if speed_kph is None or sig_kph is None or sig_kph > 8.5:
        return _suppressed_speed_estimate(
            f"Suppressed: insufficient ballistic ball track inliers ({n_inl}) or high propagated uncertainty "
            f"(sigma={f'{sig_kph:.1f}' if sig_kph is not None else 'N/A'} km/h > 8.5 km/h; "
            f"depth_rel={depth_rel_sigma*100:.1f}%, az_rel={azimuth_rel_sigma*100:.1f}%)."
        )

    ci_lo = max(0.0, speed_kph - 1.96 * sig_kph)
    ci_hi = speed_kph + 1.96 * sig_kph
    obs = ObservabilityGrade.HIGH if sig_kph <= 3.5 else (ObservabilityGrade.MODERATE if sig_kph <= 6.0 else ObservabilityGrade.LOW)
    inlier_pts = [
        (round(float(times_s[i]), 4), round(float(pts_world_xy[i, 0]), 3), round(float(pts_world_xy[i, 1]), 3), 0.0)
        for i in range(len(times_s))
        if mask[i]
    ]
    return OpticalBallSpeedResult(
        metric=MetricEstimate(
            key="ball_release_speed_kph",
            label="Optical Ball Release Speed (Wrist-ROI)",
            unit="km/h",
            value=round(speed_kph, 1),
            ci_low=round(ci_lo, 1),
            ci_high=round(ci_hi, 1),
            interval_type="analytical_uncertainty",
            sigma_total=round(sig_kph, 2),
            observability=obs,
            suppressed=False,
            definition_note=(
                "Gravity-compensated ballistic speed fitted via RANSAC to post-release down-pitch wrist-ROI ball "
                f"centroids (propagating OLS slope SE, depth_rel={depth_rel_sigma*100:.1f}%, az_rel={azimuth_rel_sigma*100:.1f}%)."
            ),
            empirical_reference="Elite male fast bowlers: ~130-150 km/h (36-42 m/s).",
            literature_source="Worthington et al. (2013)",
        ),
        optical_br_frame=events.br.frame_float,
        inlier_points_world_m=inlier_pts,
        ransac_inliers=n_inl,
        fit_residual_m=round(rmse_m, 4) if rmse_m is not None else None,
    )


def estimate_ball_release_speed_video(
    video_path: str | Path,
    joints2d_px: np.ndarray,
    joints_world_m: np.ndarray,
    audit: VideoAudit,
    cal: PitchCalibration,
    events: DeliveryEvents,
    bowling_arm: BowlingArm = BowlingArm.RIGHT,
    roi_size: int = 384,
    wrist_unc_m: Optional[float] = None,
) -> OpticalBallSpeedResult:
    """Zero-training asymmetric down-pitch wrist-ROI temporal-median differencing + ballistic RANSAC speed estimator."""
    # Check hard observability gates BEFORE decoding pixels
    if (
        (not events.is_complete_delivery)
        or audit.timebase_unreliable
        or audit.fps < 115.0
        or (not cal.is_metric_calibrated)
        or cal.is_panning_camera
        or abs(math.sin(math.radians(cal.azimuth_deg))) < 0.70
    ):
        return estimate_ball_release_speed_from_world_points(
            np.zeros(0), np.zeros((0, 2)), audit, cal, events, wrist_depth_sigma_m=wrist_unc_m
        )

    n = joints2d_px.shape[0]
    src_map = audit.source_frame_indices if (audit.source_frame_indices and len(audit.source_frame_indices) == n) else list(range(n))
    is_right = bowling_arm == BowlingArm.RIGHT
    wri_i = JOINT_IDX["rwri" if is_right else "lwri"]
    elb_i = JOINT_IDX["relb" if is_right else "lelb"]

    f_start = max(0, events.br.frame_idx - 2)
    f_end = min(n - 1, events.br.frame_idx + 12)
    if f_end - f_start < 5:
        return _suppressed_speed_estimate("Suppressed: insufficient post-release frames (<5) for optical ball tracking.")

    abs_needed = [int(src_map[i]) for i in range(f_start, f_end + 1)]
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return _suppressed_speed_estimate("Suppressed: unable to open video for wrist-ROI ball tracking.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, abs_needed[0])
    decoded = {}
    need_set = set(abs_needed)
    for f_abs in range(abs_needed[0], abs_needed[-1] + 1):
        ok, img = cap.read()
        if not ok or img is None:
            break
        if f_abs in need_set:
            decoded[f_abs] = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cap.release()

    grays = [decoded[a] for a in abs_needed if a in decoded]
    if len(grays) < 6:
        return _suppressed_speed_estimate("Suppressed: could not decode post-release frames for ball tracking.")

    h_img, w_img = grays[0].shape[:2]
    src_w = max(int(audit.width or w_img), 1)
    src_h = max(int(audit.height or h_img), 1)
    sx_vid = w_img / float(src_w)
    sy_vid = h_img / float(src_h)

    K = np.asarray(cal.camera_matrix_K, dtype=np.float64).reshape(3, 3)
    K_inv = np.linalg.inv(K)
    R = np.asarray(cal.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(cal.translation_t, dtype=np.float64).reshape(3)
    wri_world_br = joints_world_m[events.br.frame_idx, wri_i]
    z_plane = float(wri_world_br[2])

    # Asymmetric down-pitch flight corridor ROI: project wrist at BR AND 3.2 m down-pitch (+X_world)
    # so 140-155 km/h balls remain inside the crop for 8-12 frames regardless of bowling direction!
    uv_start = _project_world_point(wri_world_br, K, R, t)
    uv_end = _project_world_point(wri_world_br + np.array([3.2, -0.35, 0.0], dtype=np.float64), K, R, t)
    if uv_start is None:
        wri_uv = joints2d_px[events.br.frame_idx, wri_i]
        uv_start = (float(wri_uv[0]), float(wri_uv[1]))
    if uv_end is None:
        uv_end = (uv_start[0] + roi_size, uv_start[1])

    u0_v, v0_v = uv_start[0] * sx_vid, uv_start[1] * sy_vid
    u1_v, v1_v = uv_end[0] * sx_vid, uv_end[1] * sy_vid
    pad_x = max(80, int(round(0.22 * roi_size * sx_vid)))
    pad_y = max(120, int(round(0.45 * roi_size * sy_vid)))
    x0 = max(0, int(math.floor(min(u0_v, u1_v) - pad_x)))
    x1 = min(w_img, int(math.ceil(max(u0_v, u1_v) + pad_x)))
    y0 = max(0, int(math.floor(min(v0_v, v1_v) - pad_y)))
    y1 = min(h_img, int(math.ceil(max(v0_v, v1_v) + pad_y)))
    if (x1 - x0) < 64 or (y1 - y0) < 64:
        return _suppressed_speed_estimate("Suppressed: down-pitch wrist-ROI falls outside image boundaries.")

    crops = np.stack([g[y0:y1, x0:x1].astype(np.float32) for g in grays], axis=0)
    bg_med = np.median(crops, axis=0)
    arm_mask_rad = max(20, int(round(0.025 * max(w_img, h_img))))

    det_t: List[float] = []
    det_xy: List[Tuple[float, float]] = []
    for k_i in range(2, len(crops)):
        rel_f = min(f_start + k_i, n - 1)
        diff = np.abs(crops[k_i] - bg_med)

        # Mask out tracked bowling hand & forearm capsule so follow-through arm motion cannot be fitted as ball speed
        w_pt = joints2d_px[rel_f, wri_i]
        e_pt = joints2d_px[rel_f, elb_i]
        if np.all(np.isfinite(w_pt)) and np.all(np.isfinite(e_pt)):
            wx = int(round(float(w_pt[0]) * sx_vid - x0))
            wy = int(round(float(w_pt[1]) * sy_vid - y0))
            ex = int(round(float(e_pt[0]) * sx_vid - x0))
            ey = int(round(float(e_pt[1]) * sy_vid - y0))
            hx = int(round(wx + 0.35 * (wx - ex)))
            hy = int(round(wy + 0.35 * (wy - ey)))
            arm_mask = np.ones(diff.shape, dtype=np.uint8)
            cv2.line(arm_mask, (ex, ey), (hx, hy), 0, thickness=2 * arm_mask_rad)
            cv2.circle(arm_mask, (wx, wy), arm_mask_rad + 6, 0, -1)
            diff = diff * arm_mask

        thresh = max(18.0, float(np.percentile(diff, 99.4)))
        bw = (diff >= thresh).astype(np.uint8) * 255
        n_cc, _, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
        best_pt = None
        best_score = -1.0
        for c_i in range(1, n_cc):
            area = int(stats[c_i, cv2.CC_STAT_AREA])
            bw_c = int(stats[c_i, cv2.CC_STAT_WIDTH])
            bh_c = int(stats[c_i, cv2.CC_STAT_HEIGHT])
            if not (6 <= area <= 900):
                continue
            aspect = max(bw_c, bh_c) / max(min(bw_c, bh_c), 1)
            if aspect > 5.5:
                continue
            # Map crop pixel back to original audit resolution matching camera_matrix_K
            u_orig = (x0 + float(cents[c_i, 0])) / max(sx_vid, 1e-6)
            v_orig = (y0 + float(cents[c_i, 1])) / max(sy_vid, 1e-6)
            pw = _ray_to_release_plane_world((u_orig, v_orig), K_inv, R, t, z_plane)
            if pw is None or pw[0] < wri_world_br[0] + 0.05:
                continue
            sc = float(diff[int(cents[c_i, 1]), int(cents[c_i, 0])]) / max(aspect, 1.0)
            if sc > best_score:
                best_score = sc
                best_pt = pw
        if best_pt is not None:
            det_t.append((f_start + k_i) / max(audit.fps, 1.0))
            det_xy.append((float(best_pt[0]), float(best_pt[1])))

    if len(det_t) < 4:
        return _suppressed_speed_estimate(
            f"Suppressed: detected only {len(det_t)} flying ball candidates in asymmetric down-pitch wrist-ROI."
        )

    return estimate_ball_release_speed_from_world_points(
        np.asarray(det_t, dtype=np.float64),
        np.asarray(det_xy, dtype=np.float64),
        audit,
        cal,
        events,
        wrist_depth_sigma_m=wrist_unc_m,
    )
