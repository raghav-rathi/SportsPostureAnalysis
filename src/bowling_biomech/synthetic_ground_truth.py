"""Synthetic 3D Fast-Bowling Ground-Truth Simulator & Multi-View Camera Benchmark.

Generates physically realistic 240 Hz 3D fast-bowling deliveries with exact known ground-truth
biomechanical parameters (Worthington 2013, Bayne 2016, Alway 2021, Portus 2004), projects
both MCC Law 7 pitch landmarks and 3D anatomical joints through virtual pinhole cameras across
viewpoints and frame rates, and verifies metric RMSE and 95% CI coverage mathematically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import cv2
import numpy as np

from bowling_biomech.constants import JOINT_IDX, MCC_PITCH_LANDMARKS_M, SMPL24_JOINTS
from bowling_biomech.schemas import BowlingArm
from bowling_biomech.stage1_pitch_calibration import (
    MCC_LAW7_LANDMARKS_M,
    default_camera_matrix,
)


@dataclass
class SyntheticDeliveryParams:
    bowler_arm: BowlingArm = BowlingArm.RIGHT
    fps: float = 240.0
    duration_s: float = 1.15
    runup_speed_mps: float = 5.8           # Worthington (2013) elite male mean ~5.83 +- 0.50 m/s
    bfc_time_s: float = 0.42
    ffc_time_s: float = 0.612              # Felton (2019) elite male mean = 192 ms
    br_time_s: float = 0.720               # Felton (2019) FFC -> BR ~ 108 ms
    stride_length_m: float = 1.48          # ~80% of 1.85 m stature
    front_knee_ffc_deg: float = 160.0      # Internal knee angle (180 = straight)
    front_knee_br_deg: float = 168.0       # Bracing / extending front knee
    trunk_flexion_ffc_deg: float = 12.0    # Global sagittal trunk inclination at FFC
    trunk_flexion_br_deg: float = 44.0     # Delta = 32 deg forward trunk inclination
    thorax_lat_flex_ffc_deg: float = 18.0  # Contralateral lateral inclination at FFC
    thorax_lat_flex_br_deg: float = 36.0
    pelvis_yaw_bfc_deg: float = 215.0      # AIS semi-open alignment relative to +X (180=side, 270=front)
    shoulder_yaw_bfc_deg: float = 210.0
    shoulder_counter_rot_deg: float = 16.0
    delayed_arm_ffc_deg: float = 148.0     # Bowling arm behind global vertical at FFC
    br_phi_deg: float = 8.0                # True arm angle past vertical at BR (randomize in benchmarks!)
    rear_hip_bfc_deg: float = 145.0        # Trunk-thigh rear hip angle at BFC
    pelvic_drop_ffc_deg: float = 4.0       # Frontal pelvic tilt at FFC
    follow_through_drift_deg: float = 6.0  # Lateral drift post-release
    sprint_lean_deg: float = 4.0           # Pre-BFC forward sprint lean (set 12-18 deg for stress tests)
    approach_angle_deg: float = 0.0        # Non-zero angled run-up relative to +X pitch axis
    stature_m: float = 1.85


def _rot_y(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _rot_x(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _smoothstep(t: float, t0: float, t1: float) -> float:
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    x = (t - t0) / max(t1 - t0, 1e-9)
    return x * x * (3.0 - 2.0 * x)


def generate_synthetic_delivery(
    params: SyntheticDeliveryParams,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Synthesize a continuous 3D fast-bowling trajectory in Pitch World Frame (+X pitch, +Y up, +Z off->leg).

    Returns:
        joints_world: (T, 24, 3) array of metric 3D SMPL-24 joint positions in meters.
        ground_truth: dict of exact scalar biomechanical ground-truth values & event frames.
    """
    n_frames = int(round(params.duration_s * params.fps))
    times = np.arange(n_frames, dtype=np.float64) / params.fps
    joints = np.zeros((n_frames, 24, 3), dtype=np.float64)

    # Handedness sign: +1 for right-arm bowler (bowling arm = right, front leg = left, contralateral bend = -Z/+Z)
    is_right = params.bowler_arm == BowlingArm.RIGHT
    # Segment lengths scaled to stature (matching real SMPL-24 anatomical ratios including pelv->hip gap)
    s = params.stature_m / 1.85
    pelvis_half_w = 0.11 * s
    pelv_hip_drop = 0.08 * s
    thigh_l = 0.42 * s
    shank_l = 0.41 * s
    foot_l = 0.16 * s
    torso_l = 0.48 * s
    shoulder_half_w = 0.20 * s
    upper_arm_l = 0.31 * s
    forearm_l = 0.28 * s
    hand_l = 0.10 * s
    neck_l = 0.08 * s
    head_l = 0.11 * s

    # Anchor X position so FFC lands near popping crease X = 1.15 m
    x_ffc = 1.15
    x_bfc = x_ffc - params.stride_length_m

    # Precompute pelvis X(t): constant run-up speed up to BFC, smooth stride to FFC, decelerate after FFC
    v_run = params.runup_speed_mps
    v_stride = params.stride_length_m / max(params.ffc_time_s - params.bfc_time_s, 1e-4)

    # Track exact arm horizontal time (UAH)
    uah_frame_float = None

    for i, t in enumerate(times):
        # 1. Pelvis center position in Pitch World (strictly C0 continuous across BFC, FFC, BR;
        #    hip_y = py - 0.08*s stays strictly inside thigh_l + shank_l = 0.83*s so both feet touch Y=0!)
        if t <= params.bfc_time_s:
            px = x_bfc + v_run * (t - params.bfc_time_s)
            py = 0.87 * s + 0.03 * math.sin(2.0 * math.pi * 2.5 * (t - params.bfc_time_s))
        elif t <= params.ffc_time_s:
            u = (t - params.bfc_time_s) / (params.ffc_time_s - params.bfc_time_s)
            px = x_bfc + params.stride_length_m * u
            py = (0.87 - 0.03 * _smoothstep(u, 0.0, 1.0)) * s - 0.02 * math.sin(math.pi * u)
        else:
            dt = t - params.ffc_time_s
            px = x_ffc + v_stride * 0.65 * (1.0 - math.exp(-3.2 * dt)) / 3.2
            py = 0.84 * s - 0.03 * _smoothstep(t, params.ffc_time_s, params.br_time_s)
        if t > params.br_time_s:
            # Post-release lateral drift along -sign_arm * Z
            sign_arm_tmp = 1.0 if is_right else -1.0
            dx_ft = max(0.0, px - (x_ffc + v_stride * 0.65 * (1.0 - math.exp(-3.2 * (params.br_time_s - params.ffc_time_s))) / 3.2))
            pz = -sign_arm_tmp * dx_ft * math.tan(math.radians(params.follow_through_drift_deg))
        elif abs(params.approach_angle_deg) > 1e-4 and t <= params.ffc_time_s:
            pz = (px - x_ffc) * math.tan(math.radians(params.approach_angle_deg))
        else:
            pz = 0.0
        pelv = np.array([px, py, pz], dtype=np.float64)

        # 2. Pelvis and shoulder transverse yaw (deg: 180=side-on, 270=front-on facing +X)
        # Convention in right-handed Pitch World (+X down pitch, +Y up, +Z = bowler's right when facing +X):
        #   yaw=270 (front-on): right hip/sho = +Z, left = -Z => right_vec = (0, 0, +1)
        #   yaw=180 (right-arm side-on): left points +X, right points -X => right_vec = (-1, 0, 0)
        #   yaw=180 (left-arm side-on): right points +X, left points -X => right_vec = (+1, 0, 0)
        if t <= params.bfc_time_s:
            pelv_yaw = params.pelvis_yaw_bfc_deg
            sh_yaw = params.shoulder_yaw_bfc_deg
        elif t <= params.ffc_time_s:
            u = _smoothstep(t, params.bfc_time_s, params.ffc_time_s)
            pelv_yaw = params.pelvis_yaw_bfc_deg + 42.0 * u
            if u <= 0.40:
                sh_yaw = params.shoulder_yaw_bfc_deg - params.shoulder_counter_rot_deg * math.sin(0.5 * math.pi * (u / 0.40))
            else:
                u2 = (u - 0.40) / 0.60
                min_sh = params.shoulder_yaw_bfc_deg - params.shoulder_counter_rot_deg
                sh_yaw = min_sh + (params.shoulder_counter_rot_deg + 18.0) * (u2 * u2 * (3.0 - 2.0 * u2))
        else:
            u = _smoothstep(t, params.ffc_time_s, params.br_time_s + 0.12)
            pelv_yaw = params.pelvis_yaw_bfc_deg + 42.0 + 35.0 * u
            sh_yaw = params.shoulder_yaw_bfc_deg + 18.0 + 75.0 * u

        sign_arm = 1.0 if is_right else -1.0
        pelv_rad = math.radians(pelv_yaw - 270.0) * sign_arm
        pelv_right = np.array([math.sin(pelv_rad), 0.0, math.cos(pelv_rad)], dtype=np.float64)
        # Apply realistic pelvic drop at FFC (front hip lower than rear hip)
        drop_u = _smoothstep(t, params.bfc_time_s, params.ffc_time_s) * (1.0 - 0.5 * _smoothstep(t, params.br_time_s, params.br_time_s + 0.2))
        drop_dy = pelvis_half_w * math.tan(math.radians(params.pelvic_drop_ffc_deg)) * drop_u
        mid_hip_pos = pelv - np.array([0.0, pelv_hip_drop, 0.0])
        if is_right:
            lhip = mid_hip_pos - pelvis_half_w * pelv_right - np.array([0.0, drop_dy, 0.0])
            rhip = mid_hip_pos + pelvis_half_w * pelv_right + np.array([0.0, drop_dy, 0.0])
        else:
            lhip = mid_hip_pos - pelvis_half_w * pelv_right + np.array([0.0, drop_dy, 0.0])
            rhip = mid_hip_pos + pelvis_half_w * pelv_right - np.array([0.0, drop_dy, 0.0])

        # 3. Trunk forward sagittal flexion & contralateral lateral flexion via exact 3D Z-X Cardan rotation:
        #    s = [sin(alpha)cos(beta), cos(alpha)cos(beta), -sign_arm * sin(beta)]
        if t <= params.bfc_time_s:
            trunk_flex = params.sprint_lean_deg
            # Keep run-up upright (0 deg roll), ramping gently into 6 deg gather lean just before BFC
            lat_flex = 6.0 * _smoothstep(t, params.bfc_time_s - 0.12, params.bfc_time_s)
        elif t <= params.ffc_time_s:
            u = _smoothstep(t, params.bfc_time_s, params.ffc_time_s)
            trunk_flex = params.sprint_lean_deg + (params.trunk_flexion_ffc_deg - params.sprint_lean_deg) * u
            lat_flex = 6.0 + (params.thorax_lat_flex_ffc_deg - 6.0) * u
        elif t <= params.br_time_s:
            u = _smoothstep(t, params.ffc_time_s, params.br_time_s)
            trunk_flex = params.trunk_flexion_ffc_deg + (params.trunk_flexion_br_deg - params.trunk_flexion_ffc_deg) * u
            lat_flex = params.thorax_lat_flex_ffc_deg + (params.thorax_lat_flex_br_deg - params.thorax_lat_flex_ffc_deg) * u
        else:
            u = _smoothstep(t, params.br_time_s, params.br_time_s + 0.18)
            trunk_flex = params.trunk_flexion_br_deg + 14.0 * u
            lat_flex = params.thorax_lat_flex_br_deg + 4.0 * u

        # Rotate +Y by R_z(-alpha) @ R_x(-sign_arm * beta) -> exact unit vector on S^2
        spine_dir = _rot_z(-trunk_flex) @ (_rot_x(-sign_arm * lat_flex) @ np.array([0.0, 1.0, 0.0]))
        spine_dir /= np.linalg.norm(spine_dir)

        spi1 = pelv + 0.25 * torso_l * spine_dir
        spi2 = pelv + 0.55 * torso_l * spine_dir
        spi3 = pelv + 0.85 * torso_l * spine_dir
        neck = pelv + 1.00 * torso_l * spine_dir
        head = neck + (neck_l + head_l) * spine_dir

        sh_rad = math.radians(sh_yaw - 270.0) * sign_arm
        sh_right_horiz = np.array([math.sin(sh_rad), 0.0, math.cos(sh_rad)], dtype=np.float64)
        # Project shoulder line orthogonal to spine_dir so torso maintains rigid shoulder width
        sh_right = sh_right_horiz - np.dot(sh_right_horiz, spine_dir) * spine_dir
        sh_right /= max(float(np.linalg.norm(sh_right)), 1e-6)
        lcla = spi3 - 0.45 * shoulder_half_w * sh_right
        rcla = spi3 + 0.45 * shoulder_half_w * sh_right
        lsho = neck - shoulder_half_w * sh_right
        rsho = neck + shoulder_half_w * sh_right

        # 4. Lower limbs: Continuous, non-teleporting 2-link sagittal IK across swing & stance (EXP-007 S3!)
        front_hip = lhip if is_right else rhip
        rear_hip = rhip if is_right else lhip

        def _solve_leg_ik(hip_p: np.ndarray, ankle_p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
            """Exact 2-link sagittal IK placing ankle at `ankle_p` (clamped to reach) with knee bending forward (+X)."""
            vec = ankle_p - hip_p
            dx_v, dy_v = float(vec[0]), float(-vec[1])
            max_reach = thigh_l + shank_l - 1e-4
            min_reach = abs(thigh_l - shank_l) + 0.05
            dist = float(math.hypot(dx_v, dy_v))
            if dist > max_reach:
                scale_r = max_reach / max(dist, 1e-6)
                dx_v *= scale_r
                dy_v *= scale_r
                dist = max_reach
                ankle_p = np.array([hip_p[0] + dx_v, hip_p[1] - dy_v, ankle_p[2]], dtype=np.float64)
            elif dist < min_reach:
                dy_v = max(dy_v, min_reach)
                dist = float(math.hypot(dx_v, dy_v))
                ankle_p = np.array([hip_p[0] + dx_v, hip_p[1] - dy_v, ankle_p[2]], dtype=np.float64)
            chord_ang = math.atan2(dx_v, max(dy_v, 1e-4))
            cos_g = float(np.clip((thigh_l**2 + dist**2 - shank_l**2) / (2.0 * thigh_l * max(dist, 1e-6)), -1.0, 1.0))
            alpha_k = chord_ang + math.acos(cos_g)
            knee_p = hip_p + thigh_l * np.array([math.sin(alpha_k), -math.cos(alpha_k), 0.0], dtype=np.float64)
            return knee_p, ankle_p

        def _planted_ankle_for_knee(hip_p: np.ndarray, knee_int_deg: float, forward_sign: float = 1.0) -> np.ndarray:
            chord = math.sqrt(thigh_l**2 + shank_l**2 - 2.0 * thigh_l * shank_l * math.cos(math.radians(knee_int_deg)))
            dy_h = min(float(hip_p[1]), chord - 1e-4)
            dx_h = math.sqrt(max(chord**2 - dy_h**2, 1e-6))
            return np.array([hip_p[0] + forward_sign * dx_h, 0.0, hip_p[2]], dtype=np.float64)

        # Front foot: smooth parabolic swing descending continuously onto exact FFC ground point (ZERO teleportation!)
        if t < params.ffc_time_s:
            target_land = _planted_ankle_for_knee(front_hip, params.front_knee_ffc_deg, forward_sign=1.0)
            if t < params.bfc_time_s:
                f_y_sw = 0.14 + 0.12 * abs(math.sin(2.0 * math.pi * 2.4 * t))
                f_x_sw = front_hip[0] + 0.10 * math.sin(2.0 * math.pi * 2.4 * t)
                f_ank_tgt = np.array([f_x_sw, f_y_sw, front_hip[2]], dtype=np.float64)
            else:
                u_s = (t - params.bfc_time_s) / max(params.ffc_time_s - params.bfc_time_s, 1e-4)
                s_u = _smoothstep(u_s, 0.0, 1.0)
                f_y_bfc = 0.14 + 0.12 * abs(math.sin(2.0 * math.pi * 2.4 * params.bfc_time_s))
                f_dx_bfc = 0.10 * math.sin(2.0 * math.pi * 2.4 * params.bfc_time_s)
                # Exact C0 continuity at both BFC (u_s=0) and FFC (u_s=1, Y=0, X=target_land[0])
                f_y_sw = f_y_bfc * (1.0 - s_u) + 0.18 * math.sin(math.pi * u_s) * (1.0 - 0.20 * u_s)
                f_x_start = front_hip[0] + f_dx_bfc
                f_x_sw = f_x_start + (target_land[0] - f_x_start) * s_u
                f_ank_tgt = np.array([f_x_sw, max(0.0, f_y_sw), front_hip[2]], dtype=np.float64)
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_tgt)
        elif t <= params.br_time_s:
            u = _smoothstep(t, params.ffc_time_s, params.br_time_s)
            f_knee_int = params.front_knee_ffc_deg + (params.front_knee_br_deg - params.front_knee_ffc_deg) * u
            f_ank_tgt = _planted_ankle_for_knee(front_hip, f_knee_int, forward_sign=1.0)
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_tgt)
        else:
            u = _smoothstep(t, params.br_time_s, params.br_time_s + 0.25)
            f_ank_br = _planted_ankle_for_knee(front_hip, params.front_knee_br_deg, forward_sign=1.0)
            f_ank_tgt = np.array([f_ank_br[0] - 0.32 * u, 0.14 * u, front_hip[2]], dtype=np.float64)
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_tgt)
        f_toe = f_ankle + np.array([foot_l, 0.0, 0.0])

        # Rear foot: smooth landing continuously onto exact BFC ground point, planted ~75 ms, then smooth lift
        r_ank_bfc_tgt = _planted_ankle_for_knee(rear_hip, 152.0, forward_sign=-0.10)
        if params.bfc_time_s <= t <= params.bfc_time_s + 0.075:
            r_knee, r_ankle = _solve_leg_ik(rear_hip, r_ank_bfc_tgt)
        elif t < params.bfc_time_s:
            u_pre = min(1.0, max(0.0, (params.bfc_time_s - t) / 0.16))
            r_y_sw = 0.22 * math.sin(0.5 * math.pi * u_pre)
            r_x_sw = r_ank_bfc_tgt[0] - 0.16 * u_pre
            r_knee, r_ankle = _solve_leg_ik(rear_hip, np.array([r_x_sw, r_y_sw, rear_hip[2]], dtype=np.float64))
        else:
            u_post = _smoothstep(t, params.bfc_time_s + 0.075, params.br_time_s)
            r_y_sw = 0.20 * u_post
            r_x_sw = r_ank_bfc_tgt[0] - 0.35 * u_post
            r_knee, r_ankle = _solve_leg_ik(rear_hip, np.array([r_x_sw, r_y_sw, rear_hip[2]], dtype=np.float64))
        r_toe = r_ankle + np.array([foot_l, 0.0, 0.0])

        if is_right:
            lkne, lank, ltoe = f_knee, f_ankle, f_toe
            rkne, rank, rtoe = r_knee, r_ankle, r_toe
        else:
            rkne, rank, rtoe = f_knee, f_ankle, f_toe
            lkne, lank, ltoe = r_knee, r_ankle, r_toe

        # 5. Bowling arm circumduction in delivery plane (cleanly decelerating after follow-through — no 2nd pass!)
        phi_ffc = -params.delayed_arm_ffc_deg
        phi_br = params.br_phi_deg
        if t < params.bfc_time_s:
            phi = -255.0 + 35.0 * (t / max(params.bfc_time_s, 1e-4))
        elif t <= params.ffc_time_s:
            u = _smoothstep(t, params.bfc_time_s, params.ffc_time_s)
            phi = -220.0 + (phi_ffc - (-220.0)) * u
        elif t <= params.br_time_s:
            u = (t - params.ffc_time_s) / max(params.br_time_s - params.ffc_time_s, 1e-5)
            phi = phi_ffc + (phi_br - phi_ffc) * (0.35 * u + 0.65 * u * u)
        else:
            dt = t - params.br_time_s
            # Realistic follow-through deceleration settling smoothly ~115 deg past BR (does not loop overhead!)
            omega_br = (phi_br - phi_ffc) * 1.65 / max(params.br_time_s - params.ffc_time_s, 1e-5)
            max_ft_sweep = 115.0
            k_damp = max(omega_br / max_ft_sweep, 8.0)
            phi = phi_br + max_ft_sweep * (1.0 - math.exp(-k_damp * dt))

        if uah_frame_float is None and t >= params.ffc_time_s and phi >= -90.0:
            uah_frame_float = float(i)

        phi_rad = math.radians(phi)
        # Arm unit vector in pitch sagittal plane (slight lateral abduction outward)
        b_arm_dir = np.array([math.sin(phi_rad), math.cos(phi_rad), 0.08 * sign_arm], dtype=np.float64)
        b_arm_dir /= np.linalg.norm(b_arm_dir)

        b_sho = rsho if is_right else lsho
        nb_sho = lsho if is_right else rsho

        # Nearly straight bowling arm (172° internal elbow angle, legal <15° change)
        b_elb = b_sho + upper_arm_l * b_arm_dir
        b_wri = b_elb + forearm_l * b_arm_dir
        b_han = b_wri + hand_l * b_arm_dir

        # Non-bowling (front) arm: high at BFC/FFC (+X, +Y), pulls cleanly into ribs by BR
        if t <= params.ffc_time_s:
            nb_dir = np.array([0.72, 0.68, -0.10 * sign_arm], dtype=np.float64)
            nb_elbow_dir = np.array([0.65, 0.75, -0.05 * sign_arm], dtype=np.float64)
        else:
            u = _smoothstep(t, params.ffc_time_s, params.br_time_s)
            nb_dir = np.array([0.72 - 0.92 * u, 0.68 - 1.38 * u, -0.15 * sign_arm], dtype=np.float64)
            nb_elbow_dir = np.array([0.65 - 0.55 * u, 0.75 - 0.60 * u, -0.20 * sign_arm], dtype=np.float64)
        nb_dir /= np.linalg.norm(nb_dir)
        nb_elbow_dir /= np.linalg.norm(nb_elbow_dir)
        nb_elb = nb_sho + upper_arm_l * nb_dir
        nb_wri = nb_elb + forearm_l * nb_elbow_dir
        nb_han = nb_wri + hand_l * nb_elbow_dir

        if is_right:
            relb, rwri, rhan = b_elb, b_wri, b_han
            lelb, lwri, lhan = nb_elb, nb_wri, nb_han
        else:
            lelb, lwri, lhan = b_elb, b_wri, b_han
            relb, rwri, rhan = nb_elb, nb_wri, nb_han

        frame_j = np.array(
            [
                pelv, lhip, rhip, spi1, lkne, rkne,
                spi2, lank, rank, spi3, ltoe, rtoe,
                neck, lcla, rcla, head, lsho, rsho,
                lelb, relb, lwri, rwri, lhan, rhan,
            ],
            dtype=np.float64,
        )
        joints[i] = frame_j

    gt = {
        "bfc_frame": params.bfc_time_s * params.fps,
        "ffc_frame": params.ffc_time_s * params.fps,
        "uah_frame": float(uah_frame_float) if uah_frame_float is not None else (params.ffc_time_s + 0.06) * params.fps,
        "br_frame": params.br_time_s * params.fps,
        "runup_speed_mps": params.runup_speed_mps,
        "stride_length_m": params.stride_length_m,
        "front_knee_ffc_deg": params.front_knee_ffc_deg,
        "front_knee_br_deg": params.front_knee_br_deg,
        "front_knee_change_deg": params.front_knee_br_deg - params.front_knee_ffc_deg,
        "trunk_flexion_ffc_deg": params.trunk_flexion_ffc_deg,
        "trunk_flexion_br_deg": params.trunk_flexion_br_deg,
        "trunk_flexion_delta_deg": params.trunk_flexion_br_deg - params.trunk_flexion_ffc_deg,
        "thorax_lat_flex_ffc_deg": params.thorax_lat_flex_ffc_deg,
        "thorax_lat_flex_br_deg": params.thorax_lat_flex_br_deg,
        "shoulder_counter_rotation_deg": params.shoulder_counter_rot_deg,
        "delayed_arm_ffc_deg": params.delayed_arm_ffc_deg,
    }
    return joints, gt


def make_virtual_camera(
    azimuth_deg: float,
    elevation_deg: float,
    distance_m: float = 12.0,
    look_at: Tuple[float, float, float] = (0.61, 1.0, 0.0),
    width: int = 1280,
    height: int = 720,
    fov_deg: float = 50.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construct exact pinhole camera (K, R_world_to_cam, t, cam_pos_world) looking at pitch crease."""
    K = default_camera_matrix(width, height, fov_deg)
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    target = np.array(look_at, dtype=np.float64)

    # azimuth=0 -> behind bowler (-X); azimuth=90 -> +Z side-on; azimuth=180 -> +X behind batter
    horiz = distance_m * math.cos(el)
    cam_pos = target + np.array(
        [-horiz * math.cos(az), distance_m * math.sin(el), horiz * math.sin(az)],
        dtype=np.float64,
    )

    # OpenCV camera axes: +Z_cam forward (cam -> target), +X_cam right, +Y_cam down
    z_cam = target - cam_pos
    z_cam /= np.linalg.norm(z_cam)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_cam = np.cross(z_cam, world_up)
    if np.linalg.norm(x_cam) < 1e-6:
        x_cam = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)  # points downward in OpenCV convention

    # Rows of R are camera basis vectors expressed in world coords
    R = np.vstack([x_cam, y_cam, z_cam])
    t = -R @ cam_pos
    return K, R, t, cam_pos


def project_world_points(
    points_world: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project (..., 3) world points to (..., 2) pixel coordinates and (..., 3) camera-space meters."""
    orig_shape = points_world.shape[:-1]
    pts = points_world.reshape(-1, 3)
    pts_cam = (R @ pts.T).T + t.reshape(1, 3)
    uv_h = (K @ pts_cam.T).T
    uv = uv_h[:, :2] / np.maximum(uv_h[:, 2:3], 1e-6)
    if noise_px > 0.0:
        gen = rng if rng is not None else np.random.default_rng(42)
        uv = uv + gen.normal(0.0, noise_px, size=uv.shape)
    return uv.reshape(*orig_shape, 2), pts_cam.reshape(*orig_shape, 3)


def project_mcc_landmarks(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    landmark_keys: List[str],
    noise_px: float = 0.0,
    rng: np.random.Generator | None = None,
) -> Dict[str, Tuple[float, float]]:
    """Project named MCC Law 7 landmarks into virtual camera pixel space."""
    pts = np.array([MCC_LAW7_LANDMARKS_M[k] for k in landmark_keys], dtype=np.float64)
    uv, _ = project_world_points(pts, K, R, t, noise_px=noise_px, rng=rng)
    return {k: (float(uv[idx, 0]), float(uv[idx, 1])) for idx, k in enumerate(landmark_keys)}
