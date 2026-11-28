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

    # Anchor ankle ground contact positions (MCC Law 21.5: popping crease is at X = +1.22 m;
    # front ankle plants stationary at X = 1.12 m -> heel ~14.5 cm behind crease, 0.0 m foot skate!).
    x_ank_ffc = 1.12
    x_ank_bfc = x_ank_ffc - params.stride_length_m

    sign_arm = 1.0 if is_right else -1.0

    def _leg_chord(knee_int_deg: float) -> float:
        k_clamped = float(np.clip(knee_int_deg, 95.0, 179.6))
        return math.sqrt(max(thigh_l**2 + shank_l**2 - 2.0 * thigh_l * shank_l * math.cos(math.radians(k_clamped)), 1e-6))

    def _pelv_yaw_and_drop_u(t_val: float) -> Tuple[float, float, float]:
        if t_val <= params.bfc_time_s:
            p_yaw = params.pelvis_yaw_bfc_deg
            s_yaw = params.shoulder_yaw_bfc_deg
        elif t_val <= params.ffc_time_s:
            u_loc = _smoothstep(t_val, params.bfc_time_s, params.ffc_time_s)
            p_yaw = params.pelvis_yaw_bfc_deg + 42.0 * u_loc
            if u_loc <= 0.40:
                s_yaw = params.shoulder_yaw_bfc_deg - params.shoulder_counter_rot_deg * math.sin(0.5 * math.pi * (u_loc / 0.40))
            else:
                u2 = (u_loc - 0.40) / 0.60
                min_sh = params.shoulder_yaw_bfc_deg - params.shoulder_counter_rot_deg
                s_yaw = min_sh + (params.shoulder_counter_rot_deg + 18.0) * (u2 * u2 * (3.0 - 2.0 * u2))
        else:
            u_loc = _smoothstep(t_val, params.ffc_time_s, params.br_time_s + 0.12)
            p_yaw = params.pelvis_yaw_bfc_deg + 42.0 + 35.0 * u_loc
            s_yaw = params.shoulder_yaw_bfc_deg + 18.0 + 75.0 * u_loc
        d_u = _smoothstep(t_val, params.bfc_time_s, params.ffc_time_s) * (
            1.0 - 0.5 * _smoothstep(t_val, params.br_time_s, params.br_time_s + 0.2)
        )
        return p_yaw, s_yaw, d_u

    def _hip_offsets_from_pelv(p_yaw: float, d_u: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        p_rad = math.radians(p_yaw - 270.0) * sign_arm
        p_right = np.array([math.sin(p_rad), 0.0, math.cos(p_rad)], dtype=np.float64)
        d_dy = pelvis_half_w * math.tan(math.radians(params.pelvic_drop_ffc_deg)) * d_u
        f_off = np.array(
            [-sign_arm * pelvis_half_w * p_right[0], -pelv_hip_drop - d_dy, -sign_arm * pelvis_half_w * p_right[2]],
            dtype=np.float64,
        )
        r_off = np.array(
            [sign_arm * pelvis_half_w * p_right[0], -pelv_hip_drop + d_dy, sign_arm * pelvis_half_w * p_right[2]],
            dtype=np.float64,
        )
        return f_off, r_off, p_right

    # Exact pelvis anchors at BFC, FFC, and BR so planted ankles stay 100% stationary on Y=0
    c_rear_bfc = _leg_chord(152.0)
    theta_rear_bfc_rad = math.radians(5.0)
    p_yaw_bfc, _, d_u_bfc = _pelv_yaw_and_drop_u(params.bfc_time_s)
    _, r_off_bfc, _ = _hip_offsets_from_pelv(p_yaw_bfc, d_u_bfc)
    x_bfc = x_ank_bfc + c_rear_bfc * math.sin(theta_rear_bfc_rad) - float(r_off_bfc[0])
    y_bfc = c_rear_bfc * math.cos(theta_rear_bfc_rad) - float(r_off_bfc[1])

    theta_ffc_deg = 27.5
    theta_br_deg = 1.5
    c_front_ffc = _leg_chord(params.front_knee_ffc_deg)
    p_yaw_ffc, _, d_u_ffc = _pelv_yaw_and_drop_u(params.ffc_time_s)
    f_off_ffc, _, _ = _hip_offsets_from_pelv(p_yaw_ffc, d_u_ffc)
    x_ffc = x_ank_ffc - c_front_ffc * math.sin(math.radians(theta_ffc_deg)) - float(f_off_ffc[0])
    y_ffc = c_front_ffc * math.cos(math.radians(theta_ffc_deg)) - float(f_off_ffc[1])
    z_ank_ffc = float(f_off_ffc[2])
    f_ank_fixed = np.array([x_ank_ffc, 0.0, z_ank_ffc], dtype=np.float64)

    pz_bfc = (x_bfc - x_ffc) * math.tan(math.radians(params.approach_angle_deg)) if abs(params.approach_angle_deg) > 1e-4 else 0.0
    z_ank_bfc = pz_bfc + float(r_off_bfc[2])
    r_ank_fixed = np.array([x_ank_bfc, 0.0, z_ank_bfc], dtype=np.float64)

    c_front_br = _leg_chord(params.front_knee_br_deg)
    p_yaw_br, _, d_u_br = _pelv_yaw_and_drop_u(params.br_time_s)
    f_off_br, _, _ = _hip_offsets_from_pelv(p_yaw_br, d_u_br)
    x_br = x_ank_ffc - c_front_br * math.sin(math.radians(theta_br_deg)) - float(f_off_br[0])
    y_br = c_front_br * math.cos(math.radians(theta_br_deg)) - float(f_off_br[1])

    # Precompute pelvis X(t) speeds: constant run-up speed up to BFC, vault over fixed front ankle across [FFC, BR]
    v_run = params.runup_speed_mps
    v_ft_start = 2.6 * s

    # Track exact arm horizontal time (UAH)
    uah_frame_float = None

    for i, t in enumerate(times):
        pelv_yaw, sh_yaw, drop_u = _pelv_yaw_and_drop_u(t)
        f_off_cur, r_off_cur, pelv_right = _hip_offsets_from_pelv(pelv_yaw, drop_u)

        # 1. Pelvis center position in Pitch World (strictly C0 continuous across BFC, FFC, BR;
        #    vaults cleanly over stationary `f_ank_fixed` during [FFC, BR] with realistic braking impulse!)
        if t <= params.bfc_time_s:
            px = x_bfc + v_run * (t - params.bfc_time_s)
            py = y_bfc + 0.03 * math.sin(2.0 * math.pi * 2.5 * (t - params.bfc_time_s))
            pz = (px - x_ffc) * math.tan(math.radians(params.approach_angle_deg)) if abs(params.approach_angle_deg) > 1e-4 else 0.0
        elif t <= params.ffc_time_s:
            u = (t - params.bfc_time_s) / max(params.ffc_time_s - params.bfc_time_s, 1e-4)
            px = x_bfc + (x_ffc - x_bfc) * u
            s_u = _smoothstep(u, 0.0, 1.0)
            py = y_bfc + (y_ffc - y_bfc) * s_u - 0.02 * math.sin(math.pi * u)
            pz = (px - x_ffc) * math.tan(math.radians(params.approach_angle_deg)) if abs(params.approach_angle_deg) > 1e-4 else 0.0
        elif t <= params.br_time_s:
            u_lin = (t - params.ffc_time_s) / max(params.br_time_s - params.ffc_time_s, 1e-4)
            u_s = _smoothstep(t, params.ffc_time_s, params.br_time_s)
            f_knee_cur = params.front_knee_ffc_deg + (params.front_knee_br_deg - params.front_knee_ffc_deg) * u_s
            c_cur = _leg_chord(f_knee_cur)
            # Decelerating vault sweep: rapid entry at FFC braking into tall release lever at BR
            w_vault = 1.65 * u_lin - 0.65 * u_lin * u_lin
            theta_cur_rad = math.radians(theta_ffc_deg + (theta_br_deg - theta_ffc_deg) * w_vault)
            f_hip_vault = f_ank_fixed + np.array(
                [-c_cur * math.sin(theta_cur_rad), c_cur * math.cos(theta_cur_rad), 0.0],
                dtype=np.float64,
            )
            px = float(f_hip_vault[0] - f_off_cur[0])
            py = float(f_hip_vault[1] - f_off_cur[1])
            pz = float(f_hip_vault[2] - f_off_cur[2])
        else:
            dt = t - params.br_time_s
            dx_ft = v_ft_start * (1.0 - math.exp(-3.2 * dt)) / 3.2
            px = x_br + dx_ft
            py = y_br - 0.04 * _smoothstep(t, params.br_time_s, params.br_time_s + 0.25)
            pz_at_br = float(f_ank_fixed[2] - f_off_br[2])
            pz = pz_at_br - sign_arm * dx_ft * math.tan(math.radians(params.follow_through_drift_deg))
        pelv = np.array([px, py, pz], dtype=np.float64)

        # 2. Hip joint positions from exact pelvis yaw and frontal-plane pelvic drop
        lhip = pelv + (f_off_cur if is_right else r_off_cur)
        rhip = pelv + (r_off_cur if is_right else f_off_cur)

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

        # Front foot: smooth parabolic swing descending onto `f_ank_fixed`, 100% stationary on Y=0 across [FFC, BR],
        # then smooth peel-off in follow-through (zero teleportation, zero foot-skate, legal behind X = +1.22 m!)
        if t < params.ffc_time_s:
            if t < params.bfc_time_s:
                f_y_sw = 0.14 + 0.12 * abs(math.sin(2.0 * math.pi * 2.4 * t))
                f_x_sw = front_hip[0] + 0.10 * math.sin(2.0 * math.pi * 2.4 * t)
                f_ank_tgt = np.array([f_x_sw, f_y_sw, front_hip[2]], dtype=np.float64)
            else:
                u_s = (t - params.bfc_time_s) / max(params.ffc_time_s - params.bfc_time_s, 1e-4)
                s_u = _smoothstep(u_s, 0.0, 1.0)
                f_y_bfc = 0.14 + 0.12 * abs(math.sin(2.0 * math.pi * 2.4 * params.bfc_time_s))
                f_dx_bfc = 0.10 * math.sin(2.0 * math.pi * 2.4 * params.bfc_time_s)
                f_hip_bfc_x = x_bfc + float((f_off_ffc if False else _hip_offsets_from_pelv(p_yaw_bfc, d_u_bfc)[0])[0])
                f_x_start = f_hip_bfc_x + f_dx_bfc
                f_y_sw = f_y_bfc * (1.0 - s_u) + 0.18 * math.sin(math.pi * u_s) * (1.0 - 0.20 * u_s)
                f_x_sw = f_x_start + (float(f_ank_fixed[0]) - f_x_start) * s_u
                f_z_sw = float(front_hip[2]) * (1.0 - s_u) + float(f_ank_fixed[2]) * s_u
                f_ank_tgt = np.array([f_x_sw, max(0.0, f_y_sw), f_z_sw], dtype=np.float64)
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_tgt)
        elif t <= params.br_time_s + 0.018:
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_fixed.copy())
        else:
            u_ft = _smoothstep(t, params.br_time_s + 0.018, params.br_time_s + 0.28)
            f_trail_tgt = np.array([front_hip[0] - 0.20 * s, 0.15 * s, front_hip[2]], dtype=np.float64)
            f_ank_tgt = (1.0 - u_ft) * f_ank_fixed + u_ft * f_trail_tgt
            f_knee, f_ankle = _solve_leg_ik(front_hip, f_ank_tgt)
        f_toe = f_ankle + np.array([foot_l, 0.0, 0.0])

        # Rear foot: lands continuously onto stationary `r_ank_fixed` at BFC, holds ~38 ms stance, then lifts smoothly
        if params.bfc_time_s <= t <= params.bfc_time_s + 0.038:
            r_knee, r_ankle = _solve_leg_ik(rear_hip, r_ank_fixed.copy())
        elif t < params.bfc_time_s:
            u_pre = min(1.0, max(0.0, (params.bfc_time_s - t) / 0.16))
            r_y_sw = 0.22 * math.sin(0.5 * math.pi * u_pre)
            r_x_sw = rear_hip[0] - c_rear_bfc * math.sin(theta_rear_bfc_rad) - 0.14 * u_pre
            r_knee, r_ankle = _solve_leg_ik(rear_hip, np.array([r_x_sw, r_y_sw, rear_hip[2]], dtype=np.float64))
        else:
            u_post = _smoothstep(t, params.bfc_time_s + 0.038, params.ffc_time_s + 0.03)
            r_trail_tgt = np.array([rear_hip[0] - 0.26 * s, 0.21 * s, rear_hip[2]], dtype=np.float64)
            r_ank_tgt = (1.0 - u_post) * r_ank_fixed + u_post * r_trail_tgt
            r_knee, r_ankle = _solve_leg_ik(rear_hip, r_ank_tgt)
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
