"""Stage 6A: Truthful, Uncertainty-Aware Biomechanics Visuals & Freeze-Frame Video Renderer.

Fixes EXP-007:
  - B2 / Zero-Drift Frame Alignment: Accepts `source_frame_indices` and decodes sequentially via `cap.read()`
    so pulldown-deduplicated clips (e.g., Steyn 25-in-29.97) align 2D skeletons and freeze-frames to the
    exact raw video frame with zero off-by-N drift.
  - Section 6: Removes unverified "Uninjured Elite Lat-Flex (Bayne 2016) 32-44 deg" plot bands and replaces
               retired 0-100 Quadrant Panel 6 with an honest Cohort Z-Score / Repeatability Forest Plot.
  - B23: Imports `JOINT_IDX` from `constants.py`.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import JOINT_IDX
from .schemas import BiomechanicalFactSheet, ObservabilityGrade
from .stage3_kinematics import KinematicsResult

SKELETON_EDGES: List[Tuple[str, str, Tuple[int, int, int], str]] = [
    ("pelv", "spi1", (240, 120, 255), "#f078ff"),
    ("spi1", "spi2", (240, 120, 255), "#f078ff"),
    ("spi2", "spi3", (240, 120, 255), "#f078ff"),
    ("spi3", "neck", (240, 120, 255), "#f078ff"),
    ("neck", "head", (240, 120, 255), "#f078ff"),
    ("lsho", "rsho", (80, 220, 255), "#50dcff"),
    ("lhip", "rhip", (80, 220, 255), "#50dcff"),
    ("lhip", "lkne", (255, 190, 50), "#32beff"),
    ("lkne", "lank", (255, 190, 50), "#32beff"),
    ("lank", "ltoe", (255, 190, 50), "#32beff"),
    ("rhip", "rkne", (70, 150, 255), "#ff9646"),
    ("rkne", "rank", (70, 150, 255), "#ff9646"),
    ("rank", "rtoe", (70, 150, 255), "#ff9646"),
    ("rsho", "relb", (90, 255, 110), "#6eff5a"),
    ("relb", "rwri", (90, 255, 110), "#6eff5a"),
    ("lsho", "lelb", (255, 120, 70), "#4678ff"),
    ("lelb", "lwri", (255, 120, 70), "#4678ff"),
]


def _prepare_readable_canvas(
    frame_bgr: np.ndarray,
    joints2d_px: np.ndarray,
    target_h: int = 560,
    min_w: int = 560,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize frame BEFORE drawing text and pad narrow portrait frames so overlays stay crisp."""
    h, w = frame_bgr.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(frame_bgr, (new_w, target_h), interpolation=cv2.INTER_AREA)
    j2 = joints2d_px.copy() * scale

    if new_w < min_w:
        pad_left = (min_w - new_w) // 2
        pad_right = min_w - new_w - pad_left
        canvas = cv2.copyMakeBorder(
            resized, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(15, 19, 26)
        )
        j2[:, 0] += pad_left
        return canvas, j2
    return resized, j2


def draw_skeleton_on_frame(
    frame_bgr: np.ndarray,
    joints2d_px: np.ndarray,
    title: str,
    subtitle: str,
    callout_lines: List[str],
    draw_skeleton: bool = True,
) -> np.ndarray:
    """Draw reprojected 2D skeleton and crisp dark-banner metric callouts on an already-sized canvas."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    thick = max(2, int(round(min(h, w) / 260)))

    if draw_skeleton:
        for a_name, b_name, bgr, _ in SKELETON_EDGES:
            pa = joints2d_px[JOINT_IDX[a_name]]
            pb = joints2d_px[JOINT_IDX[b_name]]
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), bgr, thick, cv2.LINE_AA)
        for pt in joints2d_px:
            if np.all(np.isfinite(pt)):
                cv2.circle(img, (int(pt[0]), int(pt[1])), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)

    banner_h = max(62, int(0.115 * h))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (14, 17, 23), -1)
    cv2.addWeighted(overlay, 0.84, img, 0.16, 0, img)

    title_scale = float(np.clip(w / 900.0, 0.58, 0.85))
    sub_scale = float(np.clip(w / 1220.0, 0.44, 0.62))
    cv2.putText(
        img, title, (12, int(banner_h * 0.46)),
        cv2.FONT_HERSHEY_SIMPLEX, title_scale, (255, 255, 255), 2, cv2.LINE_AA,
    )
    cv2.putText(
        img, subtitle, (12, int(banner_h * 0.84)),
        cv2.FONT_HERSHEY_SIMPLEX, sub_scale, (160, 225, 255), 1, cv2.LINE_AA,
    )

    if callout_lines:
        line_h = max(24, int(0.044 * h))
        box_h = line_h * len(callout_lines) + 14
        overlay2 = img.copy()
        cv2.rectangle(overlay2, (0, h - box_h), (w, h), (14, 17, 23), -1)
        cv2.addWeighted(overlay2, 0.86, img, 0.14, 0, img)
        txt_scale = float(np.clip(w / 1220.0, 0.45, 0.66))
        for i, txt in enumerate(callout_lines):
            y = h - box_h + 20 + i * line_h
            col = (100, 196, 255) if ("Suppressed" in txt or "REFUSED" in txt) else (238, 244, 250)
            cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, txt_scale, col, 1, cv2.LINE_AA)
    return img


def _read_exact_frames_sequential(video_path: str | Path, abs_indices: List[int]) -> Dict[int, np.ndarray]:
    """Decode exact source frame indices sequentially via `cap.read()` (avoids GOP seek off-by-1 drift)."""
    if not abs_indices:
        return {}
    needed = set(int(i) for i in abs_indices)
    first_f, last_f = min(needed), max(needed)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_f)
    out: Dict[int, np.ndarray] = {}
    for f in range(first_f, last_f + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if f in needed:
            out[f] = frame
    cap.release()
    return out


def render_keyframe_panel(
    video_path: str | Path,
    start_frame: int,
    joints2d_px: np.ndarray,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
    source_frame_indices: Optional[List[int]] = None,
) -> Path:
    """Render readable 2x2 key-phase freeze-frame strip (BFC, FFC, UAH, BR) aligned to exact source frames."""
    ev = fact_sheet.events
    m = fact_sheet.metrics
    src_map = source_frame_indices or fact_sheet.audit.source_frame_indices

    def _to_abs(rel_idx: int) -> int:
        if src_map and 0 <= rel_idx < len(src_map):
            return int(src_map[rel_idx])
        return int(start_frame + rel_idx)

    def _fmt(key: str, short: str) -> str:
        me = m.get(key)
        if me is None or me.suppressed or me.value is None:
            return f"{short}: [Suppressed - {me.observability.value if me else 'N/A'}]"
        tag = "rep" if me.interval_type == "repeatability_interval" else "unc"
        if me.ci_low is not None and me.ci_high is not None:
            return f"{short}: {me.value:.1f} [{me.ci_low:.1f}, {me.ci_high:.1f}] {me.unit} ({me.observability.value}, {tag})"
        return f"{short}: {me.value:.1f} {me.unit} ({me.observability.value})"

    phases = [
        ("1. Back Foot Contact (BFC)", ev.bfc if ev.is_complete_delivery else None, [_fmt("runup_speed_mps", "Run-up"), _fmt("rear_hip_bfc_deg", "Rear Hip Proxy")]),
        (
            "2. Front Foot Contact (FFC)",
            ev.ffc if ev.is_complete_delivery else None,
            [_fmt("front_knee_ffc_deg", "Front Knee"), _fmt("delayed_arm_ffc_deg", "Arm Delay"), _fmt("thorax_lat_flex_ffc_deg", "Lat Tilt")],
        ),
        (
            "3. Upper Arm Horizontal (UAH)",
            ev.uah if ev.is_complete_delivery else None,
            [_fmt("shoulder_counter_rotation_deg", "SCR"), "ICC Elbow: REFUSED (Lab 3D Req)"],
        ),
        (
            "4. Ball Release (BR)" if ev.is_complete_delivery else "4. Ball Release (NOT VERIFIED IN CLIP)",
            ev.br if ev.is_complete_delivery else None,
            [_fmt("front_knee_br_deg", "Front Knee"), _fmt("trunk_flexion_delta_deg", "Trunk Flex d"), _fmt("thorax_lat_flex_br_deg", "Lat Tilt BR")],
        ),
    ]

    abs_needed = [_to_abs(evt.frame_idx) for _, evt, _ in phases if evt is not None]
    decoded = _read_exact_frames_sequential(video_path, abs_needed)

    panels: List[np.ndarray] = []
    target_h = 560
    min_w = 560
    canvas_w = None

    for title, evt, lines in phases:
        if evt is None:
            w_use = canvas_w or min_w
            blank = np.full((target_h, w_use, 3), (22, 26, 34), dtype=np.uint8)
            msg = "Delivery incomplete / phase not observed" if not ev.is_complete_delivery else "Phase not observed in clip"
            cv2.putText(
                blank, msg, (max(16, w_use // 10), target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.68, (90, 190, 255), 2, cv2.LINE_AA,
            )
            annotated = draw_skeleton_on_frame(
                blank, np.zeros((24, 2)), title, "Suppressed — release sanity gate not passed", lines, draw_skeleton=False
            )
            panels.append(annotated)
            continue

        abs_f = _to_abs(evt.frame_idx)
        frame = decoded.get(abs_f)
        if frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        j2 = joints2d_px[min(evt.frame_idx, len(joints2d_px) - 1)]
        canvas, j2_s = _prepare_readable_canvas(frame, j2, target_h=target_h, min_w=min_w)
        canvas_w = canvas.shape[1]
        sub = f"t={evt.timestamp_s:.3f}s (u={evt.frame_idx}, src={abs_f}, +- {evt.timing_uncertainty_ms:.1f} ms)"
        annotated = draw_skeleton_on_frame(canvas, j2_s, title, sub, lines, draw_skeleton=True)
        panels.append(annotated)

    max_w = max(p.shape[1] for p in panels)
    norm_panels = []
    for p in panels:
        if p.shape[1] != max_w:
            pad_l = (max_w - p.shape[1]) // 2
            pad_r = max_w - p.shape[1] - pad_l
            p = cv2.copyMakeBorder(p, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(15, 19, 26))
        norm_panels.append(p)

    top = np.hstack([norm_panels[0], norm_panels[1]])
    bot = np.hstack([norm_panels[2], norm_panels[3]])
    grid = np.vstack([top, bot])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return out


def render_biomechanics_dashboard(
    joints_world: np.ndarray,
    kin: KinematicsResult,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
) -> Path:
    """Render 6-panel scientific biomechanics dashboard (waveforms, virtual 3D views, cohort forest plot)."""
    fps = fact_sheet.audit.fps
    n = joints_world.shape[0]
    ev = fact_sheet.events
    t_ms = (np.arange(n) - ev.ffc.frame_float) / max(fps, 1.0) * 1000.0
    bfc_ms = (ev.bfc.frame_float - ev.ffc.frame_float) / max(fps, 1.0) * 1000.0
    br_ms = (ev.br.frame_float - ev.ffc.frame_float) / max(fps, 1.0) * 1000.0

    fig = plt.figure(figsize=(16.8, 10.2), facecolor="#0f131a")
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.40)
    text_color = "#e6edf3"
    grid_color = "#263042"

    def _style_ax(ax, title: str, xlabel: str, ylabel: str):
        ax.set_facecolor("#161c26")
        ax.set_title(title, color=text_color, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color="#9fb0c8", fontsize=8.5)
        ax.set_ylabel(ylabel, color="#9fb0c8", fontsize=8.5)
        ax.tick_params(colors="#9fb0c8", labelsize=8)
        ax.grid(True, color=grid_color, alpha=0.65, linewidth=0.7)
        for sp in ax.spines.values():
            sp.set_color("#334155")

    def _mark_events(ax):
        if not ev.is_complete_delivery:
            return
        y_lo, y_hi = ax.get_ylim()
        span = max(y_hi - y_lo, 1.0)
        marks = [(bfc_ms, "BFC", "#f59e0b", 0.96), (0.0, "FFC", "#38bdf8", 0.87), (br_ms, "BR", "#22c55e", 0.96)]
        for x_m, lbl, col, y_frac in marks:
            ax.axvline(x_m, color=col, linestyle="--", linewidth=1.1, alpha=0.85)
            ax.text(
                x_m, y_lo + y_frac * span, lbl, color=col, fontsize=7.5, fontweight="bold",
                ha="center", va="top", bbox=dict(boxstyle="round,pad=0.14", facecolor="#0f131a", edgecolor="none", alpha=0.72),
            )

    # 1. Front Knee Flexion / Bracing
    ax1 = fig.add_subplot(gs[0, 0])
    fk_m = fact_sheet.metrics["front_knee_br_deg"]
    fk_sup = fk_m.suppressed
    ax1.plot(
        t_ms, kin.series["front_knee_deg"], color="#38bdf8",
        linewidth=1.5 if fk_sup else 2.2, linestyle="--" if fk_sup else "-", alpha=0.22 if fk_sup else 1.0,
    )
    _style_ax(
        ax1,
        f"Front Knee Internal Angle ({'SUPPRESSED' if fk_sup else fk_m.observability.value.upper()})",
        "Time relative to FFC (ms)",
        "Internal Angle (deg, 180=straight)",
    )
    _mark_events(ax1)
    if fk_sup:
        ax1.text(
            0.5, 0.45, "SUPPRESSED FROM THIS VIEW / CLIP\n(Curve dimmed; excluded from cues)",
            transform=ax1.transAxes, ha="center", va="center", color="#fbbf24", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f131a", edgecolor="#fbbf24", alpha=0.88),
        )

    # 2. Global Trunk Kinematics (Z-X Cardan — no misleading Bayne elite bands!)
    ax2 = fig.add_subplot(gs[0, 1])
    tf_sup = fact_sheet.metrics["trunk_flexion_delta_deg"].suppressed
    lf_sup = fact_sheet.metrics["thorax_lat_flex_br_deg"].suppressed
    ax2.plot(
        t_ms, kin.series["trunk_flex_deg"], color="#a855f7",
        linewidth=1.4 if tf_sup else 2.1, linestyle="--" if tf_sup else "-", alpha=0.20 if tf_sup else 1.0,
        label=f"Global Sagittal Inclination{' (suppressed)' if tf_sup else ''}",
    )
    ax2.plot(
        t_ms, kin.series["thorax_lat_flex_deg"], color="#fb7185",
        linewidth=1.4 if lf_sup else 2.1, linestyle="--" if lf_sup else "-", alpha=0.20 if lf_sup else 1.0,
        label=f"Global Contralateral Inclination{' (suppressed)' if lf_sup else ''}",
    )
    _style_ax(ax2, "Global Trunk Kinematics (Z-X Cardan)", "Time relative to FFC (ms)", "Angle from Vertical (deg)")
    _mark_events(ax2)
    ax2.legend(loc="lower left", fontsize=7.0, facecolor="#161c26", labelcolor=text_color)

    # 3. Pelvis vs Shoulder Transverse Alignment
    ax3 = fig.add_subplot(gs[0, 2])
    scr_sup = fact_sheet.metrics["shoulder_counter_rotation_deg"].suppressed
    ax3.plot(
        t_ms, kin.series["pelvis_yaw_deg"], color="#22d3ee",
        linewidth=1.4 if scr_sup else 2.0, linestyle="--" if scr_sup else "-", alpha=0.22 if scr_sup else 1.0,
        label="Pelvis Alignment",
    )
    ax3.plot(
        t_ms, kin.series["shoulder_yaw_deg"], color="#facc15",
        linewidth=1.4 if scr_sup else 2.0, linestyle="--" if scr_sup else "-", alpha=0.22 if scr_sup else 1.0,
        label="Shoulder Alignment",
    )
    _style_ax(
        ax3,
        f"Transverse Alignment (180=Side, 270=Front){' [SUP]' if scr_sup else ''}",
        "Time relative to FFC (ms)",
        "Unwrapped Alignment (deg)",
    )
    if scr_sup:
        ax3.set_ylim(90.0, 420.0)
    _mark_events(ax3)
    ax3.legend(loc="lower left", fontsize=7.5, facecolor="#161c26", labelcolor=text_color)

    # Use smoothed Pitch-World trajectory for clean 3D virtual pose views (P1-5)
    jw_plot = kin.joints_world_smoothed if kin.joints_world_smoothed is not None else joints_world
    f_ank_name = "lank" if fact_sheet.bowling_arm.value == "right" else "rank"
    if ev.is_complete_delivery:
        i0_g = max(0, min(ev.ffc.frame_idx, n - 1))
        i1_g = max(i0_g + 1, min(ev.br.frame_idx + 1, n))
        y_ground_ref = float(np.median(jw_plot[i0_g:i1_g, JOINT_IDX[f_ank_name], 1]))
    else:
        y_ground_ref = float(np.percentile(jw_plot[:, JOINT_IDX[f_ank_name], 1], 10))

    # 4. Virtual 3D Side-On Projection (anchor each pose to planted front ankle X when world translation is suppressed)
    ax4 = fig.add_subplot(gs[1, 0])
    trans_sup = fact_sheet.metrics["stride_length_pct_stature"].suppressed
    plot_phases = [(ev.ffc.frame_idx, 0.45), (ev.br.frame_idx, 1.0)] if ev.is_complete_delivery else [(n // 2, 0.7)]
    for idx, col_alpha in plot_phases:
        pts = jw_plot[min(idx, n - 1)]
        x_ref = float(pts[JOINT_IDX[f_ank_name], 0]) if trans_sup else 0.0
        for a_name, b_name, _, hex_c in SKELETON_EDGES:
            pa, pb = pts[JOINT_IDX[a_name]], pts[JOINT_IDX[b_name]]
            ax4.plot(
                [pa[0] - x_ref, pb[0] - x_ref],
                [pa[1] - y_ground_ref, pb[1] - y_ground_ref],
                color=hex_c,
                alpha=col_alpha,
                linewidth=2.2,
            )
    ax4.axhline(0.0, color="#475569", linestyle=":", linewidth=0.9)
    x_lbl_4 = "+X Relative to Front Ankle (m, translation suppressed)" if trans_sup else "+X Down Pitch (m)"
    _style_ax(ax4, "Virtual 3D Pitch-World Side View (FFC -> BR)", x_lbl_4, "+Y Above Ground (m)")
    ax4.set_aspect("equal", adjustable="datalim")

    # 5. Virtual 3D Rear Coronal View
    ax5 = fig.add_subplot(gs[1, 1])
    rear_idx = ev.br.frame_idx if ev.is_complete_delivery else (n // 2)
    pts_rear = jw_plot[min(rear_idx, n - 1)]
    for a_name, b_name, _, hex_c in SKELETON_EDGES:
        pa, pb = pts_rear[JOINT_IDX[a_name]], pts_rear[JOINT_IDX[b_name]]
        ax5.plot([pa[2], pb[2]], [pa[1] - y_ground_ref, pb[1] - y_ground_ref], color=hex_c, linewidth=2.4)
    ax5.axhline(0.0, color="#475569", linestyle=":", linewidth=0.9)
    pelv_r = pts_rear[JOINT_IDX["pelv"]]
    ax5.axvline(pelv_r[2], color="#94a3b8", linestyle="--", linewidth=1.1, label="Gravity Plumb Line")
    _style_ax(
        ax5,
        f"Virtual 3D Rear View at {'BR' if ev.is_complete_delivery else 'Mid-Clip'}",
        "+Z Across Pitch (m)",
        "+Y Above Ground (m)",
    )
    ax5.set_aspect("equal", adjustable="datalim")
    ax5.legend(loc="upper left", fontsize=7.5, facecolor="#161c26", labelcolor=text_color)

    # 6. Cohort Comparison Z-Score Forest Plot (replaces retired unvalidated 0-100 quadrant!)
    ax6 = fig.add_subplot(gs[1, 2])
    comps = fact_sheet.cohort_comparisons
    short_names = {
        "front_knee_br_deg": "Front Knee BR",
        "trunk_flexion_delta_deg": "Trunk Flex Δ",
        "delayed_arm_ffc_deg": "Arm Delay FFC",
        "runup_speed_mps": "Run-up Speed",
        "thorax_lat_flex_br_deg": "LatFlex BR",
        "shoulder_counter_rotation_deg": "SCR",
    }
    if comps and ev.is_complete_delivery:
        ys = np.arange(len(comps))
        z_vals = [c.z_score for c in comps]
        z_los = [(c.ci_low - c.cohort_mean) / max(c.cohort_sd, 1e-6) for c in comps]
        z_his = [(c.ci_high - c.cohort_mean) / max(c.cohort_sd, 1e-6) for c in comps]
        ax6.axvspan(-1.0, 1.0, color="#22c55e", alpha=0.10, label="Cohort +-1 SD")
        ax6.axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1.0)
        for idx_c, c in enumerate(comps):
            col_c = "#38bdf8" if c.definition_match == "exact" else "#fbbf24"
            ax6.plot([z_los[idx_c], z_his[idx_c]], [idx_c, idx_c], color=col_c, linewidth=2.5, alpha=0.85)
            ax6.scatter([z_vals[idx_c]], [idx_c], color="#ffffff", edgecolor=col_c, s=42, zorder=5)
        ax6.set_yticks(ys)
        ax6.set_yticklabels(
            [
                f"{short_names.get(c.metric_key, c.metric_key)}\n({'exact' if c.definition_match == 'exact' else 'proxy'})"
                for c in comps
            ],
            fontsize=7.8,
        )
        ax6.set_ylim(-0.6, max(len(comps) - 0.4, 0.6))
        ax6.set_xlim(-3.5, 3.5)
        ax6.legend(loc="lower right", fontsize=7.0, facecolor="#161c26", labelcolor=text_color)
        _style_ax(ax6, f"Cohort Z-Scores ({fact_sheet.interval_type})", "Z-Score vs Published Cohort", "")
    else:
        msg = (
            "INCOMPLETE DELIVERY CLIP\nNo verified ball release;\nmetrics suppressed"
            if not ev.is_complete_delivery
            else "No cohort-matched metrics\nobservable from this view/FPS"
        )
        ax6.text(0.5, 0.5, msg, transform=ax6.transAxes, ha="center", va="center", color="#fbbf24", fontsize=9.0, fontweight="bold")
        _style_ax(ax6, "Cohort Comparison & Observability Status", "", "")

    fps_title = (
        f"{fact_sheet.audit.fps:.1f} nominal fps [UNVERIFIED TIMEBASE: {fact_sheet.audit.timebase_status}]"
        if fact_sheet.audit.timebase_unreliable
        else f"{fact_sheet.audit.fps:.1f} unique fps ({fact_sheet.audit.timebase_status})"
    )
    fig.suptitle(
        f"Cricket Fast-Bowling Biomechanics — {fact_sheet.bowler_id} "
        f"[{fact_sheet.bowling_arm.value.upper()} ARM | {fact_sheet.action_class.value.upper()} | "
        f"{fps_title} | {fact_sheet.calibration.method}]",
        color="#ffffff", fontsize=11.4, fontweight="bold", y=0.98,
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return out


def render_annotated_freeze_video(
    video_path: str | Path,
    start_frame: int,
    joints2d_px: np.ndarray,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
    freeze_seconds: float = 1.1,
    slow_factor: float = 0.5,
    source_frame_indices: Optional[List[int]] = None,
) -> Path:
    """Export annotated H.264 freeze-frame analysis video strictly aligned to `source_frame_indices`."""
    src_map = source_frame_indices or fact_sheet.audit.source_frame_indices
    n = len(joints2d_px)
    abs_list = [int(src_map[i]) for i in range(n)] if (src_map and len(src_map) == n) else [int(start_frame + i) for i in range(n)]
    decoded = _read_exact_frames_sequential(video_path, abs_list)

    first_frame = decoded.get(abs_list[0])
    if first_frame is None:
        raise RuntimeError(f"Could not read frames from {video_path} for annotated freeze video.")
    h, w = first_frame.shape[:2]

    out_fps = max(15.0, min(30.0, fact_sheet.audit.fps * slow_factor))
    freeze_n = max(6, int(round(freeze_seconds * out_fps)))
    ev = fact_sheet.events
    m = fact_sheet.metrics

    def _fmt(key: str, short: str) -> str:
        me = m.get(key)
        if me is None or me.suppressed or me.value is None:
            return f"{short}: [Suppressed]"
        if me.ci_low is not None and me.ci_high is not None:
            return f"{short}: {me.value:.1f} [{me.ci_low:.1f}, {me.ci_high:.1f}] {me.unit}"
        return f"{short}: {me.value:.1f} {me.unit}"

    freeze_map: Dict[int, Tuple[str, List[str]]] = {}
    if ev.is_complete_delivery:
        freeze_map[ev.bfc.frame_idx] = ("PAUSE: Back Foot Contact (BFC)", [_fmt("runup_speed_mps", "Run-up"), _fmt("rear_hip_bfc_deg", "Rear Hip")])
        freeze_map[ev.ffc.frame_idx] = ("PAUSE: Front Foot Contact (FFC)", [_fmt("front_knee_ffc_deg", "Front Knee"), _fmt("delayed_arm_ffc_deg", "Arm Delay")])
        freeze_map[ev.br.frame_idx] = (
            "PAUSE: Ball Release (BR)",
            [_fmt("front_knee_br_deg", "Front Knee BR"), _fmt("thorax_lat_flex_br_deg", "Lat Tilt BR"), "ICC Elbow: REFUSED (Lab 3D Req)"],
        )

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = out.with_suffix(".raw_mp4v.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_mp4), fourcc, out_fps, (w, h))

    fps_tag = "nominal fps [UNVERIFIED]" if fact_sheet.audit.timebase_unreliable else "unique fps"
    for rel_idx, abs_f in enumerate(abs_list):
        frame = decoded.get(abs_f)
        if frame is None:
            continue
        j2 = joints2d_px[rel_idx]
        if rel_idx in freeze_map:
            title, lines = freeze_map[rel_idx]
            sub = f"t={rel_idx / max(fact_sheet.audit.fps, 1.0):.3f}s | {fact_sheet.calibration.view_class.value}"
            ann = draw_skeleton_on_frame(frame, j2, title, sub, lines)
            for _ in range(freeze_n):
                writer.write(ann)
        else:
            title = f"{fact_sheet.bowler_id} ({fact_sheet.bowling_arm.value}-arm | {fact_sheet.audit.fps:.1f} {fps_tag})"
            sub = f"u={rel_idx} (src={abs_f}) | ICC Elbow Legality: REFUSED"
            ann = draw_skeleton_on_frame(frame, j2, title, sub, [])
            writer.write(ann)
    writer.release()

    # Transcode mp4v -> browser-safe H.264 (yuv420p + faststart) when ffmpeg is available (P2-1)
    if shutil.which("ffmpeg"):
        res = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_mp4),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
            ],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            tmp_mp4.unlink(missing_ok=True)
            return out
    tmp_mp4.replace(out)
    return out


def render_truthful_breakdown_reel_9x16(
    video_path: str | Path,
    start_frame: int,
    joints2d_px: np.ndarray,
    joints_world_m: np.ndarray,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
    freeze_seconds: float = 1.25,
    source_frame_indices: Optional[List[int]] = None,
) -> Path:
    """Phase 2 (M2D): 1-click 1080x1920 (9:16) Truthful Breakdown Reel MP4 (browser-safe H.264).

    Combines top video viewport with synchronized 2D skeleton, bottom dark biomechanical telemetry card
    (live mini-3D side/rear wireframe + verified `[ci_low, ci_high]` readouts), auto-freezes at
    BFC / FFC / UAH / BR, and preserves the permanent non-truncating ICC 15-degree refusal banner.
    """
    W_OUT, H_OUT = 1080, 1920
    VIDEO_H = 1120
    src_map = source_frame_indices or fact_sheet.audit.source_frame_indices
    n = len(joints2d_px)
    abs_list = [int(src_map[i]) for i in range(n)] if (src_map and len(src_map) == n) else [int(start_frame + i) for i in range(n)]
    decoded = _read_exact_frames_sequential(video_path, abs_list) if Path(video_path).exists() else {}

    out_fps = 25.0
    freeze_n = max(8, int(round(freeze_seconds * out_fps)))
    ev = fact_sheet.events
    m = fact_sheet.metrics

    def _fmt(key: str, short: str) -> str:
        me = m.get(key)
        if me is None or me.suppressed or me.value is None:
            return f"{short}: [Suppressed]"
        return f"{short}: {me.value:.1f} [{me.ci_low:.1f}, {me.ci_high:.1f}] {me.unit} ({me.observability.value})"

    freeze_map: Dict[int, Tuple[str, List[str]]] = {}
    if ev.is_complete_delivery:
        freeze_map[ev.bfc.frame_idx] = (
            "PHASE 1: BACK FOOT CONTACT (BFC)",
            [_fmt("pelvis_alignment_bfc_deg", "Pelvis BFC"), _fmt("shoulder_alignment_bfc_deg", "Shoulder BFC"), _fmt("runup_speed_mps", "Run-up")],
        )
        freeze_map[ev.ffc.frame_idx] = (
            "PHASE 2: FRONT FOOT CONTACT (FFC)",
            [_fmt("front_knee_ffc_deg", "Front Knee"), _fmt("thorax_lat_flex_ffc_deg", "Lat Tilt FFC"), _fmt("pelvic_drop_ffc_deg", "Pelvic Drop")],
        )
        if ev.uah is not None:
            freeze_map[ev.uah.frame_idx] = (
                "PHASE 3: UPPER ARM HORIZONTAL (UAH)",
                [_fmt("shoulder_counter_rotation_deg", "SCR"), _fmt("max_hip_shoulder_sep_deg", "Peak Hip-Sho Sep")],
            )
        freeze_map[ev.br.frame_idx] = (
            "PHASE 4: BALL RELEASE (BR)",
            [_fmt("front_knee_br_deg", "Front Knee BR"), _fmt("thorax_lat_flex_br_deg", "Lat Tilt BR"), _fmt("release_height_m", "Release Height")],
        )

    f_ank_i = JOINT_IDX["lank" if fact_sheet.bowling_arm.value == "right" else "rank"]
    y_ground = float(np.percentile(joints_world_m[:, f_ank_i, 1], 10)) if len(joints_world_m) > 0 else 0.0

    def _compose_frame(rel_idx: int, abs_f: int, banner: str, metric_lines: List[str]) -> np.ndarray:
        canvas = np.full((H_OUT, W_OUT, 3), (15, 19, 26), dtype=np.uint8)
        raw = decoded.get(abs_f)
        j2 = joints2d_px[rel_idx]
        if raw is not None:
            rh, rw = raw.shape[:2]
            scale = min(W_OUT / max(rw, 1), VIDEO_H / max(rh, 1))
            nw, nh = int(round(rw * scale)), int(round(rh * scale))
            resized = cv2.resize(raw, (nw, nh), interpolation=cv2.INTER_AREA)
            ox, oy = (W_OUT - nw) // 2, 110 + (VIDEO_H - nh) // 2
            canvas[oy : oy + nh, ox : ox + nw] = resized
            src_w = max(int(fact_sheet.audit.width or rw), 1)
            src_h = max(int(fact_sheet.audit.height or rh), 1)
            sx = nw / float(src_w)
            sy = nh / float(src_h)
            j2_s = j2.copy()
            j2_s[:, 0] = j2[:, 0] * sx + ox
            j2_s[:, 1] = j2[:, 1] * sy + oy
            for a_name, b_name, bgr, _ in SKELETON_EDGES:
                pa, pb = j2_s[JOINT_IDX[a_name]], j2_s[JOINT_IDX[b_name]]
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    cv2.line(canvas, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), bgr, 3, cv2.LINE_AA)
            for pt in j2_s:
                if np.all(np.isfinite(pt)):
                    cv2.circle(canvas, (int(pt[0]), int(pt[1])), 4, (255, 255, 255), -1, cv2.LINE_AA)

        # Header bar
        cv2.rectangle(canvas, (0, 0), (W_OUT, 108), (22, 28, 38), -1)
        fps_str = (
            f"{fact_sheet.audit.fps:.1f} nom. fps [UNVERIFIED]"
            if fact_sheet.audit.timebase_unreliable
            else f"{fact_sheet.audit.fps:.1f} unique fps"
        )
        cv2.putText(canvas, f"{fact_sheet.bowler_id} ({fact_sheet.bowling_arm.value.upper()}-ARM)", (36, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (245, 250, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{fps_str} | {fact_sheet.calibration.method} | {fact_sheet.interval_type}", (36, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (160, 185, 215), 1, cv2.LINE_AA)

        # Bottom telemetry panel (mini 3D rear view on right + verified metrics on left)
        y_panel = 1245
        cv2.rectangle(canvas, (24, y_panel), (W_OUT - 24, H_OUT - 24), (22, 28, 38), -1)
        cv2.putText(canvas, banner, (48, y_panel + 52), cv2.FONT_HERSHEY_SIMPLEX, 0.92, (56, 189, 248), 2, cv2.LINE_AA)

        y_txt = y_panel + 108
        for ln in metric_lines[:5]:
            col = (100, 196, 255) if "[Suppressed" in ln else (235, 245, 255)
            cv2.putText(canvas, ln, (48, y_txt), cv2.FONT_HERSHEY_SIMPLEX, 0.72, col, 2, cv2.LINE_AA)
            y_txt += 48

        # Synchronized 3D mini coronal wireframe box on bottom-right
        bx0, by0, bx1, by1 = 720, y_panel + 75, W_OUT - 48, H_OUT - 140
        cv2.rectangle(canvas, (bx0, by0), (bx1, by1), (15, 19, 26), -1)
        cv2.putText(canvas, "3D Pitch-World Rear", (bx0 + 18, by0 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (159, 176, 200), 1, cv2.LINE_AA)
        if rel_idx < len(joints_world_m):
            pts3 = joints_world_m[rel_idx]
            zc = float(pts3[JOINT_IDX["pelv"], 2])
            px_scale = 175.0
            cx_3d = (bx0 + bx1) // 2
            cy_base = by1 - 36
            for a_name, b_name, bgr, _ in SKELETON_EDGES:
                pa, pb = pts3[JOINT_IDX[a_name]], pts3[JOINT_IDX[b_name]]
                p1 = (int(cx_3d + (pa[2] - zc) * px_scale), int(cy_base - (pa[1] - y_ground) * px_scale))
                p2 = (int(cx_3d + (pb[2] - zc) * px_scale), int(cy_base - (pb[1] - y_ground) * px_scale))
                cv2.line(canvas, p1, p2, bgr, 3, cv2.LINE_AA)

        # Permanent non-truncating scientific guardrails footer
        cv2.rectangle(canvas, (24, H_OUT - 122), (W_OUT - 24, H_OUT - 24), (42, 24, 24), -1)
        cv2.putText(canvas, "ICC 15-DEGREE ELBOW LEGALITY: PERMANENTLY REFUSED (Lab >=250 Hz 3D Required)", (42, H_OUT - 82), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (120, 170, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Single-delivery mechanical stress proxy only — NOT a clinical injury diagnosis.", (42, H_OUT - 44), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (195, 205, 220), 1, cv2.LINE_AA)
        return canvas

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_mp4 = out.with_suffix(".reel_raw.mp4")
    writer = cv2.VideoWriter(str(tmp_mp4), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (W_OUT, H_OUT))

    default_lines = [
        _fmt("thorax_lat_flex_br_deg", "Lat Tilt BR"),
        _fmt("shoulder_counter_rotation_deg", "SCR"),
        _fmt("front_knee_br_deg", "Front Knee BR"),
        f"Frame u={0} | Action: {fact_sheet.action_class.value}",
    ]
    for rel_idx, abs_f in enumerate(abs_list):
        if rel_idx in freeze_map:
            title, lines = freeze_map[rel_idx]
            frame_out = _compose_frame(rel_idx, abs_f, title, lines)
            for _ in range(freeze_n):
                writer.write(frame_out)
        else:
            lines_cur = default_lines[:3] + [f"Frame u={rel_idx} (src={abs_f}) | Action: {fact_sheet.action_class.value}"]
            writer.write(_compose_frame(rel_idx, abs_f, "LIVE DELIVERY TRACKING", lines_cur))
    writer.release()

    if shutil.which("ffmpeg"):
        res = subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", str(tmp_mp4),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out),
            ],
            capture_output=True,
            check=False,
        )
        if res.returncode == 0 and out.exists() and out.stat().st_size > 0:
            tmp_mp4.unlink(missing_ok=True)
            return out
    tmp_mp4.replace(out)
    return out
