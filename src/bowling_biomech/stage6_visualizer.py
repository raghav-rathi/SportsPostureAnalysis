"""Stage 6A: Truthful Mathematical Visuals, Freeze-Frame Panels, Virtual 3D Views, & Sparklines.

Renders strictly measurement-grounded visual explanations (never generative body-warping art):
  1. Readable key-phase annotated freeze-frame strip (resize-before-draw for portrait 9:16 legibility)
  2. Synchronized kinematic waveforms with suppressed channels visually dimmed (never contradicting FactSheet)
  3. Virtual 3D canonical projections (Side-On, Rear Coronal)
  4. 2D Performance-Durability Quadrant with uncertainty ellipse & non-clipping labels
  5. Automatic H.264 freeze-frame slow-motion video export
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

from .schemas import BiomechanicalFactSheet
from .stage3_kinematics import KinematicsResult
from .synthetic_ground_truth import JOINT_IDX

SKELETON_EDGES: List[Tuple[str, str, Tuple[int, int, int], str]] = [
    ("pelv", "spi1", (210, 60, 210), "#d23cd2"),
    ("spi1", "spi2", (210, 60, 210), "#d23cd2"),
    ("spi2", "spi3", (210, 60, 210), "#d23cd2"),
    ("spi3", "neck", (210, 60, 210), "#d23cd2"),
    ("neck", "head", (210, 60, 210), "#d23cd2"),
    ("lhip", "rhip", (255, 220, 40), "#28dcff"),
    ("lsho", "rsho", (40, 220, 255), "#ffdc28"),
    ("lhip", "lkne", (240, 210, 40), "#28d2f0"),
    ("lkne", "lank", (240, 210, 40), "#28d2f0"),
    ("lank", "ltoe", (240, 210, 40), "#28d2f0"),
    ("rhip", "rkne", (40, 165, 255), "#ffa528"),
    ("rkne", "rank", (40, 165, 255), "#ffa528"),
    ("rank", "rtoe", (40, 165, 255), "#ffa528"),
    ("neck", "lsho", (255, 140, 60), "#3c8cff"),
    ("lsho", "lelb", (255, 140, 60), "#3c8cff"),
    ("lelb", "lwri", (255, 140, 60), "#3c8cff"),
    ("neck", "rsho", (60, 235, 90), "#5aeb3c"),
    ("rsho", "relb", (60, 235, 90), "#5aeb3c"),
    ("relb", "rwri", (60, 235, 90), "#5aeb3c"),
]


def _prepare_readable_canvas(
    frame_bgr: np.ndarray,
    joints2d_px: np.ndarray,
    target_h: int = 560,
    min_w: int = 520,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resize BEFORE drawing overlays and pad narrow portrait clips so HUD text stays crisp & legible."""
    h, w = frame_bgr.shape[:2]
    scale = target_h / max(h, 1)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(frame_bgr, (new_w, target_h))
    j2_scaled = joints2d_px * scale

    if new_w < min_w:
        pad_total = min_w - new_w
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        resized = cv2.copyMakeBorder(resized, 0, 0, pad_l, pad_r, cv2.BORDER_CONSTANT, value=(15, 18, 24))
        j2_scaled = j2_scaled.copy()
        j2_scaled[:, 0] += pad_l
    return resized, j2_scaled


def draw_skeleton_on_frame(
    frame_bgr: np.ndarray,
    joints2d_px: np.ndarray,
    title: str,
    subtitle: str,
    callout_lines: List[str],
    draw_skeleton: bool = True,
) -> np.ndarray:
    """Draw mathematically accurate 2D reprojected skeleton & crisp verified metric callouts onto a canvas."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    thickness = max(2, int(round(min(h, w) / 240.0)))
    radius = max(3, int(round(min(h, w) / 170.0)))

    if draw_skeleton:
        for a_name, b_name, bgr, _ in SKELETON_EDGES:
            pa = joints2d_px[JOINT_IDX[a_name]]
            pb = joints2d_px[JOINT_IDX[b_name]]
            if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
                continue
            p1 = (int(round(pa[0])), int(round(pa[1])))
            p2 = (int(round(pb[0])), int(round(pb[1])))
            cv2.line(img, p1, p2, bgr, thickness, cv2.LINE_AA)

        for j in range(min(22, joints2d_px.shape[0])):
            pt = joints2d_px[j]
            if np.all(np.isfinite(pt)):
                cv2.circle(img, (int(round(pt[0])), int(round(pt[1]))), radius, (255, 255, 255), -1, cv2.LINE_AA)

    banner_h = max(58, int(0.105 * h))
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (16, 20, 27), -1)
    cv2.addWeighted(overlay, 0.84, img, 0.16, 0, img)

    title_scale = float(np.clip(w / 950.0, 0.56, 0.92))
    sub_scale = float(np.clip(w / 1280.0, 0.44, 0.68))
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


def render_keyframe_panel(
    video_path: str | Path,
    start_frame: int,
    joints2d_px: np.ndarray,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
) -> Path:
    """Render readable 2x2 key-phase freeze-frame strip (BFC, FFC, UAH, BR) with verified metrics."""
    ev = fact_sheet.events
    m = fact_sheet.metrics

    def _fmt(key: str, short: str) -> str:
        me = m.get(key)
        if me is None or me.suppressed or me.value is None:
            return f"{short}: [Suppressed - {me.observability.value if me else 'N/A'}]"
        half = (me.ci_high - me.ci_low) / 2.0 if (me.ci_high is not None and me.ci_low is not None) else 0.0
        return f"{short}: {me.value:.1f} +- {half:.1f} {me.unit} ({me.observability.value})"

    phases = [
        ("1. Back Foot Contact (BFC)", ev.bfc, [_fmt("runup_speed_mps", "Run-up"), _fmt("rear_hip_bfc_deg", "Rear Hip")]),
        (
            "2. Front Foot Contact (FFC)",
            ev.ffc,
            [_fmt("front_knee_ffc_deg", "Front Knee"), _fmt("delayed_arm_ffc_deg", "Arm Delay"), _fmt("thorax_lat_flex_ffc_deg", "Lat Bend")],
        ),
        (
            "3. Upper Arm Horizontal (UAH)",
            ev.uah,
            [_fmt("shoulder_counter_rotation_deg", "SCR"), "ICC Elbow: REFUSED (Lab 3D Req)"],
        ),
        (
            "4. Ball Release (BR)" if ev.is_complete_delivery else "4. Ball Release (NOT IN CLIP)",
            ev.br if ev.is_complete_delivery else None,
            [_fmt("front_knee_br_deg", "Front Knee"), _fmt("trunk_flexion_delta_deg", "Trunk Flex d"), _fmt("thorax_lat_flex_br_deg", "Lat Bend BR")],
        ),
    ]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video for keyframe rendering: {video_path}")

    panels: List[np.ndarray] = []
    target_h = 560
    min_w = 560
    canvas_w = None

    for title, evt, lines in phases:
        if evt is None:
            w_use = canvas_w or min_w
            blank = np.full((target_h, w_use, 3), (22, 26, 34), dtype=np.uint8)
            cv2.putText(
                blank, "Phase not observed in clip", (w_use // 6, target_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (90, 190, 255), 2, cv2.LINE_AA,
            )
            annotated = draw_skeleton_on_frame(
                blank, np.zeros((24, 2)), title, "Truncated clip or phase outside [FFC, BR)", lines, draw_skeleton=False
            )
            panels.append(annotated)
            continue

        abs_frame = start_frame + evt.frame_idx
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ok, frame = cap.read()
        if not ok or frame is None:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        j2 = joints2d_px[min(evt.frame_idx, len(joints2d_px) - 1)]
        canvas, j2_s = _prepare_readable_canvas(frame, j2, target_h=target_h, min_w=min_w)
        canvas_w = canvas.shape[1]
        sub = f"t={evt.timestamp_s:.3f}s (frame {evt.frame_idx}, +- {evt.timing_uncertainty_ms:.1f} ms)"
        annotated = draw_skeleton_on_frame(canvas, j2_s, title, sub, lines, draw_skeleton=True)
        panels.append(annotated)
    cap.release()

    # Ensure identical width across all 4 panels before stacking
    max_w = max(p.shape[1] for p in panels)
    norm_panels = []
    for p in panels:
        if p.shape[1] != max_w:
            p = cv2.resize(p, (max_w, target_h))
        norm_panels.append(p)

    grid = np.vstack([np.hstack(norm_panels[:2]), np.hstack(norm_panels[2:])])
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), grid)
    return out


def render_biomechanics_dashboard(
    joints_world: np.ndarray,
    kin: KinematicsResult,
    fact_sheet: BiomechanicalFactSheet,
    out_path: str | Path,
) -> Path:
    """Render a 6-panel scientific biomechanics dashboard with suppressed channels dimmed."""
    fps = max(fact_sheet.audit.fps, 1.0)
    n = joints_world.shape[0]
    t_ms = (np.arange(n) - fact_sheet.events.ffc.frame_float) / fps * 1000.0
    ev = fact_sheet.events

    fig = plt.figure(figsize=(16, 10), facecolor="#0f131a")
    gs = fig.add_gridspec(2, 3, hspace=0.36, wspace=0.28)
    title_color = "#f2f5f8"
    text_color = "#cfd8e3"
    grid_color = "#283241"

    def _style_ax(ax, title: str, xlabel: str, ylabel: str):
        ax.set_facecolor("#161c26")
        ax.set_title(title, color=title_color, fontsize=10.5, fontweight="bold", pad=8)
        ax.set_xlabel(xlabel, color=text_color, fontsize=9)
        ax.set_ylabel(ylabel, color=text_color, fontsize=9)
        ax.tick_params(colors=text_color, labelsize=8)
        for sp in ax.spines.values():
            sp.set_color("#344054")
        ax.grid(True, color=grid_color, linestyle="--", alpha=0.55)

    def _mark_events(ax):
        marks = [
            ("BFC", (ev.bfc.frame_float - ev.ffc.frame_float) / fps * 1000.0, "#f59e0b", 0.94),
            ("FFC", 0.0, "#38bdf8", 0.84),
        ]
        if ev.is_complete_delivery:
            marks.append(("BR", (ev.br.frame_float - ev.ffc.frame_float) / fps * 1000.0, "#22c55e", 0.74))
        ymin, ymax = ax.get_ylim()
        span = max(ymax - ymin, 1e-3)
        for lbl, x_ms, col, y_frac in marks:
            ax.axvline(x_ms, color=col, linestyle=":", linewidth=1.4, alpha=0.9)
            ax.text(
                x_ms + 5,
                ymin + y_frac * span,
                lbl,
                color=col,
                fontsize=8,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", facecolor="#0f131a", edgecolor="none", alpha=0.72),
            )

    # 1. Front Knee Angle Waveform (dimmed if suppressed!)
    ax1 = fig.add_subplot(gs[0, 0])
    fk_m = fact_sheet.metrics["front_knee_br_deg"]
    fk_sup = fk_m.suppressed
    ax1.plot(
        t_ms,
        kin.series["front_knee_deg"],
        color="#38bdf8",
        linewidth=1.5 if fk_sup else 2.2,
        linestyle="--" if fk_sup else "-",
        alpha=0.22 if fk_sup else 1.0,
    )
    ax1.axhspan(158, 178, color="#22c55e", alpha=0.10, label="Elite Braced Zone (158-178 deg)")
    _style_ax(
        ax1,
        f"Front Knee Angle ({'SUPPRESSED' if fk_sup else fk_m.observability.value.upper()})",
        "Time relative to FFC (ms)",
        "Internal Angle (deg, 180=straight)",
    )
    _mark_events(ax1)
    if fk_sup:
        ax1.text(
            0.5, 0.45, "SUPPRESSED FROM THIS VIEW / CROP\n(Curve dimmed; not used in scoring)",
            transform=ax1.transAxes, ha="center", va="center", color="#fbbf24", fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#0f131a", edgecolor="#fbbf24", alpha=0.88),
        )
    ax1.legend(loc="lower left", fontsize=7.5, facecolor="#161c26", labelcolor=text_color)

    # 2. Thoracolumbar Kinematics (dim suppressed curve independently!)
    ax2 = fig.add_subplot(gs[0, 1])
    tf_sup = fact_sheet.metrics["trunk_flexion_delta_deg"].suppressed
    lf_sup = fact_sheet.metrics["thorax_lat_flex_br_deg"].suppressed
    ax2.plot(
        t_ms, kin.series["trunk_flex_deg"], color="#a855f7",
        linewidth=1.4 if tf_sup else 2.1, linestyle="--" if tf_sup else "-", alpha=0.20 if tf_sup else 1.0,
        label=f"Sagittal Forward Flexion{' (suppressed)' if tf_sup else ''}",
    )
    ax2.plot(
        t_ms, kin.series["thorax_lat_flex_deg"], color="#fb7185",
        linewidth=1.4 if lf_sup else 2.1, linestyle="--" if lf_sup else "-", alpha=0.20 if lf_sup else 1.0,
        label=f"Contralateral Lateral Flexion{' (suppressed)' if lf_sup else ''}",
    )
    ax2.axhspan(32, 44, color="#22c55e", alpha=0.10, label="Uninjured Elite Lat-Flex (Bayne 2016)")
    ax2.axhline(48, color="#ef4444", linestyle="--", linewidth=1.0, alpha=0.75, label="Elevated Load (~48-50 deg)")
    _style_ax(ax2, "Thoracolumbar Kinematics (Z-X Cardan)", "Time relative to FFC (ms)", "Angle from Vertical (deg)")
    _mark_events(ax2)
    ax2.legend(loc="lower left", fontsize=7.0, facecolor="#161c26", labelcolor=text_color)

    # 3. Pelvis vs Shoulder Transverse Rotation (Unwrapped Portus Convention)
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
        f"Transverse Alignment (180=Side, 270=Front){' [LOW/SUP]' if scr_sup else ''}",
        "Time relative to FFC (ms)",
        "Unwrapped Alignment (deg)",
    )
    _mark_events(ax3)
    ax3.legend(loc="lower left", fontsize=7.5, facecolor="#161c26", labelcolor=text_color)

    # 4. Virtual 3D Side-On Projection at FFC & BR
    ax4 = fig.add_subplot(gs[1, 0])
    plot_phases = [(ev.ffc.frame_idx, 0.45)]
    if ev.is_complete_delivery:
        plot_phases.append((ev.br.frame_idx, 1.0))
    for idx, col_alpha in plot_phases:
        pts = joints_world[min(idx, n - 1)]
        for a_name, b_name, _, hex_c in SKELETON_EDGES:
            pa, pb = pts[JOINT_IDX[a_name]], pts[JOINT_IDX[b_name]]
            ax4.plot([pa[0], pb[0]], [pa[1], pb[1]], color=hex_c, alpha=col_alpha, linewidth=2.2)
    _style_ax(ax4, "Virtual 3D Pitch-World Side View (FFC faded -> BR solid)", "+X Down Pitch (m)", "+Y Height (m)")
    ax4.set_aspect("equal", adjustable="datalim")

    # 5. Virtual 3D Rear Coronal View at BR (or FFC if incomplete)
    ax5 = fig.add_subplot(gs[1, 1])
    rear_idx = ev.br.frame_idx if ev.is_complete_delivery else ev.ffc.frame_idx
    pts_rear = joints_world[min(rear_idx, n - 1)]
    for a_name, b_name, _, hex_c in SKELETON_EDGES:
        pa, pb = pts_rear[JOINT_IDX[a_name]], pts_rear[JOINT_IDX[b_name]]
        ax5.plot([pa[2], pb[2]], [pa[1], pb[1]], color=hex_c, linewidth=2.4)
    pelv_r = pts_rear[JOINT_IDX["pelv"]]
    ax5.axvline(pelv_r[2], color="#94a3b8", linestyle="--", linewidth=1.1, label="Gravity Plumb Line")
    _style_ax(
        ax5,
        f"Virtual 3D Rear View at {'BR' if ev.is_complete_delivery else 'FFC'} (Gravity-Aligned)",
        "+Z Across Pitch (m)",
        "+Y Height (m)",
    )
    ax5.set_aspect("equal", adjustable="datalim")
    ax5.legend(loc="upper right", fontsize=7.5, facecolor="#161c26", labelcolor=text_color)

    # 6. 2D Performance-Durability Quadrant with Uncertainty Ellipse
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.add_patch(Rectangle((70, 70), 30, 30, facecolor="#22c55e", alpha=0.10))
    ax6.add_patch(Rectangle((70, 0), 30, 70, facecolor="#f59e0b", alpha=0.08))
    ax6.add_patch(Rectangle((0, 70), 70, 30, facecolor="#38bdf8", alpha=0.08))
    ax6.add_patch(Rectangle((0, 0), 70, 70, facecolor="#ef4444", alpha=0.08))
    ax6.axvline(70, color="#64748b", linestyle="--", linewidth=1.0)
    ax6.axhline(70, color="#64748b", linestyle="--", linewidth=1.0)

    q = fact_sheet.quadrant
    if q.kinetic_efficiency_x is not None and q.durability_safety_y is not None:
        w_e = 2.0 * (q.kinetic_efficiency_ci or 8.0)
        h_e = 2.0 * (q.durability_safety_ci or 8.0)
        ell = Ellipse(
            (q.kinetic_efficiency_x, q.durability_safety_y),
            width=w_e, height=h_e, facecolor="#38bdf8", edgecolor="#e0f2fe", alpha=0.35, linewidth=1.8,
        )
        ax6.add_patch(ell)
        ax6.scatter([q.kinetic_efficiency_x], [q.durability_safety_y], color="#ffffff", s=45, zorder=5)
        ha = "right" if q.kinetic_efficiency_x > 72.0 else "left"
        dx = -3.0 if ha == "right" else 3.0
        dy = -5.0 if q.durability_safety_y > 90.0 else 3.0
        ax6.text(
            q.kinetic_efficiency_x + dx, q.durability_safety_y + dy,
            fact_sheet.bowler_id, ha=ha, color="#ffffff", fontsize=8.5, fontweight="bold",
        )
    else:
        ax6.text(
            50, 50, "Partial Observability\n(Add complementary angle\nfor full 2D placement)",
            ha="center", va="center", color="#fbbf24", fontsize=9, fontweight="bold",
        )
    ax6.set_xlim(0, 100)
    ax6.set_ylim(0, 100)
    _style_ax(ax6, "Performance vs. Durability Quadrant (95% Ellipse)", "Kinetic & Pace Efficiency (0-100)", "Lumbar Mechanical Durability (0-100)")

    fig.suptitle(
        f"Cricket Fast-Bowling Biomechanics Dashboard — {fact_sheet.bowler_id} "
        f"[{fact_sheet.bowling_arm.value.upper()} ARM | {fact_sheet.action_class.value.upper()} | "
        f"{fact_sheet.audit.fps:.0f} fps | View: {fact_sheet.calibration.view_class.value}]",
        color="#ffffff", fontsize=12.5, fontweight="bold", y=0.98,
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
    out_fps: float = 25.0,
    freeze_seconds: float = 1.4,
) -> Path:
    """Render an annotated slow-motion H.264 MP4 pausing automatically at BFC, FFC, UAH, and BR."""
    ev = fact_sheet.events
    m = fact_sheet.metrics

    def _fmt(key: str, short: str) -> str:
        me = m.get(key)
        if me is None or me.suppressed or me.value is None:
            return f"{short}: [{me.observability.value.upper() if me else 'N/A'}]"
        half = (me.ci_high - me.ci_low) / 2.0 if (me.ci_high is not None and me.ci_low is not None) else 0.0
        return f"{short}: {me.value:.1f} +- {half:.1f} {me.unit}"

    freeze_map = {
        ev.bfc.frame_idx: (
            "PHASE 1: BACK FOOT CONTACT (BFC)",
            [_fmt("runup_speed_mps", "Run-up"), _fmt("rear_hip_bfc_deg", "Rear Hip")],
        ),
        ev.ffc.frame_idx: (
            "PHASE 2: FRONT FOOT CONTACT (FFC)",
            [_fmt("front_knee_ffc_deg", "Front Knee"), _fmt("delayed_arm_ffc_deg", "Arm Delay"), _fmt("thorax_lat_flex_ffc_deg", "Lat Bend")],
        ),
    }
    if ev.uah is not None and ev.uah.frame_idx not in freeze_map:
        freeze_map[ev.uah.frame_idx] = (
            "PHASE 3: UPPER ARM HORIZONTAL (UAH)",
            [_fmt("shoulder_counter_rotation_deg", "SCR"), "ICC 15-deg Elbow Legality: REFUSED (Lab 3D Req)"],
        )
    if ev.is_complete_delivery:
        freeze_map[ev.br.frame_idx] = (
            "PHASE 4: BALL RELEASE (BR)",
            [_fmt("front_knee_br_deg", "Front Knee BR"), _fmt("trunk_flexion_delta_deg", "Trunk Flex Delta"), _fmt("thorax_lat_flex_br_deg", "Lat Bend BR")],
        )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w_even = w - (w % 2)
    h_even = h - (h % 2)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp_raw = out.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_raw), fourcc, out_fps, (w_even, h_even))

    freeze_reps = max(1, int(round(freeze_seconds * out_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for rel_idx in range(joints2d_px.shape[0]):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame.shape[1] != w_even or frame.shape[0] != h_even:
            frame = cv2.resize(frame, (w_even, h_even))

        if rel_idx in freeze_map:
            title, lines = freeze_map[rel_idx]
            sub = f"{fact_sheet.bowler_id} | View: {fact_sheet.calibration.view_class.value} | Frame {rel_idx}"
            ann = draw_skeleton_on_frame(frame, joints2d_px[rel_idx], title, sub, lines)
            for _ in range(freeze_reps):
                writer.write(ann)
        else:
            sub = f"View: {fact_sheet.calibration.view_class.value} | {fact_sheet.audit.fps:.0f} fps | Frame {rel_idx}"
            ann = draw_skeleton_on_frame(
                frame, joints2d_px[rel_idx], f"{fact_sheet.bowler_id} — 3D Biomechanical Tracking", sub, [],
            )
            writer.write(ann)

    cap.release()
    writer.release()

    cmd = [
        "ffmpeg", "-y", "-i", str(tmp_raw),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-movflags", "+faststart",
        str(out),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if proc.returncode == 0 and out.exists():
        tmp_raw.unlink(missing_ok=True)
    else:
        tmp_raw.rename(out)
    return out
