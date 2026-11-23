"""EXP-007 diagnostics on finished pipeline runs of real clips (CPU only).

Works from what `analyze_delivery_video` already wrote (`fact_sheet.json`, `trajectory_3d.npz`) plus the source
video, and asks questions the pipeline never asks of itself:

  * Is the timebase real? Counts repeated frames inside the bowler's box and reports their cadence. A repeat
    every Nth frame means frame-rate conversion; long runs mean slow motion or a freeze frame.
  * Is the detected ball release a release? Reports the bowling-arm angle and wrist height at the BR frame.
  * How large is the bowler in pixels, and how sharp does Stage 0 think the clip is?
  * What did the run actually deliver: reported metrics, pillar scores, action class, quadrant.
  * Optional: contact sheets of raw frames around the detected events, for visual checking.
  * Optional: camera focal length and distance from two visible wickets of known height, 20.12 m apart.

Example:
  python validation/audit_real_clip_diagnostics.py \
      --run steyn=outputs/v3_steyn:data/steyn.mp4 --run mcgrath=outputs/v3_mcgrath:data/reel.mp4 \
      --sheets-dir outputs/exp007_sheets --two-wickets steyn:85 --out outputs/exp007_real_clip_diagnostics.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from bowling_biomech.stage2_event_spotting import smooth_trajectory
from bowling_biomech.synthetic_ground_truth import JOINT_IDX as J

STUMP_HEIGHT_M = 0.7112   # Law 8
PITCH_LENGTH_M = 20.12    # Law 6


def frame_differences(video: Path, start: int, n: int, box: Tuple[int, int, int, int]) -> np.ndarray:
    """Mean absolute grey-level change between consecutive frames inside `box`."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    x0, y0, x1, y1 = box
    prev, out = None, []
    for _ in range(n):
        ok, frame = cap.read()
        if not ok:
            break
        grey = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            out.append(float(np.mean(np.abs(grey - prev))))
        prev = grey
    cap.release()
    return np.asarray(out)


def held_frame_report(diffs: np.ndarray) -> Dict:
    """A frame is 'held' when it changes by less than a quarter of the clip's median change (floor 0.35)."""
    moving = diffs[diffs > 0]
    threshold = max(0.25 * float(np.median(moving)) if moving.size else 0.0, 0.35)
    held = diffs < threshold
    idx = np.flatnonzero(held)
    gaps = np.diff(idx)
    runs, run = [], 0
    for h in held:
        run = run + 1 if h else 0
        runs.append(run)
    cadence = None
    if gaps.size >= 4:
        vals, counts = np.unique(gaps, return_counts=True)
        top = int(vals[np.argmax(counts)])
        if top > 1 and counts.max() / gaps.size >= 0.7:
            cadence = top
    return {
        "held_frame_ratio": round(float(held.mean()), 3),
        "longest_held_run_frames": int(max(runs)) if runs else 0,
        "regular_repeat_every_n_frames": cadence,
        "implied_unique_frames_per_container_second_factor": (round((cadence - 1) / cadence, 3) if cadence else None),
        "median_change_moving_frames": round(float(np.median(diffs[~held])), 2) if (~held).any() else None,
        "median_change_held_frames": round(float(np.median(diffs[held])), 3) if held.any() else None,
        "pattern_first_60": "".join("H" if h else "." for h in held[:60]),
    }


def release_sanity(fs: Dict, world: np.ndarray) -> Dict:
    """Arm angle in the sagittal plane (0 = vertical, negative = behind) and wrist height at the reported BR."""
    fps = fs["audit"]["fps"]
    side = "r" if fs["bowling_arm"] == "right" else "l"
    j = smooth_trajectory(world, fps, cutoff_hz=min(18.0, 0.38 * fps))
    arm = j[:, J[side + "wri"]] - j[:, J[side + "sho"]]
    phi = np.degrees(np.arctan2(arm[:, 0], arm[:, 1]))
    arm_len = float(np.percentile(np.linalg.norm(arm, axis=1), 95))
    br = int(round(fs["events"]["br"]["frame_float"]))
    return {
        "br_frame": fs["events"]["br"]["frame_float"],
        "br_method": fs["events"]["br"]["detection_method"],
        "br_confidence": fs["events"]["br"]["confidence"],
        "is_complete_delivery_flag": fs["events"]["is_complete_delivery"],
        "arm_angle_from_vertical_at_br_deg": round(float(phi[br]), 0),
        "wrist_above_shoulder_at_br_m": round(float(arm[br, 1]), 2),
        "wrist_above_shoulder_as_fraction_of_arm_length": round(float(arm[br, 1]) / arm_len, 2),
        "stride_ms": fs["events"]["delivery_stride_duration_ms"],
        "ffc_to_br_ms": fs["events"]["ffc_to_br_duration_ms"],
    }


def delivered_outputs(fs: Dict) -> Dict:
    metrics = {k: m for k, m in fs["metrics"].items() if k != "icc_elbow_legality_verdict"}
    reported = [k for k, m in metrics.items() if not m["suppressed"]]
    return {
        "metrics_reported": len(reported),
        "metrics_total": len(metrics),
        "reported_keys": reported,
        "action_class": fs["action_class"],
        "kinetic_score": fs["kinetic_efficiency"]["score"],
        "kinetic_contributors": fs["kinetic_efficiency"]["contributing_metrics"],
        "durability_score": fs["lumbar_load_screening"]["score"],
        "durability_contributors": fs["lumbar_load_screening"]["contributing_metrics"],
        "quadrant": fs["quadrant"]["quadrant_label"],
        "limiters": [c["title"] for c in fs["limiters"]],
        "strengths": [c["title"] for c in fs["strengths"]],
    }


def contact_sheet(video: Path, start: int, joints2d: np.ndarray, valid: np.ndarray, frames: List[int],
                  marks: Dict[int, str], out: Path, cell_h: int = 320, cols: int = 8, draw: bool = True) -> None:
    """Raw frames cropped to the bowler, with NLF's 2D joints (left limbs blue, right limbs red)."""
    cap = cv2.VideoCapture(str(video))
    width, height = int(cap.get(3)), int(cap.get(4))
    pts = joints2d[frames].reshape(-1, 2)
    x0, x1 = int(max(0, pts[:, 0].min() - 40)), int(min(width, pts[:, 0].max() + 40))
    y0, y1 = int(max(0, pts[:, 1].min() - 40)), int(min(height, pts[:, 1].max() + 40))
    centre = [("pelv", "neck"), ("neck", "head"), ("lsho", "rsho"), ("lhip", "rhip")]
    left = [("lhip", "lkne"), ("lkne", "lank"), ("lsho", "lelb"), ("lelb", "lwri")]
    right = [("rhip", "rkne"), ("rkne", "rank"), ("rsho", "relb"), ("relb", "rwri")]
    cells = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + f)
        ok, img = cap.read()
        if not ok:
            img = np.zeros((height, width, 3), np.uint8)
        if draw:
            p = joints2d[f]
            for group, colour in ((centre, (255, 255, 255)), (left, (255, 120, 0)), (right, (0, 0, 255))):
                for a, b in group:
                    cv2.line(img, tuple(map(int, p[J[a]])), tuple(map(int, p[J[b]])), colour, 2, cv2.LINE_AA)
        crop = img[y0:y1, x0:x1]
        scale = cell_h / crop.shape[0]
        crop = cv2.resize(crop, (int(crop.shape[1] * scale), cell_h), interpolation=cv2.INTER_CUBIC)
        label = f"{f}" + ("" if valid[f] else " (interp)") + (f" {marks[f]}" if f in marks else "")
        cv2.rectangle(crop, (0, 0), (crop.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(crop, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cells.append(crop)
    cap.release()
    w = max(c.shape[1] for c in cells)
    cells = [cv2.copyMakeBorder(c, 0, 0, 0, w - c.shape[1], cv2.BORDER_CONSTANT) for c in cells]
    rows = []
    for i in range(0, len(cells), cols):
        row = cells[i : i + cols] + [np.zeros_like(cells[0])] * (cols - len(cells[i : i + cols]))
        rows.append(np.hstack(row))
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])


def two_wicket_geometry(video: Path, frame_idx: int, image_width: int, image_height: int) -> Optional[Dict]:
    """Focal length and camera distance from two wickets of regulation height, one pitch length apart.

    Detects tall saturated-yellow posts (training stumps), groups them into a near and a far wicket, and uses
    apparent height h = f * H / Z at both ends: Z_near = L / (h_near / h_far - 1), f = h_near * Z_near / H.
    Assumes the pitch runs roughly along the optical axis, which holds for a camera behind the bowler.
    """
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, img = cap.read()
    cap.release()
    if not ok:
        return None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.morphologyEx(cv2.inRange(hsv, (18, 120, 120), (35, 255, 255)), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    posts = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)
             if stats[i][3] > 60 and stats[i][3] / max(stats[i][2], 1) > 3.0 and stats[i][4] > 300]
    if len(posts) < 2:
        return None
    posts.sort(key=lambda p: p[3])
    split = int(np.argmax(np.diff([p[3] for p in posts]))) + 1
    far, near = posts[:split], posts[split:]
    if not far or not near:
        return None
    h_near, h_far = max(p[3] for p in near), max(p[3] for p in far)
    base_near = float(np.mean([p[1] + p[3] for p in near]))
    base_far = float(np.mean([p[1] + p[3] for p in far]))

    def solve(hn: float, hf: float) -> Dict[str, float]:
        z_near = PITCH_LENGTH_M / (hn / hf - 1.0)
        focal = hn * z_near / STUMP_HEIGHT_M
        ratio = (z_near + PITCH_LENGTH_M) / z_near
        horizon_y = (ratio * base_far - base_near) / (ratio - 1.0)
        return {
            "camera_to_near_wicket_m": z_near, "focal_px": focal,
            "fov_on_larger_side_deg": 2.0 * math.degrees(math.atan(0.5 * max(image_width, image_height) / focal)),
            "camera_height_m": (base_near - horizon_y) * z_near / focal,
            "camera_pitch_deg_negative_is_down": -math.degrees(math.atan((0.5 * image_height - horizon_y) / focal)),
        }

    best = solve(h_near, h_far)
    band = [solve(h_near + dn, h_far + df) for dn in (-4, 4) for df in (-4, 4)]
    out = {k: round(v, 2) for k, v in best.items()}
    out.update(
        frame=frame_idx, near_post_heights_px=[p[3] for p in near], far_post_heights_px=[p[3] for p in far],
        focal_px_range_for_4px_height_error=[round(min(b["focal_px"] for b in band)), round(max(b["focal_px"] for b in band))],
        distance_m_range_for_4px_height_error=[
            round(min(b["camera_to_near_wicket_m"] for b in band), 1), round(max(b["camera_to_near_wicket_m"] for b in band), 1)
        ],
    )
    if len(near) >= 2:
        xs = sorted(p[0] + 0.5 * p[2] for p in near)
        out["near_post_spacing_m_check"] = round((xs[-1] - xs[0]) * best["camera_to_near_wicket_m"] / best["focal_px"], 3)
        out["regulation_off_to_leg_stump_centre_spacing_m"] = [0.191, 0.194]
    return out


def diagnose(name: str, run_dir: Path, video: Path, sheets_dir: Optional[Path]) -> Dict:
    fs = json.loads((run_dir / "fact_sheet.json").read_text(encoding="utf-8"))
    data = np.load(run_dir / "trajectory_3d.npz")
    j2, world, cam, valid = data["joints2d_px"], data["joints3d_world_m"], data["joints3d_cam_m"], data["valid_mask"]
    width, height, n = fs["audit"]["width"], fs["audit"]["height"], len(j2)
    start = fs["metadata"]["start_frame"]
    pts = j2.reshape(-1, 2)
    box = (int(max(0, pts[:, 0].min() - 20)), int(max(0, pts[:, 1].min() - 20)),
           int(min(width, pts[:, 0].max() + 20)), int(min(height, pts[:, 1].max() + 20)))
    pelvis = cam[valid][:, J["pelv"]]
    out = {
        "clip": {
            "frames": n, "fps_assumed": fs["audit"]["fps"], "width": width, "height": height,
            "view_class": fs["calibration"]["view_class"], "is_panning_camera": fs["calibration"]["is_panning_camera"],
            "background_pan_fraction": fs["metadata"].get("background_pan_fraction"),
            "blur_grade": fs["audit"]["blur_grade"], "median_laplacian_var": fs["audit"]["median_laplacian_var"],
            "bowler_height_px_median": round(float(np.median(j2[:, :, 1].max(1) - j2[:, :, 1].min(1)))),
            "pelvis_depth_median_m": round(float(np.median(pelvis[:, 2])), 2),
            "pelvis_depth_range_m": round(float(np.ptp(pelvis[:, 2])), 2),
        },
        "timebase": held_frame_report(frame_differences(video, start, n, box)),
        "release": release_sanity(fs, world),
        "delivered": delivered_outputs(fs),
    }
    if sheets_dir is not None:
        ev = fs["events"]
        marks = {int(round(ev["bfc"]["frame_float"])): "BFC", int(round(ev["ffc"]["frame_float"])): "FFC",
                 int(round(ev["br"]["frame_float"])): "BR"}
        around = range(max(0, min(marks) - 4), min(n, max(marks) + 8))
        frames = sorted(set(around) | set(int(v) for v in np.linspace(0, n - 1, 10)))
        contact_sheet(video, start, j2, valid, frames, marks, sheets_dir / f"{name}_frames.jpg")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, metavar="NAME=RUN_DIR:VIDEO")
    ap.add_argument("--sheets-dir", type=Path, default=None)
    ap.add_argument("--two-wickets", action="append", default=[], metavar="NAME:ABSOLUTE_FRAME")
    ap.add_argument("--out", type=Path, default=Path("outputs/exp007_real_clip_diagnostics.json"))
    args = ap.parse_args()

    runs: Dict[str, Tuple[Path, Path]] = {}
    for spec in args.run:
        name, rest = spec.split("=", 1)
        run_dir, video = rest.rsplit(":", 1)
        runs[name] = (Path(run_dir), Path(video))
    result = {name: diagnose(name, run_dir, video, args.sheets_dir) for name, (run_dir, video) in runs.items()}
    for spec in args.two_wickets:
        name, frame = spec.split(":")
        clip = result[name]["clip"]
        geo = two_wicket_geometry(runs[name][1], int(frame), clip["width"], clip["height"])
        if geo is not None:
            assumed_focal = 0.5 * max(clip["width"], clip["height"]) / math.tan(math.radians(27.5))
            geo["pipeline_assumed_focal_px"] = round(assumed_focal, 1)
            geo["focal_ratio_true_over_assumed"] = round(geo["focal_px"] / assumed_focal, 2)
        result[name]["two_wicket_geometry"] = geo
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
