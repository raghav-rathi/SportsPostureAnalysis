"""EXP-007 ground-truth-free consistency checks of the pose backbone on a real clip (needs CUDA and NLF weights).

The same frames go through NLF several times with one input property changed each time:

  flip      mirrored image; the result is mirrored back and left/right joints are swapped
  aug5      NLF's own test-time augmentation (num_aug=5)
  half_res  frames downscaled 2x
  fov_XX    a different assumed field of view (NLF applies it to the larger image side)
  alt_model a second NLF checkpoint (--alt-model), for example another release of the same architecture

A consistent estimator returns the same body every time. The spread between runs is a LOWER bound on the pose
error, because errors shared by all runs cancel. It can show that a confidence interval is too narrow. It cannot
show that one is right.

Optional: --geocalib estimates the vertical field of view and gravity from the image itself with GeoCalib
(pip install git+https://github.com/cvg/GeoCalib) and compares them with the pipeline's assumptions.

Example:
  python validation/audit_nlf_self_consistency.py clip.mp4 --start 25 --end 100 --arm right \
      --panning --out outputs/exp007_nlf_consistency.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.stats import spearmanr

from bowling_biomech.schemas import BowlingArm, VideoAudit
from bowling_biomech.stage0_audit import audit_video
from bowling_biomech.stage1_pitch_calibration import calibrate_from_3d_motion_and_gravity, transform_cam_to_pitch_world
from bowling_biomech.stage2_event_spotting import detect_delivery_events
from bowling_biomech.stage2_nlf_3d import (
    _associate_tracks,
    _extract_standing_background_gravity,
    _interpolate_missing_frames,
    _load_nlf_cached,
    _select_bowler_track,
)
from bowling_biomech.stage3_kinematics import extract_fast_bowling_kinematics
from bowling_biomech.stage4_uncertainty import quantify_and_gate_metrics
from bowling_biomech.synthetic_ground_truth import JOINT_IDX, SMPL24_JOINTS

LR_PAIRS = [(JOINT_IDX["l" + n[1:]], JOINT_IDX[n]) for n in SMPL24_JOINTS if n.startswith("r")]
JOINT_GROUPS = {
    "hips": ["lhip", "rhip"], "knees": ["lkne", "rkne"], "ankles": ["lank", "rank"],
    "trunk": ["spi1", "spi2", "spi3", "neck", "head"], "shoulders": ["lsho", "rsho"],
    "elbows": ["lelb", "relb"], "wrists": ["lwri", "rwri"],
}


def load_frames(video: Path, start: int, end: Optional[int]) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = total if end is None else min(end, total)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames = []
    for _ in range(start, end):
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def run_nlf(model, frames: List[np.ndarray], fov_deg: float, num_aug: int, device: str, batch: int = 16):
    """Per-frame detections in metres, same call the pipeline makes."""
    import torch

    dets: List[List[Dict[str, np.ndarray]]] = []
    with torch.inference_mode():
        for b0 in range(0, len(frames), batch):
            x = torch.from_numpy(np.stack(frames[b0 : b0 + batch])).permute(0, 3, 1, 2).contiguous().to(device)
            out = model.detect_smpl_batched(
                x, default_fov_degrees=float(fov_deg), num_aug=int(num_aug), detector_threshold=0.30,
                max_detections=12, suppress_implausible_poses=True,
            )
            for j3, j2, un in zip(out["joints3d"], out["joints2d"], out["joint_uncertainties"]):
                j3, j2, un = (a.detach().cpu().numpy().astype(np.float64) for a in (j3, j2, un))
                dets.append([{"joints3d": j3[p] / 1000.0, "joints2d": j2[p], "unc": un[p] / 1000.0} for p in range(j3.shape[0])])
    return dets


def track_bowler(dets, width: int, height: int, anchor: Optional[np.ndarray] = None) -> Dict:
    """Pipeline tracking and selection. With `anchor` (T, 2) the track closest to that 2D pelvis path is taken,
    so every variant follows the same person as the baseline."""
    n = len(dets)
    tracks = _associate_tracks(dets, img_diag=float(np.hypot(width, height)))
    if anchor is None:
        tid, _ = _select_bowler_track(tracks, n_frames=n, img_w=width, img_h=height)
    else:
        def _dist(track) -> float:
            d = [np.linalg.norm(det["joints2d"][JOINT_IDX["pelv"]] - anchor[f]) for f, det in track.items()]
            return float(np.median(d)) - 0.5 * len(d)  # prefer long tracks that stay on the anchor path
        tid = int(np.argmin([_dist(t) if len(t) >= max(4, int(0.15 * n)) else 1e9 for t in tracks]))
    j3 = np.zeros((n, 24, 3))
    j2 = np.zeros((n, 24, 2))
    unc = np.full((n, 24), 0.12)
    valid = np.zeros(n, dtype=bool)
    in_frame = np.zeros((n, 24), dtype=bool)
    for f, det in tracks[tid].items():
        j3[f], j2[f], unc[f], valid[f] = det["joints3d"], det["joints2d"], det["unc"], True
        in_frame[f] = (
            (det["joints2d"][:, 0] >= 12) & (det["joints2d"][:, 0] <= width - 12)
            & (det["joints2d"][:, 1] >= 12) & (det["joints2d"][:, 1] <= height - 12)
        )
    return {
        "j3": _interpolate_missing_frames(j3, valid), "j2": _interpolate_missing_frames(j2, valid),
        "unc": _interpolate_missing_frames(unc, valid), "valid": valid, "in_frame": in_frame,
        "bg": _extract_standing_background_gravity(tracks, tid, width, height),
    }


def mirror_back(trk: Dict, width: int) -> Dict:
    """Undo a horizontal image flip: negate camera X, mirror pixel x, swap anatomical left and right."""
    out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in trk.items()}
    out["j3"][..., 0] *= -1.0
    out["j2"][..., 0] = (width - 1) - out["j2"][..., 0]
    for l, r in LR_PAIRS:
        for key in ("j3", "j2", "unc", "in_frame"):
            out[key][:, [l, r]] = out[key][:, [r, l]]
    out["bg"] = [v * np.array([-1.0, 1.0, 1.0]) for v in trk["bg"]]
    return out


def run_chain(trk: Dict, arm: BowlingArm, audit: VideoAudit, panning: bool, events=None) -> Dict:
    """Stages 1 to 4 exactly as the pipeline runs them after pose extraction."""
    j3 = trk["j3"]
    cal = calibrate_from_3d_motion_and_gravity(
        j3[:, JOINT_IDX["pelv"]], j3[:, JOINT_IDX["neck"]] - j3[:, JOINT_IDX["pelv"]], audit.width, audit.height,
        joints3d_cam_m=j3, joint_in_frame=trk["in_frame"], joint_uncertainties_m=trk["unc"], valid_mask=trk["valid"],
        bowling_arm_is_right=(arm == BowlingArm.RIGHT), is_panning_camera=panning,
        background_upright_vectors=trk["bg"],
    )
    world = transform_cam_to_pitch_world(j3, cal)
    ev = events or detect_delivery_events(world, fps=audit.fps, bowling_arm=arm, joint_in_frame=trk["in_frame"])
    kin = extract_fast_bowling_kinematics(
        world, ev, fps=audit.fps, bowling_arm=arm, joint_uncertainties_m=trk["unc"], joint_in_frame=trk["in_frame"]
    )
    metrics = quantify_and_gate_metrics(kin, audit, cal, events=ev)
    return {"cal": cal, "events": ev, "kin": kin, "metrics": metrics, "world": world}


def joint_spread(a: Dict, b: Dict, frames: slice) -> Dict:
    """Disagreement between two runs on the same frames, split by joint group and by camera axis."""
    pa, pb = a["j3"][frames], b["j3"][frames]
    ra, rb = pa - pa[:, :1], pb - pb[:, :1]
    d_rel = np.linalg.norm(ra - rb, axis=-1)
    unc = a["unc"][frames]
    out: Dict[str, Dict] = {"by_group": {}}
    for g, names in JOINT_GROUPS.items():
        idx = [JOINT_IDX[n] for n in names]
        out["by_group"][g] = {
            "root_relative_disagreement_mm_median": round(1000 * float(np.median(d_rel[:, idx])), 1),
            "root_relative_disagreement_mm_p90": round(1000 * float(np.percentile(d_rel[:, idx], 90)), 1),
            "nlf_uncertainty_mm_median": round(1000 * float(np.median(unc[:, idx])), 1),
        }
    body = np.arange(1, 24)
    rho = spearmanr(unc[:, body].ravel(), d_rel[:, body].ravel()).statistic
    comp = np.abs(ra - rb)[:, body]
    out.update(
        spearman_uncertainty_vs_disagreement=round(float(rho), 2),
        rms_disagreement_over_rms_uncertainty=round(
            float(np.sqrt(np.mean(d_rel[:, body] ** 2)) / np.sqrt(np.mean(unc[:, body] ** 2))), 2
        ),
        expected_ratio_if_uncertainty_were_a_calibrated_sigma=1.41,
        root_relative_rms_mm_camera_x_y_depth=[round(1000 * float(np.sqrt(np.mean(comp[..., k] ** 2))), 1) for k in range(3)],
        pelvis_absolute_rms_mm_camera_x_y_depth=[
            round(1000 * float(np.sqrt(np.mean((pa[:, 0, k] - pb[:, 0, k]) ** 2))), 1) for k in range(3)
        ],
    )
    return out


def geocalib_estimate(frames: List[np.ndarray], device: str) -> Dict:
    import torch
    from geocalib import GeoCalib

    model = GeoCalib().to(device)
    vfov, roll, pitch, up = [], [], [], []
    for fr in frames:
        img = torch.from_numpy(fr).permute(2, 0, 1).float().div(255.0).to(device)
        res = model.calibrate(img)
        r, p = (float(v) for v in res["gravity"].rp.reshape(-1))
        vfov.append(math.degrees(float(res["camera"].vfov.reshape(-1)[0])))
        roll.append(math.degrees(r))
        pitch.append(math.degrees(p))
        up.append(res["gravity"].vec3d.reshape(-1).detach().cpu().numpy().astype(np.float64))
    return {"vfov_deg": vfov, "roll_deg": roll, "pitch_deg": pitch, "gravity_vec3d": np.asarray(up)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("video", type=Path)
    ap.add_argument("--nlf-model", type=Path, default=Path(os.environ.get("NLF_MODEL_PATH", "models/nlf/nlf_l_multi.torchscript")))
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--arm", choices=["right", "left"], required=True)
    ap.add_argument("--capture-fps", type=float, default=None)
    ap.add_argument("--panning", action="store_true", help="pass the pipeline's pan verdict for this clip")
    ap.add_argument("--fovs", type=float, nargs="*", default=[30.0, 15.0, 8.0])
    ap.add_argument("--geocalib", action="store_true")
    ap.add_argument("--alt-model", type=Path, default=None, help="second TorchScript checkpoint to compare against")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, default=Path("outputs/exp007_nlf_consistency.json"))
    ap.add_argument("--save-npz", type=Path, default=None, help="also save every variant's tracked joints")
    args = ap.parse_args()

    arm = BowlingArm(args.arm)
    frames = load_frames(args.video, args.start, args.end)
    height, width = frames[0].shape[:2]
    audit = audit_video(args.video, start_frame=args.start, end_frame=args.end)
    if args.capture_fps:
        audit = audit.model_copy(update={"fps": float(args.capture_fps)})
    model = _load_nlf_cached(str(args.nlf_model), args.device)

    base = track_bowler(run_nlf(model, frames, 55.0, 1, args.device), width, height)
    anchor = base["j2"][:, JOINT_IDX["pelv"]]
    variants: Dict[str, Dict] = {"base": base}

    flipped = [np.ascontiguousarray(f[:, ::-1]) for f in frames]
    anchor_flipped = anchor * np.array([-1.0, 1.0]) + np.array([width - 1.0, 0.0])
    variants["flip"] = mirror_back(track_bowler(run_nlf(model, flipped, 55.0, 1, args.device), width, height, anchor_flipped), width)

    variants["aug5"] = track_bowler(run_nlf(model, frames, 55.0, 5, args.device), width, height, anchor)

    half = [cv2.resize(f, (width // 2, height // 2), interpolation=cv2.INTER_AREA) for f in frames]
    hr = track_bowler(run_nlf(model, half, 55.0, 1, args.device), width // 2, height // 2, anchor / 2.0)
    hr["j2"] = hr["j2"] * 2.0
    hr["in_frame"] = base["in_frame"].copy()
    variants["half_res"] = hr

    spread_keys = ["flip", "aug5", "half_res"]
    if args.alt_model is not None:
        alt = _load_nlf_cached(str(args.alt_model), args.device)
        variants["alt_model"] = track_bowler(run_nlf(alt, frames, 55.0, 1, args.device), width, height, anchor)
        spread_keys.append("alt_model")

    geo = None
    if args.geocalib:
        pick = [frames[i] for i in np.linspace(0, len(frames) - 1, 9, dtype=int)]
        geo = geocalib_estimate(pick, args.device)
        f_px = 0.5 * height / math.tan(math.radians(float(np.median(geo["vfov_deg"]))) / 2.0)
        fov_larger_side = 2.0 * math.degrees(math.atan(0.5 * max(width, height) / f_px))
        args.fovs = list(args.fovs) + [round(fov_larger_side, 1)]
    for fov in args.fovs:
        variants[f"fov_{fov:g}"] = track_bowler(run_nlf(model, frames, fov, 1, args.device), width, height, anchor)

    chain = {k: run_chain(v, arm, audit, args.panning) for k, v in variants.items()}
    chain_fixed = {k: run_chain(v, arm, audit, args.panning, events=chain["base"]["events"]) for k, v in variants.items()}

    ev = chain["base"]["events"]
    lo = max(0, int(ev.bfc.frame_float) - 3)
    hi = min(len(frames), int(math.ceil(ev.br.frame_float)) + 4)
    window = slice(lo, hi)

    base_m = chain["base"]["metrics"]
    rows = {}
    for key, m in base_m.items():
        if key == "icc_elbow_legality_verdict":
            continue
        row = {
            "base_value": round(chain["base"]["kin"].scalars[key], 2),
            "base_reported": (not m.suppressed),
            "base_ci_half_width": (round(0.5 * (m.ci_high - m.ci_low), 2) if not m.suppressed else None),
            "base_sigma_total": m.sigma_total,
            "base_observability": m.observability.value,
        }
        for name in variants:
            if name == "base":
                continue
            row[f"delta_{name}"] = round(chain[name]["kin"].scalars[key] - chain["base"]["kin"].scalars[key], 2)
            row[f"delta_{name}_same_events"] = round(chain_fixed[name]["kin"].scalars[key] - chain["base"]["kin"].scalars[key], 2)
        rows[key] = row

    def _summary(name: str) -> Dict:
        c = chain[name]
        p = variants[name]["j3"][:, JOINT_IDX["pelv"]]
        return {
            "bfc_ffc_br_frames": [c["events"].bfc.frame_float, c["events"].ffc.frame_float, c["events"].br.frame_float],
            "br_method": c["events"].br.detection_method,
            "azimuth_deg": c["cal"].azimuth_deg, "elevation_deg": c["cal"].elevation_deg,
            "view_class": c["cal"].view_class.value, "gravity_note": c["cal"].calibration_notes[0],
            "stature_m": round(c["kin"].estimated_stature_m, 3),
            "pelvis_depth_median_m": round(float(np.median(p[:, 2])), 2),
            "pelvis_depth_range_m": round(float(np.ptp(p[:, 2])), 2),
            "pelvis_lateral_range_m": round(float(np.ptp(p[:, 0])), 2),
            "valid_ratio": round(float(variants[name]["valid"].mean()), 3),
            "pelvis_2d_distance_to_base_px_median": round(
                float(np.median(np.linalg.norm(variants[name]["j2"][:, 0] - base["j2"][:, 0], axis=1))), 1
            ),
        }

    result = {
        "clip": {
            "file": args.video.name, "start": args.start, "end": args.end, "frames": len(frames), "width": width,
            "height": height, "fps_used": audit.fps, "arm": arm.value, "panning_flag": args.panning,
            "delivery_window_frames": [lo, hi],
        },
        "variants": {k: _summary(k) for k in variants},
        "joint_disagreement_delivery_window": {
            k: joint_spread(base, variants[k], window) for k in spread_keys
        },
        "joint_disagreement_whole_clip": {k: joint_spread(base, variants[k], slice(None)) for k in spread_keys},
        "metrics": rows,
    }
    if geo is not None:
        up_pipeline = np.asarray(chain["base"]["cal"].rotation_matrix_R)[:, 1]
        g = geo["gravity_vec3d"]
        g_med = np.median(g, axis=0)
        g_med /= np.linalg.norm(g_med)
        # GeoCalib's gravity vector and the pipeline's +Y are both "up" in the camera frame up to sign.
        cosang = abs(float(np.clip(g_med @ up_pipeline, -1.0, 1.0)))
        f_px = 0.5 * height / math.tan(math.radians(float(np.median(geo["vfov_deg"]))) / 2.0)
        result["geocalib"] = {
            "vfov_deg_median": round(float(np.median(geo["vfov_deg"])), 1),
            "vfov_deg_min_max": [round(float(np.min(geo["vfov_deg"])), 1), round(float(np.max(geo["vfov_deg"])), 1)],
            "equivalent_fov_on_larger_side_deg": round(2.0 * math.degrees(math.atan(0.5 * max(width, height) / f_px)), 1),
            "pipeline_assumed_fov_on_larger_side_deg": 55.0,
            "focal_ratio_assumed_over_geocalib": round((0.5 * max(width, height) / math.tan(math.radians(27.5))) / f_px, 2),
            "roll_deg_median": round(float(np.median(geo["roll_deg"])), 1),
            "pitch_deg_median": round(float(np.median(geo["pitch_deg"])), 1),
            "pitch_deg_min_max": [round(float(np.min(geo["pitch_deg"])), 1), round(float(np.max(geo["pitch_deg"])), 1)],
            "gravity_vec3d_median": [round(float(v), 3) for v in g_med],
            "pipeline_up_in_camera": [round(float(v), 3) for v in up_pipeline],
            "angle_between_geocalib_and_pipeline_gravity_deg": round(math.degrees(math.acos(cosang)), 1),
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.save_npz:
        np.savez_compressed(
            args.save_npz, **{f"{k}_j3": v["j3"] for k, v in variants.items()}, **{f"{k}_j2": v["j2"] for k, v in variants.items()},
            **{f"{k}_unc": v["unc"] for k, v in variants.items()}, **{f"{k}_valid": v["valid"] for k, v in variants.items()},
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
