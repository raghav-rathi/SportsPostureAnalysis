"""FPS Degradation Stress Test: Quantify measurement drift from high-FPS down to 30 fps on real video.

Takes a crisp high-frame-rate fast-bowling video (e.g., 60/120/240 fps), creates temporal
subsampled copies via ffmpeg (e.g. 60 fps, 30 fps, 25 fps), runs the full RTX 4090 NLF
pipeline on each, and reports empirical error vs. native high-FPS reference plus 95% CI coverage.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from bowling_biomech.pipeline import analyze_delivery_video
from bowling_biomech.schemas import BowlingArm


def downsample_video(src: Path, dst: Path, target_fps: float) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-i", str(src),
        "-vf", f"fps={target_fps}",
        "-c:v", "libx264", "-crf", "16", "-an",
        str(dst),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dst


def run_fps_stress_test(
    high_fps_video: Path,
    bowler_id: str,
    nlf_model_path: Path,
    out_root: Path,
    target_fps_list: List[float] = [60.0, 30.0, 25.0],
    bowling_arm: Optional[BowlingArm] = None,
    device: str = "cuda",
) -> Dict:
    out_root.mkdir(parents=True, exist_ok=True)
    fs_ref, _, _ = analyze_delivery_video(
        video_path=high_fps_video,
        bowler_id=f"{bowler_id}_native",
        nlf_model_path=nlf_model_path,
        output_dir=out_root / "native",
        bowling_arm=bowling_arm,
        device=device,
    )

    report = {
        "bowler_id": bowler_id,
        "native_fps": fs_ref.audit.fps,
        "native_view": fs_ref.calibration.view_class.value,
        "comparisons": [],
    }

    for tfps in target_fps_list:
        if tfps >= fs_ref.audit.fps - 1.0:
            continue
        sub_vid = downsample_video(high_fps_video, out_root / f"clip_{int(tfps)}fps.mp4", tfps)
        fs_sub, _, _ = analyze_delivery_video(
            video_path=sub_vid,
            bowler_id=f"{bowler_id}_{int(tfps)}fps",
            nlf_model_path=nlf_model_path,
            output_dir=out_root / f"run_{int(tfps)}fps",
            bowling_arm=fs_ref.bowling_arm,
            device=device,
        )

        metric_diffs = {}
        covered = 0
        compared = 0
        for k, m_ref in fs_ref.metrics.items():
            m_sub = fs_sub.metrics.get(k)
            if m_ref.suppressed or m_ref.value is None or m_sub is None:
                continue
            if m_sub.suppressed or m_sub.value is None:
                metric_diffs[k] = {"status": "suppressed_at_low_fps", "reason": m_sub.suppression_reason}
                continue
            diff = round(m_sub.value - m_ref.value, 2)
            in_ci = bool(
                m_sub.ci_low is not None
                and m_sub.ci_high is not None
                and m_sub.ci_low <= m_ref.value <= m_sub.ci_high
            )
            compared += 1
            if in_ci:
                covered += 1
            metric_diffs[k] = {
                "native_val": m_ref.value,
                "low_fps_val": m_sub.value,
                "drift": diff,
                "low_fps_ci": [m_sub.ci_low, m_sub.ci_high],
                "native_covered_by_ci": in_ci,
            }

        report["comparisons"].append(
            {
                "target_fps": tfps,
                "compared_metrics": compared,
                "ci_coverage_pct": round(100.0 * covered / max(compared, 1), 1),
                "metric_diffs": metric_diffs,
            }
        )

    out_json = out_root / "fps_stress_test_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
