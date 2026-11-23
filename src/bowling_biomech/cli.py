"""Command-Line Interface for the Automated Cricket Fast-Bowling Biomechanics Engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import typer
from rich.console import Console
from rich.table import Table

from .pipeline import analyze_delivery_video, fuse_dual_view_fact_sheets
from .schemas import BiomechanicalFactSheet, BowlingArm, WorkloadContext
from .stage6_coach_llm import generate_grounded_report, render_report_markdown

app = typer.Typer(help="Automated Cricket Fast-Bowling Biomechanics Analysis CLI")
console = Console()


def _default_nlf_model() -> Path:
    env_p = os.environ.get("NLF_MODEL_PATH")
    if env_p:
        return Path(env_p)
    return Path("models/nlf/nlf_l_multi.torchscript")


@app.command()
def analyze(
    video: Path = typer.Argument(..., help="Path to fast-bowling video file"),
    bowler_id: str = typer.Option("Bowler", "--bowler-id", "-b"),
    nlf_model: Optional[Path] = typer.Option(None, "--nlf-model", help="Path to nlf_l_multi.torchscript (or set NLF_MODEL_PATH)"),
    output_dir: Path = typer.Option(Path("outputs/run"), "--output-dir", "-o"),
    start_frame: int = typer.Option(0, "--start-frame"),
    end_frame: Optional[int] = typer.Option(None, "--end-frame"),
    arm: Optional[BowlingArm] = typer.Option(None, "--arm", help="right | left (auto if omitted)"),
    anchor_xy: Optional[str] = typer.Option(None, "--anchor-xy", help="Normalized bowler anchor 'x,y' in [0,1] (e.g. '0.45,0.60')"),
    mcc_landmarks: Optional[Path] = typer.Option(None, "--mcc-landmarks", help="JSON file mapping MCC Law 7 landmark names to [x_px, y_px]"),
    age: Optional[int] = typer.Option(None, "--age"),
    sex: str = typer.Option("male", "--sex"),
    bowls_per_week: Optional[int] = typer.Option(None, "--bowls-per-week"),
    capture_fps: Optional[float] = typer.Option(None, "--capture-fps", help="True physical capture FPS if clip is exported as slow-motion playback"),
    device: str = typer.Option("cuda", "--device"),
) -> None:
    """Analyze a fast-bowling video and generate verified metrics, visuals, and grounded report."""
    model_path = nlf_model if nlf_model is not None else _default_nlf_model()
    anchor_tuple: Optional[Tuple[float, float]] = None
    if anchor_xy:
        parts = [float(p.strip()) for p in anchor_xy.split(",")]
        if len(parts) != 2:
            raise typer.BadParameter("--anchor-xy must be formatted as 'x,y' in [0, 1].")
        anchor_tuple = (parts[0], parts[1])

    lm_dict: Optional[Dict[str, Tuple[float, float]]] = None
    if mcc_landmarks is not None:
        raw_lm = json.loads(mcc_landmarks.read_text(encoding="utf-8"))
        lm_dict = {k: (float(v[0]), float(v[1])) for k, v in raw_lm.items()}

    workload = WorkloadContext(age_years=age, sex=sex, bowls_per_week=bowls_per_week)
    fs, report, artifacts = analyze_delivery_video(
        video_path=video,
        bowler_id=bowler_id,
        nlf_model_path=model_path,
        output_dir=output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        bowling_arm=arm,
        mcc_landmarks_px=lm_dict,
        anchor_xy_norm=anchor_tuple,
        workload=workload,
        capture_fps_override=capture_fps,
        device=device,
    )
    table = Table(title=f"Verified Biomechanics Summary — {fs.bowler_id}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value (95% CI)", style="bold white")
    table.add_column("Observability", style="yellow")
    for m in fs.metrics.values():
        val = f"[{m.observability.value.upper()}]" if m.suppressed else f"{m.value:.1f} [{m.ci_low:.1f}, {m.ci_high:.1f}] {m.unit}"
        table.add_row(m.label, val, m.observability.value)
    console.print(table)
    console.print(f"\n[bold green]Action Class:[/bold green] {fs.action_class.value} | [bold green]Quadrant:[/bold green] {fs.quadrant.quadrant_label}")
    console.print(f"[bold]Outputs written to:[/bold] {output_dir}")
    for k, p in artifacts.items():
        console.print(f"  - {k}: {p}")


@app.command()
def fuse(
    side_json: Path = typer.Argument(..., help="Side-on fact_sheet.json"),
    rear_json: Path = typer.Argument(..., help="Rear-on fact_sheet.json"),
    output_dir: Path = typer.Option(Path("outputs/fused"), "--output-dir", "-o"),
    bowler_id: Optional[str] = typer.Option(None, "--bowler-id", "-b"),
) -> None:
    """Fuse complementary side-on and rear-on FactSheets into a unified multi-view assessment."""
    side_fs = BiomechanicalFactSheet.model_validate_json(side_json.read_text(encoding="utf-8"))
    rear_fs = BiomechanicalFactSheet.model_validate_json(rear_json.read_text(encoding="utf-8"))
    fused = fuse_dual_view_fact_sheets(side_fs, rear_fs, bowler_id=bowler_id)
    report = generate_grounded_report(fused)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fact_sheet_fused.json").write_text(fused.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "coaching_report_fused.md").write_text(
        render_report_markdown(fused, report), encoding="utf-8"
    )
    console.print(f"[bold green]Fused Action:[/bold green] {fused.action_class.value} | [bold green]Quadrant:[/bold green] {fused.quadrant.quadrant_label}")
    console.print(f"Saved fused outputs to {output_dir}")


if __name__ == "__main__":
    app()
