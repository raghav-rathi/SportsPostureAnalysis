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
    mcc_landmarks: Optional[Path] = typer.Option(None, "--mcc-landmarks", help="JSON file mapping MCC pitch landmark names to [x_px, y_px]"),
    auto_two_wickets: bool = typer.Option(True, "--auto-two-wickets/--no-auto-two-wickets", help="Attempt conservative Law-8 two-wicket focal/gravity calibration before NLF"),
    repeatability: bool = typer.Option(True, "--repeatability/--no-repeatability", help="Run GPU perturbation ensemble (base/mirror/aug3 + MC) for empirical Repeatability Intervals"),
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
        lm_dict = {str(k): (float(v[0]), float(v[1])) for k, v in raw_lm.items()}

    wk = WorkloadContext(age_years=age, sex=sex, bowls_per_week=bowls_per_week)
    fs, _, artifacts = analyze_delivery_video(
        video_path=video,
        bowler_id=bowler_id,
        nlf_model_path=model_path,
        output_dir=output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        bowling_arm=arm,
        mcc_landmarks_px=lm_dict,
        auto_two_wickets=auto_two_wickets,
        anchor_xy_norm=anchor_tuple,
        workload=wk,
        capture_fps_override=capture_fps,
        compute_repeatability=repeatability,
        device=device,
    )

    tbl = Table(title=f"{fs.bowler_id} ({fs.bowling_arm.value}-arm | {fs.action_class.value} | {fs.interval_type})")
    tbl.add_column("Metric")
    tbl.add_column("Value / Interval")
    tbl.add_column("Observability")
    for m in fs.metrics.values():
        val = (
            f"[{m.observability.value.upper()}]"
            if (m.suppressed or m.value is None)
            else f"{m.value:.1f} [{m.ci_low:.1f}, {m.ci_high:.1f}] {m.unit}"
        )
        tbl.add_row(m.label, val, m.observability.value)
    console.print(tbl)
    console.print(f"[bold green]Timebase:[/bold green] {fs.audit.timebase_status} ({fs.audit.fps:.2f} unique fps) | Complete delivery: {fs.events.is_complete_delivery}")
    console.print(f"[bold cyan]Calibration:[/bold cyan] {fs.calibration.method} (intrinsics_cal={fs.calibration.is_intrinsics_calibrated}, pan={fs.calibration.is_panning_camera})")
    for k, p in artifacts.items():
        console.print(f"  - {k}: {p}")


@app.command()
def fuse(
    side_fact_sheet: Path = typer.Argument(..., help="Side-on fact_sheet.json"),
    rear_fact_sheet: Path = typer.Argument(..., help="Rear-on fact_sheet.json"),
    output_dir: Path = typer.Option(Path("outputs/fused"), "--output-dir", "-o"),
    bowler_id: Optional[str] = typer.Option(None, "--bowler-id", "-b"),
) -> None:
    """Fuse complementary side-on and rear-on BiomechanicalFactSheet JSON files."""
    s_fs = BiomechanicalFactSheet.model_validate_json(side_fact_sheet.read_text(encoding="utf-8"))
    r_fs = BiomechanicalFactSheet.model_validate_json(rear_fact_sheet.read_text(encoding="utf-8"))
    fused = fuse_dual_view_fact_sheets(s_fs, r_fs, bowler_id=bowler_id)
    rep = generate_grounded_report(fused)
    md = render_report_markdown(fused, rep)

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "fact_sheet_fused.json").write_text(fused.model_dump_json(indent=2), encoding="utf-8")
    (output_dir / "coaching_report_fused.md").write_text(md, encoding="utf-8")
    console.print(f"[bold green]Dual-View Fusion Complete:[/bold green] action={fused.action_class.value}")


@app.command("serve-studio")
def serve_studio(
    host: str = typer.Option("127.0.0.1", help="Bind host for the local-first Coaching Studio API"),
    port: int = typer.Option(8000, help="Bind port for the Coaching Studio API"),
    gpu_worker_url: Optional[str] = typer.Option(None, help="Optional remote GPU worker URL (e.g., http://127.0.0.1:8090)"),
) -> None:
    """Launch the local-first FastAPI Coaching Studio backend (with SQLite store & optional Vite bundle)."""
    import uvicorn

    if gpu_worker_url:
        os.environ["GPU_WORKER_URL"] = gpu_worker_url
    console.print(f"[bold cyan]Starting Coaching Studio API at http://{host}:{port}[/bold cyan]")
    uvicorn.run("bowling_biomech.server.app:create_app", factory=True, host=host, port=port, reload=False)


@app.command("serve-gpu-worker")
def serve_gpu_worker(
    host: str = typer.Option("127.0.0.1", help="Bind host for the Warm GPU Worker daemon"),
    port: int = typer.Option(8090, help="Bind port for the Warm GPU Worker daemon"),
) -> None:
    """Launch the Warm GPU Worker keeping NLF TorchScript loaded in VRAM with SHA-256 disk caching."""
    import uvicorn

    console.print(f"[bold green]Starting Warm NLF GPU Worker at http://{host}:{port}[/bold green]")
    uvicorn.run("bowling_biomech.gpu_worker.worker:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
