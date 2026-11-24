"""Phase 2 (M2A-M2D): Local-First FastAPI Coaching Studio Server.

Serves:
  - Longitudinal SQLite store (`bowlers`, `sessions`, `deliveries`, `calibration_profiles`, `workload_logs`)
  - Pre-seeded offline Demo Session (240 fps Dual-View MCC + real Steyn/McGrath fixtures) on first boot
  - Video upload + 3-tier FPS resolver + All-Intra (`-g 1 -bf 0`) scrub proxy + Warm GPU Worker dispatch
  - `<35 ms` Instant CPU HITL recompute (4-click MCC PnP, FPS override, causal event nudging, 3-DOF pan track)
  - Multi-ball net-session auto-clipper (`scan-video`), Pillar 3 repeatability & fatigue drift (`compare`)
  - Unsynchronized Dual-Phone 3D Ray-Precision Fusion (`fuse-dual`)
  - Deterministic Numeric-Verifier Coach Copilot (`coach/ask`)
  - 1-Click `1080x1920` Truthful Breakdown Reel Export (`export-reel`)
  - Static built frontend (`web/dist`) if compiled
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field

from ..constants import MCC_PITCH_LANDMARKS_M
from ..schemas import BiomechanicalFactSheet, BowlingArm, PitchCalibration, VideoAudit, WorkloadContext
from ..stage0_audit import audit_video
from ..stage0_net_clipper import scan_net_session_video
from ..stage1_pitch_calibration import (
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
    world_up_from_calibration,
)
from ..stage2_dual_sync_fusion import fuse_dual_view_3d_trajectories
from ..stage5_evaluator import build_workload_guidance
from ..stage6_coach_copilot import ask_grounded_coach
from ..stage6_visualizer import render_truthful_breakdown_reel_9x16
from .db import StudioDatabase
from .recompute import (
    build_telemetry_v2,
    compute_session_comparison,
    project_mcc_wireframe_2d,
    recompute_delivery_cpu,
    seed_demo_session,
)
from .video_proxy import build_all_intra_scrub_proxy, probe_video_metadata, render_synthetic_proxy_video

_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
_MAX_UPLOAD_BYTES = int(os.environ.get("STUDIO_MAX_UPLOAD_MB", "512")) * 1024 * 1024


async def _save_upload_bounded(upload: UploadFile, dest_dir: Path, stem: str) -> Path:
    suffix = Path(upload.filename or f"{stem}.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video file type '{suffix}'. Allowed: {sorted(_ALLOWED_VIDEO_SUFFIXES)}",
        )
    dest = dest_dir / f"{stem}{suffix}"
    total = 0
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Video upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB cap.",
                    )
                f.write(chunk)
    except HTTPException:
        dest.unlink(missing_ok=True)
        raise
    if total == 0:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty video upload.")
    return dest


class CreateSessionRequest(BaseModel):
    bowler_name: str = Field(..., min_length=1, max_length=120)
    bowling_arm: str = Field(default="right", pattern="^(right|left)$")
    age_band: str = Field(default="adult")
    title: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="")


class RecomputeRequest(BaseModel):
    mcc_landmarks_px: Optional[Dict[str, List[float]]] = None
    two_wickets_px: Optional[Dict[str, float]] = None
    capture_fps_override: Optional[float] = None
    bowling_arm: Optional[str] = None
    event_overrides: Optional[Dict[str, Optional[float]]] = None
    workload: Optional[Dict[str, Any]] = None
    enable_pan_tracking: Optional[bool] = None
    pan_anchor_frame: Optional[int] = None


class PreviewCalibrationRequest(BaseModel):
    mcc_landmarks_px: Dict[str, List[float]]


class CoachAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    use_external_llm: bool = False


class DualFuseRequest(BaseModel):
    delivery_id_a: str
    delivery_id_b: str
    label: str = "Dual-Phone 3D Fused Delivery"


class WorkloadLogRequest(BaseModel):
    log_date: str
    deliveries_count: int = Field(..., ge=0, le=500)
    session_type: str = "net"
    notes: str = ""


def create_app(
    data_root: Optional[str | Path] = None,
    fixtures_dir: Optional[str | Path] = None,
    gpu_worker_url: Optional[str] = None,
    seed_demo: bool = True,
) -> FastAPI:
    root = Path(data_root or os.environ.get("STUDIO_DATA_DIR", "data/studio"))
    sessions_root = root / "sessions"
    uploads_root = root / "uploads"
    sessions_root.mkdir(parents=True, exist_ok=True)
    uploads_root.mkdir(parents=True, exist_ok=True)

    db = StudioDatabase(root / "studio.db")
    repo_fixtures = Path(fixtures_dir) if fixtures_dir else (Path(__file__).resolve().parents[3] / "tests" / "fixtures")
    if seed_demo:
        seed_demo_session(db, sessions_root, fixtures_dir=repo_fixtures)

    worker_url = (gpu_worker_url or os.environ.get("GPU_WORKER_URL", "")).rstrip("/")

    app = FastAPI(title="SportsPostureAnalysis Coaching Studio API", version="2.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.db = db
    app.state.sessions_root = sessions_root
    app.state.uploads_root = uploads_root
    app.state.gpu_worker_url = worker_url

    @app.get("/api/health")
    def api_health() -> Dict[str, Any]:
        worker_status: Dict[str, Any] = {"configured": bool(worker_url), "reachable": False}
        if worker_url:
            try:
                import httpx

                r = httpx.get(f"{worker_url}/health", timeout=2.0)
                if r.status_code == 200:
                    worker_status = {"configured": True, "reachable": True, **r.json()}
            except Exception as exc:
                worker_status["error"] = str(exc)
        return {
            "status": "ok",
            "version": "2.0.0",
            "mode": "local_first_studio",
            "gpu_worker": worker_status,
            "mcc_landmark_keys": list(MCC_PITCH_LANDMARKS_M.keys()),
        }

    @app.get("/api/sessions")
    def api_list_sessions() -> Dict[str, Any]:
        return {"sessions": db.list_sessions()}

    @app.post("/api/sessions")
    def api_create_session(req: CreateSessionRequest) -> Dict[str, Any]:
        bowler = db.create_bowler(req.bowler_name, bowling_arm=req.bowling_arm, age_band=req.age_band)
        ses = db.create_session(bowler["id"], title=req.title, notes=req.notes)
        return {"session": db.get_session(ses["id"])}

    @app.get("/api/sessions/{session_id}")
    def api_get_session(session_id: str) -> Dict[str, Any]:
        ses = db.get_session(session_id)
        if not ses:
            raise HTTPException(status_code=404, detail="Session not found.")
        return {"session": ses}

    @app.post("/api/sessions/{session_id}/scan-video")
    async def api_scan_video(session_id: str, video: UploadFile = File(...)) -> Dict[str, Any]:
        """Scan an uploaded multi-ball net video at ~6 fps and return candidate delivery clips + thumbnails."""
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        dest = await _save_upload_bounded(video, uploads_root, f"scan_{uuid.uuid4().hex[:10]}")
        meta = probe_video_metadata(dest)
        cands = scan_net_session_video(dest, include_thumbnails=True)
        return {
            "upload_path": str(dest),
            "video_meta": meta.__dict__,
            "candidates": [c.__dict__ for c in cands],
        }

    @app.post("/api/sessions/{session_id}/deliveries/upload")
    async def api_upload_delivery(
        session_id: str,
        video: UploadFile = File(...),
        label: str = Form("Uploaded Delivery"),
        start_frame: int = Form(0),
        end_frame: Optional[int] = Form(None),
        capture_fps_override: Optional[float] = Form(None),
        bowling_arm: Optional[str] = Form(None),
        mcc_landmarks_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        """Ingest a delivery video, build its All-Intra scrub proxy, and run Warm GPU Worker (or local NLF)."""
        ses = db.get_session(session_id)
        if not ses:
            raise HTTPException(status_code=404, detail="Session not found.")

        did = f"del_{uuid.uuid4().hex[:10]}"
        ddir = sessions_root / session_id / did
        ddir.mkdir(parents=True, exist_ok=True)

        raw_vid = await _save_upload_bounded(video, ddir, "source")

        probed = probe_video_metadata(raw_vid)
        eff_fps = capture_fps_override or (probed.quicktime_capture_fps if probed.is_probable_slomo_container else None)
        audit = audit_video(
            raw_vid,
            start_frame=start_frame,
            end_frame=end_frame,
            capture_fps_override=eff_fps,
            dedup_regular_pulldown=True,
        )
        src_indices = audit.source_frame_indices or list(range(start_frame, start_frame + audit.num_frames))
        proxy_mp4 = build_all_intra_scrub_proxy(raw_vid, ddir / "scrub_proxy.mp4", source_frame_indices=src_indices, proxy_fps=30.0)

        cal_pre = None
        if mcc_landmarks_json:
            lm_raw = json.loads(mcc_landmarks_json)
            if len(lm_raw) >= 4:
                lm_map = {k: (float(v[0]), float(v[1])) for k, v in lm_raw.items()}
                cal_pre = calibrate_from_mcc_landmarks(lm_map, audit.width, audit.height)

        # Dispatch 3D pose inference to Warm Remote RTX 4090 Worker when configured
        if worker_url:
            import httpx

            form_data: Dict[str, Any] = {
                "start_frame": str(start_frame),
                "num_aug": "1",
                "extract_bml87": "true",
            }
            if end_frame is not None:
                form_data["end_frame"] = str(end_frame)
            if eff_fps is not None:
                form_data["capture_fps_override"] = str(eff_fps)
            if bowling_arm in ("right", "left"):
                form_data["bowling_arm"] = bowling_arm
            if cal_pre is not None:
                form_data["intrinsic_matrix_json"] = json.dumps(cal_pre.camera_matrix_K)
                form_data["world_up_json"] = json.dumps(world_up_from_calibration(cal_pre).tolist())

            with raw_vid.open("rb") as vf:
                resp = httpx.post(
                    f"{worker_url}/infer",
                    files={"video": (raw_vid.name, vf, "video/mp4")},
                    data=form_data,
                    timeout=180.0,
                )
            if resp.status_code != 200:
                raise HTTPException(status_code=502, detail=f"GPU worker failed ({resp.status_code}): {resp.text[:300]}")
            (ddir / "raw_nlf.npz").write_bytes(resp.content)
            z = np.load(ddir / "raw_nlf.npz", allow_pickle=False)
            det_arm = str(z["detected_bowling_arm"]) if "detected_bowling_arm" in z else (bowling_arm or "right")
            fx_used = float(z["nlf_fx_used"]) if "nlf_fx_used" in z else 0.0
            if cal_pre is None and "auto_cal_json" in z and str(z["auto_cal_json"]).strip():
                cal_pre = PitchCalibration.model_validate_json(str(z["auto_cal_json"]))
            if cal_pre is None:
                j_cam = z["joints3d_cam_m"].astype(np.float64)
                cal_pre = calibrate_from_3d_motion_and_gravity(
                    j_cam[:, 0], j_cam[:, 12] - j_cam[:, 0], audit.width, audit.height, joints3d_cam_m=j_cam
                )
        else:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No GPU_WORKER_URL configured for raw video 3D inference. "
                    "Start the Warm RTX 4090 GPU Worker (via SSH tunnel on :8090) and set GPU_WORKER_URL=http://127.0.0.1:8090."
                ),
            )

        (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(cal_pre.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps({"bowling_arm": det_arm, "pipeline_mode": "single_view_gpu_worker", "nlf_fx_used": fx_used}),
            encoding="utf-8",
        )
        res = recompute_delivery_cpu(ddir, bowler_id=label, video_path=raw_vid, proxy_fps=30.0)
        db.upsert_delivery(
            delivery_id=did,
            session_id=session_id,
            label=label,
            data_dir=ddir,
            video_path=str(raw_vid),
            proxy_path=str(proxy_mp4),
            pipeline_mode="single_view_gpu_worker",
            start_frame=start_frame,
            end_frame=end_frame,
            status="ready",
            fact_sheet_json=res["fact_sheet"].model_dump_json(),
            isb_summary=res["isb_summary"],
            pan_summary=res["pan_summary"],
        )
        return {
            "delivery_id": did,
            "probed_meta": probed.__dict__,
            "recompute_ms": res["recompute_ms"],
            "fact_sheet": res["fact_sheet"].model_dump(),
        }

    @app.get("/api/deliveries/{delivery_id}")
    def api_get_delivery(delivery_id: str) -> Dict[str, Any]:
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        ddir = Path(rec["data_dir"])
        fs = json.loads(rec["fact_sheet_json"]) if rec.get("fact_sheet_json") else None
        tel_path = ddir / "telemetry.json"
        tel = json.loads(tel_path.read_text(encoding="utf-8")) if tel_path.exists() else None
        rep_path = ddir / "coaching_report.md"
        rep_md = rep_path.read_text(encoding="utf-8") if rep_path.exists() else ""
        return {
            "delivery": {
                "id": rec["id"],
                "session_id": rec["session_id"],
                "label": rec["label"],
                "pipeline_mode": rec["pipeline_mode"],
                "status": rec["status"],
                "overrides": rec["overrides"],
                "isb_summary": rec["isb_summary"],
                "pan_summary": rec["pan_summary"],
                "video_proxy_url": f"/api/deliveries/{delivery_id}/video",
            },
            "fact_sheet": fs,
            "telemetry": tel,
            "report_markdown": rep_md,
            "mcc_landmarks_3d": {k: list(v) for k, v in MCC_PITCH_LANDMARKS_M.items()},
        }

    @app.get("/api/deliveries/{delivery_id}/video")
    def api_stream_delivery_video(delivery_id: str) -> FileResponse:
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        p = Path(rec["proxy_path"] or rec["video_path"])
        if not p.exists():
            raise HTTPException(status_code=404, detail="Scrub proxy video not found on disk.")
        return FileResponse(str(p), media_type="video/mp4", filename=f"{delivery_id}_proxy.mp4")

    @app.post("/api/deliveries/{delivery_id}/preview-calibration")
    def api_preview_calibration(delivery_id: str, req: PreviewCalibrationRequest) -> Dict[str, Any]:
        """<5 ms live 4-click MCC PnP wireframe preview + reprojection error badge BEFORE saving."""
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        if len(req.mcc_landmarks_px) < 4:
            raise HTTPException(status_code=400, detail="Select at least 4 MCC pitch landmarks.")
        ddir = Path(rec["data_dir"])
        audit = VideoAudit.model_validate_json((ddir / "audit.json").read_text(encoding="utf-8"))
        lm_map = {k: (float(v[0]), float(v[1])) for k, v in req.mcc_landmarks_px.items()}
        try:
            cal = calibrate_from_mcc_landmarks(lm_map, width=audit.width, height=audit.height)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"MCC PnP calibration failed: {exc}")
        return {
            "calibration": cal.model_dump(),
            "mcc_wireframe_2d": project_mcc_wireframe_2d(cal),
            "reprojection_error_px": cal.reprojection_error_px,
        }

    @app.post("/api/deliveries/{delivery_id}/recompute")
    def api_recompute_delivery(delivery_id: str, req: RecomputeRequest) -> Dict[str, Any]:
        """<35 ms instant CPU HITL recompute (calibration, FPS pill, causal event nudging, pan track, workload)."""
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        merged = dict(rec.get("overrides") or {})
        patch = req.model_dump(exclude_none=True)
        if "event_overrides" in patch and isinstance(merged.get("event_overrides"), dict):
            ev_m = dict(merged["event_overrides"])
            ev_m.update(patch["event_overrides"])
            patch["event_overrides"] = ev_m
        merged.update(patch)

        res = recompute_delivery_cpu(
            data_dir=rec["data_dir"],
            bowler_id=rec["label"],
            overrides=merged,
            video_path=rec["video_path"] or None,
        )
        db.upsert_delivery(
            delivery_id=delivery_id,
            session_id=rec["session_id"],
            label=rec["label"],
            data_dir=rec["data_dir"],
            video_path=rec["video_path"],
            proxy_path=rec["proxy_path"],
            pipeline_mode=rec["pipeline_mode"],
            start_frame=rec["start_frame"],
            end_frame=rec["end_frame"],
            status="ready",
            fact_sheet_json=res["fact_sheet"].model_dump_json(),
            isb_summary=res["isb_summary"],
            pan_summary=res["pan_summary"],
            overrides=merged,
        )
        if req.mcc_landmarks_px and len(req.mcc_landmarks_px) >= 4:
            db.save_calibration_profile(
                rec["session_id"],
                name=f"4-Click MCC ({rec['label']})",
                method=res["fact_sheet"].calibration.method,
                mcc_landmarks_px=req.mcc_landmarks_px,
                calibration_dict=res["fact_sheet"].calibration.model_dump(),
            )
        return {
            "recompute_ms": res["recompute_ms"],
            "needs_gpu_refit_for_K": res["needs_gpu_refit_for_K"],
            "fact_sheet": res["fact_sheet"].model_dump(),
            "telemetry": res["telemetry"],
            "report_markdown": res["report_markdown"],
            "isb_summary": res["isb_summary"],
            "pan_summary": res["pan_summary"],
        }

    @app.post("/api/deliveries/{delivery_id}/coach/ask")
    def api_coach_ask(delivery_id: str, req: CoachAskRequest) -> Dict[str, Any]:
        """Grounded Coach Copilot guarded by deterministic post-generation numeric verifier."""
        rec = db.get_delivery(delivery_id)
        if not rec or not rec.get("fact_sheet_json"):
            raise HTTPException(status_code=404, detail="Delivery FactSheet not found.")
        fs = BiomechanicalFactSheet.model_validate_json(rec["fact_sheet_json"])
        ans = ask_grounded_coach(req.question, fs, use_external_llm=req.use_external_llm)
        return {
            "answer_markdown": ans.answer_markdown,
            "cited_metric_keys": ans.cited_metric_keys,
            "verification_passed": ans.verification_passed,
            "blocked_violations": ans.blocked_violations,
            "generator_mode": ans.generator_mode,
        }

    @app.post("/api/deliveries/{delivery_id}/export-reel")
    def api_export_reel(delivery_id: str) -> Dict[str, Any]:
        """Render 1080x1920 (9:16) Truthful Breakdown Reel MP4 (browser-safe H.264) on demand."""
        rec = db.get_delivery(delivery_id)
        if not rec or not rec.get("fact_sheet_json"):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        ddir = Path(rec["data_dir"])
        fs = BiomechanicalFactSheet.model_validate_json(rec["fact_sheet_json"])
        z = np.load(ddir / "raw_nlf.npz", allow_pickle=False)
        j2d = z["joints2d_px"].astype(np.float64)
        tel = json.loads((ddir / "telemetry.json").read_text(encoding="utf-8"))
        jw = np.asarray(tel["joints3d_world_m"], dtype=np.float64)
        vid = rec["proxy_path"] or rec["video_path"]
        out_reel = render_truthful_breakdown_reel_9x16(
            video_path=vid,
            start_frame=0,
            joints2d_px=j2d,
            joints_world_m=jw,
            fact_sheet=fs,
            out_path=ddir / "truthful_breakdown_9x16.mp4",
            source_frame_indices=list(range(len(j2d))),
        )
        return {
            "reel_url": f"/api/deliveries/{delivery_id}/reel",
            "size_bytes": out_reel.stat().st_size,
        }

    @app.get("/api/deliveries/{delivery_id}/reel")
    def api_download_reel(delivery_id: str) -> FileResponse:
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        p = Path(rec["data_dir"]) / "truthful_breakdown_9x16.mp4"
        if not p.exists():
            raise HTTPException(status_code=404, detail="Reel not exported yet; POST /export-reel first.")
        return FileResponse(str(p), media_type="video/mp4", filename=f"{delivery_id}_truthful_reel_9x16.mp4")

    @app.get("/api/sessions/{session_id}/compare")
    def api_compare_session(session_id: str) -> Dict[str, Any]:
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        deliveries = db.list_session_deliveries(session_id)
        return compute_session_comparison(deliveries)

    @app.post("/api/sessions/{session_id}/fuse-dual")
    def api_fuse_dual(session_id: str, req: DualFuseRequest) -> Dict[str, Any]:
        """M2C: Unsynchronized Dual-Phone +Y_world Sub-Frame Sync & Anisotropic Ray-Precision 3D Fusion."""
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        rec_a = db.get_delivery(req.delivery_id_a)
        rec_b = db.get_delivery(req.delivery_id_b)
        if not rec_a or not rec_b:
            raise HTTPException(status_code=404, detail="Both source deliveries must exist.")

        fs_a = BiomechanicalFactSheet.model_validate_json(rec_a["fact_sheet_json"])
        fs_b = BiomechanicalFactSheet.model_validate_json(rec_b["fact_sheet_json"])
        if fs_a.bowling_arm != fs_b.bowling_arm:
            raise HTTPException(status_code=400, detail="Cannot fuse deliveries with mismatched bowling arms.")

        za = np.load(Path(rec_a["data_dir"]) / "raw_nlf.npz", allow_pickle=False)
        zb = np.load(Path(rec_b["data_dir"]) / "raw_nlf.npz", allow_pickle=False)
        jwa = transform_cam_to_pitch_world(za["joints3d_cam_m"].astype(np.float64), fs_a.calibration)
        jwb = transform_cam_to_pitch_world(zb["joints3d_cam_m"].astype(np.float64), fs_b.calibration)

        try:
            fused = fuse_dual_view_3d_trajectories(
                bowler_id=req.label,
                bowling_arm=fs_a.bowling_arm,
                joints_world_a=jwa,
                cal_a=fs_a.calibration,
                audit_a=fs_a.audit,
                joints_world_b=jwb,
                cal_b=fs_b.calibration,
                audit_b=fs_b.audit,
                unc_a=za["joint_uncertainties_m"].astype(np.float64),
                unc_b=zb["joint_uncertainties_m"].astype(np.float64),
                in_frame_a=za["joint_in_frame"].astype(bool),
                in_frame_b=zb["joint_in_frame"].astype(bool),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        did_f = f"del_fused_{uuid.uuid4().hex[:8]}"
        ddir = sessions_root / session_id / did_f
        ddir.mkdir(parents=True, exist_ok=True)
        # Reproject fused 3D skeleton into Camera A pixels for synchronized canvas playback
        Ka = np.asarray(fs_a.calibration.camera_matrix_K, dtype=np.float64).reshape(3, 3)
        Ra = np.asarray(fs_a.calibration.rotation_matrix_R, dtype=np.float64).reshape(3, 3)
        ta = np.asarray(fs_a.calibration.translation_t, dtype=np.float64).reshape(3)
        j_cam_f = np.einsum("ij,tkj->tki", Ra, fused.joints_world_fused) + ta[None, None, :]
        z_safe = np.maximum(j_cam_f[:, :, 2:3], 0.2)
        j2d_f = np.einsum("ij,tkj->tki", Ka, j_cam_f)[:, :, :2] / z_safe

        np.savez_compressed(
            ddir / "raw_nlf.npz",
            joints3d_cam_m=j_cam_f.astype(np.float32),
            joints2d_px=j2d_f.astype(np.float32),
            joint_uncertainties_m=fused.joint_uncertainties_fused.astype(np.float32),
            joint_in_frame=fused.joint_in_frame_fused.astype(np.bool_),
            valid_mask=np.ones(len(j_cam_f), dtype=np.bool_),
            cov_world_fused=fused.cov_world_fused.astype(np.float32),
            dual_baseline_sin=np.array(float(fused.baseline_sin), dtype=np.float64),
        )
        (ddir / "audit.json").write_text(fused.fact_sheet.audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(fused.fact_sheet.calibration.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps(
                {
                    "bowling_arm": fs_a.bowling_arm.value,
                    "pipeline_mode": "dual_3d_ray_fusion",
                    "dual_baseline_sin": float(fused.baseline_sin),
                    "nlf_fx_used": float(Ka[0, 0]),
                    "source_delivery_ids": [req.delivery_id_a, req.delivery_id_b],
                }
            ),
            encoding="utf-8",
        )
        (ddir / "fact_sheet.json").write_text(fused.fact_sheet.model_dump_json(indent=2), encoding="utf-8")
        rep = ask_grounded_coach("Give a complete verified summary of this dual-view fused delivery.", fused.fact_sheet)
        (ddir / "coaching_report.md").write_text(rep.answer_markdown, encoding="utf-8")
        tel = build_telemetry_v2(
            joints_world_m=fused.joints_world_fused,
            joints2d_px=j2d_f,
            joint_uncertainties_m=fused.joint_uncertainties_fused,
            joint_in_frame=fused.joint_in_frame_fused,
            kin_series=fused.kinematics.series,
            audit=fused.fact_sheet.audit,
            cal=fused.fact_sheet.calibration,
            events=fused.events,
            proxy_fps=30.0,
        )
        (ddir / "telemetry.json").write_text(json.dumps(tel), encoding="utf-8")
        proxy_mp4 = render_synthetic_proxy_video(
            j2d_f,
            fused.fact_sheet.audit.width,
            fused.fact_sheet.audit.height,
            ddir / "scrub_proxy.mp4",
            proxy_fps=30.0,
            title=req.label,
            mcc_wireframe_2d=project_mcc_wireframe_2d(fused.fact_sheet.calibration) if fused.fact_sheet.calibration.is_intrinsics_calibrated else None,
        )
        db.upsert_delivery(
            delivery_id=did_f,
            session_id=session_id,
            label=req.label,
            data_dir=ddir,
            video_path=str(proxy_mp4),
            proxy_path=str(proxy_mp4),
            pipeline_mode="dual_3d_ray_fusion",
            start_frame=0,
            end_frame=len(j_cam_f),
            status="ready",
            fact_sheet_json=fused.fact_sheet.model_dump_json(),
        )
        return {
            "delivery_id": did_f,
            "sync": fused.sync.__dict__,
            "alignment_method": fused.alignment_method,
            "baseline_angle_deg": fused.baseline_angle_deg,
            "baseline_sin": fused.baseline_sin,
            "fact_sheet": fused.fact_sheet.model_dump(),
        }

    @app.get("/api/bowlers/{bowler_id}/workload")
    def api_get_workload(bowler_id: str) -> Dict[str, Any]:
        b = db.get_bowler(bowler_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bowler not found.")
        logs = db.list_workload_logs(bowler_id)
        ref_date = datetime.now(timezone.utc).date()
        last_7 = 0
        last_28 = 0
        for row in logs:
            try:
                log_d = date.fromisoformat(str(row["log_date"])[:10])
            except ValueError:
                continue
            delta_days = (ref_date - log_d).days
            if 0 <= delta_days < 7:
                last_7 += int(row["deliveries_count"])
            if 0 <= delta_days < 28:
                last_28 += int(row["deliveries_count"])
        band_to_age = {"u13": 13, "u15": 15, "u17": 17, "u19": 19, "adult": 24}
        age_yrs = band_to_age.get(str(b.get("age_band", "adult")).lower(), 24)
        wg = build_workload_guidance(WorkloadContext(age_years=age_yrs, bowls_per_week=last_7))
        return {
            "bowler": b,
            "logs": logs,
            "deliveries_last_7d": last_7,
            "deliveries_last_28d": last_28,
            "guidance": wg.model_dump() if wg is not None else None,
        }

    @app.post("/api/bowlers/{bowler_id}/workload")
    def api_add_workload(bowler_id: str, req: WorkloadLogRequest) -> Dict[str, Any]:
        if not db.get_bowler(bowler_id):
            raise HTTPException(status_code=404, detail="Bowler not found.")
        entry = db.add_workload_log(bowler_id, req.log_date, req.deliveries_count, req.session_type, req.notes)
        return {"entry": entry}

    # Serve compiled Vite frontend if `web/dist` exists
    web_dist = Path(__file__).resolve().parents[3] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=str(web_dist), html=True), name="studio_ui")

    return app

