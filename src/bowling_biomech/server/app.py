"""Phase 2-3: Local-First FastAPI Coaching Studio Server.

Serves:
  - Longitudinal SQLite store (`bowlers`, `sessions`, `deliveries`, `calibration_profiles`, `workload_logs`,
    `coach_notes`, `delivery_annotations`)
  - Pre-seeded offline Demo Session (240 fps Dual-View MCC + real Steyn/McGrath fixtures) on first boot
  - Video upload + 3-tier FPS resolver + All-Intra (`-g 1 -bf 0`) scrub proxy + 3-tier Pose Engine
    (Tier 1: Warm Remote GPU Worker -> Tier 2: Local NLF TorchScript -> Tier 3: Actionable offline guide)
    with full `pipeline.py` parity (bowler-ROI blur, bowler-masked 4-DOF pan detection, upright background gravity,
    calibrated `unc_scale_factor=0.35`, `held_frame_mask`, `interp_gap_frames`)
  - `<35 ms` Instant CPU HITL recompute (4-click MCC PnP, FPS override, causal event nudging, 3-DOF pan track)
  - Multi-ball net-session auto-clipper (`scan-video` + `deliveries/from-scan` zero-reupload local slicing)
  - Bowler re-selection (`retrack-bowler`), Pillar 3 repeatability & fatigue drift (`compare`)
  - Unsynchronized Dual-Phone 3D Ray-Precision Fusion (`fuse-dual`) preserving real video proxy & scaled covariances
  - Interactive Telestrator (`annotations`), Timestamped Coach Notes (`notes`), Deterministic Numeric-Verifier Copilot
  - 1-Click `1080x1920` Truthful Breakdown Reel Export (`export-reel`)
  - Static built frontend (`web/dist`) if compiled
"""

from __future__ import annotations

from datetime import date, datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
import numpy as np
from pydantic import BaseModel, Field

from ..constants import JOINT_IDX, MCC_PITCH_LANDMARKS_M
from ..schemas import (
    BiomechanicalFactSheet,
    BowlingArm,
    CoachingStory,
    DeliveryCardSummary,
    PitchCalibration,
    VideoAudit,
    WorkloadContext,
)
from ..stage0_audit import analyze_frame_cadence, audit_video, detect_camera_motion, measure_bowler_roi_blur
from ..stage0_net_clipper import scan_net_session_video
from ..stage1_pitch_calibration import (
    auto_detect_two_wickets,
    calibrate_from_3d_motion_and_gravity,
    calibrate_from_mcc_landmarks,
    transform_cam_to_pitch_world,
    world_up_from_calibration,
)
from ..stage2_ball_release import estimate_ball_release_speed_video
from ..stage2_dual_sync_fusion import fuse_dual_view_3d_trajectories
from ..stage5_evaluator import build_compare_story, build_workload_guidance, hydrate_coaching_story
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
from .video_proxy import (
    build_all_intra_scrub_proxy,
    extract_proxy_frame_jpeg,
    probe_video_metadata,
    render_synthetic_proxy_video,
    slice_video_segment,
)

_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
_MAX_UPLOAD_BYTES = int(os.environ.get("STUDIO_MAX_UPLOAD_MB", "512")) * 1024 * 1024
_SCAN_ID_RE = re.compile(r"^scan_[a-f0-9]{10}$")


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
    bowler_id: Optional[str] = None
    bowler_name: Optional[str] = Field(default=None, max_length=120)
    bowling_arm: str = Field(default="right", pattern="^(right|left)$")
    age_band: str = Field(default="adult")
    title: str = Field(..., min_length=1, max_length=200)
    notes: str = Field(default="")


class ScanClipItem(BaseModel):
    start_frame: int = Field(..., ge=0)
    end_frame: Optional[int] = Field(default=None, ge=1)
    label: str = Field(default="Net Clip", min_length=1, max_length=160)


class ImportFromScanRequest(BaseModel):
    scan_id: str
    clips: List[ScanClipItem] = Field(..., min_length=1, max_length=30)
    bowling_arm: Optional[str] = Field(default=None, pattern="^(right|left)$")
    capture_fps_override: Optional[float] = None
    mcc_landmarks_px: Optional[Dict[str, List[float]]] = None


class RetrackBowlerRequest(BaseModel):
    anchor_xy_norm: List[float] = Field(..., min_length=2, max_length=2)


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


class SaveAnnotationsRequest(BaseModel):
    annotations: List[Dict[str, Any]] = Field(default_factory=list)


class CreateNoteRequest(BaseModel):
    frame_idx: Optional[int] = Field(default=None, ge=0)
    timestamp_s: Optional[float] = Field(default=None, ge=0.0)
    tag: str = Field(default="observation", max_length=40)
    body: str = Field(..., min_length=1, max_length=2000)


class CreateBowlerRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    bowling_arm: str = Field(default="right", pattern="^(right|left)$")
    age_band: str = Field(default="adult", max_length=32)


class PatchDeliveryRequest(BaseModel):
    label: Optional[str] = Field(default=None, min_length=1, max_length=200)


def _read_dotenv_optional() -> Dict[str, str]:
    """Read non-secret pose-engine keys (`GPU_WORKER_URL`, `NLF_MODEL_PATH`) from repo `.env` if present."""
    env_file = Path(__file__).resolve().parents[3] / ".env"
    if not env_file.exists():
        return {}
    out: Dict[str, str] = {}
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            k = k.strip()
            if k in ("GPU_WORKER_URL", "NLF_MODEL_PATH"):
                out[k] = v.strip().strip("'\"")
    except Exception:
        pass
    return out


def _humanize_ingest_error(exc: Exception) -> str:
    msg = str(getattr(exc, "detail", exc))
    low = msg.lower()
    if "pose backend" in low or "gpu_worker_url" in low or "could not reach gpu worker" in low:
        return "Analysing new videos requires the 3D pose engine, which isn't connected right now. You can explore the sample deliveries meanwhile."
    if "no frames" in low or "empty video" in low or "open video" in low:
        return "We couldn't read video frames from that file. Try exporting a standard MP4 or MOV clip from your camera roll."
    if "unsupported video" in low:
        return "That file format isn't supported yet. Please upload an MP4, MOV, or WebM video."
    if "exceeds" in low and "mb" in low:
        return "That video file is larger than the upload size limit. Trim the clip around your bowling spell and try again."
    return "We couldn't complete analysis on this clip. Make sure the bowler's full body stays inside the frame from run-up through release."


def _is_synthetic_or_fixture_delivery(rec: Dict[str, Any], fs: Optional[BiomechanicalFactSheet] = None) -> bool:
    mode = str(rec.get("pipeline_mode", ""))
    if mode in ("single_view_studio", "real_clip_fixture"):
        return True
    if fs is not None and str(fs.audit.video_path).startswith(("demo:", "fixture:", "dual:")):
        return True
    return False


def _unc_scale_for_delivery(ddir: Path, rec: Dict[str, Any]) -> float:
    meta_p = ddir / "meta.json"
    meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
    mode = str(meta.get("pipeline_mode", rec.get("pipeline_mode", "single_view_studio")))
    default_scale = 0.35 if mode in ("single_view_gpu_worker", "single_view_local_nlf", "real_clip_fixture") else 1.0
    return float(meta.get("unc_scale_factor", default_scale))


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

    dotenv_vals = _read_dotenv_optional()
    worker_url = (gpu_worker_url or os.environ.get("GPU_WORKER_URL") or dotenv_vals.get("GPU_WORKER_URL", "")).rstrip("/")

    app = FastAPI(title="SportsPostureAnalysis Coaching Studio API", version="3.0.0")
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
    app.state.jobs: Dict[str, Dict[str, Any]] = {}

    def _local_nlf_model_path() -> Optional[Path]:
        p_str = (os.environ.get("NLF_MODEL_PATH") or dotenv_vals.get("NLF_MODEL_PATH", "")).strip()
        if not p_str:
            return None
        p = Path(p_str)
        return p if p.exists() else None

    def _create_job() -> str:
        jid = f"job_{uuid.uuid4().hex[:10]}"
        app.state.jobs[jid] = {
            "status": "running",
            "stage": "finding_bowler",
            "progress": 0.12,
            "message_plain": "Finding the bowler and checking video timing...",
            "delivery_id": None,
            "error_plain": None,
        }
        return jid

    def _update_job(
        jid: Optional[str],
        *,
        status: Optional[str] = None,
        stage: Optional[str] = None,
        progress: Optional[float] = None,
        message_plain: Optional[str] = None,
        delivery_id: Optional[str] = None,
        error_plain: Optional[str] = None,
    ) -> None:
        if not jid or jid not in app.state.jobs:
            return
        entry = app.state.jobs[jid]
        if status is not None:
            entry["status"] = status
        if stage is not None:
            entry["stage"] = stage
        if progress is not None:
            entry["progress"] = round(float(progress), 2)
        if message_plain is not None:
            entry["message_plain"] = message_plain
        if delivery_id is not None:
            entry["delivery_id"] = delivery_id
        if error_plain is not None:
            entry["error_plain"] = error_plain

    def _resolve_delivery_story_and_card(
        rec_in: Dict[str, Any],
    ) -> Tuple[BiomechanicalFactSheet, CoachingStory, DeliveryCardSummary]:
        rec = rec_in if "bowler_name" in rec_in else (db.get_delivery_with_bowler(rec_in["id"]) or rec_in)
        ddir = Path(rec["data_dir"])
        meta_p = ddir / "meta.json"
        meta = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        fs = BiomechanicalFactSheet.model_validate_json(rec["fact_sheet_json"])

        mode = str(rec.get("pipeline_mode", ""))
        did = str(rec["id"])
        has_real_vid = bool(meta.get("has_real_video", False))
        if mode == "dual_3d_ray_fusion":
            card_kind = "fused_3d"
            is_sample = bool(meta.get("is_noiseless_synthetic", False))
            sample_kind: Optional[str] = "synthetic" if is_sample else None
        elif did.startswith("del_real_") and has_real_vid:
            card_kind = "sample_real"
            is_sample = True
            sample_kind = "real"
        elif mode in ("single_view_studio", "real_clip_fixture") or did.startswith(("del_demo_", "del_real_")):
            card_kind = "sample_demo"
            is_sample = True
            sample_kind = "synthetic"
        else:
            card_kind = "upload"
            is_sample = False
            sample_kind = None

        story = hydrate_coaching_story(
            fs,
            delivery_id=did,
            is_sample=is_sample,
            sample_kind=sample_kind,
        )
        thumb_frame = (
            int(fs.events.br.frame_idx)
            if fs.events.is_complete_delivery
            else max(0, int(fs.audit.num_frames) // 2)
        )
        proxy_p = Path(rec.get("proxy_path") or rec.get("video_path") or "")
        thumb_url = f"/api/deliveries/{did}/frame/{thumb_frame}.jpg?w=480" if proxy_p.exists() else None
        card = DeliveryCardSummary(
            id=did,
            session_id=str(rec["session_id"]),
            bowler_id=str(rec.get("bowler_id", "bwl_demo")),
            bowler_name=str(rec.get("bowler_name", fs.bowler_id)),
            label=str(rec["label"]),
            created_at=str(rec.get("created_at", "")),
            thumbnail_url=thumb_url,
            kind=card_kind,  # type: ignore[arg-type]
            headline=story.headline,
            top_takeaway_title=story.top_takeaway.title,
            capture_grade=story.capture.grade,
            view_label=story.capture.view_label,
            checkpoint_statuses={cp.id: cp.status for cp in story.checkpoints},
        )
        return fs, story, card

    def _run_pose_and_save_npz(
        raw_vid: Path,
        ddir: Path,
        audit: VideoAudit,
        src_indices: List[int],
        start_frame: int,
        end_frame: Optional[int],
        eff_fps: Optional[float],
        bowling_arm: Optional[str],
        cal_pre: Optional[PitchCalibration],
        anchor_xy_norm: Optional[Tuple[float, float]] = None,
    ) -> str:
        """Execute Tier 1 (Warm Remote GPU Worker) -> Tier 2 (Local NLF) -> Tier 3 (503 guide)."""
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
            if anchor_xy_norm is not None:
                form_data["anchor_xy_json"] = json.dumps([float(anchor_xy_norm[0]), float(anchor_xy_norm[1])])

            try:
                with raw_vid.open("rb") as vf:
                    resp = httpx.post(
                        f"{worker_url}/infer",
                        files={"video": (raw_vid.name, vf, "video/mp4")},
                        data=form_data,
                        timeout=180.0,
                    )
            except Exception as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"Could not reach GPU worker at {worker_url}: {exc}",
                )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"GPU worker failed ({resp.status_code}): {resp.text[:300]}",
                )
            (ddir / "raw_nlf.npz").write_bytes(resp.content)
            return "single_view_gpu_worker"

        local_nlf = _local_nlf_model_path()
        if local_nlf is not None:
            import torch
            from ..stage2_nlf_3d import extract_bml_movi_87_for_boxes, extract_bowler_3d_nlf

            dev = os.environ.get("NLF_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
            K_arr = np.asarray(cal_pre.camera_matrix_K, dtype=np.float64) if cal_pre is not None else None
            up_arr = world_up_from_calibration(cal_pre) if cal_pre is not None else None
            auto_cal_json = ""
            if K_arr is None:
                cal_tw = auto_detect_two_wickets(
                    raw_vid,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    source_frame_indices=src_indices,
                )
                if cal_tw is not None:
                    K_arr = np.asarray(cal_tw.camera_matrix_K, dtype=np.float64)
                    up_arr = world_up_from_calibration(cal_tw)
                    auto_cal_json = cal_tw.model_dump_json()

            arm_ov = BowlingArm(bowling_arm) if bowling_arm in ("right", "left") else None
            tracked = extract_bowler_3d_nlf(
                video_path=raw_vid,
                model_path=local_nlf,
                source_frame_indices=src_indices,
                num_aug=1,
                intrinsic_matrix=K_arr,
                world_up_vector=up_arr,
                bowling_arm_override=arm_ov,
                anchor_xy_norm=anchor_xy_norm,
                device=dev,
            )
            bg_up_arr = (
                np.asarray(tracked.background_upright_vectors, dtype=np.float32).reshape(-1, 3)
                if tracked.background_upright_vectors
                else np.zeros((0, 3), dtype=np.float32)
            )
            held_mask_arr = np.zeros(len(tracked.valid_mask), dtype=np.bool_)
            if audit.timebase_status == "edited_or_frozen":
                cad = analyze_frame_cadence(raw_vid, start_frame=start_frame, end_frame=end_frame)
                hm = cad.get("held_mask")
                if hm is not None and len(hm) == len(held_mask_arr):
                    held_mask_arr = np.asarray(hm, dtype=np.bool_)

            bml_cam, bml_unc = extract_bml_movi_87_for_boxes(
                video_path=raw_vid,
                model_path=local_nlf,
                source_frame_indices=src_indices,
                bowler_boxes_xyxy=tracked.bowler_boxes_xyxy,
                intrinsic_matrix=K_arr,
                world_up_vector=up_arr,
                device=dev,
            )
            np.savez_compressed(
                ddir / "raw_nlf.npz",
                joints3d_cam_m=tracked.joints3d_cam_m.astype(np.float32),
                joints2d_px=tracked.joints2d_px.astype(np.float32),
                joint_uncertainties_m=tracked.joint_uncertainties_m.astype(np.float32),
                joint_in_frame=tracked.joint_in_frame.astype(np.bool_),
                valid_mask=tracked.valid_mask.astype(np.bool_),
                interp_gap_frames=tracked.interp_gap_frames.astype(np.int32),
                bowler_boxes_xyxy=tracked.bowler_boxes_xyxy.astype(np.float32),
                background_upright_vectors=bg_up_arr,
                held_frame_mask=held_mask_arr,
                all_tracks_json=np.array(json.dumps(tracked.all_tracks_snapshot)),
                audit_json=np.array(audit.model_dump_json()),
                detected_bowling_arm=np.array(tracked.detected_bowling_arm.value),
                model_sha256=np.array(tracked.model_sha256),
                nlf_fx_used=np.array(float(K_arr[0, 0]) if K_arr is not None else 0.0, dtype=np.float64),
                auto_cal_json=np.array(auto_cal_json),
                bml_movi_87_cam_m=bml_cam.astype(np.float32),
                bml_movi_87_unc_m=bml_unc.astype(np.float32),
            )
            return "single_view_local_nlf"

        raise HTTPException(
            status_code=503,
            detail=(
                "Raw video 3D pose inference needs a pose backend: either connect the Warm Remote GPU Worker "
                "(set GPU_WORKER_URL=http://127.0.0.1:8090) OR point NLF_MODEL_PATH to a local nlf_l_multi.torchscript "
                "checkpoint. Pre-loaded demo and real-clip fixtures still run instant <35 ms CPU recompute offline."
            ),
        )

    def _finalize_ingested_npz(
        raw_vid: Path,
        ddir: Path,
        audit: VideoAudit,
        src_indices: List[int],
        bowling_arm: Optional[str],
        cal_pre: Optional[PitchCalibration],
        pipeline_mode: str,
    ) -> Tuple[VideoAudit, PitchCalibration, str, float]:
        """Full `pipeline.py` post-NLF parity: bowler-ROI blur, bowler-masked 4-DOF pan, upright gravity self-cal."""
        z = np.load(ddir / "raw_nlf.npz", allow_pickle=False)
        det_arm = str(z["detected_bowling_arm"]) if "detected_bowling_arm" in z else (bowling_arm or "right")
        fx_used = float(z["nlf_fx_used"]) if "nlf_fx_used" in z else 0.0
        boxes = z["bowler_boxes_xyxy"].astype(np.float64) if "bowler_boxes_xyxy" in z else None

        # 1. Stage 0C: Refine blur strictly inside bowler ROI & refresh warnings (pipeline.py parity!)
        if boxes is not None and len(boxes) == len(src_indices):
            roi_lap, roi_blur = measure_bowler_roi_blur(raw_vid, src_indices, boxes)
            if roi_lap > 0.0:
                refreshed_warnings = [
                    w for w in audit.warnings
                    if "Low interior sharpness" not in w and "motion blur inside bowler ROI" not in w
                ]
                if roi_blur == "severe_blur":
                    refreshed_warnings.append(
                        f"Severe motion blur inside bowler ROI (Laplacian var={roi_lap:.1f}); "
                        "3D landmark uncertainty inflated and low-SNR metrics gated."
                    )
                elif roi_blur == "moderate_blur":
                    refreshed_warnings.append(
                        f"Moderate motion blur inside bowler ROI (Laplacian var={roi_lap:.1f}); "
                        "3D pose intervals widened for shutter-speed blur."
                    )
                audit = audit.model_copy(
                    update={
                        "median_laplacian_var": roi_lap,
                        "blur_grade": roi_blur,
                        "blur_roi_method": "bowler_roi",
                        "warnings": refreshed_warnings,
                    }
                )

        # 2. Bowler-masked 4-DOF camera motion check (pan + tilt + zoom — NEVER skipped on auto_cal / MCC!)
        is_panning, _, motion_metrics = detect_camera_motion(
            video_path=raw_vid,
            bowler_boxes_xyxy=boxes,
            source_frame_indices=src_indices,
            fps_hint=audit.fps,
        )

        # 3. Stage 1B: Finalize Calibration with exact `pipeline.py` panning & gravity rules
        if cal_pre is None and "auto_cal_json" in z and str(z["auto_cal_json"]).strip():
            cal_pre = PitchCalibration.model_validate_json(str(z["auto_cal_json"]))

        if cal_pre is not None:
            cal_final = cal_pre.model_copy(
                update={
                    "is_panning_camera": bool(is_panning),
                    "is_metric_calibrated": bool(cal_pre.is_intrinsics_calibrated and (not is_panning)),
                    "camera_motion_metrics": motion_metrics,
                }
            )
        else:
            j_cam = z["joints3d_cam_m"].astype(np.float64)
            j_in = z["joint_in_frame"].astype(bool) if "joint_in_frame" in z else None
            j_unc = z["joint_uncertainties_m"].astype(np.float64) if "joint_uncertainties_m" in z else None
            v_mask = z["valid_mask"].astype(bool) if "valid_mask" in z else None
            bg_up_list = (
                [np.asarray(v, dtype=np.float64) for v in z["background_upright_vectors"]]
                if ("background_upright_vectors" in z and z["background_upright_vectors"].size > 0)
                else None
            )
            cal_final = calibrate_from_3d_motion_and_gravity(
                pelvis_traj_cam_m=j_cam[:, JOINT_IDX["pelv"]],
                up_vectors_cam=j_cam[:, JOINT_IDX["neck"]] - j_cam[:, JOINT_IDX["pelv"]],
                width=audit.width,
                height=audit.height,
                joints3d_cam_m=j_cam,
                joint_in_frame=j_in,
                joint_uncertainties_m=j_unc,
                valid_mask=v_mask,
                bowling_arm_is_right=(det_arm == "right"),
                is_panning_camera=is_panning,
                background_upright_vectors=bg_up_list,
            ).model_copy(update={"camera_motion_metrics": motion_metrics})

        (ddir / "audit.json").write_text(audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(cal_final.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps(
                {
                    "bowling_arm": det_arm,
                    "pipeline_mode": pipeline_mode,
                    "nlf_fx_used": fx_used if fx_used > 0 else float(cal_final.camera_matrix_K[0][0]),
                    "unc_scale_factor": 0.35,
                    "is_noiseless_synthetic": False,
                }
            ),
            encoding="utf-8",
        )
        return audit, cal_final, det_arm, fx_used

    def _ingest_saved_video_to_delivery(
        session_id: str,
        raw_vid: Path,
        ddir: Path,
        did: str,
        label: str,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        capture_fps_override: Optional[float] = None,
        bowling_arm: Optional[str] = None,
        mcc_landmarks_px: Optional[Dict[str, List[float]]] = None,
        job_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        jid = job_id or _create_job()
        try:
            _update_job(jid, stage="finding_bowler", progress=0.18, message_plain="Checking video quality and preparing smooth playback...")
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
            if mcc_landmarks_px and len(mcc_landmarks_px) >= 4:
                lm_map = {k: (float(v[0]), float(v[1])) for k, v in mcc_landmarks_px.items()}
                cal_pre = calibrate_from_mcc_landmarks(lm_map, audit.width, audit.height)

            _update_job(jid, stage="finding_bowler", progress=0.45, message_plain="Tracking the bowler in 3D across every frame...")
            pipeline_mode = _run_pose_and_save_npz(
                raw_vid=raw_vid,
                ddir=ddir,
                audit=audit,
                src_indices=src_indices,
                start_frame=start_frame,
                end_frame=end_frame,
                eff_fps=eff_fps,
                bowling_arm=bowling_arm,
                cal_pre=cal_pre,
            )
            _update_job(jid, stage="spotting_moments", progress=0.72, message_plain="Spotting back-foot land, front-foot plant, and ball release...")
            _finalize_ingested_npz(
                raw_vid=raw_vid,
                ddir=ddir,
                audit=audit,
                src_indices=src_indices,
                bowling_arm=bowling_arm,
                cal_pre=cal_pre,
                pipeline_mode=pipeline_mode,
            )
            overrides: Dict[str, Any] = {}
            if mcc_landmarks_px and len(mcc_landmarks_px) >= 4:
                overrides["mcc_landmarks_px"] = mcc_landmarks_px
            _update_job(jid, stage="measuring", progress=0.88, message_plain="Measuring body positions and uncertainty ranges...")
            res = recompute_delivery_cpu(ddir, bowler_id=label, overrides=overrides, video_path=raw_vid, proxy_fps=30.0)
            saved_rec = db.upsert_delivery(
                delivery_id=did,
                session_id=session_id,
                label=label,
                data_dir=ddir,
                video_path=str(raw_vid),
                proxy_path=str(proxy_mp4),
                pipeline_mode=pipeline_mode,
                start_frame=start_frame,
                end_frame=end_frame,
                status="ready",
                fact_sheet_json=res["fact_sheet"].model_dump_json(),
                isb_summary=res["isb_summary"],
                pan_summary=res["pan_summary"],
                overrides=overrides,
            )
            _, story_obj, card_obj = _resolve_delivery_story_and_card(saved_rec)
            _update_job(jid, status="done", stage="writing_story", progress=1.0, message_plain="Analysis complete.", delivery_id=did)
            return {
                "job_id": jid,
                "session_id": session_id,
                "delivery_id": did,
                "label": label,
                "pipeline_mode": pipeline_mode,
                "probed_meta": probed.__dict__,
                "recompute_ms": res["recompute_ms"],
                "fact_sheet": res["fact_sheet"].model_dump(),
                "story": story_obj.model_dump(),
                "card": card_obj.model_dump(),
            }
        except Exception as exc:
            _update_job(jid, status="failed", error_plain=_humanize_ingest_error(exc), message_plain="Could not complete analysis.")
            raise

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
        has_local_nlf = _local_nlf_model_path() is not None
        if worker_status.get("reachable"):
            pose_engine = "remote_gpu_worker"
        elif has_local_nlf:
            pose_engine = "local_nlf_model"
        else:
            pose_engine = "cpu_recompute_only"
        return {
            "status": "ok",
            "version": "3.0.0",
            "mode": "local_first_studio",
            "pose_engine": pose_engine,
            "local_nlf_configured": has_local_nlf,
            "gpu_worker": worker_status,
            "mcc_landmark_keys": list(MCC_PITCH_LANDMARKS_M.keys()),
        }

    @app.get("/api/overview")
    def api_overview() -> Dict[str, Any]:
        h = api_health()
        pe = str(h.get("pose_engine", "cpu_recompute_only"))
        can_analyse = pe in ("remote_gpu_worker", "local_nlf_model")
        if pe == "remote_gpu_worker":
            engine_lbl = "Fast GPU analysis ready"
            reason_plain = None
        elif pe == "local_nlf_model":
            engine_lbl = "Local 3D pose model ready"
            reason_plain = None
        else:
            engine_lbl = "Sample mode (3D pose engine offline)"
            reason_plain = "Analysing new videos needs the 3D pose engine, which isn't connected right now. You can explore all sample deliveries immediately."

        all_rows = db.list_all_deliveries(limit=60, ascending=False)
        cards: List[DeliveryCardSummary] = []
        cards_by_id: Dict[str, DeliveryCardSummary] = {}
        for row in all_rows:
            if not row.get("fact_sheet_json"):
                continue
            try:
                _, _, card = _resolve_delivery_story_and_card(row)
                cards.append(card)
                cards_by_id[card.id] = card
            except Exception:
                continue

        has_user_upload = any(c.kind == "upload" for c in cards)
        sample_defs = [
            ("del_real_steyn", "Dale Steyn — Real Net Video", "Filmed from behind · 25 fps real clip"),
            ("del_demo_side_240", "240 fps Side-On Sample", "Calibrated slow-motion · 19 of 22 checks visible"),
            ("del_demo_side_b2", "Spell Ball 2 — Comparison Sample", "Side-on 240 fps · compare against Ball 1"),
            ("del_demo_rear_240", "240 fps Behind-Bowler Sample", "Rear slow-motion · unlocks spine & alignment"),
            ("del_real_mcgrath", "Incomplete Clip Guardrail Example", "Shows how truncated clips are safely withheld"),
        ]
        samples: List[Dict[str, Any]] = []
        for sid_d, s_title, s_sub in sample_defs:
            c = cards_by_id.get(sid_d)
            if c is None:
                continue
            samples.append(
                {
                    "delivery_id": c.id,
                    "title": s_title if (sid_d != "del_real_steyn" or c.kind == "sample_real") else "Dale Steyn — Net Sample",
                    "subtitle": s_sub,
                    "kind": "real" if c.kind == "sample_real" else "synthetic",
                    "thumbnail_url": c.thumbnail_url,
                    "capture_grade": c.capture_grade,
                }
            )

        if not has_user_upload:
            preferred_order = ["del_real_steyn", "del_demo_side_240", "del_demo_side_b2", "del_demo_rear_240", "del_real_mcgrath"]
            rank = {did: idx for idx, did in enumerate(preferred_order)}
            recent = sorted(cards, key=lambda item: rank.get(item.id, 99))[:24]
        else:
            uploads = [c for c in cards if c.kind == "upload"]
            others = [c for c in cards if c.kind != "upload"]
            recent = (uploads + others)[:24]

        return {
            "is_first_run": not has_user_upload,
            "health": {
                "pose_engine": pe,
                "engine_label": engine_lbl,
                "can_analyse_new_video": can_analyse,
                "reason_plain": reason_plain,
            },
            "bowlers": db.list_bowlers_with_stats(),
            "samples": samples,
            "recent_deliveries": [c.model_dump() for c in recent],
        }

    @app.get("/api/jobs/{job_id}")
    def api_get_job(job_id: str) -> Dict[str, Any]:
        job = app.state.jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job not found.")
        return job

    @app.get("/api/bowlers")
    def api_list_bowlers() -> Dict[str, Any]:
        return {"bowlers": db.list_bowlers_with_stats()}

    @app.post("/api/bowlers")
    def api_create_bowler(req: CreateBowlerRequest) -> Dict[str, Any]:
        clean_name = req.name.strip()
        if not clean_name:
            raise HTTPException(status_code=400, detail="Bowler name cannot be empty.")
        bowler = db.create_bowler(clean_name, bowling_arm=req.bowling_arm, age_band=req.age_band.strip().lower() or "adult")
        bowler_with_stats = {**bowler, "delivery_count": 0, "last_delivery_at": None}
        return {"bowler": bowler_with_stats}

    @app.get("/api/bowlers/{bowler_id}/progress")
    def api_bowler_progress(bowler_id: str) -> Dict[str, Any]:
        b = db.get_bowler(bowler_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bowler not found.")
        rows = db.list_all_deliveries(bowler_id=bowler_id, limit=60, ascending=True)
        deliv_items: List[Dict[str, Any]] = []
        cp_titles = {
            "front_leg_brace": "Front-Leg Brace",
            "trunk_whip": "Trunk Whip",
            "side_bend": "Side-Bend Load",
            "shoulder_alignment": "Shoulder Alignment",
            "runup_stride": "Run-Up & Stride",
        }
        series_map: Dict[str, List[Dict[str, Any]]] = {k: [] for k in cp_titles}
        complete_count = 0

        for r in rows:
            if not r.get("fact_sheet_json"):
                continue
            try:
                fs_i, st_i, card_i = _resolve_delivery_story_and_card(r)
            except Exception:
                continue
            if fs_i.events.is_complete_delivery:
                complete_count += 1
            deliv_items.append(
                {
                    "id": card_i.id,
                    "label": card_i.label,
                    "created_at": card_i.created_at,
                    "thumbnail_url": card_i.thumbnail_url,
                }
            )
            for cp in st_i.checkpoints:
                series_map[cp.id].append(
                    {
                        "delivery_id": card_i.id,
                        "status": cp.status,
                        "value": cp.value,
                        "ci_low": cp.ci_low,
                        "ci_high": cp.ci_high,
                        "value_text": cp.value_text,
                        "range_text": cp.range_text,
                    }
                )

        wkl_data = api_get_workload(bowler_id)
        if complete_count == 0:
            summary_plain = "Upload or select a completed delivery for this bowler to track checkpoint trends over time."
        elif complete_count == 1:
            summary_plain = "One completed delivery recorded so far — film a second ball from the same camera angle to compare progress."
        else:
            summary_plain = f"Tracking {complete_count} completed deliveries across 5 body checkpoints. Tap any dot to open that ball's story."

        return {
            "bowler": {**b, "delivery_count": len(deliv_items), "last_delivery_at": deliv_items[-1]["created_at"] if deliv_items else None},
            "deliveries": deliv_items,
            "series": [
                {"checkpoint_id": cid, "title": title, "points": series_map[cid]}
                for cid, title in cp_titles.items()
            ],
            "workload": wkl_data.get("guidance"),
            "summary_plain": summary_plain,
        }

    @app.post("/api/bowlers/{bowler_id}/scan-video")
    async def api_bowler_scan_video(bowler_id: str, video: UploadFile = File(...)) -> Dict[str, Any]:
        if not db.get_bowler(bowler_id):
            raise HTTPException(status_code=404, detail="Bowler not found.")
        ses = db.get_or_create_dated_session(bowler_id)
        res = await api_scan_video(ses["id"], video=video)
        return {"session_id": ses["id"], **res}

    @app.post("/api/bowlers/{bowler_id}/deliveries/from-scan")
    def api_bowler_import_from_scan(bowler_id: str, req: ImportFromScanRequest) -> Dict[str, Any]:
        b = db.get_bowler(bowler_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bowler not found.")
        if not _SCAN_ID_RE.match(req.scan_id):
            raise HTTPException(status_code=400, detail="Invalid scan_id format.")
        manifest_path = uploads_root / f"{req.scan_id}.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Scan video not found or expired.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        ses_id = str(manifest.get("session_id", ""))
        ses = db.get_session(ses_id)
        if not ses or ses.get("bowler_id") != bowler_id:
            raise HTTPException(status_code=403, detail="Scan video does not belong to this bowler.")
        req_with_arm = req.model_copy(update={"bowling_arm": req.bowling_arm or b.get("bowling_arm", "right")})
        out = api_import_from_scan(ses_id, req_with_arm)
        first_jid = out["deliveries"][0].get("job_id") if out.get("deliveries") else None
        return {"session_id": ses_id, "job_id": first_jid, **out}

    @app.post("/api/bowlers/{bowler_id}/deliveries/upload")
    async def api_bowler_upload_delivery(
        bowler_id: str,
        video: UploadFile = File(...),
        label: str = Form("Uploaded Delivery"),
        start_frame: int = Form(0),
        end_frame: Optional[int] = Form(None),
        capture_fps_override: Optional[float] = Form(None),
        bowling_arm: Optional[str] = Form(None),
        mcc_landmarks_json: Optional[str] = Form(None),
    ) -> Dict[str, Any]:
        b = db.get_bowler(bowler_id)
        if not b:
            raise HTTPException(status_code=404, detail="Bowler not found.")
        ses = db.get_or_create_dated_session(bowler_id)
        use_arm = bowling_arm if bowling_arm in ("right", "left") else b.get("bowling_arm", "right")
        return await api_upload_delivery(
            session_id=ses["id"],
            video=video,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
            capture_fps_override=capture_fps_override,
            bowling_arm=use_arm,
            mcc_landmarks_json=mcc_landmarks_json,
        )

    @app.get("/api/compare")
    def api_compare_two_deliveries(a: str, b: str) -> Dict[str, Any]:
        rec_a = db.get_delivery_with_bowler(a)
        rec_b = db.get_delivery_with_bowler(b)
        if not rec_a or not rec_a.get("fact_sheet_json"):
            raise HTTPException(status_code=404, detail=f"Delivery '{a}' not found.")
        if not rec_b or not rec_b.get("fact_sheet_json"):
            raise HTTPException(status_code=404, detail=f"Delivery '{b}' not found.")
        fs_a, st_a, card_a = _resolve_delivery_story_and_card(rec_a)
        fs_b, st_b, card_b = _resolve_delivery_story_and_card(rec_b)
        comp = build_compare_story(card_a, fs_a, st_a, card_b, fs_b, st_b)
        return comp.model_dump()

    @app.get("/api/sessions")
    def api_list_sessions() -> Dict[str, Any]:
        return {"sessions": db.list_sessions()}

    @app.post("/api/sessions")
    def api_create_session(req: CreateSessionRequest) -> Dict[str, Any]:
        if req.bowler_id:
            bowler = db.get_bowler(req.bowler_id)
            if not bowler:
                raise HTTPException(status_code=404, detail=f"Bowler '{req.bowler_id}' not found.")
        else:
            if not req.bowler_name or not req.bowler_name.strip():
                raise HTTPException(status_code=400, detail="Provide either an existing bowler_id or a new bowler_name.")
            bowler = db.create_bowler(req.bowler_name.strip(), bowling_arm=req.bowling_arm, age_band=req.age_band)
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
        """Scan an uploaded multi-ball net video at ~6 fps, cache it under `scan_id`, and return candidate clips."""
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        scan_id = f"scan_{uuid.uuid4().hex[:10]}"
        dest = await _save_upload_bounded(video, uploads_root, scan_id)
        meta = probe_video_metadata(dest)
        cands = scan_net_session_video(dest, include_thumbnails=True)
        manifest = {
            "scan_id": scan_id,
            "session_id": session_id,
            "video_path": str(dest),
            "filename": video.filename or dest.name,
            "probed_capture_fps": meta.quicktime_capture_fps if meta.is_probable_slomo_container else None,
        }
        (uploads_root / f"{scan_id}.json").write_text(json.dumps(manifest), encoding="utf-8")
        return {
            "scan_id": scan_id,
            "upload_path": str(dest),
            "video_meta": meta.__dict__,
            "candidates": [c.__dict__ for c in cands],
        }

    @app.post("/api/sessions/{session_id}/deliveries/from-scan")
    def api_import_from_scan(session_id: str, req: ImportFromScanRequest) -> Dict[str, Any]:
        """Slice selected balls locally from a scanned net-session video (`scan_id`) without double-uploading."""
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")
        if not _SCAN_ID_RE.match(req.scan_id):
            raise HTTPException(status_code=400, detail="Invalid scan_id format.")
        manifest_path = uploads_root / f"{req.scan_id}.json"
        if not manifest_path.exists():
            raise HTTPException(status_code=404, detail="Scan video not found or expired.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("session_id") != session_id:
            raise HTTPException(status_code=403, detail="Scan video does not belong to this session.")
        src_vid = Path(manifest["video_path"])
        if not src_vid.exists():
            raise HTTPException(status_code=404, detail="Scanned source video file missing on disk.")

        imported: List[Dict[str, Any]] = []
        for clip in req.clips:
            did = f"del_{uuid.uuid4().hex[:10]}"
            ddir = sessions_root / session_id / did
            ddir.mkdir(parents=True, exist_ok=True)
            sliced_vid = slice_video_segment(src_vid, ddir / "source.mp4", clip.start_frame, clip.end_frame)
            item = _ingest_saved_video_to_delivery(
                session_id=session_id,
                raw_vid=sliced_vid,
                ddir=ddir,
                did=did,
                label=clip.label,
                start_frame=0,
                end_frame=None,
                capture_fps_override=req.capture_fps_override or manifest.get("probed_capture_fps"),
                bowling_arm=req.bowling_arm,
                mcc_landmarks_px=req.mcc_landmarks_px,
            )
            imported.append(item)
        return {"scan_id": req.scan_id, "deliveries": imported}

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
        if not db.get_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found.")

        did = f"del_{uuid.uuid4().hex[:10]}"
        ddir = sessions_root / session_id / did
        ddir.mkdir(parents=True, exist_ok=True)
        raw_vid = await _save_upload_bounded(video, ddir, "source")

        lm_dict: Optional[Dict[str, List[float]]] = None
        if mcc_landmarks_json:
            parsed = json.loads(mcc_landmarks_json)
            if isinstance(parsed, dict) and len(parsed) >= 4:
                lm_dict = {k: [float(v[0]), float(v[1])] for k, v in parsed.items()}

        return _ingest_saved_video_to_delivery(
            session_id=session_id,
            raw_vid=raw_vid,
            ddir=ddir,
            did=did,
            label=label,
            start_frame=start_frame,
            end_frame=end_frame,
            capture_fps_override=capture_fps_override,
            bowling_arm=bowling_arm,
            mcc_landmarks_px=lm_dict,
        )

    @app.get("/api/deliveries/{delivery_id}")
    def api_get_delivery(delivery_id: str) -> Dict[str, Any]:
        rec = db.get_delivery_with_bowler(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        ddir = Path(rec["data_dir"])
        fs_dict: Optional[Dict[str, Any]] = None
        story_dict: Optional[Dict[str, Any]] = None
        card_dict: Optional[Dict[str, Any]] = None
        if rec.get("fact_sheet_json"):
            fs_obj, story_obj, card_obj = _resolve_delivery_story_and_card(rec)
            fs_dict = fs_obj.model_copy(update={"coaching_story": story_obj}).model_dump()
            story_dict = story_obj.model_dump()
            card_dict = card_obj.model_dump()
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
            "card": card_dict,
            "story": story_dict,
            "fact_sheet": fs_dict,
            "telemetry": tel,
            "report_markdown": rep_md,
            "notes": db.list_notes(delivery_id),
            "annotations": db.get_annotations(delivery_id),
            "mcc_landmarks_3d": {k: list(v) for k, v in MCC_PITCH_LANDMARKS_M.items()},
        }

    @app.patch("/api/deliveries/{delivery_id}")
    def api_patch_delivery(delivery_id: str, req: PatchDeliveryRequest) -> Dict[str, Any]:
        rec = db.get_delivery_with_bowler(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        new_label = (req.label or rec["label"]).strip()
        if not new_label:
            raise HTTPException(status_code=400, detail="Delivery label cannot be empty.")
        new_fs_json = rec.get("fact_sheet_json")
        if new_fs_json:
            fs_old = BiomechanicalFactSheet.model_validate_json(new_fs_json)
            fs_upd = fs_old.model_copy(update={"bowler_id": new_label})
            new_fs_json = fs_upd.model_dump_json()
            fs_disk = Path(rec["data_dir"]) / "fact_sheet.json"
            if fs_disk.exists():
                fs_disk.write_text(fs_upd.model_dump_json(indent=2), encoding="utf-8")
        updated = db.rename_delivery(delivery_id, new_label, fact_sheet_json=new_fs_json)
        if not updated:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        _, story_obj, card_obj = _resolve_delivery_story_and_card(updated)
        return {
            "delivery": card_obj.model_dump(),
            "card": card_obj.model_dump(),
            "story": story_obj.model_dump(),
        }

    @app.get("/api/deliveries/{delivery_id}/frame/{frame_idx}.jpg")
    def api_get_delivery_frame_jpg(delivery_id: str, frame_idx: int, w: int = 480) -> Response:
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        p = Path(rec["proxy_path"] or rec["video_path"])
        if not p.exists():
            raise HTTPException(status_code=404, detail="Scrub proxy video not found on disk.")
        cache_dir = Path(rec["data_dir"]) / "frames"
        try:
            jpg_bytes = extract_proxy_frame_jpeg(p, frame_idx=frame_idx, cache_dir=cache_dir, max_width=w)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not extract frame JPEG: {exc}")
        return Response(
            content=jpg_bytes,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    @app.delete("/api/deliveries/{delivery_id}")
    def api_delete_delivery(delivery_id: str) -> Dict[str, Any]:
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        deleted = db.delete_delivery(delivery_id)
        try:
            ddir = Path(rec["data_dir"]).resolve()
            if sessions_root.resolve() in ddir.parents and ddir.exists():
                shutil.rmtree(ddir, ignore_errors=True)
        except Exception:
            pass
        return {"deleted": deleted, "delivery_id": delivery_id}

    @app.post("/api/deliveries/{delivery_id}/retrack-bowler")
    def api_retrack_bowler(delivery_id: str, req: RetrackBowlerRequest) -> Dict[str, Any]:
        """Re-run 3D bowler tracking locked to a user-clicked normalized `(x, y)` anchor point."""
        rec = db.get_delivery(delivery_id)
        if not rec:
            raise HTTPException(status_code=404, detail="Delivery not found.")
        fs_prev = BiomechanicalFactSheet.model_validate_json(rec["fact_sheet_json"]) if rec.get("fact_sheet_json") else None
        if _is_synthetic_or_fixture_delivery(rec, fs_prev):
            raise HTTPException(
                status_code=400,
                detail="Bowler re-tracking requires a raw uploaded video clip (not a synthetic demo or cached fixture).",
            )
        raw_vid = Path(rec["video_path"])
        if not raw_vid.exists():
            raise HTTPException(status_code=404, detail="Source video not found on disk for re-tracking.")

        ax = float(np.clip(req.anchor_xy_norm[0], 0.0, 1.0))
        ay = float(np.clip(req.anchor_xy_norm[1], 0.0, 1.0))
        ddir = Path(rec["data_dir"])
        audit = VideoAudit.model_validate_json((ddir / "audit.json").read_text(encoding="utf-8"))
        src_indices = audit.source_frame_indices or list(range(rec.get("start_frame", 0), rec.get("start_frame", 0) + audit.num_frames))
        ov = dict(rec.get("overrides") or {})
        cal_pre = None
        if ov.get("mcc_landmarks_px") and len(ov["mcc_landmarks_px"]) >= 4:
            lm_map = {k: (float(v[0]), float(v[1])) for k, v in ov["mcc_landmarks_px"].items()}
            cal_pre = calibrate_from_mcc_landmarks(lm_map, audit.width, audit.height)

        pipeline_mode = _run_pose_and_save_npz(
            raw_vid=raw_vid,
            ddir=ddir,
            audit=audit,
            src_indices=src_indices,
            start_frame=int(rec.get("start_frame") or 0),
            end_frame=rec.get("end_frame"),
            eff_fps=ov.get("capture_fps_override"),
            bowling_arm=ov.get("bowling_arm"),
            cal_pre=cal_pre,
            anchor_xy_norm=(ax, ay),
        )
        _finalize_ingested_npz(
            raw_vid=raw_vid,
            ddir=ddir,
            audit=audit,
            src_indices=src_indices,
            bowling_arm=ov.get("bowling_arm"),
            cal_pre=cal_pre,
            pipeline_mode=pipeline_mode,
        )
        res = recompute_delivery_cpu(ddir, bowler_id=rec["label"], overrides=ov, video_path=raw_vid, proxy_fps=30.0)
        saved_rec = db.upsert_delivery(
            delivery_id=delivery_id,
            session_id=rec["session_id"],
            label=rec["label"],
            data_dir=ddir,
            video_path=rec["video_path"],
            proxy_path=rec["proxy_path"],
            pipeline_mode=pipeline_mode,
            start_frame=rec["start_frame"],
            end_frame=rec["end_frame"],
            status="ready",
            fact_sheet_json=res["fact_sheet"].model_dump_json(),
            isb_summary=res["isb_summary"],
            pan_summary=res["pan_summary"],
            overrides=ov,
        )
        _, story_obj, _ = _resolve_delivery_story_and_card(saved_rec)
        return {
            "delivery_id": delivery_id,
            "anchor_xy_norm": [round(ax, 4), round(ay, 4)],
            "recompute_ms": res["recompute_ms"],
            "fact_sheet": res["fact_sheet"].model_dump(),
            "story": story_obj.model_dump(),
            "telemetry": res["telemetry"],
            "report_markdown": res["report_markdown"],
            "isb_summary": res["isb_summary"],
            "pan_summary": res["pan_summary"],
        }

    @app.get("/api/deliveries/{delivery_id}/annotations")
    def api_get_annotations(delivery_id: str) -> Dict[str, Any]:
        if not db.get_delivery(delivery_id):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        return {"delivery_id": delivery_id, "annotations": db.get_annotations(delivery_id)}

    @app.put("/api/deliveries/{delivery_id}/annotations")
    def api_save_annotations(delivery_id: str, req: SaveAnnotationsRequest) -> Dict[str, Any]:
        if not db.get_delivery(delivery_id):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        saved = db.save_annotations(delivery_id, req.annotations)
        return {"delivery_id": delivery_id, "annotations": saved}

    @app.get("/api/deliveries/{delivery_id}/notes")
    def api_list_notes(delivery_id: str) -> Dict[str, Any]:
        if not db.get_delivery(delivery_id):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        return {"delivery_id": delivery_id, "notes": db.list_notes(delivery_id)}

    @app.post("/api/deliveries/{delivery_id}/notes")
    def api_add_note(delivery_id: str, req: CreateNoteRequest) -> Dict[str, Any]:
        if not db.get_delivery(delivery_id):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        note = db.add_note(delivery_id, req.body.strip(), frame_idx=req.frame_idx, timestamp_s=req.timestamp_s, tag=req.tag)
        return {"note": note}

    @app.delete("/api/deliveries/{delivery_id}/notes/{note_id}")
    def api_delete_note(delivery_id: str, note_id: str) -> Dict[str, Any]:
        if not db.get_delivery(delivery_id):
            raise HTTPException(status_code=404, detail="Delivery not found.")
        ok = db.delete_note(delivery_id, note_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Note not found.")
        return {"deleted": True, "note_id": note_id}

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
        saved_rec = db.upsert_delivery(
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
        _, story_obj, _ = _resolve_delivery_story_and_card(saved_rec)
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
            "story": story_obj.model_dump(),
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

        ddir_a = Path(rec_a["data_dir"])
        ddir_b = Path(rec_b["data_dir"])
        za = np.load(ddir_a / "raw_nlf.npz", allow_pickle=False)
        zb = np.load(ddir_b / "raw_nlf.npz", allow_pickle=False)
        jwa = transform_cam_to_pitch_world(za["joints3d_cam_m"].astype(np.float64), fs_a.calibration)
        jwb = transform_cam_to_pitch_world(zb["joints3d_cam_m"].astype(np.float64), fs_b.calibration)
        unc_scale_a = _unc_scale_for_delivery(ddir_a, rec_a)
        unc_scale_b = _unc_scale_for_delivery(ddir_b, rec_b)

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
                unc_scale_a=unc_scale_a,
                unc_scale_b=unc_scale_b,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        # Evaluate optical ball speed on whichever real source view is best side-on (Bug #7 / P1-2)
        fused_fs = fused.fact_sheet
        cached_ball_metric: Optional[Dict[str, Any]] = None
        real_candidates = [
            (abs(math.sin(math.radians(fs_a.calibration.azimuth_deg))), rec_a, fs_a, za, jwa, unc_scale_a),
            (abs(math.sin(math.radians(fs_b.calibration.azimuth_deg))), rec_b, fs_b, zb, jwb, unc_scale_b),
        ]
        real_candidates.sort(key=lambda item: item[0], reverse=True)
        for _, rec_s, fs_s, zs, jws, us in real_candidates:
            if _is_synthetic_or_fixture_delivery(rec_s, fs_s):
                continue
            vp = Path(rec_s["video_path"]) if rec_s.get("video_path") else None
            if vp is None or not vp.exists():
                continue
            wri_i = JOINT_IDX["rwri" if fs_s.bowling_arm == BowlingArm.RIGHT else "lwri"]
            u_arr = zs["joint_uncertainties_m"].astype(np.float64)
            wri_u = float(u_arr[min(fs_s.events.br.frame_idx, len(u_arr) - 1), wri_i]) * us
            ball_res = estimate_ball_release_speed_video(
                video_path=vp,
                joints2d_px=zs["joints2d_px"].astype(np.float64),
                joints_world_m=jws,
                audit=fs_s.audit,
                cal=fs_s.calibration,
                events=fs_s.events,
                bowling_arm=fs_s.bowling_arm,
                wrist_unc_m=max(wri_u, 0.12),
            )
            if not ball_res.metric.suppressed:
                new_metrics = dict(fused_fs.metrics)
                new_metrics["ball_release_speed_kph"] = ball_res.metric
                fused_fs = fused_fs.model_copy(update={"metrics": new_metrics})
                cached_ball_metric = ball_res.metric.model_dump()
                break

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
        (ddir / "audit.json").write_text(fused_fs.audit.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "base_calibration.json").write_text(fused_fs.calibration.model_dump_json(indent=2), encoding="utf-8")
        (ddir / "meta.json").write_text(
            json.dumps(
                {
                    "bowling_arm": fs_a.bowling_arm.value,
                    "pipeline_mode": "dual_3d_ray_fusion",
                    "dual_baseline_sin": float(fused.baseline_sin),
                    "nlf_fx_used": float(Ka[0, 0]),
                    "unc_scale_factor": 1.0,
                    "is_noiseless_synthetic": bool(_is_synthetic_or_fixture_delivery(rec_a, fs_a)),
                    "source_delivery_ids": [req.delivery_id_a, req.delivery_id_b],
                    "cached_ball_speed_metric": cached_ball_metric,
                }
            ),
            encoding="utf-8",
        )
        (ddir / "fact_sheet.json").write_text(fused_fs.model_dump_json(indent=2), encoding="utf-8")
        rep = ask_grounded_coach("Give a complete verified summary of this dual-view fused delivery.", fused_fs)
        (ddir / "coaching_report.md").write_text(rep.answer_markdown, encoding="utf-8")
        tel = build_telemetry_v2(
            joints_world_m=fused.joints_world_fused,
            joints2d_px=j2d_f,
            joint_uncertainties_m=fused.joint_uncertainties_fused,
            joint_in_frame=fused.joint_in_frame_fused,
            kin_series=fused.kinematics.series,
            audit=fused_fs.audit,
            cal=fused_fs.calibration,
            events=fused.events,
            proxy_fps=30.0,
        )
        (ddir / "telemetry.json").write_text(json.dumps(tel), encoding="utf-8")

        # Preserve Camera A's real video proxy aligned to exact fused overlap timeline `t_grid_a` (Bug #7)
        proxy_mp4: Optional[Path] = None
        if not _is_synthetic_or_fixture_delivery(rec_a, fs_a):
            real_src_a = Path(rec_a["video_path"]) if rec_a.get("video_path") else None
            if real_src_a and real_src_a.exists():
                src_a_idx = fs_a.audit.source_frame_indices or list(range(len(jwa)))
                fused_fps = max(float(fused_fs.audit.fps), 1.0)
                t_grid_a = fused.sync.overlap_start_a_s + np.arange(len(j_cam_f), dtype=np.float64) / fused_fps
                raw_a_pos = np.clip(np.round(t_grid_a * max(float(fs_a.audit.fps), 1.0)).astype(int), 0, len(src_a_idx) - 1)
                fused_idx = [int(src_a_idx[int(pos)]) for pos in raw_a_pos]
                if len(fused_idx) == len(j_cam_f):
                    try:
                        proxy_mp4 = build_all_intra_scrub_proxy(
                            real_src_a, ddir / "scrub_proxy.mp4", source_frame_indices=fused_idx, proxy_fps=30.0
                        )
                    except Exception:
                        proxy_mp4 = None
        if proxy_mp4 is None:
            proxy_mp4 = render_synthetic_proxy_video(
                j2d_f,
                fused_fs.audit.width,
                fused_fs.audit.height,
                ddir / "scrub_proxy.mp4",
                proxy_fps=30.0,
                title=req.label,
                mcc_wireframe_2d=project_mcc_wireframe_2d(fused_fs.calibration) if fused_fs.calibration.is_intrinsics_calibrated else None,
            )

        saved_fused = db.upsert_delivery(
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
            fact_sheet_json=fused_fs.model_dump_json(),
        )
        _, fused_story, _ = _resolve_delivery_story_and_card(saved_fused)
        return {
            "delivery_id": did_f,
            "sync": fused.sync.__dict__,
            "alignment_method": fused.alignment_method,
            "baseline_angle_deg": fused.baseline_angle_deg,
            "baseline_sin": fused.baseline_sin,
            "fact_sheet": fused_fs.model_dump(),
            "story": fused_story.model_dump(),
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
