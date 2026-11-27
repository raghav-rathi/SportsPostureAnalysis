"""Phase 2 (M2A): Warm Remote GPU Worker Daemon for Neural Localizer Fields (`nlf_l_multi.torchscript`).

Keeps NLF TorchScript warm in GPU VRAM (~3.4 GB) to cut per-delivery inference from ~78 s cold JIT
to ~6.8 s steady-state warm (SMPL-24 + 87-marker `bml_movi_87`), with a content-addressed SHA-256
disk cache (`raw_nlf.npz`, ~8 ms hit) strictly isolated under `GPU_WORKER_CACHE_DIR`.

Run via:
  CUDA_VISIBLE_DEVICES=1 NLF_MODEL_PATH=/path/to/nlf_l_multi.torchscript \
  GPU_WORKER_CACHE_DIR=/path/to/cache/gpu_worker \
  uvicorn bowling_biomech.gpu_worker.worker:app --host 127.0.0.1 --port 8090
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
import numpy as np

from ..schemas import BowlingArm
from ..stage0_audit import analyze_frame_cadence, audit_video
from ..stage1_pitch_calibration import auto_detect_two_wickets, world_up_from_calibration
from ..stage2_nlf_3d import (
    _file_sha256_prefix,
    _load_nlf_cached,
    extract_bml_movi_87_for_boxes,
    extract_bowler_3d_nlf,
)

_ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".webm"}
_MAX_UPLOAD_BYTES = int(os.environ.get("GPU_WORKER_MAX_UPLOAD_MB", "512")) * 1024 * 1024


def _worker_cache_dir() -> Path:
    d = Path(os.environ.get("GPU_WORKER_CACHE_DIR", "cache/gpu_worker"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _model_path() -> str:
    p = os.environ.get("NLF_MODEL_PATH", "")
    if not p or not Path(p).exists():
        raise RuntimeError("NLF_MODEL_PATH environment variable must point to nlf_l_multi.torchscript.")
    return p


def _warmup_nlf_jit(model: Any, device: str) -> None:
    """Run a 1-frame dummy pass through SMPL-24 and `bml_movi_87` heads at startup to absorb TorchScript JIT compile."""
    import torch

    with torch.inference_mode():
        dummy = torch.zeros((1, 3, 384, 384), dtype=torch.uint8, device=device)
        model.detect_smpl_batched(dummy, default_fov_degrees=55.0, num_aug=1)
        c_locs = model.crop_model.canonical_locs()
        idx_87 = model.per_skeleton_indices["bml_movi_87"].long()
        w87 = model.get_weights_for_canonical_points(c_locs[idx_87])
        dummy_box = [torch.tensor([[64.0, 32.0, 192.0, 320.0, 0.99]], dtype=torch.float32, device=device)]
        model.estimate_poses_batched(dummy, boxes=dummy_box, weights=w87, default_fov_degrees=55.0, num_aug=1)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    app.state.warm = False
    app.state.jit_warmed = False
    app.state.model_sha256 = ""
    app.state.device = os.environ.get("GPU_WORKER_DEVICE", "cuda")
    mpath = os.environ.get("NLF_MODEL_PATH", "")
    if mpath and Path(mpath).exists():
        try:
            model = _load_nlf_cached(mpath, app.state.device)
            app.state.model_sha256 = _file_sha256_prefix(mpath)
            app.state.warm = True
            if os.environ.get("GPU_WORKER_SKIP_WARMUP", "0") != "1":
                _warmup_nlf_jit(model, app.state.device)
                app.state.jit_warmed = True
        except Exception as exc:
            app.state.load_error = str(exc)
    yield


app = FastAPI(title="SportsPostureAnalysis Warm GPU Worker", version="2.0.0", lifespan=_lifespan)


@app.get("/health")
def health() -> Dict[str, Any]:
    gpu_info: Dict[str, Any] = {"cuda_available": False}
    try:
        import torch

        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            gpu_info = {
                "cuda_available": True,
                "device_name": torch.cuda.get_device_name(idx),
                "mem_allocated_mb": round(torch.cuda.memory_allocated(idx) / (1024 * 1024), 1),
                "mem_reserved_mb": round(torch.cuda.memory_reserved(idx) / (1024 * 1024), 1),
            }
    except Exception:
        pass
    return {
        "status": "warm" if getattr(app.state, "warm", False) else "cold",
        "jit_warmed": bool(getattr(app.state, "jit_warmed", False)),
        "model_sha256": getattr(app.state, "model_sha256", ""),
        "gpu": gpu_info,
    }


@app.post("/infer")
async def infer_clip(
    video: UploadFile = File(...),
    start_frame: int = Form(0),
    end_frame: Optional[int] = Form(None),
    capture_fps_override: Optional[float] = Form(None),
    bowling_arm: Optional[str] = Form(None),
    intrinsic_matrix_json: Optional[str] = Form(None),
    world_up_json: Optional[str] = Form(None),
    anchor_xy_json: Optional[str] = Form(None),
    num_aug: int = Form(1),
    extract_bml87: bool = Form(True),
) -> FileResponse:
    """Run warm NLF SMPL-24 + optional `bml_movi_87` inference and return content-addressed `raw_nlf.npz`."""
    t0 = time.perf_counter()
    suffix = Path(video.filename or "clip.mp4").suffix.lower() or ".mp4"
    if suffix not in _ALLOWED_VIDEO_SUFFIXES:
        raise HTTPException(status_code=400, detail=f"Unsupported video suffix '{suffix}'. Allowed: {sorted(_ALLOWED_VIDEO_SUFFIXES)}")

    # Stream upload to temp file with strict byte cap & incremental SHA-256 (prevents OOM DoS)
    hasher = hashlib.sha256()
    total_bytes = 0
    with tempfile.NamedTemporaryFile(dir=str(_worker_cache_dir()), suffix=suffix, delete=False) as tf:
        tmp_vid = Path(tf.name)
        while True:
            chunk = await video.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > _MAX_UPLOAD_BYTES:
                tf.close()
                tmp_vid.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail=f"Upload exceeds {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
            hasher.update(chunk)
            tf.write(chunk)

    if total_bytes == 0:
        tmp_vid.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Empty video upload.")

    try:
        mpath = _model_path()
        model_sha = _file_sha256_prefix(mpath)
        key_material = (
            f"{model_sha}|{start_frame}|{end_frame}|{capture_fps_override}|{bowling_arm}|"
            f"{intrinsic_matrix_json}|{world_up_json}|{anchor_xy_json}|{num_aug}|{extract_bml87}"
        ).encode("utf-8")
        hasher.update(key_material)
        digest = hasher.hexdigest()[:24]
        cache_npz = _worker_cache_dir() / f"nlf_{digest}.npz"

        if cache_npz.exists():
            return FileResponse(
                str(cache_npz),
                media_type="application/octet-stream",
                filename="raw_nlf.npz",
                headers={"X-Cache-Hit": "true", "X-Infer-Seconds": "0.01"},
            )

        audit = audit_video(
            tmp_vid,
            start_frame=start_frame,
            end_frame=end_frame,
            capture_fps_override=capture_fps_override,
            dedup_regular_pulldown=True,
        )
        src_indices = audit.source_frame_indices or list(range(start_frame, start_frame + audit.num_frames))

        K_arr = np.asarray(json.loads(intrinsic_matrix_json), dtype=np.float64) if intrinsic_matrix_json else None
        up_arr = np.asarray(json.loads(world_up_json), dtype=np.float64) if world_up_json else None
        auto_cal_json = ""
        if K_arr is None:
            cal_tw = auto_detect_two_wickets(tmp_vid, start_frame=start_frame, end_frame=end_frame, source_frame_indices=src_indices)
            if cal_tw is not None:
                K_arr = np.asarray(cal_tw.camera_matrix_K, dtype=np.float64)
                up_arr = world_up_from_calibration(cal_tw)
                auto_cal_json = cal_tw.model_dump_json()

        anchor_xy = tuple(json.loads(anchor_xy_json)) if anchor_xy_json else None
        arm_ov = BowlingArm(bowling_arm) if bowling_arm in ("right", "left") else None
        dev = getattr(app.state, "device", "cuda")

        tracked = extract_bowler_3d_nlf(
            video_path=tmp_vid,
            model_path=mpath,
            source_frame_indices=src_indices,
            num_aug=int(num_aug),
            intrinsic_matrix=K_arr,
            world_up_vector=up_arr,
            bowling_arm_override=arm_ov,
            anchor_xy_norm=anchor_xy,
            device=dev,
        )

        bg_up_arr = (
            np.asarray(tracked.background_upright_vectors, dtype=np.float32).reshape(-1, 3)
            if tracked.background_upright_vectors
            else np.zeros((0, 3), dtype=np.float32)
        )
        held_mask_arr = np.zeros(len(tracked.valid_mask), dtype=np.bool_)
        if audit.timebase_status == "edited_or_frozen":
            cad = analyze_frame_cadence(tmp_vid, start_frame=start_frame, end_frame=end_frame)
            hm = cad.get("held_mask")
            if hm is not None and len(hm) == len(held_mask_arr):
                held_mask_arr = np.asarray(hm, dtype=np.bool_)

        save_kwargs: Dict[str, Any] = {
            "joints3d_cam_m": tracked.joints3d_cam_m.astype(np.float32),
            "joints2d_px": tracked.joints2d_px.astype(np.float32),
            "joint_uncertainties_m": tracked.joint_uncertainties_m.astype(np.float32),
            "joint_in_frame": tracked.joint_in_frame.astype(np.bool_),
            "valid_mask": tracked.valid_mask.astype(np.bool_),
            "interp_gap_frames": tracked.interp_gap_frames.astype(np.int32),
            "bowler_boxes_xyxy": tracked.bowler_boxes_xyxy.astype(np.float32),
            "background_upright_vectors": bg_up_arr,
            "held_frame_mask": held_mask_arr,
            "all_tracks_json": np.array(json.dumps(tracked.all_tracks_snapshot)),
            "audit_json": np.array(audit.model_dump_json()),
            "detected_bowling_arm": np.array(tracked.detected_bowling_arm.value),
            "model_sha256": np.array(tracked.model_sha256),
            "nlf_fx_used": np.array(float(K_arr[0, 0]) if K_arr is not None else 0.0, dtype=np.float64),
            "auto_cal_json": np.array(auto_cal_json),
        }

        if extract_bml87:
            bml_cam, bml_unc = extract_bml_movi_87_for_boxes(
                video_path=tmp_vid,
                model_path=mpath,
                source_frame_indices=src_indices,
                bowler_boxes_xyxy=tracked.bowler_boxes_xyxy,
                intrinsic_matrix=K_arr,
                world_up_vector=up_arr,
                device=dev,
            )
            save_kwargs["bml_movi_87_cam_m"] = bml_cam.astype(np.float32)
            save_kwargs["bml_movi_87_unc_m"] = bml_unc.astype(np.float32)

        np.savez_compressed(cache_npz, **save_kwargs)
    finally:
        tmp_vid.unlink(missing_ok=True)

    elapsed = time.perf_counter() - t0
    return FileResponse(
        str(cache_npz),
        media_type="application/octet-stream",
        filename="raw_nlf.npz",
        headers={"X-Cache-Hit": "false", "X-Infer-Seconds": f"{elapsed:.2f}"},
    )
