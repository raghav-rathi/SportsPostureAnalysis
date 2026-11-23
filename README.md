# SportsPostureAnalysis

A scientifically grounded, pitch-calibrated, uncertainty-gated 3D sports biomechanics and posture analysis platform.

> **Living Project Ledger:** See **[AGENTS.md](AGENTS.md)** for the global engineering rules, scientific guardrails, and chronological **Experiment & Modification Ledger** updated after every experiment, finding, and code modification.

---

## Flagship Module (Phase 0–1.5): Cricket Fast-Bowling Biomechanics Engine (`src/bowling_biomech/`)

Built from first principles on **Neural Localizer Fields (`isarandi/nlf`, NeurIPS 2024)**, **MCC Laws 6/7/8 Pitch & Two-Wicket Geometry Calibration**, primary-source fast-bowling biomechanics (`Worthington et al. 2013`, `Portus et al. 2001/2004`, `Bayne et al. 2016`, `Felton et al. 2019`, `Alway et al. 2021`, `Glazier 2010`, `ECB 2024` / `Cricket Australia 2026`), and strictly measurement-grounded coaching reports.

### Pipeline Architecture (7 Stages, Calibration-Before-Pose)

1. **Stage 0 (`stage0_audit.py`):** Timebase integrity audit (`analyze_frame_cadence`) with automatic regular pulldown deduplication (e.g., 25-in-29.97 fps), freeze-frame / retimed slow-mo / scene-cut detection, bowler-ROI sharpness grading (`measure_bowler_roi_blur`), and short-baseline 4-DOF KLT+RANSAC camera motion detection (pan, tilt, and zoom).
2. **Stage 1 (`stage1_pitch_calibration.py`):** Pre-NLF **MCC Laws 6/7/8 PnP** (`1.22 m` crease gap, `2.64 m` return crease width, `0.7112 m` stump height, `0.192 m` outer off-to-leg stump center spacing, coplanar `SOLVEPNP_IPPE`) and zero-annotation multi-frame **Two-Wicket Telephoto Calibration** (`auto_detect_two_wickets`) recovering camera intrinsics `K`, extrinsics, and gravity `world_up_vector` *before* 3D pose inference. Uncalibrated fallback uses standing-background or sagittal-de-leaned bowler-torso gravity with explicit `heading_sigma_deg` and `gravity_sigma_deg`.
3. **Stage 2 (`stage2_nlf_3d.py`, `stage2_event_spotting.py`):** Streaming multi-person metric 3D SMPL-24 pose & native uncertainties via `nlf_l_multi.torchscript` (`CUDA`, pinned SHA-256) with calibrated `intrinsic_matrix` and `world_up_vector` passed directly into `detect_smpl_batched`, GPU repeatability ensemble (`base`, `flip`, `aug3`), `(T, 24)` `joint_in_frame` crop bounds checking, true Winter (2009) dual-pass Butterworth smoothing (`C = 1.2465`), and strict release-sanity-gated event spotting (`BFC < FFC <= UAH < BR < FT`, Felton 2019 priors `192/108 ms`, rejecting truncated/frozen archival clips).
4. **Stage 3 (`stage3_kinematics.py`):** Canonical Pitch World Frame (`+X` down pitch, `+Y` gravity-up, `+Z` off-to-leg), exact Z–X Cardan global trunk inclination, Rician/Jensen near-180° knee debiasing, Savitzky-Golay `~50 ms` arm angular velocity, local front-foot `[FFC, BR]` ground reference, `[60°, 420°)` joint-anchored Portus (2001/2004) AIS transverse alignment & SCR.
5. **Stage 4 (`stage4_uncertainty.py`):** Unified 4-source analytical Jacobian uncertainty engine (orthogonal `sigma_epistemic` + `sigma_projective`, contact-kink `sigma_temporal`, finite-difference `sigma_calibration`, `sigma_scale`) plus empirical GPU ensemble repeatability (`repeatability_interval` vs `analytical_uncertainty` — never mislabeling uncalibrated real-video intervals as 95% CIs). Hard gating for incomplete deliveries, unreliable timebases, cropped joints, camera-panning/uncalibrated translation metrics (`runup_speed_mps`, `stride_length_pct_stature`, `follow_through_drift_deg`), `<90 fps` high-velocity arm metrics, and **permanent `REFUSED` status on ICC 15-degree elbow extension legality**.
6. **Stage 5 (`stage5_evaluator.py`):** Retires unvalidated 0–100 composite scores (`score=None`) in favor of individual verified metrics, definition-matched primary-source **Cohort Comparisons**, **ECB (2024) / Cricket Australia (2026) Workload Guidance**, strict non-straddling printed-interval coaching cues (routing straddling metrics to `inconclusive_observations`), tailored capture upgrades, and Dual-View inverse-variance fusion.
7. **Stage 6 (`stage6_visualizer.py`, `stage6_coach_llm.py`):** Zero-drift sequential `source_frame_indices` 2×2 keyframe strips with `[lo, hi]` interval overlays, 6-panel dashboards (ground-zeroed / front-ankle-anchored virtual 3D views + Cohort Z-Score Forest Plot), browser-ready H.264 (`yuv420p + faststart`) freeze videos, and strictly JSON-grounded Markdown reports.

---

## Verified Baseline Results (`v4`, Post-EXP-007 Hardening)

- **Automated Unit & Real-Clip Regression Suite (`pytest -v`):** **18 / 18 passed** (`2.57 s`, `tests/test_core_biomechanics.py`), including 4 cached real-video NLF regression fixtures (`tests/fixtures/*.npz`).
- **Per-Metric Non-Circular Synthetic Metrology Benchmark (`docs/results/v4/synthetic_benchmark_results.json`):**
  - **All 22 / 22 metrics scored** (3,003 observable AR(1) correlated-noise evaluations, randomized release angle & Felton 2019 timings, non-teleporting foot landings, non-looping follow-through): **96.4%** pooled analytical coverage, per-metric range **91.1%–100.0%**, sharpness `rmse/sigma` in **`[0.71, 1.21]`**, **`0`** false incomplete trials out of 240.
  - **Event Timing MAE:** `25–60 fps` BR `3.1–3.5 ms`; `120–240 fps` BR `2.7–2.9 ms`, FFC `8.3 ms`, BFC `5.5–8.6 ms`.
  - **Noisy Realistic Self-Calibration (120 fps, 12° sprint lean, ±10° angled approach):** Signed azimuth MAE `0.9°–1.2°`, local release-height bias `+0.002–0.010 m`, `LatFlex` coverage `100.0%`.
  - **Unmodified `EXP-007` S1–S9 Diagnostic Suite (`docs/results/v4/synthetic_diagnostics.json`):** `S1` all-22 range **91.7%–100.0%** (pooled 97.1%), `S6` release-height coverage **100.0%** across all 15 lean/approach combos (up from `0–6%`), `S8a` wrap straddle **`+6.0°`** (was `-354.0°`), `S8b` **4 / 4** pan/tilt/zoom detected (`3.743` vs `3.74` true), `S9` cue precision/recall **100.0%** (`34 / 34`, zero false alarms).
- **Real-Video RTX 4090 GPU Verification (`docs/results/v4/real_clip_diagnostics.json`):**
  - **Dale Steyn Net (`25.2 unique fps` after auto 1-in-6 pulldown dedup, Rear):** Zero-annotation two-wicket calibration recovered `f = 10,208 px` (camera `24.7 m` behind wicket) and passed `K` + `world_up` to NLF (pelvis depth `24.03 m`, depth travel `10.85 m` vs prior `4.38 m` / `1.92 m`). Verified `BFC = src 63`, `FFC = src 70`, `BR = src 71`, `thorax_lat_flex_br_deg = 30.26° [26.8, 36.5]` (`high`, `repeatability_interval`), `SCR = 46.17° [36.9, 55.0]` (`moderate`), panning translation/drift metrics suppressed, `ICC = REFUSED`.
  - **Indoor Slow-Mo (`120 nominal fps [UNVERIFIED TIMEBASE]`):** Flagged `edited_or_frozen` (`30%` repeats), gated cropped knee, velocities, and drift; 0 false or contradictory cues.
  - **Glenn McGrath & Curtly Ambrose Archival Reels (`25 fps`, edited/frozen):** Both cleanly rejected by release sanity gates (`is_complete_delivery = False`), suppressing **100% (`0 / 23`)** of delivery metrics and issuing **0** false cues or scores.

| Dale Steyn Net (`25.2 unique fps`, Two-Wicket Calibrated) | Dale Steyn `v4` Biomechanics Dashboard & Cohort Z-Scores |
|---|---|
| ![Dale Steyn v4 Keyframes](docs/figures/steyn_v4_keyframes.jpg) | ![Dale Steyn v4 Dashboard](docs/figures/steyn_v4_dashboard.png) |

*(Note: Real-video outputs measure NLF repeatability and internal geometric consistency; absolute real-world 3D accuracy awaits future ground-truth validation against IMU/radar/optical mocap.)*

---

## Quickstart

```bash
# 1. Create isolated Python 3.10+ environment and install package + dev deps
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# 2. Run unit & cached real-clip regression suite (CPU, ~2.5 s)
pytest -v
python validation/run_synthetic_benchmark.py --out outputs/synthetic_benchmark_results.json

# 3. Point to downloaded NLF TorchScript weights (CC-BY-NC-4.0 non-commercial research weights)
export NLF_MODEL_PATH="/path/to/models/nlf/nlf_l_multi.torchscript"

# 4. Run full 7-stage analysis on a fast-bowling clip (with auto two-wicket calibration & repeatability)
bowling-biomech analyze path/to/delivery.mp4 \
  --bowler-id "Bowler_01" \
  --output-dir outputs/bowler_01 \
  --capture-fps 120
```

## Documentation & Evidence
- **[AGENTS.md](AGENTS.md)** — Global agent instructions, guardrails, and living Experiment & Modification Ledger (`EXP-001`–`EXP-008`)
- **[docs/research_report.md](docs/research_report.md)** — First-principles scientific investigation, forensic Reel analysis, and corrected primary-source bibliography
- **[docs/walkthrough.md](docs/walkthrough.md)** — Phase 0–1.5 build, audit hardening, and RTX 4090 validation walkthrough
- **[docs/results/v4/](docs/results/v4/)** — Current `v4` per-metric benchmark, S1–S9 diagnostics, and RTX 4090 real-video Fact Sheets & Markdown report
- **[docs/independent_audit_and_improvements.md](docs/independent_audit_and_improvements.md)** — Independent audit (`EXP-007`) documenting all 27 bugs (`B1–B27`) resolved in Phase 1.5
- **[docs/results/exp007/](docs/results/exp007/)** — Historical `EXP-007` baseline audit outputs produced by `validation/audit_*.py`
