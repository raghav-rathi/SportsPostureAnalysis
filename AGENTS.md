# AGENTS.md — Global Agent Instructions & Experiment Ledger (`SportsPostureAnalysis`)

> **Scope:** Repository-wide global engineering instructions, scientific guardrails, and living Experiment & Modification Ledger for **SportsPostureAnalysis** (`https://github.com/raghav-rathi/SportsPostureAnalysis`). All AI agents and contributors working anywhere in this repository must follow these rules and update this ledger after every experiment, finding, or modification.

---

## 1. Mandatory Standing Rules for All Agents

### 1.1 Living Experiment & Git Progress Protocol
1. **Update `AGENTS.md` after every cycle:** After every experiment run, model evaluation, bug discovery, architectural change, or parameter modification, append a new chronological entry (`EXP-XXX`) in **Section 4** documenting:
   - Hypothesis / goal and exact files modified
   - Exact quantitative results (metrics, test counts, RMSE, CI coverage, real-video outcomes)
   - Key findings, failure modes uncovered, and next actions
   - Update **Section 3 (Current Verified Baseline)** whenever baseline numbers change.
2. **Continuous Git Progress:** Commit logical milestones cleanly with descriptive messages and push progress to `origin main` (`https://github.com/raghav-rathi/SportsPostureAnalysis`) once verification passes.
3. **Single Global Ledger:** Keep **this root `AGENTS.md`** as the single repository-wide source of truth—do not fragment experiment history across scattered subfolder `AGENTS.md` files.

### 1.2 Non-Negotiable Scientific & Biomechanical Guardrails
1. **Permanent ICC 15° Elbow Legality Refusal:** `icc_elbow_legality_verdict` MUST remain permanently `ObservabilityGrade.REFUSED` on monocular or casual multi-view video. Accredited $\ge 250\text{ Hz}$ optical 3D laboratory testing is required; this caveat must never be truncated in reports or visuals.
2. **Medical Non-Diagnosis Disclaimer:** Mechanical load & durability screening estimates single-delivery mechanical stress only—never present it as a clinical injury diagnosis. Preserve workload, skeletal maturity, and physical capacity caveats.
3. **Zero Uncalibrated Hallucinations (Neuro-Symbolic Contract):** Every reported metric must carry `(value, ci_low, ci_high, observability)` or explicit `suppressed=True` with a clear `suppression_reason`. Stage 6 coaching narratives may only cite non-suppressed keys from `BiomechanicalFactSheet`.
4. **Physical Bounds & Non-Circular Verification:** Clamp confidence intervals to physical limits (`[0°, 180°]` for internal joint angles, `>= 0` for non-negative magnitudes). Never validate sign conventions solely against a simulator written with the same formula—maintain non-circular analytical vector tests in `tests/test_core_biomechanics.py` covering both right- and left-arm bowlers.

### 1.3 Compute, Storage, Licensing & Security Hygiene
1. **Zero Secrets in Git:** Never commit passwords, SSH keys, `.env`, internal LAN IPs, or personal machine paths (`/Users/...`) into this public repository. Configure remote GPU hosts strictly through environment variables or a gitignored local `.env` (`scripts/sync_to_server.sh`).
2. **No Heavy Binaries in Git:** Keep raw videos (`*.mp4`, `*.mov`), neural weights (`*.torchscript`, `*.pt`, `*.pth`, `*.ckpt`), virtual environments (`.venv/`), caches (`cache/`), and raw run directories (`outputs/`) excluded via `.gitignore`. Track only lightweight documentation figures (`docs/figures/`) and structured JSON summaries (`docs/results/`).
3. **Remote GPU Disk Isolation:** On remote compute servers, pin virtual environments, `HF_HOME`, `TORCH_HOME`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `TMPDIR`, datasets, models, and outputs strictly inside the dedicated experiment disk (`$REMOTE_DIR`) via `env.sh`—never write experiment data to root `/` or `$HOME`.
4. **NLF Runtime & Licensing:** When loading `isarandi/nlf` (`nlf_l_multi.torchscript`), always `import torchvision` *before* `torch.jit.load` to register `torchvision::nms`. Note that NLF pretrained weights are non-commercial (`CC-BY-NC-4.0`)—acceptable for this open research project; swap weights before any commercial deployment.

### 1.4 Pre-Commit Verification Gate
- Run `pytest -v` (`tests/test_core_biomechanics.py`) — 100% pass required.
- When changing calibration, event spotting, kinematics, or uncertainty propagation, re-run `validation/run_synthetic_benchmark.py` and verify 95% CI empirical coverage and uncalibrated self-calibration RMSE.
- When changing Stage 6 overlays or plots, visually inspect generated keyframes/dashboards before committing.

---

## 2. Repository Architecture (`SportsPostureAnalysis`)

Current flagship module: **Phase 0–2 Cricket Fast-Bowling Biomechanics Engine & Interactive Coaching Studio** (`src/bowling_biomech/` + `web/`, fast & fast-medium pace bowlers, left & right arm; spin bowling deferred to future discipline expansion).

| Component | Path | Responsibility |
|---|---|---|
| Shared Constants | `src/bowling_biomech/constants.py` | Canonical `SMPL24_JOINTS`, `JOINT_IDX`, `LR_JOINT_PAIRS`, MCC Laws 6/7/8 geometry (`0.0960 m` stump half-spacing), Felton (2019) timing priors (`192/108 ms`) |
| Data Contracts | `src/bowling_biomech/schemas.py` | Strict Pydantic v2 contracts (`VideoAudit` with timebase/ROI fields, `PitchCalibration` with intrinsics/sigmas, `DeliveryEvents` with `source_frame_idx`, `MetricEstimate` with `interval_type`, `CohortComparison`, `WorkloadGuidance`, retired composite pillars) |
| Stage 0 | `src/bowling_biomech/stage0_audit.py`, `stage0_net_clipper.py` | Timebase integrity (`analyze_frame_cadence` + regular pulldown deduplication + freeze/cut detection), bowler-ROI blur (`measure_bowler_roi_blur`), 4-DOF KLT+RANSAC pan/tilt/zoom motion detector, and ~6 fps multi-ball net-session auto-clipper |
| Stage 1 | `src/bowling_biomech/stage1_pitch_calibration.py`, `stage1_pan_tracker.py` | Pre-NLF MCC Laws 6/7/8 PnP (`SOLVEPNP_IPPE`, larger-side `K`), zero-annotation Law 8 `auto_detect_two_wickets`, 1-frame MCC PnP + bowler-masked KLT/RANSAC 3-DOF SO(3) camera rotation pan tracker (`median_residual_px <= 2.5 px`, fixed `C_0 = -R_t^T t_t`), standing-background / de-leaned gravity self-cal |
| Stage 2A & Anatomy | `src/bowling_biomech/stage2_nlf_3d.py`, `stage2_isb_anatomy.py`, `stage2_dual_sync_fusion.py`, `stage2_ball_release.py` | Streaming multi-person 3D SMPL-24 + `bml_movi_87` 87-marker ISB rigid-body Pelvis/Thorax/Foot anatomy (3D invariant `R_pelvis^T @ R_thorax`, correlated-depth `N_eff` guardrail), `+Y_world` 1000 Hz spline FFT NCC dual-phone sync + anisotropic ray-precision 3D fusion (PSD matrix-sqrt cross-covariance + collinearity gate), and calibrated side-on asymmetric wrist-ROI ballistic release speed estimator |
| Stage 2B | `src/bowling_biomech/stage2_event_spotting.py` | True Winter (2009) dual-pass Butterworth (`C=1.2465`), strict overhead/slingy/25fps-bracketed BR sanity gates (`is_complete_delivery`), Felton (2019) priors, continuous-landing FFC/BFC touchdown refinement, `_boundary_clamped` confidence drops |
| Stage 3 | `src/bowling_biomech/stage3_kinematics.py` | Canonical Pitch World Frame, Z–X Cardan global trunk inclination, Rician/Jensen near-180° knee debiasing, Savitzky-Golay `~50 ms` arm omega, local front-foot ground reference, `[60°, 420°)` joint-anchored Portus (2001/2004) AIS alignment/SCR |
| Stage 4 | `src/bowling_biomech/stage4_uncertainty.py` | 4-source analytical Jacobian uncertainty (orthogonal `sigma_epistemic` + `sigma_projective`, contact-kink `sigma_temporal`, FD `sigma_calibration`, `sigma_scale`), GPU ensemble `repeatability_interval` with systematic floor, incomplete/timebase/translation/crop/`<90 fps` gating, permanent ICC `REFUSED` |
| Stage 5 | `src/bowling_biomech/stage5_evaluator.py` | Retired 0–100 composites (`score=None`), definition-matched primary-source Cohort Comparisons, ECB (2024) / Cricket Australia (2026) Workload Guidance, strict non-straddling printed-interval cues (`inconclusive_observations`), tailored upgrades, Pillar 3 repeatability |
| Stage 6 | `src/bowling_biomech/stage6_visualizer.py`, `stage6_coach_llm.py`, `stage6_coach_copilot.py` | Sequential `source_frame_indices` 2×2 keyframe strips, 6-panel dashboards, H.264 freeze video & 1-click `1080×1920` 9:16 Truthful Breakdown Reel, strictly JSON-grounded Markdown reports, and deterministic numeric-verifier Coach Copilot (`verify_coach_response`) |
| Coaching Studio & GPU Worker | `src/bowling_biomech/server/{app,db,recompute,video_proxy}.py`, `src/bowling_biomech/gpu_worker/worker.py`, `web/src/` | Local-first FastAPI Studio + WAL SQLite longitudinal store, All-Intra scrub proxy, `<35 ms` CPU HITL recompute, Warm Remote GPU Worker (startup JIT warmup + SHA-256 disk cache + `auto_cal_json` preservation), and Vite/React/Three.js/uPlot Studio UI |
| Orchestration | `src/bowling_biomech/pipeline.py`, `cli.py` | Calibration-before-NLF single-view & dual-view pipelines and Typer CLI (`bowling-biomech analyze | serve-studio | serve-gpu-worker`) |
| Metrology & Tests | `src/bowling_biomech/synthetic_ground_truth.py`, `validation/`, `tests/` | Continuous non-teleporting Z–X Cardan synthetic generator, all-22 per-metric non-circular AR(1) benchmark + noise-mismatch & noisy 12° lean self-cal arms, 27-test pytest suite (`test_core_biomechanics.py` + `test_phase2_studio.py`) + 4 cached real-clip fixtures (`tests/fixtures/*.npz`) |
| Research & Reports | `docs/research_report.md`, `docs/walkthrough.md`, `docs/figures/`, `docs/results/` | Corrected primary-source scientific report, forensic Reel deconstruction, Phase 0–2 walkthrough, verified `v4` & `v5` visuals & JSON outputs |
| Audit Diagnostics (EXP-007) | `validation/audit_synthetic_diagnostics.py`, `validation/audit_real_clip_diagnostics.py`, `validation/audit_nlf_self_consistency.py`, `docs/independent_audit_and_improvements.md` | Per-metric benchmark scoring and stress cases (`S1–S9`), repeated-frame and release checks on finished runs, pose-model consistency checks, and the `EXP-007` audit report |

---

## 3. Current Verified Baseline Snapshot (`v5`, Phase II Coaching Studio + Multi-View + ISB Anatomy)

- **Unit, Real-Clip & Phase II Studio Regression Suite (`pytest -v`):** **27 / 27 PASSED** (`24.71 s` local macOS, `31.35 s` Linux RTX 4090 — 18 core/real-clip fixture tests + 9 Phase II Studio/CV/verifier/non-circular audit regression tests).
- **Frontend Production Build (`cd web && npm run build`):** **0 TypeScript errors**, clean Vite production bundle (`3.89 s`), live browser-verified with 0 console errors (`docs/figures/studio_v2_preview.png`).
- **Phase 1.5 Metrology Zero-Regression (`v4` -> `v5` identical to last decimal):**
  - **All 22 / 22 metrics scored** (3,003 observable AR(1) evaluations): **96.4%** pooled analytical coverage, per-metric **91.1%–100.0%**, `rmse/sigma` in **`[0.71, 1.21]`**, **`0 / 240`** false incomplete trials.
  - **Non-Oracle Noise Mismatch Stress (`45° 120 fps`):** `0.5x = 100.0%`, `1.0x = 96.5%`, `2.0x = 78.2%`.
  - **Event Timing MAE:** `25 fps`: BFC `11.2 ms`, FFC `14.4 ms`, BR `3.5 ms`; `30 fps`: BFC `14.2 ms`, FFC `11.5 ms`, BR `3.2 ms`; `60 fps`: BFC `11.7 ms`, FFC `8.7 ms`, BR `3.1 ms`; `120 fps`: BFC `8.6 ms`, FFC `8.3 ms`, BR `2.9 ms`; `240 fps`: BFC `5.5 ms`, FFC `8.3 ms`, BR `2.7 ms`.
  - **Unmodified `EXP-007` S1–S9:** `S1` 91.7%–100.0% (pooled 97.1%), `S2` 0% > 40 ms, `S6` 100.0% release-height coverage, `S7` 0 failures down to `6°` FOV, `S8a` `+6.0°`, `S8b` 4/4 detected, `S9` 100.0% (34/34) cue precision & recall.
- **Phase II RTX 4090 GPU 1 Verification (`docs/results/v5/phase2_gpu_verification.json`, Dale Steyn real clip):**
  - **Warm GPU Worker (`nlf_l_multi.torchscript`, SHA-256 `52bee28edb6ea914`):** Lifespan JIT warmup `26.62 s` (`jit_warmed=true`), steady-state warm SMPL-24 + 87-marker `bml_movi_87` inference **`6.83 s`** (63 frames, 6 tracks), SHA-256 disk cache hit **`7.84 ms`**, Law-8 `two_wicket_geometry` (`fx = 10,207.6 px`, cam `24.7 m` back) preserved via `auto_cal_json`.
  - **3-DOF Rotational Pan Tracking (`stage1_pan_tracker.py`):** **`is_verified = true`**, `median_residual_px = 0.188 px` (`p90 = 0.263 px`), `min_inlier_count = 227` (`95.6%` inliers), `max_pan_deg = 0.91°`, `sigma_pan_deg = 0.008°`.
  - **ISB `bml_movi_87` Anatomy (`stage2_isb_anatomy.py`, `rho = 0.85`):** `N_eff_pelvis = 1.127`, `N_eff_thorax = 1.143`, `N_eff_foot = 1.081`. Under Guardrail #3 on telephoto rear-on Steyn (`mean_marker_unc = 0.0748 m`), sagittal ISB & noisy foot angles are transparently **suppressed** (`unmeasurable`), while transverse ISB Pelvis (`258.1° [224.2, 292.0]`), Thorax (`271.5° [240.2, 302.9]`), and SCR (`41.1° [5.3, 76.9]`) are honestly graded `low`.
  - **Dual-Phone 3D Fusion, Optical Ball Speed & Numeric Verifier:** `<2.5 ms` mixed 120+240 fps sync error, `<4.5°` simultaneous sagittal+coronal accuracy, collinear dual-camera gate verified, `140 km/h` 1080p 120 fps pixel-level video ball speed verified with forearm masking, **9 / 9** adversarial Coach Copilot bypasses blocked (100%) and **30 / 30** grounded prompt responses passed (0 false blocks).
- **Verified `v4` Real-Video Clips (`docs/results/v4/`):** Steyn `thorax_lat_flex_br_deg = 30.26° [26.8, 36.5]` (`high`, `repeatability_interval`), `SCR = 46.17° [36.9, 55.0]`; Indoor flagged `[UNVERIFIED TIMEBASE]` with 0 false cues; McGrath & Ambrose rejected (`is_complete_delivery = False`, 0/23 metrics, 0 cues/scores).

---

## 4. Chronological Experiment, Audit & Modification Ledger

### EXP-001 — Reference Reel Forensic Deconstruction & First-Principles Literature Synthesis
- **Date:** 2026-11-23
- **Goal:** Reverse-engineer reference Instagram Reel (`@athleteworld.in` / Satyam Patkar) and synthesize peer-reviewed cricket fast-bowling biomechanics + 2024–2026 3D CV pose literature.
- **Key Findings:**
  - Reference creator does **not** run AI biomechanical measurement: zero numeric angles, 25 fps archival clips, manual 2D overlays, and generative-AI upscaled infographics introducing anatomical distortions (wrong arm inset on Wasim, 2D parallel bars misrepresenting 3D transverse separation on McGrath/Donald, unrealistic `0°` vertical spine on Ambrose).
  - Grounded evaluation pillars in peer-reviewed norms: `Worthington et al. (2013)` ($R^2 = 0.74$ pace predictors), `Alway et al. (2021)` (88% in-sample LBSI model vs 78% base rate), `Bayne et al. (2016)` ($50^\circ \pm 6^\circ$ injured vs. $40^\circ \pm 8^\circ$ uninjured adolescent thorax lateral flexion), `Portus et al. (2004)` ($\ge 30^\circ$ SCR mixed action), `Abraham, Feros, & Fox (2025)` (three-iPad OpenCap cricket RMSE benchmarks, *IJSSC*), and `Dobos et al. (2022/2025)` (`pitchAI`).
- **Artifacts:** `docs/research_report.md`, `docs/figures/forensic_*.jpg`.

### EXP-002 — Phase 0–1 Engine Build, RTX 4090 Provisioning & Initial 240-Delivery Synthetic Benchmark (`v1`)
- **Date:** 2026-11-23
- **Goal:** Build end-to-end 7-stage pipeline on isolated RTX 4090 environment using official NLF (`isarandi/nlf` NeurIPS 2024, `nlf_l_multi.torchscript`) and validate metrology synthetically.
- **Modifications:** Implemented Stages 0–6, Pydantic schemas, CLI, synthetic ground-truth generator, `validation/run_synthetic_benchmark.py`, and initial 8-test pytest suite. Resolved `torchvision::nms` JIT op registration by importing `torchvision` prior to `torch.jit.load`.
- **Results:**
  - `pytest`: **8 / 8 passed** (`0.77 s`).
  - Synthetic benchmark (240 deliveries / 1,060 evals, i.i.d. Gaussian noise): **94.4%** empirical 95% CI coverage; 240 fps Front-Knee RMSE `1.92°` (side), Lat-Flex RMSE `0.99°` (rear), BR MAE `0.3–1.3 ms`.

### EXP-003 — Initial Real-Video RTX 4090 Smoke Runs & First Real-World Fixes (`v1 -> v2`)
- **Date:** 2026-11-23
- **Goal:** Stress-test pipeline on diverse real footage (Dale Steyn net 30 fps, Indoor slow-mo 120 fps, McGrath 25 fps side Reel, Ambrose 25 fps rear Reel).
- **Failures Uncovered & Fixes:**
  1. Broadcast panning canceled pelvis translation on McGrath, misclassifying side-on as rear (`-15°`) -> added 3D anatomical stride-axis fallback (`front_ankle - rear_ankle`), recovering `side_on_left (-99.8°)`.
  2. Flexed-elbow gather near head falsely triggered Ball Release on Ambrose -> added arm-extension & overhead gating.
  3. Slow-motion container playback (`30 fps` container storing `120 fps` capture) required explicit `--capture-fps` override.

### EXP-004 — Independent Gemini 4 Carbon (High) End-to-End Audit & Architectural Hardening (`v2 -> v3`)
- **Date:** 2026-11-23
- **Goal:** Commission a fresh-perspective deep audit of all 14 modules, synthetic benchmarks, GPU outputs, and research report; debate mathematical trade-offs; fix every confirmed P0, P1, and P2 bug.
- **Critical P0/P1/P2 Fixes Implemented:**
  - **P0-1 (Transverse Sign & Wrapping):** Fixed inverted sign in `_transverse_alignment_deg` (`arctan2(sign_arm * lr[:, 0], lr[:, 2])`) + BFC-anchored `np.unwrap`; fixed matching inverse bug in `synthetic_ground_truth.py`; added non-circular vector unit tests for BOTH right and left arms.
  - **P0-2 (Gravity, Panning & Metric Gating):** Added standing-upright background gravity filter (`knee > 155°` & `hip > 155°`, rejecting squatting keepers/batters) + underground tilt clamp (`el >= -1°`); masked Farneback pan detector + geometric pan override when pelvis depth drift is orthogonal to in-frame stride; set `is_metric_calibrated=False` on self-cal and hard-suppressed panning/uncalibrated `runup_speed_mps` and `stride_length_pct_stature`.
  - **P0-3 (Frame-Bounds Crop Gating & Uncertainty):** Added `(T, 24)` `joint_in_frame` bounds checking, event-specific per-metric visibility/uncertainty gating, and principled 3-joint error propagation (removed arbitrary `* 0.45` discount) + physical CI clamping (`[0°, 180°]`).
  - **P0-4 (Event Causality):** Enforced strict `bfc < ffc <= uah < br < ft`, `wrist_rel_y > 0.10` + `is_complete_delivery`, slow-mo duration consistency, and ankle-visibility warnings.
  - **P1 (Kinematics, Evaluator, PnP & Correlated Benchmark):** Exact Z–X Cardan `thorax_lat_flex_deg`; fixed `_estimate_stature_m` (`pelv->mid_hip` + sole/crown -> Steyn `1.80 m` vs. actual `1.79 m`); two-sided durability knots + interior-extremum CI scan + observability-weighted pillars + gated `action_class -> UNKNOWN`; coplanar `SOLVEPNP_IPPE` with cheirality & above-ground checks; dual-view arm-mismatch guard; upgraded benchmark to temporally correlated AR(1) depth drift (`tau = 80 ms`, right + left arms) + uncalibrated self-cal arm; tightened `velocity_high` gating to `fps < 90`.
  - **P2 (Visuals, Reports & Licensing):** Resize-before-draw readable portrait 9:16 keyframe strip, dimmed suppressed waveforms, staggered event labels, `UAH=None` placeholder, non-truncating `REFUSED` ICC caveat, explicit NLF `CC-BY-NC-4.0` weight licensing note.
- **Results:**
  - `pytest`: **12 / 12 passed** (`1.22 s`).
  - AR(1) Benchmark: **97.0%** 95% CI empirical coverage across 912 observable evaluations; 60 fps `DelayArm` properly suppressed.
  - Real RTX 4090 `v3` runs: Eliminated Steyn's `-14.4°` underground tilt (`11` umpire gravity frames, `stature = 1.80 m`, true SCR `212° -> 270° -> 325°`), eliminated Indoor's `94.9°` cropped-knee hallucination & `372°` yaw wrap spike, fixed McGrath side-on pan (`-98.9°`), and dropped Ambrose's false `84.8` durability pass to `54.2`.

### EXP-005 — Auditor Round-2 Verification Catch & Final Sign-Off (`v3` Final)
- **Date:** 2026-11-23
- **Goal:** Resolve 3 surgical edge cases flagged during the auditor's post-fix verification of `v3` JSONs and visuals, plus tighten bowling-arm auto-detection.
- **Modifications:**
  1. `stage0_audit.py`: Replaced `std_img > 1.8` (which dropped static backgrounds on truly static cameras!) with per-frame Sobel texture mask (`grad_mag > 8.0`), bright static caption exclusion, adaptive frozen-overlay masking *only* when `scene_med_std > 6.0`, and dominant-direction `coherent_steps`. Added synthetic static-with-moving-TV vs. panning-with-letterbox/watermark regression test.
  2. `stage5_evaluator.py`: Gated coronal/sagittal camera-angle upgrade prompts on actual camera azimuth (`abs(cos(az)) < 0.65`, `abs(sin(az)) < 0.65`) so rear-on blurry clips no longer prompt for a rear camera.
  3. `stage6_visualizer.py`: Fixed swapped BGR amber warning tuples `(255, 196, 100) -> (100, 196, 255)` and `(255, 190, 90) -> (90, 190, 255)`.
  4. `synthetic_ground_truth.py`: Ramped pre-BFC lateral roll via `_smoothstep(t, bfc - 0.12, bfc)` so synthetic run-up stays upright.
  5. `stage2_nlf_3d.py`: Replaced 97th-percentile arm inference with top-2 straight-arm overhead reach + shoulder-relative wrist speed + vertical circumduction range.
- **Empirical Findings & Results:**
  - Raw Farneback optical flow across all textured pixels proved Steyn (`scene_med_std = 25.85`, `+25.6 px` monotonic rightward drift = `8.0%` width) and Indoor (`scene_med_std = 42.28`, `+53.5 px = 16.7%` width) were genuinely handheld panning clips, while Ambrose (`scene_med_std = 1.54`, `bg_pan = 0.0001`) was genuinely static.
  - `uncalibrated_self_cal_arm` `rear_0` `LatFlex RMSE` dropped from `7.56°` -> **`2.99°`**.
  - Ambrose upgrades cleaned to `['Switch Smartphone to 120–240 fps Slow-Motion', 'Add a Side-On (90 deg) Camera Angle']`.
  - `pytest -v`: **13 / 13 PASSED** (`1.26 s`).
  - Auditor Verdict: **FINAL SIGN-OFF APPROVED** (zero remaining P0, P1, or P2 issues).
- **Artifacts:** `docs/figures/*_v3_*`, `docs/results/v3/*`, `docs/walkthrough.md`.

### EXP-006 — Public Repository Hygiene, GitHub Initialization (`SportsPostureAnalysis`) & Global `AGENTS.md`
- **Date:** 2026-11-23
- **Goal:** Establish `https://github.com/raghav-rathi/SportsPostureAnalysis` as the canonical public repository with global `AGENTS.md`, sanitized scripts/docs, and tracked visual/JSON evidence.
- **Modifications:**
  - Removed hardcoded remote host/server paths from `scripts/sync_to_server.sh` and `src/bowling_biomech/cli.py` (parameterized via `.env` / `NLF_MODEL_PATH`).
  - Converted local absolute image paths in `docs/research_report.md` to repo-relative `figures/*.jpg` and staged verified forensic & `v3` figures (`docs/figures/`) + JSON summaries (`docs/results/v3/`).
  - Created global `AGENTS.md`, `docs/walkthrough.md`, and upgraded `README.md`.
  - Verified zero leaked credentials, IPs, or personal machine paths across the entire repository and git history prior to pushing to `origin main`.

### EXP-007 — Independent first-principles audit, real-clip forensics and diagnostic scripts
- **Date:** 2026-11-23
- **Goal / Hypothesis:** Audit the repository from scratch. Test whether the v3 baseline claims (97.0% CI coverage, four verified real-video runs, literature-grounded scoring) survive checks that do not share the pipeline's assumptions.
- **Modifications:** No change under `src/`. Added `docs/independent_audit_and_improvements.md`, three diagnostic scripts (`validation/audit_synthetic_diagnostics.py`, `validation/audit_real_clip_diagnostics.py`, `validation/audit_nlf_self_consistency.py`), result JSONs in `docs/results/exp007/`, and five figures `docs/figures/exp007_*`. README links the audit. Section 3 gained an audit-status block.
- **Results:**
  - `pytest -v`: **13 / 13 passed**. Stock benchmark reproduced exactly (97.0%, 912 evaluations).
  - Timebase: all four v3 clips contain repeated frames. Steyn every 6th (25 fps footage in a 29.97 fps container), Indoor every 5th, McGrath 64% of frame pairs, Ambrose 82%.
  - Events: McGrath BR at 56.5 is a freeze frame, arm 46° behind vertical, wrist at 68% of arm length. Ambrose BR at 36.9 is in the back-foot phase, about 40 frames before release. Steyn BFC, FFC and BR agree with the frames to about one frame.
  - Camera: two-wicket geometry on Steyn gives focal length 11,400 px (9.6° FOV) against the assumed 1,844 px (55°), camera 27.6 m behind the wicket, 1.52 m high. Run-up depth speed is 0.79 m/s as processed and 4.9 m/s after focal scaling. Body-based gravity is 4.7° from the wicket-derived vertical.
  - NLF consistency on GPU (checkpoint v0.3.2, SHA-256 prefix `52bee28e`): Steyn `thorax_lat_flex_br` moves at most 2.1° across mirror, TTA, half resolution, v0.2.2 and v0.3.2 patch 6, against a stated ±10.6°. `follow_through_drift_deg` moves 24.8°, and 53.9° under an FOV change, against a stated ±8.5°. NLF's uncertainty output is 1.2 to 8 times the measured spread by RMS and has rank correlation 0.5 to 0.7 with it.
  - Synthetic diagnostics: per-metric coverage 82.5% to 100% behind the pooled 97.0%. 15 of 22 metrics were never scored by the stock benchmark. BFC and FFC timing error is 9.8 ms and 5.5 ms at 240 fps against a stated 2.1 ms. Self-calibration gravity error is 6° to 11° with a sprint lean, and release-height coverage is 0% to 6%. The kinetic pillar label agrees with its true-value version in 52.5% of trials.
  - Literature: action-class thresholds are unsourced. Timing priors are about 2.5 SD above Felton 2019. The Alway 2021 figure of 88% is in-sample against a 78% base rate, and that paper reports no link between SCR and LBSI and less side flexion at release in injured bowlers.
- **Findings & Decisions:** 27 numbered bugs, 7 of them P0 (`docs/independent_audit_and_improvements.md` section 5). The EXP-005 sign-off does not stand. Nothing was fixed in this cycle, so the baseline numbers above are unchanged and now carry the audit status. Next actions in order: repeated-frame check in Stage 0, release sanity gates, gating of translation-dependent metrics, per-metric benchmark scoring, pitch-based calibration with `K` passed to NLF, and a first real reference capture (trunk IMU, radar, tape).
- **Artifacts / Commit:** `docs/independent_audit_and_improvements.md`, `docs/results/exp007/`, `docs/figures/exp007_*`, `validation/audit_*.py`. Commit `0115c48`.

### EXP-008 — Phase 1.5 Foundation Hardening, Full EXP-007 Bug Resolution (`B1–B27`), Pre-NLF Two-Wicket Calibration & Honest Metrology (`v4`)
- **Date:** 2026-11-23
- **Goal / Hypothesis:** Resolve all 27 numbered bugs (`B1–B27`) and foundational metrology flaws uncovered in `EXP-007` before starting Phase II: eliminate false releases on edited/truncated clips, enforce timebase integrity & zero-drift pulldown deduplication, calibrate telephoto camera intrinsics `K` & gravity *before* NLF, replace ad-hoc uncertainty constants with 4-source analytical Jacobians + empirical GPU repeatability intervals, retire unvalidated 0–100 composite scores in favor of interval-aware cues & definition-matched primary-source norms, score all 22 metrics non-circularly, and lock regression fixtures on real footage.
- **Modifications:**
  - Added `src/bowling_biomech/constants.py` (decoupling production from `synthetic_ground_truth.py`, exact MCC Law 8 `0.0960 m` stump half-spacing, Felton 2019 `192/108 ms` priors) and `tests/fixtures/{steyn,indoor,mcgrath,ambrose}_nlf_fixture.npz` (with `!tests/fixtures/*.npz` in `.gitignore`).
  - Refactored `schemas.py` (`v2.0.0`), `stage0_audit.py` (`analyze_frame_cadence`, regular pulldown dedup, `measure_bowler_roi_blur`, 4-DOF KLT+RANSAC pan/tilt/zoom), `stage1_pitch_calibration.py` (larger-side `K`, `solve_two_wicket_calibration`, multi-frame Law 8 `auto_detect_two_wickets`, `world_up_from_calibration`, `is_panning_camera` preservation, strict `cam_pos == -R.T @ t`, sagittal sprint-lean de-leaning on uncalibrated torso gravity), `stage2_nlf_3d.py` (streaming decoding, passing `intrinsic_matrix` & `world_up_vector` directly into `detect_smpl_batched`, GPU repeatability ensemble `base/flip/aug3`, gap-inflated uncertainty, pinned `model_sha256`), `stage2_event_spotting.py` (Winter `C=1.2465`, `<100 fps` peak preservation, noise-robust median segment arm length, coronal reach `r_cor` & arc-restored 25–30 fps bracketing BR sanity gates, continuous-landing FFC/BFC touchdown refinement, `_boundary_clamped` confidence drops).
  - Refactored `stage3_kinematics.py` (local front-foot `[FFC, BR]` ground reference, Rician/Jensen near-180° debiasing, Savitzky-Golay `~50 ms` arm omega, `[60°, 420°)` joint-anchored Portus 2001/2004 AIS alignment/SCR, removed dead `crunch_factor_proxy`), `stage4_uncertainty.py` (4-source analytical Jacobians with orthogonal `sigma_epistemic`/`sigma_projective` split, joint `follow_through_drift_deg` Jacobian, contact-kink RMS temporal sensitivity, GPU repeatability floor, fixed `>0.82` before `>0.65` branch order, strict incomplete/timebase/translation gating, permanent ICC `REFUSED`), `stage5_evaluator.py` (retired 0–100 composites, definition-matched Cohort Comparisons, ECB 2024 / CA 2026 Workload Guidance, strict non-straddling printed-interval cues routing straddlers to `inconclusive_observations`, tailored tripod/pan upgrades), `stage6_coach_llm.py`, `stage6_visualizer.py` (`[lo, hi]` overlays, `[UNVERIFIED TIMEBASE]` labels, ground-zeroed/ankle-anchored 3D virtual views, Cohort Z-Score Forest Plot, browser-safe H.264 `ffmpeg` freeze-video transcode), `pipeline.py`, `cli.py`, `pyproject.toml`, `synthetic_ground_truth.py` (continuous non-teleporting legs & pelvis clearance, non-looping follow-through), `validation/run_synthetic_benchmark.py` (all 22 metrics, `false_incomplete_trials`, noise-mismatch & noisy 12° lean self-cal arms), `tests/test_core_biomechanics.py`, `docs/research_report.md`, `docs/walkthrough.md`, and `README.md`.
  - Collaborated with **Gemini 4 Carbon (High)** (`p15-partner`) across design and post-implementation audit rounds, catching and fixing 6 P1 items (`follow_through_drift_deg` joint Jacobian, 25–60 fps false-incomplete elimination from `18/144 -> 0/240`, refreshed `bowler_roi` blur warnings & `[UNVERIFIED TIMEBASE]` labels, orthogonal `epistemic/projective` split & repeatability floor, front-ankle-anchored 3D side view, and primary-source documentation updates) plus 4 P2 refinements.
- **Results:**
  - `pytest -v`: **18 / 18 PASSED** (`2.57 s`), including 4 cached real-clip fixtures.
  - Per-metric synthetic benchmark (`docs/results/v4/synthetic_benchmark_results.json`): **96.4%** pooled coverage across **all 22 / 22 metrics** (3,003 evals, per-metric **91.1%–100.0%**, `rmse/sigma` in `[0.71, 1.21]`, **`0 / 240` false incomplete trials**); `25–240 fps` BR MAE `2.7–3.5 ms`; noisy 12° lean self-cal signed Az MAE `0.9°–1.2°`, RH bias `+0.002–0.010 m`.
  - Unmodified `EXP-007` `S1–S9` (`docs/results/v4/synthetic_diagnostics.json`): `S1` min/max **91.7%–100.0%** (pooled 97.1%), `S2` 240 fps `0%` > 40 ms, `S3` non-teleporting `0.030 m`, `S6` release-height coverage **`100.0%`** across all 15 lean/angle combos (was `0–6%`), `S7` `0` failures down to `6°` FOV, `S8a` `+6.0°` (was `-354.0°`), `S8b` `4 / 4` detected (`3.743` vs `3.74`), `S9` **`34 / 34 = 100.0%`** cue precision & recall (was `3.8%`).
  - RTX 4090 GPU 1 real clips (`docs/results/v4/`): Steyn auto-deduplicated 1-in-6 pulldown (`25.18 unique fps`), auto-calibrated `two_wicket_geometry` (`f = 10,208 px`, cam `24.7 m` back) into NLF (pelvis depth `24.03 m`, depth travel `10.85 m`), crisp `bowler_roi` (`433.7`), `thorax_lat_flex_br_deg = 30.26° [26.8, 36.5]` (`high`, `repeatability_interval`), `SCR = 46.17° [36.9, 55.0]`, panning/drift suppressed; Indoor flagged `[UNVERIFIED TIMEBASE]` with 0 contradictory cues; McGrath & Ambrose cleanly returned `is_complete_delivery=False` with **0 / 23 metrics** and **0 false cues/scores**.
- **Findings & Decisions:** Every one of `EXP-007`'s 27 bugs (`B1–B27`) is closed and verified against both synthetic diagnostics and real RTX 4090 footage. Phase 0–1.5 is now a solid, honest metrology foundation ready for **Phase II** (Web UI, multi-view sync, and first real IMU/radar/tape reference capture).
- **Artifacts / Commit:** `docs/results/v4/`, `docs/figures/*_v4_*`, `tests/fixtures/*.npz`, `README.md`, `docs/walkthrough.md`, `docs/research_report.md`. Commit `bd898a9`.

### EXP-009 — Phase II Interactive Coaching Studio, Unsynchronized Dual-Phone 3D Fusion, `bml_movi_87` ISB Anatomy, 3-DOF Pan Tracker, Optical Ball Speed, Verifier-Gated Coach Copilot & 2-Round Gemini 4 Carbon (High) Audit (`v5`)
- **Date:** 2026-11-23
- **Goal / Hypothesis:** Deliver full Phase II (`M2A`–`M2D`) for fast/fast-medium pace bowlers without regressing any `v4` (`EXP-008`) metrology invariants: local-first Coaching Studio with `<35 ms` CPU HITL recompute + Warm Remote RTX 4090 GPU Worker, zero-extra-weight `bml_movi_87` ISB rigid-body anatomy with correlated-depth `N_eff` guardrail, 3-DOF SO(3) rotational pan tracking, unsynchronized dual-phone `+Y_world` sync + anisotropic ray-precision 3D fusion, calibrated side-on asymmetric wrist-ROI optical ball speed, 1-click `1080×1920` Truthful Breakdown Reel, and deterministic numeric-verifier Coach Copilot — then harden end-to-end via an independent 2-round **Gemini 4 Carbon (High)** audit.
- **Modifications:**
  - **CV & Biomechanics (`src/bowling_biomech/`):** Added `stage1_pan_tracker.py` (1-frame MCC PnP + bowler-masked KLT/RANSAC SO(3) rotation, fixed `C_0 = -R_t^T t_t`, `median_residual_px <= 2.5 px` gate), `stage2_isb_anatomy.py` (`bml_movi_87` Wu et al. 2002/2005 frames, 3D coordinate-invariant `R_pelvis^T @ R_thorax` lumbopelvic flexion, unclamped external/internal front-foot angle, `N_eff = N/(1+(N-1)*rho)` MetricEstimate intervals + completeness/plane gates), `stage2_dual_sync_fusion.py` (`+Y_world` 1000 Hz spline FFT NCC sync, `dual_mcc_pnp` / `skeleton_se3_relative` alignment, `(Lambda_1 + Lambda_2)^-1` ray fusion, PSD matrix-sqrt cross-covariance `S1 @ S2`, `baseline_sin >= 0.45` collinearity gate), `stage0_net_clipper.py` (~6 fps motion-burst auto-clipper with base64 thumbnails), `stage2_ball_release.py` (asymmetric `+3.2 m` down-pitch flight-corridor ROI, forearm/hand capsule masking, depth+azimuth uncertainty propagation, strict hard gates), `stage6_coach_copilot.py` (deterministic + optional Gemini Q&A wrapped in clause-level `verify_coach_response` blocking ICC verdicts, `/100` composites, suppressed leaks, and ungrounded numbers), `stage2_nlf_3d.py` (`all_tracks_snapshot`, `extract_bml_movi_87_for_boxes` with mm/m unit guards), and `stage6_visualizer.py` (`render_truthful_breakdown_reel_9x16` with audit-to-proxy resolution scaling and permanent ICC refusal banner).
  - **Studio Backend & Warm GPU Worker (`server/`, `gpu_worker/`, `cli.py`, `pyproject.toml`):** Added WAL SQLite `db.py`, 3-tier QuickTime/ffprobe + All-Intra proxy `video_proxy.py`, Telemetry v2.0 + `<35 ms` HITL recompute + Pillar 3 repeatability/spaghetti/fatigue + demo seeder `recompute.py`, FastAPI app factory `app.py` (bounded 512 MB streaming uploads, suffix whitelist, calendar-day workload windows, dual-fusion covariance persistence), and Warm GPU Worker `worker.py` (lifespan JIT warmup, SHA-256 disk cache, `auto_cal_json` preservation).
  - **Interactive Studio Frontend (`web/`):** Built Vite 6 + React 18 + TypeScript + Tailwind + Zustand (`subscribeWithSelector`) + Three.js (`@react-three/fiber`/`drei`) + `uPlot` Coaching Studio (`VideoCanvasPlayer`, `PitchViewport3D`, `WaveformPanel`, `StudioInspector`, `App.tsx`) with 0-frame-synced aspect-preserving canvas scrubber & 4× pixel loupe, real-time `ms`-interpolated FFC-aligned Ghost Skeleton, `OrbitControls` rig reset, `Upload / Net Scan` drawer, Multi-Ball Spaghetti SVG & Workload logging, Dual-Phone 3D Fusion, and Verifier-Gated Coach AI.
  - **Independent 2-Round Gemini 4 Carbon (High) Audit (`p2-auditor`) & Patches:** Caught and fixed 5 P0s (6/6 adversarial verifier bypasses including compound-sentence exemption poisoning & unicode `15°`; ISB Guardrail #3 + 30° side-on lumbopelvic frame mismatch + 90° foot clamp + pan de-rotation; 173-frame initial video desync + `object-contain` click drift + 1.5× 1080p Reel skeleton scaling; `fuse-dual -> /recompute` covariance/handedness preservation + PSD `S1 @ S2` cross-covariance; calendar-date workload filtering & TS schema sync), 4 P1s (asymmetric down-pitch ball-speed ROI + forearm mask + wiring; 3-DOF pan `pan_anchor_frame` + `D * sigma_pan_rad` drift + `needs_gpu_refit_for_K` metric gate; `auto_cal_json` preservation + startup JIT warmup + upload caps; Ghost `ms` sync + `CameraRig` + Upload/Spaghetti/Workload UI), 3 P2s (`.gitignore`/`sync_to_server.sh` hygiene, uPlot clipped x-axis + `ResizeObserver`, expanded non-circular test suite), and 2 Round-2 catches (`R2-1` verbatim suppression/workload prose stripping + slash/range FPS regex; `R2-2` `candidate_id` vs `candidate_idx` in Net Auto-Scan drawer). Received **`FINAL SIGN-OFF APPROVED`**.
- **Results:**
  - `pytest -v`: **27 / 27 PASSED** (`24.71 s` macOS, `31.35 s` Linux RTX 4090); `cd web && npm run build`: **0 TypeScript errors** (`3.89 s`).
  - Zero regression on Phase 1.5 synthetic benchmark (`96.4%` pooled across all 22 metrics, `3,003` evals, `91.1%–100.0%` per-metric, `0 / 240` false incomplete trials).
  - RTX 4090 GPU 1 live verification on Dale Steyn (`docs/results/v5/phase2_gpu_verification.json`): Warm Worker lifespan JIT warmup `26.62 s` (`jit_warmed=true`), steady-state SMPL-24 + `bml_movi_87` inference **`6.83 s`**, SHA-256 cache hit **`7.84 ms`**, `auto_cal_json` preserved `two_wicket_geometry` (`fx=10,207.6 px`), 3-DOF pan track verified at **`0.188 px`** median residual (`max_pan=0.91°`, `227` min inliers), and `bml_movi_87` `N_eff` gating honestly suppressed sagittal/foot angles on telephoto rear view while reporting transverse Pelvis/Thorax/SCR with wide `low`-grade CIs.
  - Coach Copilot Verifier: **9 / 9** adversarial bypasses blocked (100%) and **30 / 30** grounded prompts across all 5 seeded deliveries + adolescent `U17` workload passed (`0` false blocks).
  - Live Browser UI Audit (`docs/figures/studio_v2_preview.png`): 0 console errors, 0 failed network requests, exact `Frame 173/279` sync, unclipped uPlot axes, live Dual-Phone 3D Fusion, Ghost overlay, and Coach AI verified.
- **Findings & Decisions:** Phase II (`M2A`–`M2D`) is complete, audit-hardened, and verified across local CPU, browser UI, and RTX 4090 GPU 1. Next frontier (Phase III): first real physical reference validation dataset (wearable trunk IMU + speed radar + measured tape/stump markers) to validate absolute real-world 3D accuracy beyond repeatability.
- **Artifacts / Commit:** `docs/results/v5/phase2_gpu_verification.json`, `docs/figures/studio_v2_preview.png`, `src/bowling_biomech/{stage0_net_clipper,stage1_pan_tracker,stage2_isb_anatomy,stage2_dual_sync_fusion,stage2_ball_release,stage6_coach_copilot}.py`, `src/bowling_biomech/server/*`, `src/bowling_biomech/gpu_worker/*`, `web/src/*`, `tests/test_phase2_studio.py`.

---

## 5. Entry Template for Future Experiments

```markdown
### EXP-XXX — <Short Descriptive Title>
- **Date:** YYYY-MM-DD
- **Goal / Hypothesis:** <What problem or upgrade is being tested?>
- **Modifications:** <Files changed and algorithmic/config summary>
- **Results:** <Exact metrics, pytest count, benchmark RMSE/CI95, real-video outcomes>
- **Findings & Decisions:** <What worked, what failed, what was kept/reverted, next step>
- **Artifacts / Commit:** <Relative paths in docs/results or docs/figures + git SHA>
```
