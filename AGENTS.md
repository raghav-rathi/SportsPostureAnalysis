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

Current flagship module: **Phase 0–1 Cricket Fast-Bowling Biomechanics Engine** (`src/bowling_biomech/`, fast & fast-medium pace bowlers, left & right arm; Phase 2+ expands to multi-view sync, spin bowling, interactive UI, and broader sports posture modules).

| Component | Path | Responsibility |
|---|---|---|
| Shared Constants | `src/bowling_biomech/constants.py` | Canonical `SMPL24_JOINTS`, `JOINT_IDX`, `LR_JOINT_PAIRS`, MCC Laws 6/7/8 geometry (`0.0960 m` stump half-spacing), Felton (2019) timing priors (`192/108 ms`) |
| Data Contracts | `src/bowling_biomech/schemas.py` | Strict Pydantic v2 contracts (`VideoAudit` with timebase/ROI fields, `PitchCalibration` with intrinsics/sigmas, `DeliveryEvents` with `source_frame_idx`, `MetricEstimate` with `interval_type`, `CohortComparison`, `WorkloadGuidance`, retired composite pillars) |
| Stage 0 | `src/bowling_biomech/stage0_audit.py` | Timebase integrity (`analyze_frame_cadence` + regular pulldown deduplication + freeze/cut detection), bowler-ROI blur (`measure_bowler_roi_blur`), 4-DOF KLT+RANSAC pan/tilt/zoom motion detector (`detect_camera_motion`) |
| Stage 1 | `src/bowling_biomech/stage1_pitch_calibration.py` | Pre-NLF MCC Laws 6/7/8 PnP (`SOLVEPNP_IPPE`, larger-side `K`), zero-annotation multi-frame Law 8 `auto_detect_two_wickets` + `solve_two_wicket_calibration`, `world_up_from_calibration`, standing-background / de-leaned torso gravity self-cal, strict `cam_pos == -R.T @ t` |
| Stage 2A | `src/bowling_biomech/stage2_nlf_3d.py` | Streaming multi-person metric 3D SMPL-24 + uncertainties via pinned NLF TorchScript passing `intrinsic_matrix` & `world_up_vector`, GPU repeatability ensemble (`base`/`flip`/`aug3`), gap-inflated uncertainty, `(T, 24)` `joint_in_frame` visibility |
| Stage 2B | `src/bowling_biomech/stage2_event_spotting.py` | True Winter (2009) dual-pass Butterworth (`C=1.2465`), strict overhead/slingy/25fps-bracketed BR sanity gates (`is_complete_delivery`), Felton (2019) priors, continuous-landing FFC/BFC touchdown refinement, `_boundary_clamped` confidence drops |
| Stage 3 | `src/bowling_biomech/stage3_kinematics.py` | Canonical Pitch World Frame, Z–X Cardan global trunk inclination, Rician/Jensen near-180° knee debiasing, Savitzky-Golay `~50 ms` arm omega, local front-foot ground reference, `[60°, 420°)` joint-anchored Portus (2001/2004) AIS alignment/SCR |
| Stage 4 | `src/bowling_biomech/stage4_uncertainty.py` | 4-source analytical Jacobian uncertainty (orthogonal `sigma_epistemic` + `sigma_projective`, contact-kink `sigma_temporal`, FD `sigma_calibration`, `sigma_scale`), GPU ensemble `repeatability_interval` with systematic floor, incomplete/timebase/translation/crop/`<90 fps` gating, permanent ICC `REFUSED` |
| Stage 5 | `src/bowling_biomech/stage5_evaluator.py` | Retired 0–100 composites (`score=None`), definition-matched primary-source Cohort Comparisons, ECB (2024) / Cricket Australia (2026) Workload Guidance, strict non-straddling printed-interval cues (`inconclusive_observations`), tailored upgrades, Dual-View fusion |
| Stage 6 | `src/bowling_biomech/stage6_visualizer.py`, `stage6_coach_llm.py` | Sequential `source_frame_indices` 2×2 keyframe strips (`[lo, hi]` intervals), 6-panel dashboards (ground-zeroed/ankle-anchored 3D views + Cohort Z-Score Forest Plot), H.264 (`yuv420p`) freeze video, strictly JSON-grounded Markdown reports |
| Orchestration | `src/bowling_biomech/pipeline.py`, `cli.py` | Calibration-before-NLF single-view & dual-view pipelines and Typer CLI (`bowling-biomech`) |
| Metrology & Tests | `src/bowling_biomech/synthetic_ground_truth.py`, `validation/`, `tests/` | Continuous non-teleporting Z–X Cardan synthetic generator, all-22 per-metric non-circular AR(1) benchmark + noise-mismatch & noisy 12° lean self-cal arms, 18-test pytest suite + 4 cached real-clip fixtures (`tests/fixtures/*.npz`) |
| Research & Reports | `docs/research_report.md`, `docs/walkthrough.md`, `docs/figures/`, `docs/results/` | Corrected primary-source scientific report, forensic Reel deconstruction, Phase 0–1.5 walkthrough, verified `v4` visuals & JSON outputs |
| Audit Diagnostics (EXP-007) | `validation/audit_synthetic_diagnostics.py`, `validation/audit_real_clip_diagnostics.py`, `validation/audit_nlf_self_consistency.py`, `docs/independent_audit_and_improvements.md` | Per-metric benchmark scoring and stress cases (`S1–S9`), repeated-frame and release checks on finished runs, pose-model consistency checks, and the `EXP-007` audit report |

---

## 3. Current Verified Baseline Snapshot (`v4`, Phase 1.5 Post-EXP-007 Hardening)

- **Unit & Cached Real-Clip Regression Suite (`pytest -v`):** **18 / 18 PASSED** (`2.57 s`, including 4 compact real-clip NLF fixtures in `tests/fixtures/*.npz`).
- **Per-Metric Non-Circular Synthetic Metrology Benchmark (`docs/results/v4/synthetic_benchmark_results.json`):**
  - **All 22 / 22 metrics scored** (3,003 observable AR(1) correlated-noise evaluations, randomized release angle & Felton 2019 timings, non-teleporting foot landings, non-looping follow-through): **96.4%** pooled analytical coverage, per-metric range **91.1%–100.0%**, sharpness `rmse/sigma` in **`[0.71, 1.21]`**, **`0` false incomplete trials** out of 240.
  - **Non-Oracle Noise Mismatch Stress (all 22 @ 45° 120 fps):** `0.5x = 100.0%`, `1.0x = 96.5%`, `2.0x = 78.2%`.
  - **Event Timing MAE:** `25 fps`: BFC `11.2 ms`, FFC `14.4 ms`, BR `3.5 ms`; `30 fps`: BFC `14.2 ms`, FFC `11.5 ms`, BR `3.2 ms`; `60 fps`: BFC `11.7 ms`, FFC `8.7 ms`, BR `3.1 ms`; `120 fps`: BFC `8.6 ms`, FFC `8.3 ms`, BR `2.9 ms`; `240 fps`: BFC `5.5 ms`, FFC `8.3 ms`, BR `2.7 ms`.
  - **Noisy Realistic Self-Calibration Arm (120 fps, 12° sprint lean, ±10° approach, AR(1) noise):** Signed Az MAE `0.9°–1.2°`, local release-height bias `+0.002–0.010 m`, LatFlex RMSE `4.6°–5.1°` (`100.0%` cov).
- **Unmodified `EXP-007` S1–S9 Diagnostic Suite (`docs/results/v4/synthetic_diagnostics.json`):**
  - `S1`: Pooled all-22 = **97.1%**, min metric = **91.7%**, max = **100.0%** (all 22 / 22 metrics scored, including `follow_through_drift_deg = 98.8%`).
  - `S2` (240 fps): BFC MAE `6.7 ms` (`-0.8` bias), FFC MAE `7.9 ms` (`+0.7` bias), BR MAE `1.8 ms`, **`0.0%`** errors > 40 ms.
  - `S3`: Non-teleporting FFC `0.030 m`, BFC `0.031 m`; non-looping arm follow-through `108° -> 113°`.
  - `S6` (5 lean/approach scenarios up to 18° lean & ±15° angle): Gravity MAE `1.2°–4.6°`, pooled cov `95.3%–100.0%`, **release-height coverage = `100.0%`** across all 15 scenario×view combos (up from `0%–6%`).
  - `S7`: `0` PnP failures down to `6°` FOV, Az MAE `0.2°–0.5°`.
  - `S8a`: Transverse anchor straddle = **`+6.0°`** (exact match to true `6.0°`, was `-354.0°`).
  - `S8b`: All **4 / 4** pan/tilt/zoom cases detected (`3.743` vs `3.74` true on 300-frame pan).
  - `S9`: Interval-aware cue precision & recall = **`100.0%`** (`34 / 34` correct, zero false alarms; unvalidated 0–100 composites retired).
- **Verified RTX 4090 Real-Video Runs (`docs/results/v4/`, GPU 1 — repeatability & internal consistency verified; absolute real-world 3D accuracy awaits future IMU/radar/mocap validation):**
  - `Dale_Steyn_Net_25fps`: Auto 1-in-6 pulldown dedup (`25.18 unique fps`, 63 frames), `bowler_roi` blur = `crisp (433.7)`, zero-annotation Law 8 `two_wicket_geometry` (`f = 10,208 px`, cam `24.7 m` back) passed to NLF -> pelvis depth **`24.03 m`**, depth range **`10.85 m`** (vs `4.38 m` / `1.92 m` in `v3`), `BFC = src 63`, `FFC = src 70`, `BR = src 71`, `thorax_lat_flex_br_deg = 30.26° [26.8, 36.5]` (`high`, `repeatability_interval`), `SCR = 46.17° [36.9, 55.0]` (`moderate`), panning translation/drift suppressed, `ICC = REFUSED`.
  - `Indoor_SlowMo_120fps`: Flagged `edited_or_frozen` (`timebase_unreliable = True`, `30%` repeats), labeled `120.0 nominal fps [UNVERIFIED TIMEBASE]`, `bowler_roi` `crisp (493.4)` with clean warnings, cropped knee/velocities/drift suppressed, `0` false/contradictory cues.
  - `Glenn_McGrath_Reel_SideOn` & `Curtly_Ambrose_Reel_Rear`: Both flagged `edited_or_frozen` (`66%` / `82%` repeats; McGrath scene cut @ 56; Ambrose `bowler_roi` `severe_blur = 19.1`), **`is_complete_delivery = False`**, **`0 / 23` reported metrics**, **`0` cues**, **`0` scores** (withdrawing all invalid `v3` Reel outputs cleanly).

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
- **Artifacts / Commit:** `docs/results/v4/`, `docs/figures/*_v4_*`, `tests/fixtures/*.npz`, `README.md`, `docs/walkthrough.md`, `docs/research_report.md`.

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
