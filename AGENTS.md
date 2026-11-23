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
| Data Contracts | `src/bowling_biomech/schemas.py` | Strict Pydantic models (`VideoAudit`, `PitchCalibration`, `DeliveryEvents`, `MetricEstimate`, `BiomechanicalFactSheet`, `CoachingReport`) |
| Stage 0 | `src/bowling_biomech/stage0_audit.py` | Sub-clip FPS/resolution/blur audit + Sobel-textured adaptive optical-flow camera panning detector |
| Stage 1 | `src/bowling_biomech/stage1_pitch_calibration.py` | MCC Law 7 PnP (`SOLVEPNP_IPPE` coplanar support) + standing-background/upright-torso gravity self-calibration + geometric panning override |
| Stage 2A | `src/bowling_biomech/stage2_nlf_3d.py` | Multi-person metric 3D SMPL-24 + native uncertainties via NLF TorchScript, 3D depth-aware bowler tracking, `(T, 24)` `joint_in_frame` visibility, standing background gravity extraction, arm auto-inference |
| Stage 2B | `src/bowling_biomech/stage2_event_spotting.py` | Causal sub-frame event spotting (`BFC < FFC <= UAH < BR < FT`), overhead release verification (`is_complete_delivery`), slow-mo & ankle-visibility warnings |
| Stage 3 | `src/bowling_biomech/stage3_kinematics.py` | Zero-lag Butterworth filtering, canonical Pitch World Frame (`+X` down pitch, `+Y` up, `+Z` off-to-leg), exact Z–X Cardan `thorax_lat_flex_deg`, BFC-anchored unwrapped transverse alignment/SCR, stature estimation |
| Stage 4 | `src/bowling_biomech/stage4_uncertainty.py` | Camera-view observability matrix, per-metric event-specific crop/uncertainty gating, 3-source 95% CI propagation, physical clamping, `<90 fps` high-velocity gating, permanent ICC refusal |
| Stage 5 | `src/bowling_biomech/stage5_evaluator.py` | Peer-reviewed 4-pillar evaluator (`Worthington 2013`, `Bayne 2016`, `Alway 2021`, `Portus 2004`), two-sided knots, interior-extremum CI scan, observability-weighted pillars, azimuth-gated capture upgrades, dual-view fusion |
| Stage 6 | `src/bowling_biomech/stage6_visualizer.py`, `stage6_coach_llm.py` | Portrait-safe 2×2 keyframe strips, 6-panel dashboards (virtual 3D views + dimmed suppressed curves), freeze-frame MP4 export, strictly JSON-grounded coaching reports |
| Orchestration | `src/bowling_biomech/pipeline.py`, `cli.py` | End-to-end single-view & dual-view pipelines and Typer CLI (`bowling-biomech`) |
| Metrology & Tests | `src/bowling_biomech/synthetic_ground_truth.py`, `validation/`, `tests/` | 3D Z–X Cardan synthetic delivery generator, AR(1) correlated depth-drift benchmark + uncalibrated self-cal arm, FPS degradation stress test, 13-test pytest suite |
| Research & Reports | `docs/research_report.md`, `docs/walkthrough.md`, `docs/figures/`, `docs/results/` | First-principles scientific report, forensic Reel deconstruction, build/audit walkthrough, verified visuals & JSON outputs |

---

## 3. Current Verified Baseline Snapshot (`v3`, Post-Audit)

- **Unit & Integration Suite (`pytest -v`):** **13 / 13 PASSED** (`1.26 s`)
- **Synthetic AR(1) Metrology Benchmark (`tau = 80 ms`, Right + Left Arms, 912 observable evals):**
  - **Overall 95% CI Empirical Coverage:** **97.0%**
  - **240 fps BR Timing MAE:** `0.6–2.2 ms` across `90°`, `45°`, `0°`, and `24° broadcast`
  - **240 fps Front-Knee BR RMSE:** `6.10°` (`90° side`), `5.38°` (`45° open-side`) | Rear/Broadcast properly suppressed
  - **240 fps Thorax Lat-Flex BR RMSE:** `2.79°` (`0° rear`), `2.81°` (`24° broadcast`), `3.84°` (`45° open-side`) | Side properly suppressed
  - **60 fps High-Velocity Arm Delay:** Properly suppressed (`<90 fps` gate)
  - **Uncalibrated Self-Calibration Arm (120 fps, No Oracle `K` / No MCC Landmarks):** Azimuth MAE `4.49°–4.58°`, `FK_BR RMSE = 0.03°`, `LatFlex RMSE = 2.99°` (`rear_0`) / `4.47°` (`side_on_90`, `open_side_45`)
- **Verified RTX 4090 Real-Video Runs (`docs/results/v3/`):**
  - `Dale_Steyn_Net_30fps` (native 30 fps, rear): `11` standing-umpire gravity frames, `az = -2.9°`, `el = -1.0°`, `stature = 1.80 m` (actual `1.79 m`), `BFC=38, FFC=44, UAH=45, BR=47`, `thorax_lat_flex_br = 30.25° [19.7, 40.8] (high)`, `Durability = 89.7`, handheld pan (`bg_pan = 0.0805`) & sagittal knee suppressed, `ICC = REFUSED`.
  - `Indoor_SlowMo_120fps` (120 fps, front-on): `az = 177.3°`, `el = 9.9°`, sofa-cropped front ankle caught & hard-suppressed (`[UNMEASURABLE]`), zero yaw/lat-flex spikes, `thorax_lat_flex_br = 18.51°`, `Durability = 73.8`, handheld pan (`bg_pan = 0.211`) detected.
  - `Glenn_McGrath_Reel_SideOn` (25 fps archival portrait Reel): geometric pan override locked `side_on_left (az = -98.9°, el = 2.4°, pan = True)`, causal `BFC=42, FFC=53, UAH=53, BR=57`, blurry 25 fps knee & panning run-up/stride suppressed, readable 9:16 strip with warm amber warnings.
  - `Curtly_Ambrose_Reel_Rear` (25 fps archival static broadcast): `broadcast_elevated (az = 2.5°, el = 20.4°, pan = False, bg_pan = 0.0001)`, `UAH = None` placeholder, two-sided knots flagged `Durability = 54.2`, azimuth-gated upgrades recommend `120–240 fps` and `Side-On (90 deg)` only.

---

## 4. Chronological Experiment, Audit & Modification Ledger

### EXP-001 — Reference Reel Forensic Deconstruction & First-Principles Literature Synthesis
- **Date:** 2026-11-23
- **Goal:** Reverse-engineer reference Instagram Reel (`@athleteworld.in` / Satyam Patkar) and synthesize peer-reviewed cricket fast-bowling biomechanics + 2024–2026 3D CV pose literature.
- **Key Findings:**
  - Reference creator does **not** run AI biomechanical measurement: zero numeric angles, 25 fps archival clips, manual 2D overlays, and generative-AI upscaled infographics introducing anatomical distortions (wrong arm inset on Wasim, 2D parallel bars misrepresenting 3D transverse separation on McGrath/Donald, unrealistic `0°` vertical spine on Ambrose).
  - Grounded evaluation pillars in peer-reviewed norms: `Worthington et al. (2013)` ($R^2 = 0.74$ pace predictors), `Alway et al. (2021)` (88% prospective LBSI model), `Bayne et al. (2016)` ($50^\circ \pm 6^\circ$ injured vs. $40^\circ \pm 8^\circ$ uninjured thorax lateral flexion), `Portus et al. (2004)` ($\ge 30^\circ$ SCR mixed action), `Abraham, Feros, & Fox (2025/2026)` (OpenCap cricket RMSE benchmarks), and `Bench et al. (2022)` (`pitchAI`).
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
