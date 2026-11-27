# SportsPostureAnalysis Studio v5 — Peer-Reviewed Biomechanics & Systems Architecture Specification

**Document Version:** `5.0.0-spec`  
**Prepared:** November 2026  
**Authors:** Biomechanics Research & Backend Systems Architecture (`@biomech-researcher`)  
**Target Codebase:** `src/bowling_biomech/` + `web/` (Phase 5 Studio Expansion)  
**Zero-Regression Contract:** 100% compatibility with all 30 existing `pytest` tests (`tests/test_core_biomechanics.py`, `tests/test_phase2_studio.py`), identical 22-metric AR(1) synthetic benchmark calibration (`96.4%` coverage across 3,003 observable evaluations), and strict adherence to every guardrail in `AGENTS.md` §1.2.

---

## 1. Scientific Foundation & Guardrail Architecture

### 1.1 Non-Negotiable Neuro-Symbolic & Biomechanical Guardrails (`AGENTS.md` §1.2)
Every module specified in this document is governed by five structural invariants:

1. **Permanent ICC 15° Elbow Legality Refusal:** `icc_elbow_legality_verdict` remains permanently `ObservabilityGrade.REFUSED` (`suppressed=True, value=None`) across single-view, pan-tracked, stroboscopic, archetype-compared, and dual-phone 3D fused pipelines. Accredited $\ge 250\text{ Hz}$ multi-camera laboratory 3D optical motion capture with anatomical upper-arm/forearm marker clusters is required (Elliott, Alderson, & Denver, 2007; Abraham, Feros, & Fox, 2025, *IJSSC*: `22.71°` elbow RMSE even on three 240 Hz iPads).
2. **Medical Non-Diagnosis Boundary:** Mechanical load, lumbopelvic posture, head fall-away, fatigue drift, and ACWR workload/soreness trackers quantify **mechanical exposure and training load only**—never clinical injury diagnosis or probability of lumbar bone stress injury (LBSI).
3. **Universal `MetricEstimate` Contract (Zero Naked Numbers):** Every scalar biomechanical output added in Modules 1–6 is wrapped in `MetricEstimate` carrying `(value, ci_low, ci_high, sigma_total, sigma_epistemic, sigma_projective, sigma_temporal, sigma_calibration, observability)` or explicit `suppressed=True` with an actionable `suppression_reason`.
4. **Zero Fabricated 0–100 Composite Scores:** Unvalidated `0–100` composite scores remain permanently retired (`score=None`). Sequence quality, braking efficiency, archetype comparisons, and spell consistency use physically meaningful units (`ms`, `deg`, `deg/s`, `m/s`, `%`, `cm`, `% stature`, `ratio`) and categorical interval-verified verdicts (`optimal_proximal_to_distal`, `simultaneous_rotational`, `out_of_order`, `unmeasurable`).
5. **Physical Bounds & Non-Circular Verification:** All confidence intervals clamp to physical limits (`[0°, 180°]` internal angles, `[0%, 100%]` bounded ratios, `>= 0` magnitudes). Every sign convention is verified via analytical non-circular vector tests for **both right-arm and left-arm** bowlers.

### 1.2 Canonical Pitch World Coordinate Frame & Chirality Normalization
From `src/bowling_biomech/constants.py` and `stage3_kinematics.py`:
- **Origin $(0, 0, 0)$:** Base of middle stump at the bowler's bowling crease (`MCC Law 7` popping crease at $X = +1.22\text{ m}$; striker's bowling crease at $X = +20.12\text{ m}$).
- **$+X$:** Horizontal down-pitch axis toward the striker's wicket.
- **$+Y$:** Vertical gravity-up axis.
- **$+Z$:** $+X \times +Y$, pointing horizontally to the bowler's **right** when facing $+X$ (for a right-arm bowler over the wicket: $+Z =$ leg/bowling-arm side, $-Z =$ off/contralateral/front-leg side).
- **Chirality Sign $s_{\text{arm}}$:**
  $$s_{\text{arm}} = \begin{cases} +1, & \text{bowling\_arm} = \texttt{"right"} \\ -1, & \text{bowling\_arm} = \texttt{"left"} \end{cases}$$
  - **Contralateral (non-bowling / front-leg) lateral unit vector:** $\hat{\mathbf{u}}_{\text{contra}} = (0, 0, -s_{\text{arm}})$.
  - **Ipsilateral (bowling-arm) lateral unit vector:** $\hat{\mathbf{u}}_{\text{ipsi}} = (0, 0, +s_{\text{arm}})$.
  - **Front limb indices:** `(lhip, lkne, lank, ltoe)` if $s_{\text{arm}} = +1$, else `(rhip, rkne, rank, rtoe)`.
  - **Lead (non-bowling) arm indices:** `(lsho, lelb, lwri, lhan)` if $s_{\text{arm}} = +1$, else `(rsho, relb, rwri, rhan)`.

### 1.3 Primary-Source Literature Synthesis Supporting v5 Expansion

| Domain | Primary Peer-Reviewed Sources | Key Empirical Findings & Reference Envelopes |
|---|---|---|
| **Pace Predictors & Plant Mechanics** | Worthington, King, & Ranson (2013), *J. Appl. Biomech.*; King, Worthington, & Ranson (2016), *J. Sports Sci.* | 4 predictors explain $R^2 = 0.74$ of ball speed ($n=20$ elite males): run-up speed (`5.83 ± 0.50 m/s`), straighter front knee at BR (`165° ± 14°`), upper-trunk flexion FFC→BR (`33° ± 7.5°`), delayed arm at FFC. Larger sagittal front-leg plant angle (`~30°–36°`) + heel strike moderates peak loading rate while maximizing horizontal braking impulse (`~4.5 BW`). |
| **Kinematic Sequence & Energy Flow** | Putnam (1993), *J. Biomech.*; Ferdinands, Kersting, & Marshall (2013), *J. Sports Sci.* | Proximal-to-distal summation of angular velocity & kinetic energy: **Pelvis axial $\omega$ $\to$ Thorax axial $\omega$ $\to$ Humerus circumduction $\omega$ $\to$ Forearm/Hand distal whip**. Active proximal segment deceleration prior to BR accelerates the adjacent distal segment. |
| **Phase Tempo & Sex Modifiers** | Felton, Lister, Worthington, & King (2018, *JSS*; 2019, *Sports Biomech.*); Felton, Worthington, & King (2020) | Elite male ($n=20$): BFC→FFC `191.9 ± 31.3 ms`, FFC→BR `103.3 ± 11.2 ms` (tempo ratio `1.86 ± 0.32`). Elite female ($n=20$): BFC→FFC `172.3 ± 27.6 ms`, FFC→BR `127.7 ± 11.0 ms` (tempo ratio `1.35 ± 0.25`, greater rotational reliance). Controlled COM vertical sink preserves release height. |
| **14-Segment Anthropometric COM** | Zatsiorsky et al. (1990); de Leva (1996), *J. Biomech.* 29(9), 1223–1230; Winter (2009) | Joint-center-adjusted male & female 14-segment mass fractions ($\sum m_s = 1.0000$) and longitudinal COM ratios $\lambda_s$ referenced directly to glenohumeral, elbow, wrist, hip, knee, and ankle joint centers. |
| **Crease Alignment, Head & Lead Arm** | Bartlett et al. (1996), *JSS*; Portus et al. (2001/2004), *Sports Biomech.*; Glazier (2010), *Sports Med.*; Ferdinands et al. (2010); Steffan Jones *Skill-Stability* | "Train-tracks" open stride corridor (`5–25 cm` lateral clearance) allows hip uncoiling; crossed-over foot plant (`< -5 cm`) blocks pelvis rotation and elevates SCR/lateral bend. High lead-arm gather at FFC followed by tight ribcage pull-down stabilizes head over front foot. |
| **3D Release Tunnel & Spell Fatigue** | Phillips, Portus, Davids, & Renshaw (2012), *JSMS*; Burnett, Elliott, & Marshall (1995); Portus et al. (2000) | Expert fast bowlers achieve tight 3D wrist release clustering (radial SD `~3–6 cm`) via upstream functional variability. Across multi-over spells, fatigue manifests as front-knee flexion yield, drop in release height, and drift in contralateral trunk tilt. |
| **ACWR & Governing-Body Workload** | Hulin et al. (2014), *BJSM*; Williams et al. (2017), *BJSM* (EWMA); Lolli et al. (2019) (uncoupled ACWR); Alway et al. (2019); ECB (2024); CA (2026) | Acute (7d) vs Chronic (28d) sweet spot `0.80–1.30`, high-spike alert `> 1.50` when conditioned (`chronic >= 30 balls/wk`). EWMA ($\lambda_a = 0.25, \lambda_c = 2/29$) weights recent load exponentially. Age-band spell/day/week caps + max 4 days/wk & rest-day rules mitigate adolescent LBSI exposure. |

---

## 2. Zero-Regression Modular Integration Strategy

To guarantee **0.0% numerical drift** on the existing 22-metric AR(1) synthetic benchmark (`validation/run_synthetic_benchmark.py`, which iterates `list(METRIC_SPECS.keys())` and consumes a fixed RNG stream) and **100% pass rate** on all 30 existing tests:

1. **`METRIC_SPECS` (`stage4_uncertainty.py`) remains untouched (exact 22 canonical keys).** `extract_fast_bowling_kinematics` and `quantify_and_gate_metrics` preserve bit-for-bit identical outputs and RNG consumption.
2. **New Module Stage (`src/bowling_biomech/stage3_v5_metrics.py`):**
   - Computes Modules 1–6 deterministically from `(joints_world_smoothed, markers_world_87, kin, isb, audit, cal, events, workload_ctx)` in `< 4.5 ms` CPU time (keeping total `recompute_delivery_cpu` well under the `< 35 ms` budget!).
   - Produces:
     - `v5_metrics: Dict[str, MetricEstimate]` (22 new uncertainty-quantified, observability-gated `MetricEstimate` scalars across Modules 1–6)
     - Strongly typed domain sub-models attached as optional fields on `BiomechanicalFactSheet`:
       - `kinematic_sequence: Optional[KinematicSequenceAnalysis] = None`
       - `com_braking: Optional[COMBrakingAnalysis] = None`
       - `head_stability: Optional[HeadStabilityAnalysis] = None`
       - `lead_arm: Optional[LeadArmAnalysis] = None`
       - `crease_tunnel: Optional[CreaseReleaseTunnelAnalysis] = None`
       - `tempo_rhythm: Optional[TempoRhythmAnalysis] = None`
3. **`CoachingStory` (`coaching_story/1`) & `CompareStory` Backward Compatibility:**
   - `CoachingStory.checkpoints` retains the exact 5-item tuple `["front_leg_brace", "trunk_whip", "side_bend", "shoulder_alignment", "runup_stride"]` (preserving `test_phase2_studio.py:779`).
   - `CompareStory.rows` retains the exact 5-item list (preserving `test_phase2_studio.py:895`).
   - `api_bowler_progress` retains `len(series) == 5` (preserving `test_phase2_studio.py:909`).
   - Additive fields (`v5_summary_cards`, `v5_detail_rows`, `compare.extended_rows`, `compare.sequence_delta`, `compare.tunnel_delta`, `telemetry.trails`, `telemetry.com_world_m`, `telemetry.sequence_series`) power the new Studio v5 UI without breaking a single existing contract.
4. **Numeric-Verifier Integration (`stage6_coach_copilot.py`):**
   - `_build_allowed_numbers(fact_sheet)` and `_suppressed_stems_for_fact_sheet(fact_sheet)` inspect `fact_sheet.all_metric_estimates()` (`{**fact_sheet.metrics, **fact_sheet.v5_metrics}`), registering explicit synonym stems for every v5 metric so suppressed v5 metrics can NEVER leak through Coach Copilot or `CoachingStory`.

---

## 3. Detailed Mathematical & Algorithmic Specifications (Modules 1–11)

### Module 1: Proximal-to-Distal Kinematic Sequence Engine

#### 1.1 Four-Segment Angular & Whip Velocity Signals
Let $T$ be the processed unique-frame count at effective frame rate $f_s = \texttt{audit.fps}$.

1. **Pelvis Axial Angular Velocity $\omega_{\text{pelvis}}(t)$ (`deg/s`):**
   - Source angle $\psi_{\text{pelvis}}(t)$: Use ISB 4-marker rigid-body `isb.pelvis_yaw_isb_deg` when `bml_movi_87` is available and `!isb.metrics["pelvis_alignment_bfc_isb_deg"].suppressed`; otherwise use Stage 3 `kin.series["pelvis_yaw_deg"]` (10 Hz dual-pass Winter Butterworth filtered, unwrapped in `[60°, 420°)` Portus convention where forward opening toward $+X$ yields $\dot{\psi} > 0$ for both right- and left-arm bowlers).
   - Quadratic Savitzky-Golay first derivative (`polyorder = 2, deriv = 1, delta = 1 / f_s`), window $W_{\text{axial}} = 2 \lfloor 0.0275 f_s \rceil + 1 \ge 5$ (~55 ms span).
2. **Thorax / Shoulder Axial Angular Velocity $\omega_{\text{thorax}}(t)$ (`deg/s`):**
   - Source angle $\psi_{\text{thorax}}(t)$: Use ISB 6-marker `isb.thorax_yaw_isb_deg` when available and non-suppressed; otherwise `kin.series["shoulder_yaw_deg"]`.
   - Same Savitzky-Golay derivative window $W_{\text{axial}}$.
3. **Bowling Upper-Arm Circumduction Angular Velocity $\omega_{\text{arm}}(t)$ (`deg/s`):**
   - Directly use Stage 3 `kin.series["bowling_arm_omega_dps"]` (Savitzky-Golay `~50 ms` derivative of unwrapped humerus angle $\phi_{\text{arm}}(t)$).
4. **Distal Forearm/Wrist Whip Velocity $\omega_{\text{wrist}}(t)$ (`deg/s`) & $v_{\text{wrist,rel}}(t)$ (`m/s`):**
   - Body-centric distal lever vector $\mathbf{d}_{\text{dist}}(t) = \mathbf{p}_{\text{b\_han}}(t) - \mathbf{p}_{\text{b\_sho}}(t)$ (falling back to $\mathbf{p}_{\text{b\_wri}} - \mathbf{p}_{\text{b\_sho}}$ if hand landmark not in frame), smoothed at $\min(20\text{ Hz}, 0.40 f_s)$.
   - Notice $\mathbf{d}_{\text{dist}}(t)$ is **strictly translation-invariant** (canceling 100% of camera panning drift).
   - Component-wise Savitzky-Golay derivative $\dot{\mathbf{d}}_{\text{dist}}(t)$ (`window = W_arm = max(5, 2*round(0.025*f_s)+1)`):
     $$v_{\text{wrist,rel}}(t) = \|\dot{\mathbf{d}}_{\text{dist}}(t)\| \quad (\text{m/s}), \qquad \omega_{\text{wrist}}(t) = \frac{180}{\pi} \frac{\|\mathbf{d}_{\text{dist}}(t) \times \dot{\mathbf{d}}_{\text{dist}}(t)\|}{\max(\|\mathbf{d}_{\text{dist}}(t)\|^2, 0.20^2)} \quad (\text{deg/s})$$
     Using the exact 3D cross-product angular speed $\|\mathbf{d} \times \dot{\mathbf{d}}\| / \|\mathbf{d}\|^2$ makes $\omega_{\text{wrist}}(t)$ **100% scale-invariant** (even if monocular stature scale is slightly off, $\|\mathbf{d}\|$ cancels in numerator and denominator!).

#### 1.2 Sub-Frame Peak Extraction, Sequence Verdict & Whip Gain Ratios
- **Search Windows:**
  - Proximal (`pelvis`, `thorax`): $i \in [\text{round}(i_{\text{BFC}} + 0.35(i_{\text{FFC}} - i_{\text{BFC}})),\; \min(T-1, \lceil i_{\text{BR}} + 0.025 f_s \rceil)]$.
  - Distal (`arm`, `wrist`): $i \in [i_{\text{FFC}},\; \min(T-1, \lceil i_{\text{BR}} + 0.020 f_s \rceil)]$.
- **Parabolic 3-Point Sub-Frame Vertex:** Around discrete argmax $k^*$ (when $0 < k^* < T-1$ and $y[k^*-1] - 2y[k^*] + y[k^*+1] < -10^{-6}$):
  $$\delta = \text{clip}\!\left(\frac{y[k^*-1] - y[k^*+1]}{2(y[k^*-1] - 2y[k^*] + y[k^*+1])},\; -0.5, +0.5\right), \quad f_{\text{peak}} = k^* + \delta, \quad \omega_{\text{peak}} = y[k^*] - \frac{1}{4}(y[k^*-1] - y[k^*+1])\delta$$
- **Peak Timings & Pairwise Inter-Segment Lags (`ms`):**
  - Relative to FFC: $\tau_s^{\text{FFC}} = \frac{f_{\text{peak},s} - f_{\text{FFC}}}{f_s} \times 1000\text{ ms}$; relative to BR: $\tau_s^{\text{BR}} = \frac{f_{\text{peak},s} - f_{\text{BR}}}{f_s} \times 1000\text{ ms}$.
  - Inter-segment lags:
    $$\Delta \tau_{P\to T} = \tau_{\text{thorax}}^{\text{FFC}} - \tau_{\text{pelvis}}^{\text{FFC}}, \quad \Delta \tau_{T\to A} = \tau_{\text{arm}}^{\text{FFC}} - \tau_{\text{thorax}}^{\text{FFC}}, \quad \Delta \tau_{A\to W} = \tau_{\text{wrist}}^{\text{FFC}} - \tau_{\text{arm}}^{\text{FFC}}$$
- **Sequence Classification (`sequence_verdict`):** Let temporal resolution floor $\epsilon_{\tau} = \max(500 / f_s,\; 6.0)\text{ ms}$:
  - `unmeasurable`: if `!events.is_complete_delivery` OR `audit.timebase_unreliable` OR $f_s < 90.0$ OR any of `(pelvis, thorax, arm)` peak estimates is suppressed.
  - `optimal_proximal_to_distal`: $\Delta \tau_{P\to T} > \epsilon_{\tau}$ AND $\Delta \tau_{T\to A} > \epsilon_{\tau}$ AND $\Delta \tau_{A\to W} \ge -\epsilon_{\tau}$.
  - `simultaneous_rotational`: $\Delta \tau_{P\to T} \ge -\epsilon_{\tau}$ AND $\Delta \tau_{T\to A} \ge -\epsilon_{\tau}$ AND $\Delta \tau_{A\to W} \ge -\epsilon_{\tau}$ (not strictly separated beyond $\epsilon_{\tau}$).
  - `out_of_order`: $\Delta \tau_{P\to T} < -\epsilon_{\tau}$ (`early_thorax_spin_out`) OR $\Delta \tau_{T\to A} < -\epsilon_{\tau}$ (`early_arm_rush`) OR $\Delta \tau_{A\to W} < -2\epsilon_{\tau}$.
- **Momentum Transfer & Deceleration Ratios:**
  - `seq_pelvis_peak_dps`, `seq_thorax_peak_dps`, `seq_arm_peak_dps` (= `peak_arm_angular_vel_dps`), `seq_wrist_whip_peak_dps`.
  - `seq_pelvis_to_thorax_lag_ms` ($\Delta \tau_{P\to T}$), `seq_thorax_to_arm_lag_ms` ($\Delta \tau_{T\to A}$).
  - `seq_whip_gain_arm_over_thorax` (`ratio`): $\omega_{\text{arm,peak}} / \max(\omega_{\text{thorax,peak}}, 100.0)$, clamped CI `[0.5, 6.0]`.
  - `seq_pelvis_braking_decel_pct` (`%`): $100 \times \text{clip}\!\left(\frac{\omega_{\text{pelvis,peak}} - \omega_{\text{pelvis}}(f_{\text{BR}})}{\max(\omega_{\text{pelvis,peak}}, 80.0)},\; 0.0, 1.0\right)$, measuring proximal pelvic deceleration into ball release (Putnam, 1993).
- **Uncertainty & Gating:**
  - Peak angular velocity pose $\sigma$: propagated via Stage 4 Savitzky-Golay derivative formula $\sigma_{\omega} = \frac{\sqrt{2(1 - \exp(-T_{\text{sg}}/0.08))}}{0.5 T_{\text{sg}}} \sigma_{\theta}(f_{\text{peak}})$.
  - Peak timing $\sigma_{\tau}$ (`ms`): $\sigma_{\tau,s} = \sqrt{\left(\frac{500}{f_s}\right)^2 + \left(1000 \cdot \frac{\sigma_{\omega,s}}{\max(|\ddot{\theta}_s(f_{\text{peak}})|, 2500^\circ/\text{s}^2)}\right)^2}$, capped at `45 ms`; lag uncertainty $\sigma_{\Delta\tau} = \sqrt{\sigma_{\tau,1}^2 + \sigma_{\tau,2}^2}$.
  - Hard-suppressed if $f_s < 90\text{ fps}$, `audit.timebase_unreliable`, or `!events.is_complete_delivery`.

---

### Module 2: 3D Center of Mass (COM) & Braking Impulse Analyzer

#### 2.1 Exact 14-Segment de Leva (1996) Linear Operator on SMPL-24
Using de Leva (1996, *J. Biomechanics* 29(9), Table 4) sex-specific body segment parameters (`sex` from `WorkloadContext.sex`, default `"male"`):

| # | Segment $s$ | Proximal Endpoint $\mathbf{p}_s^{\text{prox}}$ | Distal Endpoint $\mathbf{p}_s^{\text{dist}}$ | Male $m_s$ | Male $\lambda_s$ | Female $m_s$ | Female $\lambda_s$ |
|---|---|---|---|---|---|---|---|
| 1 | Head + Neck | `neck` | `head + 0.45*(head - neck)` | `0.0694` | `0.5002` | `0.0668` | `0.4841` |
| 2 | Trunk | `neck` | `0.5*(lhip + rhip)` | `0.4346` | `0.4310` | `0.4257` | `0.3782` |
| 3, 4 | L/R Upper Arm | `lsho` / `rsho` | `lelb` / `relb` | `0.0271` | `0.5772` | `0.0255` | `0.5754` |
| 5, 6 | L/R Forearm | `lelb` / `relb` | `lwri` / `rwri` | `0.0162` | `0.4574` | `0.0138` | `0.4559` |
| 7, 8 | L/R Hand | `lwri` / `rwri` | `lhan` / `rhan` | `0.0061` | `0.7900` | `0.0056` | `0.7474` |
| 9, 10 | L/R Thigh | `lhip` / `rhip` | `lkne` / `rkne` | `0.1416` | `0.4095` | `0.1478` | `0.3612` |
| 11, 12 | L/R Shank | `lkne` / `rkne` | `lank` / `rank` | `0.0433` | `0.4459` | `0.0481` | `0.4416` |
| 13, 14 | L/R Foot | `lank` / `rank` | `ltoe` / `rtoe` | `0.0137` | `0.4415` | `0.0129` | `0.4014` |

*(Check: Male $\sum m_s = 0.0694 + 0.4346 + 2(0.2480) = 1.0000$; Female normalized by exact sum $0.9999 \to 1.0000000000$.)*

Because every endpoint above is an affine combination of SMPL-24 joints, whole-body COM collapses to a **fixed 24-weight partition-of-unity vector** $\boldsymbol{\alpha} \in \mathbb{R}^{24}$ ($\alpha_j \ge 0, \sum_{j=0}^{23} \alpha_j \equiv 1$):
$$\mathbf{r}_{\text{COM}}(t) = \sum_{j=0}^{23} \alpha_j \,\mathbf{p}_j(t) \in \mathbb{R}^3$$
This guarantees exact translation equivariance $\mathbf{r}_{\text{COM}}(\mathbf{P} + \mathbf{c}) = \mathbf{r}_{\text{COM}}(\mathbf{P}) + \mathbf{c}$ and closed-form 3D world covariance under inter-joint depth correlation $\rho_{\text{body}} = 0.75$:
$$\Sigma_{\text{COM}}(t) = (1 - \rho_{\text{body}})\sum_{j=0}^{23}\alpha_j^2 \Sigma_{t,j} + \rho_{\text{body}} \left(\sum_{j=0}^{23}\alpha_j \Sigma_{t,j}^{1/2}\right)^2$$
Notice that under correlated monocular depth errors ($\rho_{\text{body}} = 0.75$), $\Sigma_{\text{COM}}$ does **not** falsely shrink by $1/\sqrt{24}$ (respecting the exact same statistical insight as our `stage2_isb_anatomy.py` $N_{\text{eff}}$ guardrail!).

#### 2.2 Extracted COM & Braking Metrics + Strict Gating
Smooth $\mathbf{r}_{\text{COM}}(t)$ at $\min(12\text{ Hz}, 0.32 f_s)$ and compute Savitzky-Golay velocity $\mathbf{v}_{\text{COM}}(t)$:

1. **`com_ffc_speed_mps` (`m/s`):** $v_{\text{COM},x}(t_{\text{FFC}})$.
   - *Gates:* `cal.is_metric_calibrated` AND `not cal.is_panning_camera` AND `not audit.timebase_unreliable` AND `sin_side >= 0.45` (or dual-fused). CI clamped `[0.0, 12.0]`.
2. **`com_braking_delta_vx_mps` (`m/s`):** $\Delta v_{\text{COM},x} = v_{\text{COM},x}(t_{\text{FFC}}) - v_{\text{COM},x}(t_{\text{BR}})$ (`+` = horizontal braking deceleration over `[FFC, BR]`; elite braced fast bowlers typically check `1.3–2.4 m/s` of horizontal COM velocity into rotational whip, Worthington et al. 2013; King et al. 2016).
   - *Gates:* Same as `com_ffc_speed_mps` plus $f_s \ge 60\text{ fps}$. CI clamped `[-3.0, 8.0]`.
3. **`com_braking_efficiency_pct` (`%`):**
   $$\eta_{\text{brake}} = 100 \times \frac{v_{\text{COM},x}(t_{\text{FFC}}) - v_{\text{COM},x}(t_{\text{BR}})}{\max(v_{\text{COM},x}(t_{\text{FFC}}),\; 1.5)}$$
   - *Gates:* Requires `not cal.is_panning_camera` AND `not audit.timebase_unreliable` AND $f_s \ge 60\text{ fps}$ AND ($v_{\text{COM},x}(t_{\text{FFC}}) \ge 2.0\text{ m/s}$) AND (`sin_side >= 0.45` or dual-fused). Because global metric scale cancels in a velocity ratio on a static camera, uncalibrated static side-on clips CAN report `com_braking_efficiency_pct` with `LOW`/`MODERATE` grade and `+4%` perspective-foreshortening $\sigma_{\text{scale}}$! CI clamped `[-25.0, 95.0]`.
4. **`com_vertical_sink_pct_stature` (`% stature`):**
   - Local ground-referenced COM heights: $h_{\text{COM,BFC}} = y_{\text{COM}}(t_{\text{BFC}}) - y_{\text{r\_ank}}(t_{\text{BFC}})$ and $h_{\text{COM,min}} = \min_{t \in [\text{FFC},\text{BR}]} (y_{\text{COM}}(t) - y_{\text{ground,local}})$.
   - $\text{Sink}_{\%} = 100 \times \frac{h_{\text{COM,BFC}} - h_{\text{COM,min}}}{\text{stature\_m}}$.
   - *Gates:* Dimensionless & local-ground-referenced! Works on static OR panning cameras (side, 45°, rear, front, dual) whenever upper+lower limb visibility $\ge 85\%$ (`metric_in_frame`) and `gravity_sigma_deg <= 6.5°`. CI clamped `[-15.0, 35.0]`.
5. **`front_leg_plant_angle_ffc_deg` (`deg`):**
   - King, Worthington, & Ranson (2016) sagittal COM-to-front-ankle braking strut angle at FFC:
     $$\theta_{\text{plant}} = \text{degrees}\!\left(\text{atan2}\!\big(x_{\text{f\_ank}}(t_{\text{FFC}}) - x_{\text{COM}}(t_{\text{FFC}}),\; \max(y_{\text{COM}}(t_{\text{FFC}}) - y_{\text{f\_ank}}(t_{\text{FFC}}), 0.25)\big)\right)$$
   - *Gates:* Single-instant (`FFC`) body-internal angle -> **immune to camera panning**! Requires `sin_side >= 0.45` (or dual-fused) and front ankle in frame at FFC (`vis_arr[i_ffc, f_ank_i]`). Elite cohort norm: `32.0° ± 4.5°` (King et al., 2016). CI clamped `[-15.0, 65.0]`.

---

### Module 3: Head Stability & Coronal Fall-Away Analyzer

All Module 3 metrics are **single-instant or body-referenced**, making them **100% immune to handheld camera panning** on rear (`0°`), front (`180°`), and dual-fused views:

1. **`head_coronal_angle_br_deg` (`deg`):**
   - Coronal inclination of the `pelv -> head` axis at BR toward the contralateral (non-bowling) side:
     $$\theta_{\text{head,BR}} = \text{degrees}\!\left(\text{atan2}\!\big(-s_{\text{arm}}(z_{\text{head}}(t_{\text{BR}}) - z_{\text{pelv}}(t_{\text{BR}})),\; \sqrt{(x_{\text{head}} - x_{\text{pelv}})^2 + \max(y_{\text{head}} - y_{\text{pelv}}, 0.1)^2}\big)\right)$$
   - *Gates:* Coronal view geometry (`cos_rear >= 0.45` or `is_dual_fused`), `head` & `pelv` in frame at BR.
2. **`head_fall_away_ffc_to_br_deg` (`deg`):**
   - Dynamic coronal head fall-away sweep across the braking/release window: $\Delta\theta_{\text{head}} = \theta_{\text{head}}(t_{\text{BR}}) - \theta_{\text{head}}(t_{\text{FFC}})$.
   - *Gates:* Coronal view geometry, `head` & `pelv` visible at both FFC and BR. Pose variance uses Stage 4 two-instant AR(1) difference covariance `Var(m2 - m1)`!
3. **`head_lateral_offset_from_ankle_br_pct_stature` (`% stature`) & `head_lateral_offset_from_ankle_br_cm` (`cm`):**
   - Contralateral horizontal displacement of `head` past the planted front-ankle vertical plumb line at BR:
     $$\Delta z_{\text{head-ank}} = -s_{\text{arm}} \cdot \big(z_{\text{head}}(t_{\text{BR}}) - z_{\text{f\_ank}}(t_{\text{BR}})\big)$$
     - `..._pct_stature` $= 100 \times \Delta z_{\text{head-ank}} / \text{stature\_m}$ (`0%` = eyes stacked directly above front ankle; `> +14% stature` / `> +25 cm` = heavy contralateral head fall-away outside the front foot; `< 0%` = head inside front-foot line).
     - `..._cm` $= 100 \times \Delta z_{\text{head-ank}}$ (reported in `cm` ONLY when `cal.is_metric_calibrated` or `cal.is_intrinsics_calibrated`, with `4%` scale $\sigma$ if intrinsics-only; suppressed on uncalibrated FOV).
   - *Gates:* Coronal view (`cos_rear >= 0.45` or dual-fused) AND both `head` and `f_ank` in frame at BR (so `indoor_nlf_fixture.npz` with cropped ankle cleanly suppresses `head_lateral_offset_from_ankle_*` while reporting `head_coronal_angle_br_deg` and `head_fall_away_ffc_to_br_deg`!).
4. **`head_vertical_drop_ffc_to_br_pct_stature` (`% stature`):**
   - $100 \times \frac{(y_{\text{head}}(t_{\text{FFC}}) - y_{\text{f\_ank}}(t_{\text{FFC}})) - (y_{\text{head}}(t_{\text{BR}}) - y_{\text{f\_ank}}(t_{\text{BR}}))}{\text{stature\_m}}$.

---

### Module 4: Lead (Non-Bowling) Arm Pull-Down Mechanics

Let $\mathbf{u}_{\text{nb}}(t) = \mathbf{p}_{\text{nb\_elb}}(t) - \mathbf{p}_{\text{nb\_sho}}(t)$ and $\theta_{\text{nb}}(t) = \text{degrees}(\text{atan2}(u_{\text{nb},y}, \hypot(u_{\text{nb},x}, u_{\text{nb},z})))$ (`kin.series["nb_arm_elev_deg"]`):

1. **`lead_arm_gather_peak_elev_deg` (`deg`):**
   - $\max_{t \in [\max(0, t_{\text{BFC}} - 40\text{ ms}),\; t_{\text{FFC}}]} \theta_{\text{nb}}(t)$.
   - *Gates:* `sin_side >= 0.45` or `cos_rear >= 0.65` or `is_dual_fused`; `nb_sho` & `nb_elb` in frame. CI clamped `[-60.0, 90.0]`.
2. **`lead_arm_early_drop_pre_ffc_deg` (`deg`):**
   - $\max(0.0,\; \texttt{lead\_arm\_gather\_peak\_elev\_deg} - \theta_{\text{nb}}(t_{\text{FFC}}))$.
   - Directly quantifies premature front-arm collapse *before* front-foot plant (`<= 18°` = strong high front-arm block held until FFC; `>= 38°` = premature front-arm drop leaking chest height prior to FFC). CI clamped `[0.0, 150.0]`.
3. **`lead_arm_pulldown_peak_vel_dps` (`deg/s`):**
   - Peak downward angular speed $\max_{t \in [\max(0, t_{\text{FFC}} - 20\text{ ms}),\; t_{\text{BR}}]} (-\dot{\theta}_{\text{nb}}(t))$ via quadratic Savitzky-Golay (`~50 ms` window).
   - *Gates:* $f_s \ge 90\text{ fps}$, `not audit.timebase_unreliable`, `sin_side >= 0.40` or `is_dual_fused`. CI clamped `[0.0, 2500.0]`.
4. **`lead_elbow_ribcage_flare_br_pct_stature` (`% stature`):**
   - Contralateral lateral flare of `nb_elb` outside the lead-hip/lead-shoulder torso line $\mathbf{p}_{\text{rib}} = \frac{1}{2}(\mathbf{p}_{\text{nb\_sho}} + \mathbf{p}_{\text{f\_hip}})$ at BR:
     $$\text{Flare}_{\%} = 100 \times \frac{\max\!\big(0.0,\; -s_{\text{arm}}(z_{\text{nb\_elb}}(t_{\text{BR}}) - z_{\text{rib}}(t_{\text{BR}}))\big)}{\text{stature\_m}}$$
     (`<= 8% stature` = tight ribcage pull-in; `>= 16% stature` = wide "flying" lead elbow spinning torso off-target).
   - *Gates:* Coronal view (`cos_rear >= 0.45` or `is_dual_fused`), `nb_elb`, `nb_sho`, `f_hip` in frame at BR. CI clamped `[0.0, 45.0]`.

---

### Module 5: Crease Alignment, Foot-Plant Geometry & 3D Release-Point Tunnel

1. **`foot_plant_corridor_offset_cm` (`cm`) & `stride_alignment_angle_deg` (`deg`):**
   - Let $\mathbf{p}_{\text{BFC}} = \mathbf{p}_{\text{r\_ank}}(t_{\text{BFC}})$ and $\mathbf{p}_{\text{FFC}} = \mathbf{p}_{\text{f\_ank}}(t_{\text{FFC}})$.
   - Anatomical lateral corridor clearance:
     $$\Delta z_{\text{open}} = -s_{\text{arm}} \cdot \big(z_{\text{FFC}} - z_{\text{BFC}}\big) \quad (\text{m})$$
     - **Sign Verification (Right-Arm):** Back (right) ankle at $z_{\text{BFC}} = +0.05\text{ m}$, Front (left) ankle on natural off-side track at $z_{\text{FFC}} = -0.10\text{ m}$ $\implies \Delta z_{\text{open}} = -(+1)(-0.10 - 0.05) = +0.15\text{ m} = +15.0\text{ cm}$ (**Open / Train-Tracks**). If front foot jumps across to $z_{\text{FFC}} = +0.17\text{ m}$ $\implies \Delta z_{\text{open}} = -12.0\text{ cm}$ (**Crossed-Over**).
     - **Sign Verification (Left-Arm Mirror):** Back (left) ankle at $z_{\text{BFC}} = -0.05\text{ m}$, Front (right) ankle on natural off-side track at $z_{\text{FFC}} = +0.10\text{ m}$ $\implies \Delta z_{\text{open}} = -(-1)(+0.10 - (-0.05)) = +0.15\text{ m} = +15.0\text{ cm}$ (**Exact Left/Right Symmetry!**).
   - `foot_plant_corridor_offset_cm` $= 100 \times \Delta z_{\text{open}}$ (`cm`): requires `cal.is_metric_calibrated` AND `not cal.is_panning_camera`, plus rear & front ankles in frame.
   - `stride_alignment_angle_deg` $= \text{degrees}(\text{atan2}(\Delta z_{\text{open}},\; \max(x_{\text{FFC}} - x_{\text{BFC}}, 0.35)))$: requires `not cal.is_panning_camera` AND `cal.is_metric_calibrated` AND `cal.heading_sigma_deg <= 6.0°`.
2. **`front_heel_crease_margin_cm` (`cm`):**
   - Using `bml_movi_87` heel marker (`lhee`/`rhee`) when available, else `f_ank_x - 0.045`:
     $$m_{\text{crease}} = 100 \times \big(1.22 - x_{\text{heel}}(t_{\text{FFC}})\big) \quad (\text{cm})$$
   - *Gates:* Strictly requires `cal.method == "mcc_law7_pnp"` AND `cal.is_metric_calibrated` AND `not cal.is_panning_camera` AND `sin_side >= 0.50` (never reported on `two_wicket_geometry` or uncalibrated clips; labeled *"Practice crease-efficiency margin, not an umpire no-ball call"*).
3. **3D Release-Point Tunnel Coordinates:**
   - Release point $\mathbf{p}_{\text{rel}} = \mathbf{p}_{\text{b\_han}}(t_{\text{BR}})$ (or $\mathbf{p}_{\text{b\_wri}}(t_{\text{BR}}) + 0.08\,\hat{\mathbf{u}}_{\text{forearm}}$ if hand cropped):
     - `release_height_pct_stature` (`% stature`): $100 \times \frac{\texttt{release\_height\_m}}{\text{stature\_m}}$ (typical `104%–116% stature`).
     - `release_lateral_from_ankle_m` (`m`): $s_{\text{arm}} \cdot (z_{\text{rel}} - z_{\text{f\_ank}}(t_{\text{BR}}))$ (`0.00–0.18 m` = high over-the-top ear-skimming slot; `0.35–0.65 m` = round-arm / slingy slot; immune to panning!).
     - `release_lateral_from_stump_m` (`m`): $z_{\text{rel}}$ in Pitch World relative to middle stump $(0,0,0)$ — ONLY when `(cal.method in ("mcc_law7_pnp", "two_wicket_geometry"))` AND `not cal.is_panning_camera` (or pan-verified with $\sigma_z \le 0.12\text{ m}$).
     - `release_extension_past_crease_m` (`m`): $x_{\text{rel}} - 1.22$ — ONLY when `cal.method == "mcc_law7_pnp"` AND `not cal.is_panning_camera`.

---

### Module 6: Delivery Tempo & Rhythm Fingerprint

1. **`tempo_stride_duration_ms` (`ms`):** $(t_{\text{FFC}} - t_{\text{BFC}}) \times 1000$, $\sigma = \hypot(\sigma_{t,\text{BFC}}, \sigma_{t,\text{FFC}})$. Cohort norm: `191.9 ± 31.3 ms` male / `172.3 ± 27.6 ms` female (Felton et al., 2019).
2. **`tempo_braking_to_br_ms` (`ms`):** $(t_{\text{BR}} - t_{\text{FFC}}) \times 1000$, $\sigma = \hypot(\sigma_{t,\text{FFC}}, \sigma_{t,\text{BR}})$. Cohort norm: `103.3 ± 11.2 ms` male / `127.7 ± 11.0 ms` female (Felton et al., 2019).
3. **`tempo_uah_to_br_whip_ms` (`ms`):** $(t_{\text{BR}} - t_{\text{UAH}}) \times 1000$ when `events.uah` is present; gated on $f_s \ge 90\text{ fps}$.
4. **`tempo_gather_to_whip_ratio` (`ratio`):**
   $$R_{\text{tempo}} = \frac{t_{\text{FFC}} - t_{\text{BFC}}}{\max(t_{\text{BR}} - t_{\text{FFC}},\; 0.025)}, \qquad \sigma_R = R_{\text{tempo}} \sqrt{\left(\frac{\sigma_{\text{stride}}}{\Delta t_{\text{stride}}}\right)^2 + \left(\frac{\sigma_{\text{brake}}}{\Delta t_{\text{brake}}}\right)^2}$$
   - Cohort norm: `1.86 ± 0.32` (male) / `1.35 ± 0.25` (female), Felton et al. (2019).
- **Hard Gates:** All 4 tempo metrics are suppressed whenever `!events.is_complete_delivery`, `audit.timebase_unreliable`, or anchor event confidence `< 0.30`.

---

### Module 7: Stroboscopic Chronophotography & Multi-Joint Trajectory Trails

#### 7.1 Server-Side Stroboscopic Multi-Exposure Compositor (`src/bowling_biomech/stage6_stroboscopic.py`)
- **Endpoint:** `GET /api/deliveries/{delivery_id}/stroboscopic.jpg?w=1280&skeleton=false&trails=true`
- **Selected Event Phases (up to 5):**
  1. `pre_bfc` ($\max(0, i_{\text{BFC}} - \text{round}(0.14 f_s))$, opacity $\alpha_1 = 0.38$) — only if spatially separated from `BFC` by $\ge 45\text{ px}$ in 2D pelvis/torso X! *(Crucial insight: on a rear-on `0°` or front-on `180°` clip like Dale Steyn or Indoor, the bowler moves toward/away from the camera with overlapping 2D silhouettes! Let's handle both side-on and rear/front clips cleanly!)*
  2. **View-Adaptive Layout Mode:**
     - **Spatial Progression Mode (`panoramic_composite`):** Used when 2D horizontal pelvis travel across `[BFC..BR]` exceeds `14%` of frame width (or after background homography stabilization on panning side-on video):
       - If camera panned (`cal.is_panning_camera`) on real video, fit RANSAC homography $H_{k \to i_{\text{BR}}}$ on bowler-masked ORB/KLT background keypoints; if median residual $\le 3.0\text{ px}$, warp keyframes onto $i_{\text{BR}}$'s canvas.
       - Compute bowler-masked median background plate $\mathbf{B}(x, y) = \text{nanmedian}_t \mathbf{I}_{\text{masked}}(t, x, y)$ across 11 evenly spaced frames where dilated 2D limb-capsule hulls ($r_{\text{dilate}} = 1.45 \times \text{capsule\_width}$) are masked out, filling any residual NaN pockets via `cv2.inpaint(..., 5, cv2.INPAINT_TELEA)`.
       - Extract foreground alpha matte $M_k(x,y) \in [0, 1]$ at each phase $k \in \{\text{Pre-BFC}, \text{BFC}, \text{FFC}, \text{UAH}, \text{BR}\}$ by combining luminance/chroma background difference $\|\mathbf{I}_k - \mathbf{B}\|_2$ sigmoid threshold with feathered 2D anatomical limb-capsule prior (or direct `_draw_capsule` silhouette rendering on synthetic studio proxies!).
       - Composite back-to-front (`0.38 -> 0.54 -> 0.72 -> 0.86 -> 1.00`) + draw smooth anti-aliased glowing bowling-wrist whip arc & COM path when `trails=true`.
     - **Coronal Sequence Filmstrip Mode (`coronal_sequence_strip`):** Automatically activated when 2D horizontal pelvis travel is `< 14%` of frame width (such as rear-on Dale Steyn `uiZN5FHWjJk.mp4` or front-on Indoor `ZpBNq8cIRoY.mp4`, where overlapping 5 silhouettes in place would turn the bowler into a muddy blob!). Crops a tight, aspect-locked vertical athlete corridor centered on the bowler across `[BFC, FFC, UAH, BR]` and stitches a seamless 4-column high-fashion chronophotographic sequence strip with continuous horizontal release-height & ground reference lines + optional wrist whip arc!
     - Disk-cached under `ddir / "frames" / f"strobo_w{w}_sk{int(skeleton)}_tr{int(trails)}.jpg"` with `Cache-Control: public, max-age=3600`.

#### 7.2 Precomputed 2D & 3D Multi-Joint Trails (`telemetry.json` v2.1)
Added additively to `build_telemetry_v2` (`schema_version` remains `"2.0"` for existing test assertion `assert det["telemetry"]["schema_version"] == "2.0"`, with `"telemetry_minor_version": "2.1"`!):
- `com_world_m`: `(T, 3)` ground-zeroed 3D COM path (mm precision, 3 decimals).
- `com_2d_px`: `(T, 2)` 2D projected COM pixels (`K @ (R @ com + t)`).
- `trails`: Window `[max(0, i_bfc - round(0.15*fps)) .. min(T-1, i_br + round(0.18*fps))]` containing `frame_indices`, `wrist_speed_mps` (for color-ramped whip ribbon), and `2d_px` / `3d_m` coordinates for `bowling_wrist`, `bowling_elbow`, `lead_wrist`, `head`, `com`, `front_ankle`, `rear_ankle`.
- `sequence_series`: `{"pelvis_omega_dps": [...], "thorax_omega_dps": [...], "arm_omega_dps": [...], "wrist_whip_dps": [...], "com_vx_mps": [...]}` aligned to `time_rel_ffc_ms`.

---

### Module 8: Peer-Reviewed Pro Archetype 3D Reference Library

- **Module:** `src/bowling_biomech/archetypes.py`
- **Endpoint:** `GET /api/archetypes?arm=right|left` (cached in-memory per `arm`; responds in `< 2 ms` after first call).
- **Four Synthesized Reference Models (`fps = 120.0`, `duration_s = 1.05` -> compact 126-frame FFC-aligned 3D skeleton + sequence + COM curves):**

| Archetype ID | Title | Action Family | Key Literature-Grounded Parameters (`SyntheticDeliveryParams`) | Primary Citations |
|---|---|---|---|---|
| `side_on_express` | **Side-On Braced Express** | `side_on` | `pelv_bfc=192°, sho_bfc=188°, scr=10°, v_run=6.1 m/s, fk_ffc=162°, fk_br=173°, tf_delta=35°, delay_arm=152°, lat_br=36°, stature=1.86 m` | Portus et al. (2004); Worthington et al. (2013) |
| `semi_open_all_round` | **Semi-Open Balanced Rhythm** | `semi_open` | `pelv_bfc=224°, sho_bfc=220°, scr=14°, v_run=5.8 m/s, fk_ffc=160°, fk_br=168°, tf_delta=32°, delay_arm=145°, lat_br=35°, stature=1.85 m` | Portus et al. (2004); Felton et al. (2019) |
| `front_on_power` | **Front-On Rotational Power** | `front_on` | `pelv_bfc=256°, sho_bfc=252°, scr=12°, v_run=5.7 m/s, fk_ffc=156°, fk_br=164°, tf_delta=30°, delay_arm=138°, lat_br=34°, stature=1.84 m` | Portus et al. (2004); Felton et al. (2018) |
| `high_arm_bounce` | **Tall High-Release Bounce** | `semi_open` | `pelv_bfc=216°, sho_bfc=212°, scr=9°, v_run=5.5 m/s, stride=1.44 m, fk_ffc=165°, fk_br=174°, tf_ffc=8°, tf_br=36°, lat_br=31°, br_phi=3°, stature=1.95 m` | Worthington et al. (2013); King et al. (2016) |

- **Explicit Non-Template Disclaimer Included on Every Archetype Object:**
  `"Literature-synthesized 3D reference model illustrating published biomechanical principles (Glazier, 2010: elite bowlers self-organize around individual anthropometry — never treat a reference archetype as a mandatory 1-size-fits-all template or 0-100 similarity target)."`

---

### Module 9: Full ACWR Bowling Workload & Wellness Engine

#### 9.1 SQLite Schema Migration (`server/db.py`)
Idempotent `PRAGMA table_info(workload_logs)` migration adds 4 columns to `workload_logs`:
```sql
ALTER TABLE workload_logs ADD COLUMN rpe INTEGER NOT NULL DEFAULT 6;
ALTER TABLE workload_logs ADD COLUMN max_spell_balls INTEGER;
ALTER TABLE workload_logs ADD COLUMN soreness_lumbar INTEGER;
ALTER TABLE workload_logs ADD COLUMN fatigue_level INTEGER;
```
Plus `db.delete_workload_log(bowler_id: str, log_id: str) -> bool`.

#### 9.2 Daily Time-Series & Three Peer-Reviewed ACWR Formulations (`src/bowling_biomech/workload_acwr.py`)
Construct a dense daily vector $L[d]$ (`balls` bowled on calendar day $d$, summing multiple sessions on the same date) over $d \in [d_{\text{today}} - 55,\; d_{\text{today}}]$ (56 days so 28-day chronic & EWMA warm up cleanly):

1. **7-Day Acute Load (`balls`):** $A_{\text{7d}}(t) = \sum_{k=0}^{6} L[t - k]$.
2. **28-Day Coupled Chronic Weekly Average (`balls/wk`, Hulin et al., 2014):**
   $$C_{\text{coupled}}(t) = \frac{1}{4}\sum_{k=0}^{27} L[t - k], \qquad \text{ACWR}_{\text{coupled}}(t) = \begin{cases} A_{\text{7d}}(t) / C_{\text{coupled}}(t), & C_{\text{coupled}}(t) \ge 30.0 \\ \texttt{None}\;(\text{low\_chronic\_base}), & C_{\text{coupled}}(t) < 30.0 \end{cases}$$
3. **21-Day Uncoupled Chronic Weekly Average (`balls/wk`, Lolli et al., 2019):**
   $$C_{\text{uncoupled}}(t) = \frac{1}{3}\sum_{k=7}^{27} L[t - k], \qquad \text{ACWR}_{\text{uncoupled}}(t) = \begin{cases} A_{\text{7d}}(t) / C_{\text{uncoupled}}(t), & C_{\text{uncoupled}}(t) \ge 30.0 \\ \texttt{None}, & \text{otherwise} \end{cases}$$
4. **Exponentially Weighted Moving Average (`EWMA ACWR`, Williams et al., 2017):**
   - Decay constants: $\lambda_a = \frac{2}{7 + 1} = 0.25$, $\lambda_c = \frac{2}{28 + 1} = \frac{2}{29} \approx 0.0689655$.
   - Initialized at $t_0 = d_{\text{today}} - 55$ with $E_a(t_0) = E_c(t_0) = \frac{1}{7} C_{\text{initial}}$ (mean daily load of earliest logged 14-day window, preventing artificial day-0 startup transients):
     $$E_a(t) = \lambda_a L[t] + (1 - \lambda_a) E_a(t-1), \qquad E_c(t) = \lambda_c L[t] + (1 - \lambda_c) E_c(t-1)$$
     $$\text{ACWR}_{\text{EWMA}}(t) = \begin{cases} E_a(t) / E_c(t), & 7 E_c(t) \ge 30.0 \\ \texttt{None}, & \text{otherwise} \end{cases}$$
5. **Zone Classification & Governing-Body Directives:**
   - Primary ratio $r = \text{ACWR}_{\text{EWMA}}(t_{\text{today}}) \text{ or } \text{ACWR}_{\text{coupled}}(t_{\text{today}})$:
     - `low_chronic_base`: $C_{\text{coupled}} < 30\text{ balls/wk}$ ("Build chronic base gradually by <=15–20% per week before high-volume spells; Alway et al. 2019").
     - `underload`: $r < 0.80$
     - `sweet_spot`: $0.80 \le r \le 1.30$ (Hulin et al., 2014)
     - `caution_elevated`: $1.30 < r \le 1.50$
     - `high_spike_alert`: $r > 1.50$
   - **ECB (2024) & Cricket Australia (2026) Structural Checks (`alerts: List[WorkloadAlert]`):**
     - `weekly_cap_exceeded`: $A_{\text{7d}} > \texttt{max\_balls\_per\_week}$
     - `daily_cap_exceeded`: any day in last 7d $> \texttt{max\_balls\_per\_day}$
     - `spell_cap_exceeded`: any `max_spell_balls` in last 7d $> \texttt{max\_balls\_per\_spell}$
     - `frequency_exceeded`: bowling days in last 7d $> 4$ (or $> 3$ for `u13/u15/u17`)
     - `consecutive_days_alert`: consecutive bowling days $\ge 2$ for `u13/u15/u17` or $\ge 3$ for `u19/adult`
     - `lumbar_stiffness_logged`: any `soreness_lumbar >= 3` in last 7d $\implies$ non-diagnostic welfare alert recommending pause & sports-physiotherapy review.

#### 9.3 Endpoints
- `GET /api/bowlers/{bowler_id}/workload` (retains `bowler`, `logs`, `deliveries_last_7d`, `deliveries_last_28d`, `guidance`, PLUS adds `acwr_summary`, `daily_series_28d`, `directive_checks`, `alerts`, `medical_disclaimer`).
- `POST /api/bowlers/{bowler_id}/workload` (accepts optional `rpe`, `max_spell_balls`, `soreness_lumbar`, `fatigue_level`).
- `DELETE /api/bowlers/{bowler_id}/workload/{log_id}`.

---

### Module 10: Spell / Over Consistency & Fatigue Analytics

- **Endpoint:** `GET /api/bowlers/{bowler_id}/spell-analytics?session_id={optional}`
- **Computations (`src/bowling_biomech/spell_analytics.py`):**
  1. **Release-Point Bullseye ("Release Tunnel"):**
     - Extracts all completed deliveries for the bowler (or specific `session_id`).
     - Supports two coordinate modes so **both calibrated and uncalibrated spells** work honestly:
       - `world_stump_m`: $(Z_{\text{rel}}, Y_{\text{rel}})$ relative to middle stump `(0, 0, 0)` for static MCC/two-wicket calibrated deliveries.
       - `body_corridor_m`: $(s_{\text{arm}}(z_{\text{rel}} - z_{\text{f\_ank,BR}}),\; \texttt{release\_height\_m})$ relative to planted front ankle & local ground (works across static AND panning clips!).
     - Computes centroid $(\bar{z}, \bar{y})$, raw radial SD, **measurement-variance-debiased spatial SD** $\sigma_{\text{true}} = \sqrt{\max(s_{\text{obs}}^2 - \overline{\sigma_i^2},\; 0)}$, and 95% confidence ellipse `(semi_major_m, semi_minor_m, angle_deg)`.
  2. **Measurement-Debiased Repeatability CV% Table:**
     - Across `front_knee_br_deg`, `trunk_flexion_delta_deg`, `thorax_lat_flex_br_deg`, `shoulder_counter_rotation_deg`, `stride_length_pct_stature`, `release_height_m`, `peak_arm_angular_vel_dps`, `tempo_gather_to_whip_ratio`:
       - Reports `n_balls`, `mean`, `raw_sd`, `mean_measurement_sigma`, `debiased_sd`, `debiased_cv_pct`, and `grade` (`tight`, `moderate`, `variable`, `measurement_limited`).
  3. **Inverse-Variance Weighted Fatigue Drift (`Ball 1 -> N`):**
     - For each tracked metric with $N \ge 3$ observable deliveries from compatible camera views, computes weighted least-squares slope $\hat{\beta}_1$ (`unit / ball`) with weights $w_i = 1 / \max(\sigma_i^2, 10^{-4})$, total spell shift $\Delta_{\text{spell}} = \hat{\beta}_1 (N - 1)$, and analytical standard error $\text{SE}(\hat{\beta}_1) = (\sum w_i (i - \bar{i}_w)^2)^{-1/2}$.
     - Drift status: `"verified_drift"` ONLY when $|\hat{\beta}_1| / \text{SE}(\hat{\beta}_1) \ge 1.96$; otherwise `"within_measurement_noise"`.

---

### Module 11: Dual Real-Video + Realistic 4-Ball Spell Demo Seeding

#### 11.1 Empirical Verification of `ZpBNq8cIRoY.mp4` vs `indoor_nlf_fixture.npz`
Direct inspection confirmed **exact 1-to-1 frame correspondence**:
- `data/public_clips/ZpBNq8cIRoY.mp4`: `1280 × 720`, `30.0 fps` container, `264` frames (`0..263`), `held_frame_ratio = 0.304`, `longest_held_run_frames = 24` (intro freeze frames `30..53` + 24-in-30 cadence).
- `tests/fixtures/indoor_nlf_fixture.npz`: `joints3d_cam_m` shape `(264, 24, 3)`, `width = 1280, height = 720`, `azimuth_deg = 177.62°` (`front_behind_batter`), BFC=`142`, FFC=`168`, UAH=`175`, BR=`179` (`rwri_2d = [680.6, 53.6]`, `head_2d = [888.3, 318.7]`, `pelv_2d = [792.3, 533.0]`).

#### 11.2 First-Boot Seeded Inventory (`seed_demo_session` in `server/recompute.py`)
To preserve **100% pass rate on all existing `test_phase2_studio.py` assertions**:
- Keep `len(bowlers) == 1` (`bwl_demo`) and `len(sessions) == 1` (`ses_demo`).
- Keep `.demo_seed_version` writing `"4.0"` (so `test_phase2_studio.py:943` passes verbatim!), and track v5 asset upgrades via a companion marker `demo_ses_dir / ".demo_v5_version"` = `"5.0"`.
- Keep `samples[0]` on `/api/overview` = `"del_real_steyn"` (`test_phase2_studio.py:760`), followed immediately by `"del_real_indoor"`, `"del_demo_side_240"` (Ball 1), `"del_demo_side_b4"` (Ball 4 Fatigue Limiter), `"del_demo_rear_240"` (Rear Dual-Sync Mate), and `"del_real_mcgrath"`.
- **Seeded Deliveries in `ses_demo` (8 total):**
  1. `del_real_steyn` — *"Dale Steyn — Real Rear-On Net Video"* (`uiZN5FHWjJk.mp4`, 63 frames @ `25.175 fps`, Law-8 Two-Wicket calibrated).
  2. `del_real_indoor` — *"Indoor Net Slow-Mo — Real Front-On Video"* (`ZpBNq8cIRoY.mp4`, 264 frames, real front-on coronal/head/lead-arm tracking + honest edited-timebase & cropped-ankle gating showcase).
  3. `del_demo_side_240` — *"Spell Ball 1 — Side-On 240 fps (Fresh Express)"* (`fk_br=171°, tf_delta=33°, delay_arm=149°, lat_br=36°, scr=18°, v_run=5.95 m/s` + synthetic `bml_movi_87` markers).
  4. `del_demo_side_b2` — *"Spell Ball 2 — Side-On 240 fps (Settled Rhythm)"* (`fk_br=165°, tf_delta=31°, delay_arm=144°, lat_br=40°, scr=22°, v_run=5.85 m/s` + `bml_movi_87` markers).
  5. `del_demo_side_b3` — *"Spell Ball 3 — Side-On 240 fps (Mid-Spell)"* (`fk_br=158°, tf_delta=29°, delay_arm=138°, lat_br=42°, scr=24°, v_run=5.72 m/s` + `bml_movi_87` markers).
  6. `del_demo_side_b4` — *"Spell Ball 4 — Side-On 240 fps (Late-Spell Fatigue)"* (`fk_br=145°, fk_ffc=159°` -> triggers verified Front-Knee Collapse limiter & Corrective Drill, `tf_delta=26°, delay_arm=131°, lat_br=45°, scr=27°, v_run=5.55 m/s` + `bml_movi_87` markers).
  7. `del_demo_rear_240` — *"Spell Ball 1 — Rear 240 fps (Dual-Sync Mate)"* (`0°` rear mate to `del_demo_side_240` + `bml_movi_87` markers).
  8. `del_real_mcgrath` — *"Glenn McGrath — Incomplete Clip Guardrail Example"* (`is_complete_delivery = False`).
- **Seeded 28-Day Workload & Wellness History on `bwl_demo`:**
  - 12 logged sessions across `today - 27d .. today - 1d`:
    - Days `-26, -24, -21`: `30, 42, 36` balls (`108/wk`)
    - Days `-19, -17, -14`: `30, 36, 42` balls (`108/wk`)
    - Days `-12, -10, -8`: `24, 42, 36` balls (`102/wk`)
    - Days `-5, -3, -1`: `36, 42, 30` balls (`108/wk` -> satisfies `deliveries_last_7d >= 100` in `test_phase2_studio.py:493`, Coupled ACWR = `1.01`, EWMA ACWR = `1.03` in Hulin `0.8–1.3` sweet spot, `rpe = 6..8`, `soreness_lumbar = 1..2`).

---

## 4. Exact Pydantic v2 Schema Additions (`src/bowling_biomech/schemas.py`)

All new fields on existing root contracts (`BiomechanicalFactSheet`, `CoachingStory`, `CompareStory`) default to `None` or `Field(default_factory=...)` so stored v2–v4 JSON payloads validate without migration errors:

```python
SequenceVerdict = Literal[
    "optimal_proximal_to_distal",
    "simultaneous_rotational",
    "out_of_order",
    "unmeasurable",
]


class SequenceSegmentPeak(BaseModel):
    segment: Literal["pelvis", "thorax", "bowling_arm", "wrist_whip"]
    label_plain: str
    peak_dps: MetricEstimate
    peak_time_rel_ffc_ms: Optional[float] = None
    peak_time_rel_br_ms: Optional[float] = None
    timing_sigma_ms: Optional[float] = None


class KinematicSequenceAnalysis(BaseModel):
    verdict: SequenceVerdict
    verdict_label_plain: str
    summary_plain: str
    segments: List[SequenceSegmentPeak] = Field(default_factory=list)
    pelvis_to_thorax_lag_ms: MetricEstimate
    thorax_to_arm_lag_ms: MetricEstimate
    arm_to_wrist_lag_ms: MetricEstimate
    whip_gain_arm_over_thorax: MetricEstimate
    pelvis_braking_decel_pct: MetricEstimate


class COMBrakingAnalysis(BaseModel):
    model_reference: str = "de Leva (1996) 14-segment anthropometric COM (correlated-depth N_eff covariance)"
    summary_plain: str
    com_ffc_speed_mps: MetricEstimate
    com_br_speed_mps: MetricEstimate
    com_braking_delta_vx_mps: MetricEstimate
    com_braking_efficiency_pct: MetricEstimate
    com_vertical_sink_pct_stature: MetricEstimate
    front_leg_plant_angle_ffc_deg: MetricEstimate


class HeadStabilityAnalysis(BaseModel):
    summary_plain: str
    head_coronal_angle_br_deg: MetricEstimate
    head_fall_away_ffc_to_br_deg: MetricEstimate
    head_lateral_offset_from_ankle_br_pct_stature: MetricEstimate
    head_lateral_offset_from_ankle_br_cm: MetricEstimate
    head_vertical_drop_ffc_to_br_pct_stature: MetricEstimate


class LeadArmAnalysis(BaseModel):
    summary_plain: str
    lead_arm_gather_peak_elev_deg: MetricEstimate
    lead_arm_early_drop_pre_ffc_deg: MetricEstimate
    lead_arm_pulldown_peak_vel_dps: MetricEstimate
    lead_elbow_ribcage_flare_br_pct_stature: MetricEstimate


class CreaseReleaseTunnelAnalysis(BaseModel):
    corridor_verdict: Literal["train_tracks_open", "tightrope_inline", "crossed_over", "unmeasurable"]
    corridor_label_plain: str
    summary_plain: str
    foot_plant_corridor_offset_cm: MetricEstimate
    stride_alignment_angle_deg: MetricEstimate
    front_heel_crease_margin_cm: MetricEstimate
    release_height_pct_stature: MetricEstimate
    release_lateral_from_ankle_m: MetricEstimate
    release_lateral_from_stump_m: MetricEstimate
    release_extension_past_crease_m: MetricEstimate


class TempoPhaseSegment(BaseModel):
    id: Literal["stride_bfc_ffc", "early_braking_ffc_uah", "whip_uah_br", "braking_ffc_br", "follow_through"]
    label_plain: str
    start_rel_ffc_ms: float
    end_rel_ffc_ms: float
    duration_ms: float
    uncertainty_ms: float
    reference_range_ms: Optional[Tuple[float, float]] = None


class TempoRhythmAnalysis(BaseModel):
    summary_plain: str
    tempo_stride_duration_ms: MetricEstimate
    tempo_braking_to_br_ms: MetricEstimate
    tempo_uah_to_br_whip_ms: MetricEstimate
    tempo_gather_to_whip_ratio: MetricEstimate
    waterfall_phases: List[TempoPhaseSegment] = Field(default_factory=list)
```

Additions on `BiomechanicalFactSheet`:
```python
    v5_metrics: Dict[str, MetricEstimate] = Field(default_factory=dict)
    kinematic_sequence: Optional[KinematicSequenceAnalysis] = None
    com_braking: Optional[COMBrakingAnalysis] = None
    head_stability: Optional[HeadStabilityAnalysis] = None
    lead_arm: Optional[LeadArmAnalysis] = None
    crease_tunnel: Optional[CreaseReleaseTunnelAnalysis] = None
    tempo_rhythm: Optional[TempoRhythmAnalysis] = None
```

Additions on `CoachingStory` & `CompareStory`:
```python
class V5InsightCard(BaseModel):
    id: Literal["kinematic_sequence", "com_braking", "head_stability", "lead_arm", "crease_tunnel", "tempo_rhythm"]
    title: str
    badge_label: str
    status: CheckpointStatus
    headline_value_text: Optional[str] = None
    headline_range_text: Optional[str] = None
    plain_summary: str
    unlock_tip: Optional[str] = None
    moment_id: Optional[MomentId] = None
    frame_idx: Optional[int] = None

# Added to CoachingStory:
    v5_insights: List[V5InsightCard] = Field(default_factory=list)
    v5_detail_rows: List[StoryMetricRow] = Field(default_factory=list)
    stroboscopic_url: Optional[str] = None

# Added to CompareStory:
    extended_rows: List[CompareExtendedRow] = Field(default_factory=list)
```

---

## 5. REST API Contract Summary (New & Upgraded Endpoints)

| Method & Route | Purpose | Key Response Payload Fields |
|---|---|---|
| `GET /api/deliveries/{id}` | Existing delivery detail (upgraded additively) | Existing fields + `fact_sheet.{v5_metrics, kinematic_sequence, com_braking, head_stability, lead_arm, crease_tunnel, tempo_rhythm}`, `story.{v5_insights, v5_detail_rows, stroboscopic_url}`, `telemetry.{com_world_m, com_2d_px, trails, sequence_series}` |
| `GET /api/deliveries/{id}/stroboscopic.jpg` | Real-video / studio multi-exposure chronophotography | `image/jpeg` (`?w=1280&skeleton=false&trails=true`), view-adaptive (`panoramic_composite` on side-on vs `coronal_sequence_strip` on rear/front clips), disk-cached |
| `GET /api/archetypes` | 4 literature-grounded 3D fast-bowling archetypes | `?arm=right\|left` -> `{archetypes: [{id, title, action_class, literature_source, disclaimer, stature_m, time_rel_ffc_ms, joints3d_world_m, com_world_m, kin_series, sequence_series, metric_envelopes}]}` |
| `GET /api/bowlers/{id}/workload` | Upgraded ACWR & governing-body workload engine | Existing keys (`bowler, logs, deliveries_last_7d, deliveries_last_28d, guidance`) + `acwr_summary` (`acute_7d, chronic_coupled_wk, chronic_uncoupled_wk, acwr_coupled, acwr_uncoupled, acwr_ewma, zone, zone_label_plain`), `daily_series_28d`, `directive_checks`, `alerts`, `medical_disclaimer` |
| `POST /api/bowlers/{id}/workload` | Add session log with RPE & wellness | Accepts `{log_date, deliveries_count, session_type, notes, rpe?, max_spell_balls?, soreness_lumbar?, fatigue_level?}` |
| `DELETE /api/bowlers/{id}/workload/{log_id}` | Delete a workload entry | `{deleted: bool, log_id: str}` |
| `GET /api/bowlers/{id}/spell-analytics` | Release tunnel bullseye, CV% repeatability & fatigue drift | `?session_id=` -> `{completed_count, release_tunnel: {mode, points, centroid, raw_radial_sd_m, debiased_radial_sd_m, ellipse_95}, repeatability_table: [...], fatigue_drift: [...]}` |
| `GET /api/compare?a={id}&b={id}` | Upgraded 2-ball comparison | Existing 5 `rows` + `extended_rows` covering tempo, kinematic sequence lag, plant angle, COM braking, head fall-away, lead-arm tuck, and release height |

---

## 6. Verification, Non-Circular Unit Tests & Metrology Benchmark Plan

### 6.1 Non-Circular Unit & Regression Suite (`tests/test_v5_biomechanics.py`)
1. **Module 1 (Kinematic Sequence Non-Circular Synthetic Chirps):**
   - Construct exact synthetic sinusoidal angular trajectories where pelvis peaks at `-55 ms`, thorax at `-30 ms`, arm at `-6 ms`, and wrist at `0 ms` relative to BR for BOTH `BowlingArm.RIGHT` and `BowlingArm.LEFT` $\implies$ verify sub-frame peak recovery within `< 2.5 ms`, `verdict == "optimal_proximal_to_distal"`, reversed trajectory (`arm` before `thorax`) $\implies$ `"out_of_order"`, and `30 fps` / `timebase_unreliable` $\implies$ `"unmeasurable"` (`suppressed=True`).
2. **Module 2 (14-Segment COM Partition-of-Unity & Braking Physics):**
   - Verify $\sum_{j=0}^{23} \alpha_j == 1.0 \pm 10^{-12}$ for both male and female de Leva (1996) tables, exact translation equivariance $\mathbf{r}_{\text{COM}}(\mathbf{P} + \mathbf{c}) == \mathbf{r}_{\text{COM}}(\mathbf{P}) + \mathbf{c}$, exact braking $\Delta v_x$ on analytical constant-deceleration trajectory, hard suppression of `com_ffc_speed_mps` & `com_braking_delta_vx_mps` under `is_panning_camera=True`, and panning invariance of `front_leg_plant_angle_ffc_deg` & `com_vertical_sink_pct_stature`.
3. **Module 3 & 4 (Head Fall-Away & Lead-Arm Left/Right Mirror Symmetry):**
   - Construct exact 3D analytical poses with `+22 cm` contralateral head offset and `18% stature` flying lead-elbow flare on right-arm (`-Z`) and mirrored left-arm (`+Z`) $\implies$ verify identical positive values to `< 0.01` precision, plus ankle-crop suppression on `indoor_nlf_fixture.npz`.
4. **Module 5 & 6 (Crease Corridor Sign, No-Ball Gate & Tempo Ratio):**
   - Verify `+16 cm` open "train tracks" vs `-12 cm` crossed-over foot plant for both arms; verify `front_heel_crease_margin_cm` is reported ONLY on static `mcc_law7_pnp` and 100% suppressed on `two_wicket_geometry` / self-cal / panning; verify `tempo_gather_to_whip_ratio` matches `(t_ffc - t_bfc)/(t_br - t_ffc)` and suppresses on `edited_or_frozen` clips.
5. **Module 7 & 8 (Stroboscopic Compositor & Archetype Library):**
   - Verify `GET /api/deliveries/del_real_steyn/stroboscopic.jpg`, `del_real_indoor/stroboscopic.jpg`, and `del_demo_side_240/stroboscopic.jpg` return valid non-blank JPEGs; verify `GET /api/archetypes?arm=right` and `?arm=left` return all 4 archetypes with valid FFC-aligned 3D skeletons and explicit non-template disclaimers.
6. **Module 9 & 10 (ACWR Math, Low-Chronic Guardrail & Debiased Spell Drift):**
   - Verify exact hand-calculated Coupled ACWR, Uncoupled ACWR, and EWMA ACWR ($\lambda_a=0.25, \lambda_c=2/29$), verify `chronic < 30 balls/wk` triggers `low_chronic_base` instead of divide-by-near-zero spike, verify `DELETE /workload/{id}`, and verify `GET /api/bowlers/bwl_demo/spell-analytics` detects Ball 1→4 front-knee fatigue slope (`< 0 deg/ball`) and computes non-negative measurement-debiased release-tunnel SD.
7. **Neuro-Symbolic Verifier & Zero-Regression Gate:**
   - Verify `verify_coach_response` blocks adversarial suppressed v5 metric claims (e.g., claiming `com_braking_delta_vx_mps` on a panning clip or `seq_arm_peak_dps` at `25 fps`) while passing 100% of `CoachingStory` (`v5_insights` + `v5_detail_rows`) and `ask_grounded_coach` prompts across all 8 seeded deliveries.
   - Run full `pytest -v` (existing 30 tests + new v5 suite = **100% pass**) and `validation/run_synthetic_benchmark.py` (verifying **exact `96.4%` pooled coverage** on canonical 22 metrics + `>= 90%` empirical 95% CI coverage across new v5 metrics in `validation/validate_v5_modules.py`).
