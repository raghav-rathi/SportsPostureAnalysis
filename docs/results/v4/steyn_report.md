# Fast-Bowling Biomechanical Analysis Report: Dale_Steyn_Net_25fps

## Executive Summary
**Dale_Steyn_Net_25fps** (right-arm pace bowler, action class: **mixed**) was analyzed from 1920x1080 @ **25.18 unique fps** (timebase: `pulldown_deduped`, view: `rear_behind_bowler`, azimuth `0.0 deg`, elevation `1.0 deg`, calibration: `two_wicket_geometry`, intrinsics calibrated: `True`, camera motion/pan: `True`).

**Interval Semantics:** Reported ranges are **Repeatability Intervals** (`repeatability_interval`). Unvalidated 0-100 composite scores are intentionally retired in favor of individual verified metrics and cohort norms.

## Delivery Phase & Observability Overview
Delivery stride duration (BFC->FFC) was **214 ms** (+- 29.7/24.7 ms) and FFC->BR took **73 ms** (+- 14.1 ms). Out of 23 tracked variables, **9** passed observability and confidence gates, **13** were withheld, and **1** were permanently refused.

## Verified Biomechanical Measurements (Repeatability Interval)

| Metric | Value (Repeatability Interval) | Observability | Literature Context & Definition Note |
|---|---|---|---|
| Run-Up Approach Speed | **[UNMEASURABLE]** | `unmeasurable` | Elite male fast bowlers (n=20): 5.83 +- 0.50 m/s (strongest pace predictor: +0.89 m/s ball speed per SD). |
| Delivery Stride Length | **[UNMEASURABLE]** | `unmeasurable` | Published adult fast-bowling stride length: ~75-88% of stature (1.35-1.65 m). |
| Front Knee Angle at FFC | **[UNMEASURABLE]** | `unmeasurable` | Typical adult fast-bowling FFC angle: ~150-170 deg (180 deg = straight). |
| Front Knee Angle at Ball Release | **[UNMEASURABLE]** | `unmeasurable` | Elite male fast bowlers (n=20): straighter front knee at BR strongly predicts speed (+0.79 m/s per SD). |
| Front Knee Bracing Change (FFC->BR) | **[UNMEASURABLE]** | `unmeasurable` | Extender/braced patterns generally conserve braking impulse better than collapsing flexors. |
| Global Trunk Forward Inclination at FFC | **[UNMEASURABLE]** | `unmeasurable` | Global trunk inclination proxy (not directly interchangeable with Worthington upper-vs-lower trunk angle). |
| Global Trunk Forward Inclination at BR | **[UNMEASURABLE]** | `unmeasurable` | Global trunk inclination proxy at release (not directly interchangeable with inter-segment trunk flexion). |
| Global Trunk Forward Flexion Change (FFC->BR) | **[UNMEASURABLE]** | `unmeasurable` | Greater forward trunk flexion FFC->BR associates with faster ball speeds (+0.58 m/s per SD in Worthington 2013). |
| Global Trunk Contralateral Inclination at FFC | **16.0 deg** (`[2.7, 25.4]`) | `high` | Whole-trunk global lateral inclination at FFC (distinct from Alway 2021 thoracolumbar joint angle). |
| Global Trunk Contralateral Inclination at BR | **30.3 deg** (`[26.8, 36.5]`) | `high` | Context-dependent: adolescent cohort (Bayne 2016, n=25) vs elite adult study (Alway 2021, n=50) disagree in direction; interpret alongside workload. |
| Pelvis Alignment at BFC | **255.8 deg** (`[241.9, 268.0]`) | `moderate` | AIS convention (Portus et al. 2001): <210 deg side-on, 210-240 deg semi-open, >240 deg front-on. |
| Shoulder Alignment at BFC | **271.4 deg** (`[258.1, 282.7]`) | `moderate` | AIS convention (Portus et al. 2001): <210 deg side-on, 210-240 deg semi-open, >240 deg front-on. |
| Shoulder Counter-Rotation (SCR, BFC->FFC) | **46.2 deg** (`[36.9, 55.0]`) | `moderate` | AIS mixed-action threshold >=30 deg (Portus 2001/2004 retrospective); note Alway (2021) prospective adult study found no SCR-LBSI link. |
| Hip-Shoulder Separation at BFC | **-15.6 deg** (`[-27.3, -3.8]`) | `moderate` | AIS mixed-action criterion includes |separation at BFC| >= 30 deg. |
| Peak Dynamic Hip-Shoulder Separation | **22.6 deg** (`[12.7, 32.6]`) | `moderate` | Descriptive transverse torso twist measure across the delivery stride. |
| Bowling Arm Behind Global Vertical at FFC | **[UNMEASURABLE]** | `unmeasurable` | Larger arm delay at FFC strongly predicts faster pace (+0.62 m/s per SD in Worthington 2013). |
| Rear Hip Angle Proxy at BFC (Trunk-to-Thigh) | **[UNMEASURABLE]** | `unmeasurable` | Proxy for Alway (2021) rear-hip predictor (more flexed rear hip at BFC associated with LBSI in 50-bowler study). |
| Frontal Pelvic Tilt at FFC | **9.2 deg** (`[-2.2, 13.4]`) | `high` | Frontal-plane pelvic control indicator during front-foot braking. |
| Non-Bowling (Front) Arm Elevation at FFC | **[UNMEASURABLE]** | `unmeasurable` | Descriptive front-arm position entering front-foot contact. |
| Peak Bowling Arm Angular Velocity | **[UNMEASURABLE]** | `unmeasurable` | High-speed kinematic descriptor; requires >=100 fps and verified timebase. |
| Ball Release Height (Local Ground) | **1.9 m** (`[1.7, 1.9]`) | `low` | Typically ~1.05-1.18x bowler stature. |
| Follow-Through Lateral Drift Angle | **[UNMEASURABLE]** | `unmeasurable` | Requires calibrated static-camera world translation; suppressed under panning or uncalibrated FOV. |
| ICC 15-Degree Elbow Extension Legality | **[REFUSED]** | `refused` | ICC limit: <=15 deg elbow extension from Upper Arm Horizontal (UAH) to Ball Release (BR). |

## Definition-Matched Cohort Comparisons

| Metric | Value | Cohort (Mean +- SD) | Z-Score | Approx %ile | Definition Match |
|---|---|---|---|---|---|
| Global Trunk Contralateral Inclination at BR | 30.3 `[26.8, 36.5]` deg | Adolescent Uninjured Cohort (Bayne 2016, ages 14-19, n=13): 40.0 +- 8.0 | -1.22 | 11% | `proxy_global_frame` |
| Shoulder Counter-Rotation (SCR, BFC->FFC) | 46.2 `[36.9, 55.0]` deg | AIS Fast-Bowling Cohort (Portus 2001/2004, n=42): 22.0 +- 10.0 | +2.42 | 99% | `exact` |

## Interval-Cleared Strengths
- No strengths cleared interval-level confidence gates from this clip.

## Interval-Cleared Limiters & Inconclusive Observations
- **Mixed-Action Shoulder Counter-Rotation (>=30 deg):** Shoulder counter-rotation (46.2 deg, interval [36.9, 55.0]) strictly clears the 30-degree AIS mixed-action threshold (Portus 2001/2004 retrospective; note Alway 2021 prospective adult study did not find an independent SCR-LBSI association). *Actionable focus:* Align hips and shoulders coherently at BFC to minimize abrupt upper-trunk recoil.

## Capture Quality Caveats & Suppressed Metrics
- Regular 1-in-6 pulldown detected (16% repeated frames); dropped 12 duplicate frames and adjusted FPS from 29.97 to 25.17 unique fps.
- Effective unique frame rate is 25.2 fps (~91.4 deg arm sweep/frame); release-instant kinematics carry elevated temporal quantization error.
- **Capture Upgrade — Switch Smartphone to 120–240 fps Slow-Motion:** Record outdoors in native 1080p 240 fps slow-motion.
- **Capture Upgrade — Lock Camera on a Static Tripod (Eliminate Handheld Pan):** Lock the tripod head during the delivery so background wickets stay stationary.
- **Capture Upgrade — Add a Side-On (90 deg) Camera Angle:** Place a tripod phone 8-12 m perpendicular to the popping crease showing full body.
- **Refused `ICC 15-Degree Elbow Extension Legality`:** PERMANENTLY REFUSED ON VIDEO: Accredited ICC bowling legality adjudication requires >=250 Hz multi-camera 3D optical motion capture with anatomical upper-arm/forearm marker clusters. Abraham, Feros, & Fox (2025, IJSSC) reported 22.71 deg elbow RMSE even on three calibrated 240 Hz iPads.
- Withheld `Run-Up Approach Speed`: Suppressed: world-grounded translation requires a static camera with calibrated MCC pitch/wicket geometry; camera pan or uncalibrated FOV distorts depth displacement.
- Withheld `Delivery Stride Length`: Suppressed: world-grounded translation requires a static camera with calibrated MCC pitch/wicket geometry; camera pan or uncalibrated FOV distorts depth displacement.
- Withheld `Front Knee Angle at FFC`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Front Knee Angle at Ball Release`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Front Knee Bracing Change (FFC->BR)`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Global Trunk Forward Inclination at FFC`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Global Trunk Forward Inclination at BR`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Global Trunk Forward Flexion Change (FFC->BR)`: Suppressed: camera azimuth (0 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Additional withheld metrics (5): Bowling Arm Behind Global Vertical at FFC, Rear Hip Angle Proxy at BFC (Trunk-to-Thigh), Non-Bowling (Front) Arm Elevation at FFC, Peak Bowling Arm Angular Velocity, Follow-Through Lateral Drift Angle.

## Scientific & Medical Guardrails
> ICC 15-degree elbow legality CANNOT be adjudicated from monocular or casual 2D/3D video; accredited >=250 Hz 3D optical lab testing is required.
> Biomechanical load screening estimates single-delivery mechanical posture only and is NOT a medical diagnosis; injury risk depends strongly on workload, age/skeletal maturity, and physical capacity.
> Intervals reported without held-out physical reference calibration represent repeatability or analytical sensitivity estimates rather than externally calibrated laboratory confidence intervals.
