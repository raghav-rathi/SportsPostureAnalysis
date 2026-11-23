# Fast-Bowling Biomechanical Analysis Report: Dale_Steyn_Net_30fps

## Executive Summary
**Dale_Steyn_Net_30fps** (right-arm pace bowler, classified as **unknown**) was analyzed from 1920x1080 @ **30.0 fps** footage (view: `rear_behind_bowler`, azimuth `-2.9 deg`, elevation `-1.0 deg`, calibration: `motion_gravity_self_cal`, panning: `True`).

**Kinetic & Pace Efficiency:** Too few primary pace-efficiency kinematics (front-knee bracing, trunk flexion, delayed arm, run-up speed) were reliably observable from this camera view and frame rate.

**Mechanical Load & Durability:** Durability index 89.7 +- 14.0 / 100 (Low Mechanical Load Flag) from 4 observable lumbar-loading markers (thorax_lat_flex_br_deg, thorax_lat_flex_ffc_deg, pelvic_drop_ffc_deg, follow_through_drift_deg).

**2D Quadrant Placement:** `Partial Observability (Add Complementary Camera Angle)` — Full 2D Performance-Durability placement requires both sagittal (side-on) kinetic evidence and coronal/transverse (rear or 45-deg) spinal-load evidence.

## Delivery Phase & Observability Overview
Delivery stride duration (BFC->FFC) was **210 ms** and front-foot contact to ball release (FFC->BR) took **106 ms** (+- 16.7 ms frame quantization). Out of 23 tracked biomechanical variables, **8** passed observability and confidence thresholds, **14** were withheld due to camera-angle, crop, panning, or frame-rate constraints, and **1** were permanently refused.

## Verified Biomechanical Measurements (with 95% Confidence Intervals)

| Metric | Value (95% CI) | Observability | Empirical Literature Reference |
|---|---|---|---|
| Run-Up Approach Speed | **[UNMEASURABLE]** | `unmeasurable` | Elite male fast bowlers: ~5.2–6.6 m/s (strong positive predictor of ball speed) |
| Delivery Stride Length | **[UNMEASURABLE]** | `unmeasurable` | Typical fast bowling stride: ~72–88% of stature (1.30–1.65 m) |
| Front Knee Angle at FFC | **[UNMEASURABLE]** | `unmeasurable` | Typical FFC contact angle: ~150–170 deg (180 deg = fully straight) |
| Front Knee Angle at Ball Release | **[UNMEASURABLE]** | `unmeasurable` | Faster bowlers maintain straighter braced front knee (~158–178 deg at BR) |
| Front Knee Bracing Change (FFC->BR) | **[UNMEASURABLE]** | `unmeasurable` | Positive (+) = extender/bracer; near 0 = braced; <=-10 deg = flexing/collapsing |
| Upper-Trunk Forward Flexion at FFC | **[UNMEASURABLE]** | `unmeasurable` | Typical FFC upright/slight-forward posture: ~5–20 deg forward of vertical |
| Upper-Trunk Forward Flexion at BR | **[UNMEASURABLE]** | `unmeasurable` | Typical elite release trunk flexion: ~35–55 deg forward of vertical |
| Trunk Forward Flexion Range (FFC->BR) | **[UNMEASURABLE]** | `unmeasurable` | Greater forward trunk flexion FFC->BR (~25–42 deg) drives higher ball speed |
| Thorax Contralateral Lateral Flexion at FFC | **13.6 deg** (`[1.8, 25.4]`) | `high` | Moderate pre-release side-flexion (~12–28 deg); excessive early bend elevates load |
| Thorax Contralateral Lateral Flexion at BR | **30.2 deg** (`[19.7, 40.8]`) | `high` | Uninjured elite norm: ~32–44 deg (40+-8 deg); elevated LBSI risk around >=48–50 deg |
| Pelvis Alignment at BFC | **250.0 deg** (`[225.6, 274.4]`) | `low` | AIS convention: <=205 deg side-on, 205–245 deg semi, >245 deg front-on |
| Shoulder Alignment at BFC | **[UNMEASURABLE]** | `unmeasurable` | AIS convention: <=205 deg side-on, 205–245 deg semi, >245 deg front-on |
| Shoulder Counter-Rotation (SCR, BFC->FFC) | **[UNMEASURABLE]** | `unmeasurable` | <25-30 deg = non-mixed alignment; >=30 deg defines mixed action linked to LBSI |
| Hip-Shoulder Separation at BFC | **-15.5 deg** (`[-35.8, 4.7]`) | `moderate` | BFC misalignment >=30 deg with counter-rotation indicates mixed action loading |
| Peak Dynamic Hip-Shoulder Separation | **22.5 deg** (`[1.0, 44.0]`) | `moderate` | Well-timed late separation (~25–50 deg) supports stretch-shortening momentum transfer |
| Delayed Bowling Arm Angle Behind Vertical at FFC | **[UNMEASURABLE]** | `unmeasurable` | Larger delay behind vertical (~125–165 deg at FFC) strongly predicts higher ball speed |
| Rear Hip Angle at BFC | **[UNMEASURABLE]** | `unmeasurable` | Key upstream lumbopelvic control predictor in prospective LBSI regression model |
| Contralateral/Ipsilateral Pelvic Tilt at FFC | **8.5 deg** (`[-10.5, 27.6]`) | `moderate` | Controlled level pelvis (|tilt| <10 deg) reduces asymmetrical lumbar shear loading |
| Non-Bowling (Front) Arm Elevation at FFC | **[UNMEASURABLE]** | `unmeasurable` | High front-arm anchor (>15 deg above horizontal at FFC) supports trunk alignment before pull |
| Peak Bowling Arm Angular Velocity | **[UNMEASURABLE]** | `unmeasurable` | Elite fast bowlers peak ~2,000–2,570 deg/s (35–45 rad/s) near ball release |
| Ball Release Height | **2.1 m** (`[1.8, 2.4]`) | `moderate` | Typically ~1.05–1.18x bowler stature (~1.95–2.20 m for 1.85 m bowler) |
| Follow-Through Lateral Drift Angle | **-4.6 deg** (`[-13.1, 3.9]`) | `high` | Driving momentum near target line (|drift| <12-15 deg) reduces lateral braking shear |
| ICC 15-Degree Elbow Legality Verdict | **[REFUSED]** | `refused` | ICC limit: <=15 deg active elbow extension strictly between UAH and BR |

## Key Strengths
- **Controlled Contralateral Trunk Clearance:** Thorax lateral flexion at release (30.2 deg) sits cleanly within the uninjured elite fast-bowling reference range (Bayne et al., 2016: 40 +- 8 deg). *Next step:* Maintain current lateral trunk stability as bowling intensity increases.

## Prioritized Mechanical Limiters & Coaching Focus
- No high-confidence mechanical limiters were flagged among the currently observable metrics.

## Capture Quality Caveats & Suppressed Metrics
- Frame rate is 30.0 fps (~76.7 deg bowling-arm sweep per frame); release-instant kinematics carry elevated temporal quantization error. Record at 120-240 fps slow-motion outdoors for precision analysis.
- **Capture Upgrade — Switch Smartphone to 120–240 fps Slow-Motion:** Record outdoors in native 1080p 240 fps slow-motion to reduce timing uncertainty by up to 8x.
- **Capture Upgrade — Use a Static Tripod Showing Pitch Creases:** Mount the camera on a stationary tripod framing both popping/bowling creases and the bowler's full body.
- **Capture Upgrade — Add a Side-On (90 deg) Camera Angle:** Place a tripod phone 8-12 m perpendicular to the popping crease showing the full run-up to follow-through.
- **Refused `ICC 15-Degree Elbow Legality Verdict`:** Permanently refused: even dual-camera markerless OpenCap exhibits 22.7 deg RMSE on cricket elbow flexion (Abraham et al., 2025/2026), exceeding the entire 15-degree ICC threshold. Authoritative legality testing requires an ICC-accredited >=250 Hz 3D optical lab.
- Withheld `Run-Up Approach Speed`: Suppressed: camera panning cancels ground-referenced translation without MCC Law 7 pitch calibration; record from a static tripod or supply pitch crease landmarks.
- Withheld `Delivery Stride Length`: Suppressed: camera panning cancels ground-referenced translation without MCC Law 7 pitch calibration; record from a static tripod or supply pitch crease landmarks.
- Withheld `Front Knee Angle at FFC`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Front Knee Angle at Ball Release`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Front Knee Bracing Change (FFC->BR)`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Upper-Trunk Forward Flexion at FFC`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Upper-Trunk Forward Flexion at BR`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Withheld `Trunk Forward Flexion Range (FFC->BR)`: Suppressed: camera azimuth (-3 deg, view=rear_behind_bowler) cannot reliably resolve sagittal (side-on) kinematics; film from a side-on (~90 deg) angle.
- Additional withheld metrics (6): Shoulder Alignment at BFC, Shoulder Counter-Rotation (SCR, BFC->FFC), Delayed Bowling Arm Angle Behind Vertical at FFC, Rear Hip Angle at BFC, Non-Bowling (Front) Arm Elevation at FFC, Peak Bowling Arm Angular Velocity.

## Scientific & Medical Guardrails
> ICC 15-degree elbow legality CANNOT be adjudicated from monocular or casual 2D/3D video; accredited >=250 Hz 3D optical lab testing is required.
> Mechanical load screening estimates stress per delivery only and is NOT a medical diagnosis; actual injury risk depends strongly on bowling workload, age/skeletal maturity, and physical capacity.
