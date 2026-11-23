"""Canonical anatomical indices, MCC Laws of Cricket (Laws 6, 7, 8) geometry, and literature timing constants.

Decouples production pipeline stages from `synthetic_ground_truth.py` (EXP-007 B23) and fixes
regulation stump center spacing per MCC Law 8 (EXP-007 B24).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical 24-joint SMPL skeleton order (matching NLF `detect_smpl_batched` output)
SMPL24_JOINTS: List[str] = [
    "pelv", "lhip", "rhip", "spi1", "lkne", "rkne", "spi2", "lank",
    "rank", "spi3", "ltoe", "rtoe", "neck", "lcla", "rcla", "head",
    "lsho", "rsho", "lelb", "relb", "lwri", "rwri", "lhan", "rhan",
]
JOINT_IDX: Dict[str, int] = {name: i for i, name in enumerate(SMPL24_JOINTS)}

# Left-right symmetric joint index pairs (for horizontal mirror consistency / repeatability sampling)
LR_JOINT_PAIRS: List[Tuple[int, int]] = [
    (JOINT_IDX["l" + name[1:]], JOINT_IDX[name])
    for name in SMPL24_JOINTS
    if name.startswith("r")
]

# MCC Laws of Cricket (2017 Code, 4th Edition):
#   Law 6: Pitch length between bowling creases = 22 yd = 20.12 m
#   Law 7: Popping crease 4 ft (1.22 m) in front of bowling crease; return creases 4 ft 4 in (1.32 m) each side
#   Law 8: Wickets 28 in (0.7112 m) tall, 9 in (0.2286 m) outer width; 3 stumps of diameter ~1.38-1.50 in
#          => outer off-to-leg stump CENTER spacing is 0.191-0.194 m (half-spacing = 0.0960 m)
MCC_PITCH_LENGTH_M: float = 20.12
MCC_POPPING_CREASE_M: float = 1.22
MCC_RETURN_CREASE_HALF_WIDTH_M: float = 1.32
MCC_STUMP_HEIGHT_M: float = 0.7112
MCC_STUMP_OUTER_WIDTH_M: float = 0.2286
MCC_STUMP_CENTER_HALF_SPACING_M: float = 0.0960

# Canonical Pitch World Frame (meters):
#   Origin (0, 0, 0): Base of middle stump on the bowler's bowling crease
#   +X: Down the pitch toward the striker's stumps
#   +Y: Vertical gravity-up
#   +Z: Across the pitch (+X x +Y = bowler's right when facing +X)
MCC_PITCH_LANDMARKS_M: Dict[str, Tuple[float, float, float]] = {
    "stump_middle_base": (0.0, 0.0, 0.0),
    "stump_middle_top": (0.0, MCC_STUMP_HEIGHT_M, 0.0),
    "stump_off_base": (0.0, 0.0, -MCC_STUMP_CENTER_HALF_SPACING_M),
    "stump_off_top": (0.0, MCC_STUMP_HEIGHT_M, -MCC_STUMP_CENTER_HALF_SPACING_M),
    "stump_leg_base": (0.0, 0.0, MCC_STUMP_CENTER_HALF_SPACING_M),
    "stump_leg_top": (0.0, MCC_STUMP_HEIGHT_M, MCC_STUMP_CENTER_HALF_SPACING_M),
    "bowling_crease_off": (0.0, 0.0, -MCC_RETURN_CREASE_HALF_WIDTH_M),
    "bowling_crease_leg": (0.0, 0.0, MCC_RETURN_CREASE_HALF_WIDTH_M),
    "popping_crease_off": (MCC_POPPING_CREASE_M, 0.0, -MCC_RETURN_CREASE_HALF_WIDTH_M),
    "popping_crease_leg": (MCC_POPPING_CREASE_M, 0.0, MCC_RETURN_CREASE_HALF_WIDTH_M),
    "popping_crease_center": (MCC_POPPING_CREASE_M, 0.0, 0.0),
    "return_crease_back_off": (-MCC_POPPING_CREASE_M, 0.0, -MCC_RETURN_CREASE_HALF_WIDTH_M),
    "return_crease_back_leg": (-MCC_POPPING_CREASE_M, 0.0, MCC_RETURN_CREASE_HALF_WIDTH_M),
    "far_popping_crease_off": (MCC_PITCH_LENGTH_M - MCC_POPPING_CREASE_M, 0.0, -MCC_RETURN_CREASE_HALF_WIDTH_M),
    "far_popping_crease_leg": (MCC_PITCH_LENGTH_M - MCC_POPPING_CREASE_M, 0.0, MCC_RETURN_CREASE_HALF_WIDTH_M),
    "far_stump_middle_base": (MCC_PITCH_LENGTH_M, 0.0, 0.0),
    "far_stump_middle_top": (MCC_PITCH_LENGTH_M, MCC_STUMP_HEIGHT_M, 0.0),
}

# Published fast-bowling phase timing means (Felton, Lister, Worthington, & King, 2019, Sports Biomechanics):
#   Elite male (n=20): BFC->FFC = 191.9 +- 31.3 ms; FFC->BR = 103.3 +- 11.2 ms
#   Elite female (n=20): BFC->FFC = 172.3 +- 27.6 ms; FFC->BR = 127.7 +- 11.0 ms
FELTON_2019_STRIDE_MS: float = 192.0
FELTON_2019_STRIDE_SD_MS: float = 31.3
FELTON_2019_FFC_BR_MS: float = 108.0  # midpoint covering male (103 ms) & female (128 ms)
FELTON_2019_FFC_BR_SD_MS: float = 14.0

# Nominal reference velocities for frame-rate temporal-aliasing diagnostics
NOMINAL_FAST_ARM_DPS: float = 2300.0
NOMINAL_FAST_BALL_MPS: float = 38.89  # 140 km/h
