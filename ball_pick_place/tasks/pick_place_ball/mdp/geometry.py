# SPDX-License-Identifier: Apache-2.0
"""
Single source of truth for scene geometry.
==========================================
Every number here is derived from scenes/bucket.usd and env_cfg.py.
Nothing else in the project should hardcode a bucket or table dimension.

bucket.usd uses USD `Cube` prims whose default `size` is 2.0, so an
`xformOp:scale` of s gives a HALF-extent of s.

    Floor : scale (0.065, 0.065, 0.0025) @ z=0.0025 -> top face at z = +0.005
    Walls : scale (*, 0.0025, 0.025)     @ z=0.03   -> top face at z = +0.055
            inner faces at +/- 0.060 in x and y

Bucket prim is placed at (0.35, 0.30, 0.42) in env-local coordinates.
"""

from __future__ import annotations

# ── Table ────────────────────────────────────────────────────────────────────
TABLE_TOP_Z: float = 0.42

# ── Ball ─────────────────────────────────────────────────────────────────────
BALL_RADIUS: float = 0.03
BALL_REST_Z_ON_TABLE: float = TABLE_TOP_Z + BALL_RADIUS          # 0.450

# ── Bucket (derived from bucket.usd) ─────────────────────────────────────────
BUCKET_ORIGIN: tuple[float, float, float] = (0.35, 0.30, 0.42)
BUCKET_X: float = BUCKET_ORIGIN[0]
BUCKET_Y: float = BUCKET_ORIGIN[1]

BUCKET_FLOOR_TOP_Z: float = BUCKET_ORIGIN[2] + 0.005             # 0.425
BUCKET_RIM_TOP_Z: float = BUCKET_ORIGIN[2] + 0.055               # 0.475
BUCKET_CAVITY_HALF: float = 0.060                                # square, not circular
BUCKET_WALL_THICKNESS: float = 0.005

# Ball centre height when resting on the bucket floor.
BALL_REST_Z_IN_BUCKET: float = BUCKET_FLOOR_TOP_Z + BALL_RADIUS  # 0.455

# Ball centre must exceed this to physically clear the rim.
BALL_CLEARS_RIM_Z: float = BUCKET_RIM_TOP_Z + BALL_RADIUS        # 0.505

# ── Derived control targets ──────────────────────────────────────────────────
# Where the ball should be hovering at the moment of release: comfortably
# above the rim so the fingers are clear of the walls, but not so high that
# the drop scatters.
RELEASE_Z: float = 0.550                                         # 4.5 cm of clearance
RELEASE_XY_TOL: float = 0.035                                    # margin inside the 0.060 cavity

# Carry waypoint the policy is guided toward before release.
CARRY_TARGET: tuple[float, float, float] = (BUCKET_X, BUCKET_Y, RELEASE_Z)

# ── Success test bounds ──────────────────────────────────────────────────────
# Box test matching the square cavity, with margin so a ball nestled in a
# corner still counts.
IN_BUCKET_XY_HALF: float = BUCKET_CAVITY_HALF - BALL_RADIUS + 0.015   # 0.045
IN_BUCKET_Z_MIN: float = BALL_REST_Z_IN_BUCKET - 0.015               # 0.440
IN_BUCKET_Z_MAX: float = BUCKET_RIM_TOP_Z                            # 0.475
IN_BUCKET_MAX_SPEED: float = 0.15

# Franka finger joint travel.
FINGER_OPEN: float = 0.04
FINGER_CLOSED_ON_BALL: float = BALL_RADIUS     # ~0.030 when gripping the ball
FINGER_RELEASE_THRESHOLD: float = 0.034        # unambiguously released
