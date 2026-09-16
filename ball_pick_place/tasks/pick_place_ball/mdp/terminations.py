# SPDX-License-Identifier: Apache-2.0
"""
Termination conditions for Ball Pick and Place.
===============================================
`ball_in_bucket` previously used bucket_pos z=0.47, inner_radius=0.08 and
height=0.10, which accepts a ball held in the gripper 8 cm off-centre and up to
10 cm above the rim. It was also never wired into TerminationsCfg, so the task
had no success signal or success metric at all.

It now shares the same bounds as the reward's `is_ball_in_bucket`, so a success
termination and a placing reward can never disagree.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg

from .geometry import (
    BUCKET_X,
    BUCKET_Y,
    IN_BUCKET_MAX_SPEED,
    IN_BUCKET_XY_HALF,
    IN_BUCKET_Z_MAX,
    IN_BUCKET_Z_MIN,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_in_bucket(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """True when the ball is settled inside the bucket cavity."""
    ball = env.scene[ball_cfg.name]
    p = ball.data.root_pos_w - env.scene.env_origins

    inside_x = (p[:, 0] - BUCKET_X).abs() < IN_BUCKET_XY_HALF
    inside_y = (p[:, 1] - BUCKET_Y).abs() < IN_BUCKET_XY_HALF
    inside_z = (p[:, 2] >= IN_BUCKET_Z_MIN) & (p[:, 2] <= IN_BUCKET_Z_MAX)

    speed = torch.norm(ball.data.root_lin_vel_w, dim=-1)
    settled = speed < IN_BUCKET_MAX_SPEED

    return inside_x & inside_y & inside_z & settled


def ball_off_table(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Failure: the ball fell off the table."""
    p = env.scene[ball_cfg.name].data.root_pos_w - env.scene.env_origins
    return p[:, 2] < (TABLE_TOP_Z - 0.07)
