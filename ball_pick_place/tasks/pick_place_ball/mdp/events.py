# SPDX-License-Identifier: Apache-2.0
"""
Reset events for Ball Pick and Place.
=====================================
Adds a reset-state curriculum, which is the single highest-leverage change for
the "policy never opens the gripper" failure.

The problem is not that placing pays too little; it is that the policy's actual
probability of a successful drop starts near zero, so the expected value of
attempting a release is below the value of simply holding. Reward tuning alone
cannot fix that -- the policy has to SEE the placed state to learn it is good.

`reset_ball_position_on_table` now spawns a fraction of environments with the
ball already resting inside the bucket. Those episodes hand the critic a high-
value terminal state for free, which propagates backward through the value
function until the actor is willing to try the release itself.

The fraction anneals from `start_fraction` to 0.0 over `anneal_steps`
environment steps, after which every episode is a full reach-to-place task.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

from .geometry import (
    BALL_RADIUS,
    BALL_REST_Z_IN_BUCKET,
    BUCKET_X,
    BUCKET_Y,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def reset_ball_position_on_table(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    table_x_range: tuple[float, float] = (0.20, 0.45),
    table_y_range: tuple[float, float] = (-0.18, 0.18),
    start_fraction: float = 0.6,
    anneal_steps: int = 6_000_000,
):
    """Randomize the ball, with an annealed chance of starting it in the bucket.

    Args:
        start_fraction: initial share of resets that begin with the ball already
            inside the bucket. 0.6 is a good default; set to 0.0 to disable the
            curriculum entirely and recover the original behaviour.
        anneal_steps: total environment steps over which the fraction decays
            linearly to zero. With 2048 envs this is roughly 3000 iterations.
    """
    ball = env.scene[ball_cfg.name]
    num_resets = len(env_ids)
    if num_resets == 0:
        return

    # ── Anneal the curriculum fraction ───────────────────────────────────────
    progress = min(float(env.common_step_counter) / float(anneal_steps), 1.0)
    in_bucket_fraction = start_fraction * (1.0 - progress)

    # ── Default: uniform random placement on the table surface ───────────────
    rand_x = math_utils.sample_uniform(
        table_x_range[0], table_x_range[1], (num_resets, 1), device=env.device
    )
    rand_y = math_utils.sample_uniform(
        table_y_range[0], table_y_range[1], (num_resets, 1), device=env.device
    )
    rand_z = torch.full(
        (num_resets, 1), TABLE_TOP_Z + BALL_RADIUS + 0.005, device=env.device
    )
    new_pos = torch.cat([rand_x, rand_y, rand_z], dim=-1)

    # ── Curriculum: override a subset to start inside the bucket ─────────────
    if in_bucket_fraction > 0.0:
        pick = torch.rand(num_resets, device=env.device) < in_bucket_fraction
        if pick.any():
            n = int(pick.sum())
            jitter = math_utils.sample_uniform(-0.015, 0.015, (n, 2), device=env.device)
            new_pos[pick, 0] = BUCKET_X + jitter[:, 0]
            new_pos[pick, 1] = BUCKET_Y + jitter[:, 1]
            new_pos[pick, 2] = BALL_REST_Z_IN_BUCKET

    root_states = ball.data.default_root_state[env_ids].clone()
    root_states[:, 0:3] = new_pos + env.scene.env_origins[env_ids]
    root_states[:, 7:13] = 0.0
    ball.write_root_state_to_sim(root_states, env_ids=env_ids)
