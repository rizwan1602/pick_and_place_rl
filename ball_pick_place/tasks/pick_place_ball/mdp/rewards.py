# SPDX-License-Identifier: Apache-2.0
"""
Reward functions for Ball Pick and Place.
=========================================
Rewritten to remove the "never let go" local optimum.

The original design paid ~38/step for holding the ball near the bucket and
~73/step for placing it, a ratio of only 1.9x. With gamma=0.995 at 50 Hz that
demanded a ~47% release success rate before opening the gripper was worth
attempting, which the policy never reached. It converged to holding.

Three structural changes:

1. Carry rewards (reach / lift / coarse / fine) are cut so their sum is small.
   Holding is now worth ~12/step instead of ~38/step.
2. `torch.max(..., in_bucket)` credit preservation is REMOVED. It existed to
   avoid punishing release, but it also handed the "holding" state most of the
   "placed" state's value. With carry rewards small and placing large, release
   is immediately profitable and the preservation is unnecessary.
3. `release_reward` is a smooth corridor instead of a product of three step
   functions, so the policy gets gradient while approaching a valid drop pose.

New per-step totals:
    holding at the carry waypoint  ~12
    ball resting in the bucket     ~37

Ratio ~3.1x, break-even success probability ~0.33. Combine with the reset
curriculum in events.py, which drives the actual success rate well above that.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import quat_apply

from .geometry import (
    BALL_RADIUS,
    BALL_REST_Z_ON_TABLE,
    BUCKET_X,
    BUCKET_Y,
    CARRY_TARGET,
    FINGER_RELEASE_THRESHOLD,
    IN_BUCKET_MAX_SPEED,
    IN_BUCKET_XY_HALF,
    IN_BUCKET_Z_MAX,
    IN_BUCKET_Z_MIN,
    RELEASE_XY_TOL,
    RELEASE_Z,
    TABLE_TOP_Z,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ── Helpers ──────────────────────────────────────────────────────────────────

def _ball_pos_env(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """Ball position in env-local coordinates."""
    return env.scene[ball_cfg.name].data.root_pos_w - env.scene.env_origins


def _ee_pos(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_tf.data.target_pos_w[..., 0, :]


def _finger_opening(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg) -> torch.Tensor:
    """Mean finger joint position, resolved by name rather than by index."""
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_finger_joint_ids"):
        ids, _ = robot.find_joints("panda_finger_joint.*")
        env._finger_joint_ids = ids
    return robot.data.joint_pos[:, env._finger_joint_ids].mean(dim=-1)


def is_ball_in_bucket(env: ManagerBasedRLEnv, ball_cfg: SceneEntityCfg) -> torch.Tensor:
    """1.0 when the ball is settled inside the bucket cavity.

    Box test against the SQUARE cavity (the original circular r=0.05 test made
    the cavity corners unreachable for credit), plus a settle-speed check so a
    ball flying through the opening does not score.
    """
    p = _ball_pos_env(env, ball_cfg)
    inside_x = (p[:, 0] - BUCKET_X).abs() < IN_BUCKET_XY_HALF
    inside_y = (p[:, 1] - BUCKET_Y).abs() < IN_BUCKET_XY_HALF
    inside_z = (p[:, 2] >= IN_BUCKET_Z_MIN) & (p[:, 2] <= IN_BUCKET_Z_MAX)

    speed = torch.norm(env.scene[ball_cfg.name].data.root_lin_vel_w, dim=-1)
    settled = speed < IN_BUCKET_MAX_SPEED

    return (inside_x & inside_y & inside_z & settled).float()


# ── Stage 1: Reach (Broad + Fine) ───────────────────────────────────────────

def reaching_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.25,
) -> torch.Tensor:
    """Broad smooth pull of the end-effector toward the ball from across the table."""
    distance = torch.norm(
        _ee_pos(env, ee_frame_cfg) - env.scene[ball_cfg.name].data.root_pos_w, dim=-1
    )
    reward = 1.0 - torch.tanh(distance / std)
    return reward * (1.0 - is_ball_in_bucket(env, ball_cfg))


def reaching_reward_fine(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.05,
) -> torch.Tensor:
    """Precision alignment pulling fingertips onto the ball surface."""
    distance = torch.norm(
        _ee_pos(env, ee_frame_cfg) - env.scene[ball_cfg.name].data.root_pos_w, dim=-1
    )
    reward = 1.0 - torch.tanh(distance / std)
    return reward * (1.0 - is_ball_in_bucket(env, ball_cfg))


# ── Stage 2: Grasp (Clamp fingers when at ball) ──────────────────────────────

def grasping_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward closing gripper fingers when the end-effector is around the ball."""
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w
    dist_to_ee = torch.norm(_ee_pos(env, ee_frame_cfg) - ball_pos, dim=-1)
    near_ball = (dist_to_ee < 0.045).float()

    finger = _finger_opening(env, robot_cfg)
    # Fingers range from 0.04 (fully open) to 0.00 (clamped)
    clamp_progress = torch.clamp((0.04 - finger) / 0.04, 0.0, 1.0)

    return near_ball * clamp_progress * (1.0 - is_ball_in_bucket(env, ball_cfg))


# ── Stage 3: Lift ────────────────────────────────────────────────────────────

def lifting_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    lift_target_height: float = 0.12,
) -> torch.Tensor:
    """Continuous lift progress + discrete milestone bonus when ball leaves table."""
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w
    p = ball_pos - env.scene.env_origins

    dist_to_ee = torch.norm(_ee_pos(env, ee_frame_cfg) - ball_pos, dim=-1)
    held = 1.0 - torch.tanh(dist_to_ee / 0.06)

    height_above_rest = p[:, 2] - BALL_REST_Z_ON_TABLE
    continuous_lift = torch.clamp(height_above_rest / lift_target_height, 0.0, 1.0)
    # Milestone: ball has clearly lifted off the table surface (> 2.5 cm)
    is_lifted = (height_above_rest > 0.025).float()

    return held * (continuous_lift + is_lifted) * (1.0 - is_ball_in_bucket(env, ball_cfg))


# ── Stage 4: Carry to Bucket ─────────────────────────────────────────────────

def bucket_tracking_coarse(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    std: float = 0.25,
) -> torch.Tensor:
    """Broad gradient pulling a held, lifted ball toward the carry waypoint."""
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w
    p = ball_pos - env.scene.env_origins

    is_lifted = (p[:, 2] > (TABLE_TOP_Z + 0.035)).float()
    dist_to_ee = torch.norm(_ee_pos(env, ee_frame_cfg) - ball_pos, dim=-1)
    held = 1.0 - torch.tanh(dist_to_ee / 0.07)

    target = torch.tensor(CARRY_TARGET, device=env.device).unsqueeze(0)
    distance = torch.norm(p - target, dim=-1)

    return is_lifted * held * (1.0 - torch.tanh(distance / std)) * (
        1.0 - is_ball_in_bucket(env, ball_cfg)
    )


# ── Stage 5: Release ─────────────────────────────────────────────────────────

def release_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward opening the fingers inside the drop corridor."""
    p = _ball_pos_env(env, ball_cfg)

    xy_dist = torch.norm(
        p[:, :2] - torch.tensor([BUCKET_X, BUCKET_Y], device=env.device), dim=-1
    )
    xy_term = 1.0 - torch.tanh(xy_dist / RELEASE_XY_TOL)
    z_term = 1.0 - torch.tanh((p[:, 2] - RELEASE_Z).abs() / 0.04)

    finger = _finger_opening(env, robot_cfg)
    openness = torch.clamp(
        (finger - FINGER_RELEASE_THRESHOLD) / (0.040 - FINGER_RELEASE_THRESHOLD),
        0.0,
        1.0,
    )

    return xy_term * z_term * openness


# ── Stage 6: Place ───────────────────────────────────────────────────────────

def placing_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Large sustained reward for the ball resting inside the cavity."""
    in_bucket = is_ball_in_bucket(env, ball_cfg)
    if hasattr(env, "extras"):
        env.extras.setdefault("log", {})["success_rate"] = in_bucket.mean().item()
    return in_bucket


# ── Posture: downward end-effector orientation ───────────────────────────────

def ee_downward_orientation_reward(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the gripper for pointing along world -Z.

    NOTE: Isaac Lab quaternions are (w, x, y, z), NOT (x, y, z, w). The original
    comment here said xyzw. The code was correct -- an Isaac Lab quat passed to
    Isaac Lab `quat_apply` -- but the comment would have caused a real bug the
    first time anyone converted it.
    """
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_tf.data.target_quat_w.torch[..., 0, :]  # (N, 4) wxyz

    local_z = torch.zeros(ee_quat.shape[0], 3, device=env.device)
    local_z[:, 2] = 1.0
    world_z = quat_apply(ee_quat, local_z)

    return torch.clamp(-world_z[:, 2], 0.0, 1.0)


# ── Posture: arm table clearance ─────────────────────────────────────────────

def arm_table_clearance_penalty(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.47,
) -> torch.Tensor:
    """Penalize forearm links resting on the table.

    Body indices are resolved by name and cached, rather than hardcoded to
    4 and 5, so this survives any change to the Franka asset.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    if not hasattr(env, "_forearm_body_ids"):
        ids, _ = robot.find_bodies(["panda_link4", "panda_link5"])
        env._forearm_body_ids = ids

    body_pos = robot.data.body_pos_w.torch
    z = body_pos[:, env._forearm_body_ids, 2]
    lowest_z = z.min(dim=-1).values
    return torch.clamp(min_height - lowest_z, min=0.0)


# ── Penalty: ball velocity ───────────────────────────────────────────────────

def ball_velocity_penalty(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Discourage batting the ball across the table.

    Suppressed once placed, so the settling bounce inside the bucket is not
    punished.
    """
    speed = torch.norm(env.scene[ball_cfg.name].data.root_lin_vel_w, dim=-1)
    return speed * (1.0 - is_ball_in_bucket(env, ball_cfg))
