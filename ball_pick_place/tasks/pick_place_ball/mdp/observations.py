# SPDX-License-Identifier: Apache-2.0
"""
Observation functions for Ball Pick and Place task.
===================================================
All observations are properly computed relative to each environment's origin.
Includes EE orientation (quaternion) so the policy can see its own gripper angle.
"""

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def ball_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position relative to robot base frame."""
    robot = env.scene[robot_cfg.name]
    ball = env.scene[ball_cfg.name]
    return ball.data.root_pos_w - robot.data.root_pos_w


def ball_position_relative_to_ee(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """Ball position relative to end-effector center."""
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_tf.data.target_pos_w[..., 0, :]
    ball_pos = env.scene[ball_cfg.name].data.root_pos_w
    return ball_pos - ee_pos


def bucket_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    bucket_pos: tuple[float, float, float] = (0.35, 0.30, 0.47),
) -> torch.Tensor:
    """Bucket target location relative to robot base in each environment."""
    robot = env.scene[robot_cfg.name]
    bucket_w = env.scene.env_origins + torch.tensor(bucket_pos, device=env.device).unsqueeze(0)
    return bucket_w - robot.data.root_pos_w


def bucket_position_relative_to_ee(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    bucket_pos: tuple[float, float, float] = (0.35, 0.30, 0.47),
) -> torch.Tensor:
    """Bucket position relative to end-effector in each environment."""
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_tf.data.target_pos_w[..., 0, :]
    bucket_w = env.scene.env_origins + torch.tensor(bucket_pos, device=env.device).unsqueeze(0)
    return bucket_w - ee_pos


def ee_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """End-effector position in robot root frame."""
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot = env.scene[robot_cfg.name]
    return ee_tf.data.target_pos_w[..., 0, :] - robot.data.root_pos_w


def ee_orientation_in_world(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """End-effector orientation quaternion in world frame (wxyz, 4 dims).
    
    This lets the policy directly observe its own gripper angle instead of
    having to infer it from joint positions through implicit forward kinematics.
    Critical for learning the downward-orientation reward efficiently.

    NOTE: Isaac Lab quaternions are (w, x, y, z), NOT (x, y, z, w).
    """
    ee_tf: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_tf.data.target_quat_w.torch[..., 0, :]  # (N, 4) wxyz format
