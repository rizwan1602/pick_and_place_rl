# SPDX-License-Identifier: Apache-2.0
"""Custom Action Terms for Franka Ball Pick and Place.

Provides LockedOrientationDifferentialIKAction which allows the RL policy to command
3D Cartesian position deltas (dX, dY, dZ) while the IK controller solves for the FULL
6-DOF pose with rigidly locked vertical downward orientation (QUAT_VERTICAL).

This completely eliminates nullspace joint drift and wrist tilting (e.g. 45-degree tilt)
without adding orientation search complexity to the RL policy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class LockedOrientationDifferentialIKAction(DifferentialInverseKinematicsAction):
    """Differential IK action with 3D translation actions and locked downward orientation.
    
    The policy action space is exactly 3 dimensions (dX, dY, dZ).
    The underlying Differential IK controller solves against a full 6-DOF target:
      - Desired Position = current_pos + scale * action
      - Desired Orientation = [1.0, 0.0, 0.0, 0.0] (strictly vertical pointing down)
    
    This constrains all 6 DOFs of the 7-DOF Franka Panda arm, ensuring zero wrist tilt
    and perfectly vertical grasps from directly above the ball.
    """

    def __init__(self, cfg: LockedOrientationDifferentialIKActionCfg, env: ManagerBasedEnv):
        # Enforce full pose mode in the controller configuration
        cfg.controller.command_type = "pose"
        cfg.controller.use_relative_mode = False

        # Initialize base class (sets up articulation, body_idx, controller)
        super().__init__(cfg, env)

        # Pre-allocate full 7D command tensor: [x, y, z, qw, qx, qy, qz]
        self._full_ik_command = torch.zeros(self.num_envs, 7, device=self.device)

        # Downward vertical quaternion in robot base frame: [w=1.0, x=0.0, y=0.0, z=0.0]
        self._quat_vertical = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1)
        self._full_ik_command[:, 3:7] = self._quat_vertical

        # Reconfigure action tensors for 3D translation policy input
        self._raw_actions = torch.zeros(self.num_envs, 3, device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # Scale tensor of shape (num_envs, 3)
        self._scale = torch.zeros((self.num_envs, 3), device=self.device)
        if isinstance(cfg.scale, (float, int)):
            self._scale[:] = float(cfg.scale)
        else:
            self._scale[:] = torch.tensor(cfg.scale[:3], device=self.device)

    @property
    def action_dim(self) -> int:
        """The action dimension exposed to the policy is 3 (dX, dY, dZ)."""
        return 3

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self._raw_actions * self._scale

        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=-1.0, max=1.0
            )

        # Obtain current end-effector pose in robot base frame
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()

        # Desired position is current position + scaled delta
        self._full_ik_command[:, :3] = ee_pos_curr + self._processed_actions
        # Desired orientation is rigidly locked to downward vertical
        self._full_ik_command[:, 3:7] = self._quat_vertical

        # Set 7D command into the IK controller
        self._ik_controller.set_command(self._full_ik_command, ee_pos_curr, ee_quat_curr)


@configclass
class LockedOrientationDifferentialIKActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for locked-orientation differential IK action."""
    class_type: type[LockedOrientationDifferentialIKAction] = LockedOrientationDifferentialIKAction
