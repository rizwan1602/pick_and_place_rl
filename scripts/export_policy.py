# SPDX-License-Identifier: Apache-2.0
"""
Export Trained Policy to Standalone TorchScript JIT (.pt).
=========================================================
Usage:
    isaaclab.bat -p scripts/export_policy.py --checkpoint logs/.../model_1500.pt --output exports/policy.pt
"""

from __future__ import annotations

import argparse
import os

# ── 1. Suppress GitPython noise ──────────────────────────────────────────────
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# ── 2. Parse arguments & launch simulation via AppLauncher ───────────────────
parser = argparse.ArgumentParser(description="Export Trained Policy to TorchScript JIT")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.pt)")
parser.add_argument("--output", type=str, default="exports/policy.pt", help="Path for exported TorchScript model")

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 3. Imports AFTER SimulationApp ───────────────────────────────────────────
import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, export_policy_as_jit, handle_deprecated_rsl_rl_cfg

# Import our custom registered task
import ball_pick_place
from ball_pick_place.agents.rsl_rl_ppo_cfg import BallPickPlacePPORunnerCfg
from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceFrankaEnvCfg_PLAY


def main():
    print(f"\n[INFO] Loading checkpoint from: {args_cli.checkpoint}")

    # Create a minimal env to get observation/action dims
    env_cfg = BallPickPlaceFrankaEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1

    env = gym.make("BallPickPlace-Franka-Play-v0", cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Build runner and load weights
    agent_cfg = BallPickPlacePPORunnerCfg()
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(args_cli.checkpoint)

    # Export actor network as TorchScript JIT using built-in runner method
    export_dir = os.path.dirname(os.path.abspath(args_cli.output))
    filename = os.path.basename(args_cli.output)
    os.makedirs(export_dir, exist_ok=True)

    runner.export_policy_to_jit(path=export_dir, filename=filename)
    final_out = os.path.join(export_dir, filename)

    print(f"\n[SUCCESS] Standalone TorchScript JIT policy exported to: {final_out}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
