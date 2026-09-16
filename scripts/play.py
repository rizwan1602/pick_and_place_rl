# SPDX-License-Identifier: Apache-2.0
"""
Visualize / Evaluate Trained Ball Pick-and-Place Policy.
========================================================
Usage:
    isaaclab.bat -p scripts/play.py --checkpoint logs/.../model_1499.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── 1. Suppress GitPython noise ──────────────────────────────────────────────
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# ── 2. Parse arguments & launch simulation via AppLauncher ───────────────────
parser = argparse.ArgumentParser(description="Play Ball Pick-Place Franka Policy")
parser.add_argument("--task", type=str, default="BallPickPlace-Franka-Play-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments for visualization (default 1)")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained checkpoint (.pt)")

# In Isaac Lab 3.0 / Isaac Sim 6.0, AppLauncher defaults to headless unless --viz kit is passed.
# For play.py, default to GUI display (--viz kit) unless explicitly overridden.
if not any(arg in sys.argv for arg in ["--viz", "--visualizer", "--headless"]):
    sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 3. Imports AFTER SimulationApp ───────────────────────────────────────────
import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Import our custom registered task
import ball_pick_place
from ball_pick_place.agents.rsl_rl_ppo_cfg import BallPickPlacePPORunnerCfg


def main():
    print("\n" + "=" * 70)
    print("  FRANKA BALL PICK-AND-PLACE — POLICY EVALUATION")
    print(f"  Task: {args_cli.task} | Envs: {args_cli.num_envs}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("=" * 70 + "\n")

    # Create play environment
    from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceFrankaEnvCfg_PLAY

    env_cfg = BallPickPlaceFrankaEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Set camera view directly in front of the robot and table
    if hasattr(env.unwrapped, "sim"):
        env.unwrapped.sim.set_camera_view(eye=(1.2, -0.6, 0.85), target=(0.35, 0.15, 0.45))

    env = RslRlVecEnvWrapper(env)

    # Build runner and load trained weights
    agent_cfg = BallPickPlacePPORunnerCfg()
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt

    # Reset environment and get initial observations
    obs = env.get_observations()

    # Simulate — matches official Isaac Lab play.py pattern
    print("[INFO]: Playing... Press Ctrl+C to stop.\n")
    step_count = 0
    try:
        while True:
            start_time = time.time()
            with torch.inference_mode():
                actions = policy(obs)
                obs, rew, dones, extras = env.step(actions)
                policy.reset(dones)

            step_count += 1
            if step_count % 100 == 0:
                print(f"  Step {step_count:>6d} | reward: {rew.mean().item():.4f} | dones: {dones.sum().item():.0f}")

            # Real-time pacing
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
