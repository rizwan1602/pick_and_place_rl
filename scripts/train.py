# SPDX-License-Identifier: Apache-2.0
"""
Train Ball Pick-and-Place Policy using RSL-RL (PPO).
====================================================
Usage:
    isaaclab.bat -p scripts/train.py --task BallPickPlace-Franka-v0 --num_envs 2048 --headless
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

# ── 1. Suppress GitPython noise (git.exe may not be on Isaac Sim's PATH) ─────
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# ── 2. Parse arguments & launch simulation via AppLauncher ───────────────────
parser = argparse.ArgumentParser(description="Train Ball Pick-Place Franka Policy")
parser.add_argument("--task", type=str, default="BallPickPlace-Franka-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=2048, help="Number of parallel environments")
parser.add_argument("--max_iterations", type=int, default=1500, help="Max RL training iterations")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint to resume training from")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--log_dir", type=str, default="logs/rsl_rl", help="Directory for logs")

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

# Performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    print("\n" + "=" * 70)
    print("  STARTING FRANKA BALL PICK-AND-PLACE RL TRAINING")
    print(f"  Task: {args_cli.task} | Envs: {args_cli.num_envs} | Iterations: {args_cli.max_iterations}")
    print("=" * 70 + "\n")

    # Create environment
    from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceFrankaEnvCfg

    env_cfg = BallPickPlaceFrankaEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    # Agent config — handle deprecated fields for installed rsl-rl version
    agent_cfg = BallPickPlacePPORunnerCfg()
    agent_cfg.max_iterations = args_cli.max_iterations

    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    # Log directory
    log_dir = os.path.join(args_cli.log_dir, agent_cfg.experiment_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[INFO] Logging experiment in directory: {os.path.abspath(log_dir)}")

    # Initialize RSL-RL Runner
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)

    # Resume from checkpoint if provided
    if args_cli.checkpoint:
        print(f"[INFO] Resuming training from checkpoint: {args_cli.checkpoint}")
        runner.load(args_cli.checkpoint)

    # Train
    start_time = time.time()
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    print(f"\n[SUCCESS] Training finished in {round(time.time() - start_time, 2)}s! Model saved to: {log_dir}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
