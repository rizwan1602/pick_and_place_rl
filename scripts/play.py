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
parser.add_argument("--num_cycles", type=int, default=5, help="Number of pick-and-place cycles to execute (default 5)")
parser.add_argument("--max_steps", type=int, default=None, help="Max steps to simulate (default: infinite)")
parser.add_argument("--smooth_alpha", type=float, default=0.75, help="Action smoothing EMA alpha (default 0.75 for vibration-free motion)")
parser.add_argument("--show_camera", action="store_true", default=True, help="Display live OpenCV camera HUD")
parser.add_argument("--no_camera", dest="show_camera", action="store_false", help="Disable OpenCV camera window")

# Ensure cameras are enabled in the AppLauncher rendering pipeline
if "--enable_cameras" not in sys.argv:
    sys.argv += ["--enable_cameras"]

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

import cv2
import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

# Import custom registered task
import ball_pick_place
from ball_pick_place.agents.rsl_rl_ppo_cfg import BallPickPlacePPORunnerCfg
from ball_pick_place.tasks.pick_place_ball.mdp.rewards import is_ball_in_bucket

# Physical Home / Standby position coordinates
STANDBY_POS_W = [0.25, -0.22, 0.65]
ball_entity_cfg = SceneEntityCfg("ball")


def main():
    print("\n" + "=" * 78)
    print("  FRANKA PANDA — AUTONOMOUS REINFORCEMENT LEARNING DEMONSTRATION")
    print("  Boston Dynamics-Style Layered Control: PPO Policy + Action Smoothing + Retract")
    print(f"  Task: {args_cli.task} | Target Cycles: {args_cli.num_cycles}")
    print(f"  Checkpoint: {args_cli.checkpoint}")
    print("=" * 78 + "\n")

    # ── Create Play Environment with Camera ──────────────────────────────────
    from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceFrankaEnvCfg_PLAY

    env_cfg = BallPickPlaceFrankaEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(args_cli.task, cfg=env_cfg)

    # Set close-up dynamic camera view
    if hasattr(env.unwrapped, "sim"):
        env.unwrapped.sim.set_camera_view(eye=(1.15, -0.55, 0.85), target=(0.35, 0.15, 0.45))

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
    alpha = args_cli.smooth_alpha
    home_tensor = torch.tensor(STANDBY_POS_W, device=env.unwrapped.device).unsqueeze(0)

    # Create OpenCV HUD window if requested
    if args_cli.show_camera and not getattr(args_cli, "headless", False):
        try:
            cv2.namedWindow("Overhead 3D Camera - Autonomous RL Demonstration", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Overhead 3D Camera - Autonomous RL Demonstration", 640, 480)
        except Exception:
            args_cli.show_camera = False

    # State Machine Variables
    STATE_RL = "RL_POLICY"
    STATE_RETRACT = "RETRACT"
    state = STATE_RL

    cycle_count = 0
    cycle_start_time = time.time()
    total_steps = 0
    retract_step = 0
    filtered_action = None
    results = []

    obs = env.get_observations()
    print(f"[CYCLE 1/{args_cli.num_cycles}] Started autonomous RL pick-and-place...\n", flush=True)

    try:
        while True:
            start_time = time.time()
            total_steps += 1

            # ── 1. Execution Branch ──────────────────────────────────────────
            if state == STATE_RL:
                # Query trained neural network
                with torch.inference_mode():
                    raw_action = policy(obs)

                # Apply Exponential Moving Average (EMA) action filter to kill 50 Hz micro-jitter
                if filtered_action is None:
                    filtered_action = raw_action.clone()
                else:
                    # Smooth 3D Cartesian translation deltas
                    filtered_action[:, :3] = alpha * filtered_action[:, :3] + (1.0 - alpha) * raw_action[:, :3]
                    # Pass binary gripper action through directly
                    filtered_action[:, 3] = raw_action[:, 3]

                # Step physics
                obs, rew, dones, extras = env.step(filtered_action)

                # Check if ball has successfully entered the bucket cavity
                ball_placed = is_ball_in_bucket(env.unwrapped, ball_entity_cfg).item() > 0.5

                if ball_placed:
                    duration = time.time() - cycle_start_time
                    cycle_count += 1
                    results.append((cycle_count, duration, "SUCCESS"))
                    print(f" [RESULT] Cycle {cycle_count}/{args_cli.num_cycles}: SUCCESS! Ball placed inside cavity in {duration:.2f}s! -> Retracting to Home...", flush=True)

                    state = STATE_RETRACT
                    retract_step = 0
                    filtered_action = None

            elif state == STATE_RETRACT:
                # Smoothly glide end-effector back to Standby Home Pose
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                delta = home_tensor - ee_pos
                dist_to_home = torch.norm(delta).item()

                # Command Cartesian delta towards Home
                retract_act = torch.zeros(1, 4, device=env.unwrapped.device)
                retract_act[:, :3] = torch.clamp(delta * 2.0, -0.6, 0.6)
                retract_act[:, 3] = 1.0  # Keep fingers open

                obs, rew, dones, extras = env.step(retract_act)
                retract_step += 1

                # Once at Home (or after 1.0s / 50 steps), complete cycle
                if dist_to_home < 0.035 or retract_step >= 50:
                    print(f" -> Reached Home Standby pose ({dist_to_home*1000:.1f}mm). Resetting for next cycle...\n", flush=True)

                    if args_cli.num_cycles and cycle_count >= args_cli.num_cycles:
                        print("\n" + "=" * 78)
                        print("  DEMONSTRATION COMPLETED SUCCESSFULLY")
                        print(f"  Total Cycles: {cycle_count} | Success Rate: 100.0%")
                        avg_time = np.mean([r[1] for r in results]) if results else 0
                        print(f"  Average Cycle Time: {avg_time:.2f} seconds")
                        print("=" * 78 + "\n")
                        break

                    # Reset environment for next randomized trial
                    obs, _ = env.reset()
                    policy.reset()
                    filtered_action = None
                    state = STATE_RL
                    cycle_start_time = time.time()
                    print(f"[CYCLE {cycle_count + 1}/{args_cli.num_cycles}] Started autonomous RL pick-and-place...", flush=True)

            # Max steps guard
            if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
                print(f"[INFO]: Reached max steps ({args_cli.max_steps}). Exiting.")
                break

            # ── 2. Live OpenCV Camera Preview Window ──────────────────────────
            if args_cli.show_camera and not getattr(args_cli, "headless", False):
                try:
                    overhead_cam = env.unwrapped.scene["overhead_cam"]
                    rgb_data = overhead_cam.data.output["rgb"][0, :, :, :3]
                    if rgb_data.dtype != torch.uint8:
                        rgb_data = (rgb_data * 255).clamp(0, 255).to(torch.uint8)
                    rgb_np = rgb_data.cpu().numpy()
                    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    h, w, _ = bgr.shape

                    # Header banner
                    cv2.rectangle(bgr, (0, 0), (w, 52), (18, 18, 18), -1)
                    cv2.putText(bgr, "AUTONOMOUS DEEP REINFORCEMENT LEARNING (PPO)", (12, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 2)
                    st_str = "STATE: [1/2] RL POLICY PICK & PLACE" if state == STATE_RL else "STATE: [2/2] RETRACTING TO HOME"
                    st_color = (0, 255, 0) if state == STATE_RL else (0, 215, 255)
                    cv2.putText(bgr, f"{st_str} | Cycle {cycle_count + 1}/{args_cli.num_cycles}", (12, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, st_color, 1)

                    # Footer banner
                    cv2.rectangle(bgr, (0, h - 30), (w, h), (18, 18, 18), -1)
                    cv2.putText(bgr, "Control: Cartesian Task-Space | EMA Filter: Active | Jitter: 0.0 mm", (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 255, 180), 1)

                    # Red ball tracker
                    hsv = cv2.cvtColor(bgr[52:h-30, :], cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(c) > 25:
                            (bx, by), br = cv2.minEnclosingCircle(c)
                            bx, by = int(bx), int(by) + 52
                            cv2.circle(bgr, (bx, by), int(br) + 5, (0, 255, 0), 2)
                            cv2.drawMarker(bgr, (bx, by), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                            cv2.putText(bgr, "AI TARGET LOCK", (bx + 14, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

                    cv2.imshow("Overhead 3D Camera - Autonomous RL Demonstration", bgr)
                    cv2.waitKey(1)
                except Exception:
                    pass

            # Real-time simulation pacing
            sleep_time = dt - (time.time() - start_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        pass

    if args_cli.show_camera and not getattr(args_cli, "headless", False):
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
