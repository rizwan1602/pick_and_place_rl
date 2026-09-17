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

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# -- 1. Suppress GitPython noise ----------------------------------------------
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
from ball_pick_place.tasks.pick_place_ball.mdp.geometry import (
    BUCKET_X,
    BUCKET_Y,
    BUCKET_RIM_TOP_Z,
    BALL_REST_Z_IN_BUCKET,
    HAND_HOVER_Z,
    IN_BUCKET_XY_HALF,
    IN_BUCKET_Z_MIN,
    IN_BUCKET_Z_MAX,
    IN_BUCKET_MAX_SPEED,
    TABLE_TOP_Z,
    BALL_RADIUS,
)

# Physical Home / Standby position coordinates (identical to run_deterministic_ik.py)
STANDBY_POS_W = [0.35, 0.00, 0.65]
ball_entity_cfg = SceneEntityCfg("ball")

# ── PLC Control System & Button Coordinates ──────────────────────────────────
PLC_IDLE = "IDLE"           # Parked at Home, awaiting START command
PLC_RUNNING = "RUNNING"     # Actively executing pick-and-place
PLC_STOPPED = "STOPPED"     # Motion halted (arm holds position)
PLC_RESETTING = "RESETTING" # Retracting arm to Home Standby pose

# Button bounding boxes on OpenCV window (x, y, w, h) - compact in top right corner
BTN_START = (315, 14, 76, 40)
BTN_STOP  = (396, 14, 72, 40)
BTN_RESET = (473, 14, 76, 40)
BTN_BALL  = (554, 14, 76, 40)

plc_pending_cmd = None


def on_mouse_click(event, x, y, flags, param):
    """Mouse click handler for OpenCV PLC control buttons and table click-to-place."""
    global plc_pending_cmd
    if event == cv2.EVENT_LBUTTONDOWN:
        if BTN_START[0] <= x <= BTN_START[0] + BTN_START[2] and BTN_START[1] <= y <= BTN_START[1] + BTN_START[3]:
            plc_pending_cmd = "START"
        elif BTN_STOP[0] <= x <= BTN_STOP[0] + BTN_STOP[2] and BTN_STOP[1] <= y <= BTN_STOP[1] + BTN_STOP[3]:
            plc_pending_cmd = "STOP"
        elif BTN_RESET[0] <= x <= BTN_RESET[0] + BTN_RESET[2] and BTN_RESET[1] <= y <= BTN_RESET[1] + BTN_RESET[3]:
            plc_pending_cmd = "RESET"
        elif BTN_BALL[0] <= x <= BTN_BALL[0] + BTN_BALL[2] and BTN_BALL[1] <= y <= BTN_BALL[1] + BTN_BALL[3]:
            plc_pending_cmd = "SPAWN_BALL"
        elif 70 <= y <= 450:
            # Click directly on the camera view to place ball at clicked position!
            norm_x = (x - 320) / 320.0
            norm_y = (y - 260) / 200.0
            target_y = float(np.clip(-norm_x * 0.22, -0.15, 0.15))
            target_x = float(np.clip(0.40 - norm_y * 0.16, 0.28, 0.44))
            plc_pending_cmd = ("MOVE_BALL", target_x, target_y)


class ActionSmoother:
    """Action Space Smoother: Eliminates 50 Hz micro-jitter while maintaining 100% full speed & torque.
    
    Filters directly in the policy's action space [-1.0, 1.0] to prevent velocity choking,
    preserving full motor authority so the Franka Panda moves at full industrial speed.
    """

    def __init__(self, alpha: float = 0.65):
        self.alpha = alpha
        self.filtered_action = None

    def reset(self):
        self.filtered_action = None

    def filter(self, raw_action: torch.Tensor) -> torch.Tensor:
        if self.filtered_action is None:
            self.filtered_action = raw_action.clone()
        else:
            self.filtered_action = self.alpha * self.filtered_action + (1.0 - self.alpha) * raw_action
        return self.filtered_action


def main():
    global plc_pending_cmd

    print("\n" + "=" * 78)
    print("  FRANKA PANDA - AUTONOMOUS ROBOTIC CELL WITH INDUSTRIAL PLC PANEL")
    print("  PLC Status: [ONLINE] (Bright Green Indicator)")
    print("  Controls:")
    print("    [START] (Click button or press 'S') : Picks ball & places into bucket")
    print("    [STOP]  (Click button or press Space) : Immediately halts robot motion")
    print("    [RESET] (Click button or press 'R') : Returns arm directly to Home Standby")
    print("    [SPAWN] (Press 'B') : Randomize ball on table (or place manually in viewport)")
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

    # ── Automatically Load & Enable Siemens PLC Bridge Extension ─────────────
    cell_bridge = None
    try:
        import omni.kit.app
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_folder = os.path.abspath("extensions")
        ext_mgr.add_path(ext_folder)
        if not ext_mgr.is_extension_enabled("com.rizwan.plc_bridge"):
            ext_mgr.set_extension_enabled_immediate("com.rizwan.plc_bridge", True)
        print("[PLC] Siemens PLC Bridge Extension (com.rizwan.plc_bridge) loaded successfully!")
    except Exception as e:
        print(f"[PLC] Note on extension loading: {e}")

    # Connect to shared state bridge
    sys.path.insert(0, os.path.abspath("extensions/com.rizwan.plc_bridge"))
    try:
        from plc_bridge.bridge_state import RobotCellBridge
        cell_bridge = RobotCellBridge.get()
        cell_bridge.plc_online = True
        cell_bridge.robot_state = PLC_IDLE
        cell_bridge.notify()
    except Exception as e:
        print(f"[PLC] Note on bridge state: {e}")

    # Build runner and load trained weights
    agent_cfg = BallPickPlacePPORunnerCfg()
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)

    print(f"[INFO]: Loading model checkpoint from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    home_tensor = torch.tensor(STANDBY_POS_W, device=env.unwrapped.device).unsqueeze(0)
    smoother = ActionSmoother(alpha=0.65)

    # Create OpenCV HUD window with mouse callback for PLC buttons
    if args_cli.show_camera and not getattr(args_cli, "headless", False):
        try:
            cv2.namedWindow("Overhead 3D Camera - Autonomous RL Demonstration", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Overhead 3D Camera - Autonomous RL Demonstration", 640, 480)
            cv2.setWindowProperty("Overhead 3D Camera - Autonomous RL Demonstration", cv2.WND_PROP_TOPMOST, 1)
            cv2.setMouseCallback("Overhead 3D Camera - Autonomous RL Demonstration", on_mouse_click)
        except Exception:
            args_cli.show_camera = False

    # State Machine Variables
    STATE_RL = "RL_POLICY"
    STATE_RELEASE = "RELEASE_AND_LIFT"
    STATE_SETTLE = "VERIFY_SETTLE"
    STATE_RETRACT = "RETRACT_HOME"
    state = STATE_RL

    # PLC State: Starts parked at Home, waiting for user to click START
    plc_state = PLC_IDLE
    last_gripper_cmd = 1.0

    cycle_count = 0
    cycle_start_time = time.time()
    total_steps = 0
    release_step = 0
    settle_step = 0
    consecutive_settled = 0
    retract_step = 0
    results = []

    obs = env.get_observations()
    print("[PLC] System initialized in IDLE mode. Place ball on table and click [START]...\n", flush=True)

    try:
        while True:
            total_steps += 1

            # ── 0. Poll Siemens PLC Bridge Extension UI & OPC UA Commands ───
            if cell_bridge:
                b_cmd = cell_bridge.pop_command()
                if b_cmd:
                    plc_pending_cmd = b_cmd

            # ── 1. Process Hotkeys & Mouse PLC Commands ──────────────────────
            key = cv2.waitKey(1) & 0xFF if (args_cli.show_camera and not getattr(args_cli, "headless", False)) else 255
            if key in [ord('s'), ord('S')]:
                plc_pending_cmd = "START"
            elif key in [ord(' '), ord('x'), ord('X')]:
                plc_pending_cmd = "STOP"
            elif key in [ord('r'), ord('R')]:
                plc_pending_cmd = "RESET"
            elif key in [ord('b'), ord('B')]:
                # Randomize ball position on table
                rx = float(torch.empty(1).uniform_(0.28, 0.42).item())
                ry = float(torch.empty(1).uniform_(-0.12, 0.12).item())
                rz = TABLE_TOP_Z + BALL_RADIUS + 0.002
                ball = env.unwrapped.scene["ball"]
                env_origins = env.unwrapped.scene.env_origins
                b_state = ball.data.default_root_state.clone()
                b_state[:, 0] = rx + env_origins[0, 0]
                b_state[:, 1] = ry + env_origins[0, 1]
                b_state[:, 2] = rz + env_origins[0, 2]
                b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                b_state[:, 7:13] = 0.0
                ball.write_root_state_to_sim(b_state)
                if cell_bridge:
                    cell_bridge.ball_x = rx
                    cell_bridge.ball_y = ry
                    cell_bridge.status_message = f"Ball randomized on table at X={rx:.3f}m, Y={ry:.3f}m"
                    cell_bridge.notify()
                print(f"[PLC] Spawned ball on table at X={rx:.3f}m, Y={ry:.3f}m", flush=True)
            elif key in [27, ord('q'), ord('Q')]:
                print("[INFO]: Quit requested. Exiting demonstration.")
                break

            if plc_pending_cmd:
                cmd = plc_pending_cmd
                plc_pending_cmd = None

                if isinstance(cmd, tuple) and cmd[0] == "MOVE_BALL":
                    _, rx, ry = cmd
                    rz = TABLE_TOP_Z + BALL_RADIUS + 0.002
                    ball = env.unwrapped.scene["ball"]
                    env_origins = env.unwrapped.scene.env_origins
                    b_state = ball.data.default_root_state.clone()
                    b_state[:, 0] = rx + env_origins[0, 0]
                    b_state[:, 1] = ry + env_origins[0, 1]
                    b_state[:, 2] = rz + env_origins[0, 2]
                    b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                    b_state[:, 7:13] = 0.0
                    ball.write_root_state_to_sim(b_state)
                    if cell_bridge:
                        cell_bridge.ball_x = rx
                        cell_bridge.ball_y = ry
                        cell_bridge.status_message = f"Ball placed at X={rx:.3f}m, Y={ry:.3f}m"
                        cell_bridge.notify()
                    print(f"[PLC] Ball positioned at click: X={rx:.3f}m, Y={ry:.3f}m", flush=True)

                elif cmd == "SPAWN_BALL":
                    rx = float(torch.empty(1).uniform_(0.28, 0.42).item())
                    ry = float(torch.empty(1).uniform_(-0.12, 0.12).item())
                    rz = TABLE_TOP_Z + BALL_RADIUS + 0.002
                    ball = env.unwrapped.scene["ball"]
                    env_origins = env.unwrapped.scene.env_origins
                    b_state = ball.data.default_root_state.clone()
                    b_state[:, 0] = rx + env_origins[0, 0]
                    b_state[:, 1] = ry + env_origins[0, 1]
                    b_state[:, 2] = rz + env_origins[0, 2]
                    b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                    b_state[:, 7:13] = 0.0
                    ball.write_root_state_to_sim(b_state)
                    if cell_bridge:
                        cell_bridge.ball_x = rx
                        cell_bridge.ball_y = ry
                        cell_bridge.status_message = f"Ball spawned at X={rx:.3f}m, Y={ry:.3f}m"
                        cell_bridge.notify()
                    print(f"[PLC] Ball spawned at random table spot: X={rx:.3f}m, Y={ry:.3f}m", flush=True)

                elif cmd == "START":
                    # If ball is already in the bucket when START is pressed, place it on the table
                    ball_pos_env = env.unwrapped.scene["ball"].data.root_pos_w - env.unwrapped.scene.env_origins
                    inside_x = abs(ball_pos_env[0, 0].item() - BUCKET_X) < IN_BUCKET_XY_HALF
                    inside_y = abs(ball_pos_env[0, 1].item() - BUCKET_Y) < IN_BUCKET_XY_HALF
                    if inside_x and inside_y:
                        rx = float(torch.empty(1).uniform_(0.28, 0.42).item())
                        ry = float(torch.empty(1).uniform_(-0.12, 0.12).item())
                        rz = TABLE_TOP_Z + BALL_RADIUS + 0.002
                        ball = env.unwrapped.scene["ball"]
                        env_origins = env.unwrapped.scene.env_origins
                        b_state = ball.data.default_root_state.clone()
                        b_state[:, 0] = rx + env_origins[0, 0]
                        b_state[:, 1] = ry + env_origins[0, 1]
                        b_state[:, 2] = rz + env_origins[0, 2]
                        b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                        b_state[:, 7:13] = 0.0
                        ball.write_root_state_to_sim(b_state)
                        print(f"[PLC] Ball auto-spawned onto table for new cycle: X={rx:.3f}m, Y={ry:.3f}m", flush=True)

                    if plc_state != PLC_RUNNING:
                        plc_state = PLC_RUNNING
                        state = STATE_RL
                        cycle_start_time = time.time()
                        smoother.reset()
                        policy.reset()
                        if cell_bridge:
                            cell_bridge.robot_state = plc_state
                            cell_bridge.status_message = f"Cycle {cycle_count + 1} initiated -> Running Pick & Place"
                            cell_bridge.notify()
                        print(f"\n[PLC PANEL] START: Cycle {cycle_count + 1} initiated!", flush=True)

                elif cmd == "STOP":
                    plc_state = PLC_STOPPED
                    if cell_bridge:
                        cell_bridge.robot_state = plc_state
                        cell_bridge.status_message = "Robot halted (Hold position)"
                        cell_bridge.notify()
                    print("\n[PLC PANEL] STOP: Arm motion halted. Holding position.", flush=True)

                elif cmd == "RESET":
                    plc_state = PLC_RESETTING
                    smoother.reset()
                    policy.reset()
                    state = STATE_RL

                    # 1. Erase ball from container and respawn anywhere on the table
                    rx = float(torch.empty(1).uniform_(0.28, 0.42).item())
                    ry = float(torch.empty(1).uniform_(-0.12, 0.12).item())
                    rz = TABLE_TOP_Z + BALL_RADIUS + 0.002
                    ball = env.unwrapped.scene["ball"]
                    env_origins = env.unwrapped.scene.env_origins
                    b_state = ball.data.default_root_state.clone()
                    b_state[:, 0] = rx + env_origins[0, 0]
                    b_state[:, 1] = ry + env_origins[0, 1]
                    b_state[:, 2] = rz + env_origins[0, 2]
                    b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
                    b_state[:, 7:13] = 0.0
                    ball.write_root_state_to_sim(b_state)

                    if cell_bridge:
                        cell_bridge.robot_state = plc_state
                        cell_bridge.ball_x = rx
                        cell_bridge.ball_y = ry
                        cell_bridge.status_message = f"RESET ALL: Ball on table (X={rx:.2f}, Y={ry:.2f}) | Homing arm..."
                        cell_bridge.notify()
                    print(f"\n[PLC PANEL] RESET ALL: Ball removed from container & placed on table (X={rx:.3f}m, Y={ry:.3f}m). Homing Franka Panda...", flush=True)

            # ── 2. PLC Execution Branch ──────────────────────────────────────
            if plc_state == PLC_STOPPED:
                # Arm holds current position firmly
                stop_act = torch.zeros(1, 4, device=env.unwrapped.device)
                stop_act[:, 3] = last_gripper_cmd
                obs, rew, dones, extras = env.step(stop_act)

            elif plc_state == PLC_IDLE:
                # Parked at Home Standby pose, fingers open
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                delta = home_tensor - ee_pos
                idle_act = torch.zeros(1, 4, device=env.unwrapped.device)
                idle_act[:, :3] = torch.clamp(delta * 2.0, -0.3, 0.3)
                idle_act[:, 3] = 1.0
                last_gripper_cmd = 1.0
                obs, rew, dones, extras = env.step(idle_act)

            elif plc_state == PLC_RESETTING:
                # Retract arm to Standby Home Pose
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                delta = home_tensor - ee_pos
                dist_to_home = torch.norm(delta).item()

                reset_act = torch.zeros(1, 4, device=env.unwrapped.device)
                reset_act[:, :3] = torch.clamp(delta * 2.5, -0.8, 0.8)
                reset_act[:, 3] = 1.0  # Open fingers
                last_gripper_cmd = 1.0
                obs, rew, dones, extras = env.step(reset_act)

                if dist_to_home < 0.035:
                    plc_state = PLC_IDLE
                    state = STATE_RL
                    smoother.reset()
                    policy.reset()
                    if cell_bridge:
                        cell_bridge.robot_state = plc_state
                        cell_bridge.status_message = "At Standby Home pose. Ready for START."
                        cell_bridge.notify()
                    print(f"[PLC] Reached Home Standby pose ({dist_to_home*1000:.1f}mm). Ready for START.", flush=True)

            elif plc_state == PLC_RUNNING:
                # ── Autonomous Pick & Place Sequence ─────────────────────────
                if state == STATE_RL:
                    with torch.inference_mode():
                        raw_action = policy(obs)

                    smooth_trans = smoother.filter(raw_action[:, :3])
                    act = torch.zeros(1, 4, device=env.unwrapped.device)
                    act[:, :3] = smooth_trans
                    act[:, 3] = raw_action[:, 3]

                    ball_pos_w = env.unwrapped.scene["ball"].data.root_pos_w
                    ee_pos_w = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :]
                    env_origins = env.unwrapped.scene.env_origins

                    ball_pos_env = ball_pos_w - env_origins
                    ee_pos_env = ee_pos_w - env_origins

                    bucket_xy = torch.tensor([BUCKET_X, BUCKET_Y], device=env.unwrapped.device)
                    dist_ee_bucket_xy = torch.norm(ee_pos_env[0, :2] - bucket_xy).item()
                    dist_ee_ball = torch.norm(ee_pos_w[0] - ball_pos_w[0]).item()

                    # Safety Interlock: keep closed when holding ball
                    is_holding = (ball_pos_env[0, 2].item() > 0.48) and (dist_ee_ball < 0.08)
                    if is_holding and dist_ee_bucket_xy > 0.045:
                        act[:, 3] = -1.0

                    last_gripper_cmd = act[0, 3].item()
                    obs, rew, dones, extras = env.step(act)

                    # Transition check to release
                    over_bucket = (dist_ee_bucket_xy <= 0.045) and (ee_pos_env[0, 2].item() <= 0.62)
                    in_bucket = is_ball_in_bucket(env.unwrapped, ball_entity_cfg).item() > 0.5

                    if (is_holding and over_bucket) or in_bucket:
                        state = STATE_RELEASE
                        release_step = 0

                elif state == STATE_RELEASE:
                    # Vertical release & lift above bucket
                    ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                    target_pose = torch.tensor([BUCKET_X, BUCKET_Y, 0.62], device=env.unwrapped.device).unsqueeze(0)
                    delta = target_pose - ee_pos

                    release_act = torch.zeros(1, 4, device=env.unwrapped.device)
                    release_act[:, :3] = torch.clamp(delta * 2.5, -0.6, 0.6)
                    release_act[:, 3] = 1.0  # Open fingers fully
                    last_gripper_cmd = 1.0

                    obs, rew, dones, extras = env.step(release_act)
                    release_step += 1

                    if ee_pos[0, 2].item() >= 0.58 and release_step >= 12:
                        state = STATE_SETTLE
                        settle_step = 0
                        consecutive_settled = 0

                elif state == STATE_SETTLE:
                    # Cavity settle verification
                    ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                    target_pose = torch.tensor([BUCKET_X, BUCKET_Y, 0.62], device=env.unwrapped.device).unsqueeze(0)
                    delta = target_pose - ee_pos

                    hover_act = torch.zeros(1, 4, device=env.unwrapped.device)
                    hover_act[:, :3] = torch.clamp(delta * 2.0, -0.2, 0.2)
                    hover_act[:, 3] = 1.0
                    last_gripper_cmd = 1.0

                    obs, rew, dones, extras = env.step(hover_act)
                    settle_step += 1

                    ball_pos_env = env.unwrapped.scene["ball"].data.root_pos_w - env.unwrapped.scene.env_origins
                    inside_x = abs(ball_pos_env[0, 0].item() - BUCKET_X) < IN_BUCKET_XY_HALF
                    inside_y = abs(ball_pos_env[0, 1].item() - BUCKET_Y) < IN_BUCKET_XY_HALF
                    inside_z = (ball_pos_env[0, 2].item() >= IN_BUCKET_Z_MIN) and (ball_pos_env[0, 2].item() <= IN_BUCKET_Z_MAX)
                    ball_vel_w = env.unwrapped.scene["ball"].data.root_lin_vel_w
                    ball_speed = torch.norm(ball_vel_w, dim=-1).item()

                    is_resting = inside_x and inside_y and inside_z and (ball_speed < 0.25)
                    if is_resting:
                        consecutive_settled += 1
                    else:
                        consecutive_settled = 0

                    if consecutive_settled >= 3:
                        duration = time.time() - cycle_start_time
                        cycle_count += 1
                        results.append((cycle_count, duration, "SUCCESS"))
                        print(f" [RESULT] Cycle {cycle_count}: SUCCESS! Ball placed & settled in cavity in {duration:.2f}s! -> Retracting to Home...", flush=True)
                        if cell_bridge:
                            cell_bridge.cycle_count = cycle_count
                            cell_bridge.status_message = f"Cycle {cycle_count} SUCCESS in {duration:.2f}s! Retracting..."
                            cell_bridge.notify()
                        state = STATE_RETRACT
                        retract_step = 0
                    elif settle_step >= 25:
                        duration = time.time() - cycle_start_time
                        cycle_count += 1
                        results.append((cycle_count, duration, "FAILED"))
                        print(f" [RESULT] Cycle {cycle_count}: FAILED! Proceeding to Retract...", flush=True)
                        if cell_bridge:
                            cell_bridge.cycle_count = cycle_count
                            cell_bridge.status_message = f"Cycle {cycle_count} FAILED! Retracting..."
                            cell_bridge.notify()
                        state = STATE_RETRACT
                        retract_step = 0

                elif state == STATE_RETRACT:
                    # Retract to Standby Home Pose
                    ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                    delta = home_tensor - ee_pos
                    dist_to_home = torch.norm(delta).item()

                    retract_act = torch.zeros(1, 4, device=env.unwrapped.device)
                    retract_act[:, :3] = torch.clamp(delta * 2.5, -0.8, 0.8)
                    retract_act[:, 3] = 1.0  # Open fingers
                    last_gripper_cmd = 1.0

                    obs, rew, dones, extras = env.step(retract_act)
                    retract_step += 1

                    if dist_to_home < 0.040 or retract_step >= 35:
                        print(f" -> Reached Home Standby pose ({dist_to_home*1000:.1f}mm). Ready for next cycle.\n", flush=True)
                        plc_state = PLC_IDLE
                        state = STATE_RL
                        smoother.reset()
                        policy.reset()
                        if cell_bridge:
                            cell_bridge.robot_state = plc_state
                            cell_bridge.status_message = "Ready for next cycle. Place ball and click START."
                            cell_bridge.notify()

            # Max steps guard
            if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
                print(f"[INFO]: Reached max steps ({args_cli.max_steps}). Exiting.")
                break

            # ── 3. Live OpenCV Camera Preview Window & Industrial PLC HUD ────
            if args_cli.show_camera and not getattr(args_cli, "headless", False):
                try:
                    overhead_cam = env.unwrapped.scene["overhead_cam"]
                    rgb_data = overhead_cam.data.output["rgb"][0, :, :, :3]
                    if rgb_data.dtype != torch.uint8:
                        rgb_data = (rgb_data * 255).clamp(0, 255).to(torch.uint8)
                    rgb_np = rgb_data.cpu().numpy()
                    bgr = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)
                    h, w, _ = bgr.shape

                    # ── PLC Control Panel Bar (Top, y: 0 to 68) ──────────────
                    cv2.rectangle(bgr, (0, 0), (w, 68), (24, 24, 24), -1)
                    cv2.line(bgr, (0, 68), (w, 68), (55, 55, 55), 1)

                    # Title & Green PLC Status Indicator
                    cv2.putText(bgr, "FRANKA PANDA - INDUSTRIAL PLC CELL", (14, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (220, 220, 220), 1, cv2.LINE_AA)

                    # Glowing Green PLC Status LED (Always Green!)
                    cv2.circle(bgr, (20, 47), 7, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(bgr, (20, 47), 10, (0, 180, 0), 1, cv2.LINE_AA)
                    cv2.putText(bgr, "PLC STATUS: ONLINE", (36, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0), 1, cv2.LINE_AA)

                    # State Mode Pill next to PLC status
                    if plc_state == PLC_RUNNING:
                        st_text = f"RUNNING [{state}]"
                        st_col = (0, 255, 255)
                    elif plc_state == PLC_IDLE:
                        st_text = "READY (Place ball -> Click START)"
                        st_col = (0, 255, 140)
                    elif plc_state == PLC_STOPPED:
                        st_text = "STOPPED (Arm Hold)"
                        st_col = (0, 100, 255)
                    elif plc_state == PLC_RESETTING:
                        st_text = "HOMING (Reset)"
                        st_col = (255, 200, 0)
                    else:
                        st_text = plc_state
                        st_col = (200, 200, 200)

                    cv2.putText(bgr, f"| {st_text}", (195, 51), cv2.FONT_HERSHEY_SIMPLEX, 0.38, st_col, 1, cv2.LINE_AA)

                    # 4 Clickable Buttons in One Single Place
                    # 1. START Button
                    bx1, by1, bw1, bh1 = BTN_START
                    start_bg = (20, 75, 20) if plc_state == PLC_RUNNING else (16, 42, 16)
                    start_border = (0, 255, 0) if plc_state == PLC_RUNNING else (0, 180, 0)
                    cv2.rectangle(bgr, (bx1, by1), (bx1 + bw1, by1 + bh1), start_bg, -1)
                    cv2.rectangle(bgr, (bx1, by1), (bx1 + bw1, by1 + bh1), start_border, 2)
                    cv2.putText(bgr, "START", (bx1 + 10, by1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 255, 0) if plc_state != PLC_RUNNING else (255, 255, 255), 2, cv2.LINE_AA)

                    # 2. STOP Button
                    bx2, by2, bw2, bh2 = BTN_STOP
                    stop_bg = (20, 20, 80) if plc_state == PLC_STOPPED else (18, 18, 40)
                    stop_border = (0, 0, 255) if plc_state == PLC_STOPPED else (0, 0, 180)
                    cv2.rectangle(bgr, (bx2, by2), (bx2 + bw2, by2 + bh2), stop_bg, -1)
                    cv2.rectangle(bgr, (bx2, by2), (bx2 + bw2, by2 + bh2), stop_border, 2)
                    cv2.putText(bgr, "STOP", (bx2 + 12, by2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (60, 60, 255) if plc_state != PLC_STOPPED else (255, 255, 255), 2, cv2.LINE_AA)

                    # 3. RESET Button
                    bx3, by3, bw3, bh3 = BTN_RESET
                    reset_bg = (70, 50, 15) if plc_state == PLC_RESETTING else (35, 28, 12)
                    reset_border = (255, 200, 0) if plc_state == PLC_RESETTING else (180, 140, 0)
                    cv2.rectangle(bgr, (bx3, by3), (bx3 + bw3, by3 + bh3), reset_bg, -1)
                    cv2.rectangle(bgr, (bx3, by3), (bx3 + bw3, by3 + bh3), reset_border, 2)
                    cv2.putText(bgr, "RESET", (bx3 + 10, by3 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 220, 50) if plc_state != PLC_RESETTING else (255, 255, 255), 2, cv2.LINE_AA)

                    # 4. BALL Button (Spawn/Move)
                    bx4, by4, bw4, bh4 = BTN_BALL
                    cv2.rectangle(bgr, (bx4, by4), (bx4 + bw4, by4 + bh4), (45, 25, 60), -1)
                    cv2.rectangle(bgr, (bx4, by4), (bx4 + bw4, by4 + bh4), (210, 120, 255), 2)
                    cv2.putText(bgr, "BALL", (bx4 + 14, by4 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (230, 160, 255), 2, cv2.LINE_AA)

                    # ── Bottom Shortcut Bar ──────────────────────────────────
                    cv2.rectangle(bgr, (0, h - 28), (w, h), (18, 18, 18), -1)
                    cv2.putText(bgr, "CONTROLS: Click buttons OR press [S]=Start  [Space]=Stop  [R]=Reset  [B]=Spawn Ball  | Click table to place ball", 
                                (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (170, 210, 170), 1, cv2.LINE_AA)

                    # Red ball tracker
                    hsv = cv2.cvtColor(bgr[68:h-28, :], cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255])) | cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
                    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if cnts:
                        c = max(cnts, key=cv2.contourArea)
                        if cv2.contourArea(c) > 25:
                            (bx, by), br = cv2.minEnclosingCircle(c)
                            bx, by = int(bx), int(by) + 68
                            cv2.circle(bgr, (bx, by), int(br) + 5, (0, 255, 0), 2)
                            cv2.drawMarker(bgr, (bx, by), (0, 255, 0), cv2.MARKER_CROSS, 16, 1)
                            cv2.putText(bgr, "BALL TARGET", (bx + 14, by - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1, cv2.LINE_AA)

                    cv2.imshow("Overhead 3D Camera - Autonomous RL Demonstration", bgr)
                except Exception:
                    pass

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

