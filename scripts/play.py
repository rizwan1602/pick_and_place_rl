# SPDX-License-Identifier: Apache-2.0
"""
Industrial Digital Twin: Franka Panda Pick & Place with RL and Siemens S7 PLC.
=============================================================================
Usage:
    isaaclab.bat -p scripts/play.py --checkpoint weights/model_499.pt
"""

from __future__ import annotations

import argparse
import importlib.metadata as metadata
import logging
import os
import sys
import time
from typing import Optional, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Suppress GitPython noise
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RobotCell")

# ── 1. Parse arguments & launch simulation via AppLauncher ───────────────────
parser = argparse.ArgumentParser(description="Franka Panda Digital Twin with RL and Siemens PLC")
parser.add_argument("--task", type=str, default="BallPickPlace-Franka-Play-v0", help="Task name")
parser.add_argument("--num_envs", type=int, default=1, help="Number of visualization environments (default: 1)")
parser.add_argument("--checkpoint", type=str, default="weights/model_499.pt", help="Path to trained checkpoint (.pt)")
parser.add_argument("--num_cycles", type=int, default=5, help="Number of pick-and-place cycles to evaluate (default: 5)")
parser.add_argument("--max_steps", type=int, default=None, help="Max simulation steps (default: infinite)")
parser.add_argument("--smooth_alpha", type=float, default=0.65, help="EMA action smoother alpha (default: 0.65)")
parser.add_argument("--show_camera", action="store_true", default=True, help="Display live OpenCV camera HUD")
parser.add_argument("--no_camera", dest="show_camera", action="store_false", help="Disable OpenCV camera window")

if "--enable_cameras" not in sys.argv:
    sys.argv += ["--enable_cameras"]

if not any(arg in sys.argv for arg in ["--viz", "--visualizer", "--headless"]):
    sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Imports AFTER SimulationApp ───────────────────────────────────────────
import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.managers import SceneEntityCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg

import ball_pick_place
from ball_pick_place.agents.rsl_rl_ppo_cfg import BallPickPlacePPORunnerCfg
from ball_pick_place.tasks.pick_place_ball.mdp.rewards import is_ball_in_bucket
from ball_pick_place.tasks.pick_place_ball.mdp.geometry import (
    BUCKET_X,
    BUCKET_Y,
    IN_BUCKET_XY_HALF,
    IN_BUCKET_Z_MIN,
    IN_BUCKET_Z_MAX,
    TABLE_TOP_Z,
    BALL_RADIUS,
)
from ball_pick_place.utils import ActionSmoother, CellCameraHUD

# Physical Home Standby pose coordinates
STANDBY_POS_W = [0.35, 0.00, 0.65]
ball_entity_cfg = SceneEntityCfg("ball")

# PLC State definitions
PLC_IDLE = "IDLE"           # Parked at Home, awaiting START
PLC_RUNNING = "RUNNING"     # Actively executing pick-and-place
PLC_STOPPED = "STOPPED"     # Motion halted (arm holds position)
PLC_RESETTING = "RESETTING" # Retracting arm to Home Standby pose

# State Machine definitions
STATE_RL = "RL_POLICY"
STATE_RELEASE = "RELEASE_AND_LIFT"
STATE_SETTLE = "VERIFY_SETTLE"
STATE_RETRACT = "RETRACT_HOME"


def spawn_or_move_ball(
    env: gym.Env,
    x: Optional[float] = None,
    y: Optional[float] = None,
    bridge: Optional[Any] = None,
    status_msg: Optional[str] = None,
) -> Tuple[float, float]:
    """Uniformly reposition or randomize the ball on the table workspace.

    Adheres to DRY (Don't Repeat Yourself) by consolidating PhysX tensor state updates.
    """
    if x is None:
        x = float(torch.empty(1).uniform_(0.28, 0.42).item())
    if y is None:
        y = float(torch.empty(1).uniform_(-0.12, 0.12).item())

    z = TABLE_TOP_Z + BALL_RADIUS + 0.002
    ball = env.unwrapped.scene["ball"]
    env_origins = env.unwrapped.scene.env_origins

    b_state = ball.data.default_root_state.clone()
    b_state[:, 0] = x + env_origins[0, 0]
    b_state[:, 1] = y + env_origins[0, 1]
    b_state[:, 2] = z + env_origins[0, 2]
    b_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.unwrapped.device)
    b_state[:, 7:13] = 0.0
    ball.write_root_state_to_sim(b_state)

    if bridge:
        bridge.ball_x = x
        bridge.ball_y = y
        if status_msg:
            bridge.status_message = status_msg
        bridge.notify()

    return x, y


def main() -> None:
    logger.info("Initializing Franka Panda Autonomous Cell with Siemens S7 PLC...")
    logger.info(f"Target checkpoint: {args_cli.checkpoint}")

    # ── Create Environment ───────────────────────────────────────────────────
    from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceFrankaEnvCfg_PLAY

    env_cfg = BallPickPlaceFrankaEnvCfg_PLAY()
    env_cfg.scene.num_envs = args_cli.num_envs
    env = gym.make(args_cli.task, cfg=env_cfg)

    if hasattr(env.unwrapped, "sim"):
        env.unwrapped.sim.set_camera_view(eye=(1.15, -0.55, 0.85), target=(0.35, 0.15, 0.45))

    env = RslRlVecEnvWrapper(env)

    # ── Load Siemens PLC Extension ───────────────────────────────────────────
    cell_bridge = None
    try:
        import omni.kit.app
        ext_mgr = omni.kit.app.get_app().get_extension_manager()
        ext_mgr.add_path(os.path.abspath("extensions"))
        if not ext_mgr.is_extension_enabled("com.rizwan.siemens_plc"):
            ext_mgr.set_extension_enabled_immediate("com.rizwan.siemens_plc", True)
        logger.info("Siemens PLC extension (com.rizwan.siemens_plc) enabled successfully.")
    except Exception as e:
        logger.debug(f"Extension manager note: {e}")

    # Connect to shared state bridge
    sys.path.insert(0, os.path.abspath("extensions/com.rizwan.siemens_plc"))
    try:
        from siemens_plc.bridge_state import RobotCellBridge
        cell_bridge = RobotCellBridge.get()
        cell_bridge.plc_online = True
        cell_bridge.robot_state = PLC_IDLE
        cell_bridge.notify()
    except Exception as e:
        logger.debug(f"Cell bridge note: {e}")

    # ── Load RL Policy ───────────────────────────────────────────────────────
    agent_cfg = BallPickPlacePPORunnerCfg()
    installed_version = metadata.version("rsl-rl-lib")
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    logger.info(f"Loading trained neural weights from: {args_cli.checkpoint}")
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    home_tensor = torch.tensor(STANDBY_POS_W, device=env.unwrapped.device).unsqueeze(0)
    smoother = ActionSmoother(alpha=args_cli.smooth_alpha)

    # ── Initialize Encapsulated Camera HUD (No Global State) ─────────────────
    hud = CellCameraHUD(
        window_name="Overhead 3D Camera - Industrial Digital Twin",
        width=640,
        height=480,
        enabled=args_cli.show_camera and not getattr(args_cli, "headless", False),
    )

    # State Machine & Telemetry Tracking
    state = STATE_RL
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
    logger.info("System ready in IDLE mode. Place ball on table and issue START command.")

    try:
        while True:
            total_steps += 1
            cmd = None

            # 1. Check Siemens PLC Bridge commands
            if cell_bridge:
                b_cmd = cell_bridge.pop_command()
                if b_cmd:
                    cmd = b_cmd

            # 2. Check OpenCV HUD commands (mouse clicks & hotkeys)
            hud_cmd = hud.pop_command()
            if hud_cmd:
                cmd = hud_cmd

            # 3. Process Command Event
            if cmd:
                if isinstance(cmd, tuple) and cmd[0] == "MOVE_BALL":
                    _, rx, ry = cmd
                    spawn_or_move_ball(env, rx, ry, cell_bridge, f"Ball repositioned at X={rx:.3f}m, Y={ry:.3f}m")
                    logger.info(f"Ball manually positioned at X={rx:.3f}m, Y={ry:.3f}m")

                elif cmd == "SPAWN_BALL":
                    rx, ry = spawn_or_move_ball(env, None, None, cell_bridge, "Ball randomized on table")
                    logger.info(f"Ball randomized on table at X={rx:.3f}m, Y={ry:.3f}m")

                elif cmd == "START":
                    # Auto-respawn ball if already inside the placement container
                    ball_pos_env = env.unwrapped.scene["ball"].data.root_pos_w - env.unwrapped.scene.env_origins
                    inside_x = abs(ball_pos_env[0, 0].item() - BUCKET_X) < IN_BUCKET_XY_HALF
                    inside_y = abs(ball_pos_env[0, 1].item() - BUCKET_Y) < IN_BUCKET_XY_HALF
                    if inside_x and inside_y:
                        spawn_or_move_ball(env, None, None, cell_bridge, "Ball auto-spawned on table for new cycle")
                        logger.info("Ball in bucket detected: auto-spawned onto table for new cycle.")

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
                        logger.info(f"START: Cycle {cycle_count + 1} initiated.")

                elif cmd == "STOP":
                    plc_state = PLC_STOPPED
                    if cell_bridge:
                        cell_bridge.robot_state = plc_state
                        cell_bridge.status_message = "Robot halted (Hold position)"
                        cell_bridge.notify()
                    logger.info("STOP: Robot motion halted. Holding pose.")

                elif cmd == "RESET":
                    plc_state = PLC_RESETTING
                    smoother.reset()
                    policy.reset()
                    state = STATE_RL
                    rx, ry = spawn_or_move_ball(env, None, None, cell_bridge, "RESET ALL: Ball on table | Homing arm...")
                    logger.info(f"RESET: Arm homing initiated, ball placed at X={rx:.3f}m, Y={ry:.3f}m.")

            # 4. State Execution Branch
            if plc_state == PLC_STOPPED:
                # Hold position
                stop_act = torch.zeros(1, 4, device=env.unwrapped.device)
                stop_act[:, 3] = last_gripper_cmd
                obs, rew, dones, extras = env.step(stop_act)

            elif plc_state == PLC_IDLE:
                # Parked at Standby Home pose
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                delta = home_tensor - ee_pos
                idle_act = torch.zeros(1, 4, device=env.unwrapped.device)
                idle_act[:, :3] = torch.clamp(delta * 2.0, -0.3, 0.3)
                idle_act[:, 3] = 1.0
                last_gripper_cmd = 1.0
                obs, rew, dones, extras = env.step(idle_act)

            elif plc_state == PLC_RESETTING:
                # Retract arm to Standby Home
                ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                delta = home_tensor - ee_pos
                dist_to_home = torch.norm(delta).item()

                reset_act = torch.zeros(1, 4, device=env.unwrapped.device)
                reset_act[:, :3] = torch.clamp(delta * 2.5, -0.8, 0.8)
                reset_act[:, 3] = 1.0
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
                    logger.info(f"Franka Panda reached Home Standby ({dist_to_home * 1000:.1f} mm error). Ready.")

            elif plc_state == PLC_RUNNING:
                # Autonomous Pick & Place Sequence
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

                    # Safety Interlock: hold ball securely until within drop radius
                    is_holding = (ball_pos_env[0, 2].item() > 0.48) and (dist_ee_ball < 0.08)
                    if is_holding and dist_ee_bucket_xy > 0.045:
                        act[:, 3] = -1.0

                    last_gripper_cmd = act[0, 3].item()
                    obs, rew, dones, extras = env.step(act)

                    over_bucket = (dist_ee_bucket_xy <= 0.045) and (ee_pos_env[0, 2].item() <= 0.62)
                    in_bucket = is_ball_in_bucket(env.unwrapped, ball_entity_cfg).item() > 0.5

                    if (is_holding and over_bucket) or in_bucket:
                        state = STATE_RELEASE
                        release_step = 0

                elif state == STATE_RELEASE:
                    # Vertical release & lift above container
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
                    # Verify ball at rest in container cavity
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
                        logger.info(f"Cycle {cycle_count}: SUCCESS! Settled in {duration:.2f}s. Retracting...")
                        if cell_bridge:
                            cell_bridge.cycle_count = cycle_count
                            cell_bridge.status_message = f"Cycle {cycle_count} SUCCESS ({duration:.2f}s)! Retracting..."
                            cell_bridge.notify()
                        state = STATE_RETRACT
                        retract_step = 0
                    elif settle_step >= 25:
                        duration = time.time() - cycle_start_time
                        cycle_count += 1
                        results.append((cycle_count, duration, "FAILED"))
                        logger.warning(f"Cycle {cycle_count}: FAILED (ball not settled). Retracting...")
                        if cell_bridge:
                            cell_bridge.cycle_count = cycle_count
                            cell_bridge.status_message = f"Cycle {cycle_count} FAILED! Retracting..."
                            cell_bridge.notify()
                        state = STATE_RETRACT
                        retract_step = 0

                elif state == STATE_RETRACT:
                    # Return to Standby Home Pose
                    ee_pos = env.unwrapped.scene["ee_frame"].data.target_pos_w[..., 0, :] - env.unwrapped.scene.env_origins
                    delta = home_tensor - ee_pos
                    dist_to_home = torch.norm(delta).item()

                    retract_act = torch.zeros(1, 4, device=env.unwrapped.device)
                    retract_act[:, :3] = torch.clamp(delta * 2.5, -0.8, 0.8)
                    retract_act[:, 3] = 1.0
                    last_gripper_cmd = 1.0

                    obs, rew, dones, extras = env.step(retract_act)
                    retract_step += 1

                    if dist_to_home < 0.040 or retract_step >= 35:
                        logger.info(f"Franka Panda returned to Home Standby. Ready for next cycle.")
                        plc_state = PLC_IDLE
                        state = STATE_RL
                        smoother.reset()
                        policy.reset()
                        if cell_bridge:
                            cell_bridge.robot_state = plc_state
                            cell_bridge.status_message = "Ready for next cycle. Place ball and click START."
                            cell_bridge.notify()

            # 5. Check step limits
            if args_cli.max_steps is not None and total_steps >= args_cli.max_steps:
                logger.info(f"Reached max steps limit ({args_cli.max_steps}). Exiting loop.")
                break

            # 6. Render Camera HUD (cleanly delegated to CellCameraHUD)
            if hud.enabled:
                overhead_cam = env.unwrapped.scene["overhead_cam"]
                rgb_data = overhead_cam.data.output["rgb"][0, :, :, :3]
                key_cmd = hud.render(rgb_data, plc_state=plc_state, fsm_state=state)
                if key_cmd == "EXIT":
                    logger.info("Exit requested via HUD shortcut. Terminating.")
                    break
                elif key_cmd:
                    cmd = key_cmd  # will process on next loop iteration

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")

    finally:
        hud.close()
        env.close()
        simulation_app.close()
        logger.info("Robotic cell shutdown complete.")


if __name__ == "__main__":
    main()
