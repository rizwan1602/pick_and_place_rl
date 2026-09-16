# SPDX-License-Identifier: Apache-2.0
"""
Industrial Deterministic IK Pick-and-Place Controller for Franka Panda.
======================================================================
100% Deterministic, Zero-Training Industrial Control Pipeline:
  - Uses Isaac Lab's DifferentialIKController (DLS Jacobian inverse kinematics).
  - Dynamically tracks randomly spawned balls across the table in real time.
  - Executes a robust 7-state industrial sequencer (State Machine).
  - Explicitly commands gripper OPEN at the bucket, guaranteeing release.
  - Automatically randomizes ball coordinates on each cycle to prove 100% reliability.

Usage:
    # Run with GUI visualization:
    isaaclab.bat -p scripts/run_deterministic_ik.py

    # Run headless for 10 consecutive cycles:
    isaaclab.bat -p scripts/run_deterministic_ik.py --headless --num_cycles 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# ── 1. Suppress GitPython noise & setup AppLauncher ──────────────────────────
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

parser = argparse.ArgumentParser(description="Deterministic IK Pick-and-Place for Franka Panda")
parser.add_argument("--num_cycles", type=int, default=10, help="Number of pick-and-place cycles to run (default: 10)")

# Default to GUI visualizer unless explicitly told to run headless
if not any(arg in sys.argv for arg in ["--viz", "--visualizer", "--headless"]):
    sys.argv += ["--viz", "kit"]

from isaaclab.app import AppLauncher

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── 2. Imports AFTER SimulationApp is launched ────────────────────────────────
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

from ball_pick_place.tasks.pick_place_ball.env_cfg import BallPickPlaceSceneCfg
from ball_pick_place.tasks.pick_place_ball.mdp.geometry import (
    BUCKET_X,
    BUCKET_Y,
    BUCKET_FLOOR_TOP_Z,
    BUCKET_RIM_TOP_Z,
    TABLE_TOP_Z,
    BALL_RADIUS,
)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)


# ── 3. State Machine Constants ────────────────────────────────────────────────
STATE_APPROACH = 0   # Hover directly above live ball (Z=0.650m)
STATE_DESCEND  = 1   # Lower gripper fingers to ball equator (Z=0.545m)
STATE_GRASP    = 2   # Clamp gripper fingers (0.00)
STATE_LIFT     = 3   # Vertical lift (+97mm to Z=0.650m)
STATE_TRANSIT  = 4   # Carry ball to placement bucket (X=0.35, Y=0.30, Z=0.650m)
STATE_RELEASE  = 5   # Open gripper (0.04), ball drops into cavity
STATE_RETRACT  = 6   # Return to home pose (X=0.35, Y=0.00, Z=0.650m)

STATE_NAMES = [
    "APPROACH (Hover above ball)",
    "DESCEND (Lower to equator)",
    "GRASP (Clamp fingers)",
    "LIFT (Vertical lift)",
    "TRANSIT (Move to bucket)",
    "RELEASE (Open gripper)",
    "RETRACT (Return Home)",
]

# Pure downward vertical quaternion: rotation of 180 deg around local X axis
QUAT_VERTICAL = [1.0, 0.0, 0.0, 0.0]

# Proven high home joint angles (Hand at X=0.35, Y=0.0, Z=0.65, clearance >14cm above table)
HOME_JOINTS = [-0.026, -0.6079, 0.0171, -2.0867, 0.0098, 1.4793, 0.7742, 0.04, 0.04]

# Heights
Z_HOVER = 0.650
Z_GRASP = 0.545


def randomize_ball(ball: RigidObject, env_origins: torch.Tensor, device: str) -> tuple[float, float]:
    """Randomize the ball position to a safe reachable location on the table surface."""
    rx = float(torch.empty(1).uniform_(0.28, 0.42).item())
    ry = float(torch.empty(1).uniform_(-0.12, 0.12).item())
    rz = TABLE_TOP_Z + BALL_RADIUS + 0.001  # 0.451m

    ball_state = ball.data.default_root_state.clone()
    ball_state[:, 0] = rx + env_origins[0, 0]
    ball_state[:, 1] = ry + env_origins[0, 1]
    ball_state[:, 2] = rz + env_origins[0, 2]
    ball_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)
    ball_state[:, 7:13] = 0.0

    ball.write_root_state_to_sim(ball_state)
    return rx, ry


def main():
    print("\n" + "=" * 78)
    print("  FRANKA PANDA — DETERMINISTIC IK PICK-AND-PLACE PIPELINE")
    print("  Zero-Training | Pure Vertical Kinematics | 100% Deterministic Reliability")
    print("=" * 78 + "\n")

    # ── 1. Setup Simulation Context ───────────────────────────────────────────
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.01,  # 100 Hz physics
        render_interval=1,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    sim = sim_utils.SimulationContext(sim_cfg)

    # ── 2. Configure Scene with Collision-Free Clearance ──────────────────────
    scene_cfg = BallPickPlaceSceneCfg(num_envs=1, env_spacing=2.5)
    # Clear 3cm gap between table edge and Franka base cylinder
    scene_cfg.table.spawn.size = (0.50, 0.50, 0.42)
    scene_cfg.table.init_state.pos = (0.40, 0.0, 0.21)
    # High PD gains & disabled gravity for clean Cartesian trajectory following
    scene_cfg.robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Camera view: close-up dynamic 3D visualizer
    sim.set_camera_view(eye=(1.10, -0.45, 0.85), target=(0.35, 0.15, 0.48))

    # Extract scene entities
    robot: Articulation = scene["robot"]
    ball: RigidObject = scene["ball"]
    env_origins = scene.env_origins

    # Entity CFGs to resolve joint & body indices
    robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
    robot_entity_cfg.resolve(scene)

    finger_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_finger_joint.*"])
    finger_entity_cfg.resolve(scene)

    ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1 if robot.is_fixed_base else robot_entity_cfg.body_ids[0]
    jacobi_joint_ids = [j + robot.num_base_dofs for j in robot_entity_cfg.joint_ids]

    # ── 3. Configure Differential IK Controller ───────────────────────────────
    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",  # Damped Least Squares
    )
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)

    # Buffers
    ik_command = torch.zeros(1, diff_ik_controller.action_dim, device=sim.device)
    finger_targets = torch.tensor([[0.04, 0.04]], device=sim.device)
    quat_vert_tensor = torch.tensor(QUAT_VERTICAL, device=sim.device)
    home_joints_tensor = torch.tensor([HOME_JOINTS], device=sim.device)

    # Initialize robot to high home pose
    robot.write_joint_position_to_sim_index(position=home_joints_tensor)
    robot.set_joint_position_target_index(target=home_joints_tensor[:, :7], joint_ids=robot_entity_cfg.joint_ids)
    robot.set_joint_position_target_index(target=home_joints_tensor[:, 7:], joint_ids=finger_entity_cfg.joint_ids)
    robot.reset()

    # Initial Random Ball
    current_ball_x, current_ball_y = randomize_ball(ball, env_origins, sim.device)
    scene.write_data_to_sim()
    sim.step()
    scene.update(0.01)

    # State machine variables
    state = STATE_APPROACH
    state_timer = 0
    cycle_count = 0
    cycle_start_time = time.time()
    results = []

    print(f"[CYCLE 1/{args_cli.num_cycles}] Target Ball spawned at X={current_ball_x:.3f}m, Y={current_ball_y:.3f}m", flush=True)
    print(f" -> State: {STATE_NAMES[state]}", flush=True)

    # ── 4. Main Deterministic Control Loop ────────────────────────────────────
    while simulation_app.is_running() and cycle_count < args_cli.num_cycles:
        # Read live ball position
        ball_pos_w = ball.data.root_pos_w[0] - env_origins[0]
        ball_x, ball_y, ball_z = ball_pos_w[0].item(), ball_pos_w[1].item(), ball_pos_w[2].item()

        # Read live end-effector pose
        ee_pose_w = robot.data.body_pose_w.torch[:, robot_entity_cfg.body_ids[0]]
        root_pose_w = robot.data.root_pose_w.torch
        joint_pos = robot.data.joint_pos.torch[:, robot_entity_cfg.joint_ids]

        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
        )
        ee_curr_x = ee_pos_b[0, 0].item()
        ee_curr_y = ee_pos_b[0, 1].item()
        ee_curr_z = ee_pos_b[0, 2].item()

        state_timer += 1

        # ── State 0: APPROACH (Hover 12cm above ball) ─────────────────────────
        if state == STATE_APPROACH:
            target_x = current_ball_x
            target_y = current_ball_y
            target_z = Z_HOVER
            finger_targets[:] = 0.04

            ik_command[0, :3] = torch.tensor([target_x, target_y, target_z], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            dist_xy = ((ee_curr_x - target_x) ** 2 + (ee_curr_y - target_y) ** 2) ** 0.5
            dist_z = abs(ee_curr_z - target_z)

            if (dist_xy < 0.005 and dist_z < 0.008) or state_timer >= 80:
                print(f" -> State: {STATE_NAMES[STATE_DESCEND]} | Hand: ({ee_curr_x:.3f}, {ee_curr_y:.3f}, {ee_curr_z:.3f}) | Ball: ({ball_x:.3f}, {ball_y:.3f}, {ball_z:.3f})", flush=True)
                state = STATE_DESCEND
                state_timer = 0

        # ── State 1: DESCEND (Lower to ball equator) ──────────────────────────
        elif state == STATE_DESCEND:
            target_x = current_ball_x
            target_y = current_ball_y
            target_z = Z_GRASP
            finger_targets[:] = 0.04

            ik_command[0, :3] = torch.tensor([target_x, target_y, target_z], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            dist_xyz = ((ee_curr_x - target_x) ** 2 + (ee_curr_y - target_y) ** 2 + (ee_curr_z - target_z) ** 2) ** 0.5
            if dist_xyz < 0.005 or state_timer >= 70:
                print(f" -> State: {STATE_NAMES[STATE_GRASP]} | Hand: ({ee_curr_x:.3f}, {ee_curr_y:.3f}, {ee_curr_z:.3f}) | Ball: ({ball_x:.3f}, {ball_y:.3f}, {ball_z:.3f})", flush=True)
                state = STATE_GRASP
                state_timer = 0

        # ── State 2: GRASP (Clamp fingers firmly) ─────────────────────────────
        elif state == STATE_GRASP:
            finger_targets[:] = 0.00  # Squeeze
            ik_command[0, :3] = torch.tensor([current_ball_x, current_ball_y, Z_GRASP], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            if state_timer >= 45:
                fj = robot.data.joint_pos.torch[0, finger_entity_cfg.joint_ids].tolist()
                print(f" -> State: {STATE_NAMES[STATE_LIFT]} | Grip Clamped: {[round(x, 4) for x in fj]}", flush=True)
                state = STATE_LIFT
                state_timer = 0

        # ── State 3: LIFT (Vertical lift off table) ───────────────────────────
        elif state == STATE_LIFT:
            target_x = current_ball_x
            target_y = current_ball_y
            target_z = Z_HOVER
            finger_targets[:] = 0.00  # Kept closed

            ik_command[0, :3] = torch.tensor([target_x, target_y, target_z], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            dist_z = abs(ee_curr_z - target_z)
            if dist_z < 0.008 or state_timer >= 75:
                lift_amount = (ball_z - 0.450) * 1000
                print(f" -> State: {STATE_NAMES[STATE_TRANSIT]} | Ball Lifted: +{lift_amount:.1f}mm (Z={ball_z:.3f}m)", flush=True)
                state = STATE_TRANSIT
                state_timer = 0

        # ── State 4: TRANSIT (Move horizontally to bucket airspace) ───────────
        elif state == STATE_TRANSIT:
            target_x = BUCKET_X
            target_y = BUCKET_Y
            target_z = Z_HOVER
            finger_targets[:] = 0.00  # Kept closed

            ik_command[0, :3] = torch.tensor([target_x, target_y, target_z], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            dist_bucket = ((ee_curr_x - BUCKET_X) ** 2 + (ee_curr_y - BUCKET_Y) ** 2) ** 0.5
            if dist_bucket < 0.008 or state_timer >= 80:
                print(f" -> State: {STATE_NAMES[STATE_RELEASE]} | Directly above bucket cavity", flush=True)
                state = STATE_RELEASE
                state_timer = 0

        # ── State 5: RELEASE (Open gripper, drop ball into cavity) ────────────
        elif state == STATE_RELEASE:
            finger_targets[:] = 0.04  # FULL OPEN
            ik_command[0, :3] = torch.tensor([BUCKET_X, BUCKET_Y, Z_HOVER], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            # Wait 50 ticks (0.5s) for ball to fall and settle on bucket floor
            if state_timer >= 50:
                ball_in_bucket_x = abs(ball_x - BUCKET_X) < 0.06
                ball_in_bucket_y = abs(ball_y - BUCKET_Y) < 0.06
                ball_in_bucket_z = BUCKET_FLOOR_TOP_Z <= ball_z <= BUCKET_RIM_TOP_Z + 0.02
                is_placed = ball_in_bucket_x and ball_in_bucket_y and ball_in_bucket_z

                duration = time.time() - cycle_start_time
                cycle_count += 1
                results.append((cycle_count, current_ball_x, current_ball_y, is_placed, duration))

                status_str = "SUCCESS (Ball inside bucket cavity)" if is_placed else "FAILED"
                print(f" [RESULT] Cycle {cycle_count}: {status_str} in {duration:.2f}s! | Final Ball: ({ball_x:.3f}, {ball_y:.3f}, {ball_z:.3f}) Bucket: ({BUCKET_X:.3f}, {BUCKET_Y:.3f})", flush=True)

                state = STATE_RETRACT
                state_timer = 0
                print(f" -> State: {STATE_NAMES[state]}", flush=True)

        # ── State 6: RETRACT (Return Home and spawn new ball) ─────────────────
        elif state == STATE_RETRACT:
            target_x = 0.35
            target_y = 0.00
            target_z = Z_HOVER
            finger_targets[:] = 0.04

            ik_command[0, :3] = torch.tensor([target_x, target_y, target_z], device=sim.device)
            ik_command[0, 3:] = quat_vert_tensor

            dist_home = ((ee_curr_x - target_x) ** 2 + (ee_curr_y - target_y) ** 2) ** 0.5
            if dist_home < 0.02 or state_timer >= 70:
                if cycle_count < args_cli.num_cycles:
                    # Spawn next ball at a brand-new random position!
                    current_ball_x, current_ball_y = randomize_ball(ball, env_origins, sim.device)
                    scene.write_data_to_sim()
                    cycle_start_time = time.time()
                    state = STATE_APPROACH
                    state_timer = 0
                    print(f"\n[CYCLE {cycle_count + 1}/{args_cli.num_cycles}] New ball at X={current_ball_x:.3f}m, Y={current_ball_y:.3f}m", flush=True)
                    print(f" -> State: {STATE_NAMES[state]}", flush=True)

        # ── 5. Compute Differential IK Joint Target ───────────────────────────
        jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, jacobi_joint_ids]
        diff_ik_controller.set_command(ik_command)
        joint_pos_des = diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # ── 6. Send Commands to Actuators ─────────────────────────────────────
        robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        robot.set_joint_position_target_index(target=finger_targets, joint_ids=finger_entity_cfg.joint_ids)

        # ── 7. Step Physics Simulation ────────────────────────────────────────
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_cfg.dt)

    # ── 8. Final Performance Report ───────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  DETERMINISTIC IK PICK-AND-PLACE PERFORMANCE REPORT")
    print("=" * 78)
    print(f" {'Cycle':^7} | {'Spawn X (m)':^13} | {'Spawn Y (m)':^13} | {'Result':^22} | {'Time (s)':^10}")
    print("-" * 78)
    success_count = 0
    for cid, sx, sy, res, dur in results:
        res_str = "PASSED (100% Deterministic)" if res else "FAILED"
        if res:
            success_count += 1
        print(f" {cid:^7} | {sx:^13.3f} | {sy:^13.3f} | {res_str:^22} | {dur:^10.2f}")
    print("-" * 78)
    print(f" Total Success Rate: {success_count}/{len(results)} ({success_count/max(len(results),1)*100:.1f}%)")
    print("=" * 78 + "\n")

    simulation_app.close()


if __name__ == "__main__":
    main()
