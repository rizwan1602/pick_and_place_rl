# SPDX-License-Identifier: Apache-2.0
"""
Manager-based RL Environment Configuration for Franka Ball Pick-and-Place.
==========================================================================
All dimensions and coordinates match scenes/runtime_scene.usd:
  - Table: 0.60m x 0.50m x 0.42m at center (0.35, 0.00, 0.21)
  - Bucket: Hollow 5-walled container at (0.35, 0.30, 0.42) with floor + 4 physical walls
  - Ball: Radius 0.03m on table surface (Z = 0.42m + 0.03m = 0.45m)
  - Robot: Base floor-mounted at (0, 0, 0)
"""

from __future__ import annotations

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    ActionTermCfg as ActionCfg,
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer import OffsetCfg
from isaaclab.utils.configclass import configclass

from isaaclab_physx.physics import PhysxCfg

import isaaclab.envs.mdp as mdp
from . import mdp as custom_mdp

from isaaclab.controllers import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab_assets.robots.franka import FRANKA_PANDA_HIGH_PD_CFG

# Path to the hollow bucket USD model with 5 physical collision walls
_BUCKET_USD_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "scenes", "bucket.usd")
).replace("\\", "/")


##
# Scene definition
##

@configclass
class BallPickPlaceSceneCfg(InteractiveSceneCfg):
    """Configuration for the ball pick and place scene."""

    # 1. Ground Plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 2. Lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )

    # 3. Table (0.50m x 0.50m x 0.42m at center (0.40, 0.0, 0.21) - 3cm clearance from base)
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.50, 0.50, 0.42),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.6, 0.55, 0.45)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.5,
                friction_combine_mode="max",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.40, 0.0, 0.21)),
    )

    # 4. Robot (Franka Panda floor-mounted at 0, 0, 0 with High Home Pose Z=0.650m)
    robot: ArticulationCfg = FRANKA_PANDA_HIGH_PD_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos={
                "panda_joint1": -0.0260,
                "panda_joint2": -0.6079,
                "panda_joint3": 0.0171,
                "panda_joint4": -2.0867,
                "panda_joint5": 0.0098,
                "panda_joint6": 1.4793,
                "panda_joint7": 0.7742,
                "panda_finger_joint.*": 0.04,
            },
        ),
    )

    # 5. Target Ball (r=0.03m, dynamic rigid sphere with maximum grip friction)
    ball: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=2,
                max_angular_velocity=1000.0,
                max_linear_velocity=1000.0,
                max_depenetration_velocity=2.5,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.057),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.004,
                rest_offset=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.15, 0.15)),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.5,
                dynamic_friction=1.2,
                restitution=0.0,
                friction_combine_mode="max",
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.35, 0.0, 0.45)),
    )

    # 6. Placement Bucket (Hollow Container with 4 collision walls + floor, open top)
    bucket = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Bucket",
        spawn=sim_utils.UsdFileCfg(
            usd_path=_BUCKET_USD_PATH,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.35, 0.30, 0.42)),
    )

    # 7. End-Effector Frame Transformer
    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                name="end_effector",
                offset=OffsetCfg(pos=(0.0, 0.0, 0.1034)),
            ),
        ],
    )


##
# MDP Settings: Actions, Observations, Rewards, Events, Terminations
##

@configclass
class ActionsCfg:
    """Cartesian Task-Space Actions for Franka: 3D Delta Position + 1 binary gripper."""
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=["panda_joint.*"],
        body_name="panda_hand",
        controller=DifferentialIKControllerCfg(command_type="position", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
        body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
    )
    gripper_action = mdp.BinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=["panda_finger.*"],
        open_command_expr={"panda_finger_.*": 0.04},
        close_command_expr={"panda_finger_.*": 0.0},
    )


@configclass
class ObservationsCfg:
    """Observation inputs fed to the policy network (41 dimensions)."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot")})
        ee_pos = ObsTerm(func=custom_mdp.ee_position_in_robot_root_frame)
        ball_pos_robot = ObsTerm(func=custom_mdp.ball_position_in_robot_root_frame)
        ball_pos_ee = ObsTerm(func=custom_mdp.ball_position_relative_to_ee)
        bucket_pos_robot = ObsTerm(func=custom_mdp.bucket_position_in_robot_root_frame)
        bucket_pos_ee = ObsTerm(func=custom_mdp.bucket_position_relative_to_ee)
        ee_orientation = ObsTerm(func=custom_mdp.ee_orientation_in_world)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False   # no ObsTerm declares noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Staged, monotonically increasing reward pipeline matching IK behavior.
    
    Potential values step-by-step:
      1. Reach:   0 -> 9 pts  (Broad 6.0 + Fine 3.0)
      2. Grasp:   0 -> 4 pts  (Finger clamp near ball)
      3. Lift:    0 -> 10 pts (Ball raised above table)
      4. Transit: 0 -> 10 pts (Carry towards bucket)
      5. Release: 0 -> 25 pts (Open fingers above bucket)
      6. Place:   100 pts     (Ball inside cavity)
    """
    # ── Stage 1: Approach the ball (Broad + Fine)
    reaching_coarse = RewTerm(func=custom_mdp.reaching_reward,      params={"std": 0.25}, weight=6.0)
    reaching_fine   = RewTerm(func=custom_mdp.reaching_reward_fine, params={"std": 0.05}, weight=3.0)

    # ── Stage 2: Grasp the ball (Clamp fingers when at ball)
    grasping        = RewTerm(func=custom_mdp.grasping_reward,                             weight=4.0)

    # ── Stage 3: Lift the ball off table
    lifting         = RewTerm(func=custom_mdp.lifting_reward,                              weight=10.0)

    # ── Stage 4: Transit to bucket
    bucket_tracking = RewTerm(func=custom_mdp.bucket_tracking_coarse, params={"std": 0.25}, weight=10.0)

    # ── Stage 5: Release above bucket (Decisive incentive to open fingers)
    release         = RewTerm(func=custom_mdp.release_reward,                              weight=25.0)

    # ── Stage 6: Place inside cavity (Dominant goal milestone: 100 > 30 holding)
    placing         = RewTerm(func=custom_mdp.placing_reward,                              weight=100.0)

    # ── Posture & Clearance
    ee_orientation  = RewTerm(func=custom_mdp.ee_downward_orientation_reward, weight=2.0)
    table_clearance = RewTerm(func=custom_mdp.arm_table_clearance_penalty,    weight=-5.0)

    # ── Regularization
    ball_velocity = RewTerm(func=custom_mdp.ball_velocity_penalty, weight=-0.2)
    action_rate   = RewTerm(func=mdp.action_rate_l2,               weight=-1e-4)


@configclass
class EventCfg:
    """Reset events: ball always spawns on table (zero bucket cheat)."""
    reset_robot = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (0.9, 1.1),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot", joint_names=["panda_joint.*"]),
        },
    )
    reset_ball = EventTerm(
        func=custom_mdp.reset_ball_position_on_table,
        mode="reset",
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "table_x_range": (0.28, 0.42),
            "table_y_range": (-0.12, 0.12),
            "start_fraction": 0.0,      # 0% bucket resets - BALL ALWAYS SPAWNS ON TABLE!
            "anneal_steps": 1,
        },
    )


@configclass
class TerminationsCfg:
    """Episode terminations: timeout and failure (ball off table).
    
    NOTE: Success does NOT terminate the episode early!
    This allows the robot to collect sustained placing reward (35.0/step)
    for all remaining episode steps (~14,000 points total), making dropping
    the ball 11x more profitable than holding it!
    """
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    failure  = DoneTerm(func=custom_mdp.ball_off_table)


@configclass
class CurriculumCfg:
    """No curriculum penalty ramp: keeps penalties static at -1e-4 to allow fluid arm motion."""
    pass


##
# Complete Environment Configuration
##

@configclass
class BallPickPlaceFrankaEnvCfg(ManagerBasedRLEnvCfg):
    """Main training configuration for Ball Pick-and-Place."""
    scene: BallPickPlaceSceneCfg = BallPickPlaceSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    events: EventCfg = EventCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 10.0  # 10 seconds per attempt
        self.sim.dt = 0.01            # 100 Hz physics
        self.sim.render_interval = self.decimation

        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )


@configclass
class BallPickPlaceCameraSceneCfg(BallPickPlaceSceneCfg):
    """Scene configuration equipped with Overhead 3D RGB-D Camera for Vision-Guided RL/IK."""
    overhead_cam: CameraCfg = CameraCfg(
        prim_path="{ENV_REGEX_NS}/OverheadCam",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 5.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.40, 0.0, 1.20),
            rot=(0.0, 0.7071068, 0.0, 0.7071068),
            convention="world",
        ),
    )


@configclass
class BallPickPlaceFrankaCameraEnvCfg(BallPickPlaceFrankaEnvCfg):
    """Camera-enabled RL / Vision evaluation environment."""
    scene: BallPickPlaceCameraSceneCfg = BallPickPlaceCameraSceneCfg(num_envs=1, env_spacing=2.5)


@configclass
class BallPickPlaceFrankaEnvCfg_PLAY(BallPickPlaceFrankaCameraEnvCfg):
    """Evaluation / Play variant equipped with camera and close-up visualization."""
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
