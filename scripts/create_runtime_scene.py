# SPDX-License-Identifier: Apache-2.0
"""
Create Runtime Scene for Ball Pick-and-Place POC
=================================================

Builds the complete USD scene and saves it for runtime inference:
  - Franka Panda robot arm
  - A table with legs
  - A draggable ball (sphere with rigid body physics)
  - A bucket/bin for placement target
  - Overhead camera
  - Proper lighting

Usage:
    "C:\\...\\isaac-sim-standalone-6.0.0\\python.bat" scripts\\create_runtime_scene.py

Output:
    scenes\\runtime_scene.usd
"""

from __future__ import annotations

# ==============================================================================
# 1. Parse args and launch SimulationApp FIRST
# ==============================================================================
import argparse
import os
import sys

parser = argparse.ArgumentParser(description="Create runtime USD scene for ball pick-and-place")
parser.add_argument("--output", type=str, default=None, help="Output .usd file path")
parser.add_argument("--headless", action="store_true", default=False, help="Run without GUI preview")
args, _ = parser.parse_known_args()

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": args.headless, "width": 1280, "height": 720})

# ==============================================================================
# 2. Import Omniverse / Isaac Sim 6.0 modules AFTER SimulationApp
# ==============================================================================
import numpy as np

import carb
import omni.usd
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.experimental.objects import GroundPlane, DistantLight
from isaacsim.storage.native import get_assets_root_path

from pxr import (
    Gf,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdLux,
    UsdPhysics,
    UsdShade,
)


# ==============================================================================
# 3. Scene Configuration — All dimensions in meters
# ==============================================================================

# --- Table ---
# Franka Panda natural workspace: arm hangs down from shoulder at z~0.33m above base.
# Best table height = slightly below shoulder = ~0.40m (when robot is at ground level).
# This lets the arm comfortably reach all corners from above.
TABLE_SIZE = (0.60, 0.50, 0.02)          # Length x Width x Thickness
TABLE_HEIGHT = 0.40                       # LOW table — matches Franka's natural reach plane
TABLE_TOP_Z = TABLE_HEIGHT + TABLE_SIZE[2] / 2
TABLE_SURFACE_Z = TABLE_HEIGHT + TABLE_SIZE[2]
TABLE_CENTER_XY = (0.35, 0.00)           # Close to robot for full coverage
TABLE_COLOR = (0.6, 0.55, 0.45)
# Table edges: X = 0.05 to 0.65 (all within Franka reach at this height)
#              Y = -0.25 to 0.25 (all within Franka reach)

# --- Robot ---
# Floor-mounted. Shoulder joint at z~0.33m, elbow workspace covers table at z=0.40m.
ROBOT_POSITION = (0.0, 0.0, 0.0)         # Robot base at ground level

# --- Ball ---
BALL_RADIUS = 0.03
BALL_MASS = 0.057
BALL_POSITION = (0.35, 0.00, TABLE_SURFACE_Z + BALL_RADIUS + 0.001)
BALL_COLOR = (1.0, 0.15, 0.15)

# --- Bucket ---
BUCKET_INNER_SIZE = 0.12
BUCKET_HEIGHT = 0.10
BUCKET_WALL = 0.005
BUCKET_POSITION = (0.35, 0.30, TABLE_SURFACE_Z)  # Right side of table
BUCKET_COLOR = (0.3, 0.3, 0.35)

# --- Camera ---
CAMERA_POSITION = (0.35, 0.00, 1.20)     # Overhead, lower since table is lower
CAMERA_FOCAL_LENGTH = 24.0


# ==============================================================================
# 4. Helper Functions
# ==============================================================================

def set_visual_material(stage, prim_path, color, name="Material"):
    """Apply a simple preview surface material with a diffuse color."""
    mat_path = f"{prim_path}/{name}"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(stage.GetPrimAtPath(prim_path)).Bind(material)


def create_physics_material(stage, prim_path, static_friction=0.6, dynamic_friction=0.5, restitution=0.2):
    """Create and bind a physics material to a prim."""
    mat_path = f"{prim_path}/PhysMat"
    UsdShade.Material.Define(stage, mat_path)
    mat_api = UsdPhysics.MaterialAPI.Apply(stage.GetPrimAtPath(mat_path))
    mat_api.CreateStaticFrictionAttr(static_friction)
    mat_api.CreateDynamicFrictionAttr(dynamic_friction)
    mat_api.CreateRestitutionAttr(restitution)
    prim = stage.GetPrimAtPath(prim_path)
    prim.CreateRelationship("material:binding:physics", custom=False).SetTargets([Sdf.Path(mat_path)])


def create_static_cube(stage, path, size, position, color, friction=0.6):
    """Create a static (non-moving) cube with collision."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.AddTranslateOp().Set(Gf.Vec3d(*position))
    cube.AddScaleOp().Set(Gf.Vec3f(size[0] / 2.0, size[1] / 2.0, size[2] / 2.0))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    set_visual_material(stage, path, color)
    create_physics_material(stage, path, static_friction=friction, dynamic_friction=friction * 0.8)
    return cube


# ==============================================================================
# 5. Build the Scene
# ==============================================================================

def build_scene(output_path):
    """Build the complete runtime scene and save to USD."""

    print("\n" + "=" * 70)
    print("  BUILDING RUNTIME SCENE: Ball Pick-and-Place")
    print("=" * 70)

    # Create a fresh stage
    stage_utils.create_new_stage()
    simulation_app.update()  # Let it initialize

    stage = omni.usd.get_context().get_stage()

    # Set up axis and units
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    # -------------------------------------------------------------------------
    # Physics Scene
    # -------------------------------------------------------------------------
    print("[1/8] Setting up physics scene...")
    physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
    physics_scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    physics_scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/World/PhysicsScene"))
    physx_scene.CreateEnableCCDAttr(True)
    physx_scene.CreateEnableStabilizationAttr(True)
    physx_scene.CreateTimeStepsPerSecondAttr(120)

    # -------------------------------------------------------------------------
    # Ground Plane
    # -------------------------------------------------------------------------
    print("[2/8] Adding ground plane...")
    GroundPlane("/World/GroundPlane", positions=[0, 0, 0])

    # -------------------------------------------------------------------------
    # Lighting
    # -------------------------------------------------------------------------
    print("[3/8] Setting up lighting...")
    DistantLight("/World/DistantLight")

    # Add dome light for ambient fill
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1500.0)
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))

    # -------------------------------------------------------------------------
    # Table
    # -------------------------------------------------------------------------
    print("[4/8] Creating table...")
    UsdGeom.Xform.Define(stage, "/World/Table")

    # Table top
    create_static_cube(
        stage, "/World/Table/Top",
        size=TABLE_SIZE,
        position=(TABLE_CENTER_XY[0], TABLE_CENTER_XY[1], TABLE_TOP_Z),
        color=TABLE_COLOR,
        friction=0.6,
    )

    # Table legs (4)
    leg_size = (0.04, 0.04, TABLE_HEIGHT)
    offsets_x = [TABLE_CENTER_XY[0] - TABLE_SIZE[0] / 2 + 0.05,
                 TABLE_CENTER_XY[0] + TABLE_SIZE[0] / 2 - 0.05]
    offsets_y = [TABLE_CENTER_XY[1] - TABLE_SIZE[1] / 2 + 0.05,
                 TABLE_CENTER_XY[1] + TABLE_SIZE[1] / 2 - 0.05]
    for i, (lx, ly) in enumerate([(x, y) for x in offsets_x for y in offsets_y]):
        create_static_cube(
            stage, f"/World/Table/Leg_{i}",
            size=leg_size,
            position=(lx, ly, TABLE_HEIGHT / 2),
            color=(0.45, 0.40, 0.33),
        )

    print(f"       Table: {TABLE_SIZE[0]}m x {TABLE_SIZE[1]}m, surface at z={TABLE_SURFACE_Z:.3f}m")

    # -------------------------------------------------------------------------
    # Franka Panda Robot
    # -------------------------------------------------------------------------
    print("[5/8] Adding Franka Panda robot...")

    assets_root = get_assets_root_path()
    if assets_root is None:
        carb.log_error("Could not find Isaac Sim assets folder!")
        print("       [ERROR] Assets root not found. Cannot add Franka robot.")
        print("       Make sure you're connected to the internet or have local assets cached.")
        simulation_app.close()
        sys.exit(1)

    franka_usd = assets_root + "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"
    stage_utils.add_reference_to_stage(
        usd_path=franka_usd,
        path="/World/Robot",
        variants=[("Gripper", "AlternateFinger"), ("Mesh", "Quality")],
    )

    # Position the robot — Franka USD already has xform ops, so set existing attribute
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    translate_attr = robot_prim.GetAttribute("xformOp:translate")
    if translate_attr:
        translate_attr.Set(Gf.Vec3d(*ROBOT_POSITION))
    else:
        UsdGeom.Xformable(robot_prim).AddTranslateOp().Set(Gf.Vec3d(*ROBOT_POSITION))

    print(f"       Robot at: {ROBOT_POSITION}")
    print(f"       USD: {franka_usd}")

    # -------------------------------------------------------------------------
    # Ball (Draggable — with rigid body physics)
    # -------------------------------------------------------------------------
    print("[6/8] Creating ball (draggable sphere)...")
    ball = UsdGeom.Sphere.Define(stage, "/World/Ball")
    ball.AddTranslateOp().Set(Gf.Vec3d(*BALL_POSITION))
    ball.CreateRadiusAttr(BALL_RADIUS)

    # Visual — bright red
    set_visual_material(stage, "/World/Ball", BALL_COLOR, "BallMaterial")

    # Rigid body physics (so it can be dragged and falls with gravity)
    UsdPhysics.RigidBodyAPI.Apply(ball.GetPrim())
    UsdPhysics.CollisionAPI.Apply(ball.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(ball.GetPrim())
    mass_api.CreateMassAttr(BALL_MASS)

    # PhysX damping — prevent excessive rolling
    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(ball.GetPrim())
    physx_rb.CreateAngularDampingAttr(0.5)
    physx_rb.CreateLinearDampingAttr(0.1)
    physx_rb.CreateMaxDepenetrationVelocityAttr(1.0)

    # High friction so gripper can hold it
    create_physics_material(stage, "/World/Ball", static_friction=0.8, dynamic_friction=0.7, restitution=0.3)

    print(f"       Ball: r={BALL_RADIUS}m, mass={BALL_MASS}kg at {BALL_POSITION}")
    print(f"       >>> DRAGGABLE — select and use Move tool (W) to place anywhere!")

    # -------------------------------------------------------------------------
    # Bucket (Open-top container — 4 walls + 1 floor)
    # -------------------------------------------------------------------------
    print("[7/8] Creating bucket...")
    UsdGeom.Xform.Define(stage, "/World/Bucket")
    bucket_xform = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Bucket"))
    bucket_xform.AddTranslateOp().Set(Gf.Vec3d(*BUCKET_POSITION))

    bi = BUCKET_INNER_SIZE
    bw = BUCKET_WALL
    bh = BUCKET_HEIGHT
    half = bi / 2.0

    # Floor
    create_static_cube(stage, "/World/Bucket/Floor",
                       size=(bi + 2 * bw, bi + 2 * bw, bw),
                       position=(0, 0, bw / 2),
                       color=BUCKET_COLOR)

    # Walls
    walls = [
        ("Front", (bi + 2 * bw, bw, bh), (0, -(half + bw / 2), bw + bh / 2)),
        ("Back",  (bi + 2 * bw, bw, bh), (0,  (half + bw / 2), bw + bh / 2)),
        ("Left",  (bw, bi, bh),           (-(half + bw / 2), 0, bw + bh / 2)),
        ("Right", (bw, bi, bh),           ( (half + bw / 2), 0, bw + bh / 2)),
    ]
    for name, size, pos in walls:
        create_static_cube(stage, f"/World/Bucket/{name}",
                           size=size, position=pos, color=BUCKET_COLOR)

    print(f"       Bucket: {bi}m x {bi}m x {bh}m at Y={BUCKET_POSITION[1]}m")

    # -------------------------------------------------------------------------
    # Overhead Camera
    # -------------------------------------------------------------------------
    print("[8/8] Adding overhead camera...")
    cam_xform = UsdGeom.Xform.Define(stage, "/World/OverheadCamera")
    cam_xform.AddTranslateOp().Set(Gf.Vec3d(*CAMERA_POSITION))
    # In USD (Z-up), camera default view direction is along -Z (straight down toward the table)
    cam_xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, 0.0))  # Looks straight down at table

    camera = UsdGeom.Camera.Define(stage, "/World/OverheadCamera/Camera")
    camera.CreateFocalLengthAttr(CAMERA_FOCAL_LENGTH)
    camera.CreateHorizontalApertureAttr(20.955)
    camera.CreateVerticalApertureAttr(15.2908)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 10.0))

    print(f"       Camera at: {CAMERA_POSITION}, looking down")

    # -------------------------------------------------------------------------
    # Save the Stage
    # -------------------------------------------------------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Update the stage before saving
    simulation_app.update()

    stage.GetRootLayer().Export(output_path)

    print("\n" + "=" * 70)
    print(f"  [OK] SCENE SAVED: {output_path}")
    print("=" * 70)
    print()
    print("  HOW TO USE:")
    print(f"  1. Open Isaac Sim -> File -> Open -> {output_path}")
    print("  2. Press PLAY to start physics")
    print("  3. Select /World/Ball -> press W (Move) -> drag it on table")
    print("  4. Enable com.rizwan.digitaltwin extension for PLC + inference")
    print()
    print("  SCENE CONTENTS:")
    print(f"   [Robot]   /World/Robot        - Franka Panda (AlternateFinger gripper)")
    print(f"   [Table]   /World/Table        - {TABLE_SIZE[0]}x{TABLE_SIZE[1]}m, surface at z={TABLE_SURFACE_Z:.2f}m")
    print(f"   [Ball]    /World/Ball         - r={BALL_RADIUS}m, {BALL_MASS}kg (DRAGGABLE)")
    print(f"   [Bucket]  /World/Bucket       - {bi}x{bi}x{bh}m at Y={BUCKET_POSITION[1]}m")
    print(f"   [Camera]  /World/OverheadCamera - looking down from z={CAMERA_POSITION[2]}m")
    print()


# ==============================================================================
# 6. Main
# ==============================================================================

if __name__ == "__main__":
    if args.output:
        output_path = args.output
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        output_path = os.path.join(project_root, "scenes", "runtime_scene.usd")

    build_scene(output_path)

    if not args.headless:
        print("[INFO] GUI mode — inspect the scene. Close the window to exit.\n")
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()
    print("[INFO] Done.")
