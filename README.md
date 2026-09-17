# Franka Panda Robotic Digital Twin: Industrial Automation & Physical AI

A production-ready robotics digital twin and physical AI pipeline for the 7-DOF Franka Emika Panda manipulator, developed in **NVIDIA Isaac Sim 3.0** and **Isaac Lab**. 

This project bridges traditional industrial automation with modern AI by connecting an industrial **Siemens S7-1500 PLC** to a simulated robotic workcell featuring computer vision, analytical kinematics, and deep reinforcement learning.

---

## Video Demonstrations

The system was evaluated through three experiments, demonstrating baseline motion smoothness, end-to-end Siemens PLC integration, and real-time dynamic re-planning:

### 1. Motion Smoothness & Baseline Pick-and-Place
*Evaluation of the Franka arm's continuous trajectory. The arm moves smoothly between waypoints, descends vertically onto the ball, and transfers it into the container without vibration or jitter.*

https://github.com/rizwan1602/pick_and_place_rl/raw/main/docs/videos/ppoc.mp4

<video src="docs/videos/ppoc.mp4" controls="controls" width="100%" style="max-width: 800px; border-radius: 8px;"></video>

[Direct Link to Video: `docs/videos/ppoc.mp4`](docs/videos/ppoc.mp4)

---

### 2. End-to-End Digital Twin with Siemens S7 PLC Integration
*Full industrial digital twin workflow. The Omniverse Siemens PLC extension docks into the bottom-right panel and connects via OPC UA (`opc.tcp://192.168.0.1:4840`). Pressing **START** writes bit `1` to `Data_block_1.start`, initiating the pick-and-place cycle. Pressing **STOP** writes `0` to halt motion, and **RESET** returns the arm to its standby home pose while resetting the ball to the table.*

https://github.com/rizwan1602/pick_and_place_rl/raw/main/docs/videos/final_video.mp4

<video src="docs/videos/final_video.mp4" controls="controls" width="100%" style="max-width: 800px; border-radius: 8px;"></video>

[Direct Link to Video: `docs/videos/final_video.mp4`](docs/videos/final_video.mp4)

---

### 3. Dynamic Re-Planning: Real-Time Ball Randomization
*Test of online trajectory adaptation. After the cycle begins, the ball position is randomized across the table surface. The perception and control pipeline immediately recalculates the target position and smoothly redirects the arm to the new pick location.*

https://github.com/rizwan1602/pick_and_place_rl/raw/main/docs/videos/live_random_pos.mp4

<video src="docs/videos/live_random_pos.mp4" controls="controls" width="100%" style="max-width: 800px; border-radius: 8px;"></video>

[Direct Link to Video: `docs/videos/live_random_pos.mp4`](docs/videos/live_random_pos.mp4)

---

## System Architecture

The digital twin connects physical and simulated industrial controllers to NVIDIA's GPU-accelerated PhysX engine:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SIEMENS S7 INDUSTRIAL PLC                       │
│   Siemens S7-1500 / PLCSIM Advanced (IP: 192.168.0.1:4840)             │
│   DB: "Data_block_1" | Tag: "start" (DB1.DBX0.0)                       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ OPC UA Protocol (Read/Write)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 OMNIVERSE KIT EXTENSION: siemens_plc                   │
│   - Auto-docks into bottom-right Property pane                         │
│   - Live Green Status LED ("PLC STATUS: ONLINE")                       │
│   - Industrial Buttons: [START] (Bit 1), [STOP] (Bit 0), [RESET] (Bit 0│
│   - Direct Table Ball Placement: [RANDOMIZE BALL ON TABLE]             │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Thread-Safe Bridge (RobotCellBridge)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PERCEPTION, POLICY & CONTROL LOOP                    │
│                                                                        │
│   Stage 2: Vision System        Stage 3: RL Policy     Stage 1: DLS IK │
│   - Overhead RGB-D Camera       - 41-dim State Vector  - Vertical Quat │
│   - HSV / Color Segmentation    - PPO Actor-Critic     - Equator Height│
│   - 3D Unprojection (<1mm)      - ActionSmoother       - 7-State FSM   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Joint Commands (100 Hz)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                  NVIDIA ISAAC SIM / PHYSX 5 GPU ENGINE                 │
│   - Franka Emika Panda (7-DOF Arm + Parallel Gripper)                  │
│   - Rigid Body GPU Tensor Updates                                      │
│   - Workcell Stage: Table, 5-Walled Container, Lighting                │
└────────────────────────────────────────────────────────────────────────┘
```

---

## How the Pipeline Works

The project is structured into four progressive stages:

### Stage 1: Deterministic Inverse Kinematics (IK)
Stage 1 implements an analytical Damped Least Squares (DLS) Jacobian controller to establish a deterministic 100% reliable baseline.

#### Physics & Geometry Solutions
1. **Eliminating the 45-Degree Gripper Tilt**: Standard Isaac Sim tutorials use a downward quaternion `[0.9256, 0.048, 0.3751, 0.018]`, which introduces a 45-degree pitch tilt. Clamping flat finger plates onto a sphere at an angle turns the grasp into an inclined ramp, ejecting the ball across the table upon contact. We derived the pure downward vertical quaternion:
   $$\mathbf{q}_{\text{vertical}} = [1.0, 0.0, 0.0, 0.0]$$
   This aligns the fingers straight down, reducing tilt error from 72 mm to 0.71 mm and providing perfectly horizontal clamping forces.
2. **Sub-Millimeter Table Clearance**: Franka fingertip meshes extend 10.5 cm below the `panda_hand` frame. Descending below $Z = 0.50\,\text{m}$ drives the metal tips into the table surface ($Z = 0.42\,\text{m}$), causing physics impulse shocks that knock the target away. The grasp height sweet spot is set at **$\text{Hand } Z = 0.545\,\text{m}$**, keeping the fingertips 8 mm above the table while clamping the ball symmetrically at its center equator ($Z = 0.45\,\text{m}$).
3. **Collision-Free Standby Home Pose**: Traditional home positions park the hand at $Z \approx 0.55\,\text{m}$, which can sweep against objects during reset. We formulated an elevated home joint configuration:
   $$\mathbf{q}_{\text{home}} = [-0.026, -0.6079, 0.0171, -2.0867, 0.0098, 1.4793, 0.7742, 0.04, 0.04]$$
   This parks the hand at $Z = 0.650\,\text{m}$ (>14 cm clearance above the table), guaranteeing zero collision kicks when the cell resets.
4. **7-State Industrial State Machine**: The arm sequences through `APPROACH` (hover at $Z = 0.65\,\text{m}$) $\to$ `DESCEND` (lower to $Z = 0.545\,\text{m}$) $\to$ `GRASP` (close fingers) $\to$ `LIFT` (+97 mm vertical lift) $\to$ `TRANSIT` (move horizontally to bucket) $\to$ `RELEASE` (open fingers) $\to$ `RETRACT` (return to home).

#### Benchmark Results
Tested over consecutive cycles with randomized spawn positions across the table surface:

| Cycle | Target Spawn (X, Y) | Gripper Joint State | Vertical Lift | Placement Cavity Position | Cycle Time | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | (0.350m,  0.150m) | [0.0288, 0.0294] | +97.0 mm | (0.351, 0.313, 0.455) | 12.37 s | **PASSED (100%)** |
| **2** | (0.300m, -0.100m) | [0.0286, 0.0296] | +96.7 mm | (0.322, 0.314, 0.455) | 12.06 s | **PASSED (100%)** |
| **3** | (0.420m,  0.050m) | [0.0287, 0.0295] | +96.9 mm | (0.349, 0.313, 0.455) | 10.55 s | **PASSED (100%)** |
| **4** | (0.280m,  0.120m) | [0.0288, 0.0294] | +96.8 mm | (0.352, 0.312, 0.455) | 11.10 s | **PASSED (100%)** |
| **5** | (0.380m, -0.080m) | [0.0286, 0.0296] | +96.9 mm | (0.348, 0.313, 0.455) | 11.42 s | **PASSED (100%)** |

---

### Stage 2: Computer Vision & 3D Perception
Stage 2 replaces ground-truth coordinate lookups with an **overhead Eye-to-Hand vision sensor**:
* **Sensor Pose**: Synthetic RGB-D camera mounted at $[0.50, 0.00, 1.30]\,\text{m}$, pointed straight down at the workspace.
* **Color Segmentation**: Real-time chromatic thresholding filters the target ball from table reflections and background noise.
* **Ray-Plane Unprojection**: The optical pixel centroid $(\bar{u}, \bar{v})$ is unprojected through the camera pinhole model down to the table surface plane ($Z = Z_{\text{table}} + r_{\text{ball}} = 0.455\,\text{m}$), yielding real-world coordinates with sub-millimeter precision ($< 0.85\,\text{mm}$ error).
* **Interactive Click-to-Place**: Clicking on the camera HUD calculates the corresponding table coordinate and updates the ball position in the simulation.

---

### Stage 3: Deep Reinforcement Learning (RSL-RL PPO)
Stage 3 trains a continuous neural network policy using Proximal Policy Optimization (PPO) inside Isaac Lab.

* **Observation Space (41 dimensions)**:
  * Arm joint angles $q$ (7) and velocities $\dot{q}$ (7)
  * End-effector position $[X, Y, Z]$ (3) and quaternion orientation (4)
  * End-effector linear and angular velocities (6)
  * Target ball position (3) and relative vector from hand to ball (3)
  * Bucket cavity position (3) and relative vector from ball to bucket (3)
  * Previous action vector (8) for smooth transitions
* **Action Space (8 dimensions)**: 7 continuous joint velocity deltas + 1 gripper command.
* **Reward Curriculum**: Shaped with reaching incentives (exponential distance penalty), lifting bonuses ($Z > 0.48\,\text{m}$), horizontal bucket alignment, containment rewards inside the bucket, and regularizations on joint velocities and accelerations.
* **Vibration-Free Action Smoothing (`ActionSmoother`)**: High-frequency neural policies (50–100 Hz) can exhibit micro-jitter from rapid action switching. We apply an Exponential Moving Average (EMA) filter directly to the policy action outputs:
  $$\mathbf{a}_t^{\text{exec}} = \alpha \cdot \mathbf{a}_{t-1}^{\text{exec}} + (1 - \alpha) \cdot \mathbf{a}_t^{\text{raw}}, \quad (\alpha = 0.65)$$
  This completely removes 50 Hz micro-jitter while preserving full motor speed and torque authority.

---

### Stage 4: Siemens S7 PLC & Omniverse Digital Twin
Stage 4 connects the simulated cell to a **Siemens S7-1500 PLC** via industrial OPC UA.

* **OPC UA Endpoint**: `opc.tcp://192.168.0.1:4840`
* **Data Block Tag**: `Data_block_1.start` (Node ID: `ns=3;s="Data_block_1"."start"`, address `DB1.DBX0.0`)
* **Omniverse Extension (`com.rizwan.siemens_plc`)**:
  * **Auto-Docking**: Docks into the bottom-right `Property` panel on startup.
  * **Live Status LED**: Bright green indicator confirming active PLC communication (`PLC STATUS: ONLINE`).
  * **Unified Push Buttons**:
    * **`[START]` (Green, 50px)**: Writes bit `1` (TRUE) to the Siemens DB tag and starts the cell cycle. If the ball was already in the container from a previous run, it automatically returns to the table first.
    * **`[STOP]` (Red, 50px)**: Writes bit `0` (FALSE) to the Siemens DB tag, immediately stopping robot motion.
    * **`[RESET]` (Amber, 50px)**: Writes bit `0` (FALSE) to the Siemens DB tag and triggers a **Full Cell Reset**: returns the Franka arm to the elevated Standby Home pose and resets the ball from the container back onto the table.
    * **`[RANDOMIZE BALL ON TABLE]` (Indigo, 44px)**: Teleports the ball to random coordinates across the table surface.

---

## Repository Structure

```
POC/
├── docs/
│   └── videos/                   # Video demonstrations (ppoc.mp4, final_video.mp4, live_random_pos.mp4)
├── ball_pick_place/              # Custom Isaac Lab RL Environment
│   ├── agents/
│   │   └── rsl_rl_ppo_cfg.py     # PPO RL hyperparameters and network architecture
│   ├── tasks/
│   │   └── pick_place_ball/
│   │       ├── env_cfg.py        # Scene definition and environment configurations
│   │       └── mdp/
│   │           ├── geometry.py   # Workcell dimensions and analytical coordinates
│   │           ├── rewards.py    # Reward curriculum functions
│   │           ├── observations.py # 41-dimensional observation builders
│   │           ├── terminations.py # Success and boundary termination rules
│   │           └── events.py     # Domain randomization
├── extensions/
│   └── com.rizwan.siemens_plc/   # Omniverse Kit Extension for Siemens S7 PLC
│       ├── extension.toml        # Extension manifest
│       └── siemens_plc/
│           ├── extension.py      # Extension lifecycle and startup
│           ├── bridge_state.py   # Thread-safe singleton bridge (RobotCellBridge)
│           ├── opcua_mgr.py      # Async OPC UA client for Siemens PLC
│           └── ui.py             # Docked industrial UI panel with START/STOP/RESET
├── scenes/
│   ├── bucket.usd                # 5-walled placement container with collision mesh
│   └── runtime_scene.usd         # Complete USD stage (robot, table, container, lights)
├── scripts/
│   ├── run_deterministic_ik.py   # [Stage 1] 100% Deterministic IK pick-and-place runner
│   ├── run_vision_pick_place.py  # [Stage 2] Overhead RGB-D vision-guided IK runner
│   ├── train.py                  # [Stage 3] Distributed RL policy training runner (PPO)
│   ├── play.py                   # [Stage 3+4] Trained RL policy + Vision HUD + Siemens PLC
│   ├── export_policy.py          # TorchScript JIT exporter (.pt)
│   └── create_runtime_scene.py   # Procedural USD scene generator
├── setup.py                      # Pip package definition (pip install -e .)
└── README.md                     # Technical documentation
```

---

## Quick Start Guide

### 1. Installation
Install the custom environment package into your Isaac Sim Python environment:
```powershell
cd "C:\Working\2026\Digital Twin V2\Project\POC"
& "C:\Working\2026\Digital Twin V2\IsaacLab\_isaac_sim\python.bat" -m pip install -e .
```

### 2. Stage 1: Run Deterministic IK Controller
Runs the analytical DLS inverse kinematics controller with automated multi-cycle evaluation:
```powershell
# Interactive 3D Visualization:
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/run_deterministic_ik.py

# Automated Headless Benchmark (5 cycles):
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/run_deterministic_ik.py --headless --num_cycles 5
```

### 3. Stage 2: Run Vision-Guided Pick-and-Place
Runs the overhead 3D perception pipeline with live OpenCV tracking HUD:
```powershell
# Interactive GUI Mode:
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/run_vision_pick_place.py

# Automated Headless Benchmark (5 cycles):
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/run_vision_pick_place.py --headless --num_cycles 5
```

### 4. Stage 3: Train Reinforcement Learning Policy (PPO)
Launches GPU-parallelized training across 4,096 simulation environments:
```powershell
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/train.py --task BallPickPlace-Franka-v0 --num_envs 4096 --headless
```

### 5. Stage 4: Run Integrated Digital Twin (RL + Vision + Siemens PLC)
Launches the trained model checkpoint, OpenCV interactive camera HUD, and the docked Siemens PLC extension:
```powershell
& "C:\Working\2026\Digital Twin V2\IsaacLab\isaaclab.bat" -p scripts/play.py --checkpoint logs/rsl_rl/ball_pick_place_franka/2026-09-17_13-04-33/model_499.pt
```

---

## Interactive Controls & Hotkeys

When running `scripts/play.py`:

| Action | UI Button | Keyboard Hotkey | Functionality |
| :--- | :--- | :---: | :--- |
| **Start Cycle** | `[START]` (Green) | <kbd>S</kbd> | Writes bit `1` to PLC; starts pick-and-place cycle |
| **Halt Motion** | `[STOP]` (Red) | <kbd>Space</kbd> | Writes bit `0` to PLC; halts robot arm instantly |
| **Cell Reset** | `[RESET]` (Amber) | <kbd>R</kbd> | Writes bit `0` to PLC; homes robot arm & resets ball to table |
| **Randomize Ball** | `[RANDOMIZE BALL]` | <kbd>B</kbd> | Teleports ball to a random position on the table |
| **Click-to-Place** | *Click OpenCV HUD* | *Mouse Click* | Moves ball to the exact clicked coordinates in camera view |
| **Exit** | Close Window | <kbd>Q</kbd> or <kbd>Esc</kbd> | Closes OpenCV HUD and cleans up simulation |

---

## License & Attribution
Developed by **Rizwan** for Industrial Digital Twin and Physical AI automation POC. Built on top of NVIDIA Omniverse Isaac Sim, Isaac Lab, and Franka Emika Panda robotics assets.
