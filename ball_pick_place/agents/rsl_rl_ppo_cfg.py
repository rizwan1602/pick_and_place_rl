# SPDX-License-Identifier: Apache-2.0
"""
RSL-RL PPO Configuration for Franka Ball Pick-and-Place.

Hyperparameters aligned with official NVIDIA IsaacLab-Arena manipulation tasks:
  - Observation normalization enabled on Actor and Critic (stabilizes value function)
  - Entropy coefficient = 0.002 (prevents action noise explosion / flailing)
  - Gamma = 0.995 (long-horizon credit assignment for multi-stage manipulation)
  - Learning rate = 3e-4 with adaptive schedule
"""

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from isaaclab.utils.configclass import configclass


@configclass
class BallPickPlacePPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner configuration for ball pick and place."""

    num_steps_per_env = 48
    max_iterations = 3000
    save_interval = 100
    experiment_name = "ball_pick_place_franka"
    run_name = ""
    logger = "tensorboard"
    neptune_project = "orbit"
    wandb_project = "orbit"

    actor = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,  # Stabilizes training and prevents divergence
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 128],
        activation="elu",
        obs_normalization=True,  # Stabilizes value function estimation
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,     # Matches NVIDIA Franka benchmarks: prevents action std explosion
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.5e-4,   # Clean, stable learning rate
        schedule="adaptive",
        gamma=0.98,             # Standard manipulation discount factor (focuses on fast completion)
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
