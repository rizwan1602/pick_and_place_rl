import gymnasium as gym

gym.register(
    id="BallPickPlace-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BallPickPlaceFrankaEnvCfg",
    },
)

gym.register(
    id="BallPickPlace-Franka-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:BallPickPlaceFrankaEnvCfg_PLAY",
    },
)
