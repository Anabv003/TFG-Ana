import gymnasium as gym
from . import env_hover, env_z_rotation, env_command

for i in range(100):
    gym.register(
        id=f"Isaac-Command1-Aerotaxi-RANDOM-{i+1}",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{env_command.__name__}.env1.env:UAVEnvCfg",
            "rsl_rl_cfg_entry_point": f"{env_command.__name__}.env1.grid_hiperparametros.cfg_files.ppo_random_command1_{i+1}_cfg:Command1PPORunnerCfg",
        },
    )