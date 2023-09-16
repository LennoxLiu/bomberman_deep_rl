import gymnasium as gym
import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
from gymnasium import spaces

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import SubprocVecEnv


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(n_input_channels, 32),
            nn.ReLU(),

            nn.Linear(32, 64),
            nn.ReLU(),
            
            nn.Linear(64, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.mlp(observations)


if __name__ == '__main__':
    model_path = "./Original/agent_code/DQN_agent/dqn_bomberman"
    option={"argv": ["play","--no-gui","--agents","user_agent",\
                                                "rule_based_agent","rule_based_agent","rule_based_agent", \
                                                "--scenario","classic"],
            "enable_rule_based_agent_reward": True}

    env = CustomEnv(options = option)
    # env_vec = make_vec_env(CustomEnv,n_envs=1,seed=np.random.randint(0, 2**31 - 1), env_kwargs={"options":option})
    # envs = [CustomEnv(option) for _ in range(16)]
    # env_vec = SubprocVecEnv(envs)

    policy_kwargs = dict(
        features_extractor_class=CustomMLP,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=[64, 32, 16]
    )

    model = DQN("MlpPolicy", env, learning_starts=0,
                device="auto",
                batch_size = 64,
                tau = 0.001, #0.8
                gamma = 0.5, # 0.9 #0.1 training by rule_based_agent, only need immediate reward
                learning_rate = 0.0003,#0.0003
                target_update_interval= 10240,
                exploration_fraction=0.99, # 0.9
                exploration_initial_eps = 1,
                exploration_final_eps = 0.2,
                stats_window_size= 100,
                policy_kwargs = policy_kwargs,
                tensorboard_log="./tb_log/",
                verbose = 0
                )

    # target_update_interval. Increasing the frequency of updates can make learning more stable but might slow down convergence, while decreasing the frequency can make learning more efficient but potentially less stable.
    # A higher γ places more emphasis on long-term rewards and encourages the agent to consider future consequences in its decision-making. A lower γ makes the agent more focused on immediate rewards.
    # tau: cita = tau* cita_new + (1-tau)*cita_old

    # (policy: str | type[DQNPolicy], env: GymEnv | str,
    #  learning_rate: float | Schedule = 0.0001,
    #  buffer_size: int = 1000000, learning_starts: int = 50000,
    #  batch_size: int = 32, tau: float = 1, gamma: float = 0.99,
    #  train_freq: int | Tuple[int, str] = 4, gradient_steps: int = 1,
    #  replay_buffer_class: type[ReplayBuffer] | None = None,
    #  replay_buffer_kwargs: Dict[str, Any] | None = None,
    #  optimize_memory_usage: bool = False, target_update_interval: int = 10000,
    #  exploration_fraction: float = 0.1, exploration_initial_eps: float = 1,
    #  exploration_final_eps: float = 0.05, max_grad_norm: float = 10,
    #  stats_window_size: int = 100, tensorboard_log: str | None = None,
    #  policy_kwargs: Dict[str, Any] | None = None, verbose: int = 0, seed: int | None = None,
    #  device: device | str = "auto", _init_setup_model: bool = True) -> None

    new_parameters = {
        "learning_rate": 0.0001,
        # "target_update_interval": 10240, # more n_steps means more robust, less tuned
        # "batch_size": 64,
        # "tau": 0.8,#0.05,
        "gamma": 0.8,
        "exploration_fraction": 1,
        "exploration_initial_eps": 0.3,
        "exploration_final_eps":0.2,
        # "stats_window_size": 100,
        "device":"cpu"
        }
    # model = DQN.load(model_path,env = env, force_reset = True, custom_objects = new_parameters) #
    # model.learn( total_timesteps=10240*2, progress_bar=True, log_interval = 100, reset_num_timesteps=True)
    while True:
        model.learn( total_timesteps=10240*5, progress_bar=True, log_interval = 100, reset_num_timesteps=False)
        # total_timesteps=61440
        model.save(model_path)
