import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
from gymnasium import spaces

import torch as th
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

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

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            "rule_based_agent","rule_based_agent","rule_based_agent", \
                                            "--scenario","classic"],
        "enable_rule_based_agent_reward": True}
model_path = "./Original/agent_code/DQN_agent/dqn_bomberman"

env = CustomEnv(options = option)
env.metadata = option


class CustomMLP(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(n_input_channels, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            # nn.Flatten(),
        )
        n_flatten = 256

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.mlp(observations))


policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=256),
)

model = DQN("MlpPolicy", env, learning_starts=0,
            tau = 0.8,
            gamma = 0.1, # training by rule_based_agent, only need immediate reward
            learning_rate = 0.0001,
            target_update_interval= 10240,
            exploration_fraction=0.9,
            exploration_initial_eps = 0.9,
            exploration_final_eps = 0.1,
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
    "target_update_interval": 10240, # more n_steps means more robust, less tuned
    "batch_size": 64,
    "tau": 0.05,
    # "gamma": 0.9,
    # "exploration_fraction": 0.99,
    # "exploration_initial_eps": 0.5,
    "exploration_final_eps":0.1,
    "stats_window_size": 100
    }
# model = DQN.load(model_path, env = env, force_reset = True, custom_objects = new_parameters) 
while True:
    model.learn( total_timesteps=102400, progress_bar=True, log_interval = 100, reset_num_timesteps=False)
    # total_timesteps=61440
    model.save(model_path)
