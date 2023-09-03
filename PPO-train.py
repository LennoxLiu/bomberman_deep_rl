import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
import torch as th
import torch.nn as nn
import settings as s
from gymnasium import spaces


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

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
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
        # self.cnn = nn.Sequential(
        #     nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 128, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[128 ,64, 32], vf=[128, 64, 32])
    # Custom actor (pi) and value function (vf) networks
    # of two layers of size 32 each with Relu activation function
    # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
)

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            "rule_based_agent","rule_based_agent","rule_based_agent", \
                                            "--scenario","classic"],
        "enable_rule_based_agent_reward": True}
model_path = "./Original/agent_code/PPO_agent/ppo_bomberman"
env = CustomEnv()
env.metadata = option
# env = gym.wrappers.NormalizeReward(env)
model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, learning_rate = 0.0003, n_steps = 512, batch_size = 64, stats_window_size = 100)


new_parameters = {
    "learning_rate": 0.0003,
    "n_steps": 512, # more n_steps means more robust, less tuned
    "batch_size": 64,
    "stats_window_size":  100,
    "clip_range": 0.2,
    }
# model = PPO.load(model_path, env = env, force_reset = True, custom_objects = new_parameters) 
while True:
    model.learn(total_timesteps=int(2048*25), progress_bar=True, log_interval = 2)
    # total_timesteps=20480
    model.save(model_path)
