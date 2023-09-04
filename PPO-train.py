import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
import torch as th
import torch.nn as nn
import settings as s
from gymnasium import spaces
import os
import shutil

def delete_contents_of_folder(folder_path):
    try:
        # Get a list of all files and folders inside the specified folder
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            
            # If it's a file, delete it directly
            if os.path.isfile(item_path):
                os.remove(item_path)
            
            # If it's a folder, recursively delete its contents and then delete the folder itself
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        
        print(f"Successfully deleted all files and folders inside {folder_path}.")
    
    except Exception as e:
        print(f"An error occurred while deleting files and folders inside {folder_path}: {str(e)}")


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

from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


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
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # nn.Flatten(),
        )
        n_flatten = 128

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.mlp(observations))


policy_kwargs = dict(
    features_extractor_class=CustomMLP,
    features_extractor_kwargs=dict(features_dim=128),
    activation_fn=th.nn.ReLU,
    net_arch=dict(pi=[128,128, 64, 64, 32], vf=[128, 128, 64, 64, 32])
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

# delete_contents_of_folder("./tb_log/")
model = PPO("MlpPolicy", env, learning_rate = 0.0001, n_steps = 2048, batch_size = 256, verbose=0,tensorboard_log="./tb_log/", policy_kwargs=policy_kwargs, stats_window_size = 100)

new_parameters = {
    "learning_rate": 0.0001,
    "n_steps": 2048, # more n_steps means more robust, less tuned
    "batch_size": 256,
    "stats_window_size":  100,
    "clip_range": 0.2,
    "tensorboard_log":"./tb_log/",
    "verbose": 0
    }

# model = PPO.load(model_path, env = env, force_reset = True, custom_objects = new_parameters) 

while True:
    model.learn(total_timesteps=int(2048*25), progress_bar=True, log_interval = 1, reset_num_timesteps=False)
    # total_timesteps=20480
    model.save(model_path)