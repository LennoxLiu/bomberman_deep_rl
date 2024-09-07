from typing import Callable
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import torch as th
import settings as s
from imitation.algorithms.dagger import BetaSchedule
from imitation.algorithms.bc import BCLogger
import torch.nn as nn
from imitation.util import logger as imit_logger
import os
import shutil
import time
import numpy as np
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from imitation.util import logger as imit_logger
from imitation.data.wrappers import RolloutInfoWrapper
import torch.nn as nn
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.nn.functional as F

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return max(progress_remaining * initial_value, 1e-5)

    return func


# beta is a float between 0 and 1 that determines the probability of using the expert policy instead of the learner policy.
class CustomBetaSchedule(BetaSchedule):
    def __init__(self, logger, delta_beta = 0.005 ,beta0 = 1, beta_final: float = 0.05):
        self.beta_final = beta_final
        self.logger = logger
        self.delta_beta = delta_beta

        self.beta = beta0

    def  __call__(self, round_num: int) -> float:
        if round_num % 5 == 0:
            self.beta -= self.delta_beta # 0.0001 too small, 0.01 too large
            self.beta = max(self.beta, self.beta_final)

        self.logger.record("dagger/beta", self.beta)
        self.logger.dump(step=round_num)
        # self.logger.add_scalar("dagger/beta", self.beta, round_num)
        return self.beta


# beta is a float between 0 and 1 that determines the probability of using the expert policy instead of the learner policy.
class CustomBetaSchedule2(BetaSchedule):
    def __init__(self, logger, decrease_beta = 0.05,increase_beta = 0.01,beta0 = 1, beta_final: float = 0.05):
        self.beta_final = beta_final
        self.logger = logger
        self.decrease_beta = decrease_beta
        self.increase_beta = increase_beta

        self.beta = beta0

    def  __call__(self, round_num: int) -> float:
        self.logger.record("dagger/beta", self.beta)
        self.logger.dump(step=round_num)
        return self.beta
    
    def decrease(self):
        self.beta -= self.decrease_beta
        self.beta = max(self.beta, self.beta_final)
    
    def increase(self):
        self.beta += self.increase_beta
        self.beta = min(self.beta, 1)
    

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, network_configs: dict):
        super().__init__(observation_space, features_dim=network_configs['dense'][-1])
        # We assume 2x1xROWxCOL image (1 channel)
        n_input_channels = 1
        self.crop_range = network_configs['crop_range']
        cnn1_config = network_configs['cnn1']
        cnn1_strides = network_configs['cnn1_strides']
        self.cnn1 = nn.Sequential()
        self.cnn1.add_module('conv0', nn.Conv2d(n_input_channels, cnn1_config[0], kernel_size=3, stride=cnn1_strides[0], padding=1))
        self.cnn1.add_module('relu', nn.ReLU())
        for i in range(1, len(cnn1_config)):
            self.cnn1.add_module('conv'+str(i), nn.Conv2d(cnn1_config[i-1], cnn1_config[i], kernel_size=3, stride=cnn1_strides[i], padding=1))
            self.cnn1.add_module('relu', nn.ReLU())
        self.cnn1.add_module('flatten', nn.Flatten())

        cnn2_config = network_configs['cnn2']
        cnn2_strides = network_configs['cnn2_strides']
        self.cnn2 = nn.Sequential()
        self.cnn2.add_module('conv0', nn.Conv2d(n_input_channels, cnn2_config[0], kernel_size=3, stride=cnn2_strides[0], padding=1))
        self.cnn2.add_module('relu', nn.ReLU())
        for i in range(1, len(cnn2_config)):
            self.cnn2.add_module('conv'+str(i), nn.Conv2d(cnn2_config[i-1], cnn2_config[i], kernel_size=3, stride=cnn2_strides[i], padding=1))
            self.cnn2.add_module('relu', nn.ReLU())
        self.cnn2.add_module('flatten', nn.Flatten())

        # # Compute shape by doing one forward pass
        with th.no_grad():
            # print("observation_space.sample().shape:", observation_space.sample().shape)
            # print("type(observation_space.sample())", type(observation_space.sample()))
            n_flatten1 = self.cnn1(
                th.as_tensor(observation_space.sample()[0].reshape(-1, 1, s.ROWS, s.COLS)).float()
            ).shape[1]
            n_flatten2 = self.cnn2(
                th.as_tensor(observation_space.sample()[1].reshape(-1, 1, s.ROWS, s.COLS)).float()
            ).shape[1]

        linear_config = network_configs['dense']
        self.dense = nn.Sequential()
        self.dense.add_module('linear0', nn.Linear(n_flatten1+n_flatten2, linear_config[0]))
        self.dense.add_module('relu', nn.ReLU())
        for i in range(1, len(linear_config)):
            self.dense.add_module('linear'+str(i), nn.Linear(linear_config[i-1], linear_config[i]))
            self.dense.add_module('relu', nn.ReLU())
    

    # observation: (batch_size, 2, ROWS, COLS)
    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs1, obs2 = observations[:,0], observations[:, 1]
        
        # Get the agent positions and crop around them
        cropped_obs1 = self.crop_around_agent(obs1, [6,7] , self.crop_range)
        cropped_obs2 = self.crop_around_agent(obs2, [s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 3], self.crop_range)


        # Reshape and standardize the input to [0,1]
        cropped_obs1 = cropped_obs1.reshape(-1, 1, self.crop_range, self.crop_range) / 8
        cropped_obs2 = cropped_obs2.reshape(-1, 1, self.crop_range, self.crop_range) / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4
        
        return self.dense(th.cat([self.cnn1(cropped_obs1), self.cnn2(cropped_obs2)], dim=1))

    # Crop the observation around the agent's position
    # field_size: size x size, must be odd
    def crop_around_agent(self, obs: th.Tensor, agent_value_list: list, field_size = 29) -> th.Tensor:
        # Initialize a list to hold the cropped observations
        cropped_obs = []
        batch_size = obs.shape[0]
        for i in range(batch_size):
            # Find the agent's position in the current observation
            agent_pos = th.where(th.isin(obs[i], th.tensor(agent_value_list).to(obs.device)))

            # If agent is found, get the coordinates (x, y)
            if len(agent_pos[0]) > 0:
                x, y = agent_pos[0][0], agent_pos[1][0]
            else:
                # If no agent is found, throw an error
                raise ValueError("Agent not found in the observation")

            small = int((field_size-1)/2)
            large = int((field_size+1)/2)
            # Define the boundaries of the 7x7 grid
            min_x = max(0, x - small)
            max_x = min(s.ROWS, x + large)
            min_y = max(0, y - small)
            max_y = min(s.COLS, y + large)

            # Crop the observation to the field_size x field_size grid
            cropped = obs[i, min_x:max_x, min_y:max_y]

            # Calculate padding required to center the agent at (3, 3)
            pad_left = max(0, small - y)
            pad_right = max(0, (y + large) - s.COLS)
            pad_top = max(0, small - x)
            pad_bottom = max(0, (x + large) - s.ROWS)

            # Apply padding to the cropped observation to make it 7x7 and center the agent
            padded = F.pad(cropped, (pad_left, pad_right, pad_top, pad_bottom), value=-1)
            # Add the padded observation to the list
            cropped_obs.append(padded)

        # Stack all cropped observations back into a tensor
        return th.stack(cropped_obs)



def save_DAgger_trainer(trainer,configs):
    trainer.scratch_dir.mkdir(parents=True, exist_ok=True)

    trainer_dict = {
        'policy': trainer.policy,
        'bc_trainer': trainer.bc_trainer,
        'current_beta': trainer.beta_schedule.beta,
        'configs': configs,  # Add any important hyperparameters
        # Exclude non-pickleable items like logger, rng, and env
    }

    # save full trainer checkpoints
    checkpoint_paths = [
        trainer.scratch_dir / f"checkpoint-{trainer.round_num:03d}.pt",
        trainer.scratch_dir / "checkpoint-latest.pt",
    ]
    for checkpoint_path in checkpoint_paths:
        th.save(trainer_dict, checkpoint_path)


def load_DAgger_trainer(checkpoint_path):
    checkpoint = th.load(checkpoint_path)
    policy = checkpoint['policy']
    bc_trainer = checkpoint['bc_trainer']
    current_beta = checkpoint['current_beta']
    configs = checkpoint['configs']
    custom_logger = imit_logger.configure(folder='logs/tensorboard_logs',format_strs=["tensorboard"],)
    betaSchedule=CustomBetaSchedule2(custom_logger, beta0=current_beta, decrease_beta=configs["dagger_trainer"]["decrease_beta"], 
                                      increase_beta=configs["dagger_trainer"]["increase_beta"], beta_final=configs["dagger_trainer"]["beta_final"])
    bc_trainer._bc_logger = BCLogger(custom_logger) # the logger causes thread lock, makes it not pickable
    rng = np.random.default_rng(configs["SEED"])
    env = make_vec_env(
        'CustomEnv-v1',
        rng=rng,
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
        log_dir='logs',
    )
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir='checkpoints',
        expert_policy=policy,
        bc_trainer=bc_trainer,
        rng=rng,
        custom_logger=custom_logger,
        beta_schedule=betaSchedule,
    )
    print(f"Loaded DAgger trainer from {checkpoint_path}")
    
    if os.path.exists('checkpoints/demos'):
        shutil.rmtree('checkpoints/demos')
        print("Removed 'checkpoints/demos' folder to generate new data")

    return dagger_trainer, current_beta, configs

def load_tensorboard_log(tag,log_dir='logs/tensorboard_logs'):
    # Load the event accumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()  # Load the data

    tag_values = [0]

    # Access scalar metrics like 'loss' or 'accuracy'
    if tag in event_acc.Tags()['scalars']:
        tag_events = event_acc.Scalars(tag)  # Get all events for 'loss'
        tag_values = [event.value for event in tag_events]

    return tag_values


if __name__ == '__main__':
    values=load_tensorboard_log('dagger/mean_episode_reward',log_dir='logs/tensorboard_logs')
    print(values)