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
        self.beta = min(self.beta, 0.95)
    

# crop_size must be odd
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
        self.crop_size = network_configs['crop_size']

        cnn1_config = network_configs['cnn1']
        cnn1_strides = network_configs['cnn1_strides']
        self.cnn1 = nn.Sequential()
        self.cnn1.add_module('conv0', nn.Conv2d(n_input_channels, cnn1_config[0], kernel_size=2, stride=cnn1_strides[0], padding=0))
        self.cnn1.add_module('relu', nn.ReLU())
        for i in range(1, len(cnn1_config)):
            self.cnn1.add_module('conv'+str(i), nn.Conv2d(cnn1_config[i-1], cnn1_config[i], kernel_size=3, stride=cnn1_strides[i], padding=0))
            self.cnn1.add_module('relu', nn.ReLU())
        
        # self.cnn1.add_module('maxpool', nn.MaxPool2d(kernel_size=2, stride=1))
        self.cnn1.add_module('flatten', nn.Flatten())

        cnn2_config = network_configs['cnn2']
        cnn2_strides = network_configs['cnn2_strides']
        self.cnn2 = nn.Sequential()
        self.cnn2.add_module('conv0', nn.Conv2d(n_input_channels, cnn2_config[0], kernel_size=2, stride=cnn2_strides[0], padding=0))
        self.cnn2.add_module('relu', nn.ReLU())
        for i in range(1, len(cnn2_config)):
            self.cnn2.add_module('conv'+str(i), nn.Conv2d(cnn2_config[i-1], cnn2_config[i], kernel_size=3, stride=cnn2_strides[i], padding=0))
            self.cnn2.add_module('relu', nn.ReLU())
        
        # self.cnn2.add_module('maxpool', nn.MaxPool2d(kernel_size=2, stride=1))
        self.cnn2.add_module('flatten', nn.Flatten())

        with th.no_grad():
            # Reshape inputs for passing through the CNNs
            obs1_sample = th.as_tensor(observation_space.sample()[0]).float()
            obs2_sample = th.as_tensor(observation_space.sample()[1]).float()

            crop_diam = int((self.crop_size - 1) / 2)
            obs1_sample = obs1_sample[s.ROWS-crop_diam:s.ROWS+crop_diam+1,s.COLS-crop_diam:s.COLS+crop_diam+1]
            obs2_sample = obs2_sample[s.ROWS-crop_diam:s.ROWS+crop_diam+1,s.COLS-crop_diam:s.COLS+crop_diam+1]
            
            # Pass through CNNs and calculate flatten sizes
            n_flatten1 = self.cnn1(obs1_sample.reshape(-1, 1, self.crop_size, self.crop_size)).shape[1]
            n_flatten2 = self.cnn2(obs2_sample.reshape(-1, 1, self.crop_size, self.crop_size)).shape[1]

            print(f"n_flatten1: {n_flatten1}, n_flatten2: {n_flatten2}")
        
        
        self.dense1 = nn.Sequential()
        self.dense1.add_module('linear0', nn.Linear(n_flatten1, network_configs['dense1']))
        self.dense1.add_module('relu', nn.ReLU())

        self.dense2 = nn.Sequential()
        self.dense2.add_module('linear0', nn.Linear(n_flatten2, network_configs['dense2']))
        self.dense2.add_module('relu', nn.ReLU())


        linear_config = network_configs['dense']
        self.dense = nn.Sequential()
        self.dense.add_module('linear0', nn.Linear(network_configs['dense1']+network_configs['dense2'], linear_config[0]))
        self.dense.add_module('relu', nn.ReLU())
        for i in range(1, len(linear_config)):
            self.dense.add_module('linear'+str(i), nn.Linear(linear_config[i-1], linear_config[i]))
            self.dense.add_module('relu', nn.ReLU())
    

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs1, obs2 = observations[:,0], observations[:, 1]
        
        crop_diam = int((self.crop_size - 1) / 2)
        # crop obs to crop_size x crop_size
        obs1 = obs1[:,s.ROWS-crop_diam:s.ROWS+crop_diam+1,s.COLS-crop_diam:s.COLS+crop_diam+1]
        obs2 = obs2[:,s.ROWS-crop_diam:s.ROWS+crop_diam+1,s.COLS-crop_diam:s.COLS+crop_diam+1]
        
        # Reshape and standardize the input to [0,1]
        obs1 = obs1.reshape(-1, 1, self.crop_size, self.crop_size) / 8
        obs2 = obs2.reshape(-1, 1, self.crop_size, self.crop_size) / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

        return self.dense(th.cat([self.dense1(self.cnn1(obs1)), self.dense2(self.cnn2(obs2))], dim=1))


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
        # trainer.scratch_dir / f"checkpoint-{trainer.round_num:03d}.pt",
        trainer.scratch_dir / "checkpoint-latest.pt",
    ]
    for checkpoint_path in checkpoint_paths:
        th.save(trainer_dict, checkpoint_path, pickle_protocol=5)


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