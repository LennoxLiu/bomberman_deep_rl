import os
import pathlib
import pickle
import shutil
import tempfile
import time
from typing import Callable
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from torch import device
import torch
from torch.cuda import is_available
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save
from imitation.data.wrappers import RolloutInfoWrapper
from RuleBasedPolicy import RuleBasedPolicy
from test_win_rate import test_against_RuleBasedAgent
from tqdm import tqdm
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
from gymnasium import spaces
import torch as th
import settings as s
from torch.utils.tensorboard import SummaryWriter

my_device = device("cuda" if is_available() else "cpu")
print("Using device:", my_device)

remove_logs_checkpoints = input("Do you want to remove existing 'logs' and 'checkpoints' folders? (y/n): ")
if remove_logs_checkpoints.lower() == 'y':
    # Remove existing 'logs' and 'checkpoints' folders
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')


# Create a SummaryWriter for logging to TensorBoard
os.makedirs('logs/tensorboard_logs', exist_ok=True)
# Configure the custom logger to use the SummaryWriter
# custom_logger = SummaryWriter(log_dir='logs/tensorboard_logs')
custom_logger = imit_logger.configure(
        folder='logs/tensorboard_logs',
        format_strs=["tensorboard"],
    )

os.makedirs('checkpoints', exist_ok=True)
def callback(round_num: int, /) -> None:
    # if checkpoint_interval > 0 and  round_num % checkpoint_interval == 0:
    save(bc_trainer, pathlib.Path(f"checkpoints/checkpoint{round_num:05d}"))

SEED = 42
rng = np.random.default_rng(SEED)
env = make_vec_env(
    'CustomEnv-v1',
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
    log_dir='logs',
)

expert = RuleBasedPolicy(env.observation_space, env.action_space)

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

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim = [128, 64]):
        super().__init__(observation_space, features_dim=sum(features_dim))
        # We assume 2x1xROWxCOL image (1 channel), but input as (HxWx2)
        n_input_channels = 1
        self.cnn1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            # print("observation_space.sample().shape:", observation_space.sample().shape)
            # print("type(observation_space.sample())", type(observation_space.sample()))
            n_flatten1 = self.cnn1(
                th.as_tensor(observation_space.sample()[0].reshape(-1, 1, s.ROWS, s.COLS)).float()
            ).shape[1]
            n_flatten2 = self.cnn2(
                th.as_tensor(observation_space.sample()[1].reshape(-1, 1, s.ROWS, s.COLS)).float()
            ).shape[1]

        self.linear1 = nn.Sequential(nn.Linear(n_flatten1, features_dim[0]), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(n_flatten2, features_dim[1]), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # print("observations.shape:", observations.shape)
        obs1, obs2 = observations[:,0], observations[:, 1]
        # print("obs1.shape:", obs1.shape)
        # print("obs2.shape:", obs2.shape)
        
        # Reshape and standardize the input to [0,1]
        obs1 = obs1.reshape(-1, 1, s.ROWS, s.COLS) / 8
        obs2 = obs2.reshape(-1, 1, s.ROWS, s.COLS) / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

        # print("obs1.shape:", obs1.shape)
        # print("obs2.shape:", obs2.shape)
        return th.cat([self.linear1(self.cnn1(obs1)), self.linear2(self.cnn2(obs2))], dim=1)

configs = {
    "bc_trainer": {
        "batch_size": 256, # The number of samples in each batch of expert data.
        "minibatch_size": 256, # if GPU memory is not enough, reduce this number to a factor of batch_size
        "l2_weight": 5e-4,
        "policy":{
            "learning_rate": 0.0003,
            "net_arch": [128, 64, 32],
            "features_extractor_class": "CustomCNN",
            "features_extractor_kwargs": {
                "features_dim": [128, 64]
        }}
    },
    "dagger_trainer": {
        "rollout_round_min_episodes": 3, # The number of episodes the must be completed completed before a dataset aggregation step ends.
        "rollout_round_min_timesteps": 1024, #The number of environment timesteps that must be completed before a dataset aggregation step ends. Also, that any round will always train for at least self.batch_size timesteps, because otherwise BC could fail to receive any batches.
        "bc_train_kwargs": {
            "n_epochs": 8,
        },
    }
}

time_steps_per_round = configs["dagger_trainer"]["rollout_round_min_timesteps"]*configs["dagger_trainer"]['bc_train_kwargs']['n_epochs']

# The agent is trained in “rounds” where each round consists of a dataset aggregation step followed by BC update step.
# During a dataset aggregation step, self.expert_policy is used to perform rollouts in the environment but there is a 1 - beta chance (beta is determined from the round number and self.beta_schedule) that the DAgger agent’s action is used instead. Regardless of whether the DAgger agent’s action is used during the rollout, the expert action and corresponding observation are always appended to the dataset. The number of environment steps in the dataset aggregation stage is determined by the rollout_round_min* arguments.
# During a BC update step, BC.train() is called to update the DAgger agent on all data collected so far.
bc_trainer = bc.BC(
    batch_size=configs['bc_trainer']['batch_size'],
    minibatch_size=configs['bc_trainer']['minibatch_size'],
    l2_weight=configs['bc_trainer']['l2_weight'],
    policy=ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        linear_schedule(configs['bc_trainer']['policy']["learning_rate"]),
        net_arch=configs['bc_trainer']['policy']["net_arch"],
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=configs['bc_trainer']['policy']["features_extractor_kwargs"],
    ),
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    device=my_device,
    custom_logger=custom_logger,
)

start_time = time.time()
win_rates = []
score_per_rounds = []
os.makedirs('checkpoints', exist_ok=True)
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir='checkpoints',
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rng,
    custom_logger=custom_logger,
)

############# Start training #############
rew_before_training, _ = evaluate_policy(dagger_trainer.policy, env, 100)
print(f"Mean reward before training:{np.mean(rew_before_training):.2f}")

# total_timesteps (int) – The number of timesteps to train inside the environment. 
# In practice this is a lower bound, because the number of timesteps is rounded up to finish the minimum number of episodes or timesteps in the last DAgger training round, and the environment timesteps are executed in multiples of self.venv.num_envs.
# for round_id in tqdm(range(30)):

round_id=1
custom_logger.record("a/win_rate", 0)
custom_logger.record("a/score_per_round", 0)
custom_logger.dump(step=0)
while True:
    dagger_trainer.train(total_timesteps = time_steps_per_round,
                            rollout_round_min_episodes=configs["dagger_trainer"]["rollout_round_min_episodes"],
                            rollout_round_min_timesteps=configs["dagger_trainer"]["rollout_round_min_timesteps"],
                            bc_train_kwargs=configs["dagger_trainer"]["bc_train_kwargs"],
                        ) # 6600 for 5 mins
    if round_id % 5 == 0:
        dagger_trainer.save_trainer()
        #  The created snapshot can be reloaded with `reconstruct_trainer()`.

        # with open(f"checkpoints/dagger_trainer-checkpoint{round_id:05d}.pkl", "wb") as file:
        #     pickle.dump(dagger_trainer, file)
    win_rate, score_per_round = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=50, verbose=False)
    print(f"Round {round_id} Win rate: {win_rate:.2f}, Score per round: {score_per_round:.2f}")
    win_rates.append(win_rate)
    score_per_rounds.append(score_per_round)
    custom_logger.record("a/win_rate", win_rate)
    custom_logger.record("a/score_per_round", score_per_round)
    custom_logger.dump(step=round_id)

    round_id += 1

rew_after_training, _ = evaluate_policy(dagger_trainer.policy, env, 100)
print(f"Mean reward before training: {np.mean(rew_before_training):.2f}, after training: {np.mean(rew_after_training):.2f}")
learner_eval_after = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=50, verbose=True)
print(f"Win rate after training: {learner_eval_after[0]:.2f}, score per round after training: {learner_eval_after[1]:.2f}")
print(f"Total time elapsed: {(time.time() - start_time)/60:.2f} min")

############# Finish training #############
import matplotlib.pyplot as plt

# Plot win rates and score per rounds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(range(len(win_rates)), win_rates)
ax1.set_xlabel('training rounds')
ax1.set_ylabel('Win Rate per round')
ax1.set_title('Win Rate vs training rounds')

ax2.plot(range(0,len(win_rates)*time_steps_per_round,time_steps_per_round), score_per_rounds)
ax2.set_xlabel('time steps')
ax2.set_ylabel('Score per Round')
ax2.set_title('Score vs time steps')

plt.tight_layout()
plt.show()
plt.savefig('logs/dagger_train.png')

