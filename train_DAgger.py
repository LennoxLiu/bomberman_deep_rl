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

my_device = device("cuda" if is_available() else "cpu")
print("Using device:", my_device)

remove_logs_checkpoints = input("Do you want to remove existing 'logs' and 'checkpoints' folders? (y/n): ")
if remove_logs_checkpoints.lower() == 'y':
    # Remove existing 'logs' and 'checkpoints' folders
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')

custom_logger = imit_logger.configure(
        folder='logs',
        format_strs=["tensorboard","stdout"],
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

    def __init__(self, observation_space: spaces.MultiDiscrete, features_dim = [128, 64]):
        super().__init__(observation_space, features_dim=sum(features_dim))
        # We assume 2x1xROWxCOL image (1 channel), but input as (HxWx2)
        n_input_channels = 1
        self.cnn1 = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=4, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=2, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten1 = self.cnn1(
                th.as_tensor(observation_space.sample().reshape(2, 1, s.ROWS, s.COLS)).float()
            ).shape[1]
            n_flatten2 = self.cnn2(
                th.as_tensor(observation_space.sample().reshape(2, 1, s.ROWS, s.COLS)).float()
            ).shape[1]

        self.linear1 = nn.Sequential(nn.Linear(n_flatten1, features_dim[0]), nn.ReLU())
        self.linear2 = nn.Sequential(nn.Linear(n_flatten2, features_dim[1]), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        print("observations.shape:", observations.shape)
        obs1, obs2 = observations[:,:s.ROWS*s.COLS], observations[:, s.ROWS*s.COLS:]
        print("obs1.shape:", obs1.shape)
        print("obs2.shape:", obs2.shape)

        obs1 = obs1.reshape(-1, 1, s.ROWS, s.COLS)
        obs2 = obs2.reshape(-1, 1, s.ROWS, s.COLS)
        print("obs1.shape:", obs1.shape)
        print("obs2.shape:", obs2.shape)
        return th.cat([self.linear1(self.cnn1(obs1)), self.linear2(self.cnn2(obs2))], dim=1)

configs = dict(
    batch_size=1024, #32
    minibatch_size=256,
    learning_rate=0.0003,
    net_arch=[64, 64],
    features_extractor_class="CustomCNN",
    features_extractor_kwargs=dict(
        features_dim=[64,32]),
)

bc_trainer = bc.BC(
    batch_size=configs['batch_size'],
    minibatch_size=configs['minibatch_size'],
    policy=ActorCriticPolicy(
        env.observation_space,
        env.action_space,
        linear_schedule(configs["learning_rate"]),
        net_arch=configs["net_arch"],
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=configs["features_extractor_kwargs"],
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
with tempfile.TemporaryDirectory(prefix="dagger_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )

    rew_before_training, _ = evaluate_policy(dagger_trainer.policy, env, 100)
    print(f"Mean reward before training:{np.mean(rew_before_training):.2f}")
    
    time_steps_per_round = 6600 # 6600 for 5 mins
    for round_id in tqdm(range(8)):
        dagger_trainer.train(time_steps_per_round) # 6600 for 5 mins
        with open(f"checkpoints/dagger_trainer-checkpoint{round_id:05d}.pkl", "wb") as file:
            pickle.dump(dagger_trainer, file)
        win_rate, score_per_round = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=10, verbose=False)
        print(f"Round {round_id} Win rate: {win_rate:.2f}, Score per round: {score_per_round:.2f}")
        win_rates.append(win_rate)
        score_per_rounds.append(score_per_round)

rew_after_training, _ = evaluate_policy(dagger_trainer.policy, env, 100)
print(f"Mean reward before training: {np.mean(rew_before_training):.2f}, after training: {np.mean(rew_after_training):.2f}")
learner_eval_after = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=20, verbose=True)
print(f"Win rate after training: {learner_eval_after[0]:.2f}, score per round after training: {learner_eval_after[1]:.2f}")
print(f"Total time elapsed: {(time.time() - start_time)/60:.2f} min")

import matplotlib.pyplot as plt

# Plot win rates and score per rounds
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(range(len(win_rates)), win_rates)
ax1.set_xlabel('Round')
ax1.set_ylabel('Win Rate')
ax1.set_title('Win Rate per Round')

ax2.plot(range(0,len(win_rates)*time_steps_per_round,time_steps_per_round), score_per_rounds)
ax2.set_xlabel('time steps')
ax2.set_ylabel('Score per Round')
ax2.set_title('Score vs steps')

plt.tight_layout()
plt.savefig('/logs/dagger_train.png')
plt.show()
