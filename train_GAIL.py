import logging
import os
import pathlib
import imitation
from imitation.data import types
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo import MlpPolicy

from imitation.algorithms.adversarial.gail import GAIL
from imitation.data import rollout
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.serialize import load_policy
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
import torch
import wandb
from CustomEnv import ACTION_MAP, CustomEnv
from gymnasium.spaces import MultiDiscrete
import settings as s
from gymnasium import spaces
from imitation.util import util
from imitation.util.logger import WandbOutputFormat, HierarchicalLogger
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save
from test_win_rate import test_against_RuleBasedAgent

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

custom_logger = imit_logger.configure(
        folder='logs',
        format_strs=["tensorboard", "stdout"],
    )
os.makedirs('checkpoints', exist_ok=True)
def callback(round_num: int, /) -> None:
    # if checkpoint_interval > 0 and  round_num % checkpoint_interval == 0:
    save(gail_trainer, pathlib.Path(f"checkpoints/checkpoint{round_num:05d}"))
# env = make_vec_env(
#     "seals:seals/CartPole-v0",
#     rng=np.random.default_rng(SEED),
#     n_envs=8,
#     post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
# )
# expert = load_policy(
#     "ppo-huggingface",
#     organization="HumanCompatibleAI",
#     env_name="seals-CartPole-v0",
#     venv=env,
# )
# rollouts = rollout.rollout(
#     expert,
#     env,
#     rollout.make_sample_until(min_timesteps=None, min_episodes=60),
#     rng=np.random.default_rng(SEED),
# )

# print(type(rollouts))
# print(type(rollouts[0]))
# exit(0)

################## test format of rollouts ##################

env = make_vec_env(
    'CustomEnv-v1',
    rng=np.random.default_rng(SEED),
    n_envs=8,
    post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
)

rollouts = np.load("rule_based_traj/rule_based_traj_combined.npy", allow_pickle=True).tolist()
configs = {'learner': {'policy': "MlpPolicy", 'batch_size': 64, 'ent_coef': 0.0, 'learning_rate': 0.0004, 'gamma': 0.95, 'n_epochs': 5, 'seed': SEED}, \
           'reward_net': {'normalize_input_layer': "RunningNorm"}, \
            'gail_trainer': {'demo_batch_size': 1024, 'gen_replay_buffer_capacity': 512, 'n_disc_updates_per_round': 8} }

os.makedirs('logs/learner', exist_ok=True)
learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=configs['learner']['batch_size'],
    ent_coef=configs['learner']['ent_coef'],
    learning_rate=configs['learner']['learning_rate'],
    gamma=configs['learner']['gamma'],
    n_epochs=configs['learner']['n_epochs'],
    seed=SEED,
    tensorboard_log='./logs/learner/'
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
os.makedirs('logs/GAIL', exist_ok=True)

gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=configs['gail_trainer']['demo_batch_size'],
    gen_replay_buffer_capacity=configs['gail_trainer']['gen_replay_buffer_capacity'],
    n_disc_updates_per_round=configs['gail_trainer']['n_disc_updates_per_round'],
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
    log_dir='./logs/GAIL/',
    init_tensorboard=True,
    init_tensorboard_graph=True,
    custom_logger=custom_logger,
)

# evaluate the learner before training
learner_eval_before = test_against_RuleBasedAgent(0, learner, rounds=20, verbose=False)

# train the learner and evaluate again
gail_trainer.train(20000*5, callback)  # Train for 800_000 steps to match expert.


learner_eval_after = test_against_RuleBasedAgent(0, learner, rounds=20, verbose=False)


print(f"Win rate before training: {learner_eval_before[0]:.2f}, score per round before training: {learner_eval_before[1]:.2f}")
print(f"Win rate after training: {learner_eval_after[0]:.2f}, score per round after training: {learner_eval_after[1]:.2f}")

os.makedirs('models', exist_ok=True)
save(learner, pathlib.Path(f'models/learner_GAIL'))
