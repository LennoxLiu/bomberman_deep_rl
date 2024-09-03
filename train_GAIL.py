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
from CustomEnv import ACTION_MAP, CustomEnv
from gymnasium.spaces import MultiDiscrete
import settings as s
from gymnasium import spaces
from imitation.util import util

SEED = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

learner = PPO(
    env=env,
    policy=MlpPolicy,
    batch_size=64,
    ent_coef=0.0,
    learning_rate=0.0004,
    gamma=0.95,
    n_epochs=5,
    seed=SEED,
)
reward_net = BasicRewardNet(
    observation_space=env.observation_space,
    action_space=env.action_space,
    normalize_input_layer=RunningNorm,
)
gail_trainer = GAIL(
    demonstrations=rollouts,
    demo_batch_size=512, #1024
    gen_replay_buffer_capacity=512,
    n_disc_updates_per_round=8,
    venv=env,
    gen_algo=learner,
    reward_net=reward_net,
    allow_variable_horizon=True,
)

reward_net.to(device)

# evaluate the learner before training
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

# train the learner and evaluate again
gail_trainer.train(50000)  # Train for 800_000 steps to match expert.

env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))