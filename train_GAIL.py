import os
import pathlib
import shutil
import numpy as np
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
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save
from test_win_rate import test_against_RuleBasedAgent
import torch as th

remove_logs_checkpoints = input("Do you want to remove existing 'logs' and 'checkpoints' folders? (y/n): ")
if remove_logs_checkpoints.lower() == 'y':
    # Remove existing 'logs' and 'checkpoints' folders
    if os.path.exists('logs'):
        shutil.rmtree('logs')
    if os.path.exists('checkpoints'):
        shutil.rmtree('checkpoints')


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
configs = {'learner': {
                'policy': "MlpPolicy", 
                'batch_size': 64, 
                'ent_coef': 0.0, 
                'learning_rate': 0.0004, 
                'gamma': 0.95, 
                'n_epochs': 5, 
                'seed': SEED,
                'policy_kwargs': {"activation_fn": "th.nn.ReLU", \
                    "pi":[256,128,64,32], "vf": [256,128,64,32]}, \
            
            }, \
            'reward_net': {'normalize_input_layer': "RunningNorm"}, \
            'gail_trainer': {
                'demo_batch_size': 1024,
                'gen_replay_buffer_capacity': 512, 
                'n_disc_updates_per_round': 8,

} }

os.makedirs('logs/learner', exist_ok=True)
if remove_logs_checkpoints.lower() == 'y':
    use_checkpoint = input("Train from previous checkpoint? (y/n): ")
if use_checkpoint.lower() == 'y':
    learner = load_policy(
        "ppo",
        venv=env,
        path="checkpoints/checkpoint00005/gen_policy/model"
    )
else:
    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=configs['learner']['batch_size'],
        ent_coef=configs['learner']['ent_coef'],
        learning_rate=configs['learner']['learning_rate'],
        gamma=configs['learner']['gamma'],
        n_epochs=configs['learner']['n_epochs'],
        seed=SEED,
        tensorboard_log='./logs/learner/',
        policy_kwargs= dict(activation_fn=th.nn.ReLU, net_arch=dict(\
            pi=configs['learner']['policy_kwargs']['pi'], \
            vf=configs['learner']['policy_kwargs']['vf'])),
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
env.seed(SEED)
learner_rewards_before_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
# train the learner and evaluate again
gail_trainer.train(20000*6, callback)  # Train for 800_000 steps to match expert.

env.seed(SEED)
learner_rewards_after_training, _ = evaluate_policy(
    learner, env, 100, return_episode_rewards=True,
)
learner_eval_after = test_against_RuleBasedAgent(0, learner, rounds=20, verbose=False)

print("mean reward after training:", np.mean(learner_rewards_after_training))
print("mean reward before training:", np.mean(learner_rewards_before_training))
print(f"Win rate after training: {learner_eval_after[0]:.2f}, score per round after training: {learner_eval_after[1]:.2f}")

os.makedirs('models', exist_ok=True)
