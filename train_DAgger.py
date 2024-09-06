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
from imitation.algorithms.dagger import BetaSchedule
from imitation.util import util
from imitation.algorithms.bc import BCLogger
from stable_baselines3.common.monitor import Monitor

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
custom_logger = imit_logger.configure(folder='logs/tensorboard_logs',format_strs=["tensorboard"],)

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
env_test = make_vec_env(
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
        # We assume 2x1xROWxCOL image (1 channel), but input as (HxWx2)
        n_input_channels = 1

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
    

    def forward(self, observations: th.Tensor) -> th.Tensor:
        obs1, obs2 = observations[:,0], observations[:, 1]
        
        # Reshape and standardize the input to [0,1]
        obs1 = obs1.reshape(-1, 1, s.ROWS, s.COLS) / 8
        obs2 = obs2.reshape(-1, 1, s.ROWS, s.COLS) / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

        return self.dense(th.cat([self.cnn1(obs1), self.cnn2(obs2)], dim=1))

configs = {
    "bc_trainer": {
        "batch_size": 256, # The number of samples in each batch of expert data.
        "minibatch_size": 256, # if GPU memory is not enough, reduce this number to a factor of batch_size
        "l2_weight": 0, # 1e-7, default: 0
        "policy":{
            "learning_rate": 0.0003, # default 3e-4
            "net_arch": [512, 512, 256, 256, 128, 128, 128, 128, 64, 64, 32, 32],
            "features_extractor_class": "CustomCNN",
            "activation_fn": "nn.ReLU", # nn.ReLU nn.LeakyReLU(slope), default: "th.nn.Tanh"
            "features_extractor_kwargs": {
                "network_configs": {"cnn1":[32,64,128,256],"cnn1_strides":[1,1,2,2], "cnn2":[32,64,128],"cnn2_strides":[1,1,2], "dense": [512, 512]}
        }}
    },
    "dagger_trainer": {
        "rollout_round_min_episodes": 3, # The number of episodes the must be completed completed before a dataset aggregation step ends.
        "rollout_round_min_timesteps": 1024, #The number of environment timesteps that must be completed before a dataset aggregation step ends. Also, that any round will always train for at least self.batch_size timesteps, because otherwise BC could fail to receive any batches.
        "bc_train_kwargs": {
            "n_epochs": 8, # default: 4
        },
        "beta0": 0.75, # The initial value of beta. The probability of using the expert policy instead of the learner policy.
        # "delta_beta": 0.05, # The amount that beta decreases by each round.
        "beta_final": 0.1, # The final value of beta. The probability of using the expert policy instead of the learner policy.
        "decrease_beta": 0.05, # The amount that beta decreases by each round.
        "increase_beta": 0.05, # The amount that beta increases by each round.
    },
    "SEED":42
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
        activation_fn=nn.ReLU,
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
os.makedirs('logs/tensorboard_logs/beta', exist_ok=True)
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir='checkpoints',
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=rng,
    custom_logger=custom_logger,
    beta_schedule=CustomBetaSchedule2(custom_logger, decrease_beta=configs["dagger_trainer"]["decrease_beta"], 
                                      increase_beta=configs["dagger_trainer"]["increase_beta"],
                                      beta0=configs["dagger_trainer"]["beta0"], beta_final=configs["dagger_trainer"]["beta_final"]),
)



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

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
def load_tensorboard_log(tag,log_dir='logs/tesnsorboard_logs'):
    # Load the event accumulator
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()  # Load the data

    # Access scalar metrics like 'loss' or 'accuracy'
    if tag in event_acc.Tags()['scalars']:
        loss_events = event_acc.Scalars('loss')  # Get all events for 'loss'

    return loss_events


if not remove_logs_checkpoints and os.path.exists("checkpoints/checkpoint-latest.pt"):
    dagger_trainer, current_beta, configs = load_DAgger_trainer("checkpoints/checkpoint-latest.pt")

############# Start training #############
learner_reward, _ = evaluate_policy(dagger_trainer.policy, env_test , n_eval_episodes=10)
print(f"Round 0 Learner reward: {learner_reward:.2f}")

round_id=1
custom_logger.record("a/win_rate", 0)
custom_logger.record("a/score_per_round", 0)
custom_logger.record("a/learner_reward", np.mean(learner_reward))
custom_logger.dump(step=0)
while True:
    dagger_trainer.train(total_timesteps = time_steps_per_round,
                            rollout_round_min_episodes=configs["dagger_trainer"]["rollout_round_min_episodes"],
                            rollout_round_min_timesteps=configs["dagger_trainer"]["rollout_round_min_timesteps"],
                            bc_train_kwargs=configs["dagger_trainer"]["bc_train_kwargs"],
                        ) # 6600 for 5 mins

    try:
        # dagger_trainer.save_trainer()
        #  The created snapshot can be reloaded with `reconstruct_trainer()`.
        save_DAgger_trainer(dagger_trainer,configs)
    except Exception as e:
        print("Error saving trainer:", e)
        continue

    # with open(f"checkpoints/dagger_trainer-checkpoint{round_id:05d}.pkl", "wb") as file:
    #     pickle.dump(dagger_trainer_copy, file)
    with open(f"checkpoints/policy-checkpoint{round_id:05d}.pkl", "wb") as file:
        pickle.dump(dagger_trainer.policy, file)

    win_rate, score_per_round = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=50, verbose=False)
    print(f"Round {round_id} Win rate: {win_rate:.2f}, Score per round: {score_per_round:.2f}")
    win_rates.append(win_rate)
    score_per_rounds.append(score_per_round)
    custom_logger.record("a/win_rate", win_rate)
    custom_logger.record("a/score_per_round", score_per_round)
    custom_logger.dump(step=round_id)

    learner_reward, _ = evaluate_policy(dagger_trainer.policy, env_test , n_eval_episodes=10)
    custom_logger.record("a/learner_reward", np.mean(learner_reward))
    custom_logger.dump(step=round_id)
    print(f"Round {round_id} Learner reward: {learner_reward:.2f}")
    
    mean_reward_list = load_tensorboard_log("dagger/mean_episode_reward")
    if mean_reward_list[-1] > np.mean(mean_reward_list[:-1]):
        dagger_trainer.beta_schedule.decrease()
    else:
        dagger_trainer.beta_schedule.increase()

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

