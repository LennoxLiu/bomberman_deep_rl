import os
import pathlib
import pickle
import shutil
import time
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.util.util import make_vec_env
from torch import device
from torch.cuda import is_available
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save
from imitation.data.wrappers import RolloutInfoWrapper
from RuleBasedPolicy import RuleBasedPolicy
from test_win_rate import test_against_RuleBasedAgent
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
import train_utils as tu

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

configs = {
    "bc_trainer": {
        "batch_size": 256, # The number of samples in each batch of expert data.
        "minibatch_size": 256, # if GPU memory is not enough, reduce this number to a factor of batch_size
        "l2_weight": 1e-8, # 1e-7, default: 0
        "policy":{
            "learning_rate": 0.0003, # default 3e-4
            "net_arch": dict(pi=[128,64,32], vf=[128, 128, 64, 32]), 
            "features_extractor_class": "CustomCNN",
            "activation_fn": "nn.ReLU", # nn.ReLU nn.LeakyReLU(slope), default: "th.nn.Tanh"
            "features_extractor_kwargs": {
                "network_configs": {"cnn1":[32,64,128],"cnn1_strides":[1,1,2], "cnn2":[32,64,128],"cnn2_strides":[1,1,2], "dense": [256]}
        }}
    },
    "dagger_trainer": {
        "rollout_round_min_episodes": 3, # The number of episodes the must be completed completed before a dataset aggregation step ends.
        "rollout_round_min_timesteps": 1024, #The number of environment timesteps that must be completed before a dataset aggregation step ends. Also, that any round will always train for at least self.batch_size timesteps, because otherwise BC could fail to receive any batches.
        "bc_train_kwargs": {
            "n_epochs": 8, # default: 4
        },
        "beta0": 0.9, # The initial value of beta. The probability of using the expert policy instead of the learner policy.
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
        tu.linear_schedule(configs['bc_trainer']['policy']["learning_rate"]),
        net_arch=configs['bc_trainer']['policy']["net_arch"],
        activation_fn=nn.ReLU,
        features_extractor_class=tu.CustomCNN,
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
    beta_schedule=tu.CustomBetaSchedule2(custom_logger, decrease_beta=configs["dagger_trainer"]["decrease_beta"], 
                                      increase_beta=configs["dagger_trainer"]["increase_beta"],
                                      beta0=configs["dagger_trainer"]["beta0"], beta_final=configs["dagger_trainer"]["beta_final"]),
)


if not remove_logs_checkpoints and os.path.exists("checkpoints/checkpoint-latest.pt"):
    dagger_trainer, current_beta, configs = tu.load_DAgger_trainer("checkpoints/checkpoint-latest.pt")

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
        tu.save_DAgger_trainer(dagger_trainer,configs)
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
    
    mean_reward_list = tu.load_tensorboard_log("dagger/mean_episode_reward")
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

