import os
import pathlib
import shutil
import tempfile

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from torch import device
from torch.cuda import is_available
from imitation.util import logger as imit_logger
from imitation.scripts.train_adversarial import save
from imitation.data.wrappers import RolloutInfoWrapper

from RuleBasedPolicy import RuleBasedPolicy
from test_win_rate import test_against_RuleBasedAgent

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
        format_strs=["tensorboard", "stdout"],
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
)

expert = RuleBasedPolicy(env.observation_space, env.action_space)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=rng,
    device=my_device,
)

with tempfile.TemporaryDirectory(prefix="dagger_") as tmpdir:
    print(tmpdir)
    dagger_trainer = SimpleDAggerTrainer(
        venv=env,
        scratch_dir=tmpdir,
        expert_policy=expert,
        bc_trainer=bc_trainer,
        rng=rng,
    )
    dagger_trainer.train(8_000)

reward, _ = evaluate_policy(dagger_trainer.policy, env, 10)
print("Reward:", reward)
learner_eval_after = test_against_RuleBasedAgent(0, dagger_trainer.policy, rounds=20, verbose=False)
print(f"Win rate after training: {learner_eval_after[0]:.2f}, score per round after training: {learner_eval_after[1]:.2f}")
