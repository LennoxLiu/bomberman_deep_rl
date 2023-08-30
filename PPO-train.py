import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
import torch as th


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float):
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            "coin_collector_agent","peaceful_agent","peaceful_agent", \
                                            "--scenario","loot-crate-6"]}
model_path = "./Original/agent_code/PPO_agent/ppo_bomberman"

env = CustomEnv(options = option)
# model = PPO("MlpPolicy", env, verbose=1, learning_rate = 0.0003, n_steps = 512, batch_size = 64, stats_window_size = 400)
# learning_rate: float | Schedule = 0.0003,
#  n_steps: int = 2048, batch_size: int = 64,
#  n_epochs: int = 10, gamma: float = 0.99,
#  gae_lambda: float = 0.95, clip_range: float | Schedule = 0.2,
#  clip_range_vf: float | Schedule | None = None,
#  normalize_advantage: bool = True, ent_coef: float = 0,
#  vf_coef: float = 0.5, max_grad_norm: float = 0.5,
#  use_sde: bool = False, sde_sample_freq: int = -1,
#  target_kl: float | None = None, stats_window_size: int = 100,
#  tensorboard_log: str | None = None,
#  policy_kwargs: Dict[str, Any] | None = None,
#  verbose: int = 0, seed: int | None = None,
#  device: device | str = "auto", _init_setup_model: bool = True) -> None

new_parameters = {
    "learning_rate": 0.0003,
    "n_steps": 512, # more n_steps means more robust, less tuned
    "batch_size": 64,
    "stats_window_size":  400,
    "clip_range": 0.1,
    }
model = PPO.load(model_path, env = env, force_reset = True, custom_objects = new_parameters) 
while True:
    model.learn(total_timesteps=int(20480/6*5), progress_bar=True, log_interval = 2)
    # total_timesteps=20480
    model.save(model_path)
