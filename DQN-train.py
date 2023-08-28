import gymnasium as gym

from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            # "coin_collector_agent", \
                                            "--scenario","coin-heaven"]}
model_path = "./Original/agent_code/DQN_agent/dqn_bomberman"

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

env = CustomEnv(options = option)
# model = DQN("MlpPolicy", env, verbose=1,learning_starts=0, learning_rate = 0.0005,target_update_interval = 1000, stats_window_size= 400)
# (policy: str | type[DQNPolicy], env: GymEnv | str,
#  learning_rate: float | Schedule = 0.0001,
#  buffer_size: int = 1000000, learning_starts: int = 50000,
#  batch_size: int = 32, tau: float = 1, gamma: float = 0.99,
#  train_freq: int | Tuple[int, str] = 4, gradient_steps: int = 1,
#  replay_buffer_class: type[ReplayBuffer] | None = None,
#  replay_buffer_kwargs: Dict[str, Any] | None = None,
#  optimize_memory_usage: bool = False, target_update_interval: int = 10000,
#  exploration_fraction: float = 0.1, exploration_initial_eps: float = 1,
#  exploration_final_eps: float = 0.05, max_grad_norm: float = 10,
#  stats_window_size: int = 100, tensorboard_log: str | None = None,
#  policy_kwargs: Dict[str, Any] | None = None, verbose: int = 0, seed: int | None = None,
#  device: device | str = "auto", _init_setup_model: bool = True) -> None

new_parameters = {
    "learning_rate": linear_schedule(0.0005),
    "target_update_interval": 1000, # more n_steps means more robust, less tuned
    "batch_size": 64,
    "exploration_fraction": 0.1,
    "exploration_initial_eps": 1,
    "exploration_final_eps":0.05,
    "stats_window_size": 400
    }
model = DQN.load(model_path, env = env, force_reset = True, custom_objects = new_parameters) 
while True:
    model.learn( total_timesteps=61440, progress_bar=True, log_interval = 100)
    # total_timesteps=61440
    model.save(model_path)
