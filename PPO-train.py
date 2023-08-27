import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            # "coin_collector_agent", \
                                            "--scenario","loot-crate-4"]}
model_path = "./Original/agent_code/PPO_agent/ppo_bomberman"

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
# model = PPO("MlpPolicy", env, verbose=1, learning_rate = linear_schedule(0.001), n_steps = 2048, batch_size = 64, stats_window_size = 10)
model = PPO.load(model_path, env = env, force_reset = True)
        
while True:
    model.learn(total_timesteps=20480, progress_bar=True)
    model.save(model_path)
