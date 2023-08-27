import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm
import settings as s

option={"argv": ["play","--no-gui","--agents","user_agent",\
                                            # "coin_collector_agent", \
                                            "--scenario","loot-crate-5"]}
model_path = "./Original/agent_code/PPO_agent/ppo_bomberman"

env = CustomEnv(options = option)
model = PPO("MultiInputPolicy", env, verbose=1, learning_rate = 0.001, n_steps = 16, batch_size = 4, stats_window_size = 400)
# model = PPO.load(model_path, env)
        
for turn in tqdm(range(20000)):
    if turn % 100 == 0 and turn != 0: # reload environment for every 100 turns
        del env
        env = CustomEnv(options = option)
        
        model.save(model_path)
        del model
        model = PPO.load(model_path, env)

    model.learn(total_timesteps=400)
    if turn % 5 == 0:
        model.save(model_path)

model.save(model_path)