import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=1, seed=42)
vec_env = CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent",\
                                        "coin_collector_agent","coin_collector_agent",\
                                            "--scenario","classic"]})

# model = PPO("MlpPolicy", vec_env, verbose=1, use_sde = True)
model = PPO.load("./Original/agent_code/PPO_agent/ppo_bomberman", vec_env)
# model.learn(total_timesteps=400)
for turn in tqdm(range(20000)):
    model.learn(total_timesteps=400)
    if turn % 10 == 0:
        model.save("./Original/agent_code/PPO_agent/ppo_bomberman")
model.save("./Original/agent_code/PPO_agent/ppo_bomberman")

# del model # remove to demonstrate saving and loading
# model = PPO.load("./agent_code/PPO_agent/_bomberman")

# obs, _ = vec_env.reset()
# for _ in range(400):
#     action, _states = model.predict(obs)
#     obs, rewards, terminated, truncated, info = vec_env.step(action)