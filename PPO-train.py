import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=1, seed=42)
vec_env = CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent","coin_collector_agent"]})

# model = PPO("MultiInputPolicy", vec_env, verbose=1)
model = PPO.load("./PPO/ppo_bomberman", vec_env)
# model.learn(total_timesteps=400)
for turn in tqdm(range(100)):
    model.learn(total_timesteps=400)
    if turn % 10 == 0:
        model.save("./PPO/ppo_bomberman")
model.save("./PPO/ppo_bomberman")

# del model # remove to demonstrate saving and loading
# model = PPO.load("./PPO/_bomberman")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()