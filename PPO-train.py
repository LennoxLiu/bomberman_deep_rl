import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

# Parallel environments
# vec_env = make_vec_env("CartPole-v1", n_envs=1, seed=42)
vec_env = CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent", "--scenario", "coin-heaven"]})

model = PPO("MultiInputPolicy", vec_env, verbose=1)
# model = PPO.load("./PPO/ppo_cartpole", vec_env)
# model.learn(total_timesteps=400)
for turn in tqdm(range(250)):
    model.learn(total_timesteps=400)
    if turn % 10 == 0:
        model.save("./PPO/ppo_cartpole")
model.save("./PPO/ppo_cartpole")

# del model # remove to demonstrate saving and loading
# model = PPO.load("./PPO/ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render()