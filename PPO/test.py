import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

# Parallel environments
vec_env = make_vec_env("CartPole-v1", n_envs=4, seed=42)

model = PPO("MlpPolicy", vec_env, verbose=1)
model.learn(total_timesteps=250000)
model.save("./PPO/ppo_cartpole")

del model # remove to demonstrate saving and loading

model = PPO.load("./PPO/ppo_cartpole")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")