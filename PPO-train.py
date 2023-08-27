import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from CustomEnv import CustomEnv
from tqdm import tqdm

env = None
model = None
for turn in tqdm(range(20000)):
    if turn % 100 == 0: # reload everything for every 100 turns
        del env
        env = CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent",\
                                            "coin_collector_agent", \
                                            "--scenario","loot-crate-3"]})
        
        if model != None:
            model.save("./Original/agent_code/PPO_agent/ppo_bomberman")
        del model
        model = PPO("MultiInputPolicy", env, verbose=1)
        # model = PPO.load("./Original/agent_code/PPO_agent/ppo_bomberman", env)
    
    model.learn(total_timesteps=400)
    if turn % 5 == 0 and turn % 100 != 0:
        model.save("./Original/agent_code/PPO_agent/ppo_bomberman")

model.save("./Original/agent_code/PPO_agent/ppo_bomberman")

# del model # remove to demonstrate saving and loading
# model = PPO.load("./agent_code/PPO_agent/_bomberman")

# obs, _ = vec_env.reset()
# for _ in range(400):
#     action, _states = model.predict(obs)
#     obs, rewards, terminated, truncated, info = vec_env.step(action)