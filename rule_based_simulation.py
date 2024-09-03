import time
import numpy as np
from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym
from imitation.data import types
import os


os.makedirs('rule_based_traj', exist_ok=True)

rule_based_agent = RuleBasedAgent()
env = gym.make('CustomEnv-v1')

rounds = 1E+4
traj_list = []
rng = np.random.default_rng(42)
start_time = time.time()
while len(traj_list) < rounds:
    observation, game_state = env.reset(seed=None) # When seed = None, it will randomly generate new seeds in reset()
    temp_obs = [observation]
    rule_based_agent.reset()
    temp_actions = []
    temp_rewards = []

    terminated = False
    truncated = False
    while not terminated and not truncated:
        action = rule_based_agent.act(game_state)
        action = CustomEnv.ACTION_MAP.index(action) # convert action from string to int
        
        observation, reward, terminated, truncated, game_state = env.step(action)
        if not terminated and not truncated:
            temp_actions.append(action)
            temp_obs.append(observation)
            temp_rewards.append(reward)

    agent_win, other_scores, user_agent_score =env.close()
    
    if agent_win:
        traj_temp = types.TrajectoryWithRew(obs=np.array(temp_obs,dtype=np.uint8), acts=np.array(temp_actions,dtype=np.uint8), rews=np.array(temp_rewards,dtype=np.float32), infos=None, terminal=True)
        # print(isinstance(traj_temp, types.TrajectoryWithRew))

        traj_list.append(traj_temp)
        
        # print("user score:",user_agent_score, "other scores:",other_scores)
        # print("rewards:",temp_rewards)
        # print("observation[-1]:",np.array(temp_obs[-1],dtype=np.float32))
    # else:
    #     print("agent lose")
        # print("user score:",user_agent_score, "other scores:",other_scores)

        if len(traj_list) % 10 == 0:
            np.save(f'rule_based_traj/traj_list_{len(traj_list)}.npy', traj_list)
            print("current trajs:", len(traj_list))
            print("Time elapsed: %.2f s" % (time.time() - start_time) )
            print("Estimated time remaining: %.2f min" % ((time.time() - start_time) / len(traj_list) * (rounds - len(traj_list)) /60) )
    
print("total trajs:",len(traj_list))

# # Load and continue the list
# loaded_traj_list = []
# for i in range(1, len(traj_list) // 100 + 1):
#     file_name = f'traj_list_{i * 100}.npy'
#     loaded_traj = np.load(file_name, allow_pickle=True)
#     loaded_traj_list.extend(loaded_traj)

# traj_list.extend(loaded_traj_list)