import time
import numpy as np
from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym
from imitation.data import types
import os
from multiprocessing import Pool

os.makedirs('rule_based_traj', exist_ok=True)

def simulate_trajectory(turn_id, rounds=10):
    rule_based_agent = RuleBasedAgent()
    env = gym.make('CustomEnv-v1')
    traj_list = []
    start_time = time.time()
    while len(traj_list) < rounds: # roundss trajectories per turn
        observation, game_state = env.reset() # seed = None will generate a random seed
        temp_obs = [observation]
        rule_based_agent.reset()
        temp_actions = []
        temp_rewards = []

        terminated = False
        truncated = False
        while not terminated and not truncated:
            action = rule_based_agent.act(game_state)
            action = CustomEnv.ACTION_MAP.index(action)
            
            observation, reward, terminated, truncated, game_state = env.step(action)
            if not terminated and not truncated:
                temp_actions.append(action)
                temp_obs.append(observation)
                temp_rewards.append(reward)

        agent_win, other_scores, user_agent_score = env.close()
        
        if agent_win:
            traj_temp = types.TrajectoryWithRew(obs=np.array(temp_obs, dtype=np.uint8), acts=np.array(temp_actions, dtype=np.uint8), rews=np.array(temp_rewards, dtype=np.float32), infos=None, terminal=True)
            traj_list.append(traj_temp)
    
    np.save(f'rule_based_traj/traj_list_{turn_id}.npy', traj_list, allow_pickle=True)
    print('Turn %i done. Time elapsed: %.2f' % (turn_id, time.time() - start_time))

turns = 100
num_processes = 19

if __name__ == '__main__':
    start_time = time.time()
    with Pool(num_processes) as pool:
        pool.map(simulate_trajectory, range(turns))

