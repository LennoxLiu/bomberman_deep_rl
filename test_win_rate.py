import time
import numpy as np
from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym
from imitation.data import types
import os
from multiprocessing import Pool
from tqdm import tqdm
from imitation.policies.serialize import load_policy
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper


def test_against_RuleBasedAgent(turn_id, agent, rounds=10, verbose=False):
    user_agent = agent
    env = gym.make('CustomEnv-v1')
    start_time = time.time()
    win_count = 0
    total_score = 0
    for i in range(rounds): # roundss trajectories per turn
        observation, game_state = env.reset() # seed = None will generate a random seed
        # user_agent.reset()

        terminated = False
        truncated = False
        while not terminated and not truncated:
            # action = user_agent.act(game_state)
            # action = CustomEnv.ACTION_MAP.index(action)

            action = user_agent.predict(observation, deterministic=True)[0]
            
            observation, reward, terminated, truncated, game_state = env.step(action)

        agent_win, other_scores, user_agent_score = env.close()
        
        total_score += user_agent_score

        if agent_win:
            win_count += 1
        
        if verbose:
            print(f'Round {i} done. Time elapsed: {time.time() - start_time:.0f}s Win rate: {win_count / rounds:.2f}')
            print(f'score: {user_agent_score}, others scores: {other_scores} ')

    return win_count / rounds, total_score / rounds


if __name__ == '__main__':
    env = gym.make('CustomEnv-v1')

    agent=load_policy(
        "ppo",
        venv=env,
        path="checkpoints/checkpoint00100/gen_policy/model.zip"
    )

    print("Win rate:", test_against_RuleBasedAgent(0,agent,20,verbose=True))
    exit(0)
########################### parallel test_against_RuleBasedAgent ###########################
    turns = 10
    num_processes = 14
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = []
        for i in range(turns):
            result = pool.apply_async(test_against_RuleBasedAgent, args=(i,agent))
            results.append(result)

        # while len(results) > 0 and any([not result.ready() for result in results]):
        #     completed = [result.ready() for result in results]
        #     percent_complete = sum(completed) / turns * 100
            # if percent_complete > 0:
            #     print(f"Progress: {percent_complete:.2f}% , Estimated time remaining: {((time.time() - start_time) / percent_complete) * (100 - percent_complete) /60 :.2f} mins")
            
            # time.sleep(10)
        
        win_rates = [result.get() for result in results]

    
    print('Win rates:', np.mean(win_rates))

