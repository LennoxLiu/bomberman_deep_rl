import pickle
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


def test_against_RuleBasedAgent(turn_id, agent, rounds=10, rule_based_agent = False, verbose=False):
    user_agent = agent
    # don't end the game early
    env = CustomEnv.CustomEnv(options = {"argv": ["play","--no-gui","--my-agent","user_agent","--train","1","--continue-without-training"]})
    start_time = time.time()
    win_count = 0
    total_score = 0
    for i in range(rounds): # roundss trajectories per turn
        observation, game_state = env.reset() # seed = None will generate a random seed
        if rule_based_agent:
            user_agent.reset() # for RuleBasedAgent

        terminated = False
        truncated = False
        temp_actions = []
        temp_rewards = []
        while not terminated and not truncated:
            if rule_based_agent:
                action = user_agent.act(observation) # for RuleBasedAgent
                action = CustomEnv.ACTION_MAP.index(action) # for imitation agent
                temp_actions.append(action)
            else:
              action = user_agent.predict(observation, deterministic=True)[0] # for imitation agent
            # if verbose:
            #     print(f'Action: {action}')
            
            observation, reward, terminated, truncated, game_state = env.step(action)
            # print("game_state in test:", game_state)
            temp_rewards.append(reward)

        agent_win, other_scores, user_agent_score, agent_events = env.close()
        
        total_score += user_agent_score

        if agent_win:
            win_count += 1
            if verbose:
                print(f'score: {user_agent_score}, others scores: {other_scores} ')
                print("agent events:",agent_events)
                print("rewards:",temp_rewards[-5:])

        if verbose:
            print(f'Round {i} done. Time elapsed: {time.time() - start_time:.0f}s Win rate: {win_count / rounds:.2f}, Score this round: {user_agent_score:.2f}')
            # print("len(actions):",len(temp_actions))
            

    return win_count / rounds, total_score / rounds


if __name__ == '__main__':
    env = gym.make('CustomEnv-v1')

    agent=pickle.load(open('checkpoints/policy-checkpoint00054.pkl','rb'))

    print("Win rate:", test_against_RuleBasedAgent(0,agent,100, rule_based_agent=False,verbose=True))
    
    # print("Win rate:", test_against_RuleBasedAgent(0,RuleBasedAgent(has_memory=False),1, rule_based_agent=True,verbose=True))
    exit(0)
########################### parallel test_against_RuleBasedAgent ###########################
    turns = 10
    num_processes = 14
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = []
        for i in range(turns):
            result = pool.apply_async(test_against_RuleBasedAgent, args=(i,RuleBasedAgent(has_memory=False),20,True,True))
            results.append(result)

        # while len(results) > 0 and any([not result.ready() for result in results]):
        #     completed = [result.ready() for result in results]
        #     percent_complete = sum(completed) / turns * 100
            # if percent_complete > 0:
            #     print(f"Progress: {percent_complete:.2f}% , Estimated time remaining: {((time.time() - start_time) / percent_complete) * (100 - percent_complete) /60 :.2f} mins")
            
            # time.sleep(10)
        
        results_list = [result.get() for result in results]

    print('Win rates, score per round:', np.mean(results_list, axis=0))

