import pickle
import time
import numpy as np
import torch
from DAggerRandomForestAgent import DAggerRandomForestAgent
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


def test_against_agent(turn_id, agent, rounds=10, env_id = 'CustomEnv_random-v0', rule_based_agent = False, verbose=False):
    user_agent = agent
    # don't end the game early
    # env = CustomEnv.CustomEnv(options = {"argv": ["play","--no-gui","--my-agent","user_agent","--train","1","--continue-without-training"]})
    # env = CustomEnv.CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent","coin_collector_agent","coin_collector_agent","coin_collector_agent","--train","1","--continue-without-training"]})
    # env = CustomEnv.CustomEnv(options = {"argv": ["play","--no-gui","--agents","user_agent","random_agent","random_agent","random_agent","--train","1","--continue-without-training"]})
    env = gym.make(env_id)

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
    agent = DAggerRandomForestAgent()
    # agent = torch.load(open('models/DAgger_RandomForest/policy-checkpoint.pkl','rb'))
    print("Win rate:", test_against_agent(0,agent,100,'CustomEnv-v1', rule_based_agent=True,verbose=False))
    
    # env = gym.make('CustomEnv-v1')
    # agent=torch.load(open(f'checkpoints/policy-checkpoint{36:05d}.pkl','rb'))
    
    # reports = []
    # for i in tqdm([35,36]):
    #     agent=torch.load(open(f'checkpoints/policy-checkpoint{i:05d}.pkl','rb'))
    #     win_rate, score_per_round = test_against_agent(i,agent,100,'CustomEnv_random-v0',False,False)
    #     reports.append((i,win_rate,score_per_round))
    #     print(f"checkpoint {i:3d} win rate: {win_rate:.2f}, score per round: {score_per_round:.2f}")

    # reports = np.array(reports)
    # print(np.argsort(reports[:, 2]))
    # print(reports)
    exit(0)
########################### parallel test_against_agent ###########################
    turns = 10
    num_processes = 14
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = []
        for i in range(turns):
            result = pool.apply_async(test_against_agent, args=(i,agent,10,False,True))
            results.append(result)

        # while len(results) > 0 and any([not result.ready() for result in results]):
        #     completed = [result.ready() for result in results]
        #     percent_complete = sum(completed) / turns * 100
            # if percent_complete > 0:
            #     print(f"Progress: {percent_complete:.2f}% , Estimated time remaining: {((time.time() - start_time) / percent_complete) * (100 - percent_complete) /60 :.2f} mins")
            
            # time.sleep(10)
        
        results_list = [result.get() for result in results]

    print('Win rates, score per round:', np.mean(results_list, axis=0))

