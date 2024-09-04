# generate data(rollout) from rule based agent for GAIL training
import time
import numpy as np
from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym
from imitation.data import types
import os
from multiprocessing import Pool
from tqdm import tqdm

os.makedirs('rule_based_traj', exist_ok=True)

def simulate_trajectory(turn_id, rounds=100):
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
        
        print("len(actions):",len(temp_actions))
        if agent_win:
            traj_temp = types.TrajectoryWithRew(obs=np.array(temp_obs, dtype=np.uint8), acts=np.array(temp_actions, dtype=np.uint8), rews=np.array(temp_rewards, dtype=np.float32), infos=None, terminal=True)
            traj_list.append(traj_temp)
            # print(temp_rewards)
    
    np.save(f'rule_based_traj/traj_list_{turn_id}.npy', traj_list, allow_pickle=True)
    print('Turn %i done. Time elapsed: %.2f' % (turn_id, time.time() - start_time))
    
    return True

turns = 100
num_processes = 14

if __name__ == '__main__':
    # simulate_trajectory(0, rounds=10)
########################### Simulate trajectories ###########################
    start_time = time.time()
    with Pool(num_processes) as pool:
        results = []
        for i in range(turns):
            result = pool.apply_async(simulate_trajectory, args=(i,))
            results.append(result)

        while len(results) > 0 and any([not result.ready() for result in results]):
            completed = [result.ready() for result in results]
            percent_complete = sum(completed) / turns * 100
            if percent_complete > 0:
                print(f"Progress: {percent_complete:.2f}% , Estimated time remaining: {((time.time() - start_time) / percent_complete) * (100 - percent_complete) /60 :.2f} mins")
            
            time.sleep(30)

########################### Combine all trajectories ###########################
    # traj_list_combined = []
    # file_list = os.listdir('rule_based_traj')
    # for file_name in tqdm(file_list):
    #     if file_name.endswith('.npy'):
    #         file_path = os.path.join('rule_based_traj', file_name)
    #         traj_list = np.load(file_path, allow_pickle=True)
    #         traj_list_combined.extend(traj_list)

    # np.save('rule_based_traj_combined.npy', traj_list_combined, allow_pickle=True)

########################## Look at data ##########################
    # traj = np.load("rule_based_traj/rule_based_traj_combined.npy", allow_pickle=True).tolist()

    # act_length = []
    # rew_sum = []
    # for i in range(len(traj)):
    #     act_length.append(len(traj[i].acts))
    #     if len(traj[i].rews) == 400:
    #         rew_sum.append(np.sum(traj[i].rews))

    # import matplotlib.pyplot as plt

    # # Draw histogram of action lengths
    # plt.figure()
    # plt.hist(act_length, bins=20)
    # plt.xlabel('Action Length')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Action Lengths')
    # plt.savefig('rule_based_traj/hist_act_length.png')
    # plt.show()

    # # Draw histogram of reward means
    # plt.figure()
    # plt.hist(rew_sum, bins=20)
    # plt.xlabel('Reward Mean')
    # plt.ylabel('Frequency')
    # plt.title('Histogram of Reward Sums')
    # plt.savefig('rule_based_traj/hist_rew_sum-400steps.png')
    # plt.show()

################# Filter data, keep only steps with 400 steps #################
    # traj = np.load("rule_based_traj/rule_based_traj_combined.npy", allow_pickle=True).tolist()
    # traj_filtered = [traj[i] for i in range(len(traj)) if len(traj[i].acts) == 400]
    # np.save('rule_based_traj/rule_based_traj_filtered.npy', traj_filtered, allow_pickle=True)