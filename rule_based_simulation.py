import numpy as np
from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym
from imitation.data import types

rule_based_agent = RuleBasedAgent()
env = gym.make('CustomEnv-v1')

rounds = 50
traj_list = []
rng = np.random.default_rng(42)
for i in range(rounds):
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

        print(observation, action,reward)

    agent_win, other_scores, user_agent_score =env.close()
    
    if agent_win:
        traj_temp = types.TrajectoryWithRew(obs=np.array(temp_obs,dtype=np.uint8), acts=np.array(temp_actions,dtype=np.uint8), rews=np.array(temp_rewards,dtype=np.float32), infos=None, terminal=True)
        print(isinstance(traj_temp, types.TrajectoryWithRew))

        traj_list.append(traj_temp)
        print("current trajs:",len(traj_list))
        # print("user score:",user_agent_score, "other scores:",other_scores)
    # else:
    #     print("agent lose")
        # print("user score:",user_agent_score, "other scores:",other_scores)
    
print("total trajs:",len(traj_list))


