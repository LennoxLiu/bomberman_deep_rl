from RuleBasedAgent import RuleBasedAgent
import CustomEnv
import gymnasium as gym

rule_based_agent = RuleBasedAgent()
env = gym.make('CustomEnv-v1')

rounds = 100
obs_list = []
actions_list = []
rewards_list = []
for i in range(rounds):
    observation, game_state = env.reset()
    temp_obs = [observation]
    rule_based_agent.reset_self()
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
    
    agent_win=env.close()
    
    if agent_win:
        obs_list.append(temp_obs)
        actions_list.append(temp_actions)
        rewards_list.append(temp_rewards)
        print("current trajs:",len(obs_list))

print("total trajs:",len(obs_list))