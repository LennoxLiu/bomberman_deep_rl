from model import RainbowAgent
from tianshou.data import Batch
import numpy as np
import events as e

def setup(self):
    """Initialize the RainbowAgent when the game starts."""
    state_shape = (17, 17) 
    action_shape = 6  # UP, DOWN, LEFT, RIGHT, WAIT, BOMB

    self.agent = RainbowAgent(state_shape, action_shape)

    self.last_min_distance = float('inf')

def act(self, game_state):
    """Choose action for each step using the Rainbow DQN model and calculate reward."""
    action, reward = self.agent.act(game_state)

    self.logger.info(f"Action chosen: {action}, Reward: {reward}")

    return action

def calculate_reward(self, game_state, events):
    """
    Calculate the reward based on the game state and events.
    
    """
    reward = 0.0

    if e.KILLED_OPPONENT in events:
        reward += 1.0  
    if e.SURVIVED_ROUND in events:
        reward += 1.0 
    if e.CRATE_DESTROYED in events:
        reward += 0.1  

    # 每次迭代的惩罚（防止无所作为）
    reward -= 0.01

    opponent_positions = [opponent[3] for opponent in game_state["others"]]
    if opponent_positions:
        self_position = game_state["self"][3]
        min_distance = min([np.linalg.norm(np.array(self_position) - np.array(opponent_pos)) for opponent_pos in opponent_positions])
        
        if min_distance < self.last_min_distance:
            reward += 0.002 
        else:
            reward -= 0.002

        self.last_min_distance = min_distance
    
    if game_state["in_bomb_danger"]:
        reward -= 0.000666

    if game_state["safe_zone"]:
        reward += 0.002

    return reward

def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    """
    Record the game events that occurred and update the model.
    
    Args:
        old_game_state (dict): The previous game state.
        self_action (str): The action taken by the agent.
        new_game_state (dict): The new game state after taking the action.
        events (list): A list of events that occurred as a result of the action.
    """
    old_state = self.agent.process_game_state(old_game_state)

    new_state = self.agent.process_game_state(new_game_state)

    reward = self.calculate_reward(new_game_state, events)

    batch = Batch(
        obs=[old_state], 
        act=[self_action], 
        rew=[reward], 
        obs_next=[new_state], 
        done=[False] 
    )

    self.agent.update(batch)

def end_of_episode(self, last_game_state, last_action, events):
    """
    Called at the end of each episode to clean up and finalize any episode-specific operations.
    
    Args:
        last_game_state (dict): The final game state at the end of the episode.
        last_action (str): The last action taken by the agent.
        events (list): A list of events that occurred in the final game step.
    """
    final_state = self.agent.process_game_state(last_game_state)
    final_reward = self.calculate_reward(last_game_state, events)

    batch = Batch(
        obs=[final_state],
        act=[last_action],
        rew=[final_reward],
        obs_next=[None],  # 没有下一个状态，因为是最后一步
        done=[True]  # 表示回合结束
    )

    self.agent.update(batch)

    self.logger.info(f"End of episode, events: {events}")
