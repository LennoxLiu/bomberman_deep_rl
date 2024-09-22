import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tianshou.policy import RainbowPolicy
from tianshou.utils.net.common import Net
from tianshou.utils.net.discrete import NoisyLinear
from tianshou.data import Batch, PrioritizedVectorReplayBuffer


class RainbowNet(nn.Module):
    """
    Neural network model for Rainbow DQN adapted to the BombeRLeWorld environment.
    """
    def __init__(self, state_shape, action_shape, atom_size=51, dueling=True):
        super(RainbowNet, self).__init__()
        self.action_shape = action_shape
        self.atom_size = atom_size
        
        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(np.prod(state_shape), 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # Dueling architecture for value and advantage streams
        self.value_stream = nn.Linear(128, atom_size)
        self.advantage_stream = nn.Linear(128, action_shape * atom_size)

    def forward(self, obs, state=None, info={}):
        # Feature extraction
        feature = self.feature_layer(obs)
        value = self.value_stream(feature).view(-1, 1, self.atom_size)
        advantage = self.advantage_stream(feature).view(-1, self.action_shape, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_atoms


def create_policy(state_shape, action_shape, Vmin, Vmax, atom_size=51, lr=1e-3, gamma=0.99):
    """
    Create a Rainbow DQN policy.
    
    Args:
        state_shape (tuple): Shape of the state input.
        action_shape (int): Number of possible actions.
        Vmin (float): Minimum possible value for Q-distribution.
        Vmax (float): Maximum possible value for Q-distribution.
        atom_size (int): Number of atoms for Q-value distribution.
        lr (float): Learning rate for optimizer.
        gamma (float): Discount factor for Q-learning.
        
    Returns:
        policy (RainbowPolicy): The Rainbow DQN policy.
    """
    # Create the network
    net = RainbowNet(state_shape, action_shape, atom_size)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    # Create the Rainbow DQN policy
    policy = RainbowPolicy(
        model=net,
        optim=optimizer,
        discount_factor=gamma,
        estimation_step=1,
        target_update_freq=500,
        reward_normalization=False,
        is_double=True,
        noise=True,
        v_min=Vmin,
        v_max=Vmax,
        n_step=3,
        target_update_mode="hard",
    )
    return policy


# Replay buffer for storing experience
def create_replay_buffer(size=20000, alpha=0.6, beta=0.4):
    """
    Create a prioritized experience replay buffer.
    
    Args:
        size (int): Maximum buffer size.
        alpha (float): Priority factor for prioritized experience replay.
        beta (float): Importance-sampling weight factor.
        
    Returns:
        replay_buffer (PrioritizedVectorReplayBuffer): Experience replay buffer.
    """
    return PrioritizedVectorReplayBuffer(
        size, buffer_num=1, alpha=alpha, beta=beta, stack_num=4
    )


class RainbowAgent:
    """
    Wrapper class for managing Rainbow DQN with training and update logic.
    """
    def __init__(self, state_shape, action_shape, lr=1e-4, gamma=0.99, n_step=3):
        """
        Initialize RainbowAgent with a RainbowDQN model.
        
        Args:
            state_shape (tuple): The shape of the input state.
            action_shape (int): Number of actions.
            lr (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            n_step (int): Multi-step learning parameter.
        """
        self.policy = create_policy(state_shape, action_shape, Vmin=-10, Vmax=10, lr=lr, gamma=gamma)
        self.replay_buffer = create_replay_buffer(size=20000)

    def process_game_state(self, game_state):
        """
        Process the game_state dictionary into a flat array for the model input.
        
        Args:
            game_state (dict): The current game state.
            
        Returns:
            np.ndarray: Flattened state vector.
        """
        field = game_state['field'].flatten()  # Extract field data
        explosion_map = game_state['explosion_map'].flatten()  # Extract explosion map
        self_position = np.array(game_state['self'][3])  # Agent's position
        others = np.concatenate([np.array(other[3]) for other in game_state['others']]) if game_state['others'] else np.zeros(2)
        bombs = np.concatenate([np.array(bomb[0]) for bomb in game_state['bombs']]) if game_state['bombs'] else np.zeros(2)
        coins = np.concatenate([np.array(coin) for coin in game_state['coins']]) if game_state['coins'] else np.zeros(2)

        # Concatenate all extracted features into a single vector
        state = np.concatenate([field, explosion_map, self_position, others, bombs, coins])
        return state

    def calculate_reward(self, game_state, action):
        """
        Calculate the reward based on the game state and action taken.
        
        Args:
            game_state (dict): The current game state.
            action (int): The action taken by the agent.
            
        Returns:
            float: The calculated reward.
        """
        reward = 0.0

        # Rewards for killing opponents, surviving, and destroying crates
        if game_state["killed_opponent"]:
            reward += 1.0
        if game_state["self_alive"]:
            reward += 1.0
        if game_state["crate_destroyed"]:
            reward += 0.1

        # Reward and penalty for moving closer to or farther from opponents
        opponent_positions = [opponent[3] for opponent in game_state["others"]]
        if opponent_positions:
            self_position = game_state["self"][3]
            min_distance = min([np.linalg.norm(np.array(self_position) - np.array(opponent_pos)) for opponent_pos in opponent_positions])
            if min_distance < getattr(self, 'last_min_distance', float('inf')):
                reward += 0.002  # Moving closer
            else:
                reward -= 0.002  # Moving farther
            self.last_min_distance = min_distance

        # Time penalty and bomb danger
        reward -= 0.01  # Time penalty
        if game_state["in_bomb_danger"]:
            reward -= 0.000666
        if game_state["safe_zone"]:
            reward += 0.002  # Staying in a safe zone

        return reward

    def act(self, game_state):
        """
        Decide action for each step using Rainbow DQN policy and calculate reward.
        
        Args:
            game_state (dict): The current game state.
            
        Returns:
            int: The action to be taken (as an integer).
            float: The calculated reward.
        """
        # Convert game state to model input
        state = self.process_game_state(game_state)
        # Select action using the policy
        action = self.policy(Batch(obs=[state])).act[0]
        # Calculate reward based on the current game state and action
        reward = self.calculate_reward(game_state, action)
        return action, reward

    def update(self, batch):
        """
        Update the model based on a batch of transitions.
        
        Args:
            batch (Batch): A batch of transitions.
            
        Returns:
            dict: A dictionary containing the loss value.
        """
        loss = self.policy.learn(batch)
        return {"loss": loss.item()}
