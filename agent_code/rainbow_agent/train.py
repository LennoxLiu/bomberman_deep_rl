import torch
from tianshou.data import Collector, PrioritizedVectorReplayBuffer
from tianshou.trainer import OffpolicyTrainer
from tianshou.env import DummyVectorEnv
from model import create_policy
import sys
import os
sys.path.append(sys.path.append('/Users/liangxinyu/Desktop/bomberman_rl-1/environment.py'))
from environment import BombeRLeWorld
import settings as s


def create_env():
    """
    Creates the game environment for training the agent.
    This is a vectorized environment using DummyVectorEnv for parallelization.

    Returns:
        env: A DummyVectorEnv object.
    """
    def init_env():
        return BombeRLeWorld()

    return DummyVectorEnv([init_env for _ in range(4)])  # 4 parallel environments for training


def train_agent(num_epochs=1000, buffer_size=20000, batch_size=64, lr=1e-4, gamma=0.99, n_step=3):
    """
    Trains the Rainbow DQN agent to play the game and beat the random agent.

    Args:
        num_epochs (int): Number of epochs to train the agent.
        buffer_size (int): The size of the replay buffer.
        batch_size (int): The batch size for training.
        lr (float): Learning rate for the optimizer.
        gamma (float): Discount factor for the DQN.
        n_step (int): N-step returns for Rainbow DQN.

    Returns:
        None
    """
    env = create_env()

    state_shape = (s.COLS, s.ROWS) 
    action_shape = 6  

    policy = create_policy(state_shape, action_shape, Vmin=-10, Vmax=10, atom_size=51, lr=lr, gamma=gamma)

    buffer = PrioritizedVectorReplayBuffer(size=buffer_size, buffer_num=1, alpha=0.6, beta=0.4)

    train_collector = Collector(policy, env, buffer)

    def save_fn(policy):
        torch.save(policy.state_dict(), 'rainbow_dqn_policy.pth')

    def stop_fn(mean_rewards):
        return mean_rewards >= 300  # Stop training

    result = OffpolicyTrainer(
        policy=policy,
        train_collector=train_collector,
        test_collector=None,  # Optional: define a test collector if you want to test the policy during training
        max_epoch=num_epochs, 
        step_per_epoch=5000, 
        step_per_collect=10, 
        update_per_step=1, 
        batch_size=batch_size, 
        stop_fn=stop_fn,  
        save_fn=save_fn
    )

    print(f"Training complete! Best reward: {result['best_reward']}")


if __name__ == '__main__':
    train_agent()
