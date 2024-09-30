
from stable_baselines3.common import policies
from collections import deque
from random import shuffle
import numpy as np
from RuleBasedAgent import RuleBasedAgent
import settings as s
import torch as th
from CustomEnv import ACTION_MAP

class RuleBasedPolicy(policies.BasePolicy):
    def __init__(self, observation_space, action_space, *args, **kwargs):
        super(RuleBasedPolicy, self).__init__(observation_space, action_space, *args, **kwargs)

        self.rule_based_agent = RuleBasedAgent(has_memory=False)
        

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        """
        Get the action according to the policy for a given observation.

        By default provides a dummy implementation -- not all BasePolicy classes
        implement this, e.g. if they are a Critic in an Actor-Critic method.

        :param observation:
        :param deterministic: Whether to use stochastic or deterministic actions
        :return: Taken action according to the policy
        """

        # Get the stacked actions
        return th.tensor([ACTION_MAP.index(self.rule_based_agent.act(observation[i])) for i in range(observation.shape[0])], dtype=th.int8)

        
