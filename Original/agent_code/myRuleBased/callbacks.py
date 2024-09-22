import pickle
import os
import sys
# Add the current folder to the system path
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
import CustomEnv
from RuleBasedAgent import RuleBasedAgent

import numpy as np
import settings as s

def setup(self):
    self.model = RuleBasedAgent(has_memory=True)
    print(self.model)


def act(self, game_state: dict):
    action = self.model.act(game_state)

    return action
