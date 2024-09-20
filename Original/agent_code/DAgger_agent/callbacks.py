import pickle
import os
import sys
# Add the current folder to the system path
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
from CustomEnv import fromStateToObservation, ACTION_MAP
import train_utils as tu
import torch

def setup(self):
    self.model = torch.load(open('policy-checkpoint00002.pkl','rb'), map_location=torch.device('cpu'))
    print(self.model)


def act(self, game_state: dict):
    observation = fromStateToObservation(game_state)
    action = self.model.predict(observation, deterministic=True)[0]

    # print(ACTION_MAP[action])
    self.logger.info("Pick action: "+ ACTION_MAP[action])

    return ACTION_MAP[action]
