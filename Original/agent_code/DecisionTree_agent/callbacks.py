import pickle
import os
import sys

# Add the current folder to the system path
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)

from CustomEnv import fromStateToObservation, ACTION_MAP
import train_utils as tu
import torch
import numpy as np
import settings as s

# Crop the observation for smaller feature space
def crop_observation(observation, crop_size_1, crop_size_2):
    obs1, obs2 = observation[0], observation[1]
        
    crop_diam_1 = int((crop_size_1 - 1) / 2)
    crop_diam_2 = int((crop_size_2 - 1) / 2)
        
    # crop obs to crop_size x crop_size
    obs1 = obs1[s.ROWS-crop_diam_1:s.ROWS+crop_diam_1+1,s.COLS-crop_diam_1:s.COLS+crop_diam_1+1]
    obs2 = obs2[s.ROWS-crop_diam_2:s.ROWS+crop_diam_2+1,s.COLS-crop_diam_2:s.COLS+crop_diam_2+1]
    
    # flatten the cropped observation
    obs1 = obs1.reshape(crop_size_1* crop_size_1)
    obs2 = obs2.reshape(crop_size_2* crop_size_2)

    # Reshape and standardize the input to [0,1]
    # May or may not need to standardize the input for decision tree
    # obs1 = obs1 / 8
    # obs2 = obs2 / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

    return np.concatenate((obs1, obs2))

def setup(self):
    self.model = pickle.load(open('decision_tree_model.pkl','rb'))
    print(self.model)


def act(self, game_state: dict):
    observation = fromStateToObservation(game_state)
    observation = crop_observation(observation, 17, 9).reshape(1, -1)
    action = self.model.predict(observation)[0]

    # print(ACTION_MAP[action])
    self.logger.info("Pick action: "+ ACTION_MAP[action])

    return ACTION_MAP[action]
