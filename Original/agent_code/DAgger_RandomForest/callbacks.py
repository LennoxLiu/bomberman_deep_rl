import pickle
import os
import sys
# Add the current folder to the system path
current_folder = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_folder)
from CustomEnv import fromStateToObservation, ACTION_MAP
import train_utils as tu
import torch
from GetFeatures import GetFeatures
import settings as s
import numpy as np

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
    obs1 = obs1 / 8
    obs2 = obs2 / s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4

    return np.concatenate((obs1, obs2))

def setup(self):
    self.deep_model = torch.load(open('policy-checkpoint.pkl','rb'), map_location=torch.device('cpu'))
    print(self.deep_model)

    self.classic_model = pickle.load(open('decision_tree_model.pkl','rb'))
    self.feature_extractor = GetFeatures()

    self.prev_position = []
    self.using_deep_model = True
    self.prev_is_classic_model = 0

def act(self, game_state: dict):
    current_pos = game_state['self'][3]
    step = game_state['step']
    
    self.prev_position.append(current_pos)
    if len(self.prev_position) > 3 and \
        self.prev_position[0] == self.prev_position[2] and \
        self.prev_position[1] == self.prev_position[3]:
        
        self.using_deep_model = not self.using_deep_model  # If the agent is stuck, switch the model
        self.prev_position= []

        model_name = "Deep" if self.using_deep_model else "Classic"
        print(f"Step {step}: Stuck detected, switch to ",model_name," model")
        self.logger.info(f"Step {step}: Stuck detected, switch to {model_name} model")
    
    if len(self.prev_position) > 3:
        self.prev_position.pop(0)
    if self.using_deep_model == False:
        self.prev_is_classic_model += 1
    else:
        self.prev_is_classic_model = 0
    
    if self.prev_is_classic_model > 3:
        self.using_deep_model = True
        print(f"Step {step}: Switch to Deep model after 3 consecutive steps of Classic model")
        self.logger.info(f"Step {step}: Switch to Deep model after 3 consecutive steps of Classic model")
    
    observation = fromStateToObservation(game_state)
    if self.using_deep_model:
        action = self.deep_model.predict(observation, deterministic=True)[0]
    else:
        observation_crop = crop_observation(observation, 9, 17)
        feature = self.feature_extractor.state_to_features(game_state)
        input = np.concatenate((feature,observation_crop)).reshape(1, -1)
        action = self.classic_model.predict(input)[0]

    # print(ACTION_MAP[action])
    self.logger.info("Pick action: "+ ACTION_MAP[action])

    return ACTION_MAP[action]
