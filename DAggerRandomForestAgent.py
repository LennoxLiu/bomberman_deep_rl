from collections import deque
from random import random, shuffle
import string
import torch
import numpy as np
from CustomEnv import ACTION_MAP, fromObservationToState, fromStateToObservation
import settings as s
import pickle
from GetFeatures import GetFeatures
from train_DecisionTree import crop_observation

class DAggerRandomForestAgent:
    def __init__(self):
        self.deep_model = torch.load(open('models/DAgger_RandomForest/policy-checkpoint.pkl','rb'))
        # print(self.deep_model)

        self.classic_model = pickle.load(open('models/DAgger_RandomForest/decision_tree_model.pkl','rb'))
        self.feature_extractor = GetFeatures()

        self.prev_position = []
        self.using_deep_model = True
        self.prev_is_classic_model = 0
    
    def act(self, observation):
        game_state = fromObservationToState(observation)
        current_pos = game_state['self'][3]
        
        self.prev_position.append(current_pos)
        if len(self.prev_position) > 3 and \
            self.prev_position[0] == self.prev_position[2] and \
            self.prev_position[1] == self.prev_position[3]:
            
            self.using_deep_model = not self.using_deep_model  # If the agent is stuck, switch the model
            self.prev_position= []

            model_name = "Deep" if self.using_deep_model else "Classic"
            
        if len(self.prev_position) > 3:
            self.prev_position.pop(0)
        if self.using_deep_model == False:
            self.prev_is_classic_model += 1
        else:
            self.prev_is_classic_model = 0
        
        if self.prev_is_classic_model > 3:
            self.using_deep_model = True
            
        observation = fromStateToObservation(game_state)
        if self.using_deep_model:
            action = self.deep_model.predict(observation, deterministic=True)[0]
        else:
            observation_crop = crop_observation(observation, 9, 17)
            feature = self.feature_extractor.state_to_features(game_state)
            input = np.concatenate((feature,observation_crop)).reshape(1, -1)
            action = self.classic_model.predict(input)[0]

        return ACTION_MAP[action]
    
    def reset(self):
        self.prev_position = []
        self.using_deep_model = True
        self.prev_is_classic_model = 0