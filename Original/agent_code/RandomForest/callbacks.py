import os
import pickle
import random

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from RuleBasedAgent import RuleBasedAgent
from GetFeatures import GetFeatures

ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
ACTION_INV_MAP = {"UP": 0, "DOWN": 1, "LEFT": 2, "RIGHT": 3, "WAIT": 4, "BOMB": 5}

def setup(self):
    """
    Setup your code. This is called once when loading each agent.
    Make sure that you prepare everything such that act(...) can be called.

    When in training mode, the separate `setup_training` in train.py is called
    after this method. This separation allows you to share your trained agent
    with other students, without revealing your training code.

    In this example, our model is a set of probabilities over actions
    that are is independent of the game state.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    if self.train and not os.path.isfile("./models/random_forest_model.joblib"):
        print("Error: model not found.")
    else:
        self.model = joblib.load('./models/random_forest_model.joblib')
    
    
    self.random_seed = np.random.randint(1, 2**31 -1)
    self.get_feature_class = GetFeatures()
    self.rule_based_model = RuleBasedAgent(self.random_seed)


def act(self, game_state: dict) -> str:
    """
    Your agent should parse the input, think, and take a decision.
    When not in training mode, the maximum execution time for this method is 0.5s.

    :param self: The same object that is passed to all of your callbacks.
    :param game_state: The dictionary that describes everything on the board.
    :return: The action to take as a string.
    """
    observation = self.get_feature_class.state_to_features(game_state)

    # todo Exploration vs exploitation
    exploration_rate = 0
    action = None
    if self.train:
        # if random.random() < exploration_rate:
            # self.logger.debug("Choosing action purely at random.")
            # 80%: walk in any direction. 10% wait. 10% bomb.
            # action =  np.random.choice(ACTION_MAP, p=[.2, .2, .2, .2, .1, .1])
        # else: 
            # do as rule_based model to generate learning data
        action, _ = self.rule_based_model.act(game_state)
            # action = self.model.predict([observation])[0]
            # action = ACTION_MAP[action]
        
        self.observations.append(observation)

        # get reward for this action needs next game_state
        # so now we get reward for last action
        if game_state["step"] > 1:
            self.rewards.append(
                self.get_reward_class.get_reward(
                    game_state, self.target_actions[-1], None))
        else: # first step
            self.get_reward_class.get_reward(game_state, None, [])

        self.target_actions.append(ACTION_INV_MAP[action])

    else: # not training
        action = self.model.predict([observation])[0]
        action = ACTION_MAP[action]
        print(action)

        
    return action


def reset_self(self):
    self.rule_based_model.reset_self()
    self.get_feature_class.reset()

def state_to_features(game_state: dict) -> np.array:
    return np.array(0)
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Converts the game state to the input of your model, i.e.
#     a feature vector.

#     You can find out about the state of the game environment via game_state,
#     which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
#     what it contains.

#     :param game_state:  A dictionary describing the current game board.
#     :return: np.array
#     """
#     # This is the dict before the game begins and after it ends
#     if game_state is None:
#         return None

#     # For example, you could construct several channels of equal shape, ...
#     channels = []
#     channels.append(...)
#     # concatenate them as a feature tensor (they must have the same shape), ...
#     stacked_channels = np.stack(channels)
#     # and return them as a vector
#     return stacked_channels.reshape(-1)
