from collections import namedtuple, deque

import pickle
from typing import List

import events as e
from .callbacks import state_to_features
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from GetFeatures import GetFeatures
from GetReward import GetReward
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
import os
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import settings as s
BATCH = 600 # 16 #2048

def setup_training(self):
    """
    Initialise self for training purpose.

    This is called after `setup` in callbacks.py.

    :param self: This object is passed to all callbacks and you can set arbitrary values.
    """
    self.observations = []
    self.target_actions = []
    self.rewards = []
    self.get_reward_class = GetReward(self.random_seed,directly_rule_based = True)
    self.get_reward_class.events = None
    # ccp_alpha
    # max_leaf_nodes
    # min_samples_leaf
    # max_depth

    self.model = RandomForestClassifier(n_estimators = 1500, ccp_alpha= 0.0001, n_jobs = -1, oob_score=True)
    self.metadata = {"global_steps": 0,"params": self.model.get_params()}
    delete_all_files_in_folder('./tb_logs')

    # self.model = joblib.load('./models/random_forest_model.joblib')
    # with open('metadata.pickle', 'rb') as file:
    #     self.metadata = pickle.load(file)
    
    # self.writer = SummaryWriter("./tb_logs")


def delete_all_files_in_folder(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    # Iterate over files in the folder and delete them
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")




def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: List[str]):
    if events == None:
        self.get_reward_class.events = []
    else:
        self.get_reward_class.events = events # store events to pass to GetReward
    
    update_rewards_from_events(self) # after act(), so the current reward is already there
#     """
#     Called once per step to allow intermediate rewards based on game events.

#     When this method is called, self.events will contain a list of all game
#     events relevant to your agent that occurred during the previous step. Consult
#     settings.py to see what events are tracked. You can hand out rewards to your
#     agent based on these events and your knowledge of the (new) game state.

#     This is *one* of the places where you could update your agent.

#     :param self: This object is passed to all callbacks and you can set arbitrary values.
#     :param old_game_state: The state that was passed to the last call of `act`.
#     :param self_action: The action that you took.
#     :param new_game_state: The state the agent is in now.
#     :param events: The events that occurred when going from  `old_game_state` to `new_game_state`
#     """
#     self.logger.debug(f'Encountered game event(s) {", ".join(map(repr, events))} in step {new_game_state["step"]}')

#     # Idea: Add your own events to hand out rewards
#     if ...:
#         events.append(PLACEHOLDER_EVENT)

#     # state_to_features is defined in callbacks.py
#     self.transitions.append(Transition(state_to_features(old_game_state), self_action, state_to_features(new_game_state), reward_from_events(self, events)))


def end_of_round(self, last_game_state: dict, last_action: str, events: List[str]):
    """
    Called at the end of each game or when the agent died to hand out final rewards.
    This replaces game_events_occurred in this round.

    This is similar to game_events_occurred. self.events will contain all events that
    occurred during your agent's final step.

    This is *one* of the places where you could update your agent.
    This is also a good place to store an agent that you updated.

    :param self: The same object that is passed to all of your callbacks.
    """
    # get reward for this action needs next game_state
    # so now we get reward for last action
    self.rewards.append(self.get_reward_class.get_reward(last_game_state,self.target_actions[-1], events))
    # print(self.rewards)
    update_rewards_from_events(self)

    if last_game_state["round"] % BATCH == 0:
        total_rewards = sum([reward for reward in self.rewards if reward > 0])

        self.observations = np.array(self.observations)
        self.target_actions = np.array(self.target_actions)
        # print(self.rewards)
        self.rewards = np.array(self.rewards) / total_rewards

        with open('train_data.pickle', 'wb') as file:
            pickle.dump([self.observations, self.target_actions, self.rewards], file)
        
        update_model(self)
        self.observations = []
        self.target_actions = []
        self.rewards = []
    
    # add data to tensorboard
    # self.writer.add_scalar("score",  self.model.score(self.observations,self.target_actions, self.rewards), self.metadata["global_steps"])
    # self.writer.add_histogram("weights", self.rewards, self.metadata["global_steps"])
    # self.writer.add_scalar("n_estimators", self.model.n_estimators, self.metadata["global_steps"])

    # reset after end round
    self.get_reward_class.reset()

def update_rewards_from_events(self):
    for event in self.get_reward_class.events:
        match(event):
            case e.COIN_COLLECTED:
                # game_event_reward += 1000
                self.rewards[-1] += 500
                for i in range(2, min(s.BOMB_TIMER, len(self.rewards)) + 1):
                    self.rewards[-i] += 500 / i
            case e.KILLED_OPPONENT:
                # game_event_reward += 5000
                self.rewards[-s.BOMB_TIMER -1] += 2500
                for i in range(1, min(s.BOMB_TIMER, len(self.rewards)) + 1):
                    self.rewards[-i] += 100 * i # after dropping the bomb
                for i in range(1, min(s.BOMB_TIMER, len(self.rewards) - s.BOMB_TIMER -1) + 1):
                    self.rewards[-s.BOMB_TIMER -1 -i] += 500 / i # before dropping the bomb
            case e.KILLED_SELF:
                # game_event_reward -= 2000
                self.rewards[-s.BOMB_TIMER -1] -= 1000
                for i in range(1, min(s.BOMB_TIMER, len(self.rewards)) + 1):
                    self.rewards[-i] -= 200
            case e.GOT_KILLED:
                # game_event_reward -= 1000
                for i in range(1, s.BOMB_TIMER + 1):
                    self.rewards[-i] -= 500

            # case e.OPPONENT_ELIMINATED:
            #     game_event_reward -= 10
            # case e.SURVIVED_ROUND:
            #     for i in range(1, min(100, len(self.rewards)) + 1):
            #         self.rewards[-i] += 200
    
def update_model(self):
    n_splits = 5

    # Initialize the KFold object
    kf = KFold(n_splits=n_splits)

    # Initialize a variable to keep track of the performance metrics (e.g., accuracy)
    total_accuracy = 0

    # Iterate through the folds
    for train_index, test_index in kf.split(self.target_actions):
        X_train, X_test = self.observations[train_index], self.observations[test_index]
        y_train, y_test = self.target_actions[train_index], self.target_actions[test_index]
        weights_train, weights_test = self.rewards[train_index], self.rewards[test_index]

        # Train the model
        self.model.fit(X_train, y_train, sample_weight=weights_train)  # Use sample weights

        # Evaluate the model
        accuracy = self.model.score(X_test, y_test, sample_weight=weights_test)  # Use sample weights
        total_accuracy += accuracy

    # Calculate the average performance metric
    average_accuracy = total_accuracy / n_splits

    # Store the model
    joblib.dump(self.model, './models/random_forest_model.joblib')
    with open('metadata.pickle', 'wb') as file:
        pickle.dump(self.metadata, file)

    self.metadata["global_steps"] += len(self.target_actions)*(n_splits - 1) # update global_steps
    
    print("\naverage_accuracy:", average_accuracy)
    print("OOB_score:", self.model.oob_score_)
    print("total_steps:",self.metadata["global_steps"])


# def reward_from_events(self, events: List[str]) -> int:
#     """
#     *This is not a required function, but an idea to structure your code.*

#     Here you can modify the rewards your agent get so as to en/discourage
#     certain behavior.
#     """
#     game_rewards = {
#         e.COIN_COLLECTED: 1,
#         e.KILLED_OPPONENT: 5,
#         PLACEHOLDER_EVENT: -.1  # idea: the custom event is bad
#     }
#     reward_sum = 0
#     for event in events:
#         if event in game_rewards:
#             reward_sum += game_rewards[event]
#     self.logger.info(f"Awarded {reward_sum} for events {', '.join(events)}")
#     return reward_sum
