from stable_baselines3 import DQN
from CustomEnv import fromStateToObservation, ACTION_MAP
from GetFeatures import GetFeatures

def setup(self):
    self.model = DQN.load("dqn_bomberman",device="cpu")
    self.get_features_class = GetFeatures()


def act(self, game_state: dict):
    observation = fromStateToObservation(self.get_features_class, game_state)

    action, _states = self.model.predict(observation, deterministic=True)
    print(ACTION_MAP[action])
    self.logger.info("Pick action: "+ ACTION_MAP[action])

    return ACTION_MAP[action]

def reset_self(self):
    self.get_features_class.reset()