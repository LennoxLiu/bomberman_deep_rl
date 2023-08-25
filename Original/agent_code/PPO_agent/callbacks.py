from stable_baselines3 import PPO
from CustomEnv import fromStateToObservation, ACTION_MAP

def setup(self):
    self.model = PPO.load("ppo_bomberman")


def act(self, game_state: dict):
    observation = fromStateToObservation(game_state)
    action, _states = self.model.predict(observation)
    print(ACTION_MAP[action])
    return ACTION_MAP[action]
