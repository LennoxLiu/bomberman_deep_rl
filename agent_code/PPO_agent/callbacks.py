from stable_baselines3 import PPO
from CustomEnv import fromStateToObservation

def setup(self):
    self.model = PPO.load("./agent_code/PPO_agent/_bomberman")


def act(self, game_state: dict):
    observation = fromStateToObservation(game_state)
    action, _states = self.model.predict(observation)
    return action
