from stable_baselines3 import PPO
from CustomEnv import fromStateToObservation, ACTION_MAP
from CustomEnv import CustomEnv

def setup(self):
    self.model = PPO.load("ppo_bomberman")
    print(self.model)


def act(self, game_state: dict):
    observation = fromStateToObservation(game_state)

    action, _states = self.model.predict(observation)
    print(ACTION_MAP[action])
    self.logger.info("Pick action: "+ ACTION_MAP[action])

    return ACTION_MAP[action]
