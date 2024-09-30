def setup(self):
    pass


def act(self, game_state: dict):
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']

def state_to_features(game_state: dict):
    pass