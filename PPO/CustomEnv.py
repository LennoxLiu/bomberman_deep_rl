import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import settings as s


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, arg1, arg2,):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(6) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        # Do not pass "round" and "user_input"
        COIN_COUNT = 50
        self.observation_space = spaces.Dict(
            {   "step": Discrete(s.MAX_STEPS), \ 
                "field": MultiDiscrete([s.COLS, s.ROWS, 3]), \
                "bombs": Dict({"position": MultiDiscrete([s.COLS, s.ROWS]), "countdown": Discrete(s.BOMB_TIMER + 1)}), \
                "explosion_map": MultiDiscrete([s.COLS, s.ROWS, s.EXPLOSION_TIMER + 1]), \
                "coins": MultiDiscrete([s.COLS, s.ROWS]), \
                "self": Dict({  "score": Box(low=0, dtype=np.float64), \
                                "bomb_possible": Discrete(2), \
                                "position": MultiDiscrete([s.COLS, s.ROWS])
                            }), \
                "others": Dict({    "id": Discrete(3),\
                                    "status": Dict({    "score": Box(low=0, dtype=np.float64),\
                                                        "bomb_possible": Discrete(2),\
                                                        "position": MultiDiscrete([s.COLS, s.ROWS])\
                                                    }) \
                                } \
                            ) \
            } \
            )

    def step(self, action):

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return observation, info

    def render(self):

    def close(self):
        None


if __name__ == "__main__":
    env = CustomEnv(arg1, ...)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)