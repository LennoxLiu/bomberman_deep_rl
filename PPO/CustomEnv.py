import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import settings as s

import main
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ACTION_MAP=['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, arg1, arg2,):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
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


    def my_render(self, wait_until_due):
        # If every step should be displayed, wait until it is due to be shown
        if wait_until_due:
            self.gui_timekeeper.wait()

        if self.gui_timekeeper.is_due():
            self.gui_timekeeper.note()
            # Render (which takes time)
            self.gui.render()
            pygame.display.flip()


    def step(self, action):
        self.world.do_step(ACTION_MAP[action])
        self.user_input = None
        
        # how to get observation in here?
        get_state_for_agent()

        return observation, reward, terminated, truncated, _

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # following documentation
        # start a new round

        # train the model using "user input"
        self.world, n_rounds, self.gui, self.every_step, \
            turn_based, self.make_video, update_interval = \
                main.my_main_parser(argv="play --my-agent PPO_agent")

        # world_controller(world, args.n_rounds,
        #                  gui=gui, every_step=every_step, turn_based=args.turn_based,
        #                  make_video=args.make_video, update_interval=args.update_interval)
        
        # my world controller
        if self.make_video and not self.gui.screenshot_dir.exists():
            self.gui.screenshot_dir.mkdir()

        self.gui_timekeeper = main.Timekeeper(update_interval)
        self.user_input = None

        # one CustomEnv only run one round
        self.world.new_round()

        self.PPO_agent = None
        for a in self.agent:
            if a.name == "user_agent":
                self.PPO_agent = a
        
        # Get first observation
        state = self.world.get_state_for_agent(self.PPO_agent)
        

        assert self.observation_space.contains(observation)
        return observation, _

    def render(self):
        if self.gui is not None:
            self.my_render(self.every_step)


    def close(self):
        None


if __name__ == "__main__":
    env = CustomEnv(arg1, ...)
    # It will check your custom environment and output additional warnings if needed
    check_env(env)