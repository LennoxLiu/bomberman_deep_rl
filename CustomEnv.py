import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete
import settings as s
import events as e

import main
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ACTION_MAP=['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        # Do not pass "round", opponent score
        self.observation_space = spaces.Dict( 
            {   
                "step": Discrete(s.MAX_STEPS), 
                "field": Box(low = 0, high = 6, shape = (s.COLS, s.ROWS), dtype = np.uint8), 
                # 0: ston walls, 1: free tiles, 2: crates, 3: coins,
                # 4: no bomb opponents, 5: has bomb opponents,
                # 6: self
                "bombs": Box(low = 0, high = s.BOMB_TIMER, shape = (s.COLS, s.ROWS), dtype = np.uint8), 
                "explosion_map": Box(low = 0, high = s.EXPLOSION_TIMER, shape = (s.COLS, s.ROWS), dtype = np.uint8), 
                "self": Dict({  "score": Box(low=0, dtype=np.uint16), 
                                "bomb_possible": Discrete(2), 
                            }), 
            } 
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
        
        # get observation
        game_state = self.world.get_state_for_agent(self.PPO_agent)
        observation = self.fromStateToObservation(game_state)

        # get reward
        # self.PPO_agent.last_game_state, self.PPO_agent.last_action, game_state, self.events
        reward = 0
        for event in self.events:
            match(event):
                case e.MOVED_LEFT | e.MOVED_RIGHT | e.MOVED_UP | e.MOVED_DOWN:
                    reward += 5
                case e.WAITED:
                    reward -= 5
                case e.INVALID_ACTION:
                    reward -= 50
                case e.BOMB_DROPPED:
                    reward += 10
                case e.BOMB_EXPLODED:
                    reward += 1
                case e.CRATE_DESTROYED:
                    reward += 10
                case e.COIN_FOUND:
                    reward += 20
                case e.COIN_COLLECTED:
                    reward += 100
                case e.KILLED_OPPONENT:
                    reward += 500
                case e.KILLED_SELF:
                    reward -= 100
                case e.GOT_KILLED:
                    reward -= 500
                case e.OPPONENT_ELIMINATED:
                    reward -= 10
                case e.SURVIVED_ROUND:
                    reward += 500

        # terminated or trunctated
        terminated = False
        truncated = False
        if self.world.running == False:
            if self.world.step == s.MAX_STEPS:
                terminated = True
            else:
                truncated = True

        return observation, reward, terminated, truncated, None


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
        self.opponent_names = []
        for a in self.agent:
            if a.name == "user_agent":
                self.PPO_agent = a
            else:
                self.opponent_names.append(a.name)
        assert self.PPO_agent != None

        # Get first observation
        game_state = self.world.get_state_for_agent(self.PPO_agent)
        observation = self.fromStateToObservation(game_state)

        return observation, _

    def fromStateToObservation(self, game_state):
        observation = {}
        observation["step"] = game_state["step"]
        
        # 0: ston walls, 1: free tiles, 2: crates, 
        observation["field"] = game_state["field"].astype(np.uint8) + 1
        #3: coins,
        for coin in game_state["coins"]:
            observation["field"][coin] = 3
        # 4: no bomb opponents, 5: has bomb opponents,
        for other in game_state["others"]:
            if other[2] == False: # bombs_left == False
                observation["field"][other[3]] = 4
            else:
                observation["field"][other[3]] = 5
        # 6: self
        observation["field"][game_state["self"][3]] = 6

        # position and countdown of bombs
        observation["bombs"] = game_state["bombs"].astype(np.uint8)

        observation["explosion_map"] = game_state["explosion_map"].astype(np.uint8)
        
        observation["self"] = {"score": game_state["self"][1], "bomb_possible": game_state["self"][2]}

        assert self.observation_space.contains(observation)
        return observation

    def render(self):
        if self.gui is not None:
            self.my_render(self.every_step)


    def close(self):
        if self.make_video:
            self.gui.make_video()
        
        # Can render end screen until next round is queried
        
        self.world.end()


if __name__ == "__main__":
    env = CustomEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)