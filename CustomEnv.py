import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, Dict, Discrete
import settings as s
import events as e
import agents

import main
from environment import BombeRLeWorld, GUI
from fallbacks import pygame, LOADED_PYGAME
from replay import ReplayWorld

ACTION_MAP=['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def fromStateToObservation(game_state):
        # 0: ston walls, 1: free tiles, 2: crates, 
        observation = game_state["field"].astype(np.uint8) + 1
        #3: coins,
        for coin in game_state["coins"]:
            observation[coin] = 3
        # 4: no bomb opponents, 5: has bomb opponents,
        for other in game_state["others"]:
            if other[2] == False: # bombs_left == False
                observation[other[3]] = 4
            else:
                observation[other[3]] = 5
        # 6: no bomb self, 7: has bomb self
        if game_state["self"][2] == False:
            observation[game_state["self"][3]] = 6
        else:
            observation[game_state["self"][3]] = 7

        # 8~8+s.EXPLOSION_TIMER: explosion map
        explosion_map = game_state["explosion_map"].astype(np.uint8)
        # Replace elements in observation with corresponding elements+8 from explosion_map if explosion_map elements are non-zero
        observation[explosion_map != 0] = explosion_map[explosion_map != 0] + 8
        
        # 8+s.EXPLOSION_TIMER ~ 8+s.EXPLOSION_TIMER+ s.BOMB_TIMER: bomb map
        for bomb in game_state["bombs"]:
            observation[bomb[0]] = 8 + s.EXPLOSION_TIMER + bomb[1]
        assert Box(low = 0, high = 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER, shape = (s.COLS, s.ROWS), dtype = np.uint8).contains(observation)

        observation = observation.reshape(-1)
        assert Box(low = 0, high = 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER, shape = (s.COLS * s.ROWS,), dtype = np.uint8).contains(observation)
    
        return observation


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, options = {"argv": ["play","--no-gui","--my-agent","user_agent"]}):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        # Do not pass "round", opponent score
        self.observation_space = spaces.Box(low = 0, high = 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER, shape = (s.COLS * s.ROWS,), dtype = np.uint8)
                # 0: ston walls, 1: free tiles, 2: crates, 3: coins,
                # 4: no bomb opponents, 5: has bomb opponents,
                # 6: no bomb self, 7: has bomb self
                # 8~8+s.EXPLOSION_TIMER: explosion map
                # 8+s.EXPLOSION_TIMER ~ 8+s.EXPLOSION_TIMER+ s.BOMB_TIMER: bomb map
        
        # train the model using "user input"
        self.world, n_rounds, self.gui, self.every_step, \
            turn_based, self.make_video, update_interval = \
                main.my_main_parser(argv=options["argv"])

        # world_controller(world, args.n_rounds,
        #                  gui=gui, every_step=every_step, turn_based=args.turn_based,
        #                  make_video=args.make_video, update_interval=args.update_interval)
        
        # my world controller
        if self.make_video and not self.gui.screenshot_dir.exists():
            self.gui.screenshot_dir.mkdir()

        self.gui_timekeeper = main.Timekeeper(update_interval)
        self.world.user_input = None

        # store my agent
        self.PPO_agent = None
        for a in self.world.agents:
            if a.name == "user_agent":
                self.PPO_agent = a
        assert isinstance(self.PPO_agent, agents.Agent)

        # start a new round
        self.world.new_round()


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

        terminated = False
        truncated = False

        # get observation
        game_state = self.world.get_state_for_agent(self.PPO_agent)
        if game_state == None: # the agent is dead
            truncated = True
            observation = fromStateToObservation(self.PPO_agent.last_game_state)
        else:
            observation = fromStateToObservation(game_state)

        # get reward
        # self.PPO_agent.last_game_state, self.PPO_agent.last_action, game_state, self.events
        reward = 0
        for event in self.PPO_agent.events:
            match(event):
                case e.MOVED_LEFT | e.MOVED_RIGHT | e.MOVED_UP | e.MOVED_DOWN:
                    reward += 2
                case e.WAITED:
                    reward += 0.5
                case e.INVALID_ACTION:
                    reward -= 50
                case e.BOMB_DROPPED:
                    reward += 1
                case e.BOMB_EXPLODED:
                    reward += 1
                case e.CRATE_DESTROYED:
                    reward += 1
                case e.COIN_FOUND:
                    reward += 5
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
        if self.world.running == False:
            if self.world.step == s.MAX_STEPS:
                terminated = True
            else:
                truncated = True

        # the reward in gym is the smaller the better
        return observation, reward, terminated, truncated, {"events" : self.PPO_agent.events}


    def reset(self, seed = None):
        super().reset(seed=seed) # following documentation
        
        # start a new round
        self.world.new_round()

        # Get first observation
        game_state = self.world.get_state_for_agent(self.PPO_agent)
        observation = fromStateToObservation(game_state)

        return observation, {"info": "reset"}
    

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