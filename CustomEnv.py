import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, MultiDiscrete
import settings as s
import events as e
import agents

import main
from fallbacks import pygame, LOADED_PYGAME
import math

ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def fromStateToObservation(game_state):
        observation = {}
        one_array = np.ones(s.COLS * s.ROWS)
        # 0: stone walls, 1: free tiles, 2: crates, 
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

        # 7~7+s.EXPLOSION_TIMER: explosion map
        explosion_map = game_state["explosion_map"].astype(np.uint8)
        # Replace elements in observation with corresponding elements+8 from explosion_map if explosion_map elements are non-zero
        observation["field"][explosion_map != 0] = explosion_map[explosion_map != 0] + 7
        
        # 8+s.EXPLOSION_TIMER ~ 8+s.EXPLOSION_TIMER+ s.BOMB_TIMER: bomb map
        for bomb in game_state["bombs"]:
            observation["field"][bomb[0]] = 8 + s.EXPLOSION_TIMER + bomb[1]
        
        # 6: self (needs to be present at any time)
        observation["field"][game_state["self"][3]] = 6

        observation["field"] = observation["field"].flatten()
        assert MultiDiscrete(nvec=one_array * ( 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER), dtype = np.uint8).contains(observation["field"])

        observation["bomb_possible"] = int(game_state["self"][2])
        
        assert spaces.Dict({"field": MultiDiscrete(nvec=one_array * ( 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER), dtype = np.uint8),"bomb_possible": spaces.Discrete(2)}).contains(observation)

        return observation


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, options = {"argv": ["play","--no-gui","--my-agent","user_agent"]}):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.trajectory = []

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        one_array = np.ones(s.COLS * s.ROWS)
        self.observation_space = spaces.Dict(
            {
                "field": MultiDiscrete(nvec=one_array * ( 8 + s.EXPLOSION_TIMER + s.BOMB_TIMER), dtype = np.uint8),
                # 0: stone walls, 1: free tiles, 2: crates, 3: coins,
                # 4: no bomb opponents, 5: has bomb opponents,
                # 6: self
                # 7~7+s.EXPLOSION_TIMER: explosion map
                # 8+s.EXPLOSION_TIMER ~ 8+s.EXPLOSION_TIMER+ s.BOMB_TIMER: bomb map
            
                "bomb_possible": spaces.Discrete(2)
            }
        )

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
        self.my_agent = None
        self.dist_to_opponent = [(s.COLS+s.ROWS, s.COLS+s.ROWS) for _ in range(s.MAX_AGENTS-1)] # closest dist and last dist
        for a in self.world.agents:
            if a.name == "user_agent":
                self.my_agent = a
        assert isinstance(self.my_agent, agents.Agent)

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

    def manhattan_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = abs(x2 - x1) + abs(y2 - y1)
        return distance

    def step(self, action):
        self.world.do_step(ACTION_MAP[action])
        self.user_input = None

        terminated = False
        truncated = False

        # get observation
        death_reward = 0
        game_state = self.world.get_state_for_agent(self.my_agent)
        if game_state['self'[0]] == 'DEAD': # the agent is dead
            truncated = True
            death_reward = -0.5
        
        observation = fromStateToObservation(game_state)

        # terminated or trunctated
        if self.world.running == False:
            if self.world.step == s.MAX_STEPS:
                terminated = True
        
        current_pos = game_state["self"][3]
        
        def in_bomb_range(bomb_x,bomb_y,x,y):
            return ((bomb_x == x) and (abs(bomb_y - y) <= s.BOMB_POWER)) or \
                      ((bomb_y == y) and (abs(bomb_x - x) <= s.BOMB_POWER))
        
        # escape from explosion reward
        escape_bomb_reward = 0
        if len(self.trajectory) > 0:
            x, y = self.trajectory[-1] # last position
            x_now, y_now =current_pos
            # Add proposal to run away from any nearby bomb about to blow
            for (xb, yb), t in game_state['bombs']:
                # if now in bomb range
                if in_bomb_range(xb,yb,x_now,y_now):
                    escape_bomb_reward -= 0.000666

                # If agent is in a safe cell when there is a bomb nearby
                if self.manhattan_distance((xb,yb),(x_now,y_now)) < s.BOMB_POWER + 2 and not in_bomb_range(xb,yb,x_now,y_now):
                    escape_bomb_reward += 0.002
        
        
        index=0
        closer_to_opponent_reward = 0
        for agent in game_state['others']:
            current_dist = self.manhattan_distance(current_pos, agent[3])
            closest_dist, last_dist = self.dist_to_opponent[index]
            if current_dist < closest_dist:
                self.dist_to_opponent[index] = (current_dist, current_dist)
                if closest_dist != s.COLS+s.ROWS:
                    closer_to_opponent_reward += 0.1 # the agent is in the closest distance so far to an opponent
                closest_dist = current_dist

            if current_dist < last_dist:
                closer_to_opponent_reward += 0.002 # the agent is getting closer to the opponent
            elif current_dist > last_dist:
                closer_to_opponent_reward -= 0.002 # the agent is getting further from the opponent
            
            self.dist_to_opponent[index] = (closest_dist, current_dist)
            index += 1
                
        
        # Get reward
        # self.my_agent.last_game_state, self.my_agent.last_action, game_state, self.events
        reward = death_reward + escape_bomb_reward + closer_to_opponent_reward
        for event in self.my_agent.events:
            if event in [e.MOVED_LEFT, e.MOVED_RIGHT, e.MOVED_UP, e.MOVED_DOWN]:
                reward += 0
            elif event == e.WAITED:
                reward += 0
            elif event == e.INVALID_ACTION:
                reward -= 0
            elif event == e.BOMB_DROPPED:
                reward += 0
            elif event == e.BOMB_EXPLODED:
                reward += 0
            elif event == e.CRATE_DESTROYED:
                reward += 0.1
            elif event == e.COIN_FOUND:
                reward += 0.05
            elif event == e.COIN_COLLECTED:
                reward += 0.2
            elif event == e.KILLED_OPPONENT:
                reward += 1.0
            elif event == e.KILLED_SELF:
                reward -= 0.5
            elif event == e.GOT_KILLED:
                reward -= 0.5
            elif event == e.OPPONENT_ELIMINATED:
                reward -= 0
            elif event == e.SURVIVED_ROUND:
                reward += 1.0

        reward =-0.01 # penalty per iteration
        self.trajectory.append(current_pos)
        return observation, reward, terminated, truncated, {"events" : self.my_agent.events}


    def reset(self, seed = None):
        super().reset(seed=seed) # following documentation
        
        self.trajectory = []
        # start a new round
        self.world.new_round()

        # Get first observation
        game_state = self.world.get_state_for_agent(self.my_agent)
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