from random import random
import string
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium import spaces
import settings as s
import events as e
import agents

import main
from fallbacks import pygame, LOADED_PYGAME
import math
from imitation.util.util import make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper

ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def fromStateToObservation(game_state):
        observation = None
        one_array = np.ones(s.COLS * s.ROWS*2)
        # 0: stone walls, 1: free tiles, 2: crates, 
        field = game_state["field"].astype(np.uint8) + 1
        #3: coins,
        for coin in game_state["coins"]:
            field[coin] = 3
        # 4: no bomb opponents, 5: has bomb opponents,
        for other in game_state["others"]:
            if other[2] == False: # bombs_left == False
                field[other[3]] = 4
            else:
                field[other[3]] = 5

        field[game_state['self'][3]] = 6 if game_state['self'][2] == False else 7

        # 0: nothing
        # 1~s.EXPLOSION_TIMER: explosion map
        explosion_field = game_state["explosion_map"].astype(np.uint8)
        # s.EXPLOSION_TIMER+1 ~ s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 1: explosion map + bomb timer 
        for bomb in game_state["bombs"]:
            explosion_field[bomb[0]] += bomb[1] + s.EXPLOSION_TIMER + 1 # overlay bomb on explosion
        
        # s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 2 opponents
        for other in game_state["others"]:
            explosion_field[other[3]] = s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 2
        
        # s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 3 self
        explosion_field[game_state['self'][3]] = s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 3
        
        # print("explosion_field in Env: \n",explosion_field)
        
        observation = np.stack((field, explosion_field))

        assert spaces.Box(low=0,high=max(8,s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4), shape=(2,s.ROWS,s.COLS), dtype = np.uint8).contains(observation)

        return observation


def fromObservationToState(observation):
    # Reconstruct the game state from the observation
    field = observation[0]
    explosion_field = observation[1]

    # Initialize game state components
    game_state = {
        'field': np.zeros((s.COLS, s.ROWS), dtype=np.int8),
        'bombs': [],
        'coins': [],
        'self': None,
        'others': [],
        'explosion_map': np.zeros((s.COLS, s.ROWS), dtype=np.int8)
    }

    # Decode the field
    for x in range(s.COLS):
        for y in range(s.ROWS):
            value = field[x, y]
            if value == 0:
                game_state['field'][x, y] = -1  # Stone walls
            elif value == 1:
                game_state['field'][x, y] = 0  # Free tile
            elif value == 2:
                game_state['field'][x, y] = 1  # Crates
            elif value == 3:
                game_state['coins'].append((x, y))  # Coins
            elif value == 4:
                game_state['others'].append(('rule_based_agent', 0, False, (x, y)))  # Opponent without bomb
            elif value == 5:
                game_state['others'].append(('rule_based_agent', 0, True, (x, y)))  # Opponent with bomb
            elif value == 6:
                game_state['self'] = ('user_agent', 0, False, (x, y))  # Self without bomb
            elif value == 7:
                game_state['self'] = ('user_agent', 0, True, (x, y))  # Self with bomb

    # Decode the explosion_field (bombs, explosions)
    for x in range(s.COLS):
        for y in range(s.ROWS):
            value = explosion_field[x, y]
            if value > 0 and value <= s.EXPLOSION_TIMER:
                game_state['explosion_map'][x, y] = value  # Explosion timer
            elif value > s.EXPLOSION_TIMER and value <= s.EXPLOSION_TIMER * 2 + s.BOMB_TIMER:
                game_state['bombs'].append(((x, y), value - s.EXPLOSION_TIMER -1))  # Bomb with timer
            elif value == s.EXPLOSION_TIMER * 2 + s.BOMB_TIMER + 2:
                # Already decoded as others
                pass
            elif value == s.EXPLOSION_TIMER * 2 + s.BOMB_TIMER + 3:
                # Already decoded as self
                pass

    return game_state


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, options = {"argv": ["play","--no-gui","--my-agent","user_agent","--train","1"]}):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        one_array = np.ones(s.COLS * s.ROWS * 2)
        self.observation_space = spaces.Box(low=0,high=max(8,s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 4), shape=(2,s.ROWS,s.COLS), dtype = np.uint8)
            # observation space consists of two parts
            # first part is the field without bomb
            # 0: stone walls, 1: free tiles, 2: crates, 3: coins,
            # 4: no bomb opponents, 5: has bomb opponents,
            # 6: self, 7: self with bomb

            # second part is the explosion_map with bomb
            # 0: nothing
            # 1~s.EXPLOSION_TIMER: explosion map
            # s.EXPLOSION_TIMER+1 ~ s.EXPLOSION_TIMER*2 + s.BOMB_TIMER + 1: explosion map + bomb timer 

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

        self.rng = np.random.default_rng()
        # start a new round
        self.world.new_round()


    def manhattan_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = abs(x2 - x1) + abs(y2 - y1)
        return distance

    def step(self, action):
        terminated = False
        truncated = False
        # terminated or trunctated
        if self.world.running == False:
            terminated = True
            if self.world.step == s.MAX_STEPS:
                truncated = True
            else:
                truncated = False
            # return np.zeros((2,s.ROWS, s.COLS),dtype=np.uint8), 0, terminated, truncated, {}
            # game_state = self.world.get_state_for_agent(self.my_agent)
        else:
            self.world.do_step(ACTION_MAP[action])
            self.user_input = None
        
        # get observation
        death_reward = 0
        game_state = self.world.get_state_for_agent(self.my_agent)
        if game_state['self'][0] == 'DEAD': # the agent is dead
            truncated = True
            # death_reward = -0.5 # deal with this later
        
        observation = fromStateToObservation(game_state)
        
        current_pos = game_state["self"][3]
        
        def in_bomb_range(bomb_x,bomb_y,x,y):
            return ((bomb_x == x) and (abs(bomb_y - y) <= s.BOMB_POWER)) or \
                      ((bomb_y == y) and (abs(bomb_x - x) <= s.BOMB_POWER))
        
        # escape from explosion reward
        escape_bomb_reward = 0
        x_now, y_now = current_pos
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
                reward += 0.05
            elif event == e.COIN_FOUND:
                reward += 0.05
            elif event == e.COIN_COLLECTED:
                reward += 0.2
            elif event == e.KILLED_OPPONENT:
                reward += 1.0
            elif event == e.KILLED_SELF:
                reward -= 0.25
            elif event == e.GOT_KILLED:
                reward -= 0.5
            elif event == e.OPPONENT_ELIMINATED:
                reward -= 0
            elif event == e.SURVIVED_ROUND:
                reward += 1.0

        reward -= 0.01 # penalty per iteration
        return observation, reward, terminated, truncated, game_state # output game_state as info


    def reset(self, **kwargs):
        seed = int(self.rng.integers(0, np.iinfo(np.int64).max))
        super().reset(seed=seed) # following documentation
        
        self.dist_to_opponent = [(s.COLS+s.ROWS, s.COLS+s.ROWS) for _ in range(s.MAX_AGENTS-1)] # closest dist and last dist
        
        # start a new round
        self.world.new_round()

        # Get first observation
        game_state = self.world.get_state_for_agent(self.my_agent)
        observation = fromStateToObservation(game_state)

        return observation, game_state # output game_state as info
    

    def render(self):
        if self.gui is not None:
            self.my_render(self.every_step)


    def close(self):
        other_scores = []
        user_agent_score = 0
        for a in self.world.agents:
            if a.name != "user_agent":
                other_scores.append(a.total_score)
            else: # user_agent
                user_agent_score = a.total_score
                agent_events = a.events

        self.world.end()

        for a in self.world.agents:
            # reset all agents
            a.reset()

        return user_agent_score > max(other_scores), other_scores, user_agent_score, agent_events # return True if user_agent wins


from gymnasium import register
import os

register(
    id='CustomEnv-v1',  # Unique identifier for the environment
    entry_point='CustomEnv:CustomEnv',  # Replace with the actual path to your CustomEnv class
)

# tmp_env = gym.make('CustomEnv-v1')

if __name__ == "__main__":
    env = CustomEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)

#     env = make_vec_env(
#     'CustomEnv-v1',
#     rng=np.random.default_rng(42),
#     n_envs=8,
#     post_wrappers=[lambda env, env_idx: RolloutInfoWrapper(env)],  # to compute rollouts
# )