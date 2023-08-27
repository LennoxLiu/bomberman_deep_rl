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
        one_array = np.ones(s.COLS * s.ROWS)
        
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

        if game_state["self"][2] == False:
            observation[game_state["self"][3]] = 6 # 6: self without bomb
        else:
            observation[game_state["self"][3]] = 7 # 7: self with bomb
        for bomb in game_state["bombs"]:
            if bomb[0] == game_state["self"][3]:
                observation[game_state["self"][3]] = 8 #8: self with bomb on top

        # 9~9+s.EXPLOSION_TIMER: explosion map
        explosion_map = game_state["explosion_map"].astype(np.uint8)
        # Replace elements in observation with corresponding elements+9 from explosion_map if explosion_map elements are non-zero
        observation[explosion_map != 0] = explosion_map[explosion_map != 0] + 9
        
        # 10+s.EXPLOSION_TIMER~ 10+s.EXPLOSION_TIMER*2: explosion on coin
        for coin in game_state["coins"]:
            if explosion_map[coin] != 0:
                observation[coin] += s.EXPLOSION_TIMER + 1

        # 11+s.EXPLOSION_TIMER*2 ~ 11+s.EXPLOSION_TIMER*2+ s.BOMB_TIMER: bomb map
        for bomb in game_state["bombs"]:
            observation[bomb[0]] = 11 + s.EXPLOSION_TIMER*2 + bomb[1]

        observation = observation.flatten()
        
        assert MultiDiscrete(nvec=one_array * ( 11 + s.EXPLOSION_TIMER*2 + s.BOMB_TIMER), dtype = np.uint8).contains(observation)

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
        self.observation_space = MultiDiscrete(nvec=one_array * ( 11 + s.EXPLOSION_TIMER*2 + s.BOMB_TIMER), dtype = np.uint8)
                # 0: stone walls, 1: free tiles, 2: crates, 3: coins,
                # 4: no bomb opponents, 5: has bomb opponents,
                # 6: self without bomb, 7: self with bomb, 8: self with bomb on top
                # 9~9+s.EXPLOSION_TIMER: explosion map
                # 10+s.EXPLOSION_TIMER~ 10+s.EXPLOSION_TIMER*2: explosion on coin
                # 11+s.EXPLOSION_TIMER*2 ~ 11+s.EXPLOSION_TIMER*2+ s.BOMB_TIMER: bomb map

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
        game_state = self.world.get_state_for_agent(self.PPO_agent)
        if game_state == None: # the agent is dead
            truncated = True
            observation = fromStateToObservation(self.PPO_agent.last_game_state)
            game_state = self.PPO_agent.last_game_state
        else:
            observation = fromStateToObservation(game_state)

                # terminated or trunctated
        if self.world.running == False:
            if self.world.step == s.MAX_STEPS:
                terminated = True
            
        a = math.log(2)
        b = 5**2
        # calculate non-explore punishment
        non_explore_punishment = 0
        current_pos = game_state["self"][3]
        for i in range(min(len(self.trajectory), 15)): # only calculate the recent 5 pos
            pos = self.trajectory[-i]
            non_explore_punishment += b* np.exp(-a *self.manhattan_distance(current_pos, pos)) * np.exp(-a*i)

        # new visit reward
        new_visit_reward = 0
        if current_pos not in self.trajectory:
            new_visit_reward = 10
        
        # escape from explosion reward
        escape_bomb_reward = 0
        def in_bomb_range(bomb_x,bomb_y,x,y):
            return ((bomb_x == x) and (abs(bomb_y - y) <= s.BOMB_POWER)) or \
                      ((bomb_y == y) and (abs(bomb_x - x) <= s.BOMB_POWER))
        
        
        meaningfull_bomb_reward = 0
        if len(self.trajectory) > 0:
            x, y = self.trajectory[-1] # last position
            x_now, y_now =current_pos
            # Add proposal to run away from any nearby bomb about to blow
            for (xb, yb), t in game_state['bombs']:
                if (xb == x) and (abs(yb - y) <= s.BOMB_POWER):
                    # Run away
                    if ((yb > y) and ACTION_MAP[action] ==  'UP') or \
                        ((yb < y) and ACTION_MAP[action] == 'DOWN'):
                        escape_bomb_reward += 20
                    # Go towards bomb or wait
                    if ((yb > y) and ACTION_MAP[action] ==  'DOWN') or \
                        ((yb < y) and ACTION_MAP[action] == 'UP') or \
                        (ACTION_MAP[action] ==  'WAIT'):
                        escape_bomb_reward -= 20
                if (yb == y) and (abs(xb - x) <= s.BOMB_POWER):
                    # Run away
                    if ((xb > x) and ACTION_MAP[action] == 'LEFT') or \
                        ((xb < x) and ACTION_MAP[action] == 'RIGHT'):
                        escape_bomb_reward += 20
                    # Go towards bomb or wait
                    if ((xb > x) and ACTION_MAP[action] == 'RIGHT') or \
                        ((xb < x) and ACTION_MAP[action] == 'LEFT') or \
                        (ACTION_MAP[action] ==  'WAIT'):
                        escape_bomb_reward -= 20

                # Try random direction if directly on top of a bomb
                if xb == x and yb == y and ACTION_MAP[action] != "WAIT" \
                    and ACTION_MAP[action] != "BOMB":
                    escape_bomb_reward += 10

                # If last pos in bomb range and now not
                if in_bomb_range(xb,yb,x,y) and not in_bomb_range(xb,yb,x_now,y_now):
                    escape_bomb_reward += 30    

            # meaningfull bomb reward
            if ACTION_MAP[action] == "BOMB":
                # if there's a agent in bomb range, reward ++
                for agent in self.world.active_agents:
                    if agent != self.PPO_agent and \
                        in_bomb_range(x,y,agent.x,agent.y): 
                        meaningfull_bomb_reward += 100
                
                field = game_state["field"]
                for x_temp in range(field.shape[0]):
                    for y_temp in range(field.shape[1]):
                        if field[x_temp,y_temp] == 1 and \
                            in_bomb_range(x,y,x_temp,y_temp): # it's a crate
                            meaningfull_bomb_reward += 50

        self.trajectory.append(current_pos)

        # Get game event reward
        # self.PPO_agent.last_game_state, self.PPO_agent.last_action, game_state, self.events
        game_event_reward = 0
        for event in self.PPO_agent.events:
            match(event):
                case e.MOVED_LEFT | e.MOVED_RIGHT | e.MOVED_UP | e.MOVED_DOWN:
                    game_event_reward += 5
                case e.WAITED:
                    game_event_reward += 1
                case e.INVALID_ACTION:
                    game_event_reward -= 50
                case e.BOMB_DROPPED:
                    game_event_reward += 50
                case e.BOMB_EXPLODED:
                    game_event_reward += 0
                case e.CRATE_DESTROYED:
                    game_event_reward += 50
                case e.COIN_FOUND:
                    game_event_reward += 50
                case e.COIN_COLLECTED:
                    game_event_reward += 1000
                case e.KILLED_OPPONENT:
                    game_event_reward += 5000
                case e.KILLED_SELF:
                    game_event_reward -= 200 * (1- game_state["step"]/s.MAX_STEPS) + 50# decrease the got killed punishment when exploring
                case e.GOT_KILLED:
                    game_event_reward -= 300 * (1- game_state["step"]/s.MAX_STEPS) + 50
                case e.OPPONENT_ELIMINATED:
                    game_event_reward -= 10
                case e.SURVIVED_ROUND:
                    game_event_reward += 500

        survive_reward = 0.25* game_state["step"] # considering invad operation punishment = 50
        reward = survive_reward + game_event_reward + new_visit_reward - non_explore_punishment + meaningfull_bomb_reward
        
        return observation, reward, terminated, truncated, {"events" : self.PPO_agent.events}


    def reset(self, seed = None):
        super().reset(seed=seed) # following documentation
        
        self.trajectory = []
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