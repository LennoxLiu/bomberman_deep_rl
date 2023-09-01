import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, MultiDiscrete, Dict, Discrete
import settings as s
import events as e
import agents

import main
import math

ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

def fromStateToObservation(game_state):
        observation = {}
        one_array = np.ones(s.COLS* s.ROWS)

        game_field = game_state["field"].astype(np.uint8) 
        # 1 for crates, −1 for stone walls and 0 for free tiles

        # 0: stone walls, 1: free tiles, 2: crates
        # 3: coin
        # 4: other agent
        # 5: self
        observation["field_and_targets"] = game_field + 1

        for coin in game_state["coins"]:
            observation["field_and_targets"][coin] = 3 # coin
        for other in game_state["others"]:
            observation["field_and_targets"][other[3]] = 4
        observation["field_and_targets"][game_state["self"][3]] = 5
        observation["field_and_targets"] = observation["field_and_targets"].flatten()
        assert MultiDiscrete(nvec= one_array * 6, dtype = np.uint8).contains(observation["field_and_targets"])

        # 0: nothing
        # 1: bomb
        # 2-3: explosion still dangerous(as > 0 in explosion_map: 2,1)
        explosion_map = game_state["explosion_map"].astype(np.uint8)
        explosion_map[explosion_map > 0] += 1
        observation["bomb_and_explosion"] = explosion_map
        for bomb in game_state["bombs"]:
            observation["bomb_and_explosion"][bomb[0]] = 1
        observation["bomb_and_explosion"] = observation["bomb_and_explosion"].flatten()
        assert MultiDiscrete(nvec= one_array * 2 + s.EXPLOSION_TIMER, dtype = np.uint8).contains(observation["bomb_and_explosion"])

        observation["bomb_possible"] = int(game_state["self"][2])
        assert Discrete(2).contains(observation["bomb_possible"])
        
        spaces.Dict({"field_and_targets": MultiDiscrete(nvec= one_array * 6, dtype = np.uint8),"bomb_and_explosion": MultiDiscrete(nvec= one_array * 2 + s.EXPLOSION_TIMER, dtype = np.uint8),"bomb_possible": Discrete(2)}).contains(observation)
        return observation


def in_bomb_range(field,bomb_x,bomb_y,x,y):
            is_in_bomb_range_x = False
            is_in_bomb_range_y = False
            if (bomb_x == x) and (abs(bomb_y - y) <= s.BOMB_POWER):
                is_in_bomb_range_x = True
                for y_temp in range(min(y,bomb_y),max(y,bomb_y)):
                    if field[x,y_temp] == -1:
                        is_in_bomb_range_x = False
            
            if (bomb_y == y) and (abs(bomb_x - x) <= s.BOMB_POWER):
                is_in_bomb_range_y = True
                for x_temp in range(min(x,bomb_x),max(x,bomb_x)):
                    if field[x_temp,y] == -1:
                        is_in_bomb_range_y = False
            return is_in_bomb_range_x or is_in_bomb_range_y


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, options = {"argv": ["play","--no-gui","--my-agent","user_agent"]}):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.trajectory = []

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        one_array = np.ones(s.COLS* s.ROWS)
        # Do not pass "round", opponent score
        self.observation_space = spaces.Dict(
            {   
                "field_and_targets": MultiDiscrete(nvec= one_array * 6, dtype = np.uint8),
                # 0: stone walls, 1: free tiles, 2: crates
                # 3: coin
                # 4: other agent
                # 5: self
                
                "bomb_and_explosion": MultiDiscrete(nvec= one_array * 2 + s.EXPLOSION_TIMER, dtype = np.uint8),
                # 0: nothing
                # 1: bomb
                # 2-3: explosion still dangerous(as > 0 in explosion_map: 2,1)

                "bomb_possible": Discrete(2)
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
        self.world.user_input = None

        # store my agent
        self.deep_agent = None
        for a in self.world.agents:
            if a.name == "user_agent":
                self.deep_agent = a
        assert isinstance(self.deep_agent, agents.Agent)

        # start a new round
        self.world.new_round()


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
            last_action =  self.deep_agent.last_action

            # get observation
            game_state = self.world.get_state_for_agent(self.deep_agent)
            if game_state == None: # the agent is dead
                truncated = True
                observation = fromStateToObservation(self.deep_agent.last_game_state)
                game_state = self.deep_agent.last_game_state
            else:
                observation = fromStateToObservation(game_state)

                    # terminated or trunctated
            if self.world.running == False:
                if self.world.step == s.MAX_STEPS:
                    terminated = True
                
            a = math.log(2)
            b = 2**2
            # calculate non-explore punishment
            non_explore_punishment = 0
            current_pos = game_state["self"][3]
            for i in range(1, len(self.trajectory)):
                pos = self.trajectory[-i] 
                non_explore_punishment -= b* np.exp(-a *self.manhattan_distance(current_pos, pos)) * np.exp(-a*i)

            # new visit reward
            new_visit_reward = 0
            if current_pos not in self.trajectory:
                new_visit_reward = 50
            
            # escape from explosion reward
            escape_bomb_reward = 0
            field = game_state["field"]
            if len(self.trajectory) > 0:
                x, y = self.trajectory[-1] # last position
                x_now, y_now = current_pos
                # Add proposal to run away from any nearby bomb about to blow
                for (xb, yb), t in game_state['bombs']:
                    if (xb == x) and (abs(yb - y) <= s.BOMB_POWER):
                        # Run away
                        if ((yb > y) and last_action ==  'UP' and field[x,y+1] == 0) or \
                            ((yb < y) and last_action == 'DOWN' and field[x,y-1] == 0):
                            escape_bomb_reward += 100
                        # Go towards bomb or wait
                        if ((yb > y) and last_action ==  'DOWN' and field[x,y-1] == 0) or \
                            ((yb < y) and last_action == 'UP' and field[x,y+1] == 0) or \
                            (last_action ==  'WAIT'):
                            escape_bomb_reward -= 100
                    if (yb == y) and (abs(xb - x) <= s.BOMB_POWER):
                        # Run away
                        if ((xb > x) and last_action == 'LEFT' and field[x-1,y] == 0) or \
                            ((xb < x) and last_action == 'RIGHT' and field[x+1,y] == 0):
                            escape_bomb_reward += 100
                        # Go towards bomb or wait
                        if ((xb > x) and last_action == 'RIGHT' and field[x+1,y] == 0) or \
                            ((xb < x) and last_action == 'LEFT' and field[x-1,y] == 0) or \
                            (last_action ==  'WAIT'):
                            escape_bomb_reward -= 100

                    # Try random direction if directly on top of a bomb
                    if xb == x and yb == y:
                        if (last_action == "UP" and field[x,y+1] == 0) or \
                            (last_action == "DOWN" and field[x,y-1] == 0) or \
                            (last_action == "LEFT" and field[x-1,y] == 0) or \
                            (last_action == "RIGHT" and field[x+1,y] == 0)    :
                            escape_bomb_reward += 50

                    # If last pos in bomb range and now not
                    if in_bomb_range(field,xb,yb,x,y) and not in_bomb_range(field,xb,yb,x_now,y_now):
                        escape_bomb_reward += 200

                    # if last pos not in bomb range and now yes
                    if in_bomb_range(field,xb,yb,x_now,y_now) and not in_bomb_range(field,xb,yb,x,y):
                        escape_bomb_reward -= 100    

            # meaning full bomb position reward
            meaningfull_bomb_reward = 0
            x, y = current_pos
            if last_action == "BOMB":
                # if there's a agent in bomb range, reward ++
                for agent in self.world.active_agents:
                    if agent != self.deep_agent and \
                        in_bomb_range(field,x,y,agent.x,agent.y): 
                        meaningfull_bomb_reward += 100
                
                for x_temp in range(field.shape[0]):
                    for y_temp in range(field.shape[1]):
                        if field[x_temp,y_temp] == 1 and \
                            in_bomb_range(field,x,y,x_temp,y_temp): # it's a crate
                            meaningfull_bomb_reward += 50

            # Get game event reward
            # self.deep_agent.last_game_state, self.deep_agent.last_action, game_state, self.events
            game_event_reward = 0
            for event in self.deep_agent.events:
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
                        game_event_reward += 500
                    case e.COIN_FOUND:
                        game_event_reward += 100
                    case e.COIN_COLLECTED:
                        game_event_reward += 1000
                    case e.KILLED_OPPONENT:
                        game_event_reward += 5000
                    case e.KILLED_SELF:
                        game_event_reward -= 500
                    case e.GOT_KILLED:
                        game_event_reward -= 1500
                    case e.OPPONENT_ELIMINATED:
                        game_event_reward -= 10
                    case e.SURVIVED_ROUND:
                        game_event_reward += 500

            survive_reward = 50 * (game_state["step"]/s.MAX_STEPS) # considering invad operation punishment = 50
            
            # to prevent agent to back and forward
            back_forward_punishment = 0
            if len(self.trajectory) > 2:
                last_pos = self.trajectory[-1]
                wait_time = 0
                for pos in reversed(self.trajectory):
                    if pos != last_pos:
                        break
                    else:
                        wait_time += 1
                if pos == current_pos:
                    back_forward_punishment -= 50
                non_explore_punishment -= wait_time * 5

            reward = back_forward_punishment + survive_reward + game_event_reward + new_visit_reward + non_explore_punishment + meaningfull_bomb_reward
            
            # maintain self.trajectory
            self.trajectory.append(current_pos)
            
            return observation, reward, terminated, truncated, {"events" : self.deep_agent.events, "reward": reward}


    def reset(self, seed = None):
        super().reset(seed=seed) # following documentation
        
        self.trajectory = []
        # start a new round
        self.world.new_round()

        # Get first observation
        game_state = self.world.get_state_for_agent(self.deep_agent)
        observation = fromStateToObservation(game_state)

        return observation, {"info": "reset"}
    

    def render(self):
        pass

    def close(self):
        self.world.end()


if __name__ == "__main__":
    env = CustomEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)