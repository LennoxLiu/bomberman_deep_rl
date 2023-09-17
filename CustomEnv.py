import gymnasium as gym
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env
from gymnasium.spaces import Box, MultiDiscrete
import settings as s
import events as e
import agents

import main
import math
from RuleBasedAgent import RuleBasedAgent
from GetFeatures import GetFeatures, in_bomb_range
from GetFeatures import FEATURE_DIM,INF
ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']


def fromStateToObservation(get_feature_class: GetFeatures, game_state):
    features = get_feature_class.state_to_features(game_state)
    try:
        assert Box(low = 0,high = 1, shape=(FEATURE_DIM,), dtype = np.float16).contains(features)
    except:
        print(len(features))
        print(features)
    return features


def fromStateToObservation_old(game_state):
        one_array = np.ones(s.COLS * s.ROWS)
        
        # 0: ston walls, 1: free tiles, 2: crates, 
        observation = game_state["field"].copy().astype(np.uint8) + 1
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
        explosion_map = game_state["explosion_map"].copy().astype(np.uint8)
        # Replace elements in observation with corresponding elements+9 from explosion_map if explosion_map elements are non-zero
        observation[explosion_map != 0] = explosion_map[explosion_map != 0] + 9 # is it correct?
        
        # 10+s.EXPLOSION_TIMER~ 10+s.EXPLOSION_TIMER*2: explosion on coin
        for coin in game_state["coins"]:
            if explosion_map[coin] != 0:
                observation[coin] += s.EXPLOSION_TIMER + 1

        # 11+s.EXPLOSION_TIMER*2 ~ 11+s.EXPLOSION_TIMER*2+ s.BOMB_TIMER: bomb map
        for bomb in game_state["bombs"]:
            observation[bomb[0]] = 11 + s.EXPLOSION_TIMER*2 + bomb[1]
        
        observation = observation.flatten()
        assert Box(low = 0, high = 11 + s.EXPLOSION_TIMER*2 + s.BOMB_TIMER, shape=(s.COLS* s.ROWS,), dtype = np.uint8).contains(observation)

        return observation


class CustomEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    # metadata = {"render_modes": ["default"], "render_fps": 30}

    def __init__(self, options = {"argv": ["play","--no-gui","--my-agent","user_agent"],"enable_rule_based_agent_reward": False}):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        self.metadata = options
        self.trajectory = []
        self.rule_based_agent = RuleBasedAgent()
        self.get_feature_class = GetFeatures()

        self.action_space = spaces.Discrete(len(ACTION_MAP)) # UP, DOWN, LEFT, RIGHT, WAIT, BOMB
        
        # one_array = np.ones(s.COLS * s.ROWS)
        # self.observation_space = Box(low = 0, high = 11 + s.EXPLOSION_TIMER*2 + s.BOMB_TIMER, shape=(s.COLS* s.ROWS,), dtype = np.uint8)
                # 0: stone walls, 1: free tiles, 2: crates, 3: coins,
                # 4: no bomb opponents, 5: has bomb opponents,
                # 6: self without bomb, 7: self with bomb, 8: self with bomb on top
                # 9~9+s.EXPLOSION_TIMER: explosion map
                # 10+s.EXPLOSION_TIMER~ 10+s.EXPLOSION_TIMER*2: explosion on coin
                # 11+s.EXPLOSION_TIMER*2 ~ 11+s.EXPLOSION_TIMER*2+ s.BOMB_TIMER: bomb map
        self.observation_space = Box(low = 0,high = 1, shape=(FEATURE_DIM,), dtype = np.float16)

        # train the model using "user input"
        self.world, n_rounds, self.gui, self.every_step, \
            turn_based, self.make_video, update_interval = \
                main.my_main_parser(argv=options["argv"])

        # world_controller(world, args.n_rounds,
        #                  gui=gui, every_step=every_step, turn_based=args.turn_based,
        #                  make_video=args.make_video, update_interval=args.update_interval)
        
        self.world.user_input = None

        # store my agent
        self.deep_agent = None
        for a in self.world.agents:
            if a.name == "user_agent":
                self.deep_agent = a # python automatically copy by reference
        assert isinstance(self.deep_agent, agents.Agent)

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
        
        if self.world.running == True:
            self.world.do_step(user_input = ACTION_MAP[action])

        # get observation
        game_state = self.world.get_state_for_agent(self.deep_agent)
        if game_state == None: # the agent is dead
            truncated = True
            observation = fromStateToObservation(self.get_feature_class, self.deep_agent.last_game_state)
            
            # check if our agent wins
            deep_agent_win = 0
            for agent in self.world.agents:
                if agent.score < self.deep_agent.score:
                    deep_agent_win += 1
            
            reward = 0
            reward += self.deep_agent.last_game_state["self"][1] * 100
            reward += deep_agent_win * 500
            reward += self.deep_agent.last_game_state["step"]

            return observation, reward, terminated, truncated, {}
        
        else: # our agent is still alive
            observation = fromStateToObservation(self.get_feature_class, game_state)

        # terminated or trunctated
        if self.world.running == False:
            if self.world.step == s.MAX_STEPS:
                terminated = True
            else:
                truncated = True

        current_pos = game_state["self"][3]
        last_action =  self.deep_agent.last_action

        if not self.metadata["enable_rule_based_agent_reward"]:
        #### start to calculate reward_goal
            a = math.log(2)
            b = 2**2
            # calculate non-explore punishment
            non_explore_punishment = 0
            
            for i in range(1, len(self.trajectory)):
                pos = self.trajectory[-i] 
                non_explore_punishment -= b* np.exp(-a *self.manhattan_distance(current_pos, pos)) * np.exp(-a*i)

            # new visit reward
            new_visit_reward = 0
            if current_pos not in self.trajectory:
                new_visit_reward = 50
            
            # escape from explosion reward
            escape_bomb_reward = 0
            field = game_state["field"].copy()
            if len(self.trajectory) > 0:
                x, y = self.trajectory[-1] # last position
                x_now, y_now = current_pos
                # Add proposal to run away from any nearby bomb about to blow
                for (xb, yb), t in game_state['bombs']:
                    if (xb == x) and (abs(yb - y) <= s.BOMB_POWER):
                        # Run away
                        if ((yb > y) and last_action ==  'UP' and field[x,y-1] == 0) or \
                            ((yb < y) and last_action == 'DOWN' and field[x,y+1] == 0):
                            escape_bomb_reward += 100
                        # Go towards bomb or wait
                        if ((yb > y) and last_action ==  'DOWN' and field[x,y+1] == 0) or \
                            ((yb < y) and last_action == 'UP' and field[x,y-1] == 0) or \
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
                        if (last_action == "UP" and field[x,y-1] == 0) or \
                            (last_action == "DOWN" and field[x,y+1] == 0) or \
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
                # maintain bomb_history in rule_based_agent
                # Keep track of chosen action for cycle detection
                self.rule_based_agent.bomb_history.append((x, y))
                
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
                        game_event_reward -= 2000
                    case e.GOT_KILLED:
                        game_event_reward -= 1000
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

            reward = game_event_reward + meaningfull_bomb_reward + survive_reward
            # back_forward_punishment
            # + non_explore_punishment 
            #  + new_visit_reward 

        if self.metadata["enable_rule_based_agent_reward"]: # enable rule_based_agent_reward
            reward = 0
            
            target_action, valid_actions = self.rule_based_agent.act(self.deep_agent.last_game_state)
            # print(target_action, valid_actions)
            
            if ACTION_MAP[action] == target_action:
                reward += 100 #1000
                if ACTION_MAP[action] == "BOMB":
                    reward += 10 # extra reward for dropping right bomb
            elif ACTION_MAP[action] != "WAIT" and ACTION_MAP[action]in valid_actions:
                reward += 1
            
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
                if pos == current_pos and ACTION_MAP[action] != target_action:
                    reward -= 10 # back and forth punishment
                if ACTION_MAP[action] != "BOMB":
                    reward -= wait_time * 5 # Waiting punishment
        
        # maintain self.trajectory
        self.trajectory.append(current_pos)
        
        return observation, reward/10, terminated, truncated, {}


    def reset(self, seed = np.random.randint(0, 2**31 - 1)):
        super().reset(seed=seed) # following documentation
        
        self.trajectory = []
        self.get_feature_class.reset()

        # start a new round
        self.world.new_round()

        # Get first observation
        game_state = self.world.get_state_for_agent(self.deep_agent)
        observation = fromStateToObservation(self.get_feature_class, game_state)
        return observation, {"info": "reset"}
    

    def render(self):
        pass


    def close(self):
        self.world.end()


if __name__ == "__main__":
    env = CustomEnv()
    # It will check your custom environment and output additional warnings if needed
    check_env(env)
    # env_vec = CustomVecEnv([env for _ in range(4)])
    # check_env(env_vec)