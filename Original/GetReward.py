import math
import numpy as np
import settings as s
from GetFeatures import in_bomb_range, manhattan_distance
import events as e
from RuleBasedAgent import RuleBasedAgent

ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']

class GetReward():
    def __init__(self, random_seed = 42, directly_rule_based = False):
        self.trajectory = []
        self.rule_based_model = RuleBasedAgent(random_seed)
        self.enable_rule_based_agent_reward = False
        self.last_game_state = None
        self.directly_rule_based = directly_rule_based # do not check the action with another rule based agent
        self.events = None

    def reset(self):
        self.trajectory = []
        self.last_game_state = None
        self.events = None

        # Reset rule based model
        self.rule_based_model.reset_self()
    
    # reward at "game_state" doing "action"
    # "game_state" is state after "action"
    # action here should be index
    def get_reward(self, game_state: dict, action: int, events = []):
        if self.last_game_state == None:
            self.last_game_state = game_state
            return 0

        if events == None:
            # In callbacks.py act(), get events from train.py
            events = self.events

        current_pos = game_state["self"][3]
    
        if self.directly_rule_based:
            reward = 100
            if ACTION_MAP[action] == "BOMB":
                reward += 20
            return reward
        
        if not self.enable_rule_based_agent_reward:
        #### start to calculate reward_goal
            a = math.log(2)
            b = 2**2
            # calculate non-explore punishment
            non_explore_punishment = 0
            
            for i in range(1, len(self.trajectory)):
                pos = self.trajectory[-i] 
                non_explore_punishment -= b* np.exp(-a *manhattan_distance(current_pos, pos)) * np.exp(-a*i)

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
                        if ((yb > y) and action ==  'UP' and field[x,y-1] == 0) or \
                            ((yb < y) and action == 'DOWN' and field[x,y+1] == 0):
                            escape_bomb_reward += 100
                        # Go towards bomb or wait
                        if ((yb > y) and action ==  'DOWN' and field[x,y+1] == 0) or \
                            ((yb < y) and action == 'UP' and field[x,y-1] == 0) or \
                            (action ==  'WAIT'):
                            escape_bomb_reward -= 100
                    if (yb == y) and (abs(xb - x) <= s.BOMB_POWER):
                        # Run away
                        if ((xb > x) and action == 'LEFT' and field[x-1,y] == 0) or \
                            ((xb < x) and action == 'RIGHT' and field[x+1,y] == 0):
                            escape_bomb_reward += 100
                        # Go towards bomb or wait
                        if ((xb > x) and action == 'RIGHT' and field[x+1,y] == 0) or \
                            ((xb < x) and action == 'LEFT' and field[x-1,y] == 0) or \
                            (action ==  'WAIT'):
                            escape_bomb_reward -= 100

                    # Try random direction if directly on top of a bomb
                    if xb == x and yb == y:
                        if (action == "UP" and field[x,y-1] == 0) or \
                            (action == "DOWN" and field[x,y+1] == 0) or \
                            (action == "LEFT" and field[x-1,y] == 0) or \
                            (action == "RIGHT" and field[x+1,y] == 0)    :
                            escape_bomb_reward += 50

                    # If last pos in bomb range and now not
                    if in_bomb_range(field,xb,yb,x,y) and not in_bomb_range(field,xb,yb,x_now,y_now):
                        escape_bomb_reward += 200

                    # if last pos not in bomb range and now yes
                    if in_bomb_range(field,xb,yb,x_now,y_now) and not in_bomb_range(field,xb,yb,x,y):
                        escape_bomb_reward -= 100    

            # Get game event reward
            game_event_reward = 0
            for event in events:
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
                    # case e.COIN_COLLECTED:
                    #     game_event_reward += 1000
                    # case e.KILLED_OPPONENT:
                    #     game_event_reward += 5000 
                    # case e.KILLED_SELF:
                    #     game_event_reward -= 2000 * (1- game_state["step"]/s.MAX_STEPS)
                    # case e.GOT_KILLED:
                    #     game_event_reward -= 1000 * (1- game_state["step"]/s.MAX_STEPS)
                    # case e.OPPONENT_ELIMINATED:
                    #     game_event_reward -= 10
                    # case e.SURVIVED_ROUND:
                        # game_event_reward += 500

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

            reward = back_forward_punishment  + game_event_reward 
            # lose meaningful_bomb_reward
            # + new_visit_reward + non_explore_punishment
            # + survive_reward

        if self.enable_rule_based_agent_reward: # enable rule_based_agent_reward
            reward = 0
            if ACTION_MAP[action] == "BOMB":
                self.rule_based_model.bomb_history.append(current_pos)
            
            target_action, valid_actions = self.rule_based_model.act(self.last_game_state)
            # print(target_action, valid_actions)
            
            if ACTION_MAP[action] == target_action:
                reward += 100 #1000
                if ACTION_MAP[action] == "BOMB":
                    reward += 10 # extra reward for dropping right bomb

            elif ACTION_MAP[action] != "WAIT" and ACTION_MAP[action]in valid_actions:
                reward += 1
            
            # # to prevent agent to back and forward
            # back_forward_punishment = 0
            # if len(self.trajectory) > 2:
            #     last_pos = self.trajectory[-1]
            #     wait_time = 0
            #     for pos in reversed(self.trajectory):
            #         if pos != last_pos:
            #             break
            #         else:
            #             wait_time += 1
            #     if pos == current_pos and ACTION_MAP[action] != target_action:
            #         reward -= 10 # back and forth punishment
            #     if ACTION_MAP[action] != "BOMB":
            #         reward -= wait_time * 5 # Waiting punishment
        
        # maintain self.trajectory
        self.trajectory.append(current_pos)
        self.last_game_state = game_state
        
        return reward
        