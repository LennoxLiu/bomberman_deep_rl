from collections import deque
import settings as s
import numpy as np

class GetFeatures():
        
        def __init__(self):
            self.coins = []
        
        def manhattan_distance(self, point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            distance = int(abs(x2 - x1) + abs(y2 - y1))
            return distance

        def get_valid_actions(self, game_state):
            _, score, bombs_left, (x, y) = game_state['self']
            bomb_xys = [xy for (xy, t) in game_state['bombs']]
            arena = game_state['field']
            others = [xy for (n, s, b, xy) in game_state['others']]
            
             # Check which moves make sense at all
            directions = [(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
            valid_tiles= []
            valid_actions = np.zeros(6)

            for d in directions:
                if ((arena[d] == 0) and
                        (not d in others) and
                        (not d in bomb_xys)):
                    valid_tiles.append(d)
            
            # ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
            if (x, y - 1) in valid_tiles: valid_actions[0] = 1 # (0,0) is at upper-left
            if (x, y + 1) in valid_tiles: valid_actions[1] = 1
            if (x - 1, y) in valid_tiles: valid_actions[2] = 1
            if (x + 1, y) in valid_tiles: valid_actions[3] = 1
            if (x, y) in valid_tiles: valid_actions[4] = 1
            # Disallow the BOMB action if agent dropped a bomb in the same spot recently
            if (bombs_left > 0) : valid_actions[5] = 1

            return valid_actions
        
        # find the shortest path between two points(coinsider wall, crates)
        # only pass at 0 in grid
        # return the length of path, return np.inf if no path
        def find_shortest_path(self, grid, start, end):
            def neighbors(node):
                x, y = node
                return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

            visited = set()
            queue = deque([(start, 0)])
            rows, cols = len(grid), len(grid[0])
            
            if not (0 <= end[0] < rows) or not (0 <= end[1] < cols):
                return np.inf  # If end point is out of bounds or not a valid tile

            while queue:
                node, distance = queue.popleft()

                if node == end:
                    return distance

                visited.add(node)

                for neighbor in neighbors(node):
                    x, y = neighbor
                    if 0 <= x < rows and 0 <= y < cols and neighbor not in visited and grid[x][y] == 0:
                        queue.append((neighbor, distance + 1))

            return np.inf  # If no path is found


        def get_distances_directions(self, grid, current_pos, target_pos_list):
            x_now, y_now = current_pos
            target_distances = [self.find_shortest_path(grid, current_pos, pos) \
                                  for pos in target_pos_list]
            combined = list(zip(target_pos_list, target_distances))

             # Sort based on the values from the second list
            sorted_combined = sorted(combined, key=lambda x: x[1])

            # Extract the first list from the sorted pairs
            closest_targets = [pair[0] for pair in sorted_combined][:3] # only take the closest 3 targets
            
            while closest_targets < 3: # padding to 3 items
                closest_targets.append((np.inf,np.inf))

            target_distances_directions = np.zeros((4,3))
            target_distances_directions.fill(np.inf)

            # ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
            neighbours = [(x_now, y_now-1),(x_now, y_now+1), (x_now-1, y_now), (x_now+1, y_now)]
            
            for i in range(4):
                for j in range(3):
                    target_distances_directions[i,j] = self.find_shortest_path(grid,
                                                                             neighbours[i], closest_targets[j] )
            
            return target_distances_directions.flatten()
    

        def state_to_features(self, game_state: dict):
            features = []
            _, score, bombs_left, (x_now, y_now) = game_state['self']

            # maintain all recorded coins
            coins_now = [(int(x), int(y)) for x, y in game_state["coins"]]
            for coin in coins_now:
                 if coin not in self.coins:
                    self.coins.append(coin)

            # get crates cnt
            crates_cnt = int(sum(row.count(1) for row in game_state["field"]))
            # get undiscovered coins cnt
            TOTAL_COINS = s.SCENARIOS["classic"]["COIN_COUNT"]
            undiscovered_coins_cnt = TOTAL_COINS - len(self.coins)
            # add expected coins per crate
            expected_coins_per_crate = 0
            if crates_cnt > 0:
                expected_coins_per_crate = undiscovered_coins_cnt / crates_cnt
            features.append(expected_coins_per_crate)

            # add remaining points
            features.append(s.REWARD_COIN * (undiscovered_coins_cnt + len(game_state["coins"])) \
                                              + s.REWARD_KILL * len(game_state["others"]))

            # add valid actions(consider crates, wall, bombs)
            valid_actions = self.get_valid_actions(game_state)
            features.append(valid_actions)

            # add nearest 3 coin distances after action, consider wall and crates
            features.append(self.get_distances_directions(game_state['field'],
                                                           (x_now,y_now), game_state["coins"]))

            # get nearest 3 opponents
            grid = game_state['field']
            for bomb in game_state["bombs"]:
                grid[bomb[0]] = -1 # cannot pass through bomb
            grid[game_state['explosion_map'] > 0] = -1 # should not pass through explosion
            opponent_pos = [op[3] for op in game_state["others"]]
            # add sorted opponent distance, consider wall, crates, bombs, explosion
            features.append(self.get_distances_directions(grid,
                                                           (x_now,y_now), opponent_pos))
            
            # if place bomb at current position, hom many crates can be exploded
            bomb_crates_cnt = 0
            for i in range(1, s.BOMB_POWER + 1):
                if y_now + i < s.COLS and game_state["field"][x_now, y_now + i] == 1: # 1 for crates
                    bomb_crates_cnt += 1
                if y_now -i >= 0 and game_state["field"][x_now, y_now - i] == 1: # 1 for crates
                    bomb_crates_cnt += 1
                if x_now + i < s.ROWS and game_state["field"][x_now + i, y_now] == 1: # 1 for crates
                    bomb_crates_cnt += 1
                if x_now - i >= 0 and game_state["field"][x_now - i, y_now] == 1: # 1 for crates
                    bomb_crates_cnt += 1
            features.append(bomb_crates_cnt)

            # add expected revaled coins if placing bomb at current position
            features.append(bomb_crates_cnt * expected_coins_per_crate)

            # add whether to drop bomb, 0 means impossible to drop or will kill ourself
            def will_suicide():
                
            can_drop_bomb = 0
            if bombs_left and not will_suicide(game_state):
                can_drop_bomb = 1
            features.append(can_drop_bomb)

            # get nearest 3 crates
            def find_indices_of_value(arr, value):
                indices = []
                for i, row in enumerate(arr):
                    for j, element in enumerate(row):
                        if element == value:
                            indices.append((i, j))
                return indices

            crates_pos = find_indices_of_value(game_state["field"], 1)
            # add nearest 3 crates
            features.append(self.get_distances_directions(game_state["field"],
                                                           (x_now,y_now), crates_pos))
            

