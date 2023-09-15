from collections import deque
import settings as s
import numpy as np

INF = s.COLS + s.ROWS
FEATURE_DIM = 34
TOTAL_COINS = s.SCENARIOS["classic"]["COIN_COUNT"]

def get_blast_coords(arena, bomb_x, bomb_y):
        x = bomb_x
        y = bomb_y
        blast_coords = [(x, y)]

        for i in range(1, s.BOMB_POWER + 1):
            if arena[x + i, y] == -1:
                break
            blast_coords.append((x + i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x - i, y] == -1:
                break
            blast_coords.append((x - i, y))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x, y + i] == -1:
                break
            blast_coords.append((x, y + i))
        for i in range(1, s.BOMB_POWER + 1):
            if arena[x, y - i] == -1:
                break
            blast_coords.append((x, y - i))

        return [(int(item[0]),int(item[1])) for item in blast_coords]


def in_bomb_range(field,bomb_x,bomb_y,x,y):
    blast_coords = get_blast_coords(field,bomb_x,bomb_y)
    return (int(x),int(y)) in blast_coords


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
            arena = game_state['field'].copy()
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
            if (bombs_left > 0) : valid_actions[5] = 1

            return valid_actions
        
        # find the shortest path between two points(coinsider wall, crates)
        # only pass at 0 in grid
        # return the length of path when length < INF, return INF if no path
        def find_shortest_path(self, grid, start, end):
            def neighbors(node):
                x, y = node
                return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

            visited = set()
            queue = deque([(start, 0)])
            rows, cols = len(grid), len(grid[0])
            
            if not (0 <= end[0] < rows) or not (0 <= end[1] < cols):
                return INF  # If end point is out of bounds or not a valid tile

            while queue:
                node, distance = queue.popleft()

                if node == end:
                    return distance

                visited.add(node)

                for neighbor in neighbors(node):
                    x, y = neighbor
                    if 0 <= x < rows and 0 <= y < cols \
                        and distance + 1 <= INF \
                        and neighbor not in visited \
                        and grid[x][y] == 0:
                        queue.append((neighbor, distance + 1))

            return INF  # If no path is found


        def get_distances_directions(self, grid, current_pos, target_pos_list, size = 3):
            x_now, y_now = current_pos
            
            # to save time, use manhattan distance for large number of targets
            target_distances = [self.manhattan_distance(current_pos, pos) \
                                  for pos in target_pos_list]
            
            combined = list(zip(target_pos_list, target_distances))
            
            # ACTION_MAP = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB']
            neighbours = [(x_now, y_now-1),(x_now, y_now+1), (x_now-1, y_now), (x_now+1, y_now), (x_now, y_now)]
            
            # Sort based on the values from the second list
            sorted_combined = sorted(combined, key=lambda x: x[1])

            # Extract the first list from the sorted pairs
            closest_targets = [pair[0] for pair in sorted_combined][:size] # only take the closest 3 targets
            
            while len(closest_targets) < size: # padding to 3 items
                closest_targets.append((INF,INF))

            target_distances_directions = np.zeros((len(neighbours),size))
            target_distances_directions.fill(INF)
            
            for i in range(len(neighbours)):
                for j in range(size):
                    target_distances_directions[i,j] = self.find_shortest_path(grid,
                                                                             neighbours[i], closest_targets[j] )
            
            target_distances_directions /= INF # scale to [0,1]
            return target_distances_directions.flatten()

        # return true if there exist a path that can escape from current bomb
        def can_escape(self, grid, start):
            rows, cols = len(grid), len(grid[0])
            visited = [[False for _ in range(cols)] for _ in range(rows)]
            queue = deque([(start[0], start[1], 0)])

            while queue:
                x, y, length = queue.popleft()

                if length <= s.BOMB_TIMER and not in_bomb_range(grid,start[0], start[1],x ,y):
                    return True

                visited[x][y] = True

                if length < s.BOMB_TIMER:
                    moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

                    for dx, dy in moves:
                        new_x, new_y = x + dx, y + dy

                        if (0 <= x < rows) and (0 <= y < cols) \
                            and grid[x][y] == 0 and not visited[x][y]:
                            queue.append((new_x, new_y, length + 1))

            return False


        def state_to_features(self, game_state: dict):
            features = []
            _, score, bombs_left, (x_now, y_now) = game_state['self']
            neighbours = [(x_now, y_now-1),(x_now, y_now+1), (x_now-1, y_now), (x_now+1, y_now), (x_now, y_now)]

            # maintain all recorded coins
            coins_now = [(int(x), int(y)) for x, y in game_state["coins"]]
            for coin in coins_now:
                 if coin not in self.coins:
                    self.coins.append(coin)

            # get crates cnt
            crates_cnt = len(game_state["field"] == 1) # count number of crates
            # get undiscovered coins cnt
            undiscovered_coins_cnt = TOTAL_COINS - len(self.coins)
            # add expected coins per crate
            expected_coins_per_crate = 0
            if crates_cnt > 0:
                expected_coins_per_crate = undiscovered_coins_cnt / crates_cnt
            if expected_coins_per_crate > 1:
                expected_coins_per_crate = 1
            features.append(expected_coins_per_crate)

            # add remaining points
            # features.append(s.REWARD_COIN * (undiscovered_coins_cnt + len(game_state["coins"])) \
            #                                   + s.REWARD_KILL * len(game_state["others"]))
            
            # add valid actions(consider crates, wall, bombs)
            valid_actions = self.get_valid_actions(game_state)
            features.append(valid_actions) # dim = 5

            # add nearest 1 coin distances after action, consider wall and crates
            features.append(self.get_distances_directions(game_state['field'],
                                                           (x_now,y_now), game_state["coins"],1))

            # get nearest 1 opponent, consider wall and crates
            opponent_pos = [op[3] for op in game_state["others"]]
            features.append(self.get_distances_directions(game_state['field'],
                                                           (x_now,y_now), opponent_pos,1))
            
            # if place bomb at current position, hom many crates can be exploded
            def get_crates_cnt(grid, x_now,y_now):
                bomb_crates_cnt = 0
                for i in range(1, s.BOMB_POWER + 1):
                    if y_now + i < s.COLS and grid[x_now, y_now + i] == 1: # 1 for crates
                        bomb_crates_cnt += 1
                    if y_now -i >= 0 and grid[x_now, y_now - i] == 1: # 1 for crates
                        bomb_crates_cnt += 1
                    if x_now + i < s.ROWS and grid[x_now + i, y_now] == 1: # 1 for crates
                        bomb_crates_cnt += 1
                    if x_now - i >= 0 and grid[x_now - i, y_now] == 1: # 1 for crates
                        bomb_crates_cnt += 1
                return bomb_crates_cnt
            
            crates_directions = []
            grid = game_state["field"].copy()
            for n in neighbours:
                crates_directions.append(get_crates_cnt(grid,n[0],n[1]))
            # add crates with directions
            features.append(np.array(crates_directions) / (s.BOMB_POWER* 4) ) # dim = 4, scale to [0,1]

            # add whether to drop bomb, 0 means impossible to drop or will kill ourself
            
            can_drop_bomb = 0
            grid = game_state["field"].copy()
            grid[grid == 1] = -1 # use 1 to represent opponents
            for op in game_state["others"]:
                grid[op[3]] = 1 # fake oppoenent as crate
            if bombs_left and self.can_escape(game_state["field"],(x_now,y_now)):
                can_drop_bomb = 1
            
            drop_bomb_score = 0
            drop_bomb_score += get_crates_cnt(game_state["field"], x_now, y_now) / (s.BOMB_POWER* 4) * 0.167 # scale to 0.5
            drop_bomb_score += get_crates_cnt(grid, x_now, y_now) / 3 * 0.833 # scale to 0.8 for opponents
            
            features.append(can_drop_bomb)
            features.append(drop_bomb_score) # dim = 1

            # get nearest 1 crates
            def find_indices_of_value(arr, value):
                indices = []
                for i, row in enumerate(arr):
                    for j, element in enumerate(row):
                        if element == value:
                            indices.append((i, j))
                return indices

            crates_pos = find_indices_of_value(game_state["field"], 1)
            # add nearest 1 crates
            features.append(self.get_distances_directions(game_state["field"],
                                                           (x_now,y_now), crates_pos, 1)) # dim = 4*n
            
            # add directions to escape from bombs
            # consider multiple bombs, check if in_bomb_range
            escape = np.zeros(len(neighbours))
            # first unite explosion_map and bomb, 
            # then find path to escape and return shortest length of path at each direction
            # or return one-hot coding for shortest
            
            # 1. get grid for different time step
            grid_list = []
            field_0 = game_state["field"].copy()
            explosion_map_0 = game_state["explosion_map"].copy()
            
            for i in range(s.BOMB_TIMER + 1):
                field = field_0.copy()

                for bomb in game_state["bombs"]:
                    if bomb[1] - i > 0:
                        field[bomb[0]] = 1 # cannot pass through bomb (0 is the only valid path)
                    
                # add step 0 explosion
                explosion_map = explosion_map_0.copy()
                explosion_map[explosion_map > 0] -= i # go i time step

                # add future explosion
                for bomb in game_state["bombs"]:
                    blast_coords = []
                    if bomb[1] - i <= 0:
                        blast_coords = get_blast_coords(field_0, bomb[0][0], bomb[0][1])
                    for (x, y) in blast_coords:
                        explosion_map[x, y] = max(explosion_map[x, y], (bomb[1] - i + s.EXPLOSION_TIMER) - 1) # the origianl code is exp.timer - 1, so here is a -1

                grid_list.append(field + explosion_map)
            
            for i in range(len(neighbours)):
                safe_cnt = 0
                for j in range(len(grid_list)):
                    if grid_list[j][neighbours[i]] <= 0: # safe
                        safe_cnt += 1
                    else:
                        break
                escape[i] = safe_cnt / len(grid_list) # scale to 1
            
            features.append(escape)


            # flatten features and convert it to np.array
            def flatten_list(lst):
                flattened_list = []
                for item in lst:
                    if isinstance(item, list):
                        flattened_list.extend(flatten_list(item))
                    elif isinstance(item, np.ndarray):
                        flattened_list.extend(item.flatten())
                    else:
                        flattened_list.append(item)
                return flattened_list
            
            return np.array(flatten_list(features), dtype = np.float16)
    
        def reset(self):
            self.coins = []
