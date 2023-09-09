# from collections import deque

# def shortest_path(grid, start, end):
#     def neighbors(node):
#         x, y = node
#         return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

#     visited = set()
#     queue = deque([(start, 0)])
#     rows, cols = len(grid), len(grid[0])

#     while queue:
#         node, distance = queue.popleft()

#         if node == end:
#             return distance

#         visited.add(node)

#         for neighbor in neighbors(node):
#             x, y = neighbor
#             if 0 <= x < rows and 0 <= y < cols and neighbor not in visited and grid[x][y] == 0:
#                 queue.append((neighbor, distance + 1))

#     return -1  # If no path is found

# # Example 2D grid (0 represents a valid tile, 1 represents an obstacle)
# grid = [
#     [0, 1, 0, 0, 0],
#     [0, 1, 0, 1, 0],
#     [0, 0, 0, 1, 0],
#     [1, 1, 1, 1, 0],
#     [0, 0, 0, 0, 0]
# ]

# # Define start and end points
# start = (0, 0)
# end = (4, 4)

# # Find the shortest path length
# shortest_length = shortest_path(grid, start, end)

# if shortest_length >= 0:
#     print(f"The shortest path length is: {shortest_length}")
# else:
#     print("No path found.")

# def find_indices_of_value(arr, value):
#     indices = []
#     for i, row in enumerate(arr):
#         for j, element in enumerate(row):
#             if element == value:
#                 indices.append((i, j))
#     return indices

# # Example 2D array
# two_d_array = [
#     [1, 0, 1],
#     [0, 1, 1],
#     [1, 1, 0]
# ]

# # Get indices of value 1
# indices_of_1 = find_indices_of_value(two_d_array, 1)

# print(f"The indices of value 1 are: {indices_of_1}")

from collections import deque
import settings as s

def in_bomb_range(field,bomb_x,bomb_y,x,y):
            is_in_bomb_range_x = False
            is_in_bomb_range_y = False
            if (bomb_x == x) and (abs(bomb_y - y) <= s.BOMB_POWER):
                is_in_bomb_range_x = True
                for y_temp in range(min(y,bomb_y),max(y,bomb_y)):
                    if field[x][y_temp] == -1:
                        is_in_bomb_range_x = False
            
            if (bomb_y == y) and (abs(bomb_x - x) <= s.BOMB_POWER):
                is_in_bomb_range_y = True
                for x_temp in range(min(x,bomb_x),max(x,bomb_x)):
                    if field[x_temp][y] == -1:
                        is_in_bomb_range_y = False
            return is_in_bomb_range_x or is_in_bomb_range_y


def bfs(grid, start, TIME):
    rows, cols = len(grid), len(grid[0])
    visited = [[False for _ in range(cols)] for _ in range(rows)]
    queue = deque([(start[0], start[1], 0)])

    def is_valid_move(grid, visited, x, y):
        rows, cols = len(grid), len(grid[0])
        return (0 <= x < rows) and (0 <= y < cols) and grid[x][y] == 0 and not visited[x][y]

    while queue:
        x, y, length = queue.popleft()

        if length <= TIME and not in_bomb_range(grid, start[0], start[1],x ,y):
            return True

        visited[x][y] = True

        if length < TIME:
            moves = [(0, 1), (0, -1), (1, 0), (-1, 0)]

            for dx, dy in moves:
                new_x, new_y = x + dx, y + dy

                if is_valid_move(grid, visited, new_x, new_y):
                    queue.append((new_x, new_y, length + 1))

    return False

# Example usage:
grid = [
    [0, 1, 0, 1, 1],
    [0, 1, 0, 0, 1],
    [0, 1, 1, 1, 0],
    [0, 0, 1, 1, 0]
]

start = (1, 2)
TIME = 2

if bfs(grid, start, TIME):
    print(f"There exists a valid endpoint within {TIME} steps and in bomb range")
else:
    print(f"No valid endpoint found within {TIME} steps and in bomb range")

