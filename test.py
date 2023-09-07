from collections import deque

def shortest_path(grid, start, end):
    def neighbors(node):
        x, y = node
        return [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]

    visited = set()
    queue = deque([(start, 0)])
    rows, cols = len(grid), len(grid[0])

    while queue:
        node, distance = queue.popleft()

        if node == end:
            return distance

        visited.add(node)

        for neighbor in neighbors(node):
            x, y = neighbor
            if 0 <= x < rows and 0 <= y < cols and neighbor not in visited and grid[x][y] == 0:
                queue.append((neighbor, distance + 1))

    return -1  # If no path is found

# Example 2D grid (0 represents a valid tile, 1 represents an obstacle)
grid = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0]
]

# Define start and end points
start = (0, 0)
end = (4, 4)

# Find the shortest path length
shortest_length = shortest_path(grid, start, end)

if shortest_length >= 0:
    print(f"The shortest path length is: {shortest_length}")
else:
    print("No path found.")

def find_indices_of_value(arr, value):
    indices = []
    for i, row in enumerate(arr):
        for j, element in enumerate(row):
            if element == value:
                indices.append((i, j))
    return indices

# Example 2D array
two_d_array = [
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 0]
]

# Get indices of value 1
indices_of_1 = find_indices_of_value(two_d_array, 1)

print(f"The indices of value 1 are: {indices_of_1}")
