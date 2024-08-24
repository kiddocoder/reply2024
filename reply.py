import numpy as np
from collections import defaultdict, deque
from itertools import combinations
import time

TILE_DIRECTIONS = {
    '3': [(0, 1), (0, -1)],
    '5': [(1, 1), (-1, -1)],
    '6': [(1, -1), (-1, 1)],
    '7': [(0, 1), (0, -1), (1, -1), (-1, 1), (1, 1), (-1, -1)],
    '9': [(-1, 1), (1, -1)],
    '96': [(1, -1), (-1, 1), (-1, 1), (1, -1)],
    'A': [(-1, -1), (1, 1)],
    'A5': [(-1, -1), (1, 1), (1, 1), (-1, -1)],
    'B': [(0, 1), (0, -1), (-1, -1), (1, 1), (-1, 1), (1, -1)],
    'C': [(1, 0), (-1, 0)],
    'C3': [(0, 1), (0, -1), (1, 0), (-1, 0)],
    'D': [(1, 0), (-1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)],
    'E': [(-1, -1), (1, 1), (1, -1), (-1, 1), (1, 0), (-1, 0)],
    'F': [(0, 1), (0, -1), (1, -1), (-1, 1), (-1, -1), (1, 1), (1, 0), (-1, 0), (-1, 1), (1, -1), (1, 1), (-1, -1)]
}

class ChickNorrisTV:
    def __init__(self, width, height, golden_points, silver_points, tiles):
        self.width = width
        self.height = height
        self.golden_points = set(map(tuple, golden_points))
        self.silver_points = {tuple(point[:2]): point[2] for point in silver_points}
        self.tiles = tiles
        self.grid = np.zeros((height, width), dtype='U2')
        self.best_score = float('-inf')
        self.best_solution = None
        self.start_time = time.time()
        self.time_limit = 300  # 5 minutes time limit

    def is_valid(self, x, y):
        return 0 <= x < self.height and 0 <= y < self.width

    def bfs(self, start, end):
        queue = deque([(start, [])])
        visited = set()

        while queue:
            (x, y), path = queue.popleft()

            if (x, y) == end:
                return path

            if (x, y) in visited:
                continue

            visited.add((x, y))

            tile_id = self.grid[x, y]
            if tile_id:
                for dx, dy in TILE_DIRECTIONS[tile_id]:
                    nx, ny = x + dx, y + dy
                    if self.is_valid(nx, ny) and (nx, ny) not in visited:
                        queue.append(((nx, ny), path + [(x, y)]))

        return None

    def calculate_score(self):
        total_score = 0
        tile_costs = defaultdict(int)
        visited_tiles = set()

        for start, end in combinations(self.golden_points, 2):
            path = self.bfs(start, end)
            if not path:
                return float('-inf')

            path_score = 0
            for x, y in path:
                tile_id = self.grid[x, y]
                if tile_id:
                    if (x, y) in visited_tiles:
                        tile_costs[tile_id] += self.tiles[tile_id][0]
                        if (x, y) in self.silver_points:
                            path_score += self.silver_points[(x, y)]
                    else:
                        tile_costs[tile_id] += self.tiles[tile_id][0]
                        visited_tiles.add((x, y))
                        if (x, y) in self.silver_points:
                            path_score += self.silver_points[(x, y)]

            total_score += path_score

        return total_score - sum(tile_costs.values())

    def place_tile(self, x, y, tile_id):
        if (x, y) in self.golden_points or (x, y) in self.silver_points:
            return False
        if self.grid[x, y] != '':
            return False
        if self.tiles[tile_id][1] <= 0:
            return False
        self.grid[x, y] = tile_id
        self.tiles[tile_id] = (self.tiles[tile_id][0], self.tiles[tile_id][1] - 1)
        return True

    def remove_tile(self, x, y):
        tile_id = self.grid[x, y]
        if tile_id:
            self.grid[x, y] = ''
            self.tiles[tile_id] = (self.tiles[tile_id][0], self.tiles[tile_id][1] + 1)

    def solve(self):
        def backtrack(x, y):
            if time.time() - self.start_time > self.time_limit:
                return

            if x == self.height:
                score = self.calculate_score()
                if score > self.best_score:
                    self.best_score = score
                    self.best_solution = self.grid.copy()
                return

            if y == self.width:
                backtrack(x + 1, 0)
                return

            backtrack(x, y + 1)

            for tile_id in self.tiles:
                if self.place_tile(x, y, tile_id):
                    backtrack(x, y + 1)
                    self.remove_tile(x, y)

        backtrack(0, 0)
        return self.best_solution, self.best_score

def parse_input(filename):
    with open(filename, 'r') as f:
        w, h, gn, sm, tl = map(int, f.readline().split())
        golden_points = [tuple(map(int, f.readline().split())) for _ in range(gn)]
        silver_points = [tuple(map(int, f.readline().split())) for _ in range(sm)]
        tiles = {}
        for _ in range(tl):
            tile_id, cost, count = f.readline().split()
            tiles[tile_id] = (int(cost), int(count))
    return w, h, golden_points, silver_points, tiles

def write_output(filename, solution, score):
    with open(filename, 'w') as f:
        for x in range(solution.shape[0]):
            for y in range(solution.shape[1]):
                if solution[x, y]:
                    f.write(f"{solution[x, y]} {y} {x}\n")
        f.write(f"# Score: {score}\n")

if __name__ == "__main__":
    w, h, golden_points, silver_points, tiles = parse_input("input.txt")
    solver = ChickNorrisTV(w, h, golden_points, silver_points, tiles)
    solution, score = solver.solve()
    write_output("output.txt", solution, score)
    print(f"Best score: {score}")
