import heapq
import numpy as np

class Node:
    def __init__(self, x, y, g=0, h=0, parent=None):
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent
    def __lt__(self, other):
        return self.f < other.f

class AStarPlanner:
    def __init__(self, grid_size=0.5, world_size=30.0):
        self.grid_size = grid_size
        self.cell_num = int(world_size / grid_size)

    def _to_grid(self, x, y):
        return (int(x / self.grid_size), int(y / self.grid_size))

    def _to_world(self, gx, gy):
        return (gx * self.grid_size + self.grid_size/2,
                gy * self.grid_size + self.grid_size/2)

    def _collision(self, x, y, obstacles):
        for ox, oy, r in obstacles:
            if np.hypot(x - ox, y - oy) < r + self.grid_size/2:
                return True
        return False

    def plan(self, start, goal, obstacles):
        start_g = self._to_grid(start[0], start[1])
        goal_g = self._to_grid(goal[0], goal[1])
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {start_g: 0}
        f_score = {start_g: self._heuristic(start_g, goal_g)}
        heapq.heappush(open_set, (f_score[start_g], start_g))

        while open_set:
            _, current = heapq.heappop(open_set)
            if current == goal_g:
                return self._reconstruct_path(came_from, current, start)
            closed_set.add(current)
            for neighbor in self._neighbors(current):
                if neighbor in closed_set:
                    continue
                # Check collision with obstacles
                wx, wy = self._to_world(neighbor[0], neighbor[1])
                if self._collision(wx, wy, obstacles):
                    continue
                tentative_g = g_score[current] + self._distance(current, neighbor)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal_g)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        # Fallback: direct line
        return [start, goal]

    def _neighbors(self, node):
        for dx in [-1,0,1]:
            for dy in [-1,0,1]:
                if dx==0 and dy==0: continue
                nx, ny = node[0]+dx, node[1]+dy
                if 0 <= nx < self.cell_num and 0 <= ny < self.cell_num:
                    yield (nx, ny)

    def _distance(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1]) * self.grid_size

    def _heuristic(self, a, b):
        return np.hypot(a[0]-b[0], a[1]-b[1]) * self.grid_size

    def _reconstruct_path(self, came_from, current, start):
        path = [self._to_world(current[0], current[1])]
        while current in came_from:
            current = came_from[current]
            path.append(self._to_world(current[0], current[1]))
        path.append(start)
        path.reverse()
        return path