import numpy as np
from shapely.geometry import Point, Polygon

class DWA:
    """
    Dynamic Window Approach for local obstacle avoidance.
    """
    def __init__(self, dt, max_v=1.0, min_v=-0.5, max_w=8.0, min_w=-8.0,
                 predict_time=1.5, resolution=0.5,
                 heading_gain=5.0, dist_gain=0.3, vel_gain=0.2,
                 safe_dist=1.5):
        self.dt = dt
        self.max_v = max_v
        self.min_v = min_v
        self.max_w = max_w
        self.min_w = min_w
        self.predict_time = predict_time
        self.resolution = resolution
        self.heading_gain = heading_gain
        self.dist_gain = dist_gain
        self.vel_gain = vel_gain
        self.safe_dist = safe_dist

    def plan(self, state, goal, obstacles, current_v, current_w):
        """
        state: (x, y, theta, v, w) current pose
        goal: (x, y) target position
        obstacles: list of (x, y, radius)
        current_v, current_w: current velocities (for dynamic window)
        returns: (v_best, w_best)
        """
        vs = np.arange(self.min_v, self.max_v + self.resolution, self.resolution)
        ws = np.arange(self.min_w, self.max_w + self.resolution, self.resolution)

        best_score = -np.inf
        best_v = current_v
        best_w = current_w

        for v in vs:
            for w in ws:
                traj = self._predict_trajectory(state, v, w)
                score = self._evaluate_trajectory(traj, goal, obstacles, v)
                if score > best_score:
                    best_score = score
                    best_v, best_w = v, w

        return best_v, best_w

    def _predict_trajectory(self, state, v, w):
        """Simulate trajectory for given (v, w) over predict_time seconds."""
        x, y, theta, _, _ = state
        dt = self.dt
        steps = int(self.predict_time / dt)
        traj = []
        for _ in range(steps):
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += w * dt
            traj.append((x, y))
        return traj

    def _evaluate_trajectory(self, traj, goal, obstacles, v):
        """
        Scoring function: 
        - heading cost: final orientation towards goal
        - obstacle cost: penalty for getting too close
        - velocity cost: prefer higher speed
        Returns a scalar score (higher is better).
        """
        if not traj:
            return -np.inf

        final_x, final_y = traj[-1]
        dx = goal[0] - final_x
        dy = goal[1] - final_y
        heading = np.hypot(dx, dy)
        heading_score = -heading

        min_obs_dist = float('inf')
        for (ox, oy, r) in obstacles:
            for (px, py) in traj:
                dist = np.hypot(px - ox, py - oy) - r
                if dist < min_obs_dist:
                    min_obs_dist = dist
        if min_obs_dist < self.safe_dist:
            obstacle_score = -np.inf if min_obs_dist < 0 else -10.0 * (self.safe_dist - min_obs_dist)
        else:
            obstacle_score = 0.0

        vel_score = v / self.max_v

        score = (self.heading_gain * heading_score +
                 self.dist_gain * obstacle_score +
                 self.vel_gain * vel_score)

        return score