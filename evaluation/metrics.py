"""
Evaluation metrics for multi-agent formation control.
Usage:
    from evaluation.metrics import MetricsCollector
    collector = MetricsCollector()
    ... during simulation, call collector.update(...) each step
    results = collector.compute_final_metrics()
"""
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

class MetricsCollector:
    """
    Collects and computes various performance metrics for multi-agent formation.
    """
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.reset()

    def reset(self):
        """Clear all recorded data."""
        # Timestamps
        self.step_times = []          # step index (or simulation time)
        self.done_flags = []          # whether episode ended at each step

        # Per-agent states
        self.positions = [[] for _ in range(self.num_agents)]      # list of (x,y)
        self.headings = [[] for _ in range(self.num_agents)]       # list of theta (rad)
        self.velocities = [[] for _ in range(self.num_agents)]     # list of v (m/s)
        self.angular_vels = [[] for _ in range(self.num_agents)]   # list of w (rad/s)

        # Collision and safety info
        self.collisions = [[] for _ in range(self.num_agents)]     # bool per step
        self.safety_signals = [[] for _ in range(self.num_agents)] # 0,1,2 per step

        # Goal reaching
        self.goal_reached = [False] * self.num_agents
        self.goal_reached_time = [None] * self.num_agents          # step index when reached

        # Formation errors (if desired positions are known)
        self.formation_errors = []      # list of scalar formation error per step

        # Additional control inputs (if available)
        self.control_v = [[] for _ in range(self.num_agents)]      # commanded v
        self.control_w = [[] for _ in range(self.num_agents)]      # commanded w

        # Episode metadata
        self.episode_duration_steps = 0
        self.episode_start_time = None
        self.episode_end_time = None

    def update(self, step_idx: int, states: List[Dict], info: Dict):
        """
        Update metrics with current step data.
        Args:
            step_idx: current step number (0-indexed).
            states: list of dicts, each contains keys: 'x','y','theta','v','w'
            info: dict from env.step(), may contain 'desired_positions', 'collision', 'safety', etc.
        """
        # Defensive check: ensure info is a dictionary, otherwise treat as empty
        if not isinstance(info, dict):
            info = {}

        # Append step
        self.step_times.append(step_idx)

        # Per-agent data
        for i in range(self.num_agents):
            s = states[i]
            self.positions[i].append((s['x'], s['y']))
            self.headings[i].append(s['theta'])
            self.velocities[i].append(s['v'])
            self.angular_vels[i].append(s['w'])

            # Collision and safety from info
            if 'collision' in info and i < len(info['collision']):
                self.collisions[i].append(info['collision'][i])
            else:
                self.collisions[i].append(False)

            if 'safety' in info and i < len(info['safety']):
                self.safety_signals[i].append(info['safety'][i])
            else:
                self.safety_signals[i].append(0)

            # Control inputs from info if available
            if 'cmd_v' in info and i < len(info['cmd_v']):
                self.control_v[i].append(info['cmd_v'][i])
            if 'cmd_w' in info and i < len(info['cmd_w']):
                self.control_w[i].append(info['cmd_w'][i])

        # Formation error (if provided in info)
        if 'formation_error' in info:
            self.formation_errors.append(info['formation_error'])

        # Check goal reaching (assume goal position known, maybe from info)
        if 'goal_position' in info and 'goal_radius' in info:
            goal = info['goal_position']
            radius = info['goal_radius']
            for i in range(self.num_agents):
                if not self.goal_reached[i]:
                    pos = (states[i]['x'], states[i]['y'])
                    dist = np.hypot(pos[0]-goal[0], pos[1]-goal[1])
                    if dist < radius:
                        self.goal_reached[i] = True
                        self.goal_reached_time[i] = step_idx

    def compute_final_metrics(self) -> Dict:
        """Compute all metrics after episode ends."""
        metrics = {}

        # ----- 1. Goal achievement -----
        success = all(self.goal_reached)
        metrics['success'] = success
        if success:
            # Time to goal: maximum time among agents (when all reached)
            metrics['goal_reached_time_steps'] = max(t for t in self.goal_reached_time if t is not None)
            # Mean time per agent
            metrics['mean_goal_time_steps'] = np.mean([t for t in self.goal_reached_time if t is not None])
        else:
            metrics['goal_reached_time_steps'] = None
            metrics['mean_goal_time_steps'] = None

        # ----- 2. Trajectory errors (relative to ideal path or desired formation) -----
        final_positions = [self.positions[i][-1] if self.positions[i] else (0,0) for i in range(self.num_agents)]
        goal_pos = (25.0, 25.0)  # should be retrieved from environment; for now hardcoded
        final_distances = [np.hypot(p[0]-goal_pos[0], p[1]-goal_pos[1]) for p in final_positions]
        metrics['final_distance_to_goal_mean'] = np.mean(final_distances)
        metrics['final_distance_to_goal_std'] = np.std(final_distances)

        start_positions = [self.positions[i][0] if self.positions[i] else (2,2) for i in range(self.num_agents)]
        traj_rmse = []
        for i in range(self.num_agents):
            if len(self.positions[i]) < 2:
                traj_rmse.append(0.0)
                continue
            p0 = np.array(start_positions[i])
            p1 = np.array(goal_pos)
            line_vec = p1 - p0
            line_len = np.linalg.norm(line_vec)
            if line_len < 1e-6:
                traj_rmse.append(0.0)
                continue
            line_unit = line_vec / line_len
            errors = []
            for pos in self.positions[i]:
                p = np.array(pos)
                proj = np.dot(p - p0, line_unit)
                closest = p0 + proj * line_unit
                error = np.linalg.norm(p - closest)
                errors.append(error)
            traj_rmse.append(np.sqrt(np.mean(np.square(errors))))
        metrics['trajectory_rmse_mean'] = np.mean(traj_rmse)
        metrics['trajectory_rmse_std'] = np.std(traj_rmse)

        # ----- 3. Control smoothness / stability -----
        for i in range(self.num_agents):
            v = np.array(self.velocities[i])
            if len(v) > 1:
                acc = np.diff(v) / 1.0
                metrics[f'agent_{i}_acc_variance'] = np.var(acc)
                metrics[f'agent_{i}_speed_variance'] = np.var(v)
            else:
                metrics[f'agent_{i}_acc_variance'] = 0.0
                metrics[f'agent_{i}_speed_variance'] = 0.0

            w = np.array(self.angular_vels[i])
            metrics[f'agent_{i}_angular_vel_variance'] = np.var(w)

            if self.control_v[i]:
                cmd_v = np.array(self.control_v[i])
                cmd_w = np.array(self.control_w[i])
                metrics[f'agent_{i}_control_effort_v'] = np.sum(np.square(cmd_v))
                metrics[f'agent_{i}_control_effort_w'] = np.sum(np.square(cmd_w))
                if len(cmd_v) > 1:
                    metrics[f'agent_{i}_cmd_v_jerk'] = np.var(np.diff(cmd_v))
                    metrics[f'agent_{i}_cmd_w_jerk'] = np.var(np.diff(cmd_w))

        # ----- 4. Formation metrics -----
        if self.formation_errors:
            metrics['formation_error_mean'] = np.mean(self.formation_errors)
            metrics['formation_error_std'] = np.std(self.formation_errors)
            metrics['formation_error_max'] = np.max(self.formation_errors)
        else:
            pairwise_errors = []
            for step in range(len(self.positions[0])):
                positions_step = [self.positions[i][step] for i in range(self.num_agents)]
                for i in range(self.num_agents):
                    for j in range(i+1, self.num_agents):
                        actual = np.hypot(positions_step[i][0]-positions_step[j][0],
                                          positions_step[i][1]-positions_step[j][1])
                        desired = 1.2
                        pairwise_errors.append(abs(actual - desired))
            metrics['formation_pairwise_error_mean'] = np.mean(pairwise_errors)
            metrics['formation_pairwise_error_std'] = np.std(pairwise_errors)

        # ----- 5. Collision and safety metrics -----
        total_collisions = sum(sum(self.collisions[i]) for i in range(self.num_agents))
        metrics['total_collisions'] = total_collisions
        metrics['collision_rate'] = total_collisions / max(1, len(self.step_times))

        for i in range(self.num_agents):
            danger_time = sum(1 for s in self.safety_signals[i] if s == 1)
            metrics[f'agent_{i}_danger_steps'] = danger_time
            metrics[f'agent_{i}_danger_ratio'] = danger_time / max(1, len(self.safety_signals[i]))

        # ----- 6. Path efficiency -----
        total_travel = [0.0] * self.num_agents
        for i in range(self.num_agents):
            if len(self.positions[i]) > 1:
                for k in range(1, len(self.positions[i])):
                    dx = self.positions[i][k][0] - self.positions[i][k-1][0]
                    dy = self.positions[i][k][1] - self.positions[i][k-1][1]
                    total_travel[i] += np.hypot(dx, dy)
        straight_line = [np.hypot(start_positions[i][0]-goal_pos[0], start_positions[i][1]-goal_pos[1])
                         for i in range(self.num_agents)]
        metrics['path_efficiency'] = [straight_line[i] / max(total_travel[i], 1e-6) for i in range(self.num_agents)]
        metrics['mean_path_efficiency'] = np.mean(metrics['path_efficiency'])

        # ----- 7. Episode duration -----
        metrics['episode_duration_steps'] = len(self.step_times)
        if self.episode_end_time is not None:
            metrics['episode_duration_seconds'] = self.episode_end_time - self.episode_start_time
        else:
            metrics['episode_duration_seconds'] = None

        return metrics

    def save_to_file(self, metrics: Dict, filepath: str):
        """Save metrics to JSON file."""
        import json
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj
        metrics_serializable = convert(metrics)
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)