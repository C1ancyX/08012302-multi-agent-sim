import numpy as np
from shapely.geometry import Polygon, Point
from shapely.strtree import STRtree
from typing import List, Tuple, Dict, Optional

class MultiAgentEnv:
    """
    Multi-Agent Environment
    - 3 rectangular vehicles (length=0.4m, width=0.2m)
    - Differential drive kinematics
    - Collision detection with safety grading
    """
    
    def __init__(self, num_agents: int = 3, dt: float = 0.1, 
                 max_steps: int = 500, render_mode: Optional[str] = None):
        self.num_agents = num_agents
        self.dt = dt
        self.max_steps = max_steps
        self.render_mode = render_mode
        
        # Vehicle dimensions (meters)
        self.car_length = 0.4
        self.car_width = 0.2
        
        # Safety distance (meters)
        self.safe_distance = 2.0
        
        # Environment boundaries
        self.boundary_min = np.array([0.0, 0.0])
        self.boundary_max = np.array([30.0, 30.0])
        
        # Goal position
        self.goal_position = np.array([25.0, 25.0])
        self.goal_radius = 1.5
        
        # Agent states: each agent has [x, y, theta, v, w]
        self.states = None
        self.v_actual = None
        self.w_actual = None
        self.current_step = 0
        self.collision_flags = None
        self.safety_signals = None
        self.online_status = None
        self.fault_codes = None
        self.operation_modes = None
        
        # Control inputs from main program
        self.set_v = None
        self.set_w = None
        self.mode_cmd = None
        self.start_stop = None
        self.obstacle_avoid_en = None
        self.tracking_en = None
        
        # Target from main program
        self.target_x = 25.0
        self.target_y = 25.0
        self.target_angle = 0.0
        
    def reset(self) -> List[List[float]]:
        """Reset environment to initial state"""
        self.current_step = 0
        
        # Initial positions (x, y, theta)
        initial_positions = np.array([
            [2.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [2.0, 8.0, 0.0]
        ])
        
        self.states = initial_positions.copy()
        self.v_actual = np.zeros(self.num_agents)
        self.w_actual = np.zeros(self.num_agents)
        self.collision_flags = np.zeros(self.num_agents, dtype=bool)
        self.safety_signals = np.zeros(self.num_agents, dtype=int)
        self.online_status = np.ones(self.num_agents, dtype=bool)
        self.fault_codes = np.zeros(self.num_agents, dtype=int)
        self.operation_modes = np.zeros(self.num_agents, dtype=int)  # 0=auto, 1=manual
        
        # Initialize control inputs
        self.set_v = np.zeros(self.num_agents)
        self.set_w = np.zeros(self.num_agents)
        self.mode_cmd = np.zeros(self.num_agents, dtype=int)
        self.start_stop = np.ones(self.num_agents, dtype=int)
        self.obstacle_avoid_en = np.zeros(self.num_agents, dtype=int)
        self.tracking_en = np.zeros(self.num_agents, dtype=int)
        
        return self._get_state_list()
    
    def step(self, actions: List[List[float]]) -> Tuple[List[List[float]], List[float], bool, Dict]:
        """Execute one step with given actions"""
        self.current_step += 1
        
        # Apply actions (only in auto mode)
        for i in range(self.num_agents):
            if self.mode_cmd[i] == 0 and self.start_stop[i] == 1:
                self.set_v[i] = np.clip(actions[i][0], -1.0, 1.0)
                self.set_w[i] = np.clip(actions[i][1], -2.0, 2.0)
        
        # Update kinematics for all agents
        for i in range(self.num_agents):
            if self.start_stop[i] == 1:
                self._update_kinematics(i)
        
        # Collision detection and safety assessment
        self._update_collision_detection()
        
        # Compute rewards
        rewards = self._compute_rewards()
        
        # Check termination condition
        done = self._check_done()
        
        # Prepare info dict
        info = self._get_info_dict()
        
        return self._get_state_list(), rewards, done, info
    
    def _update_kinematics(self, agent_id: int):
        """Update vehicle position using differential drive model"""
        v = self.set_v[agent_id]
        w = self.set_w[agent_id]
        
        x, y, theta = self.states[agent_id]
        
        # Differential drive kinematics
        x += v * np.cos(theta) * self.dt
        y += v * np.sin(theta) * self.dt
        theta += w * self.dt
        
        # Boundary clipping
        x = np.clip(x, self.boundary_min[0], self.boundary_max[0])
        y = np.clip(y, self.boundary_min[1], self.boundary_max[1])
        
        self.states[agent_id] = [x, y, theta]
        self.v_actual[agent_id] = v
        self.w_actual[agent_id] = w
    
    def _get_vehicle_polygon(self, agent_id: int) -> Polygon:
        """Get vehicle polygon for collision detection"""
        x, y, theta = self.states[agent_id]
        
        # Vehicle corner offsets (local coordinates)
        half_length = self.car_length / 2
        half_width = self.car_width / 2
        
        corners_local = np.array([
            [-half_length, -half_width],
            [ half_length, -half_width],
            [ half_length,  half_width],
            [-half_length,  half_width]
        ])
        
        # Rotation matrix
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
        
        # Transform to world coordinates
        corners_world = corners_local @ rotation.T + np.array([x, y])
        
        return Polygon(corners_world)
    
    def _update_collision_detection(self):
        """Update collision flags and safety signals"""
        self.collision_flags.fill(False)
        self.safety_signals.fill(0)
        
        # Check pairwise distances
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                min_distance = self._get_min_distance_between_vehicles(i, j)
                
                if min_distance <= 0:
                    self.collision_flags[i] = True
                    self.collision_flags[j] = True
                    self.safety_signals[i] = 2
                    self.safety_signals[j] = 2
                elif min_distance < self.safe_distance:
                    if self.safety_signals[i] < 1:
                        self.safety_signals[i] = 1
                    if self.safety_signals[j] < 1:
                        self.safety_signals[j] = 1
    
    def _get_min_distance_between_vehicles(self, i: int, j: int) -> float:
        """Calculate minimum distance between two vehicles"""
        poly_i = self._get_vehicle_polygon(i)
        poly_j = self._get_vehicle_polygon(j)
        
        if poly_i.intersects(poly_j):
            return 0.0
        
        return poly_i.distance(poly_j)
    
    def _compute_rewards(self) -> List[float]:
        """Compute rewards for each agent"""
        rewards = []
        
        for i in range(self.num_agents):
            reward = -0.01  # Step penalty
            
            # Goal reward
            pos = np.array(self.states[i][:2])
            dist_to_goal = np.linalg.norm(pos - self.goal_position)
            if dist_to_goal < self.goal_radius:
                reward += 100.0
            
            # Collision penalty
            if self.collision_flags[i]:
                reward -= 50.0
            
            # Danger penalty
            if self.safety_signals[i] == 1:
                reward -= 5.0
            
            # Formation reward (maintain distance between agents)
            formation_error = self._compute_formation_error(i)
            reward -= 0.1 * formation_error
            
            # Progress reward (closer to goal)
            prev_dist = dist_to_goal + self.v_actual[i] * self.dt
            reward += 0.1 * (prev_dist - dist_to_goal)
            
            rewards.append(reward)
        
        return rewards
    
    def _compute_formation_error(self, agent_id: int) -> float:
        """Compute formation deviation for a single agent"""
        ideal_distance = 3.0  # Ideal distance between agents
        total_error = 0.0
        count = 0
        
        for j in range(self.num_agents):
            if j != agent_id:
                pos_i = np.array(self.states[agent_id][:2])
                pos_j = np.array(self.states[j][:2])
                actual_dist = np.linalg.norm(pos_i - pos_j)
                total_error += abs(actual_dist - ideal_distance)
                count += 1
        
        return total_error / count if count > 0 else 0.0
    
    def _check_done(self) -> bool:
        """Check if episode is done"""
        # Check if all agents reached goal
        all_reached = True
        for i in range(self.num_agents):
            pos = np.array(self.states[i][:2])
            if np.linalg.norm(pos - self.goal_position) >= self.goal_radius:
                all_reached = False
                break
        
        if all_reached:
            return True
        
        if self.current_step >= self.max_steps:
            return True
        
        return False
    
    def _get_state_list(self) -> List[List[float]]:
        """Get current states as list of [x, y, theta]"""
        return [self.states[i].tolist() for i in range(self.num_agents)]
    
    def _get_info_dict(self) -> Dict:
        """Get additional information for each agent"""
        info = {
            'v_actual': self.v_actual.tolist(),
            'w_actual': self.w_actual.tolist(),
            'collision': self.collision_flags.tolist(),
            'safety': self.safety_signals.tolist(),
            'online': self.online_status.tolist(),
            'fault_code': self.fault_codes.tolist(),
            'operation_mode': self.operation_modes.tolist(),
            'target_error': self._get_target_errors(),
            'obstacle_dist': self._get_obstacle_distances(),
            'step_count': self.current_step,
            'formation_error': self._get_formation_error_all()
        }
        return info
    
    def _get_target_errors(self) -> List[float]:
        """Calculate distance error to target for each agent"""
        errors = []
        for i in range(self.num_agents):
            pos = np.array(self.states[i][:2])
            errors.append(np.linalg.norm(pos - self.goal_position))
        return errors
    
    def _get_formation_error_all(self) -> float:
        """Calculate total formation error"""
        total_error = 0.0
        ideal_distance = 3.0
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                pos_i = np.array(self.states[i][:2])
                pos_j = np.array(self.states[j][:2])
                actual_dist = np.linalg.norm(pos_i - pos_j)
                total_error += abs(actual_dist - ideal_distance)
        
        return total_error
    
    def _get_obstacle_distances(self) -> Dict[str, List[float]]:
        """
        Get obstacle distances for each agent
        Simplified: distances to boundaries and other vehicles
        """
        front_dist = []
        rear_dist = []
        left_dist = []
        right_dist = []
        
        for i in range(self.num_agents):
            x, y, theta = self.states[i]
            
            # Distances to boundaries
            front_dist.append(min(self.boundary_max[0] - x, x - self.boundary_min[0]))
            rear_dist.append(min(self.boundary_max[1] - y, y - self.boundary_min[1]))
            left_dist.append(x - self.boundary_min[0])
            right_dist.append(self.boundary_max[0] - x)
        
        return {
            'front': front_dist,
            'rear': rear_dist,
            'left': left_dist,
            'right': right_dist
        }
    
    def set_manual_control(self, agent_id: int, v: float, w: float):
        """Set manual control inputs for an agent"""
        if self.mode_cmd[agent_id] == 1:
            self.set_v[agent_id] = np.clip(v, -1.0, 1.0)
            self.set_w[agent_id] = np.clip(w, -2.0, 2.0)
    
    def get_state_dim(self) -> int:
        """Get observation dimension for each agent"""
        return 3  # x, y, theta
    
    def get_action_dim(self) -> int:
        """Get action dimension for each agent"""
        return 2  # v, w
    
    def get_global_state_dim(self) -> int:
        """Get global state dimension for critic"""
        return self.num_agents * 3  # all agents' x, y, theta