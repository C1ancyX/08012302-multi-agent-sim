import numpy as np
from models.robot1 import Robot1
from models.robot2 import Robot2
from models.robot3 import Robot3
from control.dwa import DWA
from planning.astar import AStarPlanner
from control.leader_follower import LeaderFollower

class MultiAgentEnv:
    def __init__(self, num_agents=3, dt=0.1, max_steps=500,
                 num_obstacles=20, enable_obstacles=True,
                 goal_position=(25.,25.)):
        self.num_agents = num_agents
        self.dt = dt
        self.max_steps = max_steps
        self.enable_obstacles = enable_obstacles
        self.goal = np.array(goal_position)
        self.goal_rad = 1.5

        self.robot1 = Robot1(dt=dt)
        self.robot2 = Robot2(dt=dt)
        self.robot3 = Robot3(dt=dt)
        self.robots = [self.robot1, self.robot2, self.robot3]

        self.leader_id = 1

        self.leader_dwa = DWA(dt=dt, max_v=1.5, min_v=0.0, max_w=2.0, min_w=-2.0,
                              predict_time=1.2, resolution=1.0,
                              heading_gain=2.0, dist_gain=0.4, vel_gain=2.0,
                              safe_dist=1.5)

        self.follower_dwa = DWA(dt=dt, max_v=1.5, min_v=0.0, max_w=3.0, min_w=-4.0,
                        predict_time=1.2, resolution=1.0,
                        heading_gain=1.0, dist_gain=0.5, vel_gain=2.0,
                        safe_dist=1.5)

        self.planner = AStarPlanner(grid_size=0.5, world_size=30.0)
        self.global_path = None

        self.obstacles = []
        if self.enable_obstacles:
            self._gen_obstacles(num_obstacles)

        self.step_cnt = 0
        self.collision_flags = np.zeros(num_agents, dtype=bool)
        self.safety_signals = np.zeros(num_agents, dtype=int)

        self.follower_params = {
            0: (1.2, -np.radians(60)),
            2: (1.2, np.radians(60))
        }

    def _gen_obstacles(self, num_obstacles):
        import random
        self.obstacles = []
        start_pos = [(self.robot1.x, self.robot1.y),
                     (self.robot2.x, self.robot2.y),
                     (self.robot3.x, self.robot3.y)]
        forbid = start_pos + [(self.goal[0], self.goal[1])]
        for _ in range(num_obstacles):
            for _ in range(100):
                x = random.uniform(3.,27.)
                y = random.uniform(3.,27.)
                ok = True
                for fx, fy in forbid:
                    if np.hypot(x-fx, y-fy) < 2.0:
                        ok = False
                        break
                for _, ox, oy, _ in self.obstacles:
                    if np.hypot(x-ox, y-oy) < 1.0:
                        ok = False
                        break
                if ok:
                    r = random.uniform(0.3, 0.5)
                    self.obstacles.append((None, x, y, r))
                    break

    def reset(self):
        self.step_cnt = 0
        lx, ly = 2.0, 5.0
        rho_des = 1.2
        phi1 = -np.radians(45)
        phi3 = np.radians(45)
        self.robot1.reset(lx + rho_des * np.cos(phi1), ly + rho_des * np.sin(phi1), 0.0)
        self.robot2.reset(lx, ly, 0.0)
        self.robot3.reset(lx + rho_des * np.cos(phi3), ly + rho_des * np.sin(phi3), 0.0)
        self._replan_path()
        return self._get_obs()

    def _replan_path(self):
        start = (self.robot2.x, self.robot2.y)
        goal = (self.goal[0], self.goal[1])
        obs_list = [(ox, oy, r) for _, ox, oy, r in self.obstacles]
        self.global_path = self.planner.plan(start, goal, obs_list)
        if not self.global_path or len(self.global_path) < 2:
            self.global_path = [start, goal]

    def _get_current_target(self, robot_x, robot_y):
        if not self.global_path:
            return self.goal
        dists = [np.hypot(robot_x - px, robot_y - py) for (px, py) in self.global_path]
        nearest = np.argmin(dists)
        if nearest == 0 and dists[0] < 0.5 and len(self.global_path) > 1:
            nearest = 1
        lookahead = min(nearest + 3, len(self.global_path) - 1)
        px, py = self.global_path[lookahead]
        if np.hypot(px - robot_x, py - robot_y) < 0.8 and lookahead + 1 < len(self.global_path):
            lookahead += 1
        return self.global_path[lookahead]

    def _check_obstacle_ahead(self, robot, lookahead_dist=2.0, angle_range=np.radians(100)):
        x, y, theta = robot.x, robot.y, robot.theta
        for _, ox, oy, r in self.obstacles:
            dx = ox - x
            dy = oy - y
            dist = np.hypot(dx, dy)
            if dist > lookahead_dist:
                continue
            angle_to_obs = np.arctan2(dy, dx)
            angle_diff = angle_to_obs - theta
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            if abs(angle_diff) < angle_range / 2:
                return True
        for other in self.robots:
            if other is robot:
                continue
            dx = other.x - x
            dy = other.y - y
            dist = np.hypot(dx, dy)
            if dist > lookahead_dist:
                continue
            angle_to_obs = np.arctan2(dy, dx)
            angle_diff = angle_to_obs - theta
            angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
            if abs(angle_diff) < angle_range / 2:
                return True
        return False

    def step(self, actions=None):
        self.step_cnt += 1

        if self.step_cnt % 50 == 0:
            self._replan_path()

        leader = self.robot2
        target = self._get_current_target(leader.x, leader.y)

        obs_list = [(ox, oy, r) for _, ox, oy, r in self.obstacles]
        obs_list.append((self.robot1.x, self.robot1.y, 0.4))
        obs_list.append((self.robot3.x, self.robot3.y, 0.4))

        v_leader, w_leader = self.leader_dwa.plan(
            (leader.x, leader.y, leader.theta, leader.v, leader.w),
            target,
            obs_list,
            leader.v, leader.w
        )
        v_leader = np.clip(v_leader, 0.2, 1.0)
        w_leader = np.clip(w_leader, -2.0, 2.0)
        if abs(w_leader) < 0.1:
            w_leader = 0.0
        leader.set_velocity(v_leader, w_leader)

        leader_state = leader.get_state()
        for fid, (rho_des, phi_des) in self.follower_params.items():
            follower = self.robots[fid]
            lx, ly, ltheta, lv, lw = leader_state
            target_x = lx + rho_des * np.cos(ltheta + phi_des)
            target_y = ly + rho_des * np.sin(ltheta + phi_des)

            need_avoid = self._check_obstacle_ahead(follower, lookahead_dist=2.0)

            if need_avoid:
                obs_list_f = [(ox, oy, r) for _, ox, oy, r in self.obstacles]
                for other in self.robots:
                    if other is not follower:
                        obs_list_f.append((other.x, other.y, 0.4))
                v_f, w_f = self.follower_dwa.plan(
                    (follower.x, follower.y, follower.theta, follower.v, follower.w),
                    (target_x, target_y),
                    obs_list_f,
                    follower.v, follower.w
                )
                w_f = np.clip(w_f, -3.0, 3.0)
            else:
                v_f, w_f = self._follower_control(leader_state, follower.get_state(), rho_des, phi_des)

            dx = target_x - follower.x
            dy = target_y - follower.y
            dist_error = np.hypot(dx, dy)
            speed_limit = 1.2 if dist_error > 0.8 else 0.8
            v_f = np.clip(v_f, 0.0, speed_limit)
            w_f = np.clip(w_f, -2.0, 2.0)
            follower.set_velocity(v_f, w_f)

        for robot in self.robots:
            robot.step()

        self._update_safety()
        done = self._check_done()
        rewards = self._compute_rewards()
        info = self._get_info()

        return self._get_obs(), rewards, done, info

    def _follower_control(self, leader_state, follower_state, rho_des, phi_des):
        lx, ly, ltheta, lv, lw = leader_state
        fx, fy, ftheta, fv, fw = follower_state

        target_x = lx + rho_des * np.cos(ltheta + phi_des)
        target_y = ly + rho_des * np.sin(ltheta + phi_des)

        dx = target_x - fx
        dy = target_y - fy
        dist_error = np.hypot(dx, dy)

        base_speed = max(0.2, lv)
        v_des = base_speed + 8.0 * dist_error
        v_des = np.clip(v_des, 0.2, 1.2)

        desired_theta = np.arctan2(dy, dx)
        heading_err = desired_theta - ftheta
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        w_des = 0.6 * heading_err
        w_des = np.clip(w_des, -0.6, 0.6)

        return v_des, w_des

    def _update_safety(self):
        self.safety_signals.fill(0)
        self.collision_flags.fill(False)
        safe_dist = 1.5
        robot_radius = 0.3

        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                dx = self.robots[i].x - self.robots[j].x
                dy = self.robots[i].y - self.robots[j].y
                dist = np.hypot(dx, dy)
                if dist < 2 * robot_radius:
                    self.collision_flags[i] = self.collision_flags[j] = True
                    self.safety_signals[i] = self.safety_signals[j] = 2
                elif dist < safe_dist:
                    if self.safety_signals[i] < 1: self.safety_signals[i] = 1
                    if self.safety_signals[j] < 1: self.safety_signals[j] = 1

        for i, robot in enumerate(self.robots):
            for _, ox, oy, r in self.obstacles:
                dist = np.hypot(robot.x - ox, robot.y - oy) - (robot_radius + r)
                if dist < 0:
                    self.collision_flags[i] = True
                    self.safety_signals[i] = 2
                    break
                elif dist < safe_dist and self.safety_signals[i] < 1:
                    self.safety_signals[i] = 1

    def _compute_rewards(self):
        rewards = []
        for robot in self.robots:
            dist = np.hypot(robot.x - self.goal[0], robot.y - self.goal[1])
            r = -0.01
            if dist < self.goal_rad:
                r += 100.0
            rewards.append(r)
        return rewards

    def _check_done(self):
        if all(np.hypot(r.x - self.goal[0], r.y - self.goal[1]) < self.goal_rad for r in self.robots):
            return True
        if self.step_cnt >= self.max_steps:
            return True
        return False

    def _get_obs(self):
        return [[r.x, r.y, r.theta, r.v, r.w] for r in self.robots]

    def _get_info(self):
        return {
            'v_actual': [r.v for r in self.robots],
            'w_actual': [r.w for r in self.robots],
            'collision': self.collision_flags.tolist(),
            'safety': self.safety_signals.tolist(),
            'step_count': self.step_cnt,
            'goal_position': (self.goal[0], self.goal[1]),
            'goal_radius': self.goal_rad,
            'obstacles': [{'x': ox, 'y': oy, 'radius': r} for _, ox, oy, r in self.obstacles],
            'global_path': self.global_path
        }

    def get_obstacles_for_rendering(self):
        return [{'x': ox, 'y': oy, 'radius': r} for _, ox, oy, r in self.obstacles]

    def get_global_path_for_rendering(self):
        return self.global_path

    def get_state_dim(self):
        return 5

    def get_action_dim(self):
        return 2

    def get_global_state_dim(self):
        return self.num_agents * 3