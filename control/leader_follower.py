import numpy as np

class LeaderFollower:
    """
    Leader-Follower 编队控制，基于相对距离和角度，提高角速度增益以强化队形保持。
    """
    def __init__(self, rho_des=1.2, phi_des=np.radians(60), k_rho=0.8, k_phi=1.8):
        self.rho_des = rho_des
        self.phi_des = phi_des 
        self.k_rho = k_rho
        self.k_phi = k_phi
    def compute_control(self, leader_state, follower_state, dt=0.1):
        """
        leader_state: (x, y, theta, v, w)
        follower_state: (x, y, theta, v, w)
        返回 follower 的期望线速度和角速度 (v_des, w_des)
        """
        lx, ly, ltheta, lv, lw = leader_state
        fx, fy, ftheta, fv, fw = follower_state

        #相对位置
        dx = fx - lx
        dy = fy - ly
        rho = np.hypot(dx, dy)
        if rho < 1e-6:
            rho = 1e-6
        phi = np.arctan2(dy, dx) - ltheta   #相对角度（在 leader 坐标系中）
        phi = np.arctan2(np.sin(phi), np.cos(phi))   #归一化到 [-pi, pi]

        #控制误差
        rho_err = rho - self.rho_des
        phi_err = phi - self.phi_des
        phi_err = np.arctan2(np.sin(phi_err), np.cos(phi_err))

        #线速度控制：跟随 leader 速度 + 距离误差补偿
        v_des = lv + self.k_rho * rho_err
        v_des = np.clip(v_des, 0.0, 1.5)   # 限制最大线速度

        #角速度控制：使 follower 朝向 leader 并纠正相对角度
        #期望朝向：指向 leader 的方向（即从 follower 指向 leader 的角度）
        desired_heading = np.arctan2(lx - fx, ly - fy)   #注意参数顺序
        heading_err = desired_heading - ftheta
        heading_err = np.arctan2(np.sin(heading_err), np.cos(heading_err))
        # 组合角度误差和航向误差
        w_des = 1.2 * heading_err + 0.8 * phi_err
        w_des = np.clip(w_des, -2.5, 2.5)   #放宽角速度限幅，允许快速转向

        return v_des, w_des