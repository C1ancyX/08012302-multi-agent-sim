import numpy as np

class Robot3:
    def __init__(self, dt=0.1, mass=1.0, inertia=0.1, damping=0.5, rot_damping=0.05):
        self.id = 3
        self.dt = dt
        self.mass = mass
        self.inertia = inertia
        self.damping = damping
        self.rot_damping = rot_damping

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.w = 0.0
        self.F = 0.0
        self.M = 0.0

    def reset(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = 0.0
        self.w = 0.0
        self.F = 0.0
        self.M = 0.0

    def set_control(self, F, M):
        self.F = F
        self.M = M

    def set_velocity(self, v, w):
        self.v = v
        self.w = w

    def step(self):
        self.x += self.v * np.cos(self.theta) * self.dt
        self.y += self.v * np.sin(self.theta) * self.dt
        self.theta += self.w * self.dt
        self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

    def get_state(self):
        return (self.x, self.y, self.theta, self.v, self.w)