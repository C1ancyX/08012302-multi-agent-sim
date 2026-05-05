import numpy as np

class MultiAgentEnv:
    """
    Multi-Agent Formation Simulation Environment
    - 3 agents, differential drive kinematic model
    - Action: [v, w] for each agent
    - State: [x, y, theta] for each agent
    - Target: all agents reach the goal area
    """
    def __init__(self, num_agents=3, dt=0.1, max_steps=300):
        self.num_agents = num_agents
        self.dt = dt
        self.max_steps = max_steps
        
        # Environment boundary
        self.boundary = [0, 30, 0, 30]
        
        # Goal
        self.target = np.array([25.0, 25.0])
        self.target_radius = 1.5
        
        # Step counter
        self.current_step = 0
        
        # Agent states, will be set in reset
        self.states = None
        
    def reset(self):
        self.current_step = 0
        
        starts = np.array([
            [2.0, 2.0, 0.0],
            [2.0, 5.0, 0.0],
            [2.0, 8.0, 0.0]
        ])
        self.states = starts.copy()
        
        return [self.states[i].tolist() for i in range(self.num_agents)]
    
    def step(self, actions):
        self.current_step += 1
        
        # Update each agent's state
        for i in range(self.num_agents):
            v, w = actions[i]
            x, y, theta = self.states[i]
            
            # Differential drive kinematics
            x += v * np.cos(theta) * self.dt
            y += v * np.sin(theta) * self.dt
            theta += w * self.dt
            
            # Boundary clipping
            x = np.clip(x, self.boundary[0], self.boundary[1])
            y = np.clip(y, self.boundary[2], self.boundary[3])
            
            self.states[i] = [x, y, theta]
        
        # Compute rewards
        rewards = []
        done = False
        all_arrived = True
        
        for i in range(self.num_agents):
            pos = self.states[i][:2]
            dist = np.linalg.norm(pos - self.target)
            
            if dist < self.target_radius:
                reward = 100.0
            else:
                reward = -0.1
                all_arrived = False
            
            rewards.append(reward)
        
        if all_arrived or self.current_step >= self.max_steps:
            done = True
        
        next_states = [self.states[i].tolist() for i in range(self.num_agents)]
        
        return next_states, rewards, done

    def get_state_dim(self):
        return 3 + 2 * (self.num_agents - 1)
    
    def get_action_dim(self):
        return 2