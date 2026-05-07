# agents/maddpg_agent.py（示意核心接口）
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self, obs_dim=3, action_dim=2, max_action=1.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action

    def forward(self, obs):
        return self.net(obs) * self.max_action

class Critic(nn.Module):
    def __init__(self, global_state_dim=9, all_action_dim=6):
        super().__init__()
        # global_state_dim = 3个智能体*(x,y,θ)=9, all_action_dim = 3*2=6
        self.net = nn.Sequential(
            nn.Linear(global_state_dim + all_action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state, actions):
        x = torch.cat([state, actions], dim=1)
        return self.net(x)

class MADDPGAgent:
    def __init__(self, obs_dim, action_dim, agent_id):
        self.actor = Actor(obs_dim, action_dim)
        self.target_actor = Actor(obs_dim, action_dim)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-3)

    def select_action(self, obs, noise=0.1):
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs_tensor).detach().numpy()[0]
        # 添加探索噪声
        action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)   # v 范围 [-1,1], w 需要再乘2