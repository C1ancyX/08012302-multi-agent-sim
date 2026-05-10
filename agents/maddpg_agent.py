<<<<<<< HEAD
﻿import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from agents.network import Actor, Critic

class MADDPGAgent:
    """
    MADDPG Agent for multi-agent continuous control
    Each agent has its own Actor, but shares Critic during training
    """
    
    def __init__(self, agent_id: int, state_dim: int, action_dim: int, 
                 global_state_dim: int, num_agents: int,
                 lr_actor: float = 1e-3, lr_critic: float = 1e-3,
                 gamma: float = 0.99, tau: float = 0.01):
        
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.global_state_dim = global_state_dim
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        
        # Actor networks (local and target)
        self.actor = Actor(state_dim, action_dim, max_action=1.0)
        self.target_actor = Actor(state_dim, action_dim, max_action=1.0)
        
        # Critic networks (local and target)
        self.critic = Critic(global_state_dim, action_dim, num_agents)
        self.target_critic = Critic(global_state_dim, action_dim, num_agents)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Initialize target networks with same weights
        self._init_target_networks()
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to_device()
    
    def _init_target_networks(self):
        """Initialize target networks with same weights as local networks"""
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
    
    def to_device(self):
        """Move networks to device"""
        self.actor.to(self.device)
        self.target_actor.to(self.device)
        self.critic.to(self.device)
        self.target_critic.to(self.device)
    
    def select_action(self, state: np.ndarray, noise_scale: float = 0.1) -> np.ndarray:
        """
        Select action using actor network with exploration noise
        Args:
            state: numpy array of shape (state_dim,)
            noise_scale: scale of Gaussian noise for exploration
        Returns:
            action: numpy array of shape (action_dim,)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        
        # Add exploration noise
        noise = np.random.normal(0, noise_scale, size=self.action_dim)
        action = np.clip(action + noise, -1.0, 1.0)
        
        return action
    
    def update_actor(self, states: torch.Tensor, actions: torch.Tensor, 
                     critic: nn.Module) -> torch.Tensor:
        """
        Update actor network using policy gradient
        Args:
            states: local states for this agent (batch_size, state_dim)
            actions: all actions for all agents (batch_size, num_agents * action_dim)
            critic: centralized critic network
        Returns:
            actor_loss: scalar tensor
        """
        # Get action from current actor
        new_actions = self.actor(states)
        
        # Replace this agent's action in the joint action tensor
        # This is simplified - in practice, you'd need to reconstruct the full action tensor
        # For now, we assume the critic takes current actions directly
        
        # Calculate Q-value for the new action
        q_value = critic(states, new_actions)  # Simplified
        
        # Actor loss: negative Q-value (maximize Q)
        actor_loss = -q_value.mean()
        
        # Optimize
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        return actor_loss
    
    def update_critic(self, states: torch.Tensor, actions: torch.Tensor,
                      rewards: torch.Tensor, next_states: torch.Tensor,
                      next_actions: torch.Tensor, dones: torch.Tensor,
                      target_critic: nn.Module) -> torch.Tensor:
        """
        Update critic network using TD learning
        Args:
            states: global states (batch_size, global_state_dim)
            actions: all actions (batch_size, num_agents * action_dim)
            rewards: rewards for this agent (batch_size, 1)
            next_states: next global states (batch_size, global_state_dim)
            next_actions: next all actions (batch_size, num_agents * action_dim)
            dones: done flags (batch_size, 1)
            target_critic: target critic network
        Returns:
            critic_loss: scalar tensor
        """
        # Current Q-value
        current_q = self.critic(states, actions)
        
        # Target Q-value
        with torch.no_grad():
            target_q = rewards + (1 - dones) * self.gamma * target_critic(next_states, next_actions)
        
        # MSE loss
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Optimize
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        return critic_loss
    
    def soft_update(self):
        """Soft update target networks"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def save_models(self, path_prefix: str):
        """Save models to disk"""
        torch.save(self.actor.state_dict(), f"{path_prefix}_actor_{self.agent_id}.pth")
        torch.save(self.critic.state_dict(), f"{path_prefix}_critic_{self.agent_id}.pth")
        torch.save(self.target_actor.state_dict(), f"{path_prefix}_target_actor_{self.agent_id}.pth")
        torch.save(self.target_critic.state_dict(), f"{path_prefix}_target_critic_{self.agent_id}.pth")
    
    def load_models(self, path_prefix: str):
        """Load models from disk"""
        self.actor.load_state_dict(torch.load(f"{path_prefix}_actor_{self.agent_id}.pth"))
        self.critic.load_state_dict(torch.load(f"{path_prefix}_critic_{self.agent_id}.pth"))
        self.target_actor.load_state_dict(torch.load(f"{path_prefix}_target_actor_{self.agent_id}.pth"))
        self.target_critic.load_state_dict(torch.load(f"{path_prefix}_target_critic_{self.agent_id}.pth"))
=======
# agents/maddpg_agent.py��ʾ����Ľӿڣ�
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
        # global_state_dim = 3��������*(x,y,��)=9, all_action_dim = 3*2=6
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
        # ���̽������
        action = action + np.random.normal(0, noise, size=action.shape)
        return np.clip(action, -1.0, 1.0)   # v ��Χ [-1,1], w ��Ҫ�ٳ�2
>>>>>>> 2b690f8e53359950940e1fe5342cb944a5b19dd3
