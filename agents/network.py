import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    """
    Actor network: maps observation to action
    Input: state_dim (3 for each agent: x, y, theta)
    Output: action_dim (2: v, w)
    """
    def __init__(self, state_dim: int, action_dim: int, max_action: float = 1.0):
        super(Actor, self).__init__()
        
        self.max_action = max_action
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            state: tensor of shape (batch_size, state_dim)
        Returns:
            action: tensor of shape (batch_size, action_dim) in range [-max_action, max_action]
        """
        action = self.network(state)
        return action * self.max_action


class Critic(nn.Module):
    """
    Centralized Critic network for MADDPG
    Input: global_state (all agents' states) + all actions
    Output: Q-value for each agent (or single Q-value)
    """
    def __init__(self, global_state_dim: int, action_dim: int, num_agents: int):
        super(Critic, self).__init__()
        
        self.num_agents = num_agents
        self.action_dim = action_dim
        
        # Input dimension: global_state + all actions
        input_dim = global_state_dim + (num_agents * action_dim)
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Single Q-value for the joint action
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0.0)
    
    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        Args:
            states: tensor of shape (batch_size, global_state_dim)
            actions: tensor of shape (batch_size, num_agents * action_dim)
        Returns:
            q_value: tensor of shape (batch_size, 1)
        """
        x = torch.cat([states, actions], dim=1)
        q_value = self.network(x)
        return q_value
    
    def get_q_for_agent(self, states: torch.Tensor, actions: torch.Tensor, agent_id: int) -> torch.Tensor:
        """
        Get Q-value for a specific agent (returns same as forward since centralized)
        """
        return self.forward(states, actions)