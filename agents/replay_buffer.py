import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """
    Experience replay buffer for multi-agent systems
    Stores transitions for all agents simultaneously
    """
    
    def __init__(self, capacity: int = 1000000, batch_size: int = 256):
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = deque(maxlen=capacity)
        
        # For priority-based sampling (optional)
        self.priorities = deque(maxlen=capacity)
    
    def push(self, states: np.ndarray, actions: np.ndarray, 
             rewards: np.ndarray, next_states: np.ndarray, done: bool):
        """
        Store a transition
        Args:
            states: shape (num_agents, state_dim)
            actions: shape (num_agents, action_dim)
            rewards: shape (num_agents,)
            next_states: shape (num_agents, state_dim)
            done: boolean
        """
        self.buffer.append((states, actions, rewards, next_states, done))
        
        # Initialize priority for this transition (for prioritized replay)
        # For now, use uniform priority
        self.priorities.append(1.0)
    
    def sample(self) -> tuple:
        """
        Sample a batch of transitions
        Returns:
            states: (batch_size, num_agents, state_dim)
            actions: (batch_size, num_agents, action_dim)
            rewards: (batch_size, num_agents)
            next_states: (batch_size, num_agents, state_dim)
            dones: (batch_size,)
        """
        batch = random.sample(self.buffer, min(self.batch_size, len(self.buffer)))
        
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        return states, actions, rewards, next_states, dones
    
    def sample_prioritized(self, beta: float = 0.4) -> tuple:
        """
        Sample with priority-based sampling
        Args:
            beta: importance sampling weight (0 = no correction, 1 = full correction)
        Returns:
            batch tuple plus indices and importance weights
        """
        if len(self.buffer) < self.batch_size:
            return self.sample()
        
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities / priorities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), self.batch_size, p=probs)
        
        # Compute importance sampling weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states = np.array([transition[0] for transition in batch])
        actions = np.array([transition[1] for transition in batch])
        rewards = np.array([transition[2] for transition in batch])
        next_states = np.array([transition[3] for transition in batch])
        dones = np.array([transition[4] for transition in batch])
        
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Update priorities for sampled transitions"""
        for idx, td_error in zip(indices, td_errors):
            # Priority = |TD error| + small epsilon
            priority = abs(td_error) + 1e-6
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def save(self, filepath: str):
        """Save buffer to disk"""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump({
                'buffer': self.buffer,
                'priorities': self.priorities,
                'capacity': self.capacity,
                'batch_size': self.batch_size
            }, f)
    
    def load(self, filepath: str):
        """Load buffer from disk"""
        import pickle
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.buffer = data['buffer']
            self.priorities = data['priorities']
            self.capacity = data['capacity']
            self.batch_size = data['batch_size']
    
    def clear(self):
        """Clear the buffer"""
        self.buffer.clear()
        self.priorities.clear()