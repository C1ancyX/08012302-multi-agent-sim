<<<<<<< HEAD
﻿import numpy as np
import torch
import argparse
import json
import os
from datetime import datetime

from simulation.env import MultiAgentEnv
from simulation.renderer import PygameRenderer
from simulation.manual_controller import ManualController
from agents import MADDPGAgent, ReplayBuffer

class MADDPGTrainer:
    """
    Main trainer for MADDPG algorithm
    """
    
    def __init__(self, args):
        self.args = args
        
        # Create environment
        self.env = MultiAgentEnv(
            num_agents=args.num_agents,
            dt=args.dt,
            max_steps=args.max_steps,
            render_mode='human' if args.render else None
        )
        
        # Get dimensions
        self.state_dim = self.env.get_state_dim()
        self.action_dim = self.env.get_action_dim()
        self.global_state_dim = self.env.get_global_state_dim()
        self.num_agents = args.num_agents
        
        # Create renderer (if enabled)
        self.renderer = PygameRenderer() if args.render else None
        
        # Create manual controller
        self.manual_controller = ManualController(self.num_agents) if args.manual else None
        
        # Create agents
        self.agents = []
        for i in range(self.num_agents):
            agent = MADDPGAgent(
                agent_id=i,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                global_state_dim=self.global_state_dim,
                num_agents=self.num_agents,
                lr_actor=args.lr_actor,
                lr_critic=args.lr_critic,
                gamma=args.gamma,
                tau=args.tau
            )
            self.agents.append(agent)
        
        # Create replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=args.buffer_capacity,
            batch_size=args.batch_size
        )
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.best_reward = -float('inf')
        
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('buffers', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        
        # Load checkpoint if resuming
        self.start_episode = 0
        if args.resume:
            self.load_checkpoint()
    
    def train(self):
        """Main training loop"""
        print("Starting MADDPG training...")
        print(f"Configuration: {vars(self.args)}")
        
        for episode in range(self.start_episode, self.args.num_episodes):
            # Reset environment
            states = self.env.reset()
            episode_reward = 0
            step = 0
            
            while True:
                # Select actions for all agents
                actions = []
                for i, agent in enumerate(self.agents):
                    # Check manual mode
                    if self.manual_controller and self.manual_controller.manual_active[i]:
                        cmd = self.manual_controller.get_manual_commands()
                        if cmd['commands'][i] is not None:
                            action = cmd['commands'][i]
                        else:
                            action = agent.select_action(states[i], noise_scale=self.args.noise_scale)
                    else:
                        action = agent.select_action(states[i], noise_scale=self.args.noise_scale)
                    actions.append(action)
                
                # Execute actions
                next_states, rewards, done, info = self.env.step(actions)
                
                # Store transition
                self.replay_buffer.push(
                    np.array(states),
                    np.array(actions),
                    np.array(rewards),
                    np.array(next_states),
                    done
                )
                
                # Update agents
                if len(self.replay_buffer) >= self.args.batch_size:
                    self.update_agents()
                
                # Render
                if self.renderer:
                    render_state = {
                        'states': next_states,
                        'v_actual': info['v_actual'],
                        'w_actual': info['w_actual'],
                        'safety': info['safety'],
                        'collision': info['collision'],
                        'target_error': info['target_error'],
                        'operation_mode': info['operation_mode'],
                        'step_count': info['step_count'],
                        'goal_position': (25.0, 25.0)
                    }
                    self.renderer.render(render_state)
                    
                    # Handle manual control events
                    if self.manual_controller:
                        events = pygame.event.get()
                        self.manual_controller.handle_events(events)
                        
                        # Check for quit
                        if self.renderer.should_close():
                            return
                
                # Update state and metrics
                states = next_states
                episode_reward += sum(rewards)
                step += 1
                
                if done:
                    break
            
            # Log episode results
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(step)
            
            # Print progress
            avg_reward = np.mean(self.episode_rewards[-100:])
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, "
                  f"Length = {step}, Avg Reward (100) = {avg_reward:.2f}")
            
            # Save checkpoint
            if episode % self.args.save_interval == 0:
                self.save_checkpoint(episode)
            
            # Save best model
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_models(f"models/best")
                print(f"New best model saved with reward {episode_reward:.2f}")
        
        print("Training completed!")
        self.save_metrics()
    
    def update_agents(self):
        """Update all agents using sampled batch"""
        # Sample from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.FloatTensor(dones).unsqueeze(-1)
        
        # Flatten for critic (batch_size, num_agents * dim)
        batch_size = states_tensor.shape[0]
        global_states = states_tensor.view(batch_size, -1)
        global_next_states = next_states_tensor.view(batch_size, -1)
        all_actions = actions_tensor.view(batch_size, -1)
        
        # Get next actions from target actors
        next_actions = []
        for i, agent in enumerate(self.agents):
            next_action = agent.target_actor(states_tensor[:, i, :])
            next_actions.append(next_action)
        next_actions_tensor = torch.cat(next_actions, dim=1)
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Update critic
            critic_loss = agent.update_critic(
                global_states, all_actions,
                rewards_tensor[:, i, :], global_next_states, next_actions_tensor,
                dones_tensor, agent.target_critic
            )
            
            # Update actor
            actor_loss = agent.update_actor(
                states_tensor[:, i, :], all_actions, agent.critic
            )
            
            # Soft update target networks
            agent.soft_update()
    
    def save_checkpoint(self, episode: int):
        """Save training checkpoint"""
        checkpoint = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward,
            'args': vars(self.args)
        }
        
        checkpoint_path = f"logs/checkpoint_{episode}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f)
        
        self.save_models(f"models/checkpoint_{episode}")
        self.replay_buffer.save(f"buffers/replay_buffer_{episode}.pkl")
        
        print(f"Checkpoint saved at episode {episode}")
    
    def load_checkpoint(self):
        """Load training checkpoint"""
        try:
            # Load latest checkpoint
            checkpoint_files = [f for f in os.listdir('logs') if f.startswith('checkpoint_')]
            if checkpoint_files:
                latest = max(checkpoint_files, key=lambda x: int(x.split('_')[1].split('.')[0]))
                with open(f"logs/{latest}", 'r') as f:
                    checkpoint = json.load(f)
                
                self.start_episode = checkpoint['episode'] + 1
                self.episode_rewards = checkpoint['episode_rewards']
                self.episode_lengths = checkpoint['episode_lengths']
                self.best_reward = checkpoint['best_reward']
                
                # Load models
                episode = checkpoint['episode']
                self.load_models(f"models/checkpoint_{episode}")
                
                # Load replay buffer
                self.replay_buffer.load(f"buffers/replay_buffer_{episode}.pkl")
                
                print(f"Loaded checkpoint from episode {episode}")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
    
    def save_models(self, path_prefix: str):
        """Save all agent models"""
        for agent in self.agents:
            agent.save_models(path_prefix)
    
    def load_models(self, path_prefix: str):
        """Load all agent models"""
        for agent in self.agents:
            agent.load_models(path_prefix)
    
    def save_metrics(self):
        """Save training metrics to file"""
        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'best_reward': self.best_reward,
            'final_avg_reward': np.mean(self.episode_rewards[-100:]),
            'args': vars(self.args)
        }
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"logs/training_metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f)


def main():
    parser = argparse.ArgumentParser(description="MADDPG for Multi-Agent Formation Control")
    
    # Environment parameters
    parser.add_argument('--num_agents', type=int, default=3, help="Number of agents")
    parser.add_argument('--dt', type=float, default=0.1, help="Time step")
    parser.add_argument('--max_steps', type=int, default=300, help="Maximum steps per episode")
    
    # Training parameters
    parser.add_argument('--num_episodes', type=int, default=5000, help="Number of episodes")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size")
    parser.add_argument('--buffer_capacity', type=int, default=1000000, help="Replay buffer capacity")
    parser.add_argument('--lr_actor', type=float, default=1e-3, help="Actor learning rate")
    parser.add_argument('--lr_critic', type=float, default=1e-3, help="Critic learning rate")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor")
    parser.add_argument('--tau', type=float, default=0.01, help="Soft update coefficient")
    parser.add_argument('--noise_scale', type=float, default=0.1, help="Exploration noise scale")
    
    # System parameters
    parser.add_argument('--render', action='store_true', help="Enable visualization")
    parser.add_argument('--manual', action='store_true', help="Enable manual control mode")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--save_interval', type=int, default=100, help="Save checkpoint interval")
    
    args = parser.parse_args()
    
    # Create trainer and start training
    trainer = MADDPGTrainer(args)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        trainer.save_checkpoint(trainer.start_episode + trainer.episode_rewards[-1] if trainer.episode_rewards else 0)
        print("Final checkpoint saved")


if __name__ == "__main__":
    main()
=======
# main.py ���Ŀ��
from simulation.env import MultiAgentEnv
from agents.maddpg_agent import MADDPGAgent
from agents.replay_buffer import ReplayBuffer

env = MultiAgentEnv(num_agents=3, dt=0.1, max_steps=300)
obs_dim = 3                 # ÿ��������۲��Լ��� [x,y,��]
action_dim = 2
num_agents = 3

# Ϊÿ�������崴�������� Actor �͹���� Critic��ʵ�ʿɹ�������ṹ�������Ż�����
agents = [MADDPGAgent(obs_dim, action_dim, i) for i in range(num_agents)]
# ����ʽ Critic������ȫ��״̬+���ж�����
global_critic = Critic(global_state_dim=9, all_action_dim=6)
critic_optimizer = torch.optim.Adam(global_critic.parameters(), lr=1e-3)

buffer = ReplayBuffer(capacity=100000)

for episode in range(1000):
    obs_list = env.reset()           # ���� list of [x,y,��]
    episode_reward = [0]*num_agents
    done = False
    step = 0
    
    while not done:
        # 1. ÿ���������������۲�ѡ����
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(obs_list[i])  # [v,w] ��Χ [-1,1]
            # �� w ���ŵ������ɽ��ܷ�Χ [-2,2]
            action[1] = action[1] * 2.0
            actions.append(action)
        
        # 2. �뻷������
        next_obs_list, rewards, done = env.step(actions)
        
        # 3. ����ȫ��״̬����һȫ��״̬������������concatenate��
        global_state = np.concatenate([obs[:2] for obs in obs_list])  # ֻ��λ�úͽǶȣ���9ά
        next_global_state = np.concatenate([obs[:2] for obs in next_obs_list])
        
        # 4. �洢����
        buffer.add(obs_list, actions, rewards, next_obs_list, done)
        
        # 5. ѵ����ÿ����ÿN����
        if len(buffer) > batch_size:
            # ���� batch �����������������Actor�ͼ���ʽCritic
            # ��������¹�ʽ��MADDPG����ʵ�֣��˴��ԣ�
            pass
        
        obs_list = next_obs_list
        step += 1
        if step > 300: break
    
    print(f"Episode {episode}, total reward: {sum(episode_reward)}")
>>>>>>> 2b690f8e53359950940e1fe5342cb944a5b19dd3
