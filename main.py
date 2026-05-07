# main.py 核心框架
from simulation.env import MultiAgentEnv
from agents.maddpg_agent import MADDPGAgent
from agents.replay_buffer import ReplayBuffer

env = MultiAgentEnv(num_agents=3, dt=0.1, max_steps=300)
obs_dim = 3                 # 每个智能体观测自己的 [x,y,θ]
action_dim = 2
num_agents = 3

# 为每个智能体创建独立的 Actor 和共享的 Critic（实际可共用网络结构但独立优化器）
agents = [MADDPGAgent(obs_dim, action_dim, i) for i in range(num_agents)]
# 集中式 Critic（接收全局状态+所有动作）
global_critic = Critic(global_state_dim=9, all_action_dim=6)
critic_optimizer = torch.optim.Adam(global_critic.parameters(), lr=1e-3)

buffer = ReplayBuffer(capacity=100000)

for episode in range(1000):
    obs_list = env.reset()           # 返回 list of [x,y,θ]
    episode_reward = [0]*num_agents
    done = False
    step = 0
    
    while not done:
        # 1. 每个智能体根据自身观测选择动作
        actions = []
        for i, agent in enumerate(agents):
            action = agent.select_action(obs_list[i])  # [v,w] 范围 [-1,1]
            # 将 w 缩放到环境可接受范围 [-2,2]
            action[1] = action[1] * 2.0
            actions.append(action)
        
        # 2. 与环境交互
        next_obs_list, rewards, done = env.step(actions)
        
        # 3. 构造全局状态和下一全局状态（所有智能体concatenate）
        global_state = np.concatenate([obs[:2] for obs in obs_list])  # 只用位置和角度，共9维
        next_global_state = np.concatenate([obs[:2] for obs in next_obs_list])
        
        # 4. 存储经验
        buffer.add(obs_list, actions, rewards, next_obs_list, done)
        
        # 5. 训练（每步或每N步）
        if len(buffer) > batch_size:
            # 采样 batch 并更新所有智能体的Actor和集中式Critic
            # （具体更新公式按MADDPG论文实现，此处略）
            pass
        
        obs_list = next_obs_list
        step += 1
        if step > 300: break
    
    print(f"Episode {episode}, total reward: {sum(episode_reward)}")