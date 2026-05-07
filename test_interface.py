from simulation.env import MultiAgentEnv
import numpy as np

env = MultiAgentEnv()
state = env.reset()
print("Initial state:", state)
for step in range(5):
    random_actions = [[np.random.uniform(-1,1), np.random.uniform(-2,2)] for _ in range(3)]
    next_state, rewards, done = env.step(random_actions)
    print(f"Step {step}: rewards={rewards}, done={done}")
print("接口数据流通畅。")