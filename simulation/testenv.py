
from env import MultiAgentEnv

env = MultiAgentEnv()

# test reset
states = env.reset()
print("first:", states)

# step：3 agents all go（v=1.0，theta=0）
actions = [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
next_states, rewards, done = env.step(actions)

print("next:", next_states)
print("reward:", rewards)
print("finish?:", done)

print("\nresult")
for i in range(3):
    old_x = states[i][0]
    new_x = next_states[i][0]
    print(f"3Agent{i+1}: x from {old_x:.2f} to {new_x:.2f} (bianhua {new_x-old_x:.2f})")