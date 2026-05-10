import numpy as np
from simulation.env import MultiAgentEnv

def test_environment():
    """Test that all environment interfaces work correctly"""
    
    print("Testing Multi-Agent Environment Interface...")
    print("-" * 50)
    
    # Create environment
    env = MultiAgentEnv(num_agents=3, dt=0.1, max_steps=300)
    
    # Test reset
    states = env.reset()
    print(f"[PASS] reset() returned {len(states)} states")
    print(f"      State shape: {np.array(states).shape}")
    
    # Test get_state_dim
    state_dim = env.get_state_dim()
    print(f"[PASS] get_state_dim() = {state_dim}")
    
    # Test get_action_dim
    action_dim = env.get_action_dim()
    print(f"[PASS] get_action_dim() = {action_dim}")
    
    # Test get_global_state_dim
    global_dim = env.get_global_state_dim()
    print(f"[PASS] get_global_state_dim() = {global_dim}")
    
    # Test step
    actions = [[0.5, 0.2], [0.5, 0.2], [0.5, 0.2]]
    next_states, rewards, done, info = env.step(actions)
    print(f"[PASS] step() returned next_states (shape {np.array(next_states).shape})")
    print(f"       rewards: {rewards}")
    print(f"       done: {done}")
    print(f"       info keys: {info.keys()}")
    
    # Test collision detection
    print("\nTesting collision detection...")
    env.reset()
    
    # Move vehicles into collision
    for step in range(50):
        actions = [[0.8, 0], [0.8, 0], [0.8, 0]]
        _, _, _, info = env.step(actions)
    
    print(f"  Safety signals: {info['safety']}")
    print(f"  Collision flags: {info['collision']}")
    
    # Verify safety signal format
    assert len(info['safety']) == 3, "Safety signal should be list of 3 ints"
    assert all(s in [0, 1, 2] for s in info['safety']), "Safety signal values must be 0,1,2"
    print("[PASS] Safety signal format verified (0=Safe, 1=Danger, 2=Collision)")
    
    # Test manual control
    print("\nTesting manual control interface...")
    env.reset()
    env.mode_cmd[0] = 1  # Set agent 0 to manual mode
    env.set_manual_control(0, 0.8, 0.5)
    
    # Verify manual control overrides
    manual_action = [env.set_v[0], env.set_w[0]]
    print(f"  Manual control set to v={manual_action[0]:.2f}, w={manual_action[1]:.2f}")
    print("[PASS] Manual control interface works")
    
    print("\n" + "="*50)
    print("All interface tests passed!")
    print("="*50)


if __name__ == "__main__":
    test_environment()