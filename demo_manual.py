import pygame
import numpy as np
from simulation.env import MultiAgentEnv
from simulation.renderer import PygameRenderer
from simulation.manual_controller import ManualController

def main():
    """Demonstrate manual control of vehicles"""
    
    print("Manual Control Demo")
    print("=" * 50)
    print("Controls:")
    print("  1/2/3: Select vehicle")
    print("  W/S: Increase/decrease speed")
    print("  A/D: Turn left/right")
    print("  4: Toggle team move mode")
    print("  M: Toggle auto/manual mode for selected vehicle")
    print("  R: Reset environment")
    print("  ESC: Quit")
    print("=" * 50)
    
    # Create environment
    env = MultiAgentEnv(num_agents=3, dt=0.1, max_steps=1000)
    env.reset()
    
    # Create renderer
    renderer = PygameRenderer(screen_width=1024, screen_height=768, 
                               world_width=30.0, world_height=30.0)
    
    # Create manual controller
    controller = ManualController(num_agents=3)
    
    # Enable manual mode for all agents initially
    for i in range(3):
        controller.set_manual_mode(i, True)
        env.mode_cmd[i] = 1
    
    running = True
    clock = pygame.time.Clock()
    
    while running:
        # Handle events
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    print("Environment reset")
                elif event.key == pygame.K_m:
                    # Toggle manual mode for selected vehicle
                    agent = controller.selected_agent
                    current = controller.manual_active[agent]
                    controller.set_manual_mode(agent, not current)
                    env.mode_cmd[agent] = 1 if not current else 0
                    mode = "MANUAL" if not current else "AUTO"
                    print(f"Vehicle {agent+1} switched to {mode} mode")
        
        # Get manual control commands
        commands = controller.handle_events(events)
        
        # Apply manual commands to environment
        for i in range(3):
            if controller.manual_active[i]:
                env.set_manual_control(i, commands['v'][i], commands['w'][i])
        
        # Step environment with current set_v/set_w
        actions = [[env.set_v[i], env.set_w[i]] for i in range(3)]
        states, rewards, done, info = env.step(actions)
        
        # Render
        render_state = {
            'states': states,
            'v_actual': info['v_actual'],
            'w_actual': info['w_actual'],
            'safety': info['safety'],
            'collision': info['collision'],
            'target_error': info['target_error'],
            'operation_mode': info['operation_mode'],
            'step_count': info['step_count'],
            'goal_position': (25.0, 25.0)
        }
        renderer.render(render_state)
        
        # Check if window should close
        if renderer.should_close():
            running = False
        
        # Control FPS
        clock.tick(30)
        
        if done:
            print("All vehicles reached goal! Resetting...")
            env.reset()
    
    renderer.close()
    print("Demo ended")


if __name__ == "__main__":
    main()