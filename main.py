import numpy as np
import time
import os
from simulation.env import MultiAgentEnv
from simulation.renderer import PygameRenderer
from evaluation.metrics import MetricsCollector

def main():
    #logs
    os.makedirs("logs", exist_ok=True)

    env = MultiAgentEnv(num_agents=3, dt=0.1, max_steps=1000,
                        enable_obstacles=True, num_obstacles=10,
                        goal_position=(25.,25.))
    renderer = PygameRenderer()

    #episode DEFAULT:1
    num_episodes = 1
    for episode in range(num_episodes):
        print(f"\nEpisode {episode}:")
        obs = env.reset()
        done = False
        step = 0

        #collector for metrics
        collector = MetricsCollector(num_agents=3)

        while not done:
            obs, rewards, done, info = env.step()

            #CREATE states_dict
            states_dict = [
                {'x': r.x, 'y': r.y, 'theta': r.theta, 'v': r.v, 'w': r.w}
                for r in env.robots
            ]
            collector.update(step, states_dict, info)

            leader = env.robot2
            print(f"Step {step}: Leader (Car2) v={leader.v:.2f}, w={leader.w:.2f}")

            #render
            render_state = {
                'states': [[r.x, r.y, r.theta] for r in env.robots],
                'v_actual': info['v_actual'],
                'w_actual': info['w_actual'],
                'safety': info['safety'],
                'collision': info['collision'],
                'step_count': step,
                'goal_position': (25.,25.),
                'obstacles': env.get_obstacles_for_rendering(),
                'global_path': env.get_global_path_for_rendering()
            }
            renderer.render(render_state)
            if renderer.should_close():
                return
            step += 1
            time.sleep(0.02)  #simulation speed

        #Episode over, compute and print metrics
        final_metrics = collector.compute_final_metrics()
        for key, value in final_metrics.items():
            print(f"{key}: {value}")
        collector.save_to_file(final_metrics, f"logs/episode_{episode}_metrics.json")

    renderer.close()

if __name__ == "__main__":
    main()