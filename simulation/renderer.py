import pygame
import numpy as np
from typing import List, Dict, Optional

class PygameRenderer:
    # COLOURS
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 50, 50)
    GREEN = (50, 255, 50)
    YELLOW = (255, 255, 50)
    BLUE = (50, 50, 255)
    CYAN = (50, 255, 255)
    MAGENTA = (255, 50, 255)
    ORANGE = (255, 165, 0)
    GRAY = (128, 128, 128)
    DARK_GRAY = (64, 64, 64)
    BROWN = (139, 69, 19)
    LIGHT_BROWN = (160, 82, 45)
    DARK_BROWN = (101, 67, 33)
    
    SAFE_COLOR = GREEN
    DANGER_COLOR = YELLOW
    COLLISION_COLOR = RED

    def __init__(self, screen_width: int = 1024, screen_height: int = 768,
                 world_width: float = 30.0, world_height: float = 30.0):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.world_width = world_width
        self.world_height = world_height
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Multi-Agent Formation Control - MADDPG")
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        self.clock = pygame.time.Clock()
        self.target_fps = 30
        self.camera_x = 0
        self.camera_y = 0
        self.vehicle_colors = [self.BLUE, self.CYAN, self.MAGENTA]
        self.pixels_per_meter_x = screen_width / world_width
        self.pixels_per_meter_y = screen_height / world_height

    def world_to_screen(self, x: float, y: float) -> tuple:
        screen_x = int(x * self.pixels_per_meter_x)
        screen_y = int(self.screen_height - y * self.pixels_per_meter_y)
        return (screen_x, screen_y)

    def render(self, env_state: Dict):
        self.screen.fill(self.BLACK)
        if 'states' in env_state and len(env_state['states']) > 0:
            leader_x, leader_y = env_state['states'][0][0], env_state['states'][0][1]
            self.camera_x = leader_x - self.world_width / 2
            self.camera_y = leader_y - self.world_height / 2
            self.camera_x = np.clip(self.camera_x, 0, self.world_width - self.world_width)
            self.camera_y = np.clip(self.camera_y, 0, self.world_height - self.world_height)

        self._draw_grid()
        if 'obstacles' in env_state:
            self._draw_obstacles(env_state['obstacles'])
        if 'goal_position' in env_state:
            self._draw_goal(env_state['goal_position'])
        if 'states' in env_state:
            for i, state in enumerate(env_state['states']):
                x, y, theta = state[:3]
                safety = env_state.get('safety', [0])[i] if i < len(env_state.get('safety', [])) else 0
                self._draw_vehicle(x, y, theta, self.vehicle_colors[i % len(self.vehicle_colors)], safety)
        if 'states' in env_state and 'collision_obstacle' in env_state:
            for i, state in enumerate(env_state['states']):
                if i < len(env_state['collision_obstacle']) and env_state['collision_obstacle'][i]:
                    x, y, _ = state[:3]
                    self._draw_collision_warning(x, y)
        self._draw_ui_panel(env_state)
        pygame.display.flip()
        self.clock.tick(self.target_fps)

    def _draw_grid(self):
        grid_spacing_m = 2.0
        grid_spacing_px = int(grid_spacing_m * self.pixels_per_meter_x)
        for x in range(0, self.screen_width, grid_spacing_px):
            pygame.draw.line(self.screen, self.DARK_GRAY, (x, 0), (x, self.screen_height))
        for y in range(0, self.screen_height, grid_spacing_px):
            pygame.draw.line(self.screen, self.DARK_GRAY, (0, y), (self.screen_width, y))

    def _draw_vehicle(self, x: float, y: float, theta: float, color: tuple, safety: int):
        if safety == 2:
            color = self.COLLISION_COLOR
        elif safety == 1:
            color = self.DANGER_COLOR
        else:
            color = self.GREEN if safety == 0 else color

        length_px = int(0.4 * self.pixels_per_meter_x)
        width_px = int(0.2 * self.pixels_per_meter_y)
        cx, cy = self.world_to_screen(x, y)
        rect_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        rect_surface.fill(color)
        rotated = pygame.transform.rotate(rect_surface, np.degrees(theta))
        rect = rotated.get_rect(center=(cx, cy))
        self.screen.blit(rotated, rect)
        arrow_length = 20
        end_x = cx + arrow_length * np.cos(theta)
        end_y = cy - arrow_length * np.sin(theta)
        pygame.draw.line(self.screen, self.WHITE, (cx, cy), (end_x, end_y), 2)

    def _draw_goal(self, goal_position: tuple):
        gx, gy = goal_position
        cx, cy = self.world_to_screen(gx, gy)
        radius = int(1.5 * self.pixels_per_meter_x)
        pygame.draw.circle(self.screen, self.GREEN, (cx, cy), radius, 2)
        pygame.draw.circle(self.screen, self.GREEN, (cx, cy), 5)

    def _draw_obstacles(self, obstacles: List[Dict]):
        for obs in obstacles:
            cx, cy = self.world_to_screen(obs['x'], obs['y'])
            radius = int(obs['radius'] * self.pixels_per_meter_x)
            pygame.draw.circle(self.screen, self.BROWN, (cx, cy), radius)
            pygame.draw.circle(self.screen, self.LIGHT_BROWN, (cx, cy), radius - 2)
            pygame.draw.circle(self.screen, self.DARK_BROWN, (cx, cy), radius - 4, 2)

    def _draw_collision_warning(self, x: float, y: float):
        cx, cy = self.world_to_screen(x, y)
        warning_surface = self.font_medium.render("!", True, self.RED)
        warning_rect = warning_surface.get_rect(center=(cx, cy - 25))
        self.screen.blit(warning_surface, warning_rect)
        pygame.draw.circle(self.screen, self.RED, (cx, cy), 20, 3)

    def _draw_ui_panel(self, env_state: Dict):
        panel_x = 10
        panel_y = 10
        title = self.font_large.render("Multi-Agent System Status", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 5))
        y_offset = panel_y + 35

        step_text = self.font_medium.render(f"Step: {env_state.get('step_count', 0)}", True, self.WHITE)
        self.screen.blit(step_text, (panel_x + 10, y_offset))
        y_offset += 25

        legend_texts = [("Green: Safe", self.SAFE_COLOR), ("Yellow: Danger", self.DANGER_COLOR), ("Red: Collision", self.COLLISION_COLOR)]
        for text, color in legend_texts:
            legend = self.font_small.render(text, True, color)
            self.screen.blit(legend, (panel_x + 10, y_offset))
            y_offset += 18
        y_offset += 10

        states = env_state.get('states', [])
        v_actual = env_state.get('v_actual', [])
        w_actual = env_state.get('w_actual', [])
        safety = env_state.get('safety', [])
        target_error = env_state.get('target_error', [0.0] * len(states))

        for i in range(len(states)):
            y_offset += 5
            header = self.font_medium.render(f"--- Vehicle {i+1} ---", True, self.vehicle_colors[i])
            self.screen.blit(header, (panel_x + 10, y_offset))
            y_offset += 22

            x, y, theta = states[i][:3]
            theta_deg = np.degrees(theta)
            pos_text = self.font_small.render(f"Pos: ({x:.1f}, {y:.1f}), Theta: {theta_deg:.0f} deg", True, self.WHITE)
            self.screen.blit(pos_text, (panel_x + 10, y_offset))
            y_offset += 16

            vel_text = self.font_small.render(f"V: {v_actual[i]:.2f} m/s, W: {w_actual[i]:.2f} rad/s", True, self.WHITE)
            self.screen.blit(vel_text, (panel_x + 10, y_offset))
            y_offset += 16


            safe_val = safety[i] if i < len(safety) else 0
            safety_str = ["Safe", "Danger", "Collision"][safe_val if safe_val < 3 else 0]
            status_text = self.font_small.render(f"Status: {safety_str}", True, self.WHITE)
            self.screen.blit(status_text, (panel_x + 10, y_offset))
            y_offset += 16

            err = target_error[i] if i < len(target_error) else 0.0
            error_text = self.font_small.render(f"Target Error: {err:.2f} m", True, self.WHITE)
            self.screen.blit(error_text, (panel_x + 10, y_offset))
            y_offset += 16

            signal_color = [self.GREEN, self.YELLOW, self.RED][safe_val]
            signal_text = self.font_small.render(f"Signal Safety: {safe_val}", True, signal_color)
            self.screen.blit(signal_text, (panel_x + 10, y_offset))
            y_offset += 20

    def close(self):
        pygame.quit()

    def should_close(self) -> bool:
        for event in pygame.event.get(pygame.QUIT):
            return True
        return False