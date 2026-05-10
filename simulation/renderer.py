import pygame
import numpy as np
from typing import List, Dict, Optional

class PygameRenderer:
    """
    Pygame-based visualizer for multi-agent environment
    Resolution: 1024x768, min 5 FPS
    """
    
    # Color definitions
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
    
    # Safety colors
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
        
        # Create screen
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Multi-Agent Formation Control - MADDPG")
        
        # Fonts
        self.font_small = pygame.font.Font(None, 16)
        self.font_medium = pygame.font.Font(None, 20)
        self.font_large = pygame.font.Font(None, 24)
        
        # Clock for FPS control
        self.clock = pygame.time.Clock()
        self.target_fps = 30
        
        # Camera offset (follow leader)
        self.camera_x = 0
        self.camera_y = 0
        
        # Vehicle colors
        self.vehicle_colors = [self.BLUE, self.CYAN, self.MAGENTA]
        
        # Scale factors
        self.pixels_per_meter_x = screen_width / world_width
        self.pixels_per_meter_y = screen_height / world_height
        
    def world_to_screen(self, x: float, y: float) -> tuple:
        """Convert world coordinates to screen coordinates"""
        screen_x = int(x * self.pixels_per_meter_x)
        screen_y = int(self.screen_height - y * self.pixels_per_meter_y)
        return (screen_x, screen_y)
    
    def render(self, env_state: Dict):
        """
        Render the environment
        env_state should contain:
        - states: list of [x, y, theta] for each agent
        - v_actual, w_actual: velocity and angular velocity
        - safety: safety signals (0,1,2)
        - collision: collision flags
        - target_error: distance to goal
        - operation_mode: auto/manual mode
        - step_count: current step
        - goal_position: (x, y)
        """
        self.screen.fill(self.BLACK)
        
        # Update camera to follow leader (agent 0)
        if 'states' in env_state and len(env_state['states']) > 0:
            leader_x, leader_y = env_state['states'][0][0], env_state['states'][0][1]
            self.camera_x = leader_x - self.world_width / 2
            self.camera_y = leader_y - self.world_height / 2
            self.camera_x = np.clip(self.camera_x, 0, self.world_width - self.world_width)
            self.camera_y = np.clip(self.camera_y, 0, self.world_height - self.world_height)
        
        # Draw grid
        self._draw_grid()
        
        # Draw goal
        if 'goal_position' in env_state:
            self._draw_goal(env_state['goal_position'])
        
        # Draw vehicles
        if 'states' in env_state:
            for i, state in enumerate(env_state['states']):
                safety = env_state.get('safety', [0])[i] if i < len(env_state.get('safety', [])) else 0
                self._draw_vehicle(state[0], state[1], state[2], 
                                 self.vehicle_colors[i % len(self.vehicle_colors)],
                                 safety)
        
        # Draw UI panel
        self._draw_ui_panel(env_state)
        
        pygame.display.flip()
        self.clock.tick(self.target_fps)
    
    def _draw_grid(self):
        """Draw coordinate grid"""
        grid_spacing_m = 2.0
        grid_spacing_px = int(grid_spacing_m * self.pixels_per_meter_x)
        
        for x in range(0, self.screen_width, grid_spacing_px):
            pygame.draw.line(self.screen, self.DARK_GRAY, (x, 0), (x, self.screen_height))
        
        for y in range(0, self.screen_height, grid_spacing_px):
            pygame.draw.line(self.screen, self.DARK_GRAY, (0, y), (self.screen_width, y))
    
    def _draw_vehicle(self, x: float, y: float, theta: float, color: tuple, safety: int):
        """Draw a single vehicle as rectangle with orientation"""
        # Select color based on safety
        if safety == 2:
            color = self.COLLISION_COLOR
        elif safety == 1:
            color = self.DANGER_COLOR
        else:
            color = self.GREEN if safety == 0 else color
        
        # Vehicle dimensions in pixels
        length_px = int(0.4 * self.pixels_per_meter_x)
        width_px = int(0.2 * self.pixels_per_meter_y)
        
        # Get screen coordinates
        cx, cy = self.world_to_screen(x, y)
        
        # Create rotated rectangle
        rect_surface = pygame.Surface((length_px, width_px), pygame.SRCALPHA)
        rect_surface.fill(color)
        
        # Rotate
        rotated = pygame.transform.rotate(rect_surface, -np.degrees(theta))
        rect = rotated.get_rect(center=(cx, cy))
        
        self.screen.blit(rotated, rect)
        
        # Draw direction arrow
        arrow_length = 20
        end_x = cx + arrow_length * np.cos(theta)
        end_y = cy - arrow_length * np.sin(theta)
        pygame.draw.line(self.screen, self.WHITE, (cx, cy), (end_x, end_y), 2)
    
    def _draw_goal(self, goal_position: tuple):
        """Draw goal area"""
        gx, gy = goal_position
        cx, cy = self.world_to_screen(gx, gy)
        radius = int(1.5 * self.pixels_per_meter_x)
        
        # Draw goal circle
        pygame.draw.circle(self.screen, self.GREEN, (cx, cy), radius, 2)
        pygame.draw.circle(self.screen, self.GREEN, (cx, cy), 5)
    
    def _draw_ui_panel(self, env_state: Dict):
        """Draw UI panel with vehicle information"""
        panel_x = 10
        panel_y = 10
        panel_width = 300
        panel_height = self.screen_height - 20
        panel_height = min(panel_height, 600)
        
        # Background panel
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill(self.BLACK)
        self.screen.blit(panel_surface, (panel_x, panel_y))
        
        # Title
        title = self.font_large.render("Multi-Agent System Status", True, self.WHITE)
        self.screen.blit(title, (panel_x + 10, panel_y + 5))
        
        y_offset = panel_y + 35
        
        # Step count
        step_text = self.font_medium.render(f"Step: {env_state.get('step_count', 0)}", True, self.WHITE)
        self.screen.blit(step_text, (panel_x + 10, y_offset))
        y_offset += 25
        
        # Legend
        legend_texts = [
            ("Green: Safe", self.SAFE_COLOR),
            ("Yellow: Danger", self.DANGER_COLOR),
            ("Red: Collision", self.COLLISION_COLOR)
        ]
        for text, color in legend_texts:
            legend = self.font_small.render(text, True, color)
            self.screen.blit(legend, (panel_x + 10, y_offset))
            y_offset += 18
        
        y_offset += 10
        
        # Vehicle information
        states = env_state.get('states', [])
        v_actual = env_state.get('v_actual', [])
        w_actual = env_state.get('w_actual', [])
        safety = env_state.get('safety', [])
        operation_mode = env_state.get('operation_mode', [])
        target_error = env_state.get('target_error', [])
        
        for i in range(len(states)):
            y_offset += 5
            # Vehicle header
            header = self.font_medium.render(f"--- Vehicle {i+1} ---", True, self.vehicle_colors[i])
            self.screen.blit(header, (panel_x + 10, y_offset))
            y_offset += 22
            
            # Position - fixed line without special characters
            x, y, theta = states[i]
            theta_deg = np.degrees(theta)
            pos_text = self.font_small.render(f"Pos: ({x:.1f}, {y:.1f}), Theta: {theta_deg:.0f} deg", True, self.WHITE)
            self.screen.blit(pos_text, (panel_x + 10, y_offset))
            y_offset += 16
            
            # Velocity
            vel_text = self.font_small.render(f"V: {v_actual[i]:.2f} m/s, W: {w_actual[i]:.2f} rad/s", True, self.WHITE)
            self.screen.blit(vel_text, (panel_x + 10, y_offset))
            y_offset += 16
            
            # Mode and status
            mode_str = "Manual" if operation_mode[i] == 1 else "Auto"
            safety_str = ["Safe", "Danger", "Collision"][safety[i] if safety[i] < 3 else 0]
            status_text = self.font_small.render(f"Mode: {mode_str} | Status: {safety_str}", True, self.WHITE)
            self.screen.blit(status_text, (panel_x + 10, y_offset))
            y_offset += 16
            
            # Target error
            error_text = self.font_small.render(f"Target Error: {target_error[i]:.2f} m", True, self.WHITE)
            self.screen.blit(error_text, (panel_x + 10, y_offset))
            y_offset += 16
            
            # Safety signal
            safety_val = env_state.get('safety', [0])[i] if i < len(env_state.get('safety', [])) else 0
            signal_color = [self.GREEN, self.YELLOW, self.RED][safety_val]
            signal_text = self.font_small.render(f"Signal Safety: {safety_val}", True, signal_color)
            self.screen.blit(signal_text, (panel_x + 10, y_offset))
            y_offset += 20
        
        # Control instructions
        y_offset += 10
        controls = [
            "Controls (Manual Mode):",
            "1/2/3: Select vehicle",
            "W/S: Speed +/-",
            "A/D: Steering +/-",
            "4: Toggle team move",
            "M: Toggle Auto/Manual",
            "R: Reset",
            "ESC: Quit"
        ]
        for control in controls:
            ctrl_text = self.font_small.render(control, True, self.GRAY)
            self.screen.blit(ctrl_text, (panel_x + 10, y_offset))
            y_offset += 16
    
    def close(self):
        """Close pygame window"""
        pygame.quit()
    
    def should_close(self) -> bool:
        """Check if window should close"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return True
        return False