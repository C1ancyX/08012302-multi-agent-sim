import pygame
import numpy as np
from typing import Dict, List, Optional

class ManualController:
    """
    Manual controller for vehicle operation
    Implements interrupt-based manual mode override
    """
    
    def __init__(self, num_agents: int = 3):
        self.num_agents = num_agents
        self.selected_agent = 0
        self.team_move = False  # Move all vehicles together
        self.manual_active = np.zeros(num_agents, dtype=bool)
        
        # Speed and steering values
        self.current_v = np.zeros(num_agents)
        self.current_w = np.zeros(num_agents)
        
        # Speed limits
        self.max_v = 1.0
        self.max_w = 2.0
        self.step_v = 0.05
        self.step_w = 0.1
    
    def handle_events(self, events: List[pygame.event.Event]) -> Dict:
        """
        Handle keyboard events for manual control
        Returns: dict with control commands for each agent
        """
        controls = {'v': self.current_v.copy(), 'w': self.current_w.copy()}
        
        for event in events:
            if event.type != pygame.KEYDOWN:
                continue
            
            # Vehicle selection
            if event.key == pygame.K_1:
                self.selected_agent = 0
                self.team_move = False
            elif event.key == pygame.K_2:
                self.selected_agent = 1
                self.team_move = False
            elif event.key == pygame.K_3:
                self.selected_agent = 2
                self.team_move = False
            elif event.key == pygame.K_4:
                self.team_move = not self.team_move
                if self.team_move:
                    print(f"Team move mode: ON")
                else:
                    print(f"Team move mode: OFF")
            
            # Speed control
            elif event.key == pygame.K_w:
                self._adjust_speed(+self.step_v)
            elif event.key == pygame.K_s:
                self._adjust_speed(-self.step_v)
            
            # Steering control
            elif event.key == pygame.K_a:
                self._adjust_steering(+self.step_w)
            elif event.key == pygame.K_d:
                self._adjust_steering(-self.step_w)
        
        # Apply controls
        if self.team_move:
            # Apply to all agents
            for i in range(self.num_agents):
                controls['v'][i] = self.current_v[self.selected_agent]
                controls['w'][i] = self.current_w[self.selected_agent]
        else:
            # Apply only to selected agent
            controls['v'][self.selected_agent] = self.current_v[self.selected_agent]
            controls['w'][self.selected_agent] = self.current_w[self.selected_agent]
        
        return controls
    
    def _adjust_speed(self, delta: float):
        """Adjust speed for selected agent"""
        if self.team_move:
            agent = self.selected_agent
            self.current_v[agent] = np.clip(self.current_v[agent] + delta, -self.max_v, self.max_v)
            print(f"Team speed: {self.current_v[agent]:.2f}")
        else:
            agent = self.selected_agent
            self.current_v[agent] = np.clip(self.current_v[agent] + delta, -self.max_v, self.max_v)
            print(f"Vehicle {agent+1} speed: {self.current_v[agent]:.2f}")
    
    def _adjust_steering(self, delta: float):
        """Adjust steering for selected agent"""
        if self.team_move:
            agent = self.selected_agent
            self.current_w[agent] = np.clip(self.current_w[agent] + delta, -self.max_w, self.max_w)
            print(f"Team angular speed: {self.current_w[agent]:.2f}")
        else:
            agent = self.selected_agent
            self.current_w[agent] = np.clip(self.current_w[agent] + delta, -self.max_w, self.max_w)
            print(f"Vehicle {agent+1} angular speed: {self.current_w[agent]:.2f}")
    
    def set_manual_mode(self, agent_id: int, active: bool):
        """Set manual mode for a specific agent"""
        self.manual_active[agent_id] = active
        status = "MANUAL" if active else "AUTO"
        print(f"Vehicle {agent_id+1} mode: {status}")
    
    def get_manual_commands(self) -> Dict:
        """Get current manual commands"""
        commands = []
        for i in range(self.num_agents):
            if self.manual_active[i]:
                commands.append([self.current_v[i], self.current_w[i]])
            else:
                commands.append(None)
        return {'commands': commands, 'active': self.manual_active.copy()}
    
    def reset(self):
        """Reset manual controller state"""
        self.selected_agent = 0
        self.team_move = False
        self.manual_active.fill(False)
        self.current_v.fill(0.0)
        self.current_w.fill(0.0)