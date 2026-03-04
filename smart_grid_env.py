"""
smart_grid_env.py

A Smart Grid Energy Management RL Environment.
The agent manages energy sources, storage, and grid trading to meet demand.
Extremely challenging due to renewables uncertainty, battery degradation, and price spikes.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import math

class SmartGridEnv(gym.Env):
    """
    Observation (13 variables):
    0: Current demand (MW)
    1: Solar generation (MW)
    2: Wind generation (MW)
    3: Grid price ($/MWh)
    4: Battery charge level (MWh)
    5: Battery health (0-1, degrading over time)
    6: Diesel generator available? (0/1)
    7: Time of day (hour/24)
    8: Day of year (sin component)
    9: Day of year (cos component)
    10: Forecast error demand (MW)
    11: Forecast error solar (MW)
    12: Emergency load shedding (0/1)
    
    Action space (9 discrete actions):
    0: Do nothing
    1: Charge battery from grid
    2: Discharge battery to grid
    3: Charge from solar only
    4: Discharge to load
    5: Start diesel generator
    6: Stop diesel generator
    7: Buy from grid
    8: Sell to grid
    """
    
    def __init__(self, max_steps=8760):  # 1 year of hourly steps
        super().__init__()
        self.max_steps = max_steps
        self.step_count = 0
        
        # Physical limits
        self.battery_capacity = 100  # MWh
        self.battery_max_charge = 20  # MW per hour
        self.battery_max_discharge = 20  # MW per hour
        self.diesel_capacity = 50  # MW
        self.diesel_fuel_cost = 100  # $/MWh
        self.diesel_start_cost = 500  # $ per start
        self.diesel_cooldown = 24  # hours between starts
        
        # State variables
        self.demand = 0
        self.solar = 0
        self.wind = 0
        self.grid_price = 0
        self.battery_charge = 50
        self.battery_health = 1.0
        self.diesel_available = True
        self.diesel_cooldown_counter = 0
        self.diesel_running = False
        self.hour = 0
        self.day = 0
        self.forecast_error_demand = 0
        self.forecast_error_solar = 0
        self.forecast_error_wind = 0
        self.load_shedding = 0
        self.cumulative_cost = 0
        self.battery_cycles = 0
        
        # Hidden degradation parameters
        self.hidden_degradation_rate = 0.00005  # per cycle
        self.hidden_temp_effect = 0.0
        self.hidden_forecast_bias = 0.0
        
        # Observation space: 13 continuous values
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -50, 0, 0, 0, 0, -1, -1, -50, -50, 0]),
            high=np.array([500, 200, 200, 500, 100, 1, 1, 24, 1, 1, 50, 50, 1]),
            dtype=np.float32
        )
        
        # Action space: 9 discrete actions
        self.action_space = spaces.Discrete(9)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.battery_charge = 50
        self.battery_health = 1.0
        self.diesel_available = True
        self.diesel_cooldown_counter = 0
        self.diesel_running = False
        self.hour = 0
        self.day = 0
        self.cumulative_cost = 0
        self.battery_cycles = 0
        self.hidden_temp_effect = random.uniform(-0.1, 0.1)
        self.hidden_forecast_bias = random.uniform(-5, 5)
        
        # Generate initial state
        self._update_environment()
        obs = self._get_obs()
        info = {}
        return obs, info
    
    def step(self, action):
        self.step_count += 1
        self.hour += 1
        
        # Update day and hour
        if self.hour >= 24:
            self.hour = 0
            self.day += 1
        
        # Store previous state for calculations
        prev_diesel_running = self.diesel_running
        
        # Apply action
        cost = self._apply_action(action)
        
        # Update environment (demand, solar, wind, prices)
        self._update_environment()
        
        # Calculate power balance
        total_generation = self.solar + self.wind
        if self.diesel_running:
            total_generation += self.diesel_capacity
        
        # Battery operations (if action involved battery)
        self._update_battery(action)
        
        # Grid transactions
        grid_import = 0
        grid_export = 0
        unmet_demand = 0
        
        net_load = self.demand - total_generation
        
        if net_load > 0:
            # Need to import or discharge battery
            if action == 2 or action == 4:  # Discharge actions
                discharge = min(self.battery_charge, self.battery_max_discharge, net_load)
                self.battery_charge -= discharge
                net_load -= discharge
                self.battery_cycles += 0.5
            
            if net_load > 0:
                # Still need more - import from grid
                grid_import = net_load
                cost += grid_import * self.grid_price
                
                if grid_import > 50:  # Grid capacity limit
                    unmet_demand = grid_import - 50
                    grid_import = 50
                    self.load_shedding = 1
                else:
                    self.load_shedding = 0
            else:
                self.load_shedding = 0
                
        elif net_load < 0:
            # Excess generation - can export or charge battery
            excess = -net_load
            
            if action == 1 or action == 3:  # Charge actions
                charge = min(self.battery_capacity - self.battery_charge, 
                           self.battery_max_charge, excess)
                self.battery_charge += charge
                excess -= charge
                self.battery_cycles += 0.3
            
            if excess > 0:
                # Export to grid
                grid_export = excess
                cost -= grid_export * self.grid_price * 0.9  # Sell at 90% of price
        
        # Update diesel cooldown
        if not self.diesel_running and self.diesel_cooldown_counter > 0:
            self.diesel_cooldown_counter -= 1
        elif self.diesel_cooldown_counter == 0:
            self.diesel_available = True
        
        # Battery degradation
        self.battery_health -= self.hidden_degradation_rate * self.battery_cycles
        self.battery_health = max(0.5, min(1.0, self.battery_health))
        
        # Temperature effect on battery (hidden)
        self.hidden_temp_effect += random.uniform(-0.02, 0.02)
        self.hidden_temp_effect = max(-0.3, min(0.3, self.hidden_temp_effect))
        
        # Calculate reward
        reward = self._calculate_reward(cost, unmet_demand)
        
        # Check termination
        terminated = False
        if self.battery_health < 0.5:
            terminated = True  # Battery failed
        if self.load_shedding > 5:  # Too much load shedding
            terminated = True
            
        truncated = self.step_count >= self.max_steps
        
        obs = self._get_obs()
        info = {
            'cost': cost,
            'cumulative_cost': self.cumulative_cost,
            'battery_health': self.battery_health,
            'unmet_demand': unmet_demand,
            'diesel_running': self.diesel_running
        }
        
        return obs, reward, terminated, truncated, info
    
    def _apply_action(self, action):
        """Apply action and return immediate cost"""
        cost = 0
        
        if action == 5:  # Start diesel
            if self.diesel_available and not self.diesel_running:
                self.diesel_running = True
                self.diesel_available = False
                self.diesel_cooldown_counter = self.diesel_cooldown
                cost += self.diesel_start_cost
                
        elif action == 6:  # Stop diesel
            if self.diesel_running:
                self.diesel_running = False
                
        elif action == 7:  # Buy from grid (immediate)
            # Handled in balance calculation
            pass
            
        elif action == 8:  # Sell to grid (immediate)
            # Handled in balance calculation
            pass
            
        # Other actions (battery) handled in _update_battery
        
        return cost
    
    def _update_battery(self, action):
        """Update battery based on action"""
        if action == 1:  # Charge from grid
            charge = min(self.battery_capacity - self.battery_charge, 
                        self.battery_max_charge)
            self.battery_charge += charge
            
        elif action == 2:  # Discharge to grid
            discharge = min(self.battery_charge, self.battery_max_discharge)
            self.battery_charge -= discharge
            
        elif action == 3:  # Charge from solar only
            if self.solar > 0:
                charge = min(self.battery_capacity - self.battery_charge,
                           self.battery_max_charge, self.solar * 0.5)
                self.battery_charge += charge
                
        elif action == 4:  # Discharge to load
            # Handled in balance
            pass
    
    def _update_environment(self):
        """Update demand, solar, wind, and prices with realistic patterns"""
        # Time-based demand pattern
        base_demand = 200 + 50 * math.sin(2 * math.pi * self.hour / 24 - math.pi/2)
        
        # Seasonal pattern
        season = 2 * math.pi * self.day / 365
        seasonal_factor = 1.0 + 0.2 * math.sin(season)
        
        # Random variations
        random_variation = random.uniform(-20, 20)
        
        self.demand = base_demand * seasonal_factor + random_variation
        self.demand = max(100, min(400, self.demand))
        
        # Solar generation (daytime only)
        if 6 <= self.hour <= 18:
            solar_base = 150 * math.sin(math.pi * (self.hour - 6) / 12)
            # Cloud cover effect
            cloud_factor = random.uniform(0.3, 1.0)
            self.solar = solar_base * cloud_factor
        else:
            self.solar = random.uniform(0, 10)  # Nightime residual
            
        # Wind generation (chaotic)
        wind_base = 100 + 50 * math.sin(2 * math.pi * self.day / 7) + 30 * math.sin(2 * math.pi * self.hour / 12)
        wind_variation = random.uniform(-40, 40)
        self.wind = max(0, wind_base + wind_variation)
        
        # Grid price (volatile)
        price_base = 50 + 30 * math.sin(2 * math.pi * self.hour / 24)
        price_spike = 200 if random.random() < 0.02 else 0  # 2% chance of price spike
        self.grid_price = price_base + price_spike + random.uniform(-10, 10)
        
        # Forecast errors (hidden)
        self.forecast_error_demand = random.uniform(-30, 30) + self.hidden_forecast_bias
        self.forecast_error_solar = random.uniform(-20, 20) * (1 + 0.5 * self.hidden_temp_effect)
        self.forecast_error_wind = random.uniform(-30, 30)
    
    def _calculate_reward(self, cost, unmet_demand):
        """Calculate reward based on cost and reliability"""
        # Base negative reward from cost
        reward = -cost / 1000  # Scale down cost
        
        # Heavy penalty for unmet demand
        reward -= unmet_demand * 10
        
        # Penalty for poor battery health
        reward -= (1 - self.battery_health) * 50
        
        # Small bonus for healthy battery
        if self.battery_health > 0.9:
            reward += 1
            
        # Penalty for diesel starts (encourages planning)
        if cost >= self.diesel_start_cost:
            reward -= 5
            
        return reward
    
    def _get_obs(self):
        """Return current observation"""
        # Day of year as sin/cos components (cyclical)
        day_sin = math.sin(2 * math.pi * self.day / 365)
        day_cos = math.cos(2 * math.pi * self.day / 365)
        
        return np.array([
            self.demand / 400,  # Normalize
            self.solar / 150,
            self.wind / 150,
            self.grid_price / 200,
            self.battery_charge / 100,
            self.battery_health,
            1 if self.diesel_available else 0,
            self.hour / 24,
            day_sin,
            day_cos,
            self.forecast_error_demand / 50,
            self.forecast_error_solar / 50,
            self.load_shedding
        ], dtype=np.float32)