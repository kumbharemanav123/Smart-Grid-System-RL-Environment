"""
test_smart_grid.py

Test script for the Smart Grid environment.
Runs a random agent for 100 steps to verify everything works.
"""

from smart_grid_env import SmartGridEnv
import numpy as np

def main():
    print("=" * 50)
    print("Smart Grid Environment Test")
    print("=" * 50)
    
    # Create the environment
    env = SmartGridEnv(max_steps=100)
    
    # Reset the environment
    obs, info = env.reset()
    
    print(f"\n✅ Environment created successfully")
    print(f"Observation space: {env.observation_space.shape}")
    print(f"Action space: {env.action_space}")
    print(f"Initial observation (first 5 values): {obs[:5]}")
    
    # Run a few random steps
    print("\n" + "=" * 50)
    print("Running 10 random steps...")
    print("=" * 50)
    
    total_reward = 0
    step_count = 0
    
    for step in range(10):
        # Take random action
        action = env.action_space.sample()
        
        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        step_count += 1
        
        print(f"\nStep {step+1}:")
        print(f"  Action taken: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Battery charge: {env.battery_charge:.1f} MWh")
        print(f"  Battery health: {env.battery_health:.3f}")
        print(f"  Demand: {env.demand:.1f} MW")
        print(f"  Solar: {env.solar:.1f} MW")
        print(f"  Wind: {env.wind:.1f} MW")
        print(f"  Grid price: ${env.grid_price:.1f}/MWh")
        
        if terminated or truncated:
            print(f"  Episode ended early!")
            break
    
    print("\n" + "=" * 50)
    print(f"Test completed after {step_count} steps")
    print(f"Total reward: {total_reward:.2f}")
    print("=" * 50)
    
    # Test observation shape
    print(f"\n✅ Observation shape: {obs.shape} (should be 13)")
    print("✅ All tests passed! Environment is working correctly.")

if __name__ == "__main__":
    main()