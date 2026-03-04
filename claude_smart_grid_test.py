"""
claude_smart_grid_test.py - FIXED VERSION
Tests Claude Opus 4.6 on the Smart Grid environment.
"""

import os
import time
import csv
import requests
import random
from datetime import datetime
from dotenv import load_dotenv
from smart_grid_env import SmartGridEnv

# Load API key with debugging
print("🔑 Loading API key from .env file...")
load_dotenv()
API_KEY = os.getenv("ANTHROPIC_API_KEY")

if not API_KEY:
    print("❌ No API key found in .env file")
    print("📁 Current directory:", os.getcwd())
    print("📄 Files in directory:")
    for file in os.listdir('.'):
        if file.startswith('.env'):
            print(f"   Found: {file}")
            # Try to read it
            try:
                with open(file, 'r') as f:
                    content = f.read().strip()
                    print(f"   Content: {content[:20]}...")
            except:
                print("   Could not read file")
    raise ValueError("❌ No API key found. Create a .env file with ANTHROPIC_API_KEY=your-key")
else:
    print(f"✅ API key loaded successfully: {API_KEY[:10]}...")

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-3-opus-20240229"

def call_claude(prompt, max_tokens=100):
    """Send prompt to Claude and return response"""
    headers = {
        "x-api-key": API_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    data = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}]
    }
    
    try:
        print(f"📤 Sending request to Claude...")
        response = requests.post(API_URL, headers=headers, json=data)
        print(f"📥 Response status: {response.status_code}")
        
        if response.status_code == 200:
            return response.json()["content"][0]["text"]
        else:
            print(f"⚠️ Error response: {response.text}")
            return None
    except Exception as e:
        print(f"⚠️ API call failed: {e}")
        return None

def parse_action(response_text):
    """Extract action number from Claude's response"""
    if response_text:
        import re
        match = re.search(r"Action:\s*([0-8])", response_text)
        if match:
            return int(match.group(1))
    # Fallback to random
    return random.randint(0, 8)

def build_prompt(obs, step, episode):
    """Create a detailed prompt for Claude"""
    
    # Decode observation
    demand = obs[0] * 400
    solar = obs[1] * 150
    wind = obs[2] * 150
    price = obs[3] * 200
    battery = obs[4] * 100
    health = obs[5]
    diesel_avail = "Yes" if obs[6] > 0.5 else "No"
    hour = int(obs[7] * 24)
    
    prompt = f"""You are managing a smart grid energy system. Your goal is to minimize cost while maintaining reliable power supply.

Current state (Hour {hour}, Step {step} of episode {episode}):

📊 Grid Status:
- Electricity demand: {demand:.1f} MW
- Solar generation: {solar:.1f} MW
- Wind generation: {wind:.1f} MW
- Grid electricity price: ${price:.1f}/MWh

🔋 Battery System:
- Charge level: {battery:.1f} MWh (capacity 100 MWh)
- Health: {health:.3f} (degrades with use)
- Max charge/discharge rate: 20 MW per hour

⚡ Diesel Generator:
- Available: {diesel_avail}
- Capacity: 50 MW when running
- Fuel cost: $100/MWh
- Start cost: $500 (requires 24h cooldown)

Available actions (0-8):
0: Do nothing
1: Charge battery from grid
2: Discharge battery to grid
3: Charge from solar only
4: Discharge to load
5: Start diesel generator
6: Stop diesel generator
7: Buy from grid
8: Sell to grid

Remember:
- Battery health degrades with each charge/discharge cycle
- Price spikes can occur suddenly
- Solar only works during daytime
- Diesel takes 24h to restart after stopping
- Unmet demand is heavily penalized

Reply with exactly one line: "Action: X" where X is 0-8.
Do not include any other text."""

    return prompt

def run_episode(env, episode, writer):
    """Run one episode with Claude as agent"""
    obs, info = env.reset()
    terminated = False
    total_reward = 0
    step = 0
    action_history = []
    
    print(f"\n📊 Episode {episode} starting...")
    
    while not terminated and step < 10:  # Limit to 10 steps for testing
        prompt = build_prompt(obs, step, episode)
        
        # Get action from Claude
        print(f"\n⏳ Step {step}: Getting action from Claude...")
        response = call_claude(prompt)
        action = parse_action(response)
        action_history.append(action)
        
        print(f"   Claude action: {action} (response: {response})")
        
        # Log the step
        writer.writerow([
            episode, step,
            obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7],
            action, response or "None", datetime.now().isoformat()
        ])
        
        # Take action
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"   Reward: {reward:.2f}, Battery: {env.battery_charge:.1f}MWh")
        
        step += 1
        time.sleep(1)  # Be nice to API
    
    print(f"✅ Episode {episode} finished: {step} steps, total reward={total_reward:.2f}")
    print(f"   Actions taken: {action_history}")
    return total_reward, step

def main():
    print("=" * 60)
    print("🔌 SMART GRID - CLAUDE OPUS 4.6 TEST")
    print("=" * 60)
    
    # Create environment
    env = SmartGridEnv(max_steps=8760)
    num_episodes = 2  # Run 2 episodes for testing
    
    # Setup CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    perf_file = f'claude_performance_{timestamp}.csv'
    summary_file = f'claude_summary_{timestamp}.csv'
    
    with open(perf_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'episode', 'step',
            'demand_norm', 'solar_norm', 'wind_norm', 'price_norm', 
            'battery_norm', 'battery_health', 'diesel_avail', 'hour_norm',
            'action', 'claude_response', 'timestamp'
        ])
        
        episode_rewards = []
        
        for ep in range(num_episodes):
            print(f"\n{'='*50}")
            reward, steps = run_episode(env, ep+1, writer)
            episode_rewards.append((ep+1, reward, steps))
    
    # Write summary
    with open(summary_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'steps'])
        for ep, reward, steps in episode_rewards:
            writer.writerow([ep, reward, steps])
    
    print("\n" + "=" * 60)
    print("📈 TEST COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to:")
    print(f"  📄 {perf_file}")
    print(f"  📄 {summary_file}")

if __name__ == "__main__":
    main()