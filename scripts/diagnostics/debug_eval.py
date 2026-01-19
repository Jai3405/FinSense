"""Debug PPO evaluation to see what's happening"""
import torch
import numpy as np
from agents.ppo_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from data_loader.data_loader import DataLoader
from utils.features import get_state_with_features
from utils.config import load_config

config = load_config('config.yaml')
data_config = config.get_section('data')
env_config = config.get_section('environment')

# Load data
loader = DataLoader(data_config)
data = loader.load_data()
train_ratio = config.get('training.train_ratio', 0.7)
val_ratio = config.get('training.validation_ratio', 0.15)
_, _, test_data = loader.train_test_split(data, train_ratio, val_ratio)

env = TradingEnvironment(test_data, env_config)
window_size = env_config.get("window_size", 20)
action_size = env_config.get("action_size", 3)

# Calculate state size
state_size = window_size - 1
if config.get('environment.use_volume', True):
    state_size += 1
if config.get('environment.use_technical_indicators', True):
    state_size += 9

# Load agent
device = torch.device("cpu")
agent = PPOAgent(state_size, action_size).to(device)
agent.load_state_dict(torch.load("models/ppo_final.pt", map_location=device))
agent.eval()

print("DEBUG: PPO Evaluation")
print(f"Test set size: {len(test_data['close'])}")
print(f"Starting balance: ₹{env.initial_balance}")
print()

env.reset()
steps_checked = 0
mask_stats = {0: 0, 1: 0, 2: 0}  # Count how often each action is masked
action_counts = {0: 0, 1: 0, 2: 0}

with torch.no_grad():
    while not env.is_done() and steps_checked < 100:  # Check first 100 steps
        state = get_state_with_features(test_data, env.current_step, window_size, env_config)
        state_tensor = torch.from_numpy(state).float().to(device)

        action_mask = env.get_action_mask()
        logits, _ = agent(state_tensor)
        logits = logits.squeeze(0)

        # Check mask
        for i, masked in enumerate(action_mask):
            if masked:
                mask_stats[i] += 1

        # Apply mask
        mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)
        masked_logits = logits.masked_fill(~mask_tensor, -float('inf'))

        action = torch.argmax(masked_logits).item()
        action_counts[action] += 1

        # Debug first 10 steps
        if steps_checked < 10:
            print(f"Step {steps_checked}:")
            print(f"  Balance: ₹{env.balance:.2f}, Inventory: {len(env.inventory)}")
            print(f"  Raw logits: {logits.cpu().numpy()}")
            print(f"  Action mask: {action_mask} (True=allowed)")
            print(f"  Masked logits: {masked_logits.cpu().numpy()}")
            print(f"  Chosen action: {action} ({'BUY' if action==0 else 'HOLD' if action==1 else 'SELL'})")
            print()

        reward, done, _ = env.step(action)
        steps_checked += 1

        if done:
            break

print("\n" + "="*60)
print("SUMMARY (First 100 steps):")
print("="*60)
print(f"Steps checked: {steps_checked}")
print(f"\nAction mask frequency (how often action was ALLOWED):")
print(f"  BUY  allowed: {mask_stats[0]}/{steps_checked} ({100*mask_stats[0]/steps_checked:.1f}%)")
print(f"  HOLD allowed: {mask_stats[1]}/{steps_checked} ({100*mask_stats[1]/steps_checked:.1f}%)")
print(f"  SELL allowed: {mask_stats[2]}/{steps_checked} ({100*mask_stats[2]/steps_checked:.1f}%)")
print(f"\nActions taken:")
print(f"  BUY:  {action_counts[0]} ({100*action_counts[0]/steps_checked:.1f}%)")
print(f"  HOLD: {action_counts[1]} ({100*action_counts[1]/steps_checked:.1f}%)")
print(f"  SELL: {action_counts[2]} ({100*action_counts[2]/steps_checked:.1f}%)")
print(f"\nFinal balance: ₹{env.balance:.2f}")
print(f"Final inventory: {len(env.inventory)} shares")
