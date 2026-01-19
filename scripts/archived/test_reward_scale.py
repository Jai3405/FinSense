"""
Test script to validate reward scale hypothesis.
Measures actual reward magnitudes during training.
"""
import numpy as np
import pandas as pd
from data_loader import DataLoader
from environment import TradingEnvironment
from utils import load_config

def analyze_reward_scale():
    """Analyze the scale of different reward components."""

    config = load_config('config.yaml')
    data_config = config.get_section('data')
    env_config = config.get_section('environment')

    # Load data
    loader = DataLoader(data_config)
    data = loader.load_data()
    train_data, _, _ = loader.train_test_split(data, 0.7, 0.15)

    # Create environment
    env = TradingEnvironment(train_data, env_config)
    env.reset()

    # Track reward components
    rewards = []
    equity_deltas = []
    idle_penalties = []
    transaction_costs = []

    # Simulate random trading
    window_size = env_config.get('window_size', 20)
    for step in range(window_size, min(len(train_data['close']) - 1, window_size + 500)):
        # Random action
        action = np.random.choice([0, 1, 2])

        # Store state before
        prev_balance = env.balance
        prev_inventory_size = len(env.inventory)

        # Execute
        reward, done, info = env.step(action)

        rewards.append(reward)

        # Track if this was a trade
        if action in [0, 2]:
            if env.balance != prev_balance or len(env.inventory) != prev_inventory_size:
                cost = abs(env.balance - prev_balance) - abs(len(env.inventory) - prev_inventory_size) * env.data['close'][env.current_step-1]
                transaction_costs.append(cost)

        if done:
            break

    # Analysis
    print("="*70)
    print("REWARD SCALE ANALYSIS")
    print("="*70)

    print(f"\nTotal steps simulated: {len(rewards)}")
    print(f"\nReward Statistics:")
    print(f"  Mean:   {np.mean(rewards):10.4f}")
    print(f"  Std:    {np.std(rewards):10.4f}")
    print(f"  Min:    {np.min(rewards):10.4f}")
    print(f"  Max:    {np.max(rewards):10.4f}")
    print(f"  Median: {np.median(rewards):10.4f}")

    # Count near-zero rewards (HOLD when flat)
    near_zero = np.abs(rewards) < 1.0
    print(f"\nRewards near zero (<₹1): {np.sum(near_zero)} ({100*np.mean(near_zero):.1f}%)")

    # Transaction costs
    if transaction_costs:
        print(f"\nTransaction Costs:")
        print(f"  Mean:   {np.mean(transaction_costs):10.4f}")
        print(f"  Number: {len(transaction_costs)}")

    # Calculate typical ATR
    atr_values = train_data.get('atr', [0])
    typical_atr = np.mean([a for a in atr_values if a > 0])
    print(f"\nTypical ATR: ₹{typical_atr:.2f}")
    print(f"Idle penalty per step: ₹{0.001 * typical_atr:.4f}")

    # Critical ratio
    if transaction_costs:
        avg_txn_cost = np.mean(transaction_costs)
        idle_penalty_value = 0.001 * typical_atr
        print(f"\n🚨 CRITICAL RATIO:")
        print(f"   Transaction cost / Idle penalty = {avg_txn_cost / idle_penalty_value:.1f}x")
        print(f"   → Trading is {int(avg_txn_cost / idle_penalty_value)}× more penalized than holding")

    print("="*70)

    # Recommendation
    print("\nRECOMMENDATION:")
    if transaction_costs and np.mean(transaction_costs) > 100 * (0.001 * typical_atr):
        print("  ❌ Transaction costs DOMINATE idle penalty")
        print("  → Agent rationally learns to never trade")
        print("  → SOLUTION: Use percentage-based rewards")
    else:
        print("  ✅ Reward scale is balanced")

    print("="*70)

if __name__ == '__main__':
    analyze_reward_scale()
