"""Diagnose why PPO isn't learning"""
import re
import numpy as np

# Parse training log
log_file = None
import glob
log_files = glob.glob("training_200ep_OPTIMAL_*.log")
if log_files:
    log_file = log_files[0]
else:
    log_files = glob.glob("training_200ep_*.log")
    if log_files:
        log_file = log_files[0]

if not log_file:
    print("No training log found!")
    exit(1)

print(f"Analyzing: {log_file}\n")

episodes = []
profits = []
trades = []
rewards = []

with open(log_file, 'r') as f:
    for line in f:
        if "Episode" in line and "/" in line:
            # Parse: Episode 1/200 | Profit: ₹-4456.41 | Trades: 3574 | Reward: -49.77
            match = re.search(r'Episode (\d+)/\d+ \| Profit: ₹([-\d.]+) \| Trades: (\d+) \| Reward: ([-\d.]+)', line)
            if match:
                ep, profit, trade, reward = match.groups()
                episodes.append(int(ep))
                profits.append(float(profit))
                trades.append(int(trade))
                rewards.append(float(reward))

if not episodes:
    print("Failed to parse log!")
    exit(1)

print(f"Parsed {len(episodes)} episodes\n")
print("="*70)
print("CONVERGENCE ANALYSIS")
print("="*70)

# Analyze in chunks
chunks = [(1, 50), (51, 100), (101, 150), (151, 200)]

for start, end in chunks:
    chunk_eps = [i for i, ep in enumerate(episodes) if start <= ep <= end]
    if not chunk_eps:
        continue

    chunk_trades = [trades[i] for i in chunk_eps]
    chunk_rewards = [rewards[i] for i in chunk_eps]
    chunk_profits = [profits[i] for i in chunk_eps]

    print(f"\nEpisodes {start}-{end}:")
    print(f"  Avg Trades:  {np.mean(chunk_trades):.0f} (std: {np.std(chunk_trades):.0f})")
    print(f"  Avg Reward:  {np.mean(chunk_rewards):.2f} (std: {np.std(chunk_rewards):.2f})")
    print(f"  Avg Profit:  ₹{np.mean(chunk_profits):,.0f} (std: ₹{np.std(chunk_profits):,.0f})")

print(f"\n{'='*70}")
print("CONVERGENCE VERDICT")
print(f"{'='*70}\n")

# Check if agent converged
early_trades = np.mean([trades[i] for i in range(min(50, len(trades)))])
late_trades = np.mean([trades[i] for i in range(max(0, len(trades)-50), len(trades))])
trade_reduction = (early_trades - late_trades) / early_trades * 100

early_rewards = np.mean([rewards[i] for i in range(min(50, len(rewards)))])
late_rewards = np.mean([rewards[i] for i in range(max(0, len(rewards)-50), len(rewards))])
reward_improvement = late_rewards - early_rewards

print(f"Trade Count Reduction: {trade_reduction:.1f}%")
if trade_reduction > 20:
    print("  ✅ GOOD: Agent is learning to trade less (converging)")
elif trade_reduction > 10:
    print("  ⚠️  MARGINAL: Some reduction but slow convergence")
else:
    print(f"  ❌ BAD: Agent NOT reducing trades (stuck in exploration)")

print(f"\nReward Improvement: {reward_improvement:+.2f}")
if reward_improvement > 5:
    print("  ✅ GOOD: Rewards improving significantly")
elif reward_improvement > 2:
    print("  ⚠️  MARGINAL: Slight improvement")
else:
    print("  ❌ BAD: Rewards NOT improving (policy not learning)")

print(f"\n{'='*70}")
print("DIAGNOSIS")
print(f"{'='*70}\n")

if trade_reduction < 10 and reward_improvement < 2:
    print("❌ CRITICAL: PPO IS NOT LEARNING")
    print("\nPossible causes:")
    print("1. Learning rate too low (current: 0.0003)")
    print("2. Reward signal too noisy (agent can't find gradient)")
    print("3. Batch size too small (current: 64)")
    print("4. Exploration coefficient too low (entropy: 0.01)")
    print("5. Environment is non-stationary (multi-stock may be too complex)")
    print("\nRecommendation: Try single-stock training first to isolate problem")

elif trade_reduction < 20:
    print("⚠️  PPO is learning SLOWLY")
    print("\nNeeds more episodes (250-300) OR hyperparameter tuning")
else:
    print("✅ PPO is learning properly")

# Check for overfitting
if len(episodes) >= 100:
    first_half_profit = np.mean(profits[:len(profits)//2])
    second_half_profit = np.mean(profits[len(profits)//2:])

    print(f"\n{'='*70}")
    print("OVERFITTING CHECK")
    print(f"{'='*70}")
    print(f"First half avg profit: ₹{first_half_profit:,.0f}")
    print(f"Second half avg profit: ₹{second_half_profit:,.0f}")

    if second_half_profit < first_half_profit * 0.8:
        print("⚠️  Possible overfitting (performance degrading)")
    else:
        print("✅ No overfitting detected")

print()
