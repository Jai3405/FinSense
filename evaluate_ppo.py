# evaluate_ppo.py

import torch
import numpy as np
from collections import Counter

from agents.ppo_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from data_loader.data_loader import DataLoader
from utils.features import get_state_with_features
from utils.config import load_config


def evaluate_ppo():
    config = load_config("config.yaml")

    data_config = config.get_section("data")
    env_config = config.get_section("environment")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # Load data
    # -----------------------------
    loader = DataLoader(data_config)
    data = loader.load_data()

    # Use same split ratios as training
    train_ratio = config.get('training.train_ratio', 0.7)
    val_ratio = config.get('training.validation_ratio', 0.15)
    _, _, test_data = loader.train_test_split(data, train_ratio, val_ratio)

    env = TradingEnvironment(test_data, env_config)

    window_size = env_config.get("window_size", 20)
    action_size = env_config.get("action_size", 3)

    # -----------------------------
    # Calculate state size dynamically (must match training)
    # -----------------------------
    state_size = window_size - 1  # Price diffs
    if config.get('environment.use_volume', True):
        state_size += 1
    if config.get('environment.use_technical_indicators', True):
        state_size += 9  # RSI(1) + MACD(3) + BB(1) + ATR(1) + Trend(3)

    print(f"Test set size: {len(test_data['close'])} points")
    print(f"State size: {state_size} features")

    # -----------------------------
    # Load PPO agent
    # -----------------------------
    agent = PPOAgent(state_size, action_size).to(device)
    try:
        # PPO training saves a raw state_dict, not a checkpoint object
        agent.load_state_dict(torch.load("models/ppo_final.pt", map_location=device))
        print("Evaluating PPO model (ppo_final.pt) on TEST SET...")
    except FileNotFoundError:
        print("ERROR: models/ppo_final.pt not found. Please ensure PPO training has run and a model has been saved/renamed.")
        return
        
    agent.eval()

    # -----------------------------
    # Evaluation loop
    # -----------------------------
    env.reset()
    print("\nEvaluating PPO on test set...")

    action_counter = Counter()
    total_reward = 0.0

    with torch.no_grad():
        while not env.is_done():
            state = get_state_with_features(
                test_data,
                env.current_step,
                window_size,
                env_config
            )
            state_tensor = torch.from_numpy(state).float().to(device)

            action_mask = env.get_action_mask()

            logits, _ = agent(state_tensor)
            logits = logits.squeeze(0)

            # Apply mask
            mask_tensor = torch.tensor(action_mask, dtype=torch.bool, device=device)
            logits = logits.masked_fill(~mask_tensor, -float('inf'))

            action = torch.argmax(logits).item()
            reward, done, _ = env.step(action)

            action_counter[action] += 1
            total_reward += reward

            if done:
                break

    # -----------------------------
    # Results
    # -----------------------------
    total_steps = sum(action_counter.values()) if sum(action_counter.values()) > 0 else 1

    print("\n" + "=" * 60)
    print("PPO POLICY EVALUATION RESULTS")
    print("=" * 60)

    print("\nAction Distribution:")
    print(f"  BUY :  {action_counter.get(0, 0)} ({action_counter.get(0, 0) / total_steps:.2%})")
    print(f"  HOLD:  {action_counter.get(1, 0)} ({action_counter.get(1, 0) / total_steps:.2%})")
    print(f"  SELL:  {action_counter.get(2, 0)} ({action_counter.get(2, 0) / total_steps:.2%})")

    print("\nTrading Behavior:")
    print(f"  Total executed trades: {env.episode_trades}")

    print("\nPerformance:")
    final_profit = env.get_portfolio_value() - env.initial_balance
    print(f"  Final Profit/Loss   : ₹{final_profit:.2f}")
    print(f"  Total reward signal : {total_reward:.4f}")

    print("\nInterpretation:")
    if action_counter.get(1, 0) / total_steps > 0.9:
        print("  → Policy is highly risk-averse, preferring to HOLD.")
    elif env.episode_trades < 5:
        print("  → Policy trades extremely selectively.")
    else:
        print("  → Policy actively trades under learned constraints.")

    print("=" * 60)


if __name__ == "__main__":
    evaluate_ppo()
