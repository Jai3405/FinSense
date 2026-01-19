"""
Quick multi-stock evaluation matching training setup.
"""

import torch
import numpy as np
from agents import DQNAgent
from data_loader import DataLoader
from environment import TradingEnvironment
from utils import load_config, TradingMetrics, get_state_with_features

def main():
    # Load config
    config = load_config('config.yaml')

    # Load multi-stock data (same as training)
    data_config = config.get_section('data')
    data_loader = DataLoader(data_config)

    print("Loading multi-stock data (same as training)...")
    data = data_loader.load_data()

    # Split data
    train_data, val_data, test_data = data_loader.train_test_split(
        data,
        train_ratio=config.get('training.train_ratio', 0.7),
        validation_ratio=config.get('training.validation_ratio', 0.15)
    )

    print(f"Test set size: {len(test_data['close'])} points")

    # Create environment
    env_config = config.get_section('environment')
    test_env = TradingEnvironment(test_data, env_config)
    window_size = env_config.get('window_size', 20)

    # Calculate state size (same as train.py)
    state_size = window_size - 1  # Price diffs
    if env_config.get('use_volume', True):
        state_size += 1
    if env_config.get('use_technical_indicators', True):
        state_size += 6  # RSI(1) + MACD(3) + BB(1) + ATR(1)

    # Action size is always 3 (Buy, Hold, Sell)
    action_size = 3

    # Load agent
    agent = DQNAgent(state_size, action_size, config.get_section('agent'))
    agent.load('models/best_model.pt')
    agent.epsilon = 0.0  # No exploration during evaluation

    print(f"\nEvaluating Episode 282 model on TEST SET...")
    print(f"State size: {state_size}, Action size: {action_size}")

    # Evaluate
    test_env.reset()
    done = False
    total_reward = 0
    trades = 0
    actions_taken = {'buy': 0, 'hold': 0, 'sell': 0}

    # Loop through test data
    for t in range(window_size, len(test_data['close'])):
        # Get current state
        state = get_state_with_features(test_data, t, window_size, env_config)

        # Agent selects action (training=False means greedy, no exploration)
        action = agent.act(state, training=False)

        # Execute action in environment
        reward, done, info = test_env.step(action)

        # Track actions
        if action == 0:
            actions_taken['buy'] += 1
        elif action == 1:
            actions_taken['hold'] += 1
        elif action == 2:
            actions_taken['sell'] += 1

        if info.get('success', False):
            trades += 1

        total_reward += reward

        if done:
            break

    # Calculate final metrics
    final_balance = test_env.balance
    final_portfolio_value = test_env._calculate_portfolio_value(
        test_env.data['close'][test_env.current_step]
    )
    profit = final_portfolio_value - test_env.initial_balance

    # Full metrics
    metrics_calc = TradingMetrics(
        risk_free_rate=config.get('evaluation.risk_free_rate', 0.02)
    )

    # Calculate metrics if we have trades
    if len(test_env.trades) > 0:
        metrics = metrics_calc.calculate_all_metrics(
            test_env.portfolio_values,
            test_env.trades
        )
    else:
        # No trades - just calculate basic metrics
        metrics = {
            'total_profit': profit,
            'total_return': profit / test_env.initial_balance,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'max_drawdown_percent': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'avg_profit_per_trade': 0.0
        }

    # Print results
    print("\n" + "="*70)
    print("TEST SET EVALUATION RESULTS (Multi-Stock, Same as Training)")
    print("="*70)
    print(f"\nTotal Timesteps: {test_env.current_step}")
    print(f"Total Trades: {len(test_env.trades)}")
    print(f"\nActions:")
    print(f"  Buy:  {actions_taken['buy']}")
    print(f"  Hold: {actions_taken['hold']}")
    print(f"  Sell: {actions_taken['sell']}")

    print(f"\n{'Metric':<30} {'Value':<20}")
    print("-"*50)
    print(f"{'Total Profit':<30} ₹{metrics['total_profit']:,.2f}")
    print(f"{'Total Return':<30} {metrics['total_return']*100:.2f}%")
    print(f"{'Sharpe Ratio':<30} {metrics['sharpe_ratio']:.4f}")
    print(f"{'Sortino Ratio':<30} {metrics.get('sortino_ratio', 0):.4f}")
    print(f"{'Max Drawdown':<30} {metrics['max_drawdown_percent']:.2f}%")
    print(f"{'Win Rate':<30} {metrics['win_rate']*100:.2f}%")
    print(f"{'Profit Factor':<30} {metrics.get('profit_factor', 0):.2f}")
    print(f"{'Avg Profit/Trade':<30} ₹{metrics.get('avg_profit_per_trade', 0):.2f}")

    print("\n" + "="*70)

    # Compare to Episode 282 validation metrics
    print("\nCOMPARISON TO EPISODE 282 (Validation Set):")
    print("-"*70)
    print(f"{'Metric':<30} {'Ep 282 (Val)':<20} {'Test Set':<20}")
    print("-"*70)
    print(f"{'Profit':<30} ₹26,456{'':<14} ₹{metrics['total_profit']:,.2f}")
    print(f"{'Trades':<30} 290{'':<16} {len(test_env.trades)}")
    print(f"{'Win Rate':<30} 68.97%{'':<14} {metrics['win_rate']*100:.2f}%")
    print(f"{'Sharpe':<30} -0.543{'':<15} {metrics['sharpe_ratio']:.3f}")
    print("="*70)

    if len(test_env.trades) == 0:
        print("\n🚨 CRITICAL: Agent made ZERO trades on test set!")
        print("This indicates severe overfitting or distribution mismatch.")
    elif len(test_env.trades) > 1000:
        print("\n⚠️  WARNING: Agent made too many trades (>1000)!")
        print("This indicates the model is confused on test data.")
    else:
        print(f"\n✅ Agent made {len(test_env.trades)} trades (reasonable range)")

if __name__ == '__main__':
    main()
