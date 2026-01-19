"""
Comprehensive PPO Evaluation with Full Metrics
Senior quant-level analysis before scaling to 200 episodes
"""

import torch
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

from agents.ppo_agent import PPOAgent
from environment.trading_env import TradingEnvironment
from data_loader.data_loader import DataLoader
from utils.features import get_state_with_features
from utils.config import load_config
from utils.metrics import TradingMetrics


def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio."""
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1)
    
    if std_return == 0:
        return 0.0
    
    # Annualize (assuming daily returns)
    sharpe = (mean_return - risk_free_rate / 252) / std_return * np.sqrt(252)
    return sharpe


def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sortino ratio (downside deviation)."""
    if len(returns) < 2:
        return 0.0
    
    mean_return = np.mean(returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf
    
    downside_std = np.std(downside_returns, ddof=1)
    
    if downside_std == 0:
        return 0.0
    
    # Annualize
    sortino = (mean_return - risk_free_rate / 252) / downside_std * np.sqrt(252)
    return sortino


def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown."""
    if len(portfolio_values) < 2:
        return 0.0
    
    portfolio_values = np.array(portfolio_values)
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    max_dd = np.min(drawdown)
    
    return max_dd


def analyze_trades(env):
    """Analyze trade statistics."""
    trades = env.trades  # List of profit floats

    if len(trades) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0
        }

    profits = np.array(trades)

    if len(profits) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0.0,
            'loss_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'expectancy': 0.0
        }
    
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    
    win_rate = len(wins) / len(profits) if len(profits) > 0 else 0
    loss_rate = len(losses) / len(profits) if len(profits) > 0 else 0
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = np.mean(losses) if len(losses) > 0 else 0
    
    total_wins = sum(wins) if len(wins) > 0 else 0
    total_losses = abs(sum(losses)) if len(losses) > 0 else 0
    
    profit_factor = total_wins / total_losses if total_losses > 0 else np.inf
    expectancy = np.mean(profits)
    
    return {
        'total_trades': len(trades),
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'expectancy': expectancy
    }


def evaluate_ppo_comprehensive():
    """Comprehensive PPO evaluation."""
    config = load_config("config.yaml")
    
    data_config = config.get_section("data")
    env_config = config.get_section("environment")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    loader = DataLoader(data_config)
    data = loader.load_data()
    
    # Use same split as training
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
    
    # Load PPO agent
    agent = PPOAgent(state_size, action_size).to(device)
    agent.load_state_dict(torch.load("models/ppo_final.pt", map_location=device))
    agent.eval()
    
    print("="*80)
    print("COMPREHENSIVE PPO EVALUATION (50 Episodes)")
    print("="*80)
    print(f"\nTest Set: {len(test_data['close'])} points")
    print(f"State Size: {state_size} features")
    print(f"Starting Balance: ₹{env.initial_balance:,.2f}\n")
    
    # Evaluation loop
    env.reset()
    
    action_counter = Counter()
    portfolio_values = [env.initial_balance]
    
    with torch.no_grad():
        while not env.is_done():
            state = get_state_with_features(test_data, env.current_step, window_size, env_config)
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
            portfolio_values.append(env.get_portfolio_value())
            
            if done:
                break
    
    # Calculate metrics
    final_value = env.get_portfolio_value()
    total_profit = final_value - env.initial_balance
    total_return = (final_value / env.initial_balance - 1) * 100
    
    # Calculate returns
    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    
    sharpe = calculate_sharpe_ratio(returns)
    sortino = calculate_sortino_ratio(returns)
    max_dd = calculate_max_drawdown(portfolio_values)
    
    trade_stats = analyze_trades(env)
    
    # Print results
    print("─"*80)
    print("PERFORMANCE METRICS")
    print("─"*80)
    print(f"\n💰 P&L:")
    print(f"   Total Profit/Loss    : ₹{total_profit:,.2f}")
    print(f"   Total Return         : {total_return:.2f}%")
    print(f"   Final Portfolio Value: ₹{final_value:,.2f}")
    
    print(f"\n📊 Risk-Adjusted Returns:")
    print(f"   Sharpe Ratio         : {sharpe:.4f}")
    print(f"   Sortino Ratio        : {sortino:.4f}")
    print(f"   Max Drawdown         : {max_dd*100:.2f}%")
    
    print(f"\n📈 Trading Statistics:")
    print(f"   Total Trades         : {trade_stats['total_trades']}")
    print(f"   Win Rate             : {trade_stats['win_rate']*100:.2f}%")
    print(f"   Loss Rate            : {trade_stats['loss_rate']*100:.2f}%")
    print(f"   Profit Factor        : {trade_stats['profit_factor']:.2f}")
    print(f"   Avg Win              : ₹{trade_stats['avg_win']:.2f}")
    print(f"   Avg Loss             : ₹{trade_stats['avg_loss']:.2f}")
    print(f"   Expectancy/Trade     : ₹{trade_stats['expectancy']:.2f}")
    
    total_actions = sum(action_counter.values())
    print(f"\n🎯 Action Distribution:")
    print(f"   BUY  : {action_counter.get(0, 0):4d} ({action_counter.get(0, 0)/total_actions*100:5.2f}%)")
    print(f"   HOLD : {action_counter.get(1, 0):4d} ({action_counter.get(1, 0)/total_actions*100:5.2f}%)")
    print(f"   SELL : {action_counter.get(2, 0):4d} ({action_counter.get(2, 0)/total_actions*100:5.2f}%)")
    
    # Production readiness assessment
    print("\n" + "─"*80)
    print("PRODUCTION READINESS ASSESSMENT")
    print("─"*80)
    
    checks = []
    checks.append(("Trades on unseen data", trade_stats['total_trades'] > 20, trade_stats['total_trades']))
    checks.append(("Positive expectancy", total_profit > 0, f"₹{total_profit:.2f}"))
    checks.append(("Sharpe > 0.3", sharpe > 0.3, f"{sharpe:.4f}"))
    checks.append(("Max Drawdown < 20%", abs(max_dd) < 0.20, f"{max_dd*100:.2f}%"))
    checks.append(("Balanced actions", action_counter.get(1, 0)/total_actions < 0.90, f"{action_counter.get(1, 0)/total_actions*100:.1f}% HOLD"))
    checks.append(("Win rate > 48%", trade_stats['win_rate'] > 0.48, f"{trade_stats['win_rate']*100:.2f}%"))
    
    passed = sum(1 for _, check, _ in checks if check)
    total_checks = len(checks)
    
    print(f"\nCriteria Met: {passed}/{total_checks}\n")
    
    for criterion, passed_check, value in checks:
        status = "✅" if passed_check else "❌"
        print(f"{status} {criterion:.<50} {value}")
    
    # Recommendation
    print("\n" + "─"*80)
    print("RECOMMENDATION")
    print("─"*80)
    
    if passed >= 5:
        print("\n🚀 PROCEED WITH 200-EPISODE TRAINING")
        print(f"   Confidence: {passed/total_checks*100:.0f}%")
        print("   Expected outcome: Profitable production agent")
        print("   Command: python train_ppo.py --episodes 200 --verbose > training_PPO_200ep.log 2>&1 &")
    elif passed >= 4:
        print("\n⚠️  CONDITIONAL APPROVAL")
        print(f"   Confidence: {passed/total_checks*100:.0f}%")
        print("   Recommendation: Tune idle_penalty_coefficient, then scale to 200 episodes")
    else:
        print("\n❌ NEEDS IMPROVEMENT")
        print(f"   Confidence: {passed/total_checks*100:.0f}%")
        print("   Recommendation: Refine reward function before scaling")
    
    print("\n" + "="*80)
    
    # Save equity curve plot
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_values, linewidth=2)
    plt.axhline(y=env.initial_balance, color='r', linestyle='--', alpha=0.5, label='Initial Balance')
    plt.title('PPO Agent - Equity Curve (Test Set)', fontsize=14, fontweight='bold')
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Portfolio Value (₹)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('ppo_equity_curve.png', dpi=150)
    print("\n📊 Equity curve saved to: ppo_equity_curve.png")
    
    return {
        'total_profit': total_profit,
        'total_return': total_return,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'trade_stats': trade_stats,
        'passed_checks': passed,
        'total_checks': total_checks
    }


if __name__ == "__main__":
    results = evaluate_ppo_comprehensive()
