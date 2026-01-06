"""
Unified evaluation script for FinSense DQN trading agent.

Usage:
    python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS
    python evaluate.py --model models/best_model.pt --data data/custom.csv --output results/
"""

import argparse
import logging
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from agents import DQNAgent
from data_loader import DataLoader
from environment import TradingEnvironment
from utils import (
    load_config,
    get_state_with_features,
    TradingMetrics,
    setup_logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate FinSense DQN Agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate best model on RELIANCE.NS
  python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS

  # Evaluate with custom data
  python evaluate.py --model models/best_model.pt --data data/custom.csv

  # Save results to specific directory
  python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS --output results/

  # Verbose output
  python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS --verbose
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint (e.g., models/best_model.pt)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        default=None,
        help='Stock ticker symbol (e.g., RELIANCE.NS, TCS.NS)'
    )

    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Path to custom data CSV file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='evaluation_results',
        help='Output directory for results (default: evaluation_results/)'
    )

    parser.add_argument(
        '--split',
        type=str,
        choices=['test', 'validation', 'train', 'full'],
        default='test',
        help='Data split to evaluate on (default: test)'
    )

    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def calculate_buy_and_hold_metrics(data, starting_balance, config):
    """
    Calculate buy-and-hold baseline performance.

    Args:
        data (dict): Price data
        starting_balance (float): Starting capital
        config (Config): Configuration object

    Returns:
        dict: Buy-and-hold metrics
    """
    close_prices = data['close']

    # Buy at first price, sell at last price
    first_price = close_prices[0]
    last_price = close_prices[-1]

    # Calculate shares we can buy
    shares = starting_balance / first_price

    # Final portfolio value
    final_value = shares * last_price
    profit = final_value - starting_balance
    profit_percent = (profit / starting_balance) * 100

    # Calculate portfolio values over time
    portfolio_values = shares * close_prices

    # Calculate metrics
    risk_free_rate = config.get('metrics.risk_free_rate', 0.02)
    metrics_calc = TradingMetrics(risk_free_rate=risk_free_rate)

    # Calculate returns
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    max_dd, _, _ = metrics_calc.max_drawdown(portfolio_values)

    metrics = {
        'total_profit': profit,
        'total_profit_percent': profit_percent,
        'final_portfolio_value': final_value,
        'sharpe_ratio': metrics_calc.sharpe_ratio(returns),
        'sortino_ratio': metrics_calc.sortino_ratio(returns),
        'max_drawdown': max_dd,
        'max_drawdown_percent': max_dd * 100,
        'calmar_ratio': metrics_calc.calmar_ratio(returns, portfolio_values),
        'total_trades': 2,  # Buy once, sell once
    }

    return metrics, portfolio_values


def evaluate_agent(agent, data, config, starting_balance, logger):
    """
    Evaluate agent on data.

    Args:
        agent: Trained DQN agent
        data (dict): Price data
        config (Config): Configuration object
        starting_balance (float): Starting capital
        logger: Logger instance

    Returns:
        dict: Evaluation metrics
        dict: Trading info (portfolio values, actions, etc.)
    """
    window_size = config.get('environment.window_size', 10)

    # Create environment
    env = TradingEnvironment(data, config.get_section('environment'))
    env.reset()

    # Track metrics
    portfolio_values = [starting_balance]
    actions_taken = []
    rewards = []
    positions = []  # Track inventory over time

    logger.info(f"Evaluating on {len(data['close'])} data points...")

    # Run evaluation
    for t in range(window_size, len(data['close'])):
        # Get state
        state = get_state_with_features(data, t, window_size, config.get_section('environment'))

        # Agent acts (no exploration - greedy policy)
        action = agent.act(state, training=False)

        # Execute action
        reward, done, info = env.step(action)

        # Track metrics
        portfolio_value = env.get_portfolio_value()
        portfolio_values.append(portfolio_value)
        actions_taken.append(action)
        rewards.append(reward)
        positions.append(env.inventory)

        if done:
            break

    # Calculate comprehensive metrics
    risk_free_rate = config.get('metrics.risk_free_rate', 0.02)
    metrics_calc = TradingMetrics(risk_free_rate=risk_free_rate)

    portfolio_values = np.array(portfolio_values)
    returns = np.diff(portfolio_values) / portfolio_values[:-1]

    metrics = metrics_calc.calculate_all_metrics(
        portfolio_values=portfolio_values,
        trades=env.trades
    )

    # Add additional info
    metrics['total_profit_percent'] = metrics['total_return'] * 100
    metrics['total_timesteps'] = len(actions_taken)
    metrics['buy_actions'] = actions_taken.count(0)
    metrics['hold_actions'] = actions_taken.count(1)
    metrics['sell_actions'] = actions_taken.count(2)

    trading_info = {
        'portfolio_values': portfolio_values.tolist(),
        'actions': actions_taken,
        'rewards': rewards,
        'positions': positions,
        'trades': env.trades,
        'final_balance': env.balance,
        'final_inventory': env.inventory,
        'final_portfolio_value': portfolio_values[-1]
    }

    return metrics, trading_info


def generate_visualizations(agent_metrics, agent_info, bnh_metrics, bnh_values,
                            data, output_dir, logger):
    """
    Generate evaluation visualizations.

    Args:
        agent_metrics (dict): Agent metrics
        agent_info (dict): Agent trading info
        bnh_metrics (dict): Buy-and-hold metrics
        bnh_values (array): Buy-and-hold portfolio values
        data (dict): Price data
        output_dir (Path): Output directory
        logger: Logger instance
    """
    logger.info("Generating visualizations...")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FinSense DQN Agent Evaluation', fontsize=16, fontweight='bold')

    # 1. Portfolio Value Over Time
    ax1 = axes[0, 0]
    ax1.plot(agent_info['portfolio_values'], label='DQN Agent', linewidth=2, color='#2E86AB')
    ax1.plot(bnh_values, label='Buy & Hold', linewidth=2, color='#A23B72', linestyle='--')
    ax1.set_title('Portfolio Value Over Time', fontweight='bold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Portfolio Value (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Drawdown Chart
    ax2 = axes[0, 1]
    agent_portfolio = np.array(agent_info['portfolio_values'])
    agent_peak = np.maximum.accumulate(agent_portfolio)
    agent_drawdown = (agent_portfolio - agent_peak) / agent_peak * 100

    bnh_peak = np.maximum.accumulate(bnh_values)
    bnh_drawdown = (bnh_values - bnh_peak) / bnh_peak * 100

    ax2.fill_between(range(len(agent_drawdown)), agent_drawdown, 0,
                      alpha=0.5, color='#2E86AB', label='DQN Agent')
    ax2.fill_between(range(len(bnh_drawdown)), bnh_drawdown, 0,
                      alpha=0.5, color='#A23B72', label='Buy & Hold')
    ax2.set_title('Drawdown Over Time', fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Drawdown (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Action Distribution
    ax3 = axes[1, 0]
    actions = ['Buy', 'Hold', 'Sell']
    action_counts = [
        agent_metrics['buy_actions'],
        agent_metrics['hold_actions'],
        agent_metrics['sell_actions']
    ]
    colors = ['#06A77D', '#F5B700', '#D62246']
    ax3.bar(actions, action_counts, color=colors, alpha=0.7)
    ax3.set_title('Action Distribution', fontweight='bold')
    ax3.set_ylabel('Count')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. Metrics Comparison
    ax4 = axes[1, 1]
    metrics_to_compare = [
        ('Profit %', agent_metrics['total_profit_percent'], bnh_metrics['total_profit_percent']),
        ('Sharpe', agent_metrics['sharpe_ratio'], bnh_metrics['sharpe_ratio']),
        ('Max DD %', abs(agent_metrics['max_drawdown']*100), abs(bnh_metrics['max_drawdown']*100))
    ]

    x = np.arange(len(metrics_to_compare))
    width = 0.35

    agent_values = [m[1] for m in metrics_to_compare]
    bnh_values_comp = [m[2] for m in metrics_to_compare]

    ax4.bar(x - width/2, agent_values, width, label='DQN Agent', color='#2E86AB', alpha=0.7)
    ax4.bar(x + width/2, bnh_values_comp, width, label='Buy & Hold', color='#A23B72', alpha=0.7)
    ax4.set_title('Performance Metrics Comparison', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([m[0] for m in metrics_to_compare])
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'evaluation_plot.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {plot_path}")

    plt.close()


def print_evaluation_report(agent_metrics, bnh_metrics, agent_info, ticker, logger):
    """
    Print formatted evaluation report.

    Args:
        agent_metrics (dict): Agent metrics
        bnh_metrics (dict): Buy-and-hold metrics
        agent_info (dict): Agent trading info
        ticker (str): Ticker symbol
        logger: Logger instance
    """
    print("\n" + "="*70)
    print(" FinSense DQN Agent - Evaluation Report")
    print("="*70)
    print(f"\nTicker: {ticker}")
    print(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Timesteps: {agent_metrics['total_timesteps']}")
    print(f"Total Trades: {agent_metrics['total_trades']}")

    print("\n" + "-"*70)
    print(" Performance Comparison")
    print("-"*70)

    # Format table
    print(f"\n{'Metric':<30} {'DQN Agent':<20} {'Buy & Hold':<20}")
    print("-"*70)

    metrics_to_show = [
        ('Total Profit', f"₹{agent_metrics['total_profit']:.2f}", f"₹{bnh_metrics['total_profit']:.2f}"),
        ('Profit %', f"{agent_metrics['total_profit_percent']:.2f}%", f"{bnh_metrics['total_profit_percent']:.2f}%"),
        ('Sharpe Ratio', f"{agent_metrics['sharpe_ratio']:.4f}", f"{bnh_metrics['sharpe_ratio']:.4f}"),
        ('Sortino Ratio', f"{agent_metrics['sortino_ratio']:.4f}", f"{bnh_metrics['sortino_ratio']:.4f}"),
        ('Max Drawdown', f"{agent_metrics['max_drawdown_percent']:.2f}%", f"{bnh_metrics['max_drawdown_percent']:.2f}%"),
        ('Calmar Ratio', f"{agent_metrics['calmar_ratio']:.4f}", f"{bnh_metrics['calmar_ratio']:.4f}"),
        ('Win Rate', f"{agent_metrics['win_rate']*100:.2f}%", "N/A"),
        ('Profit Factor', f"{agent_metrics['profit_factor']:.4f}", "N/A"),
    ]

    for metric, agent_val, bnh_val in metrics_to_show:
        print(f"{metric:<30} {agent_val:<20} {bnh_val:<20}")

    print("\n" + "-"*70)
    print(" Trading Activity")
    print("-"*70)
    print(f"\nBuy Actions:  {agent_metrics['buy_actions']}")
    print(f"Hold Actions: {agent_metrics['hold_actions']}")
    print(f"Sell Actions: {agent_metrics['sell_actions']}")
    print(f"\nFinal Balance: ₹{agent_info['final_balance']:.2f}")
    print(f"Final Inventory: {agent_info['final_inventory']} shares")
    print(f"Final Portfolio Value: ₹{agent_info['final_portfolio_value']:.2f}")

    # Determine winner
    print("\n" + "="*70)
    if agent_metrics['total_profit'] > bnh_metrics['total_profit']:
        improvement = ((agent_metrics['total_profit'] / bnh_metrics['total_profit']) - 1) * 100
        print(f" ✅ DQN Agent outperformed Buy & Hold by {improvement:.2f}%")
    else:
        underperformance = ((bnh_metrics['total_profit'] / agent_metrics['total_profit']) - 1) * 100
        print(f" ⚠️  Buy & Hold outperformed DQN Agent by {underperformance:.2f}%")
    print("="*70 + "\n")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('evaluate', level=log_level)

    logger.info("="*70)
    logger.info(" FinSense Evaluation Started")
    logger.info("="*70)

    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Create output directory
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(config.get_section('data'))

        if args.data:
            # Custom CSV data
            data = data_loader.load_from_csv(args.data)
            ticker = Path(args.data).stem
        elif args.ticker:
            # yfinance data
            data = data_loader.load_data(ticker=args.ticker)
            ticker = args.ticker
        else:
            # Use default ticker from config
            ticker = config.get('data.ticker', 'RELIANCE.NS')
            data = data_loader.load_data(ticker=ticker)

        # Split data
        train_data, val_data, test_data = data_loader.train_test_split(data)

        # Select split to evaluate on
        if args.split == 'test':
            eval_data = test_data
        elif args.split == 'validation':
            eval_data = val_data
        elif args.split == 'train':
            eval_data = train_data
        else:  # full
            eval_data = data

        logger.info(f"Evaluating on {args.split} split: {len(eval_data['close'])} points")
        logger.info(f"Price range: ₹{eval_data['close'].min():.2f} - ₹{eval_data['close'].max():.2f}")

        # Load model
        logger.info(f"Loading model from {args.model}...")
        model_path = Path(args.model)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Get state size from config
        window_size = config.get('environment.window_size', 10)
        use_volume = config.get('environment.use_volume', True)
        use_indicators = config.get('environment.use_technical_indicators', True)

        # Calculate state size (same as in train.py)
        state_size = window_size - 1  # Price differences
        if use_volume:
            state_size += 1
        if use_indicators:
            state_size += 6  # RSI(1) + MACD(3) + BB(1) + ATR(1)

        # Create agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=3,
            config=config.get_section('agent')
        )

        # Load weights
        agent.load(model_path)
        logger.info(f"Model loaded successfully (state_size={state_size})")

        # Get starting balance
        starting_balance = config.get('environment.starting_balance', 50000)

        # Evaluate agent
        logger.info("\n" + "="*70)
        logger.info(" Evaluating DQN Agent")
        logger.info("="*70)
        agent_metrics, agent_info = evaluate_agent(agent, eval_data, config, starting_balance, logger)

        # Calculate buy-and-hold baseline
        logger.info("\n" + "="*70)
        logger.info(" Calculating Buy & Hold Baseline")
        logger.info("="*70)
        bnh_metrics, bnh_values = calculate_buy_and_hold_metrics(eval_data, starting_balance, config)

        # Print report
        print_evaluation_report(agent_metrics, bnh_metrics, agent_info, ticker, logger)

        # Save results to JSON
        results = {
            'ticker': ticker,
            'split': args.split,
            'evaluation_date': datetime.now().isoformat(),
            'model_path': str(model_path),
            'agent_metrics': agent_metrics,
            'buy_and_hold_metrics': bnh_metrics,
            'agent_info': {
                'total_timesteps': agent_metrics['total_timesteps'],
                'final_balance': agent_info['final_balance'],
                'final_inventory': agent_info['final_inventory'],
                'final_portfolio_value': agent_info['final_portfolio_value']
            }
        }

        results_path = output_dir / f'evaluation_results_{ticker}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults saved to {results_path}")

        # Generate visualizations if requested
        if args.visualize:
            generate_visualizations(
                agent_metrics, agent_info, bnh_metrics, bnh_values,
                eval_data, output_dir, logger
            )

        logger.info("\n" + "="*70)
        logger.info(" Evaluation Complete!")
        logger.info("="*70)

        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
