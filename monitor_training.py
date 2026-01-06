#!/usr/bin/env python3
"""
Real-time training monitor for FinSense.
Shows live progress, metrics, and TensorBoard link.
"""

import time
import sys
from pathlib import Path
from datetime import datetime
import re


def parse_log_line(line):
    """Parse training log line to extract metrics."""
    # Episode X/100 | Profit: ₹123.45 | Trades: 5 | Sharpe: 0.123 | Epsilon: 0.456 | Loss: 0.000123

    if "Episode" not in line:
        return None

    metrics = {}

    # Extract episode
    episode_match = re.search(r'Episode (\d+)/(\d+)', line)
    if episode_match:
        metrics['episode'] = int(episode_match.group(1))
        metrics['total_episodes'] = int(episode_match.group(2))

    # Extract profit
    profit_match = re.search(r'Profit: ₹?([-\d.]+)', line)
    if profit_match:
        metrics['profit'] = float(profit_match.group(1))

    # Extract trades
    trades_match = re.search(r'Trades: (\d+)', line)
    if trades_match:
        metrics['trades'] = int(trades_match.group(1))

    # Extract Sharpe
    sharpe_match = re.search(r'Sharpe: ([-\d.]+)', line)
    if sharpe_match:
        metrics['sharpe'] = float(sharpe_match.group(1))

    # Extract epsilon
    epsilon_match = re.search(r'Epsilon: ([-\d.]+)', line)
    if epsilon_match:
        metrics['epsilon'] = float(epsilon_match.group(1))

    # Extract loss
    loss_match = re.search(r'Loss: ([-\d.]+)', line)
    if loss_match:
        metrics['loss'] = float(loss_match.group(1))

    return metrics if metrics else None


def format_progress_bar(current, total, width=40):
    """Create ASCII progress bar."""
    filled = int(width * current / total)
    bar = '█' * filled + '░' * (width - filled)
    percent = 100 * current / total
    return f"[{bar}] {percent:.1f}%"


def print_dashboard(metrics_history, start_time):
    """Print live training dashboard."""
    # Clear screen (works on Unix/Mac/Windows)
    print('\033[2J\033[H', end='')

    print("=" * 80)
    print(" " * 20 + "🚀 FinSense Training Monitor 🚀")
    print("=" * 80)
    print()

    if not metrics_history:
        print("Waiting for training to start...")
        return

    latest = metrics_history[-1]

    # Progress
    print("📊 PROGRESS:")
    print(f"   Episode: {latest['episode']}/{latest['total_episodes']}")
    progress_bar = format_progress_bar(latest['episode'], latest['total_episodes'])
    print(f"   {progress_bar}")

    # Time info
    elapsed = time.time() - start_time
    elapsed_mins = elapsed / 60
    avg_time_per_ep = elapsed / latest['episode'] if latest['episode'] > 0 else 0
    remaining_eps = latest['total_episodes'] - latest['episode']
    eta_mins = (avg_time_per_ep * remaining_eps) / 60

    print(f"   Elapsed: {elapsed_mins:.1f} min | ETA: {eta_mins:.1f} min")
    print()

    # Latest metrics
    print("📈 LATEST METRICS (Episode {}):", latest['episode'])
    print(f"   Profit: ₹{latest.get('profit', 0):.2f}")
    print(f"   Trades: {latest.get('trades', 0)}")
    print(f"   Sharpe Ratio: {latest.get('sharpe', 0):.4f}")
    print(f"   Epsilon: {latest.get('epsilon', 0):.4f}")
    print(f"   Loss: {latest.get('loss', 0):.6f}")
    print()

    # Summary statistics (last 10 episodes)
    if len(metrics_history) >= 2:
        recent = metrics_history[-10:]
        avg_profit = sum(m.get('profit', 0) for m in recent) / len(recent)
        avg_trades = sum(m.get('trades', 0) for m in recent) / len(recent)
        avg_sharpe = sum(m.get('sharpe', 0) for m in recent) / len(recent)
        best_profit = max(m.get('profit', 0) for m in metrics_history)
        best_sharpe = max(m.get('sharpe', 0) for m in metrics_history)

        print("📊 STATISTICS (Last 10 Episodes):")
        print(f"   Avg Profit: ₹{avg_profit:.2f}")
        print(f"   Avg Trades: {avg_trades:.1f}")
        print(f"   Avg Sharpe: {avg_sharpe:.4f}")
        print()

        print("🏆 BEST SO FAR:")
        print(f"   Best Profit: ₹{best_profit:.2f}")
        print(f"   Best Sharpe: {best_sharpe:.4f}")
        print()

    # Recent trend (mini chart of last 20 profits)
    if len(metrics_history) > 1:
        print("📉 PROFIT TREND (Last 20 Episodes):")
        recent_profits = [m.get('profit', 0) for m in metrics_history[-20:]]

        if recent_profits:
            max_profit = max(recent_profits) if max(recent_profits) > 0 else 1
            min_profit = min(recent_profits) if min(recent_profits) < 0 else 0
            range_profit = max_profit - min_profit if max_profit - min_profit > 0 else 1

            # Simple ASCII chart
            print("   ", end="")
            for p in recent_profits:
                # Normalize to 0-5 range
                normalized = int(5 * (p - min_profit) / range_profit)
                bars = ['_', '▁', '▃', '▅', '▇', '█']
                print(bars[normalized], end="")
            print()
        print()

    # Instructions
    print("=" * 80)
    print("💡 TIP: Open TensorBoard for detailed visualizations:")
    print("   tensorboard --logdir runs/")
    print()
    print("🛑 Press Ctrl+C to stop monitoring (training will continue)")
    print("=" * 80)


def monitor_training(log_file=None):
    """Monitor training progress from log file."""

    if log_file is None:
        # Find most recent log file
        log_dir = Path('logs')
        if not log_dir.exists():
            log_dir.mkdir(exist_ok=True)

        log_files = sorted(log_dir.glob('training_*.log'), key=lambda x: x.stat().st_mtime)
        if not log_files:
            print("No training log found. Start training first!")
            return

        log_file = log_files[-1]
    else:
        log_file = Path(log_file)

    print(f"Monitoring: {log_file}")
    time.sleep(1)

    metrics_history = []
    start_time = time.time()

    try:
        with open(log_file, 'r') as f:
            # Move to end of file
            f.seek(0, 2)

            while True:
                line = f.readline()

                if line:
                    # Parse line
                    metrics = parse_log_line(line)
                    if metrics:
                        metrics_history.append(metrics)
                        print_dashboard(metrics_history, start_time)
                else:
                    # No new line, wait a bit
                    time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped (training continues in background)")
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Monitor FinSense training progress')
    parser.add_argument('--log', type=str, default=None, help='Path to log file')
    args = parser.parse_args()

    monitor_training(args.log)
