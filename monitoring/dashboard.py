"""
Performance monitoring and dashboard for paper trading.

Tracks metrics, generates reports, and visualizes performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List
import logging
from pathlib import Path


class PaperTradingMonitor:
    """
    Real-time performance monitoring for paper trading.

    Tracks metrics, generates charts, and provides analytics.
    """

    def __init__(self, output_dir: str = 'logs/paper_trading'):
        """
        Initialize monitor.

        Args:
            output_dir: Directory to save logs and charts
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Metrics history
        self.metrics_history: List[Dict] = []

        # Backtest comparison (for divergence analysis)
        self.backtest_metrics = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Monitor initialized: {output_dir}")

    def log_metrics(self, timestamp: datetime, metrics: Dict):
        """
        Log current metrics.

        Args:
            timestamp: Current timestamp
            metrics: Performance metrics dictionary
        """
        metrics_with_time = {'timestamp': timestamp, **metrics}
        self.metrics_history.append(metrics_with_time)

        # Save to CSV (append mode)
        self._save_metrics_csv()

    def _save_metrics_csv(self):
        """Save metrics history to CSV."""
        if not self.metrics_history:
            return

        df = pd.DataFrame(self.metrics_history)
        csv_path = self.output_dir / 'metrics_history.csv'
        df.to_csv(csv_path, index=False)

    def plot_equity_curve(self, equity_curve: pd.DataFrame, save_path: str = None):
        """
        Plot equity curve with drawdown.

        Args:
            equity_curve: DataFrame with 'timestamp', 'portfolio_value', 'drawdown'
            save_path: Optional path to save chart
        """
        if equity_curve.empty:
            self.logger.warning("No equity curve data to plot")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Portfolio value
        ax1.plot(equity_curve['timestamp'], equity_curve['portfolio_value'],
                 label='Portfolio Value', color='blue', linewidth=2)
        ax1.axhline(y=equity_curve['portfolio_value'].iloc[0],
                    color='gray', linestyle='--', label='Starting Value')
        ax1.set_ylabel('Portfolio Value (₹)', fontsize=12)
        ax1.set_title('Paper Trading Performance', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'₹{x:,.0f}'))

        # Drawdown
        ax2.fill_between(equity_curve['timestamp'], 0,
                          -equity_curve['drawdown'] * 100,
                          color='red', alpha=0.3)
        ax2.plot(equity_curve['timestamp'], -equity_curve['drawdown'] * 100,
                 color='red', linewidth=1.5)
        ax2.set_ylabel('Drawdown (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.invert_yaxis()

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f'equity_curve_{datetime.now().strftime("%Y%m%d")}.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Equity curve saved: {save_path}")

    def plot_trade_analysis(self, trades_df: pd.DataFrame, save_path: str = None):
        """
        Plot trade analysis (P&L distribution, win/loss analysis).

        Args:
            trades_df: DataFrame with trade history
            save_path: Optional path to save chart
        """
        if trades_df.empty:
            return

        # Filter only completed trades (BUY/SELL with P&L)
        completed_trades = trades_df[trades_df['action'] == 'SELL'].copy()

        if completed_trades.empty:
            self.logger.warning("No completed trades to analyze")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

        # 1. P&L per trade
        ax1.bar(range(len(completed_trades)), completed_trades['pnl'],
                color=['green' if x > 0 else 'red' for x in completed_trades['pnl']])
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_xlabel('Trade Number')
        ax1.set_ylabel('P&L (₹)')
        ax1.set_title('P&L per Trade')
        ax1.grid(True, alpha=0.3)

        # 2. P&L distribution histogram
        ax2.hist(completed_trades['pnl'], bins=20, color='steelblue', edgecolor='black')
        ax2.axvline(x=0, color='red', linestyle='--', label='Break-even')
        ax2.axvline(x=completed_trades['pnl'].mean(), color='green',
                     linestyle='--', label=f'Mean: ₹{completed_trades["pnl"].mean():.2f}')
        ax2.set_xlabel('P&L (₹)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('P&L Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Cumulative P&L
        cumulative_pnl = completed_trades['pnl'].cumsum()
        ax3.plot(range(len(cumulative_pnl)), cumulative_pnl,
                 color='blue', linewidth=2)
        ax3.fill_between(range(len(cumulative_pnl)), 0, cumulative_pnl,
                          alpha=0.3, color='blue')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Trade Number')
        ax3.set_ylabel('Cumulative P&L (₹)')
        ax3.set_title('Cumulative P&L')
        ax3.grid(True, alpha=0.3)

        # 4. Win/Loss statistics
        wins = completed_trades[completed_trades['pnl'] > 0]
        losses = completed_trades[completed_trades['pnl'] < 0]

        win_count = len(wins)
        loss_count = len(losses)
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

        categories = ['Win Count', 'Loss Count', 'Avg Win', 'Avg Loss']
        values = [win_count, loss_count, avg_win, avg_loss]
        colors = ['green', 'red', 'lightgreen', 'lightcoral']

        ax4.bar(categories, values, color=colors, edgecolor='black')
        ax4.set_ylabel('Count / Amount (₹)')
        ax4.set_title('Win/Loss Analysis')
        ax4.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            ax4.text(i, v, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()

        # Save
        if save_path is None:
            save_path = self.output_dir / f'trade_analysis_{datetime.now().strftime("%Y%m%d")}.png'

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        self.logger.info(f"Trade analysis saved: {save_path}")

    def calculate_divergence(self, live_metrics: Dict, backtest_metrics: Dict) -> Dict:
        """
        Calculate divergence between live and backtest performance.

        Args:
            live_metrics: Current live trading metrics
            backtest_metrics: Backtest metrics from training

        Returns:
            dict: Divergence analysis
        """
        divergence = {}

        # Key metrics to compare
        metrics_to_compare = [
            'sharpe_ratio',
            'total_return_pct',
            'max_drawdown',
            'win_rate',
            'profit_factor'
        ]

        for metric in metrics_to_compare:
            live_value = live_metrics.get(metric, 0)
            backtest_value = backtest_metrics.get(metric, 0)

            if backtest_value != 0:
                diff = live_value - backtest_value
                diff_pct = (diff / backtest_value) * 100
            else:
                diff = live_value
                diff_pct = 0

            divergence[metric] = {
                'live': live_value,
                'backtest': backtest_value,
                'diff': diff,
                'diff_pct': diff_pct
            }

        return divergence

    def generate_daily_report(
        self,
        executor,
        current_price: float,
        backtest_metrics: Dict = None
    ) -> str:
        """
        Generate daily performance report.

        Args:
            executor: PaperTradingExecutor instance
            current_price: Current stock price
            backtest_metrics: Optional backtest metrics for comparison

        Returns:
            str: Formatted report
        """
        metrics = executor.get_metrics(current_price)
        timestamp = datetime.now()

        report = f"""
{'='*80}
PAPER TRADING DAILY REPORT - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

Portfolio Status:
  Starting Balance:       ₹{metrics['starting_balance']:>12,.2f}
  Current Balance:        ₹{metrics['current_balance']:>12,.2f}
  Current Inventory:      {metrics['inventory']:>4} shares @ ₹{current_price:.2f}
  Inventory Value:        ₹{metrics['inventory_value']:>12,.2f}
  Portfolio Value:        ₹{metrics['portfolio_value']:>12,.2f}
  Total Return:           ₹{metrics['total_return']:>12,.2f} ({metrics['total_return_pct']:+.2f}%)

Risk Metrics:
  Sharpe Ratio:           {metrics['sharpe_ratio']:>8.4f}
  Max Drawdown:           {metrics['max_drawdown']:>7.2f}%
  Peak Portfolio Value:   ₹{metrics['peak_value']:>12,.2f}

Trading Statistics:
  Total Actions:          {metrics['total_trades']:>4}
  Completed Trades:       {metrics['completed_trades']:>4} ({metrics['winning_trades']}W / {metrics['losing_trades']}L)
  Win Rate:               {metrics['win_rate']:>7.2f}%
  Profit Factor:          {metrics['profit_factor']:>7.2f}
  Expectancy/Trade:       ₹{metrics['expectancy_per_trade']:>8.2f}

Profit & Loss:
  Total Profit:           ₹{metrics['total_profit']:>12,.2f}
  Total Loss:             ₹{metrics['total_loss']:>12,.2f}
  Net P&L:                ₹{metrics['total_return']:>12,.2f}
"""

        # Add backtest comparison if available
        if backtest_metrics:
            divergence = self.calculate_divergence(metrics, backtest_metrics)

            report += f"""
{'='*80}
BACKTEST vs LIVE COMPARISON
{'='*80}

"""
            for metric, data in divergence.items():
                metric_name = metric.replace('_', ' ').title()
                report += f"  {metric_name}:\n"
                report += f"    Backtest:  {data['backtest']:>8.2f}\n"
                report += f"    Live:      {data['live']:>8.2f}\n"
                report += f"    Divergence: {data['diff']:>7.2f} ({data['diff_pct']:+.1f}%)\n\n"

            # Alert on high divergence
            sharpe_div = abs(divergence['sharpe_ratio']['diff_pct'])
            if sharpe_div > 20:
                report += f"\n⚠️  WARNING: Sharpe divergence >20% ({sharpe_div:.1f}%)\n"

        report += f"\n{'='*80}\n"

        return report

    def save_daily_report(self, report: str):
        """Save daily report to file."""
        filename = f"daily_report_{datetime.now().strftime('%Y%m%d')}.txt"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            f.write(report)

        self.logger.info(f"Daily report saved: {filepath}")

    def get_summary_stats(self) -> Dict:
        """Get summary statistics from metrics history."""
        if not self.metrics_history:
            return {}

        df = pd.DataFrame(self.metrics_history)

        return {
            'total_duration_days': (df['timestamp'].max() - df['timestamp'].min()).days,
            'total_updates': len(df),
            'final_sharpe': df['sharpe_ratio'].iloc[-1] if 'sharpe_ratio' in df else 0,
            'avg_sharpe': df['sharpe_ratio'].mean() if 'sharpe_ratio' in df else 0,
            'max_drawdown_overall': df['max_drawdown'].max() if 'max_drawdown' in df else 0,
            'final_return_pct': df['total_return_pct'].iloc[-1] if 'total_return_pct' in df else 0,
            'total_trades': df['total_trades'].iloc[-1] if 'total_trades' in df else 0
        }
