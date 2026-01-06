"""
Performance metrics for trading strategies.
Implements Sharpe ratio, drawdown, win rate, Sortino, Calmar, etc.
"""

import numpy as np
import pandas as pd
import logging
from typing import List, Tuple, Dict


logger = logging.getLogger(__name__)


class TradingMetrics:
    """
    Calculate comprehensive trading performance metrics.

    Provides risk-adjusted return metrics including:
    - Sharpe ratio
    - Sortino ratio
    - Calmar ratio
    - Maximum drawdown
    - Win rate / Loss rate
    - Profit factor
    - Average profit per trade
    """

    def __init__(self, risk_free_rate=0.02):
        """
        Initialize metrics calculator.

        Args:
            risk_free_rate (float): Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate

    def sharpe_ratio(self, returns, periods_per_year=252):
        """
        Calculate Sharpe ratio (risk-adjusted returns).

        Sharpe = (Mean Return - Risk Free Rate) / Std Dev of Returns

        Args:
            returns (np.ndarray or pd.Series): Period returns
            periods_per_year (int): Trading periods per year (252 for daily, 52 for weekly)

        Returns:
            float: Sharpe ratio
        """
        if len(returns) < 2:
            return 0.0

        returns = np.array(returns)

        # Annualized mean return
        mean_return = np.mean(returns) * periods_per_year

        # Annualized standard deviation
        std_return = np.std(returns) * np.sqrt(periods_per_year)

        if std_return == 0:
            return 0.0

        sharpe = (mean_return - self.risk_free_rate) / std_return

        return sharpe

    def sortino_ratio(self, returns, periods_per_year=252):
        """
        Calculate Sortino ratio (like Sharpe but only penalizes downside volatility).

        Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

        Args:
            returns (np.ndarray or pd.Series): Period returns
            periods_per_year (int): Trading periods per year

        Returns:
            float: Sortino ratio
        """
        if len(returns) < 2:
            return 0.0

        returns = np.array(returns)

        # Annualized mean return
        mean_return = np.mean(returns) * periods_per_year

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return np.inf if mean_return > self.risk_free_rate else 0.0

        downside_std = np.std(downside_returns) * np.sqrt(periods_per_year)

        if downside_std == 0:
            return 0.0

        sortino = (mean_return - self.risk_free_rate) / downside_std

        return sortino

    def max_drawdown(self, portfolio_values):
        """
        Calculate maximum drawdown (worst peak-to-trough decline).

        Args:
            portfolio_values (np.ndarray or pd.Series): Portfolio values over time

        Returns:
            tuple: (max_drawdown, peak_idx, trough_idx)
        """
        if len(portfolio_values) < 2:
            return 0.0, 0, 0

        portfolio_values = np.array(portfolio_values)

        # Calculate running maximum
        peak = np.maximum.accumulate(portfolio_values)

        # Calculate drawdown at each point
        drawdown = (portfolio_values - peak) / peak

        # Find maximum drawdown
        max_dd = drawdown.min()
        max_dd_idx = drawdown.argmin()

        # Find the peak that led to this drawdown
        peak_idx = np.argmax(portfolio_values[:max_dd_idx+1]) if max_dd_idx > 0 else 0

        return max_dd, peak_idx, max_dd_idx

    def calmar_ratio(self, returns, portfolio_values, periods_per_year=252):
        """
        Calculate Calmar ratio (return / maximum drawdown).

        Args:
            returns (np.ndarray): Period returns
            portfolio_values (np.ndarray): Portfolio values
            periods_per_year (int): Trading periods per year

        Returns:
            float: Calmar ratio
        """
        if len(returns) < 2:
            return 0.0

        # Annualized return
        total_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
        years = len(returns) / periods_per_year
        ann_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0

        # Maximum drawdown
        max_dd, _, _ = self.max_drawdown(portfolio_values)

        if max_dd == 0:
            return np.inf if ann_return > 0 else 0.0

        return ann_return / abs(max_dd)

    def win_rate(self, trades):
        """
        Calculate win rate (percentage of profitable trades).

        Args:
            trades (list): List of trade profits/losses

        Returns:
            float: Win rate (0-1)
        """
        if len(trades) == 0:
            return 0.0

        winning_trades = sum(1 for t in trades if t > 0)
        return winning_trades / len(trades)

    def loss_rate(self, trades):
        """Calculate loss rate (percentage of losing trades)."""
        return 1 - self.win_rate(trades)

    def profit_factor(self, trades):
        """
        Calculate profit factor (gross profit / gross loss).

        Args:
            trades (list): List of trade profits/losses

        Returns:
            float: Profit factor
        """
        if len(trades) == 0:
            return 0.0

        gross_profit = sum(t for t in trades if t > 0)
        gross_loss = abs(sum(t for t in trades if t < 0))

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def average_profit_per_trade(self, trades):
        """
        Calculate average profit per trade.

        Args:
            trades (list): List of trade profits/losses

        Returns:
            float: Average profit per trade
        """
        if len(trades) == 0:
            return 0.0

        return np.mean(trades)

    def average_win(self, trades):
        """Calculate average winning trade."""
        winning_trades = [t for t in trades if t > 0]
        return np.mean(winning_trades) if winning_trades else 0.0

    def average_loss(self, trades):
        """Calculate average losing trade."""
        losing_trades = [t for t in trades if t < 0]
        return np.mean(losing_trades) if losing_trades else 0.0

    def expectancy(self, trades):
        """
        Calculate expectancy (expected value per trade).

        Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)

        Args:
            trades (list): List of trade profits/losses

        Returns:
            float: Expectancy
        """
        if len(trades) == 0:
            return 0.0

        win_rate = self.win_rate(trades)
        avg_win = self.average_win(trades)
        avg_loss = abs(self.average_loss(trades))

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def calculate_all_metrics(self, portfolio_values, trades, returns=None):
        """
        Calculate all metrics at once.

        Args:
            portfolio_values (list/np.ndarray): Portfolio values over time
            trades (list): List of individual trade profits/losses
            returns (list/np.ndarray): Period returns (calculated if not provided)

        Returns:
            dict: Dictionary of all metrics
        """
        portfolio_values = np.array(portfolio_values)

        # Calculate returns if not provided
        if returns is None:
            returns = np.diff(portfolio_values) / portfolio_values[:-1]
        else:
            returns = np.array(returns)

        # Calculate all metrics
        max_dd, peak_idx, trough_idx = self.max_drawdown(portfolio_values)

        metrics = {
            # Profit metrics
            'total_profit': portfolio_values[-1] - portfolio_values[0],
            'total_return': (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],

            # Risk-adjusted metrics
            'sharpe_ratio': self.sharpe_ratio(returns),
            'sortino_ratio': self.sortino_ratio(returns),
            'calmar_ratio': self.calmar_ratio(returns, portfolio_values),

            # Drawdown metrics
            'max_drawdown': max_dd,
            'max_drawdown_percent': max_dd * 100,

            # Trade metrics
            'total_trades': len(trades),
            'win_rate': self.win_rate(trades),
            'loss_rate': self.loss_rate(trades),
            'profit_factor': self.profit_factor(trades),
            'avg_profit_per_trade': self.average_profit_per_trade(trades),
            'avg_win': self.average_win(trades),
            'avg_loss': self.average_loss(trades),
            'expectancy': self.expectancy(trades),

            # Volatility
            'volatility': np.std(returns) * np.sqrt(252),  # Annualized
        }

        return metrics

    def print_metrics(self, metrics):
        """
        Print metrics in a formatted way.

        Args:
            metrics (dict): Metrics dictionary from calculate_all_metrics()
        """
        print("\n" + "=" * 60)
        print("PERFORMANCE METRICS")
        print("=" * 60)

        print(f"\nProfit Metrics:")
        print(f"  Total Profit: ₹{metrics['total_profit']:.2f}")
        print(f"  Total Return: {metrics['total_return']*100:.2f}%")

        print(f"\nRisk-Adjusted Metrics:")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Calmar Ratio: {metrics['calmar_ratio']:.3f}")

        print(f"\nRisk Metrics:")
        print(f"  Max Drawdown: {metrics['max_drawdown_percent']:.2f}%")
        print(f"  Volatility (Annual): {metrics['volatility']*100:.2f}%")

        print(f"\nTrade Metrics:")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"  Profit Factor: {metrics['profit_factor']:.3f}")
        print(f"  Avg Profit/Trade: ₹{metrics['avg_profit_per_trade']:.2f}")
        print(f"  Avg Win: ₹{metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ₹{metrics['avg_loss']:.2f}")
        print(f"  Expectancy: ₹{metrics['expectancy']:.2f}")

        print("=" * 60 + "\n")
