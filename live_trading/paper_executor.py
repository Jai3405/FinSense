"""
Paper Trading Executor.

Simulates trade execution without real money for validation.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging


class PaperTradingExecutor:
    """
    Paper trading simulator with realistic transaction costs.

    Tracks portfolio, executes simulated trades, and calculates
    performance metrics.
    """

    def __init__(
        self,
        starting_balance: float = 50000,
        max_positions: int = 40,
        max_position_value: float = 0.95,
        transaction_costs: bool = True
    ):
        """
        Initialize paper trading executor.

        Args:
            starting_balance: Initial cash balance
            max_positions: Maximum shares to hold
            max_position_value: Maximum % of capital to invest
            transaction_costs: Include realistic transaction costs
        """
        self.starting_balance = starting_balance
        self.max_positions = max_positions
        self.max_position_value = max_position_value
        self.include_transaction_costs = transaction_costs

        # Portfolio state
        self.balance = starting_balance
        self.inventory = 0  # Current shares held
        self.avg_buy_price = 0.0  # Average price of shares in inventory

        # Trading history
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []

        # Performance tracking
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0

        # Drawdown tracking
        self.peak_value = starting_balance
        self.max_drawdown = 0.0

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Paper Trading Executor initialized: ₹{starting_balance:,.2f}")

    def _calculate_transaction_cost(self, price: float, shares: int) -> float:
        """
        Calculate realistic transaction costs (Zerodha-style).

        Includes:
        - Brokerage: ₹20 or 0.03% (whichever is lower)
        - STT: 0.025% on sell
        - Exchange charges: 0.00297%
        - SEBI charges: 0.0001%
        - Stamp duty: 0.003% on buy
        - GST: 18% on (brokerage + exchange + SEBI)

        Returns:
            float: Total transaction cost
        """
        if not self.include_transaction_costs:
            return 0.0

        trade_value = price * shares

        # Brokerage: ₹20 or 0.03%, whichever is lower
        brokerage = min(20, trade_value * 0.0003)

        # Exchange charges
        exchange_charges = trade_value * 0.0000297

        # SEBI charges
        sebi_charges = trade_value * 0.000001

        # GST on (brokerage + exchange + SEBI)
        gst = (brokerage + exchange_charges + sebi_charges) * 0.18

        # Total cost (will be doubled for buy+sell round trip)
        total = brokerage + exchange_charges + sebi_charges + gst

        return total

    def can_buy(self, price: float, shares: int = 1) -> bool:
        """Check if buy order is valid."""
        if self.inventory >= self.max_positions:
            return False

        total_cost = price * shares
        transaction_cost = self._calculate_transaction_cost(price, shares)
        required_capital = total_cost + transaction_cost

        # Check balance
        if self.balance < required_capital:
            return False

        # Check max position value constraint
        future_inventory_value = (self.inventory + shares) * price
        future_portfolio_value = self.balance - required_capital + future_inventory_value

        if future_inventory_value > future_portfolio_value * self.max_position_value:
            return False

        return True

    def can_sell(self, shares: int = 1) -> bool:
        """Check if sell order is valid."""
        return self.inventory >= shares

    def get_action_mask(self, current_price: float) -> np.ndarray:
        """
        Get action mask for PPO agent.

        Returns:
            np.ndarray: [buy_allowed, hold_allowed, sell_allowed]
        """
        return np.array([
            self.can_buy(current_price, shares=1),
            True,  # HOLD always allowed
            self.can_sell(shares=1)
        ], dtype=bool)

    def execute_buy(self, price: float, timestamp: datetime, shares: int = 1) -> Dict:
        """
        Execute BUY order.

        Args:
            price: Current stock price
            timestamp: Trade timestamp
            shares: Number of shares to buy

        Returns:
            dict: Trade information
        """
        if not self.can_buy(price, shares):
            return {
                'success': False,
                'reason': 'Insufficient balance or max positions reached'
            }

        # Calculate costs
        trade_value = price * shares
        transaction_cost = self._calculate_transaction_cost(price, shares)
        total_cost = trade_value + transaction_cost

        # Update portfolio
        self.balance -= total_cost

        # Update average buy price (weighted average)
        total_shares = self.inventory + shares
        self.avg_buy_price = (
            (self.avg_buy_price * self.inventory + price * shares) / total_shares
        )
        self.inventory += shares

        # Record trade
        trade = {
            'timestamp': timestamp,
            'action': 'BUY',
            'price': price,
            'shares': shares,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost,
            'total_cost': total_cost,
            'balance_after': self.balance,
            'inventory_after': self.inventory,
            'portfolio_value': self.get_portfolio_value(price),
            'success': True
        }

        self.trades.append(trade)
        self.total_trades += 1

        self.logger.info(
            f"BUY: {shares} shares @ ₹{price:.2f} | "
            f"Cost: ₹{total_cost:.2f} (TX: ₹{transaction_cost:.2f}) | "
            f"Balance: ₹{self.balance:.2f} | Inventory: {self.inventory}"
        )

        return trade

    def execute_sell(self, price: float, timestamp: datetime, shares: int = 1) -> Dict:
        """
        Execute SELL order.

        Args:
            price: Current stock price
            timestamp: Trade timestamp
            shares: Number of shares to sell

        Returns:
            dict: Trade information
        """
        if not self.can_sell(shares):
            return {
                'success': False,
                'reason': 'No shares to sell'
            }

        # Calculate revenue
        trade_value = price * shares
        transaction_cost = self._calculate_transaction_cost(price, shares)

        # Add STT for sell (0.025%)
        stt = trade_value * 0.00025
        transaction_cost += stt

        # Add stamp duty for sell (0.003%)
        stamp_duty = trade_value * 0.00003
        transaction_cost += stamp_duty

        net_revenue = trade_value - transaction_cost

        # Calculate P&L for this trade
        cost_basis = self.avg_buy_price * shares
        pnl = net_revenue - cost_basis
        pnl_pct = (pnl / cost_basis) * 100 if cost_basis > 0 else 0

        # Update portfolio
        self.balance += net_revenue
        self.inventory -= shares

        # Update performance stats
        if pnl > 0:
            self.winning_trades += 1
            self.total_profit += pnl
        else:
            self.losing_trades += 1
            self.total_loss += abs(pnl)

        # Record trade
        trade = {
            'timestamp': timestamp,
            'action': 'SELL',
            'price': price,
            'shares': shares,
            'trade_value': trade_value,
            'transaction_cost': transaction_cost,
            'net_revenue': net_revenue,
            'cost_basis': cost_basis,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'balance_after': self.balance,
            'inventory_after': self.inventory,
            'portfolio_value': self.get_portfolio_value(price),
            'success': True
        }

        self.trades.append(trade)
        self.total_trades += 1

        self.logger.info(
            f"SELL: {shares} shares @ ₹{price:.2f} | "
            f"Revenue: ₹{net_revenue:.2f} (TX: ₹{transaction_cost:.2f}) | "
            f"P&L: ₹{pnl:+.2f} ({pnl_pct:+.2f}%) | "
            f"Balance: ₹{self.balance:.2f} | Inventory: {self.inventory}"
        )

        return trade

    def execute_hold(self, price: float, timestamp: datetime) -> Dict:
        """Record HOLD action (no trade)."""
        portfolio_value = self.get_portfolio_value(price)

        # Still track equity curve
        return {
            'timestamp': timestamp,
            'action': 'HOLD',
            'price': price,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'inventory': self.inventory,
            'success': True
        }

    def execute_action(
        self,
        action: int,
        price: float,
        timestamp: datetime,
        shares: int = 1
    ) -> Dict:
        """
        Execute action (BUY/HOLD/SELL).

        Args:
            action: 0=BUY, 1=HOLD, 2=SELL
            price: Current price
            timestamp: Timestamp
            shares: Number of shares

        Returns:
            dict: Trade result
        """
        if action == 0:  # BUY
            return self.execute_buy(price, timestamp, shares)
        elif action == 2:  # SELL
            return self.execute_sell(price, timestamp, shares)
        else:  # HOLD
            return self.execute_hold(price, timestamp)

    def get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        return self.balance + (self.inventory * current_price)

    def update_equity_curve(self, timestamp: datetime, current_price: float):
        """Update equity curve for plotting."""
        portfolio_value = self.get_portfolio_value(current_price)

        # Track drawdown
        if portfolio_value > self.peak_value:
            self.peak_value = portfolio_value

        current_drawdown = (self.peak_value - portfolio_value) / self.peak_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        self.equity_curve.append({
            'timestamp': timestamp,
            'portfolio_value': portfolio_value,
            'balance': self.balance,
            'inventory': self.inventory,
            'inventory_value': self.inventory * current_price,
            'drawdown': current_drawdown
        })

    def get_metrics(self, current_price: float) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Returns:
            dict: Performance metrics
        """
        portfolio_value = self.get_portfolio_value(current_price)
        total_return = portfolio_value - self.starting_balance
        total_return_pct = (total_return / self.starting_balance) * 100

        # Win rate
        completed_trades = self.winning_trades + self.losing_trades
        win_rate = (self.winning_trades / completed_trades * 100) if completed_trades > 0 else 0

        # Profit factor
        profit_factor = (self.total_profit / self.total_loss) if self.total_loss > 0 else float('inf')

        # Expectancy per trade
        expectancy = total_return / completed_trades if completed_trades > 0 else 0

        # Calculate Sharpe ratio from equity curve
        sharpe = self._calculate_sharpe_ratio()

        return {
            'starting_balance': self.starting_balance,
            'current_balance': self.balance,
            'inventory': self.inventory,
            'inventory_value': self.inventory * current_price,
            'avg_cost': self.avg_buy_price,
            'unrealized_pnl': (current_price - self.avg_buy_price) * self.inventory if self.inventory > 0 else 0,
            'portfolio_value': portfolio_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': self.total_trades,
            'completed_trades': completed_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_profit': self.total_profit,
            'total_loss': self.total_loss,
            'profit_factor': profit_factor,
            'expectancy_per_trade': expectancy,
            'max_drawdown': self.max_drawdown * 100,
            'sharpe_ratio': sharpe,
            'peak_value': self.peak_value
        }

    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from equity curve."""
        if len(self.equity_curve) < 2:
            return 0.0

        # Calculate returns
        values = [point['portfolio_value'] for point in self.equity_curve]
        returns = np.diff(values) / values[:-1]

        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0

        # Sharpe = mean(returns) / std(returns) * sqrt(periods per year)
        # For daily data, periods = 252 trading days
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252)

        return sharpe

    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        return pd.DataFrame(self.trades)

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        return pd.DataFrame(self.equity_curve)

    def save_trades(self, filepath: str):
        """Save trade history to CSV."""
        df = self.get_trade_history()
        df.to_csv(filepath, index=False)
        self.logger.info(f"Trades saved to {filepath}")

    def get_summary(self, current_price: float) -> str:
        """Get formatted summary of performance."""
        metrics = self.get_metrics(current_price)

        summary = f"""
{'='*70}
PAPER TRADING SUMMARY
{'='*70}

Portfolio Status:
  Starting Balance:  ₹{metrics['starting_balance']:,.2f}
  Current Balance:   ₹{metrics['current_balance']:,.2f}
  Current Inventory: {metrics['inventory']} shares @ ₹{current_price:.2f}
  Inventory Value:   ₹{metrics['inventory_value']:,.2f}
  Portfolio Value:   ₹{metrics['portfolio_value']:,.2f}

Performance:
  Total Return:      ₹{metrics['total_return']:+,.2f} ({metrics['total_return_pct']:+.2f}%)
  Max Drawdown:      {metrics['max_drawdown']:.2f}%
  Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}

Trading Statistics:
  Total Trades:      {metrics['total_trades']}
  Completed Trades:  {metrics['completed_trades']} ({metrics['winning_trades']}W / {metrics['losing_trades']}L)
  Win Rate:          {metrics['win_rate']:.2f}%
  Profit Factor:     {metrics['profit_factor']:.2f}
  Expectancy/Trade:  ₹{metrics['expectancy_per_trade']:.2f}

Profit/Loss:
  Total Profit:      ₹{metrics['total_profit']:,.2f}
  Total Loss:        ₹{metrics['total_loss']:,.2f}
  Net P&L:           ₹{metrics['total_return']:+,.2f}

{'='*70}
"""
        return summary
