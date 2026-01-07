"""
Trading environment for FinSense.
Encapsulates trading logic and state management.
"""

import numpy as np
import logging
from typing import Tuple, Dict, Optional


logger = logging.getLogger(__name__)


class TradingEnvironment:
    """
    Trading environment for reinforcement learning.

    Manages:
    - Portfolio state (balance, inventory)
    - Trading actions (buy, hold, sell)
    - Reward calculation
    - Transaction costs
    """

    def __init__(self, data, config=None):
        """
        Initialize trading environment.

        Args:
            data (dict): Market data (close, high, low, volume)
            config (dict): Environment configuration
        """
        self.data = data
        self.config = config or self._default_config()

        # Environment parameters
        self.initial_balance = self.config.get('starting_balance', 50000)
        self.window_size = self.config.get('window_size', 10)

        # Position limits (prevents overtrading)
        self.max_positions = self.config.get('max_positions', 5)  # Max open positions
        self.max_position_value = self.config.get('max_position_value', 0.3)  # 30% of balance per position

        # Transaction costs
        self.transaction_cost = self.config.get('transaction_cost', 0.001)
        self.brokerage_per_trade = self.config.get('brokerage_per_trade', 20)
        self.brokerage_percentage = self.config.get('brokerage_percentage', 0.0003)

        # State
        self.current_step = 0
        self.balance = self.initial_balance
        self.inventory = []
        self.portfolio_values = [self.initial_balance]
        self.trades = []

        # Episode tracking
        self.episode_profit = 0.0
        self.episode_trades = 0

        # Track previous action for change penalty
        self.prev_action = 1  # Start with Hold

        logger.debug(f"Trading environment initialized with balance={self.initial_balance}")

    @staticmethod
    def _default_config():
        """Default environment configuration."""
        return {
            'starting_balance': 50000,
            'window_size': 10,
            'max_positions': 5,
            'max_position_value': 0.3,
            'transaction_cost': 0.001,
            'brokerage_per_trade': 20,
            'brokerage_percentage': 0.0003
        }

    def reset(self):
        """
        Reset environment for new episode.

        Returns:
            int: Starting timestep
        """
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.inventory = []
        self.portfolio_values = [self.initial_balance]
        self.trades = []
        self.episode_profit = 0.0
        self.episode_trades = 0
        self.prev_action = 1  # Reset to Hold

        logger.debug("Environment reset")
        return self.current_step

    def step(self, action):
        """
        Execute trading action with a robust penalty for failed attempts.

        Args:
            action (int): 0=Buy, 1=Hold, 2=Sell

        Returns:
            tuple: (reward, done, info)
        """
        done = False
        
        # --- Store state BEFORE action to detect failed trades ---
        prev_balance = self.balance
        prev_inventory_size = len(self.inventory)
        # ---------------------------------------------------------

        current_price = self.data['close'][self.current_step]
        prev_portfolio_value = self._calculate_portfolio_value(current_price)

        # Execute action
        _, info = self._execute_action(action, current_price)

        new_portfolio_value = self._calculate_portfolio_value(current_price)

        # 1. Base reward is equity delta
        equity_delta = new_portfolio_value - prev_portfolio_value
        reward = equity_delta

        # 2. Penalize changing action to discourage churn
        if action != self.prev_action:
            atr_value = self.data.get('atr', [1.0] * len(self.data['close']))[self.current_step]
            reward -= 0.001 * atr_value

        # 3. Penalize holding with no positions (opportunity cost)
        if len(self.inventory) == 0 and action == 1:
            atr_value = self.data.get('atr', [1.0] * len(self.data['close']))[self.current_step]
            idle_coeff = self.config.get('idle_penalty_coefficient', 0.001)
            reward -= idle_coeff * atr_value

        # 4. --- Penalize INVALID trade attempts (the robust fix) ---
        if action in [0, 2]:  # If a BUY or SELL was attempted
            if self.balance == prev_balance and len(self.inventory) == prev_inventory_size:
                # If nothing changed, the trade failed. Apply a small penalty.
                reward -= 0.001
        # -------------------------------------------------------------

        # Update tracking
        self.prev_action = action
        self.portfolio_values.append(new_portfolio_value)
        self.current_step += 1

        # Check if episode is done
        if self.current_step >= len(self.data['close']) - 1:
            done = True
            info['final_portfolio_value'] = new_portfolio_value
            info['total_profit'] = new_portfolio_value - self.initial_balance
            info['total_trades'] = self.episode_trades

        return reward, done, info

    def _execute_action(self, action, price):
        """Execute the action and return info (no reward calculation)."""
        if action == 0:  # Buy
            return self._execute_buy(price)
        elif action == 1:  # Hold
            return self._execute_hold(price)
        elif action == 2:  # Sell
            return self._execute_sell(price)
        else:
            return 0.0, {'action': 'unknown'}

    def _execute_buy(self, price):
        """Execute buy action (no reward calculation - handled by step())."""
        info = {'action': 'buy', 'price': price}

        # Check position limits (prevents overtrading)
        if len(self.inventory) >= self.max_positions:
            info['success'] = False
            info['reason'] = 'max_positions_reached'
            logger.debug(f"BUY failed: max positions ({self.max_positions}) reached")
            return 0.0, info

        # Check if we have enough balance
        total_cost = price + self._calculate_transaction_cost(price, is_buy=True)

        # Check if position size exceeds limit
        max_allowed = self.initial_balance * self.max_position_value
        if price > max_allowed:
            info['success'] = False
            info['reason'] = 'position_too_large'
            logger.debug(f"BUY failed: position size {price:.2f} exceeds limit {max_allowed:.2f}")
            return 0.0, info

        if self.balance >= total_cost:
            self.balance -= total_cost
            self.inventory.append(price)
            self.episode_trades += 1

            info['success'] = True
            info['transaction_cost'] = self._calculate_transaction_cost(price, is_buy=True)

            logger.debug(f"BUY at {price:.2f}, balance={self.balance:.2f}, inventory={len(self.inventory)}")
        else:
            info['success'] = False
            info['reason'] = 'insufficient_balance'
            logger.debug(f"BUY failed: insufficient balance")

        return 0.0, info

    def _execute_hold(self, price):
        """Execute hold action (no reward calculation - handled by step())."""
        info = {'action': 'hold', 'price': price}
        # No state changes on hold
        return 0.0, info

    def _execute_sell(self, price):
        """Execute sell action (no reward calculation - handled by step())."""
        info = {'action': 'sell', 'price': price}

        if len(self.inventory) == 0:
            info['success'] = False
            logger.debug(f"SELL failed: no inventory")
            return 0.0, info

        # Sell oldest position (FIFO)
        bought_price = self.inventory.pop(0)

        # Calculate profit
        gross_profit = price - bought_price
        transaction_cost = self._calculate_transaction_cost(price, is_buy=False)
        net_profit = gross_profit - transaction_cost

        self.balance += price - transaction_cost
        self.episode_profit += net_profit
        self.episode_trades += 1
        self.trades.append(net_profit)

        info['success'] = True
        info['bought_price'] = bought_price
        info['gross_profit'] = gross_profit
        info['net_profit'] = net_profit
        info['transaction_cost'] = transaction_cost

        logger.debug(f"SELL at {price:.2f}, bought at {bought_price:.2f}, profit={net_profit:.2f}")

        return 0.0, info

    def _calculate_transaction_cost(self, trade_value, is_buy=True):
        """
        Calculate Zerodha-style transaction costs.

        Args:
            trade_value (float): Trade value
            is_buy (bool): Whether this is a buy transaction

        Returns:
            float: Total transaction cost
        """
        # Brokerage: ₹20 or 0.03% whichever is lower
        brokerage = min(self.brokerage_per_trade, trade_value * self.brokerage_percentage)

        # STT (Securities Transaction Tax) - only on sell
        stt = trade_value * 0.00025 if not is_buy else 0.0

        # Exchange charges
        exchange_charges = trade_value * 0.0000297

        # SEBI charges
        sebi_charges = trade_value * 0.000001

        # Stamp duty - only on buy
        stamp_duty = trade_value * 0.00003 if is_buy else 0.0

        # GST on brokerage + charges
        gst = (brokerage + sebi_charges + exchange_charges) * 0.18

        total_cost = brokerage + stt + exchange_charges + sebi_charges + stamp_duty + gst

        return total_cost

    def _calculate_portfolio_value(self, current_price):
        """Calculate total portfolio value."""
        inventory_value = sum(current_price for _ in self.inventory)
        return self.balance + inventory_value

    def get_portfolio_value(self):
        """Get current portfolio value."""
        if self.current_step < len(self.data['close']):
            current_price = self.data['close'][self.current_step]
            return self._calculate_portfolio_value(current_price)
        return self.portfolio_values[-1]

    def get_state(self):
        """Get current environment state info."""
        return {
            'step': self.current_step,
            'balance': self.balance,
            'inventory_size': len(self.inventory),
            'portfolio_value': self.get_portfolio_value(),
            'episode_profit': self.episode_profit,
            'episode_trades': self.episode_trades
        }

    def is_done(self):
        """Check if episode is finished."""
        return self.current_step >= len(self.data['close']) - 1
