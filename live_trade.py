"""
Live trading script for FinSense.

Uses real-time data streaming to make trading decisions.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
import time
import logging

from agents import DQNAgent
from data_loader import LiveDataAdapter
from utils import (
    load_config,
    get_state_with_features,
    setup_logger
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Live Trading with FinSense DQN Agent')

    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        required=True,
        help='Stock ticker symbol (e.g., RELIANCE.NS)'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='5m',
        choices=['1m', '5m', '15m'],
        help='Trading interval (1m, 5m, 15m)'
    )

    parser.add_argument(
        '--paper-trade',
        action='store_true',
        help='Paper trading mode (no real orders)'
    )

    parser.add_argument(
        '--starting-balance',
        type=float,
        default=50000,
        help='Starting balance for paper trading'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--max-trades',
        type=int,
        default=None,
        help='Maximum number of trades (for testing)'
    )

    return parser.parse_args()


class LiveTrader:
    """Live trading engine."""

    def __init__(self, agent, data_adapter, config, starting_balance=50000):
        """
        Initialize live trader.

        Args:
            agent: Trained DQN agent
            data_adapter: LiveDataAdapter instance
            config: Configuration object
            starting_balance (float): Starting capital
        """
        self.agent = agent
        self.data_adapter = data_adapter
        self.config = config
        self.window_size = config.get('environment.window_size', 10)

        # Portfolio state
        self.balance = starting_balance
        self.inventory = 0
        self.trades = []

        # Statistics
        self.total_trades = 0
        self.winning_trades = 0
        self.total_profit = 0.0

        self.logger = logging.getLogger(__name__)

    def get_portfolio_value(self, current_price):
        """Calculate current portfolio value."""
        return self.balance + (self.inventory * current_price)

    def execute_action(self, action, current_price):
        """
        Execute trading action.

        Args:
            action (int): 0=Buy, 1=Hold, 2=Sell
            current_price (float): Current stock price

        Returns:
            dict: Trade info
        """
        trade_info = {
            'timestamp': datetime.now(),
            'action': ['BUY', 'HOLD', 'SELL'][action],
            'price': current_price,
            'portfolio_value': self.get_portfolio_value(current_price)
        }

        if action == 0:  # BUY
            if self.balance >= current_price:
                # Buy 1 share (can be modified for position sizing)
                self.inventory += 1
                self.balance -= current_price
                self.total_trades += 1

                trade_info['shares'] = 1
                trade_info['cost'] = current_price

                self.logger.info(f"BUY: 1 share @ ₹{current_price:.2f} | "
                                f"Balance: ₹{self.balance:.2f} | "
                                f"Inventory: {self.inventory}")
            else:
                self.logger.warning(f"Insufficient balance for BUY: ₹{self.balance:.2f}")
                trade_info['action'] = 'HOLD'  # Can't buy

        elif action == 2:  # SELL
            if self.inventory > 0:
                # Sell 1 share
                self.inventory -= 1
                self.balance += current_price
                self.total_trades += 1

                trade_info['shares'] = 1
                trade_info['revenue'] = current_price

                # Calculate profit for this trade
                if len(self.trades) > 0:
                    # Find corresponding buy
                    buy_trades = [t for t in self.trades if t['action'] == 'BUY']
                    if buy_trades:
                        last_buy = buy_trades[-1]
                        profit = current_price - last_buy['cost']
                        trade_info['profit'] = profit
                        self.total_profit += profit

                        if profit > 0:
                            self.winning_trades += 1

                self.logger.info(f"SELL: 1 share @ ₹{current_price:.2f} | "
                                f"Balance: ₹{self.balance:.2f} | "
                                f"Inventory: {self.inventory}")
            else:
                self.logger.warning("Cannot SELL: No inventory")
                trade_info['action'] = 'HOLD'

        else:  # HOLD
            self.logger.debug(f"HOLD | Price: ₹{current_price:.2f} | "
                             f"Portfolio: ₹{self.get_portfolio_value(current_price):.2f}")

        self.trades.append(trade_info)
        return trade_info

    def get_statistics(self):
        """Get trading statistics."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'win_rate': self.winning_trades / max(self.total_trades, 1),
            'total_profit': self.total_profit,
            'current_balance': self.balance,
            'current_inventory': self.inventory
        }


def run_live_trading(args):
    """Main live trading function."""
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger = setup_logger('live_trade', level=log_level)

    logger.info("="*60)
    logger.info(" FinSense Live Trading")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Interval: {args.interval}")
    logger.info(f"Mode: {'PAPER TRADING' if args.paper_trade else 'LIVE TRADING'}")

    if not args.paper_trade:
        logger.error("Real trading not yet implemented. Use --paper-trade flag.")
        return 1

    try:
        # Load configuration
        config = load_config(args.config)

        # Load trained model
        logger.info(f"Loading model from {args.model}...")
        model_path = Path(args.model)

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Calculate state size (same as training)
        window_size = config.get('environment.window_size', 10)
        use_volume = config.get('environment.use_volume', True)
        use_indicators = config.get('environment.use_technical_indicators', True)

        state_size = window_size - 1
        if use_volume:
            state_size += 1
        if use_indicators:
            state_size += 6

        # Create agent
        agent = DQNAgent(
            state_size=state_size,
            action_size=3,
            config=config.get_section('agent')
        )
        agent.load(model_path)

        logger.info(f"Model loaded (state_size={state_size})")

        # Initialize live data stream
        logger.info(f"Connecting to real-time data for {args.ticker}...")
        data_adapter = LiveDataAdapter(
            args.ticker,
            interval=args.interval,
            buffer_size=500
        )

        logger.info("Live data stream connected")

        # Initialize trader
        trader = LiveTrader(
            agent,
            data_adapter,
            config,
            starting_balance=args.starting_balance
        )

        logger.info(f"Starting balance: ₹{args.starting_balance:.2f}")
        logger.info("\nLive trading started. Press Ctrl+C to stop.\n")

        # Main trading loop
        trade_count = 0

        while True:
            # Wait for next candle update
            logger.debug("Waiting for next update...")
            if not data_adapter.wait_for_update(timeout=300):
                logger.warning("No update received in 5 minutes")
                continue

            # Get current data
            data = data_adapter.get_data()

            if len(data['close']) < window_size:
                logger.debug(f"Insufficient data: {len(data['close'])} < {window_size}")
                continue

            # Get current state for agent
            t = len(data['close']) - 1
            state = get_state_with_features(
                data, t, window_size,
                config.get_section('environment')
            )

            # Agent makes decision (no exploration in live trading)
            action = agent.act(state, training=False)

            # Get current price
            current_price = data['close'][-1]

            # Execute action
            trade_info = trader.execute_action(action, current_price)

            # Log status
            if trade_info['action'] != 'HOLD':
                stats = trader.get_statistics()
                logger.info(f"Stats: Trades={stats['total_trades']}, "
                           f"Win Rate={stats['win_rate']*100:.1f}%, "
                           f"Profit=₹{stats['total_profit']:.2f}")

            trade_count += 1

            # Check if max trades reached (for testing)
            if args.max_trades and trade_count >= args.max_trades:
                logger.info(f"Reached max trades ({args.max_trades})")
                break

    except KeyboardInterrupt:
        logger.info("\nStopping live trading (Ctrl+C)...")

    except Exception as e:
        logger.error(f"Live trading failed: {e}", exc_info=True)
        return 1

    finally:
        # Cleanup
        if 'data_adapter' in locals():
            data_adapter.close()

        # Final statistics
        if 'trader' in locals():
            stats = trader.get_statistics()
            logger.info("\n" + "="*60)
            logger.info(" Final Statistics")
            logger.info("="*60)
            logger.info(f"Total Trades: {stats['total_trades']}")
            logger.info(f"Winning Trades: {stats['winning_trades']}")
            logger.info(f"Win Rate: {stats['win_rate']*100:.2f}%")
            logger.info(f"Total Profit: ₹{stats['total_profit']:.2f}")
            logger.info(f"Final Balance: ₹{stats['current_balance']:.2f}")
            logger.info(f"Final Inventory: {stats['current_inventory']} shares")

            current_price = data['close'][-1] if 'data' in locals() else 0
            final_value = trader.get_portfolio_value(current_price)
            logger.info(f"Final Portfolio Value: ₹{final_value:.2f}")
            logger.info(f"Return: {((final_value - args.starting_balance) / args.starting_balance * 100):.2f}%")
            logger.info("="*60)

    return 0


def main():
    """Entry point."""
    args = parse_args()
    sys.exit(run_live_trading(args))


if __name__ == '__main__':
    main()
