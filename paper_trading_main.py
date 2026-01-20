"""
Paper Trading System - Main Entry Point

Runs paper trading using trained PPO model with real-time or simulated data.
"""

import argparse
import sys
import logging
from pathlib import Path
from datetime import datetime
import time
import yaml

from live_data.streamer import LiveDataStreamer, HistoricalDataSimulator
from live_trading.ppo_inference import PPOInference
from live_trading.paper_executor import PaperTradingExecutor
from monitoring.dashboard import PaperTradingMonitor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/paper_trading/paper_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Paper Trading System for FinSense PPO Agent'
    )

    parser.add_argument(
        '--model',
        type=str,
        default='models/ppo_final.pt',
        help='Path to trained PPO model'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )

    parser.add_argument(
        '--ticker',
        type=str,
        default='RELIANCE.NS',
        help='Stock ticker to trade'
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['live', 'simulate'],
        default='simulate',
        help='Trading mode: live (real-time) or simulate (historical backtest)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for simulation (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for simulation (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--interval',
        type=str,
        default='1d',
        choices=['1m', '5m', '15m', '1h', '1d'],
        help='Data interval'
    )

    parser.add_argument(
        '--balance',
        type=float,
        default=50000,
        help='Starting balance'
    )

    parser.add_argument(
        '--max-steps',
        type=int,
        help='Maximum steps (for testing)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    parser.add_argument(
        '--backtest-sharpe',
        type=float,
        default=0.2972,
        help='Backtest Sharpe ratio for comparison'
    )

    return parser.parse_args()


def run_paper_trading(args):
    """Main paper trading function."""
    logger = setup_logging(args.verbose)

    logger.info("="*80)
    logger.info(" FINSENSE PAPER TRADING SYSTEM")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Ticker: {args.ticker}")
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Starting Balance: ₹{args.balance:,.2f}")

    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config(args.config)

        # Initialize PPO inference engine
        logger.info("Loading PPO model...")
        ppo_agent = PPOInference(args.model, config)
        model_info = ppo_agent.get_model_info()
        logger.info(f"Model loaded: {model_info['parameters']:,} parameters")

        # Initialize data streamer
        logger.info(f"Initializing data streamer ({args.mode} mode)...")

        if args.mode == 'live':
            # Real-time live trading
            streamer = LiveDataStreamer(
                ticker=args.ticker,
                interval=args.interval,
                buffer_size=500
            )
            streamer.initialize_buffer(days=30)
            is_simulation = False

        else:
            # Historical simulation
            if not args.start_date or not args.end_date:
                logger.error("--start-date and --end-date required for simulate mode")
                return 1

            streamer = HistoricalDataSimulator(
                ticker=args.ticker,
                start_date=args.start_date,
                end_date=args.end_date,
                interval=args.interval
            )
            streamer.initialize_buffer(window_size=20)
            is_simulation = True

            progress = streamer.get_progress()
            logger.info(f"Simulation: {progress['total_candles']} candles loaded")

        # Initialize paper trading executor
        logger.info("Initializing paper trading executor...")
        executor = PaperTradingExecutor(
            starting_balance=args.balance,
            max_positions=config.get('environment', {}).get('max_positions', 40),
            max_position_value=config.get('environment', {}).get('max_position_value', 0.95),
            transaction_costs=True
        )

        # Initialize monitoring
        logger.info("Initializing performance monitor...")
        monitor = PaperTradingMonitor()

        # Backtest metrics for comparison
        backtest_metrics = {
            'sharpe_ratio': args.backtest_sharpe,
            'total_return_pct': 4.51,
            'max_drawdown': 11.57,
            'win_rate': 74.0,
            'profit_factor': 7.80
        }

        logger.info("\nPaper trading started. Press Ctrl+C to stop.\n")

        # Main trading loop
        step = 0

        while True:
            # Get new data
            if is_simulation:
                # Simulation: step through historical data
                if not streamer.update():
                    logger.info("End of simulation data reached")
                    break
            else:
                # Live: wait for new candle
                logger.debug("Waiting for new data...")
                if not streamer.wait_for_update(timeout=600, poll_interval=30):
                    logger.warning("Timeout waiting for data")
                    continue

            # Get current data
            data = streamer.get_data()
            current_price = streamer.get_latest_price()
            timestamp = streamer.get_latest_timestamp()

            if current_price is None:
                logger.warning("No price data available")
                continue

            # Get action mask (what actions are valid)
            action_mask = executor.get_action_mask(current_price)

            # Agent predicts action
            action, action_probs, value = ppo_agent.predict(data, action_mask=action_mask)
            action_name = ppo_agent.get_action_name(action)

            # Execute action
            trade_result = executor.execute_action(action, current_price, timestamp)

            # Update equity curve
            executor.update_equity_curve(timestamp, current_price)

            # Log trade
            if trade_result['action'] in ['BUY', 'SELL']:
                logger.info(
                    f"Step {step} | {timestamp} | "
                    f"Action: {action_name} | Price: ₹{current_price:.2f} | "
                    f"Probs: [BUY:{action_probs[0]:.2f} HOLD:{action_probs[1]:.2f} SELL:{action_probs[2]:.2f}]"
                )

            # Periodic logging (every 10 steps or on trade)
            if step % 10 == 0 or trade_result['action'] in ['BUY', 'SELL']:
                metrics = executor.get_metrics(current_price)
                logger.info(
                    f"Portfolio: ₹{metrics['portfolio_value']:,.2f} | "
                    f"Return: {metrics['total_return_pct']:+.2f}% | "
                    f"DD: {metrics['max_drawdown']:.2f}% | "
                    f"Sharpe: {metrics['sharpe_ratio']:.4f} | "
                    f"Trades: {metrics['completed_trades']} ({metrics['win_rate']:.1f}% WR)"
                )

                # Log to monitor
                monitor.log_metrics(timestamp, metrics)

            step += 1

            # Check max steps (for testing)
            if args.max_steps and step >= args.max_steps:
                logger.info(f"Reached max steps ({args.max_steps})")
                break

            # Simulation: no sleep needed
            # Live: sleep handled by wait_for_update()

    except KeyboardInterrupt:
        logger.info("\nStopping paper trading (Ctrl+C pressed)...")

    except Exception as e:
        logger.error(f"Paper trading failed: {e}", exc_info=True)
        return 1

    finally:
        # Generate final report
        if 'executor' in locals() and 'monitor' in locals():
            logger.info("\n" + "="*80)
            logger.info(" GENERATING FINAL REPORT")
            logger.info("="*80)

            # Get final metrics
            final_metrics = executor.get_metrics(current_price)

            # Print summary
            summary = executor.get_summary(current_price)
            print(summary)

            # Generate daily report
            report = monitor.generate_daily_report(
                executor,
                current_price,
                backtest_metrics=backtest_metrics
            )
            print(report)

            # Save report
            monitor.save_daily_report(report)

            # Plot equity curve
            equity_df = executor.get_equity_curve_df()
            if not equity_df.empty:
                monitor.plot_equity_curve(equity_df)

            # Plot trade analysis
            trades_df = executor.get_trade_history()
            if not trades_df.empty:
                monitor.plot_trade_analysis(trades_df)

            # Save trade history
            trades_csv_path = Path('logs/paper_trading') / f'trades_{datetime.now().strftime("%Y%m%d")}.csv'
            executor.save_trades(str(trades_csv_path))

            logger.info("\nAll reports and charts saved to logs/paper_trading/")

            # Final verdict
            final_sharpe = final_metrics['sharpe_ratio']
            final_dd = final_metrics['max_drawdown']

            logger.info("\n" + "="*80)
            logger.info(" PAPER TRADING VERDICT")
            logger.info("="*80)

            if final_sharpe > 0.25 and final_dd < 20:
                logger.info("✅ SUCCESS: Paper trading performance meets targets!")
                logger.info("   Ready for real money deployment (start with ₹10,000)")
            elif final_sharpe > 0.15:
                logger.info("⚠️  MARGINAL: Performance is acceptable but below target")
                logger.info("   Consider extending paper trading period")
            else:
                logger.info("❌ FAILED: Performance below minimum threshold")
                logger.info("   Review strategy before real money deployment")

            logger.info("="*80)

    return 0


def main():
    """Entry point."""
    args = parse_args()
    sys.exit(run_paper_trading(args))


if __name__ == '__main__':
    main()
