"""
Unified training script for FinSense.
Consolidates all training logic with modern best practices.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import DQNAgent
from utils import (
    load_config,
    setup_logger,
    get_state_with_features,
    get_reward_function,
    TradingMetrics
)
from utils.checkpoint import CheckpointManager
from data_loader import DataLoader
from environment import TradingEnvironment


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train FinSense DQN Agent')

    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--ticker', type=str, default=None,
                       help='Stock ticker to train on')
    parser.add_argument('--episodes', type=int, default=None,
                       help='Number of training episodes')
    parser.add_argument('--output', type=str, default='models',
                       help='Output directory for models')
    parser.add_argument('--tensorboard', type=str, default='runs',
                       help='TensorBoard log directory')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint (path or "auto" for latest)')
    parser.add_argument('--auto-resume', action='store_true',
                       help='Automatically resume from latest checkpoint if available')

    return parser.parse_args()


def train():
    """Main training function."""
    # Parse arguments
    args = parse_args()

    # Load configuration
    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Override config with command line args
    if args.ticker:
        config.update('data.ticker', args.ticker)
    if args.episodes:
        config.update('training.episodes', args.episodes)

    # Setup logging
    log_level = 'DEBUG' if args.verbose else config.get('logging.level', 'INFO')
    logger = setup_logger(
        'finsense',
        config.get('logging.file', 'logs/training.log'),
        getattr(__import__('logging'), log_level)
    )

    logger.info("="*60)
    logger.info("FinSense Training Started")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Ticker: {config.get('data.ticker')}")
    logger.info(f"Episodes: {config.get('training.episodes')}")

    # Setup TensorBoard
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = Path(args.tensorboard) / f"{config.get('data.ticker')}_{timestamp}"
    writer = SummaryWriter(tensorboard_dir)
    logger.info(f"TensorBoard logs: {tensorboard_dir}")

    try:
        # Load data
        logger.info("Loading data...")
        data_loader = DataLoader(config.get_section('data'))

        # Determine what data to load
        data_config = config.get_section('data')

        if data_config.get('multi_stock', False):
            # Multi-stock training
            stock_list = data_config.get('stock_list', ['RELIANCE.NS'])
            logger.info(f"Multi-stock training mode: {len(stock_list)} stocks")
            data = data_loader.load_data(ticker=stock_list)
        elif data_config.get('use_market_index', False):
            # Load with market index
            ticker = args.ticker or data_config.get('ticker', 'RELIANCE.NS')
            market_index = data_config.get('market_index', 'NIFTY50')
            logger.info(f"Loading {ticker} with {market_index} index")
            data = data_loader.load_with_market_index(ticker, index=market_index)
        else:
            # Standard single-stock loading
            ticker = args.ticker or data_config.get('ticker', 'RELIANCE.NS')
            data = data_loader.load_data(ticker=ticker)

        # Data augmentation (if enabled)
        if data_config.get('augment_data', False):
            noise = data_config.get('augmentation_noise', 0.01)
            copies = data_config.get('augmentation_copies', 3)
            logger.info(f"Augmenting data: noise={noise}, copies={copies}")
            data = data_loader.augment_data(data, noise_level=noise, n_augmented=copies)

        # Split data
        train_data, val_data, test_data = data_loader.train_test_split(
            data,
            train_ratio=config.get('training.train_ratio', 0.7),
            validation_ratio=config.get('training.validation_ratio', 0.15)
        )

        logger.info(f"Train: {len(train_data['close'])} points")
        logger.info(f"Val: {len(val_data['close'])} points")
        logger.info(f"Test: {len(test_data['close'])} points")

        # Get data stats
        stats = data_loader.get_data_stats(train_data)
        logger.info(f"Price range: ₹{stats['min_price']:.2f} - ₹{stats['max_price']:.2f}")
        logger.info(f"Total return: {stats['total_return']:.2f}%")

        # Create environment
        env = TradingEnvironment(train_data, config.get_section('environment'))

        # Calculate state size
        window_size = config.get('environment.window_size')
        use_volume = config.get('environment.use_volume', True)
        use_indicators = config.get('environment.use_technical_indicators', True)

        # Estimate state size
        state_size = window_size - 1  # Price diffs
        if use_volume:
            state_size += 1
        if use_indicators:
            state_size += 6  # RSI(1) + MACD(3) + BB(1) + ATR(1)

        logger.info(f"State size: {state_size} features")

        # Create agent
        logger.info("Creating DQN agent...")
        agent = DQNAgent(
            state_size=state_size,
            action_size=3,
            config=config.get_section('agent')
        )

        # Create checkpoint manager
        checkpoint_manager = CheckpointManager(
            checkpoint_dir=args.output,
            config=config.get_section('checkpoints')
        )

        # Handle resume logic
        start_episode = 0
        resume_checkpoint = None

        if args.auto_resume or args.resume == 'auto':
            # Auto-resume from latest checkpoint
            resume_checkpoint, start_episode = checkpoint_manager.get_latest_checkpoint()
            if resume_checkpoint:
                logger.info(f"Auto-resuming from episode {start_episode}: {resume_checkpoint}")
                agent.load(resume_checkpoint)
            else:
                logger.info("No checkpoint found, starting from scratch")

        elif args.resume:
            # Resume from specific checkpoint
            logger.info(f"Resuming from {args.resume}")
            agent.load(args.resume)
            # Try to extract episode number from filename
            import re
            match = re.search(r'ep(\d+)', str(args.resume))
            if match:
                start_episode = int(match.group(1))
                logger.info(f"Resuming from episode {start_episode}")

        # Create reward function
        reward_func = get_reward_function(
            config.get('reward.type', 'profit_with_risk'),
            config.get_section('reward')
        )

        # Create metrics calculator
        metrics_calc = TradingMetrics(
            risk_free_rate=config.get('evaluation.risk_free_rate', 0.02)
        )

        # Training loop
        episodes = config.get('training.episodes')
        if start_episode > 0:
            logger.info(f"Resuming training from episode {start_episode+1}/{episodes}")
        else:
            logger.info(f"Starting training for {episodes} episodes...")

        best_profit = float('-inf')

        for episode in range(start_episode, episodes):
            # Reset environment and reward function
            env.reset()
            if hasattr(reward_func, 'reset'):
                reward_func.reset()

            episode_reward = 0.0
            episode_loss = 0.0
            loss_count = 0

            # Episode loop
            for t in range(window_size, len(train_data['close'])):
                # Get state
                state = get_state_with_features(
                    train_data, t, window_size,
                    config.get_section('environment')
                )

                # Agent selects action
                action = agent.act(state, training=True)

                # Execute action in environment
                reward, done, info = env.step(action)

                # Calculate risk-adjusted reward
                if hasattr(reward_func, 'calculate'):
                    portfolio_value = env.get_portfolio_value()
                    prev_portfolio_value = env.portfolio_values[-2] if len(env.portfolio_values) > 1 else env.initial_balance

                    reward = reward_func.calculate(
                        profit=reward,
                        portfolio_value=portfolio_value,
                        prev_portfolio_value=prev_portfolio_value,
                        holding_losing_position=(len(env.inventory) > 0 and env.inventory[0] > train_data['close'][t]),
                        transaction_cost=info.get('transaction_cost', 0.0)
                    )

                episode_reward += reward

                # Get next state
                if not done:
                    next_state = get_state_with_features(
                        train_data, t+1, window_size,
                        config.get_section('environment')
                    )
                else:
                    next_state = state

                # Remember experience
                agent.remember(state, action, reward, next_state, done)

                # Train agent
                if agent.can_replay():
                    loss = agent.replay()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1

                if done:
                    break

            # Episode finished
            episode_info = env.get_state()
            final_value = episode_info['portfolio_value']
            profit = final_value - env.initial_balance
            num_trades = episode_info['episode_trades']

            # Calculate metrics
            if len(env.trades) > 0:
                episode_metrics = metrics_calc.calculate_all_metrics(
                    env.portfolio_values,
                    env.trades
                )
            else:
                episode_metrics = {'total_profit': profit, 'sharpe_ratio': 0.0}

            # Logging
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0

            logger.info(
                f"Episode {episode+1}/{episodes} | "
                f"Profit: ₹{profit:.2f} | "
                f"Trades: {num_trades} | "
                f"Sharpe: {episode_metrics.get('sharpe_ratio', 0):.3f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Loss: {avg_loss:.6f}"
            )

            # TensorBoard logging
            writer.add_scalar('Profit/episode', profit, episode)
            writer.add_scalar('Portfolio_Value/episode', final_value, episode)
            writer.add_scalar('Trades/episode', num_trades, episode)
            writer.add_scalar('Loss/episode', avg_loss, episode)
            writer.add_scalar('Epsilon/episode', agent.epsilon, episode)
            writer.add_scalar('Sharpe_Ratio/episode', episode_metrics.get('sharpe_ratio', 0), episode)
            writer.add_scalar('Max_Drawdown/episode', episode_metrics.get('max_drawdown', 0), episode)

            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                agent, episode+1, episode_metrics
            )

            # Track best profit
            if profit > best_profit:
                best_profit = profit

            # Decay epsilon after each episode
            agent.decay_epsilon()

        # Training complete
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Best profit: ₹{best_profit:.2f}")
        logger.info(f"Final epsilon: {agent.epsilon:.3f}")

        # Save final model
        final_path = Path(args.output) / 'final_model.pt'
        agent.save(final_path)
        logger.info(f"Final model saved to {final_path}")

        # Close TensorBoard writer
        writer.close()

        logger.info(f"TensorBoard logs saved to {tensorboard_dir}")
        logger.info("View with: tensorboard --logdir runs/")

        return 0

    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(train())
