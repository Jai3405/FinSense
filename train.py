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


def validate_agent(agent, val_data, config, logger):
    """
    Run a validation loop on the agent in evaluation mode.

    Args:
        agent: The agent to validate.
        val_data (dict): The validation dataset.
        config (dict): The configuration object.
        logger: The logger instance.

    Returns:
        tuple: (validation_profit, validation_trades)
    """
    logger.info("--- Running Validation ---")
    
    # Set agent to evaluation mode
    agent.q_network.eval()
    
    env_config = config.get_section('environment')
    window_size = env_config.get('window_size', 10)
    
    val_env = TradingEnvironment(val_data, env_config)
    val_env.reset()
    
    done = False
    while not done:
        state = get_state_with_features(
            val_data, val_env.current_step, window_size, env_config
        )
        action = agent.act(state, training=False)  # Use greedy policy
        _, done, _ = val_env.step(action)

    # Set agent back to training mode
    agent.q_network.train()

    final_portfolio_value = val_env.get_portfolio_value()
    val_profit = final_portfolio_value - val_env.initial_balance
    val_trades = val_env.episode_trades
    
    logger.info("--- Validation Complete ---")
    return val_profit, val_trades


def train():
    """Main training function."""
    args = parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    if args.episodes:
        config.update('training.episodes', args.episodes)

    log_level = 'DEBUG' if args.verbose else config.get('logging.level', 'INFO')
    logger = setup_logger(
        'finsense',
        config.get('logging.file', 'logs/training.log'),
        getattr(__import__('logging'), log_level)
    )

    logger.info("="*60)
    logger.info("FinSense Training Started")
    logger.info("="*60)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    tensorboard_dir = Path(args.tensorboard) / f"{config.get('data.ticker', 'FNSN')}_{timestamp}"
    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    logger.info(f"TensorBoard logs: {tensorboard_dir}")

    try:
        data_loader = DataLoader(config.get_section('data'))
        data = data_loader.load_data()
        
        train_data, val_data, _ = data_loader.train_test_split(
            data,
            train_ratio=config.get('training.train_ratio', 0.7),
            validation_ratio=config.get('training.validation_ratio', 0.15)
        )

        logger.info(f"Train: {len(train_data['close'])} points, Val: {len(val_data['close'])} points")

        env = TradingEnvironment(train_data, config.get_section('environment'))
        
        window_size = config.get('environment.window_size')
        state_size = window_size - 1
        if config.get('environment.use_volume', True): state_size += 1
        if config.get('environment.use_technical_indicators', True):
            state_size += 9  # RSI(1) + MACD(3) + BB(1) + ATR(1) + Trend(3)
        logger.info(f"State size: {state_size} features")

        agent = DQNAgent(state_size, 3, config.get_section('agent'))
        checkpoint_manager = CheckpointManager(args.output, config.get_section('checkpoints'))

        # Create metrics calculator
        metrics_calc = TradingMetrics(
            risk_free_rate=config.get('evaluation.risk_free_rate', 0.02)
        )

        episodes = config.get('training.episodes')
        logger.info(f"Starting training for {episodes} episodes...")

        best_val_profit = float('-inf')
        VALIDATION_INTERVAL = 5
        MIN_VAL_TRADES = 4      # Minimum trades to be considered a valid policy

        for episode in range(episodes):
            env.reset()
            episode_loss, loss_count = 0.0, 0
            
            for t in range(window_size, len(train_data['close'])):
                state = get_state_with_features(train_data, t, window_size, config.get_section('environment'))
                action_mask = env.get_action_mask()
                action = agent.act(state, training=True, action_mask=action_mask)
                reward, done, _ = env.step(action)
                next_state = get_state_with_features(train_data, env.current_step, window_size, config.get_section('environment')) if not done else state
                agent.remember(state, action, reward, next_state, done)

                if agent.can_replay():
                    loss = agent.replay()
                    if loss is not None:
                        episode_loss += loss
                        loss_count += 1
                if done:
                    break
            
            train_info = env.get_state()
            train_profit = train_info['portfolio_value'] - env.initial_balance
            avg_loss = episode_loss / loss_count if loss_count > 0 else 0.0
            
            logger.info(f"Episode {episode+1}/{episodes} [TRAIN] | Profit: ₹{train_profit:.2f} | Trades: {train_info['episode_trades']} | Epsilon: {agent.epsilon:.3f} | Loss: {avg_loss:.6f}")
            writer.add_scalar('Profit/train', train_profit, episode)
            writer.add_scalar('Trades/train', train_info['episode_trades'], episode)
            writer.add_scalar('Loss/episode', avg_loss, episode)

            if (episode + 1) % VALIDATION_INTERVAL == 0:
                val_profit, val_trades = validate_agent(agent, val_data, config, logger)
                logger.info(f"Episode {episode+1}/{episodes} [VALIDATION] | Profit: ₹{val_profit:.2f} | Trades: {val_trades}")
                writer.add_scalar('Profit/validation', val_profit, episode)
                writer.add_scalar('Trades/validation', val_trades, episode)

                if val_trades >= MIN_VAL_TRADES and val_profit > best_val_profit:
                    best_val_profit = val_profit
                    checkpoint_manager.save_checkpoint(
                        agent,
                        episode + 1,
                        {
                            "val_profit": val_profit,
                            "val_trades": val_trades
                        }
                    )
                    logger.info(
                        f"✅ New best model saved | Val Profit: ₹{val_profit:.2f} | Trades: {val_trades}"
                    )

            # Calculate training metrics for logging and periodic checkpoints
            if len(env.trades) > 0:
                episode_metrics = metrics_calc.calculate_all_metrics(
                    env.portfolio_values,
                    env.trades
                )
            else:
                episode_metrics = {'total_profit': train_profit, 'sharpe_ratio': 0.0}

            # Save periodic checkpoint (will only save if freq matches config)
            checkpoint_manager.save_checkpoint(agent, episode + 1, episode_metrics)
            
            # Decay epsilon
            agent.decay_epsilon()

        logger.info("="*60)
        logger.info("Training Complete!")
        agent.save(Path(args.output) / 'final_model.pt')
        writer.close()
        return 0

    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(train())
