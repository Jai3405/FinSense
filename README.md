# FinSense

A reinforcement learning-based algorithmic trading system for Indian stock markets using PPO (Proximal Policy Optimization).

## Features

- **Multi-Stock Training**: Train on multiple NSE stocks simultaneously for better generalization
- **PPO Agent**: State-of-the-art policy gradient algorithm with action masking
- **Percentage-Based Rewards**: Properly scaled reward function preventing dead policy
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, trend features
- **Comprehensive Evaluation**: Sharpe ratio, Sortino ratio, max drawdown, win rate metrics
- **Production Ready**: Built-in transaction costs, position limits, risk management

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Jai3405/FinSense.git
cd FinSense

# Create virtual environment
python3 -m venv finsense_env
source finsense_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Training

Train PPO agent on 5 NSE stocks:

```bash
# Standard training (200 episodes)
python train_ppo.py --episodes 200 --verbose

# Background training with nohup
nohup python train_ppo.py --episodes 200 --verbose > training.log 2>&1 &

# Monitor progress
tail -f training.log
```

### Evaluation

Evaluate trained model on test set:

```bash
# Comprehensive evaluation with all metrics
python comprehensive_ppo_eval.py

# Quick evaluation
python evaluate_ppo.py
```

## Project Structure

```
FinSense/
├── agents/             # PPO and DQN agent implementations
├── data_loader/        # Data loading and preprocessing
├── environment/        # Trading environment with proper reward scaling
├── models/             # Trained model checkpoints
├── utils/              # Feature engineering, metrics, logging
├── docs/               # Development documentation
├── scripts/            # Utility and archived scripts
├── config.yaml         # Main configuration file
└── train_ppo.py        # Training script
```

## Configuration

Edit `config.yaml` to customize:
- Stock tickers (currently: RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK)
- Training episodes and hyperparameters
- PPO parameters (learning rate, GAE lambda, clip epsilon)
- Reward coefficients and penalties
- Position limits and transaction costs

## Current Status

**Branch**: `experimental/reward-tuning`

- ✅ Fixed dead policy bug (percentage-based rewards)
- ✅ Multi-stock data split (per-stock 70/15/15 split)
- ✅ 200-episode training completed
- 🔄 Investigating training results

## Development

See [docs/development/](docs/development/) for detailed documentation on:
- Reward function design and fixes
- Multi-stock implementation
- Training guides and best practices
- Algorithm comparisons (DQN vs PPO)

## License

MIT License

## Contributing

This is a personal project. Feel free to fork and adapt for your own use.
