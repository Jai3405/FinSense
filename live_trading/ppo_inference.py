"""
PPO Inference Engine for Paper Trading.

Loads trained PPO model and makes predictions on live data.
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.ppo_agent import PPOAgent
from utils.features import (
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    sigmoid
)


class PPOInference:
    """
    PPO agent inference engine for paper trading.

    Loads trained model and makes predictions on live data
    with proper feature engineering.
    """

    def __init__(self, model_path: str, config: Dict):
        """
        Initialize PPO inference engine.

        Args:
            model_path: Path to trained PPO model (.pt file)
            config: Configuration dictionary with environment settings
        """
        self.model_path = model_path
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Environment settings
        self.window_size = config.get('environment', {}).get('window_size', 20)
        self.use_volume = config.get('environment', {}).get('use_volume', True)
        self.use_technical_indicators = config.get('environment', {}).get('use_technical_indicators', True)

        # Calculate state size (must match training)
        self.state_size = self._calculate_state_size()

        # Initialize PPO agent
        self.agent = self._load_model()

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.agent.to(self.device)
        self.agent.eval()  # Set to evaluation mode

        self.logger.info(f"PPO Inference Engine initialized")
        self.logger.info(f"Model: {model_path}")
        self.logger.info(f"State size: {self.state_size}")
        self.logger.info(f"Device: {self.device}")

    def _calculate_state_size(self) -> int:
        """Calculate state size based on features."""
        state_size = self.window_size - 1  # Price differences

        if self.use_volume:
            state_size += 1  # Volume ratio

        if self.use_technical_indicators:
            # RSI(1) + MACD(3) + BB(1) + ATR(1) + Trend(3) = 9 features
            state_size += 9

        return state_size

    def _load_model(self) -> PPOAgent:
        """Load trained PPO model from checkpoint."""
        try:
            model_path = Path(self.model_path)

            if not model_path.exists():
                raise FileNotFoundError(f"Model not found: {model_path}")

            # Create PPO agent architecture
            agent = PPOAgent(
                state_size=self.state_size,
                action_size=3,  # BUY, HOLD, SELL
                hidden_size=self.config.get('agent', {}).get('hidden_size', 128)
            )

            # Load model weights
            checkpoint = torch.load(model_path, map_location='cpu')

            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    agent.load_state_dict(checkpoint['model_state_dict'])
                elif 'policy_state_dict' in checkpoint:
                    agent.load_state_dict(checkpoint['policy_state_dict'])
                else:
                    agent.load_state_dict(checkpoint)
            else:
                agent.load_state_dict(checkpoint)

            self.logger.info("Model loaded successfully")
            return agent

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise

    def create_state(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Create state vector from raw market data.

        This MUST match the exact state creation from training
        (environment/trading_env.py _get_observation).

        Args:
            data: Dictionary with 'open', 'high', 'low', 'close', 'volume'

        Returns:
            np.ndarray: State vector for agent (shape: [state_size])
        """
        prices = data['close']
        t = len(prices) - 1  # Current timestep

        if t < self.window_size:
            raise ValueError(f"Insufficient data: need {self.window_size}, got {t+1}")

        state = []

        # 1. Price differences (normalized percentage changes)
        window_start = max(0, t - self.window_size + 1)
        price_window = prices[window_start:t+1]

        for i in range(1, len(price_window)):
            price_diff = (price_window[i] - price_window[i-1]) / price_window[i-1]
            state.append(price_diff)

        # 2. Volume ratio (if enabled)
        if self.use_volume:
            volumes = data['volume']
            window_volumes = volumes[window_start:t+1]
            avg_volume = np.mean(window_volumes[:-1]) if len(window_volumes) > 1 else 1.0
            current_volume = volumes[t]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
            state.append(sigmoid(np.log(volume_ratio)))

        # 3. Technical indicators (if enabled)
        if self.use_technical_indicators:
            # Need sufficient data for indicators
            indicator_window = max(50, self.window_size)
            indicator_start = max(0, t - indicator_window + 1)
            indicator_prices = prices[indicator_start:t+1]

            # RSI (normalized to 0-1)
            rsi = calculate_rsi(indicator_prices, period=14)
            state.append(rsi / 100.0)

            # MACD (3 values, normalized)
            macd_line, signal_line, histogram = calculate_macd(
                indicator_prices, fast=12, slow=26, signal=9
            )
            current_price = prices[t]
            state.append(sigmoid(macd_line / current_price))
            state.append(sigmoid(signal_line / current_price))
            state.append(sigmoid(histogram / current_price))

            # Bollinger Bands (position within bands)
            _, _, _, percent_b = calculate_bollinger_bands(
                indicator_prices, period=20, std_dev=2
            )
            state.append(percent_b)

            # ATR (volatility, normalized)
            atr = calculate_atr(
                data['high'][indicator_start:t+1],
                data['low'][indicator_start:t+1],
                indicator_prices,
                period=14
            )
            state.append(sigmoid(atr / current_price))

            # Trend indicators (3 values: short, medium, long-term)
            for ma_period in [5, 20, 50]:
                if len(indicator_prices) >= ma_period:
                    ma = np.mean(indicator_prices[-ma_period:])
                    trend = (current_price - ma) / ma
                    state.append(sigmoid(trend))
                else:
                    state.append(0.5)  # Neutral

        return np.array(state, dtype=np.float32)

    def predict(
        self,
        data: Dict[str, np.ndarray],
        action_mask: Optional[np.ndarray] = None
    ) -> Tuple[int, np.ndarray, float]:
        """
        Get action prediction from PPO agent.

        Args:
            data: Market data dictionary
            action_mask: Boolean mask for valid actions [buy_allowed, hold_allowed, sell_allowed]

        Returns:
            tuple: (action, action_probs, value)
                - action: 0=BUY, 1=HOLD, 2=SELL
                - action_probs: Probabilities for each action
                - value: State value estimate
        """
        try:
            # Create state
            state = self.create_state(data)

            # Convert to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            # Get action from agent (no gradient needed for inference)
            with torch.no_grad():
                action, log_prob, entropy, value = self.agent.act(
                    state_tensor,
                    action_mask=action_mask
                )

            # Get action probabilities for logging/monitoring
            with torch.no_grad():
                logits, _ = self.agent.forward(state_tensor)

                if action_mask is not None:
                    mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
                    logits = logits.masked_fill(~mask, -1e9)

                action_probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

            return action, action_probs, value.item()

        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            raise

    def get_action_name(self, action: int) -> str:
        """Convert action index to name."""
        return ['BUY', 'HOLD', 'SELL'][action]

    def validate_state_size(self, data: Dict[str, np.ndarray]) -> bool:
        """
        Validate that created state matches expected size.

        Useful for debugging state creation issues.
        """
        try:
            state = self.create_state(data)
            expected_size = self.state_size
            actual_size = len(state)

            if actual_size != expected_size:
                self.logger.error(
                    f"State size mismatch! Expected: {expected_size}, Got: {actual_size}"
                )
                return False

            self.logger.debug(f"State size validated: {actual_size}")
            return True

        except Exception as e:
            self.logger.error(f"State validation failed: {e}")
            return False

    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            'model_path': self.model_path,
            'state_size': self.state_size,
            'action_size': 3,
            'window_size': self.window_size,
            'use_volume': self.use_volume,
            'use_technical_indicators': self.use_technical_indicators,
            'device': str(self.device),
            'parameters': sum(p.numel() for p in self.agent.parameters())
        }
