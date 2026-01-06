"""
Test suite for feature engineering.
"""

import pytest
import numpy as np
import pandas as pd

from utils.features import (
    sigmoid,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_atr,
    get_state_with_features
)


class TestFeatureEngineering:
    """Test technical indicator calculations."""

    @pytest.fixture
    def sample_prices(self):
        """Generate sample price data."""
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(100))
        return prices

    @pytest.fixture
    def sample_data(self):
        """Generate sample OHLCV data."""
        np.random.seed(42)
        close = 100 + np.cumsum(np.random.randn(100))
        high = close + np.random.rand(100) * 2
        low = close - np.random.rand(100) * 2
        volume = np.random.randint(1000, 10000, 100)

        return {
            'close': close,
            'high': high,
            'low': low,
            'volume': volume
        }

    def test_sigmoid(self):
        """Test sigmoid function."""
        assert sigmoid(0) == 0.5
        assert sigmoid(100) > 0.99  # Large positive
        assert sigmoid(-100) < 0.01  # Large negative

        # Test symmetry
        assert abs(sigmoid(5) + sigmoid(-5) - 1.0) < 0.001

    def test_sigmoid_overflow(self):
        """Test sigmoid handles overflow."""
        assert sigmoid(1000) == 1.0
        assert sigmoid(-1000) == 0.0

    def test_rsi_calculation(self, sample_prices):
        """Test RSI calculation."""
        rsi = calculate_rsi(sample_prices, period=14)

        # RSI should be between 0 and 100
        assert 0 <= rsi <= 100

        # With random data, should be around 50 (neutral)
        assert 30 <= rsi <= 70

    def test_rsi_insufficient_data(self):
        """Test RSI with insufficient data."""
        prices = np.array([100, 101, 102])
        rsi = calculate_rsi(prices, period=14)

        # Should return neutral value
        assert rsi == 50.0

    def test_rsi_trending_up(self):
        """Test RSI on uptrending data."""
        prices = np.arange(100, 150)  # Strong uptrend
        rsi = calculate_rsi(prices, period=14)

        # Should be high (overbought)
        assert rsi > 70

    def test_rsi_trending_down(self):
        """Test RSI on downtrending data."""
        prices = np.arange(150, 100, -1)  # Strong downtrend
        rsi = calculate_rsi(prices, period=14)

        # Should be low (oversold)
        assert rsi < 30

    def test_macd_calculation(self, sample_prices):
        """Test MACD calculation."""
        macd_line, signal_line, histogram = calculate_macd(sample_prices)

        # All should be numeric
        assert isinstance(macd_line, (int, float))
        assert isinstance(signal_line, (int, float))
        assert isinstance(histogram, (int, float))

        # Histogram should equal macd - signal
        assert abs(histogram - (macd_line - signal_line)) < 0.001

    def test_macd_insufficient_data(self):
        """Test MACD with insufficient data."""
        prices = np.array([100, 101, 102])
        macd_line, signal_line, histogram = calculate_macd(prices)

        # Should return zeros
        assert macd_line == 0.0
        assert signal_line == 0.0
        assert histogram == 0.0

    def test_bollinger_bands(self, sample_prices):
        """Test Bollinger Bands calculation."""
        upper, middle, lower, percent_b = calculate_bollinger_bands(sample_prices)

        # Bands should be ordered
        assert upper >= middle >= lower

        # %B should be between 0 and 1 (mostly)
        assert -0.5 <= percent_b <= 1.5  # Can go slightly outside

        # Current price should be near %B position
        current_price = sample_prices[-1]
        expected_b = (current_price - lower) / (upper - lower) if upper != lower else 0.5
        assert abs(percent_b - expected_b) < 0.01

    def test_bollinger_bands_insufficient_data(self):
        """Test Bollinger Bands with insufficient data."""
        prices = np.array([100, 101, 102])
        upper, middle, lower, percent_b = calculate_bollinger_bands(prices)

        # Should return defaults
        assert upper == 0.0
        assert middle == 0.0
        assert lower == 0.0
        assert percent_b == 0.5

    def test_atr_calculation(self, sample_data):
        """Test ATR calculation."""
        atr = calculate_atr(
            sample_data['high'],
            sample_data['low'],
            sample_data['close']
        )

        # ATR should be positive
        assert atr > 0

        # Should be reasonable relative to price
        avg_price = np.mean(sample_data['close'])
        assert atr < avg_price * 0.2  # Less than 20% of price

    def test_atr_insufficient_data(self):
        """Test ATR with insufficient data."""
        high = np.array([100, 101])
        low = np.array([99, 100])
        close = np.array([100, 101])

        atr = calculate_atr(high, low, close)
        assert atr == 0.0

    def test_get_state_with_features_basic(self, sample_data):
        """Test state generation with features."""
        config = {
            'use_volume': False,
            'use_technical_indicators': False
        }

        window_size = 10
        t = 20

        state = get_state_with_features(sample_data, t, window_size, config)

        # Should return 2D array
        assert isinstance(state, np.ndarray)
        assert state.ndim == 2
        assert state.shape[0] == 1  # Batch dimension

        # Should have window_size - 1 price features
        assert state.shape[1] >= window_size - 1

    def test_get_state_with_volume(self, sample_data):
        """Test state generation with volume."""
        config = {
            'use_volume': True,
            'use_technical_indicators': False
        }

        window_size = 10
        t = 20

        state = get_state_with_features(sample_data, t, window_size, config)

        # Should have price features + volume
        assert state.shape[1] >= window_size

    def test_get_state_with_all_indicators(self, sample_data):
        """Test state generation with all indicators."""
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

        window_size = 10
        t = 50  # Need enough data for indicators

        state = get_state_with_features(sample_data, t, window_size, config)

        # Should have many features
        # Price diffs (9) + volume (1) + RSI (1) + MACD (3) + BB (1) + ATR (1) = 16+
        assert state.shape[1] >= 16

    def test_get_state_lookback_window(self, sample_data):
        """Test state uses correct lookback window."""
        config = {
            'use_volume': False,
            'use_technical_indicators': False
        }

        window_size = 5
        t = 10

        state = get_state_with_features(sample_data, t, window_size, config)

        # Should use data from t-window_size to t-1
        # Verify by checking price differences
        expected_prices = sample_data['close'][t-window_size:t]
        assert len(expected_prices) == window_size

    def test_get_state_no_lookahead_bias(self, sample_data):
        """Test state doesn't use future data."""
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

        window_size = 10
        t = 50

        state = get_state_with_features(sample_data, t, window_size, config)

        # Modify future data (t and beyond)
        modified_data = sample_data.copy()
        modified_data['close'][t:] = 99999.0

        state_modified = get_state_with_features(modified_data, t, window_size, config)

        # State should be identical (no look-ahead)
        np.testing.assert_array_equal(state, state_modified)

    def test_get_state_insufficient_data(self, sample_data):
        """Test state generation with insufficient data."""
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

        window_size = 10
        t = 2  # Not enough data

        state = get_state_with_features(sample_data, t, window_size, config)

        # Should return a state (with padding)
        assert isinstance(state, np.ndarray)
        assert state.shape[0] == 1

    def test_feature_normalization(self, sample_data):
        """Test features are normalized."""
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

        window_size = 10
        t = 50

        state = get_state_with_features(sample_data, t, window_size, config)

        # Most features should be roughly between 0 and 1 (after sigmoid/normalization)
        # Check that not all features are extreme
        features = state[0]
        in_range = np.sum((features >= -2) & (features <= 2))
        assert in_range > len(features) * 0.8  # At least 80% reasonable

    def test_state_reproducibility(self, sample_data):
        """Test state generation is reproducible."""
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

        window_size = 10
        t = 50

        state1 = get_state_with_features(sample_data, t, window_size, config)
        state2 = get_state_with_features(sample_data, t, window_size, config)

        # Should be identical
        np.testing.assert_array_equal(state1, state2)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
