"""
Advanced feature engineering for trading states.
Implements technical indicators: RSI, MACD, Bollinger Bands, ATR, etc.
"""

import numpy as np
import pandas as pd
import math
import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


def sigmoid(x):
    """Sigmoid activation function."""
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI).

    Args:
        prices (np.ndarray or pd.Series): Price data
        period (int): RSI period

    Returns:
        float: RSI value (0-100)
    """
    if len(prices) < period + 1:
        return 50.0  # Neutral RSI

    prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    # Calculate RS and RSI
    rs = gain / loss.replace(0, 1e-10)  # Avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50.0


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        prices (np.ndarray or pd.Series): Price data
        fast (int): Fast EMA period
        slow (int): Slow EMA period
        signal (int): Signal line period

    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    if len(prices) < slow:
        return 0.0, 0.0, 0.0

    prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices

    # Calculate EMAs
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return (
        macd_line.iloc[-1],
        signal_line.iloc[-1],
        histogram.iloc[-1]
    )


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Args:
        prices (np.ndarray or pd.Series): Price data
        period (int): Moving average period
        std_dev (float): Number of standard deviations

    Returns:
        tuple: (Upper band, Middle band, Lower band, %B position)
    """
    if len(prices) < period:
        return 0.0, 0.0, 0.0, 0.5

    prices = pd.Series(prices) if isinstance(prices, np.ndarray) else prices

    # Middle band (SMA)
    middle_band = prices.rolling(window=period).mean()

    # Standard deviation
    std = prices.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    # %B indicator (position within bands)
    current_price = prices.iloc[-1]
    upper = upper_band.iloc[-1]
    lower = lower_band.iloc[-1]
    middle = middle_band.iloc[-1]

    if upper == lower:
        percent_b = 0.5
    else:
        percent_b = (current_price - lower) / (upper - lower)

    return upper, middle, lower, percent_b


def calculate_atr(high, low, close, period=14):
    """
    Calculate Average True Range (ATR) - volatility indicator.

    Args:
        high (np.ndarray or pd.Series): High prices
        low (np.ndarray or pd.Series): Low prices
        close (np.ndarray or pd.Series): Close prices
        period (int): ATR period

    Returns:
        float: ATR value
    """
    if len(close) < period + 1:
        return 0.0

    high = pd.Series(high) if isinstance(high, np.ndarray) else high
    low = pd.Series(low) if isinstance(low, np.ndarray) else low
    close = pd.Series(close) if isinstance(close, np.ndarray) else close

    # True Range components
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())

    # True Range
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # ATR
    atr = tr.rolling(window=period).mean()

    return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0


def get_state_with_features(data, t, window_size, config=None):
    """
    Generate state representation with advanced technical features.

    Fixes look-ahead bias by only using data up to t-1 (not including t).

    Args:
        data (dict): Dictionary with 'close', 'high', 'low', 'volume' keys
        t (int): Current timestep
        window_size (int): Number of historical days
        config (dict): Configuration for features

    Returns:
        np.ndarray: State vector with shape (1, feature_dim)
    """
    if config is None:
        config = {
            'use_volume': True,
            'use_technical_indicators': True,
            'indicators': ['rsi', 'macd', 'bollinger_bands', 'atr']
        }

    # Extract price data (IMPORTANT: up to t-1, not t)
    start_idx = max(0, t - window_size)
    close_prices = data['close'][start_idx:t]  # Up to but not including t

    # Ensure we have enough data
    if len(close_prices) < 2:
        logger.warning(f"Insufficient data at t={t}, returning zero state")
        return np.zeros((1, window_size + 10))  # Approximate feature size

    # Pad if needed
    if len(close_prices) < window_size:
        padding = [close_prices[0]] * (window_size - len(close_prices))
        close_prices = np.array(padding + list(close_prices))
    else:
        close_prices = np.array(close_prices)

    features = []

    # 1. Price differences (sigmoid normalized) - CORE FEATURES
    for i in range(len(close_prices) - 1):
        diff = close_prices[i + 1] - close_prices[i]
        features.append(sigmoid(diff))

    # 2. Volume features (if enabled)
    if config.get('use_volume', False) and 'volume' in data:
        volume = data['volume'][start_idx:t]
        if len(volume) >= 2:
            # Volume change ratio
            vol_change = volume[-1] / (volume[-2] + 1e-10)
            features.append(sigmoid(vol_change - 1))
        else:
            features.append(0.5)

    # 3. Technical indicators (if enabled)
    if config.get('use_technical_indicators', False):
        indicators = config.get('indicators', [])

        # RSI
        if 'rsi' in indicators:
            rsi = calculate_rsi(close_prices, period=14)
            features.append(rsi / 100.0)  # Normalize to 0-1

        # MACD
        if 'macd' in indicators:
            macd_line, signal_line, histogram = calculate_macd(close_prices)
            # Normalize MACD values
            price_range = close_prices[-1]
            features.append(sigmoid(macd_line / (price_range + 1e-10)))
            features.append(sigmoid(signal_line / (price_range + 1e-10)))
            features.append(sigmoid(histogram / (price_range + 1e-10)))

        # Bollinger Bands
        if 'bollinger_bands' in indicators:
            upper, middle, lower, percent_b = calculate_bollinger_bands(close_prices)
            features.append(percent_b)  # Already 0-1 normalized

        # ATR (volatility)
        if 'atr' in indicators and all(k in data for k in ['high', 'low']):
            high = data['high'][start_idx:t]
            low = data['low'][start_idx:t]
            if len(high) >= 15 and len(low) >= 15:
                atr = calculate_atr(high, low, close_prices, period=14)
                # Normalize ATR by current price
                features.append(sigmoid(atr / (close_prices[-1] + 1e-10)))
            else:
                features.append(0.5)

    # --- Trend Features (EDGE INJECTION) ---
    full_close = pd.Series(data['close'][:t])
    
    if len(full_close) > 26:  # Ensure enough data for slow EMA
        ema_fast = full_close.ewm(span=12, adjust=False).mean()
        ema_slow = full_close.ewm(span=26, adjust=False).mean()
        
        # 1. EMA Difference (normalized by price)
        ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / (full_close.iloc[-1] + 1e-8)
        
        # 2. EMA Slope (normalized by price)
        ema_slope = ema_fast.diff().iloc[-1] / (full_close.iloc[-1] + 1e-8) if len(ema_fast) > 1 else 0.0
        
        # 3. Trend Strength (ADX-lite)
        # This requires high/low data, so we get it from the main `data` dict
        full_high = pd.Series(data['high'][:t])
        full_low = pd.Series(data['low'][:t])
        true_range = np.maximum(full_high - full_low, np.maximum(abs(full_high - full_close.shift(1)), abs(full_low - full_close.shift(1))))
        atr = true_range.rolling(14).mean().iloc[-1]
        trend_strength = abs(full_close.diff()).rolling(14).mean().iloc[-1] / (atr + 1e-8)
        
        features.extend([ema_diff, ema_slope, trend_strength])
    else:
        # Not enough data for trend indicators, append neutral values
        features.extend([0.0, 0.0, 0.5]) # Use 0.5 for trend_strength as it's sigmoid-like

    state_array = np.array([features])
    # NaN safety (mandatory)
    state_array = np.nan_to_num(state_array, nan=0.0, posinf=0.0, neginf=0.0)

    return state_array


def get_state(data, t, n):
    """
    DEPRECATED: Old state generation function (kept for backward compatibility).
    Use get_state_with_features() instead.

    Returns an n-day state representation ending at time t.
    """
    logger.warning("get_state() is deprecated. Use get_state_with_features() instead.")

    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else [-d * [data[0]]] + data[0:t + 1]
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])
