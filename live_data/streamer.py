"""
Real-time data streaming for paper trading.

Uses yfinance for free, real-time (5-minute delayed) stock data.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import logging
from typing import Dict, Optional


class LiveDataStreamer:
    """
    Streams real-time stock data for paper trading.

    Uses yfinance with 5-minute candles. Data is delayed by ~15 minutes
    (acceptable for paper trading validation).
    """

    def __init__(self, ticker: str, interval: str = '5m', buffer_size: int = 500):
        """
        Initialize live data streamer.

        Args:
            ticker: Stock ticker (e.g., 'RELIANCE.NS')
            interval: Data interval ('1m', '5m', '15m', '1h', '1d')
            buffer_size: Number of historical candles to maintain
        """
        self.ticker = ticker
        self.interval = interval
        self.buffer_size = buffer_size

        # Initialize yfinance ticker
        self.yf_ticker = yf.Ticker(ticker)

        # Data buffer
        self.data_buffer = {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': [],
            'timestamp': []
        }

        # State
        self.is_initialized = False
        self.last_update = None

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"LiveDataStreamer initialized: {ticker} @ {interval}")

    def initialize_buffer(self, days: int = 30):
        """
        Initialize buffer with historical data.

        Args:
            days: Number of days of historical data to fetch
        """
        self.logger.info(f"Fetching {days} days of historical data...")

        try:
            # Fetch historical data
            hist = self.yf_ticker.history(period=f'{days}d', interval=self.interval)

            if hist.empty:
                raise ValueError(f"No data returned for {self.ticker}")

            # Populate buffer
            self.data_buffer['open'] = hist['Open'].tolist()
            self.data_buffer['high'] = hist['High'].tolist()
            self.data_buffer['low'] = hist['Low'].tolist()
            self.data_buffer['close'] = hist['Close'].tolist()
            self.data_buffer['volume'] = hist['Volume'].tolist()
            self.data_buffer['timestamp'] = hist.index.tolist()

            # Trim to buffer size
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][-self.buffer_size:]

            self.last_update = datetime.now()
            self.is_initialized = True

            self.logger.info(f"Buffer initialized with {len(self.data_buffer['close'])} candles")
            self.logger.info(f"Date range: {self.data_buffer['timestamp'][0]} to {self.data_buffer['timestamp'][-1]}")

        except Exception as e:
            self.logger.error(f"Failed to initialize buffer: {e}")
            raise

    def update(self) -> bool:
        """
        Fetch latest data and update buffer.

        Returns:
            bool: True if new data received, False otherwise
        """
        if not self.is_initialized:
            self.logger.warning("Buffer not initialized. Call initialize_buffer() first.")
            return False

        try:
            # Fetch recent data (last 1 day)
            hist = self.yf_ticker.history(period='1d', interval=self.interval)

            if hist.empty:
                self.logger.warning("No new data available")
                return False

            # Get latest candle
            latest = hist.iloc[-1]
            latest_timestamp = hist.index[-1]

            # Check if this is actually new data
            if self.data_buffer['timestamp'] and latest_timestamp <= self.data_buffer['timestamp'][-1]:
                # No new candle yet
                return False

            # Append new candle
            self.data_buffer['open'].append(float(latest['Open']))
            self.data_buffer['high'].append(float(latest['High']))
            self.data_buffer['low'].append(float(latest['Low']))
            self.data_buffer['close'].append(float(latest['Close']))
            self.data_buffer['volume'].append(int(latest['Volume']))
            self.data_buffer['timestamp'].append(latest_timestamp)

            # Trim buffer to size
            for key in self.data_buffer:
                self.data_buffer[key] = self.data_buffer[key][-self.buffer_size:]

            self.last_update = datetime.now()

            self.logger.debug(f"New candle: {latest_timestamp} | Close: ₹{latest['Close']:.2f} | Volume: {latest['Volume']}")

            return True

        except Exception as e:
            self.logger.error(f"Failed to update data: {e}")
            return False

    def get_data(self) -> Dict[str, np.ndarray]:
        """
        Get current data buffer as numpy arrays.

        Returns:
            dict: Data arrays (open, high, low, close, volume)
        """
        return {
            'open': np.array(self.data_buffer['open']),
            'high': np.array(self.data_buffer['high']),
            'low': np.array(self.data_buffer['low']),
            'close': np.array(self.data_buffer['close']),
            'volume': np.array(self.data_buffer['volume'])
        }

    def get_latest_price(self) -> Optional[float]:
        """Get most recent close price."""
        if not self.data_buffer['close']:
            return None
        return self.data_buffer['close'][-1]

    def get_latest_timestamp(self) -> Optional[datetime]:
        """Get most recent timestamp."""
        if not self.data_buffer['timestamp']:
            return None
        return self.data_buffer['timestamp'][-1]

    def wait_for_update(self, timeout: int = 600, poll_interval: int = 30) -> bool:
        """
        Wait for new data to arrive.

        Args:
            timeout: Maximum seconds to wait
            poll_interval: Seconds between polling attempts

        Returns:
            bool: True if new data received, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.update():
                return True

            # Wait before next poll
            time.sleep(poll_interval)

        self.logger.warning(f"Timeout waiting for new data ({timeout}s)")
        return False

    def is_market_hours(self) -> bool:
        """
        Check if current time is within Indian market hours (9:15 AM - 3:30 PM IST).

        Note: This is a simplified check. Doesn't account for holidays.
        """
        now = datetime.now()

        # Convert to IST if needed (assuming system is in IST)
        market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
        market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)

        # Check if weekday (Monday=0, Sunday=6)
        is_weekday = now.weekday() < 5

        # Check if within market hours
        is_trading_time = market_open <= now <= market_close

        return is_weekday and is_trading_time

    def get_stats(self) -> Dict:
        """Get streamer statistics."""
        if not self.data_buffer['close']:
            return {'status': 'empty'}

        return {
            'status': 'active' if self.is_initialized else 'not_initialized',
            'ticker': self.ticker,
            'interval': self.interval,
            'buffer_size': len(self.data_buffer['close']),
            'latest_price': self.get_latest_price(),
            'latest_timestamp': self.get_latest_timestamp(),
            'last_update': self.last_update,
            'is_market_hours': self.is_market_hours()
        }


class HistoricalDataSimulator(LiveDataStreamer):
    """
    Simulates live trading using historical data.

    Useful for testing paper trading system with historical data
    before deploying to real-time markets.
    """

    def __init__(self, ticker: str, start_date: str, end_date: str, interval: str = '1d'):
        """
        Initialize historical data simulator.

        Args:
            ticker: Stock ticker
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval
        """
        super().__init__(ticker, interval)

        self.start_date = start_date
        self.end_date = end_date
        self.current_index = 0
        self.simulation_data = None

    def initialize_buffer(self, window_size: int = 20):
        """Initialize with historical data."""
        self.logger.info(f"Loading historical data: {self.start_date} to {self.end_date}")

        try:
            # Fetch all historical data
            hist = self.yf_ticker.history(start=self.start_date, end=self.end_date, interval=self.interval)

            if hist.empty:
                raise ValueError(f"No data for {self.ticker} from {self.start_date} to {self.end_date}")

            # Store full simulation data
            self.simulation_data = {
                'open': hist['Open'].tolist(),
                'high': hist['High'].tolist(),
                'low': hist['Low'].tolist(),
                'close': hist['Close'].tolist(),
                'volume': hist['Volume'].tolist(),
                'timestamp': hist.index.tolist()
            }

            # Initialize buffer with first window_size candles
            for key in self.data_buffer:
                self.data_buffer[key] = self.simulation_data[key][:window_size]

            self.current_index = window_size
            self.is_initialized = True

            self.logger.info(f"Simulation initialized with {len(self.simulation_data['close'])} candles")
            self.logger.info(f"Starting simulation at index {self.current_index}")

        except Exception as e:
            self.logger.error(f"Failed to initialize simulator: {e}")
            raise

    def update(self) -> bool:
        """
        Simulate receiving next candle from historical data.

        Returns:
            bool: True if more data available, False if end reached
        """
        if not self.is_initialized:
            self.logger.warning("Simulator not initialized")
            return False

        if self.current_index >= len(self.simulation_data['close']):
            self.logger.info("End of simulation data reached")
            return False

        # Append next candle
        for key in self.data_buffer:
            self.data_buffer[key].append(self.simulation_data[key][self.current_index])

        # Trim to buffer size
        for key in self.data_buffer:
            self.data_buffer[key] = self.data_buffer[key][-self.buffer_size:]

        self.current_index += 1
        self.last_update = datetime.now()

        return True

    def get_progress(self) -> Dict:
        """Get simulation progress."""
        if not self.simulation_data:
            return {'progress': 0}

        total = len(self.simulation_data['close'])
        return {
            'current_index': self.current_index,
            'total_candles': total,
            'progress_pct': (self.current_index / total) * 100,
            'remaining': total - self.current_index
        }
