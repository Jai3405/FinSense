"""
Real-time data streaming for live trading.

Supports:
- Yahoo Finance live data
- WebSocket connections (future: Groww API)
- Data buffering and streaming
- Live price feeds
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
import time
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
import asyncio


logger = logging.getLogger(__name__)


class RealTimeDataStream:
    """
    Real-time data streaming for live trading.

    Features:
    - Live price feeds from Yahoo Finance
    - Asynchronous data updates
    - Ring buffer for historical context
    - Thread-safe operations
    - Automatic reconnection
    """

    def __init__(self, ticker, interval='1m', buffer_size=500, config=None):
        """
        Initialize real-time data stream.

        Args:
            ticker (str): Stock ticker symbol
            interval (str): Update interval ('1m', '5m', '15m')
            buffer_size (int): Number of historical candles to keep
            config (dict): Optional configuration
        """
        self.ticker = ticker
        self.interval = interval
        self.buffer_size = buffer_size
        self.config = config or {}

        # Data buffers (ring buffers)
        self.close_buffer = []
        self.high_buffer = []
        self.low_buffer = []
        self.open_buffer = []
        self.volume_buffer = []
        self.timestamp_buffer = []

        # Thread management
        self.running = False
        self.update_thread = None
        self.data_queue = Queue()

        # Callbacks
        self.on_update_callbacks = []

        # Statistics
        self.last_update = None
        self.update_count = 0
        self.error_count = 0

        logger.info(f"RealTimeDataStream initialized: {ticker}, interval={interval}, buffer={buffer_size}")

    def start(self):
        """Start the real-time data stream."""
        if self.running:
            logger.warning("Stream already running")
            return

        self.running = True
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

        logger.info(f"Started real-time stream for {self.ticker}")

    def stop(self):
        """Stop the real-time data stream."""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=5)

        logger.info(f"Stopped real-time stream for {self.ticker}")

    def _update_loop(self):
        """Main update loop (runs in separate thread)."""
        interval_seconds = self._interval_to_seconds(self.interval)

        # Initial load of historical data
        self._load_initial_data()

        while self.running:
            try:
                # Fetch latest data
                new_data = self._fetch_latest_candle()

                if new_data:
                    # Add to buffers
                    self._add_to_buffer(new_data)

                    # Trigger callbacks
                    self._trigger_callbacks(new_data)

                    self.last_update = datetime.now()
                    self.update_count += 1

                # Wait for next interval
                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                self.error_count += 1
                time.sleep(5)  # Wait before retry

    def _load_initial_data(self):
        """Load initial historical data to populate buffer."""
        try:
            logger.info(f"Loading initial data for {self.ticker}...")

            # Download recent history
            stock = yf.Ticker(self.ticker)

            # Determine period based on buffer size and interval
            if self.interval == '1m':
                period = '7d'  # 1-minute data, last 7 days
            elif self.interval == '5m':
                period = '60d'
            elif self.interval == '15m':
                period = '60d'
            else:
                period = '60d'

            df = stock.history(period=period, interval=self.interval)

            if df.empty:
                raise ValueError(f"No data returned for {self.ticker}")

            # Take last buffer_size candles
            df = df.tail(self.buffer_size)

            # Initialize buffers
            self.close_buffer = df['Close'].tolist()
            self.high_buffer = df['High'].tolist()
            self.low_buffer = df['Low'].tolist()
            self.open_buffer = df['Open'].tolist()
            self.volume_buffer = df['Volume'].tolist()
            self.timestamp_buffer = df.index.tolist()

            logger.info(f"Loaded {len(self.close_buffer)} initial candles")

        except Exception as e:
            logger.error(f"Failed to load initial data: {e}")
            raise

    def _fetch_latest_candle(self):
        """
        Fetch the latest price candle.

        Returns:
            dict: Latest OHLCV data or None
        """
        try:
            stock = yf.Ticker(self.ticker)

            # Get very recent data (last few candles)
            df = stock.history(period='1d', interval=self.interval)

            if df.empty:
                return None

            # Get the latest candle
            latest = df.iloc[-1]

            candle = {
                'close': float(latest['Close']),
                'high': float(latest['High']),
                'low': float(latest['Low']),
                'open': float(latest['Open']),
                'volume': float(latest['Volume']),
                'timestamp': latest.name
            }

            # Check if this is actually a new candle
            if self.timestamp_buffer and candle['timestamp'] <= self.timestamp_buffer[-1]:
                return None  # Not a new candle

            return candle

        except Exception as e:
            logger.debug(f"Error fetching latest candle: {e}")
            return None

    def _add_to_buffer(self, candle):
        """
        Add new candle to ring buffer.

        Args:
            candle (dict): New OHLCV candle
        """
        # Add to buffers
        self.close_buffer.append(candle['close'])
        self.high_buffer.append(candle['high'])
        self.low_buffer.append(candle['low'])
        self.open_buffer.append(candle['open'])
        self.volume_buffer.append(candle['volume'])
        self.timestamp_buffer.append(candle['timestamp'])

        # Trim to buffer size (ring buffer)
        if len(self.close_buffer) > self.buffer_size:
            self.close_buffer.pop(0)
            self.high_buffer.pop(0)
            self.low_buffer.pop(0)
            self.open_buffer.pop(0)
            self.volume_buffer.pop(0)
            self.timestamp_buffer.pop(0)

    def get_current_data(self):
        """
        Get current buffered data.

        Returns:
            dict: Current OHLCV data in standard format
        """
        return {
            'close': np.array(self.close_buffer),
            'high': np.array(self.high_buffer),
            'low': np.array(self.low_buffer),
            'open': np.array(self.open_buffer),
            'volume': np.array(self.volume_buffer),
            'dates': self.timestamp_buffer.copy()
        }

    def get_latest_price(self):
        """
        Get the most recent price.

        Returns:
            float: Latest close price
        """
        if not self.close_buffer:
            return None
        return self.close_buffer[-1]

    def get_latest_candle(self):
        """
        Get the most recent complete candle.

        Returns:
            dict: Latest OHLCV candle
        """
        if not self.close_buffer:
            return None

        return {
            'close': self.close_buffer[-1],
            'high': self.high_buffer[-1],
            'low': self.low_buffer[-1],
            'open': self.open_buffer[-1],
            'volume': self.volume_buffer[-1],
            'timestamp': self.timestamp_buffer[-1]
        }

    def register_callback(self, callback: Callable):
        """
        Register callback for new data updates.

        Args:
            callback (callable): Function to call on new data
                                 Signature: callback(candle_dict)
        """
        self.on_update_callbacks.append(callback)
        logger.info(f"Registered callback: {callback.__name__}")

    def _trigger_callbacks(self, candle):
        """Trigger all registered callbacks."""
        for callback in self.on_update_callbacks:
            try:
                callback(candle)
            except Exception as e:
                logger.error(f"Error in callback {callback.__name__}: {e}")

    @staticmethod
    def _interval_to_seconds(interval):
        """Convert interval string to seconds."""
        if interval == '1m':
            return 60
        elif interval == '5m':
            return 300
        elif interval == '15m':
            return 900
        elif interval == '1h':
            return 3600
        else:
            return 300  # Default to 5 minutes

    def get_status(self):
        """
        Get stream status.

        Returns:
            dict: Status information
        """
        return {
            'ticker': self.ticker,
            'running': self.running,
            'buffer_size': len(self.close_buffer),
            'last_update': self.last_update,
            'update_count': self.update_count,
            'error_count': self.error_count,
            'latest_price': self.get_latest_price()
        }


class MultiTickerStream:
    """
    Manage multiple real-time data streams.

    Use for monitoring multiple stocks simultaneously.
    """

    def __init__(self, tickers, interval='1m', buffer_size=500):
        """
        Initialize multi-ticker stream.

        Args:
            tickers (list): List of ticker symbols
            interval (str): Update interval
            buffer_size (int): Buffer size for each ticker
        """
        self.tickers = tickers
        self.interval = interval
        self.buffer_size = buffer_size

        # Create stream for each ticker
        self.streams = {}
        for ticker in tickers:
            self.streams[ticker] = RealTimeDataStream(
                ticker, interval, buffer_size
            )

        logger.info(f"MultiTickerStream initialized for {len(tickers)} tickers")

    def start_all(self):
        """Start all streams."""
        for ticker, stream in self.streams.items():
            stream.start()
            time.sleep(0.1)  # Stagger starts to avoid rate limits

        logger.info(f"Started {len(self.streams)} streams")

    def stop_all(self):
        """Stop all streams."""
        for stream in self.streams.values():
            stream.stop()

        logger.info("Stopped all streams")

    def get_data(self, ticker):
        """
        Get data for specific ticker.

        Args:
            ticker (str): Ticker symbol

        Returns:
            dict: Current data for ticker
        """
        if ticker not in self.streams:
            raise ValueError(f"Ticker {ticker} not in stream")

        return self.streams[ticker].get_current_data()

    def get_all_latest_prices(self):
        """
        Get latest price for all tickers.

        Returns:
            dict: {ticker: price}
        """
        return {
            ticker: stream.get_latest_price()
            for ticker, stream in self.streams.items()
        }

    def get_status_all(self):
        """Get status for all streams."""
        return {
            ticker: stream.get_status()
            for ticker, stream in self.streams.items()
        }


class LiveDataAdapter:
    """
    Adapter to make real-time data compatible with backtesting interface.

    Allows seamless switching between historical and live data.
    """

    def __init__(self, ticker, interval='1m', buffer_size=500):
        """
        Initialize live data adapter.

        Args:
            ticker (str): Stock ticker
            interval (str): Update interval
            buffer_size (int): Historical buffer size
        """
        self.stream = RealTimeDataStream(ticker, interval, buffer_size)
        self.stream.start()

        # Wait for initial data
        time.sleep(5)

        logger.info(f"LiveDataAdapter ready for {ticker}")

    def get_data(self):
        """
        Get data in standard format (compatible with backtesting).

        Returns:
            dict: OHLCV data
        """
        return self.stream.get_current_data()

    def get_latest_state(self, window_size=10):
        """
        Get latest state for agent (last window_size candles).

        Args:
            window_size (int): Number of candles

        Returns:
            dict: Recent data window
        """
        data = self.stream.get_current_data()

        # Take last window_size candles
        return {
            'close': data['close'][-window_size:],
            'high': data['high'][-window_size:],
            'low': data['low'][-window_size:],
            'open': data['open'][-window_size:],
            'volume': data['volume'][-window_size:]
        }

    def wait_for_update(self, timeout=300):
        """
        Wait for next data update.

        Args:
            timeout (int): Max wait time in seconds

        Returns:
            bool: True if update received, False if timeout
        """
        start_time = time.time()
        last_count = self.stream.update_count

        while time.time() - start_time < timeout:
            if self.stream.update_count > last_count:
                return True
            time.sleep(0.5)

        return False

    def close(self):
        """Close the live data stream."""
        self.stream.stop()


# Example usage and testing functions

def example_basic_stream():
    """Example: Basic real-time streaming."""
    print("Starting real-time stream for RELIANCE.NS...")

    stream = RealTimeDataStream('RELIANCE.NS', interval='1m', buffer_size=100)

    # Register callback
    def on_new_data(candle):
        print(f"New candle: {candle['timestamp']} | Close: ₹{candle['close']:.2f}")

    stream.register_callback(on_new_data)

    # Start stream
    stream.start()

    # Monitor for 5 minutes
    try:
        for i in range(5):
            time.sleep(60)
            status = stream.get_status()
            print(f"Status: {status}")
    finally:
        stream.stop()


def example_multi_ticker():
    """Example: Multiple tickers streaming."""
    print("Starting multi-ticker stream...")

    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
    multi_stream = MultiTickerStream(tickers, interval='5m')

    multi_stream.start_all()

    try:
        for i in range(3):
            time.sleep(60)
            prices = multi_stream.get_all_latest_prices()
            print(f"\nLatest prices: {prices}")
    finally:
        multi_stream.stop_all()


def example_live_trading_adapter():
    """Example: Live trading with adapter."""
    print("Starting live trading adapter...")

    adapter = LiveDataAdapter('RELIANCE.NS', interval='1m', buffer_size=200)

    try:
        # Get current data (compatible with backtesting format)
        data = adapter.get_data()
        print(f"Buffer size: {len(data['close'])} candles")
        print(f"Latest price: ₹{data['close'][-1]:.2f}")

        # Get state for agent (last 10 candles)
        state_data = adapter.get_latest_state(window_size=10)
        print(f"State window: {len(state_data['close'])} candles")

        # Wait for update
        print("Waiting for next update...")
        if adapter.wait_for_update(timeout=120):
            print("Update received!")
            new_price = adapter.stream.get_latest_price()
            print(f"New price: ₹{new_price:.2f}")

    finally:
        adapter.close()


if __name__ == '__main__':
    # Run examples
    print("="*60)
    print("Real-Time Data Streaming Examples")
    print("="*60)

    # Uncomment to run examples:
    # example_basic_stream()
    # example_multi_ticker()
    # example_live_trading_adapter()

    print("\nImport this module to use real-time streaming in your trading bot!")
