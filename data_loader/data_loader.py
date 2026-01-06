"""
Data loading and preprocessing pipeline for FinSense.
Handles multiple data sources: yfinance, CSV, Groww API.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class DataLoader:
    """
    Unified data loader for financial data.

    Supports multiple sources:
    - Yahoo Finance (yfinance)
    - CSV files
    - Groww API (future)
    """

    def __init__(self, config=None):
        """
        Initialize data loader.

        Args:
            config (dict): Data configuration
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config():
        """Default data configuration."""
        return {
            'source': 'yfinance',
            'ticker': 'RELIANCE.NS',
            'interval': '1d',
            'start_date': '2020-01-01',
            'end_date': None,
            'normalize': True,
            'fill_missing': 'forward',
            'remove_outliers': False
        }

    def load_data(self, ticker=None, start_date=None, end_date=None, interval=None):
        """
        Load financial data for single or multiple tickers.

        Args:
            ticker (str or list): Stock ticker symbol(s)
            start_date (str): Start date (YYYY-MM-DD)
            end_date (str): End date (YYYY-MM-DD) or None for today
            interval (str): Data interval (1d, 5m, 15m, 1h)

        Returns:
            dict: Data dictionary with close, high, low, volume, open keys
                  For multiple tickers, returns concatenated data

        Raises:
            ValueError: If data loading fails
        """
        ticker = ticker or self.config['ticker']
        interval = interval or self.config['interval']
        start_date = start_date or self.config['start_date']
        end_date = end_date or self.config['end_date']

        # Handle multiple tickers
        if isinstance(ticker, list):
            return self.load_multiple_tickers(ticker, start_date, end_date, interval)

        source = self.config['source']

        logger.info(f"Loading {ticker} data from {source}, interval: {interval}")

        if source == 'yfinance':
            data = self._load_from_yfinance(ticker, start_date, end_date, interval)
        elif source == 'csv':
            data = self._load_from_csv(ticker)
        else:
            raise ValueError(f"Unsupported data source: {source}")

        # Preprocessing
        data = self._preprocess(data)

        # Validation
        data = self._validate_data(data, ticker)

        logger.info(f"Loaded {len(data['close'])} data points for {ticker}")

        return data

    def _load_from_yfinance(self, ticker, start_date, end_date, interval):
        """
        Load data from Yahoo Finance.

        Args:
            ticker (str): Stock ticker
            start_date (str): Start date
            end_date (str): End date or None
            interval (str): Data interval

        Returns:
            dict: OHLCV data

        Raises:
            ValueError: If download fails
        """
        try:
            stock = yf.Ticker(ticker)

            # Determine period or dates
            if end_date is None:
                # Use period instead of dates for intraday
                if interval in ['5m', '15m', '1h']:
                    period = '60d'  # Last 60 days for intraday
                    df = stock.history(period=period, interval=interval)
                else:
                    df = stock.history(start=start_date, interval=interval)
            else:
                df = stock.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Convert to our format
            data = {
                'close': df['Close'].values,
                'high': df['High'].values,
                'low': df['Low'].values,
                'volume': df['Volume'].values,
                'open': df['Open'].values,
                'dates': df.index.tolist()
            }

            return data

        except Exception as e:
            logger.error(f"yfinance download failed for ticker '{ticker}'. Original error: {type(e).__name__}: {e}")
            raise ValueError(f"yfinance download failed for ticker '{ticker}'. See logs for details.")

    def _load_from_csv(self, filepath):
        """
        Load data from CSV file.

        Args:
            filepath (str): Path to CSV file

        Returns:
            dict: OHLCV data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV format is invalid
        """
        filepath = Path(filepath)

        if not filepath.exists():
            # Try in data/ directory
            filepath = Path('data') / filepath.name
            if not filepath.exists():
                raise FileNotFoundError(f"CSV file not found: {filepath}")

        try:
            df = pd.read_csv(filepath)

            # Detect column names (flexible)
            close_col = self._find_column(df, ['Close', 'close', 'CLOSE'])
            high_col = self._find_column(df, ['High', 'high', 'HIGH'])
            low_col = self._find_column(df, ['Low', 'low', 'LOW'])
            volume_col = self._find_column(df, ['Volume', 'volume', 'VOLUME'])
            open_col = self._find_column(df, ['Open', 'open', 'OPEN'])

            data = {
                'close': df[close_col].values if close_col else None,
                'high': df[high_col].values if high_col else None,
                'low': df[low_col].values if low_col else None,
                'volume': df[volume_col].values if volume_col else None,
                'open': df[open_col].values if open_col else None,
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            if 'close' not in data:
                raise ValueError("CSV must have Close price column")

            return data

        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise ValueError(f"CSV loading failed: {e}")

    @staticmethod
    def _find_column(df, possible_names):
        """Find column by multiple possible names."""
        for name in possible_names:
            if name in df.columns:
                return name
        return None

    def _preprocess(self, data):
        """
        Preprocess data: handle missing values, outliers, normalization.

        Args:
            data (dict): Raw data

        Returns:
            dict: Preprocessed data
        """
        # Handle missing values
        if self.config['fill_missing']:
            data = self._fill_missing(data)

        # Remove outliers
        if self.config['remove_outliers']:
            data = self._remove_outliers(data)

        return data

    def _fill_missing(self, data):
        """Fill missing values."""
        method = self.config['fill_missing']

        for key in ['close', 'high', 'low', 'open', 'volume']:
            if key not in data:
                continue

            arr = data[key]
            df = pd.DataFrame({key: arr})

            if method == 'forward':
                df = df.fillna(method='ffill')
            elif method == 'backward':
                df = df.fillna(method='bfill')
            elif method == 'mean':
                df = df.fillna(df.mean())

            data[key] = df[key].values

        return data

    def _remove_outliers(self, data):
        """Remove outliers using IQR method."""
        for key in ['close', 'high', 'low', 'open']:
            if key not in data:
                continue

            arr = data[key]
            q1 = np.percentile(arr, 25)
            q3 = np.percentile(arr, 75)
            iqr = q3 - q1

            lower_bound = q1 - (1.5 * iqr)
            upper_bound = q3 + (1.5 * iqr)

            # Clip outliers
            data[key] = np.clip(arr, lower_bound, upper_bound)

        return data

    def train_test_split(self, data, train_ratio=0.8, validation_ratio=0.1):
        """
        Split data into train/validation/test sets (temporal split).

        Args:
            data (dict): Full dataset
            train_ratio (float): Proportion for training
            validation_ratio (float): Proportion for validation

        Returns:
            tuple: (train_data, val_data, test_data)
        """
        n = len(data['close'])

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + validation_ratio))

        train_data = self._slice_data(data, 0, train_end)
        val_data = self._slice_data(data, train_end, val_end)
        test_data = self._slice_data(data, val_end, n)

        logger.info(f"Split: Train={len(train_data['close'])}, "
                   f"Val={len(val_data['close'])}, "
                   f"Test={len(test_data['close'])}")

        return train_data, val_data, test_data

    @staticmethod
    def _slice_data(data, start, end):
        """Slice data dictionary."""
        return {
            key: value[start:end] if isinstance(value, np.ndarray) else value
            for key, value in data.items()
        }

    def get_data_stats(self, data):
        """
        Get statistics about the data.

        Args:
            data (dict): Data dictionary

        Returns:
            dict: Statistics
        """
        close_prices = data['close']

        stats = {
            'total_points': len(close_prices),
            'start_price': close_prices[0],
            'end_price': close_prices[-1],
            'min_price': np.min(close_prices),
            'max_price': np.max(close_prices),
            'mean_price': np.mean(close_prices),
            'std_price': np.std(close_prices),
            'total_return': (close_prices[-1] - close_prices[0]) / close_prices[0] * 100,
        }

        if 'volume' in data:
            stats['avg_volume'] = np.mean(data['volume'])

        return stats

    def load_multiple_tickers(self, tickers, start_date=None, end_date=None, interval=None):
        """
        Load data for multiple tickers and concatenate.

        Args:
            tickers (list): List of ticker symbols
            start_date (str): Start date
            end_date (str): End date
            interval (str): Data interval

        Returns:
            dict: Concatenated data from all tickers
        """
        logger.info(f"Loading data for {len(tickers)} tickers: {tickers}")

        all_data = []

        for ticker in tickers:
            try:
                data = self._load_from_yfinance(ticker, start_date, end_date, interval)
                data = self._preprocess(data)
                data = self._validate_data(data, ticker)
                all_data.append(data)
                logger.info(f"  ✓ {ticker}: {len(data['close'])} points")
            except Exception as e:
                logger.warning(f"  ✗ {ticker}: Failed to load - {e}")
                continue

        if not all_data:
            raise ValueError("Failed to load data for any ticker")

        # Concatenate all data
        combined_data = self._concatenate_data(all_data)

        logger.info(f"Combined data: {len(combined_data['close'])} total points")

        return combined_data

    def _concatenate_data(self, data_list):
        """
        Concatenate multiple data dictionaries.

        Args:
            data_list (list): List of data dicts

        Returns:
            dict: Combined data
        """
        combined = {}

        for key in ['close', 'high', 'low', 'open', 'volume']:
            arrays = [d[key] for d in data_list if key in d]
            if arrays:
                combined[key] = np.concatenate(arrays)

        # Combine dates if present
        if 'dates' in data_list[0]:
            combined['dates'] = []
            for d in data_list:
                if 'dates' in d:
                    combined['dates'].extend(d['dates'])

        return combined

    def _validate_data(self, data, ticker):
        """
        Validate data quality and detect issues.

        Args:
            data (dict): Data to validate
            ticker (str): Ticker symbol

        Returns:
            dict: Validated data (with warnings logged)
        """
        close_prices = data['close']

        # 1. Check for NaN/Inf values
        if np.any(np.isnan(close_prices)) or np.any(np.isinf(close_prices)):
            logger.warning(f"{ticker}: Found NaN/Inf values in data")
            # Replace with forward fill
            close_prices = pd.Series(close_prices).fillna(method='ffill').fillna(method='bfill').values
            data['close'] = close_prices

        # 2. Check for zeros
        if np.any(close_prices <= 0):
            logger.warning(f"{ticker}: Found zero or negative prices")
            # Replace zeros with previous non-zero value
            close_prices = pd.Series(close_prices).replace(0, np.nan).fillna(method='ffill').values
            data['close'] = close_prices

        # 3. Detect potential stock splits (>20% single-day change)
        if len(close_prices) > 1:
            pct_changes = np.abs(np.diff(close_prices) / close_prices[:-1])
            split_candidates = np.where(pct_changes > 0.2)[0]

            if len(split_candidates) > 0:
                logger.warning(f"{ticker}: Potential stock splits detected at indices: {split_candidates.tolist()}")

        # 4. Check for data gaps (if dates available)
        if 'dates' in data and len(data['dates']) > 1:
            dates = pd.to_datetime(data['dates'])
            date_diffs = np.diff(dates).astype('timedelta64[D]').astype(int)

            # For daily data, gaps > 5 days (excluding weekends)
            if 'interval' in self.config and self.config['interval'] == '1d':
                large_gaps = np.where(date_diffs > 5)[0]
                if len(large_gaps) > 0:
                    logger.warning(f"{ticker}: {len(large_gaps)} data gaps detected (>5 days)")

        # 5. Check volatility (too low might indicate stale data)
        if len(close_prices) > 20:
            volatility = np.std(np.diff(close_prices) / close_prices[:-1])
            if volatility < 0.0001:  # Less than 0.01% daily volatility
                logger.warning(f"{ticker}: Unusually low volatility ({volatility:.6f}) - possible stale data")

        return data

    def load_with_market_index(self, ticker, index='NIFTY50', start_date=None, end_date=None, interval=None):
        """
        Load stock data along with market index for context.

        Args:
            ticker (str): Stock ticker
            index (str): Market index ('NIFTY50', 'BANKNIFTY', 'SENSEX')
            start_date (str): Start date
            end_date (str): End date
            interval (str): Data interval

        Returns:
            dict: Data with market index included
        """
        # Index ticker mapping
        index_tickers = {
            'NIFTY50': '^NSEI',
            'BANKNIFTY': '^NSEBANK',
            'SENSEX': '^BSESN'
        }

        if index not in index_tickers:
            logger.warning(f"Unknown index {index}, skipping")
            return self.load_data(ticker, start_date, end_date, interval)

        index_ticker = index_tickers[index]

        logger.info(f"Loading {ticker} with {index} index")

        # Load stock data
        stock_data = self.load_data(ticker, start_date, end_date, interval)

        # Load index data
        try:
            index_data = self._load_from_yfinance(index_ticker, start_date, end_date, interval)
            index_data = self._preprocess(index_data)

            # Align lengths (take minimum)
            min_len = min(len(stock_data['close']), len(index_data['close']))

            # Add index data as additional features
            stock_data['index_close'] = index_data['close'][:min_len]
            stock_data['index_volume'] = index_data.get('volume', np.zeros(min_len))[:min_len]

            # Trim stock data to match
            for key in ['close', 'high', 'low', 'open', 'volume']:
                if key in stock_data:
                    stock_data[key] = stock_data[key][:min_len]

            logger.info(f"Added {index} data ({min_len} points)")

        except Exception as e:
            logger.warning(f"Failed to load {index} data: {e}")

        return stock_data

    def augment_data(self, data, noise_level=0.01, n_augmented=3):
        """
        Create augmented versions of data for better generalization.

        Args:
            data (dict): Original data
            noise_level (float): Noise standard deviation (as fraction of price)
            n_augmented (int): Number of augmented versions to create

        Returns:
            dict: Augmented data (original + augmented concatenated)
        """
        logger.info(f"Augmenting data with {n_augmented} noisy versions (noise={noise_level})")

        augmented_list = [data]  # Start with original

        close_prices = data['close']

        for i in range(n_augmented):
            aug_data = {}

            # Add price noise
            noise = np.random.normal(0, noise_level, len(close_prices)) * close_prices
            aug_data['close'] = close_prices + noise

            # Augment OHLV if available
            if 'high' in data:
                noise_h = np.random.normal(0, noise_level, len(data['high'])) * data['high']
                aug_data['high'] = data['high'] + noise_h

            if 'low' in data:
                noise_l = np.random.normal(0, noise_level, len(data['low'])) * data['low']
                aug_data['low'] = data['low'] + noise_l

            if 'open' in data:
                noise_o = np.random.normal(0, noise_level, len(data['open'])) * data['open']
                aug_data['open'] = data['open'] + noise_o

            if 'volume' in data:
                # Volume noise (log-normal to keep positive)
                aug_data['volume'] = data['volume'] * np.random.lognormal(0, noise_level, len(data['volume']))

            augmented_list.append(aug_data)

        # Concatenate all
        combined = self._concatenate_data(augmented_list)

        logger.info(f"Augmented data: {len(data['close'])} → {len(combined['close'])} points")

        return combined


def load_data_from_config(config):
    """
    Convenience function to load data from config.

    Args:
        config: Config object or dict

    Returns:
        dict: Loaded data
    """
    if hasattr(config, 'get_section'):
        data_config = config.get_section('data')
    else:
        data_config = config.get('data', {})

    loader = DataLoader(data_config)
    return loader.load_data()
