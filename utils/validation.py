"""
Walk-forward validation for time series.
Prevents overfitting and provides realistic performance estimates.
"""

import numpy as np
import logging
from typing import List, Tuple, Dict


logger = logging.getLogger(__name__)


class WalkForwardValidator:
    """
    Walk-forward validation for time series trading strategies.

    Implements:
    - Rolling window approach
    - Train on past N months
    - Test on next M months
    - Step forward by S months
    """

    def __init__(self, config=None):
        """
        Initialize walk-forward validator.

        Args:
            config (dict): Validation configuration
        """
        self.config = config or self._default_config()

        self.train_period = self.config.get('train_period', 6)  # months
        self.test_period = self.config.get('test_period', 1)  # months
        self.step_size = self.config.get('step_size', 1)  # months

        logger.info(f"Walk-forward validator: train={self.train_period}m, "
                   f"test={self.test_period}m, step={self.step_size}m")

    @staticmethod
    def _default_config():
        """Default configuration."""
        return {
            'train_period': 6,  # 6 months training
            'test_period': 1,   # 1 month testing
            'step_size': 1,     # Step forward 1 month
            'min_train_size': 100  # Minimum data points for training
        }

    def split_data(self, data, interval='1d'):
        """
        Split data into walk-forward windows.

        Args:
            data (dict): Full dataset with 'close' prices
            interval (str): Data interval ('1d', '5m', etc.)

        Returns:
            list: List of (train_data, test_data) tuples
        """
        total_points = len(data['close'])

        # Calculate points per month based on interval
        points_per_month = self._get_points_per_month(interval)

        train_size = self.train_period * points_per_month
        test_size = self.test_period * points_per_month
        step = self.step_size * points_per_month

        windows = []
        current_start = 0

        while True:
            train_end = current_start + train_size
            test_end = train_end + test_size

            # Check if we have enough data
            if test_end > total_points:
                break

            # Check minimum training size
            if train_size < self.config.get('min_train_size', 100):
                logger.warning(f"Training window too small: {train_size} points")
                break

            # Extract windows
            train_data = self._slice_data(data, current_start, train_end)
            test_data = self._slice_data(data, train_end, test_end)

            windows.append((train_data, test_data))

            logger.debug(f"Window {len(windows)}: train[{current_start}:{train_end}], "
                        f"test[{train_end}:{test_end}]")

            # Step forward
            current_start += step

        logger.info(f"Created {len(windows)} walk-forward windows")

        return windows

    @staticmethod
    def _get_points_per_month(interval):
        """
        Get approximate data points per month.

        Args:
            interval (str): Data interval

        Returns:
            int: Points per month
        """
        if interval == '1d':
            return 21  # ~21 trading days per month
        elif interval == '5m':
            return 21 * 75  # 21 days * 75 intervals per day
        elif interval == '15m':
            return 21 * 25  # 21 days * 25 intervals per day
        elif interval == '1h':
            return 21 * 6   # 21 days * 6 hours per day
        else:
            logger.warning(f"Unknown interval {interval}, defaulting to 21")
            return 21

    @staticmethod
    def _slice_data(data, start, end):
        """Slice data dictionary."""
        sliced = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                sliced[key] = value[start:end]
            elif isinstance(value, list):
                sliced[key] = value[start:end]
            else:
                sliced[key] = value

        return sliced

    def validate(self, train_fn, evaluate_fn, data, interval='1d'):
        """
        Run walk-forward validation.

        Args:
            train_fn: Function to train model (train_fn(train_data) -> model)
            evaluate_fn: Function to evaluate model (evaluate_fn(model, test_data) -> metrics)
            data (dict): Full dataset
            interval (str): Data interval

        Returns:
            dict: Aggregated validation results
        """
        windows = self.split_data(data, interval)

        if len(windows) == 0:
            raise ValueError("No validation windows created - data may be too short")

        all_metrics = []

        for i, (train_data, test_data) in enumerate(windows):
            logger.info(f"Validating window {i+1}/{len(windows)}...")

            # Train model on this window
            model = train_fn(train_data)

            # Evaluate on test period
            metrics = evaluate_fn(model, test_data)
            all_metrics.append(metrics)

            logger.info(f"Window {i+1} metrics: {metrics}")

        # Aggregate results
        aggregated = self._aggregate_metrics(all_metrics)

        logger.info(f"Walk-forward validation complete: {aggregated}")

        return aggregated

    @staticmethod
    def _aggregate_metrics(all_metrics):
        """
        Aggregate metrics across all windows.

        Args:
            all_metrics (list): List of metric dictionaries

        Returns:
            dict: Aggregated metrics (mean, std, min, max)
        """
        if not all_metrics:
            return {}

        # Get all metric names
        metric_names = set()
        for metrics in all_metrics:
            metric_names.update(metrics.keys())

        aggregated = {}

        for name in metric_names:
            values = [m.get(name, 0.0) for m in all_metrics if name in m]

            if values:
                aggregated[f'{name}_mean'] = np.mean(values)
                aggregated[f'{name}_std'] = np.std(values)
                aggregated[f'{name}_min'] = np.min(values)
                aggregated[f'{name}_max'] = np.max(values)

        # Add number of windows
        aggregated['num_windows'] = len(all_metrics)

        return aggregated


def create_temporal_splits(data, n_splits=5):
    """
    Create temporal train/test splits (simpler than walk-forward).

    Args:
        data (dict): Full dataset
        n_splits (int): Number of splits

    Returns:
        list: List of (train_data, test_data) tuples
    """
    total_points = len(data['close'])
    points_per_split = total_points // (n_splits + 1)

    splits = []

    for i in range(n_splits):
        train_end = points_per_split * (i + 1)
        test_end = min(train_end + points_per_split, total_points)

        train_data = WalkForwardValidator._slice_data(data, 0, train_end)
        test_data = WalkForwardValidator._slice_data(data, train_end, test_end)

        splits.append((train_data, test_data))

    return splits
