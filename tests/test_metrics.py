"""
Test suite for trading metrics.
"""

import pytest
import numpy as np

from utils.metrics import TradingMetrics


class TestTradingMetrics:
    """Test performance metrics calculations."""

    @pytest.fixture
    def metrics(self):
        """Create metrics calculator."""
        return TradingMetrics(risk_free_rate=0.02)

    @pytest.fixture
    def sample_returns(self):
        """Generate sample returns."""
        np.random.seed(42)
        return np.random.randn(100) * 0.01  # 1% daily volatility

    @pytest.fixture
    def sample_portfolio_values(self):
        """Generate sample portfolio values."""
        np.random.seed(42)
        returns = np.random.randn(100) * 0.01
        portfolio = [100000]
        for ret in returns:
            portfolio.append(portfolio[-1] * (1 + ret))
        return np.array(portfolio)

    @pytest.fixture
    def sample_trades(self):
        """Generate sample trades."""
        np.random.seed(42)
        return list(np.random.randn(50) * 100)  # 50 trades

    def test_sharpe_ratio_positive(self, metrics):
        """Test Sharpe ratio with positive returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])  # Positive returns
        sharpe = metrics.sharpe_ratio(returns, periods_per_year=252)

        # Should be positive
        assert sharpe > 0

    def test_sharpe_ratio_negative(self, metrics):
        """Test Sharpe ratio with negative returns."""
        returns = np.array([-0.01, -0.02, -0.015, -0.01, -0.02])  # Negative returns
        sharpe = metrics.sharpe_ratio(returns, periods_per_year=252)

        # Should be negative
        assert sharpe < 0

    def test_sharpe_ratio_zero_volatility(self, metrics):
        """Test Sharpe ratio with zero volatility."""
        returns = np.array([0.01, 0.01, 0.01, 0.01])  # Constant returns
        sharpe = metrics.sharpe_ratio(returns)

        # Should return 0 (or handle gracefully)
        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self, metrics):
        """Test Sharpe ratio with insufficient data."""
        returns = np.array([0.01])
        sharpe = metrics.sharpe_ratio(returns)

        assert sharpe == 0.0

    def test_sortino_ratio_positive(self, metrics):
        """Test Sortino ratio with mixed returns."""
        returns = np.array([0.02, -0.01, 0.03, -0.005, 0.015])
        sortino = metrics.sortino_ratio(returns)

        # Should be positive (positive mean return)
        assert sortino > 0

    def test_sortino_ratio_no_downside(self, metrics):
        """Test Sortino ratio with no negative returns."""
        returns = np.array([0.01, 0.02, 0.015, 0.01, 0.02])
        sortino = metrics.sortino_ratio(returns)

        # Should be very high (infinite or large)
        assert sortino > 10 or np.isinf(sortino)

    def test_max_drawdown_basic(self, metrics):
        """Test maximum drawdown calculation."""
        portfolio = np.array([100, 110, 105, 115, 95, 105, 110])
        max_dd, peak_idx, trough_idx = metrics.max_drawdown(portfolio)

        # Should be negative
        assert max_dd < 0

        # Peak should be before trough
        assert peak_idx <= trough_idx

        # Drawdown should be from 115 to 95 = -17.4%
        expected_dd = (95 - 115) / 115
        assert abs(max_dd - expected_dd) < 0.01

    def test_max_drawdown_no_drawdown(self, metrics):
        """Test max drawdown with only gains."""
        portfolio = np.array([100, 110, 120, 130])
        max_dd, peak_idx, trough_idx = metrics.max_drawdown(portfolio)

        # Should be zero
        assert max_dd == 0.0

    def test_max_drawdown_single_value(self, metrics):
        """Test max drawdown with single value."""
        portfolio = np.array([100])
        max_dd, peak_idx, trough_idx = metrics.max_drawdown(portfolio)

        assert max_dd == 0.0

    def test_calmar_ratio(self, metrics, sample_portfolio_values):
        """Test Calmar ratio calculation."""
        returns = np.diff(sample_portfolio_values) / sample_portfolio_values[:-1]
        calmar = metrics.calmar_ratio(returns, sample_portfolio_values)

        # Should be numeric
        assert isinstance(calmar, (int, float)) or np.isinf(calmar)

    def test_calmar_ratio_no_drawdown(self, metrics):
        """Test Calmar ratio with no drawdown."""
        portfolio = np.array([100, 110, 120, 130])
        returns = np.diff(portfolio) / portfolio[:-1]
        calmar = metrics.calmar_ratio(returns, portfolio)

        # Should be infinite (no drawdown)
        assert np.isinf(calmar)

    def test_win_rate(self, metrics):
        """Test win rate calculation."""
        trades = [100, -50, 200, -30, 150, -20, 80]
        win_rate = metrics.win_rate(trades)

        # 4 wins out of 7 = 57%
        assert abs(win_rate - 4/7) < 0.01

    def test_win_rate_all_wins(self, metrics):
        """Test win rate with all winning trades."""
        trades = [100, 50, 200, 30, 150]
        win_rate = metrics.win_rate(trades)

        assert win_rate == 1.0

    def test_win_rate_all_losses(self, metrics):
        """Test win rate with all losing trades."""
        trades = [-100, -50, -200, -30]
        win_rate = metrics.win_rate(trades)

        assert win_rate == 0.0

    def test_win_rate_no_trades(self, metrics):
        """Test win rate with no trades."""
        trades = []
        win_rate = metrics.win_rate(trades)

        assert win_rate == 0.0

    def test_profit_factor(self, metrics):
        """Test profit factor calculation."""
        trades = [100, -50, 200, -30, 150]
        pf = metrics.profit_factor(trades)

        # Gross profit = 450, Gross loss = 80
        expected_pf = 450 / 80
        assert abs(pf - expected_pf) < 0.01

    def test_profit_factor_no_losses(self, metrics):
        """Test profit factor with no losses."""
        trades = [100, 50, 200]
        pf = metrics.profit_factor(trades)

        # Should be infinite
        assert np.isinf(pf)

    def test_profit_factor_no_gains(self, metrics):
        """Test profit factor with no gains."""
        trades = [-100, -50, -200]
        pf = metrics.profit_factor(trades)

        # Should be zero
        assert pf == 0.0

    def test_average_profit_per_trade(self, metrics):
        """Test average profit per trade."""
        trades = [100, -50, 200, -30, 150]
        avg = metrics.average_profit_per_trade(trades)

        expected_avg = (100 - 50 + 200 - 30 + 150) / 5
        assert abs(avg - expected_avg) < 0.01

    def test_average_win(self, metrics):
        """Test average winning trade."""
        trades = [100, -50, 200, -30, 150]
        avg_win = metrics.average_win(trades)

        # Average of 100, 200, 150 = 150
        assert avg_win == 150.0

    def test_average_loss(self, metrics):
        """Test average losing trade."""
        trades = [100, -50, 200, -30, 150]
        avg_loss = metrics.average_loss(trades)

        # Average of -50, -30 = -40
        assert avg_loss == -40.0

    def test_expectancy(self, metrics):
        """Test expectancy calculation."""
        trades = [100, -50, 200, -30, 150, -20]
        expectancy = metrics.expectancy(trades)

        # (Win Rate × Avg Win) - (Loss Rate × Avg Loss)
        win_rate = 4/6  # 4 wins out of 6
        avg_win = 150.0
        avg_loss = 100/3  # Average of 50, 30, 20

        expected = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        assert abs(expectancy - expected) < 0.1

    def test_calculate_all_metrics(self, metrics, sample_portfolio_values, sample_trades):
        """Test comprehensive metrics calculation."""
        all_metrics = metrics.calculate_all_metrics(
            sample_portfolio_values,
            sample_trades
        )

        # Check all expected keys present
        expected_keys = [
            'total_profit', 'total_return', 'sharpe_ratio', 'sortino_ratio',
            'calmar_ratio', 'max_drawdown', 'max_drawdown_percent',
            'total_trades', 'win_rate', 'loss_rate', 'profit_factor',
            'avg_profit_per_trade', 'avg_win', 'avg_loss', 'expectancy',
            'volatility'
        ]

        for key in expected_keys:
            assert key in all_metrics

        # Check values are reasonable
        assert isinstance(all_metrics['total_profit'], (int, float))
        assert isinstance(all_metrics['sharpe_ratio'], (int, float))
        assert 0 <= all_metrics['win_rate'] <= 1
        assert all_metrics['total_trades'] == len(sample_trades)

    def test_metrics_with_perfect_strategy(self, metrics):
        """Test metrics with perfect (all-win) strategy."""
        portfolio = np.array([100, 110, 121, 133.1])  # 10% gain each period
        trades = [10, 11, 12.1]  # All winning trades

        all_metrics = metrics.calculate_all_metrics(portfolio, trades)

        assert all_metrics['win_rate'] == 1.0
        assert all_metrics['max_drawdown'] == 0.0
        assert np.isinf(all_metrics['profit_factor'])
        assert all_metrics['total_profit'] > 0

    def test_metrics_with_losing_strategy(self, metrics):
        """Test metrics with losing strategy."""
        portfolio = np.array([100, 95, 90, 85])  # Consistent losses
        trades = [-5, -5, -5]  # All losing trades

        all_metrics = metrics.calculate_all_metrics(portfolio, trades)

        assert all_metrics['win_rate'] == 0.0
        assert all_metrics['total_profit'] < 0
        assert all_metrics['sharpe_ratio'] < 0
        assert all_metrics['profit_factor'] == 0.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
