"""
Advanced reward functions for RL trading.
Implements risk-adjusted rewards with drawdown penalties, Sharpe components, etc.
"""

import numpy as np
import logging
from collections import deque


logger = logging.getLogger(__name__)


class RewardFunction:
    """Base class for reward functions."""

    def __init__(self, config=None):
        """
        Initialize reward function.

        Args:
            config (dict): Reward configuration
        """
        self.config = config or self._default_config()

    @staticmethod
    def _default_config():
        return {
            'type': 'profit_with_risk',
            'risk_penalty_factor': 0.1,
            'holding_penalty': 0.001,
            'transaction_cost_aware': True,
            'max_drawdown_penalty': 0.5,
            'volatility_penalty': 0.2,
            'sharpe_weight': 0.3
        }

    def calculate(self, **kwargs):
        """Calculate reward. To be implemented by subclasses."""
        raise NotImplementedError


class SimpleProfitRewardFunction(RewardFunction):
    """Simple profit-based reward (baseline)."""

    def calculate(self, profit, done=False):
        """
        Calculate simple profit reward.

        Args:
            profit (float): Trade profit/loss
            done (bool): Whether episode is done

        Returns:
            float: Reward value
        """
        # Clip reward to prevent extreme values
        return max(profit, -1.0)


class ProfitWithRiskRewardFunction(RewardFunction):
    """
    Risk-adjusted profit reward.

    Incorporates:
    - Base profit
    - Drawdown penalty
    - Volatility penalty
    - Holding penalty for losing positions
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.portfolio_history = deque(maxlen=100)
        self.returns_history = deque(maxlen=50)

    def calculate(self, profit, portfolio_value, prev_portfolio_value,
                  holding_losing_position=False, transaction_cost=0.0):
        """
        Calculate risk-adjusted reward.

        Args:
            profit (float): Current trade profit
            portfolio_value (float): Current portfolio value
            prev_portfolio_value (float): Previous portfolio value
            holding_losing_position (bool): Whether holding a losing position
            transaction_cost (float): Transaction costs incurred

        Returns:
            float: Risk-adjusted reward
        """
        reward = 0.0

        # 1. Base profit component
        reward += profit

        # 2. Transaction cost penalty
        if self.config['transaction_cost_aware']:
            reward -= transaction_cost

        # 3. Holding penalty for losing positions
        if holding_losing_position:
            reward -= self.config['holding_penalty']

        # 4. Drawdown penalty
        self.portfolio_history.append(portfolio_value)
        if len(self.portfolio_history) > 1:
            peak = max(self.portfolio_history)
            drawdown = (portfolio_value - peak) / peak if peak > 0 else 0.0

            if drawdown < 0:
                drawdown_penalty = abs(drawdown) * self.config['max_drawdown_penalty']
                reward -= drawdown_penalty

        # 5. Volatility penalty
        if prev_portfolio_value > 0:
            period_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
            self.returns_history.append(period_return)

            if len(self.returns_history) >= 10:
                volatility = np.std(list(self.returns_history))
                volatility_penalty = volatility * self.config['volatility_penalty']
                reward -= volatility_penalty

        return reward

    def reset(self):
        """Reset internal state for new episode."""
        self.portfolio_history.clear()
        self.returns_history.clear()


class SharpeRewardFunction(RewardFunction):
    """
    Sharpe ratio-based reward.

    Encourages high returns with low volatility.
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.returns_history = deque(maxlen=100)
        self.risk_free_rate = 0.02 / 252  # Daily risk-free rate

    def calculate(self, portfolio_value, prev_portfolio_value):
        """
        Calculate Sharpe-based reward.

        Args:
            portfolio_value (float): Current portfolio value
            prev_portfolio_value (float): Previous portfolio value

        Returns:
            float: Sharpe-based reward
        """
        if prev_portfolio_value == 0:
            return 0.0

        # Calculate return
        period_return = (portfolio_value - prev_portfolio_value) / prev_portfolio_value
        self.returns_history.append(period_return)

        # Need enough history for Sharpe calculation
        if len(self.returns_history) < 10:
            return period_return  # Return simple return initially

        # Calculate Sharpe ratio
        returns = np.array(list(self.returns_history))
        excess_returns = returns - self.risk_free_rate
        mean_excess = np.mean(excess_returns)
        std_excess = np.std(excess_returns)

        if std_excess == 0:
            sharpe = 0.0
        else:
            sharpe = mean_excess / std_excess

        # Combine Sharpe with current return
        reward = (
            self.config['sharpe_weight'] * sharpe +
            (1 - self.config['sharpe_weight']) * period_return
        )

        return reward

    def reset(self):
        """Reset internal state."""
        self.returns_history.clear()


class MultiObjectiveRewardFunction(RewardFunction):
    """
    Multi-objective reward combining multiple components.

    Balances:
    - Profit maximization
    - Risk minimization
    - Transaction cost minimization
    - Sharpe ratio optimization
    """

    def __init__(self, config=None):
        super().__init__(config)
        self.profit_reward = ProfitWithRiskReward(config)
        self.sharpe_reward = SharpeReward(config)

    def calculate(self, profit, portfolio_value, prev_portfolio_value,
                  holding_losing_position=False, transaction_cost=0.0):
        """
        Calculate multi-objective reward.

        Args:
            profit (float): Current trade profit
            portfolio_value (float): Current portfolio value
            prev_portfolio_value (float): Previous portfolio value
            holding_losing_position (bool): Whether holding losing position
            transaction_cost (float): Transaction costs

        Returns:
            float: Combined reward
        """
        # Get profit-based reward
        profit_component = self.profit_reward.calculate(
            profit=profit,
            portfolio_value=portfolio_value,
            prev_portfolio_value=prev_portfolio_value,
            holding_losing_position=holding_losing_position,
            transaction_cost=transaction_cost
        )

        # Get Sharpe-based reward
        sharpe_component = self.sharpe_reward.calculate(
            portfolio_value=portfolio_value,
            prev_portfolio_value=prev_portfolio_value
        )

        # Combine components
        reward = 0.6 * profit_component + 0.4 * sharpe_component

        return reward

    def reset(self):
        """Reset all components."""
        self.profit_reward.reset()
        self.sharpe_reward.reset()


def get_reward_function(reward_type='profit_with_risk', config=None):
    """
    Factory function to get reward function by type.

    Args:
        reward_type (str): Type of reward function
        config (dict): Configuration

    Returns:
        RewardFunction: Reward function instance
    """
    reward_functions = {
        'profit_only': SimpleProfitRewardFunction,
        'profit_with_risk': ProfitWithRiskRewardFunction,
        'sharpe_based': SharpeRewardFunction,
        'multi_objective': MultiObjectiveRewardFunction
    }

    if reward_type not in reward_functions:
        logger.warning(f"Unknown reward type '{reward_type}', using profit_with_risk")
        reward_type = 'profit_with_risk'

    return reward_functions[reward_type](config)
