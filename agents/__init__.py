"""
Agents module for FinSense.
Provides DQN implementations with Double DQN and target networks.
"""

from .base_agent import BaseAgent
from .dqn_agent import DQNAgent

__all__ = ['BaseAgent', 'DQNAgent']
