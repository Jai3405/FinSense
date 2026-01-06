"""
Base Agent class for DQN implementations.
Provides common interface and utilities for all agent types.
"""

import numpy as np
import random
from collections import deque
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    """Abstract base class for all DQN agents."""

    def __init__(self, state_size, action_size=3, config=None):
        """
        Initialize base agent.

        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions (default: 3 for Buy/Hold/Sell)
            config (dict): Configuration dictionary with hyperparameters
        """
        self.state_size = state_size
        self.action_size = action_size
        self.inventory = []

        # Load config or use defaults
        if config is None:
            config = self._default_config()

        self.gamma = config.get('gamma', 0.95)
        self.epsilon = config.get('epsilon_start', 1.0)
        self.epsilon_min = config.get('epsilon_min', 0.01)
        self.epsilon_decay = config.get('epsilon_decay', 0.995)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.batch_size = config.get('batch_size', 32)
        self.memory_size = config.get('memory_size', 2000)
        self.target_update_frequency = config.get('target_update_frequency', 10)

        # Experience replay memory
        self.memory = deque(maxlen=self.memory_size)

        # Training step counter for target network updates
        self.train_step = 0

    @staticmethod
    def _default_config():
        """Return default hyperparameter configuration."""
        return {
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 2000,
            'target_update_frequency': 10,
            'hidden_size': 64,
            'dropout': 0.2
        }

    @abstractmethod
    def act(self, state, training=True):
        """
        Select action using epsilon-greedy policy.

        Args:
            state: Current state
            training (bool): If False, uses greedy policy (no exploration)

        Returns:
            int: Selected action index
        """
        pass

    @abstractmethod
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        pass

    @abstractmethod
    def replay(self, batch_size=None):
        """
        Train on a batch of experiences from replay buffer.

        Args:
            batch_size (int): Size of training batch (uses self.batch_size if None)

        Returns:
            float: Training loss
        """
        pass

    @abstractmethod
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Save model to file."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Load model from file."""
        pass

    def decay_epsilon(self):
        """Decay exploration rate."""
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def can_replay(self, batch_size=None):
        """Check if enough experiences for training."""
        batch_size = batch_size or self.batch_size
        return len(self.memory) >= batch_size
