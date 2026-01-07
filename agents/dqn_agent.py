"""
Double DQN Agent with Target Networks (PyTorch implementation).
Implements state-of-the-art DQN with:
- Target network for stable learning
- Double DQN to reduce Q-value overestimation
- Proper random sampling from replay buffer
- GPU acceleration support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import logging
from pathlib import Path

from .base_agent import BaseAgent


logger = logging.getLogger(__name__)


class DQNNetwork(nn.Module):
    """Dueling Deep Q-Network architecture."""

    def __init__(self, state_size, action_size, hidden_size=64, dropout=0.2): # dropout is unused but kept for API consistency
        super(DQNNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_size, hidden_size)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))

        value = self.value_stream(x)
        advantage = self.advantage_stream(x)

        # Combine value and advantage streams
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values


class DQNAgent(BaseAgent):
    """
    Double DQN Agent with target network.

    Features:
    - Double DQN algorithm to reduce overestimation bias
    - Separate target network updated periodically
    - Experience replay with random sampling
    - GPU support
    - Configurable hyperparameters
    """

    def __init__(self, state_size, action_size=3, config=None):
        """
        Initialize DQN Agent.

        Args:
            state_size (int): Dimension of state space
            action_size (int): Number of possible actions
            config (dict): Configuration dictionary
        """
        super().__init__(state_size, action_size, config)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Get network config
        if config is None:
            config = self._default_config()
        hidden_size = config.get('hidden_size', 64)
        dropout = config.get('dropout', 0.2)

        # Initialize Q-network and target network
        self.q_network = DQNNetwork(
            state_size, action_size, hidden_size, dropout
        ).to(self.device)

        self.target_network = DQNNetwork(
            state_size, action_size, hidden_size, dropout
        ).to(self.device)

        # Copy Q-network weights to target network
        self.update_target_network()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        logger.info(f"Initialized DQN Agent with state_size={state_size}, "
                   f"action_size={action_size}")

    def act(self, state, training=True, action_mask=None):
        """
        Select action using epsilon-greedy policy, with action masking.

        Args:
            state (np.ndarray): Current state
            training (bool): If True, uses epsilon-greedy; if False, uses greedy
            action_mask (list, optional): Boolean mask of valid actions.

        Returns:
            int: Selected action index
        """
        # Epsilon-greedy exploration during training
        if training and np.random.random() <= self.epsilon:
            if action_mask is not None:
                valid_actions = [i for i, valid in enumerate(action_mask) if valid]
                return random.choice(valid_actions) if valid_actions else 1 # Default to Hold
            return random.randrange(self.action_size)

        # Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)

        if action_mask is not None:
            masked_q = q_values.clone()
            for i, valid in enumerate(action_mask):
                if not valid:
                    masked_q[0, i] = -float('inf')
            return int(torch.argmax(masked_q).item())

        return int(torch.argmax(q_values).item())

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
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size=None):
        """
        Train on random batch from replay buffer using Double DQN.

        Double DQN formula:
        Q_target = reward + gamma * Q_target(next_state, argmax_a Q(next_state, a))

        This reduces overestimation by using Q-network for action selection
        but target network for value estimation.

        Args:
            batch_size (int): Batch size for training

        Returns:
            float: Training loss (None if not enough experiences)
        """
        batch_size = batch_size or self.batch_size

        if not self.can_replay(batch_size):
            return None

        # Random sampling from replay buffer
        batch = random.sample(self.memory, batch_size)

        # Unpack batch
        states = np.vstack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.vstack([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)

        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Double DQN: Use Q-network to select actions, target network to evaluate
        with torch.no_grad():
            # Select best actions using Q-network
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            # Evaluate those actions using target network
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()

        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # Compute loss
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # NOTE: Epsilon decay moved to end of episode in train.py
        # Don't decay here to avoid double decay bug (was causing epsilon to hit min at episode 20)

        # Update target network periodically
        self.train_step += 1
        if self.train_step % self.target_update_frequency == 0:
            self.update_target_network()

        return loss.item()

    def update_target_network(self):
        """Copy Q-network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
        logger.debug("Target network updated")

    def save(self, filepath):
        """
        Save agent state to file.

        Args:
            filepath (str): Path to save model
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
            'state_size': self.state_size,
            'action_size': self.action_size,
            'config': {
                'gamma': self.gamma,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay': self.epsilon_decay,
                'learning_rate': self.learning_rate,
                'batch_size': self.batch_size,
                'memory_size': self.memory_size,
                'target_update_frequency': self.target_update_frequency
            }
        }, filepath)

        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """
        Load agent state from file.

        Args:
            filepath (str): Path to model file

        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=self.device)

        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.train_step = checkpoint['train_step']

        logger.info(f"Model loaded from {filepath}")

    def get_q_values(self, state):
        """
        Get Q-values for a given state.

        Args:
            state (np.ndarray): State

        Returns:
            np.ndarray: Q-values for all actions
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.cpu().numpy()[0]
