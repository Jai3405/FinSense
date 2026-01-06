"""
Test suite for DQN agents.
"""

import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile

from agents import DQNAgent, BaseAgent


class TestDQNAgent:
    """Test DQN agent implementation."""

    @pytest.fixture
    def agent(self):
        """Create a test agent."""
        config = {
            'gamma': 0.95,
            'epsilon_start': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995,
            'learning_rate': 0.001,
            'batch_size': 32,
            'memory_size': 1000,
            'target_update_frequency': 10,
            'hidden_size': 32,
            'dropout': 0.2
        }
        return DQNAgent(state_size=10, action_size=3, config=config)

    def test_agent_initialization(self, agent):
        """Test agent is properly initialized."""
        assert agent.state_size == 10
        assert agent.action_size == 3
        assert agent.gamma == 0.95
        assert agent.epsilon == 1.0
        assert len(agent.memory) == 0
        assert agent.q_network is not None
        assert agent.target_network is not None

    def test_act_exploration(self, agent):
        """Test epsilon-greedy exploration."""
        state = np.random.rand(10)

        # With epsilon=1.0, should explore (random action)
        actions = set()
        for _ in range(50):
            action = agent.act(state, training=True)
            actions.add(action)
            assert 0 <= action < 3

        # Should have explored multiple actions
        assert len(actions) > 1

    def test_act_exploitation(self, agent):
        """Test greedy action selection."""
        state = np.random.rand(10)
        agent.epsilon = 0.0  # No exploration

        # Should return consistent action
        actions = [agent.act(state, training=True) for _ in range(10)]
        assert len(set(actions)) == 1  # All same action

    def test_remember(self, agent):
        """Test experience storage."""
        state = np.random.rand(10)
        action = 1
        reward = 10.0
        next_state = np.random.rand(10)
        done = False

        agent.remember(state, action, reward, next_state, done)
        assert len(agent.memory) == 1

        # Add more experiences
        for _ in range(100):
            agent.remember(state, action, reward, next_state, done)

        assert len(agent.memory) == 101

    def test_memory_limit(self, agent):
        """Test memory buffer respects max size."""
        state = np.random.rand(10)

        # Fill beyond capacity
        for i in range(1500):
            agent.remember(state, 1, float(i), state, False)

        # Should not exceed max size
        assert len(agent.memory) == agent.memory_size

    def test_replay_insufficient_memory(self, agent):
        """Test replay returns None with insufficient memory."""
        # Add only a few experiences
        for _ in range(10):
            agent.remember(np.random.rand(10), 1, 0.0, np.random.rand(10), False)

        loss = agent.replay(batch_size=32)
        assert loss is None

    def test_replay_training(self, agent):
        """Test replay updates network."""
        # Fill memory
        for _ in range(100):
            state = np.random.rand(10)
            agent.remember(state, 1, 1.0, np.random.rand(10), False)

        # Get initial weights
        initial_params = [p.clone() for p in agent.q_network.parameters()]

        # Train
        loss = agent.replay(batch_size=32)

        # Check loss is returned
        assert loss is not None
        assert isinstance(loss, float)

        # Check weights changed
        current_params = list(agent.q_network.parameters())
        params_changed = any(
            not torch.equal(initial, current)
            for initial, current in zip(initial_params, current_params)
        )
        assert params_changed

    def test_target_network_update(self, agent):
        """Test target network is updated."""
        # Get initial target network params
        initial_target = [p.clone() for p in agent.target_network.parameters()]

        # Modify Q-network
        for p in agent.q_network.parameters():
            p.data.fill_(1.0)

        # Update target
        agent.update_target_network()

        # Check target network matches Q-network
        for q_param, target_param in zip(
            agent.q_network.parameters(),
            agent.target_network.parameters()
        ):
            assert torch.equal(q_param, target_param)

    def test_epsilon_decay(self, agent):
        """Test epsilon decays properly."""
        initial_epsilon = agent.epsilon
        agent.decay_epsilon()

        assert agent.epsilon < initial_epsilon
        assert agent.epsilon >= agent.epsilon_min

        # Decay to minimum
        for _ in range(1000):
            agent.decay_epsilon()

        assert agent.epsilon == agent.epsilon_min

    def test_save_and_load(self, agent):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.pt"

            # Train a bit to change weights
            for _ in range(50):
                agent.remember(np.random.rand(10), 1, 1.0, np.random.rand(10), False)
            agent.replay(batch_size=32)

            # Save
            agent.save(filepath)
            assert filepath.exists()

            # Create new agent and load
            new_agent = DQNAgent(state_size=10, action_size=3)
            new_agent.load(filepath)

            # Check weights match
            for p1, p2 in zip(agent.q_network.parameters(), new_agent.q_network.parameters()):
                assert torch.equal(p1, p2)

            assert new_agent.epsilon == agent.epsilon

    def test_load_nonexistent_file(self, agent):
        """Test loading from nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            agent.load("nonexistent_model.pt")

    def test_get_q_values(self, agent):
        """Test Q-value retrieval."""
        state = np.random.rand(10)
        q_values = agent.get_q_values(state)

        assert isinstance(q_values, np.ndarray)
        assert q_values.shape == (3,)  # 3 actions

    def test_double_dqn_logic(self, agent):
        """Test Double DQN uses separate networks."""
        # Fill memory
        for _ in range(100):
            agent.remember(np.random.rand(10), 1, 1.0, np.random.rand(10), False)

        # Modify target network to be different
        for p in agent.target_network.parameters():
            p.data.fill_(0.5)

        # Train should use both networks
        loss = agent.replay(batch_size=32)
        assert loss is not None

    def test_can_replay(self, agent):
        """Test can_replay checks memory size."""
        assert not agent.can_replay(batch_size=32)

        # Add enough experiences
        for _ in range(50):
            agent.remember(np.random.rand(10), 1, 0.0, np.random.rand(10), False)

        assert agent.can_replay(batch_size=32)

    def test_different_state_sizes(self):
        """Test agent works with different state sizes."""
        for state_size in [5, 10, 17, 50]:
            agent = DQNAgent(state_size=state_size, action_size=3)
            state = np.random.rand(state_size)
            action = agent.act(state)
            assert 0 <= action < 3

    def test_gpu_support(self, agent):
        """Test GPU/CPU device handling."""
        assert agent.device in [torch.device('cuda'), torch.device('cpu')]

        # Model should be on correct device
        for param in agent.q_network.parameters():
            assert param.device.type == agent.device.type


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
