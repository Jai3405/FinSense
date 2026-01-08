# agents/ppo_agent.py

import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOAgent(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, state):
        x = self.shared(state)
        # Detach value head during policy update, but not here
        return self.policy_head(x), self.value_head(x)

    def act(self, state, action_mask=None):
        # The caller is now responsible for ensuring state is a tensor on the correct device
        logits, value = self.forward(state)

        # ACTION MASKING (CORRECT PPO WAY)
        if action_mask is not None:
            mask = torch.tensor(action_mask, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~mask, -1e9)

        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action),
            dist.entropy(),
            value.squeeze(-1)
        )

    def evaluate_action(self, state, action, action_mask=None):
        """
        Evaluate an action taken in a state, used during PPO update.
        """
        logits, value = self.forward(state)

        if action_mask is not None:
            logits = logits.masked_fill(~action_mask, -1e9)

        dist = Categorical(logits=logits)

        log_prob = dist.log_prob(action)
        entropy = dist.entropy()

        return log_prob, entropy, value
