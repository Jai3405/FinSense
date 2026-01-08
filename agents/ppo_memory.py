# agents/ppo_memory.py

import torch
import numpy as np

class PPOMemory:
    def __init__(self, batch_size, device):
        self.batch_size = batch_size
        self.device = device
        self.clear()

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.masks = []

    def add(self, state, action, log_prob, reward, done, value, mask):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)
        self.masks.append(mask)

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        return batches

    def compute_gae(self, last_value, gamma, lam):
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        gae = 0
        
        # Use last_value for the final state, ensuring it's a scalar
        last_value_scalar = last_value.detach().cpu().item() if isinstance(last_value, torch.Tensor) else last_value
        values_np = torch.cat(self.values).detach().cpu().numpy()
        
        # Correctly create a list of scalar values for GAE calculation
        value_scalars = list(values_np) + [last_value_scalar]

        for t in reversed(range(len(self.rewards))):
            delta = (
                self.rewards[t]
                + gamma * value_scalars[t + 1] * (1 - self.dones[t])
                - value_scalars[t]
            )
            gae = delta + gamma * lam * (1 - self.dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values_np
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)

        return torch.tensor(advantages).to(self.device), torch.tensor(returns).to(self.device)
