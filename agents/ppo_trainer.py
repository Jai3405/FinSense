# agents/ppo_trainer.py

import torch
import torch.nn.functional as F

class PPOTrainer:
    def __init__(self, agent, optimizer, ppo_config):
        self.agent = agent
        self.optimizer = optimizer
        self.clip_eps = ppo_config.get('clip_eps', 0.2)
        self.value_coef = ppo_config.get('value_coef', 0.5)
        self.entropy_coef = ppo_config.get('entropy_coef', 0.01)

    def update(self, memory, advantages, returns):
        ppo_epochs = 4
        
        # Create full tensors from memory
        states = torch.cat(memory.states).to(advantages.device)
        actions = torch.tensor(memory.actions, dtype=torch.long).to(advantages.device)
        old_log_probs = torch.stack(memory.log_probs).to(advantages.device)
        masks = torch.tensor(memory.masks, dtype=torch.bool).to(advantages.device)
        
        # Get shuffled batches of indices
        batches = memory.generate_batches()
        
        for _ in range(ppo_epochs):
            for batch_indices in batches:
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_masks = masks[batch_indices]

                # Get new log_probs, entropy, and values from the agent
                log_probs, entropy, new_values = self.agent.evaluate_action(
                    batch_states, batch_actions, action_mask=batch_masks
                )
                
                new_values = new_values.squeeze()

                # Ratio of new to old policy
                ratio = (log_probs - batch_old_log_probs).exp()

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * batch_advantages
                
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(new_values, batch_returns)

                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy.mean()
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.agent.parameters(), 0.5)
                self.optimizer.step()
