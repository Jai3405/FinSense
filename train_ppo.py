# train_ppo.py
# Main training loop for the PPO agent.

import torch
from agents.ppo_agent import PPOAgent
from agents.ppo_memory import PPOMemory
from agents.ppo_trainer import PPOTrainer
from environment import TradingEnvironment
from utils import load_config, setup_logger
from data_loader import DataLoader
import numpy as np

def train_ppo():
    """Main PPO training loop."""
    config = load_config('config.yaml')
    logger = setup_logger('ppo_train', 'logs/ppo_training.log')
    
    # Config sections
    data_config = config.get_section('data')
    ppo_config = config.get_section('ppo')
    env_config = config.get_section('environment')

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Loading
    data_loader = DataLoader(data_config)
    data = data_loader.load_data()
    train_data, val_data, _ = data_loader.train_test_split(data)

    env = TradingEnvironment(train_data, env_config)
    
    # Get state size dynamically
    sample_state = env.reset()
    # This is not right, reset() returns an int. We need to get a real state.
    # Let's get it manually for now.
    sample_state = env.get_state() 
    # The state from get_state is a dict. The input to the network is the feature vector.
    # We need to call get_state_with_features
    from utils.features import get_state_with_features
    window_size = env_config.get('window_size', 20)
    sample_feature_state = get_state_with_features(train_data, window_size, window_size, env_config)
    state_size = sample_feature_state.shape[1]
    
    action_size = env_config.get('action_size', 3)
    
    # Agent and Trainer
    agent = PPOAgent(state_size, action_size).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=ppo_config.get('lr', 0.0003))
    trainer = PPOTrainer(agent, optimizer, ppo_config)
    
    # Memory
    memory = PPOMemory(batch_size=ppo_config.get('batch_size', 64), device=device)

    logger.info("Starting PPO Training...")

    for episode in range(config.get('training.episodes', 100)):
        env.reset()
        state = get_state_with_features(train_data, env.current_step, window_size, env_config)
        
        done = False
        episode_reward = 0
        
        # --- Rollout Collection ---
        for t in range(config.get('training.max_steps', 10000)):
            state_tensor = torch.from_numpy(state).float().to(device)
            action_mask = env.get_action_mask()
            
            action, log_prob, _, value = agent.act(state_tensor, action_mask)
            
            reward, done, info = env.step(action)
            next_state = get_state_with_features(train_data, env.current_step, window_size, env_config)

            memory.add(state_tensor.cpu(), action, log_prob.detach().cpu(), reward, done, value.detach().cpu(), action_mask)
            
            state = next_state
            episode_reward += reward
            
            # PPO update is typically done after a certain number of steps (e.g., 2048)
            # For this simpler loop, we'll update when the episode is done.
            if done:
                break
        
        # --- PPO Update ---
        # Get value of the last state for GAE calculation
        last_state_tensor = torch.from_numpy(next_state).float().to(device)
        _, last_value = agent(last_state_tensor.unsqueeze(0))
        
        advantages, returns = memory.compute_gae(
            last_value.item(), 
            gamma=ppo_config.get('gamma', 0.99), 
            lam=ppo_config.get('gae_lambda', 0.95)
        )
        trainer.update(memory, advantages, returns)
        memory.clear()

        logger.info(f"Episode {episode+1} | Total Reward: {episode_reward:.2f}")

    # --- Save the final trained model ---
    save_path = "models/ppo_final.pt"
    torch.save(agent.state_dict(), save_path)
    logger.info(f"PPO Training Complete! Final model saved to {save_path}")

if __name__ == '__main__':
    train_ppo()
