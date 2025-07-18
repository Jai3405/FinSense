import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import yfinance as yf
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import datetime
from pathlib import Path
import argparse

# PyTorch Agent Implementation
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class Agent:
    def __init__(self, state_size, action_size=3, lr=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.inventory = []
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    
    def expReplay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        # Convert to numpy arrays first, then to tensors
        states = np.vstack([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch], dtype=np.float32)
        next_states = np.vstack([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.q_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (0.95 * next_q_values * ~dones)
        
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, filepath):
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
    
    def load_model(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

# Helper functions
def getState(data, t, n):
    """Get state representation for time t with window size n"""
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else np.concatenate([np.repeat(data[0], -d), data[0:t + 1]])
    res = []
    for i in range(n - 1):
        diff = block[i + 1] - block[i]
        if hasattr(diff, 'item'):
            res.append(diff.item())
        else:
            res.append(float(diff))
    return np.array(res, dtype=np.float32)

def formatPrice(n):
    """Format price for display"""
    if isinstance(n, np.ndarray):
        n = n.item()
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def get_data_by_date(ticker, interval, target_date):
    """Get data for a specific date"""
    target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    end_date = target_date + datetime.timedelta(days=1)
    
    data_df = yf.download(tickers=ticker, interval=interval, start=target_date, end=end_date)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    
    if target_date not in unique_dates:
        raise ValueError(f"Target date {target_date} not found in data.")
    
    return data_df[data_df['Date'] == target_date]['Close'].dropna().values

def train_agent(ticker, interval, train_date, window_size, episode_count, batch_size, starting_balance):
    """Train the agent on specified date"""
    print(f"Training on {train_date}")
    train_data = get_data_by_date(ticker, interval, train_date)
    print(f"Training samples: {len(train_data)}")
    
    # Initialize agent
    agent = Agent(window_size)
    l = len(train_data) - 1
    
    # Precompute states
    states = [getState(train_data, t, window_size + 1) for t in range(window_size, len(train_data))]
    
    # Create models folder
    os.makedirs("models", exist_ok=True)
    
    # Tracking
    reward_history = []
    profit_history = []
    
    for e in range(episode_count + 1):
        print(f"Episode {e}/{episode_count}")
        balance = starting_balance
        state = states[0]
        
        total_profit = 0
        total_reward = 0
        agent.inventory = []
        
        for t in range(window_size, l):
            action = agent.act(state)
            next_state = states[t + 1 - window_size]
            reward = 0
            
            if action == 1 and balance >= train_data[t]:  # Buy
                price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
                agent.inventory.append(price)
                balance -= price
                print(f"Buy: {formatPrice(price)} | Balance: {formatPrice(balance)}")
                
            elif action == 2 and len(agent.inventory) > 0:  # Sell
                price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
                bought_price = agent.inventory.pop(0)
                reward = max(price - bought_price, -1)
                total_profit += price - bought_price
                balance += price
                print(f"Sell: {formatPrice(price)} | Profit: {formatPrice(price - bought_price)} | Balance: {formatPrice(balance)}")
            
            total_reward += reward
            done = True if t == l - 1 else False
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                print(f"Episode {e} - Balance: {formatPrice(balance)} | Profit: {formatPrice(total_profit)} | Reward: {formatPrice(total_reward)}")
                print("--------------------------------")
            
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
        
        reward_history.append(total_reward)
        profit_history.append(total_profit)
        
        # Save model every episode
        agent.save_model(f"models/model_ep{e}.pth")
    
    return reward_history, profit_history

def evaluate_single_model(model_path, ticker, interval, eval_date, window_size, start_balance):
    """Evaluate a single model on specified date"""
    eval_data = get_data_by_date(ticker, interval, eval_date)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = DQNNetwork(window_size, 3).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    l = len(eval_data) - 1
    state = getState(eval_data, window_size, window_size + 1)
    
    balance = start_balance
    inventory = []
    total_profit = 0
    buy_count = 0
    sell_count = 0
    profit_over_time = []
    
    for t in range(window_size, l):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state = getState(eval_data, t + 1, window_size + 1)
        price = eval_data[t].item() if hasattr(eval_data[t], 'item') else float(eval_data[t])
        
        if action == 1 and balance >= price:  # Buy
            inventory.append(price)
            balance -= price
            buy_count += 1
        elif action == 2 and len(inventory) > 0:  # Sell
            bought_price = inventory.pop(0)
            profit = price - bought_price
            total_profit += profit
            balance += price
            sell_count += 1
        
        state = next_state
        profit_over_time.append(total_profit)
    
    final_inventory_value = sum(inventory) if inventory else 0
    final_portfolio_value = balance + final_inventory_value
    final_profit = final_portfolio_value - start_balance
    
    return {
        'final_profit': final_profit,
        'total_profit': total_profit,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'profit_over_time': profit_over_time
    }

def evaluate_all_models(ticker, interval, eval_date, window_size, start_balance, episode_count):
    """Evaluate all models generated during training"""
    print(f"Evaluating all models on {eval_date}")
    
    results = []
    model_numbers = []
    
    for episode in range(episode_count + 1):
        model_path = f"models/model_ep{episode}.pth"
        if os.path.exists(model_path):
            try:
                result = evaluate_single_model(model_path, ticker, interval, eval_date, window_size, start_balance)
                results.append(result)
                model_numbers.append(episode)
                print(f"Episode {episode}: Profit = {formatPrice(result['final_profit'])}")
            except Exception as e:
                print(f"Error evaluating episode {episode}: {e}")
    
    return results, model_numbers

def plot_training_and_evaluation(reward_history, profit_history, eval_results, model_numbers):
    """Plot training progress and evaluation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training progress
    ax1.plot(profit_history, label="Training Profit", color="green")
    ax1.plot(reward_history, label="Training Reward", color="blue")
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Value")
    ax1.set_title("Training Progress")
    ax1.grid(True)
    ax1.legend()
    
    # Evaluation profits
    eval_profits = [r['final_profit'] for r in eval_results]
    ax2.plot(model_numbers, eval_profits, 'o-', color="red", label="Evaluation Profit")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Profit ($)")
    ax2.set_title("Evaluation Profit per Episode")
    ax2.grid(True)
    ax2.legend()
    
    # Buy/Sell counts
    buy_counts = [r['buy_count'] for r in eval_results]
    sell_counts = [r['sell_count'] for r in eval_results]
    ax3.plot(model_numbers, buy_counts, 'o-', color="green", label="Buy Count")
    ax3.plot(model_numbers, sell_counts, 'o-', color="red", label="Sell Count")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Count")
    ax3.set_title("Trading Activity per Episode")
    ax3.grid(True)
    ax3.legend()
    
    # Best model profit over time
    best_idx = np.argmax(eval_profits)
    best_model_profit = eval_results[best_idx]['profit_over_time']
    ax4.plot(best_model_profit, color="purple", label=f"Best Model (Ep {model_numbers[best_idx]})")
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Cumulative Profit ($)")
    ax4.set_title("Best Model Profit Over Time")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig("unified_training_evaluation_results.png", dpi=300, bbox_inches='tight')
    plt.show()


def validate_date(date_str):
    """Validate date format"""
    try:
        datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main():

    STOCK_TICKER = 'WIPRO.NS'
    TRAIN_DATE = '2025-06-02'
    EVAL_DATE = '2025-06-27'
    INTERVAL = '15m'
    WINDOW_SIZE = 7
    EPISODES = 25
    BATCH_SIZE = 32
    STARTING_BALANCE = 1000

    
    # Validate dates
    if not validate_date(TRAIN_DATE):
        print(f"❌ Error: Invalid training date format '{TRAIN_DATE}'. Please use YYYY-MM-DD format.")
        return
    
    if not validate_date(EVAL_DATE):
        print(f"❌ Error: Invalid evaluation date format '{EVAL_DATE}'. Please use YYYY-MM-DD format.")
        return
    
    # Check if eval_date is after train_date
    train_dt = datetime.datetime.strptime(TRAIN_DATE, "%Y-%m-%d")
    eval_dt = datetime.datetime.strptime(EVAL_DATE, "%Y-%m-%d")
    if eval_dt <= train_dt:
        print("❌ Error: Evaluation date must be after training date.")
        return
    
    print("=" * 50)
    print("UNIFIED TRAINING AND EVALUATION")
    print("=" * 50)
    print(f"Ticker: {STOCK_TICKER}")
    print(f"Interval: {INTERVAL}")
    print(f"Training Date: {TRAIN_DATE}")
    print(f"Evaluation Date: {EVAL_DATE}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Episodes: {EPISODES}")
    print(f"Starting Balance: {formatPrice(STARTING_BALANCE)}")
    print("=" * 50)
    
    # Training phase
    print("\n🚀 TRAINING PHASE")
    print("-" * 30)
    reward_history, profit_history = train_agent(
        STOCK_TICKER, INTERVAL, TRAIN_DATE, 
        WINDOW_SIZE, EPISODES, BATCH_SIZE, STARTING_BALANCE
    )
    
    # Evaluation phase
    print("\n📊 EVALUATION PHASE")
    print("-" * 30)
    eval_results, model_numbers = evaluate_all_models(
        STOCK_TICKER, INTERVAL, EVAL_DATE,
        WINDOW_SIZE, STARTING_BALANCE, EPISODES
    )
    
    # Results summary
    print("\n📈 RESULTS SUMMARY")
    print("-" * 30)
    best_idx = np.argmax([r['final_profit'] for r in eval_results])
    best_episode = model_numbers[best_idx]
    best_profit = eval_results[best_idx]['final_profit']
    
    print(f"Best Model: Episode {best_episode}")
    print(f"Best Profit: {formatPrice(best_profit)}")
    print(f"Best Buy Count: {eval_results[best_idx]['buy_count']}")
    print(f"Best Sell Count: {eval_results[best_idx]['sell_count']}")
    
    # Plot results
    plot_training_and_evaluation(reward_history, profit_history, eval_results, model_numbers)
    
    print("\n✅ Analysis complete! Check 'unified_training_evaluation_results.png' for plots.")

if __name__ == "__main__":
    main()