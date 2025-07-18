import os
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
    return ("-₹" if n < 0 else "₹") + "{0:.2f}".format(abs(n))

def calculate_groww_charges(trade_value):
    """Calculate Groww intraday trading charges"""
    # Brokerage: ₹20 or 0.1% of trade value, whichever is lower
    brokerage = min(20, trade_value * 0.001)
    # Minimum charge: ₹5
    brokerage = max(5, brokerage)
    
    # Statutory charges
    stt = trade_value * 0.001  # 0.1% STT
    exchange_charges = trade_value * 0.0000297  # 0.00297% for NSE
    sebi_charges = trade_value * 0.000001  # 0.0001%
    
    # GST on brokerage
    gst = brokerage * 0.18  # 18% GST on brokerage
    
    total_charges = brokerage + stt + exchange_charges + sebi_charges + gst
    
    return {
        'brokerage': brokerage,
        'stt': stt,
        'exchange_charges': exchange_charges,
        'sebi_charges': sebi_charges,
        'gst': gst,
        'total_charges': total_charges
    }

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

def train_agent(ticker, interval, train_date, window_size, episode_count, batch_size, starting_balance, include_charges=False):
    """Train the agent on specified date"""
    print(f"Training on {train_date} {'with' if include_charges else 'without'} charges")
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
    charges_history = []
    
    for e in range(episode_count + 1):
        print(f"Episode {e}/{episode_count}")
        balance = starting_balance
        state = states[0]
        
        total_profit = 0
        total_charges = 0
        total_reward = 0
        agent.inventory = []
        
        for t in range(window_size, l):
            action = agent.act(state)
            next_state = states[t + 1 - window_size]
            reward = 0
            
            if action == 1:  # Buy
                price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
                
                if include_charges:
                    charges = calculate_groww_charges(price)
                    total_trade_cost = price + charges['total_charges']
                    
                    if balance >= total_trade_cost:
                        agent.inventory.append(price)
                        balance -= total_trade_cost
                        total_charges += charges['total_charges']
                        print(f"Buy: {formatPrice(price)} | Charges: {formatPrice(charges['total_charges'])} | Balance: {formatPrice(balance)}")
                else:
                    if balance >= price:
                        agent.inventory.append(price)
                        balance -= price
                        print(f"Buy: {formatPrice(price)} | Balance: {formatPrice(balance)}")
                
            elif action == 2 and len(agent.inventory) > 0:  # Sell
                price = train_data[t].item() if hasattr(train_data[t], 'item') else float(train_data[t])
                bought_price = agent.inventory.pop(0)
                
                if include_charges:
                    charges = calculate_groww_charges(price)
                    net_proceeds = price - charges['total_charges']
                    balance += net_proceeds
                    total_charges += charges['total_charges']
                    profit = price - bought_price
                    total_profit += profit
                    reward = max(profit - charges['total_charges'], -1)
                    print(f"Sell: {formatPrice(price)} | Charges: {formatPrice(charges['total_charges'])} | Profit: {formatPrice(profit)} | Balance: {formatPrice(balance)}")
                else:
                    balance += price
                    profit = price - bought_price
                    total_profit += profit
                    reward = max(profit, -1)
                    print(f"Sell: {formatPrice(price)} | Profit: {formatPrice(profit)} | Balance: {formatPrice(balance)}")
            
            total_reward += reward
            done = True if t == l - 1 else False
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                net_profit = total_profit - total_charges if include_charges else total_profit
                print(f"Episode {e} - Balance: {formatPrice(balance)} | Gross Profit: {formatPrice(total_profit)} | Charges: {formatPrice(total_charges)} | Net Profit: {formatPrice(net_profit)}")
                print("--------------------------------")
            
            if len(agent.memory) > batch_size:
                agent.expReplay(batch_size)
        
        reward_history.append(total_reward)
        profit_history.append(total_profit)
        charges_history.append(total_charges)
        
        # Save model every episode
        agent.save_model(f"models/model_ep{e}.pth")
    
    return reward_history, profit_history, charges_history

def evaluate_single_model(model_path, ticker, interval, eval_date, window_size, start_balance, include_charges=True):
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
    total_charges = 0
    buy_count = 0
    sell_count = 0
    profit_over_time = []
    net_profit_over_time = []
    
    for t in range(window_size, l):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
            action = torch.argmax(q_values).item()
        
        next_state = getState(eval_data, t + 1, window_size + 1)
        price = eval_data[t].item() if hasattr(eval_data[t], 'item') else float(eval_data[t])
        
        if action == 1:  # Buy
            if include_charges:
                charges = calculate_groww_charges(price)
                total_trade_cost = price + charges['total_charges']
                
                if balance >= total_trade_cost:
                    inventory.append(price)
                    balance -= total_trade_cost
                    total_charges += charges['total_charges']
                    buy_count += 1
            else:
                if balance >= price:
                    inventory.append(price)
                    balance -= price
                    buy_count += 1
                    
        elif action == 2 and len(inventory) > 0:  # Sell
            bought_price = inventory.pop(0)
            
            if include_charges:
                charges = calculate_groww_charges(price)
                net_proceeds = price - charges['total_charges']
                balance += net_proceeds
                total_charges += charges['total_charges']
            else:
                balance += price
                
            profit = price - bought_price
            total_profit += profit
            sell_count += 1
        
        state = next_state
        profit_over_time.append(total_profit)
        net_profit_over_time.append(total_profit - total_charges)
    
    final_inventory_value = sum(inventory) if inventory else 0
    final_portfolio_value = balance + final_inventory_value
    final_profit = final_portfolio_value - start_balance
    net_profit_after_charges = total_profit - total_charges
    
    return {
        'final_profit': final_profit,
        'total_profit': total_profit,
        'net_profit_after_charges': net_profit_after_charges,
        'total_charges': total_charges,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'profit_over_time': profit_over_time,
        'net_profit_over_time': net_profit_over_time
    }

def evaluate_all_models(ticker, interval, eval_date, window_size, start_balance, episode_count, include_charges=True):
    """Evaluate all models generated during training"""
    print(f"Evaluating all models on {eval_date} {'with' if include_charges else 'without'} charges")
    
    results = []
    model_numbers = []
    
    for episode in range(episode_count + 1):
        model_path = f"models/model_ep{episode}.pth"
        if os.path.exists(model_path):
            try:
                result = evaluate_single_model(model_path, ticker, interval, eval_date, window_size, start_balance, include_charges)
                results.append(result)
                model_numbers.append(episode)
                if include_charges:
                    print(f"Episode {episode}: Net Profit = {formatPrice(result['net_profit_after_charges'])}, Charges = {formatPrice(result['total_charges'])}")
                else:
                    print(f"Episode {episode}: Profit = {formatPrice(result['final_profit'])}")
            except Exception as e:
                print(f"Error evaluating episode {episode}: {e}")
    
    return results, model_numbers

def plot_training_and_evaluation(reward_history, profit_history, charges_history, eval_results, model_numbers, include_charges=True):
    """Plot training progress and evaluation results"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training progress
    ax1.plot(profit_history, label="Gross Training Profit", color="green")
    if include_charges:
        net_profit = [p - c for p, c in zip(profit_history, charges_history)]
        ax1.plot(net_profit, label="Net Training Profit", color="darkgreen", linestyle="--")
        ax1.plot(charges_history, label="Training Charges", color="red", alpha=0.7)
    ax1.plot(reward_history, label="Training Reward", color="blue", alpha=0.7)
    ax1.set_xlabel("Episodes")
    ax1.set_ylabel("Value (₹)")
    ax1.set_title("Training Progress")
    ax1.grid(True)
    ax1.legend()
    
    # Evaluation profits
    if include_charges:
        eval_profits = [r['net_profit_after_charges'] for r in eval_results]
        eval_charges = [r['total_charges'] for r in eval_results]
        ax2.plot(model_numbers, eval_profits, 'o-', color="darkgreen", label="Net Evaluation Profit")
        ax2.plot(model_numbers, eval_charges, 'o-', color="red", alpha=0.7, label="Evaluation Charges")
        ax2.set_title("Net Evaluation Profit per Episode")
    else:
        eval_profits = [r['final_profit'] for r in eval_results]
        ax2.plot(model_numbers, eval_profits, 'o-', color="red", label="Evaluation Profit")
        ax2.set_title("Evaluation Profit per Episode")
    
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Profit (₹)")
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
    best_model_gross_profit = eval_results[best_idx]['profit_over_time']
    ax4.plot(best_model_gross_profit, color="green", alpha=0.7, label=f"Gross Profit (Ep {model_numbers[best_idx]})")
    
    if include_charges:
        best_model_net_profit = eval_results[best_idx]['net_profit_over_time']
        ax4.plot(best_model_net_profit, color="darkgreen", label=f"Net Profit (Ep {model_numbers[best_idx]})")
    
    ax4.set_xlabel("Time Steps")
    ax4.set_ylabel("Cumulative Profit (₹)")
    ax4.set_title("Best Model Profit Over Time")
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    suffix = "_with_charges" if include_charges else "_without_charges"
    plt.savefig(f"unified_training_evaluation_results{suffix}.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Unified Training and Evaluation with Groww Charges')
    parser.add_argument('--ticker', default='WIPRO.NS', help='Stock ticker')
    parser.add_argument('--interval', default='5m', help='Data interval')
    parser.add_argument('--train-date', required=True, help='Training date (YYYY-MM-DD)')
    parser.add_argument('--eval-date', required=True, help='Evaluation date (YYYY-MM-DD)')
    parser.add_argument('--window-size', type=int, default=7, help='State window size')
    parser.add_argument('--episodes', type=int, default=25, help='Training episodes')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--balance', type=float, default=2000, help='Starting balance')
    parser.add_argument('--include-charges', action='store_true', help='Include Groww brokerage charges')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("UNIFIED TRAINING AND EVALUATION WITH GROWW CHARGES")
    print("=" * 60)
    print(f"Ticker: {args.ticker}")
    print(f"Interval: {args.interval}")
    print(f"Training Date: {args.train_date}")
    print(f"Evaluation Date: {args.eval_date}")
    print(f"Window Size: {args.window_size}")
    print(f"Episodes: {args.episodes}")
    print(f"Starting Balance: {formatPrice(args.balance)}")
    print(f"Include Charges: {'Yes' if args.include_charges else 'No'}")
    print("=" * 60)
    
    # Training phase
    print("\n🚀 TRAINING PHASE")
    print("-" * 30)
    reward_history, profit_history, charges_history = train_agent(
        args.ticker, args.interval, args.train_date, 
        args.window_size, args.episodes, args.batch_size, args.balance, args.include_charges
    )
    
    # Evaluation phase
    print("\n📊 EVALUATION PHASE")
    print("-" * 30)
    eval_results, model_numbers = evaluate_all_models(
        args.ticker, args.interval, args.eval_date,
        args.window_size, args.balance, args.episodes, args.include_charges
    )
    
    # Results summary
    print("\n📈 RESULTS SUMMARY")
    print("-" * 30)
    
    if args.include_charges:
        best_idx = np.argmax([r['net_profit_after_charges'] for r in eval_results])
        best_episode = model_numbers[best_idx]
        best_profit = eval_results[best_idx]['net_profit_after_charges']
        best_charges = eval_results[best_idx]['total_charges']
        
        print(f"Best Model: Episode {best_episode}")
        print(f"Best Net Profit: {formatPrice(best_profit)}")
        print(f"Total Charges: {formatPrice(best_charges)}")
        print(f"Best Buy Count: {eval_results[best_idx]['buy_count']}")
        print(f"Best Sell Count: {eval_results[best_idx]['sell_count']}")
        
        # Training summary
        total_training_charges = sum(charges_history)
        total_training_profit = sum(profit_history)
        print(f"\nTraining Summary:")
        print(f"Total Training Charges: {formatPrice(total_training_charges)}")
        print(f"Total Training Profit: {formatPrice(total_training_profit)}")
        print(f"Net Training Profit: {formatPrice(total_training_profit - total_training_charges)}")
        
    else:
        best_idx = np.argmax([r['final_profit'] for r in eval_results])
        best_episode = model_numbers[best_idx]
        best_profit = eval_results[best_idx]['final_profit']
        
        print(f"Best Model: Episode {best_episode}")
        print(f"Best Profit: {formatPrice(best_profit)}")
        print(f"Best Buy Count: {eval_results[best_idx]['buy_count']}")
        print(f"Best Sell Count: {eval_results[best_idx]['sell_count']}")
    
    # Plot results
    plot_training_and_evaluation(reward_history, profit_history, charges_history, eval_results, model_numbers, args.include_charges)
    
    suffix = "_with_charges" if args.include_charges else "_without_charges"
    print(f"\n✅ Analysis complete! Check 'unified_training_evaluation_results{suffix}.png' for plots.")

if __name__ == "__main__":
    main()