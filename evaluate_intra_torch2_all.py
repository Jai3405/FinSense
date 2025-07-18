import datetime
import os
from pathlib import Path
import glob

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the same classes from training code
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

# Helper functions
def getState(data, t, n):
    """Get state representation for time t with window size n"""
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else np.concatenate([np.repeat(data[0], -d), data[0:t + 1]])
    res = []
    for i in range(n - 1):
        diff = block[i + 1] - block[i]
        # Handle numpy arrays properly
        if hasattr(diff, 'item'):
            res.append(diff.item())
        else:
            res.append(float(diff))
    return np.array(res, dtype=np.float32)

def formatPrice(n):
    """Format price for display"""
    # Convert to scalar if it's a numpy array
    if isinstance(n, np.ndarray):
        n = n.item()
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def evaluate_single_model(model_path, data, window_size, start_balance, device):
    """Evaluate a single model and return performance metrics"""
    try:
        # Load the model checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Create model with correct architecture
        action_size = 3
        model = DQNNetwork(window_size, action_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Evaluation variables
        l = len(data) - 1
        balance = start_balance
        inventory = []
        total_profit = 0
        buy_count = 0
        sell_count = 0
        
        # Buy and hold comparison
        shares_bought = start_balance / data[0] if data[0] > 0 else 1
        buy_and_hold_profit = (data[-1] - data[0]) * shares_bought
        
        for t in range(window_size, l):
            state = getState(data, t, window_size + 1)
            
            # Get action from model
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()
            
            price = data[t].item() if hasattr(data[t], 'item') else float(data[t])
            
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
        
        # Final calculations
        final_inventory_value = sum(inventory) if inventory else 0
        final_portfolio_value = balance + final_inventory_value
        final_profit = final_portfolio_value - start_balance
        
        return {
            "total_profit": total_profit,
            "final_profit": final_profit,
            "final_portfolio_value": final_portfolio_value,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "buy_and_hold_profit": buy_and_hold_profit,
            "success": True
        }
        
    except Exception as e:
        print(f"Error evaluating {model_path}: {e}")
        return {"success": False, "error": str(e)}

def plot_model_comparison(model_results, ticker):
    """Plot comparison of all models"""
    # Extract data for plotting
    model_names = list(model_results.keys())
    final_profits = [model_results[name]["final_profit"] for name in model_names]
    total_profits = [model_results[name]["total_profit"] for name in model_names]
    buy_and_hold_profits = [model_results[name]["buy_and_hold_profit"] for name in model_names]
    buy_counts = [model_results[name]["buy_count"] for name in model_names]
    sell_counts = [model_results[name]["sell_count"] for name in model_names]
    
    # Sort by model number if possible
    try:
        model_numbers = []
        for name in model_names:
            if "model_ep" in name:
                num = int(name.split("model_ep")[1].split(".pth")[0])
                model_numbers.append(num)
            elif "balanced_model_ep" in name:
                num = int(name.split("balanced_model_ep")[1].split(".pth")[0])
                model_numbers.append(num)
            else:
                model_numbers.append(0)
        
        # Sort all lists by model number
        sorted_data = sorted(zip(model_numbers, model_names, final_profits, total_profits, buy_and_hold_profits))
        model_numbers, model_names, final_profits, total_profits, buy_and_hold_profits = zip(*sorted_data)
    except:
        # If sorting fails, use original order
        model_numbers = range(len(model_names))
    
    # Create plots
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2)
    
    # Final profit comparison
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(model_numbers, final_profits, marker='o', label='Final Profit', color='blue')
    ax1.axhline(y=buy_and_hold_profits[0], color='orange', linestyle='--', label='Buy & Hold')
    ax1.set_title('Final Profit vs Buy & Hold')
    ax1.set_xlabel('Model Number')
    ax1.set_ylabel('Final Profit ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Total trading profit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(model_numbers, total_profits, marker='s', label='Trading Profit', color='green')
    ax2.set_title('Total Trading Profit')
    ax2.set_xlabel('Model Number')
    ax2.set_ylabel('Trading Profit ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Portfolio values
    ax3 = fig.add_subplot(gs[1, 0])
    portfolio_values = [2000 + profit for profit in final_profits]
    ax3.plot(model_numbers, portfolio_values, marker='^', label='Portfolio Value', color='purple')
    ax3.axhline(y=2000 + buy_and_hold_profits[0], color='orange', linestyle='--', label='Buy & Hold Value')
    ax3.set_title('Final Portfolio Value')
    ax3.set_xlabel('Model Number')
    ax3.set_ylabel('Portfolio Value ($)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Best vs worst models
    ax4 = fig.add_subplot(gs[1, 1])
    best_idx = np.argmax(final_profits)
    worst_idx = np.argmin(final_profits)
    
    categories = ['Best Model', 'Worst Model', 'Buy & Hold']
    values = [float(final_profits[best_idx]), float(final_profits[worst_idx]), float(buy_and_hold_profits[0])]
    colors = ['green', 'red', 'orange']
    
    bars = ax4.bar(categories, values, color=colors, alpha=0.7)
    ax4.set_title('Performance Comparison')
    ax4.set_ylabel('Final Profit ($)')
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + (max(values) - min(values)) * 0.01,
                f'${value:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'model_comparison_{ticker}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Constants
TICKER = "WIPRO.NS"
START_BALANCE = 2000
INTERVAL = "5m"
VISUALS_DIR = "visuals"
MODELS_DIR = "models"

# Mode selection for data slicing
MODE = "date"  # options: "date", "relative", "yesterday", "today"
TARGET_DATE = "2025-06-20"  # Used if MODE == "date"

TARGET_DATE = datetime.datetime.strptime(TARGET_DATE, "%Y-%m-%d").date()
END_DATE = TARGET_DATE + datetime.timedelta(days=1)

PERIOD = "2d"
SKIP = 1

# Setup
Path(VISUALS_DIR).mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Find all .pth model files
model_files = glob.glob(os.path.join(MODELS_DIR, "*.pth"))
print(f"Found {len(model_files)} .pth model files")

if not model_files:
    print("No .pth files found in models directory!")
    exit(1)

# Data Loading
print(f"Downloading {TICKER} data...")

if MODE == "date":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, start=TARGET_DATE, end=END_DATE)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if TARGET_DATE not in unique_dates:
        raise ValueError(f"Target date {TARGET_DATE} not found in data.")
    data = data_df[data_df['Date'] == TARGET_DATE]['Close'].dropna().values

elif MODE == "relative":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period=PERIOD)
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough days of data to evaluate.")
    data = data_df[data_df['Date'].isin(unique_dates[:-SKIP])]['Close'].dropna().values

elif MODE == "yesterday":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period="2d")
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 2:
        raise ValueError("Not enough data to evaluate yesterday.")
    data = data_df[data_df['Date'] == unique_dates[-2]]['Close'].dropna().values

elif MODE == "today":
    data_df = yf.download(tickers=TICKER, interval=INTERVAL, period="1d")
    data_df['Date'] = data_df.index.date
    unique_dates = sorted(data_df['Date'].unique())
    if len(unique_dates) < 1:
        raise ValueError("No data to evaluate today.")
    data = data_df[data_df['Date'] == unique_dates[-1]]['Close'].dropna().values

else:
    raise ValueError("Invalid MODE selected. Choose 'date', 'relative', 'yesterday' or 'today'.")

print(f"Loaded {len(data)} data points for evaluation")

# Evaluate all models
window_size = 7  # Default window size
model_results = {}

print("\nEvaluating all models...")
print("=" * 60)

for model_file in model_files:
    model_name = os.path.basename(model_file)
    print(f"Evaluating {model_name}...")
    
    result = evaluate_single_model(model_file, data, window_size, START_BALANCE, device)
    
    if result["success"]:
        model_results[model_name] = result
        print(f" {model_name}: Final Profit = {formatPrice(result['final_profit'])}")
    else:
        print(f" {model_name}: Failed - {result['error']}")

print("=" * 60)

# Summary of results
if model_results:
    print(f"\nSUMMARY - Evaluated {len(model_results)} models successfully")
    print("=" * 60)
    
    # Find best and worst performing models
    best_model = max(model_results.items(), key=lambda x: x[1]['final_profit'])
    worst_model = min(model_results.items(), key=lambda x: x[1]['final_profit'])
    
    print(f"Best Model: {best_model[0]}")
    print(f"  Final Profit: {formatPrice(best_model[1]['final_profit'])}")
    print(f"  Trading Profit: {formatPrice(best_model[1]['total_profit'])}")
    print(f"  Buy/Sell Actions: {best_model[1]['buy_count']}/{best_model[1]['sell_count']}")
    
    print(f"\nWorst Model: {worst_model[0]}")
    print(f"  Final Profit: {formatPrice(worst_model[1]['final_profit'])}")
    print(f"  Trading Profit: {formatPrice(worst_model[1]['total_profit'])}")
    print(f"  Buy/Sell Actions: {worst_model[1]['buy_count']}/{worst_model[1]['sell_count']}")
    
    # Buy and hold comparison
    buy_and_hold_profit = list(model_results.values())[0]['buy_and_hold_profit']
    print(f"\nBuy & Hold Profit: {formatPrice(buy_and_hold_profit)}")
    
    # Count models that beat buy and hold
    beat_buy_hold = sum(1 for result in model_results.values() 
                       if result['final_profit'] > buy_and_hold_profit)
    print(f"Models beating Buy & Hold: {beat_buy_hold}/{len(model_results)}")
    
    print("=" * 60)
    
    # Plot comparison
    plot_model_comparison(model_results, TICKER)
    
    # Save detailed results
    with open(f'model_evaluation_results_{TICKER}.txt', 'w') as f:
        f.write(f"Model Evaluation Results for {TICKER}\n")
        f.write("=" * 60 + "\n\n")
        
        for model_name, result in sorted(model_results.items()):
            f.write(f"Model: {model_name}\n")
            f.write(f"  Final Profit: {formatPrice(result['final_profit'])}\n")
            f.write(f"  Trading Profit: {formatPrice(result['total_profit'])}\n")
            f.write(f"  Portfolio Value: {formatPrice(result['final_portfolio_value'])}\n")
            f.write(f"  Buy/Sell Actions: {result['buy_count']}/{result['sell_count']}\n")
            f.write(f"  vs Buy & Hold: {formatPrice(result['final_profit'] - buy_and_hold_profit)}\n\n")
        
        f.write(f"Buy & Hold Profit: {formatPrice(buy_and_hold_profit)}\n")
        f.write(f"Models beating Buy & Hold: {beat_buy_hold}/{len(model_results)}\n")
    
    print(f"Detailed results saved to: model_evaluation_results_{TICKER}.txt")
    
else:
    print("No models were successfully evaluated!")