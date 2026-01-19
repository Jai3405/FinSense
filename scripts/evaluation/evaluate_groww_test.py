import datetime
import time
import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from growwapi import GrowwAPI
import pyotp

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
    return ("-₹" if n < 0 else "₹") + "{0:.2f}".format(abs(n))

def is_market_open():
    """Check if market is currently open (9:15 AM - 3:30 PM IST on weekdays)"""
    now = datetime.datetime.now()
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    market_start = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    
    return market_start <= now <= market_end

def get_groww_live_price(groww_api, symbol):
    """Get live price from Groww API"""
    try:
        # Get live price data
        live_data = groww_api.get_live_price(symbol)
        if live_data and 'ltp' in live_data:
            return float(live_data['ltp'])
        else:
            print(f"No live price data available for {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching live price for {symbol}: {e}")
        return None

def plot_live_results(results, ticker):
    """Plot live trading results"""
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1])
    
    # Main price chart with buy/sell signals
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(results['timestamps'], results['all_prices'], label='Live Price', color='blue', alpha=0.7)
    
    # Mark buy and sell points
    for i, timestamp in enumerate(results['buy_timestamps']):
        idx = results['timestamps'].index(timestamp) if timestamp in results['timestamps'] else None
        if idx is not None:
            ax1.scatter(timestamp, results['all_prices'][idx], color='green', marker='^', s=100, zorder=5)
    
    for i, timestamp in enumerate(results['sell_timestamps']):
        idx = results['timestamps'].index(timestamp) if timestamp in results['timestamps'] else None
        if idx is not None:
            ax1.scatter(timestamp, results['all_prices'][idx], color='red', marker='v', s=100, zorder=5)
    
    ax1.set_title(f'{ticker} Live Trading Session')
    ax1.set_ylabel('Price (₹)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Profit over time
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(results['timestamps'], results['profit_over_time'], label='Trading Profit', color='green')
    ax2.set_title('Live Profit Tracking')
    ax2.set_ylabel('Profit (₹)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Portfolio value
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(results['timestamps'], results['portfolio_values'], label='Portfolio Value', color='purple')
    ax3.set_title('Live Portfolio Value')
    ax3.set_ylabel('Value (₹)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'live_trading_results_{ticker}_{timestamp_str}.png', dpi=300, bbox_inches='tight')
    plt.show()

# Constants
MODEL_NUMBER = 25
TICKER = "WIPRO"  # Remove .NS for Groww API
START_BALANCE = 2000
VISUALS_DIR = "visuals"
UPDATE_INTERVAL = 60  # seconds between price updates
MAX_RUNTIME = 7200  # 2 hours max runtime

def load_groww_config():
    """Load Groww API configuration from JSON file"""
    config_file = "groww_config.json"
    
    if not Path(config_file).exists():
        print(f"Error: {config_file} not found!")
        print("Please copy groww_config_template.json to groww_config.json and add your API credentials.")
        return None
    
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def initialize_groww_api(config):
    """Initialize Groww API with config"""
    try:
        if config['auth_method'] == 'access_token':
            if config['access_token'] == "YOUR_ACCESS_TOKEN_HERE":
                raise ValueError("Please set your actual access token in groww_config.json")
            return GrowwAPI(config['access_token'])
        
        elif config['auth_method'] == 'totp':
            if config['api_key'] == "YOUR_API_KEY_HERE" or config['api_secret'] == "YOUR_API_SECRET_HERE":
                raise ValueError("Please set your actual API key and secret in groww_config.json")
            
            totp_gen = pyotp.TOTP(config['api_secret'])
            totp = totp_gen.now()
            access_token = GrowwAPI.get_access_token(config['api_key'], totp)
            return GrowwAPI(access_token)
        
        else:
            raise ValueError("Invalid auth_method in config. Use 'access_token' or 'totp'")
    
    except Exception as e:
        print(f"Error initializing Groww API: {e}")
        return None

# Model Setup
Path(VISUALS_DIR).mkdir(exist_ok=True)

model_name = f"model_ep{MODEL_NUMBER}.pth"
model_path = f"models/{model_name}"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load the model checkpoint
try:
    checkpoint = torch.load(model_path, map_location=device)
    window_size = 7
    action_size = 3
    model = DQNNetwork(window_size, action_size).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the model file exists and is properly trained.")
    exit(1)

# Load configuration and initialize Groww API
config = load_groww_config()
if config is None:
    exit(1)

groww = initialize_groww_api(config)
if groww is None:
    exit(1)

print("Groww API initialized successfully")

def live_trading_simulation():
    """Run live trading simulation using real-time Groww data"""
    print(f"Starting live trading simulation for {TICKER}")
    print(f"Update interval: {UPDATE_INTERVAL} seconds")
    print(f"Max runtime: {MAX_RUNTIME} seconds")
    print("=" * 50)
    
    # Initialize tracking variables
    balance = START_BALANCE
    inventory = []
    total_profit = 0
    price_history = []
    timestamps = []
    portfolio_values = []
    profit_over_time = []
    buy_timestamps = []
    sell_timestamps = []
    
    start_time = datetime.datetime.now()
    
    try:
        while True:
            current_time = datetime.datetime.now()
            
            # Check if we've exceeded max runtime
            if (current_time - start_time).total_seconds() > MAX_RUNTIME:
                print("Max runtime reached. Stopping simulation.")
                break
            
            # Check if market is open
            if not is_market_open():
                print(f"Market is closed. Current time: {current_time.strftime('%H:%M:%S')}")
                print("Waiting for market to open (9:15 AM - 3:30 PM IST on weekdays)...")
                time.sleep(300)  # Wait 5 minutes before checking again
                continue
            
            # Get live price
            current_price = get_groww_live_price(groww, TICKER)
            if current_price is None:
                print("Failed to get live price. Retrying in 30 seconds...")
                time.sleep(30)
                continue
            
            # Add to price history
            price_history.append(current_price)
            timestamps.append(current_time)
            
            print(f"[{current_time.strftime('%H:%M:%S')}] {TICKER}: ₹{current_price:.2f}")
            
            # Only make decisions if we have enough history
            if len(price_history) >= window_size + 1:
                # Get current state
                state = getState(np.array(price_history), len(price_history) - 1, window_size + 1)
                
                # Get action from model
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    q_values = model(state_tensor)
                    action = torch.argmax(q_values).item()
                
                # Execute action (simulation only)
                if action == 1 and balance >= current_price:  # Buy
                    inventory.append(current_price)
                    balance -= current_price
                    buy_timestamps.append(current_time)
                    print(f"🟢 BUY: {formatPrice(current_price)} | Balance: {formatPrice(balance)} | Inventory: {len(inventory)}")
                
                elif action == 2 and len(inventory) > 0:  # Sell
                    bought_price = inventory.pop(0)
                    profit = current_price - bought_price
                    total_profit += profit
                    balance += current_price
                    sell_timestamps.append(current_time)
                    print(f"🔴 SELL: {formatPrice(current_price)} | Profit: {formatPrice(profit)} | Balance: {formatPrice(balance)}")
                
                else:  # Hold
                    print(f"⚪ HOLD | Balance: {formatPrice(balance)} | Inventory: {len(inventory)}")
            
            # Calculate portfolio value
            inventory_value = sum(inventory) if inventory else 0
            portfolio_value = balance + inventory_value
            portfolio_values.append(portfolio_value)
            profit_over_time.append(total_profit)
            
            # Print status every 10 updates
            if len(price_history) % 10 == 0:
                print(f"📊 Portfolio Value: {formatPrice(portfolio_value)} | Total Profit: {formatPrice(total_profit)}")
                print("-" * 50)
            
            # Wait for next update
            time.sleep(UPDATE_INTERVAL)
    
    except KeyboardInterrupt:
        print("\nSimulation stopped by user.")
    
    except Exception as e:
        print(f"Error during simulation: {e}")
    
    finally:
        # Final summary
        final_inventory_value = sum(inventory) if inventory else 0
        final_portfolio_value = balance + final_inventory_value
        final_profit = final_portfolio_value - START_BALANCE
        
        print("\n" + "=" * 50)
        print("LIVE TRADING SIMULATION SUMMARY")
        print("=" * 50)
        print(f"Duration: {(datetime.datetime.now() - start_time).total_seconds()/60:.1f} minutes")
        print(f"Price updates: {len(price_history)}")
        print(f"Buy orders: {len(buy_timestamps)}")
        print(f"Sell orders: {len(sell_timestamps)}")
        print(f"Final Portfolio Value: {formatPrice(final_portfolio_value)}")
        print(f"Total Profit: {formatPrice(final_profit)}")
        print(f"Return: {(final_profit/START_BALANCE)*100:.2f}%")
        
        # Plot results if we have data
        if len(price_history) > 0:
            results = {
                'timestamps': timestamps,
                'all_prices': price_history,
                'portfolio_values': portfolio_values,
                'profit_over_time': profit_over_time,
                'buy_timestamps': buy_timestamps,
                'sell_timestamps': sell_timestamps
            }
            plot_live_results(results, TICKER)

if __name__ == "__main__":
    print("Groww Live Trading Simulation")
    print("IMPORTANT: This is a simulation only - no actual trades will be placed")
    print("Make sure you have:")
    print("1. Valid Groww API access token")
    print("2. Trained model file in models/ directory")
    print("3. Stable internet connection")
    print()
    
    # Configuration is already loaded and validated above
    
    response = input("Ready to start live simulation? (y/n): ")
    if response.lower() == 'y':
        live_trading_simulation()
    else:
        print("Simulation cancelled.")