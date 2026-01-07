"""Check Q-values on test set to understand policy behavior"""
import torch
import numpy as np
from agents import DQNAgent
from data_loader import DataLoader
from environment import TradingEnvironment
from utils import load_config, get_state_with_features

config = load_config("config.yaml")
data_config = config.get_section("data")
env_config = config.get_section("environment")

print("Loading data...")
loader = DataLoader(data_config)
data = loader.load_data()

train_ratio = config.get('training.train_ratio', 0.7)
val_ratio = config.get('training.validation_ratio', 0.15)
train_data, val_data, test_data = loader.train_test_split(data, train_ratio, val_ratio)

# Calculate state size from config to match the trained model
window_size = env_config.get('window_size', 20)
state_size = window_size - 1
if config.get('environment.use_volume', True): state_size += 1
if config.get('environment.use_technical_indicators', True):
    state_size += 9  # RSI(1) + MACD(3) + BB(1) + ATR(1) + Trend(3)

# Load agent
agent = DQNAgent(state_size=state_size, action_size=3, config=config.get_section("agent"))
agent.load("models/final_model.pt")
agent.epsilon = 0.0

window_size = env_config.get("window_size", 20)

# Sample 10 random points from test set
test_indices = np.random.choice(range(window_size, len(test_data['close'])), 10, replace=False)

print("\nQ-Values at 10 random test set points:")
print("="*70)

q_values_all = {'buy': [], 'hold': [], 'sell': []}

for i, t in enumerate(test_indices):
    state = get_state_with_features(test_data, t, window_size, env_config)

    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_vals = agent.q_network(state_tensor).squeeze().cpu().numpy()

    q_values_all['buy'].append(q_vals[0])
    q_values_all['hold'].append(q_vals[1])
    q_values_all['sell'].append(q_vals[2])

    print(f"\nPoint {i+1} (t={t}):")
    print(f"  Q(Buy):  {q_vals[0]:8.4f}")
    print(f"  Q(Hold): {q_vals[1]:8.4f} {'← HIGHEST' if np.argmax(q_vals) == 1 else ''}")
    print(f"  Q(Sell): {q_vals[2]:8.4f}")
    print(f"  Action: {['Buy', 'Hold', 'Sell'][np.argmax(q_vals)]}")

print("\n" + "="*70)
print("AVERAGE Q-VALUES ACROSS TEST SET:")
print(f"  Q(Buy):  {np.mean(q_values_all['buy']):8.4f}")
print(f"  Q(Hold): {np.mean(q_values_all['hold']):8.4f}")
print(f"  Q(Sell): {np.mean(q_values_all['sell']):8.4f}")
print("="*70)

if np.mean(q_values_all['hold']) > np.mean(q_values_all['buy']) and \
   np.mean(q_values_all['hold']) > np.mean(q_values_all['sell']):
    print("\n🚨 DIAGNOSIS: Q(Hold) dominates - idle penalty still too weak")
    print("   OR: Train/test distribution mismatch is severe")
else:
    print("\n✓ Q-values are balanced - issue may be elsewhere")
