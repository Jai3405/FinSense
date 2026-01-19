"""Manual evaluation script - as instructed by expert"""
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

# Split data into train/val/test
train_ratio = config.get('training.train_ratio', 0.7)
val_ratio = config.get('training.validation_ratio', 0.15)
train_data, val_data, test_data = loader.train_test_split(data, train_ratio, val_ratio)

print(f"Test set size: {len(test_data['close'])} points")

# Create environment with TEST data
env = TradingEnvironment(test_data, env_config)

# Calculate state size dynamically (must match training configuration)
window_size = env_config.get('window_size', 20)
state_size = window_size - 1  # Price diffs
if env_config.get('use_volume', True):
    state_size += 1
if env_config.get('use_technical_indicators', True):
    state_size += 9  # RSI(1) + MACD(3) + BB(1) + ATR(1) + Trend(3)

print(f"State size: {state_size} features")

# Load agent
agent = DQNAgent(state_size=state_size, action_size=3, config=config.get_section("agent"))
agent.load("models/final_model.pt")
agent.epsilon = 0.0  # Deterministic evaluation

print("Evaluating final_model.pt on TEST SET...")

state = get_state_with_features(test_data, window_size, window_size, env_config)
env.reset()

done = False
trades = 0
actions = {'buy': 0, 'hold': 0, 'sell': 0}

for t in range(window_size, len(test_data['close'])):
    state = get_state_with_features(test_data, t, window_size, env_config)
    action = agent.act(state, training=False)

    # Track actions
    if action == 0:
        actions['buy'] += 1
    elif action == 1:
        actions['hold'] += 1
    elif action == 2:
        actions['sell'] += 1

    reward, done, info = env.step(action)

    if info.get('success', False):
        trades += 1

    if done:
        break

print("\n" + "="*60)
print("EVALUATION COMPLETE")
print("="*60)
print(f"Total trades: {trades}")
print(f"Actions: Buy={actions['buy']}, Hold={actions['hold']}, Sell={actions['sell']}")
print(f"Final balance: ₹{env.balance:.2f}")
print("="*60)
