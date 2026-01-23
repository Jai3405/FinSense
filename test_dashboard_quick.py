"""
Quick test to verify dashboard backend works
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from live_data.streamer import HistoricalDataSimulator
from live_trading.ppo_inference import PPOInference
from live_trading.paper_executor import PaperTradingExecutor
from utils.config import load_config

print("="*60)
print(" SPIKE TERMINAL - QUICK BACKEND TEST")
print("="*60)
print()

try:
    # Load config
    print("1. Loading config...")
    config = load_config('config.yaml')
    print("   ✓ Config loaded")

    # Initialize streamer
    print("2. Initializing data streamer...")
    streamer = HistoricalDataSimulator(
        ticker='RELIANCE.NS',
        start_date='2024-01-01',
        end_date='2024-01-31',
        interval='1d'
    )
    streamer.initialize_buffer(window_size=20)
    print("   ✓ Streamer initialized")

    # Load PPO agent
    print("3. Loading PPO agent...")
    agent = PPOInference('models/ppo_final.pt', config)
    print("   ✓ Agent loaded")

    # Initialize executor
    print("4. Initializing paper trading executor...")
    executor = PaperTradingExecutor(
        starting_balance=50000,
        max_positions=40,
        max_position_value=0.95,
        transaction_costs=True
    )
    print("   ✓ Executor initialized")

    # Run a few steps
    print()
    print("5. Running 3 test steps...")
    print()

    for i in range(3):
        if not streamer.update():
            break

        data = streamer.get_data()
        current_price = streamer.get_latest_price()
        timestamp = streamer.get_latest_timestamp()

        action_mask = executor.get_action_mask(current_price)
        action, action_probs, value = agent.predict(data, action_mask=action_mask)
        action_name = agent.get_action_name(action)

        trade_result = executor.execute_action(action, current_price, timestamp)
        executor.update_equity_curve(timestamp, current_price)

        metrics = executor.get_metrics(current_price)

        print(f"   Step {i+1}:")
        print(f"   - Price: ₹{current_price:.2f}")
        print(f"   - Action: {action_name}")
        print(f"   - AI Confidence: BUY={action_probs[0]*100:.1f}% | HOLD={action_probs[1]*100:.1f}% | SELL={action_probs[2]*100:.1f}%")
        print(f"   - Portfolio: ₹{metrics['portfolio_value']:.2f}")
        print(f"   - Inventory: {metrics['inventory']} shares")
        print(f"   - Avg Cost: ₹{metrics.get('avg_cost', 0):.2f}")
        print(f"   - Unrealized P&L: ₹{metrics.get('unrealized_pnl', 0):.2f}")
        print()

    print("="*60)
    print(" ✓ ALL TESTS PASSED - DASHBOARD SHOULD WORK")
    print("="*60)
    print()
    print("Now run: ./start_dashboard.sh")
    print()

except Exception as e:
    print()
    print("="*60)
    print(" ✗ ERROR DETECTED")
    print("="*60)
    print()
    print(f"Error: {e}")
    print()
    import traceback
    traceback.print_exc()
    print()
    print("Fix this error before running dashboard.")
    print()
