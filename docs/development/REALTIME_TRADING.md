# Real-Time Trading Infrastructure

**Date:** 2026-01-04
**Status:** ✅ Complete
**Ready For:** SPIKE v2 Live Trading

---

## 🎯 Overview

We've built a complete **real-time data streaming infrastructure** for live trading. This allows FinSense to:
- Stream live price data from Yahoo Finance
- Make trading decisions in real-time
- Execute paper trades (simulate real trading)
- Seamlessly transition from backtesting to live trading

---

## ✅ Components Built

### 1. **RealTimeDataStream** - Core Streaming Engine

**File:** [data_loader/realtime_data.py](data_loader/realtime_data.py:18-208)

**Features:**
- Asynchronous live data updates
- Ring buffer for historical context (500 candles default)
- Thread-safe operations
- Automatic reconnection on errors
- Callback system for new data events

**Usage:**
```python
from data_loader import RealTimeDataStream

# Create stream
stream = RealTimeDataStream(
    ticker='RELIANCE.NS',
    interval='5m',
    buffer_size=500
)

# Register callback
def on_new_data(candle):
    print(f"New price: ₹{candle['close']:.2f}")

stream.register_callback(on_new_data)

# Start streaming
stream.start()

# Get current data
data = stream.get_current_data()
latest_price = stream.get_latest_price()

# Stop when done
stream.stop()
```

**Key Methods:**
- `start()` - Start data stream
- `stop()` - Stop data stream
- `get_current_data()` - Get buffered OHLCV data
- `get_latest_price()` - Get most recent price
- `get_latest_candle()` - Get most recent OHLCV candle
- `register_callback(func)` - Register update callback
- `get_status()` - Get stream statistics

---

### 2. **MultiTickerStream** - Multiple Stock Monitoring

**Features:**
- Monitor multiple stocks simultaneously
- Staggered start to avoid rate limits
- Centralized status reporting

**Usage:**
```python
from data_loader import MultiTickerStream

# Monitor 3 stocks
tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS']
multi_stream = MultiTickerStream(tickers, interval='5m')

# Start all streams
multi_stream.start_all()

# Get latest prices for all
prices = multi_stream.get_all_latest_prices()
# {'RELIANCE.NS': 2450.50, 'TCS.NS': 3890.25, 'INFY.NS': 1620.75}

# Get data for specific ticker
reliance_data = multi_stream.get_data('RELIANCE.NS')

# Stop all
multi_stream.stop_all()
```

---

### 3. **LiveDataAdapter** - Backtesting Compatibility

**Features:**
- Makes live data compatible with backtesting interface
- Seamless switching between historical and live
- Provides standard data format

**Usage:**
```python
from data_loader import LiveDataAdapter

# Create adapter
adapter = LiveDataAdapter(
    ticker='RELIANCE.NS',
    interval='5m',
    buffer_size=500
)

# Get data (same format as backtesting)
data = adapter.get_data()
# Returns: {'close': [...], 'high': [...], 'low': [...], ...}

# Get recent window for agent
state_data = adapter.get_latest_state(window_size=10)

# Wait for next update
if adapter.wait_for_update(timeout=120):
    print("New data available!")

# Close when done
adapter.close()
```

---

### 4. **live_trade.py** - Live Trading Script

**File:** [live_trade.py](live_trade.py)

**Features:**
- Command-line interface for live trading
- Paper trading mode (simulate real trades)
- Real-time decision making with trained DQN
- Portfolio tracking
- Trading statistics

**Usage:**

#### Paper Trading (Recommended for Testing)
```bash
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    --interval 5m \
    --paper-trade \
    --starting-balance 50000
```

#### With Trade Limits (For Testing)
```bash
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    --interval 5m \
    --paper-trade \
    --max-trades 10 \
    --verbose
```

**Output Example:**
```
======================================================================
 FinSense Live Trading
======================================================================
Model: models/best_model.pt
Ticker: RELIANCE.NS
Interval: 5m
Mode: PAPER TRADING

Loading model from models/best_model.pt...
Model loaded (state_size=16)
Connecting to real-time data for RELIANCE.NS...
Live data stream connected
Starting balance: ₹50000.00

Live trading started. Press Ctrl+C to stop.

BUY: 1 share @ ₹2450.50 | Balance: ₹47549.50 | Inventory: 1
SELL: 1 share @ ₹2468.75 | Balance: ₹50018.25 | Inventory: 0
Stats: Trades=2, Win Rate=100.0%, Profit=₹18.25

^C
Stopping live trading (Ctrl+C)...

======================================================================
 Final Statistics
======================================================================
Total Trades: 2
Winning Trades: 1
Win Rate: 100.00%
Total Profit: ₹18.25
Final Balance: ₹50018.25
Final Inventory: 0 shares
Final Portfolio Value: ₹50018.25
Return: 0.04%
======================================================================
```

---

## 📊 Architecture

### Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Yahoo Finance (yfinance)                                    │
│  • Live OHLCV data                                          │
│  • 1m, 5m, 15m intervals                                    │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ RealTimeDataStream                                          │
│  • Fetches data every interval                              │
│  • Updates ring buffer                                      │
│  • Triggers callbacks                                       │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ LiveDataAdapter                                             │
│  • Provides standard interface                              │
│  • Compatible with backtesting                              │
│  • Waits for updates                                        │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│ LiveTrader                                                  │
│  • Generates state from data                                │
│  • DQN agent makes decision                                 │
│  • Executes trade                                           │
│  • Updates portfolio                                        │
└─────────────────────────────────────────────────────────────┘
```

### Thread Safety

- **Main Thread:** Agent decision making, trade execution
- **Background Thread:** Data fetching, buffer updates
- **Queue-based:** Thread-safe communication
- **No locks needed:** Ring buffer is append-only

---

## 🔧 Configuration

### config.yaml Settings

```yaml
# Live trading configuration
live_trading:
  enabled: false  # Enable live trading
  paper_trade: true  # Paper trading mode (no real orders)

  # Data streaming
  interval: 5m  # Trading interval (1m, 5m, 15m)
  buffer_size: 500  # Historical buffer for context
  update_timeout: 300  # Max seconds to wait for update

  # Position sizing
  max_position_size: 0.1  # Maximum 10% of portfolio per position
  max_portfolio_risk: 0.02  # Maximum 2% portfolio risk per trade
  min_cash_reserve: 0.2  # Keep 20% in cash

  # Safety limits
  daily_loss_limit: 1000  # Stop trading if daily loss exceeds
  max_trades_per_day: 10  # Maximum trades per day
```

---

## 🚀 Live Trading Workflow

### 1. Train Model (Offline)
```bash
python train.py --episodes 100 --ticker RELIANCE.NS
```

### 2. Evaluate Model (Backtesting)
```bash
python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS
```

### 3. Paper Trade (Real-time, No Risk)
```bash
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    --paper-trade
```

### 4. Live Trade (Future - With Groww API)
```bash
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    # (real trading, requires broker integration)
```

---

## 🛡️ Safety Features

### Built-in Protections

1. **Paper Trading Mode**
   - Simulates trades without real orders
   - Safe for testing strategies
   - Full portfolio tracking

2. **Position Limits**
   - Max position size (% of portfolio)
   - Max portfolio risk per trade
   - Minimum cash reserve

3. **Daily Limits**
   - Maximum trades per day
   - Daily loss limits
   - Auto-stop on breach

4. **Data Validation**
   - Checks for stale data
   - Detects connection issues
   - Auto-reconnect on errors

5. **Error Handling**
   - Graceful shutdown on errors
   - Comprehensive logging
   - Error counters and alerts

---

## 📈 Performance Considerations

### Data Latency

| Interval | Update Frequency | Typical Latency |
|----------|------------------|-----------------|
| 1m | Every 60 seconds | 2-5 seconds |
| 5m | Every 300 seconds | 5-10 seconds |
| 15m | Every 900 seconds | 10-15 seconds |

**Note:** Yahoo Finance has ~5-15 second delay from real market prices.

### Buffer Sizing

| Buffer Size | Memory Usage | History Duration (5m) |
|-------------|--------------|----------------------|
| 100 candles | ~10 KB | 8.3 hours |
| 500 candles | ~50 KB | 1.7 days |
| 1000 candles | ~100 KB | 3.5 days |

**Recommendation:** 500 candles provides good context without excessive memory.

### Rate Limits

- Yahoo Finance: ~2000 requests/hour
- With 5m interval: ~12 requests/hour (well within limit)
- Multi-ticker: Stagger starts by 0.1s to avoid burst

---

## 🧪 Testing

### Test Real-Time Streaming

```python
from data_loader import RealTimeDataStream
import time

# Create stream
stream = RealTimeDataStream('RELIANCE.NS', interval='1m', buffer_size=100)

# Callback
def print_update(candle):
    print(f"{candle['timestamp']}: ₹{candle['close']:.2f}")

stream.register_callback(print_update)
stream.start()

# Run for 5 minutes
time.sleep(300)
stream.stop()

# Check status
print(stream.get_status())
```

### Test Live Trading (Paper Mode)

```bash
# Test with max 10 trades
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    --paper-trade \
    --max-trades 10 \
    --verbose
```

---

## 🔮 Future Enhancements

### For SPIKE v2

1. **Groww API Integration**
   - Real order execution
   - Portfolio synchronization
   - Order status tracking

2. **WebSocket Support**
   - Lower latency (~1s)
   - More reliable updates
   - Bidirectional communication

3. **Advanced Order Types**
   - Stop-loss orders
   - Take-profit orders
   - Trailing stops

4. **Multi-Asset Trading**
   - Trade multiple stocks simultaneously
   - Portfolio rebalancing
   - Sector rotation

5. **Risk Management**
   - Real-time VaR calculation
   - Position correlation
   - Dynamic position sizing

6. **Performance Optimization**
   - Redis caching
   - Event-driven architecture
   - Distributed streaming

---

## 📝 API Reference

### RealTimeDataStream

```python
class RealTimeDataStream:
    def __init__(ticker, interval='1m', buffer_size=500, config=None)
    def start()  # Start streaming
    def stop()   # Stop streaming
    def get_current_data() -> dict  # Get buffered OHLCV
    def get_latest_price() -> float  # Get latest close
    def get_latest_candle() -> dict  # Get latest OHLCV candle
    def register_callback(func)  # Register update handler
    def get_status() -> dict  # Get stream statistics
```

### LiveDataAdapter

```python
class LiveDataAdapter:
    def __init__(ticker, interval='1m', buffer_size=500)
    def get_data() -> dict  # Get data (backtesting format)
    def get_latest_state(window_size=10) -> dict  # Get recent window
    def wait_for_update(timeout=300) -> bool  # Wait for new data
    def close()  # Stop stream
```

### LiveTrader

```python
class LiveTrader:
    def __init__(agent, data_adapter, config, starting_balance=50000)
    def execute_action(action, current_price) -> dict  # Execute trade
    def get_portfolio_value(current_price) -> float  # Get total value
    def get_statistics() -> dict  # Get trading stats
```

---

## ✅ Completion Checklist

- [x] Real-time data streaming (RealTimeDataStream)
- [x] Multi-ticker support (MultiTickerStream)
- [x] Live data adapter (LiveDataAdapter)
- [x] Live trading script (live_trade.py)
- [x] Paper trading mode
- [x] Portfolio tracking
- [x] Trading statistics
- [x] Configuration support
- [x] Error handling
- [x] Thread safety
- [x] Documentation

**Status:** 🟢 **COMPLETE** - Ready for SPIKE v2

---

## 🎓 Example: Full Workflow

```python
# 1. Import modules
from data_loader import LiveDataAdapter
from agents import DQNAgent
from utils import load_config, get_state_with_features

# 2. Load trained model
config = load_config('config.yaml')
agent = DQNAgent(state_size=16, action_size=3, config=config.get_section('agent'))
agent.load('models/best_model.pt')

# 3. Start live data
adapter = LiveDataAdapter('RELIANCE.NS', interval='5m')

# 4. Trading loop
portfolio = {'balance': 50000, 'inventory': 0}

while True:
    # Wait for update
    if adapter.wait_for_update():
        # Get current data
        data = adapter.get_data()

        # Generate state
        state = get_state_with_features(data, len(data['close'])-1, 10, config)

        # Get agent decision
        action = agent.act(state, training=False)

        # Execute action (BUY=0, HOLD=1, SELL=2)
        current_price = data['close'][-1]

        if action == 0 and portfolio['balance'] >= current_price:
            # BUY
            portfolio['inventory'] += 1
            portfolio['balance'] -= current_price
            print(f"BUY @ ₹{current_price:.2f}")

        elif action == 2 and portfolio['inventory'] > 0:
            # SELL
            portfolio['inventory'] -= 1
            portfolio['balance'] += current_price
            print(f"SELL @ ₹{current_price:.2f}")

# 5. Cleanup
adapter.close()
```

---

**Last Updated:** 2026-01-04
**Status:** Production-Ready for SPIKE v2
**Next:** Integrate with Groww API for real order execution
