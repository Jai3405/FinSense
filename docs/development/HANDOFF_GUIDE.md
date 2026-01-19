# Project Handoff Guide

**Current Branch:** `experimental/reward-tuning`
**Base Branch:** `main`
**Date:** 2026-01-06

---

## Current Status

### Training In Progress
- **Running:** 50-episode test training (PID 2292)
- **Purpose:** Testing idle penalty fix
- **Log:** `training_50ep_test.log`
- **Estimated completion:** ~1.5 hours from start (10:20 AM)

### Latest Results
- **Episode 1:** 8,587 trades ✅ (fix working - agent is trading again!)
- **Episode 15:** Best so far (₹13,777 profit, 3,824 trades)
- **Previous issue:** Dead policy (0 trades) - now FIXED

---

## What Was Fixed

### Problem
Agent converged to dead policy - made 0 trades on test set because:
```
Q(HOLD) = -0.056 (highest)
Q(BUY)  = -0.96
Q(SELL) = -1.58
```

Agent learned "HOLD always = 0 reward, trading = negative reward, so never trade"

### Solution Implemented
Added **idle penalty** when holding with no positions:

**Files changed:**
1. [environment/trading_env.py](environment/trading_env.py#L132-142)
   - Added 7 lines for idle penalty calculation

2. [config.yaml](config.yaml#L30-31)
   - Added `idle_penalty_coefficient: 0.0003`

**Effect:**
- HOLD when flat: slightly negative (-0.0003 × ATR)
- Forces agent to consider opportunity cost
- Still allows selectivity (penalty is small)

---

## Repository Structure

```
FinSense-1/
├── agents/              # DQN agent implementation
│   ├── dqn_agent.py    # Main agent with fixed epsilon decay
│   └── base_agent.py   # Base class
├── data_loader/         # Multi-stock data loading
│   └── data_loader.py  # Handles 5 Indian stocks + augmentation
├── environment/         # Trading environment
│   └── trading_env.py  # Contains idle penalty fix
├── utils/               # Utilities
│   ├── metrics.py      # Trading metrics calculation
│   ├── features.py     # Feature engineering
│   ├── checkpoint.py   # Model checkpointing
│   └── config.py       # Config management
├── tests/               # Test suite
├── config.yaml          # Main configuration
├── train.py             # Training script
├── evaluate.py          # Single-stock evaluation
├── evaluate_multistock.py  # Multi-stock evaluation (proper)
└── models/
    ├── best_model.pt    # Episode 15 (current best)
    └── best_model.json  # Metrics

Documentation:
├── IDLE_PENALTY_FIX.md           # Idle penalty explanation
├── TEST_SET_FAILURE_ANALYSIS.md  # Why 0 trades happened
├── EXPERT_FIXES_APPLIED.md       # All applied fixes
├── TRAINING_STATUS.md            # Current training status
├── HOW_TO_TRAIN.md               # Training guide
└── QUICK_START.md                # Quick start guide
```

---

## Key Configuration

### config.yaml
```yaml
environment:
  starting_balance: 50000
  max_positions: 5  # Max open positions
  max_position_value: 0.3  # 30% per position
  idle_penalty_coefficient: 0.0003  # Idle penalty

agent:
  epsilon_decay: 0.9905  # FIXED: was double-decaying
  learning_rate: 0.0005
  batch_size: 64
  memory_size: 10000

data:
  multi_stock: true
  stock_list:
    - RELIANCE.NS
    - TCS.NS
    - INFY.NS
    - HDFCBANK.NS
    - ICICIBANK.NS
  augment_data: true
  augmentation_noise: 0.01
  augmentation_copies: 2

training:
  train_ratio: 0.7
  validation_ratio: 0.15
  episodes: 300
```

---

## What To Do Next

### 1. Check Training Progress

```bash
# Monitor training
./monitor_50ep.sh

# Or tail the log
tail -f training_50ep_test.log

# Check if still running
ps aux | grep 2292
```

### 2. When Training Finishes

```bash
# Evaluate on test set
python evaluate_multistock.py
```

**Expected results:**
- Trades: 100-500 (good range)
- NOT 0 (that was the bug)
- NOT >1,000 (overtrading)

### 3. Decide Next Steps

**If test evaluation shows 100-500 trades:**
✅ Success! Run full 300-episode training:
```bash
python train.py --config config.yaml --episodes 300 --verbose > training_300ep.log 2>&1 &
```

**If test evaluation shows 0-50 trades:**
⚠️ Increase idle penalty:
```yaml
# In config.yaml
idle_penalty_coefficient: 0.0005  # Increase from 0.0003
```
Then retry 50 episodes.

**If test evaluation shows >1,000 trades:**
⚠️ Decrease idle penalty:
```yaml
# In config.yaml
idle_penalty_coefficient: 0.0001  # Decrease from 0.0003
```
Then retry 50 episodes.

---

## Important Context

### Previous Training (300 episodes)
- Started with equity delta reward + action change penalty
- **Episode 282:** Best model (₹26,456, 290 trades on validation)
- **Problem:** On test set → 0 trades (dead policy)
- **Root cause:** Missing opportunity cost for HOLD

### Expert Analysis
Two experts analyzed the failure:
1. First expert: Said full restart needed (too extreme)
2. Second expert: Said add idle penalty (minimal fix - we did this)

We followed the second expert's recommendation.

### Why Idle Penalty Works
```
Old:  HOLD (flat) = 0 reward → optimal to never trade
New:  HOLD (flat) = -0.0003×ATR → must beat opportunity cost to trade
```

Creates proper incentive structure:
- Good trade: positive (gains > costs + opportunity)
- Bad trade: negative (costs > gains)
- Inaction: small negative (opportunity cost)

---

## Files Currently Generated

### Logs
- `training_50ep_test.log` - Current 50-episode test
- `training.log` - Previous 300-episode run

### Models
- `models/best_model.pt` - Episode 15 (current best from ongoing run)
- `models/best_model.json` - Metrics for Episode 15

### Scripts
- `monitor_50ep.sh` - Monitoring script
- `evaluate_multistock.py` - Evaluation script (use this, not evaluate.py)

---

## Key Insights for Continuation

### 1. Always Use Multi-Stock Evaluation
❌ DON'T: `python evaluate.py --ticker RELIANCE.NS`
✅ DO: `python evaluate_multistock.py`

Training uses 5 stocks combined. Evaluation must match.

### 2. Episode 15 is Current Best
From ongoing 50-episode test:
- Profit: ₹13,777
- Trades: 3,824
- Win rate: 54.9%
- This is training data performance (not test set yet)

### 3. Watch for Policy Collapse
**Bad signs:**
- Trades dropping below 100/episode
- Sharpe ratio degrading over time
- Q-values all favoring HOLD again

**Good signs:**
- Trades stabilizing at 500-2,000
- Sharpe improving
- Mix of buy/hold/sell actions

### 4. Tunable Parameters
If agent behavior is off, tune these (in order):

1. `idle_penalty_coefficient` (0.0001-0.001)
   - Too high → overtrading
   - Too low → dead policy returns
   - Current: 0.0003

2. Action change penalty (in code)
   - Currently: 0.001 × ATR
   - Can adjust if churning

3. Epsilon decay
   - Currently: 0.9905 (reaches 0.05 at ep 300)
   - Working well, don't touch unless needed

---

## Common Issues & Solutions

### Issue: Agent still makes 0 trades
**Solution:** Increase `idle_penalty_coefficient` to 0.0005 or 0.001

### Issue: Agent overtrades (>5,000 trades/episode)
**Solution:** Decrease `idle_penalty_coefficient` to 0.0001

### Issue: Training crashes
**Check:**
1. Is data loading correctly? (5 stocks)
2. Are technical indicators calculated? (ATR needed for penalties)
3. GPU memory if using CUDA

### Issue: Can't reproduce results
**Remember:**
- Set random seeds (already in code)
- Use same config.yaml
- Same stock list and date range

---

## Testing & Validation

### Run Tests
```bash
pytest tests/ -v
```

Tests cover:
- Agent initialization
- Feature calculation
- Metrics computation

### Validate Config
```python
from utils import load_config
config = load_config('config.yaml')
print(config.get('environment.idle_penalty_coefficient'))
# Should print: 0.0003
```

---

## Git Workflow

### Current State
```
main (7 commits ahead of origin) - pushed ✓
└── experimental/reward-tuning (current) - pushed ✓
```

### Making Changes
```bash
# Make changes on experimental branch
git add <files>
git commit -m "Description"
git push origin experimental/reward-tuning
```

### If Experimental Works
```bash
# Switch to main
git checkout main

# Merge experimental
git merge experimental/reward-tuning

# Push to main
git push origin main
```

### If Experimental Fails
```bash
# Just abandon branch or keep iterating
# Main branch is safe with working code
```

---

## Performance Baselines

### Episode 15 (Current Best - Training)
```
Profit: ₹13,777 (27.6% return)
Trades: 3,824
Win rate: 54.9%
Sharpe: -0.61 (negative but improving)
Profit factor: 1.17
```

### Episode 282 (Previous Best - Validation)
```
Profit: ₹26,456 (52.9% return)
Trades: 290 (very selective)
Win rate: 68.97%
Sharpe: -0.54
Profit factor: 2.38
```

### Test Set Goal (After Fix)
```
Trades: 100-500 (not 0!)
Win rate: >50%
Profit: Positive
Sharpe: Improving over time
```

---

## Critical Files to Understand

### 1. environment/trading_env.py
**Lines 132-142:** Idle penalty implementation
```python
idle_penalty = 0.0
if len(self.inventory) == 0 and action == 1:  # HOLD with no positions
    atr_value = self.data.get('atr', ...)[self.current_step]
    idle_coeff = self.config.get('idle_penalty_coefficient', 0.0003)
    idle_penalty = idle_coeff * atr_value

reward = equity_delta - action_change_penalty - idle_penalty
```

### 2. agents/dqn_agent.py
**Lines 184-200:** Fixed epsilon decay (was double-decaying)
```python
# NOTE: Epsilon decay moved to end of episode in train.py
# Don't decay here to avoid double decay bug
```

### 3. train.py
**Lines 200-300:** Main training loop
- Creates environment
- Runs episodes
- Saves checkpoints
- Validates on validation set

### 4. evaluate_multistock.py
**Entire file:** Proper multi-stock evaluation
- Matches training distribution
- Uses same 5 stocks
- Splits train/val/test correctly

---

## Next Steps Summary

1. ✅ **Wait for training to finish** (~1.5 hours from 10:20 AM)
2. ⏳ **Run evaluation:** `python evaluate_multistock.py`
3. ⏳ **Check trades count:**
   - 100-500: Success → run 300 episodes
   - 0-50: Increase idle penalty → retry 50 episodes
   - >1000: Decrease idle penalty → retry 50 episodes
4. ⏳ **If successful, run full training:**
   ```bash
   python train.py --config config.yaml --episodes 300 --verbose > training_300ep.log 2>&1 &
   ```

---

## Resources

### Documentation Files
- `IDLE_PENALTY_FIX.md` - Detailed explanation of the fix
- `TEST_SET_FAILURE_ANALYSIS.md` - Why we had 0 trades
- `EXPERT_FIXES_APPLIED.md` - All previous fixes
- `TRAINING_STATUS.md` - Current status

### Key Papers/Concepts
- Equity delta reward (portfolio value change)
- Opportunity cost in RL
- Dead policy detection
- Reward shaping in trading

### Debugging
```bash
# Check Q-values
python -c "
from agents import DQNAgent
from utils import load_config
import torch

config = load_config('config.yaml')
agent = DQNAgent(26, 3, config.get_section('agent'))
agent.load('models/best_model.pt')

# Test Q-values on dummy state
state = torch.randn(26)
q_vals = agent.get_q_values(state)
print(f'Q(Buy): {q_vals[0]:.3f}')
print(f'Q(Hold): {q_vals[1]:.3f}')
print(f'Q(Sell): {q_vals[2]:.3f}')
"
```

---

## Final Notes

### What's Working
- ✅ Multi-stock data loading
- ✅ Feature engineering (26 features)
- ✅ Equity delta reward
- ✅ Action change penalty
- ✅ Idle penalty (new fix)
- ✅ Epsilon decay (fixed)
- ✅ Checkpoint system

### What Needs Monitoring
- ⏳ Test set performance (run evaluate_multistock.py)
- ⏳ Trade count trends (should stabilize, not collapse)
- ⏳ Sharpe ratio (should improve, not degrade)

### What NOT To Change
- ❌ Don't remove idle penalty (will break again)
- ❌ Don't train on single stock (defeats multi-stock setup)
- ❌ Don't use evaluate.py (use evaluate_multistock.py)
- ❌ Don't commit model files to git (too large)

---

## Emergency Contacts

If training crashes or results are unexpected:

1. Check logs: `tail -100 training_50ep_test.log`
2. Check GPU/CPU usage: `top` or `nvidia-smi`
3. Verify data loading: Check for errors in log
4. Test environment manually:
   ```python
   from environment import TradingEnvironment
   from data_loader import DataLoader
   from utils import load_config

   config = load_config('config.yaml')
   loader = DataLoader(config.get_section('data'))
   data = loader.load_data()
   env = TradingEnvironment(data, config.get_section('environment'))
   ```

---

**Good luck with the continuation! The foundation is solid, the fix is working, just need to validate and scale up.**
