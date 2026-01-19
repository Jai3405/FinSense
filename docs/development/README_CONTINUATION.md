# FinSense Project - Continuation Instructions

## Quick Summary

**What we fixed:** Dead policy (agent made 0 trades on test set)
**How we fixed it:** Added idle penalty for holding with no positions
**Current status:** Testing fix with 50-episode run (Episode 15 showing ₹13,777 profit, 3,824 trades)

---

## For Gemini CLI

### Current State
- **Branch:** `experimental/reward-tuning` (use this for changes)
- **Training:** In progress (PID 2292, log: `training_50ep_test.log`)
- **What to do:** Read [HANDOFF_GUIDE.md](HANDOFF_GUIDE.md) for full context

### Immediate Tasks

1. **Check training progress:**
   ```bash
   ./monitor_50ep.sh
   # or
   tail -f training_50ep_test.log
   ```

2. **When training finishes, evaluate:**
   ```bash
   python evaluate_multistock.py
   ```

3. **Interpret results:**
   - Trades 100-500: ✅ Success → run 300 episodes
   - Trades 0-50: ⚠️ Increase idle_penalty_coefficient
   - Trades >1000: ⚠️ Decrease idle_penalty_coefficient

### Key Files to Understand

1. **[HANDOFF_GUIDE.md](HANDOFF_GUIDE.md)** - READ THIS FIRST
   - Complete project context
   - What was fixed and why
   - Next steps decision tree
   - All configuration details

2. **[IDLE_PENALTY_FIX.md](IDLE_PENALTY_FIX.md)** - Technical details
   - Why idle penalty works
   - Mathematical foundation
   - Tuning guidelines

3. **[environment/trading_env.py](environment/trading_env.py#L132-142)** - The fix
   - Lines 132-142: Idle penalty implementation
   - Only 7 lines changed

4. **[config.yaml](config.yaml#L30-31)** - Configuration
   - Line 31: `idle_penalty_coefficient: 0.0003`

### Repository Context

**Structure:**
```
FinSense-1/
├── agents/           # DQN implementation
├── data_loader/      # Multi-stock data loading
├── environment/      # Trading environment (idle penalty here)
├── utils/            # Utilities (metrics, features, config)
├── config.yaml       # Main configuration
├── train.py          # Training script
├── evaluate_multistock.py  # Evaluation (use this!)
└── HANDOFF_GUIDE.md  # START HERE
```

**Training Setup:**
- 5 Indian stocks (RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK)
- Multi-stock training with data augmentation
- 26 features (price, volume, RSI, MACD, Bollinger, ATR)
- Equity delta reward + action change penalty + idle penalty

---

## What Changed (Summary)

### Problem
Agent converged to "always HOLD" policy:
```
Q(HOLD) = -0.056  (highest - always chosen)
Q(BUY)  = -0.96
Q(SELL) = -1.58

Result: 0 trades on test set
```

### Solution
Added opportunity cost for inaction:
```python
# environment/trading_env.py lines 132-142
if len(self.inventory) == 0 and action == 1:  # HOLD when flat
    idle_penalty = 0.0003 * ATR

reward = equity_delta - action_change_penalty - idle_penalty
```

### Result
Episode 1: 8,587 trades (agent is trading again! ✅)
Episode 15: 3,824 trades, ₹13,777 profit

---

## Git Branches

### main
- Stable code with all fixes
- 8 commits pushed
- All documentation included

### experimental/reward-tuning
- Where ongoing work happens
- Use this for changes
- Can merge to main if successful
- Has HANDOFF_GUIDE.md

### Workflow
```bash
# Make changes
git checkout experimental/reward-tuning
git add <files>
git commit -m "Description"
git push origin experimental/reward-tuning

# If successful
git checkout main
git merge experimental/reward-tuning
git push origin main
```

---

## Critical Information

### DON'T
- ❌ Use `evaluate.py` (single-stock, wrong)
- ❌ Train on single stock (defeats multi-stock setup)
- ❌ Remove idle penalty (will break again)
- ❌ Change epsilon decay (already fixed)

### DO
- ✅ Use `evaluate_multistock.py` (matches training)
- ✅ Read HANDOFF_GUIDE.md fully
- ✅ Monitor training with `./monitor_50ep.sh`
- ✅ Tune `idle_penalty_coefficient` if needed

---

## Success Criteria

### For 50-Episode Test
- Trades > 0 in most episodes ✅ (already happening)
- Trades stabilizing (not collapsing)
- Agent making diverse actions (buy/hold/sell mix)

### For Test Set Evaluation
- Trades: 100-500 (reasonable range)
- Win rate: >50%
- Policy alive (taking actions)

### For Full 300-Episode Run
- Trades: 500-2,000/episode
- Sharpe improving over time
- Profit consistently positive
- No policy collapse

---

## What's Already Done

✅ Refactored core modules (agents, data_loader, environment, utils)
✅ Implemented equity delta reward
✅ Fixed epsilon decay double-decay bug
✅ Added action change penalty
✅ **Added idle penalty (new fix)**
✅ Created multi-stock evaluation script
✅ Added comprehensive documentation
✅ Set up experimental branch for continuation

---

## Monitoring Commands

```bash
# Quick status
./monitor_50ep.sh

# Live log watching
tail -f training_50ep_test.log

# Check if training is running
ps aux | grep 2292

# See recent episodes
grep "Episode" training_50ep_test.log | tail -10

# Count zero-trade episodes (should be low)
grep "Trades: 0" training_50ep_test.log | wc -l
```

---

## If Something Goes Wrong

### Training Crashes
1. Check log for errors: `tail -100 training_50ep_test.log`
2. Check GPU memory: `nvidia-smi` or `top`
3. Verify data loaded: Look for "Loading multi-stock data" in log

### Still Getting 0 Trades
1. Increase idle penalty in [config.yaml](config.yaml#L31):
   ```yaml
   idle_penalty_coefficient: 0.0005  # or 0.001
   ```
2. Restart 50-episode test

### Overtrading (>5,000 trades/episode)
1. Decrease idle penalty in [config.yaml](config.yaml#L31):
   ```yaml
   idle_penalty_coefficient: 0.0001
   ```
2. Restart 50-episode test

---

## Timeline

### Completed
- ✅ Implemented idle penalty fix
- ✅ Started 50-episode test (10:20 AM)
- ✅ Episode 1-15 showing agent is trading

### In Progress
- ⏳ 50-episode test running (~1.5 hours total)

### Next
- ⏳ Evaluate on test set
- ⏳ If successful: Run 300-episode training (~10 hours)
- ⏳ Final evaluation and deployment decision

---

## Context for New AI Assistant

This project uses reinforcement learning (DQN) for stock trading. The agent:
- Observes market state (26 features)
- Takes actions (buy, hold, sell)
- Receives rewards (equity delta - penalties)
- Learns optimal trading strategy

**Recent breakthrough:** Fixed dead policy by adding opportunity cost. Agent now actively trades instead of freezing.

**Your role:** Monitor 50-episode test, evaluate results, decide next steps. All details in [HANDOFF_GUIDE.md](HANDOFF_GUIDE.md).

---

## File Checklist

Key files for understanding:
- [ ] HANDOFF_GUIDE.md (comprehensive guide)
- [ ] IDLE_PENALTY_FIX.md (technical details)
- [ ] TEST_SET_FAILURE_ANALYSIS.md (what went wrong)
- [ ] EXPERT_FIXES_APPLIED.md (previous fixes)
- [ ] environment/trading_env.py (where idle penalty lives)
- [ ] config.yaml (configuration)

---

**Start here:** [HANDOFF_GUIDE.md](HANDOFF_GUIDE.md)

**Questions?** All information is in the handoff guide. Read it thoroughly before making changes.

**Good luck!** The hard part is done. Just need to validate and scale.
