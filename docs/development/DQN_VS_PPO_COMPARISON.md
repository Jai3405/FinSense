# DQN vs PPO Comparison - Fixed Percentage Rewards

## Current Status

**DQN:** 50-episode training completed ✅
**PPO:** 50-episode training in progress (Episode 8/50)

---

## DQN Results (50 Episodes)

### Training Performance:
- Episodes 1-10: High exploration, learning basic patterns
- Episodes 11-50: Converging policy, selective trading
- Final training: 500-700 trades per episode

### Test Set Evaluation:
- **Total trades: 33**
- **Action distribution:** Buy=99, Hold=91, Sell=14
- **Final balance:** ₹43,610 (started at ₹50,000)
- **P&L:** -₹6,389 (-12.8%)

### Key Findings:
✅ Dead policy FIXED - Agent trades actively
✅ Uses all three actions (not just HOLD)
⚠️ Negative P&L on test set (needs more training)

---

## PPO Results (In Progress - Episodes 1-8)

### Training Performance So Far:
- **Trades per episode:** 692-716 (very active)
- **Profits per episode:** ₹395-₹2,435
- **Cumulative rewards:** -8.90 to -11.17

### Observations:
✅ Trading very actively (more than DQN)
✅ Positive profits on training set
📊 Need to complete 50 episodes and evaluate on test set

---

## Expected Timeline

**PPO Training:** ~1.5 hours for 50 episodes (same as DQN)
- Current: Episode 8/50 (~16% complete)
- Estimated completion: ~1 hour 20 minutes from now

**Next Steps:**
1. Wait for PPO training to complete
2. Run evaluation on test set: `python evaluate_ppo.py`
3. Compare metrics:
   - Test set trades (DQN: 33 vs PPO: ?)
   - Test set P&L (DQN: -12.8% vs PPO: ?)
   - Sharpe ratio
   - Action distribution
4. Pick winner for full 200-episode training

---

## How to Monitor PPO Progress

```bash
# Watch live updates
tail -f training_PPO_FIXED_50ep.log

# Check current episode
grep "Episode" training_PPO_FIXED_50ep.log | tail -1

# Check when it completes
ls -lh models/ppo_final.pt
```

---

## Decision Criteria

**Pick DQN if:**
- PPO shows similar or worse test set performance
- PPO overtrades (>500 test trades)
- PPO has higher variance

**Pick PPO if:**
- Better test set Sharpe ratio
- Better risk-adjusted returns
- More stable policy

**Run both for 200 episodes if:**
- Results are very close
- Want to ensemble predictions
