# Production-Ready Fix Plan

## Root Cause
Reward components are misscaled by 4-5 orders of magnitude, making rational policy = "never trade".

## The Fix (3-Step Process)

### Step 1: Normalize All Rewards to Percentage Returns (2 hours)

**File:** `environment/trading_env.py`

**Change:**
```python
# OLD (BROKEN)
reward = equity_delta  # In rupees

# NEW (FIXED)
portfolio_return = (new_portfolio_value / prev_portfolio_value - 1)  * 100  # In percent
reward = portfolio_return
```

**Why:** All rewards now in same scale (percentage points)

### Step 2: Recalibrate Penalties (30 minutes)

```python
# Action change penalty
if action != self.prev_action:
    reward -= 0.01  # 0.01% penalty (was 0.001 × ATR)

# Idle penalty
if len(self.inventory) == 0 and action == 1:
    reward -= 0.02  # 0.02% opportunity cost

# Invalid trade penalty
if failed_trade:
    reward -= 0.01  # 0.01% penalty
```

**Why:** All penalties now comparable to typical returns (0.1-0.5%/day)

### Step 3: Remove Dominating Bonuses (15 minutes)

**REMOVE:**
- Holding reward for winners (creates "never sell" bias)
- Realized PnL multiplier (creates asymmetric incentive)
- Trend alignment bonus (adds noise)

**KEEP:**
- Equity delta (core signal)
- Action change penalty (reduces churn)
- Idle penalty (encourages participation)

---

## Implementation Timeline

**Day 1 (Today):**
- [ ] Implement percentage-based rewards
- [ ] Recalibrate penalties
- [ ] Remove bonus terms
- [ ] Test on 10 episodes

**Day 2:**
- [ ] If trades > 50: Run 100-episode training
- [ ] If trades < 50: Adjust idle penalty
- [ ] Evaluate on test set

**Day 3:**
- [ ] If Sharpe > 0: Scale to 300 episodes
- [ ] If Sharpe < 0: Add Sharpe-aware reward term
- [ ] Production validation

---

## Expected Results

**With percentage rewards:**
- Transaction cost: ~0.04% per trade
- Idle penalty: ~0.02% per step
- Ratio: 2:1 (reasonable)

**Agent will learn:**
- Trade when expected return > 0.06% (transaction + opportunity cost)
- This happens ~20-30% of timesteps in trending markets
- Expected test set trades: 50-150 (selective, not dead)

---

## Fallback Plan

If percentage rewards still don't work:

### Option B: Transaction Cost Subsidy
```python
# Temporarily reduce costs for learning
transaction_cost = 0.0001  # 10× lower than real
```

Train with subsidized costs, then:
1. Gradually increase to real costs over episodes
2. Use curriculum learning

###Option C: Sharpe-Aware Reward
```python
# Calculate rolling 20-step Sharpe
returns = np.diff(portfolio_values[-20:]) / portfolio_values[-20:-1]
sharpe = np.mean(returns) / (np.std(returns) + 1e-8)
reward = sharpe * 10  # Directly optimize risk-adjusted returns
```

---

## Success Metrics

**Minimum Viable Product:**
- Test set trades: 50-200
- Sharpe ratio: > 0.3
- Max drawdown: < 20%
- Win rate: > 48%

**Production Ready:**
- Test set Sharpe: > 0.5
- Max drawdown: < 15%
- Consistent across 5 random seeds

---

## What NOT To Do

❌ Add more reward bonuses
❌ Implement new architectures (Dueling DQN, etc.)
❌ Add more features
❌ Try different algorithms (PPO, A2C, etc.)

✅ Fix the reward scale
✅ Test rigorously
✅ Iterate on ONE variable (idle penalty coefficient)
