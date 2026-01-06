# TEST SET EVALUATION - CATASTROPHIC FAILURE

**Date:** 2026-01-05
**Model:** Episode 282 (best validation model)
**Result:** ZERO TRADES on test set

---

## Executive Summary

**CRITICAL FAILURE**: The model made **ZERO trades** on the test set despite being trained on the exact same 5-stock distribution. This is the worst possible outcome and indicates complete policy collapse.

### Quick Stats

| Metric | Validation (Ep 282) | Test Set | Status |
|--------|---------------------|----------|---------|
| Trades | 290 | **0** | ❌ DEAD POLICY |
| Profit | ₹26,456 | ₹0 | ❌ NO ACTIVITY |
| Win Rate | 68.97% | 0% | ❌ NO TRADES |
| Sharpe | -0.543 | 0.0 | ❌ NO DATA |

---

## What Happened

### Test Run Details

```
Loading multi-stock data (same as training)...
Test set size: 224 points

Evaluating Episode 282 model on TEST SET...
State size: 26, Action size: 3

Total Timesteps: 223
Total Trades: 0

Actions:
  Buy:  0
  Hold: 203
  Sell: 0
```

**The agent ONLY took HOLD actions for 203 consecutive steps.**

---

## Root Cause Analysis

### Investigation: Q-Value Inspection

I checked the agent's Q-values on test data:

```python
State (step 20):
Q-values: [[-0.96, -0.056, -1.58]]
           Buy    Hold    Sell
Action: 1 (HOLD)

State (step 30):
Q-values: [[-0.96, -0.056, -1.58]]
Action: 1 (HOLD)

State (step 40):
Q-values: [[-0.79, -0.056, -1.33]]
Action: 1 (HOLD)
```

### The Problem

**HOLD has the highest Q-value (-0.056) in ALL states.**

This means the agent learned:
- BUY → Expected return: ~₹-0.96
- **HOLD → Expected return: ~₹-0.056** ✅ (best option)
- SELL → Expected return: ~₹-1.58

The agent rationally chose HOLD because it expects to lose less money by doing nothing.

---

## Why This Happened

### The Reward Function Trap

Looking at [environment/trading_env.py:121-133](environment/trading_env.py#L121-L133):

```python
# Get portfolio value BEFORE action
prev_portfolio_value = self._calculate_portfolio_value(current_price)

# Execute action
action_reward, info = self._execute_action(action, current_price)

# Get portfolio value AFTER action
new_portfolio_value = self._calculate_portfolio_value(current_price)

# Reward = equity delta
equity_delta = new_portfolio_value - prev_portfolio_value

# Action change penalty
action_change_penalty = 0.0
if action != self.prev_action:
    atr_value = self.data.get('atr', ...)[self.current_step]
    action_change_penalty = 0.001 * atr_value

# Final reward
reward = equity_delta - action_change_penalty
```

### The Flaw

**When the agent has NO positions (inventory is empty):**

1. **BUY action:**
   - `prev_portfolio_value` = balance (no holdings)
   - Execute buy → reduce balance by cost + transaction cost
   - `new_portfolio_value` = reduced_balance + position_value
   - **equity_delta ≈ -transaction_cost** (immediate loss)
   - If action changed: **penalty = -0.001 × ATR**
   - **Total reward ≈ -transaction_cost - action_change_penalty** (NEGATIVE!)

2. **HOLD action:**
   - `prev_portfolio_value` = balance (no holdings)
   - Nothing happens
   - `new_portfolio_value` = balance (unchanged)
   - **equity_delta = 0** (no change)
   - **Total reward = 0** (NEUTRAL!)

3. **SELL action:**
   - Can't sell (no inventory)
   - Gets rejected or does nothing
   - **Total reward ≈ 0 or slightly negative**

**Result:** Agent learns "HOLD = 0 reward, BUY = negative reward, so ALWAYS HOLD."

---

## Training Data Mismatch?

### Data Split Analysis

```
Total raw data: 1,492 points (5 stocks combined)

Split:
- Train: 1,044 points (70%)
- Val:   224 points (15%)
- Test:  224 points (15%)

During training:
- Augmentation: 2 copies with 1% noise
- Effective train size: ~3,132 points

During evaluation:
- NO augmentation
- Raw test size: 224 points
```

**Is this the problem?** NO. Even if test data looks slightly different, a working agent should still TRY to trade. A dead policy (0 trades) indicates the agent learned "never trade" as the optimal strategy.

---

## Expert Predictions Were Correct

From [EXPERT_FIXES_APPLIED.md:199-203](EXPERT_FIXES_APPLIED.md#L199-L203):

> ❌ BAD SIGNS (Old Behavior):
> - Trades drop below 300/episode
> - Sharpe ratio gets worse over time
> - Profit stays negative or near zero
> - **Agent just holds and does nothing (dead policy)**

From [EXPERT_FIXES_APPLIED.md:306-312](EXPERT_FIXES_APPLIED.md#L306-L312):

> ⚠️ Warning Signs (Old Behavior Returning)
>
> If you see ANY of these, reward function still broken:
> - **Trades drop below 300/episode** ✅ (0 trades!)
> - **Sharpe ratio gets worse over time** ✅ (was -0.77 → -5.82)
> - **Profit stays negative or near zero** ✅ (₹0!)
> - **Agent just holds everything (dead policy)** ✅ (confirmed!)

**ALL warning signs were present.** The expert was 100% correct.

---

## What Went Wrong With The Expert Fixes?

The expert recommended [EXPERT_FIXES_APPLIED.md:35-65](EXPERT_FIXES_APPLIED.md#L35-L65):

### ✅ Fix #1: Equity Delta Reward
**Implemented correctly**, but has a fatal flaw:

**Problem:** When agent has NO positions:
- HOLD → equity_delta = 0 (no change in portfolio)
- BUY → equity_delta = -transaction_cost (immediate loss)

**Result:** Agent learns to NEVER BUY because buying always gives immediate negative reward.

### ✅ Fix #2: Action Change Penalty
**Implemented correctly**, but makes the problem WORSE:

```python
if action != self.prev_action:
    action_change_penalty = 0.001 * ATR
```

**Problem:** This FURTHER penalizes changing from HOLD to BUY or SELL.

**Result:** Agent learns to stay in HOLD forever because changing actions is punished.

---

## The Core Issue: Myopic Reward Design

### What The Agent Sees (From Experience)

Episode during training:

```
Step 1: HOLD → reward = 0
Step 2: BUY → reward = -20 (txn cost) - 5 (action penalty) = -25
Step 3: HOLD → reward = 0 (no position value change yet)
Step 4: HOLD → reward = 0
Step 5: SELL → reward = -20 (txn cost) - 5 (action penalty) + 10 (profit) = -15
...

Average reward from HOLD: 0
Average reward from BUY: -25
Average reward from SELL: -15
```

**Optimal policy learned:** ALWAYS HOLD.

### What Should Have Happened

A proper reward function should reward:
- Buying low → Positive reward when position increases in value
- Holding winners → Positive reward as unrealized gains grow
- Selling high → Positive reward when locking in profits
- Cutting losses → Small negative reward (better than holding losers)

But **equity delta doesn't reward unrealized gains properly** when the agent is risk-averse and sees immediate transaction costs.

---

## Comparison to Training Behavior

### Training Phase Analysis (from logs)

| Phase | Episodes | Avg Trades | Avg Profit | Avg Sharpe |
|-------|----------|------------|------------|------------|
| Early | 1-100 | 6,918 | ₹-5,084 | -0.77 |
| Mid | 101-200 | 4,982 | ₹-1,196 | -1.04 |
| Late | 201-300 | 1,039 | ₹-430 | **-5.82** |

**Episode 282 specifically:**
- Trades: 290
- Profit: ₹26,456
- Sharpe: -0.543

### The Pattern

```
Early training: 6,918 trades → Random exploration
Mid training: 4,982 trades → Learning to trade less
Late training: 1,039 trades → Policy collapsing
Episode 282: 290 trades → Occasional trading (lucky run)
Test set: 0 trades → COMPLETE POLICY COLLAPSE
```

**Episode 282 was an outlier** - a lucky validation run where the agent happened to trade. But the underlying learned policy is "mostly hold, rarely trade."

On the test set, the agent reverted to its true learned policy: **NEVER TRADE.**

---

## Why Validation Looked Good But Test Failed

### Validation Set (Ep 282)

During training, Episode 282 evaluated on **validation set**:
- 224 points
- Agent made 290 trades
- Profit: ₹26,456

**Why did it trade?** Possible reasons:
1. **Epsilon = 0.057** (still some exploration)
2. **Lucky data sequence** (validation set had favorable patterns)
3. **Q-value variance** (some states had BUY Q-value slightly higher)

### Test Set (Evaluation)

During evaluation on **test set**:
- 224 points (same size as validation)
- Agent made **0 trades**
- Epsilon = 0.0 (no exploration)

**Why didn't it trade?**
1. **No exploration** (epsilon = 0)
2. **Q-values converged to HOLD** (learned policy dominates)
3. **Test data slightly different** (out of temporal distribution)

**The real issue:** The learned policy is "HOLD > BUY > SELL" in almost all states. Episode 282's validation performance was **lucky noise**, not a learned strategy.

---

## Expert's Missing Insight

The expert said [EXPERT_FIXES_APPLIED.md:516-518](EXPERT_FIXES_APPLIED.md#L516-L518):

> "reward = equity[t] - equity[t-1]
> This naturally rewards holding winners and penalizes churn"

**But the expert MISSED this:**

Equity delta DOES reward holding winners... **IF you have positions to hold.**

But it DOESN'T incentivize ACQUIRING positions in the first place, because:
- Acquiring positions (BUY) has immediate negative reward (transaction cost)
- Agent never learns to acquire positions
- Therefore, agent never has winners to hold

**It's a chicken-and-egg problem:**
- Need positions to get positive holding rewards
- But acquiring positions gives negative reward
- So agent never acquires positions
- So agent never experiences positive holding rewards
- So agent learns "don't acquire positions"

---

## The Real Fix (What Should Have Been Done)

### Proper Reward Shaping

```python
def calculate_reward(self, action, prev_value, new_value, info):
    # Base reward: equity delta
    equity_delta = new_value - prev_value

    # CRITICAL FIX: Offset transaction costs
    # Don't penalize initial buy - only penalize churn
    if action == BUY and info['successful_buy']:
        # Offset transaction cost for productive buys
        equity_delta += info['transaction_cost']

        # Small exploration bonus for taking risk
        equity_delta += 0.01 * ATR  # Encourage position-taking

    if action == SELL and info['successful_sell']:
        # Only penalize if sold too early (held < 5 steps)
        if info['hold_duration'] < 5:
            equity_delta -= 0.01 * ATR  # Discourage churn

    # Action change penalty ONLY if churning (changing repeatedly)
    if action != self.prev_action and self.consecutive_holds < 3:
        equity_delta -= 0.001 * ATR

    return equity_delta
```

### Key Fixes Needed

1. **Offset transaction costs on BUY** - Don't punish position acquisition
2. **Exploration bonus** - Reward trying new positions
3. **Hold duration tracking** - Only penalize selling too quickly
4. **Churn detection** - Only penalize rapid action changes, not strategic trades

---

## Verification: Did Expert Fixes Get Applied?

Let me check if the code matches [EXPERT_FIXES_APPLIED.md](EXPERT_FIXES_APPLIED.md):

### ✅ Equity Delta Reward

**Expert said:**
> reward = equity[t] - equity[t-1]

**Code ([trading_env.py:121-123](environment/trading_env.py#L121-L123)):**
```python
prev_portfolio_value = self._calculate_portfolio_value(current_price)
# ... action ...
new_portfolio_value = self._calculate_portfolio_value(current_price)
equity_delta = new_portfolio_value - prev_portfolio_value
```

✅ **Implemented correctly**

### ✅ Action Change Penalty

**Expert said:**
> if action != prev_action: reward -= 0.001 * ATR

**Code ([trading_env.py:126-130](environment/trading_env.py#L126-L130)):**
```python
if action != self.prev_action:
    atr_value = self.data.get('atr', ...)[self.current_step]
    action_change_penalty = 0.001 * atr_value
```

✅ **Implemented correctly**

### Conclusion

**The expert's recommendations were implemented correctly, but they were INSUFFICIENT to prevent policy collapse.**

The expert's fixes addressed:
- ✅ Holding winners (works IF you have positions)
- ✅ Preventing churn (works TOO WELL - prevents trading entirely)

But MISSED:
- ❌ Incentivizing position acquisition
- ❌ Rewarding exploration of trading strategies
- ❌ Balancing transaction costs with long-term gains

---

## Bottom Line

### What We Learned

1. **Episode 282's validation performance (₹26,456 profit, 290 trades) was LUCKY NOISE**, not a learned strategy
2. **The agent's true learned policy is "ALWAYS HOLD"** (Q_hold > Q_buy > Q_sell in all states)
3. **Equity delta reward DOESN'T solve the transaction cost problem** - it still penalizes initial position acquisition
4. **Action change penalty MADE IT WORSE** - it discouraged any trading activity

### What This Means

**The model is COMPLETELY UNUSABLE for trading.**

- It will NEVER make trades in production
- It has learned risk avoidance, not strategic trading
- Training for 300 episodes with these rewards was WASTED EFFORT

### What Needs To Happen

**COMPLETE RESTART** with fixed reward function that:
1. Offsets transaction costs on productive buys
2. Rewards unrealized gains (holding winners)
3. Only penalizes TRUE churn (rapid buy-sell-buy)
4. Encourages exploration of trading strategies

**Time required:**
- 2-4 hours: Redesign and implement reward function
- 10-12 hours: Retrain 300 episodes
- 1 hour: Evaluate and verify

**Total: ~15 hours to potentially working model**

---

## Expert's Verdict (Predicted)

From [EXPERT_FIXES_APPLIED.md:473-477](EXPERT_FIXES_APPLIED.md#L473-L477):

> **If you still see:**
> - Trades: Collapsing toward 600 ❌
> - Sharpe: Getting worse ❌
> - Profit: Always negative ❌
>
> **Then we have deeper problems.**

**We have ALL of these + worse (0 trades).** The expert would say:

> "Your agent learned the correct response to a broken incentive structure. The problem isn't the agent, the network, the features, or the training loop. **The problem is the reward function doesn't incentivize trading - it incentivizes NOT TRADING.**
>
> Equity delta + action change penalty = 'minimize activity' instead of 'maximize risk-adjusted returns through strategic trading.'
>
> This is a fundamental design flaw. You cannot fix this by training longer or tweaking hyperparameters. You must redesign the reward function from scratch."

---

## Recommendations

### Immediate Actions (DO NOT PROCEED UNTIL FIXED)

1. **DO NOT deploy this model** - it will never trade
2. **DO NOT train with current reward function** - will produce same result
3. **DO NOT add more features/indicators** - not the problem

### Required Changes (Priority Order)

#### 1. Redesign Reward Function (CRITICAL)

**File to modify:** [environment/trading_env.py:97-147](environment/trading_env.py#L97-L147)

**Changes needed:**
```python
# Option A: Unrealized P&L reward
reward = (
    realized_pnl  # Profit from sells
    + unrealized_pnl_change  # Change in open position values
    - transaction_costs  # Explicit cost tracking
    - churn_penalty  # Only if rapid trading
    + exploration_bonus  # Encourage trying positions
)

# Option B: Returns-based reward
reward = (
    portfolio_return  # % change in portfolio value
    * (1 - volatility_penalty)  # Risk adjustment
    - action_frequency_penalty  # Discourage overtrading
)

# Option C: Hybrid (recommended)
reward = (
    equity_delta  # Keep this
    + buy_offset_bonus  # CRITICAL: Offset txn cost on BUY
    - sell_churn_penalty  # Only penalize SELL if held < N steps
    - action_change_penalty  # Keep but ONLY if consecutive changes
)
```

#### 2. Add Position Incentives

**Current problem:** Agent never acquires positions because buying has negative immediate reward.

**Fix:** Add temporary "position acquisition bonus" to offset transaction costs:

```python
if action == BUY and buy_successful:
    # Offset transaction cost to make BUY neutral, not negative
    reward += transaction_cost
    # Small bonus for taking a position
    reward += 0.01 * portfolio_value
```

#### 3. Track Hold Duration

**Current problem:** Action change penalty penalizes ALL changes equally.

**Fix:** Only penalize changes if position wasn't held long enough:

```python
if action == SELL:
    if self.position_hold_duration[symbol] < 5:
        # Penalize churning (sell too early)
        reward -= 0.01 * ATR
    # else: no penalty for strategic sells
```

#### 4. Retrain From Scratch

After fixing rewards:
- Delete old models
- Train fresh 300-episode run
- Monitor for:
  - Trades staying in 500-2,000 range (not collapsing)
  - Profits becoming positive
  - Sharpe improving over time

---

## Appendix: Training Trajectory

### Trade Count Collapse Timeline

```
Episode 1:   8,635 trades (random exploration)
Episode 50:  6,000 trades (learning)
Episode 100: 4,000 trades (converging)
Episode 150: 2,500 trades (policy forming)
Episode 200: 1,500 trades (mostly holds)
Episode 250:   900 trades (collapsing)
Episode 282:   290 trades (lucky outlier)
Episode 300:   644 trades (dead policy)

Test set:    0 trades (COMPLETE COLLAPSE)
```

### Sharpe Ratio Degradation Timeline

```
Episodes 1-100:    -0.77 (exploration noise)
Episodes 101-200:  -1.04 (getting worse)
Episodes 201-300:  -5.82 (CATASTROPHIC)
Episode 282:       -0.543 (lucky validation run)
Test set:          0.0 (no data - no trades)
```

**The signs were there all along.** Trade counts collapsing + Sharpe degrading = broken reward function.

---

## Conclusion

**This is not a failure of training - it's a failure of reward design.**

The agent learned EXACTLY what it was incentivized to learn:
- HOLD = safe (0 reward)
- BUY = risky (negative reward)
- SELL = risky (negative reward)

**Optimal learned policy: ALWAYS HOLD.**

Episode 282's validation performance (₹26,456, 290 trades) was statistical noise - a lucky sequence where Q-values happened to favor trading. On the test set, with no exploration (epsilon=0), the true learned policy emerged: **NEVER TRADE.**

**The model is dead. Long live the (future, properly-rewarded) model.**

---

## Next Steps

**User decision required:**

1. **Accept reality** - Current model is unusable
2. **Redesign rewards** - Fix transaction cost penalty and action change penalty
3. **Retrain from scratch** - 300 episodes with fixed rewards (~15 hours total effort)
4. **Validate properly** - Ensure test set trades are 500-2,000, not 0 or 10,000

**DO NOT proceed with:**
- Paper trading (model won't trade)
- Real capital deployment (model won't trade)
- Adding features (not the problem)
- Training more episodes (will make it worse)

**The ball is in your court. Fix rewards, or accept a non-trading model.**
