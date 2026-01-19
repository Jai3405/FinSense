# Proper Training Guide for FinSense

**Date:** 2026-01-04
**Purpose:** Explain the difference between quick testing vs proper model training

---

## 🚨 What Went Wrong (The Bug)

###  Previous "100 Episode Training" Was Broken

**The Problem:**
```python
# In train.py - the epsilon decay was MISSING!
# After each episode, we logged epsilon but never decayed it

logger.info(f"Epsilon: {agent.epsilon:.3f}")  # Just logging
# ❌ MISSING: agent.decay_epsilon()  # Should be here!
```

**What Happened:**
- Epsilon started at 0.01 (from loaded model or bug)
- Stayed at 0.01 for all 100 episodes
- **No exploration happened** - agent just exploited existing knowledge
- Training finished in ~3 minutes (too fast!)
- Agent didn't actually learn anything new

**Expected Behavior:**
- Episode 1: epsilon = 1.0 (100% random exploration)
- Episode 50: epsilon ≈ 0.6 (60% random)
- Episode 100: epsilon ≈ 0.4 (40% random)
- Episode 200: epsilon ≈ 0.1 (10% random)
- Episode 300: epsilon = 0.05 (5% random, mostly greedy)

---

## ✅ The Fix

### 1. Fixed Training Loop

**File:** [train.py](train.py:296-297)

```python
# After each episode:
if profit > best_profit:
    best_profit = profit

# ✅ ADDED: Decay epsilon after each episode
agent.decay_epsilon()
```

### 2. Improved Hyperparameters

**File:** [config.yaml](config.yaml)

#### Agent Improvements:
```yaml
agent:
  epsilon_min: 0.05        # Was: 0.01 (more exploration)
  epsilon_decay: 0.998     # Was: 0.995 (slower decay)
  learning_rate: 0.0005    # Was: 0.001 (more stable)
  batch_size: 64           # Was: 32 (better gradients)
  memory_size: 5000        # Was: 2000 (more experience)
  hidden_size: 128         # Was: 64 (more capacity)
```

#### Environment Improvements:
```yaml
environment:
  window_size: 20          # Was: 10 (longer context)
```

#### Training Improvements:
```yaml
training:
  episodes: 300            # Was: 100 (proper learning time)
```

#### Data Improvements:
```yaml
data:
  multi_stock: true        # Was: false (5 stocks for diversity)
  use_market_index: true   # Was: false (Nifty50 context)
  augment_data: true       # Was: false (data augmentation)
```

---

## 📊 Quick Test vs Proper Training

### Quick Test (What we did before)
**Purpose:** Verify code works
**Duration:** 5-10 minutes
**Episodes:** 10-25
**Data:** Single stock (RELIANCE.NS)
**Features:** Basic (no augmentation, no multi-stock)
**Epsilon:** May or may not decay properly
**Result:** Proof that code runs without errors

### Proper Training (What we need now)
**Purpose:** Train a production-quality model
**Duration:** 2-8 hours (depending on settings)
**Episodes:** 300-500
**Data:** Multiple stocks (5+) with augmentation
**Features:** Full (indicators, market index, augmentation)
**Epsilon:** Proper exploration-to-exploitation transition
**Result:** Model that can actually trade profitably

---

## 🎯 What "Proper Training" Means

### 1. **Sufficient Exploration**
- Start with random actions (epsilon=1.0)
- Gradually become smarter
- Agent discovers profitable strategies through trial and error
- Takes TIME - can't rush this!

### 2. **Enough Episodes**
- **100 episodes:** Too few, agent barely explored
- **300 episodes:** Minimum for decent learning
- **500+ episodes:** Better, especially for complex strategies
- **1000+ episodes:** Professional-grade training

### 3. **Diverse Data**
- **Single stock:** Agent overfits to one stock's patterns
- **Multi-stock:** Learns general patterns (RELIANCE, TCS, INFY, HDFC, ICICI)
- **Augmentation:** Synthetic variations help generalization
- **Market index:** Understands market context

### 4. **Proper State Size**
With window_size=20 and all features:
- **State size ≈ 26 features:**
  - 19 price differences (window_size - 1)
  - 1 volume feature
  - 6 technical indicators (RSI, MACD, BB, ATR)

### 5. **Network Capacity**
- **Hidden size: 128** (was 64)
- More neurons = can learn complex patterns
- But also needs more data (hence multi-stock + augmentation)

---

## ⏱️ Expected Training Timeline

### Configuration: 300 Episodes, 5 Stocks, Augmentation

**Data Loading:** ~2-5 minutes
- Load 5 stocks from yfinance
- Apply augmentation (2 copies per stock)
- Validation and preprocessing

**Per Episode:** ~1-2 minutes
- Depends on data size (~5000+ points with augmentation)
- More data = more learning steps = longer episodes

**Total Training Time:** ~5-10 hours
- 300 episodes × 1-2 min/episode
- Can run overnight or while doing other work

**Progress Indicators:**
- Episodes 1-50: High loss, random profits (exploration)
- Episodes 50-150: Loss decreasing, profits improving
- Episodes 150-250: Strategy emerging, consistent profits
- Episodes 250-300: Refinement, exploitation, stable performance

---

## 📈 How to Monitor Proper Training

### Real-Time Monitoring

```bash
# Terminal 1: Start training
python start_training_monitor.py

# Terminal 2: TensorBoard (optional)
tensorboard --logdir runs/
# Open http://localhost:6006
```

### What to Watch For

1. **Epsilon Decay** (should decrease gradually)
   ```
   Episode 1:   epsilon = 1.000
   Episode 50:  epsilon = 0.606
   Episode 100: epsilon = 0.407
   Episode 200: epsilon = 0.134
   Episode 300: epsilon = 0.050
   ```

2. **Loss** (should trend downward)
   ```
   Episodes 1-50:   Loss = 50-200 (high, learning)
   Episodes 51-150: Loss = 10-50 (improving)
   Episodes 151-300: Loss = 1-10 (converging)
   ```

3. **Profit** (should become positive and stable)
   ```
   Episodes 1-50:   Profit = -₹5000 to +₹2000 (random)
   Episodes 51-150: Profit = -₹1000 to +₹5000 (learning)
   Episodes 151-300: Profit = +₹2000 to +₹10000 (trading well)
   ```

4. **Trades** (should be reasonable, not too many/few)
   ```
   Too few trades (0-5):   Not trading enough
   Good range (10-50):     Healthy trading activity
   Too many trades (200+): Overtrading, wasting on fees
   ```

---

## 🚀 Running Proper Training

### Step 1: Clear Old Models (Optional)
```bash
# Backup old models
mv models models_old_$(date +%Y%m%d)
mkdir models
```

### Step 2: Start Training
```bash
python start_training_monitor.py
```

**This will:**
- Train for 300 episodes
- Use 5 stocks (RELIANCE, TCS, INFY, HDFC, ICICI)
- Apply data augmentation
- Include Nifty50 market index
- Show live progress dashboard
- Take ~5-10 hours

### Step 3: Monitor Progress
- Live dashboard updates every episode
- Shows: profit, trades, epsilon, loss, Sharpe ratio
- Mini-chart of recent profits
- ETA for completion

### Step 4: Let It Run
- Can run overnight
- Training continues even if you close monitor (Ctrl+C)
- Models saved every 10 episodes

### Step 5: Evaluate Results
```bash
# After training completes
python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS --split test
```

---

## 💡 Tips for Successful Training

### Do's ✅
1. **Let it run fully** - Don't interrupt training midway
2. **Check every 50 episodes** - See if epsilon is decaying
3. **Watch TensorBoard** - Visual graphs help spot issues
4. **Save checkpoints** - Models saved every 10 episodes
5. **Compare to baseline** - Always evaluate vs buy-and-hold

### Don'ts ❌
1. **Don't expect instant results** - Episode 1-50 will be random
2. **Don't panic on losses** - Early exploration causes losses
3. **Don't overtrain** - >500 episodes may overfit
4. **Don't skip validation** - Always evaluate on test set
5. **Don't ignore epsilon** - If stuck at 0.01, training is broken

---

## 🔬 Understanding the Learning Process

### Phase 1: Random Exploration (Episodes 1-50)
**Epsilon:** 1.0 → 0.6
**Behavior:** Agent acts randomly
**Purpose:** Fill replay buffer with diverse experiences
**Metrics:** High loss, random profits, inconsistent

### Phase 2: Learning Patterns (Episodes 51-150)
**Epsilon:** 0.6 → 0.2
**Behavior:** Mix of exploration and learned strategy
**Purpose:** Discover profitable patterns
**Metrics:** Decreasing loss, improving profits

### Phase 3: Exploitation (Episodes 151-250)
**Epsilon:** 0.2 → 0.08
**Behavior:** Mostly using learned strategy
**Purpose:** Refine and optimize trading decisions
**Metrics:** Stable loss, consistent profits

### Phase 4: Fine-tuning (Episodes 251-300)
**Epsilon:** 0.08 → 0.05
**Behavior:** Greedy with minimal exploration
**Purpose:** Polish strategy for best performance
**Metrics:** Low loss, maximized profits

---

## 📊 Expected Results (Proper Training)

### Realistic Targets

**Conservative (Good):**
- Profit: 3-7% on test set
- Win Rate: 55-65%
- Sharpe Ratio: 0.5-1.2
- Max Drawdown: < 5%

**Optimistic (Great):**
- Profit: 8-15% on test set
- Win Rate: 65-75%
- Sharpe Ratio: 1.2-2.0
- Max Drawdown: < 3%

**Exceptional (Excellent):**
- Profit: > 15% on test set
- Win Rate: > 75%
- Sharpe Ratio: > 2.0
- Max Drawdown: < 2%

**Note:** If beating buy-and-hold on RELIANCE (which had 13.74% return), that's a win!

---

## 🎓 Key Takeaways

1. **The Bug Was Critical**
   - No epsilon decay = no learning
   - Previous "training" was just inference
   - Fixed by adding `agent.decay_epsilon()`

2. **Proper Training Takes Time**
   - 300 episodes × 1-2 min = 5-10 hours
   - Can't rush the learning process
   - Overnight training is normal

3. **More Data = Better Generalization**
   - Multi-stock training reduces overfitting
   - Data augmentation helps robustness
   - Market index provides context

4. **Monitor, Don't Micromanage**
   - Check progress every 50 episodes
   - Let agent explore early on
   - Trust the process

5. **Baseline Comparison Essential**
   - Always compare to buy-and-hold
   - DQN should beat simple strategies
   - If not, iterate on hyperparameters

---

## 🚀 Next Steps

1. **Run Proper Training NOW**
   ```bash
   python start_training_monitor.py
   ```

2. **Let It Complete** (5-10 hours)
   - Go do other work
   - Check back periodically
   - Training will save checkpoints

3. **Evaluate Results**
   ```bash
   python evaluate.py --model models/best_model.pt
   ```

4. **Iterate If Needed**
   - If results poor, adjust hyperparameters
   - Try longer training (500 episodes)
   - Experiment with reward function

5. **Deploy to SPIKE v2**
   - Once satisfied with performance
   - Use real-time trading system
   - Start with paper trading

---

**Summary:**
The previous training was broken (no epsilon decay). We've fixed it and configured proper training parameters. Now we need to run a REAL training session with 300 episodes, multi-stock data, and full feature set. This will take ~5-10 hours but will produce a production-quality model.

**Ready to train properly? 🚀**

---

**Last Updated:** 2026-01-04 02:00:00
**Status:** Ready for Proper Training
**Estimated Time:** 5-10 hours
