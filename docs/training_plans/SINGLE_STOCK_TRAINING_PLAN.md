# Single-Stock PPO Training Plan (RELIANCE.NS)
**Scientific Approach: Isolate Variables Before Scaling**

---

## Why Single-Stock First?

**After 200 episodes of multi-stock training showed NO convergence**, we need to isolate the problem.

**Two possible causes:**
1. Multi-stock environment too complex for PPO to learn
2. PPO hyperparameters are wrong

**Solution:** Train on RELIANCE.NS only
- If it learns → Multi-stock was the problem
- If it fails → PPO hyperparameters need fixing

---

## Configuration Changes

### Data:
```yaml
BEFORE (Multi-stock):
  ticker: [RELIANCE, TCS, INFY, HDFCBANK, ICICIBANK]
  multi_stock: true
  Total points: 7,510 (1,502 per stock)

AFTER (Single-stock):
  ticker: RELIANCE.NS
  multi_stock: false
  Total points: 1,502
  Train/Val/Test: 1,051 / 225 / 226
```

### PPO Hyperparameters (OPTIMIZED):
```yaml
BEFORE (Too Conservative):
  learning_rate: 0.0003  ← Too slow
  entropy_coef: 0.01     ← Not enough exploration
  batch_size: 64         ← Noisy gradients

AFTER (Aggressive):
  learning_rate: 0.001   ← 3× faster learning
  entropy_coef: 0.05     ← 5× more exploration
  batch_size: 128        ← More stable updates
```

### Episodes:
**150 episodes** (reduced from 200)
- Single stock converges faster
- Less data = less time per episode
- Can iterate faster

---

## Expected Behavior

### If PPO is Working (Episodes 1-150):

**Episodes 1-30:** Exploration
- Trade counts: 1,800-2,000
- Rewards: -25 to -30
- Random profits

**Episodes 30-80:** Learning
- Trade counts: DECREASE to 1,200-1,500
- Rewards: IMPROVE to -20 to -15
- Profits becoming consistent

**Episodes 80-150:** Convergence
- Trade counts: Stabilize at 800-1,200
- Rewards: -10 to -5
- Consistent profitable pattern

### If PPO Still Fails:
- Trade counts stay at 1,800-2,000 throughout
- Rewards stuck at -25 to -30
- No improvement visible

→ Then we switch to DQN or redesign rewards

---

## Training Command

```bash
cd /Users/jay/FinSense-1
source finsense_env/bin/activate

nohup python train_ppo.py --episodes 150 --verbose > training_150ep_SINGLESTOCK_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $!

tail -f training_150ep_SINGLESTOCK_*.log
```

---

## Timeline

**Expected duration:** ~18 hours (much faster than multi-stock)

**Why faster?**
- Single stock: ~12 minutes per episode
- Multi-stock was: ~18 minutes per episode
- 150 episodes × 12 min = ~30 hours
- But with faster learning rate, may converge in 100 episodes = ~20 hours

---

## Success Criteria

**After 150 episodes, evaluate on test set:**

### Minimum Success (Proves PPO Works):
✅ Test set trades: >50 (not 0-1)
✅ Sharpe ratio: >0.2 (positive)
✅ Trade count reduction: >20% from episode 1 to 150
✅ Reward improvement: >5 points

→ **If achieved:** PPO works! Multi-stock was too complex. Scale gradually.

### Failure (PPO Broken):
❌ Test set trades: <10
❌ Sharpe: <0 (negative)
❌ No convergence (trades stay at 2,000)
❌ No reward improvement

→ **If this happens:** Switch to DQN or completely redesign reward function

---

## Next Steps After Single-Stock

### If Single-Stock Succeeds:

**Phase 1:** Optimize single-stock
- Train 200-300 episodes for best performance
- Target: Sharpe >0.5 on RELIANCE

**Phase 2:** Add 1 more stock (2-stock training)
- RELIANCE.NS + TCS.NS
- Verify it still converges
- 150 episodes

**Phase 3:** Gradually scale
- 3 stocks (add INFY)
- 4 stocks (add HDFCBANK)
- 5 stocks (add ICICIBANK)
- Each time: verify convergence

**Phase 4:** Full multi-stock production
- All 5 stocks
- 200-300 episodes
- Production-ready

### If Single-Stock Fails:

**Option A:** Try DQN instead
- More robust to hyperparameters
- Off-policy learning
- Proven for trading

**Option B:** Redesign reward function
- Add drawdown penalty
- Add Sharpe ratio component
- Simplify to pure profit maximization

**Option C:** Different features
- Remove technical indicators
- Add momentum features
- Try different window sizes

---

## Monitoring

### Check convergence every 30 episodes:

```bash
# Episodes 1-30
grep "Episode.*/" training_150ep_SINGLESTOCK_*.log | sed -n '1,30p' | awk '{print $8}' | awk -F'|' '{print $1}'

# Episodes 31-60
grep "Episode.*/" training_150ep_SINGLESTOCK_*.log | sed -n '31,60p' | awk '{print $8}' | awk -F'|' '{print $1}'

# etc.
```

**What to look for:**
- Trade counts should DECREASE over time
- If stuck at same level after 60 episodes → Stop and pivot

---

## Risk Management

**Compute budget:** 18-30 hours
**API limits:** Minimal (yfinance only once at start)
**Disk space:** ~20MB log file

**If training fails:**
- Only lost 18 hours
- Learned PPO doesn't work for this problem
- Can pivot to DQN quickly

**If training succeeds:**
- Validated PPO works
- Have single-stock baseline
- Can scale to multi-stock systematically

---

## Why This Is The Right Approach

**30 years of experience says:**

1. **Always isolate variables** in research
2. **Start simple, add complexity** gradually
3. **Multi-stock failed** → Go back to single-stock
4. **Can't debug 5 stocks** → Debug 1 stock first
5. **Fast iteration** beats slow optimization

**This is textbook scientific method.**

---

## Expected Outcome

**Probability estimates:**

- **60%:** Single-stock converges, proves multi-stock was the issue
- **30%:** Single-stock also fails, need to fix PPO hyperparameters further or switch algorithm
- **10%:** Something else (bug, data issue, etc.)

**Either way, we'll have clarity in 18 hours instead of guessing.**

---

## Final Checklist

✅ Single-stock configuration (RELIANCE.NS only)
✅ Optimized PPO hyperparameters (3× faster learning)
✅ 150 episodes (enough to prove convergence)
✅ Training command ready
✅ Success criteria defined
✅ Fallback plan if it fails

**Ready to train. This is the right move.**

---

**Start when ready. Come back in 18 hours for results.**
