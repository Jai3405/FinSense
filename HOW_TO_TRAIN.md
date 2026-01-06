# How to Run Training (Simple Guide)

## 🚀 The Simple Way

### ✨ Auto-Resume is ON

The training **automatically saves checkpoints every 10 episodes** and **resumes from where it left off** if interrupted!

- Computer crashes? Just restart → resumes from last checkpoint ✅
- Accidentally Ctrl+C? Just restart → picks up where you left off ✅
- No progress lost! Ever! ✅

### Option 1: Direct in Terminal (Easiest)

Just run this and watch it live:

```bash
./run_training.sh
```

You'll see:
```
Episode 1/300 | Profit: ₹123.45 | Trades: 10 | Sharpe: 0.123 | Epsilon: 1.000 | Loss: 45.123
Episode 2/300 | Profit: ₹234.56 | Trades: 12 | Sharpe: 0.234 | Epsilon: 0.998 | Loss: 40.567
Episode 3/300 | Profit: ₹345.67 | Trades: 8 | Sharpe: 0.345 | Epsilon: 0.996 | Loss: 38.234
...
```

**To stop:** Press `Ctrl+C`

---

## 🔍 Option 2: Run in Background + Monitor

If you want to do other things while training:

### Terminal 1 - Start Training:
```bash
nohup ./run_training.sh > training_output.log 2>&1 &
```

This runs training in the background.

### Terminal 2 - Watch Progress:
```bash
./watch_progress.sh
```

This shows live updates as training progresses.

**To stop watching:** Press `Ctrl+C` (training continues)

**To stop training:**
```bash
# Find the process
ps aux | grep train.py

# Kill it
kill <PID>
```

---

## 📊 What You'll See

### Data Loading (First 2-5 minutes):
```
Loading data...
Loading data for 5 tickers: ['RELIANCE.NS', 'TCS.NS', 'INFY.NS', 'HDFCBANK.NS', 'ICICIBANK.NS']
  ✓ RELIANCE.NS: 1260 points
  ✓ TCS.NS: 1260 points
  ✓ INFY.NS: 1260 points
  ✓ HDFCBANK.NS: 1260 points
  ✓ ICICIBANK.NS: 1260 points
Combined data: 6300 total points
Augmenting data with 2 noisy versions (noise=0.01)
Augmented data: 6300 → 18900 points
```

### Training Episodes (5-10 hours):
```
Episode 1/300 | Profit: ₹601.73 | Trades: 360 | Sharpe: -1.902 | Epsilon: 1.000 | Loss: 268.118
Episode 2/300 | Profit: ₹2154.72 | Trades: 342 | Sharpe: -0.376 | Epsilon: 0.998 | Loss: 350.558
Episode 3/300 | Profit: ₹1234.56 | Trades: 280 | Sharpe: -0.123 | Epsilon: 0.996 | Loss: 320.123
...
Episode 100/300 | Profit: ₹3456.78 | Trades: 150 | Sharpe: 0.567 | Epsilon: 0.670 | Loss: 45.234
...
Episode 200/300 | Profit: ₹5678.90 | Trades: 120 | Sharpe: 1.234 | Epsilon: 0.223 | Loss: 12.345
...
Episode 300/300 | Profit: ₹7890.12 | Trades: 100 | Sharpe: 1.567 | Epsilon: 0.050 | Loss: 5.678
```

### Completion:
```
============================================================
Training Complete!
Best profit: ₹8174.01
Final epsilon: 0.050
Final model saved to models/final_model.pt
TensorBoard logs saved to runs/RELIANCE.NS_20260104_123456
View with: tensorboard --logdir runs/
============================================================
```

---

## ⏱️ Timeline

| Time | Episode Range | What's Happening | Epsilon Range |
|------|--------------|------------------|---------------|
| **0-5 min** | Setup | Loading 5 stocks, augmentation | - |
| **0-2 hrs** | 1-100 | Random exploration | 1.0 → 0.67 |
| **2-4 hrs** | 101-200 | Learning patterns | 0.67 → 0.22 |
| **4-6 hrs** | 201-300 | Exploitation/refinement | 0.22 → 0.05 |

**Total:** ~5-10 hours (varies by machine)

---

## 🎯 What to Look For

### Good Signs ✅
- **Epsilon decreases gradually:** 1.0 → 0.998 → 0.996 → ... → 0.05
- **Loss trends downward:** 200 → 100 → 50 → 20 → 10
- **Profits become positive:** After ~100 episodes
- **Trades stabilize:** Not too many (500+), not too few (0-5)

### Bad Signs ❌
- **Epsilon stuck:** Stays at 1.0 or 0.01 (bug!)
- **Loss explodes:** Goes to 1000+ and stays there
- **No trades:** Agent never buys/sells
- **Too many trades:** 500+ trades per episode (overtrading)

---

## 📈 After Training

### 1. Check Results:
```bash
python evaluate.py --model models/best_model.pt --ticker RELIANCE.NS --split test
```

### 2. View TensorBoard:
```bash
tensorboard --logdir runs/
# Open http://localhost:6006 in browser
```

### 3. Test Live Trading (Paper Mode):
```bash
python live_trade.py \
    --model models/best_model.pt \
    --ticker RELIANCE.NS \
    --interval 5m \
    --paper-trade \
    --max-trades 10
```

---

## 🐛 Troubleshooting

### Training finishes too fast (<5 min):
**Problem:** Epsilon not decaying (bug)
**Check:** Look at Episode 1 - is epsilon already at 0.01?
**Fix:** Make sure [train.py](train.py:297) has `agent.decay_epsilon()`

### Out of memory:
**Problem:** Too much data
**Fix:** Reduce augmentation copies in config:
```yaml
augmentation_copies: 1  # Instead of 2
```

### Training crashes:
**Check logs:**
```bash
tail -50 logs/finsense.log
```

### Can't find models:
**Check directory:**
```bash
ls -lh models/
```

---

## 💡 Pro Tips

1. **Run overnight:** Start before bed, check in morning
2. **Use tmux/screen:** Terminal session survives disconnects
3. **Check every 50 episodes:** Quick sanity check
4. **Save backups:** Copy `models/` before retraining
5. **Compare results:** Keep old models to compare performance

---

## 🎓 Quick Commands Cheat Sheet

```bash
# Start training (foreground)
./run_training.sh

# Start training (background)
nohup ./run_training.sh > training.log 2>&1 &

# Watch progress
./watch_progress.sh

# Check if training is running
ps aux | grep train.py

# Kill training
kill $(pgrep -f train.py)

# View last 20 episodes
tail -20 logs/finsense.log | grep Episode

# Evaluate model
python evaluate.py --model models/best_model.pt

# TensorBoard
tensorboard --logdir runs/
```

---

**Ready to train? Just run:**
```bash
./run_training.sh
```

**And watch it happen live in your terminal! 🚀**
