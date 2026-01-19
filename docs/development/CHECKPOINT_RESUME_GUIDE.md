# Checkpoint & Resume Guide

## 🎯 The Problem You Solved

**Scenario:** Training crashes at episode 250/300
- **Without checkpoints:** Lose 8+ hours, start over ❌
- **With checkpoints:** Resume from episode 250, only lose ~20 minutes ✅

---

## ✅ How It Works (Automatic)

### Auto-Resume is ON by Default

When you run:
```bash
./run_training.sh
```

The system **automatically**:
1. Checks for existing checkpoints in `models/`
2. Finds the latest checkpoint (e.g., `model_ep250.pt`)
3. Loads that checkpoint
4. Resumes from episode 251/300

**No manual intervention needed!**

---

## 💾 What Gets Saved

### Every 10 Episodes:
```
models/
├── model_ep10.pt          # Episode 10 checkpoint
├── model_ep10.json        # Metadata (profit, epsilon, etc.)
├── model_ep20.pt          # Episode 20 checkpoint
├── model_ep20.json
├── model_ep30.pt
├── ...
├── model_ep290.pt         # Latest checkpoint
├── model_ep290.json
└── best_model.pt          # Best performing model
└── best_model.json
```

### Metadata includes:
- Episode number
- Profit/loss
- Sharpe ratio
- All performance metrics
- Epsilon value (for resume)

---

## 🔄 Resume Scenarios

### Scenario 1: Training Crashes
```bash
# You're at episode 250/300
# Computer crashes / power outage / accidental Ctrl+C

# Just run again
./run_training.sh

# Output:
# "Auto-resuming from episode 250: models/model_ep250.pt"
# "Resuming training from episode 251/300"
```

### Scenario 2: Intentional Stop
```bash
# Training at episode 150/300
# You press Ctrl+C to stop

# Come back later
./run_training.sh

# Resumes from episode 150
```

### Scenario 3: Overnight Training Interrupted
```bash
# Start before bed
nohup ./run_training.sh > training.log 2>&1 &

# Laptop died / WiFi dropped / process killed
# Next morning, just run again

./run_training.sh
# Picks up where it left off!
```

---

## 🎛️ Manual Resume Options

### Option 1: Auto-Resume (Default)
```bash
# Finds and loads latest checkpoint automatically
python train.py --auto-resume
```

### Option 2: Resume from Specific Checkpoint
```bash
# Resume from episode 200
python train.py --resume models/model_ep200.pt
```

### Option 3: Resume from "auto" keyword
```bash
# Same as --auto-resume
python train.py --resume auto
```

### Option 4: Start Fresh (Ignore Checkpoints)
```bash
# Don't use --auto-resume flag
python train.py --config config.yaml --verbose
# This will start from episode 0 even if checkpoints exist
```

---

## 🗑️ Clean Checkpoint Strategy

The system keeps only the **5 most recent** checkpoints to save disk space.

**Example:**
- Episodes 10, 20, 30, 40, 50, 60 completed
- Only keeps: ep20, ep30, ep40, ep50, ep60
- Deletes: ep10 (oldest)

**Plus:**
- `best_model.pt` is ALWAYS kept (best profit)

---

## 📊 Example Training Flow

### Fresh Start:
```
Episode 1/300   → Save to models/ (no checkpoint yet)
Episode 10/300  → Save checkpoint: model_ep10.pt ✅
Episode 20/300  → Save checkpoint: model_ep20.pt ✅
Episode 30/300  → Save checkpoint: model_ep30.pt ✅
...
Episode 150/300 → [CRASH! Power outage] ⚡
```

### Resume:
```bash
./run_training.sh
```

```
Auto-resuming from episode 150: models/model_ep150.pt ✅
Resuming training from episode 151/300
Episode 151/300 → Continue...
Episode 160/300 → Save checkpoint: model_ep160.pt ✅
...
Episode 300/300 → Training Complete! 🎉
```

**Time saved:** ~7 hours (would have lost all progress)

---

## 🧪 Testing Checkpoint Resume

### Quick Test (5 episodes):
```bash
# 1. Start small training
python train.py --config config.yaml --episodes 5 --auto-resume

# Output: Episodes 1-5 complete

# 2. Run again (should skip)
python train.py --config config.yaml --episodes 5 --auto-resume

# Output: "Auto-resuming from episode 5"
#         Training already complete!
```

### Proper Test (20 episodes):
```bash
# 1. Start 20 episode training
python train.py --config config.yaml --episodes 20 --auto-resume

# 2. After episode 10, press Ctrl+C

# 3. Check checkpoint exists
ls -lh models/model_ep10.*
# Should see: model_ep10.pt and model_ep10.json

# 4. Resume
python train.py --config config.yaml --episodes 20 --auto-resume

# Output: "Auto-resuming from episode 10"
#         "Resuming from episode 11/20"
```

---

## 🔍 Checking Checkpoint Status

### View All Checkpoints:
```bash
ls -lh models/model_ep*.pt
```

```
models/model_ep10.pt   95K
models/model_ep20.pt   95K
models/model_ep30.pt   95K
models/model_ep40.pt   95K
models/model_ep50.pt   95K
models/best_model.pt   95K
```

### View Latest Checkpoint Metadata:
```bash
cat models/model_ep290.json
```

```json
{
  "episode": 290,
  "metrics": {
    "total_profit": 5234.56,
    "sharpe_ratio": 1.234,
    "max_drawdown": -0.02,
    "total_trades": 45
  },
  "is_best": false,
  "metric_name": "total_profit",
  "metric_value": 5234.56
}
```

---

## 🚨 Edge Cases

### Case 1: Change Config Mid-Training
```bash
# Episode 150/300 complete
# You change config.yaml (e.g., increase episodes to 500)

./run_training.sh

# ✅ Resumes from 150, trains to 500 (new total)
```

### Case 2: Corrupt Checkpoint
```bash
# If checkpoint is corrupted, training will error

# Solution: Resume from earlier checkpoint
python train.py --resume models/model_ep140.pt

# Or start fresh
rm models/model_ep150.pt
./run_training.sh
```

### Case 3: Want Fresh Start
```bash
# Clear all checkpoints
rm models/model_ep*.pt models/model_ep*.json

# Or backup first
mv models models_backup_$(date +%Y%m%d)
mkdir models

# Then start fresh
./run_training.sh
```

---

## 💡 Best Practices

### 1. Let Checkpoints Work
✅ Don't worry about crashes - just restart
✅ Checkpoints save every 10 episodes automatically
✅ Auto-resume is intelligent

### 2. Check Progress
```bash
# View last checkpoint
ls -lt models/model_ep*.pt | head -1

# View training log
tail -20 logs/finsense.log
```

### 3. Backup Before Big Changes
```bash
# Before changing hyperparameters significantly
cp -r models models_backup_good_run
```

### 4. Clean Up Old Runs
```bash
# After successful training
mv models models_final_300ep
mkdir models
```

---

## 📝 Configuration

In `config.yaml`:
```yaml
checkpoints:
  save_frequency: 10     # Save every 10 episodes
  max_keep: 5            # Keep last 5 checkpoints
  save_best: true        # Always save best model
  metric_name: total_profit  # Metric to track for "best"
```

**Recommendation:** Keep defaults, they work well!

---

## 🎯 Summary

**What you get:**
- ✅ Auto-save every 10 episodes
- ✅ Auto-resume on restart
- ✅ Best model always saved
- ✅ Disk space managed (keeps 5 recent)
- ✅ No manual intervention needed

**How to use:**
```bash
# Just run this - everything else is automatic!
./run_training.sh
```

**If training stops:**
```bash
# Just run again - picks up where it left off!
./run_training.sh
```

**That's it! 🚀**

---

**Last Updated:** 2026-01-04
**Status:** Production-Ready
**Tested:** ✅ Works perfectly
