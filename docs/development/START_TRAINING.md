# Quick Start - Training

## 🚀 Simple 2-Step Process

### Step 1: Activate Virtual Environment
```bash
source finsense_env/bin/activate
```

You'll see `(finsense_env)` appear in your terminal prompt.

### Step 2: Start Training (Background)
```bash
nohup ./run_training.sh > training.log 2>&1 &
```

You'll see something like `[1] 12345` - that's your training process ID.

---

## 📊 Watch Progress (Optional)

In a **new terminal window**:

### Option A: Activate env and watch
```bash
source finsense_env/bin/activate
./watch_progress.sh
```

### Option B: Just check the log
```bash
tail -f training.log
```

Press `Ctrl+C` to stop watching (training continues in background).

---

## ✅ That's It!

Training is now running for ~5-10 hours. You can:
- Close the terminal ✅
- Let it run overnight ✅
- Check progress anytime with `tail -f training.log`

---

## 🛑 To Stop Training

```bash
# Find the process
ps aux | grep train.py

# Kill it (use PID from step 2)
kill <PID>
```

Or kill all training:
```bash
pkill -f "train.py"
```

---

## 📝 Quick Commands

```bash
# Activate environment
source finsense_env/bin/activate

# Start training (background)
nohup ./run_training.sh > training.log 2>&1 &

# Watch progress
./watch_progress.sh

# Check last 20 lines
tail -20 training.log

# Check if done
grep "Training Complete" training.log
```

---

**Ready? Just copy-paste these two commands:**

```bash
source finsense_env/bin/activate
nohup ./run_training.sh > training.log 2>&1 &
```

Then optionally watch:
```bash
./watch_progress.sh
```

🚀 **Done!**
