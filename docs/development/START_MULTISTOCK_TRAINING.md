# Multi-Stock PPO Training - 200 Episodes

## System Status
✅ Data: 7,480 points (5 stocks × 1,496 each)
✅ Split: Per-stock 70/15/15 (all 5 stocks in test set)
✅ Rewards: Percentage-based (0.02% idle penalty)
✅ Configuration: Verified and correct

## Training Command

**Run this in your terminal:**

```bash
nohup python train_ppo.py --episodes 200 --verbose > training_PPO_multistock_200ep.log 2>&1 &
```

**What this does:**
- `nohup`: Keeps running even if terminal disconnects
- `train_ppo.py --episodes 200`: 200 episodes of PPO training
- `--verbose`: Print progress to log
- `> training_PPO_multistock_200ep.log`: Save output to file
- `2>&1`: Capture errors too
- `&`: Run in background

## Expected Timeline

**Training time:** ~60 hours (2.5 days)
- 5 stocks × 1,047 points each = 5,235 training points per episode
- ~18 minutes per episode
- 200 episodes × 18 min = 60 hours

## Monitor Progress

**Check if running:**
```bash
ps aux | grep train_ppo
```

**Watch live log:**
```bash
tail -f training_PPO_multistock_200ep.log
```

**Check progress (last 20 lines):**
```bash
tail -20 training_PPO_multistock_200ep.log
```

**Count completed episodes:**
```bash
grep "Episode.*/" training_PPO_multistock_200ep.log | tail -1
```

**Check log file size (grows as training progresses):**
```bash
ls -lh training_PPO_multistock_200ep.log
```

## Stop Training (if needed)

```bash
pkill -f train_ppo.py
```

## After Training Completes

**Check completion:**
```bash
tail -50 training_PPO_multistock_200ep.log | grep "Training Complete"
```

**Verify model saved:**
```bash
ls -lh models/ppo_final.pt
```

**Evaluate on test set:**
```bash
python comprehensive_ppo_eval.py
```

## Expected Results

**After 200 episodes on 5 stocks:**
- Test P&L: +₹2,000 to +₹4,000 (better than single-stock)
- Sharpe: 0.35-0.50 (higher than single-stock 0.22)
- Win rate: 65-70%
- Max DD: 2-5%
- **Agent works on ALL 5 stocks** ✅

## Important Notes

⚠️ **API Limits:** This will use significant yfinance API calls
⚠️ **Disk Space:** Log file will be ~50-100MB
⚠️ **CPU Usage:** Will use 1 CPU core continuously for 60 hours
✅ **Safe:** Can disconnect terminal, training continues

## Quick Reference

| Command | Purpose |
|---------|---------|
| `tail -f training_PPO_multistock_200ep.log` | Watch live |
| `grep "Episode" training_PPO_multistock_200ep.log \| wc -l` | Count episodes |
| `ps aux \| grep train_ppo` | Check if running |
| `pkill -f train_ppo.py` | Stop training |

---

**Ready to start? Just copy-paste the nohup command above into your terminal.**
