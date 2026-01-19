# Clear Action Plan - What To Do Next

## Current Situation (Plain English)

**Where you are:**
- Branch: `experimental/reward-tuning`
- Fixed: Core reward bug (agent now trades!)
- Status: 10-episode test PASSED (106 trades on validation)

**What's working:**
- DQN agent with fixed rewards
- PPO agent exists but not tested with fixed rewards yet

**What's NOT working yet:**
- Haven't tested on final TEST SET (the ultimate proof)
- Haven't done full training (only 10 episodes so far)

---

## The Simple 3-Step Plan

### Step 1: Verify DQN Works (Today - 2 hours)

**Run 50-episode training with fixed DQN:**
```bash
cd /Users/jay/FinSense-1
source finsense_env/bin/activate
python train.py --config config.yaml --episodes 50 --verbose > training_DQN_FIXED_50ep.log 2>&1 &
```

**What to expect:**
- Training episodes: 500-700 trades/episode
- Validation episodes: 80-120 trades
- Loss decreasing

**How to monitor:**
```bash
tail -f training_DQN_FIXED_50ep.log
# Press Ctrl+C to stop watching
```

**When it finishes (1.5 hours):**
```bash
# Test on held-out TEST SET (the final proof)
python manual_eval.py
```

**Success criteria:**
- Test set trades: > 50 (proves dead policy is fixed)
- Trades < 300 (proves agent is selective, not random)
- Profit: any (don't care about profit yet, just that it trades)

---

### Step 2: If DQN Works, Test PPO (Tomorrow - 2 hours)

**Why test PPO:**
- PPO might be better for trading (on-policy, more stable)
- We already implemented it (Gemini's work)
- Should take advantage of the fixed rewards

**Run 50-episode PPO training:**
```bash
python train_ppo.py --config config.yaml --episodes 50 --verbose > training_PPO_FIXED_50ep.log 2>&1 &
```

**Then evaluate:**
```bash
python evaluate_ppo.py
```

**Compare DQN vs PPO:**
- Which trades more sensibly?
- Which has better Sharpe ratio?
- Which is more stable?

**Pick the winner and use that for full training.**

---

### Step 3: Full Training with Winner (This Week - 1 day)

**Once you know which works better (DQN or PPO):**

```bash
# If DQN wins:
python train.py --config config.yaml --episodes 200 --verbose > training_FINAL_200ep.log 2>&1 &

# If PPO wins:
python train_ppo.py --config config.yaml --episodes 200 --verbose > training_PPO_FINAL_200ep.log 2>&1 &
```

**Final evaluation:**
```bash
# Test on held-out test set
python manual_eval.py  # For DQN
# OR
python evaluate_ppo.py  # For PPO
```

**Success = Deployable Agent:**
- Test trades: 50-200
- Sharpe ratio: > 0.3
- Max drawdown: < 20%
- Win rate: > 48%

---

## What To Do About The Branches

### Option A: Keep Experimenting (Recommended for now)

**Stay on `experimental/reward-tuning` until we prove it works:**
- Run Step 1 (DQN 50 episodes)
- If test set shows 50+ trades → IT WORKED
- Then merge to main

### Option B: Create Clean Branch

**If you want a fresh start with just the fix:**
```bash
# Create new branch from main
git checkout main
git checkout -b fix/percentage-rewards

# Cherry-pick just the reward fix
git cherry-pick <commit-hash-of-my-fix>

# Test from clean slate
```

**I recommend Option A** - stay on experimental, prove it works, then merge.

---

## What About All Gemini's Changes?

**Good stuff to keep:**
1. ✅ Validation-driven checkpointing (saves models that actually trade)
2. ✅ PPO implementation (worth testing)
3. ✅ Action masking (prevents invalid trades)
4. ✅ Better logging and monitoring

**Stuff that doesn't matter anymore:**
1. ❌ Dueling DQN (reward bug was the problem, not architecture)
2. ❌ Trend features (reward bug was the problem, not features)
3. ❌ Complex reward bonuses (removed them in the fix)

**Decision:** Keep everything for now, test DQN with fixed rewards, THEN decide what to clean up.

---

## Simple Decision Tree

```
┌─────────────────────────┐
│ Run DQN 50 episodes     │
│ (Step 1)                │
└───────────┬─────────────┘
            │
            ▼
    ┌───────────────┐
    │ Test set eval │
    └───────┬───────┘
            │
     ┌──────┴──────┐
     │             │
     ▼             ▼
Trades > 50    Trades < 50
     │             │
     ▼             ▼
  SUCCESS!      Adjust idle_penalty
     │          (increase to 0.03)
     │             │
     ▼             └──────┐
Test PPO                  │
(Step 2)                  │
     │                    │
     ▼                    │
Compare DQN vs PPO        │
     │                    │
     ▼                    │
Pick winner              ▼
     │              Retry DQN 50ep
     ▼                    │
Full 200ep training       │
(Step 3)                  │
     │                    │
     ▼                    ▼
 DONE! 🎉          Back to start
```

---

## What I Need From You (Clear Instructions)

**Right now, tell me:**

1. **Do you want me to start Step 1?** (Run DQN 50 episodes)
   - I can kick it off right now
   - Takes 1.5 hours to complete
   - Then we evaluate on test set

2. **Do you want to wait and watch?**
   - I can explain how to monitor it yourself
   - You run the commands when you're ready

3. **Do you want me to clean up branches first?**
   - Merge to main
   - Or create a clean branch
   - Then proceed

**Just say:**
- "Start Step 1" → I'll launch 50-episode DQN training
- "Explain how to monitor" → I'll teach you the commands
- "Clean up first" → I'll organize the branches

**Whatever you choose, I'll handle it step by step.**

---

## The Bottom Line (Super Simple)

**What was broken:** Reward math was wrong (rupees vs percentages)

**What I fixed:** Made everything percentages

**What works now:** Agent trades (106 trades, not 5!)

**What's next:** Prove it on test set, then full training

**End goal:** Deployable trading agent with positive Sharpe

**Current branch:** experimental/reward-tuning (safe to experiment)

**Main branch:** Still has old code (safe fallback)

---

**I'm ready to execute whatever you decide. Just say the word.** 🚀
