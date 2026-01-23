# SPIKE Terminal - Pre-Presentation Test Guide

## Quick Start Test (2 Minutes)

### Step 1: Start the Dashboard
```bash
cd /Users/jay/FinSense-1
./start_dashboard.sh
```

**Expected Output:**
```
==========================================
 SPIKE TERMINAL - CAPSTONE PRESENTATION
==========================================

Installing dashboard dependencies...
Starting SPIKE Terminal server...

Dashboard will open on: http://localhost:5000
Open this URL in your browser

Press Ctrl+C to stop the server
==========================================
```

### Step 2: Open Your Browser
Open: **http://localhost:5000**

---

## Visual Inspection Checklist

### Header (Top Bar)
- [ ] Logo shows "SPIKE" in slanted italic style with tight letter spacing
- [ ] Subtitle shows "TERMINAL" in monospace font
- [ ] RELIANCE.NS ticker displays "₹0.00"
- [ ] Portfolio shows "₹50,000"
- [ ] P&L shows "₹0.00"
- [ ] Status shows "System Idle" with gray dot

### Metric Cards (4 Cards in Center-Top)
- [ ] Portfolio Value card has dashboard grid icon (top-right)
- [ ] Sharpe Ratio card has trending up icon
- [ ] Win Rate card has checkmark icon
- [ ] Max Drawdown card has trending down icon
- [ ] All cards show "₹50,000", "0.000", "0.0%", "0.00%" respectively
- [ ] Icons are subtle (60% opacity) and scale up on hover

### Charts (Center)
- [ ] Portfolio Equity Curve has line chart icon next to title
- [ ] Agent Actions has action icon next to title
- [ ] Both charts are empty (will fill during simulation)

### Right Panel
- [ ] "Live Activity" section with badge showing "0"
- [ ] "AI Confidence" panel with three bars (BUY/HOLD/SELL at 0%)
- [ ] "Current Position" section showing RELIANCE.NS details

### Left Panel
- [ ] Simulation Controls with date inputs
- [ ] Start Date: 2024-01-01
- [ ] End Date: 2024-03-31
- [ ] Initial Capital: ₹50,000
- [ ] Green "START" button
- [ ] Gray disabled "STOP" button
- [ ] Market Statistics showing all zeros
- [ ] "Export CSV" button at bottom

---

## Functional Test (30 Seconds)

### Test 1: Start Simulation

1. **Click the green "START" button**

**Expected behavior:**
- Loading spinner appears with "Initializing AI Agent..." message
- Spinner disappears after 1-2 seconds
- Status dot turns green
- Status text changes to "Running"
- START button becomes disabled
- STOP button becomes enabled

### Test 2: Watch Real-Time Updates

**Within 5 seconds, you should see:**

**Header Updates:**
- [ ] Live price changes from ₹0.00 to actual price (e.g., ₹2,450.75)
- [ ] Portfolio value updates in real-time
- [ ] P&L shows positive or negative value with color

**Metric Cards:**
- [ ] Portfolio Value increases/decreases
- [ ] Percentage change shows in green or red
- [ ] Sharpe Ratio updates (starts at 0.000, increases slowly)
- [ ] Win Rate updates after first trades
- [ ] Max Drawdown shows percentage

**AI Confidence Bars (KEY FEATURE):**
- [ ] BUY bar animates to show percentage (e.g., 78.5%)
- [ ] HOLD bar animates to show percentage (e.g., 15.2%)
- [ ] SELL bar animates to show percentage (e.g., 6.3%)
- [ ] Bars have shimmer animation effect
- [ ] Percentages add up to 100%
- [ ] Bar colors: Green (BUY), Orange (HOLD), Red (SELL)

**Activity Feed:**
- [ ] New entries appear at top
- [ ] BUY actions show in green
- [ ] SELL actions show in red
- [ ] HOLD actions show in orange
- [ ] Each entry shows timestamp and action

**Portfolio Equity Chart:**
- [ ] Line graph fills from left to right
- [ ] Shows portfolio value over time
- [ ] Green gradient fill under line

**Agent Actions Chart:**
- [ ] Bar chart shows BUY/HOLD/SELL counts
- [ ] Updates as agent takes actions

**Current Position:**
- [ ] Stock shows RELIANCE.NS
- [ ] Price updates in real-time
- [ ] Inventory shows share count
- [ ] Avg Cost shows entry price when position is opened
- [ ] Unrealized P&L shows in green (profit) or red (loss)
- [ ] Realized P&L updates when shares are sold

### Test 3: Let It Run (15-20 seconds)

Just watch it work. You should see:
- Activity feed scrolling with new trades
- AI confidence bars changing every step
- Portfolio value fluctuating
- Charts filling with data
- Position details updating

### Test 4: Export Data

1. **Scroll down left panel and click "Export CSV"**

**Expected behavior:**
- [ ] File downloads automatically
- [ ] File name: `finsense_trades_YYYY-MM-DD_HHMMSS.csv`
- [ ] File contains columns: Timestamp, Price, Portfolio Value, Action, P&L
- [ ] File has data for all steps so far

### Test 5: Stop Simulation

1. **Click the "STOP" button**

**Expected behavior:**
- [ ] Status changes to "System Idle"
- [ ] Status dot turns gray
- [ ] Updates stop
- [ ] START button becomes enabled again
- [ ] STOP button becomes disabled

---

## Performance Expectations

After a full simulation (2024-01-01 to 2024-03-31):

**Expected Final Metrics:**
- Portfolio Value: ₹50,000 - ₹52,000 (varies by market)
- Return: +0.3% to +4%
- Sharpe Ratio: 0.20 - 0.35
- Win Rate: 50% - 75%
- Max Drawdown: -0.5% to -2%
- Total Trades: 15-30

**Note:** Exact numbers vary based on PPO model and market data.

---

## Troubleshooting

### Issue: "Error: Config file not found"
**Solution:** Make sure you're running from the FinSense-1 directory:
```bash
cd /Users/jay/FinSense-1
./start_dashboard.sh
```

### Issue: Port 5000 already in use
**Solution:** The app will automatically try ports 5001-5009. Check the terminal output for the actual port.

### Issue: AI confidence bars not updating
**Solution:** Check browser console (F12) for errors. Refresh the page and try again.

### Issue: Charts not rendering
**Solution:** Ensure you have internet connection (Chart.js loads from CDN).

### Issue: Loading spinner stuck
**Solution:** Backend likely crashed. Check terminal for errors. Common issue: model file not found.

---

## Quick Debug Commands

**Check if config exists:**
```bash
ls -la /Users/jay/FinSense-1/config.yaml
```

**Check if model exists:**
```bash
ls -la /Users/jay/FinSense-1/models/ppo_final.pt
```

**Check if virtual env is activated:**
```bash
which python3
# Should show: /Users/jay/FinSense-1/finsense_env/bin/python3
```

**Restart dashboard:**
```bash
# Press Ctrl+C to stop
./start_dashboard.sh
```

---

## Presentation Demo Script

### Opening (10 seconds)
"This is SPIKE Terminal - an enterprise AI trading platform powered by reinforcement learning. What you're seeing is a trained PPO agent making real-time trading decisions."

### Start Simulation (5 seconds)
Click START button.

"Notice the loading state - production-ready user experience."

### During Simulation (30 seconds)
**Point to AI Confidence bars:**
"Here's the killer feature - real-time AI decision probabilities. The agent is 78% confident to BUY right now. This isn't a black box - you can see the agent's reasoning."

**Point to Portfolio Equity Curve:**
"Portfolio value updating live as trades execute."

**Point to Activity Feed:**
"Every BUY and SELL decision logged with timestamps."

**Point to Current Position:**
"Complete position tracking - average cost, unrealized P&L, realized P&L. Professional-level metrics."

### Export Data (5 seconds)
Click "Export CSV"

"All trading data exportable for post-analysis. Production-ready."

### Key Talking Points
1. "AI confidence visualization shows probabilistic decision-making"
2. "450 episodes of PPO training achieved 0.29 Sharpe ratio"
3. "Complete error handling with auto-reconnection"
4. "This could be deployed to paper trading tomorrow"

---

## What Evaluators Will Notice

### Visual Impact
- Bloomberg Terminal-level professional design
- No emojis - serious enterprise software
- Stylized SPIKE logo with dynamic slant
- Cohesive color palette throughout

### Technical Depth
- Real-time WebSocket communication
- AI confidence visualization (unique feature)
- Complete trading metrics
- Professional error handling

### Completeness
- Full system: data → AI → execution → reporting
- Export capability
- Loading states
- Responsive design

### Production Quality
- Clean, polished UI
- Smooth animations
- No obvious bugs or incomplete features
- Professional terminology and metrics

---

## Pre-Presentation Final Check (Morning Of)

**30 minutes before:**
```bash
cd /Users/jay/FinSense-1
./start_dashboard.sh
```

1. Open http://localhost:5000
2. Click START
3. Watch for 10 seconds
4. Verify:
   - [ ] AI confidence bars animate
   - [ ] Charts fill with data
   - [ ] Activity feed updates
   - [ ] Position details update
   - [ ] No console errors (F12)
5. Click STOP
6. Click "Export CSV"
7. Verify CSV downloads and has data
8. Close dashboard (Ctrl+C in terminal)

**10 minutes before:**
```bash
./start_dashboard.sh
```

1. Open http://localhost:5000
2. Press F11 for fullscreen
3. Pre-fill dates: 2024-01-01 to 2024-03-31
4. Pre-fill capital: ₹50,000
5. Don't click START yet - wait for presentation

**During presentation:**
1. Click START
2. Let run for 30 seconds
3. Point out features as they update
4. Click "Export CSV"
5. Show final metrics

---

## Success Criteria

Your test is successful if:
- [ ] Dashboard loads without errors
- [ ] SPIKE logo displays correctly with slant
- [ ] All 6 SVG icons visible (4 metrics + 2 charts)
- [ ] START button works
- [ ] AI confidence bars animate smoothly
- [ ] Charts fill with data
- [ ] Activity feed updates
- [ ] Position tracking works (avg cost, P&L)
- [ ] Export CSV downloads valid data
- [ ] STOP button works
- [ ] No emojis anywhere
- [ ] Professional appearance throughout

---

## Backup Plan

If dashboard fails during presentation:

**Option 1: Screenshots**
Take screenshots now of:
1. Dashboard before simulation
2. AI confidence bars in action
3. Mid-simulation with trades
4. Final results
5. Exported CSV in Excel

**Option 2: Video Recording**
Record a 1-minute video of the simulation running.

**Option 3: Terminal Output**
Show the backend running with:
```bash
python3 test_paper_trading.py
```

---

## Confidence Boosters

**You built:**
- A working AI trading agent (74% backtest win rate)
- Enterprise-grade web interface
- Real-time visualization system
- Production-ready infrastructure

**This is:**
- A-grade capstone work
- Portfolio-worthy project
- Actually deployable system
- Commercially viable (with validation)

**Expected reactions:**
- "This looks professional"
- "The AI confidence bars are impressive"
- "How long did this take?"
- "Are you going to deploy it?"

**Remember:**
- You know this system inside and out
- The work is solid (0.29 Sharpe, 74% win rate)
- Results speak for themselves
- It's okay to be proud

---

## Final Words

Test it right now. Run through the entire checklist. If everything works, you're ready.

The AI confidence visualization is your differentiator. When evaluators see those probability bars updating in real-time, they'll understand this is real machine learning.

**You got this.**
