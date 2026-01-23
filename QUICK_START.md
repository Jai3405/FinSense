# SPIKE Terminal - Quick Start

## The Dashboard is Currently Running!

**Open this URL in your browser:**
http://localhost:5001

(Port 5000 was in use, so it auto-selected 5001)

---

## What to Do Now

### 1. Open the Dashboard
- Open your browser
- Go to: **http://localhost:5001**
- You should see SPIKE Terminal with the slanted italic logo

### 2. Start the Simulation
- Click the green **START** button
- Watch the AI confidence bars animate (BUY/HOLD/SELL percentages)
- See real-time trades in the activity feed
- Watch charts fill with data

### 3. What Should Happen
Within 5 seconds you should see:
- ✓ AI Confidence bars animating with percentages
- ✓ Live price updating in header
- ✓ Portfolio value changing
- ✓ Activity feed showing BUY/SELL trades
- ✓ Charts filling from left to right
- ✓ Position details updating

---

## If Nothing Happens When You Click START

Open browser console (press F12) and look for errors. Common issues:

### Issue 1: "Config file not found"
**Already fixed** - I updated the path handling

### Issue 2: WebSocket not connecting
Check the browser console for connection errors

### Issue 3: Loading spinner stuck
Backend crashed - check terminal for Python errors

---

## Current Status

✓ Dashboard is running on port 5001
✓ Backend test passed
✓ All paths fixed (config.yaml, model file)
✓ All metrics added to backend response
✓ SPIKE branding complete
✓ Professional SVG icons added

---

## To Stop the Dashboard

Press `Ctrl+C` in the terminal where it's running

---

## To Restart

```bash
cd /Users/jay/FinSense-1
./start_dashboard.sh
```

It will find an available port automatically.

---

## What I Fixed

**Problem:** Dashboard wasn't updating when you clicked START

**Root Causes:**
1. Missing metrics in backend response (avg_cost, unrealized_pnl, etc.)
2. Config file path issue

**Solution:**
1. Added all missing metrics to the backend data payload
2. Fixed config path to use parent directory
3. Fixed model path to use parent directory

**Test:** Backend test now passes completely

---

## Try It Now

1. Open: http://localhost:5001
2. Click START
3. Watch the AI confidence bars - they should animate immediately
4. Let it run for 30 seconds

If the AI confidence bars show percentages and animate, **IT'S WORKING!**

That's the key feature for your presentation.
