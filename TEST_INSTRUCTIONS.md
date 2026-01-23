# DEBUG INSTRUCTIONS FOR SPIKE TERMINAL

## The Problem
You clicked START but nothing is updating on the dashboard - no live trades, no AI confidence bars animating, nothing.

## What I've Done
1. Created a debug version of the app with extensive logging
2. The debug app is running on **port 5001**
3. The backend API is working (I tested it with curl - it returns success)
4. Created a simple WebSocket test page

## What You Need To Do

### Step 1: Test WebSocket Connection

Open in your browser:
**http://localhost:5001/test_socket**

This is a simple test page I created.

1. You should see "Connected" in green
2. Click the "Start Simulation" button
3. Watch the log area

**Expected result:**
You should see messages like:
```
✓ Socket.IO connected
Starting test simulation...
✓ Simulation started successfully
📊 Trading update received: Step=0, Price=2450.75, Action=BUY
   AI Confidence: BUY=78.5% HOLD=15.2% SELL=6.3%
📊 Trading update received: Step=1, Price=2455.30, Action=HOLD
   AI Confidence: BUY=12.1% HOLD=85.3% SELL=2.6%
...
```

### Step 2: Check Results

**If you see trading updates on the test page:**
- The WebSocket is working
- The backend is working
- The problem is in the main dashboard's JavaScript

**If you DON'T see trading updates:**
- Check browser console (F12) for errors
- Tell me what errors you see

### Step 3: Check Main Dashboard

After testing, go back to the main dashboard:
**http://localhost:5001**

Open browser console (press F12) and look for:
1. Any red error messages
2. "Connected to FinSense Terminal" message
3. Any WebSocket connection errors

## Common Issues

### Issue 1: "ERR_CONNECTION_REFUSED"
The server isn't running. Run:
```bash
cd /Users/jay/FinSense-1
pkill -f app_debug
source finsense_env/bin/activate
cd dashboard
python3 app_debug.py
```

### Issue 2: Socket.IO not connecting
Browser console shows "transport error" or similar. This means Socket.IO isn't configured correctly.

### Issue 3: No console messages at all
JavaScript isn't loading. Check Network tab in F12 for failed file loads.

## What To Report Back

Tell me:
1. What do you see on http://localhost:5001/test_socket when you click "Start Simulation"?
2. Do you see trading updates in the log?
3. What errors (if any) show in browser console (F12)?
4. Does the browser console show "Connected to FinSense Terminal"?

This will help me identify exactly where the problem is.
