#!/bin/bash

clear

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║           SPIKE TERMINAL - PRESENTATION READY                 ║"
echo "║                  FINAL VERSION TEST                           ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "WHAT WAS FIXED:"
echo "  ✓ Rebuilt with Tailwind CSS (bypasses browser cache)"
echo "  ✓ Added ALL visual styling from WhatsApp screenshot"
echo "  ✓ Gradient backgrounds on all cards"
echo "  ✓ Hover effects with glow"
echo "  ✓ Activity feed slide-in animations"
echo "  ✓ Confidence bar gradients"
echo "  ✓ Button press effects"
echo "  ✓ Custom fonts (Inter + JetBrains Mono)"
echo "  ✓ Loading spinner"
echo "  ✓ All transitions and effects"
echo "  ✓ WebSocket functionality INTACT"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Kill any existing server
echo "1. Stopping any running servers..."
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2
echo "   ✓ Port 8000 cleared"
echo ""

# Activate environment
echo "2. Activating Python environment..."
cd /Users/jay/FinSense-1
source finsense_env/bin/activate
cd dashboard
echo "   ✓ Environment activated"
echo ""

# Start server
echo "3. Starting FastAPI server..."
echo ""
python3 app_fastapi.py &
SERVER_PID=$!
sleep 3
echo ""

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                     SERVER RUNNING!                           ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "URL: http://localhost:8000"
echo "PID: $SERVER_PID"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "CRITICAL: OPEN IN INCOGNITO/PRIVATE MODE"
echo ""
echo "Arc Browser:"
echo "  1. Press Cmd+Shift+N (opens incognito)"
echo "  2. Go to: http://localhost:8000"
echo ""
echo "Safari:"
echo "  1. File → New Private Window (or Cmd+Shift+N)"
echo "  2. Go to: http://localhost:8000"
echo ""
echo "Chrome:"
echo "  1. Press Cmd+Shift+N (opens incognito)"
echo "  2. Go to: http://localhost:8000"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "WHAT YOU SHOULD SEE:"
echo ""
echo "  ✓ 4 metrics cards in ONE horizontal row"
echo "  ✓ Dark gradient backgrounds"
echo "  ✓ Cards glow on hover"
echo "  ✓ Green START button with pulse effect"
echo "  ✓ Activity feed on the right"
echo "  ✓ AI Confidence panel below activity"
echo ""
echo "TESTING THE START BUTTON:"
echo ""
echo "  1. Click the green START button"
echo "  2. Button should show spinner"
echo "  3. Within 2-3 seconds you should see:"
echo "     - Chart populating with candlesticks"
echo "     - Metrics updating (Portfolio Value, Sharpe, Win Rate)"
echo "     - Activity feed showing BUY/SELL/HOLD actions"
echo "     - Confidence bars animating"
echo "  4. Everything updates every 0.5 seconds"
echo ""
echo "IF IT WORKS: You're 100% ready for your presentation!"
echo "IF IT DOESN'T: Take a screenshot and send it to me"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop server when done testing"
echo ""

wait $SERVER_PID
