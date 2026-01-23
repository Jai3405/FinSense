#!/bin/bash

clear

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                                                               ║"
echo "║         SPIKE TERMINAL - TAILWIND CSS VERSION                ║"
echo "║              READY FOR YOUR PRESENTATION                      ║"
echo "║                                                               ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "✓ Rebuilt with Tailwind CSS (no cache issues)"
echo "✓ Exact layout from WhatsApp screenshot"
echo "✓ All 4 metrics cards in ONE row"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Kill existing
lsof -ti:8000 | xargs kill -9 2>/dev/null
sleep 2

# Start
source finsense_env/bin/activate
cd dashboard

echo ""
echo "Starting server..."
python3 app_fastapi.py &
sleep 3

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║                  SERVER RUNNING!                              ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Open: http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop"
echo ""

wait
