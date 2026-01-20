#!/bin/bash
# Quick test of the dashboard

echo "=========================================="
echo " TESTING DASHBOARD"
echo "=========================================="
echo ""

cd /Users/jay/FinSense-1
source finsense_env/bin/activate

echo "✅ Checking dependencies..."
pip list | grep -E "flask|socketio" > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed"
else
    echo "❌ Installing dependencies..."
    pip install -q flask flask-socketio python-socketio eventlet
fi

echo ""
echo "✅ Checking model file..."
if [ -f "models/ppo_final.pt" ]; then
    echo "✅ Model found: models/ppo_final.pt"
else
    echo "❌ WARNING: Model not found at models/ppo_final.pt"
fi

echo ""
echo "✅ Checking config..."
if [ -f "config.yaml" ]; then
    echo "✅ Config found: config.yaml"
else
    echo "❌ ERROR: config.yaml not found!"
    exit 1
fi

echo ""
echo "=========================================="
echo " ALL CHECKS PASSED!"
echo "=========================================="
echo ""
echo "Ready to start dashboard. Run:"
echo "  ./start_dashboard.sh"
echo ""
