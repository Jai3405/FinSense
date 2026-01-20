#!/bin/bash
# Start FinSense Dashboard for Capstone Presentation

echo "=========================================="
echo " FINSENSE - CAPSTONE PRESENTATION"
echo "=========================================="
echo ""
echo "Installing dashboard dependencies..."
source finsense_env/bin/activate
pip install -q flask flask-socketio python-socketio eventlet

echo ""
echo "Starting dashboard server..."
echo ""
echo "🚀 Dashboard will open on: http://localhost:5000"
echo "📊 Open this URL in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

cd dashboard
python app.py
