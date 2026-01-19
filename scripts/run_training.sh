#!/bin/bash
# Simple training starter - outputs directly to terminal
# Automatically resumes from latest checkpoint if available

echo "======================================================================"
echo "  Starting FinSense Training - 300 Episodes"
echo "======================================================================"
echo ""
echo "Configuration:"
echo "  - 300 episodes"
echo "  - 5 stocks (RELIANCE, TCS, INFY, HDFC, ICICI)"
echo "  - Multi-stock training enabled"
echo "  - Data augmentation enabled"
echo "  - Window size: 20 days"
echo "  - Auto-resume: ON (resumes from latest checkpoint)"
echo ""
echo "Expected duration: 5-10 hours"
echo ""
echo "Checkpoints saved every 10 episodes to models/"
echo "Press Ctrl+C to stop training at any time (progress is saved)"
echo "======================================================================"
echo ""

# Activate virtual environment
source finsense_env/bin/activate

# Run training with auto-resume enabled
python train.py --config config.yaml --verbose --auto-resume
