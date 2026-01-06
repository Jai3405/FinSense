#!/bin/bash
# Watch training progress in real-time
# Run this in a SEPARATE terminal while training runs

echo "======================================================================"
echo "  FinSense Training Progress Monitor"
echo "======================================================================"
echo ""

# Check if training.log exists (from nohup), otherwise use finsense.log
if [ -f "training.log" ]; then
    LOG_FILE="training.log"
else
    LOG_FILE="logs/finsense.log"
fi

echo "Watching: $LOG_FILE"
echo "Press Ctrl+C to stop watching (training will continue)"
echo ""
echo "======================================================================"
echo ""

# Watch the log file in real-time
tail -f "$LOG_FILE" | grep --line-buffered "Episode"
