#!/bin/bash
# Live training log viewer with timestamps

LOG_FILE=${1:-"training.log"}

echo "========================================"
echo "Training Monitor - Live Updates"
echo "Log file: $LOG_FILE"
echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
echo ""

# Function to add timestamp to each line
tail -f "$LOG_FILE" | while read line; do
    echo "[$(date '+%H:%M:%S')] $line"
done
