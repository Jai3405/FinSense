#!/bin/bash
# Start FinSense training with live monitoring

set -e

echo "======================================================================"
echo "  FinSense Training - 100 Episodes"
echo "======================================================================"
echo ""

# Create logs directory
mkdir -p logs

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/training_100ep_${TIMESTAMP}.log"

echo "📝 Log file: $LOG_FILE"
echo ""

# Start training in background
echo "🚀 Starting training in background..."
nohup python train.py --config config.yaml --verbose > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

echo "✓ Training started (PID: $TRAIN_PID)"
echo ""

# Wait for log file to be created
sleep 2

# Start monitor in foreground
echo "📊 Starting live monitor..."
echo "   (Press Ctrl+C to stop monitoring - training will continue)"
echo ""
sleep 1

python monitor_training.py --log "$LOG_FILE"

echo ""
echo "======================================================================"
echo "✓ Monitoring stopped"
echo ""
echo "Training continues in background (PID: $TRAIN_PID)"
echo ""
echo "To check progress again:"
echo "  python monitor_training.py --log $LOG_FILE"
echo ""
echo "To view in TensorBoard:"
echo "  tensorboard --logdir runs/"
echo ""
echo "To stop training:"
echo "  kill $TRAIN_PID"
echo "======================================================================"
