#!/usr/bin/env python3
"""
Start training with live monitoring (cross-platform).
"""

import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
import os


def start_training_with_monitor():
    """Start training and monitor in a user-friendly way."""

    print("=" * 80)
    print(" " * 20 + "🚀 FinSense Training - 100 Episodes")
    print("=" * 80)
    print()

    # Create logs directory
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)

    # Generate log filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'training_100ep_{timestamp}.log'

    print(f"📝 Log file: {log_file}")
    print()

    # Start training in background
    print("🚀 Starting training in background...")

    # Open log file
    log_handle = open(log_file, 'w')

    # Start training process
    train_process = subprocess.Popen(
        [sys.executable, 'train.py', '--config', 'config.yaml', '--verbose'],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        cwd=Path.cwd()
    )

    print(f"✓ Training started (PID: {train_process.pid})")
    print()

    # Wait for log file to have content
    print("⏳ Waiting for training to initialize...")
    time.sleep(3)

    # Start monitor
    print("📊 Starting live monitor...")
    print("   (Press Ctrl+C to stop monitoring - training will continue)")
    print()
    time.sleep(1)

    try:
        # Run monitor
        monitor_process = subprocess.run(
            [sys.executable, 'monitor_training.py', '--log', str(log_file)],
            cwd=Path.cwd()
        )

    except KeyboardInterrupt:
        print("\n\nMonitoring stopped by user")

    # Monitor exited
    print()
    print("=" * 80)
    print("✓ Monitoring stopped")
    print()

    # Check if training is still running
    if train_process.poll() is None:
        print(f"Training continues in background (PID: {train_process.pid})")
        print()
        print("To check progress again:")
        print(f"  python monitor_training.py --log {log_file}")
        print()
        print("To view in TensorBoard:")
        print("  tensorboard --logdir runs/")
        print()
        print("To stop training:")
        if sys.platform == 'win32':
            print(f"  taskkill /PID {train_process.pid} /F")
        else:
            print(f"  kill {train_process.pid}")
    else:
        print("Training has finished!")
        print()
        print("Check results in:")
        print(f"  - Log: {log_file}")
        print("  - Models: models/")
        print("  - TensorBoard: runs/")

    print("=" * 80)

    # Close log file
    log_handle.close()


if __name__ == '__main__':
    try:
        start_training_with_monitor()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
