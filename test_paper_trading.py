"""
Test Paper Trading System

Quick test to validate the paper trading system works end-to-end.
"""

import sys
from pathlib import Path

# Test on historical data (faster than live)
test_start = '2024-01-01'
test_end = '2024-03-31'  # 3 months test

print("="*80)
print(" TESTING PAPER TRADING SYSTEM")
print("="*80)
print(f"Test Period: {test_start} to {test_end}")
print(f"Mode: Historical Simulation (Fast)")
print("="*80)
print()

# Run paper trading in simulate mode
sys.argv = [
    'test_paper_trading.py',
    '--mode', 'simulate',
    '--start-date', test_start,
    '--end-date', test_end,
    '--ticker', 'RELIANCE.NS',
    '--interval', '1d',
    '--balance', '50000',
    '--verbose'
]

# Import and run main
from paper_trading_main import main

if __name__ == '__main__':
    sys.exit(main())
