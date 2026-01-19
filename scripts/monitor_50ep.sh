#!/bin/bash
# Quick monitoring script for 50-episode test

echo "=== TRAINING PROGRESS ==="
echo ""
echo "Recent episodes:"
tail -100 training_50ep_test.log | grep "Episode" | tail -5
echo ""
echo "=== TRADE COUNT ANALYSIS ==="
echo "Episodes with trade counts:"
grep "Trades:" training_50ep_test.log | tail -10
echo ""
echo "=== CHECKING FOR DEAD POLICY ==="
zero_trades=$(grep "Trades: 0" training_50ep_test.log | wc -l)
if [ $zero_trades -gt 5 ]; then
    echo "⚠️  WARNING: Found $zero_trades episodes with 0 trades"
    echo "Idle penalty may need to be increased"
else
    echo "✅ No dead policy detected (only $zero_trades episodes with 0 trades)"
fi
echo ""
echo "=== LATEST LOG TAIL ==="
tail -20 training_50ep_test.log
