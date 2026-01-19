"""Verify multi-stock data split is correct"""
from data_loader.data_loader import DataLoader
from utils.config import load_config

config = load_config('config.yaml')
data_config = config.get_section('data')

loader = DataLoader(data_config)
data = loader.load_data()

print("="*70)
print("MULTI-STOCK DATA SPLIT VERIFICATION")
print("="*70)

# Get configured stocks
tickers = data_config.get('ticker')
print(f"\nConfigured stocks: {tickers}")
print(f"Multi-stock enabled: {data_config.get('multi_stock', False)}")

# Total data
total_points = len(data['close'])
print(f"\nTotal data points loaded: {total_points}")

# Expected per stock
if isinstance(tickers, list):
    expected_per_stock = total_points // len(tickers)
    print(f"Expected points per stock: ~{expected_per_stock}")
else:
    print("Single stock mode")

# Perform split
train_ratio = config.get('training.train_ratio', 0.7)
val_ratio = config.get('training.validation_ratio', 0.15)
test_ratio = 1 - train_ratio - val_ratio

print(f"\nSplit ratios: {train_ratio:.0%} / {val_ratio:.0%} / {test_ratio:.0%}")

train_data, val_data, test_data = loader.train_test_split(data, train_ratio, val_ratio)

print(f"\n{'='*70}")
print("SPLIT RESULTS:")
print(f"{'='*70}")
print(f"Train set:      {len(train_data['close']):,} points")
print(f"Validation set: {len(val_data['close']):,} points")
print(f"Test set:       {len(test_data['close']):,} points")
print(f"Total:          {len(train_data['close']) + len(val_data['close']) + len(test_data['close']):,} points")

# Verify per-stock split for multi-stock
if isinstance(tickers, list) and len(tickers) > 1:
    print(f"\n{'='*70}")
    print("PER-STOCK VERIFICATION:")
    print(f"{'='*70}")

    expected_train_per_stock = int(expected_per_stock * train_ratio)
    expected_val_per_stock = int(expected_per_stock * val_ratio)
    expected_test_per_stock = expected_per_stock - expected_train_per_stock - expected_val_per_stock

    print(f"\nExpected per stock:")
    print(f"  Train: {expected_train_per_stock} points per stock")
    print(f"  Val:   {expected_val_per_stock} points per stock")
    print(f"  Test:  {expected_test_per_stock} points per stock")

    print(f"\nActual (should match expected × {len(tickers)} stocks):")
    print(f"  Train: {len(train_data['close'])} total = {len(train_data['close'])//len(tickers)} per stock")
    print(f"  Val:   {len(val_data['close'])} total = {len(val_data['close'])//len(tickers)} per stock")
    print(f"  Test:  {len(test_data['close'])} total = {len(test_data['close'])//len(tickers)} per stock")

    # Check if test set contains data from ALL stocks
    print(f"\n{'='*70}")
    print("CRITICAL CHECK: Does test set contain ALL stocks?")
    print(f"{'='*70}")

    # Sample test set prices to see diversity
    test_prices = test_data['close']
    print(f"\nTest set price range: ₹{min(test_prices):.2f} - ₹{max(test_prices):.2f}")
    print(f"Test set price std dev: ₹{test_prices.std():.2f}")

    # Show first and last 5 prices to verify mix
    print(f"\nFirst 10 test prices: {test_prices[:10].round(2).tolist()}")
    print(f"Last 10 test prices:  {test_prices[-10:].round(2).tolist()}")

    if len(set([round(p, -2) for p in test_prices])) >= len(tickers):
        print("\n✅ PASS: Test set appears to contain multiple stocks (diverse prices)")
    else:
        print("\n❌ FAIL: Test set may only contain one stock (similar prices)")

print(f"\n{'='*70}")
print("VERIFICATION COMPLETE")
print(f"{'='*70}\n")
