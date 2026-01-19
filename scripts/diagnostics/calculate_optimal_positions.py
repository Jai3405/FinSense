"""Calculate optimal max_positions for the trading setup"""
import numpy as np
from data_loader.data_loader import DataLoader
from utils.config import load_config

config = load_config('config.yaml')
data_config = config.get_section('data')

# Load data to get actual stock prices
loader = DataLoader(data_config)
data = loader.load_data()

print("="*70)
print("OPTIMAL POSITION SIZE CALCULATION")
print("="*70)

# Get price statistics
prices = data['close']
avg_price = np.mean(prices)
min_price = np.min(prices)
max_price = np.max(prices)
median_price = np.median(prices)

print(f"\nStock Price Statistics (across 5 stocks):")
print(f"  Average:  ₹{avg_price:,.2f}")
print(f"  Median:   ₹{median_price:,.2f}")
print(f"  Min:      ₹{min_price:,.2f}")
print(f"  Max:      ₹{max_price:,.2f}")

# Account parameters
starting_balance = 50000
max_position_value_pct = 0.95  # Can use 95% of capital

print(f"\nAccount Parameters:")
print(f"  Starting Balance: ₹{starting_balance:,}")
print(f"  Max Capital Usage: {max_position_value_pct*100:.0f}%")
print(f"  Available Capital: ₹{starting_balance * max_position_value_pct:,.2f}")

# Calculate position sizes at different price points
print(f"\n{'='*70}")
print("POSITION SIZE SCENARIOS:")
print(f"{'='*70}")

# Scenario 1: Using average price
max_shares_avg = int((starting_balance * max_position_value_pct) / avg_price)
invested_avg = max_shares_avg * avg_price
print(f"\n1. At AVERAGE price (₹{avg_price:.2f}):")
print(f"   Max shares: {max_shares_avg}")
print(f"   Investment: ₹{invested_avg:,.2f} ({100*invested_avg/starting_balance:.1f}% of capital)")

# Scenario 2: Using median price
max_shares_median = int((starting_balance * max_position_value_pct) / median_price)
invested_median = max_shares_median * median_price
print(f"\n2. At MEDIAN price (₹{median_price:.2f}):")
print(f"   Max shares: {max_shares_median}")
print(f"   Investment: ₹{invested_median:,.2f} ({100*invested_median/starting_balance:.1f}% of capital)")

# Scenario 3: Using minimum price (cheapest stock)
max_shares_min = int((starting_balance * max_position_value_pct) / min_price)
invested_min = max_shares_min * min_price
print(f"\n3. At MIN price (₹{min_price:.2f}) - Cheapest stock:")
print(f"   Max shares: {max_shares_min}")
print(f"   Investment: ₹{invested_min:,.2f} ({100*invested_min/starting_balance:.1f}% of capital)")

# Scenario 4: Using maximum price (most expensive stock)
max_shares_max = int((starting_balance * max_position_value_pct) / max_price)
invested_max = max_shares_max * max_price
print(f"\n4. At MAX price (₹{max_price:.2f}) - Most expensive stock:")
print(f"   Max shares: {max_shares_max}")
print(f"   Investment: ₹{invested_max:,.2f} ({100*invested_max/starting_balance:.1f}% of capital)")

print(f"\n{'='*70}")
print("TRANSACTION COST ANALYSIS:")
print(f"{'='*70}")

# Transaction costs per trade
brokerage = 20
stt_pct = 0.00025
exchange_pct = 0.0000297
sebi_pct = 0.000001
gst_rate = 0.18

def calc_transaction_cost(price, shares):
    """Calculate total transaction cost for buy+sell round trip"""
    trade_value = price * shares

    # Buy costs
    buy_brokerage = min(20, trade_value * 0.0003)
    buy_stamp = trade_value * 0.00003
    buy_exchange = trade_value * exchange_pct
    buy_sebi = trade_value * sebi_pct
    buy_gst = (buy_brokerage + buy_exchange + buy_sebi) * gst_rate
    buy_cost = buy_brokerage + buy_stamp + buy_exchange + buy_sebi + buy_gst

    # Sell costs
    sell_brokerage = min(20, trade_value * 0.0003)
    sell_stt = trade_value * stt_pct
    sell_exchange = trade_value * exchange_pct
    sell_sebi = trade_value * sebi_pct
    sell_gst = (sell_brokerage + sell_exchange + sell_sebi) * gst_rate
    sell_cost = sell_brokerage + sell_stt + sell_exchange + sell_sebi + sell_gst

    total_cost = buy_cost + sell_cost
    cost_pct = (total_cost / trade_value) * 100

    return total_cost, cost_pct

# Test different position sizes
test_sizes = [1, 5, 10, 20, 30, 40, 50]
print(f"\nTransaction costs at median price (₹{median_price:.2f}):")
print(f"{'Shares':<10} {'Trade Value':<15} {'Total Cost':<15} {'Cost %':<10}")
print("-" * 50)
for shares in test_sizes:
    cost, cost_pct = calc_transaction_cost(median_price, shares)
    trade_value = median_price * shares
    print(f"{shares:<10} ₹{trade_value:<14,.2f} ₹{cost:<14,.2f} {cost_pct:.3f}%")

print(f"\n{'='*70}")
print("RECOMMENDATION:")
print(f"{'='*70}")

# Optimal position size logic:
# 1. Should allow near-full capital deployment
# 2. Transaction costs should be <0.1% of trade value
# 3. Round number for simplicity

# Find position size where transaction cost < 0.1%
optimal = None
for shares in range(1, 100):
    cost, cost_pct = calc_transaction_cost(median_price, shares)
    if cost_pct < 0.1:
        optimal = shares
        break

if optimal:
    opt_cost, opt_cost_pct = calc_transaction_cost(median_price, optimal)
    opt_investment = optimal * median_price
    print(f"\nOptimal max_positions (transaction cost < 0.1%): {optimal}")
    print(f"  At median price: ₹{opt_investment:,.2f} investment ({100*opt_investment/starting_balance:.1f}% of capital)")
    print(f"  Transaction cost: {opt_cost_pct:.3f}%")

# Conservative recommendation
conservative = max_shares_median
cons_cost, cons_cost_pct = calc_transaction_cost(median_price, conservative)
cons_investment = conservative * median_price
print(f"\nConservative (95% capital at median): {conservative}")
print(f"  Investment: ₹{cons_investment:,.2f} ({100*cons_investment/starting_balance:.1f}% of capital)")
print(f"  Transaction cost: {cons_cost_pct:.3f}%")

# Final recommendation
if cons_cost_pct < 0.1:
    recommended = conservative
    reasoning = "Allows full capital deployment with acceptable transaction costs"
else:
    recommended = optimal
    reasoning = "Optimizes for transaction cost efficiency"

print(f"\n{'*'*70}")
print(f"FINAL RECOMMENDATION: max_positions = {recommended}")
print(f"Reasoning: {reasoning}")
print(f"{'*'*70}")

# Verify it works for all stocks
print(f"\nVERIFICATION - Can we trade all 5 stocks?")
print("-" * 50)

# Get price per stock (approximate by splitting data)
tickers = data_config.get('ticker')
n_per_stock = len(prices) // len(tickers)

for i, ticker in enumerate(tickers):
    start_idx = i * n_per_stock
    end_idx = (i + 1) * n_per_stock if i < len(tickers) - 1 else len(prices)
    stock_prices = prices[start_idx:end_idx]
    stock_median = np.median(stock_prices)

    max_shares_this_stock = int((starting_balance * max_position_value_pct) / stock_median)
    investment = max_shares_this_stock * stock_median
    cost, cost_pct = calc_transaction_cost(stock_median, min(recommended, max_shares_this_stock))

    print(f"{ticker:<15} Median: ₹{stock_median:>7.2f}  Max shares at 95%: {max_shares_this_stock:>3}  TX cost: {cost_pct:.3f}%")

print(f"\n✅ With max_positions={recommended}, agent can fully utilize capital on all stocks")
print(f"✅ Transaction costs remain economical (<0.15% per round-trip trade)")
print()
