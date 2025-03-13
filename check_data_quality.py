#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from utils.data_cache import DataCache

def analyze_data_quality(timeframe='1h', days=60):
    """Analyze the quality of cached data"""
    print(f"Analyzing data quality for {timeframe} timeframe over {days} days")
    
    # Get data
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = cache.get_cached_data(
        symbol='BTC/USDT', 
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if data is None or data.empty:
        print("No data available for analysis")
        return
    
    # 1. Basic statistics
    print("\n=== BASIC STATISTICS ===")
    print(f"Total rows: {len(data)}")
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Expected rows: ~{days * 24 // {'1h': 1, '4h': 4, '1d': 24, '15m': 4, '5m': 12}[timeframe]}")
    print(f"Actual rows: {len(data)}")
    
    # 2. Check for missing values
    print("\n=== MISSING VALUES CHECK ===")
    missing = data.isnull().sum()
    
    if missing.sum() > 0:
        print("Missing values detected:")
        for col, count in missing.items():
            if count > 0:
                print(f"  {col}: {count} missing values ({count/len(data)*100:.2f}%)")
    else:
        print("✓ No missing values detected")
    
    # 3. Check for price anomalies
    print("\n=== PRICE ANOMALY CHECK ===")
    
    # Calculate price changes
    data['price_change_pct'] = data['close'].pct_change() * 100
    
    # Identify large price changes (more than 5%)
    large_changes = data[abs(data['price_change_pct']) > 5]
    
    if len(large_changes) > 0:
        print(f"⚠️ Found {len(large_changes)} large price changes (>5%)")
        for i, row in large_changes.iterrows():
            print(f"  {i}: {row['price_change_pct']:.2f}% change from {row['open']} to {row['close']}")
    else:
        print("✓ No unusually large price changes detected")
    
    # 4. Check for consecutive identical prices
    print("\n=== IDENTICAL PRICES CHECK ===")
    
    identical_closes = 0
    max_streak = 0
    current_streak = 1
    
    for i in range(1, len(data)):
        if data['close'].iloc[i] == data['close'].iloc[i-1]:
            current_streak += 1
            identical_closes += 1
        else:
            max_streak = max(max_streak, current_streak)
            current_streak = 1
    
    max_streak = max(max_streak, current_streak)
    
    if identical_closes > 0:
        print(f"⚠️ Found {identical_closes} instances of identical consecutive closing prices")
        print(f"  Maximum streak of identical prices: {max_streak}")
    else:
        print("✓ No unusual patterns of identical prices")
    
    # 5. Check for gaps in time series
    print("\n=== TIME SERIES CONTINUITY CHECK ===")
    
    # Expected time difference
    expected_diff_minutes = {'1h': 60, '4h': 240, '1d': 1440, '15m': 15, '5m': 5}[timeframe]
    
    # Calculate actual time differences
    time_diffs = []
    for i in range(1, len(data)):
        diff = (data.index[i] - data.index[i-1]).total_seconds() / 60
        time_diffs.append(diff)
    
    # Identify gaps
    gaps = [i+1 for i, diff in enumerate(time_diffs) if diff > expected_diff_minutes * 1.5]
    
    if gaps:
        print(f"⚠️ Found {len(gaps)} gaps in time series")
        for i in gaps[:5]:  # Show first 5
            gap_size = (data.index[i] - data.index[i-1]).total_seconds() / 60
            print(f"  Gap at {data.index[i]}: {gap_size:.1f} minutes (expected {expected_diff_minutes})")
        
        if len(gaps) > 5:
            print(f"  ... and {len(gaps)-5} more gaps")
    else:
        print("✓ Time series is continuous")
    
    # 6. Visualize data
    plt.figure(figsize=(15, 10))
    
    # Price chart
    plt.subplot(3, 1, 1)
    plt.plot(data.index, data['close'])
    plt.title(f'BTC/USDT Price - {timeframe} ({days} days)')
    plt.ylabel('Price (USDT)')
    plt.grid(True, alpha=0.3)
    
    # Volume chart
    plt.subplot(3, 1, 2)
    plt.bar(data.index, data['volume'], color='blue', alpha=0.5)
    plt.title('Trading Volume')
    plt.ylabel('Volume')
    plt.grid(True, alpha=0.3)
    
    # Price change distribution
    plt.subplot(3, 1, 3)
    plt.hist(data['price_change_pct'].dropna(), bins=50, alpha=0.7)
    plt.axvline(x=0, color='red', linestyle='--')
    plt.title('Price Change Distribution')
    plt.xlabel('Price Change (%)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = '/home/panal/Documents/dashboard-trading/reports/diagnostics'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"data_quality_{timeframe}_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(output_file)
    
    print(f"\nData quality visualization saved to: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze data quality')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to analyze')
    parser.add_argument('--days', type=int, default=60, help='Number of days to analyze')
    
    args = parser.parse_args()
    analyze_data_quality(args.timeframe, args.days)
