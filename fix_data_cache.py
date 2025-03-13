#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def check_cache_files():
    """Check and verify all cache files in both directories"""
    cache_dirs = [
        '/home/panal/Documents/dashboard-trading/data/cache',
        '/home/panal/Documents/bot-machine-learning-main/data/cache'
    ]
    
    valid_files = []
    problematic_files = []
    
    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            print(f"Directory does not exist: {cache_dir}")
            continue
            
        print(f"\nChecking files in {cache_dir}:")
        for file in os.listdir(cache_dir):
            if not file.endswith('.pkl'):
                continue
                
            filepath = os.path.join(cache_dir, file)
            try:
                # Try loading the file
                df = pd.read_pickle(filepath)
                if isinstance(df, pd.DataFrame) and len(df) > 0:
                    # Check if this is OHLCV data
                    required_cols = ['open', 'high', 'low', 'close']
                    has_required = all(col in df.columns for col in required_cols)
                    
                    if has_required:
                        print(f"✓ {file}: Valid ({len(df)} rows, {df.index[0]} to {df.index[-1]})")
                        valid_files.append({
                            'file': filepath, 
                            'rows': len(df),
                            'start': df.index[0],
                            'end': df.index[-1],
                            'timeframe': detect_timeframe(df)
                        })
                    else:
                        print(f"⚠ {file}: Missing required columns {[col for col in required_cols if col not in df.columns]}")
                        problematic_files.append(filepath)
                else:
                    print(f"⚠ {file}: Empty DataFrame")
                    problematic_files.append(filepath)
            except Exception as e:
                print(f"✗ {file}: Error loading file: {str(e)}")
                problematic_files.append(filepath)
    
    return valid_files, problematic_files

def detect_timeframe(df):
    """Detect the likely timeframe of a DataFrame based on time difference between rows"""
    if len(df) < 2:
        return "unknown"
    
    # Calculate median time difference in minutes
    diffs = []
    for i in range(1, min(100, len(df))):
        diff = (df.index[i] - df.index[i-1]).total_seconds() / 60
        diffs.append(diff)
    
    median_diff = np.median(diffs)
    
    # Map to common timeframes
    if median_diff < 3:
        return "1m"
    elif median_diff < 10:
        return "5m"
    elif median_diff < 20:
        return "15m"
    elif median_diff < 45:
        return "30m"
    elif median_diff < 120:
        return "1h"
    elif median_diff < 360:
        return "4h"
    elif median_diff < 1440:
        return "1d"
    else:
        return "higher"

def create_4h_data_for_low_risk():
    """Create a dedicated 4h timeframe dataset for low_risk_strategy"""
    print("\nCreating dedicated 4h data for low_risk_strategy")
    
    # Find source data - preferably 1h data
    source_data = None
    source_path = None
    cache_dirs = [
        '/home/panal/Documents/dashboard-trading/data/cache',
        '/home/panal/Documents/bot-machine-learning-main/data/cache'
    ]
    
    # First try to find 1h data
    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            continue
            
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl') and "BTC_USDT" in file and "1h" in file:
                try:
                    filepath = os.path.join(cache_dir, file)
                    df = pd.read_pickle(filepath)
                    if isinstance(df, pd.DataFrame) and len(df) > 0:
                        source_data = df
                        source_path = filepath
                        print(f"Found 1h source data: {filepath} ({len(df)} rows)")
                        break
                except Exception:
                    continue
        
        if source_data is not None:
            break
    
    # If no 1h data, try 15m data
    if source_data is None:
        for cache_dir in cache_dirs:
            if not os.path.exists(cache_dir):
                continue
                
            for file in os.listdir(cache_dir):
                if file.endswith('.pkl') and "BTC_USDT" in file and "15m" in file:
                    try:
                        filepath = os.path.join(cache_dir, file)
                        df = pd.read_pickle(filepath)
                        if isinstance(df, pd.DataFrame) and len(df) > 0:
                            source_data = df
                            source_path = filepath
                            print(f"Found 15m source data: {filepath} ({len(df)} rows)")
                            break
                    except Exception:
                        continue
            
            if source_data is not None:
                break
    
    if source_data is None:
        print("❌ Could not find source data for resampling")
        return False
    
    # Resample to 4h
    try:
        # Ensure we have a datetime index
        if not isinstance(source_data.index, pd.DatetimeIndex):
            print("❌ Source data does not have a datetime index")
            return False
            
        # Map timeframe string to pandas offset string
        offset = '4H'
        
        # Resample
        resampled = source_data.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop any rows with NaN values
        resampled = resampled.dropna()
        
        # Save to dedicated file
        main_cache_dir = '/home/panal/Documents/dashboard-trading/data/cache'
        os.makedirs(main_cache_dir, exist_ok=True)
        
        target_file = os.path.join(main_cache_dir, "BTC_USDT_4h_lowrisk_fixed.pkl")
        resampled.to_pickle(target_file)
        
        print(f"✓ Successfully created 4h data for low_risk_strategy: {target_file}")
        print(f"  Data range: {resampled.index[0]} to {resampled.index[-1]}")
        print(f"  Number of candles: {len(resampled)}")
        
        # Create a simple plot to verify
        plt.figure(figsize=(12, 6))
        plt.plot(resampled.index, resampled['close'])
        plt.title('BTC/USDT 4h Close Price')
        plt.xlabel('Date')
        plt.ylabel('Price (USDT)')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(main_cache_dir, 'btc_usdt_4h_verification.png'))
        plt.close()
        
        print(f"✓ Created verification chart at: {os.path.join(main_cache_dir, 'btc_usdt_4h_verification.png')}")
        
        return True
    except Exception as e:
        print(f"❌ Error creating 4h data: {str(e)}")
        return False

def create_low_risk_strategy_symlink():
    """Create a symlink for the low_risk_strategy data in both cache directories"""
    main_cache_dir = '/home/panal/Documents/dashboard-trading/data/cache'
    alt_cache_dir = '/home/panal/Documents/bot-machine-learning-main/data/cache'
    
    source_file = os.path.join(main_cache_dir, "BTC_USDT_4h_lowrisk_fixed.pkl")
    
    if not os.path.exists(source_file):
        print(f"❌ Source file does not exist: {source_file}")
        return False
    
    # Create standard timestamped filename that the strategy analyzer expects
    import hashlib
    identifier = "BTC_USDT_4h"
    hash_suffix = hashlib.md5(identifier.encode()).hexdigest()[:8]
    target_name = f"BTC_USDT_4h_{hash_suffix}.pkl"
    
    # Create copies or symlinks in both directories
    for cache_dir in [main_cache_dir, alt_cache_dir]:
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
            
        target_file = os.path.join(cache_dir, target_name)
        
        # Create a copy instead of symlink for better compatibility
        if not os.path.exists(target_file):
            try:
                import shutil
                shutil.copy2(source_file, target_file)
                print(f"✓ Created copy at: {target_file}")
            except Exception as e:
                print(f"❌ Error creating copy: {str(e)}")
    
    return True

if __name__ == "__main__":
    # Step 1: Check existing cache files
    print("=== CHECKING CACHE FILES ===")
    valid_files, problematic_files = check_cache_files()
    
    print(f"\nFound {len(valid_files)} valid files and {len(problematic_files)} problematic files")
    
    # Step 2: Create dedicated 4h data for low_risk_strategy
    print("\n=== CREATING 4H DATA FOR LOW RISK STRATEGY ===")
    success = create_4h_data_for_low_risk()
    
    if success:
        # Step 3: Create symlinks for both cache directories
        print("\n=== CREATING STANDARD FILENAMES FOR CACHE DIRECTORIES ===")
        create_low_risk_strategy_symlink()
        
        print("\n=== SETUP COMPLETE ===")
        print("Now run the following commands to analyze the low_risk_strategy:")
        print("python -m utils.strategy_analyzer --config low_risk_strategy --timeframe 4h --days 90")
    else:
        print("\n❌ Failed to create 4h data. Please check the logs above for errors.")
