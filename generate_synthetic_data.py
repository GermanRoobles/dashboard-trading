#!/usr/bin/env python
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_synthetic_btc_data(timeframe, days=180, volatility=0.02):
    """Generate synthetic Bitcoin price data for testing"""
    # Determine the number of periods based on timeframe
    periods_per_day = {
        '1m': 24 * 60,
        '5m': 24 * 12,
        '15m': 24 * 4,
        '30m': 24 * 2,
        '1h': 24,
        '4h': 6,
        '1d': 1
    }
    
    periods = periods_per_day.get(timeframe, 24) * days
    
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    if timeframe == '1d':
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
    elif timeframe == '4h':
        dates = pd.date_range(start=start_date, end=end_date, freq='4h')  # use 'h' instead of 'H'
    elif timeframe == '1h':
        dates = pd.date_range(start=start_date, end=end_date, freq='1h')  # use 'h' instead of 'H'
    elif timeframe == '30m':
        dates = pd.date_range(start=start_date, end=end_date, freq='30min')
    elif timeframe == '15m':
        dates = pd.date_range(start=start_date, end=end_date, freq='15min')
    elif timeframe == '5m':
        dates = pd.date_range(start=start_date, end=end_date, freq='5min')
    else:
        dates = pd.date_range(start=start_date, end=end_date, freq='min')
    
    # Generate simulated price data
    price = 45000  # Starting price
    prices = [price]
    
    for i in range(1, len(dates)):
        # Random price change with slight upward bias
        change = np.random.normal(0.0001, volatility)
        price = price * (1 + change)
        prices.append(price)
    
    # Create dataframe
    df = pd.DataFrame(index=dates)
    df['close'] = prices
    
    # Generate open, high, low
    df['open'] = df['close'].shift(1)
    df.loc[df.index[0], 'open'] = df['close'].iloc[0] * 0.9995  # use iloc instead of []
    
    # Add some randomness to high/low
    daily_volatility = df['close'].pct_change().std()
    df['high'] = df[['open', 'close']].max(axis=1) * (1 + abs(np.random.normal(0, daily_volatility, len(df))))
    df['low'] = df[['open', 'close']].min(axis=1) * (1 - abs(np.random.normal(0, daily_volatility, len(df))))
    
    # Make sure high is always highest and low is always lowest
    for i in range(len(df)):
        high = max(df.loc[df.index[i], 'open'], df.loc[df.index[i], 'close'], df.loc[df.index[i], 'high'])
        low = min(df.loc[df.index[i], 'open'], df.loc[df.index[i], 'close'], df.loc[df.index[i], 'low'])
        df.loc[df.index[i], 'high'] = high
        df.loc[df.index[i], 'low'] = low
    
    # Add volume
    df['volume'] = np.random.normal(1000, 500, len(df)) * df['close']
    df['volume'] = df['volume'].abs()
    
    return df

def save_synthetic_data():
    """Generate and save synthetic data for backtesting"""
    # Create cache directory if it doesn't exist
    cache_dir = '/home/panal/Documents/dashboard-trading/data/cache'
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate data for different timeframes
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    for tf in timeframes:
        print(f"Generating synthetic data for {tf} timeframe...")
        df = generate_synthetic_btc_data(tf, days=365)
        
        # Save to pickle file
        import hashlib
        identifier = f"BTC_USDT_{tf}"
        hash_suffix = hashlib.md5(identifier.encode()).hexdigest()[:8]
        filename = f"BTC_USDT_{tf}_{hash_suffix}.pkl"
        filepath = os.path.join(cache_dir, filename)
        
        df.to_pickle(filepath)
        print(f"Created synthetic data file: {filepath} with {len(df)} candles")
        
        # Create special 4h file for low risk strategy
        if tf == '4h':
            special_file = os.path.join(cache_dir, "BTC_USDT_4h_lowrisk_fixed.pkl")
            df.to_pickle(special_file)
            print(f"Created special 4h file: {special_file}")

if __name__ == "__main__":
    print("Generating synthetic data for backtesting...")
    save_synthetic_data()
    print("Data generation complete! You can now run backtests.")
