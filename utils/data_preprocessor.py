import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

class DataPreprocessor:
    """Utility to preprocess and resample data to different timeframes"""
    
    @staticmethod
    def resample_timeframe(df: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """
        Resample OHLCV data to a new timeframe
        
        Args:
            df: DataFrame with OHLCV data with datetime index
            target_timeframe: Target timeframe (e.g., '4h', '1d')
            
        Returns:
            DataFrame with resampled data
        """
        if df.empty or not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DateTimeIndex and contain data")
            
        # Map timeframe string to pandas offset string
        timeframe_map = {
            '1m': '1min', '3m': '3min', '5m': '5min', '15m': '15min', '30m': '30min',
            '1h': '1H', '2h': '2H', '4h': '4H', '6h': '6H', '8h': '8H', '12h': '12H',
            '1d': '1D', '3d': '3D', '1w': '1W'
        }
        
        if target_timeframe not in timeframe_map:
            raise ValueError(f"Unsupported timeframe: {target_timeframe}")
            
        offset = timeframe_map[target_timeframe]
        
        # Resample using pandas
        resampled = df.resample(offset).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Filter out rows with NaN values
        resampled = resampled.dropna()
        
        return resampled
        
    @staticmethod
    def create_missing_timeframes(source_timeframe: str, 
                                 target_timeframes: list, 
                                 cache_dir: str,
                                 symbol: str = 'BTC/USDT'):
        """
        Create missing timeframe data by resampling from existing data
        
        Args:
            source_timeframe: Source timeframe data to use (e.g., '1h')
            target_timeframes: List of timeframes to create (e.g., ['4h', '1d'])
            cache_dir: Directory where cached data is stored
            symbol: Trading pair symbol
        """
        print(f"Creating missing timeframes from {source_timeframe} data...")
        
        # Find cached source data
        source_files = []
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl') and source_timeframe in file and symbol.replace('/', '_') in file:
                source_files.append(os.path.join(cache_dir, file))
        
        if not source_files:
            print(f"No source data found for {symbol} {source_timeframe}")
            return False
            
        # Use the most recent file
        source_files.sort(key=os.path.getmtime, reverse=True)
        source_file = source_files[0]
        
        try:
            # Load source data
            source_data = pd.read_pickle(source_file)
            print(f"Loaded source data: {len(source_data)} candles from {source_file}")
            
            # Create each target timeframe
            for tf in target_timeframes:
                # Check if this timeframe already exists
                target_exists = False
                for file in os.listdir(cache_dir):
                    if file.endswith('.pkl') and tf in file and symbol.replace('/', '_') in file:
                        print(f"Target timeframe {tf} already exists: {file}")
                        target_exists = True
                        break
                
                if not target_exists:
                    # Resample the data
                    resampled = DataPreprocessor.resample_timeframe(source_data, tf)
                    
                    # Generate a new filename
                    clean_symbol = symbol.replace('/', '_')
                    import hashlib
                    identifier = f"{clean_symbol}_{tf}"
                    hash_suffix = hashlib.md5(identifier.encode()).hexdigest()[:8]
                    target_file = os.path.join(cache_dir, f"{clean_symbol}_{tf}_{hash_suffix}.pkl")
                    
                    # Save the resampled data
                    resampled.to_pickle(target_file)
                    print(f"Created {tf} data with {len(resampled)} candles, saved to {target_file}")
            
            return True
            
        except Exception as e:
            print(f"Error processing data: {str(e)}")
            return False

def create_missing_data():
    """Create missing timeframe data for analysis"""
    # Define cache directories to check
    cache_dirs = [
        '/home/panal/Documents/dashboard-trading/data/cache',
        '/home/panal/Documents/bot-machine-learning-main/data/cache'
    ]
    
    # Select the first existing directory
    cache_dir = None
    for directory in cache_dirs:
        if os.path.exists(directory):
            cache_dir = directory
            break
    
    if not cache_dir:
        print("No cache directory found")
        return False
    
    print(f"Using cache directory: {cache_dir}")
    
    # Create necessary timeframes from available data
    processor = DataPreprocessor()
    
    # First try creating from the lowest timeframe available
    for source_tf in ['1m', '5m', '15m', '1h']:
        # Check if we have this timeframe
        has_source = False
        for file in os.listdir(cache_dir):
            if file.endswith('.pkl') and source_tf in file and 'BTC_USDT' in file:
                has_source = True
                break
        
        if has_source:
            print(f"Found source data for {source_tf}")
            # Create higher timeframes from this source
            target_tfs = []
            if source_tf == '1m':
                target_tfs = ['3m', '5m', '15m', '30m', '1h', '4h', '1d']
            elif source_tf == '5m':
                target_tfs = ['15m', '30m', '1h', '4h', '1d']
            elif source_tf == '15m':
                target_tfs = ['30m', '1h', '4h', '1d']
            elif source_tf == '1h':
                target_tfs = ['4h', '1d']
            
            result = processor.create_missing_timeframes(source_tf, target_tfs, cache_dir)
            if result:
                print(f"Successfully created missing timeframes from {source_tf}")
                break
    
    return True

if __name__ == "__main__":
    create_missing_data()
