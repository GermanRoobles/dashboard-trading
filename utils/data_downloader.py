import os
import pandas as pd
import ccxt
import time
from datetime import datetime, timedelta

class DataDownloader:
    """Downloads and caches historical market data"""
    
    def __init__(self, exchange_id='binance'):
        """Initialize with the exchange to use"""
        self.exchange = getattr(ccxt, exchange_id)({
            'enableRateLimit': True,
        })
        
        self.cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            'data',
            'cache'
        )
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
    def download_data(self, symbol, timeframe, days=180):
        """
        Download historical data and save to cache
        
        Args:
            symbol: Trading pair (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            days: Number of days of history to download
            
        Returns:
            DataFrame with OHLCV data
        """
        print(f"Downloading {days} days of {timeframe} data for {symbol}...")
        
        # Calculate time range
        end_time = int(time.time() * 1000)  # Current time in milliseconds
        start_time = end_time - (days * 24 * 60 * 60 * 1000)  # days ago
        
        # Fetch data in chunks to avoid rate limits
        all_candles = []
        chunk_size = 1000  # Maximum number of candles per request
        current_start = start_time
        
        while current_start < end_time:
            candles = self.exchange.fetch_ohlcv(
                symbol, 
                timeframe=timeframe,
                since=current_start,
                limit=chunk_size
            )
            
            if not candles:
                break
                
            all_candles.extend(candles)
            
            # Update progress
            downloaded_days = len(all_candles) * self.get_timeframe_minutes(timeframe) / (24 * 60)
            print(f"Downloaded approximately {downloaded_days:.1f} days of data...")
            
            # Update start time for next request
            current_start = candles[-1][0] + 1
            
            # Respect rate limits
            time.sleep(self.exchange.rateLimit / 1000)
        
        # Convert to DataFrame
        if not all_candles:
            print("No data retrieved")
            return None
            
        df = pd.DataFrame(
            all_candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Convert timestamp to datetime index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Save to cache
        cache_file = self.get_cache_filename(symbol, timeframe)
        df.to_pickle(cache_file)
        
        print(f"Downloaded {len(df)} candles for {symbol} {timeframe}")
        print(f"Data range: {df.index.min()} to {df.index.max()}")
        print(f"Saved to: {cache_file}")
        
        return df
    
    def get_cache_filename(self, symbol, timeframe):
        """Generate a cache filename for the given symbol and timeframe"""
        # Clean symbol name for filenames
        clean_symbol = symbol.replace('/', '_')
        
        # Create a unique identifier based on symbol and timeframe
        identifier = f"{clean_symbol}_{timeframe}"
        
        # Generate a short hash for uniqueness
        import hashlib
        hash_suffix = hashlib.md5(identifier.encode()).hexdigest()[:8]
        
        return os.path.join(self.cache_dir, f"{clean_symbol}_{timeframe}_{hash_suffix}.pkl")
    
    def get_timeframe_minutes(self, timeframe):
        """Convert timeframe string to minutes"""
        unit = timeframe[-1]
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 60 * 24
        else:
            return 60  # Default to 1h
    
    def download_multiple_timeframes(self, symbol='BTC/USDT', timeframes=None, days=180):
        """Download data for multiple timeframes"""
        if timeframes is None:
            timeframes = ['5m', '15m', '1h', '4h', '1d']
            
        results = {}
        for tf in timeframes:
            print(f"\nDownloading {tf} data...")
            df = self.download_data(symbol, tf, days)
            results[tf] = df
            
        return results

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Download historical market data')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', help='Trading pair')
    parser.add_argument('--timeframe', type=str, default=None, help='Single timeframe to download')
    parser.add_argument('--days', type=int, default=180, help='Days of history')
    
    args = parser.parse_args()
    
    downloader = DataDownloader()
    
    if args.timeframe:
        downloader.download_data(args.symbol, args.timeframe, args.days)
    else:
        print("Downloading data for all standard timeframes...")
        downloader.download_multiple_timeframes(args.symbol)
        
if __name__ == "__main__":
    main()
