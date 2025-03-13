#!/usr/bin/env python
from utils.data_downloader import DataDownloader

def main():
    """Download data for all standard timeframes for BTC/USDT"""
    print("Starting comprehensive data download...")
    
    # Initialize the downloader
    downloader = DataDownloader()
    
    # Define trading pairs and timeframes to download
    symbols = ['BTC/USDT', 'ETH/USDT']
    timeframes = ['5m', '15m', '1h', '4h', '1d']
    
    # Download data for each combination
    for symbol in symbols:
        print(f"\n=== Downloading data for {symbol} ===")
        for tf in timeframes:
            print(f"\nDownloading {tf} timeframe...")
            try:
                df = downloader.download_data(symbol, tf, days=365)
                if df is not None:
                    print(f"Successfully downloaded {len(df)} candles")
            except Exception as e:
                print(f"Error downloading {symbol} {tf}: {str(e)}")
    
    print("\nDownload complete!")
    print("Use the cached data for backtesting with the low_risk_strategy and other configurations")

if __name__ == "__main__":
    main()
