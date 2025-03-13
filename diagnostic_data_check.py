#!/usr/bin/env python
import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

def check_cached_data():
    """Check all cached data and report on availability"""
    # Cache directories to check
    cache_dirs = [
        '/home/panal/Documents/dashboard-trading/data/cache',
        '/home/panal/Documents/bot-machine-learning-main/data/cache'
    ]
    
    # Lists to store available timeframes by exchange/symbol
    available_data = {}
    
    # Check each directory
    for cache_dir in cache_dirs:
        if not os.path.exists(cache_dir):
            continue
            
        print(f"\nChecking directory: {cache_dir}")
        
        # Check each file in the directory
        for filename in os.listdir(cache_dir):
            if not filename.endswith('.pkl'):
                continue
                
            # Parse filename to get symbol and timeframe
            parts = filename.split('_')
            if len(parts) >= 3:
                symbol = f"{parts[0]}_{parts[1]}"
                timeframe = parts[2]
                
                # Skip if timeframe contains non-alphanumeric chars (probably a hash)
                if not timeframe.isalnum():
                    continue
                
                # Try to load and check the data
                try:
                    filepath = os.path.join(cache_dir, filename)
                    data = pd.read_pickle(filepath)
                    
                    if symbol not in available_data:
                        available_data[symbol] = []
                    
                    # Add timeframe information
                    available_data[symbol].append({
                        'timeframe': timeframe,
                        'filepath': filepath,
                        'candles': len(data),
                        'start_date': data.index.min(),
                        'end_date': data.index.max(),
                        'days_range': (data.index.max() - data.index.min()).days
                    })
                    
                except Exception as e:
                    print(f"Error loading {filename}: {str(e)}")
    
    # Print summary report
    print("\n==== DATA AVAILABILITY REPORT ====")
    for symbol, timeframes in available_data.items():
        print(f"\nSymbol: {symbol}")
        print("-" * 60)
        print(f"{'Timeframe':<10} {'Candles':<10} {'Date Range':<30} {'File Location'}")
        print("-" * 60)
        
        # Sort by timeframe
        timeframes.sort(key=lambda x: x['timeframe'])
        
        for tf_data in timeframes:
            tf = tf_data['timeframe']
            candles = tf_data['candles']
            start = tf_data['start_date'].strftime('%Y-%m-%d')
            end = tf_data['end_date'].strftime('%Y-%m-%d')
            date_range = f"{start} to {end}"
            
            # Get just the filename for display
            filename = os.path.basename(tf_data['filepath'])
            
            print(f"{tf:<10} {candles:<10} {date_range:<30} {filename}")
    
    # Create a visualization of data coverage
    plt.figure(figsize=(12, 8))
    
    y_pos = 0
    symbols = list(available_data.keys())
    
    for symbol in symbols:
        timeframes = available_data[symbol]
        
        for tf_data in timeframes:
            tf = tf_data['timeframe']
            start_date = tf_data['start_date']
            end_date = tf_data['end_date']
            
            # Plot a horizontal bar for this timeframe
            plt.barh(
                y_pos, 
                (end_date - start_date).days, 
                left=start_date,
                height=0.8,
                label=f"{symbol} {tf}" if y_pos == 0 else "",
                alpha=0.7
            )
            
            # Add text label
            plt.text(
                start_date + (end_date - start_date) / 2,
                y_pos,
                f"{tf} ({tf_data['candles']} candles)",
                va='center',
                ha='center',
                fontsize=8
            )
            
            y_pos += 1
    
    # Format the plot
    plt.yticks(range(y_pos), [f"{s}_{t['timeframe']}" for s in symbols for t in available_data[s]])
    plt.xlabel('Date Range')
    plt.title('Data Availability by Symbol and Timeframe')
    
    # Format x-axis as dates
    plt.gcf().autofmt_xdate()
    
    # Save the visualization
    report_dir = '/home/panal/Documents/dashboard-trading/reports/diagnostics'
    os.makedirs(report_dir, exist_ok=True)
    plt.savefig(os.path.join(report_dir, f"data_coverage_{datetime.now().strftime('%Y%m%d')}.png"))
    
    print(f"\nData coverage visualization saved to {report_dir}")
    print("\nTo create missing timeframes, run:\npython -m utils.data_preprocessor")

if __name__ == "__main__":
    check_cached_data()
