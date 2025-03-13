#!/usr/bin/env python
import os
import json
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def monitor_paper_trading(log_file=None):
    """Monitor and visualize ongoing paper trading results"""
    if log_file is None:
        # Find the most recent paper trading log
        report_dir = '/home/panal/Documents/dashboard-trading/reports/paper_trading'
        if not os.path.exists(report_dir):
            print(f"Paper trading report directory not found: {report_dir}")
            return
            
        # Find all json result files
        result_files = [f for f in os.listdir(report_dir) if f.endswith('.json') and f.startswith('results_')]
        
        if not result_files:
            print("No paper trading results found.")
            return
            
        # Sort by modification time (newest first)
        result_files.sort(key=lambda x: os.path.getmtime(os.path.join(report_dir, x)), reverse=True)
        log_file = os.path.join(report_dir, result_files[0])
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    print(f"Monitoring paper trading log: {log_file}")
    print("Press Ctrl+C to stop monitoring\n")
    
    last_read_time = 0
    signals_seen = 0
    
    try:
        while True:
            # Load the file if it has been modified
            current_mod_time = os.path.getmtime(log_file)
            
            if current_mod_time > last_read_time:
                with open(log_file, 'r') as f:
                    paper_trading_data = json.load(f)
                
                trades = paper_trading_data.get('trades', [])
                
                if len(trades) > signals_seen:
                    # We have new signals!
                    new_signals = trades[signals_seen:]
                    signals_seen = len(trades)
                    
                    # Display the new signals
                    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {len(new_signals)} new signals detected:")
                    for signal in new_signals:
                        print(f"  {signal['timestamp']} - {signal['signal']} at {signal['price']}")
                        
                    # Generate visualization if we have enough signals
                    if len(trades) >= 3:
                        generate_visualization(trades, os.path.dirname(log_file))
                
                last_read_time = current_mod_time
            
            # Sleep for a bit to avoid high CPU usage
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\nStopped monitoring")
        if signals_seen > 0:
            print(f"Total signals detected: {signals_seen}")
    except Exception as e:
        print(f"Error monitoring paper trading: {e}")

def generate_visualization(trades, output_dir):
    """Generate visualization for paper trading signals"""
    # Convert trades to DataFrame
    try:
        df = pd.DataFrame(trades)
        
        # Convert timestamp strings to datetime if needed
        if isinstance(df['timestamp'][0], str):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Convert price to numeric if needed
        df['price'] = pd.to_numeric(df['price'])
        
        # Plot signals over time
        plt.figure(figsize=(12, 6))
        
        # Plot price as a line
        plt.plot(df['timestamp'], df['price'], 'k-', alpha=0.6, label='Price')
        
        # Plot buy and sell signals
        buys = df[df['signal'] == 'BUY']
        sells = df[df['signal'] == 'SELL']
        
        if not buys.empty:
            plt.scatter(buys['timestamp'], buys['price'], marker='^', color='green', s=100, label='BUY')
        
        if not sells.empty:
            plt.scatter(sells['timestamp'], sells['price'], marker='v', color='red', s=100, label='SELL')
        
        plt.title('Paper Trading Signals')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Format dates on x-axis
        plt.gcf().autofmt_xdate()
        
        # Save the plot
        output_file = os.path.join(output_dir, 'paper_trading_signals.png')
        plt.savefig(output_file)
        plt.close()
        
        print(f"Updated visualization saved to: {output_file}")
        
    except Exception as e:
        print(f"Error generating visualization: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monitor paper trading results')
    parser.add_argument('--log', type=str, help='Path to paper trading log file')
    
    args = parser.parse_args()
    monitor_paper_trading(args.log)
