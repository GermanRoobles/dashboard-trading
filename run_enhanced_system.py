#!/usr/bin/env python
import os
import argparse
import json
import pandas as pd
from datetime import datetime, timedelta
import subprocess
import time
import threading
from utils.data_cache import DataCache
from strategies.multi_timeframe_coordinator import MultiTimeframeCoordinator

def run_dashboard(port=8050):
    """Run the performance dashboard in a separate process"""
    print(f"Starting performance dashboard on port {port}...")
    subprocess.Popen(["python", "apps/performance_dashboard.py", "--port", str(port)])

def run_strategy_coordinator(config=None, timeframes=None, days=30):
    """Run the multi-timeframe strategy coordinator"""
    print(f"Starting multi-timeframe strategy coordinator...")
    
    # Fix: Construct full path to config file if only name is provided
    if config and not config.endswith('.json') and not os.path.exists(config):
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config}.json"
    else:
        config_path = config
    
    # Initialize coordinator with the proper config path
    coordinator = MultiTimeframeCoordinator(config_path=config_path)
    
    # If timeframes are specified, override defaults
    if timeframes:
        coordinator.timeframes = timeframes.split(',')
        print(f"Using timeframes: {coordinator.timeframes}")
        
    # Get data
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Load data for each timeframe
    data = {}
    for tf in coordinator.timeframes:
        print(f"Loading {tf} data...")
        data[tf] = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=tf,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data[tf] is None:
            print(f"⚠️ No data available for timeframe {tf}")
            
    # Update data for each timeframe
    for tf, df in data.items():
        if df is not None:
            print(f"Updating {tf} signals...")
            coordinator.update_data(tf, df)
    
    # Get consolidated signal
    signal = coordinator.get_consolidated_signal()
    signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
    
    # Get dynamic position size
    pos_size = coordinator.get_position_size()
    
    print("\n=== MULTI-TIMEFRAME STRATEGY SIGNAL ===")
    print(f"Consolidated signal: {signal_text}")
    print(f"Recommended position size: {pos_size:.4f}")
    
    # Generate report
    report_dir = f"/home/panal/Documents/dashboard-trading/reports/multi_timeframe_{datetime.now().strftime('%Y%m%d')}"
    os.makedirs(report_dir, exist_ok=True)
    
    print(f"\nGenerating strategy report...")
    report_df = coordinator.generate_report(report_dir)
    
    print(f"\nTimeframe weights:")
    for _, row in report_df.iterrows():
        tf = row['timeframe']
        weight = row['weight']
        tf_signal = row['signal']
        tf_signal_text = "BUY" if tf_signal == 1 else "SELL" if tf_signal == -1 else "HOLD"
        print(f"  {tf}: {weight:.2f} - Signal: {tf_signal_text}")
    
    print(f"\nReport saved to: {report_dir}")
    print("\nStrategy coordinator complete.")
    
    return report_df, signal, pos_size

def run_hybrid_backtest(config, timeframes, days):
    """Run the hybrid backtest for the specified timeframes"""
    print("Starting hybrid strategy backtest...")
    
    # Run the backtest script
    cmd = ["python", "run_hybrid_backtest.py"]
    
    if config:
        cmd.extend(["--config", config])
        
    if timeframes:
        cmd.extend(["--timeframes", timeframes])
        
    if days:
        cmd.extend(["--days", str(days)])
    
    # Run process and capture output
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()
    
    # Print output
    print(stdout)
    
    if stderr:
        print("Errors:")
        print(stderr)
        
    # Return success/failure
    return process.returncode == 0

def main():
    """Main function to run the enhanced trading system"""
    parser = argparse.ArgumentParser(description='Run the enhanced trading system')
    parser.add_argument('--config', type=str, default='hybrid_strategy', help='Strategy configuration')
    parser.add_argument('--timeframes', type=str, default='15m,1h,4h', help='Comma-separated timeframes')
    parser.add_argument('--days', type=int, default=30, help='Days of historical data')
    parser.add_argument('--dashboard', action='store_true', help='Launch the dashboard')
    parser.add_argument('--backtest', action='store_true', help='Run backtests')
    parser.add_argument('--live', action='store_true', help='Run live signal coordination')
    parser.add_argument('--all', action='store_true', help='Run everything')
    parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    
    args = parser.parse_args()
    
    # Run the requested components
    if args.all or args.dashboard:
        # Start dashboard in a separate thread
        dashboard_thread = threading.Thread(target=run_dashboard, args=(args.port,))
        dashboard_thread.daemon = True
        dashboard_thread.start()
        print("Dashboard started in background")
        time.sleep(2)  # Give the dashboard time to start
    
    if args.all or args.backtest:
        # Run backtests
        print("\n=== RUNNING BACKTESTS ===")
        success = run_hybrid_backtest(args.config, args.timeframes, args.days)
        if not success:
            print("⚠️ Backtest failed or had errors")
    
    if args.all or args.live:
        # Run live strategy coordinator
        print("\n=== RUNNING LIVE STRATEGY COORDINATION ===")
        report_df, signal, pos_size = run_strategy_coordinator(args.config, args.timeframes, args.days)
        
        # Output final recommendation
        print("\n=== FINAL RECOMMENDATION ===")
        signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
        print(f"Signal: {signal_text}")
        print(f"Position Size: {pos_size:.4f}")
        print(f"Based on analysis of {args.timeframes} timeframes")
    
    # If dashboard was started and this is the main thread, keep program running
    if (args.all or args.dashboard) and not (args.backtest or args.live):
        print("\nDashboard is running. Press Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nExiting...")

if __name__ == "__main__":
    main()
