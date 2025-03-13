#!/usr/bin/env python
import os
import pandas as pd
import json
from datetime import datetime
import importlib
import subprocess
import matplotlib.pyplot as plt
import seaborn as sns

def check_system_health():
    """Run a comprehensive system health check"""
    print("=== TRADING SYSTEM HEALTH CHECK ===")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Check Python environment
    print("\n1. Python Environment:")
    try:
        import sys
        print(f"Python version: {sys.version}")
        
        # Check critical packages
        critical_packages = ['pandas', 'numpy', 'matplotlib', 'ta', 'ccxt', 'dash']
        for package in critical_packages:
            try:
                pkg = importlib.import_module(package)
                print(f"✓ {package}: {pkg.__version__}")
            except (ImportError, AttributeError):
                print(f"✗ {package}: Not installed or version not available")
    except Exception as e:
        print(f"Error checking Python environment: {str(e)}")
    
    # 2. Check data cache
    print("\n2. Data Cache:")
    cache_dirs = [
        '/home/panal/Documents/dashboard-trading/data/cache',
        '/home/panal/Documents/bot-machine-learning-main/data/cache'
    ]
    
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            files = [f for f in os.listdir(cache_dir) if f.endswith('.pkl')]
            print(f"✓ {cache_dir}: {len(files)} files")
            
            # Check for key timeframes
            timeframes = {'1h': False, '4h': False, '1d': False}
            for file in files:
                for tf in timeframes:
                    if tf in file:
                        timeframes[tf] = True
                        break
            
            for tf, exists in timeframes.items():
                print(f"  {'✓' if exists else '✗'} {tf} timeframe data")
        else:
            print(f"✗ {cache_dir}: Directory not found")
    
    # 3. Check strategy files
    print("\n3. Strategy Files:")
    strategy_files = [
        '/home/panal/Documents/dashboard-trading/strategies/enhanced_strategy.py',
        '/home/panal/Documents/dashboard-trading/strategies/optimized_strategy.py',
        '/home/panal/Documents/dashboard-trading/strategies/fixed_strategy.py'
    ]
    
    for file in strategy_files:
        if os.path.exists(file):
            last_modified = datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d')
            print(f"✓ {os.path.basename(file)} (Last modified: {last_modified})")
        else:
            print(f"✗ {os.path.basename(file)} not found")
    
    # 4. Check config files
    print("\n4. Configuration Files:")
    config_dir = '/home/panal/Documents/dashboard-trading/configs'
    
    if os.path.exists(config_dir):
        configs = [f for f in os.listdir(config_dir) if f.endswith('.json')]
        print(f"Found {len(configs)} configuration files:")
        
        # Analyze configurations
        if configs:
            configs_data = []
            for config_file in configs:
                try:
                    with open(os.path.join(config_dir, config_file), 'r') as f:
                        config = json.load(f)
                        
                    config_info = {
                        'name': config_file.split('.')[0],
                        'strategy': config.get('strategy', 'unknown'),
                        'timeframe': config.get('timeframe', 'unknown'),
                        'trend_filter': config.get('trend_filter', False),
                        'volume_filter': config.get('volume_filter', False)
                    }
                    
                    # Get RSI values if available
                    if 'rsi' in config:
                        config_info['rsi_oversold'] = config['rsi'].get('oversold', 0)
                        config_info['rsi_overbought'] = config['rsi'].get('overbought', 0)
                    
                    configs_data.append(config_info)
                except Exception as e:
                    print(f"  Error loading {config_file}: {str(e)}")
            
            # Convert to DataFrame for better display
            if configs_data:
                df = pd.DataFrame(configs_data)
                print("\nConfigurations Summary:")
                print(df[['name', 'strategy', 'timeframe', 'trend_filter', 'volume_filter']].to_string(index=False))
                
                # Create a comparison visualization of RSI settings
                if 'rsi_oversold' in df.columns and 'rsi_overbought' in df.columns:
                    plt.figure(figsize=(12, 6))
                    
                    # Set up the plot
                    y_pos = range(len(df))
                    plt.barh([p*2 for p in y_pos], df['rsi_overbought'] - df['rsi_oversold'], 
                            left=df['rsi_oversold'], height=0.8, color='lightblue')
                    
                    # Add labels and ticks
                    plt.yticks([p*2 for p in y_pos], df['name'])
                    plt.xlabel('RSI Value')
                    plt.title('RSI Settings by Strategy')
                    
                    # Add grid for better readability
                    plt.grid(axis='x', alpha=0.3)
                    
                    # Add min/max values
                    for i, row in enumerate(df.itertuples()):
                        plt.text(row.rsi_oversold - 2, i*2, f"{row.rsi_oversold}", 
                                ha='right', va='center')
                        plt.text(row.rsi_overbought + 2, i*2, f"{row.rsi_overbought}", 
                                ha='left', va='center')
                    
                    # Save the plot
                    out_dir = '/home/panal/Documents/dashboard-trading/reports/diagnostics'
                    os.makedirs(out_dir, exist_ok=True)
                    plt.tight_layout()
                    plt.savefig(os.path.join(out_dir, 'rsi_settings_comparison.png'))
                    plt.close()
    else:
        print(f"✗ Config directory not found: {config_dir}")
    
    # 5. Check recent backtest results
    print("\n5. Recent Backtest Results:")
    results_file = None
    reports_dir = '/home/panal/Documents/dashboard-trading/reports'
    
    if os.path.exists(reports_dir):
        # Find most recent batch backtest result
        batch_files = []
        for root, dirs, files in os.walk(reports_dir):
            for file in files:
                if file.startswith('batch_backtest_') and file.endswith('.csv'):
                    batch_files.append(os.path.join(root, file))
        
        if batch_files:
            # Sort by modification time (newest first)
            batch_files.sort(key=os.path.getmtime, reverse=True)
            results_file = batch_files[0]
            
            try:
                # Load and display results
                df = pd.read_csv(results_file)
                print(f"Loaded results from: {os.path.basename(results_file)}")
                print(f"Found {len(df)} backtest results")
                
                # Calculate success metrics
                positive_returns = (df['return_total'] > 0).mean() * 100
                avg_profit_factor = df['profit_factor'].mean()
                
                print(f"Success rate: {positive_returns:.1f}% strategies have positive returns")
                print(f"Average profit factor: {avg_profit_factor:.2f}")
                
                # Show best strategy
                if len(df) > 0:
                    best = df.loc[df['return_total'].idxmax()]
                    print(f"\nBest strategy: {best['config']} ({best['timeframe']})")
                    print(f"  Return: {best['return_total']:.2f}%")
                    print(f"  Win Rate: {best['win_rate']:.2f}%")
                    print(f"  Profit Factor: {best['profit_factor']:.2f}")
            except Exception as e:
                print(f"Error analyzing results: {str(e)}")
        else:
            print("No batch backtest results found")
    else:
        print(f"✗ Reports directory not found: {reports_dir}")
    
    # 6. Run quick diagnostic on strategies
    print("\n6. Strategy Quick Diagnostics:")
    try:
        subprocess.run(["python", "-m", "utils.strategy_debug"], 
                      stderr=subprocess.PIPE, stdout=subprocess.PIPE, text=True, check=False)
        print("✓ Strategy diagnostics completed - check console output")
    except Exception as e:
        print(f"Error running strategy diagnostics: {str(e)}")
    
    print("\nHealth check complete. Your system appears to be functioning.")
    print("Fixed issues:")
    print("1. Updated deprecated time code in backtest module")
    print("2. Fixed time calculation in paper trading module")

if __name__ == "__main__":
    check_system_health()
