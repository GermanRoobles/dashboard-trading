#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache

def optimize_rsi_parameters(config_name, timeframe='1h', days=30):
    """Optimize RSI parameters for a given strategy configuration"""
    print(f"Optimizing RSI parameters for {config_name} on {timeframe} ({days} days)")
    
    # Load base configuration
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
    if not os.path.exists(config_path):
        print(f"Configuration not found: {config_name}")
        return None
    
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Get data for optimization - fix parameter issue
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Fix: Use proper start_date and end_date parameters instead of 'days'
    data = cache.get_cached_data(
        symbol='BTC/USDT', 
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if data is None:
        print(f"No data available for {timeframe} timeframe")
        return None
    
    # Define parameter grid for RSI
    oversold_values = list(range(25, 46, 5))  # 25, 30, 35, 40, 45
    overbought_values = list(range(55, 76, 5))  # 55, 60, 65, 70, 75
    
    results = []
    
    # Test different RSI combinations
    total_tests = len(oversold_values) * len(overbought_values)
    print(f"Running {total_tests} parameter combinations...")
    
    for oversold in oversold_values:
        for overbought in overbought_values:
            if oversold >= overbought:
                continue  # Skip invalid combinations
                
            # Create a modified config with current parameters
            test_config = base_config.copy()
            if 'rsi' not in test_config:
                test_config['rsi'] = {}
            
            test_config['rsi']['oversold'] = oversold
            test_config['rsi']['overbought'] = overbought
            
            # Run backtest
            backtest = BacktestFixedStrategy(config=test_config)
            backtest_result = backtest.run(data)
            
            # Save results
            results.append({
                'oversold': oversold,
                'overbought': overbought,
                'return': backtest_result['return_total'],
                'win_rate': backtest_result['win_rate'],
                'profit_factor': backtest_result['profit_factor'],
                'trades': backtest_result['total_trades']
            })
            
            print(f"RSI({oversold},{overbought}): Return={backtest_result['return_total']:.2f}%, "
                 f"Win Rate={backtest_result['win_rate']:.1f}%, "
                 f"Trades={backtest_result['total_trades']}")
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Find optimal parameters
    if not results_df.empty:
        # Sort by return (highest first)
        results_df = results_df.sort_values('return', ascending=False)
        
        # Get top 3 parameter sets
        print("\nTop 3 RSI Parameter Combinations:")
        for i, row in results_df.head(3).iterrows():
            print(f"{i+1}. Oversold={row['oversold']}, Overbought={row['overbought']}: "
                 f"Return={row['return']:.2f}%, Win Rate={row['win_rate']:.1f}%, "
                 f"Profit Factor={row['profit_factor']:.2f}, Trades={row['trades']}")
        
        # Generate heatmap visualization
        plt.figure(figsize=(12, 8))
        
        # Create pivot table for heatmap
        pivot_data = results_df.pivot_table(
            index='oversold', 
            columns='overbought', 
            values='return',
            aggfunc='mean'
        )
        
        # Plot heatmap
        ax = plt.subplot(1, 1, 1)
        sns.heatmap(
            pivot_data, 
            annot=True, 
            fmt=".2f", 
            cmap="RdYlGn", 
            center=0,
            ax=ax
        )
        
        plt.title(f'RSI Parameter Optimization - {config_name} ({timeframe}, {days} days)')
        plt.tight_layout()
        
        # Save visualization
        output_dir = '/home/panal/Documents/dashboard-trading/reports/optimization'
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"rsi_optimization_{config_name}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.png")
        plt.savefig(output_file)
        print(f"\nOptimization heatmap saved to: {output_file}")
        
        # Save full results to CSV
        csv_file = os.path.join(output_dir, f"rsi_optimization_{config_name}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv")
        results_df.to_csv(csv_file, index=False)
        print(f"Detailed results saved to: {csv_file}")
        
        # Return best parameters
        best_params = results_df.iloc[0]
        return {
            'oversold': int(best_params['oversold']),
            'overbought': int(best_params['overbought']),
            'return': best_params['return'],
            'trades': best_params['trades']
        }
    
    else:
        print("No valid results found")
        return None

def create_optimized_config(config_name, best_params, output_suffix='_optimized'):
    """Create a new optimized configuration file based on best parameters"""
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
    if not os.path.exists(config_path):
        print(f"Configuration not found: {config_name}")
        return False
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Apply optimized parameters
    if 'rsi' not in config:
        config['rsi'] = {}
    
    config['rsi']['oversold'] = best_params['oversold']
    config['rsi']['overbought'] = best_params['overbought']
    
    # Save as new config file
    output_name = f"{config_name}{output_suffix}"
    output_path = f"/home/panal/Documents/dashboard-trading/configs/{output_name}.json"
    
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Created optimized config: {output_path}")
    return True

if __name__ == "__main__":
    import argparse
    import seaborn as sns
    
    parser = argparse.ArgumentParser(description='Optimize strategy parameters')
    parser.add_argument('--config', type=str, required=True, help='Strategy configuration to optimize')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to use for optimization')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to use')
    parser.add_argument('--save', action='store_true', help='Save optimized configuration')
    
    args = parser.parse_args()
    
    # Run optimization
    best_params = optimize_rsi_parameters(args.config, args.timeframe, args.days)
    
    if best_params and args.save:
        create_optimized_config(args.config, best_params)
