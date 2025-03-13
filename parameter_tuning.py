#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache

def tune_multiple_parameters(config_name, timeframe='1h', days=30):
    """Tune multiple parameters for a strategy to improve performance"""
    print(f"=== PARAMETER TUNING FOR {config_name.upper()} ===")
    
    # Load base configuration
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
    if not os.path.exists(config_path):
        print(f"Configuration not found: {config_name}")
        return None
    
    with open(config_path, 'r') as f:
        base_config = json.load(f)
    
    # Get data for optimization
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = cache.get_cached_data(
        symbol='BTC/USDT', 
        timeframe=timeframe,
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if data is None:
        print(f"No data available for {timeframe} timeframe")
        return None
    
    print(f"Loaded {len(data)} bars of data from {data.index[0]} to {data.index[-1]}")
    
    # 1. First run a baseline test
    print("\n=== BASELINE TEST ===")
    backtest = BacktestFixedStrategy(config=base_config)
    baseline_result = backtest.run(data)
    
    print(f"Baseline performance:")
    print(f"Return: {baseline_result['return_total']:.2f}%")
    print(f"Win Rate: {baseline_result['win_rate']:.2f}%")
    print(f"Profit Factor: {baseline_result['profit_factor']:.2f}")
    print(f"Trades: {baseline_result['total_trades']}")
    
    # 2. Test different combinations of parameters
    print("\n=== TESTING PARAMETER COMBINATIONS ===")
    
    # Define parameter tests based on strategy type
    strategy_type = base_config.get('strategy', 'enhanced')
    
    parameter_grid = {
        # RSI parameters
        'rsi_oversold': [25, 30, 35, 40],
        'rsi_overbought': [60, 65, 70, 75],
        
        # EMA parameters
        'ema_short': [8, 9, 10, 12],
        'ema_long': [21, 24, 26, 30],
        
        # Holding time
        'holding_time': [3, 4, 5, 6]
    }
    
    # Create test configurations
    test_configs = []
    
    # Test RSI parameters
    for oversold in parameter_grid['rsi_oversold']:
        for overbought in parameter_grid['rsi_overbought']:
            if oversold >= overbought:
                continue
                
            # Create config copy
            test_config = base_config.copy()
            if 'rsi' not in test_config:
                test_config['rsi'] = {}
            
            test_config['rsi']['oversold'] = oversold
            test_config['rsi']['overbought'] = overbought
            
            test_configs.append({
                'config': test_config,
                'description': f"RSI({oversold},{overbought})"
            })
    
    # Test EMA parameters
    for short in parameter_grid['ema_short']:
        for long in parameter_grid['ema_long']:
            if short >= long:
                continue
                
            # Create config copy
            test_config = base_config.copy()
            if 'ema' not in test_config:
                test_config['ema'] = {}
            
            test_config['ema']['short'] = short
            test_config['ema']['long'] = long
            
            test_configs.append({
                'config': test_config,
                'description': f"EMA({short},{long})"
            })
    
    # Test holding time
    for holding_time in parameter_grid['holding_time']:
        # Create config copy
        test_config = base_config.copy()
        test_config['holding_time'] = holding_time
        
        test_configs.append({
            'config': test_config,
            'description': f"Holding({holding_time})"
        })
    
    # Add a fully optimized config
    optimized_config = base_config.copy()
    
    if 'rsi' not in optimized_config:
        optimized_config['rsi'] = {}
    optimized_config['rsi']['oversold'] = 35
    optimized_config['rsi']['overbought'] = 70
    
    if 'ema' not in optimized_config:
        optimized_config['ema'] = {}
    optimized_config['ema']['short'] = 9
    optimized_config['ema']['long'] = 26
    
    optimized_config['holding_time'] = 4
    optimized_config['trend_filter'] = True
    optimized_config['volume_filter'] = True
    
    test_configs.append({
        'config': optimized_config,
        'description': f"Fully Optimized"
    })
    
    # Run tests
    results = []
    
    for i, test in enumerate(test_configs):
        print(f"Testing configuration {i+1}/{len(test_configs)}: {test['description']}...")
        
        try:
            backtest = BacktestFixedStrategy(config=test['config'])
            result = backtest.run(data)
            
            # Save result
            results.append({
                'description': test['description'],
                'return': result['return_total'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'trades': result['total_trades'],
                'config': test['config']
            })
            
            print(f"  Return: {result['return_total']:.2f}%, Win Rate: {result['win_rate']:.2f}%, Trades: {result['total_trades']}")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    # 3. Analyze results
    print("\n=== RESULTS ===")
    
    if not results:
        print("No valid results found.")
        return
        
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by return (highest first)
    results_df = results_df.sort_values('return', ascending=False)
    
    # Top 5 configurations
    print("\nTop 5 configurations:")
    for i, row in results_df.head(5).iterrows():
        print(f"{i+1}. {row['description']}: Return={row['return']:.2f}%, Win Rate={row['win_rate']:.2f}%, Profit Factor={row['profit_factor']:.2f}, Trades={row['trades']}")
    
    # Visualize results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    
    # Return comparison
    sns.barplot(x='description', y='return', data=results_df.head(10), ax=axs[0])
    axs[0].set_title('Return by Configuration')
    axs[0].set_ylabel('Return (%)')
    axs[0].set_xlabel('')
    axs[0].tick_params(axis='x', rotation=45)
    axs[0].axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Win rate vs return
    scatter = axs[1].scatter(
        results_df['win_rate'],
        results_df['return'],
        s=results_df['trades'] * 2,
        alpha=0.7
    )
    
    axs[1].set_title('Win Rate vs Return')
    axs[1].set_xlabel('Win Rate (%)')
    axs[1].set_ylabel('Return (%)')
    axs[1].grid(True, alpha=0.3)
    axs[1].axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Add labels for top configurations
    for i, row in results_df.head(5).iterrows():
        axs[1].annotate(
            row['description'],
            (row['win_rate'], row['return']),
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.tight_layout()
    
    # Save visualization
    output_dir = '/home/panal/Documents/dashboard-trading/reports/tuning'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"parameter_tuning_{config_name}_{timeframe}_{datetime.now().strftime('%Y%m%d')}.png")
    plt.savefig(output_file)
    print(f"\nVisualization saved to: {output_file}")
    
    # 4. Create optimized config
    best_config = results_df.iloc[0]['config']
    output_name = f"{config_name}_tuned"
    output_path = f"/home/panal/Documents/dashboard-trading/configs/{output_name}.json"
    
    with open(output_path, 'w') as f:
        json.dump(best_config, f, indent=2)
    
    print(f"\nCreated optimized config: {output_path}")
    
    # Summary comparison with baseline
    best_return = results_df.iloc[0]['return']
    return_difference = best_return - baseline_result['return_total']
    
    print("\n=== IMPROVEMENT SUMMARY ===")
    print(f"Baseline Return: {baseline_result['return_total']:.2f}%")
    print(f"Optimized Return: {best_return:.2f}%")
    print(f"Improvement: {return_difference:.2f}% ({return_difference / abs(baseline_result['return_total']) * 100:.1f}% change)")
    
    return results_df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Tune strategy parameters')
    parser.add_argument('--config', type=str, required=True, help='Strategy configuration to tune')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days of data to use')
    
    args = parser.parse_args()
    
    # Run tuning
    tune_multiple_parameters(args.config, args.timeframe, args.days)
