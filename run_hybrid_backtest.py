#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import sys
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache
from strategies.adaptive_hybrid_strategy import AdaptiveHybridStrategy

def run_hybrid_backtest(config_name='hybrid_strategy', timeframes=['1h', '4h'], days=90, debug=False):
    """Run backtest with the new hybrid strategy across multiple timeframes"""
    print(f"=== RUNNING HYBRID STRATEGY BACKTEST ===")
    
    # Load configuration
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
    if not os.path.exists(config_path):
        print(f"Configuration not found: {config_path}")
        return None
        
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Add debug flag to config
    config['debug'] = debug
    
    # Prepare results collection
    results_by_timeframe = {}
    
    for timeframe in timeframes:
        print(f"\nTesting {config_name} on {timeframe} timeframe ({days} days)")
        
        # Get data
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
            print(f"No data available for {timeframe}")
            continue
            
        print(f"Loaded {len(data)} bars from {data.index[0]} to {data.index[-1]}")
        
        # Create backtest instance
        backtest = BacktestFixedStrategy(config=config)
        
        # Run backtest
        start_time = datetime.now()
        results = backtest.run(data)
        end_time = datetime.now()
        
        # Ensure equity curve exists in results
        if 'equity_curve' not in results:
            results['equity_curve'] = pd.Series(1.0, index=data.index)
        
        # Save results
        results_by_timeframe[timeframe] = {
            'return_total': results['return_total'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor'],
            'max_drawdown': results['max_drawdown'],
            'total_trades': results['total_trades'],
            'execution_time': (end_time - start_time).total_seconds(),
            'equity_curve': results['equity_curve']  # Add equity curve to results
        }
        
        # Report results
        print(f"\n--- Results for {timeframe} ---")
        print(f"Return: {results['return_total']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        
        # Generate visualization
        backtest.generate_report()
    
    # Create comparative visualization
    create_comparison_chart(results_by_timeframe, config_name)
    
    # Save overall results
    save_results(results_by_timeframe, config_name)
    
    return results_by_timeframe

def create_comparison_chart(results, config_name):
    """Create a comparison chart of results across timeframes"""
    if not results:
        return
        
    # Create directory
    output_dir = f"/home/panal/Documents/dashboard-trading/reports/hybrid_strategy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data
    timeframes = list(results.keys())
    returns = [results[tf]['return_total'] for tf in timeframes]
    win_rates = [results[tf]['win_rate'] for tf in timeframes]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create two subplots
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)
    
    # Plot returns
    bars1 = ax1.bar(timeframes, returns, color=['green' if r > 0 else 'red' for r in returns])
    ax1.set_title('Returns by Timeframe')
    ax1.set_ylabel('Return (%)')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Plot win rates
    bars2 = ax2.bar(timeframes, win_rates, color='blue')
    ax2.set_title('Win Rate by Timeframe')
    ax2.set_ylabel('Win Rate (%)')
    ax2.set_ylim([0, 100])  # Win rate is a percentage
    
    # Add value labels
    for bar in bars2:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}%',
                    xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{config_name}_comparison.png"))
    plt.close()
    
    print(f"Comparison chart saved to: {output_dir}/{config_name}_comparison.png")

def save_results(results, config_name):
    """Save backtest results to file"""
    output_dir = f"/home/panal/Documents/dashboard-trading/reports/hybrid_strategy"
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame
    data = []
    for timeframe, result in results.items():
        result_copy = result.copy()
        result_copy['timeframe'] = timeframe
        data.append(result_copy)
    
    if data:
        df = pd.DataFrame(data)
        
        # Save to CSV
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        csv_path = os.path.join(output_dir, f"{config_name}_results_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run hybrid strategy backtest')
    parser.add_argument('--config', type=str, default='hybrid_strategy', help='Configuration to use')
    parser.add_argument('--timeframes', type=str, default='1h,4h', help='Comma-separated list of timeframes')
    parser.add_argument('--days', type=int, default=90, help='Number of days for backtest')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')  # Add debug flag
    
    args = parser.parse_args()
    
    # Parse timeframes
    timeframe_list = args.timeframes.split(',')
    
    # Run backtest with debug flag
    run_hybrid_backtest(args.config, timeframe_list, args.days, args.debug)
