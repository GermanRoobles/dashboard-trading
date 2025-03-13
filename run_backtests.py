#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import time

# Import our components
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache
from utils.strategy_comparison import compare_strategies

def run_backtest(config_name, timeframe='1h', days=90, output=True):
    """Run backtest for a specific configuration"""
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
    
    # Check if the configuration exists
    if not os.path.exists(config_path):
        print(f"Configuration not found: {config_name}")
        return None
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Get data for backtest
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
        
    # Create backtest engine
    backtest = BacktestFixedStrategy(config=config)
    
    # Run backtest
    print(f"Running backtest for {config_name} on {timeframe} ({days} days)...")
    start_time = time.time()
    results = backtest.run(data)
    end_time = time.time()
    
    # Report results
    if output:
        print("\n==== BACKTEST RESULTS ====")
        print(f"Strategy: {config_name}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {days} days")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Return: {results['return_total']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Avg Trade: {results['avg_trade_pnl']:.2f}%")
        print(f"Execution time: {end_time - start_time:.2f} seconds")
        
        # Generate visual report
        report_info = backtest.generate_report()
        print(f"Report saved to: {report_info['output_dir']}")
        
    return results, backtest

def run_multiple_configs(configs, timeframes, periods):
    """Run backtests for multiple configurations"""
    all_results = []
    
    for config_name in configs:
        for timeframe in timeframes:
            for days in periods:
                try:
                    result, _ = run_backtest(config_name, timeframe, days)
                    if result:
                        all_results.append({
                            'config': config_name,
                            'timeframe': timeframe,
                            'days': days,
                            **result
                        })
                except Exception as e:
                    print(f"Error running backtest for {config_name} on {timeframe} ({days} days): {str(e)}")
    
    # Convert to DataFrame
    if all_results:
        result_df = pd.DataFrame(all_results)
        
        # Save to CSV
        output_file = f"reports/batch_backtest_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        result_df.to_csv(output_file, index=False)
        print(f"Batch results saved to {output_file}")
        
        # Create comparison chart
        plt.figure(figsize=(12, 7))
        plt.bar(
            result_df['config'] + ' (' + result_df['timeframe'] + ')', 
            result_df['return_total'],
            color=['green' if x > 0 else 'red' for x in result_df['return_total']]
        )
        plt.title('Backtest Returns by Strategy and Timeframe')
        plt.xticks(rotation=45)
        plt.ylabel('Return (%)')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        chart_file = f"reports/batch_comparison_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        plt.savefig(chart_file)
        print(f"Comparison chart saved to {chart_file}")
        
        return result_df
    
    return None

def main():
    parser = argparse.ArgumentParser(description='Run backtests for trading strategies')
    parser.add_argument('--config', type=str, default=None, help='Configuration to backtest')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe for analysis')
    parser.add_argument('--days', type=int, default=90, help='Number of days for backtest')
    parser.add_argument('--batch', action='store_true', help='Run batch of backtests')
    parser.add_argument('--compare', action='store_true', help='Compare existing results')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('reports', exist_ok=True)
    
    if args.compare:
        print("Comparing existing strategy results...")
        compare_strategies()
    elif args.batch:
        # Get all configurations
        configs = [f.split('.')[0] for f in os.listdir('configs') 
                if f.endswith('.json') and not f.startswith('.')]
        
        # Use selected config or all configs
        if args.config:
            if args.config in configs:
                configs = [args.config]
            else:
                print(f"Configuration not found: {args.config}")
                return
                
        timeframes = ['1h', '4h'] if args.timeframe == 'all' else [args.timeframe]
        periods = [30, 90, 180] if args.days == 0 else [args.days]
        
        print(f"Running batch test with {len(configs)} configs, {len(timeframes)} timeframes, {len(periods)} periods")
        result_df = run_multiple_configs(configs, timeframes, periods)
    else:
        # Single backtest
        if not args.config:
            print("Please specify a configuration with --config")
            return
        
        run_backtest(args.config, args.timeframe, args.days)

if __name__ == '__main__':
    main()
