#!/usr/bin/env python
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache

def compare_strategies(timeframe='1h', days=30, debug=False):
    """Compare all strategies in the configs directory"""
    print(f"=== COMPARING ALL STRATEGIES ===")
    print(f"Timeframe: {timeframe}")
    print(f"Period: {days} days")

    # Load all config files
    config_dir = "/home/panal/Documents/dashboard-trading/configs"
    config_files = [f for f in os.listdir(config_dir) if f.endswith('.json')]
    
    print(f"\nFound {len(config_files)} configurations to test")
    
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
        print("No data available for testing")
        return
    
    # Initialize empty DataFrame with required columns
    df = pd.DataFrame(columns=[
        'config_name',
        'return',
        'win_rate',
        'profit_factor',
        'max_drawdown',
        'trades',
        'parameters'
    ])
    
    # Test each configuration
    for config_file in config_files:
        config_name = config_file.replace('.json', '')
        print(f"\nTesting {config_name}...")
        
        try:
            # Load config with its specific parameters
            with open(os.path.join(config_dir, config_file), 'r') as f:
                config = json.load(f)
                
                # Verify required parameters exist
                if 'rsi' not in config:
                    print(f"Warning: No RSI parameters in {config_name}, skipping...")
                    continue
                    
                # Add debug flag if needed
                config['debug'] = debug
                
                if debug:
                    print(f"Configuration parameters:")
                    print(f"RSI Oversold: {config['rsi'].get('oversold', 'not set')}")
                    print(f"RSI Overbought: {config['rsi'].get('overbought', 'not set')}")
                    print(f"EMA Short: {config.get('ema', {}).get('short', 'not set')}")
                    print(f"EMA Long: {config.get('ema', {}).get('long', 'not set')}")
                
                # Run backtest with config's specific parameters
                backtest = BacktestFixedStrategy(config=config)
                result = backtest.run(data)
                
                # Add results to DataFrame
                df = pd.concat([df, pd.DataFrame([{
                    'config_name': config_name,
                    'return': result['return_total'],
                    'win_rate': result['win_rate'],
                    'profit_factor': result['profit_factor'],
                    'max_drawdown': result['max_drawdown'],
                    'trades': result['total_trades'],
                    'parameters': {
                        'rsi_oversold': config['rsi'].get('oversold'),
                        'rsi_overbought': config['rsi'].get('overbought'),
                        'ema_short': config.get('ema', {}).get('short'),
                        'ema_long': config.get('ema', {}).get('long')
                    }
                }])], ignore_index=True)
                
                print(f"Return: {result['return_total']:.2f}%")
                print(f"Win Rate: {result['win_rate']:.2f}%")
                print(f"Trades: {result['total_trades']}")
                
        except Exception as e:
            print(f"Error testing {config_name}: {str(e)}")
            continue

    # Ensure we have results before sorting
    if len(df) > 0:
        df_sorted = df.sort_values('return', ascending=False)
    else:
        print("No valid results to compare")
        return df
    
    # Print summary
    print("\n=== STRATEGY COMPARISON SUMMARY ===")
    print("\nTop 5 Strategies by Return:")
    for i, row in df_sorted.head().iterrows():
        print(f"\n{i+1}. {row['config_name']}:")
        print(f"   Return: {row['return']:.2f}%")
        print(f"   Win Rate: {row['win_rate']:.2f}%")
        print(f"   Profit Factor: {row['profit_factor']:.2f}")
        print(f"   Trades: {row['trades']}")
    
    # Create visualizations
    output_dir = "/home/panal/Documents/dashboard-trading/reports/comparisons"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Returns comparison
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df_sorted['config_name'], df_sorted['return'])
    for bar in bars:
        if bar.get_height() > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
            
    plt.title('Strategy Returns Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"returns_comparison_{timeframe}.png"))
    plt.close()
    
    # 2. Risk-Return scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['max_drawdown'], df['return'], alpha=0.6)
    
    # Add labels for each point
    for i, row in df.iterrows():
        plt.annotate(row['config_name'], 
                    (row['max_drawdown'], row['return']),
                    xytext=(5, 5), textcoords='offset points')
    
    plt.title('Risk-Return Analysis')
    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('Return (%)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"risk_return_{timeframe}.png"))
    plt.close()
    
    # Save results to CSV
    csv_file = os.path.join(output_dir, f"strategy_comparison_{timeframe}_{datetime.now().strftime('%Y%m%d')}.csv")
    df.to_csv(csv_file, index=False)
    
    print(f"\nResults saved to: {csv_file}")
    return df

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare all trading strategies')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe to test')
    parser.add_argument('--days', type=int, default=30, help='Number of days to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    results_df = compare_strategies(args.timeframe, args.days, args.debug)
