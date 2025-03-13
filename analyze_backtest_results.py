import os
import pandas as pd
import argparse

def analyze_backtest_results():
    """Analyze the backtest results to find the best configuration"""
    results_dir = "/home/panal/Documents/dashboard-trading/reports/hybrid_strategy"
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    if not results_files:
        print("No backtest results found.")
        return
    
    # Load the most recent results file
    results_files.sort(reverse=True)
    latest_results_file = os.path.join(results_dir, results_files[0])
    results_df = pd.read_csv(latest_results_file)
    
    # Find the configuration with the highest return
    best_config = results_df.loc[results_df['return_total'].idxmax()]
    
    print("Best Configuration:")
    print(best_config)
    
    # Add detailed trade analysis when --trades-only flag is used
    parser = argparse.ArgumentParser()
    parser.add_argument('--trades-only', action='store_true', help='Show detailed trade analysis')
    args = parser.parse_args()

    if args.trades_only:
        print("\n=== DETAILED TRADE ANALYSIS ===")
        print(f"Total trades: {best_config['total_trades']}")
        print(f"Win rate: {best_config['win_rate']:.2f}%")
        print(f"Return: {best_config['return_total']:.2f}%")
        print(f"Profit factor: {best_config['profit_factor']:.2f}")
        print(f"Max drawdown: {best_config['max_drawdown']:.2f}%")
        
        # Add trade distribution analysis
        if 'trades' in best_config:
            trades = pd.DataFrame(best_config['trades'])
            print("\nTrade Statistics:")
            print(f"Average profit per trade: {trades['pnl'].mean():.2f}%")
            print(f"Largest win: {trades['pnl'].max():.2f}%")
            print(f"Largest loss: {trades['pnl'].min():.2f}%")
            print(f"Average holding time: {trades['bars_held'].mean():.1f} bars")

    return best_config

if __name__ == "__main__":
    best_config = analyze_backtest_results()
