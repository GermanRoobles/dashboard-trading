import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime

def load_analysis_results(analysis_dir='/home/panal/Documents/dashboard-trading/reports/analysis'):
    """Load all analysis JSON files from the directory"""
    results = []
    
    for filename in os.listdir(analysis_dir):
        if filename.startswith('analysis_') and filename.endswith('.json'):
            filepath = os.path.join(analysis_dir, filename)
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # Extract key information
                config_name = filename.split('_')[1]
                
                if 'results' in data:
                    strategy_data = {
                        'config_name': config_name,
                        'filepath': filepath,
                        'timestamp': os.path.getmtime(filepath),
                        'date': datetime.fromtimestamp(os.path.getmtime(filepath)).strftime('%Y-%m-%d')
                    }
                    
                    # Add result metrics
                    for key, value in data['results'].items():
                        if key in ['return_total', 'win_rate', 'profit_factor', 'max_drawdown', 'total_trades']:
                            try:
                                strategy_data[key] = float(value)
                            except (ValueError, TypeError):
                                strategy_data[key] = 0
                    
                    results.append(strategy_data)
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
    
    return pd.DataFrame(results)

def compare_strategies(results=None):
    """Compare strategy performance and create visualizations"""
    if results is None:
        results = load_analysis_results()
        
    if results.empty:
        print("No analysis results found")
        return
    
    # Sort by return (best first)
    results = results.sort_values(by='return_total', ascending=False)
    
    print("\n=== STRATEGY COMPARISON ===")
    print(f"Found {len(results)} analyzed strategies")
    
    # Create output directory
    output_dir = '/home/panal/Documents/dashboard-trading/reports/comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 1. Performance table
    print("\nPerformance Summary (sorted by return):")
    summary = results[['config_name', 'return_total', 'win_rate', 'profit_factor', 'max_drawdown', 'total_trades']]
    summary.columns = ['Strategy', 'Return (%)', 'Win Rate (%)', 'Profit Factor', 'Max DD (%)', 'Total Trades']
    print(summary)
    
    # Save to CSV
    summary.to_csv(os.path.join(output_dir, f'strategy_comparison_{timestamp}.csv'), index=False)
    
    # 2. Return comparison chart
    plt.figure(figsize=(12, 7))
    ax = sns.barplot(x='config_name', y='return_total', data=results, palette='viridis')
    plt.title('Strategy Return Comparison')
    plt.xlabel('Strategy')
    plt.ylabel('Return (%)')
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(results['return_total']):
        ax.text(
            i, 
            v + (0.5 if v >= 0 else -1.0),  # Position text above or below bar
            f"{v:.2f}%", 
            ha='center'
        )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'return_comparison_{timestamp}.png'))
    
    # 3. Risk-return scatter plot
    plt.figure(figsize=(10, 8))
    
    # Add size dimension based on total trades
    size_scale = results['total_trades'] / results['total_trades'].max() * 200 + 50
    
    # Create scatter plot
    scatter = plt.scatter(
        x=results['max_drawdown'],
        y=results['return_total'],
        s=size_scale,  # Size based on number of trades
        c=results['profit_factor'],  # Color based on profit factor
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar for profit factor
    cbar = plt.colorbar(scatter)
    cbar.set_label('Profit Factor')
    
    # Add annotation for each strategy
    for i, row in results.iterrows():
        plt.annotate(
            row['config_name'],
            (row['max_drawdown'], row['return_total']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Risk-Return Profile of Trading Strategies')
    plt.xlabel('Max Drawdown (%)')
    plt.ylabel('Return (%)')
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'risk_return_scatter_{timestamp}.png'))
    
    # 4. Win rate vs profit factor
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot
    scatter = plt.scatter(
        x=results['win_rate'],
        y=results['profit_factor'],
        s=size_scale,  # Size based on number of trades
        c=results['return_total'],  # Color based on return
        cmap='RdYlGn',
        alpha=0.7
    )
    
    # Add colorbar for return
    cbar = plt.colorbar(scatter)
    cbar.set_label('Return (%)')
    
    # Add annotation for each strategy
    for i, row in results.iterrows():
        plt.annotate(
            row['config_name'],
            (row['win_rate'], row['profit_factor']),
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title('Win Rate vs Profit Factor')
    plt.xlabel('Win Rate (%)')
    plt.ylabel('Profit Factor')
    plt.axhline(y=1, color='red', linestyle='--', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'winrate_vs_profitfactor_{timestamp}.png'))
    
    print(f"\nComparison visualizations saved to: {output_dir}")
    
    # Calculate best strategies for different metrics
    print("\nBest Strategies by Metric:")
    metrics = {
        'Return': results.loc[results['return_total'].idxmax()],
        'Win Rate': results.loc[results['win_rate'].idxmax()],
        'Profit Factor': results.loc[results['profit_factor'].idxmax()],
        'Min Drawdown': results.loc[results['max_drawdown'].idxmin()]
    }
    
    for metric, row in metrics.items():
        print(f"- Best {metric}: {row['config_name']} ({row['return_total']:.2f}% return, " +
              f"{row['win_rate']:.2f}% win rate, {row['profit_factor']:.2f} profit factor)")
              
    return results

def main():
    """Run strategy comparison from command line"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare trading strategy performance')
    parser.add_argument('--dir', type=str, default=None, help='Directory with analysis result files')
    
    args = parser.parse_args()
    
    # Run comparison
    if args.dir:
        results = load_analysis_results(args.dir)
    else:
        results = None
        
    compare_strategies(results)

if __name__ == "__main__":
    main()
