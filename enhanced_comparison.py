#!/usr/bin/env python
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from utils.strategy_comparison import load_analysis_results

def create_enhanced_comparisons():
    """Generate enhanced strategy comparison visualizations"""
    print("Generating enhanced strategy comparisons...")
    
    # Load results from all analysis files
    results = load_analysis_results()
    
    if results.empty:
        print("No analysis results found")
        return
        
    # Create output directory
    output_dir = '/home/panal/Documents/dashboard-trading/reports/enhanced_comparisons'
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # 1. Strategy Performance Quadrant Chart
    print("Creating Performance Quadrant Chart...")
    plt.figure(figsize=(12, 10))
    
    # Calculate risk-adjusted return (return / max_drawdown)
    results['risk_adjusted'] = results['return_total'] / results['max_drawdown'].replace(0, 0.01)
    
    # Create scatter plot
    scatter = plt.scatter(
        results['max_drawdown'],
        results['return_total'],
        s=results['total_trades'] * 3,  # Size based on number of trades
        c=results['profit_factor'],    # Color based on profit factor
        cmap='viridis',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Profit Factor')
    
    # Add labels for each point
    for i, row in results.iterrows():
        plt.annotate(
            row['config_name'],
            (row['max_drawdown'], row['return_total']),
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    # Add quadrant lines and labels
    plt.axhline(y=results['return_total'].median(), color='gray', linestyle='--', alpha=0.5)
    plt.axvline(x=results['max_drawdown'].median(), color='gray', linestyle='--', alpha=0.5)
    
    # Label quadrants
    plt.text(
        results['max_drawdown'].max() * 0.9, 
        results['return_total'].max() * 0.9,
        "High Return\nHigh Risk",
        ha='right'
    )
    plt.text(
        results['max_drawdown'].min() * 1.1, 
        results['return_total'].max() * 0.9,
        "High Return\nLow Risk\n(Optimal)",
        ha='left'
    )
    plt.text(
        results['max_drawdown'].max() * 0.9, 
        results['return_total'].min() * 1.1,
        "Low Return\nHigh Risk\n(Worst)",
        ha='right'
    )
    plt.text(
        results['max_drawdown'].min() * 1.1, 
        results['return_total'].min() * 1.1,
        "Low Return\nLow Risk",
        ha='left'
    )
    
    plt.title("Risk-Return Quadrant Analysis")
    plt.xlabel("Risk (Max Drawdown %)")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"risk_return_quadrant_{timestamp}.png"))
    
    # 2. Win Rate vs Return (instead of Avg Trade PnL since that field is missing)
    print("Creating Win Rate vs Return Chart...")
    plt.figure(figsize=(12, 8))
    
    scatter = plt.scatter(
        results['win_rate'],
        results['return_total'],  # Use return_total instead of avg_trade_pnl
        s=results['total_trades'] * 3,
        c=results['profit_factor'],
        cmap='RdYlGn',
        alpha=0.7
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter)
    cbar.set_label('Profit Factor')
    
    # Add labels for each point
    for i, row in results.iterrows():
        plt.annotate(
            row['config_name'],
            (row['win_rate'], row['return_total']),  # Use return_total instead of avg_trade_pnl
            fontsize=9,
            xytext=(5, 5),
            textcoords='offset points'
        )
    
    plt.title("Win Rate vs Return")
    plt.xlabel("Win Rate (%)")
    plt.ylabel("Return (%)")
    plt.axhline(y=0, color='red', linestyle='-', alpha=0.3)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"win_rate_vs_return_{timestamp}.png"))
    
    # 3. Strategy Ranking Table
    print("Creating Strategy Ranking...")
    
    # Calculate composite score based on multiple metrics
    results['return_rank'] = results['return_total'].rank(ascending=False)
    results['risk_adj_rank'] = results['risk_adjusted'].rank(ascending=False)
    results['win_rate_rank'] = results['win_rate'].rank(ascending=False)
    results['profit_factor_rank'] = results['profit_factor'].rank(ascending=False)
    
    # Calculate overall score (lower is better)
    results['overall_score'] = (results['return_rank'] + 
                              results['risk_adj_rank'] + 
                              results['win_rate_rank'] +
                              results['profit_factor_rank'])
    
    # Sort by overall score
    ranked_df = results.sort_values('overall_score')
    
    # Remove any duplicates by config_name
    ranked_df = ranked_df.drop_duplicates(subset='config_name')
    
    # Create a nice summary table
    summary = ranked_df[['config_name', 'return_total', 'win_rate', 
                        'profit_factor', 'max_drawdown', 'total_trades', 'overall_score']]
    summary.columns = ['Strategy', 'Return (%)', 'Win Rate (%)', 'Profit Factor', 
                      'Max DD (%)', 'Total Trades', 'Score']
    
    # Save ranking to CSV
    summary.to_csv(os.path.join(output_dir, f"strategy_ranking_{timestamp}.csv"), index=False)
    
    # Print top 3 strategies with correct enumeration
    print("\nTop 3 Strategies:")
    for i, row in summary.head(3).iterrows():
        performance_indicator = "⚠️ NEGATIVE RETURN" if row['Return (%)'] < 0 else "✅"
        print(f"{i+1}. {row['Strategy']}: Return={row['Return (%)']:.2f}%, Win Rate={row['Win Rate (%)']:.2f}%, "
              f"Profit Factor={row['Profit Factor']:.2f} {performance_indicator}")
    
    # Add overall performance assessment
    negative_returns = (summary['Return (%)'] < 0).sum()
    positive_returns = (summary['Return (%)'] > 0).sum()
    total_strategies = len(summary)
    
    print("\nPerformance Overview:")
    print(f"Total strategies analyzed: {total_strategies}")
    print(f"Strategies with positive returns: {positive_returns} ({positive_returns/total_strategies*100:.1f}%)")
    print(f"Strategies with negative returns: {negative_returns} ({negative_returns/total_strategies*100:.1f}%)")
    
    if negative_returns > positive_returns:
        print("\n⚠️ WARNING: Most strategies have negative returns. Consider:")
        print("  - Adjusting strategy parameters")
        print("  - Using a different timeframe")
        print("  - Checking for data quality issues")
        print("  - Evaluating market conditions")
    
    # 4. Strategy similarity analysis
    print("Creating Strategy Similarity Analysis...")
    
    # Create correlation matrix based on trade patterns
    similarity_report = "Strategy similarity analysis is not yet available.\n"
    similarity_report += "This feature would analyze the correlation between strategies\n"
    similarity_report += "based on their trade entry/exit patterns."
    
    with open(os.path.join(output_dir, f"strategy_similarity_{timestamp}.txt"), 'w') as f:
        f.write(similarity_report)
    
    print(f"Enhanced comparisons saved to: {output_dir}")

if __name__ == "__main__":
    create_enhanced_comparisons()
