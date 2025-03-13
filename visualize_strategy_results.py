#!/usr/bin/env python
import os
import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def visualize_strategy_analysis(analysis_file):
    """Visualize strategy analysis results from a JSON file"""
    # Load analysis data
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    # Extract config name from filename
    config_name = os.path.basename(analysis_file).split('_')[1]
    
    # Create output directory
    output_dir = os.path.join(
        os.path.dirname(analysis_file),
        f"visualizations_{config_name}_{datetime.now().strftime('%Y%m%d')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot regime distribution
    if 'regimes' in analysis:
        regimes = analysis['regimes']
        regime_names = list(regimes.keys())
        regime_counts = list(regimes.values())
        
        plt.figure(figsize=(10, 6))
        plt.bar(regime_names, regime_counts, color=sns.color_palette("viridis", len(regime_names)))
        plt.title(f'Market Regimes Distribution - {config_name}')
        plt.ylabel('Count')
        plt.xlabel('Regime')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'regime_distribution.png'))
        plt.close()
    
    # Plot performance metrics
    if 'results' in analysis:
        results = analysis['results']
        metrics = {
            'Return (%)': float(results.get('return_total', 0)),
            'Win Rate (%)': float(results.get('win_rate', 0)),
            'Profit Factor': float(results.get('profit_factor', 0)),
            'Max Drawdown (%)': float(results.get('max_drawdown', 0)),
            'Trades/Month': float(results.get('trades_per_month', 0))
        }
        
        # Plot metrics as a horizontal bar chart
        plt.figure(figsize=(10, 6))
        bars = plt.barh(
            list(metrics.keys()),
            list(metrics.values()),
            color=['green' if v > 0 else 'red' for v in metrics.values()]
        )
        
        # Add value labels to bars
        for bar in bars:
            width = bar.get_width()
            label_x_pos = width if width > 0 else 0
            plt.text(label_x_pos + 0.5, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', va='center')
        
        plt.title(f'Performance Metrics - {config_name}')
        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'performance_metrics.png'))
        plt.close()
    
    # Return path to the output directory
    return output_dir

def main():
    parser = argparse.ArgumentParser(description='Visualize strategy analysis results')
    parser.add_argument('file', help='Path to strategy analysis JSON file')
    
    args = parser.parse_args()
    
    # Check if file exists
    if not os.path.exists(args.file):
        print(f"Error: File not found: {args.file}")
        return
    
    # Visualize results
    output_dir = visualize_strategy_analysis(args.file)
    print(f"Visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()
