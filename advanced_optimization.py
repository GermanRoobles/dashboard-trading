#!/usr/bin/env python
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache
from utils.market_analyzer import MarketAnalyzer

class AdvancedOptimizer:
    """Advanced parameter optimization with cross-validation and regime-specific tuning"""
    
    def __init__(self, config_name, base_timeframe='1h', debug=False):
        """Initialize optimizer with configuration and base timeframe"""
        self.config_name = config_name
        self.base_timeframe = base_timeframe
        self.debug = debug
        self.config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
        
        # Output directory for results
        self.output_dir = f"/home/panal/Documents/dashboard-trading/reports/optimization/{config_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load base configuration
        with open(self.config_path, 'r') as f:
            self.base_config = json.load(f)
            
        # Initialize market analyzer for regime detection
        self.market_analyzer = MarketAnalyzer()
        
        # Initialize data cache
        self.cache = DataCache()
        
    def run_optimization_pipeline(self, days=90, validation_split=0.3, regimes=True):
        """Run the complete optimization pipeline with validation"""
        print(f"=== ADVANCED OPTIMIZATION FOR {self.config_name} ===")
        print(f"Base timeframe: {self.base_timeframe}")
        print(f"Data period: {days} days with {validation_split*100:.0f}% validation split")
        
        # Load data
        data = self._load_data(days)
        if data is None:
            return
            
        # Split data into training and validation sets
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        valid_data = data.iloc[split_idx:]
        
        print(f"Training data: {len(train_data)} bars from {train_data.index[0]} to {train_data.index[-1]}")
        print(f"Validation data: {len(valid_data)} bars from {valid_data.index[0]} to {valid_data.index[-1]}")
        
        # Step 1: Run standard parameter grid search on training data
        print("\n=== STEP 1: PARAMETER GRID SEARCH ===")
        grid_results = self._run_parameter_grid_search(train_data)
        
        # Find top configurations
        top_configs = grid_results.sort_values('return', ascending=False).head(5)
        print("\nTop 5 configurations from grid search:")
        for i, row in top_configs.iterrows():
            print(f"{i+1}. {row['config_desc']} - Return: {row['return']:.2f}%, Win Rate: {row['win_rate']:.2f}%")
        
        # Step 2: Validate top configurations
        print("\n=== STEP 2: VALIDATING TOP CONFIGURATIONS ===")
        validation_results = []
        
        for i, row in top_configs.iterrows():
            config = row['config']
            print(f"\nValidating configuration: {row['config_desc']}")
            
            # Run backtest on validation data
            backtest = BacktestFixedStrategy(config=config)
            val_result = backtest.run(valid_data)
            
            # Calculate validation metrics
            val_return = val_result['return_total']
            train_return = row['return']
            consistency = val_return / train_return if train_return > 0 and val_return > 0 else 0
            robustness = val_return / abs(train_return - val_return) if abs(train_return - val_return) > 0 else 0
            
            print(f"Training return: {train_return:.2f}%")
            print(f"Validation return: {val_return:.2f}%")
            print(f"Consistency score: {consistency:.2f}")
            print(f"Robustness score: {robustness:.2f}")
            
            validation_results.append({
                'config': config,
                'config_desc': row['config_desc'],
                'train_return': train_return,
                'val_return': val_return,
                'consistency': consistency,
                'robustness': robustness,
                'train_win_rate': row['win_rate'],
                'val_win_rate': val_result['win_rate'],
                'val_profit_factor': val_result['profit_factor']
            })
        
        # Convert to DataFrame for analysis
        val_df = pd.DataFrame(validation_results)
        
        # Step 3: Regime-specific optimization (optional)
        if regimes:
            print("\n=== STEP 3: REGIME-SPECIFIC OPTIMIZATION ===")
            regime_results = self._optimize_by_regime(train_data, valid_data)
        else:
            regime_results = None
            
        # Step 4: Finalize optimal configuration
        print("\n=== STEP 4: FINALIZING OPTIMAL CONFIGURATION ===")
        
        # Select best configuration based on consistency and validation return
        val_df['combined_score'] = val_df['val_return'] * val_df['consistency']
        best_idx = val_df['combined_score'].idxmax()
        best_config = val_df.loc[best_idx]['config']
        best_desc = val_df.loc[best_idx]['config_desc']
        
        print(f"Selected optimal configuration: {best_desc}")
        print(f"Training return: {val_df.loc[best_idx]['train_return']:.2f}%")
        print(f"Validation return: {val_df.loc[best_idx]['val_return']:.2f}%")
        print(f"Consistency score: {val_df.loc[best_idx]['consistency']:.2f}")
        print(f"Validation win rate: {val_df.loc[best_idx]['val_win_rate']:.2f}%")
        print(f"Validation profit factor: {val_df.loc[best_idx]['val_profit_factor']:.2f}")
        
        # Save optimized configuration
        optimized_name = f"{self.config_name}_optimized"
        optimized_path = f"/home/panal/Documents/dashboard-trading/configs/{optimized_name}.json"
        
        with open(optimized_path, 'w') as f:
            json.dump(best_config, f, indent=2)
            
        print(f"\nOptimized configuration saved to: {optimized_path}")
        
        # Create visualization of results
        self._visualize_results(grid_results, val_df, regime_results)
        
        return best_config
        
    def _load_data(self, days):
        """Load historical data for optimization"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=self.base_timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None or len(data) < days * 12:  # Rough check for sufficient data
            print(f"Error: Insufficient data for timeframe {self.base_timeframe} over {days} days")
            return None
            
        return data
        
    def _run_parameter_grid_search(self, data):
        """Run parameter grid search with improved debugging"""
        if self.debug:
            print("\nDEBUG: Starting parameter grid search")
            print(f"DEBUG: Data shape: {data.shape}")
            print(f"DEBUG: Date range: {data.index[0]} to {data.index[-1]}")

        parameter_grid = {
            'rsi': {
                'window': [7, 10, 14, 21],
                'oversold': [20, 25, 30, 35, 40],
                'overbought': [60, 65, 70, 75, 80]
            },
            'ema': {
                'short': [5, 7, 9, 12, 15],
                'long': [21, 26, 30, 40, 50]
            },
            'holding_time': [2, 3, 4, 5, 6],
            'trend_filter': [True, False],
            'volume_filter': [True, False]
        }
        
        # Add parameters for bidirectional trading
        parameter_grid['trading_bias'] = ['neutral', 'long_bias', 'short_bias']
        parameter_grid['signal_threshold'] = [0.5, 0.6, 0.7]
        
        # Generate test configurations
        test_configs = []
        
        # 1. RSI parameter combinations
        print("\nGenerating RSI parameter combinations...")
        for window in parameter_grid['rsi']['window']:
            for oversold in parameter_grid['rsi']['oversold']:
                for overbought in parameter_grid['rsi']['overbought']:
                    if oversold >= overbought:
                        continue
                        
                    config = self.base_config.copy()
                    if 'rsi' not in config:
                        config['rsi'] = {}
                    config['rsi']['window'] = window
                    config['rsi']['oversold'] = oversold
                    config['rsi']['overbought'] = overbought
                    
                    desc = f"RSI(window={window},os={oversold},ob={overbought})"
                    test_configs.append({'config': config, 'desc': desc})
        
        # 2. EMA parameter combinations
        print("Generating EMA parameter combinations...")
        for short in parameter_grid['ema']['short']:
            for long in parameter_grid['ema']['long']:
                if short >= long:
                    continue
                    
                config = self.base_config.copy()
                if 'ema' not in config:
                    config['ema'] = {}
                config['ema']['short'] = short
                config['ema']['long'] = long
                
                desc = f"EMA(short={short},long={long})"
                test_configs.append({'config': config, 'desc': desc})
        
        # 3. Holding time combinations
        print("Generating holding time combinations...")
        for holding_time in parameter_grid['holding_time']:
            config = self.base_config.copy()
            config['holding_time'] = holding_time
            
            desc = f"HoldingTime={holding_time}"
            test_configs.append({'config': config, 'desc': desc})
        
        # 4. Filter combinations
        print("Generating filter combinations...")
        for trend_filter in parameter_grid['trend_filter']:
            for volume_filter in parameter_grid['volume_filter']:
                config = self.base_config.copy()
                config['trend_filter'] = trend_filter
                config['volume_filter'] = volume_filter
                
                desc = f"Filters(trend={trend_filter},volume={volume_filter})"
                test_configs.append({'config': config, 'desc': desc})
        
        # 5. Combined optimized parameters (selected combinations)
        print("Generating combined parameter settings...")
        
        # Best RSI parameters based on preliminary testing
        rsi_params = [
            {'window': 14, 'oversold': 30, 'overbought': 70},
            {'window': 14, 'oversold': 25, 'overbought': 75},
            {'window': 10, 'oversold': 30, 'overbought': 70}
        ]
        
        # Best EMA parameters based on preliminary testing
        ema_params = [
            {'short': 9, 'long': 26},
            {'short': 12, 'long': 26},
            {'short': 9, 'long': 21}
        ]
        
        for rsi in rsi_params:
            for ema in ema_params:
                for holding_time in [3, 4]:
                    config = self.base_config.copy()
                    if 'rsi' not in config:
                        config['rsi'] = {}
                    if 'ema' not in config:
                        config['ema'] = {}
                        
                    config['rsi']['window'] = rsi['window']
                    config['rsi']['oversold'] = rsi['oversold']
                    config['rsi']['overbought'] = rsi['overbought']
                    config['ema']['short'] = ema['short']
                    config['ema']['long'] = ema['long']
                    config['holding_time'] = holding_time
                    config['trend_filter'] = True
                    config['volume_filter'] = True
                    
                    desc = f"Combined(RSI={rsi['oversold']}-{rsi['overbought']},EMA={ema['short']}-{ema['long']},HT={holding_time})"
                    test_configs.append({'config': config, 'desc': desc})
        
        # Add trading bias variations
        print("Generating trading bias combinations...")
        for bias in parameter_grid['trading_bias']:
            for threshold in parameter_grid['signal_threshold']:
                config = self.base_config.copy()
                
                # Add trading bias to config
                config['trading_bias'] = bias
                config['signal_threshold'] = threshold
                
                desc = f"TradingBias={bias},Threshold={threshold}"
                test_configs.append({'config': config, 'desc': desc})
        
        # Convert RSI parameters to float
        for test in test_configs:
            if 'rsi' in test['config']:
                test['config']['rsi']['oversold'] = float(test['config']['rsi']['oversold'])
                test['config']['rsi']['overbought'] = float(test['config']['rsi']['overbought'])
        
        # Run tests
        print(f"\nRunning grid search with {len(test_configs)} parameter combinations...")
        results = []
        
        # Add debug output during testing
        for i, test in enumerate(test_configs):
            if self.debug and i % 5 == 0:
                print(f"\nDEBUG: Testing configuration {i+1}/{len(test_configs)}")
                print(f"DEBUG: Config parameters: {test['desc']}")

            try:
                # Convert numeric parameters
                if 'rsi' in test['config']:
                    test['config']['rsi']['oversold'] = float(test['config']['rsi']['oversold'])
                    test['config']['rsi']['overbought'] = float(test['config']['rsi']['overbought'])
                    test['config']['rsi']['window'] = int(test['config']['rsi']['window'])
                    
                if 'ema' in test['config']:
                    test['config']['ema']['short'] = int(test['config']['ema']['short'])
                    test['config']['ema']['long'] = int(test['config']['ema']['long'])
                
                # Create backtest instance
                backtest = BacktestFixedStrategy(config=test['config'])
                
                # Run backtest
                result = backtest.run(data)
                
                if self.debug:
                    print(f"DEBUG: Result - Return: {result['return_total']:.2f}%, "
                          f"Win Rate: {result['win_rate']:.2f}%, "
                          f"Trades: {result['total_trades']}")

                # Validate result structure and convert values
                if isinstance(result, dict) and 'return_total' in result:
                    results.append({
                        'config': test['config'],
                        'config_desc': test['desc'],
                        'return_total': float(result['return_total']),
                        'win_rate': float(result['win_rate']),
                        'profit_factor': float(result['profit_factor']),
                        'max_drawdown': float(result['max_drawdown']),
                        'trades': int(result['total_trades'])
                    })
                else:
                    print(f"Invalid result format for {test['desc']}")
                    
            except Exception as e:
                if self.debug:
                    print(f"DEBUG: Error in config {test['desc']}: {str(e)}")
                continue
        
        # Create DataFrame with correct column names
        if results:
            results_df = pd.DataFrame(results)
            results_df = results_df.rename(columns={'return_total': 'return'})
            
            # Save results to CSV
            csv_path = os.path.join(self.output_dir, f"grid_search_results_{datetime.now().strftime('%Y%m%d')}.csv")
            results_df.sort_values('return', ascending=False).to_csv(csv_path, index=False)
            
            print(f"\nGrid search complete. Found {len(results)} valid configurations.")
            print(f"Results saved to: {csv_path}")
            
            return results_df
        else:
            print("\nNo valid results found during grid search.")
            return pd.DataFrame({'return': [], 'win_rate': [], 'profit_factor': [], 'max_drawdown': [], 'trades': []})
        
    def _optimize_by_regime(self, train_data, valid_data):
        """Optimize parameters for different market regimes"""
        # Identify regimes in the data
        regimes = ['trending_up', 'trending_down', 'ranging', 'volatile']
        regime_data = self._extract_regime_segments(train_data)
        
        regime_results = {}
        
        # Optimize for each regime
        for regime, data in regime_data.items():
            if data is None or len(data) < 50:  # Skip regimes with insufficient data
                print(f"Skipping regime '{regime}' due to insufficient data")
                continue
                
            print(f"\nOptimizing for {regime} market regime ({len(data)} bars)")
            
            # Use a simplified parameter grid for regime-specific optimization
            parameter_grid = {
                'rsi': {
                    'oversold': [25, 30, 35] if regime != 'volatile' else [20, 25, 30],
                    'overbought': [65, 70, 75] if regime != 'volatile' else [70, 75, 80]
                },
                'ema': {
                    'short': [7, 9, 12],
                    'long': [21, 26, 30]
                }
            }
            
            # Generate test configurations
            test_configs = []
            
            for oversold in parameter_grid['rsi']['oversold']:
                for overbought in parameter_grid['rsi']['overbought']:
                    if oversold >= overbought:
                        continue
                        
                    for short in parameter_grid['ema']['short']:
                        for long in parameter_grid['ema']['long']:
                            if short >= long:
                                continue
                                
                            config = self.base_config.copy()
                            if 'rsi' not in config:
                                config['rsi'] = {}
                            if 'ema' not in config:
                                config['ema'] = {}
                                
                            config['rsi']['oversold'] = oversold
                            config['rsi']['overbought'] = overbought
                            config['ema']['short'] = short
                            config['ema']['long'] = long
                            
                            desc = f"{regime}: RSI({oversold},{overbought}), EMA({short},{long})"
                            test_configs.append({'config': config, 'desc': desc})
            
            # Run tests for this regime
            results = []
            
            for test in test_configs:
                try:
                    backtest = BacktestFixedStrategy(config=test['config'])
                    result = backtest.run(data)
                    
                    results.append({
                        'config': test['config'],
                        'config_desc': test['desc'],
                        'return': result['return_total'],
                        'win_rate': result['win_rate'],
                        'profit_factor': result['profit_factor'],
                        'trades': result['total_trades']
                    })
                    
                except Exception as e:
                    print(f"Error testing {test['desc']}: {e}")
            
            # Find best configuration for this regime
            if results:
                results_df = pd.DataFrame(results)
                best_idx = results_df['return'].idxmax()
                best_result = results_df.loc[best_idx]
                
                print(f"Best configuration for {regime}:")
                print(f"  {best_result['config_desc']}")
                print(f"  Return: {best_result['return']:.2f}%, Win Rate: {best_result['win_rate']:.2f}%")
                
                # Store result
                regime_results[regime] = {
                    'config': best_result['config'],
                    'config_desc': best_result['config_desc'],
                    'return': best_result['return'],
                    'win_rate': best_result['win_rate'],
                    'profit_factor': best_result['profit_factor']
                }
        
        # Save regime-specific configurations
        if regime_results:
            # Create adaptive_regime config that incorporates regime-specific parameters
            adaptive_config = self.base_config.copy()
            
            if 'regime_params' not in adaptive_config:
                adaptive_config['regime_params'] = {}
                
            for regime, result in regime_results.items():
                adaptive_config['regime_params'][regime] = {
                    'rsi': {
                        'oversold': result['config']['rsi']['oversold'],
                        'overbought': result['config']['rsi']['overbought']
                    },
                    'ema': {
                        'short': result['config']['ema']['short'],
                        'long': result['config']['ema']['long']
                    }
                }
                
            # Save the adaptive config
            adaptive_name = f"{self.config_name}_adaptive_regimes"
            adaptive_path = f"/home/panal/Documents/dashboard-trading/configs/{adaptive_name}.json"
            
            with open(adaptive_path, 'w') as f:
                json.dump(adaptive_config, f, indent=2)
                
            print(f"\nAdaptive regime configuration saved to: {adaptive_path}")
                
        return regime_results
            
    def _extract_regime_segments(self, data, min_segment_size=50):
        """Extract data segments for different market regimes"""
        segments = {
            'trending_up': None,
            'trending_down': None,
            'ranging': None,
            'volatile': None
        }
        
        # Analyze each window of data to detect regime
        window_size = min_segment_size
        detected_regimes = []
        
        for i in range(0, len(data) - window_size, min_segment_size // 2):
            segment = data.iloc[i:i+window_size]
            regime = self.market_analyzer.detect_market_regime(segment)
            detected_regimes.append((i, i+window_size, regime))
        
        # Extract representative segments for each regime
        for regime in segments.keys():
            # Find all segments of this regime type
            matching_segments = [s for s in detected_regimes if s[2] == regime]
            
            if matching_segments:
                # Select the largest segment
                largest = max(matching_segments, key=lambda s: s[1] - s[0])
                start, end, _ = largest
                segments[regime] = data.iloc[start:end]
                
        return segments
        
    def _visualize_results(self, grid_results, val_results, regime_results=None):
        """Create visualizations of optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d')
        
        # 1. Return vs Win Rate scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(
            grid_results['win_rate'], 
            grid_results['return'],
            alpha=0.5,
            s=grid_results['trades'] / 5
        )
        
        # Highlight top 5 configurations
        top_5 = grid_results.sort_values('return', ascending=False).head()
        plt.scatter(
            top_5['win_rate'], 
            top_5['return'],
            color='red',
            s=100,
            marker='*',
            label='Top 5 Configurations'
        )
        
        plt.title(f'Return vs Win Rate - {self.config_name} Optimization')
        plt.xlabel('Win Rate (%)')
        plt.ylabel('Return (%)')
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
        plt.axvline(x=50, color='r', linestyle='-', alpha=0.3)
        plt.legend()
        
        # Save plot
        plt.savefig(os.path.join(self.output_dir, f"return_vs_winrate_{timestamp}.png"))
        plt.close()
        
        # 2. Training vs Validation Performance
        if not val_results.empty:
            plt.figure(figsize=(12, 6))
            
            # Create bar chart comparing training vs validation returns
            x = range(len(val_results))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], val_results['train_return'], width, label='Training Return')
            plt.bar([i + width/2 for i in x], val_results['val_return'], width, label='Validation Return')
            
            plt.title(f'Training vs Validation Returns - {self.config_name}')
            plt.xlabel('Configuration')
            plt.ylabel('Return (%)')
            plt.xticks(x, [f"Config {i+1}" for i in range(len(val_results))])
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, f"train_vs_validation_{timestamp}.png"))
            plt.close()
        
        # 3. Regime-specific performance if available
        if regime_results:
            regimes = list(regime_results.keys())
            returns = [regime_results[r]['return'] for r in regimes]
            win_rates = [regime_results[r]['win_rate'] for r in regimes]
            
            plt.figure(figsize=(12, 6))
            
            # Create grouped bar chart
            x = range(len(regimes))
            width = 0.35
            
            plt.bar([i - width/2 for i in x], returns, width, label='Return (%)')
            plt.bar([i + width/2 for i in x], win_rates, width, label='Win Rate (%)')
            
            plt.title(f'Performance by Market Regime - {self.config_name}')
            plt.xlabel('Market Regime')
            plt.ylabel('Value')
            plt.xticks(x, regimes)
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot
            plt.savefig(os.path.join(self.output_dir, f"regime_performance_{timestamp}.png"))
            plt.close()

def main():
    """Run the advanced optimization process"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run advanced strategy optimization')
    parser.add_argument('--config', type=str, required=True, help='Base configuration to optimize')
    parser.add_argument('--timeframe', type=str, default='1h', help='Base timeframe to optimize for')
    parser.add_argument('--days', type=int, default=90, help='Number of days of data to use')
    parser.add_argument('--no-regimes', action='store_true', help='Skip regime-specific optimization')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    # Create optimizer with debug flag
    optimizer = AdvancedOptimizer(args.config, args.timeframe, debug=args.debug)
    
    # Run optimization
    optimizer.run_optimization_pipeline(
        days=args.days,
        validation_split=0.3,  # 30% of data for validation
        regimes=not args.no_regimes
    )

if __name__ == "__main__":
    main()
