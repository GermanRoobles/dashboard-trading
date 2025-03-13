#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache

class EnhancedTestingFramework:
    """
    A framework for more robust strategy testing with:
    1. Multiple market regimes
    2. Varying market conditions
    3. Monte Carlo simulation
    4. Walk-forward optimization
    """
    
    def __init__(self, config_name):
        # Load the strategy configuration
        self.config_name = config_name
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Set up paths
        self.output_dir = f"/home/panal/Documents/dashboard-trading/reports/enhanced_tests/{config_name}"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data cache
        self.cache = DataCache()
        
    def run_regime_tests(self, timeframe='4h'):
        """Test strategy across different synthetic market regimes"""
        print(f"\n=== TESTING {self.config_name} ACROSS MARKET REGIMES ===")
        
        # Generate different market regime data
        regimes = {
            'uptrend': self._generate_regime_data('uptrend', timeframe),
            'downtrend': self._generate_regime_data('downtrend', timeframe),
            'ranging': self._generate_regime_data('ranging', timeframe),
            'volatile': self._generate_regime_data('volatile', timeframe)
        }
        
        # Test strategy on each regime
        results = {}
        for regime_name, regime_data in regimes.items():
            print(f"\nTesting on {regime_name} market...")
            
            # Run backtest
            backtest = BacktestFixedStrategy(config=self.config)
            result = backtest.run(regime_data)
            
            # Save results
            results[regime_name] = {
                'return': result['return_total'],
                'win_rate': result['win_rate'],
                'profit_factor': result['profit_factor'],
                'max_drawdown': result['max_drawdown'],
                'trades': result['total_trades']
            }
            
            print(f"  Return: {result['return_total']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Trades: {result['total_trades']}")
        
        # Visualize results
        self._visualize_regime_results(results)
        
        return results
    
    def run_monte_carlo(self, timeframe='4h', days=30, iterations=100):
        """Run Monte Carlo simulation to test strategy robustness"""
        print(f"\n=== MONTE CARLO SIMULATION FOR {self.config_name} ===")
        
        # Get base data
        data = self._get_data(timeframe, days)
        if data is None:
            return None
            
        # Run initial backtest
        backtest = BacktestFixedStrategy(config=self.config)
        base_result = backtest.run(data)
        
        print(f"Base backtest result:")
        print(f"  Return: {base_result['return_total']:.2f}%")
        print(f"  Win Rate: {base_result['win_rate']:.2f}%")
        print(f"  Trades: {base_result['total_trades']}")
        
        # Run Monte Carlo simulations
        if not backtest.trades or len(backtest.trades) < 5:
            print("Not enough trades for Monte Carlo simulation")
            return None
            
        # Extract trade results - fix key name
        trade_returns = [t.get('pnl', 0) for t in backtest.trades]  # Changed from pnl_pct to pnl
        
        # Run simulations
        print(f"Running {iterations} Monte Carlo simulations...")
        mc_results = []
        
        for i in range(iterations):
            # Resample trades with replacement
            sampled_returns = np.random.choice(trade_returns, size=len(trade_returns), replace=True)
            cumulative_return = (1 + np.array(sampled_returns) / 100).prod() - 1
            mc_results.append(cumulative_return * 100)  # convert to percentage
            
        # Calculate statistics
        mc_results.sort()
        median_return = np.median(mc_results)
        lower_5pct = np.percentile(mc_results, 5)
        upper_95pct = np.percentile(mc_results, 95)
        
        print(f"Monte Carlo results (from {iterations} simulations):")
        print(f"  Median return: {median_return:.2f}%")
        print(f"  5% worst case: {lower_5pct:.2f}%")
        print(f"  95% best case: {upper_95pct:.2f}%")
        print(f"  Probability of profit: {len([r for r in mc_results if r > 0]) / iterations * 100:.1f}%")
        
        # Visualize results
        self._visualize_monte_carlo(mc_results, base_result['return_total'])
        
        return {
            'base_return': base_result['return_total'],
            'median_return': median_return,
            'lower_5pct': lower_5pct,
            'upper_95pct': upper_95pct,
            'prob_profit': len([r for r in mc_results if r > 0]) / iterations * 100
        }
        
    def test_parameter_sensitivity(self, timeframe='4h', days=30):
        """Test strategy sensitivity to small parameter changes"""
        print(f"\n=== PARAMETER SENSITIVITY ANALYSIS FOR {self.config_name} ===")
        
        # Get data
        data = self._get_data(timeframe, days)
        if data is None:
            return None
        
        # Run base test
        backtest = BacktestFixedStrategy(config=self.config)
        base_result = backtest.run(data)
        
        print(f"Base backtest result:")
        print(f"  Return: {base_result['return_total']:.2f}%")
        print(f"  Win Rate: {base_result['win_rate']:.2f}%")
        print(f"  Trades: {base_result['total_trades']}")
        
        # Test parameter variations
        variations = []
        
        # Test RSI variations if present
        if 'rsi' in self.config:
            base_oversold = self.config['rsi'].get('oversold', 30)
            base_overbought = self.config['rsi'].get('overbought', 70)
            
            # Try small variations in RSI parameters
            for oversold_delta in [-3, -1, 0, 1, 3]:
                for overbought_delta in [-3, -1, 0, 1, 3]:
                    # Skip invalid combinations
                    if base_oversold + oversold_delta >= base_overbought + overbought_delta:
                        continue
                        
                    varied_config = self.config.copy()
                    varied_config['rsi'] = self.config['rsi'].copy()
                    varied_config['rsi']['oversold'] = base_oversold + oversold_delta
                    varied_config['rsi']['overbought'] = base_overbought + overbought_delta
                    
                    variation_name = f"RSI({base_oversold + oversold_delta},{base_overbought + overbought_delta})"
                    
                    try:
                        backtest = BacktestFixedStrategy(config=varied_config)
                        result = backtest.run(data)
                        
                        variations.append({
                            'name': variation_name,
                            'return': result['return_total'],
                            'win_rate': result['win_rate'],
                            'trades': result['total_trades'],
                            'return_diff': result['return_total'] - base_result['return_total']
                        })
                    except Exception as e:
                        print(f"Error testing {variation_name}: {str(e)}")
        
        # Test EMA variations if present
        if 'ema' in self.config:
            base_short = self.config['ema'].get('short', 9)
            base_long = self.config['ema'].get('long', 21)
            
            # Try small variations in EMA parameters
            for short_delta in [-2, -1, 0, 1, 2]:
                for long_delta in [-2, -1, 0, 1, 2]:
                    # Skip invalid combinations
                    if base_short + short_delta >= base_long + long_delta:
                        continue
                        
                    varied_config = self.config.copy()
                    varied_config['ema'] = self.config['ema'].copy()
                    varied_config['ema']['short'] = base_short + short_delta
                    varied_config['ema']['long'] = base_long + long_delta
                    
                    variation_name = f"EMA({base_short + short_delta},{base_long + long_delta})"
                    
                    try:
                        backtest = BacktestFixedStrategy(config=varied_config)
                        result = backtest.run(data)
                        
                        variations.append({
                            'name': variation_name,
                            'return': result['return_total'],
                            'win_rate': result['win_rate'],
                            'trades': result['total_trades'],
                            'return_diff': result['return_total'] - base_result['return_total']
                        })
                    except Exception as e:
                        print(f"Error testing {variation_name}: {str(e)}")
        
        # Convert to DataFrame and visualize
        if variations:
            variations_df = pd.DataFrame(variations)
            variations_df = variations_df.sort_values('return', ascending=False)
            
            print("\nParameter sensitivity results:")
            for i, row in variations_df.head().iterrows():
                print(f"  {row['name']}: {row['return']:.2f}% ({row['return_diff']:+.2f}%)")
                
            # Visualize results
            self._visualize_sensitivity(variations_df, base_result['return_total'])
            return variations_df
        else:
            print("No parameter variations to test")
            return None
    
    def _get_data(self, timeframe, days):
        """Get data from cache or download if necessary"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.cache.get_cached_data(
            symbol='BTC/USDT', 
            timeframe=timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None:
            print(f"No data available for {timeframe} timeframe")
            
        return data
    
    def _generate_regime_data(self, regime, timeframe, days=30, points=180):
        """Generate synthetic data for different market regimes"""
        # Get base data structure from real data
        real_data = self._get_data(timeframe, days)
        if real_data is None:
            return None
            
        # Start with the first point from real data
        start_price = real_data['close'].iloc[0]
        dates = pd.date_range(
            start=datetime.now() - timedelta(days=days),
            periods=points, 
            freq=timeframe.replace('4h', '4h').replace('1h', '1h')
        )
        
        # Generate regime-specific prices
        prices = [start_price]
        
        if regime == 'uptrend':
            # Create uptrend with occasional small pullbacks
            for i in range(1, points):
                change = np.random.normal(0.002, 0.01)  # Positive drift
                # Add occasional pullbacks
                if i % 15 == 0:  
                    change = np.random.normal(-0.01, 0.005)
                price = prices[-1] * (1 + change)
                prices.append(price)
                
        elif regime == 'downtrend':
            # Create downtrend with occasional small rallies
            for i in range(1, points):
                change = np.random.normal(-0.002, 0.01)  # Negative drift
                # Add occasional rallies
                if i % 15 == 0:
                    change = np.random.normal(0.01, 0.005)
                price = prices[-1] * (1 + change)
                prices.append(price)
                
        elif regime == 'ranging':
            # Create ranging market around a mean price
            mean_price = start_price
            for i in range(1, points):
                # Mean reversion factor - the further from mean, the stronger pull back
                deviation = (prices[-1] - mean_price) / mean_price
                mean_reversion = -deviation * 0.1
                
                # Random noise
                noise = np.random.normal(0, 0.005)
                
                # Combine for price change
                change = mean_reversion + noise
                price = prices[-1] * (1 + change)
                prices.append(price)
                
        elif regime == 'volatile':
            # Create volatile market with large moves both ways
            for i in range(1, points):
                change = np.random.normal(0, 0.025)  # Higher volatility
                price = prices[-1] * (1 + change)
                prices.append(price)
                
        # Create dataframe
        df = pd.DataFrame(index=dates)
        df['close'] = prices
        
        # Generate open, high, low based on close
        df['open'] = df['close'].shift(1)
        df.loc[df.index[0], 'open'] = df['close'].iloc[0] * 0.9995
        
        # Daily volatility for high/low calculation
        if regime == 'volatile':
            volatility_factor = 0.015
        elif regime == 'ranging':
            volatility_factor = 0.005
        else:
            volatility_factor = 0.01
            
        df['high'] = df[['open', 'close']].max(axis=1) * (1 + volatility_factor * np.random.random(len(df)))
        df['low'] = df[['open', 'close']].min(axis=1) * (1 - volatility_factor * np.random.random(len(df)))
        
        # Generate volume
        if regime == 'uptrend':
            # Higher volume on up days
            df['volume'] = np.where(
                df['close'] > df['open'],
                np.random.normal(1200, 300, len(df)),
                np.random.normal(800, 200, len(df))
            ) * df['close']
        elif regime == 'downtrend':
            # Higher volume on down days
            df['volume'] = np.where(
                df['close'] < df['open'],
                np.random.normal(1200, 300, len(df)),
                np.random.normal(800, 200, len(df))
            ) * df['close']
        else:
            df['volume'] = np.random.normal(1000, 250, len(df)) * df['close']
        
        df['volume'] = df['volume'].abs()
        
        return df
    
    def _visualize_regime_results(self, results):
        """Visualize strategy performance across regimes"""
        regimes = list(results.keys())
        returns = [results[r]['return'] for r in regimes]
        win_rates = [results[r]['win_rate'] for r in regimes]
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot returns
        bars1 = ax1.bar(regimes, returns, color=['green' if r > 0 else 'red' for r in returns])
        ax1.set_title(f'Returns by Market Regime - {self.config_name}')
        ax1.set_ylabel('Return (%)')
        ax1.set_ylim(min(min(returns) * 1.1, -0.5), max(max(returns) * 1.1, 0.5))
        
        # Add values on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        # Plot win rates
        bars2 = ax2.bar(regimes, win_rates, color='blue', alpha=0.7)
        ax2.set_title(f'Win Rates by Market Regime')
        ax2.set_ylabel('Win Rate (%)')
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.3)
        
        # Add values on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"regime_analysis_{datetime.now().strftime('%Y%m%d')}.png"))
        plt.close()
        
        print(f"Regime analysis chart saved to: {self.output_dir}")
    
    def _visualize_monte_carlo(self, mc_results, base_return):
        """Visualize Monte Carlo simulation results"""
        plt.figure(figsize=(12, 6))
        
        # Plot histogram of returns
        plt.hist(mc_results, bins=30, alpha=0.7, color='blue')
        
        # Add vertical lines for key values
        plt.axvline(x=base_return, color='red', linestyle='-', label=f'Actual Return: {base_return:.2f}%')
        plt.axvline(x=np.median(mc_results), color='green', linestyle='--', 
                   label=f'Median Return: {np.median(mc_results):.2f}%')
        plt.axvline(x=np.percentile(mc_results, 5), color='orange', linestyle=':', 
                   label=f'5% Worst Case: {np.percentile(mc_results, 5):.2f}%')
        
        plt.title(f'Monte Carlo Simulation - {self.config_name}')
        plt.xlabel('Return (%)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"monte_carlo_{datetime.now().strftime('%Y%m%d')}.png"))
        plt.close()
        
        print(f"Monte Carlo analysis chart saved to: {self.output_dir}")
    
    def _visualize_sensitivity(self, variations_df, base_return):
        """Visualize parameter sensitivity results"""
        plt.figure(figsize=(14, 8))
        
        # Sort by return for better visualization
        sorted_df = variations_df.sort_values('return', ascending=False)
        
        # Create bar chart
        bars = plt.bar(sorted_df['name'], sorted_df['return'], 
                      color=['green' if r > 0 else 'red' for r in sorted_df['return']])
        
        # Add horizontal line for base return
        plt.axhline(y=base_return, color='blue', linestyle='--', 
                   label=f'Base Return: {base_return:.2f}%')
        
        # Add labels
        plt.title(f'Parameter Sensitivity Analysis - {self.config_name}')
        plt.xlabel('Parameter Combination')
        plt.ylabel('Return (%)')
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"sensitivity_{datetime.now().strftime('%Y%m%d')}.png"))
        plt.close()
        
        print(f"Parameter sensitivity chart saved to: {self.output_dir}")

def run_enhanced_tests():
    """Run enhanced tests for a strategy"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run enhanced strategy tests')
    parser.add_argument('--config', type=str, required=True, help='Strategy configuration to test')
    parser.add_argument('--timeframe', type=str, default='4h', help='Timeframe to use')
    parser.add_argument('--days', type=int, default=30, help='Number of days to test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    parser.add_argument('--regimes', action='store_true', help='Run regime tests')
    parser.add_argument('--monte-carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--sensitivity', action='store_true', help='Run parameter sensitivity analysis')
    
    args = parser.parse_args()
    
    # Initialize framework
    framework = EnhancedTestingFramework(args.config)
    
    # Run requested tests
    print(f"\n=== ENHANCED TESTING FOR {args.config} ===")
    
    if args.all or args.regimes:
        framework.run_regime_tests(args.timeframe)
        
    if args.all or args.monte_carlo:
        framework.run_monte_carlo(args.timeframe, args.days)
        
    if args.all or args.sensitivity:
        framework.test_parameter_sensitivity(args.timeframe, args.days)
        
    print("\nEnhanced testing completed. Results saved to:", framework.output_dir)

if __name__ == "__main__":
    run_enhanced_tests()
