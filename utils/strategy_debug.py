import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
import os

def analyze_conditions(data: pd.DataFrame, strategy_name: str = 'optimized'):
    """Analyze why a strategy might not be producing signals"""
    if strategy_name.lower() == 'optimized':
        strategy = OptimizedStrategy()
    else:
        strategy = EnhancedStrategy()
    
    # Save original parameters
    original_params = strategy.params.copy()
    
    # Modify strategy to produce more signals for testing
    strategy.params['rsi']['oversold'] = 45
    strategy.params['rsi']['overbought'] = 55
    
    # Calculate RSI for reference
    import ta
    rsi = ta.momentum.RSIIndicator(data['close'], window=14).rsi()
    
    # Count potential trigger points
    low_rsi_count = (rsi < strategy.params['rsi']['oversold']).sum()
    high_rsi_count = (rsi > strategy.params['rsi']['overbought']).sum()
    
    # Generate signals with relaxed parameters
    signals = strategy.generate_signals(data)
    
    # Print diagnostic information
    print(f"\n----- STRATEGY DIAGNOSTIC: {strategy_name} -----")
    print(f"Data points: {len(data)}")
    print(f"RSI below {strategy.params['rsi']['oversold']}: {low_rsi_count} times ({low_rsi_count/len(data)*100:.1f}%)")
    print(f"RSI above {strategy.params['rsi']['overbought']}: {high_rsi_count} times ({high_rsi_count/len(data)*100:.1f}%)")
    print(f"Signals generated: {(signals != 0).sum()} ({(signals != 0).sum()/len(data)*100:.2f}%)")
    print(f"LONG signals: {(signals == 1).sum()}")
    print(f"SHORT signals: {(signals == -1).sum()}")
    
    # Restore original parameters
    strategy.params = original_params
    
    return {
        'low_rsi_count': low_rsi_count,
        'high_rsi_count': high_rsi_count,
        'long_signals': (signals == 1).sum(),
        'short_signals': (signals == -1).sum()
    }

def run_diagnostic_analysis(data_path: str = None):
    """Run full diagnostic analysis on strategies"""
    # Load data from cache or use provided path
    from utils.data_cache import DataCache
    from datetime import datetime, timedelta
    
    if data_path:
        data = pd.read_pickle(data_path)
    else:
        # Fixed: Use the correct method to get cached data
        cache = DataCache()
        # Try to find a recent cache file
        try:
            # Try to get cached data for BTC/USDT for a 30-day period
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            data = cache.get_cached_data(
                symbol='BTC/USDT',
                timeframe='1h',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if data is None:
                # Try with 15m timeframe if 1h is not available
                data = cache.get_cached_data(
                    symbol='BTC/USDT',
                    timeframe='15m',
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
            if data is not None:
                print(f"Found cached data with {len(data)} entries")
            else:
                print("No cached data found. Please run a backtest first to generate cache.")
                return
                
        except Exception as e:
            print(f"Error retrieving cached data: {str(e)}")
            print("Tip: Run a backtest first to ensure data is cached")
            return
    
    if data is None or data.empty:
        print("No data available for analysis")
        return
        
    # Run analysis on both strategies
    optimize_stats = analyze_conditions(data, 'optimized')
    enhanced_stats = analyze_conditions(data, 'enhanced')
    
    # Generate recommendations
    print("\n----- RECOMMENDATIONS -----")
    
    if optimize_stats['long_signals'] < 5 or enhanced_stats['long_signals'] < 5:
        print("✅ RECOMMENDATION: Relax RSI oversold threshold to 45-50")
        print("✅ RECOMMENDATION: Remove AND conditions in long signal generation")
    
    if optimize_stats['short_signals'] < 5 or enhanced_stats['short_signals'] < 5:
        print("✅ RECOMMENDATION: Relax RSI overbought threshold to 50-55")
        print("✅ RECOMMENDATION: Remove AND conditions in short signal generation")
        
    # Check for extremely low signal counts
    if (optimize_stats['long_signals'] + optimize_stats['short_signals'] < 10):
        print("⚠️ WARNING: Strategy is extremely conservative - signals are rare")
        print("✅ RECOMMENDATION: Consider using combinatorial signal conditions (OR instead of AND)")
        
    print("\nRun this script directly to perform diagnostic analysis on your strategies.")

if __name__ == "__main__":
    run_diagnostic_analysis()
