#!/usr/bin/env python
import pandas as pd
import numpy as np
from backtest.run_fixed_backtest import BacktestFixedStrategy
from utils.data_cache import DataCache
import json
from datetime import datetime, timedelta

def validate_strategy_calculations(debug=False):
    """Validate strategy calculations with enhanced debugging"""
    print("=== STRATEGY CALCULATION VALIDATION ===\n")
    
    # 1. Create test dataset with more extreme movements
    print("Creating test dataset...")
    index = pd.date_range(start='2024-01-01', periods=200, freq='15min')
    base_price = 100.0
    prices = []
    
    # Generate prices with more extreme movements
    for i in range(200):
        if i < 50:  # Strong uptrend
            noise = np.random.normal(0, 0.2)
            price = base_price * (1 + 0.003 * i + noise)
        elif i < 100:  # Strong downtrend
            noise = np.random.normal(0, 0.2)
            price = base_price * (1.15 - 0.003 * (i - 50) + noise)
        elif i < 150:  # Ranging with clear reversals
            cycle = np.sin(i/10) * 5
            noise = np.random.normal(0, 0.3)
            price = base_price * (1 + cycle/100 + noise)
        else:  # Volatile
            noise = np.random.normal(0, 0.8)
            price = base_price * (1 + noise)
        prices.append(price)
    
    # Create DataFrame with OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.001))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.001))) for p in prices],
        'close': prices,
        'volume': [1000 * (1 + abs(np.random.normal(0, 0.5))) for _ in prices]
    }, index=index)
    
    # 2. Create test configuration with more sensitive parameters
    test_config = {
        'rsi': {
            'window': 14,
            'oversold': 35,  # More sensitive oversold
            'overbought': 65  # More sensitive overbought
        },
        'ema': {
            'short': 9,
            'long': 21
        },
        'holding_time': 4
    }
    
    # 3. Run backtest with debug output
    print("\nRunning backtest with test data...")
    backtest = BacktestFixedStrategy(config=test_config)
    
    # Enable detailed debug mode
    backtest.debug = debug
    result = backtest.run(data)
    
    # 4. Validate indicators
    print("\nValidating indicators...")
    rsi = backtest.debug_data.get('rsi') if hasattr(backtest, 'debug_data') else None
    ema_short = backtest.debug_data.get('ema_short') if hasattr(backtest, 'debug_data') else None
    ema_long = backtest.debug_data.get('ema_long') if hasattr(backtest, 'debug_data') else None
    
    if rsi is not None:
        print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
        if rsi.min() < 0 or rsi.max() > 100:
            print("⚠️ ERROR: RSI values outside valid range (0-100)")
    else:
        print("⚠️ ERROR: No RSI data available")
        
    if ema_short is not None and ema_long is not None:
        print(f"EMA Short range: {ema_short.min():.2f} to {ema_short.max():.2f}")
        print(f"EMA Long range: {ema_long.min():.2f} to {ema_long.max():.2f}")
    else:
        print("⚠️ ERROR: No EMA data available")
    
    # 5. Validate trade calculations
    print("\nValidating trade calculations...")
    trades = result.get('trades', [])
    print(f"Total trades: {len(trades)}")
    
    if trades:
        for i, trade in enumerate(trades):
            print(f"\nTrade {i+1}:")
            print(f"Entry price: {trade.get('entry_price')}")
            print(f"Exit price: {trade.get('exit_price')}")
            print(f"P&L: {trade.get('pnl'):.2f}%")
            
            # Validate P&L calculation
            expected_pnl = ((trade.get('exit_price') - trade.get('entry_price')) / 
                          trade.get('entry_price') * 100 * trade.get('direction', 1))
            
            if abs(expected_pnl - trade.get('pnl', 0)) > 0.01:
                print(f"⚠️ ERROR: P&L calculation mismatch")
                print(f"Expected: {expected_pnl:.2f}%")
                print(f"Got: {trade.get('pnl'):.2f}%")
    else:
        print("⚠️ WARNING: No trades generated with test data")
    
    # Add signal validation
    if hasattr(backtest, 'debug_data') and 'signals' in backtest.debug_data:
        signals = backtest.debug_data['signals']
        print("\nSignal Generation Statistics:")
        print(f"Total signal points: {len(signals[signals != 0])}")
        print(f"Long signals: {len(signals[signals == 1])}")
        print(f"Short signals: {len(signals[signals == -1])}")
        
        if len(signals[signals != 0]) == 0:
            print("⚠️ ERROR: No signals generated")
    
    # Add detailed debug during indicator calculation
    print("\nDebugging signal generation...")
    if hasattr(backtest, 'debug_data'):
        rsi = backtest.debug_data.get('rsi')
        ema_short = backtest.debug_data.get('ema_short')
        ema_long = backtest.debug_data.get('ema_long')
        
        if rsi is not None and ema_short is not None and ema_long is not None:
            for i in range(1, len(data)):
                current_rsi = rsi.iloc[i]
                prev_rsi = rsi.iloc[i-1]
                ema_cross = ema_short.iloc[i] > ema_long.iloc[i]
                
                # Log potential signal conditions
                if current_rsi < test_config['rsi']['oversold']:
                    print(f"Potential long signal at {i}: RSI={current_rsi:.2f}, EMA Cross={ema_cross}")
                elif current_rsi > test_config['rsi']['overbought']:
                    print(f"Potential short signal at {i}: RSI={current_rsi:.2f}, EMA Cross={not ema_cross}")
    
    # Add detailed condition checking
    if debug and hasattr(backtest, 'debug_logger'):
        print("\nSignal Condition Analysis:")
        for condition in backtest.debug_logger.signal_conditions:
            timestamp = condition['timestamp']
            if 'long_conditions' in condition:
                if all(condition['long_conditions'].values()):
                    print(f"\nPotential long signal at {timestamp}:")
                    for name, value in condition['long_conditions'].items():
                        print(f"  {name}: {value}")
                        
            if 'short_conditions' in condition:
                if all(condition['short_conditions'].values()):
                    print(f"\nPotential short signal at {timestamp}:")
                    for name, value in condition['short_conditions'].items():
                        print(f"  {name}: {value}")

    # Add trade execution analysis
    if debug and hasattr(backtest, 'debug_logger'):
        print("\nTrade Execution Analysis:")
        for trade in backtest.debug_logger.trades:
            print(f"\nTrade at {trade['timestamp']}:")
            print(f"Type: {trade['type']}")
            print(f"Direction: {trade['direction']}")
            print(f"Price: {trade['price']:.2f}")
            if 'conditions' in trade:
                print("Entry conditions:")
                for name, value in trade['conditions'].items():
                    print(f"  {name}: {value}")

    # 6. Validate return calculation
    print("\nValidating return calculation...")
    if trades:
        expected_return = (1 + sum(t.get('pnl', 0)/100 for t in trades))
        print(f"Expected return: {(expected_return - 1) * 100:.2f}%")
        print(f"Reported return: {result.get('return_total', 0):.2f}%")
        
        if abs(expected_return * 100 - result.get('return_total', 0)) > 0.01:
            print("⚠️ ERROR: Return calculation mismatch")
    
    # 7. Test with real data
    print("\n=== TESTING WITH REAL DATA ===")
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    real_data = cache.get_cached_data(
        symbol='BTC/USDT',
        timeframe='15m',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if real_data is not None:
        print(f"\nRunning backtest with {len(real_data)} bars of real data...")
        real_result = backtest.run(real_data)
        
        print(f"Trades generated: {real_result.get('total_trades', 0)}")
        print(f"Return: {real_result.get('return_total', 0):.2f}%")
        print(f"Win rate: {real_result.get('win_rate', 0):.2f}%")
        
        real_trades = real_result.get('trades', [])
        if real_trades:
            print("\nSample of real trades:")
            for trade in real_trades[:3]:
                print(f"P&L: {trade.get('pnl', 0):.2f}%, Bars held: {trade.get('bars_held', 0)}")
    
    return result

def main():
    """Run validation with command line arguments"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate backtest calculations')
    parser.add_argument('--config', type=str, help='Configuration to test')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    if args.config:
        with open(f"/home/panal/Documents/dashboard-trading/configs/{args.config}.json", 'r') as f:
            config = json.load(f)
    else:
        config = None
    
    # Pass debug flag to validation function
    result = validate_strategy_calculations(debug=args.debug)
    
    # Print summary with more detail in debug mode
    print("\n=== VALIDATION SUMMARY ===")
    trades = result.get('total_trades', 0)
    returns = result.get('return_total', 0)
    win_rate = result.get('win_rate', 0)
    
    if args.debug:
        print(f"Total trades: {trades}")
        print(f"Return: {returns:.2f}%")
        print(f"Win rate: {win_rate:.2f}%")
    
    print(f"Strategy functioning: {'✓' if trades > 0 else '✗'}")
    print(f"Signal generation: {'✓' if trades > 0 else '✗'}")
    print(f"P&L calculation: {'✓' if abs(returns) > 0 else '✗'}")
    print(f"Win rate calculation: {'✓' if win_rate > 0 else '✗'}")

if __name__ == "__main__":
    main()
