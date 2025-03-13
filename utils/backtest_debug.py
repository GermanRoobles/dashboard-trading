import pandas as pd

class BacktestDebugger:
    """Debug helper for backtesting"""
    
    def __init__(self):
        self.signal_conditions = []
        self.debug_data = {}
        self.trade_log = []
        
    def log_signal_conditions(self, index, price, rsi, prev_rsi, 
                            ema_short, ema_long, signal=0):
        """Log conditions that could generate signals"""
        self.signal_conditions.append({
            'index': index,
            'price': price,
            'rsi': rsi,
            'rsi_change': rsi - prev_rsi,
            'ema_short': ema_short,
            'ema_long': ema_long,
            'ema_cross': ema_short > ema_long,
            'signal': signal
        })
    
    def analyze_signal_generation(self):
        """Analyze why signals are or aren't being generated"""
        if not self.signal_conditions:
            return "No signal conditions logged"
            
        df = pd.DataFrame(self.signal_conditions)
        analysis = []
        
        # Check RSI movements
        rsi_changes = df['rsi_change'].abs()
        if rsi_changes.max() < 1:
            analysis.append("WARNING: Very small RSI movements detected")
            
        # Check EMA crosses
        ema_crosses = (df['ema_cross'] != df['ema_cross'].shift(1)).sum()
        if ema_crosses == 0:
            analysis.append("WARNING: No EMA crosses detected")
            
        # Check signal distribution
        signals = df['signal'].value_counts()
        analysis.append(f"\nSignal distribution:")
        analysis.append(f"Long signals: {signals.get(1, 0)}")
        analysis.append(f"Short signals: {signals.get(-1, 0)}")
        analysis.append(f"No signals: {signals.get(0, 0)}")
        
        return "\n".join(analysis)

    def print_debug_summary(self):
        """Print summary of debugging information"""
        print("\n=== BACKTEST DEBUG SUMMARY ===")
        
        # Print signal analysis
        print("\nSignal Generation Analysis:")
        print(self.analyze_signal_generation())
        
        # Print indicator statistics
        if 'rsi' in self.debug_data:
            rsi = self.debug_data['rsi']
            print(f"\nRSI Statistics:")
            print(f"Range: {rsi.min():.2f} to {rsi.max():.2f}")
            print(f"Mean: {rsi.mean():.2f}")
            print(f"Crosses below 30: {(rsi < 30).sum()}")
            print(f"Crosses above 70: {(rsi > 70).sum()}")
        
        # Print trade statistics
        if self.trade_log:
            print(f"\nTrade Log Analysis:")
            print(f"Total trade attempts: {len(self.trade_log)}")
            successful = sum(1 for t in self.trade_log if t.get('executed', False))
            print(f"Successful trades: {successful}")
            print(f"Failed trades: {len(self.trade_log) - successful}")
