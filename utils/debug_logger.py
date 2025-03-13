import pandas as pd
from datetime import datetime

class DebugLogger:
    def __init__(self):
        self.logs = []
        self.signal_conditions = []
        self.trades = []
        
    def log_bar(self, index, data):
        """Log data for each bar"""
        self.logs.append({
            'timestamp': index,
            'price': data.get('price'),
            'rsi': data.get('rsi'),
            'rsi_prev': data.get('rsi_prev'),
            'ema_short': data.get('ema_short'),
            'ema_long': data.get('ema_long'),
            'signal': data.get('signal', 0)
        })
        
    def log_signal_condition(self, index, condition_data):
        """Log signal generation conditions"""
        self.signal_conditions.append({
            'timestamp': index,
            **condition_data
        })
        
    def log_trade(self, trade_data):
        """Log trade execution details"""
        self.trades.append({
            'timestamp': datetime.now(),
            **trade_data
        })
        
    def get_summary(self):
        """Get debug summary"""
        df_logs = pd.DataFrame(self.logs)
        df_signals = pd.DataFrame(self.signal_conditions)
        
        summary = {
            'total_bars': len(self.logs),
            'signals_generated': len(df_logs[df_logs['signal'] != 0]),
            'long_signals': len(df_logs[df_logs['signal'] == 1]),
            'short_signals': len(df_logs[df_logs['signal'] == -1]),
            'conditions_checked': len(self.signal_conditions),
            'trades_attempted': len(self.trades)
        }
        
        if not df_logs.empty:
            summary.update({
                'rsi_range': f"{df_logs['rsi'].min():.2f} to {df_logs['rsi'].max():.2f}",
                'ema_short_range': f"{df_logs['ema_short'].min():.2f} to {df_logs['ema_short'].max():.2f}",
                'ema_long_range': f"{df_logs['ema_long'].min():.2f} to {df_logs['ema_long'].max():.2f}"
            })
            
        return summary
