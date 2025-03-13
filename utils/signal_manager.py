import pandas as pd
import numpy as np
import ta
from utils.market_regime_detector import MarketRegimeDetector  # Add this import

class BacktestSignalManager:
    def __init__(self, config, debug=False):
        self.config = config
        self.debug = debug
        self.signal_log = []
        self.regime_detector = MarketRegimeDetector()
        
    def generate_signals(self, data, rsi, ema_short, ema_long):
        """Generate signals with less restrictive conditions"""
        signals = pd.Series(0, index=data.index)
        last_signal = 0
        
        # Debug counters
        long_signals = 0
        short_signals = 0
        
        for i in range(1, len(data)):
            try:
                current_rsi = float(rsi.iloc[i])
                prev_rsi = float(rsi.iloc[i-1])
                
                # Condiciones más agresivas para entradas largas
                if (current_rsi < 20 or  # RSI muy bajo
                    (current_rsi < 30 and current_rsi > prev_rsi)) and \
                    ema_short.iloc[i] > ema_long.iloc[i] * 0.99 and \
                    last_signal != 1:
                    
                    signals.iloc[i] = 1
                    last_signal = 1
                    long_signals += 1
                    
                # Condiciones más agresivas para entradas cortas    
                elif (current_rsi > 80 or  # RSI muy alto
                      (current_rsi > 70 and current_rsi < prev_rsi)) and \
                    ema_short.iloc[i] < ema_long.iloc[i] * 1.01 and \
                    last_signal != -1:
                    
                    signals.iloc[i] = -1
                    last_signal = -1
                    short_signals += 1
                    
                # Reset más rápido
                elif 40 < current_rsi < 60:
                    last_signal = 0

            except Exception as e:
                print(f"Error at bar {i}: {str(e)}")
                continue
                
        print(f"Generated signals - Long: {long_signals}, Short: {short_signals}")
        return signals
    
    def _log_conditions(self, index, price, rsi, prev_rsi, ema_short, ema_long):
        """Log trading conditions for analysis"""
        if self.debug:
            self.signal_log.append({
                'bar': index,
                'price': price,
                'rsi': rsi,
                'rsi_change': rsi - prev_rsi,
                'ema_short': ema_short,
                'ema_long': ema_long,
                'ema_cross': ema_short > ema_long
            })
            
    def get_signal_summary(self):
        """Get summary of signal generation"""
        if not self.signal_log:
            return "No signals logged"
            
        df = pd.DataFrame(self.signal_log)
        return {
            'total_bars': len(df),
            'rsi_range': f"{df['rsi'].min():.2f} to {df['rsi'].max():.2f}",
            'rsi_changes': f"{df['rsi_change'].min():.2f} to {df['rsi_change'].max():.2f}",
            'ema_crosses': len(df[df['ema_cross'] != df['ema_cross'].shift(1)])
        }
