#!/usr/bin/env python
import pandas as pd

class SignalValidator:
    def __init__(self):
        self.min_trades = 20
        self.min_win_rate = 0.5
        self.min_profit_factor = 1.0
        
    def validate_signals(self, signals, trades):
        """Validate trading signals with improved error handling"""
        try:
            if not trades or len(trades) < self.min_trades:
                return False
                
            # Calculate win rate safely
            winning_trades = sum(1 for t in trades if isinstance(t, dict) and t.get('pnl', 0) > 0)
            win_rate = winning_trades / len(trades)
            
            if win_rate < self.min_win_rate:
                return False
                
            # Calculate profit factor safely
            total_profit = sum(t.get('pnl', 0) for t in trades if isinstance(t, dict) and t.get('pnl', 0) > 0)
            total_loss = abs(sum(t.get('pnl', 0) for t in trades if isinstance(t, dict) and t.get('pnl', 0) < 0))
            
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            if profit_factor < self.min_profit_factor:
                return False
                
            return True
            
        except Exception as e:
            print(f"Error validating signals: {str(e)}")
            return False
