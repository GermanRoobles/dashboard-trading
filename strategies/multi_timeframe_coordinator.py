#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from datetime import datetime
from strategies.adaptive_hybrid_strategy import AdaptiveHybridStrategy

class MultiTimeframeCoordinator:
    """
    Coordinates trading signals across multiple timeframes,
    leveraging the strengths of each timeframe based on market conditions
    """
    
    def __init__(self, config_path=None, timeframes=None):
        """Initialize with configuration and timeframes to monitor"""
        # Default timeframes - based on testing results
        self.timeframes = timeframes or ['15m', '1h', '4h']
        
        # Load configuration
        if config_path:
            # Verify the config path exists
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                import json
                self.config = json.load(f)
                print(f"Loaded configuration from: {config_path}")
        else:
            # Use hybrid_strategy as default
            config_path = "/home/panal/Documents/dashboard-trading/configs/hybrid_strategy.json"
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Default configuration file not found: {config_path}")
                
            with open(config_path, 'r') as f:
                import json
                self.config = json.load(f)
                print(f"Loaded default configuration from: {config_path}")
        
        # Initialize strategies for each timeframe
        self.strategies = {}
        for tf in self.timeframes:
            strategy = AdaptiveHybridStrategy()
            # Clone config but adapt to specific timeframe
            tf_config = self.config.copy()
            tf_config['timeframe'] = tf
            
            # Apply timeframe-specific adjustments based on test results
            if tf == '15m':
                # 15m performed best with higher return but more volatility
                # Adjust for more trades but more aggressive filtering
                if 'risk_controls' in tf_config and 'regime_adjustments' in tf_config['risk_controls']:
                    # Adjust position size for the higher volatility
                    tf_config['risk_controls']['regime_adjustments']['volatile']['position_size_multiplier'] = 0.3
                    # Be more selective with entries
                    if 'filters' not in tf_config:
                        tf_config['filters'] = {}
                    tf_config['filters']['minimum_adr'] = 0.9
            
            elif tf == '4h':
                # 4h performed best with high win rate and low drawdown
                # Optimize for quality over quantity
                if 'position_size' not in tf_config:
                    tf_config['position_size'] = {}
                # Can use larger positions due to higher accuracy
                tf_config['position_size']['default'] = 0.03
                
                # Extend holding time to capture larger moves
                tf_config['holding_time'] = 4
            
            self.strategies[tf] = strategy
        
        # Performance metrics for each timeframe
        self.performance = {
            '15m': {'weight': 0.4, 'return': 5.56, 'win_rate': 52.07, 'profit_factor': 1.26},
            '1h': {'weight': 0.3, 'return': 1.67, 'win_rate': 57.89, 'profit_factor': 1.71},
            '4h': {'weight': 0.3, 'return': 0.93, 'win_rate': 72.41, 'profit_factor': 3.02}
        }
        
        # Current regime detection
        self.current_regime = 'ranging'
        
        # Signal history
        self.signals = {tf: pd.DataFrame() for tf in self.timeframes}
        
    def update_data(self, timeframe, data):
        """Update data for a specific timeframe and generate signals"""
        if timeframe not in self.strategies:
            return None
            
        strategy = self.strategies[timeframe]
        
        # Detect regime
        regime = strategy.detect_market_regime(data)
        
        # Generate signals
        signals = strategy.generate_signals(data)
        
        # Store signals
        self.signals[timeframe] = pd.DataFrame({
            'timestamp': data.index,
            'signal': signals,
            'price': data['close'],
            'regime': regime
        })
        
        # Update current regime (use 1h as reference)
        if timeframe == '1h':
            self.current_regime = regime
            
        return signals
            
    def get_consolidated_signal(self):
        """Get consolidated trading signal across timeframes"""
        # If we don't have signals for all timeframes yet, return neutral
        for tf in self.timeframes:
            if self.signals[tf].empty:
                return 0
                
        # Get latest signals
        latest_signals = {}
        for tf, signal_df in self.signals.items():
            if not signal_df.empty:
                latest_signals[tf] = signal_df.iloc[-1]
                
        # Calculate weights based on performance and current regime
        weights = self._calculate_weights()
        
        # Calculate weighted signal
        weighted_sum = 0
        for tf, weight in weights.items():
            if tf in latest_signals:
                weighted_sum += latest_signals[tf]['signal'] * weight
        
        # Determine final signal (-1, 0, 1)
        if weighted_sum > 0.5:
            return 1
        elif weighted_sum < -0.5:
            return -1
        else:
            return 0
    
    def _calculate_weights(self):
        """Calculate timeframe weights based on performance and current regime"""
        weights = {tf: self.performance[tf]['weight'] for tf in self.timeframes}
        
        # Adjust weights based on current market regime
        if self.current_regime == 'volatile':
            # In volatile regimes, prefer 4h (more stable) and 15m (quicker reactions)
            weights['4h'] += 0.1
            weights['15m'] += 0.05
            weights['1h'] -= 0.15
        elif self.current_regime == 'trending_up' or self.current_regime == 'trending_down':
            # In trending regimes, prefer 1h and 4h
            weights['1h'] += 0.1
            weights['4h'] += 0.05
            weights['15m'] -= 0.15
        elif self.current_regime == 'ranging':
            # In ranging regimes, prefer 15m for quick reversals
            weights['15m'] += 0.1
            weights['1h'] += 0.05
            weights['4h'] -= 0.15
            
        # Normalize weights to sum to 1
        weight_sum = sum(weights.values())
        weights = {tf: w/weight_sum for tf, w in weights.items()}
        
        return weights
        
    def get_position_size(self, base_size=0.02):
        """Calculate dynamic position size based on timeframe consensus"""
        # Get signal strength (agreement between timeframes)
        signal_counts = {
            1: 0,   # long signals
            0: 0,   # neutral
            -1: 0   # short signals
        }
        
        for tf, signal_df in self.signals.items():
            if not signal_df.empty:
                latest_signal = signal_df.iloc[-1]['signal']
                signal_counts[latest_signal] = signal_counts.get(latest_signal, 0) + 1
        
        # Calculate position size modifier based on signal agreement
        total_timeframes = len(self.timeframes)
        
        # If all timeframes agree, increase position size
        if signal_counts[1] == total_timeframes or signal_counts[-1] == total_timeframes:
            return base_size * 1.5
        # If majority agree, use normal position size
        elif signal_counts[1] > total_timeframes/2 or signal_counts[-1] > total_timeframes/2:
            return base_size
        # Otherwise, reduce position size
        else:
            return base_size * 0.75
    
    def generate_report(self, output_dir=None):
        """Generate a report of multi-timeframe signals and performance"""
        if output_dir is None:
            output_dir = f"/home/panal/Documents/dashboard-trading/reports/multi_timeframe_{datetime.now().strftime('%Y%m%d')}"
            os.makedirs(output_dir, exist_ok=True)
        
        # Create report dataframe
        report_data = []
        for tf, signal_df in self.signals.items():
            if not signal_df.empty:
                last_row = signal_df.iloc[-1]
                report_data.append({
                    'timeframe': tf,
                    'signal': last_row['signal'],
                    'regime': last_row['regime'],
                    'return': self.performance[tf]['return'],
                    'win_rate': self.performance[tf]['win_rate'],
                    'profit_factor': self.performance[tf]['profit_factor'],
                    'weight': self._calculate_weights()[tf]
                })
        
        # Convert to DataFrame
        report_df = pd.DataFrame(report_data)
        
        # Save to CSV
        report_df.to_csv(os.path.join(output_dir, 'multi_timeframe_report.csv'), index=False)
        
        # Create visualization
        if not report_df.empty:
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Bar colors based on signal
            colors = {1: 'green', 0: 'gray', -1: 'red'}
            bar_colors = [colors[signal] for signal in report_df['signal']]
            
            # Create bars
            bars = ax.bar(report_df['timeframe'], report_df['weight'], color=bar_colors)
            
            # Add signal annotations
            for i, bar in enumerate(bars):
                signal = report_df.iloc[i]['signal']
                signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        signal_text, ha='center', va='bottom')
            
            ax.set_title('Multi-Timeframe Signal Weights')
            ax.set_xlabel('Timeframe')
            ax.set_ylabel('Weight')
            
            plt.savefig(os.path.join(output_dir, 'signal_weights.png'))
            plt.close()
        
        return report_df

def test_coordinator():
    """Test the multi-timeframe coordinator with sample data"""
    from utils.data_cache import DataCache
    from datetime import datetime, timedelta
    
    print("Testing multi-timeframe coordinator...")
    
    # Initialize coordinator
    coordinator = MultiTimeframeCoordinator()
    
    # Get data for each timeframe
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    data = {}
    for tf in coordinator.timeframes:
        data[tf] = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=tf,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
    
    # Update data for each timeframe
    for tf, df in data.items():
        if df is not None:
            print(f"Updating {tf} data...")
            coordinator.update_data(tf, df)
    
    # Get consolidated signal
    signal = coordinator.get_consolidated_signal()
    signal_text = "BUY" if signal == 1 else "SELL" if signal == -1 else "HOLD"
    
    print(f"Consolidated signal: {signal_text}")
    
    # Get dynamic position size
    pos_size = coordinator.get_position_size()
    print(f"Recommended position size: {pos_size:.3f}")
    
    # Generate report
    print("Generating report...")
    coordinator.generate_report()
    
    print("Test complete!")

if __name__ == "__main__":
    test_coordinator()
