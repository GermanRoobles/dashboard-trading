#!/usr/bin/env python
import json
import os

def create_alternative_config():
    """Create an alternative configuration with different parameters"""
    base_path = "/home/panal/Documents/dashboard-trading/configs/hybrid_strategy.json"
    
    # Load base config
    with open(base_path, 'r') as f:
        config = json.load(f)
    
    # Create alternative with different parameters
    alt_config = config.copy()
    
    # Adjust RSI parameters - wider range
    alt_config['rsi'] = {
        'window': 14,
        'oversold': 35,  # Less sensitive to oversold
        'overbought': 75  # Less sensitive to overbought
    }
    
    # Adjust EMA parameters
    alt_config['ema'] = {
        'short': 8,  # Slightly faster
        'long': 21   # Slightly faster
    }
    
    # Adjust holding time based on optimization result
    alt_config['holding_time'] = 2
    
    # Add new parameters for market regime detection
    alt_config['market_regime_weights'] = {
        'trending_up': 1.2,
        'trending_down': 0.8,
        'ranging': 1.0,
        'volatile': 0.6
    }
    
    # Save new configuration
    output_path = "/home/panal/Documents/dashboard-trading/configs/hybrid_strategy_alternative.json"
    with open(output_path, 'w') as f:
        json.dump(alt_config, f, indent=2)
    
    print(f"Alternative configuration created: {output_path}")

if __name__ == "__main__":
    create_alternative_config()
