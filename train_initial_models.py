#!/usr/bin/env python
from ml.model_trainer import MLModelTrainer
from utils.data_cache import DataCache
from datetime import datetime, timedelta

def train_initial_models(days=180):
    """Train initial ML models with historical data"""
    print("=== TRAINING INITIAL ML MODELS ===")
    
    # Get training data
    cache = DataCache()
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Get data for different timeframes
    timeframes = ['15m', '1h', '4h']
    models = {}
    
    for tf in timeframes:
        print(f"\nTraining models for {tf} timeframe...")
        data = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=tf,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is not None:
            trainer = MLModelTrainer()
            trainer.train(data)
            print(f"Models trained and saved for {tf}")
            
    print("\nInitial training complete!")

if __name__ == "__main__":
    train_initial_models()
