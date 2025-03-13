import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import joblib

class MLModelTrainer:
    def __init__(self):
        self.scalp_model = RandomForestClassifier(n_estimators=100)
        self.swing_model = RandomForestClassifier(n_estimators=100)
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """Prepare features with consistent naming"""
        try:
            features = pd.DataFrame()
            
            # Price features
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['range'] = (data['high'] - data['low']) / data['close']
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # RSI
            rsi = self._calculate_rsi(data['close'])
            features['rsi'] = rsi
            
            # MACD
            macd = self._calculate_macd(data['close'])
            features['macd'] = macd
            # Don't include macd_signal as it's causing the mismatch
            
            # Bollinger Bands
            bb_pos = self._calculate_bb_position(data['close'])
            features['bb_position'] = bb_pos
            
            # Drop any NaN values
            features = features.dropna()
            
            # Log feature columns for debugging
            print(f"Generated features: {features.columns.tolist()}")
            
            return features
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            return None

    def prepare_labels(self, data, threshold=0.001, swing_period=8):  # Ajustados los parámetros
        # Scalping labels (short-term price movements)
        future_returns = data['close'].pct_change(2).shift(-2)
        scalp_labels = pd.Series(0, index=data.index)
        scalp_labels[future_returns > threshold] = 1
        scalp_labels[future_returns < -threshold] = -1
        
        # Swing labels (longer-term trends)
        swing_returns = data['close'].pct_change(swing_period).shift(-swing_period)
        swing_labels = pd.Series(0, index=data.index)
        swing_labels[swing_returns > threshold*2] = 1  # Reducido el multiplicador
        swing_labels[swing_returns < -threshold*2] = -1
        
        return scalp_labels, swing_labels

    def train(self, data):
        features = self.prepare_features(data)
        scalp_labels, swing_labels = self.prepare_labels(data)
        
        # Remove NaN values
        valid_idx = features.dropna().index
        features = features.loc[valid_idx]
        scalp_labels = scalp_labels.loc[valid_idx]
        swing_labels = swing_labels.loc[valid_idx]
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Train models
        self.scalp_model.fit(features_scaled, scalp_labels)
        self.swing_model.fit(features_scaled, swing_labels)
        
        # Save models
        joblib.dump(self.scalp_model, 'models/scalp_model.pkl')
        joblib.dump(self.swing_model, 'models/swing_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
        
    def _calculate_macd(self, prices):
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        return macd
        
    def _calculate_bb_position(self, prices, period=20):
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + std * 2
        lower = sma - std * 2
        return (prices - lower) / (upper - lower)
