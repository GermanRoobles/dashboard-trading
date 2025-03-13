import pandas as pd
import numpy as np
import joblib
import ta  # Add missing import

class BacktestSignalGenerator:
    def __init__(self, debug=False):
        self.debug = debug
        self.debug_log = []
        self.ml_confidence_threshold = 0.6  # Minimum confidence for ML signals
        self.ml_enabled = True

    def generate_signals(self, data, config):
        """Generate trading signals"""
        signals = pd.Series(0, index=data.index)
        
        try:
            # Calculate technical indicators
            rsi = ta.momentum.RSIIndicator(data['close']).rsi()
            ema_short = data['close'].ewm(span=config['ema']['short']).mean()
            ema_long = data['close'].ewm(span=config['ema']['long']).mean()
            
            # Generate signals
            long_signals = 0
            short_signals = 0
            last_signal = 0
            
            for i in range(1, len(data)):
                try:
                    current_rsi = float(rsi.iloc[i])
                    prev_rsi = float(rsi.iloc[i-1])
                    
                    # Long signal conditions
                    if (current_rsi < float(config['rsi']['oversold']) and
                        last_signal != 1):
                        signals.iloc[i] = 1
                        long_signals += 1
                        last_signal = 1
                        
                    # Short signal conditions
                    elif (current_rsi > float(config['rsi']['overbought']) and
                          last_signal != -1):
                        signals.iloc[i] = -1
                        short_signals += 1
                        last_signal = -1
                    
                    # Reset signal conditions
                    elif (current_rsi > 45 and current_rsi < 55):
                        last_signal = 0
                        
                except Exception as e:
                    print(f"Error processing signal at bar {i}: {str(e)}")
                    continue
            
            print(f"Generated {long_signals} long signals and {short_signals} short signals")
            return signals  # Return only the signals Series
            
        except Exception as e:
            print(f"Error generating signals: {str(e)}")
            return pd.Series(0, index=data.index)  # Return empty signals on error
    
    def _calculate_indicators(self, data, config):
        """Calculate technical indicators"""
        print("Calculating technical indicators...")
        
        # Get RSI parameters from config with validation
        rsi_params = config.get('rsi', {})
        rsi_oversold = float(rsi_params.get('oversold', 35))  # More conservative default
        rsi_overbought = float(rsi_params.get('overbought', 65))
        
        # Calculate RSI with improved error handling
        close = data['close']
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.ewm(com=13, adjust=False).mean()  # Changed to EWM for better responsiveness
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))
        
        # Get EMA parameters
        ema_params = config.get('ema', {})
        ema_short = int(ema_params.get('short', 9))
        ema_long = int(ema_params.get('long', 21))
        
        # Calculate EMAs with minimum periods
        ema_short_series = close.ewm(span=ema_short, adjust=False, min_periods=1).mean()
        ema_long_series = close.ewm(span=ema_long, adjust=False, min_periods=1).mean()
        
        return rsi, ema_short_series, ema_long_series
    
    def _get_ml_predictions(self, data):
        """Get ML model predictions"""
        try:
            # Load ML models
            scalp_model = joblib.load('models/scalp_model.pkl')
            swing_model = joblib.load('models/swing_model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            # Prepare features
            features = self._prepare_features(data)
            if features is None:
                return None
                
            # Get predictions
            features_scaled = scaler.transform(features)
            scalp_probs = scalp_model.predict_proba(features_scaled)
            swing_probs = swing_model.predict_proba(features_scaled)
            
            # Get predictions with less restrictive thresholds
            long_prob = 0.7 * scalp_probs[:, 2] + 0.3 * swing_probs[:, 2]  # Weight scalp signals higher
            short_prob = 0.7 * scalp_probs[:, 0] + 0.3 * swing_probs[:, 0]
            
            # Create predictions DataFrame aligned with input data
            predictions = pd.DataFrame({
                'long_prob': long_prob,
                'short_prob': short_prob
            }, index=features.index)
            
            # Forward fill any missing values
            predictions = predictions.reindex(data.index).ffill().fillna(0)
            
            return predictions
            
        except Exception as e:
            print(f"Error getting ML predictions: {str(e)}")
            return None

    def _prepare_features(self, data):
        """Prepare features with consistent naming"""
        try:
            features = pd.DataFrame(index=data.index)
            
            # Price features
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['range'] = (data['high'] - data['low']) / data['close']
            
            # Volume features
            if 'volume' in data.columns:
                features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            else:
                features['volume_sma_ratio'] = 1.0  # Default value when volume not available
            
            # RSI 
            features['rsi'] = ta.momentum.RSIIndicator(close=data['close']).rsi()
            
            # MACD (only main line)
            macd = ta.trend.MACD(close=data['close'])
            features['macd'] = macd.macd()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(close=data['close'])
            features['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            # Forward fill NaN values then backward fill any remaining
            features = features.ffill().bfill()
            
            # Log features for debugging
            if self.debug:
                print(f"Generated features: {features.columns.tolist()}")
                
            return features
            
        except Exception as e:
            if self.debug:
                print(f"Error preparing ML features: {str(e)}")
            return None
