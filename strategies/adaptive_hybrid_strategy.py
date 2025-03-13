from strategies.enhanced_strategy import EnhancedStrategy
import numpy as np
import ta
import pandas as pd
from ml.model_trainer import MLModelTrainer
import joblib

class AdaptiveHybridStrategy(EnhancedStrategy):
    """
    A hybrid strategy that adapts parameters based on detected market regimes
    Combines the best elements of low_risk_strategy and adaptive_strategy
    """
    
    def __init__(self):
        """Initialize strategy with default parameters"""
        super().__init__()
        self.name = "adaptive_hybrid"
        self.description = "Adaptive Hybrid Strategy with regime-based parameter adjustment"
        
        # Set default parameters from low_risk strategy (best performer)
        self.params = {
            'rsi': {
                'window': 14,
                'oversold': 30,
                'overbought': 70
            },
            'ema': {
                'short': 9,
                'long': 26
            },
            'holding_time': 3,
            'trend_filter': True,
            'volume_filter': True,
            'market_regime_detection': True
        }
        
        # Add regime-specific parameters
        self.regime_params = {
            'trending_up': {
                'rsi': {'oversold': 35, 'overbought': 75},
                'position_bias': 0.7  # Bias toward longs in uptrends
            },
            'trending_down': {
                'rsi': {'oversold': 25, 'overbought': 65},
                'position_bias': -0.7  # Bias toward shorts in downtrends
            },
            'ranging': {
                'rsi': {'oversold': 30, 'overbought': 70},
                'position_bias': 0  # No bias in ranging markets
            },
            'volatile': {
                'rsi': {'oversold': 20, 'overbought': 80},
                'position_bias': 0  # No bias but wider RSI in volatile markets
            }
        }
        
        # Current detected regime
        self.current_regime = 'ranging'  # Default to ranging

        # Add ML components
        self.ml_trainer = None
        self.scalp_model = None
        self.swing_model = None
        self.scaler = None
        
        # Load ML models if available
        try:
            self.scalp_model = joblib.load('models/scalp_model.pkl')
            self.swing_model = joblib.load('models/swing_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
        except:
            print("Warning: ML models not found, will use traditional signals only")
        
    def detect_market_regime(self, data):
        """
        Detect current market regime based on price action
        Returns: 'trending_up', 'trending_down', 'ranging', or 'volatile'
        """
        # Use at least 30 bars for regime detection
        window = min(30, len(data) - 1)
        
        # Calculate volatility (standard deviation of returns)
        returns = data['close'].pct_change().dropna()
        volatility = returns[-window:].std() * 100  # Convert to percentage
        
        # Calculate trend strength using linear regression slope
        y = data['close'][-window:].values
        x = np.arange(len(y))
        
        # Fix: Get just the slope coefficient correctly from np.polyfit
        coeffs = np.polyfit(x, y, 1)
        slope = coeffs[0]  # First coefficient is the slope
        
        trend_strength = slope / data['close'].mean() * 1000  # Normalized slope
        
        # Calculate range ratio (how much price stays within a range)
        price_range = data['high'][-window:].max() / data['low'][-window:].min() - 1
        
        # Determine regime based on volatility and trend strength
        if volatility > 3.0:
            regime = 'volatile'
        elif abs(trend_strength) > 1.0:
            regime = 'trending_up' if trend_strength > 0 else 'trending_down'
        else:
            regime = 'ranging'
            
        # Store the current regime for reference
        self.current_regime = regime
        return regime
        
    def apply_regime_parameters(self, regime):
        """Apply parameters specific to the detected market regime"""
        if regime in self.regime_params:
            # Apply RSI adjustments
            if 'rsi' in self.regime_params[regime]:
                self.params['rsi']['oversold'] = self.regime_params[regime]['rsi']['oversold']
                self.params['rsi']['overbought'] = self.regime_params[regime]['rsi']['overbought']
            
            # Store position bias for signal generation
            self.position_bias = self.regime_params[regime].get('position_bias', 0)
        
    def generate_signals(self, data):
        """Generate trading signals using ML models and traditional indicators"""
        # Get traditional signals first
        traditional_signals = super().generate_signals(data)
        
        # If ML models aren't loaded, return traditional signals only
        if None in (self.scalp_model, self.swing_model, self.scaler):
            return traditional_signals
            
        try:
            # Prepare features for ML
            features = self._prepare_ml_features(data)
            if features is None:
                return traditional_signals
                
            features_scaled = self.scaler.transform(features)
            
            # Get ML predictions
            scalp_probs = self.scalp_model.predict_proba(features_scaled)
            swing_probs = self.swing_model.predict_proba(features_scaled)
            
            # Combine signals
            final_signals = self._combine_signals(
                traditional_signals,
                scalp_probs,
                swing_probs,
                data
            )
            
            return final_signals
            
        except Exception as e:
            print(f"Error in ML signal generation: {str(e)}")
            return traditional_signals
    
    def _prepare_ml_features(self, data):
        """Prepare features for ML models"""
        try:
            features = pd.DataFrame()
            
            # Price action features
            features['returns'] = data['close'].pct_change()
            features['volatility'] = features['returns'].rolling(20).std()
            features['range'] = (data['high'] - data['low']) / data['close']
            
            # Volume features
            features['volume_sma_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            
            # Technical indicators
            features['rsi'] = ta.momentum.RSIIndicator(data['close']).rsi()
            
            # MACD
            macd = ta.trend.MACD(data['close'])
            features['macd'] = macd.macd()
            features['macd_signal'] = macd.macd_signal()
            
            # Bollinger Bands
            bb = ta.volatility.BollingerBands(data['close'])
            features['bb_position'] = (data['close'] - bb.bollinger_lband()) / (bb.bollinger_hband() - bb.bollinger_lband())
            
            return features.dropna()
            
        except Exception as e:
            print(f"Error preparing ML features: {str(e)}")
            return None
            
    def _combine_signals(self, traditional, scalp_probs, swing_probs, data):
        """Combine traditional and ML signals"""
        signals = pd.Series(0, index=data.index)
        
        for i in range(len(data)):
            try:
                # Get probabilities for each type of signal
                scalp_long_prob = scalp_probs[i][2] if len(scalp_probs[i]) > 2 else 0
                scalp_short_prob = scalp_probs[i][0] if len(scalp_probs[i]) > 0 else 0
                
                swing_long_prob = swing_probs[i][2] if len(swing_probs[i]) > 2 else 0
                swing_short_prob = swing_probs[i][0] if len(swing_probs[i]) > 0 else 0
                
                # Get traditional signal
                trad_signal = traditional.iloc[i]
                
                # Calculate weighted signal based on current regime
                if self.current_regime == 'volatile':
                    # In volatile markets, prefer scalping signals
                    long_prob = 0.7 * scalp_long_prob + 0.3 * swing_long_prob
                    short_prob = 0.7 * scalp_short_prob + 0.3 * swing_short_prob
                elif self.current_regime in ['trending_up', 'trending_down']:
                    # In trending markets, prefer swing signals
                    long_prob = 0.3 * scalp_long_prob + 0.7 * swing_long_prob
                    short_prob = 0.3 * scalp_short_prob + 0.7 * swing_short_prob
                else:
                    # In ranging markets, balance between both
                    long_prob = 0.5 * scalp_long_prob + 0.5 * swing_long_prob
                    short_prob = 0.5 * scalp_short_prob + 0.5 * swing_short_prob
                
                # Generate final signal
                if trad_signal != 0:  # If traditional signal exists
                    if ((trad_signal == 1 and long_prob > 0.6) or 
                        (trad_signal == -1 and short_prob > 0.6)):
                        # Confirm traditional signal with ML
                        signals.iloc[i] = trad_signal
                elif long_prob > 0.8:  # Strong ML long signal
                    signals.iloc[i] = 1
                elif short_prob > 0.8:  # Strong ML short signal
                    signals.iloc[i] = -1
                    
            except Exception as e:
                if self.debug:
                    print(f"Error combining signals at index {i}: {str(e)}")
                continue
                
        return signals
    
    def apply_filters(self, data, signals):
        """Apply additional filters based on the detected regime"""
        # Apply basic filters from parent class
        filtered_signals = super().apply_filters(data, signals)
        
        # Add regime-specific filters
        if self.current_regime == 'volatile':
            # In volatile regimes, be more selective with signals
            for i in range(1, len(filtered_signals)):
                if filtered_signals.iloc[i] != 0:
                    # Check for larger price movements to confirm signals
                    price_change = abs(data['close'].iloc[i] / data['close'].iloc[i-1] - 1)
                    if price_change < 0.005:  # 0.5% minimum move
                        filtered_signals.iloc[i] = 0  # Filter out weak signals
        
        return filtered_signals
