import pandas as pd
import numpy as np
import ta

class MarketRegimeDetector:
    def __init__(self):
        self.lookback_period = 5  # Más corto
        self.vol_threshold = 0.5  # Más sensible
        self.trend_threshold = 0.1  # Más sensible
        
    def detect_regime(self, data):
        """Detecta el régimen actual del mercado"""
        # Calcular indicadores
        volatility = self._calculate_volatility(data)
        trend_strength = self._calculate_trend_strength(data)
        volume_profile = self._calculate_volume_profile(data)
        
        # Clasificar régimen con umbrales más bajos
        if volatility > 0.5:  # Más sensible a volatilidad
            return 'volatile'
        elif trend_strength > 0.1:  # Más sensible a tendencias
            return 'trending_up' if trend_strength > 0 else 'trending_down'
        else:  # Default a ranging
            return 'ranging'
            
    def _calculate_volatility(self, data):
        """Calcula la volatilidad usando ATR normalizado"""
        atr = ta.volatility.AverageTrueRange(
            high=data['high'],
            low=data['low'],
            close=data['close'],
            window=self.lookback_period
        ).average_true_range()
        
        return (atr / data['close']) * 100
        
    def _calculate_trend_strength(self, data):
        """Calcula la fuerza de la tendencia"""
        # Usar regresión lineal sobre precios
        y = data['close'].values[-self.lookback_period:]
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalizar pendiente
        return slope / np.mean(y)
        
    def _calculate_volume_profile(self, data):
        """Analiza el perfil de volumen"""
        avg_volume = data['volume'].rolling(self.lookback_period).mean()
        return data['volume'] / avg_volume
