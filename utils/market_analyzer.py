import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
from enum import Enum
import ta

class MarketRegime(Enum):
    TRENDING_UP = 'trending_up'
    TRENDING_DOWN = 'trending_down'
    RANGING = 'ranging'
    VOLATILE = 'volatile'
    UNKNOWN = 'unknown'

class MarketAnalyzer:
    """Analizador de regímenes de mercado y condiciones para optimización de estrategias"""
    
    @staticmethod
    def detect_market_regime(data: pd.DataFrame, window: int = 20) -> MarketRegime:
        """
        Detecta el régimen actual del mercado analizando precios y volatilidad
        
        Args:
            data: DataFrame con datos OHLCV
            window: Ventana para el análisis (por defecto 20 periodos)
            
        Returns:
            MarketRegime: Régimen de mercado detectado
        """
        # Reduce window size requirements
        min_data_points = max(10, min(window, len(data) // 3))
        
        if len(data) < min_data_points:
            return MarketRegime.UNKNOWN
            
        # Make copy to avoid modifying original data
        df = data.copy().tail(min_data_points * 2)  # Use much fewer data points
        
        # Check for required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            # Attempt a simple solution if columns are missing
            if 'close' in df.columns:
                # Use close price for all price columns if they're missing
                for col in ['open', 'high', 'low']:
                    if col not in df.columns:
                        df[col] = df['close']
                # Use a default volume if missing
                if 'volume' not in df.columns:
                    df['volume'] = 1.0
            else:
                return MarketRegime.UNKNOWN
        
        try:
            # Calculate simple metrics that don't require sophisticated analysis
            # 1. Calculate price trend - rate of change over window
            price_change = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
            
            # 2. Calculate volatility - simple std dev / mean
            volatility = df['close'].pct_change().std()
            
            # 3. Calculate range vs trend - using High-Low range
            avg_range = (df['high'] - df['low']).mean() / df['close'].mean()
            
            # Simplify regime detection logic
            if volatility > 0.02:  # High volatility threshold
                return MarketRegime.VOLATILE
            elif abs(price_change) > 0.03:  # Significant price change (3%+)
                return MarketRegime.TRENDING_UP if price_change > 0 else MarketRegime.TRENDING_DOWN
            else:
                return MarketRegime.RANGING
                
        except Exception as e:
            # Silently fallback to UNKNOWN without printing errors
            return MarketRegime.UNKNOWN
    
    def detect_market_regime(self, data, window=30):
        """Enhanced market regime detection"""
        if len(data) < window:
            return 'unknown'
            
        # Calculate key metrics
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)  # Annualized volatility
        
        # Calculate trend strength
        ema_short = ta.trend.ema_indicator(data['close'], window=9)
        ema_long = ta.trend.ema_indicator(data['close'], window=21)
        trend_strength = (ema_short - ema_long) / ema_long * 100
        
        # Determine regime
        if volatility > 0.8:  # High volatility threshold
            return 'volatile'
        elif abs(trend_strength.iloc[-1]) > 1.0:
            return 'trending_up' if trend_strength.iloc[-1] > 0 else 'trending_down'
        else:
            return 'ranging'
    
    @staticmethod
    def optimize_parameters(regime: MarketRegime) -> Dict:
        """
        Devuelve parámetros optimizados según el régimen de mercado detectado
        
        Args:
            regime: Régimen de mercado detectado
            
        Returns:
            Dict: Parámetros optimizados para ese régimen
        """
        # Parámetros por defecto (régimen desconocido)
        default_params = {
            'rsi': {'oversold': 40, 'overbought': 60},
            'ema': {'short': 9, 'long': 21},
            'holding_time': 4,
            'atr_multiplier': 1.5
        }
        
        # Parámetros por régimen
        params_map = {
            MarketRegime.TRENDING_UP: {
                'rsi': {'oversold': 40, 'overbought': 70},  # RSI más extremo para tendencia alcista
                'ema': {'short': 8, 'long': 21},  # EMAs rápidas para capturar tendencia
                'holding_time': 6,  # Mantener más tiempo en tendencia
                'atr_multiplier': 2.0,  # Stops más amplios
                'position_sizing': {'min': 0.04, 'max': 0.08}
            },
            MarketRegime.TRENDING_DOWN: {
                'rsi': {'oversold': 30, 'overbought': 60},  # RSI más extremo para tendencia bajista
                'ema': {'short': 8, 'long': 21},  # EMAs rápidas
                'holding_time': 5,  # Mantener tiempo medio
                'atr_multiplier': 1.8,
                'position_sizing': {'min': 0.03, 'max': 0.06}
            },
            MarketRegime.RANGING: {
                'rsi': {'oversold': 35, 'overbought': 65},  # RSI para capturar extremos en rango
                'ema': {'short': 9, 'long': 25},  # EMAs estándares
                'holding_time': 3,  # Salir más rápido en mercados de rango
                'atr_multiplier': 1.3,  # Stops más ajustados
                'position_sizing': {'min': 0.03, 'max': 0.05}
            },
            MarketRegime.VOLATILE: {
                'rsi': {'oversold': 30, 'overbought': 70},  # RSI extremo para evitar falsas señales
                'ema': {'short': 12, 'long': 30},  # EMAs más lentas para filtrar ruido
                'holding_time': 2,  # Salir rápido en alta volatilidad
                'atr_multiplier': 2.5,  # Stops más amplios para volatilidad
                'position_sizing': {'min': 0.02, 'max': 0.04}  # Posiciones más pequeñas
            }
        }
        
        # Obtener parámetros específicos para el régimen o usar valores predeterminados
        return params_map.get(regime, default_params)
    
    @staticmethod
    def get_market_statistics(data: pd.DataFrame) -> Dict:
        """
        Calcula estadísticas clave del mercado para informes y visualización
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Dict: Estadísticas del mercado
        """
        if len(data) < 20:
            return {}
            
        # Calcular retornos
        returns = data['close'].pct_change().dropna()
        
        # Estadísticas básicas
        stats = {
            'volatility_daily': returns.std() * np.sqrt(252),  # Anualizada
            'returns_mean': returns.mean() * 100,              # En porcentaje 
            'returns_median': returns.median() * 100,          # En porcentaje
            'returns_skew': returns.skew(),                    # Asimetría
            'returns_kurtosis': returns.kurtosis(),            # Curtosis
            'returns_positive_pct': (returns > 0).mean() * 100 # Porcentaje de retornos positivos
        }
        
        # Calcular correlación con el tiempo para medir la tendencia
        time_idx = np.arange(len(data['close']))
        stats['trend_correlation'] = np.corrcoef(time_idx, data['close'].values)[0, 1]
        
        return stats
