"""
Módulo para proporcionar funciones compatibles con TA-LIB usando la biblioteca 'ta'
"""

import numpy as np
import pandas as pd
from ta import momentum, trend, volatility

def RSI(close_prices, timeperiod=14):
    """Implementación compatible con TA-LIB para el indicador RSI"""
    rsi = momentum.RSIIndicator(close=close_prices, window=timeperiod, fillna=True)
    return rsi.rsi()

def MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9):
    """Implementación compatible con TA-LIB para el indicador MACD"""
    macd_indicator = trend.MACD(
        close=close_prices, 
        window_fast=fastperiod, 
        window_slow=slowperiod, 
        window_sign=signalperiod,
        fillna=True
    )
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    macd_histogram = macd_indicator.macd_diff()
    
    return macd_line, signal_line, macd_histogram

def BBANDS(close_prices, timeperiod=20, nbdevup=2, nbdevdn=2):
    """Implementación compatible con TA-LIB para Bollinger Bands"""
    bollinger = volatility.BollingerBands(
        close=close_prices,
        window=timeperiod,
        window_dev=nbdevup,
        fillna=True
    )
    upper_band = bollinger.bollinger_hband()
    middle_band = bollinger.bollinger_mavg()
    lower_band = bollinger.bollinger_lband()
    
    return upper_band, middle_band, lower_band

def MA(close_prices, timeperiod=20, matype=0):
    """Implementación compatible con TA-LIB para Moving Average"""
    if matype == 0 or matype == 'sma':
        # Simple Moving Average
        ma = close_prices.rolling(window=timeperiod).mean()
    else:
        # Exponential Moving Average (por defecto para otros tipos)
        ma = close_prices.ewm(span=timeperiod, adjust=False).mean()
        
    return ma

def ATR(high, low, close, timeperiod=14):
    """Implementación compatible con TA-LIB para Average True Range"""
    atr_indicator = volatility.AverageTrueRange(
        high=high,
        low=low,
        close=close,
        window=timeperiod,
        fillna=True
    )
    return atr_indicator.average_true_range()
