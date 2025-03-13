"""
Módulo para aplicar monkey patching y reemplazar TA-LIB con nuestra implementación alternativa
"""

def apply_patches():
    """Aplica monkey patching para reemplazar TA-LIB con nuestra implementación"""
    import sys
    
    class MockTaLib:
        """Clase mock para reemplazar talib"""
        
        @staticmethod
        def RSI(*args, **kwargs):
            from utils.talib_alternative import RSI
            return RSI(*args, **kwargs)
            
        @staticmethod
        def MACD(*args, **kwargs):
            from utils.talib_alternative import MACD
            return MACD(*args, **kwargs)
            
        @staticmethod
        def BBANDS(*args, **kwargs):
            from utils.talib_alternative import BBANDS
            return BBANDS(*args, **kwargs)
        
        @staticmethod
        def MA(*args, **kwargs):
            from utils.talib_alternative import MA
            return MA(*args, **kwargs)
            
        @staticmethod
        def ATR(*args, **kwargs):
            from utils.talib_alternative import ATR
            return ATR(*args, **kwargs)
    
    # Crear un módulo ficticio talib
    sys.modules['talib'] = MockTaLib()
    print("TA-LIB patched with alternative implementation")
