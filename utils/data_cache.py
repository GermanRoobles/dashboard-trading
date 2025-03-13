import pandas as pd
import os
import pickle
from datetime import datetime, timedelta
import hashlib
import glob
import ccxt  # Add this import
import time  # Add this import

class DataCache:
    """Clase para gestionar el caché de datos OHLCV"""
    
    def __init__(self, cache_dir='/home/panal/Documents/dashboard-trading/data/cache'):
        self.cache_dir = cache_dir
        self.alt_cache_dir = '/home/panal/Documents/bot-machine-learning-main/data/cache'
        self.ccxt_client = None
        self._init_exchange()
        
    def _init_exchange(self):
        """Initialize CCXT exchange client"""
        import ccxt
        self.ccxt_client = ccxt.binance()
        
    def get_cached_data(self, symbol='BTC/USDT', timeframe='1h', 
                       start_date=None, end_date=None):
        """
        Obtiene datos del caché basado en símbolo y timeframe
        
        Args:
            symbol: Par de trading
            timeframe: Timeframe de los datos
            start_date: Fecha inicio (formato YYYY-MM-DD)
            end_date: Fecha fin (formato YYYY-MM-DD)
            
        Returns:
            DataFrame con datos históricos o None si no existen
        """
        # Preparar fechas si se especifican
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
            
        # Try special case for 4h data first
        if timeframe == '4h' and 'low_risk' in os.listdir(self.cache_dir):
            special_file = os.path.join(self.cache_dir, "BTC_USDT_4h_lowrisk_fixed.pkl")
            if os.path.exists(special_file):
                try:
                    data = pd.read_pickle(special_file)
                    print(f"Usando datos especiales para 4h: {special_file}")
                    return self._filter_by_date(data, start_date, end_date)
                except Exception as e:
                    print(f"Error cargando datos especiales 4h: {str(e)}")

        # Buscar en las dos posibles ubicaciones de caché
        for cache_path in [self.cache_dir, self.alt_cache_dir]:
            if not os.path.exists(cache_path):
                continue
                
            # Patrón para archivos que coinciden con el símbolo y timeframe
            clean_symbol = symbol.replace('/', '_')
            patterns = [
                f"{clean_symbol}_{timeframe}_*.pkl",  # Formato hash 
                f"{clean_symbol}_{timeframe}_*_*.pkl",  # Formato fecha
                f"*{clean_symbol}*{timeframe}*.pkl",   # Formato flexible
            ]
            
            # Buscar archivos que coincidan con los patrones
            for pattern in patterns:
                matching_files = glob.glob(os.path.join(cache_path, pattern))
                if matching_files:
                    # Ordenar por fecha de modificación (más reciente primero)
                    matching_files.sort(key=os.path.getmtime, reverse=True)
                    
                    # Intentar cargar el archivo más reciente
                    for file_path in matching_files:
                        try:
                            print(f"Usando datos en caché para {symbol} {timeframe}")
                            data = pd.read_pickle(file_path)
                            return self._filter_by_date(data, start_date, end_date)
                        except Exception as e:
                            print(f"Error al cargar {file_path}: {str(e)}")
                            continue
        
        print(f"No se encontraron datos en caché para {symbol} {timeframe}")
        return None
    
    def _filter_by_date(self, data, start_date=None, end_date=None):
        """Filtrar datos por fecha"""
        if data is None or data.empty:
            return data
            
        # Asegurarse de que el índice es de tipo datetime
        if not isinstance(data.index, pd.DatetimeIndex):
            print("Advertencia: El índice no es DateTimeIndex, no se puede filtrar por fechas")
            return data
            
        # Filtrar por fecha de inicio
        if start_date is not None:
            data = data[data.index >= start_date]
            
        # Filtrar por fecha de fin
        if end_date is not None:
            data = data[data.index <= end_date]
            
        return data
        
    def save_to_cache(self, data: pd.DataFrame, symbol: str, timeframe: str, start_date, end_date):
        """Guardar datos en caché"""
        if data is None or data.empty:
            return
            
        cache_file = self._get_cache_filename(symbol, timeframe, start_date, end_date)
        
        try:
            # Optimizaciones para datasets grandes
            if len(data) > 10000:
                # Para periodos largos, reducir la precisión de los datos
                data = data.copy()
                numeric_cols = data.select_dtypes(include=['float64']).columns
                data[numeric_cols] = data[numeric_cols].astype('float32')
                
            data.to_pickle(cache_file)
            print(f"Datos guardados en caché: {cache_file}")
        except Exception as e:
            print(f"Error guardando caché: {e}")
            
    def _get_cache_filename(self, symbol, timeframe, start_date, end_date):
        """Generar nombre de archivo de caché único"""
        symbol_clean = symbol.replace('/', '_')
        
        # Para periodos muy largos, usar hash en vez de fechas completas
        if (datetime.fromisoformat(end_date) - datetime.fromisoformat(start_date)).days > 365:
            date_hash = hashlib.md5(f"{start_date}_{end_date}".encode()).hexdigest()[:8]
            return os.path.join(self.cache_dir, f"{symbol_clean}_{timeframe}_{date_hash}.pkl")
        else:
            return os.path.join(self.cache_dir, f"{symbol_clean}_{timeframe}_{start_date}_{end_date}.pkl")
    
    def get_cached_data(self, symbol, timeframe, start_date, end_date):
        """Get data from cache or download if not available"""
        try:
            # Fix timeframe format
            timeframe_map = {
                '1m': '1m',
                '3m': '3m', 
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '2h': '2h',
                '4h': '4h',
                '6h': '6h',
                '8h': '8h',
                '12h': '12h',
                '1d': '1d',
                '3d': '3d',
                '1w': '1w',
            }

            # Validate timeframe
            if timeframe not in timeframe_map:
                print(f"Invalid timeframe: {timeframe}")
                return None

            binance_timeframe = timeframe_map[timeframe]
            
            # Try to download data with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    print(f"Downloading fresh data for {symbol} {binance_timeframe}")
                    # Get Binance exchange info first to validate symbol
                    exchange_info = ccxt.binance().load_markets()
                    
                    if symbol not in exchange_info:
                        print(f"Invalid symbol: {symbol}")
                        return None
                    
                    exchange = ccxt.binance({
                        'enableRateLimit': True,
                        'options': {
                            'defaultType': 'future'
                        }
                    })
                    
                    # Convert dates to milliseconds timestamp
                    since = int(pd.Timestamp(start_date).timestamp() * 1000)
                    end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
                    
                    ohlcv = []
                    while since < end_ts:
                        data_chunk = exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=binance_timeframe,
                            since=since,
                            limit=1000
                        )
                        
                        if not data_chunk:
                            break
                            
                        ohlcv.extend(data_chunk)
                        since = data_chunk[-1][0] + 1
                        
                    # Convert to DataFrame
                    df = pd.DataFrame(
                        ohlcv, 
                        columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    )
                    
                    # Set index
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.set_index('timestamp', inplace=True)
                    
                    # Filter by date range
                    df = df[start_date:end_date]
                    
                    if len(df) > 0:
                        print(f"Downloaded {len(df)} new bars")
                        return df
                    else:
                        print("No data available in specified date range")
                        return None
                        
                except Exception as e:
                    print(f"Error downloading data: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retrying
                    continue
                    
            print(f"Failed to download data after {max_retries} attempts")
            return None
            
        except Exception as e:
            print(f"Error in get_cached_data: {str(e)}")
            return None

    def _download_data(self, symbol, timeframe, start_date, end_date):
        """Download data from exchange"""
        try:
            # Convert dates to timestamps
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp(end_date).timestamp() * 1000)
            
            # Download data in chunks
            all_candles = []
            current_ts = start_ts
            
            while current_ts < end_ts:
                candles = self.ccxt_client.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_ts,
                    limit=1000
                )
                
                if not candles:
                    break
                    
                all_candles.extend(candles)
                current_ts = candles[-1][0] + 1
                
            # Convert to DataFrame
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            print(f"Error downloading data: {str(e)}")
            return None
