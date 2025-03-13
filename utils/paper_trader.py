import pandas as pd
import time
import datetime
import ccxt
from strategies.optimized_strategy import OptimizedStrategy
from utils.data_cache import DataCache

class PaperTrader:
    def __init__(self, strategy_params, timeframe='15m', symbol='BTC/USDT'):
        self.strategy = OptimizedStrategy()
        self.strategy.params.update(strategy_params)
        self.timeframe = timeframe
        self.symbol = symbol
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.trades = []
        self.log_file = f"paper_trading_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        
        # Inicializar archivo de log
        with open(self.log_file, 'w') as f:
            f.write("timestamp,signal,price,reason\n")
    
    def run(self, hours=24):
        """Ejecuta paper trading por X horas"""
        print(f"Iniciando paper trading por {hours} horas...")
        
        end_time = datetime.datetime.now() + datetime.timedelta(hours=hours)
        
        while datetime.datetime.now() < end_time:
            try:
                # Obtener datos recientes
                ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Generar señal para el último punto
                signal = self.strategy.generate_signals(df).iloc[-1]
                current_price = df['close'].iloc[-1]
                current_time = datetime.datetime.now()
                
                # Si hay señal, registrarla
                if signal != 0:
                    reason = "RSI extremo y alineamiento EMAs" if abs(signal) == 1 else "Divergencia detectada"
                    self.log_signal(current_time, signal, current_price, reason)
                    print(f"[{current_time}] Señal: {'COMPRA' if signal > 0 else 'VENTA'} a {current_price}")
                
                # Esperar hasta el próximo período
                wait_time = self._calculate_wait_time()
                time.sleep(wait_time)
                
            except Exception as e:
                print(f"Error en paper trading: {str(e)}")
                time.sleep(60)  # Esperar un minuto y reintentar
    
    def log_signal(self, timestamp, signal, price, reason):
        """Registrar señal en el log"""
        signal_type = "BUY" if signal > 0 else "SELL"
        with open(self.log_file, 'a') as f:
            f.write(f"{timestamp},{signal_type},{price},{reason}\n")
        
        self.trades.append({
            'timestamp': timestamp,
            'signal': signal_type,
            'price': price,
            'reason': reason
        })
    
    def _calculate_wait_time(self):
        """Calcular tiempo de espera hasta próxima vela"""
        if self.timeframe == '15m':
            minutes = 15
        elif self.timeframe == '1h':
            minutes = 60
        else:
            minutes = 5  # Por defecto
            
        now = datetime.datetime.now()
        
        # Calculate minutes until next candle (safer method)
        remaining_minutes = minutes - (now.minute % minutes)
        if remaining_minutes == minutes:
            remaining_minutes = 0
        
        # Calculate the next candle time
        next_candle = now + datetime.timedelta(minutes=remaining_minutes)
        next_candle = next_candle.replace(second=5, microsecond=0)
        
        # Calculate wait time in seconds
        wait_seconds = (next_candle - now).total_seconds()
        return max(1, wait_seconds)  # Mínimo 1 segundo

if __name__ == "__main__":
    # Ejemplo de parámetros conservadores
    params = {
        'rsi': {'oversold': 38, 'overbought': 62},
        'ema': {'short': 10, 'long': 25},
        'holding_time': 4,
        'trend_filter': True,
        'volume_filter': True
    }
    
    trader = PaperTrader(params, timeframe='15m')
    trader.run(hours=24)  # Ejecutar por 24 horas
