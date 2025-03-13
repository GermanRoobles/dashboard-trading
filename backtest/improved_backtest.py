import pandas as pd
import numpy as np
import ta
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from analytics.risk_manager import RiskManager
import ccxt
import os
from strategies.optimized_strategy import OptimizedStrategy

class ImprovedBacktest:
    """Clase para ejecución de backtest mejorado con características avanzadas"""

    def __init__(self, params: Dict):
        self.params = params
        self.strategy = OptimizedStrategy()
        
        # Configurar estrategia
        self.strategy.params = self._prepare_strategy_params()
        
        # Variables para almacenar resultados
        self.trades = []
        self.equity_curve = [1.0]
        self.current_balance = float(params.get('initial_balance', 100000))
        
        # Crear risk manager para gestionar posiciones y apalancamiento
        from analytics.risk_manager import RiskManager
        self.risk_manager = RiskManager(params)
        
        # Variables para gestión de capital y apalancamiento
        self.position_sizes = []  # Almacena tamaños de posición
        self.leverages = []       # Almacena apalancamientos
        
    def _prepare_strategy_params(self) -> Dict:
        """Preparar los parámetros de configuración de estrategia"""
        # Valores por defecto
        params = {
            'rsi': {
                'window': 14,
                'oversold': 40,
                'overbought': 60
            },
            'macd': {
                'window_fast': 12,
                'window_slow': 26,
                'window_sign': 9
            },
            'volume': {
                'ma_period': 10,
                'min_threshold': 0.7
            },
            'atr': {
                'window': 14,
                'multiplier': 1.5
            },
            'holding_time': 6,
            'trend_filter': True
        }
        
        # Actualizar con cualquier parámetro proporcionado
        if 'rsi' in self.params:
            params['rsi'].update(self.params['rsi'])
        if 'macd' in self.params:
            params['macd'].update(self.params['macd'])
        if 'volume' in self.params:
            params['volume'].update(self.params['volume'])
        if 'atr' in self.params:
            params['atr'].update(self.params['atr'])
        if 'holding_time' in self.params:
            params['holding_time'] = self.params['holding_time']
        if 'trend_filter' in self.params:
            params['trend_filter'] = self.params['trend_filter']
        
        return params

    def get_historical_data(self, symbol: str = 'BTC/USDT', timeframe: str = '15m', days: int = 30) -> pd.DataFrame:
        """Obtener datos históricos"""
        # Los valores posibles de timeframe son: 1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M
        
        # Si el archivo existe, usarlo
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/cache')
        os.makedirs(data_dir, exist_ok=True)
        symbol_format = symbol.replace('/', '_')
        file_path = os.path.join(data_dir, f"{symbol_format}_{timeframe}_{days}d.csv")
        
        if os.path.exists(file_path) and (time.time() - os.path.getmtime(file_path)) < 3600 * 12:  # 12 horas
            try:
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                print(f"Usando datos en caché de {file_path}")
                return data
            except Exception as e:
                print(f"Error al leer datos en caché: {e}")
        
        print(f"Obteniendo datos desde {(datetime.now() - timedelta(days=days)).date()} hasta {datetime.now().date()}")
        
        # Verificar si necesitamos obtener en bloques debido a límites
        is_small_tf = timeframe in ['1m', '3m', '5m', '15m', '30m']
        is_long_period = days > 30
        
        if is_small_tf and is_long_period:
            return self._get_data_in_chunks(symbol, timeframe, days)
        else:
            return self._get_single_data_block(symbol, timeframe, days)

    def _get_single_data_block(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Obtener datos históricos en un solo bloque"""
        try:
            exchange = ccxt.binance({'enableRateLimit': True})
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            all_candles = []
            
            while True:
                candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
                if not candles:
                    break
                    
                all_candles.extend(candles)
                if len(candles) < 1000:
                    break
                    
                since = candles[-1][0] + 1  # Continuar desde el siguiente timestamp
            
            if not all_candles:
                print("No se obtuvieron datos")
                return pd.DataFrame()
                
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Guardar en caché para futuros usos
            symbol_format = symbol.replace('/', '_')
            file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  'data/cache', f"{symbol_format}_{timeframe}_{days}d.csv")
            df.to_csv(file_path)
            
            return df
            
        except Exception as e:
            print(f"Error al obtener datos: {e}")
            return pd.DataFrame()

    def _get_data_in_chunks(self, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
        """Obtener datos en fragmentos debido a límites de la API"""
        print(f"Solicitando periodo largo ({days} días) o timeframe pequeño ({timeframe}). Procesando en partes...")
        
        # Dividir en fragmentos más pequeños
        chunk_days = 5 if timeframe in ['1m', '3m', '5m'] else (7 if timeframe == '15m' else 14)
        n_chunks = (days + chunk_days - 1) // chunk_days  # Redondear hacia arriba
        
        print(f"Descargando datos en {n_chunks} fragmentos de {chunk_days} días cada uno")
        
        all_data = []
        end_date = datetime.now()
        
        for i in range(n_chunks):
            start_date = end_date - timedelta(days=chunk_days)
            if i == n_chunks - 1:  # último fragmento
                start_date = end_date - timedelta(days=days % chunk_days or chunk_days)
                
            print(f"Fragmento {i+1}/{n_chunks}: {start_date.strftime('%Y-%m-%d %H:%M')} a {end_date.strftime('%Y-%m-%d %H:%M')}")
            
            exchange = ccxt.binance({'enableRateLimit': True})
            since = int(start_date.timestamp() * 1000)
            until = int(end_date.timestamp() * 1000)
            
            # Obtener velas para este período
            try:
                candles = []
                current_since = since
                
                while current_since < until:
                    chunk = exchange.fetch_ohlcv(symbol, timeframe, since=current_since, limit=1000)
                    if not chunk:
                        break
                        
                    candles.extend(chunk)
                    
                    if len(chunk) < 1000:
                        break
                        
                    current_since = chunk[-1][0] + 1
                
                if candles:
                    df_chunk = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                    df_chunk['timestamp'] = pd.to_datetime(df_chunk['timestamp'], unit='ms')
                    df_chunk.set_index('timestamp', inplace=True)
                    
                    # Filtrar para asegurar que no superamos end_date
                    df_chunk = df_chunk[df_chunk.index <= pd.to_datetime(until, unit='ms')]
                    
                    all_data.append(df_chunk)
                    print(f"  ✓ Obtenidas {len(df_chunk)} velas de {start_date.date()} a {end_date.date()}")
                    
            except Exception as e:
                print(f"Error descargando fragmento {i+1}: {e}")
            
            # Actualizar end_date para el siguiente fragmento
            end_date = start_date
            time.sleep(1)  # Evitar límites de API
        
        if not all_data:
            print("No se obtuvieron datos en ningún fragmento")
            return pd.DataFrame()
            
        # Concatenar todos los fragmentos
        final_data = pd.concat(all_data)
        
        # Ordenar y eliminar duplicados
        final_data = final_data.sort_index()
        final_data = final_data[~final_data.index.duplicated(keep='first')]
        
        # Guardar en caché
        symbol_format = symbol.replace('/', '_')
        file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              'data/cache', f"{symbol_format}_{timeframe}_{days}d.csv")
        final_data.to_csv(file_path)
        
        return final_data

    def run(self, data):
        """Ejecutar backtest con funcionalidades avanzadas y aplicar tamaño de posición y apalancamiento"""
        if data.empty:
            print("ERROR: No hay datos para ejecutar el backtest")
            return {
                'return_total': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trades_per_month': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'avg_leverage': 0,
                'avg_position_size': 0
            }
        
        # Verificar si estamos operando en un dataframe muy grande
        is_large_dataset = len(data) > 30000
        if is_large_dataset:
            print(f"Dataset muy grande ({len(data)} velas). Activando optimizaciones de memoria.")
            
        # Generar señales
        signals = self.strategy.generate_signals(data)
        
        # Para análisis
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        signal_ratio = ((signals != 0).sum() / len(signals)) * 100
        
        print("\nOptimized Strategy Analysis:")
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        print(f"Signal ratio: {signal_ratio:.2f}%\n")
        
        # Ignorar señales en los primeros días (indicadores incompletos)
        start_idx = 20
        
        # Reset de variables de backtest
        self.trades = []
        self.equity_curve = [1.0]
        self.current_balance = float(self.params.get('initial_balance', 100000))
        self.position_sizes = []
        self.leverages = []
        
        # Bucle principal optimizado con gestión de capital
        daily_trades = {}
        active_trade = None
        trades_executed = 0
        max_daily_trades = 3
        rejected_daily = 0
        
        # Para cálculo de volatilidad de mercado
        price_changes = []
        
        # Bucle principal
        for i in range(start_idx, len(data)):
            current_date = data.index[i].date()
            current_price = data['close'].iloc[i]
            
            # Calcular volatilidad del mercado (últimas 20 velas)
            if i > 20:
                window = data['close'].iloc[i-20:i]
                volatility = window.pct_change().std()
                price_changes.append(volatility)
            else:
                volatility = 0.01  # Valor por defecto si no tenemos suficientes datos
            
            # Actualizar daily_trades
            if current_date not in daily_trades:
                daily_trades = {current_date: 0}
                
            # Si hay un trade activo, verificar si debe cerrarse
            if active_trade:
                # Calcular cuánto tiempo ha estado abierto
                entry_idx = active_trade['entry_idx']
                bars_held = i - entry_idx
                
                # Cerrar si se alcanzó el holding_time
                if bars_held >= self.strategy.params['holding_time']:
                    # Cerrar trade
                    exit_price = current_price
                    
                    # Calcular PnL usando apalancamiento y tamaño de posición
                    entry_price = active_trade['entry_price']
                    position_size = active_trade['position_size']
                    leverage = active_trade['leverage']
                    
                    # Cálculo del PnL en puntos base
                    price_change_pct = ((exit_price / entry_price) - 1) * active_trade['direction']
                    
                    # PnL con apalancamiento como porcentaje del capital total
                    pnl_pct = price_change_pct * leverage * position_size
                    
                    # PnL en valor absoluto del capital
                    pnl_amount = self.current_balance * pnl_pct
                    
                    # Actualizar balance
                    self.current_balance += pnl_amount
                    
                    # Actualizar equity curve (como valor absoluto)
                    equity_value = self.current_balance / float(self.params.get('initial_balance', 100000))
                    self.equity_curve.append(equity_value)
                    
                    # Registrar trade completo
                    trade = {
                        'entry_time': data.index[entry_idx],
                        'exit_time': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': active_trade['direction'],
                        'position_size': position_size,
                        'leverage': leverage,
                        'pnl': pnl_pct * 100,  # Convertir a porcentaje para compatibilidad
                        'pnl_amount': pnl_amount,
                        'bars_held': bars_held
                    }
                    
                    self.trades.append(trade)
                    
                    # Reset trade activo
                    active_trade = None
            
            # Si no hay trade activo y hay una señal, abrir nuevo trade
            if active_trade is None and signals.iloc[i] != 0:
                # Verificar límites diarios
                if daily_trades.get(current_date, 0) >= max_daily_trades:
                    rejected_daily += 1
                    continue
                
                # Calcular tamaño de posición (3-7% del capital)
                current_drawdown = 0
                if len(self.equity_curve) > 1:
                    peak = max(self.equity_curve)
                    current_drawdown = (peak - self.equity_curve[-1]) / peak
                
                position_size = self.risk_manager.calculate_position_size(
                    self.current_balance, 
                    current_drawdown
                )
                
                # Calcular apalancamiento (10-30x) según volatilidad y riesgo
                market_volatility = np.std(price_changes[-20:]) * 100 if len(price_changes) >= 20 else 0.5
                trade_risk = 0.5  # Valor por defecto, podría calcularse mejor
                
                leverage = self.risk_manager.calculate_leverage(
                    self.current_balance,
                    market_volatility,
                    trade_risk
                )
                
                # Guardar valores para métricas
                self.position_sizes.append(position_size)
                self.leverages.append(leverage)
                
                # Abrir nuevo trade
                active_trade = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'direction': signals.iloc[i],  # 1 para long, -1 para short
                    'position_size': position_size,
                    'leverage': leverage
                }
                
                # Actualizar contador diario
                daily_trades[current_date] = daily_trades.get(current_date, 0) + 1
                trades_executed += 1
        
        # Cerrar trade activo al final si existe
        if active_trade:
            exit_price = data['close'].iloc[-1]
            entry_price = active_trade['entry_price']
            position_size = active_trade['position_size']
            leverage = active_trade['leverage']
            
            price_change_pct = ((exit_price / entry_price) - 1) * active_trade['direction']
            pnl_pct = price_change_pct * leverage * position_size
            pnl_amount = self.current_balance * pnl_pct
            
            self.current_balance += pnl_amount
            equity_value = self.current_balance / float(self.params.get('initial_balance', 100000))
            self.equity_curve.append(equity_value)
            
            trade = {
                'entry_time': data.index[active_trade['entry_idx']],
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': active_trade['direction'],
                'position_size': position_size,
                'leverage': leverage,
                'pnl': pnl_pct * 100,
                'pnl_amount': pnl_amount,
                'bars_held': len(data) - active_trade['entry_idx']
            }
            
            self.trades.append(trade)
        
        print("Improved Backtest Analysis:")
        print(f"Long signals: {long_signals}")
        print(f"Short signals: {short_signals}")
        print(f"Signal ratio: {signal_ratio:.2f}%")
        
        # Calcular métricas de rendimiento
        return self.calculate_performance_metrics()

    def calculate_performance_metrics(self, forced_days=None):
        """Calcular métricas de rendimiento con métricas adicionales de apalancamiento y tamaño de posición"""
        if not self.trades:
            return {
                'return_total': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trades_per_month': 0,
                'max_drawdown': 0,
                'total_trades': 0,
                'avg_leverage': 0,
                'avg_position_size': 0
            }
        
        # Calcular días totales para métricas mensuales
        if forced_days:
            days = forced_days
        else:
            start_date = pd.to_datetime(self.trades[0]['entry_time'])
            end_date = pd.to_datetime(self.trades[-1]['exit_time'])
            days = max(1, (end_date - start_date).days)  # Evitar división por cero
        
        # Calcular retorno total desde el capital inicial
        return_total = (self.equity_curve[-1] - 1.0) * 100
        
        # Calcular win rate
        wins = sum(1 for trade in self.trades if trade['pnl'] > 0)
        win_rate = (wins / len(self.trades)) * 100 if self.trades else 0
        
        # Calcular profit factor
        gross_profit = sum(max(0, trade['pnl_amount']) for trade in self.trades)
        gross_loss = abs(sum(min(0, trade['pnl_amount']) for trade in self.trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Calcular drawdown máximo
        peak = 1.0
        max_dd = 0
        
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
        
        # Calcular promedios de apalancamiento y tamaño de posición
        avg_leverage = sum(trade.get('leverage', 0) for trade in self.trades) / len(self.trades)
        
        # CORRECCIÓN AQUÍ: Dividir por el número de trades y luego convertir a porcentaje
        avg_position_size = sum(trade.get('position_size', 0) for trade in self.trades) / len(self.trades) * 100
        
        # También usar las listas para verificar
        if self.leverages and not avg_leverage:
            avg_leverage = sum(self.leverages) / len(self.leverages)
            
        if self.position_sizes and not avg_position_size:
            avg_position_size = sum(self.position_sizes) * 100 / len(self.position_sizes)
        
        return {
            'return_total': return_total,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'trades_per_month': (len(self.trades) / days) * 30,
            'max_drawdown': max_dd * 100,  # En porcentaje
            'total_trades': len(self.trades),
            'avg_leverage': avg_leverage,
            'avg_position_size': avg_position_size
        }
