#!/usr/bin/env python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from utils.data_cache import DataCache
from backtest.run_fixed_backtest import BacktestFixedStrategy
import json
import matplotlib.pyplot as plt
import os

class SystemVerifier:
    def __init__(self):
        self.cache = DataCache()
        self.output_dir = "/home/panal/Documents/dashboard-trading/reports/verification"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def verify_data_quality(self, symbol="BTC/USDT", timeframe="1h", days=30):
        """Verificar la calidad de los datos"""
        print("\n=== VERIFICACIÓN DE DATOS ===")
        
        # Obtener datos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.cache.get_cached_data(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None:
            print("❌ Error: No se pudieron obtener datos")
            return False
            
        print(f"✓ Datos obtenidos: {len(data)} barras")
        
        # Verificar columnas requeridas
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            print(f"❌ Error: Faltan columnas: {missing_columns}")
            return False
            
        print("✓ Todas las columnas requeridas están presentes")
        
        # Verificar valores nulos
        null_counts = data.isnull().sum()
        if null_counts.any():
            print("❌ Se encontraron valores nulos:")
            print(null_counts[null_counts > 0])
            return False
            
        print("✓ No hay valores nulos")
        
        # Verificar gaps en el tiempo
        time_diffs = pd.Series(data.index).diff()
        expected_diff = pd.Timedelta(timeframe.replace('m', 'min').replace('h', 'hour'))
        gaps = time_diffs[time_diffs > expected_diff * 1.5]
        
        if not gaps.empty:
            print("\n⚠️ Se encontraron gaps en los datos:")
            for idx, gap in gaps.items():
                print(f"  Gap de {gap} en {data.index[idx]}")
        
        # Verificar precios anómalos
        price_changes = data['close'].pct_change().abs()
        anomalies = price_changes[price_changes > 0.1]  # Cambios de más del 10%
        
        if not anomalies.empty:
            print("\n⚠️ Movimientos de precio inusuales detectados:")
            for idx, change in anomalies.items():
                print(f"  Cambio de {change*100:.2f}% en {idx}")
        
        # Visualizar datos
        self._plot_data_verification(data)
        
        return True
        
    def verify_signal_generation(self, data=None, config_name="hybrid_strategy"):
        """Verificar la generación de señales"""
        print("\n=== VERIFICACIÓN DE SEÑALES ===")
        
        if data is None:
            # Obtener datos de prueba
            data = self.cache.get_cached_data(
                symbol="BTC/USDT",
                timeframe="1h",
                start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                end_date=datetime.now().strftime('%Y-%m-%d')
            )
        
        # Cargar configuración
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_name}.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Activar modo debug
        config['debug'] = True
        
        # Ejecutar backtest
        backtest = BacktestFixedStrategy(config)
        results = backtest.run(data)
        
        # Verificar resultados
        print("\nEstadísticas de señales:")
        print(f"Total de trades: {results['total_trades']}")
        print(f"Win rate: {results['win_rate']:.2f}%")
        print(f"Profit factor: {results['profit_factor']:.2f}")
        print(f"Return total: {results['return_total']:.2f}%")
        
        # Analizar trades
        trades = results.get('trades', [])
        if trades:
            durations = [t['bars_held'] for t in trades]
            profits = [t['pnl'] for t in trades]
            
            print("\nAnálisis de trades:")
            print(f"Duración promedio: {np.mean(durations):.1f} barras")
            print(f"PnL promedio: {np.mean(profits):.2f}%")
            print(f"PnL máximo: {max(profits):.2f}%")
            print(f"PnL mínimo: {min(profits):.2f}%")
            
            # Visualizar distribución de trades
            self._plot_trade_analysis(trades)
        
        # Verificar coherencia de señales
        equity_curve = results.get('equity_curve')
        if isinstance(equity_curve, pd.Series):
            returns = equity_curve.pct_change()
            sharpe = np.sqrt(252) * returns.mean() / returns.std()
            print(f"\nSharpe Ratio: {sharpe:.2f}")
            
            # Visualizar equity curve
            self._plot_equity_verification(equity_curve)
        
        return results
        
    def _plot_data_verification(self, data):
        """Crear gráficos de verificación de datos"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
        
        # Precio y volumen
        ax1.plot(data.index, data['close'], label='Close')
        ax1.set_title('Precio de cierre')
        ax1.grid(True)
        
        # Volumen
        ax2.bar(data.index, data['volume'], color='gray', alpha=0.5)
        ax2.set_title('Volumen')
        ax2.grid(True)
        
        # Cambios porcentuales
        changes = data['close'].pct_change() * 100
        ax3.hist(changes, bins=50, color='blue', alpha=0.6)
        ax3.set_title('Distribución de cambios de precio (%)')
        ax3.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'data_verification.png'))
        plt.close()
        
    def _plot_trade_analysis(self, trades):
        """Crear gráficos de análisis de trades"""
        profits = [t['pnl'] for t in trades]
        durations = [t['bars_held'] for t in trades]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribución de PnL
        ax1.hist(profits, bins=30, color='green', alpha=0.6)
        ax1.axvline(x=0, color='red', linestyle='--')
        ax1.set_title('Distribución de PnL')
        ax1.grid(True)
        
        # Duración de trades
        ax2.hist(durations, bins=30, color='blue', alpha=0.6)
        ax2.set_title('Duración de trades')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'trade_analysis.png'))
        plt.close()
        
    def _plot_equity_verification(self, equity_curve):
        """Crear gráfico de verificación de equity curve"""
        plt.figure(figsize=(12, 6))
        
        # Equity curve
        plt.plot(equity_curve.index, equity_curve.values, color='blue')
        
        # Línea base
        plt.axhline(y=1.0, color='red', linestyle='--')
        
        # Drawdown
        drawdown = (equity_curve / equity_curve.cummax() - 1) * 100
        plt.fill_between(drawdown.index, 0, drawdown.values, color='red', alpha=0.3)
        
        plt.title('Equity Curve y Drawdown')
        plt.grid(True)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'equity_verification.png'))
        plt.close()

def main():
    """Ejecutar verificación completa del sistema"""
    verifier = SystemVerifier()
    
    # Verificar datos para múltiples timeframes
    timeframes = ['15m', '1h', '4h']
    for tf in timeframes:
        print(f"\nVerificando datos para timeframe {tf}")
        verifier.verify_data_quality(timeframe=tf, days=30)
    
    # Verificar generación de señales
    print("\nVerificando generación de señales")
    latest_data = verifier.cache.get_cached_data(
        symbol="BTC/USDT",
        timeframe="1h",
        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
        end_date=datetime.now().strftime('%Y-%m-%d')
    )
    
    if latest_data is not None:
        results = verifier.verify_signal_generation(latest_data)
        
        print("\n=== VERIFICACIÓN COMPLETADA ===")
        print(f"Reportes guardados en: {verifier.output_dir}")
    else:
        print("❌ Error: No se pudieron obtener datos para la verificación")

if __name__ == "__main__":
    main()
