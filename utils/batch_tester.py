import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from backtest.improved_backtest import ImprovedBacktest
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from utils.data_cache import DataCache

class BatchTester:
    """Sistema para ejecutar múltiples backtests en lote y comparar resultados"""
    
    def __init__(self, max_workers=4):
        self.cache = DataCache()
        self.max_workers = max_workers
        self.results_dir = '/home/panal/Documents/dashboard-trading/reports/batch_tests'
        self.config_dir = '/home/panal/Documents/dashboard-trading/configs'
        os.makedirs(self.results_dir, exist_ok=True)
        self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    def run_batch(self, configs, timeframes, periods):
        """
        Ejecutar múltiples backtests en paralelo
        
        Args:
            configs: Lista de configuraciones {'name': 'name_in_json', 'params': {...}}
            timeframes: Lista de timeframes ['15m', '1h', '4h', etc.]
            periods: Lista de períodos en días [30, 60, 180, etc.]
            
        Returns:
            DataFrame con resultados comparativos
        """
        print(f"Iniciando batch test con {len(configs)} configs, {len(timeframes)} timeframes, {len(periods)} períodos")
        print(f"Total de tests: {len(configs) * len(timeframes) * len(periods)}")
        
        all_tasks = []
        for config in configs:
            for tf in timeframes:
                for days in periods:
                    # Obtener config de archivo o usar directamente
                    if isinstance(config, dict) and 'name' in config:
                        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config['name']}.json"
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                params = json.load(f)
                                params['name'] = config['name']
                        else:
                            print(f"Config no encontrada: {config['name']}")
                            continue
                    else:
                        # Si la configuración es un string (nombre de archivo)
                        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config}.json"
                        if os.path.exists(config_path):
                            with open(config_path, 'r') as f:
                                params = json.load(f)
                                params['name'] = config
                        else:
                            print(f"Config no encontrada: {config}")
                            continue
                    
                    all_tasks.append((params, tf, days))
        
        # Ejecutar tests en secuencia (sin paralelismo por ahora)
        results = []
        total = len(all_tasks)
        completed = 0
        
        for task in all_tasks:
            params, tf, days = task
            try:
                print(f"\nEjecutando test con {params['name']}, {tf}, {days}d...")
                result = self.run_single_test(params, tf, days)
                
                # Logging more details about the result
                if 'error' in result:
                    print(f"❌ Error en test: {result['error']}")
                else:
                    print(f"✓ Test completado: Return={result['return_total']:.2f}%, Trades={result['total_trades']}")
                    results.append(result)
                    
            except Exception as e:
                import traceback
                print(f"❌ Error en {params['name']}, {tf}, {days}d: {str(e)}")
                print(traceback.format_exc())
            
            # Actualizar progreso
            completed += 1
            progress = completed / total * 100
            print(f"Progreso: {completed}/{total} ({progress:.1f}%)")
        
        # Convertir resultados a DataFrame
        if results:
            results_df = pd.DataFrame(results)
            
            # Guardar resultados
            csv_path = os.path.join(self.results_dir, f"batch_results_{self.session_id}.csv")
            results_df.to_csv(csv_path, index=False)
            
            # Generar visualizaciones
            self._generate_batch_visualizations(results_df)
            
            return results_df
        else:
            print("No se obtuvieron resultados válidos de ninguna configuración.")
            print("Verifique que la caché de datos esté actualizada ejecutando primero un backtest normal.")
            return pd.DataFrame()

    def run_single_test(self, config, timeframe, days):
        """Ejecuta un solo backtest y devuelve resultados"""
        # Obtener datos históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        try:
            # First try with standard format
            data = self.cache.get_cached_data(
                symbol='BTC/USDT', 
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            # If data is None, try with dashboard cache format 
            if data is None:
                print(f"No se encontraron datos en caché estándar para {timeframe}, intentando con caché del dashboard...")
                # Check for dashboard cache files
                dashboard_cache_path = '/home/panal/Documents/bot-machine-learning-main/data/cache'
                if os.path.exists(dashboard_cache_path):
                    cache_files = [f for f in os.listdir(dashboard_cache_path) if f.endswith('.pkl') and timeframe in f]
                    if cache_files:
                        latest_file = sorted(cache_files)[-1]
                        cache_path = os.path.join(dashboard_cache_path, latest_file)
                        print(f"Usando caché alternativa: {cache_path}")
                        data = pd.read_pickle(cache_path)
            
            if data is None:
                return {
                    'config': config['name'],
                    'timeframe': timeframe,
                    'days': days,
                    'error': 'No se encontraron datos en caché'
                }
                
            if data.empty:
                return {
                    'config': config['name'],
                    'timeframe': timeframe,
                    'days': days,
                    'error': 'Datos en caché vacíos'
                }
                
            print(f"Datos cargados: {len(data)} velas, desde {data.index.min()} hasta {data.index.max()}")
                
            # Ensure RSI window parameter exists in config
            if 'rsi' in config:
                if 'window' not in config['rsi']:
                    print(f"Añadiendo parámetro RSI window predeterminado...")
                    config['rsi']['window'] = 14
            
            # Seleccionar la estrategia correcta
            strategy_name = config.get('strategy', 'optimized')
            print(f"Ejecutando con estrategia: {strategy_name}")
            
            if strategy_name == 'enhanced':
                class TestBacktest(ImprovedBacktest):
                    def __init__(self, params):
                        super().__init__(params)
                        self.strategy = EnhancedStrategy()
            else:
                class TestBacktest(ImprovedBacktest):
                    def __init__(self, params):
                        super().__init__(params)
                        self.strategy = OptimizedStrategy()
                        
            # Ejecutar backtest
            backtest = TestBacktest(config)
            results = backtest.run(data)
            
            # Guardar información básica de trades
            trade_count = len(backtest.trades)
            print(f"Backtest completado: {trade_count} trades generados")
            
            # Extraer métricas principales
            metrics = {
                'config': config['name'],
                'timeframe': timeframe,
                'days': days,
                'return_total': results['return_total'],
                'win_rate': results['win_rate'],
                'profit_factor': results['profit_factor'],
                'max_drawdown': results['max_drawdown'],
                'trades_per_month': results['trades_per_month'],
                'total_trades': trade_count,
                'avg_leverage': results.get('avg_leverage', 0),
                'avg_position_size': results.get('avg_position_size', 0)
            }
            
            return metrics
            
        except Exception as e:
            import traceback
            print(f"Error ejecutando backtest: {str(e)}")
            print(traceback.format_exc())
            return {
                'config': config['name'],
                'timeframe': timeframe,
                'days': days,
                'error': f'Error durante ejecución: {str(e)}'
            }

    def _generate_batch_visualizations(self, df):
        """Generar visualizaciones para los resultados del batch test"""
        # 1. Heatmap de retornos por timeframe y período
        if len(df['timeframe'].unique()) > 1 and len(df['days'].unique()) > 1:
            plt.figure(figsize=(12, 8))
            pivot_data = df.pivot_table(
                index='timeframe', 
                columns='days', 
                values='return_total',
                aggfunc='mean'
            )
            
            # Crear heatmap
            ax = sns.heatmap(
                pivot_data, 
                cmap='RdYlGn', 
                annot=True, 
                fmt='.2f',
                linewidths=0.5,
                center=0
            )
            plt.title('Retorno (%) por Timeframe y Período')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f"heatmap_returns_{self.session_id}.png"))
        
        # 2. Gráfico de barras para comparar retornos entre configuraciones
        plt.figure(figsize=(14, 8))
        sns.barplot(x='config', y='return_total', hue='timeframe', data=df)
        plt.title('Retornos por Configuración y Timeframe')
        plt.xlabel('Configuración')
        plt.ylabel('Retorno (%)')
        plt.xticks(rotation=45)
        plt.legend(title='Timeframe')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"returns_by_config_{self.session_id}.png"))
        
        # 3. Gráfico de dispersión: Trades vs Return
        plt.figure(figsize=(12, 8))
        for config in df['config'].unique():
            config_data = df[df['config'] == config]
            plt.scatter(
                config_data['total_trades'], 
                config_data['return_total'],
                s=80,
                label=config,
                alpha=0.7
            )
        
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        plt.title('Relación entre Número de Trades y Retorno')
        plt.xlabel('Número de Trades')
        plt.ylabel('Retorno (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"trades_vs_return_{self.session_id}.png"))
        
        plt.close('all')

    def run_mode_analysis(self, config_name, timeframe='1h', days=30):
        """Analizar estrategia con diferentes modos y opciones"""
        # Cargar configuración base
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        if not os.path.exists(config_path):
            return {"error": f"Archivo de configuración no encontrado: {config_path}"}
            
        with open(config_path, 'r') as f:
            base_config = json.load(f)
        
        # Obtener datos históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None:
            return {"error": "No se encontraron datos en caché"}
        
        # Variables de modo a probar
        modes = [
            {"name": "normal", "config": {}},
            {"name": "sin_filtro_tendencia", "config": {"trend_filter": False}},
            {"name": "sin_filtro_volumen", "config": {"volume_filter": False}},
            {"name": "con_trailing", "config": {"use_trailing": True}},
            {"name": "con_salidas_parciales", "config": {"partial_exits": True}},
            {"name": "todo_activado", "config": {
                "trend_filter": True, 
                "volume_filter": True,
                "use_trailing": True,
                "partial_exits": True
            }},
            {"name": "todo_desactivado", "config": {
                "trend_filter": False, 
                "volume_filter": False,
                "use_trailing": False,
                "partial_exits": False
            }}
        ]
        
        # Ejecutar tests para cada modo
        results = []
        for mode in modes:
            # Crear copia de la configuración y aplicar modificaciones
            test_config = base_config.copy()
            for key, value in mode["config"].items():
                test_config[key] = value
                
            # Preparar backtest según estrategia
            strategy_name = test_config.get('strategy', 'optimized')
            
            if strategy_name == 'enhanced':
                class TestBacktest(ImprovedBacktest):
                    def __init__(self, params):
                        super().__init__(params)
                        self.strategy = EnhancedStrategy()
            else:
                class TestBacktest(ImprovedBacktest):
                    def __init__(self, params):
                        super().__init__(params)
                        self.strategy = OptimizedStrategy()
            
            # Ejecutar backtest
            backtest = TestBacktest(test_config)
            backtest_results = backtest.run(data)
            
            # Guardar resultados
            mode_result = {
                "mode": mode["name"],
                "config": config_name,
                **{k: v for k, v in backtest_results.items()},
                "total_trades": len(backtest.trades),
                "long_count": sum(1 for t in backtest.trades if t['position'] == 1),
                "short_count": sum(1 for t in backtest.trades if t['position'] == -1),
                "long_pct": sum(t.get('pnl_pct', 0) for t in backtest.trades if t['position'] == 1),
                "short_pct": sum(t.get('pnl_pct', 0) for t in backtest.trades if t['position'] == -1)
            }
            results.append(mode_result)
        
        # Convertir a DataFrame
        results_df = pd.DataFrame(results)
        
        # Guardar resultados a CSV
        csv_path = os.path.join(
            self.results_dir,
            f"modes_analysis_{config_name}_{timeframe}_{days}d_{self.session_id}.csv"
        )
        results_df.to_csv(csv_path, index=False)
        
        # Generar visualizaciones
        plt.figure(figsize=(14, 10))
        
        # Gráfico de barras para retorno por modo
        plt.subplot(2, 1, 1)
        sns.barplot(x='mode', y='return_total', data=results_df, palette='viridis')
        plt.title(f'Retorno por Modo - {config_name} ({timeframe}, {days}d)')
        plt.xlabel('Modo')
        plt.ylabel('Retorno (%)')
        plt.xticks(rotation=45)
        
        # Gráfico de barras para trades por modo
        plt.subplot(2, 1, 2)
        
        # Crear barras apiladas para long/short trades
        bottoms = np.zeros(len(results_df))
        
        # Trades LONG
        bars_long = plt.bar(
            results_df.index, 
            results_df['long_count'], 
            label='LONG',
            bottom=bottoms,
            color='green',
            alpha=0.7
        )
        
        # Actualizar bottoms para LONG
        bottoms = results_df['long_count'].values
        
        # Trades SHORT
        bars_short = plt.bar(
            results_df.index, 
            results_df['short_count'], 
            label='SHORT',
            bottom=bottoms,
            color='red',
            alpha=0.7
        )
        
        # Añadir etiquetas al gráfico
        for i, row in enumerate(results_df.itertuples()):
            plt.text(
                i, 
                row.long_count / 2, 
                str(row.long_count),
                ha='center', 
                va='center',
                color='white',
                fontweight='bold'
            )
            
            plt.text(
                i, 
                row.long_count + row.short_count / 2, 
                str(row.short_count),
                ha='center', 
                va='center',
                color='white',
                fontweight='bold'
            )
        
        plt.title(f'Trades por Modo - {config_name} ({timeframe}, {days}d)')
        plt.xlabel('Modo')
        plt.ylabel('Número de Trades')
        plt.xticks(range(len(results_df)), results_df['mode'], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(
            self.results_dir,
            f"modes_comparison_{config_name}_{timeframe}_{days}d.png"
        ))
        plt.close()
        
        return results_df

# Función helper para ejecutar lotes de pruebas
def run_batch_tests(configs=None, timeframes=None, periods=None):
    """Ejecuta un batch de pruebas con varias configuraciones"""
    # Valores por defecto
    if configs is None:
        # Obtener todas las configuraciones disponibles
        config_dir = '/home/panal/Documents/dashboard-trading/configs'
        configs = [f.split('.')[0] for f in os.listdir(config_dir) 
                 if f.endswith('.json') and not f.startswith('.')]
    
    if timeframes is None:
        timeframes = ['1h', '15m']  # Timeframes más comunes
        
    if periods is None:
        periods = [30, 180]  # Períodos corto y largo
    
    # Crear y ejecutar batch
    tester = BatchTester(max_workers=4)  # Ajusta según tu CPU
    
    print(f"Ejecutando pruebas en lote para {len(configs)} configuraciones")
    print(f"Timeframes: {timeframes}")
    print(f"Períodos: {periods} días")
    
    # Convertir strings a dicts con nombre
    config_objects = [{"name": config, "params": {}} for config in configs]
    
    results = tester.run_batch(config_objects, timeframes, periods)
    
    # Ejecutar análisis de modos para la mejor configuración
    if len(results) > 0:
        best_config = results.loc[results['return_total'].idxmax()]['config']
        print(f"\nEjecutando análisis de modos para la mejor configuración: {best_config}")
        
        for tf in timeframes:
            mode_results = tester.run_mode_analysis(best_config, tf, periods[0])
            print(f"Análisis de modos completado para {best_config} en {tf}. Guardado en: {tester.results_dir}")
    
    return results

if __name__ == "__main__":
    run_batch_tests()