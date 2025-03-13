import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
from backtest.run_fixed_backtest import BacktestFixedStrategy
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from backtest.improved_backtest import ImprovedBacktest
from utils.data_cache import DataCache

class DiagnosticTool:
    """Herramienta para diagnosticar problemas en los resultados de backtests"""
    
    def __init__(self, config_dir='/home/panal/Documents/dashboard-trading/configs'):
        self.config_dir = config_dir
        self.cache = DataCache()
        self.results_dir = '/home/panal/Documents/dashboard-trading/reports/diagnostics'
        
        # Crear directorio para resultados si no existe
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_metric_consistency(self, backtest_result: Dict):
        """Analizar consistencia entre métricas"""
        inconsistencies = []
        
        # Revisar consistencia entre profit factor y return
        profit_factor = backtest_result.get('profit_factor', 0)
        return_total = backtest_result.get('return_total', 0)
        
        if profit_factor > 1 and return_total < 0:
            inconsistencies.append({
                'type': 'profit_factor_return_mismatch',
                'severity': 'high',
                'message': f"Profit Factor ({profit_factor:.2f}) > 1 pero Return ({return_total:.2f}%) < 0",
                'possible_cause': "Posible problema con el cálculo del profit factor o con el capital inicial"
            })
        
        if profit_factor < 1 and return_total > 0:
            inconsistencies.append({
                'type': 'profit_factor_return_mismatch',
                'severity': 'high',
                'message': f"Profit Factor ({profit_factor:.2f}) < 1 pero Return ({return_total:.2f}%) > 0",
                'possible_cause': "Posible problema con el tratamiento de comisiones o salidas parciales"
            })
            
        # Revisar consistencia entre win rate y profit factor
        win_rate = backtest_result.get('win_rate', 0)
        if win_rate > 60 and profit_factor < 1:
            inconsistencies.append({
                'type': 'winrate_profitfactor_mismatch',
                'severity': 'medium',
                'message': f"Win Rate alto ({win_rate:.2f}%) pero Profit Factor bajo ({profit_factor:.2f})",
                'possible_cause': "Pérdidas grandes en operaciones perdedoras, ganancias pequeñas en ganadoras"
            })
        
        if win_rate < 40 and profit_factor > 1.5:
            inconsistencies.append({
                'type': 'winrate_profitfactor_mismatch',
                'severity': 'medium',
                'message': f"Win Rate bajo ({win_rate:.2f}%) pero Profit Factor alto ({profit_factor:.2f})",
                'possible_cause': "Ganancias grandes en operaciones ganadoras, pérdidas pequeñas en perdedoras"
            })
            
        return inconsistencies
    
    def analyze_trade_list(self, trades: List[Dict]):
        """Analizar la lista de operaciones para encontrar anomalías"""
        if not trades:
            return {"error": "No hay operaciones para analizar"}
        
        # Stats básicos
        trade_count = len(trades)
        pnl_values = []
        holding_times = []
        leverage_values = []
        position_sizes = []
        
        # Extraer valores
        for trade in trades:
            if 'pnl_pct' in trade:
                pnl_values.append(trade['pnl_pct'])
            elif 'pnl' in trade:
                pnl_values.append(trade['pnl'])
                
            if 'entry_time' in trade and 'exit_time' in trade:
                entry = pd.to_datetime(trade['entry_time'])
                exit = pd.to_datetime(trade['exit_time'])
                holding_time = (exit - entry).total_seconds() / 3600  # en horas
                holding_times.append(holding_time)
            
            if 'leverage' in trade:
                leverage_values.append(trade['leverage'])
                
            if 'position_size' in trade:
                position_sizes.append(trade['position_size'])
        
        # Análisis de resultados
        if pnl_values:
            win_rate = sum(1 for x in pnl_values if x > 0) / len(pnl_values) * 100
            avg_win = np.mean([x for x in pnl_values if x > 0]) if any(x > 0 for x in pnl_values) else 0
            avg_loss = np.mean([x for x in pnl_values if x < 0]) if any(x < 0 for x in pnl_values) else 0
            profit_factor = abs(sum(x for x in pnl_values if x > 0) / sum(x for x in pnl_values if x < 0)) if sum(x for x in pnl_values if x < 0) != 0 else float('inf')
            
            # Analizar rachas
            streaks = []
            current_streak = {'type': None, 'count': 0}
            
            for pnl in pnl_values:
                streak_type = 'win' if pnl > 0 else 'loss'
                
                if current_streak['type'] == streak_type:
                    current_streak['count'] += 1
                else:
                    if current_streak['type'] is not None:
                        streaks.append(current_streak.copy())
                    current_streak = {'type': streak_type, 'count': 1}
            
            # Añadir la última racha
            if current_streak['count'] > 0:
                streaks.append(current_streak)
            
            # Encontrar rachas más largas
            max_win_streak = max([s['count'] for s in streaks if s['type'] == 'win'], default=0)
            max_loss_streak = max([s['count'] for s in streaks if s['type'] == 'loss'], default=0)
            
            return {
                "trade_count": trade_count,
                "win_rate": win_rate,
                "avg_win": avg_win,
                "avg_loss": avg_loss,
                "profit_factor": profit_factor,
                "avg_holding_time": np.mean(holding_times) if holding_times else 0,
                "avg_leverage": np.mean(leverage_values) if leverage_values else 0,
                "avg_position_size": np.mean(position_sizes) * 100 if position_sizes else 0,
                "max_win_streak": max_win_streak,
                "max_loss_streak": max_loss_streak,
                "win_loss_ratio": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
            }
        else:
            return {"error": "No se encontraron valores PnL en las operaciones"}
    
    def run_validation_test(self, config_name: str, timeframe: str = '1h', days: int = 30):
        """Ejecutar test de validación con configuración específica"""
        # Cargar configuración
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        if not os.path.exists(config_path):
            return {"error": f"Archivo de configuración no encontrado: {config_path}"}
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
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
            
        # Preparar backtest según estrategia
        strategy_name = config.get('strategy', 'optimized')
        
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
        
        # Analizar consistencia de resultados
        inconsistencies = self.analyze_metric_consistency(results)
        trade_analysis = self.analyze_trade_list(backtest.trades)
        
        # Preparar reporte
        report = {
            "config_name": config_name,
            "timeframe": timeframe,
            "days": days,
            "results": results,
            "inconsistencies": inconsistencies,
            "trade_analysis": trade_analysis,
            "total_trades": len(backtest.trades)
        }
        
        # Guardar reporte
        report_path = os.path.join(
            self.results_dir, 
            f"validation_{config_name}_{timeframe}_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        
        with open(report_path, 'w') as f:
            # Convert complex types to strings for JSON serialization
            report_json = report.copy()
            report_json['results'] = {k: str(v) for k, v in report['results'].items()}
            json.dump(report_json, f, indent=2)
        
        # Generar visualizaciones
        if len(backtest.trades) > 0:
            self._generate_trade_visualizations(backtest.trades, config_name, timeframe, days)
            
        return report
    
    def _generate_trade_visualizations(self, trades: List[Dict], config_name: str, timeframe: str, days: int):
        """Generar visualizaciones para análisis de trades"""
        # Extraer datos para análisis
        trade_data = []
        
        for trade in trades:
            trade_info = {
                'entry_time': pd.to_datetime(trade['entry_time']),
                'position': trade['position'],
                'leverage': trade.get('leverage', 1),
                'position_size': trade.get('position_size', 0) * 100,
            }
            
            # PnL en diferentes formatos posibles
            if 'pnl_pct' in trade:
                trade_info['pnl'] = trade['pnl_pct']
            elif 'pnl' in trade:
                trade_info['pnl'] = trade['pnl']
            else:
                # Calcular PnL si no está presente
                entry = trade['entry_price']
                exit = trade['exit_price']
                direction = trade['position']
                trade_info['pnl'] = ((exit / entry - 1) * direction) * 100
            
            trade_data.append(trade_info)
            
        # Convertir a DataFrame para análisis
        trades_df = pd.DataFrame(trade_data)
        
        # 1. Visualizar PnL por operación
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(trades_df)), 
                trades_df['pnl'], 
                color=[('green' if x > 0 else 'red') for x in trades_df['pnl']])
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.title(f'PnL por Operación - {config_name} ({timeframe}, {days}d)')
        plt.xlabel('# Operación')
        plt.ylabel('PnL (%)')
        plt.tight_layout()
        
        # Guardar gráfico
        plt.savefig(os.path.join(self.results_dir, f"pnl_chart_{config_name}_{timeframe}_{days}d.png"))
        
        # 2. Equity curve
        cumulative_returns = (1 + trades_df['pnl'] / 100).cumprod()
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative_returns.values, color='blue')
        plt.title(f'Equity Curve - {config_name} ({timeframe}, {days}d)')
        plt.xlabel('# Operación')
        plt.ylabel('Equity (1 = 100%)')
        plt.tight_layout()
        
        # Guardar equity curve
        plt.savefig(os.path.join(self.results_dir, f"equity_curve_{config_name}_{timeframe}_{days}d.png"))
        
        # 3. PnL por dirección (Long vs Short)
        if 'position' in trades_df.columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='position', y='pnl', data=trades_df)
            plt.title(f'PnL por Dirección - {config_name} ({timeframe}, {days}d)')
            plt.xlabel('Dirección (1=Long, -1=Short)')
            plt.ylabel('PnL (%)')
            plt.tight_layout()
            
            # Guardar boxplot
            plt.savefig(os.path.join(self.results_dir, f"pnl_direction_{config_name}_{timeframe}_{days}d.png"))
            
        # 4. Distribución de PnL
        plt.figure(figsize=(10, 6))
        sns.histplot(trades_df['pnl'], bins=20, kde=True)
        plt.axvline(x=0, color='red', linestyle='--')
        plt.title(f'Distribución de PnL - {config_name} ({timeframe}, {days}d)')
        plt.xlabel('PnL (%)')
        plt.ylabel('Frecuencia')
        plt.tight_layout()
        
        # Guardar histograma
        plt.savefig(os.path.join(self.results_dir, f"pnl_dist_{config_name}_{timeframe}_{days}d.png"))
        
        plt.close('all')
    
    def compare_configurations(self, configs: List[str], timeframe: str = '1h', days: int = 30):
        """Comparar varias configuraciones en el mismo período"""
        all_results = {}
        
        for config_name in configs:
            print(f"Ejecutando validación para {config_name}...")
            result = self.run_validation_test(config_name, timeframe, days)
            if 'error' not in result:
                all_results[config_name] = result
            else:
                print(f"Error en {config_name}: {result['error']}")
                
        # Crear tabla comparativa
        comparison_data = []
        for config_name, result in all_results.items():
            metrics = result['results']
            trade_analysis = result['trade_analysis']
            
            row = {
                'Configuración': config_name,
                'Retorno (%)': metrics.get('return_total', 0),
                'Win Rate (%)': metrics.get('win_rate', 0),
                'Profit Factor': metrics.get('profit_factor', 0),
                'Trades/Mes': metrics.get('trades_per_month', 0),
                'Max DD (%)': metrics.get('max_drawdown', 0),
                'Total Trades': metrics.get('total_trades', 0),
                'Apalancamiento': metrics.get('avg_leverage', 0),
                'Tamaño Pos. (%)': metrics.get('avg_position_size', 0),
                'Win/Loss Ratio': trade_analysis.get('win_loss_ratio', 0),
                'Max Win Streak': trade_analysis.get('max_win_streak', 0),
                'Max Loss Streak': trade_analysis.get('max_loss_streak', 0)
            }
            comparison_data.append(row)
            
        comparison_df = pd.DataFrame(comparison_data)
        
        # Guardar comparativa a CSV
        csv_path = os.path.join(
            self.results_dir, 
            f"comparison_{timeframe}_{days}d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        comparison_df.to_csv(csv_path, index=False)
        
        # Generar gráfico comparativo de retornos
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Configuración', y='Retorno (%)', data=comparison_df)
        plt.title(f'Comparación de Retornos por Configuración ({timeframe}, {days}d)')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"returns_comparison_{timeframe}_{days}d.png"))
        
        # Generar gráfico comparativo de win rate
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Configuración', y='Win Rate (%)', data=comparison_df)
        plt.title(f'Comparación de Win Rate por Configuración ({timeframe}, {days}d)')
        plt.xticks(rotation=45)
        plt.axhline(y=50, color='red', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"winrate_comparison_{timeframe}_{days}d.png"))
        
        plt.close('all')
        
        return comparison_df
    
    def run_robustness_test(self, config_name: str, timeframes: List[str] = ['1h'], 
                          periods: List[int] = [30, 60, 180]):
        """Ejecutar test de robustez en múltiples timeframes y períodos"""
        results = []
        
        for timeframe in timeframes:
            for days in periods:
                print(f"Ejecutando {config_name} en {timeframe}, {days} días...")
                report = self.run_validation_test(config_name, timeframe, days)
                
                if 'error' not in report:
                    summary = {
                        'timeframe': timeframe,
                        'days': days,
                        'return': report['results']['return_total'],
                        'win_rate': report['results']['win_rate'],
                        'profit_factor': report['results']['profit_factor'],
                        'trades': report['total_trades'],
                        'inconsistencies': len(report['inconsistencies'])
                    }
                    results.append(summary)
                else:
                    print(f"Error: {report['error']}")
        
        # Convertir a DataFrame
        results_df = pd.DataFrame(results)
        
        # Guardar resultados
        csv_path = os.path.join(
            self.results_dir, 
            f"robustness_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        results_df.to_csv(csv_path, index=False)
        
        # Crear visualización de robustez
        if not results_df.empty:
            plt.figure(figsize=(12, 8))
            
            # Crear una matriz de tamaño y color
            plt.scatter(
                results_df['timeframe'], 
                results_df['days'],
                s=results_df['trades'] * 3,  # Tamaño basado en número de trades
                c=results_df['return'],      # Color basado en retorno
                cmap='RdYlGn',               # Colormap: rojo (negativo) a verde (positivo)
                alpha=0.7
            )
            
            # Añadir etiquetas con retorno
            for _, row in results_df.iterrows():
                plt.annotate(
                    f"{row['return']:.1f}%\n({row['trades']}t)",
                    (row['timeframe'], row['days']),
                    ha='center'
                )
                
            plt.title(f'Mapa de Robustez - {config_name}')
            plt.xlabel('Timeframe')
            plt.ylabel('Período (días)')
            plt.colorbar(label='Retorno (%)')
            plt.tight_layout()
            
            # Guardar el mapa de robustez
            plt.savefig(os.path.join(self.results_dir, f"robustness_map_{config_name}.png"))
            plt.close()
            
        return results_df
                

# Función de ayuda para ejecutar tests rápidos
def run_diagnostics():
    """Ejecutar diagnósticos completos para probar consistencia de estrategias"""
    tool = DiagnosticTool()
    
    # 1. Analizar resultados inconsistentes
    print("Analizando las diferentes configuraciones...")
    all_configs = [f.split('.')[0] for f in os.listdir(tool.config_dir) 
                 if f.endswith('.json') and not f.startswith('.')]
    
    comparison = tool.compare_configurations(all_configs, timeframe='1h', days=30)
    print("\nResumen de resultados:")
    print(comparison)
    
    # 2. Realizar prueba de robustez en la mejor configuración
    best_config = comparison.loc[comparison['Retorno (%)'].idxmax()]['Configuración']
    print(f"\nRealizando prueba de robustez en {best_config}...")
    robustness = tool.run_robustness_test(
        best_config,
        timeframes=['15m', '1h', '4h'],
        periods=[30, 60, 180]
    )
    
    # 3. Analizar posibles inconsistencias en profundidad
    inconsistent_configs = []
    for _, row in comparison.iterrows():
        # Identificar configuraciones con potenciales inconsistencias
        if (row['Profit Factor'] > 1 and row['Retorno (%)'] < 0) or \
           (row['Profit Factor'] < 1 and row['Retorno (%)'] > 0):
            inconsistent_configs.append(row['Configuración'])
            
    if inconsistent_configs:
        print("\n⚠️ Se encontraron inconsistencias en las siguientes configuraciones:")
        for config in inconsistent_configs:
            print(f"  - {config}")
            
        print("\nRealizando análisis detallado de inconsistencias...")
        for config in inconsistent_configs:
            report = tool.run_validation_test(config, timeframe='1h', days=60)
            if 'inconsistencies' in report:
                print(f"\nInconsistencias en {config}:")
                for issue in report['inconsistencies']:
                    print(f"  - {issue['message']}")
                    print(f"    Posible causa: {issue['possible_cause']}")
    
    print("\nDiagnóstico completado. Resultados guardados en:", tool.results_dir)
    
    return {
        "comparison": comparison,
        "robustness": robustness,
        "inconsistent_configs": inconsistent_configs
    }

if __name__ == "__main__":
    run_diagnostics()
