import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Union
from utils.data_cache import DataCache
from backtest.improved_backtest import ImprovedBacktest
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from utils.market_analyzer import MarketAnalyzer, MarketRegime

class StrategyAnalyzer:
    """
    Herramienta avanzada para analizar rendimiento de estrategias y optimizar parámetros
    basado en análisis de regímenes de mercado y comportamiento por condición.
    """
    
    def __init__(self, config_dir='/home/panal/Documents/dashboard-trading/configs'):
        self.config_dir = config_dir
        self.cache = DataCache()
        self.results_dir = '/home/panal/Documents/dashboard-trading/reports/analysis'
        os.makedirs(self.results_dir, exist_ok=True)
        self.market_analyzer = MarketAnalyzer()
        
    def run_analysis(self, config_name: str, timeframe: str = '1h', days: int = 90):
        """Ejecutar análisis detallado de una estrategia"""
        print(f"\n==== ANÁLISIS DE ESTRATEGIA: {config_name} ====")
        
        # 1. Cargar configuración
        config_path = os.path.join(self.config_dir, f"{config_name}.json")
        if not os.path.exists(config_path):
            print(f"❌ Configuración no encontrada: {config_path}")
            return None
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 2. Obtener datos históricos
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        data = self.cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None:
            # Intentar con cache alternativo
            dashboard_cache_path = '/home/panal/Documents/bot-machine-learning-main/data/cache'
            if os.path.exists(dashboard_cache_path):
                cache_files = [f for f in os.listdir(dashboard_cache_path) if f.endswith('.pkl') and timeframe in f]
                if cache_files:
                    latest_file = sorted(cache_files)[-1]
                    cache_path = os.path.join(dashboard_cache_path, latest_file)
                    print(f"Usando caché alternativa: {cache_path}")
                    data = pd.read_pickle(cache_path)
        
        if data is None or data.empty:
            print(f"❌ No se encontraron datos para análisis")
            return None
        
        # 3. Preparar backtest
        strategy_name = config.get('strategy', 'optimized')
        
        if 'rsi' in config and 'window' not in config['rsi']:
            config['rsi']['window'] = 14
        
        if strategy_name == 'enhanced':
            class TestBacktest(ImprovedBacktest):
                def __init__(self, params):
                    super().__init__(params)
                    self.strategy = EnhancedStrategy()
                    # Añadir variable para rastrear estado del mercado
                    self.market_regimes = []
                    self.conditions_hit = {}
        else:
            class TestBacktest(ImprovedBacktest):
                def __init__(self, params):
                    super().__init__(params)
                    self.strategy = OptimizedStrategy()
                    # Añadir variable para rastrear estado del mercado
                    self.market_regimes = []
                    self.conditions_hit = {}
        
        # 4. Ejecutar backtest
        backtest = TestBacktest(config)
        results = backtest.run(data)
        
        # 5. Analizar rendimiento de la estrategia
        print("\n1. ESTADÍSTICAS GENERALES")
        print(f"Retorno total: {results['return_total']:.2f}%")
        print(f"Win Rate: {results['win_rate']:.2f}%")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Total de operaciones: {results['total_trades']}")
        
        # 6. Detección de regímenes de mercado para todo el período
        print("\n2. ANÁLISIS DE REGÍMENES DE MERCADO")
        # Use smaller window sizes for better detection with limited data
        window_size = min(10, max(5, len(data) // 20))
        step_size = max(1, window_size // 2)
        
        market_regimes = []
        periods = []
        
        # Detectar regímenes en ventanas deslizantes
        for i in range(0, len(data) - window_size, step_size):
            window_data = data.iloc[i:i+window_size]
            regime = self.market_analyzer.detect_market_regime(window_data)
            start_date = window_data.index[0]
            end_date = window_data.index[-1]
            
            market_regimes.append({
                'regime': regime,
                'start': start_date,
                'end': end_date,
                'days': (end_date - start_date).days
            })
            periods.append((start_date, end_date))
        
        # Contar regímenes detectados
        regime_counts = {}
        for entry in market_regimes:
            regime_name = entry['regime'].value
            if regime_name in regime_counts:
                regime_counts[regime_name] += 1
            else:
                regime_counts[regime_name] = 1
        
        # Mostrar distribución de regímenes
        total_regimes = sum(regime_counts.values())
        for regime, count in regime_counts.items():
            print(f"{regime}: {count} períodos ({count/total_regimes*100:.1f}%)")
        
        # 7. Analizar rendimiento por régimen de mercado
        print("\n3. RENDIMIENTO POR RÉGIMEN DE MERCADO")
        
        # Clasificar trades por régimen
        trades_by_regime = {k.value: [] for k in MarketRegime}
        
        for trade in backtest.trades:
            entry_time = pd.to_datetime(trade['entry_time'])
            exit_time = pd.to_datetime(trade['exit_time'])
            
            # Determinar en qué período/régimen ocurrió la operación
            assigned_regime = None
            
            for i, (period_start, period_end) in enumerate(periods):
                # Si la entrada está dentro del período, asignar ese régimen
                if period_start <= entry_time <= period_end:
                    assigned_regime = market_regimes[i]['regime'].value
                    break
            
            if assigned_regime:
                trades_by_regime[assigned_regime].append(trade)
            else:
                # Si no se puede asignar un régimen, usar 'unknown'
                if 'unknown' not in trades_by_regime:
                    trades_by_regime['unknown'] = []
                trades_by_regime['unknown'].append(trade)
        
        # Calcular rendimiento por régimen
        for regime, trades in trades_by_regime.items():
            if not trades:
                continue
            
            trade_count = len(trades)
            
            # Fix win rate calculation by handling different PnL field names
            win_count = 0
            pnl_values = []
            
            for t in trades:
                pnl = None
                if 'pnl_pct' in t:
                    pnl = t['pnl_pct']
                elif 'pnl' in t:
                    pnl = t['pnl']
                elif 'profit_pct' in t:
                    pnl = t['profit_pct']
                
                if pnl is not None:
                    pnl_values.append(pnl)
                    if pnl > 0:
                        win_count += 1
            
            win_rate = (win_count / trade_count) * 100 if trade_count > 0 else 0
            avg_pnl = sum(pnl_values) / len(pnl_values) if pnl_values else 0
            
            print(f"\nRégimen: {regime}")
            print(f"  Trades: {trade_count}")
            print(f"  Win Rate: {win_rate:.2f}%")
            print(f"  PnL Promedio: {avg_pnl:.2f}%")
        
        # 8. Analizar patrones de entradas y salidas
        print("\n4. ANÁLISIS DE ENTRADAS Y SALIDAS")
        
        # Calcular duración promedio de trades
        durations = []
        for trade in backtest.trades:
            entry = pd.to_datetime(trade['entry_time'])
            exit = pd.to_datetime(trade['exit_time'])
            duration = (exit - entry).total_seconds() / 3600  # hours
            durations.append(duration)
            
        avg_duration = sum(durations) / len(durations) if durations else 0
        print(f"Duración promedio de trades: {avg_duration:.1f} horas")
        
        # Calcular PnL por dirección
        long_trades = [t for t in backtest.trades if t['position'] == 1]
        short_trades = [t for t in backtest.trades if t['position'] == -1]
        
        long_win_count = 0
        short_win_count = 0
        long_pnl = []
        short_pnl = []
        
        for trade in long_trades:
            pnl = None
            if 'pnl_pct' in trade:
                pnl = trade['pnl_pct']
            elif 'pnl' in trade:
                pnl = trade['pnl']
            elif 'profit_pct' in trade:
                pnl = trade['profit_pct']
                
            if pnl is not None:
                long_pnl.append(pnl)
                if pnl > 0:
                    long_win_count += 1
        
        for trade in short_trades:
            pnl = None
            if 'pnl_pct' in trade:
                pnl = trade['pnl_pct']
            elif 'pnl' in trade:
                pnl = trade['pnl']
            elif 'profit_pct' in trade:
                pnl = trade['profit_pct']
                
            if pnl is not None:
                short_pnl.append(pnl)
                if pnl > 0:
                    short_win_count += 1
        
        long_win_rate = (long_win_count / len(long_trades)) * 100 if long_trades else 0
        short_win_rate = (short_win_count / len(short_trades)) * 100 if short_trades else 0
        
        print(f"Long trades: {len(long_trades)}, Win rate: {long_win_rate:.2f}%")
        print(f"Short trades: {len(short_trades)}, Win rate: {short_win_rate:.2f}%")
        
        # 9. Generar recomendaciones basadas en análisis
        print("\n5. RECOMENDACIONES PARA OPTIMIZACIÓN")
        
        recommendations = self._generate_recommendations(
            config,
            results,
            trades_by_regime,
            {
                'long_win_rate': long_win_rate,
                'short_win_rate': short_win_rate,
                'regime_distribution': regime_counts
            }
        )
        
        for i, rec in enumerate(recommendations):
            print(f"{i+1}. {rec}")
        
        # 10. Guardar resultados de análisis
        self._save_analysis_results(
            config_name, 
            {
                'config': config,
                'results': {k: str(v) if isinstance(v, (float, np.float64)) 
                           and not np.isnan(v) and not np.isinf(v) 
                           else v for k, v in results.items()},
                'regimes': regime_counts,
                'long_win_rate': long_win_rate,
                'short_win_rate': short_win_rate,
                'recommendations': recommendations
            }
        )
        
        # Generar gráficos de análisis
        self._generate_analysis_charts(
            backtest.trades, 
            market_regimes,
            trades_by_regime,
            config_name
        )
        
        return {
            'config': config,
            'results': results,
            'recommendations': recommendations
        }

    def _generate_recommendations(self, config, results, trades_by_regime, stats):
        """Generar recomendaciones basadas en análisis"""
        recommendations = []
        strategy_type = config.get('strategy', 'optimized')
        
        # 1. Analizar resultados generales
        if results['return_total'] < 0:
            recommendations.append(
                "Ajustar parámetros para mejorar rendimiento general (resultado negativo actual)"
            )
        
        # 2. Analizar win rate
        if results['win_rate'] < 45:
            recommendations.append(
                f"Win rate bajo ({results['win_rate']:.1f}%). Considere ajustar los filtros para mayor precisión"
            )
        
        # 3. Analizar desbalance entre long y short
        long_short_ratio = abs(stats['long_win_rate'] - stats['short_win_rate'])
        if long_short_ratio > 15:  # Más de 15% de diferencia
            better_direction = "LONG" if stats['long_win_rate'] > stats['short_win_rate'] else "SHORT"
            worse_direction = "SHORT" if better_direction == "LONG" else "LONG"
            recommendations.append(
                f"Desbalance significativo: {better_direction} ({stats['long_win_rate']:.1f}% vs {stats['short_win_rate']:.1f}% en {worse_direction}). " +
                f"Ajustar criterios para {worse_direction} o considerar una estrategia direccional"
            )
        
        # 4. Recomendaciones basadas en régimen
        regime_counts = stats['regime_distribution']
        
        # Encontrar régimen predominante
        if regime_counts:
            predominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            
            # Si hay suficientes trades en ese régimen
            if predominant_regime in trades_by_regime and trades_by_regime[predominant_regime]:
                regime_trades = trades_by_regime[predominant_regime]
                regime_win_rate = sum(1 for t in regime_trades if t.get('pnl_pct', 0) > 0) / len(regime_trades) * 100
                
                if regime_win_rate < 45:
                    if predominant_regime == 'trending_up':
                        recommendations.append(
                            "Optimizar para mercados alcistas: considere EMAs más cortas (8,20) y mantener trades por más tiempo"
                        )
                    elif predominant_regime == 'trending_down':
                        recommendations.append(
                            "Optimizar para mercados bajistas: considere filtros más estrictos para LONG y más agresivos para SHORT"
                        )
                    elif predominant_regime == 'ranging':
                        recommendations.append(
                            "Optimizar para mercados laterales: ajustar RSI a rangos más amplios (30-70)"
                        )
                    elif predominant_regime == 'volatile':
                        recommendations.append(
                            "Optimizar para mercados volátiles: reducir tamaño de posición y usar stops más amplios"
                        )
        
        # 5. Recomendaciones específicas por estrategia
        if strategy_type == 'enhanced':
            recommendations.append(
                "Para estrategia Enhanced: considere ajustar los filtros de tendencia y use más divergencias"
            )
        else:
            recommendations.append(
                "Para estrategia Optimized: considere refinar los filtros de volatilidad y volumen"
            )
        
        # 6. Recomendaciones para la gestión de riesgos
        if results['max_drawdown'] > 10:
            recommendations.append(
                f"Drawdown elevado ({results['max_drawdown']:.1f}%). Reduzca tamaño de posición y mejore sistema de stops"
            )
        
        # 7. Recomendación específica para el timeframe
        recommendations.append(
            "Realice backtesting en múltiples timeframes para confirmar consistencia de la estrategia"
        )
        
        # 8. Propuesta de optimización paramétrica
        optimal_params = self._suggest_optimal_parameters(config, stats)
        param_changes = []
        
        for k, v in optimal_params.items():
            if k == 'rsi' and 'rsi' in config:
                if v['oversold'] != config['rsi'].get('oversold'):
                    param_changes.append(f"RSI oversold: {config['rsi'].get('oversold')} → {v['oversold']}")
                if v['overbought'] != config['rsi'].get('overbought'):
                    param_changes.append(f"RSI overbought: {config['rsi'].get('overbought')} → {v['overbought']}")
            elif k == 'ema' and 'ema' in config:
                if v['short'] != config['ema'].get('short'):
                    param_changes.append(f"EMA corta: {config['ema'].get('short')} → {v['short']}")
                if v['long'] != config['ema'].get('long'):
                    param_changes.append(f"EMA larga: {config['ema'].get('long')} → {v['long']}")
        
        if param_changes:
            rec = "Parámetros optimizados sugeridos: " + ", ".join(param_changes)
            recommendations.append(rec)
        
        return recommendations
            
    def _suggest_optimal_parameters(self, config, stats):
        """Sugerir parámetros optimizados basados en rendimiento y regímenes"""
        # Copia de parámetros actuales
        optimal_params = config.copy()
        
        # Adaptaciones según régimen predominante
        regime_counts = stats['regime_distribution']
        if regime_counts:
            predominant_regime = max(regime_counts.items(), key=lambda x: x[1])[0]
            
            # Ajustar RSI según régimen predominante
            if 'rsi' in optimal_params:
                if predominant_regime == 'trending_up':
                    optimal_params['rsi'] = {
                        'window': 14,
                        'oversold': 40,  # Menos sensible en tendencia alcista
                        'overbought': 70  # Más sensible para salidas en tendencia
                    }
                elif predominant_regime == 'trending_down':
                    optimal_params['rsi'] = {
                        'window': 14,
                        'oversold': 30,  # Más sensible para compras en tendencia bajista
                        'overbought': 60  # Menos sensible para ventas
                    }
                elif predominant_regime == 'ranging':
                    optimal_params['rsi'] = {
                        'window': 14,
                        'oversold': 35,  # Valores equilibrados para mercado de rango
                        'overbought': 65
                    }
                elif predominant_regime == 'volatile':
                    optimal_params['rsi'] = {
                        'window': 21,  # Ventana más larga para filtrar ruido
                        'oversold': 30,  # Valores más extremos para evitar falsas señales
                        'overbought': 70
                    }
            
            # Ajustar EMAs según régimen
            if 'ema' in optimal_params:
                if predominant_regime == 'trending_up':
                    optimal_params['ema'] = {
                        'short': 8,   # EMAs más cortas para seguir tendencia rápidamente
                        'long': 21
                    }
                elif predominant_regime == 'trending_down':
                    optimal_params['ema'] = {
                        'short': 8,
                        'long': 21
                    }
                elif predominant_regime == 'ranging':
                    optimal_params['ema'] = {
                        'short': 10,  # EMAs más equilibradas para mercado de rango
                        'long': 25
                    }
                elif predominant_regime == 'volatile':
                    optimal_params['ema'] = {
                        'short': 12,  # EMAs más lentas para mercados volátiles
                        'long': 30
                    }
        
        return optimal_params
            
    def _save_analysis_results(self, config_name, analysis_data):
        """Guardar resultados de análisis a archivo JSON"""
        file_path = os.path.join(
            self.results_dir,
            f"analysis_{config_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        )
        
        with open(file_path, 'w') as f:
            json.dump(analysis_data, f, indent=2, default=str)
        
        print(f"\nResultados guardados en: {file_path}")
            
    def _generate_analysis_charts(self, trades, market_regimes, trades_by_regime, config_name):
        """Generar gráficos de análisis"""
        # 1. Gráfico de PnL por operación
        if trades:
            pnl_values = []
            entry_times = []
            
            for trade in trades:
                pnl = None
                if 'pnl_pct' in trade:
                    pnl = trade['pnl_pct']
                elif 'pnl' in trade:
                    pnl = trade['pnl']
                    
                if pnl is not None:
                    pnl_values.append(pnl)
                    entry_times.append(pd.to_datetime(trade['entry_time']))
            
            if pnl_values:
                plt.figure(figsize=(12, 6))
                plt.bar(range(len(pnl_values)), 
                        pnl_values, 
                        color=[('green' if x > 0 else 'red') for x in pnl_values])
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
                plt.title(f'PnL por Operación - {config_name}')
                plt.xlabel('# Operación')
                plt.ylabel('PnL (%)')
                plt.tight_layout()
                
                # Guardar gráfico
                plt.savefig(os.path.join(self.results_dir, f"trade_pnl_{config_name}.png"))
                plt.close()
                
                # 2. Gráfico de equity curve
                cumulative_returns = (1 + np.array(pnl_values) / 100).cumprod()
                plt.figure(figsize=(12, 6))
                plt.plot(cumulative_returns, color='blue')
                plt.title(f'Equity Curve - {config_name}')
                plt.xlabel('# Operación')
                plt.ylabel('Equity (1 = 100%)')
                plt.tight_layout()
                
                # Guardar equity curve
                plt.savefig(os.path.join(self.results_dir, f"equity_curve_{config_name}.png"))
                plt.close()
                
                # 3. Rendimiento por régimen de mercado
                if trades_by_regime:
                    plt.figure(figsize=(10, 8))
                    
                    # Para cada régimen, calcular retorno promedio
                    regimes = []
                    avg_returns = []
                    trade_counts = []
                    
                    for regime, regime_trades in trades_by_regime.items():
                        if not regime_trades:
                            continue
                            
                        pnls = []
                        for t in regime_trades:
                            if 'pnl_pct' in t:
                                pnls.append(t['pnl_pct'])
                            elif 'pnl' in t:
                                pnls.append(t['pnl'])
                        
                        if pnls:
                            regimes.append(regime)
                            avg_returns.append(np.mean(pnls))
                            trade_counts.append(len(pnls))
                    
                    if regimes:
                        # Gráfico de barras con retorno promedio por régimen
                        bar_colors = ['green' if x > 0 else 'red' for x in avg_returns]
                        
                        plt.subplot(2, 1, 1)
                        plt.bar(regimes, avg_returns, color=bar_colors)
                        plt.title(f'Retorno Promedio por Régimen - {config_name}')
                        plt.ylabel('PnL Promedio (%)')
                        plt.xticks(rotation=45)
                        
                        # Gráfico de número de trades por régimen
                        plt.subplot(2, 1, 2)
                        plt.bar(regimes, trade_counts, color='blue', alpha=0.7)
                        plt.title(f'Número de Trades por Régimen')
                        plt.ylabel('Número de Trades')
                        plt.xticks(rotation=45)
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(self.results_dir, f"regime_performance_{config_name}.png"))
                        plt.close()


def run_strategy_analysis(config_name=None, timeframe='1h', days=90):
    """Ejecutar análisis de estrategia desde línea de comandos"""
    # First try to create missing timeframe data if needed
    try:
        from utils.data_preprocessor import create_missing_data
        create_missing_data()
    except Exception as e:
        print(f"Warning: Could not preprocess data: {str(e)}")
    
    analyzer = StrategyAnalyzer()
    
    if config_name:
        # Analizar configuración específica
        analyzer.run_analysis(config_name, timeframe, days)
    else:
        # Analizar todas las configuraciones disponibles
        config_dir = '/home/panal/Documents/dashboard-trading/configs'
        configs = [f.split('.')[0] for f in os.listdir(config_dir) 
                  if f.endswith('.json') and not f.startswith('.')]
        
        for config in configs:
            print(f"\nAnalizando configuración: {config}")
            analyzer.run_analysis(config, timeframe, days)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analizar y optimizar estrategias de trading')
    parser.add_argument('--config', type=str, help='Configuración a analizar')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe para análisis')
    parser.add_argument('--days', type=int, default=90, help='Número de días para análisis')
    
    args = parser.parse_args()
    run_strategy_analysis(args.config, args.timeframe, args.days)
