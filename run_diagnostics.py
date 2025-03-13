#!/usr/bin/env python
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from utils.diagnostic_tool import DiagnosticTool, run_diagnostics
from utils.batch_tester import BatchTester, run_batch_tests
from utils.walk_forward_test import walk_forward_test
from utils.data_cache import DataCache
from datetime import datetime, timedelta

def main():
    """Ejecutar diagnóstico completo del sistema de trading"""
    parser = argparse.ArgumentParser(description='Herramienta de diagnóstico para sistema de trading')
    parser.add_argument('--mode', choices=['quick', 'full', 'consistency', 'walk_forward', 'batch'],
                       default='quick', help='Modo de ejecución')
    parser.add_argument('--config', type=str, help='Configuración específica para analizar')
    parser.add_argument('--timeframe', type=str, default='1h', help='Timeframe para análisis')
    parser.add_argument('--days', type=int, default=30, help='Número de días para análisis')
    
    args = parser.parse_args()
    
    # Verificar existencia de directorios
    reports_dir = '/home/panal/Documents/dashboard-trading/reports/diagnostics'
    os.makedirs(reports_dir, exist_ok=True)
    
    # Crear herramientas
    tool = DiagnosticTool()
    tester = BatchTester()
    
    print("==" * 30)
    print(f"DIAGNÓSTICO DE SISTEMA DE TRADING - {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("==" * 30)
    
    # Modo rápido - análisis básico
    if args.mode == 'quick':
        print("\n[MODO RÁPIDO] Ejecutando análisis básico...")
        
        # Obtener todas las configuraciones
        config_dir = '/home/panal/Documents/dashboard-trading/configs'
        configs = [f.split('.')[0] for f in os.listdir(config_dir) 
                 if f.endswith('.json') and not f.startswith('.')]
        
        # Si se especificó una configuración, usar solo esa
        if args.config:
            if args.config in configs:
                configs = [args.config]
            else:
                print(f"⚠️ Configuración '{args.config}' no encontrada. Usando todas las configuraciones.")
        
        print(f"Analizando {len(configs)} configuraciones en {args.timeframe}, {args.days} días...")
        comparison = tool.compare_configurations(configs, args.timeframe, args.days)
        
        print("\nResumen de resultados:")
        print(comparison[['Configuración', 'Retorno (%)', 'Win Rate (%)', 'Profit Factor', 
                         'Trades/Mes', 'Max DD (%)', 'Total Trades']])
        
        # Identificar la mejor configuración
        best_config = comparison.loc[comparison['Retorno (%)'].idxmax()]
        print(f"\n✅ Mejor configuración: {best_config['Configuración']}")
        print(f"   Retorno: {best_config['Retorno (%)']}%")
        print(f"   Win Rate: {best_config['Win Rate (%)']}%")
        print(f"   Profit Factor: {best_config['Profit Factor']}")
        print(f"   Trades/Mes: {best_config['Trades/Mes']}")
        
        # Identificar inconsistencias
        inconsistencies = []
        for _, row in comparison.iterrows():
            config = row['Configuración']
            pf = row['Profit Factor']
            ret = row['Retorno (%)']
            
            if (pf > 1 and ret < 0) or (pf < 1 and ret > 0):
                inconsistencies.append(config)
        
        if inconsistencies:
            print("\n⚠️ Posibles inconsistencias detectadas en:")
            for config in inconsistencies:
                print(f"   - {config}")
            print("   Ejecute el modo 'consistency' para análisis detallado")
        
    # Modo completo - todos los análisis
    elif args.mode == 'full':
        print("\n[MODO COMPLETO] Ejecutando análisis exhaustivo...")
        
        # 1. Análisis de consistencia
        results = run_diagnostics()
        
        # 2. Pruebas en lote
        configs = [
            'ultra_conservative_optimized',
            'conservative',
            'moderate',
            'aggressive',
            'aggressive_balanced',
            'production_ready'
        ]
        
        timeframes = ['15m', '1h', '4h']
        periods = [30, 90, 180]
        
        batch_results = run_batch_tests(configs, timeframes, periods)
        
        # 3. Robustness test para la mejor configuración
        best_config = batch_results.loc[batch_results['return_total'].idxmax()]['config']
        robustness = tool.run_robustness_test(
            best_config,
            timeframes=timeframes,
            periods=periods
        )
        
        print(f"\nAnálisis completo finalizado. Resultados guardados en {reports_dir}")
        print(f"Mejor configuración: {best_config}")
    
    # Análisis de consistencia
    elif args.mode == 'consistency':
        print("\n[ANÁLISIS DE CONSISTENCIA] Verificando métricas...")
        
        if args.config:
            # Analizar una configuración específica
            print(f"Analizando configuración: {args.config}")
            result = tool.run_validation_test(args.config, args.timeframe, args.days)
            
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Configuración: {args.config}")
                print(f"Retorno: {result['results']['return_total']:.2f}%")
                print(f"Win Rate: {result['results']['win_rate']:.2f}%")
                print(f"Profit Factor: {result['results']['profit_factor']:.2f}")
                print(f"Trades: {result['total_trades']}")
                
                if result['inconsistencies']:
                    print("\n⚠️ Inconsistencias detectadas:")
                    for issue in result['inconsistencies']:
                        print(f"  - {issue['message']}")
                        print(f"    Causa posible: {issue['possible_cause']}")
                else:
                    print("\n✅ No se detectaron inconsistencias")
        else:
            # Analizar todas las configuraciones
            run_diagnostics()
    
    # Walk-forward testing
    elif args.mode == 'walk_forward':
        print("\n[WALK-FORWARD TESTING] Analizando estabilidad de estrategia...")
        
        # Obtener datos
        cache = DataCache()
        end_date = datetime.now()
        start_date = end_date - timedelta(days=180)  # Usar 180 días para walk-forward test
        
        data = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=args.timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if data is None:
            print("❌ No se encontraron datos en caché. Ejecute primero un backtest.")
            return
        
        # Cargar configuración
        import json
        config_file = args.config if args.config else 'moderate'
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{config_file}.json"
        
        if not os.path.exists(config_path):
            print(f"❌ Configuración no encontrada: {config_path}")
            return
            
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Ejecutar walk-forward test
        print(f"Ejecutando walk-forward test para {config_file} en {args.timeframe}...")
        results = walk_forward_test(data, config, window_size=14, step_size=7)
        
        print(f"\nAnálisis walk-forward completado. Gráfico guardado como walk_forward_results.png")
    
    # Pruebas en lote
    elif args.mode == 'batch':
        print("\n[MODO BATCH] Ejecutando tests en lote...")
        
        # Determinar configuraciones a probar
        configs = None
        if args.config:
            configs = [args.config]
            
        # Determinar timeframes
        timeframes = [args.timeframe]
        
        # Determinar períodos
        periods = [args.days]
            
        # Ejecutar batch
        run_batch_tests(configs, timeframes, periods)
    
    print("\nDiagnóstico completado. Consulta la carpeta 'reports/diagnostics' para ver resultados.")

if __name__ == "__main__":
    main()
