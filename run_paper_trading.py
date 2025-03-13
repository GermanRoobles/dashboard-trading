#!/usr/bin/env python
import os
import argparse
import json
from datetime import datetime
import time
from utils.paper_trader import PaperTrader

def main():
    """Ejecutar paper trading para verificar estrategia en tiempo real"""
    parser = argparse.ArgumentParser(description='Herramienta de paper trading')
    parser.add_argument('--config', type=str, default='production_ready', 
                       help='Configuración a usar para paper trading')
    parser.add_argument('--timeframe', type=str, default='15m', 
                       help='Timeframe para operaciones')
    parser.add_argument('--symbol', type=str, default='BTC/USDT', 
                       help='Par a operar')
    parser.add_argument('--hours', type=int, default=24, 
                       help='Horas de ejecución')
    
    args = parser.parse_args()
    
    # Verificar existencia de configuración
    config_path = f"/home/panal/Documents/dashboard-trading/configs/{args.config}.json"
    if not os.path.exists(config_path):
        print(f"❌ Configuración no encontrada: {args.config}")
        print("Configuraciones disponibles:")
        config_dir = '/home/panal/Documents/dashboard-trading/configs'
        for file in os.listdir(config_dir):
            if file.endswith('.json') and not file.startswith('.'):
                print(f"  - {file.split('.')[0]}")
        return
    
    # Cargar configuración
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Crear directorio de logs si no existe
    log_dir = '/home/panal/Documents/dashboard-trading/reports/paper_trading'
    os.makedirs(log_dir, exist_ok=True)
    
    print(f"=== INICIANDO PAPER TRADING ===")
    print(f"Configuración: {args.config}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Símbolo: {args.symbol}")
    print(f"Duración: {args.hours} horas")
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==============================")
    
    # Iniciar paper trading
    trader = PaperTrader(
        config, 
        timeframe=args.timeframe,
        symbol=args.symbol
    )
    
    try:
        trader.run(hours=args.hours)
    except KeyboardInterrupt:
        print("\n⚠️ Paper trading interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
    
    # Guardar resultados
    results_file = os.path.join(log_dir, f"results_{args.config}_{args.timeframe}_{datetime.now().strftime('%Y%m%d_%H%M')}.json")
    
    with open(results_file, 'w') as f:
        json.dump({
            "config": args.config,
            "timeframe": args.timeframe,
            "symbol": args.symbol,
            "duration_hours": args.hours,
            "start_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "trades": [
                {k: str(v) if isinstance(v, (datetime, pd.Timestamp)) else v 
                 for k, v in trade.items()}
                for trade in trader.trades
            ]
        }, f, indent=2)
    
    print(f"\nPaper trading completado. Resultados guardados en: {results_file}")
    print(f"Total de señales detectadas: {len(trader.trades)}")
    
    # Análisis básico
    if trader.trades:
        long_signals = sum(1 for t in trader.trades if t['signal'] == 'BUY')
        short_signals = sum(1 for t in trader.trades if t['signal'] == 'SELL')
        
        print(f"Señales LONG: {long_signals}")
        print(f"Señales SHORT: {short_signals}")
        print(f"Ratio LONG/SHORT: {long_signals/short_signals if short_signals else 'N/A'}")

if __name__ == "__main__":
    import pandas as pd  # Importar pandas para el manejo de fechas en la serialización
    main()
