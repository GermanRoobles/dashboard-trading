import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy

def walk_forward_test(data, strategy_params, window_size=30, step_size=5):
    """
    Realiza una prueba walk-forward para verificar robustez
    
    Args:
        data: DataFrame con datos históricos
        strategy_params: Parámetros de la estrategia
        window_size: Tamaño de ventana en días
        step_size: Paso entre ventanas en días
    """
    # Crear estrategia
    if strategy_params.get('strategy') == 'enhanced':
        strategy = EnhancedStrategy()
    else:
        strategy = OptimizedStrategy()
    
    # Ensure RSI window parameter exists
    if 'rsi' in strategy_params and 'window' not in strategy_params['rsi']:
        strategy_params['rsi']['window'] = 14  # Use default value
    
    # Update strategy params with provided ones
    strategy.params.update(strategy_params)
    
    # Obtener fecha inicial y final
    start_date = data.index.min()
    end_date = data.index.max()
    
    # Crear ventanas
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        window_end = current_start + timedelta(days=window_size)
        if window_end > end_date:
            window_end = end_date
        
        windows.append((current_start, window_end))
        current_start += timedelta(days=step_size)
        
        if len(windows) >= 20:  # Limitar a 20 ventanas máximo
            break
    
    # Ejecutar estrategia en cada ventana
    results = []
    for i, (start, end) in enumerate(windows):
        window_data = data[(data.index >= start) & (data.index <= end)]
        if len(window_data) < 20:  # Ventana muy pequeña
            continue
            
        # Generar señales
        signals = strategy.generate_signals(window_data)
        
        # Calcular estadísticas
        long_signals = (signals == 1).sum()
        short_signals = (signals == -1).sum()
        signal_ratio = (long_signals + short_signals) / len(signals) * 100
        
        results.append({
            'window': i+1,
            'start': start,
            'end': end,
            'days': (end - start).days,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'total_signals': long_signals + short_signals,
            'signal_ratio': signal_ratio
        })
    
    # Crear DataFrame de resultados
    results_df = pd.DataFrame(results)
    
    # Visualizar resultados
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Gráfico 1: Señales por ventana
    ax1.bar(results_df['window'], results_df['long_signals'], color='green', label='Long')
    ax1.bar(results_df['window'], results_df['short_signals'], color='red', bottom=results_df['long_signals'], label='Short')
    ax1.set_xlabel('Ventana')
    ax1.set_ylabel('Número de señales')
    ax1.set_title('Señales por ventana')
    ax1.legend()
    
    # Gráfico 2: Ratio de señales
    ax2.plot(results_df['window'], results_df['signal_ratio'], marker='o')
    ax2.set_xlabel('Ventana')
    ax2.set_ylabel('Ratio de señales (%)')
    ax2.set_title('Ratio de señales por ventana')
    
    plt.tight_layout()
    
    # Guardar gráfico
    fig.savefig('walk_forward_results.png')
    print(f"Gráfico guardado en walk_forward_results.png")
    
    # Estadísticas de estabilidad
    std_signals = results_df['total_signals'].std() / results_df['total_signals'].mean()
    print(f"\nEstabilidad de señales (menor es mejor): {std_signals:.2f}")
    print(f"- Valor < 0.3: Excelente estabilidad")
    print(f"- Valor < 0.5: Buena estabilidad")
    print(f"- Valor > 0.7: Estrategia potencialmente inestable")
    
    # Devolver DataFrame con resultados
    return results_df

if __name__ == "__main__":
    # Ejemplo de uso
    from utils.data_cache import DataCache
    
    cache = DataCache()
    # Obtener datos de 180 días
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    data = cache.get_cached_data(
        symbol='BTC/USDT',
        timeframe='1h',
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d')
    )
    
    if data is not None:
        params = {
            'rsi': {'oversold': 40, 'overbought': 60},
            'ema': {'short': 9, 'long': 21}
        }
        results = walk_forward_test(data, params)
        print(results)
    else:
        print("No se encontraron datos en caché. Ejecute primero un backtest.")
