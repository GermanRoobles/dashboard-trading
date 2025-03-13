import json
import pandas as pd
import sys
import os
from datetime import datetime, timedelta

def validate_strategy_config(config_dict):
    """
    Valida una configuración de estrategia y ofrece recomendaciones
    
    Args:
        config_dict: Diccionario con configuración de la estrategia
    
    Returns:
        dict: Resultado con estado y recomendaciones
    """
    # Validaciones básicas
    issues = []
    warnings = []
    recommendations = []
    
    # 1. Verificar RSI
    if 'rsi' in config_dict:
        rsi_oversold = config_dict['rsi'].get('oversold')
        rsi_overbought = config_dict['rsi'].get('overbought')
        
        if rsi_oversold and rsi_overbought:
            # Verificar que rsi_oversold < rsi_overbought
            if rsi_oversold >= rsi_overbought:
                issues.append("RSI oversold debe ser menor que overbought")
            
            # Verificar rangos recomendados
            if rsi_oversold < 30:
                warnings.append(f"RSI oversold ({rsi_oversold}) es extremadamente bajo")
            if rsi_overbought > 70:
                warnings.append(f"RSI overbought ({rsi_overbought}) es extremadamente alto")
                
            # Verificar diferencia entre valores
            diff = rsi_overbought - rsi_oversold
            if diff < 15:
                warnings.append(f"Diferencia entre RSI overbought y oversold ({diff}) es pequeña")
            elif diff > 35:
                warnings.append(f"Diferencia entre RSI overbought y oversold ({diff}) es grande")
    else:
        issues.append("Falta configuración de RSI")
    
    # 2. Verificar EMAs
    if 'ema' in config_dict:
        ema_short = config_dict['ema'].get('short')
        ema_long = config_dict['ema'].get('long')
        
        if ema_short and ema_long:
            # Verificar que ema_short < ema_long
            if ema_short >= ema_long:
                issues.append("EMA short debe ser menor que EMA long")
                
            # Verificar diferencia entre valores
            diff = ema_long - ema_short
            if diff < 8:
                warnings.append(f"Diferencia entre EMA long y short ({diff}) es pequeña")
            elif diff > 30:
                warnings.append(f"Diferencia entre EMA long y short ({diff}) es grande")
    else:
        issues.append("Falta configuración de EMAs")
    
    # 3. Verificar holding_time
    holding_time = config_dict.get('holding_time')
    if holding_time:
        if holding_time < 2:
            issues.append(f"Holding time ({holding_time}) es demasiado corto")
        elif holding_time > 12:
            warnings.append(f"Holding time ({holding_time}) es bastante largo")
    else:
        issues.append("Falta holding_time")
    
    # 4. Verificar filtros
    trend_filter = config_dict.get('trend_filter')
    volume_filter = config_dict.get('volume_filter')
    
    if trend_filter is None:
        warnings.append("No se especificó filtro de tendencia")
    
    if volume_filter is None:
        recommendations.append("El filtro de volumen puede mejorar la calidad de señales")
    
    # 5. Evaluar perfil de riesgo
    risk_profile = config_dict.get('risk_profile', 'moderate')
    
    if risk_profile == 'conservative':
        recommendations.append("Perfil conservador: considere EMAs más lentas (EMA long > 25)")
    elif risk_profile == 'aggressive':
        recommendations.append("Perfil agresivo: considere ajustar stops para limitar drawdown")
    
    # Preparar resultado
    result = {
        'status': 'invalid' if issues else ('warning' if warnings else 'valid'),
        'issues': issues,
        'warnings': warnings,
        'recommendations': recommendations
    }
    
    return result

def load_and_validate_config(config_path):
    """Carga y valida un archivo de configuración"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        result = validate_strategy_config(config)
        
        # Imprimir resultados bonitos
        print(f"\n==== VALIDACIÓN DE CONFIGURACIÓN ====")
        print(f"Archivo: {config_path}")
        
        if result['status'] == 'valid':
            print("\n✅ CONFIGURACIÓN VÁLIDA")
        elif result['status'] == 'warning':
            print("\n⚠️  CONFIGURACIÓN CON ADVERTENCIAS")
        else:
            print("\n❌ CONFIGURACIÓN INVÁLIDA")
        
        if result['issues']:
            print("\nProblemas que deben corregirse:")
            for issue in result['issues']:
                print(f"  - {issue}")
        
        if result['warnings']:
            print("\nAdvertencias a considerar:")
            for warning in result['warnings']:
                print(f"  - {warning}")
        
        if result['recommendations']:
            print("\nRecomendaciones:")
            for rec in result['recommendations']:
                print(f"  - {rec}")
        
        print("\n======================================")
        
        return result
    except Exception as e:
        print(f"Error al cargar o validar configuración: {str(e)}")
        return None

if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
        load_and_validate_config(config_path)
    else:
        print("Uso: python config_validator.py <ruta_a_config.json>")
        
        # Ejemplo de configuración
        example = {
            'rsi': {'oversold': 40, 'overbought': 60},
            'ema': {'short': 9, 'long': 21},
            'holding_time': 4,
            'trend_filter': True,
            'volume_filter': True,
            'risk_profile': 'moderate'
        }
        
        print("\nValidando configuración de ejemplo...")
        validate_strategy_config(example)
