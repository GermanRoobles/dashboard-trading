#!/bin/bash

echo "=== EJECUTANDO PRUEBAS DE DASHBOARD ==="

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Verificar e instalar pytest primero
echo "Verificando pytest..."
python -c "import pytest" 2>/dev/null || pip install pytest pytest-cov pytest-asyncio

# Verificar e instalar otras dependencias
echo "Verificando dependencias..."
python -m pip install --upgrade pip
pip install -r requirements.txt 2>/dev/null

# Limpiar caché y archivos temporales
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Configurar PYTHONPATH y variables de entorno
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1

# Crear estructura de directorios y archivos necesarios
mkdir -p {logs,data/{cache,test},reports/hybrid_strategy,configs}

# Crear archivo de configuración de ejemplo
cat > configs/moderate_strategy.json << EOF
{
    "risk_profile": "moderate",
    "position_size": {"min": 0.03, "max": 0.06},
    "leverage": {"min": 2, "max": 5}
}
EOF

# Crear datos de prueba
mkdir -p data/test
python - <<EOF
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Create sample OHLCV data
dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='1h')
data = pd.DataFrame({
    'open': np.random.uniform(100, 110, len(dates)),
    'high': np.random.uniform(105, 115, len(dates)),
    'low': np.random.uniform(95, 105, len(dates)),
    'close': np.random.uniform(100, 110, len(dates)),
    'volume': np.random.uniform(1000, 2000, len(dates))
}, index=dates)

data.to_csv('data/test/sample_data.csv')
EOF

# Ejecutar pruebas con pytest y opciones mejoradas
echo "Ejecutando pruebas..."
PYTHONPATH="${PYTHONPATH}" python -X dev \
    -W ignore::DeprecationWarning \
    -c "import pytest; pytest.main(['tests/test_dashboard.py', '-v', '--no-cov', '--capture=no', '--tb=short'])" \
    2>&1 | tee logs/tests.log

# Verificar resultado
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Todas las pruebas pasaron correctamente"
else
    echo "❌ Las pruebas fallaron (código: $EXIT_CODE)"
    echo "Ver detalles completos en: logs/tests.log"
fi

echo "=== PRUEBAS COMPLETADAS ==="
