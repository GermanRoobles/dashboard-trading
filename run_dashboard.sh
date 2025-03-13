#!/bin/bash

echo "=== INICIANDO DASHBOARD DE TRADING ==="

# Crear directorios necesarios
mkdir -p logs
mkdir -p data/cache
mkdir -p reports

# Activar entorno virtual si existe
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Entorno virtual activado"
else
    echo "⚠️ Entorno virtual no encontrado. Ejecuta setup.sh primero"
    exit 1
fi

# Añadir directorio actual al PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Verificar dependencias críticas
python -c "import dash" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Instalando dependencias faltantes..."
    pip install dash plotly pandas numpy
fi

# Obtener IP local para acceso en red
LOCAL_IP=$(hostname -I | cut -d' ' -f1)
PORT=8050

# Generar código QR para acceso móvil
QR_PATH="reports/dashboard_access_qr.png"
mkdir -p reports
if ! command -v qrencode &> /dev/null; then
    sudo apt-get install -y qrencode
fi
qrencode -o "$QR_PATH" "http://$LOCAL_IP:$PORT"

# Definir directorio de caché
export CACHE_DIR="/home/panal/Documents/dashboard-trading/data/cache"
echo "Using cache directory: $CACHE_DIR"

echo "Iniciando dashboard de trading..."

# Parchar TA-Lib si es necesario
python -c "import talib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "TA-LIB patched with alternative implementation"
fi

echo "✅ Acceso Dashboard - IP: $LOCAL_IP:$PORT"
echo "📱 Escanea el código QR para acceder rápidamente desde tu móvil"
echo "   QR guardado en: $(pwd)/$QR_PATH"

echo -e "\n=== DASHBOARD INICIADO ==="
echo "Accede desde tu ordenador: http://localhost:$PORT"
echo "Accede desde dispositivos en la misma red: http://$LOCAL_IP:$PORT"
echo "Código QR para acceso rápido: $(pwd)/$QR_PATH"
echo "=============================="

# Limpiar caché antes de iniciar
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

# Limpiar archivos de caché antes de iniciar
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null
find . -type f -name "*.pyc" -delete

# Verificar directorios necesarios
mkdir -p {data/cache,reports/hybrid_strategy,configs,assets}

# Asegurar que existe al menos una configuración
if [ ! -f "configs/hybrid_strategy.json" ]; then
    echo "{\"risk_profile\":\"moderate\"}" > configs/hybrid_strategy.json
fi

# Suprimir warnings de deprecación
export PYTHONWARNINGS="ignore::DeprecationWarning"

# Configurar variables de entorno para Dash
export DASH_DEBUG=false
export DASH_SILENCE_ROUTES_LOGGING=true

# Asegurarnos de que usamos el dashboard correcto
PYTHONPATH="${PYTHONPATH}:$(pwd)" python3 -X dev apps/run_interface.py 2>&1 | tee logs/dashboard.log

# Manejar señal de interrupción
trap 'echo -e "\n✋ Dashboard detenido"; exit 0' INT

exit 0
