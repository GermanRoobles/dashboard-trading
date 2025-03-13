#!/bin/bash
# Script de instalación segura para el dashboard

echo "=== Configurando entorno para dashboard de trading ==="

# 1. Crear entorno virtual
if [ ! -d "venv" ]; then
    echo "Creando entorno virtual..."
    python3 -m venv venv
    source venv/bin/activate
    
    echo "Actualizando pip..."
    pip install --upgrade pip
else
    echo "Usando entorno virtual existente..."
    source venv/bin/activate
fi

# 2. Instalar dependencias con límite de memoria
echo "Instalando dependencias (con límite de memoria)..."
pip install numpy --no-binary numpy
pip install pandas
pip install scikit-learn
pip install optuna
pip install ta
pip install matplotlib
pip install seaborn
pip install joblib

# 3. Crear directorios necesarios
mkdir -p models reports/comparisons reports/enhanced_tests

echo "=== Instalación completa ==="
echo "Ejecuta ./run_dashboard.sh para iniciar el dashboard"
