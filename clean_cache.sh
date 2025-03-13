#!/bin/bash

echo "Limpiando archivos de caché Python..."

# Eliminar archivos .pyc
find . -type f -name "*.pyc" -delete

# Eliminar carpetas __pycache__
find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null

echo "Limpieza completada"
