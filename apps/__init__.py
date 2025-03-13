"""
Paquete de aplicaciones con parche para TA-LIB
"""
# Aplicar el parche para TA-LIB al importar el paquete apps
from utils.patch_imports import apply_patches
apply_patches()

# Empty file to make Python treat the directory as a package
