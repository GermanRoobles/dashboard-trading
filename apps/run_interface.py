import sys
import os

# Configurar paths
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

def main():
    print("\nINICIALIZANDO DASHBOARD")
    print(f"Root dir: {root_dir}")

    try:
        # Cambiar aquí para asegurarnos de que importamos el correcto
        from apps.performance_dashboard import create_dashboard
        
        print("\nCreando aplicación...")
        app = create_dashboard(port=8050)
        
        print("\nIniciando servidor...")
        app.run_server(
            debug=True,
            host='0.0.0.0',
            port=8050,
            dev_tools_hot_reload=False,  # Deshabilitar hot reload
            dev_tools_serve_dev_bundles=False,  # Reducir warnings
            use_reloader=False  # Añadido para evitar problemas de recargas
        )
    except Exception as e:
        print(f"\nERROR AL INICIAR DASHBOARD: {str(e)}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app = create_dashboard(port=port)
    app.run_server(
        host='0.0.0.0',
        port=port,
        debug=False
    )
