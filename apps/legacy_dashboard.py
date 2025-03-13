import dash
import dash_bootstrap_components as dbc
from dash import html, dcc, callback, Input, Output
import plotly.graph_objects as go
import pandas as pd
import os
from datetime import datetime, timedelta
import qrcode
from PIL import Image
import socket
import subprocess
from backtest.run_fixed_backtest import BacktestFixedStrategy
from strategies.fixed_strategy import FixedStrategy
from strategies.optimized_strategy import OptimizedStrategy 
from strategies.enhanced_strategy import EnhancedStrategy

# Patch for TA-Lib if not installed
try:
    import talib
except ImportError:
    print("TA-LIB patched with alternative implementation")
    
# Helper functions
def get_local_ip():
    """Get local IP address to access dashboard from other devices"""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except:
        return "127.0.0.1"

def create_qr_code(url, output_path):
    """Create QR code for easy dashboard access"""
    try:
        # Use Python's qrcode library directly instead of GIMP
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=10,
            border=4,
        )
        qr.add_data(url)
        qr.make(fit=True)
        
        # Create an image from the QR Code and save
        img = qr.make_image(fill_color="black", back_color="white")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        return True
    except Exception as e:
        print(f"Error creating QR code: {str(e)}")
        return False

def load_backtest_results():
    """Load available backtest results"""
    results = []
    reports_dir = '/home/panal/Documents/dashboard-trading/reports'
    
    # Look for batch results
    batch_files = []
    for filename in os.listdir(reports_dir):
        if filename.startswith('batch_backtest_') and filename.endswith('.csv'):
            batch_files.append(os.path.join(reports_dir, filename))
    
    # Sort by modification time (newest first)
    batch_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    # Load the newest file
    if batch_files:
        try:
            df = pd.read_csv(batch_files[0])
            return df
        except Exception as e:
            print(f"Error loading batch results: {str(e)}")
    
    return pd.DataFrame()

def load_available_configs():
    """Load available strategy configurations"""
    configs = []
    config_dir = '/home/panal/Documents/dashboard-trading/configs'
    
    if os.path.exists(config_dir):
        for filename in os.listdir(config_dir):
            if filename.endswith('.json') and not filename.startswith('.'):
                configs.append(filename.split('.')[0])
    
    return configs

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
server = app.server
app.title = "Trading Dashboard"

# Get local IP for access
local_ip = get_local_ip()
dashboard_url = f"http://{local_ip}:8050"

# Create QR code for mobile access
qr_path = '/home/panal/Documents/dashboard-trading/reports/dashboard_access_qr.png'
qr_created = create_qr_code(dashboard_url, qr_path)

if qr_created:
    print(f"✅ Acceso Dashboard - IP: {local_ip}:8050")
    print(f"📱 Escanea el código QR para acceder rápidamente desde tu móvil")
    print(f"   QR guardado en: {qr_path}")
    print()
    print("=== DASHBOARD INICIADO ===")
    print(f"Accede desde tu ordenador: http://localhost:8050")
    print(f"Accede desde dispositivos en la misma red: {dashboard_url}")
    print(f"Código QR para acceso rápido: {qr_path}")
    print("=" * 30)

# Load initial data
backtest_results = load_backtest_results()
available_configs = load_available_configs()

# Create the layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Trading Dashboard", className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Estrategias Disponibles"),
                dbc.CardBody([
                    html.P(f"Configuraciones encontradas: {len(available_configs)}"),
                    dbc.ListGroup([
                        dbc.ListGroupItem(config) for config in available_configs
                    ])
                ])
            ], className="mb-4")
        ], width=4),
        
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Resultados de Backtests"),
                dbc.CardBody([
                    html.Div([
                        html.P(f"Total de backtests: {len(backtest_results)}"),
                        html.Div(id="backtest-summary")
                    ])
                ])
            ])
        ], width=8)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparación de Retornos"),
                dbc.CardBody([
                    dcc.Graph(id="returns-chart")
                ])
            ])
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H4("Opciones", className="mt-4"),
                dbc.Button("Ejecutar Nuevo Backtest", id="run-backtest-btn", color="primary", className="me-2"),
                dbc.Button("Ver Análisis", id="view-analysis-btn", color="info", className="me-2"),
                dbc.Button("Actualizar Datos", id="update-data-btn", color="success")
            ], className="d-flex flex-column")
        ])
    ]),

    # Añadir nuevo componente para comparación de estrategias
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Comparación de Estrategias"),
                dbc.CardBody([
                    dbc.Button(
                        "Comparar Todas las Estrategias",
                        id="compare-strategies-btn",
                        color="primary",
                        className="mb-3"
                    ),
                    html.Div(id="strategy-comparison-results"),
                    dcc.Graph(id="strategy-comparison-chart")
                ])
            ])
        ])
    ]),
], fluid=True)

@app.callback(
    [Output("backtest-summary", "children"),
     Output("returns-chart", "figure")],
    [Input("update-data-btn", "n_clicks")]
)
def update_dashboard(n_clicks):
    # Load latest data
    results = load_backtest_results()
    
    # Create summary
    if not results.empty:
        # Get best and worst strategies
        best_config = results.loc[results['return_total'].idxmax()]
        worst_config = results.loc[results['return_total'].idxmin()]
        
        summary = [
            html.P(f"Mejor estrategia: {best_config['config']} ({best_config['return_total']:.2f}%)"),
            html.P(f"Peor estrategia: {worst_config['config']} ({worst_config['return_total']:.2f}%)")
        ]
        
        # Create returns comparison chart
        fig = go.Figure()
        for config in results['config'].unique():
            config_data = results[results['config'] == config]
            fig.add_trace(go.Bar(
                x=config_data['timeframe'] + " " + config_data['days'].astype(str) + "d",
                y=config_data['return_total'],
                name=config
            ))
        
        fig.update_layout(
            title="Comparación de Retornos por Estrategia",
            xaxis_title="Timeframe y Período",
            yaxis_title="Retorno (%)",
            template="plotly_dark"
        )
    else:
        summary = [html.P("No se encontraron resultados de backtest")]
        fig = go.Figure()
    
    return summary, fig

# Add callback for run-backtest-btn
@app.callback(
    Output("run-backtest-btn", "disabled"),
    [Input("run-backtest-btn", "n_clicks")]
)
def run_backtest(n_clicks):
    if not n_clicks:
        return False
        
    try:
        subprocess.Popen(["python", "run_backtests.py", "--batch"])
        return True  # Disable button during execution
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        return False

@app.callback(
    [Output("strategy-comparison-results", "children"),
     Output("strategy-comparison-chart", "figure")],
    [Input("compare-strategies-btn", "n_clicks")],
    [State("date-picker-range", "start_date"),
     State("date-picker-range", "end_date"),
     State("timeframe-selector", "value")]
)
def compare_all_strategies(n_clicks, start_date, end_date, timeframe):
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    try:
        # Cargar datos
        cache = DataCache()
        data = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date
        )

        if data is None:
            return html.Div("No hay datos disponibles"), {}

        # Configurar estrategias
        config = {'timeframe': timeframe, 'initial_balance': 1000}
        backtest = BacktestFixedStrategy(config)
        
        # Ejecutar comparación
        comparison_df = backtest.compare_strategies(data)
        
        # Crear tabla de resultados
        table = dbc.Table.from_dataframe(
            comparison_df.reset_index(),
            striped=True,
            bordered=True,
            hover=True
        )
        
        # Crear gráfico de comparación
        fig = go.Figure()
        for strategy in comparison_df.index:
            fig.add_trace(go.Bar(
                name=strategy,
                x=['Return', 'Win Rate', 'Max Drawdown'],
                y=[
                    comparison_df.loc[strategy, 'Total Return (%)'],
                    comparison_df.loc[strategy, 'Win Rate (%)'],
                    comparison_df.loc[strategy, 'Max Drawdown (%)']
                ]
            ))
            
        fig.update_layout(
            title="Comparación de Métricas por Estrategia",
            barmode='group',
            template='plotly_dark'
        )
        
        return table, fig
        
    except Exception as e:
        print(f"Error en comparación de estrategias: {str(e)}")
        return html.Div(f"Error: {str(e)}"), {}

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", debug=True)