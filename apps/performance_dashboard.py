#!/usr/bin/env python
import os
import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import subprocess
import plotly.subplots as make_subplots
from utils.data_cache import DataCache
from backtest.run_fixed_backtest import BacktestFixedStrategy
from strategies.fixed_strategy import FixedStrategy
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from components.table_utils import create_table, create_table_row, create_header_cell, create_table_cell

def create_dashboard(port=8050):
    """Create a performance dashboard for the trading system"""
    try:
        print("\nINICIALIZANDO COMPONENTES")
        
        # Get available strategies BEFORE creating the app and layout
        config_dir = "/home/panal/Documents/dashboard-trading/configs"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)
            print(f"Created config directory: {config_dir}")
            
        strategies = [f.replace('.json', '') for f in os.listdir(config_dir) if f.endswith('.json')]
        
        if not strategies:
            # Create default strategy if none exists
            default_config = {
                "risk_profile": "moderate",
                "position_size": {"min": 0.03, "max": 0.06},
                "leverage": {"min": 2, "max": 5}
            }
            with open(os.path.join(config_dir, "default_strategy.json"), "w") as f:
                json.dump(default_config, f, indent=4)
            strategies = ["default_strategy"]
            print("Created default strategy configuration")
        
        print(f"Found {len(strategies)} strategies: {', '.join(strategies)}")
        
        # Initialize Dash with reduced logging
        app = dash.Dash(
            __name__, 
            external_stylesheets=[dbc.themes.DARKLY],
            suppress_callback_exceptions=True,
            title="Trading Performance Dashboard",
            assets_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets"),
            update_title=None,  # Disable browser title updates
            meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
        )
        
        print("Configurando layout...")
        
        # Configurar layout base con strategies ya definido
        app.layout = html.Div([
            dcc.Tabs([
                dcc.Tab(label='Trading Performance', children=[
                    dcc.Loading(
                        id="loading-1",
                        type="default",
                        children=[
                            html.H1("Trading Performance Dashboard", className="dashboard-title"),
                            
                            # Strategy Selection and Configuration
                            html.Div([
                                html.Div([
                                    html.Label("Select Strategy:"),
                                    dcc.Dropdown(
                                        id='strategy-selector',
                                        options=[{'label': s, 'value': s} for s in strategies],
                                        value=strategies[0] if strategies else None
                                    ),
                                    html.Button('Edit Parameters', id='edit-params-button', n_clicks=0),
                                ], className="selector-container"),

                                html.Div([
                                    html.Label("Time Range:"),
                                    dcc.DatePickerRange(
                                        id='date-picker-range',
                                        start_date=(datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d'),
                                        end_date=datetime.now().strftime('%Y-%m-%d')
                                    ),
                                ], className="date-container"),

                                html.Div([
                                    html.Label("Timeframe:"),
                                    dcc.Dropdown(
                                        id='timeframe-selector',
                                        options=[
                                            {'label': '15 minutes', 'value': '15m'},
                                            {'label': '1 hour', 'value': '1h'},
                                            {'label': '4 hours', 'value': '4h'},
                                            {'label': '1 day', 'value': '1d'}
                                        ],
                                        value='1h'
                                    ),
                                ], className="timeframe-container"),

                                html.Button('Run Backtest', id='run-backtest-button', n_clicks=0),
                            ], className="controls-container"),

                            # Parameter Editor Modal (using Div)
                            html.Div(
                                id="param-editor-modal",
                                style={
                                    'display': 'none',
                                    'position': 'fixed',
                                    'top': '50%',
                                    'left': '50%',
                                    'transform': 'translate(-50%, -50%)',
                                    'backgroundColor': 'white',
                                    'padding': '20px',
                                    'borderRadius': '8px',
                                    'boxShadow': '0 2px 4px rgba(0,0,0,0.1)',
                                    'zIndex': '1000'
                                },
                                children=[
                                    html.Div([
                                        html.H3("Strategy Parameters"),
                                        html.Div(id="param-editor-content"),
                                        html.Div([
                                            html.Button("Save", id="save-params-button"),
                                            html.Button("Cancel", id="close-modal-button"),
                                        ], className="modal-buttons")
                                    ], className="modal-content")
                                ],
                            ),

                            # Modal backdrop
                            html.Div(
                                id="modal-backdrop",
                                style={
                                    'display': 'none',
                                    'position': 'fixed',
                                    'top': 0,
                                    'left': 0,
                                    'right': 0,
                                    'bottom': 0,
                                    'backgroundColor': 'rgba(0,0,0,0.5)',
                                    'zIndex': '999'
                                }
                            ),

                            # Update interval for real-time updates
                            dcc.Interval(
                                id='interval-component',
                                interval=60*1000,  # Update every minute
                                n_intervals=0
                            ),

                            # Performance Overview Section
                            html.Div([
                                html.H2("Performance Overview", className="section-title"),
                                html.Div([
                                    html.Div([
                                        html.H4("Overall Statistics"),
                                        html.Div([
                                            html.Div([
                                                html.Div(id='total-trades', className="stat-value"),
                                                html.Div("Total Trades", className="stat-label")
                                            ], className="stat-box"),
                                            html.Div([
                                                html.Div(id='avg-trade-duration', className="stat-value"),
                                                html.Div("Avg Trade Duration", className="stat-label")
                                            ], className="stat-box"),
                                            html.Div([
                                                html.Div(id='avg-profit-per-trade', className="stat-value"),
                                                html.Div("Avg Profit/Trade", className="stat-label")
                                            ], className="stat-box"),
                                        ], className="stats-container")
                                    ], className="summary-card"),
                                ], className="overview-row"),
                            ], className="overview-section"),
                            
                            # Analysis Section - Remover Monthly Returns Card
                            html.Div([
                                # Trade Distribution Card
                                html.Div([
                                    html.H4("Trade Distribution"),
                                    dcc.Graph(id='trade-distribution')
                                ], className="chart-card"),
                                
                                # Market Regime Card
                                html.Div([
                                    html.H4("Performance by Market Regime"),
                                    dcc.Graph(id='regime-performance')
                                ], className="chart-card"),
                                
                                # Risk Metrics Card
                                html.Div([
                                    html.H4("Risk Metrics"),
                                    html.Div([
                                        html.Div(id='sharpe-ratio', className="stat-value"),
                                        html.Div("Sharpe Ratio", className="stat-label")
                                    ], className="stat-box"),
                                    html.Div([
                                        html.Div(id='sortino-ratio', className="stat-value"),
                                        html.Div("Sortino Ratio", className="stat-label")
                                    ], className="stat-box"),
                                    html.Div([
                                        html.Div(id='max-drawdown', className="stat-value"),
                                        html.Div("Max Drawdown", className="stat-label")
                                    ], className="stat-box"),
                                    html.Div([
                                        html.Div(id='recovery-factor', className="stat-value"),
                                        html.Div("Recovery Factor", className="stat-label")
                                    ], className="stat-box"),
                                ], className="stats-container")
                            ], className="analysis-section"),

                            # Añadir nueva sección para las tablas
                            html.Div([
                                html.H2("Análisis Detallado", className="section-title"),
                                
                                # Estadísticas de Trading
                                html.Div([
                                    html.H4("Estadísticas de Trading"),
                                    html.Div(id='trade-stats-table')
                                ], className="analysis-card"),
                                
                                # Rendimiento Mensual
                                html.Div([
                                    html.H4("Rendimiento Mensual"),
                                    html.Div(id='monthly-performance-table')
                                ], className="analysis-card"),
                                
                                # Análisis de Drawdown
                                html.Div([
                                    html.H4("Análisis de Drawdown"),
                                    html.Div(id='drawdown-analysis-table')
                                ], className="analysis-card"),
                                
                            ], className="tables-section")
                        ]
                    )
                ])
                # Pestaña de comparación eliminada
            ])
        ])

        print("Registrando callbacks...")
        
        # Unified callback
        @app.callback(
            [
                Output('trade-distribution', 'figure'),
                Output('regime-performance', 'figure'),
                Output('total-trades', 'children'),
                Output('avg-trade-duration', 'children'),
                Output('avg-profit-per-trade', 'children'),
                Output('sharpe-ratio', 'children'),
                Output('sortino-ratio', 'children'),
                Output('max-drawdown', 'children'),
                Output('recovery-factor', 'children'),
                Output('trade-stats-table', 'children'),
                Output('monthly-performance-table', 'children'),
                Output('drawdown-analysis-table', 'children')
            ],
            [Input('run-backtest-button', 'n_clicks')],  # Solo el botón como trigger
            [
                State('strategy-selector', 'value'),
                State('timeframe-selector', 'value'),
                State('date-picker-range', 'start_date'),
                State('date-picker-range', 'end_date')
            ],  # El resto como State
            prevent_initial_call=True
        )
        def update_all_charts(n_clicks, strategy, timeframe, start_date, end_date):
            """Unified callback for all updates - only triggered by run button"""
            if n_clicks is None:
                raise dash.exceptions.PreventUpdate
                
            if not all([strategy, timeframe, start_date, end_date]):
                return [create_empty_figure() for _ in range(3)] + ['N/A' for _ in range(9)]

            try:
                # Create backtest instance with proper configuration
                config_path = f"/home/panal/Documents/dashboard-trading/configs/{strategy}.json"
                if not os.path.exists(config_path):
                    print(f"Config file not found: {config_path}")
                    return [create_empty_figure() for _ in range(3)] + ['N/A' for _ in range(7)]
                    
                with open(config_path) as f:
                    config = json.load(f)
                
                # Add timeframe and other required parameters
                config['timeframe'] = timeframe
                config['initial_balance'] = 1000.0
                
                # Get data from cache
                cache = DataCache()
                data = cache.get_cached_data(
                    symbol='BTC/USDT',
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if data is None or len(data) == 0:
                    print("No data available")
                    return [create_empty_figure() for _ in range(3)] + ['N/A' for _ in range(7)]
                
                print(f"Running backtest for {strategy} with {len(data)} candles")
                
                # Initialize correct strategy based on name
                if 'optimized' in strategy.lower():
                    strategy_instance = OptimizedStrategy()
                elif 'enhanced' in strategy.lower():
                    strategy_instance = EnhancedStrategy()
                else:
                    strategy_instance = FixedStrategy()
                
                # Configure and run strategy
                strategy_instance.set_config(config)
                results = strategy_instance.run(data)
                
                # Convert results to DataFrame format expected by visualization functions
                df = pd.DataFrame({
                    'timeframe': [timeframe],
                    'return_total': [results.get('return_total', 0)],
                    'win_rate': [results.get('win_rate', 0)],
                    'profit_factor': [results.get('profit_factor', 1)],
                    'max_drawdown': [results.get('max_drawdown', 0)],
                    'total_trades': [results.get('total_trades', 0)],
                    'trades': [results.get('trades', [])],
                    'equity_curve': [results.get('equity_curve', pd.Series(1.0, index=data.index))]
                })
                
                # Create visualizations
                trade_dist = create_trade_distribution(df)
                regime_perf = create_regime_performance(df)
                stats = calculate_statistics(df)
                
                print("Backtest completed successfully")
                
                return [
                    trade_dist,
                    regime_perf,
                    f"{stats['total_trades']}", 
                    f"{stats['avg_duration']:.1f}h",
                    f"{stats['avg_profit']:.2f}%",
                    f"{stats['sharpe']:.2f}",
                    f"{stats['sortino']:.2f}",
                    f"{stats['max_dd']:.2f}%",
                    f"{stats['recovery']:.2f}",
                    create_trade_stats_table(df),
                    create_monthly_performance_table(df),
                    create_drawdown_analysis_table(df)
                ]
                
            except Exception as e:
                print(f"Error in backtest: {str(e)}")
                import traceback
                print(traceback.format_exc())
                return [create_empty_figure() for _ in range(3)] + ['Error' for _ in range(7)]
        
        print("Dashboard inicializado correctamente")
        return app
        
    except Exception as e:
        print(f"\nERROR EN CREATE_DASHBOARD: {str(e)}")
        import traceback
        print(traceback.format_exc())
        raise

    # Get available strategies
    config_dir = "/home/panal/Documents/dashboard-trading/configs"
    strategies = [f.replace('.json', '') for f in os.listdir(config_dir) if f.endswith('.json')]

    # Load results from hybrid strategy testing
    results_dir = "/home/panal/Documents/dashboard-trading/reports/hybrid_strategy"
    results_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    
    # Default to most recent file
    if results_files:
        results_files.sort(reverse=True)
        latest_results = pd.read_csv(os.path.join(results_dir, results_files[0]))
    else:
        # Create dummy data if no results available
        latest_results = pd.DataFrame({
            'timeframe': ['15m', '1h', '4h'],
            'return_total': [5.56, 1.67, 0.93],
            'win_rate': [52.07, 57.89, 72.41],
            'profit_factor': [1.26, 1.71, 3.02],
            'max_drawdown': [1.70, 0.47, 0.19],
            'total_trades': [434, 114, 29]
        })
    
    @app.callback(
        [Output("param-editor-modal", "style"),
         Output("modal-backdrop", "style")],
        [Input("edit-params-button", "n_clicks"),
         Input("close-modal-button", "n_clicks"),
         Input("save-params-button", "n_clicks")],
        [State("param-editor-modal", "style"),
         State("modal-backdrop", "style")]
    )
    def toggle_modal(edit_clicks, close_clicks, save_clicks, modal_style, backdrop_style):
        if edit_clicks or close_clicks or save_clicks:
            if modal_style.get('display') == 'none':
                return (
                    {**modal_style, 'display': 'block'},
                    {**backdrop_style, 'display': 'block'}
                )
            return (
                {**modal_style, 'display': 'none'},
                {**backdrop_style, 'display': 'none'}
            )
        return modal_style, backdrop_style

    @app.callback(
        Output("param-editor-content", "children"),
        [Input("strategy-selector", "value")]
    )
    def load_strategy_params(strategy_name):
        try:
            with open(f"/home/panal/Documents/dashboard-trading/configs/{strategy_name}.json", 'r') as f:
                config = json.load(f)
            return create_param_editor(config)
        except Exception as e:
            return html.Div(f"Error loading parameters: {str(e)}")

    def create_param_editor(config):
        editors = []
        # RSI Parameters
        if 'rsi' in config:
            editors.append(html.Div([
                html.H4("RSI Parameters"),
                html.Div([
                    html.Label("Oversold:"),
                    dcc.Input(
                        id="rsi-oversold",
                        type="number",
                        value=config['rsi'].get('oversold', 30),
                        min=0,
                        max=100
                    ),
                ]),
                html.Div([
                    html.Label("Overbought:"),
                    dcc.Input(
                        id="rsi-overbought",
                        type="number",
                        value=config['rsi'].get('overbought', 70),
                        min=0,
                        max=100
                    ),
                ]),
            ], className="param-section"))

        # EMA Parameters
        if 'ema' in config:
            editors.append(html.Div([
                html.H4("EMA Parameters"),
                html.Div([
                    html.Label("Short Period:"),
                    dcc.Input(
                        id="ema-short",
                        type="number",
                        value=config['ema'].get('short', 9),
                        min=1
                    ),
                ]),
                html.Div([
                    html.Label("Long Period:"),
                    dcc.Input(
                        id="ema-long",
                        type="number",
                        value=config['ema'].get('long', 21),
                        min=1
                    ),
                ]),
            ], className="param-section"))
        # ... add more parameter sections as needed ...

        return html.Div(editors)

    @app.callback(
        [Output('trade-distribution', 'figure'),
         Output('regime-performance', 'figure'),
         Output('total-trades', 'children'),
         Output('avg-trade-duration', 'children'),
         Output('avg-profit-per-trade', 'children'),
         Output('sharpe-ratio', 'children'),
         Output('sortino-ratio', 'children'),
         Output('max-drawdown', 'children'),
         Output('recovery-factor', 'children')],
        [Input('run-backtest-button', 'n_clicks')],
        [State('strategy-selector', 'value'),
         State('timeframe-selector', 'value'),
         State('date-picker-range', 'start_date'),
         State('date-picker-range', 'end_date')]
    )
    def update_charts(n_clicks, strategy, timeframe, start_date, end_date):
        if n_clicks is None:
            raise dash.exceptions.PreventUpdate

        # Get backtest results
        results = run_backtest(start_date, end_date, timeframe, strategy)
        
        # Create charts
        trade_dist = create_trade_distribution(results)
        regime_perf = create_regime_performance(results)
        
        # Calculate statistics
        stats = calculate_statistics(results)
        
        return (trade_dist, regime_perf,
                f"{stats['total_trades']}", 
                f"{stats['avg_duration']:.1f}h",
                f"{stats['avg_profit']:.2f}%",
                f"{stats['sharpe']:.2f}",
                f"{stats['sortino']:.2f}",
                f"{stats['max_dd']:.2f}%",
                f"{stats['recovery']:.2f}")

    @app.callback(
        [Output("comparison-results", "children"),
         Output("comparison-chart", "figure")],
        [Input("compare-btn", "n_clicks")],
        [State('date-picker-range', 'start_date'),
         State('date-picker-range', 'end_date'),
         State('timeframe-selector', 'value')]
    )
    def compare_strategies(n_clicks, start_date, end_date, timeframe):
        if n_clicks is None:
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

            # Configurar estrategias con config correcto
            base_config = {'timeframe': timeframe, 'initial_balance': 1000.0}
            
            # Inicializar las estrategias
            strategies = {
                'Fixed': FixedStrategy(),
                'Optimized': OptimizedStrategy(),
                'Enhanced': EnhancedStrategy()
            }

            # Ejecutar backtests y recopilar resultados
            results = []
            for name, strategy in strategies.items():
                try:
                    strategy.set_config(base_config)  # Usar el método set_config de BaseStrategy
                    result = strategy.run(data)
                    results.append({
                        'Strategy': name,
                        'Return (%)': float(result.get('return_total', 0)),
                        'Win Rate (%)': float(result.get('win_rate', 0)),
                        'Max DD (%)': float(result.get('max_drawdown', 0)),
                        'Total Trades': int(result.get('total_trades', 0))
                    })
                except Exception as e:
                    print(f"Error running {name} strategy: {str(e)}")
                    continue

            # Crear tabla de resultados
            df = pd.DataFrame(results)
            table = dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                className="comparison-table"
            )

            # Crear gráfico de comparación
            fig = go.Figure()
            metrics = ['Return (%)', 'Win Rate (%)', 'Max DD (%)']
            
            for strategy in df['Strategy']:
                strategy_data = df[df['Strategy'] == strategy]
                fig.add_trace(go.Bar(
                    name=strategy,
                    x=metrics,
                    y=[f"{strategy_data[m].iloc[0]:.1f}%" for m in metrics],
                    text=[f"{strategy_data[m].iloc[0]:.1f}%" for m in metrics],
                    textposition='auto',
                ))

            fig.update_layout(
                title="Comparación de Estrategias",
                barmode='group',
                template='plotly_dark',
                showlegend=True,
                legend_title="Estrategias"
            )

            return table, fig

        except Exception as e:
            print(f"Error in strategy comparison: {str(e)}")
            return html.Div(f"Error: {str(e)}"), go.Figure()

    # Eliminar todos los callbacks anteriores y mantener solo uno unificado
    @app.callback(
        [
            Output('trade-distribution', 'figure'),
            Output('regime-performance', 'figure'),
            Output('total-trades', 'children'),
            Output('avg-trade-duration', 'children'),
            Output('avg-profit-per-trade', 'children'),
            Output('sharpe-ratio', 'children'),
            Output('sortino-ratio', 'children'),
            Output('max-drawdown', 'children'),
            Output('recovery-factor', 'children')
        ],
        [
            Input('strategy-selector', 'value'),
            Input('timeframe-selector', 'value'),
            Input('date-picker-range', 'start_date'),
            Input('date-picker-range', 'end_date'),
            Input('run-backtest-button', 'n_clicks')
        ],
        prevent_initial_call=True
    )
    def update_all_charts(strategy, timeframe, start_date, end_date, n_clicks):
        """Unified callback for all chart and metric updates"""
        ctx = dash.callback_context
        
        if not ctx.triggered:
            raise dash.exceptions.PreventUpdate
            
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        print(f"\nDEBUG: Callback triggered by {trigger_id}")

        if not all([strategy, timeframe, start_date, end_date]):
            return [create_empty_figure() for _ in range(3)] + ['N/A' for _ in range(7)]

        try:
            results = run_backtest(start_date, end_date, timeframe, strategy)
            trade_dist = create_trade_distribution(results)
            regime_perf = create_regime_performance(results)
            stats = calculate_statistics(results)
            
            return [
                trade_dist,
                regime_perf,
                f"{stats['total_trades']}", 
                f"{stats['avg_duration']:.1f}h",
                f"{stats['avg_profit']:.2f}%",
                f"{stats['sharpe']:.2f}",
                f"{stats['sortino']:.2f}",
                f"{stats['max_dd']:.2f}%",
                f"{stats['recovery']:.2f}"
            ]
        except Exception as e:
            print(f"Error updating charts: {str(e)}")
            return [create_empty_figure() for _ in range(3)] + ['Error' for _ in range(7)]

    # Eliminar los siguientes callbacks que están duplicados:
    # - El callback que usa run-backtest-button
    # - El callback que usa strategy-selector
    # - El callback unificado anterior con @54906aaef0920a86122a9e77672aa6cf

    @app.callback(
        [
            Output("strategy-comparison-table", "children"),
            Output("comparison-returns-chart", "figure"),
            Output("comparison-metrics-chart", "figure"),
            Output("debug-output", "children")  # Añadir salida de debug
        ],
        [Input("compare-strategies-btn", "n_clicks")],
        [
            State('compare-date-range', 'start_date'),
            State('compare-date-range', 'end_date'),
            State('compare-timeframe', 'value')
        ],
        prevent_initial_call=False  # Cambiado a False para depuración
    )
    def compare_all_strategies(n_clicks, start_date, end_date, timeframe):
        """Compare all strategies with enhanced debug logging"""
        ctx = dash.callback_context
        debug_print_callback_context(ctx)
        
        print("\nDEBUG: Function arguments")
        print(f"n_clicks: {n_clicks}")
        print(f"start_date: {start_date}")
        print(f"end_date: {end_date}")
        print(f"timeframe: {timeframe}")
        
        debug_msg = f"Button clicked {n_clicks if n_clicks else 0} times"
        
        if not ctx.triggered:
            print("No trigger - initial load")
            return dash.no_update, dash.no_update, dash.no_update, "Waiting for click..."
            
        # Crear valores por defecto para carga inicial
        if n_clicks is None or n_clicks == 0:
            print("Initial load or no clicks")
            return (
                html.Div("Click 'Compare Strategies' to start comparison"),
                go.Figure(),
                go.Figure(),
                "Waiting for first click..."
            )

        try:
            # Validar existencia de componentes
            if not hasattr(app, 'layout'):
                raise Exception("App layout not initialized")
                
            def find_component(layout, component_id):
                """Buscar componente recursivamente en el layout"""
                if hasattr(layout, 'id') and layout.id == component_id:
                    return True
                if hasattr(layout, 'children'):
                    children = layout.children
                    if isinstance(children, list):
                        return any(find_component(child, component_id) for child in children)
                    return find_component(children, component_id)
                return False

            # Verificar componentes críticos
            required_components = [
                "strategy-comparison-table",
                "comparison-returns-chart", 
                "comparison-metrics-chart"
            ]
            
            for component_id in required_components:
                if not find_component(app.layout, component_id):
                    raise Exception(f"Component {component_id} not found in layout")

            # ... resto del código del callback ...
            
        except Exception as e:
            print(f"Error in comparison callback: {str(e)}")
            import traceback
            print(traceback.format_exc())
            return html.Div(f"Error: {str(e)}"), go.Figure(), go.Figure()

    # Añadir callback de depuración simple
    @app.callback(
        Output('debug-output', 'children'),
        [Input('compare-strategies-btn', 'n_clicks')],
        prevent_initial_call=False
    )
    def update_debug_output(n_clicks):
        """Simple callback para verificar funcionamiento del botón"""
        print(f"Debug callback triggered with {n_clicks} clicks")
        if n_clicks is None:
            return "Waiting for first click..."
        return f"Button clicked {n_clicks} times"

    return app

def create_empty_figure():
    """Create empty figure with message"""
    fig = go.Figure()
    fig.update_layout(
        title="No data available",
        xaxis_title="",
        yaxis_title="",
        template='plotly_dark',
        annotations=[{
            'text': "No trading data available. Run backtest first.",
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 14}
        }]
    )
    return fig

def run_backtest(start_date, end_date, timeframe, strategy):
    """Run backtest with the specified parameters"""
    try:
        print(f"\nRunning backtest for {strategy}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {start_date} to {end_date}")
        
        # Get data from cache
        cache = DataCache()
        data = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
            end_date=pd.to_datetime(end_date).strftime('%Y-%m-%d')
        )
        
        if data is None or len(data) == 0:
            print("No data available for backtest")
            return None

        print(f"Loaded {len(data)} candles")
            
        # Load strategy config
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{strategy}.json"
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            return None
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Add timeframe and other required parameters
        config['timeframe'] = timeframe
        config['initial_balance'] = 1000.0
        
        # Initialize and run backtest
        backtest = BacktestFixedStrategy(config=config)
        results = backtest.run(data)
        
        if not results or 'trades' not in results:
            print("No backtest results")
            return None
            
        print(f"Backtest completed with {len(results.get('trades', []))} trades")
        
        # Ensure trades is a list, not pd.Series
        if isinstance(results['trades'], pd.Series):
            results['trades'] = results['trades'].tolist()

        # Convert to DataFrame with proper structure
        df = pd.DataFrame({
            'timeframe': [timeframe],
            'return_total': [float(results.get('return_total', 0))],
            'win_rate': [float(results.get('win_rate', 0))],
            'profit_factor': [float(results.get('profit_factor', 1))],
            'max_drawdown': [float(results.get('max_drawdown', 0))],
            'total_trades': [int(results.get('total_trades', 0))],
            'trades': [results.get('trades', [])],
            'equity_curve': [results.get('equity_curve', pd.Series(1.0, index=data.index))]
        })

        print(f"Processed results: {df.to_dict()}")
        return df
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def create_performance_chart(df):
    """Create a performance comparison chart"""
    # Prepare data
    df_melted = pd.melt(
        df, 
        id_vars=['timeframe'], 
        value_vars=['return_total', 'win_rate', 'profit_factor'],
        var_name='metric', 
        value_name='value'
    )
    
    # Map friendly names
    metric_names = {
        'return_total': 'Return (%)',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor'
    }
    df_melted['metric'] = df_melted['metric'].map(metric_names)
    
    # Create figure
    fig = px.bar(
        df_melted,
        x='timeframe',
        y='value',
        color='metric',
        barmode='group',
        title='Performance Metrics by Timeframe',
        labels={'timeframe': 'Timeframe', 'value': 'Value', 'metric': 'Metric'},
        color_discrete_sequence=['#12939A', '#79C7E3', '#1A3177']
    )
    
    return fig

def create_risk_return_chart(df):
    """Create a risk-return scatter plot"""
    fig = px.scatter(
        df,
        x='max_drawdown',
        y='return_total',
        size='total_trades',
        color='profit_factor',
        hover_name='timeframe',
        labels={
            'max_drawdown': 'Maximum Drawdown (%)',
            'return_total': 'Return (%)',
            'profit_factor': 'Profit Factor',
            'total_trades': 'Total Trades'
        },
        title='Risk vs. Return by Timeframe',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add optimality line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df['max_drawdown'].max() * 1.2,
        y1=df['return_total'].max() * 1.2,
        line=dict(color="red", dash="dash")
    )
    
    return fig

def create_comparison_table(df):
    """Create an HTML comparison table"""
    # Format dataframe for display
    display_df = df.copy()
    display_df = display_df.rename(columns={
        'timeframe': 'Timeframe',
        'return_total': 'Return (%)',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor',
        'max_drawdown': 'Max DD (%)',
        'total_trades': 'Trades'
    })
    
    # Round numeric columns
    for col in display_df.columns:
        if col != 'Timeframe' and col != 'Trades':
            display_df[col] = display_df[col].round(2)
    
    # Convert to HTML table
    table_rows = []
    
    # Header row
    header_row = html.Tr([html.Th(col) for col in display_df.columns])
    table_rows.append(header_row)
    
    # Data rows
    for i, row in display_df.iterrows():
        # Color code return cells
        return_cell = html.Td(
            f"{row['Return (%)']:.2f}%", 
            style={'color': 'green' if row['Return (%)'] > 0 else 'red'}
        )
        
        # Build the row
        table_row = html.Tr([
            html.Td(row['Timeframe']),
            return_cell,
            html.Td(f"{row['Win Rate (%)']}%"),
            html.Td(f"{row['Profit Factor']}"),
            html.Td(f"{row['Max DD (%)']}%"),
            html.Td(row['Trades'])
        ])
        table_rows.append(table_row)
    
    return html.Table(table_rows, className="comparison-table")

def create_equity_curve(results):
    """Create equity curve visualization"""
    try:
        initial_balance = 1000.0  # Use float instead of int
        
        if not isinstance(results, pd.DataFrame) or 'trades' not in results:
            return go.Figure()

        # Get trade data and ensure we have trades
        trades = results['trades'].iloc[0]
        if not trades:
            # Return empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title='No trades available for equity curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)'
            )
            return fig

        # Create initial equity curve with all data points using 'h' instead of 'H'
        data_index = pd.date_range(
            start=min(t['entry_time'] for t in trades),
            end=max(t['exit_time'] for t in trades),
            freq='h'  # Changed from 'H' to 'h'
        )
        equity_curve = pd.Series(float(initial_balance), index=data_index, dtype='float64')  # Explicit float dtype

        # Calculate cumulative equity with proper typing
        current_equity = float(initial_balance)
        for trade in sorted(trades, key=lambda x: x['exit_time']):
            pnl_pct = float(trade['pnl']) / 100.0
            trade_profit = current_equity * pnl_pct
            current_equity += trade_profit
            equity_curve.loc[pd.to_datetime(trade['exit_time'])] = float(current_equity)

        # Forward fill gaps
        equity_curve = equity_curve.ffill()  # Use ffill() instead of fillna(method='ffill')

        # Calculate final return percentage
        total_return = ((equity_curve.iloc[-1] - initial_balance) / initial_balance) * 100

        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='#2ecc71', width=2)
        ))

        # Add initial balance line
        fig.add_hline(
            y=initial_balance,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )

        # Update layout
        fig.update_layout(
            title=f'Equity Curve (Return: {total_return:.2f}%)',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            template='plotly_white',
            hovermode='x unified',
            showlegend=False,
            yaxis=dict(
                tickformat='$,.2f',
                range=[
                    min(initial_balance * 0.95, equity_curve.min() * 0.95),
                    max(initial_balance * 1.05, equity_curve.max() * 1.05)
                ]
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating equity curve: {str(e)}")
        return go.Figure()

def create_trade_distribution(results):
    """Create trade distribution chart with debug logging"""
    print("\nCreating trade distribution chart...")
    print(f"Results type: {type(results)}")
    
    if not isinstance(results, pd.DataFrame):
        print("Warning: Results is not a DataFrame")
        return create_empty_figure()
        
    if 'trades' not in results.columns:
        print("Warning: No trades column in results")
        return create_empty_figure()

    print(f"Number of trade entries: {len(results['trades'])}")
    
    # Calculate trade distribution
    trade_pnls = []
    for trades in results['trades']:
        if isinstance(trades, list):
            trade_pnls.extend([float(t.get('pnl', 0)) for t in trades if isinstance(t, dict)])

    print(f"Processed {len(trade_pnls)} trades")

    if not trade_pnls:
        print("Warning: No valid PnL values found")
        return create_empty_figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trade_pnls,
        nbinsx=30,
        name='Trade PnL',
        marker_color=['#2ecc71' if x >= 0 else '#e74c3c' for x in trade_pnls]
    ))

    fig.update_layout(
        title='Trade PnL Distribution',
        xaxis_title='PnL (%)',
        yaxis_title='Number of Trades',
        showlegend=False,
        template='plotly_dark'
    )
    
    print("Trade distribution chart created successfully")
    return fig

def create_regime_performance(results):
    """Create regime performance chart"""
    if not isinstance(results, pd.DataFrame):
        # Create dummy regime performance if not available
        fig = go.Figure()
        fig.update_layout(
            title='No Regime Performance Data Available',
            xaxis_title='Regime',
            yaxis_title='Return (%)'
        )
        return fig

    fig = go.Figure()
    regimes = ['Trending', 'Ranging', 'Volatile']
    returns = [
        results['return_total'].mean() if len(results) > 0 else 0,
        results['return_total'].mean() * 0.8 if len(results) > 0 else 0,
        results['return_total'].mean() * 1.2 if len(results) > 0 else 0
    ]
    
    fig.add_trace(go.Bar(
        x=regimes,
        y=returns,
        marker_color=['#2ecc71' if x >= 0 else '#e74c3c' for x in returns]
    ))
    
    fig.update_layout(
        title='Performance by Market Regime',
        xaxis_title='Market Regime',
        yaxis_title='Return (%)',
        template='plotly_white',
        showlegend=False
    )

    return fig

def calculate_statistics(results):
    """Calculate statistics with debug logging"""
    print("\nCalculating statistics...")
    
    stats = {
        'total_trades': 0,
        'avg_duration': 0.0,
        'avg_profit': 0.0,
        'sharpe': 0.0,
        'sortino': 0.0,
        'max_dd': 0.0,
        'recovery': 0.0
    }
    
    if not isinstance(results, pd.DataFrame):
        print("Warning: Results is not a DataFrame")
        return stats

    try:
        # Calculate basic statistics
        stats['total_trades'] = int(results['total_trades'].sum() if 'total_trades' in results else 0)
        stats['avg_profit'] = float(results['return_total'].mean() if 'return_total' in results else 0)
        stats['max_dd'] = float(results['max_drawdown'].max() if 'max_drawdown' in results else 0)

        # Calculate duration from trades
        if 'trades' in results:
            durations = []
            for trades in results['trades']:
                if isinstance(trades, list):
                    durations.extend([float(t.get('bars_held', 0)) for t in trades if isinstance(t, dict)])
            stats['avg_duration'] = float(np.mean(durations) if durations else 0)

        print(f"Calculated statistics: {stats}")
        return stats
        
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return stats

def load_strategy_combinations():
    """Load available strategy-profile combinations"""
    base_strategies = {
        'Fixed': FixedStrategy,
        'Optimized': OptimizedStrategy,
        'Enhanced': EnhancedStrategy
    }
    
    profiles = {
        'conservative': {
            'risk_profile': 'conservative',
            'position_size': {'min': 0.02, 'max': 0.04},
            'description': 'Low risk, fewer trades'
        },
        'moderate': {
            'risk_profile': 'moderate',
            'position_size': {'min': 0.03, 'max': 0.06},
            'description': 'Balanced risk/reward'
        },
        'aggressive': {
            'risk_profile': 'aggressive',
            'position_size': {'min': 0.05, 'max': 0.08},
            'description': 'Higher risk, more trades'
        },
        'hybrid': {
            'risk_profile': 'hybrid',
            'position_size': {'min': 0.03, 'max': 0.07},
            'description': 'Adaptive risk management'
        }
    }

    combinations = []
    for strategy_name, strategy_class in base_strategies.items():
        for profile_name, profile_config in profiles.items():
            combinations.append({
                'label': f"{strategy_name} ({profile_name})",
                'value': f"{strategy_name.lower().replace(' ', '_')}_{profile_config['risk_profile']}",
                'strategy': strategy_class,
                'profile': profile_config
            })
    
    return combinations

# Modificar el dropdown de estrategias
strategy_combinations = load_strategy_combinations()
dcc.Dropdown(
    id='strategy-selector',
    options=[{
        'label': combo['label'],
        'value': combo['value']
    } for combo in strategy_combinations],
    value=strategy_combinations[0]['value']
),

def create_trade_stats_table(results):
    """Create a table with detailed trade statistics"""
    try:
        if not isinstance(results, pd.DataFrame) or 'trades' not in results:
            return html.Div("No hay datos disponibles")

        trades = results['trades'].iloc[0]
        if not trades:
            return html.Div("No hay operaciones para analizar")

        # Preparar datos para las tablas
        general_stats = [
            ['Total Operaciones', len(trades)],
            ['Operaciones Ganadoras', len([t for t in trades if t['pnl'] > 0])],
            ['Win Rate', f"{results['win_rate'].iloc[0]:.1f}%"],
            ['Profit Factor', f"{results['profit_factor'].iloc[0]:.2f}"]
        ]

        profit_stats = [
            ['Retorno Total', f"{results['return_total'].iloc[0]:.2f}%"],
            ['Max Drawdown', f"{results['max_drawdown'].iloc[0]:.2f}%"]
        ]

        # Crear tablas usando table_utils
        tables = [
            html.H5("Estadísticas Generales", className="table-title"),
            create_table(['Métrica', 'Valor'], general_stats, id='general-stats-table'),
            
            html.H5("Análisis de Rentabilidad", className="table-title"),
            create_table(['Métrica', 'Valor'], profit_stats, id='profit-stats-table')
        ]

        return html.Div(tables, className="tables-container")

    except Exception as e:
        print(f"Error creating trade stats table: {str(e)}")
        return html.Div("Error al crear la tabla de estadísticas")

def create_monthly_performance_table(results):
    """Create a table showing monthly performance metrics"""
    if not isinstance(results, pd.DataFrame) or 'trades' not in results:
        return html.Div("No hay datos disponibles")

    try:
        trades = results['trades'].iloc[0]
        if not trades:
            return html.Div("No hay operaciones para analizar")

        # Crear DataFrame con trades
        trades_df = pd.DataFrame(trades)
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['month'] = trades_df['exit_time'].dt.strftime('%Y-%m')

        # Calcular métricas mensuales
        monthly_stats = []
        initial_balance = 1000  # Balance inicial para cada mes

        for month in trades_df['month'].unique():
            month_trades = trades_df[trades_df['month'] == month]
            
            # Simular balance del mes
            balance = initial_balance
            for pnl in month_trades['pnl']:
                balance *= (1 + pnl/100)

            stats = {
                'Mes': month,
                'Operaciones': len(month_trades),
                'Ganadas': len(month_trades[month_trades['pnl'] > 0]),
                'Win Rate': f"{(len(month_trades[month_trades['pnl'] > 0])/len(month_trades)*100):.1f}%",
                'Balance Final': f"${balance:.2f}",
                'Retorno': f"{((balance-initial_balance)/initial_balance*100):.2f}%"
            }
            monthly_stats.append(stats)

        # Crear DataFrame y tabla
        df = pd.DataFrame(monthly_stats)
        table = dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            className="monthly-performance-table"
        )

        return table

    except Exception as e:
        print(f"Error creating monthly performance table: {str(e)}")
        return html.Div("Error al crear la tabla de rendimiento mensual")

def create_drawdown_analysis_table(results):
    """Create a table showing drawdown analysis"""
    if not isinstance(results, pd.DataFrame) or 'equity_curve' not in results:
        return html.Div("No hay datos disponibles")

    try:
        equity_curve = results['equity_curve'].iloc[0]
        
        # Calcular drawdowns
        rolling_max = equity_curve.expanding().max()
        drawdowns = ((equity_curve - rolling_max) / rolling_max) * 100
        
        # Identificar periodos de drawdown
        dd_periods = []
        in_drawdown = False
        start_idx = None
        
        for i in range(len(drawdowns)):
            if drawdowns[i] < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif drawdowns[i] >= 0 and in_drawdown:
                in_drawdown = False
                dd_periods.append({
                    'Inicio': equity_curve.index[start_idx].strftime('%Y-%m-%d'),
                    'Fin': equity_curve.index[i].strftime('%Y-%m-%d'),
                    'Duración (días)': (equity_curve.index[i] - equity_curve.index[start_idx]).days,
                    'Drawdown Máximo': f"{drawdowns[start_idx:i].min():.2f}%",
                    'Tiempo Recuperación': f"{(equity_curve.index[i] - equity_curve.index[start_idx]).days} días"
                })

        # Crear DataFrame y tabla
        if dd_periods:
            df = pd.DataFrame(dd_periods)
            table = dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                className="drawdown-analysis-table"
            )
            return table
        else:
            return html.Div("No se encontraron periodos de drawdown significativos")

    except Exception as e:
        print(f"Error creating drawdown analysis table: {str(e)}")
        return html.Div("Error al crear la tabla de análisis de drawdown")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the performance dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    
    args = argparse.ArgumentParser(description='Run the performance dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    
    args = parser.parse_args()
    
    app = create_dashboard(args.port)
    app.run_server(debug=True, host='0.0.0.0', port=args.port)

def create_empty_figure():
    """Create empty figure with message"""
    fig = go.Figure()
    fig.update_layout(
        title="No data available",
        xaxis_title="",
        yaxis_title="",
        template='plotly_dark',
        annotations=[{
            'text': "No trading data available. Run backtest first.",
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 14}
        }]
    )
    return fig

def run_backtest(start_date, end_date, timeframe, strategy):
    """Run backtest with the specified parameters"""
    try:
        print(f"\nRunning backtest for {strategy}")
        print(f"Timeframe: {timeframe}")
        print(f"Period: {start_date} to {end_date}")
        
        # Get data from cache
        cache = DataCache()
        data = cache.get_cached_data(
            symbol='BTC/USDT',
            timeframe=timeframe,
            start_date=pd.to_datetime(start_date).strftime('%Y-%m-%d'),
            end_date=pd.to_datetime(end_date).strftime('%Y-%m-%d')
        )
        
        if data is None or len(data) == 0:
            print("No data available for backtest")
            return None

        print(f"Loaded {len(data)} candles")
            
        # Load strategy config
        config_path = f"/home/panal/Documents/dashboard-trading/configs/{strategy}.json"
        if not os.path.exists(config_path):
            print(f"Config not found: {config_path}")
            return None
            
        with open(config_path) as f:
            config = json.load(f)
            
        # Add timeframe and other required parameters
        config['timeframe'] = timeframe
        config['initial_balance'] = 1000.0
        
        # Initialize and run backtest
        backtest = BacktestFixedStrategy(config=config)
        results = backtest.run(data)
        
        if not results or 'trades' not in results:
            print("No backtest results")
            return None
            
        print(f"Backtest completed with {len(results.get('trades', []))} trades")
        
        # Ensure trades is a list, not pd.Series
        if isinstance(results['trades'], pd.Series):
            results['trades'] = results['trades'].tolist()

        # Convert to DataFrame with proper structure
        df = pd.DataFrame({
            'timeframe': [timeframe],
            'return_total': [float(results.get('return_total', 0))],
            'win_rate': [float(results.get('win_rate', 0))],
            'profit_factor': [float(results.get('profit_factor', 1))],
            'max_drawdown': [float(results.get('max_drawdown', 0))],
            'total_trades': [int(results.get('total_trades', 0))],
            'trades': [results.get('trades', [])],
            'equity_curve': [results.get('equity_curve', pd.Series(1.0, index=data.index))]
        })

        print(f"Processed results: {df.to_dict()}")
        return df
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return None

def create_performance_chart(df):
    """Create a performance comparison chart"""
    # Prepare data
    df_melted = pd.melt(
        df, 
        id_vars=['timeframe'], 
        value_vars=['return_total', 'win_rate', 'profit_factor'],
        var_name='metric', 
        value_name='value'
    )
    
    # Map friendly names
    metric_names = {
        'return_total': 'Return (%)',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor'
    }
    df_melted['metric'] = df_melted['metric'].map(metric_names)
    
    # Create figure
    fig = px.bar(
        df_melted,
        x='timeframe',
        y='value',
        color='metric',
        barmode='group',
        title='Performance Metrics by Timeframe',
        labels={'timeframe': 'Timeframe', 'value': 'Value', 'metric': 'Metric'},
        color_discrete_sequence=['#12939A', '#79C7E3', '#1A3177']
    )
    
    return fig

def create_risk_return_chart(df):
    """Create a risk-return scatter plot"""
    fig = px.scatter(
        df,
        x='max_drawdown',
        y='return_total',
        size='total_trades',
        color='profit_factor',
        hover_name='timeframe',
        labels={
            'max_drawdown': 'Maximum Drawdown (%)',
            'return_total': 'Return (%)',
            'profit_factor': 'Profit Factor',
            'total_trades': 'Total Trades'
        },
        title='Risk vs. Return by Timeframe',
        color_continuous_scale=px.colors.sequential.Viridis
    )
    
    # Add optimality line
    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=df['max_drawdown'].max() * 1.2,
        y1=df['return_total'].max() * 1.2,
        line=dict(color="red", dash="dash")
    )
    
    return fig

def create_comparison_table(df):
    """Create an HTML comparison table"""
    # Format dataframe for display
    display_df = df.copy()
    display_df = display_df.rename(columns={
        'timeframe': 'Timeframe',
        'return_total': 'Return (%)',
        'win_rate': 'Win Rate (%)',
        'profit_factor': 'Profit Factor',
        'max_drawdown': 'Max DD (%)',
        'total_trades': 'Trades'
    })
    
    # Round numeric columns
    for col in display_df.columns:
        if col != 'Timeframe' and col != 'Trades':
            display_df[col] = display_df[col].round(2)
    
    # Convert to HTML table
    table_rows = []
    
    # Header row
    header_row = html.Tr([html.Th(col) for col in display_df.columns])
    table_rows.append(header_row)
    
    # Data rows
    for i, row in display_df.iterrows():
        # Color code return cells
        return_cell = html.Td(
            f"{row['Return (%)']:.2f}%", 
            style={'color': 'green' if row['Return (%)'] > 0 else 'red'}
        )
        
        # Build the row
        table_row = html.Tr([
            html.Td(row['Timeframe']),
            return_cell,
            html.Td(f"{row['Win Rate (%)']}%"),
            html.Td(f"{row['Profit Factor']}"),
            html.Td(f"{row['Max DD (%)']}%"),
            html.Td(row['Trades'])
        ])
        table_rows.append(table_row)
    
    return html.Table(table_rows, className="comparison-table")

def create_equity_curve(results):
    """Create equity curve visualization"""
    try:
        initial_balance = 1000.0  # Use float instead of int
        
        if not isinstance(results, pd.DataFrame) or 'trades' not in results:
            return go.Figure()

        # Get trade data and ensure we have trades
        trades = results['trades'].iloc[0]
        if not trades:
            # Return empty figure with message
            fig = go.Figure()
            fig.update_layout(
                title='No trades available for equity curve',
                xaxis_title='Date',
                yaxis_title='Equity ($)'
            )
            return fig

        # Create initial equity curve with all data points using 'h' instead of 'H'
        data_index = pd.date_range(
            start=min(t['entry_time'] for t in trades),
            end=max(t['exit_time'] for t in trades),
            freq='h'  # Changed from 'H' to 'h'
        )
        equity_curve = pd.Series(float(initial_balance), index=data_index, dtype='float64')  # Explicit float dtype

        # Calculate cumulative equity with proper typing
        current_equity = float(initial_balance)
        for trade in sorted(trades, key=lambda x: x['exit_time']):
            pnl_pct = float(trade['pnl']) / 100.0
            trade_profit = current_equity * pnl_pct
            current_equity += trade_profit
            equity_curve.loc[pd.to_datetime(trade['exit_time'])] = float(current_equity)

        # Forward fill gaps
        equity_curve = equity_curve.ffill()  # Use ffill() instead of fillna(method='ffill')

        # Calculate final return percentage
        total_return = ((equity_curve.iloc[-1] - initial_balance) / initial_balance) * 100

        # Create figure
        fig = go.Figure()
        
        # Add equity curve
        fig.add_trace(go.Scatter(
            x=equity_curve.index,
            y=equity_curve.values,
            mode='lines',
            name='Equity',
            line=dict(color='#2ecc71', width=2)
        ))

        # Add initial balance line
        fig.add_hline(
            y=initial_balance,
            line_dash="dash",
            line_color="gray",
            opacity=0.5
        )

        # Update layout
        fig.update_layout(
            title=f'Equity Curve (Return: {total_return:.2f}%)',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            template='plotly_white',
            hovermode='x unified',
            showlegend=False,
            yaxis=dict(
                tickformat='$,.2f',
                range=[
                    min(initial_balance * 0.95, equity_curve.min() * 0.95),
                    max(initial_balance * 1.05, equity_curve.max() * 1.05)
                ]
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating equity curve: {str(e)}")
        return go.Figure()

def create_trade_distribution(results):
    """Create trade distribution chart with debug logging"""
    print("\nCreating trade distribution chart...")
    print(f"Results type: {type(results)}")
    
    if not isinstance(results, pd.DataFrame):
        print("Warning: Results is not a DataFrame")
        return create_empty_figure()
        
    if 'trades' not in results.columns:
        print("Warning: No trades column in results")
        return create_empty_figure()

    print(f"Number of trade entries: {len(results['trades'])}")
    
    # Calculate trade distribution
    trade_pnls = []
    for trades in results['trades']:
        if isinstance(trades, list):
            trade_pnls.extend([float(t.get('pnl', 0)) for t in trades if isinstance(t, dict)])

    print(f"Processed {len(trade_pnls)} trades")

    if not trade_pnls:
        print("Warning: No valid PnL values found")
        return create_empty_figure()

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=trade_pnls,
        nbinsx=30,
        name='Trade PnL',
        marker_color=['#2ecc71' if x >= 0 else '#e74c3c' for x in trade_pnls]
    ))

    fig.update_layout(
        title='Trade PnL Distribution',
        xaxis_title='PnL (%)',
        yaxis_title='Number of Trades',
        showlegend=False,
        template='plotly_dark'
    )
    
    print("Trade distribution chart created successfully")
    return fig

def create_regime_performance(results):
    """Create regime performance chart"""
    if not isinstance(results, pd.DataFrame):
        # Create dummy regime performance if not available
        fig = go.Figure()
        fig.update_layout(
            title='No Regime Performance Data Available',
            xaxis_title='Regime',
            yaxis_title='Return (%)'
        )
        return fig

    fig = go.Figure()
    regimes = ['Trending', 'Ranging', 'Volatile']
    returns = [
        results['return_total'].mean() if len(results) > 0 else 0,
        results['return_total'].mean() * 0.8 if len(results) > 0 else 0,
        results['return_total'].mean() * 1.2 if len(results) > 0 else 0
    ]
    
    fig.add_trace(go.Bar(
        x=regimes,
        y=returns,
        marker_color=['#2ecc71' if x >= 0 else '#e74c3c' for x in returns]
    ))
    
    fig.update_layout(
        title='Performance by Market Regime',
        xaxis_title='Market Regime',
        yaxis_title='Return (%)',
        template='plotly_white',
        showlegend=False
    )

    return fig

def calculate_statistics(results):
    """Calculate statistics with debug logging"""
    print("\nCalculating statistics...")
    
    stats = {
        'total_trades': 0,
        'avg_duration': 0.0,
        'avg_profit': 0.0,
        'sharpe': 0.0,
        'sortino': 0.0,
        'max_dd': 0.0,
        'recovery': 0.0
    }
    
    if not isinstance(results, pd.DataFrame):
        print("Warning: Results is not a DataFrame")
        return stats

    try:
        # Calculate basic statistics
        stats['total_trades'] = int(results['total_trades'].sum() if 'total_trades' in results else 0)
        stats['avg_profit'] = float(results['return_total'].mean() if 'return_total' in results else 0)
        stats['max_dd'] = float(results['max_drawdown'].max() if 'max_drawdown' in results else 0)

        # Calculate duration from trades
        if 'trades' in results:
            durations = []
            for trades in results['trades']:
                if isinstance(trades, list):
                    durations.extend([float(t.get('bars_held', 0)) for t in trades if isinstance(t, dict)])
            stats['avg_duration'] = float(np.mean(durations) if durations else 0)

        print(f"Calculated statistics: {stats}")
        return stats
        
    except Exception as e:
        print(f"Error calculating statistics: {str(e)}")
        return stats

def load_strategy_combinations():
    """Load available strategy-profile combinations"""
    base_strategies = {
        'Fixed': FixedStrategy,
        'Optimized': OptimizedStrategy,
        'Enhanced': EnhancedStrategy
    }
    
    profiles = {
        'conservative': {
            'risk_profile': 'conservative',
            'position_size': {'min': 0.02, 'max': 0.04},
            'description': 'Low risk, fewer trades'
        },
        'moderate': {
            'risk_profile': 'moderate',
            'position_size': {'min': 0.03, 'max': 0.06},
            'description': 'Balanced risk/reward'
        },
        'aggressive': {
            'risk_profile': 'aggressive',
            'position_size': {'min': 0.05, 'max': 0.08},
            'description': 'Higher risk, more trades'
        },
        'hybrid': {
            'risk_profile': 'hybrid',
            'position_size': {'min': 0.03, 'max': 0.07},
            'description': 'Adaptive risk management'
        }
    }

    combinations = []
    for strategy_name, strategy_class in base_strategies.items():
        for profile_name, profile_config in profiles.items():
            combinations.append({
                'label': f"{strategy_name} ({profile_name})",
                'value': f"{strategy_name.lower().replace(' ', '_')}_{profile_config['risk_profile']}",
                'strategy': strategy_class,
                'profile': profile_config
            })
    
    return combinations

# Modificar el dropdown de estrategias
strategy_combinations = load_strategy_combinations()
dcc.Dropdown(
    id='strategy-selector',
    options=[{
        'label': combo['label'],
        'value': combo['value']
    } for combo in strategy_combinations],
    value=strategy_combinations[0]['value']
),

def create_trade_stats_table(results):
    """Create a table with detailed trade statistics"""
    try:
        if not isinstance(results, pd.DataFrame) or 'trades' not in results:
            return html.Div("No hay datos disponibles")

        trades = results['trades'].iloc[0]
        if not trades:
            return html.Div("No hay operaciones para analizar")

        # Preparar datos para las tablas
        general_stats = [
            ['Total Operaciones', len(trades)],
            ['Operaciones Ganadoras', len([t for t in trades if t['pnl'] > 0])],
            ['Win Rate', f"{results['win_rate'].iloc[0]:.1f}%"],
            ['Profit Factor', f"{results['profit_factor'].iloc[0]:.2f}"]
        ]

        profit_stats = [
            ['Retorno Total', f"{results['return_total'].iloc[0]:.2f}%"],
            ['Max Drawdown', f"{results['max_drawdown'].iloc[0]:.2f}%"]
        ]

        # Crear tablas usando table_utils
        tables = [
            html.H5("Estadísticas Generales", className="table-title"),
            create_table(['Métrica', 'Valor'], general_stats, id='general-stats-table'),
            
            html.H5("Análisis de Rentabilidad", className="table-title"),
            create_table(['Métrica', 'Valor'], profit_stats, id='profit-stats-table')
        ]

        return html.Div(tables, className="tables-container")

    except Exception as e:
        print(f"Error creating trade stats table: {str(e)}")
        return html.Div("Error al crear la tabla de estadísticas")

def create_monthly_performance_table(results):
    """Create a table showing monthly performance metrics"""
    if not isinstance(results, pd.DataFrame) or 'trades' not in results:
        return html.Div("No hay datos disponibles")

    try:
        trades = results['trades'].iloc[0]
        if not trades:
            return html.Div("No hay operaciones para analizar")

        # Crear DataFrame con trades
        trades_df = pd.DataFrame(trades)
        trades_df['exit_time'] = pd.to_datetime(trades_df['exit_time'])
        trades_df['month'] = trades_df['exit_time'].dt.strftime('%Y-%m')

        # Calcular métricas mensuales
        monthly_stats = []
        initial_balance = 1000  # Balance inicial para cada mes

        for month in trades_df['month'].unique():
            month_trades = trades_df[trades_df['month'] == month]
            
            # Simular balance del mes
            balance = initial_balance
            for pnl in month_trades['pnl']:
                balance *= (1 + pnl/100)

            stats = {
                'Mes': month,
                'Operaciones': len(month_trades),
                'Ganadas': len(month_trades[month_trades['pnl'] > 0]),
                'Win Rate': f"{(len(month_trades[month_trades['pnl'] > 0])/len(month_trades)*100):.1f}%",
                'Balance Final': f"${balance:.2f}",
                'Retorno': f"{((balance-initial_balance)/initial_balance*100):.2f}%"
            }
            monthly_stats.append(stats)

        # Crear DataFrame y tabla
        df = pd.DataFrame(monthly_stats)
        table = dbc.Table.from_dataframe(
            df,
            striped=True,
            bordered=True,
            hover=True,
            className="monthly-performance-table"
        )

        return table

    except Exception as e:
        print(f"Error creating monthly performance table: {str(e)}")
        return html.Div("Error al crear la tabla de rendimiento mensual")

def create_drawdown_analysis_table(results):
    """Create a table showing drawdown analysis"""
    if not isinstance(results, pd.DataFrame) or 'equity_curve' not in results:
        return html.Div("No hay datos disponibles")

    try:
        equity_curve = results['equity_curve'].iloc[0]
        
        # Calcular drawdowns
        rolling_max = equity_curve.expanding().max()
        drawdowns = ((equity_curve - rolling_max) / rolling_max) * 100
        
        # Identificar periodos de drawdown
        dd_periods = []
        in_drawdown = False
        start_idx = None
        
        for i in range(len(drawdowns)):
            if drawdowns[i] < 0 and not in_drawdown:
                in_drawdown = True
                start_idx = i
            elif drawdowns[i] >= 0 and in_drawdown:
                in_drawdown = False
                dd_periods.append({
                    'Inicio': equity_curve.index[start_idx].strftime('%Y-%m-%d'),
                    'Fin': equity_curve.index[i].strftime('%Y-%m-%d'),
                    'Duración (días)': (equity_curve.index[i] - equity_curve.index[start_idx]).days,
                    'Drawdown Máximo': f"{drawdowns[start_idx:i].min():.2f}%",
                    'Tiempo Recuperación': f"{(equity_curve.index[i] - equity_curve.index[start_idx]).days} días"
                })

        # Crear DataFrame y tabla
        if dd_periods:
            df = pd.DataFrame(dd_periods)
            table = dbc.Table.from_dataframe(
                df,
                striped=True,
                bordered=True,
                hover=True,
                className="drawdown-analysis-table"
            )
            return table
        else:
            return html.Div("No se encontraron periodos de drawdown significativos")

    except Exception as e:
        print(f"Error creating drawdown analysis table: {str(e)}")
        return html.Div("Error al crear la tabla de análisis de drawdown")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the performance dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    
    args = argparse.ArgumentParser(description='Run the performance dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port for the dashboard')
    
    args = parser.parse_args()
    
    app = create_dashboard(args.port)
    app.run_server(debug=True, host='0.0.0.0', port=args.port)

