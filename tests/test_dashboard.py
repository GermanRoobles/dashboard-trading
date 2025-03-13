import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Importaciones directas
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Importar desde apps.performance_dashboard directamente después de las importaciones básicas
from apps.performance_dashboard import (
    create_monthly_returns, 
    create_dashboard,
    create_trade_distribution,
    create_regime_performance,
    calculate_statistics
)

# Importaciones fundamentales que no deben ser limpiadas
import sys
import os
from importlib import reload

# Función para recargar módulos críticos
def reload_critical_modules():
    """Reload critical modules to ensure clean state"""
    import pandas as pd
    import numpy as np
    import plotly.graph_objs as go
    reload(pd)
    reload(np)
    reload(go)
    
    # También recargar nuestros módulos
    from apps import performance_dashboard
    reload(performance_dashboard)

def setup_module():
    """Setup function that runs before all tests"""
    reload_critical_modules()

# Función de prueba para verificar importaciones
def test_imports():
    """Verify all required imports are working"""
    try:
        import pandas as pd
        import numpy as np
        import plotly.graph_objs as go
        from apps.performance_dashboard import (
            create_monthly_returns,
            create_dashboard,
            create_trade_distribution
        )
        return "Import test passed"
    except Exception as e:
        return f"Import test failed: {str(e)}"

def create_test_trades():
    """Create a more comprehensive test dataset"""
    return [
        {
            'entry_time': '2024-01-01 10:00:00',
            'exit_time': '2024-01-01 15:00:00',
            'pnl': 1.5,
            'bars_held': 5,
            'entry_price': 100,
            'exit_price': 101.5
        },
        {
            'entry_time': '2024-01-15 12:00:00',
            'exit_time': '2024-01-15 16:00:00',
            'pnl': -0.8,
            'bars_held': 4,
            'entry_price': 101,
            'exit_price': 100.2
        },
        {
            'entry_time': '2024-02-01 09:00:00',
            'exit_time': '2024-02-01 14:00:00',
            'pnl': 2.1,
            'bars_held': 5,
            'entry_price': 100.5,
            'exit_price': 102.6
        }
    ]

def create_test_data():
    """Create test data with proper structure"""
    trades = create_test_trades()
    
    return pd.DataFrame({
        'trades': [trades],
        'timeframe': ['1h'],
        'return_total': [2.8],
        'win_rate': [66.7],
        'profit_factor': [1.5],
        'max_drawdown': [0.8],
        'total_trades': [3],
        'equity_curve': [pd.Series(
            [100, 101.5, 100.7, 102.8],
            index=pd.date_range('2024-01-01', periods=4)
        )]
    })

def test_monthly_returns():
    """Test monthly returns calculation"""
    results = create_test_data()
    
    fig = create_monthly_returns(results)
    
    # Verificaciones básicas
    assert fig is not None, "Monthly returns figure should not be None"
    assert len(fig.data) == 1, "Should have one bar trace"
    assert fig.data[0].type == "bar", "Should be a bar chart"
    
    # Verificar datos mensuales
    x_data = fig.data[0].x
    y_data = fig.data[0].y
    
    assert len(x_data) == 2, "Should have 2 months of data"
    assert "2024-01" in x_data, "Should include January 2024"
    assert "2024-02" in x_data, "Should include February 2024"
    
    # Verificar valores
    jan_return = y_data[list(x_data).index("2024-01")]
    feb_return = y_data[list(x_data).index("2024-02")]
    
    assert jan_return == 0.7, f"January return should be 0.7%, got {jan_return}%"
    assert feb_return == 2.1, f"February return should be 2.1%, got {feb_return}%"
    
    return "Monthly returns test passed"

def test_create_trade_distribution():
    """Test trade distribution chart creation"""
    results = create_test_data()
    
    fig = create_trade_distribution(results)
    assert fig is not None, "Trade distribution figure should not be None"
    assert len(fig.data) == 1, "Should have one histogram trace"
    assert fig.data[0].type == "histogram", "Should be a histogram"
    
    return "Trade distribution test passed"

def test_all_charts():
    """Test all chart creation functions"""
    results = create_test_data()
    
    monthly_rets = create_monthly_returns(results)
    trade_dist = create_trade_distribution(results)
    regime_perf = create_regime_performance(results)
    
    assert all(fig is not None for fig in [monthly_rets, trade_dist, regime_perf]), \
        "All charts should be created successfully"
    
    return "All charts test passed"

def test_dashboard_creation():
    """Test dashboard creation"""
    app = create_dashboard(port=8051)
    assert app is not None, "Dashboard app should not be None"
    return "Dashboard creation test passed"

def test_dashboard_components():
    """Test all dashboard components"""
    app = create_dashboard(port=8051)
    
    # Get all component IDs recursively
    def get_component_ids(layout):
        ids = []
        if hasattr(layout, 'children'):
            if isinstance(layout.children, list):
                for child in layout.children:
                    ids.extend(get_component_ids(child))
            else:
                ids.extend(get_component_ids(layout.children))
        if hasattr(layout, 'id'):
            ids.append(layout.id)
        return ids
    
    component_ids = get_component_ids(app.layout)
    
    # Verify required components exist
    required_components = [
        'strategy-selector',
        'date-picker-range',
        'timeframe-selector',
        'comparison-chart'
    ]
    
    for component in required_components:
        assert component in component_ids, f"Missing component: {component}"
    
    return "Dashboard components test passed"

def test_callbacks():
    """Test callback functionality"""
    app = create_dashboard(port=8051)
    
    # Get all registered callbacks and handle composite callbacks
    callbacks = []
    for key, value in app.callback_map.items():
        if isinstance(key, str):
            callbacks.append(key)
        else:
            # Handle tuple keys (multiple outputs)
            try:
                # Extract individual callbacks from composite string
                composite_callbacks = str(key).split('...')
                callbacks.extend([cb.strip() for cb in composite_callbacks if cb.strip()])
            except Exception as e:
                print(f"Error processing callback {key}: {e}")
    
    # Clean up callback strings
    callbacks = [
        cb.replace("'", "").replace('"', "").strip()
        for cb in callbacks
    ]
    
    # Print available callbacks for debugging
    print("\nAvailable callbacks:")
    for cb in callbacks:
        print(f"  - {cb}")
    
    # Define essential callback patterns instead of exact matches
    essential_patterns = [
        'trade-distribution',
        'monthly-returns',
        'regime-performance'
    ]
    
    # Check for patterns in callbacks
    for pattern in essential_patterns:
        matching_callbacks = [cb for cb in callbacks if pattern in cb]
        assert any(matching_callbacks), f"No callbacks found matching pattern: {pattern}"
        print(f"\nFound callbacks for {pattern}:")
        for cb in matching_callbacks:
            print(f"  - {cb}")
    
    # Verify minimum number of callbacks
    min_callbacks = 3
    assert len(callbacks) >= min_callbacks, \
        f"Expected at least {min_callbacks} callbacks, found {len(callbacks)}"
    
    return "Callback registration test passed"

def main():
    """Run all tests with proper module reloading"""
    print("\nRunning dashboard tests...")
    
    # First verify imports
    print("\nVerifying imports...")
    reload_critical_modules()
    import_result = test_imports()
    if "failed" in import_result:
        print(f"❌ {import_result}")
        return 1
    
    print(f"✅ {import_result}")
    
    # Ensure clean state
    if 'app' in globals():
        del globals()['app']
    
    tests = [
        test_monthly_returns,
        test_create_trade_distribution,
        test_all_charts,
        test_dashboard_creation,
        test_dashboard_components,
        test_callbacks
    ]
    
    success = True
    for test in tests:
        try:
            print(test())
        except Exception as e:
            print(f"❌ {test.__name__} failed: {str(e)}")
            success = False
    
    print("\nAll tests completed")
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
