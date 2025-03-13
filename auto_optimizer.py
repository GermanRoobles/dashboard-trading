import optuna
from backtest.run_fixed_backtest import BacktestFixedStrategy
import json
import numpy as np

class StrategyOptimizer:
    def __init__(self, base_config, data):
        self.base_config = base_config
        self.data = data
        
    def objective(self, trial):
        # Create trial configuration
        config = self.base_config.copy()
        
        # Optimize scalping parameters
        config['models']['scalp'].update({
            'threshold': trial.suggest_float('scalp_threshold', 0.6, 0.9),
            'max_trades_per_hour': trial.suggest_int('scalp_max_trades', 2, 6)
        })
        
        # Optimize swing parameters
        config['models']['swing'].update({
            'threshold': trial.suggest_float('swing_threshold', 0.7, 0.95),
            'min_holding_time': trial.suggest_int('swing_holding', 4, 12)
        })
        
        # Optimize position sizing
        config['position_sizing']['scalp']['base_size'] = trial.suggest_float('scalp_size', 0.01, 0.05)
        config['position_sizing']['swing']['base_size'] = trial.suggest_float('swing_size', 0.03, 0.10)
        
        # Run backtest
        backtest = BacktestFixedStrategy(config)
        result = backtest.run(self.data)
        
        # Optimize for return while considering drawdown
        return result['return_total'] * (1 - result['max_drawdown']/100)
    
    def optimize(self, n_trials=100):
        study = optuna.create_study(direction='maximize')
        study.optimize(self.objective, n_trials=n_trials)
        
        # Get best parameters
        best_params = study.best_params
        best_config = self.base_config.copy()
        
        # Update config with best parameters
        best_config['models']['scalp'].update({
            'threshold': best_params['scalp_threshold'],
            'max_trades_per_hour': best_params['scalp_max_trades']
        })
        # ...rest of parameter updates...
        
        return best_config, study.best_value
