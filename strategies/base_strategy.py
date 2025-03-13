from typing import Dict, Any
import pandas as pd

class BaseStrategy:
    def __init__(self):
        self.config = {}
        self.data = None
        self.risk_profiles = {
            'conservative': {
                'position_size': {'min': 0.02, 'max': 0.04},
                'leverage': {'min': 1, 'max': 3},
                'stop_loss': 0.02,
                'take_profit': 0.04
            },
            'moderate': {
                'position_size': {'min': 0.03, 'max': 0.06},
                'leverage': {'min': 2, 'max': 5},
                'stop_loss': 0.03,
                'take_profit': 0.06
            },
            'aggressive': {
                'position_size': {'min': 0.05, 'max': 0.08},
                'leverage': {'min': 3, 'max': 8},
                'stop_loss': 0.04,
                'take_profit': 0.08
            },
            'hybrid': {
                'position_size': {'min': 0.03, 'max': 0.07},
                'leverage': {'min': 2, 'max': 6},
                'stop_loss': 0.03,
                'take_profit': 0.07
            }
        }

    def set_config(self, config: Dict[str, Any]) -> None:
        self.config = config

    def set_risk_profile(self, profile_name: str) -> None:
        """Establish risk parameters based on profile"""
        if profile_name in self.risk_profiles:
            self.config.update(self.risk_profiles[profile_name])
        
    def run(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run strategy backtest
        
        Args:
            data: OHLCV data as pandas DataFrame
            
        Returns:
            Dict with backtest results
        """
        self.data = data
        return {
            'return_total': 0.0,
            'win_rate': 0.0, 
            'max_drawdown': 0.0,
            'profit_factor': 1.0,
            'total_trades': 0
        }

    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        required_fields = {
            'position_size': dict,
            'leverage': dict,
            'risk_profile': str
        }
        
        return all(
            isinstance(self.config.get(field), type_)
            for field, type_ in required_fields.items()
        )

    def set_risk_parameters(self) -> None:
        """Set risk parameters based on profile"""
        if not hasattr(self, 'config'):
            self.config = {}
            
        if 'risk_profile' not in self.config:
            self.config['risk_profile'] = 'moderate'
