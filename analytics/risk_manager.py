from typing import Dict, Optional, List
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

class RiskManager:
    def __init__(self, config: Dict):
        self.config = self._prepare_config(config)
        self.current_drawdown = 0
        self.peak_balance = float(self.config['initial_balance'])
        self.trades_today = 0
        self.daily_loss = 0
        self.last_reset = datetime.now().date()
        self.trade_history = []  # Almacena últimas operaciones para análisis de racha
        self.consecutive_losses = 0  # Contador de pérdidas consecutivas
        
    def _prepare_config(self, config: Dict) -> Dict:
        """Ensure configuration has required fields with proper types"""
        # Set default values for missing required fields
        if 'initial_balance' not in config:
            config['initial_balance'] = 100000  # Default starting balance
            
        required_fields = {
            'initial_balance': float(config['initial_balance']),
            'max_drawdown': float(config.get('max_drawdown', 0.05)),
            'warning_drawdown': float(config.get('warning_drawdown', 0.035)),
            'daily_loss_limit': float(config.get('daily_loss_limit', 0.02))
        }
        
        # Añadir configuración de control de riesgo avanzada
        if 'risk_controls' not in config:
            config['risk_controls'] = {
                'position_reduction': {
                    'start_at_drawdown': 0.02,
                    'reduction_rate': 2.0,
                    'min_size': 0.2
                },
                'consecutive_losses': {
                    'max_full_size': 3,     # Máximo de pérdidas consecutivas antes de reducir tamaño
                    'reduction_factor': 0.5  # Factor de reducción después de pérdidas consecutivas
                },
                'volatility_scaling': True,  # Ajustar tamaño por volatilidad del mercado
                'profit_reinvestment': {
                    'enabled': True,         # Reinvertir parte de ganancias en tamaño
                    'scale_factor': 0.2      # 20% de ganancias reinvertidas en aumento de tamaño
                }
            }
            
        # Ajustar configuración de tamaño de posición según perfil de riesgo
        if 'risk_profile' in config:
            # FIX: Check if risk_profile is a string or dictionary
            if isinstance(config['risk_profile'], str):
                # When risk_profile is a string ('conservative', 'moderate', 'aggressive')
                profile_configs = {
                    'conservative': {'min': 0.02, 'max': 0.04, 'leverage_min': 5, 'leverage_max': 15},
                    'moderate': {'min': 0.03, 'max': 0.06, 'leverage_min': 10, 'leverage_max': 20},
                    'aggressive': {'min': 0.05, 'max': 0.08, 'leverage_min': 15, 'leverage_max': 30}
                }
                profile = config['risk_profile']
                if profile in profile_configs:
                    config['position_size'] = {
                        'min': profile_configs[profile]['min'],
                        'max': profile_configs[profile]['max']
                    }
                    config['leverage_range'] = {
                        'min': profile_configs[profile]['leverage_min'],
                        'max': profile_configs[profile]['leverage_max']
                    }
            else:
                # When risk_profile is already a dictionary with configurations
                if 'position_size' in config['risk_profile']:
                    config['position_size'] = config['risk_profile']['position_size']
                if 'leverage' in config['risk_profile']:
                    config['leverage_range'] = config['risk_profile']['leverage']
        else:
            # Valores por defecto si no hay perfil específico
            config['position_size'] = {'min': 0.03, 'max': 0.06}
            config['leverage_range'] = {'min': 10, 'max': 20}
            
        return {**config, **required_fields}
        
    def can_trade(self, current_balance: float) -> bool:
        """Verificar si podemos operar con controles avanzados"""
        # Reset daily stats if needed
        today = datetime.now().date()
        if today > self.last_reset:
            self._reset_daily_stats()
            
        # Actualizar drawdown
        self.peak_balance = max(self.peak_balance, current_balance)
        self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
        
        # Verificar límite de drawdown crítico
        if self.current_drawdown >= self.config['max_drawdown']:
            print(f"Trading detenido - Drawdown crítico: {self.current_drawdown:.1%}")
            return False
            
        # Verificar pérdida diaria límite
        if self.daily_loss >= self.config['daily_loss_limit']:
            print(f"Trading detenido - Pérdida diaria límite alcanzada: {self.daily_loss:.1%}")
            return False
            
        # Verificar rachas perdedoras críticas
        if self.consecutive_losses >= 5:
            # Obtener límite de pérdidas consecutivas desde configuración
            max_consecutive = self.config.get('risk_controls', {}).get('consecutive_losses', {}).get('max_streak', 7)
            if self.consecutive_losses >= max_consecutive:
                print(f"Trading detenido - Racha perdedora crítica: {self.consecutive_losses} pérdidas consecutivas")
                return False
        
        # Verificar rendimiento reciente (últimos 10 trades)
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            win_rate = sum(1 for t in recent_trades if t > 0) / len(recent_trades)
            
            # Si rendimiento es malo y estamos en drawdown, pausar trading
            if win_rate < 0.3 and self.current_drawdown > 0.03:
                print(f"Trading pausado - Bajo rendimiento reciente (Win rate: {win_rate:.1%}) durante drawdown")
                return False
        
        return True
        
    def calculate_position_size(self, current_balance: float, current_drawdown: float, 
                              market_volatility: float = None, winning_streak: int = 0) -> float:
        """
        Calcular tamaño de posición dinámico con múltiples factores
        
        Args:
            current_balance: Balance actual
            current_drawdown: Drawdown actual (0-1)
            market_volatility: Volatilidad relativa del mercado (0-1, opcional)
            winning_streak: Número de operaciones ganadoras consecutivas
            
        Returns:
            float: Tamaño de posición como fracción del capital (0-1)
        """
        # Obtener límites de tamaño de posición del config
        min_size = self.config['position_size'].get('min', 0.03)
        max_size = self.config['position_size'].get('max', 0.06)
        
        # Factor base: comienza con el tamaño medio entre min y max
        position_size = (min_size + max_size) / 2
        
        # Factor 1: Ajuste por drawdown
        if current_drawdown > 0.01:  # Si hay drawdown > 1%
            # Reducción lineal según drawdown
            drawdown_factor = max(0.5, 1 - (current_drawdown * 5))  # Reducir hasta 50% con 10% DD
            position_size *= drawdown_factor
            
        # Factor 2: Ajuste por volatilidad del mercado
        if market_volatility is not None and self.config.get('risk_controls', {}).get('volatility_scaling', True):
            # Volatilidad alta = tamaño menor, volatilidad baja = tamaño mayor
            vol_factor = 1.0
            
            if market_volatility > 0.03:  # Volatilidad alta (>3%)
                vol_factor = 0.7  # Reducir tamaño al 70%
            elif market_volatility < 0.01:  # Volatilidad baja (<1%)
                vol_factor = 1.2  # Aumentar tamaño al 120%
                
            position_size *= vol_factor
                
        # Factor 3: Ajuste por racha de resultados
        if self.consecutive_losses > 0:
            # Reducción por pérdidas consecutivas
            max_full_size = self.config.get('risk_controls', {}).get('consecutive_losses', {}).get('max_full_size', 3)
            
            if self.consecutive_losses > max_full_size:
                # Aplicar reducción progresiva
                reduction = self.config.get('risk_controls', {}).get('consecutive_losses', {}).get('reduction_factor', 0.5)
                position_size *= reduction
        
        # Factor 4: Escala con rachas ganadoras (opcional)
        if winning_streak > 2 and self.config.get('risk_controls', {}).get('profit_reinvestment', {}).get('enabled', False):
            # Escalar incremento con rachas ganadoras, máximo 30% adicional
            scale_factor = self.config.get('risk_controls', {}).get('profit_reinvestment', {}).get('scale_factor', 0.2)
            increase = min(0.3, winning_streak * 0.1 * scale_factor)  # Máx 30% adicional
            position_size *= (1 + increase)
        
        # Asegurar que estamos dentro de límites
        position_size = min(max_size, max(min_size, position_size))
        
        return float(position_size)
    
    def calculate_leverage(self, current_balance: float, market_volatility: float = 0.0, 
                          trade_risk: float = 0.0, regime: str = 'unknown') -> int:
        """
        Calcula el apalancamiento óptimo con lógica avanzada
        
        Args:
            current_balance: Balance actual
            market_volatility: Índice de volatilidad del mercado (0-1)
            trade_risk: Nivel de riesgo estimado de la operación (0-1)
            regime: Régimen de mercado detectado ('trending_up', 'trending_down', 'ranging', 'volatile')
            
        Returns:
            int: Factor de apalancamiento óptimo
        """
        # Obtener rango de apalancamiento del config
        min_leverage = self.config.get('leverage_range', {}).get('min', 10)
        max_leverage = self.config.get('leverage_range', {}).get('max', 20)
        
        # Si no hay datos de volatilidad, usar un valor conservador
        if market_volatility <= 0:
            market_volatility = 0.5
        
        # Si no hay estimación de riesgo, usar valor medio
        if trade_risk <= 0:
            trade_risk = 0.5
        
        # Factor base: apalancamiento disminuye con mayor volatilidad y riesgo
        risk_factor = (market_volatility + trade_risk) / 2
        base_leverage = max_leverage - risk_factor * (max_leverage - min_leverage)
        
        # Ajuste adicional por régimen de mercado
        regime_multipliers = {
            'trending_up': 1.2,    # Mayor apalancamiento en tendencia alcista
            'trending_down': 0.9,  # Menor apalancamiento en tendencia bajista
            'ranging': 1.0,        # Apalancamiento normal en rango
            'volatile': 0.7        # Apalancamiento reducido en volatilidad
        }
        
        # Aplicar modificador de régimen
        regime_multiplier = regime_multipliers.get(regime, 1.0)
        adjusted_leverage = base_leverage * regime_multiplier
        
        # Ajuste por drawdown actual
        if self.current_drawdown > 0.02:  # Si hay drawdown > 2%
            drawdown_factor = max(0.7, 1 - self.current_drawdown * 5)  # Reducir hasta 70% con 6% DD
            adjusted_leverage *= drawdown_factor
            
        # Asegurar que estamos dentro de límites y redondear
        final_leverage = max(min_leverage, min(max_leverage, adjusted_leverage))
        
        return int(round(final_leverage))
    
    def update_trade_history(self, pnl: float) -> None:
        """
        Actualizar historial de trades y rachas
        
        Args:
            pnl: Profit/Loss de la operación (en % o valor monetario)
        """
        # Añadir resultado a historial
        self.trade_history.append(pnl)
        
        # Mantener un máximo de 30 trades en historial
        if len(self.trade_history) > 30:
            self.trade_history.pop(0)
            
        # Actualizar rachas
        if pnl > 0:
            # Trade ganador
            self.consecutive_losses = 0
        else:
            # Trade perdedor
            self.consecutive_losses += 1
            
        # Actualizar estadísticas diarias
        self.trades_today += 1
        if pnl < 0:
            self.daily_loss += abs(pnl)
    
    def _reset_daily_stats(self):
        """Resetear estadísticas diarias"""
        self.daily_loss = 0
        self.trades_today = 0
        self.last_reset = datetime.now().date()
        
    def get_risk_stats(self) -> Dict:
        """
        Obtener estadísticas de gestión de riesgo para reporting
        
        Returns:
            Dict: Estadísticas de riesgo actuales
        """
        return {
            'current_drawdown': self.current_drawdown * 100,  # Convertir a porcentaje
            'daily_loss': self.daily_loss,
            'trades_today': self.trades_today,
            'consecutive_losses': self.consecutive_losses,
            'win_rate_recent': self._calculate_recent_win_rate(),
            'max_drawdown_limit': self.config['max_drawdown'] * 100  # Convertir a porcentaje
        }
        
    def _calculate_recent_win_rate(self) -> float:
        """Calcula win rate de operaciones recientes"""
        if not self.trade_history:
            return 0.0
            
        # Usar últimas 10 operaciones o todas si hay menos
        recent = self.trade_history[-min(10, len(self.trade_history)):]
        return sum(1 for t in recent if t > 0) / len(recent) * 100  # En porcentaje
