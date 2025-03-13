import pandas as pd
import numpy as np
import ccxt
import ta
import time
import os  # Añadido para corregir el error
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from strategies.fixed_strategy import FixedStrategy
import matplotlib.pyplot as plt
import json
from utils.data_cache import DataCache
from utils.market_analyzer import MarketAnalyzer, MarketRegime
from strategies.optimized_strategy import OptimizedStrategy
from strategies.enhanced_strategy import EnhancedStrategy
from utils.backtest_debug import BacktestDebugger
from utils.signal_generator import BacktestSignalGenerator
from utils.debug_logger import DebugLogger
from utils.signal_manager import BacktestSignalManager

class BacktestFixedStrategy:
    def __init__(self, config=None):
        """Initialize backtest with configuration"""
        self.config = config or {}
        self.signal_generator = BacktestSignalGenerator(debug=self.config.get('debug', False))
        self.debug = self.config.get('debug', False)
        self.initial_balance = float(self.config.get('initial_balance', 1000.0))  # Explicit float conversion
        self.gross_profit = 0
        self.gross_loss = 0
        self.net_profit = 0
        self.profit_factor = 1.0
        self.win_rate = 0
        self.avg_trade_pnl = 0
        self.max_drawdown = 0
        
    def validate_config(self):
        """Validate configuration parameters"""
        if not self.config:
            self.config = {}
            
        # Ensure RSI parameters exist and are valid
        if 'rsi' not in self.config:
            self.config['rsi'] = {}
        self.config['rsi'].setdefault('window', 14)
        self.config['rsi'].setdefault('oversold', 30)
        self.config['rsi'].setdefault('overbought', 70)
        
        # Ensure EMA parameters exist and are valid
        if 'ema' not in self.config:
            self.config['ema'] = {}
        self.config['ema'].setdefault('short', 9)
        self.config['ema'].etdefault('long', 21)
        
        # Convert parameters to proper types
        self.config['rsi']['window'] = int(self.config['rsi']['window'])
        self.config['rsi']['oversold'] = float(self.config['rsi']['oversold'])
        self.config['rsi']['overbought'] = float(self.config['rsi']['overbought'])
        self.config['ema']['short'] = int(self.config['ema']['short'])
        self.config['ema']['long'] = int(self.config['ema']['long'])

    def reset_state(self):
        """Reset all strategy state variables"""
        self.trades = []
        self.current_position = 0
        self.entry_price = 0
        self.entry_index = None
        self.equity = 100.0
        self.peak_equity = 100.0
        self.total_trades = 0
        self.winning_trades = 0

    def _log_signal_condition(self, index, rsi_val, ema_short_val, ema_long_val, price):
        """Log detailed signal conditions for debugging"""
        if self.debug:
            self.signal_log.append({
                'index': index,
                'rsi': rsi_val,
                'ema_short': ema_short_val,
                'ema_long': ema_long_val,
                'price': price,
                'rsi_oversold': self.config.get('rsi', {}).get('oversold', 30),
                'rsi_overbought': self.config.get('rsi', {}).get('overbought', 70)
            })

    def run(self, data):
        """Run backtest with improved trade data structure"""
        try:
            # Initialize tracking variables with explicit float types
            trades = []
            equity_curve = pd.Series(float(self.initial_balance), index=data.index)
            current_equity = float(self.initial_balance)
            position = None
            max_equity = float(self.initial_balance)

            # Usar fechas reales del DataFrame de datos
            available_dates = data.index

            # Asegurarse de que tenemos suficientes datos
            if len(available_dates) < 6:
                print("Insufficient data for backtesting")
                return self._get_default_results()

            # Process each bar usando fechas reales
            for i in range(0, len(available_dates)-5, 20):  # Step 20, hold 5 bars
                if i+5 >= len(available_dates):
                    break
                    
                # Usar fechas reales del índice
                entry_time = available_dates[i]
                exit_time = available_dates[i+5]
                
                entry_price = float(data['close'].iloc[i])
                exit_price = float(data['close'].iloc[i+5])
                
                # Calcular PnL como porcentaje
                pnl = ((exit_price - entry_price) / entry_price) * 100
                
                trade = {
                    'entry_time': entry_time,
                    'exit_time': exit_time,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': float(pnl),
                    'bars_held': 5
                }
                trades.append(trade)

                # Update equity curve
                current_equity *= (1 + pnl/100)
                equity_curve[exit_time] = current_equity
                max_equity = max(max_equity, current_equity)

            # Calculate final statistics
            if trades:
                win_rate = (sum(1 for t in trades if t['pnl'] > 0) / len(trades) * 100)
                total_return = ((current_equity - self.initial_balance) / self.initial_balance) * 100
                profit_factor = self._calculate_profit_factor(trades)
                max_drawdown = self._calculate_max_drawdown(equity_curve)
            else:
                win_rate = 0
                total_return = 0
                profit_factor = 1
                max_drawdown = 0

            return {
                'return_total': total_return,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'max_drawdown': max_drawdown,
                'total_trades': len(trades),
                'trades': trades,
                'equity_curve': equity_curve
            }

        except Exception as e:
            print(f"Error in backtest: {str(e)}")
            return self._get_default_results()

    def _calculate_win_rate(self, trades):
        """Calculate win rate from trades"""
        if not trades:
            return 0.0
        winning_trades = sum(1 for t in trades if t['pnl'] > 0)
        return (winning_trades / len(trades)) * 100
    
    def _calculate_profit_factor(self, trades):
        """Calculate profit factor from trades"""
        if not trades:
            return 1.0
            
        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 1.0
        
        return total_profit / total_loss

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown from equity curve"""
        rolling_max = equity_curve.expanding().max()
        drawdowns = (equity_curve - rolling_max) / rolling_max * 100
        return abs(drawdowns.min())

    def _get_default_results(self):
        """Return default results when backtest fails"""
        default_equity = pd.Series(self.initial_balance)
        return {
            'return_total': 0,
            'win_rate': 0,
            'profit_factor': 1,
            'max_drawdown': 0,
            'total_trades': 0,
            'trades': [],
            'equity_curve': pd.Series(self.initial_balance, index=[pd.Timestamp.now()])
        }

    def _generate_signal(self, current_rsi, prev_rsi, ema_short, ema_long):
        """Generate trading signal based on conditions"""
        rsi_os = self.config.get('rsi', {}).get('oversold', 30)
        rsi_ob = self.config.get('rsi', {}).get('overbought', 70)
        
        # Long signal conditions
        if (current_rsi < rsi_os and 
            current_rsi > prev_rsi and  # RSI turning up
            ema_short > ema_long):  # Uptrend confirmation
            return 1
            
        # Short signal conditions
        elif (current_rsi > rsi_ob and
              current_rsi < prev_rsi and  # RSI turning down
              ema_short < ema_long):  # Downtrend confirmation
            return -1
                        
        return 0

    def _print_debug_summary(self):
        """Print detailed debug information"""
        if not hasattr(self, 'debug_data'):
            return
            
        rsi = self.debug_data['rsi']
        signals = self.debug_data['signals']
        
        print("\nStrategy Run Summary:")
        print(f"RSI range: {rsi.min():.2f} to {rsi.max():.2f}")
        print(f"RSI crosses above 70: {len(rsi[rsi > 70])}")
        print(f"Long signals generated: {len(signals[signals == 1])}")
        print(f"Short signals generated: {len(signals[signals == -1])}")
        print(f"Total trades: {len(self.trades)}")

    def _process_entry(self, direction, price, index):
        """Process trade entry"""
        self.current_position = direction
        self.entry_price = price
        self.entry_index = index

    def _process_exit(self, price, index):
        """Process trade exit"""
        if self.current_position == 0:
            return
            
        # Calculate trade P&L
        pnl = ((price - self.entry_price) / self.entry_price * 100) * self.current_position
        
        # Store trade
        self.trades.append({
            'entry_price': self.entry_price,
            'exit_price': price,
            'entry_index': self.entry_index,
            'exit_index': index,
            'direction': self.current_position,
            'pnl': pnl,
            'bars_held': index - self.entry_index
        })
        
        # Update statistics
        self.total_trades += 1
        if pnl > 0:
            self.winning_trades += 1
            self.gross_profit += pnl
        else:
            self.gross_loss += abs(pnl)
            
        # Reset position
        self.current_position = 0
        self.entry_price = 0
        self.entry_index = None

    def _calculate_results(self):
        """Calculate backtest results"""
        if not self.trades:
            return self._empty_result()
            
        return {
            'return_total': self.equity - 100,
            'win_rate': (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0,
            'profit_factor': self.gross_profit / self.gross_loss if self.gross_loss > 0 else 1,
            'max_drawdown': ((self.peak_equity - self.equity) / self.peak_equity * 100) if self.peak_equity > self.equity else 0,
            'total_trades': self.total_trades,
            'trades': self.trades
        }

    def _empty_result(self):
        """Return empty result structure"""
        return {
            'return_total': 0.0,
            'win_rate': 0.0,
            'profit_factor': 1.0,
            'max_drawdown': 0.0,
            'total_trades': 0,
            'trades': []
        }

    def generate_report(self):
        """Generate backtest report with complete metrics"""
        if not hasattr(self, 'trades') or not self.trades:
            print("No trades to generate report")
            return
            
        # Create continuous equity curve
        dates = pd.date_range(
            start=self.trades[0]['entry_time'],
            end=self.trades[-1]['exit_time'],
            freq='D'
        )
        equity_curve = pd.Series(self.initial_balance, index=dates)
        
        # Update equity curve with actual trade impacts
        for trade in self.trades:
            mask = equity_curve.index >= trade['exit_time']
            equity_curve[mask] *= (1 + trade['pnl']/100)
        
        # Generate analysis tables
        monthly_returns = self._calculate_monthly_returns(equity_curve)
        equity_table = self._generate_equity_table(equity_curve)
        
        # Save tables
        report_dir = f"/home/panal/Documents/dashboard-trading/reports/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(report_dir, exist_ok=True)
        
        monthly_returns.to_csv(os.path.join(report_dir, 'monthly_returns.csv'), index=False)
        equity_table.to_csv(os.path.join(report_dir, 'equity_curve.csv'), index=False)
        
        # Add tables to report data
        report_data = self._prepare_report_data()
        report_data['monthly_returns'] = monthly_returns.to_dict('records')
        report_data['equity_curve'] = equity_table.to_dict('records')

    def _create_performance_chart(self, report_dir):
        """Create performance visualization chart"""
        if not self.trades:
            return

        # Create equity curve
        equity = pd.Series(1.0)
        for trade in self.trades:
            equity = equity * (1 + trade['pnl']/100)

        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(equity.values, label='Equity Curve')
        plt.title('Backtest Performance')
        plt.xlabel('Trade Number')
        plt.ylabel('Equity (starting at 1.0)')
        plt.grid(True)
        plt.legend()

        # Save chart
        chart_file = os.path.join(report_dir, 'performance_chart.png')
        plt.savefig(chart_file)
        plt.close()

    def generate_signals(self, data):
        """Generate trading signals using configured signal generator"""
        if self.signal_generator is None:
            self._initialize_signal_generator()
            
        # Generate signals using signal generator
        signals, rsi, ema_short, ema_long = self.signal_generator.generate_signals(data, self.config)
        
        # Store indicators for analysis
        self.indicators = {
            'rsi': rsi,
            'ema_short': ema_short,
            'ema_long': ema_long
        }
                
        return signals

    def _calculate_profit_factor(self, trades):
        """Calculate profit factor from trades"""
        if not trades:
            return 1.0
            
        total_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] < 0))
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 1.0
        
        return total_profit / total_loss

    def _calculate_max_drawdown(self, equity_curve):
        """Calculate maximum drawdown percentage"""
        peaks = equity_curve.cummax()
        drawdowns = (equity_curve - peaks) / peaks * 100
        return abs(drawdowns.min()) if len(drawdowns) > 0 else 0

    def _calculate_monthly_returns(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Calculate and format monthly returns with detailed statistics"""
        try:
            # Ensure data is sorted and indexed properly
            equity_curve = equity_curve.sort_index()
            
            # Calculate monthly returns
            monthly_equity = equity_curve.resample('M').last().dropna()
            monthly_returns = monthly_equity.pct_change().dropna()
            
            # Create detailed monthly returns table
            monthly_stats = pd.DataFrame({
                'Year': monthly_returns.index.year,
                'Month': monthly_returns.index.month,
                'Initial Equity': monthly_equity.shift(1).round(2),
                'Final Equity': monthly_equity.round(2),
                'Return (%)': (monthly_returns * 100).round(2),
                'Cumulative Return (%)': ((monthly_equity / monthly_equity.iloc[0] - 1) * 100).round(2)
            }).reset_index()
            
            # Format dates properly
            monthly_stats['Period'] = monthly_stats['index'].dt.strftime('%Y-%m')
            monthly_stats = monthly_stats.drop('index', axis=1)
            
            # Calculate summary statistics
            total_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100).round(2)
            avg_monthly_return = monthly_returns.mean() * 100
            monthly_std = monthly_returns.std() * 100
            
            # Add summary row
            summary = pd.DataFrame([{
                'Period': 'Total',
                'Year': '',
                'Month': '',
                'Initial Equity': equity_curve.iloc[0].round(2),
                'Final Equity': equity_curve.iloc[-1].round(2),
                'Return (%)': total_return,
                'Cumulative Return (%)': total_return
            }])
            
            monthly_stats = pd.concat([monthly_stats, summary], ignore_index=True)
            
            return monthly_stats
            
        except Exception as e:
            print(f"Error calculating monthly returns: {str(e)}")
            return pd.DataFrame()

    def _generate_equity_table(self, equity_curve: pd.Series) -> pd.DataFrame:
        """Generate comprehensive equity curve analysis table"""
        try:
            # Calculate daily metrics
            daily_returns = equity_curve.pct_change().fillna(0)
            cum_returns = (1 + daily_returns).cumprod()
            high_water_mark = equity_curve.cummax()
            drawdowns = ((equity_curve - high_water_mark) / high_water_mark) * 100
            
            # Calculate rolling metrics
            rolling_vol = daily_returns.rolling(window=20).std() * np.sqrt(252) * 100
            rolling_return = daily_returns.rolling(window=20).mean() * 252 * 100
            
            # Create detailed equity table
            equity_table = pd.DataFrame({
                'Date': equity_curve.index,
                'Equity': equity_curve.round(2),
                'Daily Return (%)': (daily_returns * 100).round(2),
                'Cumulative Return (%)': ((cum_returns - 1) * 100).round(2),
                'Drawdown (%)': drawdowns.round(2),
                'High Water Mark': high_water_mark.round(2),
                'Rolling Annual Vol (%)': rolling_vol.round(2),
                'Rolling Annual Return (%)': rolling_return.round(2)
            })
            
            # Add daily drawdown duration
            equity_table['Drawdown Duration'] = 0
            duration = 0
            
            for i in range(len(equity_table)):
                if equity_table['Drawdown (%)'].iloc[i] < 0:
                    duration += 1
                else:
                    duration = 0
                equity_table.loc[equity_table.index[i], 'Drawdown Duration'] = duration
            
            return equity_table
            
        except Exception as e:
            print(f"Error generating equity table: {str(e)}")
            return pd.DataFrame()

    def _calculate_drawdown_series(self, equity_curve):
        """Calculate running drawdown series"""
        highs = equity_curve.cummax()
        drawdowns = ((equity_curve - highs) / highs) * 100
        return drawdowns

    def _generate_performance_tables(self, equity_curve: pd.Series) -> Dict[str, pd.DataFrame]:
        """Generate key performance tables"""
        try:
            # 1. Trade Performance Summary
            trade_summary = pd.DataFrame({
                'Metric': [
                    'Total Trades',
                    'Winning Trades',
                    'Losing Trades',
                    'Win Rate (%)',
                    'Average Win (%)',
                    'Average Loss (%)',
                    'Largest Win (%)',
                    'Largest Loss (%)',
                    'Average Holding Period',
                    'Profit Factor'
                ],
                'Value': [
                    len(self.trades),
                    len([t for t in self.trades if t['pnl'] > 0]),
                    len([t for t in self.trades if t['pnl'] < 0]),
                    self.win_rate,
                    np.mean([t['pnl'] for t in self.trades if t['pnl'] > 0]).round(2),
                    np.mean([t['pnl'] for t in self.trades if t['pnl'] < 0]).round(2),
                    max([t['pnl'] for t in self.trades]).round(2),
                    min([t['pnl'] for t in self.trades]).round(2),
                    np.mean([t['bars_held'] for t in self.trades]).round(1),
                    self.profit_factor
                ]
            })

            # 2. Risk Metrics Table
            risk_metrics = pd.DataFrame({
                'Metric': [
                    'Max Drawdown (%)',
                    'Average Drawdown (%)',
                    'Longest Drawdown Duration',
                    'Sharpe Ratio',
                    'Sortino Ratio',
                    'Calmar Ratio',
                    'Recovery Factor',
                    'Risk-Adjusted Return'
                ],
                'Value': self._calculate_risk_metrics(equity_curve)
            })

            # 3. Trading Patterns Table
            patterns = self._analyze_trading_patterns()

            return {
                'trade_summary': trade_summary,
                'risk_metrics': risk_metrics,
                'trading_patterns': patterns
            }

        except Exception as e:
            print(f"Error generating performance tables: {str(e)}")
            return {}

    def _calculate_risk_metrics(self, equity_curve: pd.Series) -> list:
        """Calculate comprehensive risk metrics"""
        returns = equity_curve.pct_change().dropna()
        
        max_dd = self._calculate_max_drawdown(equity_curve)
        avg_dd = np.mean(self._calculate_drawdown_series(equity_curve))
        
        # Calculate advanced metrics
        sharpe = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0
        sortino = np.sqrt(252) * (returns.mean() / returns[returns < 0].std()) if len(returns[returns < 0]) > 0 else 0
        calmar = (returns.mean() * 252) / max_dd if max_dd != 0 else 0
        
        return [
            max_dd.round(2),
            avg_dd.round(2),
            self._get_longest_drawdown(equity_curve),
            sharpe.round(2),
            sortino.round(2),
            calmar.round(2),
            self._calculate_recovery_factor(equity_curve).round(2),
            (sharpe * (1 - max_dd/100)).round(2)
        ]

    def _analyze_trading_patterns(self) -> pd.DataFrame:
        """Analyze trading patterns and success rates"""
        patterns = []
        
        # Analyze trading patterns by time
        for hour in range(24):
            trades_in_hour = [t for t in self.trades if t['entry_time'].hour == hour]
            if trades_in_hour:
                win_rate = len([t for t in trades_in_hour if t['pnl'] > 0]) / len(trades_in_hour) * 100
                patterns.append({
                    'Pattern': f'Hour {hour:02d}',
                    'Trades': len(trades_in_hour),
                    'Win Rate (%)': win_rate.round(2),
                    'Avg PnL (%)': np.mean([t['pnl'] for t in trades_in_hour]).round(2)
                })
        
        return pd.DataFrame(patterns)

    def compare_strategies(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compare different trading strategies on the same data"""
        strategies = {
            'Fixed': FixedStrategy(self.config),
            'Optimized': OptimizedStrategy(self.config),
            'Enhanced': EnhancedStrategy(self.config)
        }
        
        results = []
        
        for name, strategy in strategies.items():
            try:
                # Run backtest with each strategy
                result = strategy.run(data)
                
                # Extract key metrics
                metrics = {
                    'Strategy': name,
                    'Total Return (%)': result['return_total'],
                    'Win Rate (%)': result['win_rate'],
                    'Profit Factor': result['profit_factor'],
                    'Max Drawdown (%)': result['max_drawdown'],
                    'Total Trades': result['total_trades']
                }
                
                results.append(metrics)
                
            except Exception as e:
                print(f"Error running {name} strategy: {str(e)}")
                
        comparison = pd.DataFrame(results)
        comparison.set_index('Strategy', inplace=True)
        
        return comparison

    def generate_report(self):
        """Generate enhanced backtest report"""
        if not hasattr(self, 'trades') or not self.trades:
            print("No trades to generate report")
            return

        # Create equity curve
        equity_curve = self._create_equity_curve()
        
        # Generate performance tables
        performance_tables = self._generate_performance_tables(equity_curve)
        
        # Compare strategies
        strategy_comparison = self.compare_strategies(self.data)
        
        # Save results
        report_dir = f"/home/panal/Documents/dashboard-trading/reports/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(report_dir, exist_ok=True)
        
        # Save all tables
        for name, table in performance_tables.items():
            table.to_csv(os.path.join(report_dir, f'{name}.csv'), index=False)
        
        strategy_comparison.to_csv(os.path.join(report_dir, 'strategy_comparison.csv'))
        
        # Create report data
        report_data = self._prepare_report_data()
        report_data.update({
            'performance_tables': {
                k: v.to_dict('records') for k, v in performance_tables.items()
            },
            'strategy_comparison': strategy_comparison.to_dict('records')
        })
        
        # Save report
        with open(os.path.join(report_dir, 'backtest_report.json'), 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return report_data