"""
Backtesting module for evaluating portfolio performance.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import copy

from .strategies import PortfolioStrategy, create_strategy
from .optimizer import PortfolioManager
from .scheduler import PortfolioScheduler

def normalize_datetime(dt):
    """Convert datetime to timezone-naive."""
    if pd.api.types.is_datetime64_dtype(dt):
        # For pandas datetime objects
        return pd.Timestamp(dt).to_pydatetime().replace(tzinfo=None)
    elif isinstance(dt, pd.Timestamp):
        # For pandas Timestamp objects
        return dt.to_pydatetime().replace(tzinfo=None)
    elif hasattr(dt, 'tzinfo') and dt.tzinfo is not None:
        # For timezone-aware datetime objects
        return dt.replace(tzinfo=None)
    return dt

class PortfolioBacktester:
    """Backtester for evaluating portfolio strategies."""
    
    def __init__(self, risk_profile: str):
        """Initialize the backtester."""
        self.portfolio_manager = PortfolioManager(risk_profile)
        self.risk_profile = risk_profile
        self.scheduler = PortfolioScheduler()
        self.results = {}
        self.logger = logging.getLogger(__name__)
        self.monthly_deals = {}  # Track deals by month to respect LowTurnoverStrategy limit
        
    def run_backtest(
        self,
        market_data: pd.DataFrame,
        initial_capital: float,
        start_date: datetime = datetime(2015, 1, 1),
        end_date: datetime = datetime(2024, 12, 31),
        training_end_date: datetime = datetime(2022, 12, 31),
        constraints: Optional[Dict] = None
    ) -> Dict:
        """Run backtest simulation."""
        try:
            # Normalize dates to avoid timezone issues
            start_date = normalize_datetime(start_date)
            end_date = normalize_datetime(end_date)
            training_end_date = normalize_datetime(training_end_date)
            
            # Normalize market_data index if it's DatetimeIndex
            if isinstance(market_data.index, pd.DatetimeIndex):
                market_data = market_data.copy()
                market_data.index = market_data.index.map(normalize_datetime)
            
            # Check for any remaining NA values and handle them
            if market_data.isna().any().any():
                self.logger.warning("NA values detected in market data, applying linear interpolation")
                
                # Handle NAs in price columns using linear interpolation
                price_columns = [col for col in market_data.columns if any(
                    suffix in col.lower() for suffix in ['_close', '_open', '_high', '_low', '_adj_close']
                )]
                
                for col in price_columns:
                    if market_data[col].isna().any():
                        market_data[col] = market_data[col].interpolate(method='linear')
                        # Handle edge cases with forward/backward fill
                        market_data[col] = market_data[col].fillna(method='ffill').fillna(method='bfill')
                
                # Recalculate returns columns based on interpolated prices
                for col in market_data.columns:
                    if '_close' in col:
                        symbol = col.split('_')[0]
                        returns_col = f"{symbol}_returns"
                        if returns_col in market_data.columns:
                            market_data[returns_col] = market_data[col].pct_change(fill_method=None)
                
                # Fill remaining NA values in returns with zeros
                returns_columns = [col for col in market_data.columns if '_returns' in col]
                for col in returns_columns:
                    market_data[col] = market_data[col].fillna(0)
                
                # Fill any other columns with forward/backward fill
                market_data = market_data.fillna(method='ffill').fillna(method='bfill')
                
                # As a last resort, fill any remaining NAs with zeros
                market_data = market_data.fillna(0)
                
                self.logger.info("NA handling complete in market data")
            
            # Initialize scheduler with training and evaluation periods
            self.scheduler = PortfolioScheduler(start_date, end_date, training_end_date)
            
            # Split run into training and evaluation phases
            training_results = self._run_training_phase(
                market_data, 
                initial_capital, 
                start_date,
                training_end_date
            )
            
            evaluation_results = self._run_evaluation_phase(
                market_data,
                training_results['portfolio']['total_value'][-1],
                training_end_date + timedelta(days=1),
                end_date,
                training_results['trained_strategy']
            )
            
            # Combine results
            combined_results = self._combine_results(training_results, evaluation_results)
            self.results = combined_results
            
            return combined_results
            
        except Exception as e:
            self.logger.error(f"Error in backtest: {str(e)}")
            raise
    
    def _run_training_phase(
        self, 
        market_data: pd.DataFrame, 
        initial_capital: float,
        start_date: datetime,
        end_date: datetime
    ) -> Dict:
        """Run the training phase of the backtest."""
        try:
            self.logger.info(f"Starting training phase from {start_date} to {end_date}")
            
            # Initialize portfolio
            portfolio = {
                'cash': initial_capital,
                'holdings': {},
                'total_value': [initial_capital],
                'dates': [start_date],
                'trades': []
            }
            
            # Get training days
            training_days = self.scheduler.get_training_days()
            
            # Run simulation for each training day
            for trading_day in training_days:
                # Normalize trading day
                trading_day = normalize_datetime(trading_day)
                
                # Filter market data up to current trading day
                current_data = market_data[market_data.index <= trading_day]
                
                if len(current_data) < 30:  # Need at least 30 days of data for analysis
                    continue
                
                # Prepare portfolio data
                portfolio_df = pd.DataFrame({
                    'date': portfolio['dates'],
                    'total_value': portfolio['total_value']
                })
                
                # Generate trading signals
                signals = self.portfolio_manager.strategy.generate_signals(
                    current_data, portfolio_df
                )
                
                # Calculate current portfolio state
                current_portfolio = self._calculate_portfolio_value(
                    portfolio, current_data, trading_day
                )
                
                # Generate trades based on signals
                trades = self._generate_trades(
                    current_portfolio, signals, current_data, trading_day
                )
                
                # Execute trades
                portfolio = self._execute_trades(
                    portfolio, trades, current_data, trading_day
                )
                
                # Record portfolio value
                portfolio_value = portfolio['cash']
                for symbol, quantity in portfolio['holdings'].items():
                    if quantity > 0 and f'{symbol}_close' in current_data.columns:
                        price = current_data[f'{symbol}_close'].iloc[-1]
                        portfolio_value += price * quantity
                
                portfolio['total_value'].append(portfolio_value)
                portfolio['dates'].append(trading_day)
            
            # Store trained strategy parameters
            trained_strategy = copy.deepcopy(self.portfolio_manager.strategy)
            
            # Calculate training metrics
            training_metrics = self._calculate_performance_metrics(portfolio)
            
            self.logger.info(f"Training phase completed with return: {training_metrics['total_return']:.2%}")
            
            return {
                'portfolio': portfolio,
                'metrics': training_metrics,
                'trained_strategy': trained_strategy
            }
            
        except Exception as e:
            self.logger.error(f"Error in training phase: {str(e)}")
            raise
    
    def _run_evaluation_phase(
        self, 
        market_data: pd.DataFrame, 
        initial_capital: float,
        start_date: datetime,
        end_date: datetime,
        trained_strategy: PortfolioStrategy
    ) -> Dict:
        """Run the evaluation phase of the backtest."""
        try:
            self.logger.info(f"Starting evaluation phase from {start_date} to {end_date}")
            
            # Clone the portfolio manager with the trained strategy
            evaluation_manager = PortfolioManager(self.risk_profile)
            evaluation_manager.strategy = trained_strategy
            
            # Initialize portfolio
            portfolio = {
                'cash': initial_capital,
                'holdings': {},
                'total_value': [initial_capital],
                'dates': [start_date],
                'trades': []
            }
            
            # Get evaluation days
            evaluation_days = self.scheduler.get_evaluation_days()
            
            # Run simulation for each evaluation day
            for trading_day in evaluation_days:
                # Normalize trading day
                trading_day = normalize_datetime(trading_day)
                
                # Filter market data up to current trading day
                current_data = market_data[market_data.index <= trading_day]
                
                if len(current_data) < 30:  # Need at least 30 days of data for analysis
                    continue
                
                # Prepare portfolio data
                portfolio_df = pd.DataFrame({
                    'date': portfolio['dates'],
                    'total_value': portfolio['total_value']
                })
                
                # Generate trading signals using trained strategy
                signals = trained_strategy.generate_signals(
                    current_data, portfolio_df
                )
                
                # Calculate current portfolio state
                current_portfolio = self._calculate_portfolio_value(
                    portfolio, current_data, trading_day
                )
                
                # Generate trades based on signals
                trades = self._generate_trades(
                    current_portfolio, signals, current_data, trading_day
                )
                
                # Execute trades
                portfolio = self._execute_trades(
                    portfolio, trades, current_data, trading_day
                )
                
                # Record portfolio value
                portfolio_value = portfolio['cash']
                for symbol, quantity in portfolio['holdings'].items():
                    if quantity > 0 and f'{symbol}_close' in current_data.columns:
                        price = current_data[f'{symbol}_close'].iloc[-1]
                        portfolio_value += price * quantity
                
                portfolio['total_value'].append(portfolio_value)
                portfolio['dates'].append(trading_day)
            
            # Calculate evaluation metrics
            evaluation_metrics = self._calculate_performance_metrics(portfolio)
            
            self.logger.info(f"Evaluation phase completed with return: {evaluation_metrics['total_return']:.2%}")
            
            return {
                'portfolio': portfolio,
                'metrics': evaluation_metrics
            }
            
        except Exception as e:
            self.logger.error(f"Error in evaluation phase: {str(e)}")
            raise
    
    def _combine_results(self, training_results: Dict, evaluation_results: Dict) -> Dict:
        """Combine training and evaluation results."""
        try:
            # Combine portfolios
            combined_portfolio = {
                'cash': evaluation_results['portfolio']['cash'],
                'holdings': evaluation_results['portfolio']['holdings'],
                'total_value': training_results['portfolio']['total_value'] + evaluation_results['portfolio']['total_value'][1:],
                'dates': training_results['portfolio']['dates'] + evaluation_results['portfolio']['dates'][1:],
                'trades': training_results['portfolio']['trades'] + evaluation_results['portfolio']['trades']
            }
            
            # Calculate overall metrics
            combined_metrics = self._calculate_performance_metrics(combined_portfolio)
            
            return {
                'portfolio': combined_portfolio,
                'training_metrics': training_results['metrics'],
                'evaluation_metrics': evaluation_results['metrics'],
                'combined_metrics': combined_metrics,
                'trained_strategy': training_results['trained_strategy']
            }
            
        except Exception as e:
            self.logger.error(f"Error combining results: {str(e)}")
            raise
            
    def _calculate_portfolio_value(
        self, portfolio: Dict, market_data: pd.DataFrame, date: datetime
    ) -> Dict:
        """Calculate current portfolio value and weights."""
        try:
            # Start with cash
            total_value = portfolio['cash']
            current_weights = {}
            
            # Add value of each holding
            for symbol, quantity in portfolio['holdings'].items():
                if quantity > 0 and f'{symbol}_close' in market_data.columns:
                    price = market_data[f'{symbol}_close'].iloc[-1]
                    value = price * quantity
                    total_value += value
                    current_weights[symbol] = value
            
            # Convert values to weights
            if total_value > 0:
                for symbol in current_weights:
                    current_weights[symbol] /= total_value
            
            return {
                'total_value': total_value,
                'weights': current_weights
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio value: {str(e)}")
            raise
    
    def _generate_trades(
        self, 
        current_portfolio: Dict, 
        target_weights: Dict[str, float], 
        market_data: pd.DataFrame,
        date: datetime
    ) -> List[Tuple[str, float]]:
        """Generate trades to rebalance the portfolio."""
        try:
            trades = []
            total_value = current_portfolio['total_value']
            current_weights = current_portfolio['weights']
            
            # Calculate trades needed for each symbol
            for symbol, target_weight in target_weights.items():
                current_weight = current_weights.get(symbol, 0.0)
                weight_diff = target_weight - current_weight
                
                # Skip small trades
                if abs(weight_diff) < 0.01:
                    continue
                
                # Calculate target value and quantity
                if f'{symbol}_close' in market_data.columns:
                    price = market_data[f'{symbol}_close'].iloc[-1]
                    target_value = total_value * target_weight
                    current_value = total_value * current_weight
                    value_diff = target_value - current_value
                    
                    # Calculate shares to trade
                    quantity = value_diff / price
                    
                    # Add to trades list
                    if abs(quantity) > 1e-6:  # Minimum quantity threshold
                        trades.append((symbol, quantity))
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error generating trades: {str(e)}")
            raise
    
    def _execute_trades(
        self, 
        portfolio: Dict, 
        trades: List[Tuple[str, float]], 
        market_data: pd.DataFrame,
        date: datetime
    ) -> Dict:
        """Execute trades and update portfolio."""
        try:
            # Normalize training_end_date for comparison
            training_end_date = normalize_datetime(self.scheduler.training_end_date) if self.scheduler.training_end_date else None
            
            # Check if we're using LowTurnoverStrategy
            is_low_turnover = self.risk_profile == 'Low Turnover'
            
            # Check if we're using LowRiskStrategy
            is_low_risk = self.risk_profile == 'Low Risk'
            
            # For LowTurnoverStrategy, track monthly deals
            if is_low_turnover:
                # Create month key for tracking
                month_key = f"{date.year}-{date.month}"
                
                # Initialize monthly tracking if needed
                if month_key not in self.monthly_deals:
                    self.monthly_deals[month_key] = set()
                    self.logger.info(f"Backtester: Starting new month {month_key} for trade tracking")
                
                # Check if already at limit
                if len(self.monthly_deals[month_key]) >= 2:
                    self.logger.warning(f"Backtester: Monthly limit of 2 deals already reached for {month_key}. Skipping all trades.")
                    return portfolio  # Skip all trades
                
                # Calculate how many more deals we can do this month
                remaining_deals = 2 - len(self.monthly_deals[month_key])
                
                # Limit trades to remaining deals
                if len(trades) > remaining_deals:
                    self.logger.warning(f"Backtester: Limiting trades to remaining {remaining_deals} deals for month {month_key}")
                    trades = trades[:remaining_deals]
            
            # For LowRiskStrategy, check if proposed trades would exceed 10% volatility
            if is_low_risk and trades and len(trades) > 0:
                # First, simulate the portfolio after proposed trades
                simulated_portfolio = portfolio.copy()
                simulated_portfolio['holdings'] = portfolio['holdings'].copy()
                simulated_portfolio['cash'] = portfolio['cash']
                
                # Apply trades to simulation
                for symbol, quantity in trades:
                    # Check if symbol has price data
                    if f'{symbol}_close' not in market_data.columns:
                        continue
                    
                    price = market_data[f'{symbol}_close'].iloc[-1]
                    trade_value = price * quantity
                    
                    # Check if enough cash for buy
                    if quantity > 0 and trade_value > simulated_portfolio['cash']:
                        # Scale down trade to available cash
                        quantity = simulated_portfolio['cash'] / price
                        trade_value = simulated_portfolio['cash']
                    
                    # Update simulation cash
                    simulated_portfolio['cash'] -= trade_value
                    
                    # Update simulation holdings
                    if symbol not in simulated_portfolio['holdings']:
                        simulated_portfolio['holdings'][symbol] = 0
                    
                    simulated_portfolio['holdings'][symbol] += quantity
                
                # Calculate portfolio volatility after proposed trades
                if len(market_data) >= 20:  # Need at least 20 days for meaningful volatility
                    # Calculate current portfolio weights after proposed trades
                    total_value = simulated_portfolio['cash']
                    weights = {}
                    
                    for symbol, quantity in simulated_portfolio['holdings'].items():
                        if quantity > 0 and f'{symbol}_close' in market_data.columns:
                            price = market_data[f'{symbol}_close'].iloc[-1]
                            value = price * quantity
                            total_value += value
                            weights[symbol] = value
                    
                    # Convert values to weights
                    if total_value > 0:
                        for symbol in weights:
                            weights[symbol] /= total_value
                    
                    # Prepare returns data for volatility calculation
                    returns = pd.DataFrame()
                    for symbol in weights.keys():
                        if f'{symbol}_returns' in market_data.columns:
                            returns[symbol] = market_data[f'{symbol}_returns'].tail(60)  # Use last 60 days
                    
                    # If we have enough returns data, calculate portfolio volatility
                    if not returns.empty and returns.shape[0] >= 20:
                        port_volatility = self._calculate_portfolio_volatility(returns, weights)
                        
                        # STRICT ENFORCEMENT: Check if volatility exceeds 10%
                        if port_volatility > 0.10:
                            self.logger.warning(f"Backtester: Proposed trades would result in {port_volatility*100:.2f}% volatility, exceeding 10% limit!")
                            
                            # Scale down all trades proportionally to target the 10% volatility
                            scale_factor = 0.10 / port_volatility
                            trades = [(symbol, quantity * scale_factor * 0.95) for symbol, quantity in trades]  # Apply 5% safety margin
                            
                            self.logger.info(f"Backtester: Scaled down trades by factor {scale_factor*0.95:.3f} to enforce 10% volatility limit")
            
            for symbol, quantity in trades:
                # Check if symbol has price data
                if f'{symbol}_close' not in market_data.columns:
                    continue
                
                price = market_data[f'{symbol}_close'].iloc[-1]
                trade_value = price * quantity
                
                # Check if enough cash for buy
                if quantity > 0 and trade_value > portfolio['cash']:
                    # Scale down trade to available cash
                    quantity = portfolio['cash'] / price
                    trade_value = portfolio['cash']
                
                # Update cash
                portfolio['cash'] -= trade_value
                
                # Update holdings
                if symbol not in portfolio['holdings']:
                    portfolio['holdings'][symbol] = 0
                
                portfolio['holdings'][symbol] += quantity
                
                # Remove holdings with zero or very small quantities
                if abs(portfolio['holdings'][symbol]) < 1e-6:
                    portfolio['holdings'].pop(symbol)
                
                # Normalize date for comparison
                normalized_date = normalize_datetime(date)
                
                # Record trade with period information
                period = 'Training' if (training_end_date and normalized_date <= training_end_date) else 'Evaluation'
                
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'shares': quantity,
                    'price': price,
                    'amount': trade_value,
                    'action': 'BUY' if quantity > 0 else 'SELL',
                    'period': period
                })
                
                # For LowTurnoverStrategy, record this deal
                if is_low_turnover:
                    month_key = f"{date.year}-{date.month}"
                    self.monthly_deals[month_key].add(symbol)
                    self.logger.warning(f"Backtester: Executed deal {len(self.monthly_deals[month_key])}/2 for month {month_key}: {symbol}")
            
            return portfolio
            
        except Exception as e:
            self.logger.error(f"Error executing trades: {str(e)}")
            raise
    
    def _calculate_portfolio_volatility(self, returns: pd.DataFrame, weights: Dict[str, float]) -> float:
        """Calculate portfolio volatility for risk check."""
        try:
            # Ensure returns is not empty
            if returns.empty:
                return 0.20  # Default high volatility
                
            # Ensure we have valid weights
            if not weights or sum(weights.values()) == 0:
                return 0.20  # Default high volatility
            
            # Make sure all columns in weights exist in returns DataFrame
            valid_columns = [col for col in weights.keys() if col in returns.columns]
            if not valid_columns:
                return 0.20  # Default high volatility
                
            # Normalize weights to sum to 1
            total_weight = sum(weights[col] for col in valid_columns)
            if total_weight == 0:
                return 0.20  # Default high volatility
                
            # Calculate weighted returns for each asset
            weighted_returns = pd.Series(0, index=returns.index)
            for col in valid_columns:
                normalized_weight = weights[col] / total_weight
                weighted_returns += returns[col] * normalized_weight
                
            # Calculate annualized volatility
            volatility = weighted_returns.std() * np.sqrt(252)
            
            # Return 0.20 if result is NaN or too small
            if np.isnan(volatility) or volatility < 0.01:
                return 0.20
                
            return volatility
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio volatility: {str(e)}")
            # Return a default high volatility to trigger conservative allocation
            return 0.20
    
    def _calculate_performance_metrics(self, portfolio: Dict) -> Dict:
        """Calculate performance metrics for the backtest."""
        try:
            dates = portfolio['dates']
            values = portfolio['total_value']
            
            # Convert to pandas Series
            equity_curve = pd.Series(values, index=dates)
            
            # Calculate returns with explicit fill_method=None to avoid deprecation warning
            returns = equity_curve.pct_change(fill_method=None).dropna()
            cum_returns = (1 + returns).cumprod()
            
            # Calculate metrics
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1
            volatility = returns.std() * np.sqrt(252)
            sharpe_ratio = (returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
            
            # Calculate drawdown
            drawdown = (equity_curve / equity_curve.cummax()) - 1
            max_drawdown = drawdown.min()
            
            # Calculate number of trades per month
            trades_df = pd.DataFrame(portfolio['trades'])
            if not trades_df.empty:
                trades_df['date'] = pd.to_datetime(trades_df['date'])
                trades_df['year_month'] = trades_df['date'].dt.to_period('M')
                trades_per_month = trades_df.groupby('year_month').size()
                avg_trades_per_month = trades_per_month.mean()
                max_trades_per_month = trades_per_month.max()
            else:
                avg_trades_per_month = 0
                max_trades_per_month = 0
            
            metrics = {
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'trades_count': len(portfolio['trades']),
                'avg_trades_per_month': avg_trades_per_month,
                'max_trades_per_month': max_trades_per_month
            }
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            raise
            
    def get_trade_log(self) -> pd.DataFrame:
        """Get the trade log as a DataFrame."""
        if not self.results or 'portfolio' not in self.results:
            return pd.DataFrame()
        
        return pd.DataFrame(self.results['portfolio']['trades'])
        
    def get_training_metrics(self) -> Dict:
        """Get training period metrics."""
        if not self.results or 'training_metrics' not in self.results:
            return {}
        
        return self.results['training_metrics']
        
    def get_evaluation_metrics(self) -> Dict:
        """Get evaluation period metrics."""
        if not self.results or 'evaluation_metrics' not in self.results:
            return {}
        
        return self.results['evaluation_metrics']
