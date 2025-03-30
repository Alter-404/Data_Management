"""
Portfolio Manager module for managing portfolio construction and strategy integration.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .strategies import create_strategy
from .optimizer import PortfolioOptimizer
from .backtester import PortfolioBacktester

class PortfolioManager:
    """
    Portfolio Manager that handles construction, strategy selection,
    and integration with optimization and backtesting.
    """
    
    def __init__(self, db_manager):
        """Initialize the portfolio manager."""
        self.db_manager = db_manager
        self.logger = logging.getLogger(__name__)
    
    def construct_portfolio(
        self,
        portfolio_id: int,
        strategy_name: str,
        universe_name: str,
        initial_capital: float,
        max_positions: int,
        train_start: datetime,
        train_end: datetime,
        eval_start: datetime,
        eval_end: datetime,
        frequency: str
    ) -> Dict[str, Any]:
        """
        Construct a portfolio based on the specified parameters.
        
        Args:
            portfolio_id: ID of the portfolio to use
            strategy_name: Name of the strategy to use (e.g., "Low Risk")
            universe_name: Name of the asset universe (e.g., "US Equities")
            initial_capital: Initial capital amount
            max_positions: Maximum number of positions
            train_start: Start date for training period
            train_end: End date for training period
            eval_start: Start date for evaluation period
            eval_end: End date for evaluation period
            frequency: Trading frequency (e.g., "Weekly")
            
        Returns:
            Dict containing construction results
        """
        try:
            self.logger.info(f"Starting portfolio construction for {strategy_name} strategy")
            
            # Create strategy instance
            strategy = create_strategy(strategy_name)
            
            # Fetch market data
            market_data = self._fetch_market_data(universe_name, train_start, eval_end)
            
            # Initialize the diversified strategy wrapper
            strategy_wrapper = self._create_diversified_strategy(strategy, max_positions)
            
            # Create and configure backtester
            backtester = self._create_modified_backtester(portfolio_id, strategy_wrapper)
            
            # Run backtest
            results = backtester.run_backtest(
                market_data=market_data,
                initial_capital=initial_capital,
                start_date=train_start,
                end_date=eval_end,
                training_end_date=train_end
            )
            
            # Store results in database
            self._store_results(portfolio_id, results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in portfolio construction: {str(e)}")
            raise
    
    def _fetch_market_data(self, universe_name: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch market data for the specified universe and date range."""
        try:
            # Get tickers for the specified universe
            tickers = self._get_universe_tickers(universe_name)
            
            # Fetch market data using database manager
            market_data = pd.DataFrame()
            
            for ticker in tickers:
                data = self.db_manager.get_market_data(ticker, start_date, end_date)
                if data is not None and not data.empty:
                    # Process data for each ticker
                    for col in ['open', 'high', 'low', 'close', 'volume']:
                        if col in data.columns:
                            market_data[f'{ticker}_{col}'] = data[col]
                    
                    # Calculate returns
                    if 'close' in data.columns:
                        market_data[f'{ticker}_returns'] = data['close'].pct_change()
            
            # Set index to datetime if not already
            if not market_data.empty and not isinstance(market_data.index, pd.DatetimeIndex):
                market_data.index = pd.to_datetime(market_data.index)
            
            # Fill NaN values
            market_data = market_data.fillna(method='ffill').fillna(method='bfill').fillna(0)
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            raise
    
    def _get_universe_tickers(self, universe_name: str) -> List[str]:
        """Get tickers for the specified universe."""
        # This is a simplified implementation - in a real app, this would
        # pull from a configuration file or database table
        if universe_name == "US Equities":
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V"]
        elif universe_name == "Global Equities":
            return ["AAPL", "MSFT", "ASML", "SONY", "TCEHY", "BABA", "TSM", "SAP", "LVMUY"]
        elif universe_name == "Cryptocurrencies":
            return ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD"]
        elif universe_name == "Mixed Assets":
            return ["SPY", "QQQ", "GLD", "TLT", "BTC-USD", "VWO", "VEA"]
        else:
            # Fallback to a default set
            return ["SPY", "QQQ", "DIA", "IWM", "VTI"]
    
    def _create_diversified_strategy(self, base_strategy, min_assets=3):
        """Create a diversified strategy wrapper around the base strategy."""
        # This is a strategy wrapper that ensures diversification
        class DiversifiedStrategyWrapper:
            def __init__(self, strategy, min_assets=3):
                self.strategy = strategy
                self.min_assets = min_assets
                self.risk_profile = strategy.risk_profile
            
            def generate_signals(self, market_data, portfolio_data):
                # Get base strategy signals
                weights = self.strategy.generate_signals(market_data, portfolio_data)
                
                # Ensure minimum diversification
                if len(weights) < self.min_assets:
                    # Add more assets with small weights
                    available_assets = [col.split('_')[0] for col in market_data.columns if '_close' in col]
                    for asset in available_assets:
                        if asset not in weights and len(weights) < self.min_assets:
                            weights[asset] = 0.01
                
                # Normalize weights
                total = sum(weights.values())
                if total > 0:
                    return {k: v/total for k, v in weights.items()}
                return weights
                
            def train(self, market_data, portfolio_data):
                self.strategy.train(market_data, portfolio_data)
                
            def is_evaluation_period(self, date):
                return self.strategy.is_evaluation_period(date)
                
            def is_training_period(self, date):
                return self.strategy.is_training_period(date)
        
        return DiversifiedStrategyWrapper(base_strategy, min_assets)
    
    def _create_modified_backtester(self, portfolio_id, strategy):
        """Create a modified backtester for portfolio construction."""
        # This extends the backtester to track additional data
        class ModifiedBacktester(PortfolioBacktester):
            def __init__(self, portfolio_id, strategy, db_manager=None):
                super().__init__(strategy.risk_profile)
                self.portfolio_id = portfolio_id
                self.strategy = strategy
                self.db_manager = db_manager
            
            def _generate_trades(self, current_portfolio, target_weights, market_data, date):
                # Generate trades with max position constraint
                trades = []
                cash = current_portfolio['cash']
                holdings = current_portfolio['holdings']
                
                # Get latest prices
                latest_prices = {}
                for symbol in target_weights:
                    price_col = f"{symbol}_close"
                    if price_col in market_data.columns:
                        # Get the last non-NaN price
                        prices = market_data[price_col].dropna()
                        if not prices.empty:
                            latest_prices[symbol] = prices.iloc[-1]
                
                # Calculate current portfolio value
                portfolio_value = cash
                for symbol, quantity in holdings.items():
                    if symbol in latest_prices:
                        portfolio_value += quantity * latest_prices[symbol]
                
                # Calculate target position values
                target_values = {s: w * portfolio_value for s, w in target_weights.items()}
                
                # Generate trades
                for symbol, target_value in target_values.items():
                    if symbol in latest_prices:
                        current_quantity = holdings.get(symbol, 0)
                        current_value = current_quantity * latest_prices[symbol]
                        
                        # Calculate trade
                        value_diff = target_value - current_value
                        
                        # Only trade if the difference is significant (> 1% of portfolio)
                        if abs(value_diff) > 0.01 * portfolio_value:
                            price = latest_prices[symbol]
                            quantity = value_diff / price
                            
                            # Add trade to list
                            trades.append((symbol, quantity))
                
                return trades
        
        return ModifiedBacktester(portfolio_id, strategy, self.db_manager)
    
    def _store_results(self, portfolio_id: int, results: Dict[str, Any]) -> None:
        """Store backtest results in the database."""
        try:
            # Store portfolio values
            if 'portfolio' in results and 'total_value' in results['portfolio']:
                values = results['portfolio']['total_value']
                dates = results['portfolio']['dates']
                
                portfolio_values = pd.Series(values, index=dates)
                self.db_manager.store_performance_metrics(
                    portfolio_id, 
                    {'portfolio_values': portfolio_values}
                )
            
            # Store results in database if portfolio_id is provided
            if portfolio_id and self.db_manager:
                # Store trades
                if 'portfolio' in results and 'trades' in results['portfolio']:
                    trades = results['portfolio']['trades']
                    trades_df = pd.DataFrame(trades)
                    
                    if not trades_df.empty:
                        # Add portfolio_id column
                        trades_df['portfolio_id'] = portfolio_id
                        # Rename columns to match database schema
                        col_mapping = {
                            'symbol': 'symbol',
                            'date': 'date',
                            'action': 'action', 
                            'quantity': 'shares',
                            'price': 'price',
                            'value': 'amount'
                        }
                        # Rename columns
                        trades_df = trades_df.rename(columns=col_mapping)
                        # Store in database
                        self.db_manager.store_deals(trades_df)
            
            # Store metrics
            if 'combined_metrics' in results:
                self.db_manager.store_performance_metrics(
                    portfolio_id,
                    {'metrics': {'combined': results['combined_metrics']}}
                )
                
            self.logger.info(f"Stored portfolio construction results for portfolio {portfolio_id}")
            
        except Exception as e:
            self.logger.error(f"Error storing portfolio construction results: {str(e)}")
            raise 