import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union
import logging
from abc import ABC, abstractmethod
import numpy as np
import os

class DataProvider(ABC):
    """Abstract base class for data providers."""
    
    @abstractmethod
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data for a given symbol."""
        pass

    @abstractmethod
    def get_current_price(self, symbol: str) -> float:
        """Get current price for a given symbol."""
        pass

    @abstractmethod
    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch market data for given symbols and date range."""
        pass

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider implementation."""
    
    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)
            
            if df.empty:
                logging.warning(f"No data returned from Yahoo Finance for {symbol}")
                return pd.DataFrame()
            
            # Calculate returns
            if 'Close' in df.columns:
                df['Returns'] = df['Close'].pct_change(fill_method=None)
            
            # Ensure we have a proper date index
            if not isinstance(df.index, pd.DatetimeIndex):
                logging.warning(f"Data for {symbol} does not have a DatetimeIndex")
                return pd.DataFrame()
            
            return df
        except Exception as e:
            logging.error(f"Error fetching historical data for {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            return ticker.info.get('regularMarketPrice')
        except Exception as e:
            logging.error(f"Error fetching current price for {symbol}: {str(e)}")
            raise

    def fetch_data(self, symbols: List[str], start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Fetch market data from Yahoo Finance."""
        try:
            # Download data for all symbols
            data = yf.download(
                symbols,
                start=start_date,
                end=end_date,
                group_by='ticker',
                progress=False
            )
            
            # Reset index to make date a column
            data = data.reset_index()
            
            # Rename columns to match our schema
            new_columns = ['date']
            for col in data.columns[1:]:
                symbol = col[1]  # Get symbol from multi-index
                metric = col[0].lower()  # Get metric from multi-index
                new_columns.append(f'{symbol}_{metric}')
            
            data.columns = new_columns
            
            # Ensure date column is datetime
            data['date'] = pd.to_datetime(data['date'])
            
            return data
            
        except Exception as e:
            raise Exception(f"Error fetching data from Yahoo Finance: {str(e)}")

class DataCollector:
    """Main data collector class that manages data fetching from various providers."""
    
    def __init__(self, provider: Optional[DataProvider] = None):
        """Initialize the data collector with a specific provider."""
        self.provider = provider or YahooFinanceProvider()
        self.logger = logging.getLogger(__name__)

    def set_provider(self, provider: DataProvider):
        """Set a new data provider."""
        self.provider = provider

    def get_historical_data(self, symbol: str, start_date: datetime, 
                          end_date: datetime) -> pd.DataFrame:
        """Get historical data for a symbol."""
        try:
            # Get the data from the provider
            df = self.provider.get_historical_data(symbol, start_date, end_date)
            
            if df is None or df.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Reset index to make date a column
            if isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
                # Ensure the column is named 'Date' (with capital D)
                if 'index' in df.columns:
                    df.rename(columns={'index': 'Date'}, inplace=True)
                elif 'date' in df.columns:
                    df.rename(columns={'date': 'Date'}, inplace=True)
            
            # Add Symbol column if not present
            if 'Symbol' not in df.columns:
                df['Symbol'] = symbol
            
            # Ensure Date column is properly formatted
            if 'Date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['Date']):
                # Keep as datetime for calculations, will be formatted for display later
                pass
            elif 'date' in df.columns and pd.api.types.is_datetime64_any_dtype(df['date']):
                # Rename to 'Date' for consistency
                df.rename(columns={'date': 'Date'}, inplace=True)
            
            # Rename any Yahoo Finance specific columns to more standard names
            column_rename_map = {
                'Adj Close': 'Adjusted_Close',
                'Adj. Close': 'Adjusted_Close',
                'Adj_Close': 'Adjusted_Close'
            }
            df.rename(columns=column_rename_map, inplace=True, errors='ignore')
            
            return df
        
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {str(e)}")
            # Return empty DataFrame instead of raising, to handle errors more gracefully
            return pd.DataFrame()

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        return self.provider.get_current_price(symbol)

    def get_multiple_symbols_data(self, symbols: List[str], start_date: datetime,
                                end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols."""
        data_dict = {}
        for symbol in symbols:
            try:
                data_dict[symbol] = self.get_historical_data(symbol, start_date, end_date)
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        return data_dict

    def get_market_data(self, symbols: List[str], start_date: datetime,
                       end_date: datetime) -> pd.DataFrame:
        """Get market data for multiple symbols and combine into a single DataFrame."""
        data_dict = self.get_multiple_symbols_data(symbols, start_date, end_date)
        
        if not data_dict:
            raise ValueError("No data was successfully fetched for any symbols")
        
        # Combine all data into a single DataFrame
        combined_data = pd.DataFrame()
        
        for symbol, df in data_dict.items():
            if not df.empty:
                # Add symbol column
                df['Symbol'] = symbol
                combined_data = pd.concat([combined_data, df])
        
        return combined_data

    def get_risk_free_rate(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get risk-free rate data (using 3-month Treasury Bill as proxy)."""
        try:
            return self.get_historical_data('^IRX', start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error fetching risk-free rate: {str(e)}")
            raise

    def get_market_index(self, index_symbol: str, start_date: datetime,
                        end_date: datetime) -> pd.DataFrame:
        """Get market index data (e.g., S&P 500)."""
        try:
            return self.get_historical_data(index_symbol, start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error fetching market index data: {str(e)}")
            raise

    def get_sector_data(self, sector_symbols: List[str], start_date: datetime,
                       end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Get data for sector ETFs or indices."""
        return self.get_multiple_symbols_data(sector_symbols, start_date, end_date)

    def get_asset_correlation_matrix(self, symbols: List[str], start_date: datetime,
                                   end_date: datetime) -> pd.DataFrame:
        """Calculate correlation matrix for a list of assets."""
        data_dict = self.get_multiple_symbols_data(symbols, start_date, end_date)
        
        # Create returns DataFrame
        returns_df = pd.DataFrame()
        for symbol, df in data_dict.items():
            if not df.empty:
                returns_df[symbol] = df['Returns']
        
        return returns_df.corr()

    def get_volatility_data(self, symbols: List[str], start_date: datetime,
                          end_date: datetime, window: int = 252) -> pd.DataFrame:
        """Calculate rolling volatility for a list of assets."""
        data_dict = self.get_multiple_symbols_data(symbols, start_date, end_date)
        
        volatility_df = pd.DataFrame()
        for symbol, df in data_dict.items():
            if not df.empty:
                volatility_df[symbol] = df['Returns'].rolling(window=window).std() * np.sqrt(252)
        
        return volatility_df

    def collect_market_data(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime
    ) -> pd.DataFrame:
        """Collect market data for given symbols and date range."""
        try:
            # Get data for each symbol
            data_dict = self.get_multiple_symbols_data(symbols, start_date, end_date)
            
            if not data_dict:
                raise ValueError("No data was successfully fetched for any symbols")
            
            # Create a DataFrame with the date index
            combined_data = pd.DataFrame()
            
            for symbol, df in data_dict.items():
                if not df.empty:
                    # Make a copy to avoid modifying the original
                    symbol_df = df.copy()
                    
                    # Ensure Date column exists and is consistent
                    date_col = None
                    if 'Date' in symbol_df.columns:
                        date_col = 'Date'
                    elif 'date' in symbol_df.columns:
                        date_col = 'date'
                    
                    if date_col:
                        # Set date as index for joining
                        symbol_df.set_index(date_col, inplace=True)
                    
                    # Rename other columns to match our schema
                    new_columns = {}
                    for col in symbol_df.columns:
                        if col != date_col:  # Skip date column
                            new_columns[col] = f'{symbol}_{col.lower()}'
                    symbol_df.rename(columns=new_columns, inplace=True)
                    
                    if combined_data.empty:
                        combined_data = symbol_df
                    else:
                        # Join on the index (date)
                        combined_data = combined_data.join(symbol_df, how='outer')
            
            # Reset index to make date a column
            if not combined_data.empty:
                combined_data.reset_index(inplace=True)
                
                # Ensure the date column is named 'date' (lowercase)
                if 'index' in combined_data.columns:
                    combined_data.rename(columns={'index': 'date'}, inplace=True)
                elif 'Date' in combined_data.columns:
                    combined_data.rename(columns={'Date': 'date'}, inplace=True)
                
                # Ensure date column is datetime
                if 'date' in combined_data.columns:
                    combined_data['date'] = pd.to_datetime(combined_data['date'])
                else:
                    raise ValueError("No date column found in the collected data")
                
                # Calculate returns for each symbol
                returns_data = pd.DataFrame()
                returns_data['date'] = combined_data['date']
                
                for symbol in symbols:
                    close_col = f'{symbol}_close'
                    if close_col in combined_data.columns:
                        # Calculate daily returns
                        combined_data[f'{symbol}_returns'] = combined_data[close_col].pct_change()
                        
                        # Add to returns data
                        returns_data[symbol] = combined_data[f'{symbol}_returns']
                
                # Convert returns_data to format for database storage
                returns_for_db = []
                for symbol in symbols:
                    if symbol in returns_data.columns:
                        symbol_returns = returns_data[['date', symbol]].dropna()
                        symbol_returns = symbol_returns.rename(columns={symbol: 'daily_return'})
                        symbol_returns['symbol'] = symbol
                        
                        # Calculate cumulative returns
                        symbol_returns['cumulative_return'] = (1 + symbol_returns['daily_return']).cumprod() - 1
                        
                        returns_for_db.append(symbol_returns)
                
                # Combine all returns data
                if returns_for_db:
                    all_returns = pd.concat(returns_for_db)
                    
                    # Store returns in database if available
                    try:
                        from src.data_management.database_manager import DatabaseManager
                        db_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..', 'outputs', 'database', 'database.db')
                        if os.path.exists(os.path.dirname(db_path)):
                            db_manager = DatabaseManager(db_path)
                            db_manager.store_returns(all_returns)
                            self.logger.info(f"Stored returns data for {len(symbols)} symbols")
                    except Exception as e:
                        self.logger.warning(f"Could not store returns in database: {str(e)}")
            else:
                raise ValueError("No data was collected after processing")
            
            # Ensure all required columns exist
            required_columns = ['date']
            for symbol in symbols:
                required_columns.extend([
                    f'{symbol}_open',
                    f'{symbol}_high',
                    f'{symbol}_low',
                    f'{symbol}_close',
                    f'{symbol}_volume'
                ])
            
            missing_columns = [col for col in required_columns if col not in combined_data.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return combined_data
            
        except Exception as e:
            raise Exception(f"Error collecting market data: {str(e)}")
    
    def collect_historical_data(
        self,
        symbols: List[str],
        lookback_days: int = 3650,  # Default to 10 years of data to cover 2015-2024
        start_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        """Collect historical market data for given symbols.
        
        Args:
            symbols: List of ticker symbols to collect data for
            lookback_days: Number of days to look back from today (if start_date is None)
            start_date: Explicit start date (overrides lookback_days if provided)
            
        Returns:
            DataFrame with market data for specified symbols
        """
        end_date = datetime.now()
        if start_date:
            # Use explicit start date if provided
            actual_start_date = start_date
        else:
            # Otherwise calculate from lookback days
            actual_start_date = end_date - timedelta(days=lookback_days)
            
        # Ensure we have data from at least 2015 for training purposes
        min_start_date = datetime(2015, 1, 1)
        if actual_start_date > min_start_date:
            self.logger.info(f"Adjusting start date from {actual_start_date} to {min_start_date} to ensure sufficient training data")
            actual_start_date = min_start_date
            
        try:
            return self.collect_market_data(symbols, actual_start_date, end_date)
        except Exception as e:
            self.logger.error(f"Error collecting historical data: {str(e)}")
            # Return empty DataFrame on error
            return pd.DataFrame()
