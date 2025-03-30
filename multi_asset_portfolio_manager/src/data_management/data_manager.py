"""
Data Manager module for handling data operations needed by the GUI.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import List, Dict, Optional, Union, Tuple

from .database_manager import DatabaseManager
from .data_collector import DataCollector

class DataManager:
    """
    Data manager class that handles data operations needed by the GUI components.
    This acts as a facade over database and data collection operations.
    """
    
    def __init__(self, db_manager: DatabaseManager, data_collector: DataCollector):
        """Initialize the data manager."""
        self.db_manager = db_manager
        self.data_collector = data_collector
        self.logger = logging.getLogger(__name__)
    
    def get_universe_tickers(self, universe: str) -> List[str]:
        """
        Get tickers for the specified universe.
        
        Args:
            universe: The universe name (e.g., "US Equities", "Global Equities")
            
        Returns:
            List of tickers in the universe
        """
        try:
            # Get tickers already in the database
            db_tickers = self.db_manager.get_available_tickers()
            
            # Define universe specific tickers
            if universe == "US Equities":
                tickers = [
                    # Large Cap
                    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "BRK-B", "JPM", "V", "JNJ", "PG", "MA",
                    # Mid Cap
                    "AMD", "SNAP", "UBER", "LYFT", "SQ", "ROKU", "PINS", "TWTR", "SPOT", "ZM", "ETSY", "DOCU",
                    # Small Cap
                    "FSLY", "FVRR", "CHGG", "PTON", "CRSR", "U", "UPWK", "NET", "MDB", "TTD", "DDOG",
                    # ETFs
                    "SPY", "QQQ", "DIA", "IWM", "VTI"
                ]
                # Add any tickers from database
                if db_tickers:
                    us_tickers = [t for t in db_tickers if "." not in t and "-" not in t]
                    for ticker in us_tickers:
                        if ticker not in tickers:
                            tickers.append(ticker)
                            
            elif universe == "Global Equities":
                tickers = [
                    # US Stocks
                    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", 
                    # European Stocks
                    "BP.L", "HSBA.L", "GSK.L", "ULVR.L", "AZN.L", "LLOY.L", "VOD.L", "BARC.L", "RIO.L", "STAN.L",
                    # Asian Stocks
                    "7203.T", "9984.T", "6758.T", "6861.T", "6501.T", "7751.T", "9432.T", "6902.T", "7267.T",
                    # German Stocks
                    "VOW3.DE", "DAI.DE", "SAP.DE", "BAYN.DE", "DTE.DE", "DBK.DE", "CON.DE", "LHA.DE", "BAS.DE", "BMW.DE",
                    # French Stocks
                    "AIR.PA", "BNP.PA", "SAN.PA", "MC.PA", "OR.PA", "CS.PA", "ACA.PA", "DSY.PA", "SGO.PA", "KER.PA",
                    # Dutch Stocks
                    "ASML.AS", "REN.AS", "AD.AS", "INGA.AS", "RDSA.AS", "KPN.AS", "PHIA.AS", "ABN.AS", "URW.AS", "RAND.AS",
                    # Swiss Stocks
                    "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "SREN.SW", "SCMN.SW", "GIVN.SW"
                ]
                # Add any EU tickers from database
                if db_tickers:
                    eu_tickers = [t for t in db_tickers if any(t.endswith(ext) for ext in 
                                  ['.DE', '.PA', '.AS', '.SW', '.L', '.MI', '.MC', '.T', '.ST'])]
                    for ticker in eu_tickers:
                        if ticker not in tickers:
                            tickers.append(ticker)
                            
            elif universe == "Cryptocurrencies":
                tickers = [
                    "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
                    "UNI-USD", "AAVE-USD", "MKR-USD", "SUSHI-USD", "YFI-USD", "CRV-USD", "SNX-USD",
                    "MATIC-USD", "OP-USD", "ARB-USD"
                ]
                # Add any crypto tickers from database
                if db_tickers:
                    crypto_tickers = [t for t in db_tickers if "-USD" in t]
                    for ticker in crypto_tickers:
                        if ticker not in tickers:
                            tickers.append(ticker)
                            
            elif universe == "Commodities":
                tickers = [
                    "GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV",
                    "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "UNG",
                    "HG=F", "ZC=F", "ZS=F", "ZW=F",
                    "KC=F", "CT=F", "OJ=F", "SB=F", "CC=F"
                ]
                # Add any commodity tickers from database
                if db_tickers:
                    commodity_tickers = [t for t in db_tickers if "=F" in t]
                    for ticker in commodity_tickers:
                        if ticker not in tickers:
                            tickers.append(ticker)
                            
            elif universe == "ETFs":
                tickers = [
                    "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "VWO", "EFA", "EEM", "ACWI", "VT",
                    "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU", "XLC", "XBI", "XHB"
                ]
                # Add any ETF tickers from database
                if db_tickers:
                    etf_tickers = [t for t in db_tickers if all(c not in t for c in ['.', '=', '-'])]
                    for ticker in etf_tickers:
                        if ticker not in tickers:
                            tickers.append(ticker)
            else:
                # Custom or unknown universe, use database tickers
                tickers = db_tickers if db_tickers else []
                
            return tickers
            
        except Exception as e:
            self.logger.error(f"Error getting universe tickers: {str(e)}")
            return []
    
    def fetch_market_data(self, tickers: List[str], start_date: datetime, end_date: datetime, 
                         update_progress_callback=None) -> Dict[str, pd.DataFrame]:
        """
        Fetch market data for the specified tickers.
        
        Args:
            tickers: List of tickers to fetch data for
            start_date: Start date for data
            end_date: End date for data
            update_progress_callback: Optional callback to update progress
            
        Returns:
            Dictionary mapping tickers to DataFrames with market data
        """
        try:
            results = {}
            total_tickers = len(tickers)
            
            for i, ticker in enumerate(tickers):
                # Skip if already in database
                existing_data = self.db_manager.get_market_data(ticker, start_date, end_date)
                if existing_data is not None and not existing_data.empty:
                    results[ticker] = existing_data
                    if update_progress_callback:
                        update_progress_callback((i + 1) / total_tickers * 100, ticker, True)
                    continue
                    
                # Fetch from external source
                try:
                    self.logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
                    data = self.data_collector.fetch_market_data(ticker, start_date, end_date)
                    
                    if data is not None and not data.empty:
                        # Process and prepare for database
                        processed_data = self._prepare_data_for_db(ticker, data)
                        
                        # Store in database
                        self.db_manager.store_market_data(processed_data)
                        
                        # Add to results
                        results[ticker] = data
                        
                except Exception as e:
                    self.logger.error(f"Error fetching data for {ticker}: {str(e)}")
                
                # Update progress
                if update_progress_callback:
                    update_progress_callback((i + 1) / total_tickers * 100, ticker, False)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Error fetching market data: {str(e)}")
            return {}
    
    def _prepare_data_for_db(self, symbol: str, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare market data for database storage."""
        try:
            # Create a copy to avoid modifying the original
            df = data.copy()
            
            # Add symbol column
            df['symbol'] = symbol
            
            # Convert index to date column if needed
            if isinstance(df.index, pd.DatetimeIndex):
                df.reset_index(inplace=True)
                df.rename(columns={'index': 'date'}, inplace=True)
            
            # Ensure date is in string format for SQLite
            if 'date' in df.columns and isinstance(df['date'].iloc[0], datetime):
                df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            
            # Ensure column names are lowercase for consistency
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure required columns exist
            required_cols = ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    if col in ['open', 'high', 'low', 'volume']:
                        # Fill with reasonable defaults if needed
                        df[col] = df['close'] if 'close' in df.columns else 0
                    elif col == 'volume':
                        df[col] = 0
            
            # Return only needed columns
            return df[required_cols]
            
        except Exception as e:
            self.logger.error(f"Error preparing data for DB: {str(e)}")
            return pd.DataFrame()
    
    def get_available_portfolios(self) -> Tuple[List[int], List[str]]:
        """
        Get available portfolios from the database.
        
        Returns:
            Tuple containing lists of portfolio IDs and portfolio names
        """
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.portfolio_id, p.name, c.name
                FROM Portfolios p
                JOIN Clients c ON p.client_id = c.client_id
                ORDER BY p.name
            """)
            portfolios = cursor.fetchall()
            
            # Prepare result
            portfolio_ids = [p[0] for p in portfolios]
            portfolio_names = [f"{p[1]} ({p[2]})" for p in portfolios]
            
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
            return portfolio_ids, portfolio_names
            
        except Exception as e:
            self.logger.error(f"Error getting available portfolios: {str(e)}")
            return [], []
    
    def get_available_clients(self) -> Tuple[List[int], List[str]]:
        """
        Get available clients from the database.
        
        Returns:
            Tuple containing lists of client IDs and client names
        """
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT client_id, name FROM Clients ORDER BY name")
            clients = cursor.fetchall()
            
            # Prepare result
            client_ids = [c[0] for c in clients]
            client_names = [c[1] for c in clients]
            
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
            return client_ids, client_names
            
        except Exception as e:
            self.logger.error(f"Error getting available clients: {str(e)}")
            return [], [] 