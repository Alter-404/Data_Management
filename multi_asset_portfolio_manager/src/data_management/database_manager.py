import sqlite3
from datetime import datetime
import pandas as pd
from typing import List, Dict, Optional, Union
import logging
import json
import os

class DatabaseManager:
    def __init__(self, db_path: str = "database.db"):
        """Initialize the database manager with the specified database path."""
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self.conn = None
        self._initialize_database()
        self._migrate_database()

    def _get_connection(self):
        """Get a database connection. For in-memory databases, reuse the existing connection."""
        if self.db_path == ":memory:":
            if self.conn is None:
                self.conn = sqlite3.connect(self.db_path)
                self.conn.execute("PRAGMA foreign_keys = ON")
            return self.conn
        else:
            # For file-based databases, create a new connection each time
            conn = sqlite3.connect(self.db_path)
            conn.execute("PRAGMA foreign_keys = ON")
            return conn

    def _migrate_database(self):
        """Migrate the database schema to add missing columns to existing tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Check if Portfolios table has strategy and asset_universe columns
            cursor.execute("PRAGMA table_info(Portfolios)")
            columns = [column[1] for column in cursor.fetchall()]
            
            # Add missing columns to Portfolios table if needed
            if 'strategy' not in columns:
                cursor.execute("ALTER TABLE Portfolios ADD COLUMN strategy TEXT")
                self.logger.info("Added 'strategy' column to Portfolios table")
                
            if 'asset_universe' not in columns:
                cursor.execute("ALTER TABLE Portfolios ADD COLUMN asset_universe TEXT")
                self.logger.info("Added 'asset_universe' column to Portfolios table")
            
            conn.commit()
        except Exception as e:
            self.logger.error(f"Error migrating database: {str(e)}")
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def _initialize_database(self):
        """Initialize the database with required tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create Clients table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Clients (
                client_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                risk_profile TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Managers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Managers (
                manager_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                email TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Products table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Products (
                asset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL UNIQUE,
                asset_type TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create Returns table for storing asset returns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Returns (
                return_id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                daily_return REAL NOT NULL,
                cumulative_return REAL,
                period TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(symbol, date)
            )
        """)
        
        # Create Portfolios table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Portfolios (
                portfolio_id INTEGER PRIMARY KEY AUTOINCREMENT,
                client_id INTEGER,
                manager_id INTEGER,
                name TEXT NOT NULL,
                strategy TEXT,
                asset_universe TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (client_id) REFERENCES Clients (client_id),
                FOREIGN KEY (manager_id) REFERENCES Managers (manager_id)
            )
        """)
        
        # Create PortfolioMetadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PortfolioMetadata (
                metadata_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES Portfolios (portfolio_id),
                UNIQUE(portfolio_id, key)
            )
        """)
        
        # Create PortfolioHoldings table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PortfolioHoldings (
                holding_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                asset_id INTEGER,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES Portfolios (portfolio_id),
                FOREIGN KEY (asset_id) REFERENCES Products (asset_id)
            )
        """)
        
        # Create MarketData table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS MarketData (
                data_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                symbol TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, symbol)
            )
        """)
        
        # Create Positions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Positions (
                position_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER,
                date DATE NOT NULL,
                symbol TEXT NOT NULL,
                shares REAL NOT NULL,
                entry_price REAL NOT NULL,
                current_price REAL NOT NULL,
                weight REAL NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES Portfolios (portfolio_id)
            )
        """)
        
        # Create PerformanceMetrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS PerformanceMetrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                date TIMESTAMP NOT NULL,
                total_value REAL,
                daily_return REAL,
                cumulative_return REAL,
                volatility REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                total_return REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES Portfolios (portfolio_id)
            )
        """)
        
        # Create Deals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Deals (
                deal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                portfolio_id INTEGER NOT NULL,
                date DATE NOT NULL,
                symbol TEXT NOT NULL,
                action TEXT NOT NULL,
                shares REAL NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                period TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (portfolio_id) REFERENCES Portfolios (portfolio_id),
                FOREIGN KEY (symbol) REFERENCES Products (symbol)
            )
        """)
        
        # Clear Deals table on initialization
        cursor.execute("DELETE FROM Deals")
        
        # Populate Products table with predefined tickers
        self._populate_products_table(cursor)
        
        conn.commit()
        if self.db_path != ":memory:":
            conn.close()
            
    def _populate_products_table(self, cursor):
        """Populate the Products table with predefined tickers."""
        # Define asset types and their tickers
        asset_types = {
            "US Equities": [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "LYFT",
                "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW", "USB", "WFC", "C", "PGR", "AIG",
                "JNJ", "UNH", "PFE", "MRK", "ABBV", "ABT", "BMY", "AMGN", "GILD", "MRNA", "BNTX", "DHR", "SYK", "BSX",
                "PG", "KO", "PEP", "WMT", "HD", "NKE", "MCD", "SBUX", "DIS", "NFLX", "CMCSA", "TGT", "COST", "LMT",
                "CAT", "BA", "GE", "MMM", "HON", "UPS", "FDX", "DE", "WM", "CSX", "UNP", "RTX", "ETN", "EMR"
            ],
            "EU Equities": [
                "BMW.DE", "SIE.DE", "ALV.DE", "BAYN.DE", "DTE.DE", "EOAN.DE", "HEI.DE", "MUV2.DE",
                "BNP.PA", "SAN.PA", "AIR.PA", "OR.PA", "MC.PA", "CS.PA", "BN.PA", "AI.PA", "DG.PA", "KER.PA",
                "ASML", "INGA.AS", "PHIA.AS", "ABN.AS", "KPN.AS", "WKL.AS", "AD.AS",
                "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "SREN.SW", "SCMN.SW", "GIVN.SW", "AMS.SW",
                "ERIC-B.ST", "SAND.ST", "VOLV-B.ST", "ATCO-A.ST", "ESSITY-B.ST", "HM-B.ST", "SBB-B.ST", "TELIA.ST",
                "SAN.MC", "BBVA.MC", "TEF.MC", "IBE.MC", "ITX.MC", "REP.MC", "ACS.MC", "AENA.MC", "MAP.MC", "MEL.MC",
                "UCG.MI", "ENI.MI", "ISP.MI", "G.MI", "BMPS.MI", "MB.MI", "PIRC.MI", "AMP.MI"
            ],
            "Global Equities": [
                "0700.HK", "9988.HK", "BABA", "JD", "PDD", "BIDU", "7203.T", "9984.T", "005930.KS", "035720.KS",
                "HSBC", "BP.L", "XOM", "CVX", "COP", "EOG", "CNQ.TO", "SU.TO", "ENB.TO",
                "NVO", "AZN.L", "GSK.L", "NVS", "NSRGY"
            ],
            "Commodities": [
                "GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV",
                "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "UNG", "BNO", "UCO", "SCO",
                "HG=F", "ZC=F", "ZS=F", "ZW=F",
                "KC=F", "CT=F", "OJ=F", "SB=F", "CC=F",
                "HE=F"
            ],
            "ETFs": [
                "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "VWO", "EFA", "EEM", "ACWI", "VT",
                "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU", "XLC", "XBI", "XHB",
                "AGG", "TLT", "LQD", "HYG", "BND", "BNDX", "MUB", "TIP", "SHY", "IEI", "IEF", "TLH",
                "MTUM", "QUAL", "VLUE", "SIZE", "USMV", "SPLV", "SPHB", "SPHQ", "SPHD", "SPYD",
                "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWH", "EWY", "EWT", "EWM", "EWS", "EWP", "EWQ",
                "VNQ", "REM", "MLPA", "AMLP", "GDX", "GDXJ", "IAU", "SIVR"
            ],
            "Cryptocurrencies": [
                "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
                "UNI-USD", "AAVE-USD", "MKR-USD", "SUSHI-USD", "YFI-USD", "CRV-USD", "SNX-USD",
                "MATIC-USD", "OP-USD", "ARB-USD", "LRC-USD", "BOBA-USD",
                "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "ALICE-USD",
                "XMR-USD", "ZEC-USD", "DASH-USD", "ROSE-USD",
                "DOT-USD", "LINK-USD", "ATOM-USD", "NEAR-USD", "FTM-USD", "ONE-USD", "EGLD-USD"
            ]
        }
        
        # Insert tickers into Products table
        for asset_type, tickers in asset_types.items():
            for symbol in tickers:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO Products (symbol, asset_type)
                        VALUES (?, ?)
                    """, (symbol, asset_type))
                except Exception as e:
                    print(f"Error inserting asset {symbol}: {str(e)}")

    def __del__(self):
        """Cleanup method to close the connection when the object is destroyed."""
        if self.conn is not None:
            self.conn.close()

    def add_client(self, name: str, risk_profile: str, email: str) -> int:
        """Add a new client to the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "INSERT INTO Clients (name, risk_profile, email) VALUES (?, ?, ?)",
                (name, risk_profile, email)
            )
            
            conn.commit()
            return cursor.lastrowid
            
        except Exception as e:
            if self.db_path != ":memory:":
                conn.rollback()
            raise e
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def add_manager(self, name: str, email: str) -> int:
        """Add a new portfolio manager to the database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Managers (name, email) VALUES (?, ?)",
                (name, email)
            )
            conn.commit()
            cursor.close()
            if self.db_path != ":memory:":
                conn.close()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding manager: {str(e)}")
            raise

    def add_product(self, symbol: str, asset_type: str, description: Optional[str] = None) -> int:
        """Add a new product to the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT INTO Products (symbol, asset_type, description) VALUES (?, ?, ?)",
                    (symbol, asset_type, description)
                )
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error adding product: {str(e)}")
            raise

    def add_returns(self, product_id: int, date: datetime, return_value: float):
        """Add returns data for a product."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "INSERT OR REPLACE INTO Returns (product_id, date, return_value) VALUES (?, ?, ?)",
                    (product_id, date, return_value)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error adding returns: {str(e)}")
            raise

    def create_portfolio(self, client_id: int, manager_id: int, name: str, strategy: str = None, asset_universe: str = None) -> int:
        """Create a new portfolio for a client."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO Portfolios (client_id, manager_id, name, strategy, asset_universe) VALUES (?, ?, ?, ?, ?)",
                (client_id, manager_id, name, strategy, asset_universe)
            )
            conn.commit()
            cursor.close()
            if self.db_path != ":memory:":
                conn.close()
            return cursor.lastrowid
        except Exception as e:
            self.logger.error(f"Error creating portfolio: {str(e)}")
            raise

    def store_deals(self, deals_df: pd.DataFrame) -> bool:
        """Store trade data in the database.
        
        Args:
            deals_df: DataFrame with columns portfolio_id, symbol, date, action, 
                     shares, price, amount, and optional period
        
        Returns:
            bool: True if successful, False otherwise
        """
        required_columns = ['portfolio_id', 'symbol', 'date', 'action', 
                            'shares', 'price', 'amount']
        
        # Validate required columns
        for col in required_columns:
            if col not in deals_df.columns:
                self.logger.error(f"Missing required column in deals data: {col}")
                return False
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            for _, row in deals_df.iterrows():
                # Check if portfolio exists
                cursor.execute("SELECT 1 FROM Portfolios WHERE portfolio_id = ?", (row['portfolio_id'],))
                if not cursor.fetchone():
                    self.logger.warning(f"Portfolio ID {row['portfolio_id']} does not exist. Skipping deal.")
                    continue
                
                # Check if asset exists
                cursor.execute("SELECT 1 FROM Products WHERE symbol = ?", (row['symbol'],))
                if not cursor.fetchone():
                    self.logger.warning(f"Asset symbol {row['symbol']} does not exist. Skipping deal.")
                    continue
                
                # Get period if available, otherwise NULL
                period = row.get('period', None)
                
                # Insert deal
                cursor.execute("""
                    INSERT INTO Deals (portfolio_id, symbol, date, action, 
                                     shares, price, amount, period)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    row['portfolio_id'],
                    row['symbol'],
                    row['date'],
                    row['action'],
                    row['shares'],
                    row['price'],
                    row['amount'],
                    period
                ))
            
            conn.commit()
            self.logger.info(f"Successfully stored {len(deals_df)} deals")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing deals: {str(e)}")
            return False
        finally:
            if self.db_path != ":memory:":
                conn.close()
    
    def get_deals(self, portfolio_id=None, start_date=None, end_date=None, symbols=None) -> pd.DataFrame:
        """Retrieve deal data from the database with optional filtering.
        
        Args:
            portfolio_id: Filter by portfolio ID
            start_date: Filter by deals after this date
            end_date: Filter by deals before this date
            symbols: Filter by list of symbols
        
        Returns:
            DataFrame with deal data
        """
        try:
            conn = self._get_connection()
            
            query = "SELECT * FROM Deals WHERE 1=1"
            params = []
            
            if portfolio_id is not None:
                query += " AND portfolio_id = ?"
                params.append(portfolio_id)
            
            if start_date is not None:
                query += " AND date >= ?"
                params.append(start_date)
            
            if end_date is not None:
                query += " AND date <= ?"
                params.append(end_date)
            
            if symbols is not None and len(symbols) > 0:
                placeholders = ', '.join(['?' for _ in symbols])
                query += f" AND symbol IN ({placeholders})"
                params.extend(symbols)
            
            query += " ORDER BY date DESC"
            
            deals_df = pd.read_sql_query(query, conn, params=params)
            return deals_df
            
        except Exception as e:
            self.logger.error(f"Error retrieving deals: {str(e)}")
            return pd.DataFrame()
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def record_deal(self, portfolio_id: int, symbol: str, date: str, 
                   action: str, shares: float, price: float, amount: float, 
                   period: str = None) -> int:
        """Record a single trade in the database.
        
        Args:
            portfolio_id: ID of the portfolio
            symbol: Asset symbol
            date: Date of the trade
            action: BUY or SELL
            shares: Number of units traded
            price: Price per unit
            amount: Total amount of the trade
            period: Optional period designation
        
        Returns:
            int: Deal ID if successful, -1 otherwise
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if portfolio exists
            cursor.execute("SELECT 1 FROM Portfolios WHERE portfolio_id = ?", (portfolio_id,))
            if not cursor.fetchone():
                self.logger.warning(f"Portfolio ID {portfolio_id} does not exist. Cannot record deal.")
                return -1
            
            # Check if asset exists
            cursor.execute("SELECT 1 FROM Products WHERE symbol = ?", (symbol,))
            if not cursor.fetchone():
                self.logger.warning(f"Asset symbol {symbol} does not exist. Cannot record deal.")
                return -1
            
            # Insert deal
            cursor.execute("""
                INSERT INTO Deals (portfolio_id, symbol, date, action, 
                                 shares, price, amount, period)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                portfolio_id,
                symbol,
                date,
                action,
                shares,
                price,
                amount,
                period
            ))
            
            conn.commit()
            deal_id = cursor.lastrowid
            self.logger.info(f"Successfully recorded deal ID {deal_id}")
            return deal_id
            
        except Exception as e:
            self.logger.error(f"Error recording deal: {str(e)}")
            return -1
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def update_portfolio_holdings(self, portfolio_id: int, product_id: int,
                                quantity: float, average_price: float):
        """Update portfolio holdings after a deal."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """INSERT OR REPLACE INTO PortfolioHoldings 
                       (portfolio_id, product_id, quantity, average_price)
                       VALUES (?, ?, ?, ?)""",
                    (portfolio_id, product_id, quantity, average_price)
                )
                conn.commit()
        except Exception as e:
            self.logger.error(f"Error updating portfolio holdings: {str(e)}")
            raise

    def get_portfolio_holdings(self, portfolio_id: int) -> pd.DataFrame:
        """Get current holdings for a specific portfolio."""
        try:
            conn = self._get_connection()
            query = """
                SELECT ph.*, p.symbol, p.asset_type
                FROM PortfolioHoldings ph
                JOIN Products p ON ph.asset_id = p.asset_id
                WHERE ph.portfolio_id = ?
            """
            df = pd.read_sql_query(query, conn, params=(portfolio_id,))
            if self.db_path != ":memory:":
                conn.close()
            return df
        except Exception as e:
            self.logger.error(f"Error getting portfolio holdings: {str(e)}")
            raise

    def get_portfolio_returns(self, portfolio_id: int, start_date: datetime,
                            end_date: datetime) -> pd.DataFrame:
        """Get historical returns for a portfolio."""
        try:
            conn = self._get_connection()
            query = """
                SELECT d.deal_date, d.deal_type, d.quantity, d.price,
                       p.symbol, p.asset_type
                FROM Deals d
                JOIN Products p ON d.asset_id = p.asset_id
                WHERE d.portfolio_id = ? AND d.deal_date BETWEEN ? AND ?
                ORDER BY d.deal_date
            """
            df = pd.read_sql_query(query, conn, params=(portfolio_id, start_date, end_date))
            if self.db_path != ":memory:":
                conn.close()
            return df
        except Exception as e:
            self.logger.error(f"Error getting portfolio returns: {str(e)}")
            raise

    def get_client_portfolios(self, client_id: int) -> pd.DataFrame:
        """Get all portfolios for a specific client."""
        conn = None
        try:
            conn = self._get_connection()
            query = """
                SELECT p.*, m.name as manager_name, m.email
                FROM Portfolios p
                JOIN Managers m ON p.manager_id = m.manager_id
                WHERE p.client_id = ?
            """
            return pd.read_sql_query(query, conn, params=(client_id,))
        except Exception as e:
            self.logger.error(f"Error getting client portfolios: {str(e)}")
            raise
        finally:
            if conn and self.db_path != ":memory:":
                conn.close()

    def get_product_returns(self, product_id: int, start_date: datetime,
                          end_date: datetime) -> pd.DataFrame:
        """Get historical returns for a specific product."""
        conn = None
        try:
            conn = self._get_connection()
            query = """
                SELECT date, return_value
                FROM Returns
                WHERE product_id = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            return pd.read_sql_query(
                query,
                conn,
                params=(product_id, start_date, end_date)
            )
        except Exception as e:
            self.logger.error(f"Error getting product returns: {str(e)}")
            raise
        finally:
            if conn and self.db_path != ":memory:":
                conn.close()

    def cleanup_test_data(self):
        """Clean up test data from the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete data from all tables in reverse order of dependencies
                cursor.execute("DELETE FROM Deals")
                cursor.execute("DELETE FROM PerformanceMetrics")
                cursor.execute("DELETE FROM PortfolioHoldings")
                cursor.execute("DELETE FROM Products")
                cursor.execute("DELETE FROM Portfolios")
                cursor.execute("DELETE FROM Managers")
                cursor.execute("DELETE FROM Clients")
                
                conn.commit()
                self.logger.info("Test data cleaned up successfully")
                
        except Exception as e:
            self.logger.error(f"Error cleaning up test data: {str(e)}")
            raise

    def store_market_data(self, data: pd.DataFrame):
        """Store market data in the database."""
        try:
            # Validate input data - explicitly check if data is None, empty, or has no rows
            if data is None:
                raise ValueError("Cannot store None as market data")
            if isinstance(data, pd.DataFrame):
                if data.empty:
                    raise ValueError("Cannot store empty DataFrame")
                if len(data) == 0 or len(data.columns) == 0:
                    raise ValueError("Market data DataFrame has no data or columns")
            else:
                raise ValueError("Market data must be a pandas DataFrame")
                
            # Ensure 'date' column exists
            if 'date' not in data.columns:
                raise ValueError("Market data must contain a 'date' column")
                
            # Check if we have any valid symbol data
            symbol_columns = [col for col in data.columns 
                            if col != 'date' and '_' in col]
            if not symbol_columns:
                raise ValueError("No valid market data columns found")
                
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get unique symbols from column names (excluding 'date')
            symbols = set(col.split('_')[0] for col in data.columns 
                        if col != 'date' and '_' in col)
            
            # Store data for each symbol
            for symbol in symbols:
                for idx, row in data.iterrows():
                    date = row['date'] if isinstance(row['date'], str) else row['date'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check for required columns
                    required_columns = [f'{symbol}_open', f'{symbol}_high', f'{symbol}_low', 
                                     f'{symbol}_close', f'{symbol}_volume']
                    missing_columns = [col for col in required_columns if col not in data.columns]
                    if missing_columns:
                        raise ValueError(f"Missing required columns for symbol {symbol}: {missing_columns}")
                    
                    # Handle NaN values and convert to appropriate types
                    open_price = float(row[f'{symbol}_open']) if pd.notna(row[f'{symbol}_open']) else 0.0
                    high_price = float(row[f'{symbol}_high']) if pd.notna(row[f'{symbol}_high']) else 0.0
                    low_price = float(row[f'{symbol}_low']) if pd.notna(row[f'{symbol}_low']) else 0.0
                    close_price = float(row[f'{symbol}_close']) if pd.notna(row[f'{symbol}_close']) else 0.0
                    volume = int(row[f'{symbol}_volume']) if pd.notna(row[f'{symbol}_volume']) else 0
                    
                    try:
                        cursor.execute("""
                            INSERT OR REPLACE INTO MarketData (
                                symbol, date, open, high, low, close, volume
                            ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """, (
                            symbol,
                            date,
                            open_price,
                            high_price,
                            low_price,
                            close_price,
                            volume
                        ))
                    except sqlite3.Error as e:
                        self.logger.error(f"Error inserting data for symbol {symbol} at date {date}: {str(e)}")
                        raise
            
            conn.commit()
            cursor.close()
            if self.db_path != ":memory:":
                conn.close()
            
            self.logger.info(f"Market data stored successfully for symbols: {', '.join(symbols)}")
            
        except Exception as e:
            self.logger.error(f"Error storing market data: {str(e)}")
            raise

    def store_portfolio_features(self, portfolio_id: int, data: pd.DataFrame):
        """Store portfolio features in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate cumulative returns if portfolio_returns exists
                if 'portfolio_returns' in data.columns:
                    cumulative_returns = data['portfolio_returns'].cumsum()
                
                for idx, row in data.iterrows():
                    cursor.execute("""
                        INSERT OR REPLACE INTO PerformanceMetrics (
                            portfolio_id, date, total_value, daily_return, cumulative_return
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        portfolio_id,
                        row['date'].strftime('%Y-%m-%d %H:%M:%S'),  # Convert timestamp to string
                        float(row['portfolio_value']) if 'portfolio_value' in row else 0.0,
                        float(row['portfolio_returns']) if 'portfolio_returns' in row else 0.0,
                        float(cumulative_returns.iloc[idx]) if 'portfolio_returns' in row else 0.0
                    ))
                
                conn.commit()
                self.logger.info("Portfolio features stored successfully")
                
        except Exception as e:
            self.logger.error(f"Error storing portfolio features: {str(e)}")
            raise

    def store_risk_metrics(self, portfolio_id: int, risk_metrics: Dict[str, float]):
        """Store risk metrics in the database."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO PerformanceMetrics (
                        portfolio_id, date, volatility, sharpe_ratio, total_value, daily_return, cumulative_return
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    portfolio_id,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),  # Convert timestamp to string
                    risk_metrics.get('volatility', 0.0),
                    risk_metrics.get('sharpe_ratio', 0.0),
                    0.0,  # total_value (not available in risk metrics)
                    0.0,  # daily_return (not available in risk metrics)
                    0.0   # cumulative_return (not available in risk metrics)
                ))
                
                conn.commit()
                self.logger.info("Risk metrics stored successfully")
                
        except Exception as e:
            self.logger.error(f"Error storing risk metrics: {str(e)}")
            raise

    def get_market_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Get market data for a specific symbol and date range."""
        try:
            conn = self._get_connection()
            query = """
                SELECT date, open, high, low, close, volume
                FROM MarketData
                WHERE symbol = ? AND date BETWEEN ? AND ?
                ORDER BY date
            """
            
            # Convert dates to string format
            start_str = start_date.strftime('%Y-%m-%d %H:%M:%S')
            end_str = end_date.strftime('%Y-%m-%d %H:%M:%S')
            
            df = pd.read_sql_query(
                query, 
                conn, 
                params=(symbol, start_str, end_str),
                parse_dates=['date']
            )
            
            # Calculate returns
            df['returns'] = df['close'].pct_change(fill_method=None)
            
            if self.db_path != ":memory:":
                conn.close()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {str(e)}")
            raise

    def get_available_tickers(self) -> list:
        """Get a list of all available ticker symbols."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT symbol FROM Products ORDER BY symbol ASC")
            tickers = [row[0] for row in cursor.fetchall()]
            
            return tickers
            
        except Exception as e:
            if self.db_path != ":memory:":
                conn.rollback()
            raise e
        finally:
            if self.db_path != ":memory:":
                conn.close()
    
    def get_client_ids(self) -> list:
        """Get a list of all client IDs."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT client_id FROM Clients ORDER BY client_id ASC")
            client_ids = [row[0] for row in cursor.fetchall()]
            
            return client_ids
            
        except Exception as e:
            self.logger.error(f"Error getting client IDs: {str(e)}")
            return []
        finally:
            if self.db_path != ":memory:":
                conn.close()
    
    def get_manager_ids(self) -> list:
        """Get a list of all manager IDs."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT manager_id FROM Managers ORDER BY manager_id ASC")
            manager_ids = [row[0] for row in cursor.fetchall()]
            
            return manager_ids
            
        except Exception as e:
            self.logger.error(f"Error getting manager IDs: {str(e)}")
            return []
        finally:
            if self.db_path != ":memory:":
                conn.close()
    
    def get_client_name(self, client_id: int) -> str:
        """Get the name of a client by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM Clients WHERE client_id = ?", (client_id,))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            return "Unknown Client"
            
        except Exception as e:
            self.logger.error(f"Error getting client name: {str(e)}")
            return "Error"
        finally:
            if self.db_path != ":memory:":
                conn.close()
    
    def get_manager_name(self, manager_id: int) -> str:
        """Get the name of a manager by ID."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT name FROM Managers WHERE manager_id = ?", (manager_id,))
            result = cursor.fetchone()
            
            if result:
                return result[0]
            return "Unknown Manager"
            
        except Exception as e:
            self.logger.error(f"Error getting manager name: {str(e)}")
            return "Error"
        finally:
            if self.db_path != ":memory:":
                conn.close()

    def store_positions(self, portfolio_id: int, positions_df: pd.DataFrame):
        """Store positions for a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio
            positions_df: DataFrame containing position information
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if required columns exist, otherwise skip
            required_columns = ['symbol', 'shares']
            if not all(col in positions_df.columns for col in required_columns):
                missing = [col for col in required_columns if col not in positions_df.columns]
                self.logger.warning(f"Missing required columns for positions: {missing}")
                return
            
            # First, delete existing positions for this portfolio
            cursor.execute("DELETE FROM Positions WHERE portfolio_id = ?", (portfolio_id,))
            
            # Insert each position
            for _, position in positions_df.iterrows():
                try:
                    # Get current date
                    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Get optional values with defaults
                    entry_price = float(position.get('entry_price', 0.0))
                    current_price = float(position.get('current_price', 0.0))
                    weight = float(position.get('weight', 0.0))
                    
                    cursor.execute("""
                        INSERT INTO Positions (
                            portfolio_id, date, symbol, shares, entry_price, current_price, weight
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        portfolio_id,
                        current_date,
                        position['symbol'],
                        float(position['shares']),
                        entry_price,
                        current_price,
                        weight
                    ))
                except Exception as e:
                    self.logger.error(f"Error inserting position: {str(e)}")
                    raise
            
            conn.commit()
            self.logger.info(f"Stored {len(positions_df)} positions for portfolio {portfolio_id}")
            
            if self.db_path != ":memory:":
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing positions: {str(e)}")
            raise

    def store_performance_metrics(self, portfolio_id: int, performance_data: dict):
        """Store performance metrics for a portfolio.
        
        Args:
            portfolio_id: ID of the portfolio
            performance_data: Dictionary containing performance metrics and portfolio values
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Store portfolio values over time
            if 'portfolio_values' in performance_data:
                portfolio_values = performance_data['portfolio_values']
                
                # Check if we have a Series with DatetimeIndex
                if isinstance(portfolio_values, pd.Series) and isinstance(portfolio_values.index, pd.DatetimeIndex):
                    for date, value in portfolio_values.items():
                        date_str = date.strftime('%Y-%m-%d %H:%M:%S')
                        
                        # Check if PerformanceMetrics table has the required columns
                        cursor.execute("PRAGMA table_info(PerformanceMetrics)")
                        columns = [column[1] for column in cursor.fetchall()]
                        
                        # Only store total_value if the column exists
                        if 'total_value' in columns:
                            try:
                                cursor.execute("""
                                    INSERT OR REPLACE INTO PerformanceMetrics (
                                        portfolio_id, date, total_value
                                    ) VALUES (?, ?, ?)
                                """, (
                                    portfolio_id,
                                    date_str,
                                    float(value)
                                ))
                            except Exception as e:
                                self.logger.error(f"Error inserting performance metric at {date_str}: {str(e)}")
            
            # Store overall metrics
            if 'metrics' in performance_data:
                metrics = performance_data['metrics']
                
                # Get the combined metrics (overall performance)
                combined_metrics = metrics.get('combined', {})
                if combined_metrics:
                    # Get current date
                    current_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Check which metrics we can store
                    cursor.execute("PRAGMA table_info(PerformanceMetrics)")
                    columns = [column[1] for column in cursor.fetchall()]
                    
                    # Build column list and values dynamically based on available columns
                    available_metrics = {
                        'sharpe_ratio': combined_metrics.get('sharpe_ratio', 0.0),
                        'volatility': combined_metrics.get('volatility', 0.0),
                        'max_drawdown': combined_metrics.get('max_drawdown', 0.0),
                        'total_return': combined_metrics.get('total_return', 0.0)
                    }
                    
                    # Filter metrics to only include those with corresponding columns
                    insert_columns = ['portfolio_id', 'date']
                    insert_values = [portfolio_id, current_date]
                    
                    for metric_name, metric_value in available_metrics.items():
                        if metric_name in columns:
                            insert_columns.append(metric_name)
                            insert_values.append(float(metric_value))
                    
                    # Only proceed if we have metrics to insert
                    if len(insert_columns) > 2:
                        columns_str = ', '.join(insert_columns)
                        placeholders = ', '.join(['?'] * len(insert_values))
                        
                        try:
                            cursor.execute(
                                f"INSERT OR REPLACE INTO PerformanceMetrics ({columns_str}) VALUES ({placeholders})",
                                insert_values
                            )
                        except Exception as e:
                            self.logger.error(f"Error inserting performance metrics: {str(e)}")
            
            conn.commit()
            self.logger.info(f"Stored performance metrics for portfolio {portfolio_id}")
            
            if self.db_path != ":memory:":
                conn.close()
            
        except Exception as e:
            self.logger.error(f"Error storing performance metrics: {str(e)}")
            raise

    def store_returns(self, returns_df: pd.DataFrame):
        """Store asset returns in the database.
        
        Args:
            returns_df: DataFrame with columns [symbol, date, daily_return]
                        and optionally [cumulative_return, period]
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Ensure required columns are present
            required_cols = ['symbol', 'date', 'daily_return']
            if not all(col in returns_df.columns for col in required_cols):
                missing = [col for col in required_cols if col not in returns_df.columns]
                raise ValueError(f"Missing required columns for returns: {missing}")
            
            # Insert returns data
            for _, row in returns_df.iterrows():
                # Convert date to string if needed
                date = row['date']
                if isinstance(date, datetime):
                    date = date.strftime('%Y-%m-%d')
                
                # Get optional values
                cumulative_return = row.get('cumulative_return', None)
                period = row.get('period', None)
                
                # Insert/update the return record
                cursor.execute("""
                    INSERT OR REPLACE INTO Returns 
                    (symbol, date, daily_return, cumulative_return, period)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    row['symbol'],
                    date,
                    float(row['daily_return']),
                    float(cumulative_return) if cumulative_return is not None else None,
                    period
                ))
            
            conn.commit()
            self.logger.info(f"Stored {len(returns_df)} return records")
            
            if self.db_path != ":memory:":
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Error storing returns: {str(e)}")
            raise
            
    def get_returns(self, symbol: str = None, start_date: datetime = None, 
                   end_date: datetime = None, period: str = None) -> pd.DataFrame:
        """Get asset returns from the database with optional filtering.
        
        Args:
            symbol: Optional symbol to filter by
            start_date: Optional start date for range
            end_date: Optional end date for range
            period: Optional period to filter by (e.g., 'Training', 'Evaluation')
            
        Returns:
            DataFrame with returns data
        """
        try:
            conn = self._get_connection()
            
            # Build query with parameters
            query = "SELECT * FROM Returns WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
                
            if start_date:
                query += " AND date >= ?"
                params.append(start_date.strftime('%Y-%m-%d'))
                
            if end_date:
                query += " AND date <= ?"
                params.append(end_date.strftime('%Y-%m-%d'))
                
            if period:
                query += " AND period = ?"
                params.append(period)
                
            query += " ORDER BY date"
            
            # Execute query
            df = pd.read_sql_query(query, conn, params=tuple(params), parse_dates=['date'])
            
            if self.db_path != ":memory:":
                conn.close()
                
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting returns: {str(e)}")
            raise
    
    def calculate_asset_returns(self, update_database: bool = True) -> pd.DataFrame:
        """Calculate asset returns from price data and optionally store in Returns table.
        
        Args:
            update_database: If True, calculated returns will be stored in the database
            
        Returns:
            DataFrame with calculated returns
        """
        try:
            conn = self._get_connection()
            
            # Get unique symbols from MarketData
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM MarketData")
            symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                self.logger.warning("No market data found to calculate returns")
                return pd.DataFrame()
            
            # Prepare returns DataFrame
            all_returns = pd.DataFrame()
            
            for symbol in symbols:
                # Get price data ordered by date
                query = """
                    SELECT date, close 
                    FROM MarketData 
                    WHERE symbol = ? 
                    ORDER BY date
                """
                
                prices_df = pd.read_sql_query(query, conn, params=(symbol,), parse_dates=['date'])
                
                if len(prices_df) <= 1:
                    self.logger.warning(f"Not enough price data for {symbol} to calculate returns")
                    continue
                
                # Calculate daily returns
                prices_df['daily_return'] = prices_df['close'].pct_change()
                
                # Calculate cumulative returns
                prices_df['cumulative_return'] = (1 + prices_df['daily_return'].fillna(0)).cumprod() - 1
                
                # Create returns DataFrame for this symbol
                symbol_returns = pd.DataFrame({
                    'symbol': symbol,
                    'date': prices_df['date'],
                    'daily_return': prices_df['daily_return'],
                    'cumulative_return': prices_df['cumulative_return']
                }).dropna()
                
                # Add to combined DataFrame
                all_returns = pd.concat([all_returns, symbol_returns])
            
            # Store returns in database if requested
            if update_database and not all_returns.empty:
                self.store_returns(all_returns)
                
            if self.db_path != ":memory:":
                conn.close()
                
            return all_returns
            
        except Exception as e:
            self.logger.error(f"Error calculating asset returns: {str(e)}")
            raise
