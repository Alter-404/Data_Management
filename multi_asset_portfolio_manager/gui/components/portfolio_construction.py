"""
Portfolio Construction Component for Portfolio Manager GUI
"""
import os
import sys
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import project modules
from src.data_management.database_manager import DatabaseManager
from src.data_management.data_collector import DataCollector
from src.portfolio_optimization.backtester import normalize_datetime, PortfolioBacktester
from src.portfolio_optimization.strategies import (
    LowRiskStrategy, LowTurnoverStrategy, HighYieldEquityStrategy, create_strategy
)
from src.portfolio_optimization.optimizer import PortfolioOptimizer, PortfolioManager

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PortfolioConstructionFrame(ttk.Frame):
    """Frame for constructing portfolios using trained strategies."""
    
    def __init__(self, parent, db_manager, data_collector, status_var):
        """Initialize the portfolio construction frame."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.data_collector = data_collector
        self.status_var = status_var
        self.construction_results = None
        
        # Create instance variables
        self.logger = logging.getLogger(__name__)
        self.progress_var = tk.DoubleVar(value=0)
        
        # Portfolio dropdown
        self.portfolio_var = tk.StringVar()
        self.portfolio_ids = {}  # Map from display name to ID
        self.portfolio_details = {}  # Map from portfolio ID to details (name, client, strategy, universe)
        
        # Strategy dropdown
        self.strategy_var = tk.StringVar(value="Low Risk")
        self.strategies = ["Low Risk", "Low Turnover", "High Yield Equity"]
        
        # Universe dropdown
        self.universe_var = tk.StringVar(value="US Equities")
        self.universes = ["EU Equities", "US Equities", "Global Equities",  "Commodities", "Cryptocurrencies", "ETFs", "Mixed Assets"]
        
        # Period variables
        self.train_start_var = tk.StringVar(value="2020-01-01")
        self.train_end_var = tk.StringVar(value="2022-12-31")
        self.eval_start_var = tk.StringVar(value="2023-01-01")
        self.eval_end_var = tk.StringVar(value="2024-12-31")
        
        # Settings variables
        self.capital_var = tk.StringVar(value="100000")
        self.max_positions_var = tk.StringVar(value="10")
        self.frequency_var = tk.StringVar(value="Weekly")
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create the frame widgets."""
        # Main layout - split into left and right panels
        self.left_frame = ttk.Frame(self, width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        self.left_frame.pack_propagate(False)  # Maintain width
        
        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create sections in the left panel
        self.create_portfolio_selection_section()
        self.create_strategy_section()
        self.create_construction_params_section()
        self.create_action_buttons()
        
        # Create sections in the right panel
        self.create_results_section()
        
    def create_portfolio_selection_section(self):
        """Create the portfolio selection section."""
        section = ttk.LabelFrame(self.left_frame, text="Portfolio Selection")
        section.pack(fill=tk.X, padx=5, pady=5)
        
        # Portfolio selection
        ttk.Label(section, text="Portfolio:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.portfolio_var = tk.StringVar()
        self.portfolio_combo = ttk.Combobox(section, textvariable=self.portfolio_var, state="readonly", width=25)
        self.portfolio_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        self.portfolio_combo.bind("<<ComboboxSelected>>", self.on_portfolio_selected)
        
        # Client selection
        ttk.Label(section, text="Client:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.client_var = tk.StringVar()
        self.client_combo = ttk.Combobox(section, textvariable=self.client_var, state="readonly", width=25)
        self.client_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Load available portfolios and clients
        ttk.Button(section, text="Refresh", command=self.load_portfolios).grid(row=2, column=1, sticky="e", padx=5, pady=5)
        
        # Load portfolios and clients initially
        self.load_portfolios()
        
    def on_portfolio_selected(self, event):
        """Handler for portfolio selection events."""
        selected_portfolio = self.portfolio_var.get()
        if not selected_portfolio or selected_portfolio not in self.portfolio_ids:
            return
            
        try:
            portfolio_id = self.portfolio_ids[selected_portfolio]
            
            # Use stored portfolio details from the portfolio_details dictionary
            if portfolio_id in self.portfolio_details:
                details = self.portfolio_details[portfolio_id]
                client_name = details["client"]
                strategy = details["strategy"]
                asset_universe = details["universe"]
                
                # Set client
                if client_name in self.client_names:
                    self.client_var.set(client_name)
                
                # Set strategy
                if strategy and strategy in self.strategies:
                    self.strategy_var.set(strategy)
                
                # Set asset universe
                if asset_universe and asset_universe in self.universes:
                    self.universe_var.set(asset_universe)
            
        except Exception as e:
            self.logger.error(f"Error setting portfolio details: {str(e)}")
        
    def create_strategy_section(self):
        """Create the strategy selection section."""
        section = ttk.LabelFrame(self.left_frame, text="Strategy Selection")
        section.pack(fill=tk.X, padx=5, pady=5)
        
        # Strategy selection
        ttk.Label(section, text="Strategy:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.strategy_var = tk.StringVar()
        strategies = ["Low Risk", "Low Turnover", "High Yield Equity"]
        self.strategy_combo = ttk.Combobox(section, textvariable=self.strategy_var, values=strategies, state="readonly", width=25)
        self.strategy_combo.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Asset universe
        ttk.Label(section, text="Asset Universe:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.universe_var = tk.StringVar()
        universes = ["US Equities", "Global Equities", "EU Equities", "Cryptocurrencies", "Commodities", "ETFs", "Mixed Assets"]
        self.universe_combo = ttk.Combobox(section, textvariable=self.universe_var, values=universes, state="readonly", width=25)
        self.universe_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
    def create_construction_params_section(self):
        """Create the portfolio construction parameters section."""
        section = ttk.LabelFrame(self.left_frame, text="Construction Parameters")
        section.pack(fill=tk.X, padx=5, pady=5)
        
        # Initial capital
        ttk.Label(section, text="Initial Capital:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.capital_var = tk.StringVar(value="100000")
        ttk.Entry(section, textvariable=self.capital_var, width=15).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Training period
        ttk.Label(section, text="Training Period:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        training_frame = ttk.Frame(section)
        training_frame.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        self.train_start_var = tk.StringVar(value="2023-01-01")
        ttk.Entry(training_frame, textvariable=self.train_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(training_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.train_end_var = tk.StringVar(value="2023-12-31")
        ttk.Entry(training_frame, textvariable=self.train_end_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Evaluation period
        ttk.Label(section, text="Evaluation Period:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        eval_frame = ttk.Frame(section)
        eval_frame.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        self.eval_start_var = tk.StringVar(value="2024-01-01")
        ttk.Entry(eval_frame, textvariable=self.eval_start_var, width=10).pack(side=tk.LEFT, padx=2)
        ttk.Label(eval_frame, text="to").pack(side=tk.LEFT, padx=2)
        self.eval_end_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(eval_frame, textvariable=self.eval_end_var, width=10).pack(side=tk.LEFT, padx=2)
        
        # Trading frequency
        ttk.Label(section, text="Trading Frequency:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.frequency_var = tk.StringVar(value="Weekly")
        frequencies = ["Daily", "Weekly", "Monthly"]
        ttk.Combobox(section, textvariable=self.frequency_var, values=frequencies, state="readonly", width=15).grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        # Max positions
        ttk.Label(section, text="Max Positions:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.max_positions_var = tk.StringVar(value="10")
        ttk.Entry(section, textvariable=self.max_positions_var, width=15).grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
    def create_action_buttons(self):
        """Create action buttons."""
        button_frame = ttk.Frame(self.left_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        ttk.Button(button_frame, text="Construct Portfolio", command=self.run_portfolio_construction).pack(fill=tk.X, pady=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_portfolio_results).pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar(value=0.0)
        ttk.Progressbar(button_frame, variable=self.progress_var, mode="determinate").pack(fill=tk.X, pady=10)
        
    def create_results_section(self):
        """Create the results section."""
        # Notebook for different result views
        self.results_notebook = ttk.Notebook(self.right_frame)
        self.results_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Performance tab
        self.performance_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.performance_tab, text="Performance")
        
        # Create matplotlib figure for performance chart
        self.performance_figure = plt.Figure(figsize=(10, 6), dpi=100)
        self.performance_canvas = FigureCanvasTkAgg(self.performance_figure, self.performance_tab)
        self.performance_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Metrics tab
        self.metrics_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.metrics_tab, text="Metrics")
        
        # Create treeview for metrics
        columns = ("Metric", "Training", "Evaluation", "Combined")
        self.metrics_tree = ttk.Treeview(self.metrics_tab, columns=columns, show="headings", height=15)
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=150, anchor=tk.CENTER)
        self.metrics_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Holdings tab
        self.holdings_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.holdings_tab, text="Holdings")
        
        # Create treeview for holdings
        columns = ("Symbol", "Shares", "Entry Price", "Current Price", "Value", "Weight", "Return")
        self.holdings_tree = ttk.Treeview(self.holdings_tab, columns=columns, show="headings", height=15)
        for col in columns:
            self.holdings_tree.heading(col, text=col)
            self.holdings_tree.column(col, width=100, anchor=tk.CENTER)
        self.holdings_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Trades tab
        self.trades_tab = ttk.Frame(self.results_notebook)
        self.results_notebook.add(self.trades_tab, text="Trades")
        
        # Create treeview for trades
        columns = ("Date", "Symbol", "Action", "Shares", "Price", "Amount", "Period")
        self.trades_tree = ttk.Treeview(self.trades_tab, columns=columns, show="headings", height=15)
        for col in columns:
            self.trades_tree.heading(col, text=col)
            self.trades_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.trades_tab, orient="vertical", command=self.trades_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.trades_tree.configure(yscrollcommand=scrollbar.set)
        self.trades_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
    def load_portfolios(self):
        """Load portfolios and clients from database."""
        try:
            # Get clients
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # Load clients
            cursor.execute("SELECT client_id, name FROM Clients ORDER BY name")
            clients = cursor.fetchall()
            self.client_ids = [c[0] for c in clients]
            self.client_names = [c[1] for c in clients]
            self.client_combo['values'] = self.client_names
            
            if self.client_names:
                self.client_combo.current(0)
            
            # Load portfolios with strategy and asset_universe
            cursor.execute("""
                SELECT p.portfolio_id, p.name, c.name, p.strategy, p.asset_universe 
                FROM Portfolios p
                JOIN Clients c ON p.client_id = c.client_id
                ORDER BY p.name
            """)
            portfolios = cursor.fetchall()
            
            # Store portfolios, their IDs, and their details
            self.portfolio_ids = {f"{p[1]} ({p[2]})": p[0] for p in portfolios}
            self.portfolio_details = {p[0]: {"name": p[1], "client": p[2], "strategy": p[3], "universe": p[4]} for p in portfolios}
            self.portfolio_combo['values'] = list(self.portfolio_ids.keys())
            
            if self.portfolio_combo['values']:
                self.portfolio_combo.current(0)
                # Trigger the selection handler to populate client and strategy fields
                self.on_portfolio_selected(None)
                
            self.status_var.set(f"Loaded {len(clients)} clients and {len(portfolios)} portfolios")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolios and clients: {str(e)}")
            self.status_var.set("Error loading portfolios and clients")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def run_portfolio_construction(self):
        """Run the portfolio construction process."""
        # Validate inputs
        try:
            # Check if a portfolio is selected
            if not self.portfolio_var.get():
                messagebox.showwarning("Input Error", "Please select a portfolio.")
                return
                
            # Check if a strategy is selected
            if not self.strategy_var.get():
                messagebox.showwarning("Input Error", "Please select a strategy.")
                return
                
            # Parse dates
            train_start = datetime.strptime(self.train_start_var.get(), "%Y-%m-%d")
            train_end = datetime.strptime(self.train_end_var.get(), "%Y-%m-%d")
            eval_start = datetime.strptime(self.eval_start_var.get(), "%Y-%m-%d")
            eval_end = datetime.strptime(self.eval_end_var.get(), "%Y-%m-%d")
            
            # Validate date ranges
            if train_start >= train_end:
                messagebox.showwarning("Date Error", "Training start date must be before training end date.")
                return
                
            if eval_start >= eval_end:
                messagebox.showwarning("Date Error", "Evaluation start date must be before evaluation end date.")
                return
                
            if eval_start < train_end:
                messagebox.showwarning("Date Error", "Evaluation period should start after training period ends.")
                return
                
            # Parse other numeric inputs
            initial_capital = float(self.capital_var.get())
            max_positions = int(self.max_positions_var.get())
            
            if initial_capital <= 0:
                messagebox.showwarning("Input Error", "Initial capital must be positive.")
                return
                
            if max_positions <= 0:
                messagebox.showwarning("Input Error", "Maximum positions must be positive.")
                return
                
            # Get selected portfolio ID
            selected_portfolio = self.portfolio_var.get()
            if selected_portfolio not in self.portfolio_ids:
                messagebox.showwarning("Portfolio Error", "Please select a valid portfolio.")
                return
                
            portfolio_id = self.portfolio_ids[selected_portfolio]
            strategy_name = self.strategy_var.get()
            universe_name = self.universe_var.get()
            
            # Start construction in a separate thread
            threading.Thread(
                target=self._run_construction_thread,
                args=(portfolio_id, strategy_name, universe_name, 
                      initial_capital, max_positions,
                      train_start, train_end, 
                      eval_start, eval_end,
                      self.frequency_var.get())
            ).start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid input: {str(e)}")
        except Exception as e:
            messagebox.showerror("Error", f"Error preparing portfolio construction: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
    
    def _run_construction_thread(self, portfolio_id, strategy_name, universe_name, 
                              initial_capital, max_positions,
                              train_start, train_end, 
                              eval_start, eval_end,
                              frequency):
        """Run the portfolio construction in a background thread."""
        try:
            # Update status
            self.after(0, lambda: self.status_var.set("Fetching market data..."))
            self.after(0, lambda: self.progress_var.set(20))
            
            # Normalize dates to avoid timezone issues
            train_start = normalize_datetime(train_start)
            train_end = normalize_datetime(train_end)
            eval_start = normalize_datetime(eval_start)
            eval_end = normalize_datetime(eval_end)
            
            # Get available tickers from database using src package
            available_tickers = self.db_manager.get_available_tickers()
            
            # If no tickers in database, use default lists
            if not available_tickers:
                self.after(0, lambda: self.status_var.set("No market data in database. Using default tickers."))
                
                # Determine symbols based on universe
                if universe_name == "US Equities":
                    symbols = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM", 
                        "JNJ", "V", "PG", "UNH", "HD", "BAC", "MA", "DIS", "ADBE", "CRM"
                    ]
                elif universe_name == "EU Equities":
                    symbols = [
                        "BMW.DE", "SIE.DE", "BAYN.DE", "DTE.DE", "SAP.DE", 
                        "BNP.PA", "SAN.PA", "AIR.PA", "MC.PA", 
                        "ASML", "INGA.AS", "PHIA.AS", 
                        "NESN.SW", "ROG.SW", "NOVN.SW", 
                        "BP.L", "HSBA.L", "GSK.L", "AZN.L"
                    ]
                elif universe_name == "Global Equities":
                    symbols = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "ASML", 
                        "BABA", "HSBC", "SHEL", "NVO", "SONY", "TTE", "SAP"
                    ]
                elif universe_name == "Cryptocurrencies":
                    symbols = [
                        "BTC-USD", "ETH-USD", "BNB-USD", "ADA-USD", "XRP-USD", "SOL-USD", 
                        "DOT-USD", "DOGE-USD", "AVAX-USD", "MATIC-USD", "LINK-USD"
                    ]
                elif universe_name == "Commodities":
                    symbols = [
                        "GC=F", "SI=F", "PL=F", "GLD", "SLV",
                        "CL=F", "BZ=F", "NG=F", "USO", "UNG",
                        "HG=F", "ZC=F", "ZS=F", "ZW=F",
                        "KC=F", "CT=F", "SB=F", "CC=F"
                    ]
                elif universe_name == "ETFs":
                    symbols = [
                        "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", 
                        "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY",
                        "AGG", "TLT", "LQD", "HYG", "BND",
                        "GDX", "IAU", "SIVR", "VNQ"
                    ]
                else:  # Mixed Assets
                    symbols = [
                        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "BTC-USD", "ETH-USD",
                        "GLD", "SLV", "USO", "UNG", "SPY", "QQQ", "IWM", "EFA", "EEM", "AGG"
                    ]
            else:
                # Filter tickers based on universe
                if universe_name == "US Equities":
                    # Basic filtering for US equities - no hyphens, typically all caps
                    symbols = [t for t in available_tickers if "-" not in t and not any(c in t for c in ['.', ':']) and "=" not in t]
                elif universe_name == "EU Equities":
                    # European equities typically have exchange suffixes like .DE, .PA, .AS, .SW, .L
                    symbols = [t for t in available_tickers if any(t.endswith(ext) for ext in ['.DE', '.PA', '.AS', '.SW', '.L', '.MI', '.MC', '.ST'])]
                elif universe_name == "Global Equities":
                    # Include major equities from around the world
                    symbols = [t for t in available_tickers if any(c in t for c in ['.', ':']) or ("-" not in t and "=" not in t and t.isupper())]
                elif universe_name == "Cryptocurrencies":
                    # Crypto assets typically have a -USD suffix
                    symbols = [t for t in available_tickers if "-USD" in t]
                elif universe_name == "Commodities":
                    # Commodities often use =F suffix or are ETFs that track commodities
                    symbols = [t for t in available_tickers if "=" in t or t in ["GLD", "SLV", "USO", "UNG", "DBC"]]
                elif universe_name == "ETFs":
                    # ETFs are typically all caps with no special characters
                    etf_prefixes = ["SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "XL", "IW", "EW", "VN", "AGG", "BND", "GDX", "IAU"]
                    symbols = [t for t in available_tickers if any(t.startswith(prefix) for prefix in etf_prefixes) or 
                              t in ["SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "AGG", "TLT", "LQD", "HYG"]]
                else:  # Mixed Assets
                    # Use all available tickers
                    symbols = available_tickers.copy()
            
            # Ensure we have a minimum of 5 symbols, regardless of max_positions
            min_symbols = min(5, len(symbols))
            
            # Limit to max_positions, but ensure at least min_symbols
            if len(symbols) > max_positions:
                symbols = symbols[:max_positions]
            
            # If we still have no symbols, show error and return
            if not symbols:
                self.after(0, lambda: self.status_var.set("No symbols match the selected universe."))
                self.after(0, lambda: messagebox.showerror("Data Error", "No symbols match the selected universe. Please fetch market data first."))
                return
            
            self.after(0, lambda: self.status_var.set(f"Using {len(symbols)} symbols for construction: {', '.join(symbols[:5])}..."))
            
            # Use the DataCollector from src packages to collect market data directly
            self.after(0, lambda: self.status_var.set(f"Collecting market data for {len(symbols)} symbols..."))
            
            # Create a combined DataFrame for all symbols
            all_data = pd.DataFrame()
            successful_symbols = []
            
            for i, symbol in enumerate(symbols):
                progress = 20 + (i / len(symbols) * 20)  # Progress from 20% to 40%
                self.after(0, lambda p=progress: self.progress_var.set(p))
                self.after(0, lambda s=symbol: self.status_var.set(f"Fetching data for {s}..."))
                
                try:
                    # Get data for this symbol using src package's method
                    symbol_data = self.data_collector.get_historical_data(symbol, train_start, eval_end)
                    
                    if symbol_data is not None and not symbol_data.empty:
                        # Format data for backtester (it expects a specific structure)
                        if isinstance(symbol_data.index, pd.DatetimeIndex):
                            # Already has date index, but normalize it
                            df = symbol_data.copy()
                            # Normalize the DatetimeIndex to remove timezone info
                            if df.index.tz is not None:
                                df.index = df.index.map(normalize_datetime)
                        else:
                            # Convert 'Date' column to index if it exists
                            df = symbol_data.copy()
                            if 'Date' in df.columns:
                                # Normalize the Date column values
                                df['Date'] = df['Date'].map(normalize_datetime)
                                df.set_index('Date', inplace=True)
                        
                        # Handle overlapping columns by adding a suffix to any column not in our expected format
                        expected_columns = ['open', 'high', 'low', 'close', 'volume', 'adj close']
                        for col in df.columns:
                            if col.lower() not in expected_columns:
                                # Add suffix to avoid column overlap
                                df = df.rename(columns={col: f"{col}_{symbol}"})
                        
                        # Rename columns to match expected format
                        column_mapping = {}
                        for col in df.columns:
                            if col.lower() in expected_columns:
                                # Format: SYMBOL_lowercase_column_name
                                column_mapping[col] = f"{symbol}_{col.lower().replace(' ', '_')}"
                        
                        df = df.rename(columns=column_mapping)
                        
                        # Make sure we have close data before adding
                        if f"{symbol}_close" in df.columns:
                            # Calculate returns explicitly to ensure it's available
                            df[f"{symbol}_returns"] = df[f"{symbol}_close"].pct_change(fill_method=None)
                            
                            # Merge into all_data
                            if all_data.empty:
                                all_data = df
                            else:
                                # Join on index (date), adding suffixes to avoid column overlaps
                                all_data = all_data.join(df, how='outer', lsuffix='', rsuffix='_right')
                        
                        # Add to successful symbols list
                        successful_symbols.append(symbol)
                
                except Exception as e:
                    self.after(0, lambda s=symbol, err=str(e): self.status_var.set(f"Error fetching data for {s}: {err}"))
                    self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            
            if all_data.empty:
                self.after(0, lambda: self.status_var.set("No data available for selected symbols."))
                self.after(0, lambda: messagebox.showerror("Data Error", "No data available for selected symbols."))
                return
            
            # Fill NA values using linear interpolation for all price columns
            self.after(0, lambda: self.status_var.set("Filling missing values with linear interpolation..."))
            
            # First, identify price columns (like close, open, high, low)
            price_columns = [col for col in all_data.columns if any(
                pattern in col.lower() for pattern in ['_close', '_open', '_high', '_low', '_adj_close', '_price']
            )]
            
            # Fill NA values for each price column using linear interpolation
            for col in price_columns:
                if all_data[col].isna().any():
                    # Use linear interpolation first
                    all_data[col] = all_data[col].interpolate(method='linear')
                    
                    # For any remaining NAs at the start or end of the series, use forward/backward fill
                    all_data[col] = all_data[col].fillna(method='ffill').fillna(method='bfill')
            
            # Recalculate returns based on the interpolated prices
            for symbol in successful_symbols:
                close_col = f"{symbol}_close"
                returns_col = f"{symbol}_returns"
                if close_col in all_data.columns:
                    all_data[returns_col] = all_data[close_col].pct_change(fill_method=None)
            
            # Fill any remaining NAs in returns with zeros
            returns_columns = [col for col in all_data.columns if '_returns' in col]
            for col in returns_columns:
                all_data[col] = all_data[col].fillna(0)
            
            # Fill any other remaining NAs with forward fill then backward fill
            remaining_na_columns = all_data.columns[all_data.isna().any()]
            if len(remaining_na_columns) > 0:
                self.logger.info(f"Filling remaining NA values in columns: {remaining_na_columns}")
                all_data = all_data.fillna(method='ffill').fillna(method='bfill')
            
            # Log NA value count after filling
            na_count = all_data.isna().sum().sum()
            if na_count > 0:
                self.logger.warning(f"After filling, {na_count} NA values remain in the data")
                # As a last resort, fill remaining NAs with zeros
                all_data = all_data.fillna(0)
            else:
                self.logger.info("All NA values successfully filled")
            
            # Ensure all_data has a proper DatetimeIndex for the backtester
            if not isinstance(all_data.index, pd.DatetimeIndex):
                if 'date' in all_data.columns:
                    all_data.set_index('date', inplace=True)
                elif 'Date' in all_data.columns:
                    all_data.set_index('Date', inplace=True)
            
            # Gather the list of successful symbols
            successful_symbols = []
            for column in all_data.columns:
                if '_close' in column:
                    symbol = column.split('_')[0]
                    if symbol not in successful_symbols:
                        successful_symbols.append(symbol)
            
            # Ensure we have at least min_symbols with data
            if len(successful_symbols) < min_symbols:
                self.after(0, lambda: self.status_var.set(f"Warning: Only {len(successful_symbols)} symbols have data, expected at least {min_symbols}."))
                self.logger.warning(f"Only {len(successful_symbols)} symbols have data: {successful_symbols}")
            
            # Update the status with the symbols that have data
            self.after(0, lambda syms=successful_symbols: self.status_var.set(
                f"Successfully loaded data for {len(syms)} symbols: {', '.join(syms[:5])}" + 
                ("..." if len(syms) > 5 else "")
            ))
            
            # Create strategy instance using the imported create_strategy function
            self.after(0, lambda: self.status_var.set(f"Creating {strategy_name} strategy..."))
            risk_profile = strategy_name  # Use strategy name as risk profile
            
            # Update status
            self.after(0, lambda: self.status_var.set("Running backtest..."))
            self.after(0, lambda: self.progress_var.set(40))
            
            # Define a diversification enforcer wrapper for strategies
            class DiversifiedStrategyWrapper:
                def __init__(self, strategy, min_assets=3):
                    self.strategy = strategy
                    self.min_assets = min_assets
                    self.logger = logging.getLogger(__name__)
                
                def generate_signals(self, market_data, portfolio_data):
                    """Generate signals ensuring a minimum level of diversification."""
                    # Get original signals from the base strategy
                    original_weights = self.strategy.generate_signals(market_data, portfolio_data)
                    
                    # Log the original weights for debugging
                    self.logger.info(f"Original strategy weights: {original_weights}")
                    
                    # Count assets with significant allocation
                    significant_assets = [s for s, w in original_weights.items() if w > 0.01]
                    
                    # If we already have enough assets with significant weights, return original
                    if len(significant_assets) >= self.min_assets:
                        return original_weights
                    
                    # Otherwise, need to ensure at least min_assets receive allocation
                    adjusted_weights = {}
                    all_symbols = list(original_weights.keys())
                    
                    # If we have fewer than min_assets total symbols, use all available
                    if len(all_symbols) <= self.min_assets:
                        # Equal allocation to all symbols
                        weight_per_asset = 1.0 / len(all_symbols)
                        for symbol in all_symbols:
                            adjusted_weights[symbol] = weight_per_asset
                        return adjusted_weights
                    
                    # Sort symbols by their original weight (descending)
                    sorted_symbols = sorted(all_symbols, key=lambda s: original_weights.get(s, 0), reverse=True)
                    
                    # Ensure top min_assets symbols have at least some minimum weight
                    min_weight = 0.05  # Minimum 5% allocation per asset
                    remaining_weight = 1.0 - (min_weight * self.min_assets)
                    
                    if remaining_weight < 0:
                        # If minimum weights would exceed 100%, adjust minimum weight
                        min_weight = 1.0 / self.min_assets
                        remaining_weight = 0
                    
                    # Assign minimum weight to top assets and distribute remaining proportionally
                    top_assets = sorted_symbols[:self.min_assets]
                    other_assets = sorted_symbols[self.min_assets:]
                    
                    # First, ensure each top asset gets the minimum
                    for symbol in top_assets:
                        adjusted_weights[symbol] = min_weight
                    
                    # Distribute remaining weight proportionally based on original weights
                    if remaining_weight > 0 and other_assets:
                        # Calculate total weight of remaining assets in original allocation
                        original_other_weight = sum(original_weights.get(s, 0) for s in other_assets)
                        
                        if original_other_weight > 0:
                            # Distribute proportionally
                            for symbol in other_assets:
                                orig_weight = original_weights.get(symbol, 0)
                                if orig_weight > 0:
                                    adjusted_weights[symbol] = (orig_weight / original_other_weight) * remaining_weight
                        else:
                            # If no other assets had weight, distribute equally
                            weight_per_other = remaining_weight / len(other_assets)
                            for symbol in other_assets:
                                adjusted_weights[symbol] = weight_per_other
                    
                    # Normalize to ensure weights sum to 1.0
                    total_weight = sum(adjusted_weights.values())
                    normalized_weights = {s: w/total_weight for s, w in adjusted_weights.items()}
                    
                    self.logger.info(f"Adjusted weights to ensure diversification: {normalized_weights}")
                    
                    return normalized_weights
            
            # Create a custom backtester with modified trade generation
            class ModifiedBacktester(PortfolioBacktester):
                def _generate_trades(
                    self, 
                    current_portfolio, 
                    target_weights, 
                    market_data,
                    date
                ):
                    """Generate trades with lower threshold to include more assets."""
                    trades = []
                    total_value = current_portfolio['total_value']
                    current_weights = current_portfolio['weights']
                    
                    # Log target weights for debugging
                    self.logger.info(f"Target weights for {date}: {target_weights}")
                    
                    # Calculate trades needed for each symbol
                    for symbol, target_weight in target_weights.items():
                        current_weight = current_weights.get(symbol, 0.0)
                        weight_diff = target_weight - current_weight
                        
                        # Use a smaller threshold for small trades
                        if abs(weight_diff) < 0.005:  # Reduced from 0.01
                            continue
                        
                        # Calculate target value and quantity
                        if f'{symbol}_close' in market_data.columns:
                            price = market_data[f'{symbol}_close'].iloc[-1]
                            target_value = total_value * target_weight
                            current_value = total_value * current_weight
                            value_diff = target_value - current_value
                            
                            # Calculate shares to trade
                            quantity = value_diff / price
                            
                            # Add to trades list with even smaller minimum threshold
                            if abs(quantity) > 1e-8:  # Reduced from 1e-6
                                trades.append((symbol, quantity))
                    
                    return trades
            
            # Create backtester with the appropriate strategy
            self.after(0, lambda: self.status_var.set(f"Creating backtester with {strategy_name} strategy..."))
            
            # Create backtester using the imported class
            backtester = ModifiedBacktester(risk_profile=risk_profile)
            
            # Wrap the backtester's strategy with our diversification enforcer
            original_strategy = backtester.portfolio_manager.strategy
            backtester.portfolio_manager.strategy = DiversifiedStrategyWrapper(original_strategy, min_assets=3)
            
            # Convert frequency to trading interval
            if frequency == "Daily":
                trading_interval = 1
            elif frequency == "Weekly":
                trading_interval = 5
            else:  # Monthly
                trading_interval = 20
                
            # Run backtest with separate training and evaluation periods
            self.after(0, lambda: self.status_var.set("Running backtest with market data..."))
            results = backtester.run_backtest(
                market_data=all_data,
                initial_capital=initial_capital,
                start_date=train_start,
                end_date=eval_end,
                training_end_date=train_end  # This ensures proper training/evaluation split
            )
            
            # Store portfolio ID in results
            results['portfolio_id'] = portfolio_id
            
            # Store market data for later use
            results['market_data'] = all_data
            
            # Also store all relevant date info for easy reference
            results['dates'] = {
                'train_start': train_start,
                'train_end': train_end,
                'eval_start': eval_start,
                'eval_end': eval_end,
            }
            
            # Add the list of used symbols for reference
            results['symbols_used'] = successful_symbols
            
            # Log the final holdings
            if 'portfolio' in results and 'holdings' in results['portfolio']:
                holdings = results['portfolio']['holdings']
                self.logger.info(f"Final portfolio holdings: {holdings}")
                total_holdings = len(holdings)
                self.after(0, lambda: self.status_var.set(f"Portfolio construction completed with {total_holdings} assets"))
            
            # Update progress
            self.after(0, lambda: self.progress_var.set(80))
            
            # Store results for later use
            self.construction_results = results
            
            # Update the UI with results
            self.after(0, lambda: self._update_results_display(results))
            
            # Update status
            self.after(0, lambda: self.status_var.set("Portfolio construction completed successfully"))
            self.after(0, lambda: self.progress_var.set(100))
            
            # Automatically save results to database
            self.after(0, lambda r=results: self._auto_save_results(r))
            
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            self.after(0, lambda err=str(e), tb=error_traceback: self.status_var.set(f"Error in portfolio construction: {err}"))
            self.after(0, lambda err=str(e), tb=error_traceback: messagebox.showerror("Construction Error", f"Failed to construct portfolio: {err}\n\nDetails:\n{tb}"))
    
    def _update_results_display(self, results):
        """Update the UI with the construction results."""
        try:
            # Update performance chart
            self._update_performance_chart(results)
            
            # Update metrics
            self._update_metrics_display(results)
            
            # Update holdings
            self._update_holdings_display(results)
            
            # Update trades
            self._update_trades_display(results)
            
            # Set focus to the results notebook
            self.results_notebook.select(0)  # Show the performance tab first
        except Exception as e:
            messagebox.showerror("Display Error", f"Error updating results display: {str(e)}")
    
    def _update_performance_chart(self, results):
        """Update the performance chart."""
        try:
            # Clear the figure
            self.performance_figure.clear()
            
            # Create axes
            ax = self.performance_figure.add_subplot(111)
            
            # Get portfolio values
            if 'portfolio' in results and 'total_value' in results['portfolio']:
                # Get data from portfolio dict structure
                portfolio = results['portfolio']
                dates = portfolio['dates']
                values = portfolio['total_value']
                
                # Convert to pandas Series for easier plotting if needed
                # portfolio_values = pd.Series(values, index=dates)
                
                # Determine which points belong to training vs evaluation
                training_end = None
                
                # Try to get training end date from different possible sources
                if 'dates' in results and 'train_end' in results['dates']:
                    training_end = results['dates']['train_end']
                elif hasattr(results, 'training_end_date'):
                    training_end = results.training_end_date
                
                if training_end:
                    # Create masks for training and evaluation periods
                    training_dates = [d for d in dates if d <= training_end]
                    eval_dates = [d for d in dates if d > training_end]
                    
                    training_values = [values[i] for i, d in enumerate(dates) if d <= training_end]
                    eval_values = [values[i] for i, d in enumerate(dates) if d > training_end]
                    
                    # Plot with different colors for training and evaluation
                    if training_dates:
                        ax.plot(training_dates, training_values, label='Training Period', color='blue')
                    if eval_dates:
                        ax.plot(eval_dates, eval_values, label='Evaluation Period', color='green')
                    
                    # Add vertical line at training end
                    ax.axvline(x=training_end, color='red', linestyle='--', 
                            label='Training/Evaluation Split')
                else:
                    # Just plot everything in one color
                    ax.plot(dates, values, label='Portfolio Value', color='blue')
                
                # Add labels and legend
                ax.set_title('Portfolio Performance')
                ax.set_xlabel('Date')
                ax.set_ylabel('Portfolio Value ($)')
                ax.legend()
                ax.grid(True)
                
                # Format x-axis dates
                self.performance_figure.autofmt_xdate()
                
                # Redraw canvas
                self.performance_canvas.draw()
        except Exception as e:
            self.logger.error(f"Error updating performance chart: {str(e)}")
    
    def _update_metrics_display(self, results):
        """Update the metrics display."""
        try:
            # Clear existing items
            for item in self.metrics_tree.get_children():
                self.metrics_tree.delete(item)
            
            # Add training metrics
            if 'training_metrics' in results:
                training_metrics = results['training_metrics']
                self._add_metrics_to_tree(training_metrics, 'Training')
            
            # Add evaluation metrics
            if 'evaluation_metrics' in results:
                evaluation_metrics = results['evaluation_metrics']
                self._add_metrics_to_tree(evaluation_metrics, 'Evaluation')
            
            # Add combined metrics
            if 'combined_metrics' in results:
                combined_metrics = results['combined_metrics']
                self._add_metrics_to_tree(combined_metrics, 'Combined')
        except Exception as e:
            self.logger.error(f"Error updating metrics display: {str(e)}")
    
    def _add_metrics_to_tree(self, metrics, period):
        """Add metrics for a specific period to the tree."""
        # Format and insert metrics
        for metric_name, value in metrics.items():
            # Format the value based on its type
            if isinstance(value, float):
                if metric_name in ['total_return', 'annualized_return', 'max_drawdown']:
                    formatted_value = f"{value:.2%}"
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            # Check if item exists
            item_exists = False
            for item in self.metrics_tree.get_children():
                if self.metrics_tree.item(item, 'values')[0] == metric_name:
                    item_exists = True
                    # Update existing item
                    values = list(self.metrics_tree.item(item, 'values'))
                    if period == 'Training':
                        values[1] = formatted_value
                    elif period == 'Evaluation':
                        values[2] = formatted_value
                    else:  # Combined
                        values[3] = formatted_value
                    self.metrics_tree.item(item, values=values)
                    break
            
            # Insert new item if it doesn't exist
            if not item_exists:
                values = [metric_name, '', '', '']
                if period == 'Training':
                    values[1] = formatted_value
                elif period == 'Evaluation':
                    values[2] = formatted_value
                else:  # Combined
                    values[3] = formatted_value
                self.metrics_tree.insert("", "end", values=tuple(values))
    
    def _update_holdings_display(self, results):
        """Update the holdings display."""
        try:
            # Clear existing items
            for item in self.holdings_tree.get_children():
                self.holdings_tree.delete(item)
            
            # Get holdings from portfolio
            if 'portfolio' in results and 'holdings' in results['portfolio']:
                holdings = results['portfolio']['holdings']
                
                # Get market data for current prices
                if not holdings:
                    return
                
                # For each holding, add to treeview
                total_value = 0
                for symbol, quantity in holdings.items():
                    # Get current price
                    current_price = self._get_last_price(results, symbol)
                    value = quantity * current_price
                    total_value += value
                    
                    # Calculate entry price (approximation)
                    entry_price = 0
                    if 'trades' in results['portfolio']:
                        trades = [t for t in results['portfolio']['trades'] if t['symbol'] == symbol]
                        if trades:
                            # Check for both 'type' and 'action' keys to handle different trade structures
                            buy_trades = []
                            for t in trades:
                                if ('type' in t and t['type'].upper() == 'BUY') or ('action' in t and t['action'].upper() == 'BUY'):
                                    buy_trades.append(t)
                            
                            if buy_trades:
                                # Calculate average purchase price, checking for different key names
                                total_cost = 0
                                total_shares = 0
                                for t in buy_trades:
                                    # Handle different possible key names
                                    cost = t.get('amount', t.get('value', 0))
                                    shares_count = t.get('shares', t.get('quantity', 0))
                                    total_cost += cost
                                    total_shares += shares_count
                                
                                if total_shares > 0:
                                    entry_price = total_cost / total_shares
                
                    # Calculate return
                    return_pct = ((current_price / entry_price) - 1) * 100 if entry_price > 0 else 0
                    
                    self.holdings_tree.insert("", "end", values=(
                        symbol,
                        f"{quantity:.2f}",
                        f"${entry_price:.2f}",
                        f"${current_price:.2f}",
                        f"${value:.2f}",
                        f"{(value/total_value*100):.2f}%" if total_value > 0 else "0.00%",
                        f"{return_pct:.2f}%"
                    ))
                
                # Add a total row
                self.holdings_tree.insert("", "end", values=(
                    "TOTAL",
                    "",
                    "",
                    "",
                    f"${total_value:.2f}",
                    "100.00%",
                    ""
                ))
        except Exception as e:
            self.logger.error(f"Error updating holdings display: {str(e)}")
    
    def _get_last_price(self, results, symbol):
        """Get the last price of a symbol from results."""
        # Try to get price from last trade
        if 'portfolio' in results and 'trades' in results['portfolio']:
            trades = [t for t in results['portfolio']['trades'] if t['symbol'] == symbol]
            if trades:
                last_trade = max(trades, key=lambda t: t['date'])
                return last_trade.get('price', 0)
        
        # Fallback to last data point of market data
        if 'market_data' in results:
            market_data = results['market_data']
            if f'{symbol}_close' in market_data.columns:
                return market_data[f'{symbol}_close'].iloc[-1]
        
        # Default fallback
        return 0
    
    def _update_trades_display(self, results):
        """Update the trades display."""
        try:
            # Clear existing items
            for item in self.trades_tree.get_children():
                self.trades_tree.delete(item)
            
            # Add trades
            if 'portfolio' in results and 'trades' in results['portfolio']:
                trades = results['portfolio']['trades']
                
                # Sort trades by date (newest first)
                trades_sorted = sorted(trades, key=lambda t: t['date'], reverse=True)
                
                for trade in trades_sorted:
                    date = trade.get('date', '')
                    symbol = trade.get('symbol', '')
                    action = trade.get('action', 'BUY')
                    shares = abs(trade.get('shares', 0))
                    price = trade.get('price', 0)
                    amount = abs(trade.get('amount', 0))
                    period = trade.get('period', 'Unknown')
                    
                    self.trades_tree.insert("", "end", values=(
                        date.strftime('%Y-%m-%d') if isinstance(date, datetime) else date,
                        symbol,
                        action,
                        f"{shares:.2f}",
                        f"${price:.2f}",
                        f"${amount:.2f}",
                        period
                    ))
        except Exception as e:
            self.logger.error(f"Error updating trades display: {str(e)}")
    
    def save_portfolio_results(self):
        """Save the portfolio construction results to the database."""
        if not hasattr(self, 'construction_results'):
            messagebox.showwarning("No Results", "No portfolio construction results to save.")
            return
            
        try:
            # Get the portfolio ID
            selected_portfolio = self.portfolio_var.get()
            if not selected_portfolio:
                messagebox.showwarning("Selection Error", "Please select a portfolio.")
                return
                
            portfolio_id = self.portfolio_ids[selected_portfolio]
            
            # Update status
            self.status_var.set("Saving portfolio results...")
            
            # Extract results
            results = self.construction_results
            
            # Save trades to database
            if 'portfolio' in results and 'trades' in results['portfolio']:
                trades = results['portfolio']['trades']
                # Convert to DataFrame for easier database storage
                trades_df = pd.DataFrame(trades)
                
                # Log original DataFrame info
                print("\n=== Original DataFrame Info ===")
                print(trades_df.info())
                print("\nOriginal columns:")
                print(trades_df.columns.tolist())
                
                # Rename columns to match SQLite table requirements - use original Trades columns
                column_mapping = {}
                
                # If any of our columns are named differently than what's expected, rename them
                if 'quantity' in trades_df.columns and 'shares' not in trades_df.columns:
                    column_mapping['quantity'] = 'shares'
                if 'value' in trades_df.columns and 'amount' not in trades_df.columns:
                    column_mapping['value'] = 'amount'
                if 'type' in trades_df.columns and 'action' not in trades_df.columns:
                    column_mapping['type'] = 'action'
                
                # Only rename if we have mappings
                if column_mapping:
                    trades_df = trades_df.rename(columns=column_mapping)
                
                # Add portfolio_id column
                trades_df['portfolio_id'] = portfolio_id
                
                # Log modified DataFrame info
                print("\n=== Modified DataFrame Info ===")
                print(trades_df.info())
                print("\nModified columns:")
                print(trades_df.columns.tolist())
                print("\nFirst few rows:")
                print(trades_df.head())
                
                # Store new trades
                print("\n=== Storing new trades ===")
                self.db_manager.store_deals(trades_df)
                print(f"Stored {len(trades_df)} new trades")
                
                self.status_var.set("Trades stored successfully")
            else:
                self.status_var.set("No trades found in results")
                print("\nNo trades found in results")
            
            # Save holdings/positions to database
            if 'portfolio' in results and 'holdings' in results['portfolio']:
                holdings = results['portfolio']['holdings']
                # Create a list of holdings data for DataFrame conversion
                holdings_data = []
                for symbol, quantity in holdings.items():
                    # Get current price
                    current_price = self._get_last_price(results, symbol)
                    
                    # Calculate entry price from trades
                    entry_price = 0
                    trades = []
                    if 'trades' in results['portfolio']:
                        trades = [t for t in results['portfolio']['trades'] if t['symbol'] == symbol]
                        buy_trades = []
                        for t in trades:
                            if ('type' in t and t['type'].upper() == 'BUY') or ('action' in t and t['action'].upper() == 'BUY'):
                                buy_trades.append(t)
                        
                        if buy_trades:
                            total_cost = 0
                            total_shares = 0
                            for t in buy_trades:
                                cost = t.get('amount', t.get('value', 0))
                                shares_count = t.get('shares', t.get('quantity', 0))
                                total_cost += cost
                                total_shares += shares_count
                            
                            if total_shares > 0:
                                entry_price = total_cost / total_shares
                    
                    # Calculate position value and weight
                    value = quantity * current_price
                    weight = value / (results['portfolio'].get('total_value', [])[-1] if 'total_value' in results['portfolio'] else 1.0) * 100
                    
                    holdings_data.append({
                        'symbol': symbol,
                        'shares': quantity,
                        'current_price': current_price,
                        'entry_price': entry_price,
                        'value': value,
                        'weight': weight,
                        'portfolio_id': portfolio_id
                    })
                
                if holdings_data:
                    holdings_df = pd.DataFrame(holdings_data)
                    
                    # Store positions
                    self.db_manager.store_positions(portfolio_id, holdings_df)
            
            # Save performance metrics
            if 'combined_metrics' in results and 'portfolio' in results and 'total_value' in results['portfolio']:
                metrics = results['combined_metrics']
                
                # Build daily performance data
                if 'dates' in results['portfolio'] and 'total_value' in results['portfolio']:
                    dates = results['portfolio']['dates']
                    values = results['portfolio']['total_value']
                    
                    # Create a DataFrame with dates and values
                    perf_df = pd.DataFrame({
                        'date': dates,
                        'total_value': values
                    })
                    
                    # Calculate daily returns
                    perf_df['daily_return'] = perf_df['total_value'].pct_change()
                    perf_df.loc[0, 'daily_return'] = 0  # First day has no return
                    
                    # Calculate cumulative returns
                    perf_df['cumulative_return'] = (1 + perf_df['daily_return']).cumprod() - 1
                    
                    # Calculate rolling volatility (20-day window)
                    if len(perf_df) >= 20:
                        perf_df['volatility'] = perf_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
                        
                        # Calculate rolling Sharpe ratio (20-day window)
                        risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
                        excess_return = perf_df['daily_return'] - risk_free_rate
                        perf_df['sharpe_ratio'] = excess_return.rolling(window=20).mean() / perf_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
                    
                    # Add maximum drawdown
                    running_max = perf_df['total_value'].cummax()
                    drawdown = (perf_df['total_value'] / running_max - 1.0)
                    perf_df['max_drawdown'] = drawdown.cummin()
                    
                    # Add portfolio_id for database storage
                    perf_df['portfolio_id'] = portfolio_id
                    
                    # Store each day's metrics in the PerformanceMetrics table
                    conn = self.db_manager._get_connection()
                    cursor = conn.cursor()
                    
                    # First, delete existing metrics for this portfolio
                    cursor.execute("DELETE FROM PerformanceMetrics WHERE portfolio_id = ?", (portfolio_id,))
                    
                    # Prepare date format for SQL
                    perf_df['date_str'] = [d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, datetime) else d for d in perf_df['date']]
                    
                    # Insert each row
                    for _, row in perf_df.iterrows():
                        # Build insert values dynamically based on available columns
                        columns = ['portfolio_id', 'date']
                        values = [portfolio_id, row['date_str']]
                        
                        # Add metrics if available
                        if 'total_value' in row and not pd.isna(row['total_value']):
                            columns.append('total_value')
                            values.append(float(row['total_value']))
                            
                        if 'daily_return' in row and not pd.isna(row['daily_return']):
                            columns.append('daily_return')
                            values.append(float(row['daily_return']))
                            
                        if 'cumulative_return' in row and not pd.isna(row['cumulative_return']):
                            columns.append('cumulative_return')
                            values.append(float(row['cumulative_return']))
                            
                        if 'volatility' in row and not pd.isna(row['volatility']):
                            columns.append('volatility')
                            values.append(float(row['volatility']))
                            
                        if 'sharpe_ratio' in row and not pd.isna(row['sharpe_ratio']):
                            columns.append('sharpe_ratio')
                            values.append(float(row['sharpe_ratio']))
                            
                        if 'max_drawdown' in row and not pd.isna(row['max_drawdown']):
                            columns.append('max_drawdown')
                            values.append(float(row['max_drawdown']))
                        
                        # Also add the total_return from combined metrics to the last day
                        if _ == len(perf_df) - 1 and 'total_return' in metrics:
                            columns.append('total_return')
                            values.append(float(metrics['total_return']))
                        
                        # Create the SQL insert statement
                        columns_str = ', '.join(columns)
                        placeholders = ', '.join(['?'] * len(values))
                        
                        # Insert the metrics
                        cursor.execute(
                            f"INSERT INTO PerformanceMetrics ({columns_str}) VALUES ({placeholders})",
                            values
                        )
                    
                    conn.commit()
                    if self.db_manager.db_path != ":memory:":
                        conn.close()
                    
                    print(f"Stored {len(perf_df)} daily performance metrics in database")
                else:
                    # If we don't have daily data, at least store the overall metrics
                    # Convert to DataFrame for storage
                    portfolio_df = pd.DataFrame({
                        'date': results['portfolio']['dates'],
                        'value': results['portfolio']['total_value']
                    })
                    
                    # Prepare data for database
                    performance_data = {
                        'portfolio_id': portfolio_id,
                        'metrics': metrics,
                        'portfolio_values': portfolio_df
                    }
                    
                    # Store metrics
                    self.db_manager.store_performance_metrics(portfolio_id, performance_data)
            
            messagebox.showinfo("Success", "Portfolio results saved successfully.")
            self.status_var.set("Portfolio results saved successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save portfolio results: {str(e)}")
            self.status_var.set(f"Error saving results: {str(e)}")
    
    def _auto_save_results(self, results):
        """Automatically save portfolio results to database after construction."""
        try:
            # Get the portfolio ID
            selected_portfolio = self.portfolio_var.get()
            if not selected_portfolio or selected_portfolio not in self.portfolio_ids:
                self.logger.error("Cannot auto-save: No valid portfolio selected")
                return
                
            portfolio_id = self.portfolio_ids[selected_portfolio]
            self.status_var.set("Saving portfolio results to database...")
            
            # Save trades to database
            saved_items = []
            
            if 'portfolio' in results and 'trades' in results['portfolio']:
                trades = results['portfolio']['trades']
                
                # Log original trades data
                self.logger.info(f"Original trades data: {len(trades)} trades found")
                if len(trades) > 0:
                    self.logger.info(f"Sample trade keys: {trades[0].keys()}")
                
                trades_df = pd.DataFrame(trades)
                
                # Check if we have trade data with required fields
                if not trades_df.empty and 'symbol' in trades_df.columns:
                    # Ensure date column is properly formatted
                    if 'date' in trades_df.columns:
                        trades_df['date'] = pd.to_datetime(trades_df['date'])
                        trades_df['date'] = trades_df['date'].dt.strftime('%Y-%m-%d')
                    
                    # Rename columns to match database schema - use original Trades columns
                    column_mapping = {}
                    
                    # If any of our columns are named differently than what's expected, rename them
                    if 'quantity' in trades_df.columns and 'shares' not in trades_df.columns:
                        column_mapping['quantity'] = 'shares'
                    if 'value' in trades_df.columns and 'amount' not in trades_df.columns:
                        column_mapping['value'] = 'amount'
                    if 'type' in trades_df.columns and 'action' not in trades_df.columns:
                        column_mapping['type'] = 'action'
                    
                    # Only rename if we have mappings
                    if column_mapping:
                        trades_df = trades_df.rename(columns=column_mapping)
                    
                    # Add portfolio_id column
                    trades_df['portfolio_id'] = portfolio_id
                    
                    # Log processed data
                    self.logger.info(f"Processed trades columns: {trades_df.columns.tolist()}")
                    
                    # Store new trades
                    try:
                        self.db_manager.store_deals(trades_df)
                        self.logger.info(f"Stored {len(trades_df)} trades in database")
                        saved_items.append(f"{len(trades_df)} trades")
                        
                        # Display confirmation in UI
                        msg = f"Saved {len(trades_df)} trades to database"
                        self.status_var.set(msg)
                    except Exception as trade_error:
                        self.logger.error(f"Error storing trades: {str(trade_error)}")
                        self.status_var.set(f"Error storing trades: {str(trade_error)}")
                else:
                    self.logger.warning("Trades DataFrame is empty or missing required columns")
                    self.status_var.set("Warning: Trade data format is invalid")
            else:
                self.logger.warning("No trades data found in results")
                self.status_var.set("Warning: No trades data to save")
            
            # Save holdings/positions to database
            if 'portfolio' in results and 'holdings' in results['portfolio']:
                holdings = results['portfolio']['holdings']
                holdings_data = []
                
                for symbol, quantity in holdings.items():
                    try:
                        current_price = self._get_last_price(results, symbol)
                        
                        # Calculate entry price from trades
                        entry_price = 0
                        trades = []
                        if 'trades' in results['portfolio']:
                            trades = [t for t in results['portfolio']['trades'] if t['symbol'] == symbol]
                            buy_trades = []
                            for t in trades:
                                if ('type' in t and t['type'].upper() == 'BUY') or ('action' in t and t['action'].upper() == 'BUY'):
                                    buy_trades.append(t)
                            
                            if buy_trades:
                                total_cost = 0
                                total_shares = 0
                                for t in buy_trades:
                                    cost = t.get('amount', t.get('value', 0))
                                    shares_count = t.get('shares', t.get('quantity', 0))
                                    total_cost += cost
                                    total_shares += shares_count
                                
                                if total_shares > 0:
                                    entry_price = total_cost / total_shares
                        
                        # Calculate position value and weight
                        value = quantity * current_price
                        weight = value / (results['portfolio'].get('total_value', [])[-1] if 'total_value' in results['portfolio'] else 1.0) * 100
                        
                        holdings_data.append({
                            'symbol': symbol,
                            'shares': quantity,
                            'current_price': current_price,
                            'entry_price': entry_price,
                            'value': value,
                            'weight': weight,
                            'portfolio_id': portfolio_id
                        })
                    except Exception as holding_error:
                        self.logger.error(f"Error processing holding {symbol}: {str(holding_error)}")
                
                if holdings_data:
                    try:
                        holdings_df = pd.DataFrame(holdings_data)
                        self.db_manager.store_positions(portfolio_id, holdings_df)
                        self.logger.info(f"Stored {len(holdings_data)} positions in database")
                        saved_items.append(f"{len(holdings_data)} positions")
                        self.status_var.set(f"Saved {len(holdings_data)} positions to database")
                    except Exception as positions_error:
                        self.logger.error(f"Error storing positions: {str(positions_error)}")
                        self.status_var.set(f"Error storing positions: {str(positions_error)}")
            
            # Save performance metrics
            if 'combined_metrics' in results and 'portfolio' in results and 'total_value' in results['portfolio']:
                try:
                    # Get overall metrics
                    metrics = results['combined_metrics']
                    self.logger.info(f"Performance metrics: {list(metrics.keys())}")
                    
                    # Build daily performance data
                    if 'dates' in results['portfolio'] and 'total_value' in results['portfolio']:
                        dates = results['portfolio']['dates']
                        values = results['portfolio']['total_value']
                        
                        # Create a DataFrame with dates and values
                        perf_df = pd.DataFrame({
                            'date': dates,
                            'total_value': values
                        })
                        
                        # Calculate daily returns
                        perf_df['daily_return'] = perf_df['total_value'].pct_change()
                        perf_df.loc[0, 'daily_return'] = 0  # First day has no return
                        
                        # Calculate cumulative returns
                        perf_df['cumulative_return'] = (1 + perf_df['daily_return']).cumprod() - 1
                        
                        # Calculate rolling volatility (20-day window)
                        if len(perf_df) >= 20:
                            perf_df['volatility'] = perf_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
                            
                            # Calculate rolling Sharpe ratio (20-day window)
                            risk_free_rate = 0.02 / 252  # Assuming 2% annual risk-free rate
                            excess_return = perf_df['daily_return'] - risk_free_rate
                            perf_df['sharpe_ratio'] = excess_return.rolling(window=20).mean() / perf_df['daily_return'].rolling(window=20).std() * np.sqrt(252)
                        
                        # Add maximum drawdown
                        running_max = perf_df['total_value'].cummax()
                        drawdown = (perf_df['total_value'] / running_max - 1.0)
                        perf_df['max_drawdown'] = drawdown.cummin()
                        
                        # Add portfolio_id for database storage
                        perf_df['portfolio_id'] = portfolio_id
                        
                        # Store each day's metrics in the PerformanceMetrics table
                        conn = self.db_manager._get_connection()
                        cursor = conn.cursor()
                        
                        # First, delete existing metrics for this portfolio
                        cursor.execute("DELETE FROM PerformanceMetrics WHERE portfolio_id = ?", (portfolio_id,))
                        
                        # Prepare date format for SQL
                        perf_df['date_str'] = [d.strftime('%Y-%m-%d %H:%M:%S') if isinstance(d, datetime) else d for d in perf_df['date']]
                        
                        # Insert each row
                        for _, row in perf_df.iterrows():
                            # Build insert values dynamically based on available columns
                            columns = ['portfolio_id', 'date']
                            values = [portfolio_id, row['date_str']]
                            
                            # Add metrics if available
                            if 'total_value' in row and not pd.isna(row['total_value']):
                                columns.append('total_value')
                                values.append(float(row['total_value']))
                                
                            if 'daily_return' in row and not pd.isna(row['daily_return']):
                                columns.append('daily_return')
                                values.append(float(row['daily_return']))
                                
                            if 'cumulative_return' in row and not pd.isna(row['cumulative_return']):
                                columns.append('cumulative_return')
                                values.append(float(row['cumulative_return']))
                                
                            if 'volatility' in row and not pd.isna(row['volatility']):
                                columns.append('volatility')
                                values.append(float(row['volatility']))
                                
                            if 'sharpe_ratio' in row and not pd.isna(row['sharpe_ratio']):
                                columns.append('sharpe_ratio')
                                values.append(float(row['sharpe_ratio']))
                                
                            if 'max_drawdown' in row and not pd.isna(row['max_drawdown']):
                                columns.append('max_drawdown')
                                values.append(float(row['max_drawdown']))
                            
                            # Also add the total_return from combined metrics to the last day
                            if _ == len(perf_df) - 1 and 'total_return' in metrics:
                                columns.append('total_return')
                                values.append(float(metrics['total_return']))
                            
                            # Create the SQL insert statement
                            columns_str = ', '.join(columns)
                            placeholders = ', '.join(['?'] * len(values))
                            
                            # Insert the metrics
                            cursor.execute(
                                f"INSERT INTO PerformanceMetrics ({columns_str}) VALUES ({placeholders})",
                                values
                            )
                        
                        conn.commit()
                        if self.db_manager.db_path != ":memory:":
                            conn.close()
                            
                        self.logger.info(f"Stored {len(perf_df)} daily performance metrics in database")
                        saved_items.append(f"{len(perf_df)} daily performance metrics")
                        self.status_var.set(f"Saved {len(perf_df)} daily performance metrics to database")
                    else:
                        # If we don't have daily data, at least store the overall metrics
                        # Convert to DataFrame for storage
                        portfolio_df = pd.DataFrame({
                            'date': results['portfolio']['dates'],
                            'value': results['portfolio']['total_value']
                        })
                        
                        # Prepare data for database
                        performance_data = {
                            'portfolio_id': portfolio_id,
                            'metrics': metrics,
                            'portfolio_values': portfolio_df
                        }
                        
                        # Store metrics
                        self.db_manager.store_performance_metrics(portfolio_id, performance_data)
                
                except Exception as metrics_error:
                    self.logger.error(f"Error storing performance metrics: {str(metrics_error)}")
                    self.status_var.set(f"Error storing metrics: {str(metrics_error)}")
            else:
                self.logger.warning("No performance metrics found in results")
                self.status_var.set("Warning: No performance metrics to save")
            
            # Show final confirmation message based on what was saved
            if saved_items:
                messagebox.showinfo("Database Updated", 
                                   f"Portfolio '{selected_portfolio}' data has been automatically saved to the database:\n"
                                   f"• {', '.join(saved_items)}")
                self.status_var.set("Portfolio results saved to database successfully")
            else:
                messagebox.showwarning("No Data Saved",
                                     "No portfolio data was saved to the database. Please check the logs for details.")
                self.status_var.set("Warning: No portfolio data was saved")
            
        except Exception as e:
            self.logger.error(f"Error auto-saving portfolio results: {str(e)}")
            self.status_var.set(f"Error auto-saving results: {str(e)}")
            messagebox.showerror("Auto-Save Error", f"Could not automatically save to database: {str(e)}\n\nPlease use the 'Save Results' button to try again.") 