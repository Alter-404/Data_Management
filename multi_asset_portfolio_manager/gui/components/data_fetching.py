"""
Data Fetching Component for Portfolio Manager GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
from datetime import datetime, timedelta
import threading
import yfinance as yf

class DataFetchingFrame(ttk.Frame):
    """Frame for fetching and managing data."""
    
    def __init__(self, parent, db_manager, data_collector, status_var):
        """Initialize the data fetching frame."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.data_collector = data_collector
        self.status_var = status_var
        
        # Initialize asset lists
        self.available_assets = []
        self.selected_assets = []
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create the frame widgets."""
        # Create main layout with tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create market data tab
        self.market_data_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.market_data_tab, text="Market Data")
        
        # Create market data widgets
        self.create_market_data_widgets()
        
    def create_market_data_widgets(self):
        """Create widgets for the market data tab."""
        # Create frame for asset selection
        asset_selection_frame = ttk.LabelFrame(self.market_data_tab, text="Asset Selection")
        asset_selection_frame.pack(fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        # Universe selection
        universe_frame = ttk.Frame(asset_selection_frame)
        universe_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(universe_frame, text="Asset Universe:").pack(side=tk.LEFT, padx=5, pady=5)
        self.universe_var = tk.StringVar(value="US Equities")
        universe_options = ["All", "US Equities", "EU Equities", "Global Equities", "Commodities", "ETFs", "Cryptocurrencies", "Custom"]
        universe_dropdown = ttk.Combobox(universe_frame, textvariable=self.universe_var, values=universe_options, state="readonly")
        universe_dropdown.pack(side=tk.LEFT, padx=5, pady=5)
        universe_dropdown.bind("<<ComboboxSelected>>", lambda e: self.on_universe_change())
        
        # Add refresh button next to universe dropdown
        refresh_button = ttk.Button(universe_frame, text="Refresh Tickers", command=self.refresh_available_tickers)
        refresh_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Asset selection lists
        list_frame = ttk.Frame(asset_selection_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Available assets
        available_frame = ttk.LabelFrame(list_frame, text="Available Assets")
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.available_assets_var = tk.StringVar()
        self.available_assets_listbox = tk.Listbox(available_frame, listvariable=self.available_assets_var, selectmode=tk.MULTIPLE, height=10)
        self.available_assets_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to available assets listbox
        available_scrollbar = ttk.Scrollbar(available_frame, orient="vertical", command=self.available_assets_listbox.yview)
        available_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.available_assets_listbox.configure(yscrollcommand=available_scrollbar.set)
        
        # Buttons frame
        button_frame = ttk.Frame(list_frame)
        button_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Add buttons
        ttk.Button(button_frame, text=">", command=self.add_selected_assets).pack(pady=2)
        ttk.Button(button_frame, text="<", command=self.remove_selected_assets).pack(pady=2)
        ttk.Button(button_frame, text=">>", command=self.select_all_assets).pack(pady=2)
        ttk.Button(button_frame, text="<<", command=self.clear_selected_assets).pack(pady=2)
        
        # Selected assets
        selected_frame = ttk.LabelFrame(list_frame, text="Selected Assets")
        selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.selected_assets_var = tk.StringVar()
        self.selected_assets_listbox = tk.Listbox(selected_frame, listvariable=self.selected_assets_var, selectmode=tk.MULTIPLE, height=10)
        self.selected_assets_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar to selected assets listbox
        selected_scrollbar = ttk.Scrollbar(selected_frame, orient="vertical", command=self.selected_assets_listbox.yview)
        selected_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.selected_assets_listbox.configure(yscrollcommand=selected_scrollbar.set)
        
        # Date range selection
        date_frame = ttk.LabelFrame(self.market_data_tab, text="Date Range")
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Start date
        start_frame = ttk.Frame(date_frame)
        start_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(start_frame, text="Start Date:").pack(side=tk.LEFT, padx=5)
        #self.start_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"))
        self.start_date_var = tk.StringVar(value="2020-01-01")
        ttk.Entry(start_frame, textvariable=self.start_date_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # End date
        end_frame = ttk.Frame(date_frame)
        end_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Label(end_frame, text="End Date:").pack(side=tk.LEFT, padx=5)
        #self.end_date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        self.end_date_var = tk.StringVar(value="2024-12-31")
        ttk.Entry(end_frame, textvariable=self.end_date_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # Quick date range buttons
        quick_dates_frame = ttk.Frame(date_frame)
        quick_dates_frame.pack(fill=tk.X, padx=5, pady=5)
        ttk.Button(quick_dates_frame, text="1M", command=lambda: self.set_date_range(30)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_dates_frame, text="3M", command=lambda: self.set_date_range(90)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_dates_frame, text="6M", command=lambda: self.set_date_range(180)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_dates_frame, text="1Y", command=lambda: self.set_date_range(365)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_dates_frame, text="3Y", command=lambda: self.set_date_range(1095)).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_dates_frame, text="5Y", command=lambda: self.set_date_range(1825)).pack(side=tk.LEFT, padx=2)
        
        # Action buttons
        action_frame = ttk.Frame(self.market_data_tab)
        action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(action_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Fetch button
        ttk.Button(action_frame, text="Fetch Data", command=self.fetch_data).pack(side=tk.LEFT, padx=5, pady=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(self.market_data_tab, text="Data Preview")
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create a treeview for data preview
        columns = ("Symbol", "Date", "Open", "High", "Low", "Close", "Volume")
        self.preview_tree = ttk.Treeview(preview_frame, columns=columns, show="headings", height=10)
        for col in columns:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100, anchor=tk.CENTER)
        
        # Add scrollbar to treeview
        scrollbar = ttk.Scrollbar(preview_frame, orient="vertical", command=self.preview_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.preview_tree.configure(yscrollcommand=scrollbar.set)
        self.preview_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Initialize asset lists based on selected universe
        self.initialize_asset_lists()
        
    def initialize_asset_lists(self):
        """Initialize the asset lists based on selected universe."""
        universe = self.universe_var.get()
        
        # Get tickers from database first
        db_tickers = self.db_manager.get_available_tickers()
        
        if universe == "All":
            # Combine all assets from different universes
            all_assets = set()  # Using set to avoid duplicates
            
            # US Equities
            us_equities = [
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "LYFT",
                "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW", "USB", "WFC", "C", "PGR", "AIG",
                "JNJ", "UNH", "PFE", "MRK", "ABBV", "ABT", "BMY", "AMGN", "GILD", "MRNA", "BNTX", "DHR", "SYK", "BSX",
                "PG", "KO", "PEP", "WMT", "HD", "NKE", "MCD", "SBUX", "DIS", "NFLX", "CMCSA", "TGT", "COST", "LMT",
                "CAT", "BA", "GE", "MMM", "HON", "UPS", "FDX", "DE", "WM", "CSX", "UNP", "RTX", "ETN", "EMR"
            ]
            all_assets.update(us_equities)
            
            # EU Equities
            eu_equities = [
                "BMW.DE", "SIE.DE", "ALV.DE", "BAYN.DE", "DTE.DE", "EOAN.DE", "HEI.DE", "MUV2.DE",
                "BNP.PA", "SAN.PA", "AIR.PA", "OR.PA", "MC.PA", "CS.PA", "BN.PA", "AI.PA", "DG.PA", "KER.PA",
                "ASML", "INGA.AS", "PHIA.AS", "ABN.AS", "KPN.AS", "WKL.AS", "AD.AS",
                "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "SREN.SW", "SCMN.SW", "GIVN.SW", "AMS.SW",
                "ERIC-B.ST", "SAND.ST", "VOLV-B.ST", "ATCO-A.ST", "ESSITY-B.ST", "HM-B.ST", "SBB-B.ST", "TELIA.ST",
                "SAN.MC", "BBVA.MC", "TEF.MC", "IBE.MC", "ITX.MC", "REP.MC", "ACS.MC", "AENA.MC", "MAP.MC", "MEL.MC",
                "UCG.MI", "ENI.MI", "ISP.MI", "G.MI", "BMPS.MI", "MB.MI", "PIRC.MI", "AMP.MI"
            ]
            all_assets.update(eu_equities)
            
            # Global Equities (excluding duplicates)
            global_equities = [
                "0700.HK", "9988.HK", "BABA", "JD", "PDD", "BIDU", "7203.T", "9984.T", "005930.KS", "035720.KS",
                "HSBC", "BP.L", "XOM", "CVX", "COP", "EOG", "CNQ.TO", "SU.TO", "ENB.TO",
                "NVO", "AZN.L", "GSK.L", "NVS", "NSRGY"
            ]
            all_assets.update(global_equities)
            
            # Commodities
            commodities = [
                "GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV",
                "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "UNG", "BNO", "UCO", "SCO",
                "HG=F", "ZC=F", "ZS=F", "ZW=F",
                "KC=F", "CT=F", "OJ=F", "SB=F", "CC=F",
                "HE=F"
            ]
            all_assets.update(commodities)
            
            # ETFs
            etfs = [
                "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "VWO", "EFA", "EEM", "ACWI", "VT",
                "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU", "XLC", "XBI", "XHB",
                "AGG", "TLT", "LQD", "HYG", "BND", "BNDX", "MUB", "TIP", "SHY", "IEI", "IEF", "TLH",
                "MTUM", "QUAL", "VLUE", "SIZE", "USMV", "SPLV", "SPHB", "SPHQ", "SPHD", "SPYD",
                "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWH", "EWY", "EWT", "EWM", "EWS", "EWP", "EWQ",
                "VNQ", "REM", "MLPA", "AMLP", "GDX", "GDXJ", "IAU", "SIVR"
            ]
            all_assets.update(etfs)
            
            # Cryptocurrencies
            cryptocurrencies = [
                "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
                "UNI-USD", "AAVE-USD", "MKR-USD", "SUSHI-USD", "YFI-USD", "CRV-USD", "SNX-USD",
                "MATIC-USD", "OP-USD", "ARB-USD", "LRC-USD", "BOBA-USD",
                "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "ALICE-USD",
                "XMR-USD", "ZEC-USD", "DASH-USD", "ROSE-USD",
                "DOT-USD", "LINK-USD", "ATOM-USD", "NEAR-USD", "FTM-USD", "ONE-USD", "EGLD-USD"
            ]
            all_assets.update(cryptocurrencies)
            
            # Add any database tickers that aren't already included
            if db_tickers:
                all_assets.update(db_tickers)
            
            # Convert set back to sorted list
            self.available_assets = sorted(list(all_assets))
            
        elif universe == "US Equities":
            self.available_assets = [
                # Technology
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC", "CRM", "ADBE", "PYPL", "UBER", "LYFT",
                # Financials
                "JPM", "BAC", "GS", "MS", "V", "MA", "AXP", "BLK", "SCHW", "USB", "WFC", "C", "PGR", "AIG",
                # Healthcare
                "JNJ", "UNH", "PFE", "MRK", "ABBV", "ABT", "BMY", "AMGN", "GILD", "MRNA", "BNTX", "DHR", "SYK", "BSX",
                # Consumer
                "PG", "KO", "PEP", "WMT", "HD", "NKE", "MCD", "SBUX", "DIS", "NFLX", "CMCSA", "TGT", "COST", "LMT",
                # Industrial
                "CAT", "BA", "GE", "MMM", "HON", "UPS", "FDX", "DE", "WM", "CSX", "UNP", "RTX", "ETN", "EMR"
            ]
            # Add any US tickers from database
            if db_tickers:
                # Filter for US equities (no hyphens, typically all caps)
                db_us_tickers = [t for t in db_tickers if "-" not in t and not any(c in t for c in ['.', ':'])]
                # Add to list if not already present
                for ticker in db_us_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "Global Equities":
            self.available_assets = [
                # US Tech
                "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC",
                # Asian Tech
                "0700.HK", "9988.HK", "BABA", "JD", "PDD", "BIDU", "7203.T", "9984.T", "005930.KS", "035720.KS",
                # European Tech
                "ASML", "SAP", "SIE.DE", "ERIC-B.ST", "NOKIA.HE", "TELIA.ST", "TEL2-B.ST", "TDC.CO",
                # Global Financials
                "HSBC", "JPM", "BAC", "GS", "MS", "BNP.PA", "SAN.MC", "INGA.AS", "UBS", "CS", "DB", "AXP", "V", "MA",
                # Global Energy
                "SHEL", "TTE", "BP.L", "XOM", "CVX", "COP", "EOG", "CNQ.TO", "SU.TO", "ENB.TO",
                # Global Healthcare
                "NVO", "JNJ", "UNH", "PFE", "MRK", "ABBV", "ABT", "BMY", "AMGN", "AZN.L", "GSK.L", "NVS",
                # Global Consumer
                "NSRGY", "KO", "PEP", "WMT", "HD", "NKE", "MCD", "SBUX", "DIS", "NFLX", "CMCSA", "TGT", "COST",
                # Global Industrial
                "CAT", "BA", "GE", "MMM", "HON", "UPS", "FDX", "DE", "WM", "CSX", "UNP", "RTX", "ETN", "EMR"
            ]
            # Add any global tickers from database
            if db_tickers:
                # Filter for global (includes exchange identifiers)
                db_global_tickers = [t for t in db_tickers if "." in t or ":" in t or ("-" not in t and t.isupper())]
                # Add to list if not already present
                for ticker in db_global_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "EU Equities":
            self.available_assets = [
                # German Stocks
                "BMW.DE", "SIE.DE", "ALV.DE", "BAYN.DE", "DTE.DE", "EOAN.DE", "HEI.DE", "MUV2.DE",
                # French Stocks
                "BNP.PA", "SAN.PA", "AIR.PA", "OR.PA", "MC.PA", "CS.PA", "BN.PA", "AI.PA", "DG.PA", "KER.PA",
                # Dutch Stocks
                "ASML", "INGA.AS", "PHIA.AS", "ABN.AS", "KPN.AS", "WKL.AS", "AD.AS",
                # Swiss Stocks
                "NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW", "ZURN.SW", "ABBN.SW", "SREN.SW", "SCMN.SW", "GIVN.SW", "AMS.SW",
                # Nordic Stocks
                "ERIC-B.ST", "SAND.ST", "VOLV-B.ST", "ATCO-A.ST", "ESSITY-B.ST", "HM-B.ST", "SBB-B.ST", "TELIA.ST",
                # Spanish Stocks
                "SAN.MC", "BBVA.MC", "TEF.MC", "IBE.MC", "ITX.MC", "REP.MC", "ACS.MC", "AENA.MC", "MAP.MC", "MEL.MC",
                # Italian Stocks
                "UCG.MI", "ENI.MI", "ISP.MI", "G.MI", "BMPS.MI", "MB.MI", "PIRC.MI", "AMP.MI"
            ]
            # Add any EU tickers from database
            if db_tickers:
                # Filter for EU equities (typically have .DE, .FR, .NL, etc.)
                db_eu_tickers = [t for t in db_tickers if any(t.endswith(ext) for ext in ['.DE', '.FR', '.NL', '.ES', '.IT', '.CH', '.SE', '.DK', '.NO', '.BE', '.IE', '.PT', '.FI', '.AT', '.GR'])]
                # Add to list if not already present
                for ticker in db_eu_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "Commodities":
            self.available_assets = [
                # Precious Metals
                "GC=F", "SI=F", "PL=F", "PA=F", "GLD", "SLV",
                # Energy
                "CL=F", "BZ=F", "NG=F", "RB=F", "HO=F", "USO", "UNG", "BNO", "UCO", "SCO",
                # Industrial Metals
                "HG=F", "ZC=F", "ZS=F", "ZW=F",
                # Agricultural
                "KC=F", "CT=F", "OJ=F", "SB=F", "CC=F",
                # Livestock
                "HE=F"
            ]
            # Add any commodity tickers from database
            if db_tickers:
                # Filter for commodities (typically have =F suffix)
                db_commodity_tickers = [t for t in db_tickers if "=F" in t]
                # Add to list if not already present
                for ticker in db_commodity_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "ETFs":
            self.available_assets = [
                # Broad Market ETFs
                "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "IVV", "VEA", "VWO", "EFA", "EEM", "ACWI", "VT",
                # Sector ETFs
                "XLK", "XLF", "XLE", "XLV", "XLI", "XLP", "XLY", "XLB", "XLRE", "XLU", "XLC", "XBI", "XHB",
                # Fixed Income ETFs
                "AGG", "TLT", "LQD", "HYG", "BND", "BNDX", "MUB", "TIP", "SHY", "IEI", "IEF", "TLH", "TLT",
                # Commodity ETFs
                "GLD", "SLV", "USO", "UNG", "PPLT", "PALL", "JJN", "CPER", "JO", "CORN", "WEAT", "SOYB",
                # Factor ETFs
                "MTUM", "QUAL", "VLUE", "SIZE", "USMV", "SPLV", "SPHB", "SPHQ", "SPHD", "SPYD",
                # International ETFs
                "EWJ", "EWG", "EWU", "EWC", "EWA", "EWZ", "EWH", "EWY", "EWT", "EWM", "EWS", "EWP", "EWQ",
                # Alternative ETFs
                "VNQ", "REM", "MLPA", "AMLP", "GDX", "GDXJ", "IAU", "SIVR"
            ]
            # Add any ETF tickers from database
            if db_tickers:
                # Filter for ETFs (typically no special characters)
                db_etf_tickers = [t for t in db_tickers if "-" not in t and not any(c in t for c in ['.', ':', '='])]
                # Add to list if not already present
                for ticker in db_etf_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "Cryptocurrencies":
            self.available_assets = [
                # Major Cryptocurrencies
                "BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "AVAX-USD", "DOGE-USD",
                # DeFi Tokens
                "UNI-USD", "AAVE-USD", "MKR-USD", "SUSHI-USD", "YFI-USD", "CRV-USD", "SNX-USD",
                # Layer 2 Solutions
                "MATIC-USD", "OP-USD", "ARB-USD", "LRC-USD", "BOBA-USD", "METIS-USD",
                # Gaming & Metaverse
                "SAND-USD", "MANA-USD", "AXS-USD", "ENJ-USD", "ILV-USD", "ALICE-USD", "TLM-USD",
                # Privacy Coins
                "XMR-USD", "ZEC-USD", "DASH-USD", "SCRT-USD", "ROSE-USD", "BEAM-USD", "ARRR-USD", "PIVX-USD",
                # Smart Contract Platforms
                "DOT-USD", "LINK-USD", "ATOM-USD", "NEAR-USD", "FTM-USD", "ONE-USD", "EGLD-USD"
            ]
            # Add any crypto tickers from database
            if db_tickers:
                # Filter for crypto (typically has -USD suffix)
                db_crypto_tickers = [t for t in db_tickers if "-USD" in t]
                # Add to list if not already present
                for ticker in db_crypto_tickers:
                    if ticker not in self.available_assets:
                        self.available_assets.append(ticker)
        elif universe == "Custom":
            # For custom universe, just use all database tickers
            self.available_assets = db_tickers if db_tickers else []
        
        # Update listbox
        self.available_assets_var.set(self.available_assets)
        
        # Clear selected assets
        self.selected_assets = []
        self.selected_assets_var.set(self.selected_assets)
        
        # Show info about available assets
        self.status_var.set(f"Found {len(self.available_assets)} assets for {universe} universe")
        
    def on_universe_change(self):
        """Handle change in asset universe selection."""
        self.initialize_asset_lists()
        
    def add_selected_assets(self):
        """Add selected assets to the selected list."""
        selections = self.available_assets_listbox.curselection()
        for i in selections:
            asset = self.available_assets[i]
            if asset not in self.selected_assets:
                self.selected_assets.append(asset)
        
        # Update the listbox
        self.selected_assets_var.set(self.selected_assets)
        
    def remove_selected_assets(self):
        """Remove selected assets from the selected list."""
        selections = self.selected_assets_listbox.curselection()
        for i in sorted(selections, reverse=True):
            del self.selected_assets[i]
        
        # Update the listbox
        self.selected_assets_var.set(self.selected_assets)
        
    def select_all_assets(self):
        """Select all available assets."""
        # Select all items in the available assets listbox
        self.available_assets_listbox.select_set(0, tk.END)
        # Add all available assets to selected assets
        self.selected_assets = self.available_assets.copy()
        # Update the selected assets listbox
        self.selected_assets_var.set(self.selected_assets)

    def clear_selected_assets(self):
        """Clear all selected assets."""
        self.selected_assets = []
        self.selected_assets_var.set(self.selected_assets)
        
    def set_date_range(self, days):
        """Set the date range to the specified number of days from today."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        self.start_date_var.set(start_date.strftime("%Y-%m-%d"))
        self.end_date_var.set(end_date.strftime("%Y-%m-%d"))
        
    def fetch_data(self):
        """Fetch market data for selected assets."""
        # Get selected symbols
        if not self.selected_assets:
            messagebox.showinfo("No Selection", "Please select at least one asset.")
            return
            
        # Validate date range
        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            
            if start_date >= end_date:
                messagebox.showwarning("Date Error", "Start date must be before end date.")
                return
                
        except ValueError:
            messagebox.showwarning("Date Error", "Please enter valid dates in YYYY-MM-DD format.")
            return
            
        # Clear the preview
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
            
        # Start background thread for data collection
        thread = threading.Thread(
            target=self._fetch_data_thread,
            args=(self.selected_assets, start_date, end_date),
            daemon=True
        )
        thread.start()

    def _fetch_data_thread(self, symbols, start_date, end_date):
        """Background thread for fetching data."""
        try:
            # Use data collector from src module to fetch data
            self.after(0, lambda: self.status_var.set(f"Fetching market data for {len(symbols)} symbols..."))
            
            # Progress updates
            total_symbols = len(symbols)
            all_data_for_preview = pd.DataFrame()  # For preview only
            all_data_for_db = pd.DataFrame()       # Format required by database
            
            for i, symbol in enumerate(symbols):
                progress_pct = (i / total_symbols) * 100
                self.after(0, lambda pct=progress_pct: self.progress_var.set(pct))
                self.after(0, lambda s=symbol: self.status_var.set(f"Fetching data for {s}..."))
                
                try:
                    # Fetch data for a single symbol
                    clean_symbol = symbol.strip()  # Remove any whitespace
                    if clean_symbol:
                        # Fetch data using the data collector
                        symbol_data = self.data_collector.get_historical_data(clean_symbol, start_date, end_date)
                        
                        # Store data in database
                        if symbol_data is not None and not symbol_data.empty:
                            # Add symbol column for preview
                            symbol_data_with_symbol = symbol_data.copy()
                            if 'Symbol' not in symbol_data_with_symbol.columns:
                                symbol_data_with_symbol['Symbol'] = clean_symbol
                            
                            # Keep a copy for preview
                            all_data_for_preview = pd.concat([all_data_for_preview, symbol_data_with_symbol])
                            
                            # Format data for database storage
                            db_data = self._prepare_data_for_db(clean_symbol, symbol_data)
                            if not all_data_for_db.empty:
                                all_data_for_db = pd.merge(all_data_for_db, db_data, on='date', how='outer')
                            else:
                                all_data_for_db = db_data.copy()
                            
                            self.after(0, lambda s=clean_symbol: self.status_var.set(f"Data for {s} processed successfully."))
                        else:
                            self.after(0, lambda s=clean_symbol: self.status_var.set(f"No data found for {s}."))
                except Exception as e:
                    self.after(0, lambda s=symbol, err=str(e): self.status_var.set(f"Error fetching data for {s}: {err}"))
            
            # Store all data in database at once
            if not all_data_for_db.empty:
                try:
                    self.after(0, lambda: self.status_var.set("Storing data in database..."))
                    self.db_manager.store_market_data(all_data_for_db)
                    self.after(0, lambda: self.status_var.set("All data saved to database successfully."))
                    
                    # Calculate and store returns from the fetched data
                    self.after(0, lambda: self.status_var.set("Calculating and storing returns..."))
                    
                    # Create returns data for each symbol
                    returns_data = []
                    for symbol in symbols:
                        close_col = f'{symbol}_close'
                        if close_col in all_data_for_db.columns:
                            # Calculate returns using pct_change
                            returns = all_data_for_db[close_col].pct_change()
                            returns = returns.dropna()  # Drop NaN values (first row)
                            
                            # Create DataFrame with returns, skipping first row to match the dropna() result
                            symbol_returns = pd.DataFrame({
                                'symbol': symbol,
                                'date': all_data_for_db['date'].iloc[1:],  # Skip first row
                                'daily_return': returns.values
                            })
                            
                            # Calculate cumulative returns
                            symbol_returns['cumulative_return'] = (1 + symbol_returns['daily_return']).cumprod() - 1
                            
                            returns_data.append(symbol_returns)
                    
                    # Combine all returns and store in database
                    if returns_data:
                        all_returns = pd.concat(returns_data)
                        self.db_manager.store_returns(all_returns)
                        self.after(0, lambda: self.status_var.set(f"Stored returns data for {len(symbols)} symbols"))
                    
                    # Verify data was stored
                    conn = self.db_manager._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM MarketData")
                    count = cursor.fetchone()[0]
                    self.after(0, lambda c=count: self.status_var.set(f"Database now contains {c} market data records."))
                    
                    if self.db_manager.db_path != ":memory:":
                        conn.close()
                except Exception as e:
                    self.after(0, lambda err=str(e): self.status_var.set(f"Error saving data to database: {err}"))
                    self.after(0, lambda err=str(e): messagebox.showerror("Database Error", f"Failed to save data to database: {err}"))
            else:
                self.after(0, lambda: self.status_var.set("No data to store in database."))
            
            # Set progress to 100% when done
            self.after(0, lambda: self.progress_var.set(100))
            
            # Update preview with collected data
            if not all_data_for_preview.empty:
                self.after(0, lambda d=all_data_for_preview: self._update_preview(d))
            
            # Display returns data in a separate tab
            self.after(0, self.display_returns_data)
            
            self.after(0, lambda: self.status_var.set(f"Data fetching completed for {len(symbols)} symbols."))
            
        except Exception as e:
            self.after(0, lambda err=str(e): self.status_var.set(f"Error in data fetching: {err}"))
            self.after(0, lambda err=str(e): messagebox.showerror("Error", f"Failed to fetch data: {err}"))
            
    def display_returns_data(self):
        """Display the returns data in a separate tab."""
        try:
            # Check if we need to create the results notebook first
            if not hasattr(self, 'result_notebook'):
                self.result_notebook = ttk.Notebook(self.market_data_tab)
                self.result_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
                
                # Move preview tree to a tab
                preview_tab = ttk.Frame(self.result_notebook)
                self.result_notebook.add(preview_tab, text="Market Data")
                self.preview_tree.master.pack_forget()
                self.preview_tree.master = preview_tab
                self.preview_tree.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Check if returns tab exists, create if not
            returns_tab_exists = False
            for tab in self.result_notebook.tabs():
                if self.result_notebook.tab(tab, "text") == "Returns":
                    returns_tab_exists = True
                    returns_tab = tab
                    break
            
            if not returns_tab_exists:
                returns_tab = ttk.Frame(self.result_notebook)
                self.result_notebook.add(returns_tab, text="Returns")
                
                # Create treeview for returns
                returns_columns = ("Symbol", "Date", "Daily Return", "Cumulative Return")
                self.returns_tree = ttk.Treeview(returns_tab, columns=returns_columns, show="headings", height=20)
                
                # Configure columns
                for col in returns_columns:
                    self.returns_tree.heading(col, text=col)
                    width = 150 if col == "Date" else 100
                    self.returns_tree.column(col, width=width, anchor=tk.CENTER)
                
                # Add scrollbars
                y_scrollbar = ttk.Scrollbar(returns_tab, orient="vertical", command=self.returns_tree.yview)
                y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
                self.returns_tree.configure(yscrollcommand=y_scrollbar.set)
                
                # Pack treeview
                self.returns_tree.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
            else:
                # Clear existing returns data
                self.returns_tree.delete(*self.returns_tree.get_children())
            
            # Get returns data from database
            returns_df = self.db_manager.get_returns()
            
            # If returns data exists, display it
            if not returns_df.empty:
                # Sort by date (newest first)
                returns_df = returns_df.sort_values(by=['date', 'symbol'], ascending=[False, True])
                
                # Add data to treeview
                for _, row in returns_df.iterrows():
                    self.returns_tree.insert("", "end", values=(
                        row['symbol'],
                        row['date'].strftime('%Y-%m-%d'),
                        f"{row['daily_return']:.4f}",
                        f"{row['cumulative_return']:.4f}" if 'cumulative_return' in row and pd.notna(row['cumulative_return']) else "N/A"
                    ))
                
                # Select the Returns tab
                self.result_notebook.select(returns_tab)
                
                # Update status
                self.status_var.set(f"Displayed {len(returns_df)} return records")
            else:
                self.status_var.set("No returns data available")
        
        except Exception as e:
            self.status_var.set(f"Error displaying returns: {str(e)}")
            messagebox.showerror("Error", f"Error displaying returns: {str(e)}")
    
    def _prepare_data_for_db(self, symbol, data):
        """Prepare data for database storage in the format expected by store_market_data."""
        # Create dataframe with required structure
        db_data = pd.DataFrame()
        
        # Check if date is in the dataframe
        if 'Date' in data.columns:
            db_data['date'] = data['Date']
        else:
            # Try to get date from index if it's a DatetimeIndex
            if isinstance(data.index, pd.DatetimeIndex):
                db_data['date'] = data.index
            else:
                # Create a date column with today's date as fallback
                db_data['date'] = datetime.now()
        
        # Add required columns with symbol prefix
        if 'Open' in data.columns:
            db_data[f'{symbol}_open'] = data['Open']
        
        if 'High' in data.columns:
            db_data[f'{symbol}_high'] = data['High']
        
        if 'Low' in data.columns:
            db_data[f'{symbol}_low'] = data['Low']
        
        if 'Close' in data.columns:
            db_data[f'{symbol}_close'] = data['Close']
        
        if 'Volume' in data.columns:
            db_data[f'{symbol}_volume'] = data['Volume']
        
        return db_data
    
    def _update_preview(self, data):
        """Update the data preview with fetched data."""
        # Clear existing data
        for item in self.preview_tree.get_children():
            self.preview_tree.delete(item)
        
        if data.empty:
            self.status_var.set("No data fetched")
            return
        
        # Make a copy to avoid modifying original data
        preview_data = data.copy()
        
        # Format date column if exists
        if 'Date' in preview_data.columns:
            try:
                # Check if Date column is datetime type before using dt accessor
                if pd.api.types.is_datetime64_any_dtype(preview_data['Date']):
                    preview_data['Date'] = preview_data['Date'].dt.strftime('%Y-%m-%d')
            except Exception as e:
                # If any error occurs, just keep the Date column as is
                pass
        
        # Format numeric columns to fewer decimal places
        for col in preview_data.select_dtypes(include=['float']).columns:
            preview_data[col] = preview_data[col].round(4)
        
        # Configure columns - get column names from dataframe
        columns = list(preview_data.columns)
        
        # Clear existing columns from treeview
        self.preview_tree["columns"] = []
        
        # Set new columns
        self.preview_tree["columns"] = columns
        
        # Configure each column
        for col in columns:
            # Set column heading
            self.preview_tree.heading(col, text=col)
            
            # Calculate column width based on content
            col_values = preview_data[col].astype(str)
            max_width = max(len(str(col)), col_values.str.len().max()) * 7
            width = min(max_width, 150)  # Cap width
            
            # Set column width
            self.preview_tree.column(col, width=width, minwidth=50)
        
        # Add data rows (limit to 100 rows for performance)
        for idx, row in preview_data.head(100).iterrows():
            # Convert to string to handle any data type
            values = [str(row[col]) for col in columns]
            self.preview_tree.insert("", "end", values=values)
        
        self.status_var.set(f"Preview showing {min(100, len(preview_data))} of {len(preview_data)} records")
    
    def refresh_available_tickers(self):
        """Refresh the list of available tickers from the database."""
        try:
            db_tickers = self.db_manager.get_available_tickers()
            if db_tickers:
                self.status_var.set(f"Found {len(db_tickers)} tickers in database")
            else:
                self.status_var.set("No tickers found in database")
            
            # Re-initialize asset lists based on current universe
            self.initialize_asset_lists()
        except Exception as e:
            self.status_var.set(f"Error refreshing tickers: {str(e)}")
            messagebox.showerror("Error", f"Failed to refresh tickers: {str(e)}")

# For simpledialog
from tkinter import simpledialog 