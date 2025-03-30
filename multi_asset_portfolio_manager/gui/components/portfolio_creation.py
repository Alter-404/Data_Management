"""
Portfolio Creation Component for Portfolio Manager GUI
"""
import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
from datetime import datetime
import sqlite3

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '...'))
sys.path.append(project_root)

# Import core modules
from src.portfolio_optimization.optimizer import PortfolioManager
from src.portfolio_optimization.strategies import create_strategy

class PortfolioCreationFrame(ttk.Frame):
    """Frame for creating and managing portfolios."""
    
    def __init__(self, parent, db_manager, status_var):
        """Initialize the portfolio creation frame."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.status_var = status_var
        
        # Map strategy names to strategy classes
        self.strategy_mapping = {
            "Low Risk": "Low Risk",
            "Low Turnover": "Low Turnover",
            "High Yield Equity": "High Yield Equity"
        }
        
        # Initialize all variables needed for the UI
        self.client_names = []
        self.client_ids = []
        self.manager_names = []
        self.manager_ids = []
        
        # Initialize UI variables
        self.client_var = tk.StringVar()
        self.manager_var = tk.StringVar()
        self.strategy_var = tk.StringVar()
        self.universe_var = tk.StringVar()
        self.portfolio_name_var = tk.StringVar()
        self.risk_var = tk.StringVar()
        
        # Create UI components
        self.create_widgets()
        
    def create_widgets(self):
        """Create the frame widgets."""
        # Create main layout
        self.left_frame = ttk.Frame(self, width=400)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.right_frame = ttk.Frame(self, width=400)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create notebook for client and manager selection
        self.selection_notebook = ttk.Notebook(self.left_frame)
        self.selection_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create client tab
        self.client_tab = ttk.Frame(self.selection_notebook)
        self.selection_notebook.add(self.client_tab, text="View Clients")
        
        # Create manager tab
        self.manager_tab = ttk.Frame(self.selection_notebook)
        self.selection_notebook.add(self.manager_tab, text="View Managers")
        
        # Create client list view
        self._create_clients_list(self.client_tab)
        
        # Create manager list view
        self._create_managers_list(self.manager_tab)
        
        # Create portfolio section
        self.create_portfolio_section(self.right_frame)
        
        # Load data
        self.load_clients()
        self.load_managers()
        self.load_portfolios()
        
    def _create_clients_list(self, parent):
        """Create the list view for clients."""
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview with scrollbar for clients
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("client_id", "name", "email", "risk_profile")
        self.clients_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=10, yscrollcommand=y_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.clients_tree.yview)
        
        # Set column headings
        self.clients_tree.heading("client_id", text="ID")
        self.clients_tree.heading("name", text="Name")
        self.clients_tree.heading("email", text="Email")
        self.clients_tree.heading("risk_profile", text="Strategy")
        
        # Set column widths
        self.clients_tree.column("client_id", width=50)
        self.clients_tree.column("name", width=150)
        self.clients_tree.column("email", width=200)
        self.clients_tree.column("risk_profile", width=100)
        
        # Add double click event binding for editing
        self.clients_tree.bind("<Double-1>", lambda event: self.edit_client())
        
        # Pack treeview
        self.clients_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Action buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Add Client", command=self.show_add_client_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Client", command=self.edit_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Client", command=self.delete_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_all_clients).pack(side=tk.LEFT, padx=5)
        
    def _create_managers_list(self, parent):
        """Create the list view for portfolio managers."""
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview with scrollbar for managers
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("manager_id", "name", "email")
        self.managers_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15, yscrollcommand=y_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.managers_tree.yview)
        
        # Set column headings
        self.managers_tree.heading("manager_id", text="ID")
        self.managers_tree.heading("name", text="Name")
        self.managers_tree.heading("email", text="Email")
        
        # Set column widths
        self.managers_tree.column("manager_id", width=40, anchor=tk.CENTER)
        self.managers_tree.column("name", width=120)
        self.managers_tree.column("email", width=150)
        
        # Add double click event binding for editing
        self.managers_tree.bind("<Double-1>", lambda event: self.edit_manager())
        
        # Pack treeview
        self.managers_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Action buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Add Manager", command=self.show_add_manager_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Manager", command=self.edit_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Manager", command=self.delete_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_all_managers).pack(side=tk.LEFT, padx=5)
    
    def create_portfolio_section(self, parent):
        """Create the section for portfolio creation."""
        # Create a notebook to contain tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create the Create Portfolio tab
        create_tab = ttk.Frame(notebook)
        notebook.add(create_tab, text="Create Portfolio")
        
        # Create the View Portfolios tab
        view_tab = ttk.Frame(notebook)
        notebook.add(view_tab, text="View Portfolios")
        
        # Create the form for portfolio creation
        self._create_new_portfolio_form(create_tab)
        
        # Create the portfolios list view
        self._create_portfolios_list(view_tab)
        
        # Load data for both forms
        self.load_clients()
        self.load_managers()
        self.load_portfolios()
        
        return notebook

    def _create_new_portfolio_form(self, parent):
        """Create the form for creating a new portfolio."""
        form_frame = ttk.LabelFrame(parent, text="New Portfolio")
        form_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create a simple 2-column grid layout
        ttk.Label(form_frame, text="Portfolio Name:").grid(row=0, column=0, sticky="w", padx=10, pady=8)
        self.portfolio_name_var = tk.StringVar()
        ttk.Entry(form_frame, textvariable=self.portfolio_name_var, width=30).grid(row=0, column=1, sticky="ew", padx=10, pady=8)
        
        # Client selection
        ttk.Label(form_frame, text="Client:").grid(row=1, column=0, sticky="w", padx=10, pady=8)
        self.client_combo = ttk.Combobox(form_frame, textvariable=self.client_var, state="readonly", width=30)
        self.client_combo.grid(row=1, column=1, sticky="ew", padx=10, pady=8)
        
        # Manager selection
        ttk.Label(form_frame, text="Manager:").grid(row=2, column=0, sticky="w", padx=10, pady=8) 
        self.manager_combo = ttk.Combobox(form_frame, textvariable=self.manager_var, state="readonly", width=30)
        self.manager_combo.grid(row=2, column=1, sticky="ew", padx=10, pady=8)
        
        # Strategy
        ttk.Label(form_frame, text="Strategy:").grid(row=3, column=0, sticky="w", padx=10, pady=8)
        strategies = ["Low Risk", "Low Turnover", "High Yield Equity"]
        self.strategy_combo = ttk.Combobox(form_frame, textvariable=self.strategy_var, values=strategies, state="readonly", width=30)
        self.strategy_combo.grid(row=3, column=1, sticky="ew", padx=10, pady=8)
        self.strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_change)
        
        # Asset Universe
        ttk.Label(form_frame, text="Asset Universe:").grid(row=4, column=0, sticky="w", padx=10, pady=8)
        self.all_universes = ["EU Equities", "US Equities", "Global Equities",  "Commodities", "Cryptocurrencies", "ETFs", "Mixed Assets"]
        self.universe_combo = ttk.Combobox(form_frame, textvariable=self.universe_var, values=self.all_universes, state="readonly", width=30)
        self.universe_combo.grid(row=4, column=1, sticky="ew", padx=10, pady=8)
        
        # Create button
        button_frame = ttk.Frame(form_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=15)
        
        create_button = ttk.Button(button_frame, text="Create Portfolio", command=self.create_portfolio)
        create_button.pack(pady=5)
        
        # Add some help text
        help_text = ttk.Label(form_frame, text="Fill in all fields above and click 'Create Portfolio' to create a new portfolio.\nUse the panels on the left to add new clients or managers if needed.",
                              foreground="gray", wraplength=400, justify="center")
        help_text.grid(row=6, column=0, columnspan=2, pady=10)

    def _create_portfolios_list(self, parent):
        """Create the list view for existing portfolios."""
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create treeview with scrollbar for portfolios
        tree_frame = ttk.Frame(list_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbars
        y_scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL)
        y_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create treeview
        columns = ("portfolio_id", "name", "client", "manager", "strategy", "universe")
        self.portfolios_tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=15, yscrollcommand=y_scrollbar.set)
        
        # Configure scrollbars
        y_scrollbar.config(command=self.portfolios_tree.yview)
        
        # Set column headings
        self.portfolios_tree.heading("portfolio_id", text="ID")
        self.portfolios_tree.heading("name", text="Name")
        self.portfolios_tree.heading("client", text="Client")
        self.portfolios_tree.heading("manager", text="Manager")
        self.portfolios_tree.heading("strategy", text="Strategy")
        self.portfolios_tree.heading("universe", text="Universe")
        
        # Set column widths
        self.portfolios_tree.column("portfolio_id", width=40, anchor=tk.CENTER)
        self.portfolios_tree.column("name", width=120)
        self.portfolios_tree.column("client", width=120)
        self.portfolios_tree.column("manager", width=120)
        self.portfolios_tree.column("strategy", width=120)
        self.portfolios_tree.column("universe", width=120)
        
        # Add double click event binding for editing
        self.portfolios_tree.bind("<Double-1>", lambda event: self.edit_portfolio())
        
        # Pack treeview
        self.portfolios_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Action buttons
        btn_frame = ttk.Frame(list_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Edit Portfolio", command=self.edit_portfolio).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Portfolio", command=self.delete_portfolio).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh List", command=self.refresh_portfolios).pack(side=tk.LEFT, padx=5)
    
    def load_clients(self):
        """Load clients from the database."""
        try:
            # Get clients from database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT client_id, name, email, risk_profile FROM Clients ORDER BY name")
            clients = cursor.fetchall()
            
            # Clear existing items
            for item in self.clients_tree.get_children():
                self.clients_tree.delete(item)
            
            # Insert clients into treeview
            for client in clients:
                self.clients_tree.insert("", tk.END, values=client)
            
            # Update client names list for portfolio creation
            self.client_ids = [client[0] for client in clients]
            self.client_names = [client[1] for client in clients]
            
            # Update client combobox in portfolio creation form
            if hasattr(self, 'client_combo'):
                self.client_combo['values'] = self.client_names
            
            self.status_var.set(f"Loaded {len(clients)} clients")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load clients: {str(e)}")
            self.status_var.set("Error loading clients")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def on_client_select(self, event):
        """Handle client selection event."""
        if not hasattr(self, 'client_names') or not self.client_names:
            return
            
        selected_idx = self.client_combo.current()
        if selected_idx >= 0:
            client_id = self.client_ids[selected_idx]
            
            # Get client risk profile
            try:
                conn = self.db_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT risk_profile FROM Clients WHERE client_id = ?", (client_id,))
                risk_profile = cursor.fetchone()[0]
                self.risk_var.set(risk_profile)
                
                # Update strategy to match risk profile if it's one of our strategy options
                # Only update strategy_var if it exists and is properly initialized
                if hasattr(self, 'strategy_var') and hasattr(self, 'strategy_mapping'):
                    if risk_profile in self.strategy_mapping:
                        self.strategy_var.set(risk_profile)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load client details: {str(e)}")
            finally:
                if self.db_manager.db_path != ":memory:":
                    conn.close()
    
    def load_managers(self):
        """Load portfolio managers from the database."""
        try:
            # Get managers from database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT manager_id, name, email FROM Managers ORDER BY name")
            managers = cursor.fetchall()
            
            # Clear existing items
            for item in self.managers_tree.get_children():
                self.managers_tree.delete(item)
            
            # Insert managers into treeview
            for manager in managers:
                self.managers_tree.insert("", tk.END, values=manager)
            
            # Update manager names list for portfolio creation
            self.manager_ids = [manager[0] for manager in managers]
            self.manager_names = [manager[1] for manager in managers]
            
            # Update manager combobox in portfolio creation form
            if hasattr(self, 'manager_combo'):
                self.manager_combo['values'] = self.manager_names
            
            self.status_var.set(f"Loaded {len(managers)} portfolio managers")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolio managers: {str(e)}")
            self.status_var.set("Error loading portfolio managers")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def load_portfolios(self):
        """Load existing portfolios from the database."""
        try:
            # Clear treeview
            for item in self.portfolios_tree.get_children():
                self.portfolios_tree.delete(item)
                
            # Get portfolios from database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # First check if strategy and asset_universe columns exist
            cursor.execute("PRAGMA table_info(Portfolios)")
            columns = [column[1] for column in cursor.fetchall()]
            has_strategy = 'strategy' in columns
            has_universe = 'asset_universe' in columns
            
            # Build the appropriate query based on column existence
            if has_strategy and has_universe:
                query = """
                    SELECT p.portfolio_id, p.name, p.strategy, p.asset_universe, 
                           c.name as client_name, m.name as manager_name
                    FROM Portfolios p
                    LEFT JOIN Clients c ON p.client_id = c.client_id
                    LEFT JOIN Managers m ON p.manager_id = m.manager_id
                    ORDER BY p.name
                """
            else:
                # Use metadata for missing columns
                query = """
                    SELECT p.portfolio_id, p.name, p.strategy, p.asset_universe, 
                           c.name as client_name, m.name as manager_name
                    FROM Portfolios p
                    LEFT JOIN Clients c ON p.client_id = c.client_id
                    LEFT JOIN Managers m ON p.manager_id = m.manager_id
                    ORDER BY p.name
                """
            
            cursor.execute(query)
            
            portfolios = cursor.fetchall()
            for portfolio in portfolios:
                self.portfolios_tree.insert("", "end", values=portfolio)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolios: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def show_add_client_dialog(self):
        """Show dialog to add a new client."""
        # Create dialog
        dialog = ClientDialog(self)
        self.wait_window(dialog)
        
        # Refresh clients list
        self.load_clients()
    
    def show_add_manager_dialog(self):
        """Show dialog to add a new portfolio manager."""
        # Create dialog
        dialog = ManagerDialog(self)
        self.wait_window(dialog)
        
        # Refresh managers list
        self.load_managers()
    
    def create_portfolio(self):
        """Create a new portfolio."""
        try:
            # Validate inputs
            portfolio_name = self.portfolio_name_var.get().strip()
            if not portfolio_name:
                messagebox.showerror("Error", "Please enter a portfolio name.")
                return
                
            # Get selected client
            selected_client_idx = self.client_combo.current()
            if selected_client_idx < 0:
                messagebox.showerror("Error", "Please select a client.")
                return
                
            client_id = self.client_ids[selected_client_idx]
            
            # Get selected manager
            selected_manager_idx = self.manager_combo.current()
            if selected_manager_idx < 0:
                messagebox.showerror("Error", "Please select a portfolio manager.")
                return
                
            manager_id = self.manager_ids[selected_manager_idx]
            
            # Get strategy and asset universe
            strategy = self.strategy_var.get()
            universe = self.universe_var.get()
            
            if not strategy or not universe:
                messagebox.showerror("Error", "Please select a strategy and asset universe.")
                return
                
            # Make sure the strategy is a valid one from our mapping
            if strategy not in self.strategy_mapping:
                messagebox.showerror("Error", f"Invalid strategy: {strategy}")
                return
            
            # Create the portfolio in the database using the database manager
            try:
                # Begin transaction
                conn = self.db_manager._get_connection()
                conn.execute("BEGIN TRANSACTION")
                
                # Create the portfolio with strategy and asset universe
                portfolio_id = self.db_manager.create_portfolio(
                    client_id=client_id,
                    manager_id=manager_id,
                    name=portfolio_name,
                    strategy=strategy,
                    asset_universe=universe
                )
                
                # Create initial portfolio metadata
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO PortfolioMetadata (portfolio_id, key, value)
                    VALUES (?, ?, ?)
                """, (portfolio_id, "initial_capital", "100000.0"))
                
                cursor.execute("""
                    INSERT INTO PortfolioMetadata (portfolio_id, key, value)
                    VALUES (?, ?, ?)
                """, (portfolio_id, "start_date", datetime.now().isoformat()))
                
                # Create initial performance record
                cursor.execute("""
                    INSERT INTO PerformanceMetrics (portfolio_id, date, total_value)
                    VALUES (?, ?, ?)
                """, (portfolio_id, datetime.now().isoformat(), 100000.0))
                
                # Commit transaction
                conn.commit()
                
                messagebox.showinfo("Success", f"Portfolio '{portfolio_name}' created successfully.")
                self.status_var.set(f"Portfolio '{portfolio_name}' created successfully.")
                
                # Refresh portfolios list
                self.load_portfolios()
                
            except Exception as e:
                # Rollback transaction if needed
                if 'conn' in locals():
                    conn.rollback()
                messagebox.showerror("Error", f"Failed to create portfolio: {str(e)}")
                self.status_var.set("Error creating portfolio.")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create portfolio: {str(e)}")
            self.status_var.set("Error creating portfolio.")
            
        finally:
            if 'conn' in locals() and self.db_manager.db_path != ":memory:":
                conn.close()

    def edit_portfolio(self):
        """Edit the selected portfolio."""
        selected_items = self.portfolios_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a portfolio to edit.")
            return
            
        # Get portfolio ID
        portfolio_id = self.portfolios_tree.item(selected_items[0], "values")[0]
        
        # Open edit dialog
        edit_dialog = EditPortfolioDialog(self, self.db_manager, portfolio_id)
        self.wait_window(edit_dialog)
        
        # Refresh portfolios list
        self.load_portfolios()
    
    def delete_portfolio(self):
        """Delete the selected portfolio."""
        selected_items = self.portfolios_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a portfolio to delete.")
            return
            
        # Get portfolio ID and name
        portfolio_values = self.portfolios_tree.item(selected_items[0], "values")
        portfolio_id = portfolio_values[0]
        portfolio_name = portfolio_values[1]
        
        # Confirm deletion
        if not messagebox.askyesno("Confirm Deletion", f"Are you sure you want to delete portfolio '{portfolio_name}'?"):
            return
            
        try:
            # Delete portfolio from database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # Check if Deals table exists before deleting from it
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Deals'")
            if cursor.fetchone():
                # Only delete from Deals if the table exists
                cursor.execute("DELETE FROM Deals WHERE portfolio_id = ?", (portfolio_id,))
            
            # Delete from other tables
            cursor.execute("DELETE FROM PerformanceMetrics WHERE portfolio_id = ?", (portfolio_id,))
            cursor.execute("DELETE FROM PortfolioHoldings WHERE portfolio_id = ?", (portfolio_id,))
            cursor.execute("DELETE FROM Portfolios WHERE portfolio_id = ?", (portfolio_id,))
            conn.commit()
            
            messagebox.showinfo("Success", f"Portfolio '{portfolio_name}' deleted successfully.")
            self.status_var.set(f"Portfolio '{portfolio_name}' deleted successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete portfolio: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
        # Refresh portfolios list
        self.load_portfolios()

    def delete_client(self):
        """Delete the currently selected client."""
        # Get selected item from treeview
        selected_items = self.clients_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a client to delete.")
            return
            
        # Get client data
        item = selected_items[0]
        values = self.clients_tree.item(item)['values']
        client_id = values[0]
        client_name = values[1]
        
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete client '{client_name}'?\n\nThis will also delete all portfolios associated with this client."
        )
        
        if not confirm:
            return
        
        try:
            # Check if client has portfolios
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM Portfolios WHERE client_id = ?", (client_id,))
            portfolio_count = cursor.fetchone()[0]
            
            if portfolio_count > 0:
                # Confirm deletion of portfolios
                confirm_portfolios = messagebox.askyesno(
                    "Portfolios Found",
                    f"Client '{client_name}' has {portfolio_count} portfolios that will also be deleted. Continue?",
                    icon="warning"
                )
                
                if not confirm_portfolios:
                    return
                
                # Delete related data
                try:
                    conn.execute("BEGIN TRANSACTION")
                    
                    # Delete portfolio metadata
                    cursor.execute(
                        "DELETE FROM PortfolioMetadata WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)",
                        (client_id,)
                    )
                    
                    # Delete performance metrics
                    cursor.execute(
                        "DELETE FROM PerformanceMetrics WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)",
                        (client_id,)
                    )
                    
                    # Delete positions
                    cursor.execute(
                        "DELETE FROM PortfolioHoldings WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)",
                        (client_id,)
                    )
                    
                    # Check if Deals table exists before deleting from it
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Deals'")
                    if cursor.fetchone():
                        # Only delete from Deals if the table exists
                        cursor.execute("DELETE FROM Deals WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)", (client_id,))
                    
                    # Delete portfolios
                    cursor.execute("DELETE FROM Portfolios WHERE client_id = ?", (client_id,))
                    
                    # Delete client
                    cursor.execute("DELETE FROM Clients WHERE client_id = ?", (client_id,))
                    
                    conn.commit()
                    
                    messagebox.showinfo("Success", f"Client '{client_name}' and all associated portfolios deleted successfully.")
                    self.status_var.set(f"Client '{client_name}' deleted successfully.")
                    
                    # Refresh client list
                    self.load_clients()
                    # Refresh portfolios list
                    self.load_portfolios()
                    
                except Exception as e:
                    conn.rollback()
                    messagebox.showerror("Error", f"Failed to delete client: {str(e)}")
                    self.status_var.set(f"Error deleting client: {str(e)}")
            else:
                # No portfolios, just delete the client
                cursor.execute("DELETE FROM Clients WHERE client_id = ?", (client_id,))
                conn.commit()
                
                messagebox.showinfo("Success", f"Client '{client_name}' deleted successfully.")
                self.status_var.set(f"Client '{client_name}' deleted successfully.")
                
                # Refresh client list
                self.load_clients()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

    def delete_manager(self):
        """Delete the currently selected portfolio manager."""
        # Get selected item from treeview
        selected_items = self.managers_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a portfolio manager to delete.")
            return
            
        # Get manager data
        item = selected_items[0]
        values = self.managers_tree.item(item)['values']
        manager_id = values[0]
        manager_name = values[1]
        
        # Confirm deletion
        confirm = messagebox.askyesno(
            "Confirm Deletion", 
            f"Are you sure you want to delete portfolio manager '{manager_name}'?"
        )
        
        if not confirm:
            return
        
        try:
            # Check if manager has portfolios
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM Portfolios WHERE manager_id = ?", (manager_id,))
            portfolio_count = cursor.fetchone()[0]
            
            if portfolio_count > 0:
                # Confirm reassignment of portfolios
                confirm_reassign = messagebox.askyesno(
                    "Portfolios Found",
                    f"Manager '{manager_name}' has {portfolio_count} portfolios assigned. Do you want to reassign these portfolios to another manager?",
                    icon="warning"
                )
                
                if confirm_reassign:
                    # Get other managers
                    other_managers = []
                    for item in self.managers_tree.get_children():
                        values = self.managers_tree.item(item)['values']
                        if values[0] != manager_id:
                            other_managers.append((values[0], values[1]))
                    
                    if not other_managers:
                        messagebox.showwarning("Warning", "There are no other managers to reassign portfolios to. Please create another manager first.")
                        return
                    
                    # Show dialog to select new manager
                    from tkinter import simpledialog
                    
                    # Create a simple dialog for manager selection
                    new_manager_dialog = tk.Toplevel(self)
                    new_manager_dialog.title("Select New Manager")
                    new_manager_dialog.transient(self)
                    new_manager_dialog.grab_set()
                    
                    ttk.Label(new_manager_dialog, text="Select new manager for portfolios:").pack(padx=10, pady=10)
                    
                    new_manager_var = tk.StringVar()
                    other_manager_names = [m[1] for m in other_managers]
                    new_manager_combo = ttk.Combobox(new_manager_dialog, textvariable=new_manager_var, values=other_manager_names, state="readonly", width=30)
                    new_manager_combo.pack(padx=10, pady=10)
                    
                    if other_manager_names:
                        new_manager_combo.current(0)
                    
                    def on_ok():
                        new_manager_dialog.result = new_manager_var.get()
                        new_manager_dialog.destroy()
                    
                    def on_cancel():
                        new_manager_dialog.result = None
                        new_manager_dialog.destroy()
                    
                    btn_frame = ttk.Frame(new_manager_dialog)
                    btn_frame.pack(padx=10, pady=10)
                    
                    ttk.Button(btn_frame, text="OK", command=on_ok).pack(side=tk.LEFT, padx=5)
                    ttk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side=tk.LEFT, padx=5)
                    
                    # Center dialog
                    new_manager_dialog.geometry("400x200")
                    new_manager_dialog.resizable(False, False)
                    
                    # Wait for dialog to close
                    self.wait_window(new_manager_dialog)
                    
                    if not hasattr(new_manager_dialog, 'result') or not new_manager_dialog.result:
                        return
                    
                    # Get new manager ID
                    new_manager_idx = other_manager_names.index(new_manager_dialog.result)
                    new_manager_id = other_managers[new_manager_idx][0]
                    
                    # Reassign portfolios
                    cursor.execute("UPDATE Portfolios SET manager_id = ? WHERE manager_id = ?", (new_manager_id, manager_id))
                    
                    # Delete the manager
                    cursor.execute("DELETE FROM Managers WHERE manager_id = ?", (manager_id,))
                    conn.commit()
                    
                    messagebox.showinfo("Success", f"Manager '{manager_name}' deleted and portfolios reassigned successfully.")
                    self.status_var.set(f"Manager '{manager_name}' deleted and portfolios reassigned to {new_manager_dialog.result}.")
                    
                else:
                    # User doesn't want to reassign, warn that portfolios will be deleted
                    confirm_delete_all = messagebox.askyesno(
                        "Warning",
                        f"Without reassignment, all {portfolio_count} portfolios managed by '{manager_name}' will be deleted. Continue?",
                        icon="warning"
                    )
                    
                    if not confirm_delete_all:
                        return
                    
                    try:
                        conn.execute("BEGIN TRANSACTION")
                        
                        # Delete portfolio related data
                        cursor.execute(
                            "DELETE FROM PortfolioMetadata WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE manager_id = ?)",
                            (manager_id,)
                        )
                        
                        cursor.execute(
                            "DELETE FROM PerformanceMetrics WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE manager_id = ?)",
                            (manager_id,)
                        )
                        
                        cursor.execute(
                            "DELETE FROM PortfolioHoldings WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE manager_id = ?)",
                            (manager_id,)
                        )
                        
                        # Check if Deals table exists before deleting from it
                        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='Deals'")
                        if cursor.fetchone():
                            # Only delete from Deals if the table exists
                            cursor.execute("DELETE FROM Deals WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE manager_id = ?)", (manager_id,))
                        
                        # Delete portfolios
                        cursor.execute("DELETE FROM Portfolios WHERE manager_id = ?", (manager_id,))
                        
                        # Delete manager
                        cursor.execute("DELETE FROM Managers WHERE manager_id = ?", (manager_id,))
                        
                        conn.commit()
                        
                        messagebox.showinfo("Success", f"Manager '{manager_name}' and all associated portfolios deleted successfully.")
                        self.status_var.set(f"Manager '{manager_name}' and portfolios deleted successfully.")
                        
                    except Exception as e:
                        conn.rollback()
                        messagebox.showerror("Error", f"Failed to delete manager and portfolios: {str(e)}")
                        self.status_var.set(f"Error deleting manager: {str(e)}")
                        return
            else:
                # No portfolios, just delete the manager
                cursor.execute("DELETE FROM Managers WHERE manager_id = ?", (manager_id,))
                conn.commit()
                
                messagebox.showinfo("Success", f"Manager '{manager_name}' deleted successfully.")
                self.status_var.set(f"Manager '{manager_name}' deleted successfully.")
            
            # Refresh manager list
            self.load_managers()
            # Refresh portfolios list
            self.load_portfolios()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

    def refresh_all_clients(self):
        """Refresh all client data and update all related UI elements."""
        self.load_clients()
        
        # Also refresh portfolios since they depend on clients
        self.load_portfolios()
        
        # Update status
        self.status_var.set("All client data refreshed")
        
    def refresh_all_managers(self):
        """Refresh all manager data and update all related UI elements."""
        self.load_managers()
        
        # Also refresh portfolios since they depend on managers
        self.load_portfolios()
        
        # Update status
        self.status_var.set("All manager data refreshed")

    def refresh_portfolios(self):
        """Refresh the portfolios list with the latest data."""
        self.load_portfolios()
        count = len(self.portfolios_tree.get_children())
        self.status_var.set(f"Portfolio list refreshed. {count} portfolios found.")

    def edit_client(self):
        """Edit the selected client."""
        # Get selected item
        selected_items = self.clients_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a client to edit.")
            return
            
        # Get client data
        item = selected_items[0]
        values = self.clients_tree.item(item)['values']
        client_id = values[0]
        
        # Create edit dialog
        dialog = EditClientDialog(self, self.db_manager, client_id)
        self.wait_window(dialog)
        
        # Refresh the list
        self.refresh_all_clients()
        
    def edit_manager(self):
        """Edit the selected portfolio manager."""
        # Get selected item
        selected_items = self.managers_tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select a portfolio manager to edit.")
            return
            
        # Get manager data
        item = selected_items[0]
        values = self.managers_tree.item(item)['values']
        manager_id = values[0]
        
        # Create edit dialog
        dialog = EditManagerDialog(self, self.db_manager, manager_id)
        self.wait_window(dialog)
        
        # Refresh the list
        self.refresh_all_managers()

    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        selected_strategy = self.strategy_var.get()
        
        # Filter universe options based on strategy
        if selected_strategy == "High Yield Equity":
            # For High Yield Equity, only allow universes with "Equities" in the name
            equity_universes = [u for u in self.all_universes if "Equities" in u]
            self.universe_combo['values'] = equity_universes
            
            # If current selection is not in the filtered list, select the first option
            if self.universe_var.get() not in equity_universes:
                if equity_universes:
                    self.universe_var.set(equity_universes[0])
                else:
                    self.universe_var.set("")
        else:
            # For other strategies, allow all universes
            self.universe_combo['values'] = self.all_universes
        
        # Set risk profile to match strategy
        if selected_strategy in self.strategy_mapping:
            self.risk_var.set(selected_strategy)

class EditClientDialog(tk.Toplevel):
    """Dialog for editing a client."""
    
    def __init__(self, parent, db_manager, client_id):
        """Initialize the dialog."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.client_id = client_id
        
        # Set dialog properties
        self.title("Edit Client")
        self.geometry("400x250")
        self.resizable(False, False)
        
        # Create widgets
        self.create_widgets()
        
        # Load client data
        self.load_client_data()
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Client details
        details_frame = ttk.LabelFrame(main_frame, text="Client Details")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Name
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Email
        ttk.Label(details_frame, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.email_var, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Risk Profile
        ttk.Label(details_frame, text="Strategy:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.risk_var = tk.StringVar()
        risk_profiles = ["Low Risk", "Low Turnover", "High Yield Equity"]
        ttk.Combobox(details_frame, textvariable=self.risk_var, values=risk_profiles, state="readonly", width=27).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_client_data(self):
        """Load the client's data into the form."""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, email, risk_profile FROM Clients WHERE client_id = ?", (self.client_id,))
            client = cursor.fetchone()
            
            if client:
                self.name_var.set(client[0])
                self.email_var.set(client[1])
                self.risk_var.set(client[2])
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load client data: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
    def save_changes(self):
        """Save the changes to the client."""
        try:
            # Validate inputs
            name = self.name_var.get().strip()
            email = self.email_var.get().strip()
            risk_profile = self.risk_var.get()
            
            if not all([name, email, risk_profile]):
                messagebox.showerror("Error", "All fields are required.")
                return
                
            # Update client in database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE Clients 
                SET name = ?, email = ?, risk_profile = ?
                WHERE client_id = ?
            """, (name, email, risk_profile, self.client_id))
            
            conn.commit()
            messagebox.showinfo("Success", "Client updated successfully.")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update client: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

class EditManagerDialog(tk.Toplevel):
    """Dialog for editing a portfolio manager."""
    
    def __init__(self, parent, db_manager, manager_id):
        """Initialize the dialog."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.manager_id = manager_id
        
        # Set dialog properties
        self.title("Edit Portfolio Manager")
        self.geometry("400x200")
        self.resizable(False, False)
        
        # Create widgets
        self.create_widgets()
        
        # Load manager data
        self.load_manager_data()
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Manager details
        details_frame = ttk.LabelFrame(main_frame, text="Manager Details")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Name
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Email
        ttk.Label(details_frame, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.email_var, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_manager_data(self):
        """Load the manager's data into the form."""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, email FROM Managers WHERE manager_id = ?", (self.manager_id,))
            manager = cursor.fetchone()
            
            if manager:
                name, email = manager
                self.name_var.set(name)
                self.email_var.set(email if email else "")
            
            conn.close()
        except Exception as e:
            messagebox.showerror("Error", f"Error loading manager data: {str(e)}")
        
    def save_changes(self):
        """Save the changes to the manager."""
        try:
            # Validate inputs
            name = self.name_var.get().strip()
            email = self.email_var.get().strip()
            
            if not all([name, email]):
                messagebox.showerror("Error", "All fields are required.")
                return
                
            # Update manager in database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE Managers
                SET name = ?, email = ?
                WHERE manager_id = ?
            """, (name, email, self.manager_id))
            conn.commit()
            conn.close()
            messagebox.showinfo("Success", "Manager updated successfully")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update manager: {str(e)}")

class ClientDialog(tk.Toplevel):
    """Dialog for adding a new client."""
    
    def __init__(self, parent):
        """Initialize the dialog."""
        super().__init__(parent)
        self.title("Add Client")
        self.parent = parent
        self.db_manager = parent.db_manager
        
        # Set as modal dialog
        self.transient(parent)
        self.grab_set()
        
        # Create dialog content
        self.create_widgets()
        
        # Center the dialog
        self.geometry("400x200")
        self.resizable(False, False)
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Fields
        ttk.Label(main_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(main_frame, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.email_var, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(main_frame, text="Risk Profile:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.risk_var = tk.StringVar()
        risk_profiles = ["Low Risk", "Low Turnover", "High Yield Equity"]
        ttk.Combobox(main_frame, textvariable=self.risk_var, values=risk_profiles, state="readonly", width=28).grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=3, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Save", command=self.save_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def save_client(self):
        """Save the client data."""
        # Validate inputs
        name = self.name_var.get().strip()
        email = self.email_var.get().strip()
        risk_profile = self.risk_var.get()
        
        if not name:
            messagebox.showwarning("Validation Error", "Name is required.")
            return
        
        if not risk_profile:
            messagebox.showwarning("Validation Error", "Risk profile is required.")
            return
        
        try:
            # Add client to database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # Check if risk_profile column exists
            cursor.execute("PRAGMA table_info(Clients)")
            columns = [column[1] for column in cursor.fetchall()]
            has_risk_profile = 'risk_profile' in columns
            
            # Insert client with appropriate columns
            if has_risk_profile:
                cursor.execute("""
                    INSERT INTO Clients (name, email, risk_profile)
                    VALUES (?, ?, ?)
                """, (name, email, risk_profile))
            else:
                cursor.execute("""
                    INSERT INTO Clients (name, email)
                    VALUES (?, ?)
                """, (name, email))
                
                # Get client ID
                client_id = cursor.lastrowid
                
                # Store risk profile in metadata
                cursor.execute("""
                    INSERT INTO ClientMetadata (client_id, key, value)
                    VALUES (?, ?, ?)
                """, (client_id, "risk_profile", risk_profile))
            
            conn.commit()
            
            # Close dialog
            self.destroy()
            
            messagebox.showinfo("Success", f"Client '{name}' added successfully.")
            self.parent.status_var.set(f"Client '{name}' added successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add client: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

class ManagerDialog(tk.Toplevel):
    """Dialog for adding a new portfolio manager."""
    
    def __init__(self, parent):
        """Initialize the dialog."""
        super().__init__(parent)
        self.title("Add Portfolio Manager")
        self.parent = parent
        self.db_manager = parent.db_manager
        
        # Set as modal dialog
        self.transient(parent)
        self.grab_set()
        
        # Create dialog content
        self.create_widgets()
        
        # Center the dialog
        self.geometry("400x200")
        self.resizable(False, False)
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Fields
        ttk.Label(main_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(main_frame, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.email_var, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        ttk.Button(btn_frame, text="Save", command=self.save_manager).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)
        
    def save_manager(self):
        """Save the manager data."""
        # Validate inputs
        name = self.name_var.get().strip()
        email = self.email_var.get().strip()
        
        if not name:
            messagebox.showwarning("Validation Error", "Name is required.")
            return
            
        if not email:
            messagebox.showwarning("Validation Error", "Email is required.")
            return
        
        try:
            # Add manager to database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            # Insert manager
            cursor.execute("""
                INSERT INTO Managers (name, email)
                VALUES (?, ?)
            """, (name, email))
            
            conn.commit()
            
            # Close dialog
            self.destroy()
            
            messagebox.showinfo("Success", f"Portfolio manager '{name}' added successfully.")
            self.parent.status_var.set(f"Portfolio manager '{name}' added successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add manager: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

class EditPortfolioDialog(tk.Toplevel):
    """Dialog for editing a portfolio."""
    
    def __init__(self, parent, db_manager, portfolio_id):
        """Initialize the dialog."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.portfolio_id = portfolio_id
        
        # Set dialog properties
        self.title("Edit Portfolio")
        self.geometry("400x250")
        self.resizable(False, False)
        
        # Strategy mapping
        self.strategy_mapping = {
            "Low Risk": "Low Risk",
            "Low Turnover": "Low Turnover",
            "High Yield Equity": "High Yield Equity"
        }
        
        # Create widgets
        self.create_widgets()
        
        # Load portfolio data
        self.load_portfolio_data()
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Portfolio details
        details_frame = ttk.LabelFrame(main_frame, text="Portfolio Details")
        details_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Name
        ttk.Label(details_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(details_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Client selection
        ttk.Label(details_frame, text="Client:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.client_combo = ttk.Combobox(details_frame, state="readonly", width=30)
        self.client_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Manager selection
        ttk.Label(details_frame, text="Manager:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.manager_combo = ttk.Combobox(details_frame, state="readonly", width=30)
        self.manager_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Strategy
        ttk.Label(details_frame, text="Strategy:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.strategy_var = tk.StringVar()
        strategies = ["Low Risk", "Low Turnover", "High Yield Equity"]
        self.strategy_combo = ttk.Combobox(details_frame, textvariable=self.strategy_var, values=strategies, state="readonly", width=30)
        self.strategy_combo.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        self.strategy_combo.bind("<<ComboboxSelected>>", self.on_strategy_change)
        
        # Asset Universe
        ttk.Label(details_frame, text="Asset Universe:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.universe_var = tk.StringVar()
        self.all_universes = ["US Equities", "Global Equities", "EU Equities", "Cryptocurrencies", "Commodities", "ETFs", "Mixed Assets"]
        self.universe_combo = ttk.Combobox(details_frame, textvariable=self.universe_var, values=self.all_universes, state="readonly", width=30)
        self.universe_combo.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Save", command=self.save_changes).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
        
    def load_portfolio_data(self):
        """Load the portfolio's data into the form."""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT name, client_id, manager_id, strategy, asset_universe FROM Portfolios WHERE portfolio_id = ?", (self.portfolio_id,))
            portfolio = cursor.fetchone()
            
            if portfolio:
                self.name_var.set(portfolio[0])
                self.client_combo['values'] = [self.db_manager.get_client_name(client_id) for client_id in self.db_manager.get_client_ids()]
                self.client_combo.current(self.db_manager.get_client_ids().index(portfolio[1]))
                self.manager_combo['values'] = [self.db_manager.get_manager_name(manager_id) for manager_id in self.db_manager.get_manager_ids()]
                self.manager_combo.current(self.db_manager.get_manager_ids().index(portfolio[2]))
                self.strategy_var.set(portfolio[3])
                self.universe_var.set(portfolio[4])
                
                # If High Yield Equity strategy is selected, filter universe options
                if portfolio[3] == "High Yield Equity":
                    self.on_strategy_change(None)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolio data: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
    def save_changes(self):
        """Save the changes to the portfolio."""
        try:
            # Validate inputs
            name = self.name_var.get().strip()
            client_index = self.client_combo.current()
            manager_index = self.manager_combo.current()
            strategy = self.strategy_var.get().strip()
            universe = self.universe_var.get().strip()
            
            if client_index < 0 or manager_index < 0 or not all([name, strategy, universe]):
                messagebox.showerror("Error", "All fields are required.")
                return
            
            # Get the actual client_id and manager_id from the database
            client_ids = self.db_manager.get_client_ids()
            manager_ids = self.db_manager.get_manager_ids()
            
            if client_index >= len(client_ids) or manager_index >= len(manager_ids):
                messagebox.showerror("Error", "Invalid client or manager selection.")
                return
                
            client_id = client_ids[client_index]
            manager_id = manager_ids[manager_index]
                
            # Update portfolio in database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE Portfolios 
                SET name = ?, client_id = ?, manager_id = ?, strategy = ?, asset_universe = ?
                WHERE portfolio_id = ?
            """, (name, client_id, manager_id, strategy, universe, self.portfolio_id))
            
            conn.commit()
            messagebox.showinfo("Success", "Portfolio updated successfully.")
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update portfolio: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
    def on_strategy_change(self, event):
        """Handle strategy selection change."""
        selected_strategy = self.strategy_var.get()
        
        # Filter universe options based on strategy
        if selected_strategy == "High Yield Equity":
            # For High Yield Equity, only allow universes with "Equities" in the name
            equity_universes = [u for u in self.all_universes if "Equities" in u]
            self.universe_combo['values'] = equity_universes
            
            # If current selection is not in the filtered list, select the first option
            if self.universe_var.get() not in equity_universes:
                if equity_universes:
                    self.universe_var.set(equity_universes[0])
                else:
                    self.universe_var.set("")
        else:
            # For other strategies, allow all universes
            self.universe_combo['values'] = self.all_universes