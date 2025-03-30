"""
Client Management Component for Portfolio Manager GUI
"""

import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd

class ClientManagementFrame(ttk.Frame):
    """Frame for managing clients."""
    
    def __init__(self, parent, db_manager, status_var):
        """Initialize the client management frame."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.status_var = status_var
        
        # Client list on left, details on right
        self.create_widgets()
        self.load_clients()
        
    def create_widgets(self):
        """Create the frame widgets."""
        # Split into left and right frames
        self.left_frame = ttk.Frame(self, width=300)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=5, pady=5)
        
        self.right_frame = ttk.Frame(self)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left frame: Client list and action buttons
        self.create_client_list()
        self.create_action_buttons()
        
        # Right frame: Client details
        self.create_client_details()
        
    def create_client_list(self):
        """Create the client list view."""
        # Label
        ttk.Label(self.left_frame, text="Clients", font=("Arial", 12, "bold")).pack(pady=5)
        
        # Create Treeview for client list
        self.client_tree = ttk.Treeview(self.left_frame, columns=("ID", "Name", "Risk Profile"), show="headings", height=20)
        self.client_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(self.left_frame, orient="vertical", command=self.client_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.client_tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure columns
        self.client_tree.heading("ID", text="ID")
        self.client_tree.heading("Name", text="Name")
        self.client_tree.heading("Risk Profile", text="Risk Profile")
        
        self.client_tree.column("ID", width=50)
        self.client_tree.column("Name", width=150)
        self.client_tree.column("Risk Profile", width=100)
        
        # Bind selection event
        self.client_tree.bind("<<TreeviewSelect>>", self.on_client_select)
        
    def create_action_buttons(self):
        """Create action buttons for client management."""
        btn_frame = ttk.Frame(self.left_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Add Client", command=self.add_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Edit Client", command=self.edit_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Delete Client", command=self.delete_client).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Refresh", command=self.load_clients).pack(side=tk.LEFT, padx=5)
        
    def create_client_details(self):
        """Create client details view."""
        # Client Details Frame
        detail_frame = ttk.LabelFrame(self.right_frame, text="Client Details")
        detail_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Client information fields
        info_frame = ttk.Frame(detail_frame)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Define fields
        fields = [
            ("ID:", "id_var"),
            ("Name:", "name_var"),
            ("Email:", "email_var"),
            ("Risk Profile:", "risk_var"),
            ("Created:", "created_var")
        ]
        
        # Create fields
        self.client_vars = {}
        row = 0
        for label_text, var_name in fields:
            ttk.Label(info_frame, text=label_text, font=("Arial", 10, "bold")).grid(row=row, column=0, sticky="w", padx=5, pady=5)
            self.client_vars[var_name] = tk.StringVar()
            ttk.Label(info_frame, textvariable=self.client_vars[var_name]).grid(row=row, column=1, sticky="w", padx=5, pady=5)
            row += 1
        
        # Portfolios Frame
        port_frame = ttk.LabelFrame(self.right_frame, text="Client Portfolios")
        port_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Portfolios Table
        self.portfolio_tree = ttk.Treeview(port_frame, columns=("ID", "Name", "Manager", "Value"), show="headings", height=8)
        self.portfolio_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(port_frame, orient="vertical", command=self.portfolio_tree.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.portfolio_tree.configure(yscrollcommand=scrollbar.set)
        
        # Configure columns
        self.portfolio_tree.heading("ID", text="ID")
        self.portfolio_tree.heading("Name", text="Name")
        self.portfolio_tree.heading("Manager", text="Manager")
        self.portfolio_tree.heading("Value", text="Value")
        
        self.portfolio_tree.column("ID", width=50)
        self.portfolio_tree.column("Name", width=150)
        self.portfolio_tree.column("Manager", width=150)
        self.portfolio_tree.column("Value", width=100)
        
    def load_clients(self):
        """Load clients from the database."""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT client_id, name, email, risk_profile, created_at
                FROM Clients
                ORDER BY name
            """)
            clients = cursor.fetchall()
            
            # Clear existing items
            for item in self.client_tree.get_children():
                self.client_tree.delete(item)
            
            # Insert clients into treeview
            for client in clients:
                self.client_tree.insert("", tk.END, values=client)
            
            self.status_var.set(f"Loaded {len(clients)} clients")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load clients: {str(e)}")
            self.status_var.set("Error loading clients")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def on_client_select(self, event):
        """Handle client selection event."""
        selected_items = self.client_tree.selection()
        if not selected_items:
            return
            
        # Get selected client ID
        client_id = self.client_tree.item(selected_items[0])['values'][0]
        
        try:
            # Get client details
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT client_id, name, email, risk_profile, created_at 
                FROM Clients WHERE client_id = ?
            """, (client_id,))
            client = cursor.fetchone()
            
            if client:
                # Update detail view
                self.client_vars["id_var"].set(client[0])
                self.client_vars["name_var"].set(client[1])
                self.client_vars["email_var"].set(client[2])
                self.client_vars["risk_var"].set(client[3])
                self.client_vars["created_var"].set(client[4])
                
                # Load portfolios
                self.load_client_portfolios(client_id)
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load client details: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def load_client_portfolios(self, client_id):
        """Load portfolios for the selected client."""
        # Clear existing items
        for item in self.portfolio_tree.get_children():
            self.portfolio_tree.delete(item)
        
        try:
            # Execute query to get portfolios
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.portfolio_id, p.name, m.name, COALESCE(pm.total_value, 0) 
                FROM Portfolios p 
                LEFT JOIN PortfolioManagers m ON p.manager_id = m.manager_id 
                LEFT JOIN (
                    SELECT portfolio_id, MAX(total_value) as total_value 
                    FROM PerformanceMetrics 
                    GROUP BY portfolio_id
                ) pm ON p.portfolio_id = pm.portfolio_id 
                WHERE p.client_id = ?
            """, (client_id,))
            portfolios = cursor.fetchall()
            
            # Add portfolios to tree
            for portfolio in portfolios:
                self.portfolio_tree.insert("", "end", values=portfolio)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load portfolios: {str(e)}")
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()
    
    def add_client(self):
        """Add a new client."""
        try:
            # Get input values
            name = self.client_vars["name_var"].get().strip()
            email = self.client_vars["email_var"].get().strip()
            risk_profile = self.client_vars["risk_var"].get()
            
            # Validate inputs
            if not name:
                messagebox.showerror("Error", "Name is required.")
                return
            if not email:
                messagebox.showerror("Error", "Email is required.")
                return
            if not risk_profile:
                messagebox.showerror("Error", "Risk profile is required.")
                return
            
            # Add client to database
            client_id = self.db_manager.add_client(name, risk_profile, email)
            
            # Clear input fields
            for var_name in self.client_vars:
                self.client_vars[var_name].set("")
            
            # Refresh client list
            self.load_clients()
            
            messagebox.showinfo("Success", f"Client '{name}' added successfully.")
            self.status_var.set(f"Client '{name}' added successfully.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add client: {str(e)}")
            self.status_var.set("Error adding client")
    
    def edit_client(self):
        """Edit selected client."""
        selected_items = self.client_tree.selection()
        if not selected_items:
            messagebox.showinfo("Selection Required", "Please select a client to edit.")
            return
            
        # Get selected client ID
        client_id = self.client_tree.item(selected_items[0])['values'][0]
        
        # Create dialog with client data
        dialog = ClientDialog(self, "Edit Client", self.db_manager, client_id)
        self.wait_window(dialog)
        
        # Refresh client list and details
        self.load_clients()
        self.on_client_select(None)  # Refresh details
    
    def delete_client(self):
        """Delete selected client."""
        selected_items = self.client_tree.selection()
        if not selected_items:
            messagebox.showinfo("Selection Required", "Please select a client to delete.")
            return
            
        # Get selected client ID
        client_id = self.client_tree.item(selected_items[0])['values'][0]
        client_name = self.client_tree.item(selected_items[0])['values'][1]
        
        try:
            # First check if client has portfolios
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM Portfolios WHERE client_id = ?", (client_id,))
            portfolio_count = cursor.fetchone()[0]
            
            if portfolio_count > 0:
                # Client has portfolios - ask what to do
                response = messagebox.askyesnocancel(
                    "Client Has Portfolios",
                    f"Client '{client_name}' has {portfolio_count} portfolios associated with them.\n\n"
                    "• Click 'Yes' to delete the client AND all their portfolios\n"
                    "• Click 'No' to keep the client\n"
                    "• Click 'Cancel' to view their portfolios first",
                    icon="warning"
                )
                
                if response is None:  # User clicked Cancel - show portfolios
                    self.load_client_portfolios(client_id)
                    return
                elif response is False:  # User clicked No - keep client
                    return
                # User clicked Yes - proceed with deletion (cascade)
            else:
                # No portfolios - simple confirmation
                confirm = messagebox.askyesno(
                    "Confirm Deletion", 
                    f"Are you sure you want to delete client '{client_name}'?"
                )
                if not confirm:
                    return
                    
            # Begin transaction for deletion
            conn.execute("BEGIN TRANSACTION")
            
            try:
                # Delete all related records first
                # Check if tables exist before deleting from them
                tables_to_check = [
                    'PortfolioHoldings',
                    'PortfolioMetadata',
                    'PerformanceMetrics',
                    'Deals',
                    'PortfolioRisk'
                ]
                
                existing_tables = {}
                for table in tables_to_check:
                    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
                    existing_tables[table] = cursor.fetchone() is not None
                
                # Delete portfolio holdings if table exists
                if existing_tables['PortfolioHoldings']:
                    cursor.execute("""
                        DELETE FROM PortfolioHoldings 
                        WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)
                    """, (client_id,))
                
                # Delete portfolio metadata if table exists
                if existing_tables['PortfolioMetadata']:
                    cursor.execute("""
                        DELETE FROM PortfolioMetadata 
                        WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)
                    """, (client_id,))
                
                # Delete performance metrics if table exists
                if existing_tables['PerformanceMetrics']:
                    cursor.execute("""
                        DELETE FROM PerformanceMetrics 
                        WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)
                    """, (client_id,))
                
                # Delete deals/trades if table exists
                if existing_tables['Deals']:
                    cursor.execute("""
                        DELETE FROM Deals 
                        WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)
                    """, (client_id,))
                
                # Delete portfolio risk data if table exists
                if existing_tables['PortfolioRisk']:
                    cursor.execute("""
                        DELETE FROM PortfolioRisk 
                        WHERE portfolio_id IN (SELECT portfolio_id FROM Portfolios WHERE client_id = ?)
                    """, (client_id,))
                
                # Delete portfolios
                cursor.execute("DELETE FROM Portfolios WHERE client_id = ?", (client_id,))
                
                # Finally delete the client
                cursor.execute("DELETE FROM Clients WHERE client_id = ?", (client_id,))
                
                # Commit the transaction
                conn.commit()
                
                self.status_var.set(f"Deleted client '{client_name}' and all associated data")
                
                # Refresh client list
                self.load_clients()
                
                # Clear detail view
                for var_name in self.client_vars:
                    self.client_vars[var_name].set("")
                    
                # Clear portfolios view
                for item in self.portfolio_tree.get_children():
                    self.portfolio_tree.delete(item)
                
            except Exception as e:
                # Rollback transaction on error
                conn.rollback()
                messagebox.showerror("Error", f"Failed to delete client: {str(e)}")
                self.status_var.set("Error deleting client")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to check client portfolios: {str(e)}")
            self.status_var.set("Error checking client portfolios")
            
        finally:
            if self.db_manager.db_path != ":memory:":
                conn.close()

class ClientDialog(tk.Toplevel):
    """Dialog for adding or editing a client."""
    
    def __init__(self, parent, title, db_manager, client_id=None):
        """Initialize the dialog."""
        super().__init__(parent)
        self.title(title)
        self.db_manager = db_manager
        self.client_id = client_id
        self.result = None
        
        # Set as modal dialog
        self.transient(parent)
        self.grab_set()
        
        # Create dialog content
        self.create_widgets()
        
        # Load client data if editing
        if self.client_id:
            self.load_client_data()
        
        # Center the dialog
        self.geometry("400x400")
        self.resizable(False, False)
        
    def create_widgets(self):
        """Create the dialog widgets."""
        # Main frame
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Fields
        fields_frame = ttk.Frame(main_frame)
        fields_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Name field
        ttk.Label(fields_frame, text="Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.name_var, width=30).grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Email field
        ttk.Label(fields_frame, text="Email:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.email_var = tk.StringVar()
        ttk.Entry(fields_frame, textvariable=self.email_var, width=30).grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # Risk Profile field
        ttk.Label(fields_frame, text="Risk Profile:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.risk_var = tk.StringVar()
        risk_options = ["Low Risk", "Low Turnover", "High Yield Equity Only"]
        risk_dropdown = ttk.Combobox(fields_frame, textvariable=self.risk_var, values=risk_options, state="readonly", width=28)
        risk_dropdown.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        if not self.client_id:  # Default for new clients
            risk_dropdown.current(0)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="Save", command=self.save_client).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=self.destroy).pack(side=tk.RIGHT, padx=5)
    
    def load_client_data(self):
        """Load client data for editing."""
        try:
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT name, email, risk_profile 
                FROM Clients WHERE client_id = ?
            """, (self.client_id,))
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
    
    def save_client(self):
        """Save the client data."""
        # Validate inputs
        name = self.name_var.get().strip()
        email = self.email_var.get().strip()
        risk_profile = self.risk_var.get()
        
        if not name:
            messagebox.showwarning("Validation Error", "Name is required.")
            return
        
        if not email:
            messagebox.showwarning("Validation Error", "Email is required.")
            return
        
        if not risk_profile:
            messagebox.showwarning("Validation Error", "Risk Profile is required.")
            return
        
        try:
            # Save client data
            if self.client_id:
                # Update existing client
                conn = self.db_manager._get_connection()
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE Clients 
                    SET name = ?, email = ?, risk_profile = ? 
                    WHERE client_id = ?
                """, (name, email, risk_profile, self.client_id))
                conn.commit()
                
                if self.db_manager.db_path != ":memory:":
                    conn.close()
                    
                self.result = "updated"
                
            else:
                # Add new client
                client_id = self.db_manager.add_client(name, risk_profile, email)
                self.result = "added"
            
            # Close dialog
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save client: {str(e)}") 