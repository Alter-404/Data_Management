"""
Portfolio comparison frame for visualizing and comparing multiple portfolios.
"""

import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
import threading
import logging

# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

# Import core modules
from src.visualization.portfolio_visualizer import PortfolioVisualizer

class PortfolioComparisonFrame(ttk.Frame):
    """Frame for comparing multiple portfolios performance metrics."""
    
    def __init__(self, parent, db_manager, visualizer, status_var):
        """Initialize the portfolio comparison frame."""
        super().__init__(parent)
        self.db_manager = db_manager
        self.visualizer = visualizer
        self.status_var = status_var
        self.logger = logging.getLogger(__name__)
        
        # Store selected portfolios
        self.selected_portfolios = []
        self.portfolio_ids = {}
        self.portfolio_names = []
        
        # Default date range
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime.now()
        
        # Initialize UI
        self._init_ui()
        
    def _init_ui(self):
        """Initialize the user interface."""
        # Create control panel
        self.control_panel = ttk.LabelFrame(self, text="Portfolio Comparison Controls")
        self.control_panel.pack(fill=tk.X, padx=10, pady=5)
        
        # Create portfolio selector
        self._create_portfolio_selector()
        
        # Create date range selector
        self._create_date_selector()
        
        # Create metrics selector
        self._create_metrics_selector()
        
        # Create visualization panel
        self.viz_panel = ttk.LabelFrame(self, text="Portfolio Comparison")
        self.viz_panel.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create notebook for visualization tabs
        self.notebook = ttk.Notebook(self.viz_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Create visualization tabs
        self._create_tabs()
        
        # Load portfolios on startup
        self._load_portfolios()
        
    def _create_portfolio_selector(self):
        """Create the portfolio selector panel with multi-select capability."""
        selector_frame = ttk.Frame(self.control_panel)
        selector_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create a frame for portfolio selection
        portfolio_frame = ttk.LabelFrame(selector_frame, text="Select Portfolios to Compare")
        portfolio_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Available portfolios list
        available_frame = ttk.Frame(portfolio_frame)
        available_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(available_frame, text="Available Portfolios:").pack(anchor=tk.W)
        
        # Scrollable listbox for available portfolios
        scrollbar1 = ttk.Scrollbar(available_frame)
        scrollbar1.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.available_list = tk.Listbox(available_frame, selectmode=tk.EXTENDED, 
                                        yscrollcommand=scrollbar1.set, height=5, width=30)
        self.available_list.pack(fill=tk.BOTH, expand=True)
        scrollbar1.config(command=self.available_list.yview)
        
        # Button frame for adding/removing portfolios
        btn_frame = ttk.Frame(portfolio_frame)
        btn_frame.pack(side=tk.LEFT, padx=10, pady=5)
        
        ttk.Button(btn_frame, text="Add >", command=self._add_portfolio).pack(pady=5)
        ttk.Button(btn_frame, text="< Remove", command=self._remove_portfolio).pack(pady=5)
        ttk.Button(btn_frame, text="Refresh", command=self._load_portfolios).pack(pady=5)
        
        # Selected portfolios list
        selected_frame = ttk.Frame(portfolio_frame)
        selected_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(selected_frame, text="Selected Portfolios:").pack(anchor=tk.W)
        
        # Scrollable listbox for selected portfolios
        scrollbar2 = ttk.Scrollbar(selected_frame)
        scrollbar2.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.selected_list = tk.Listbox(selected_frame, selectmode=tk.EXTENDED, 
                                       yscrollcommand=scrollbar2.set, height=5, width=30)
        self.selected_list.pack(fill=tk.BOTH, expand=True)
        scrollbar2.config(command=self.selected_list.yview)
        
        # Compare button
        ttk.Button(selector_frame, text="Compare Portfolios", 
                  command=self._compare_portfolios).pack(pady=5)
        
    def _create_date_selector(self):
        """Create date range selector."""
        date_frame = ttk.Frame(self.control_panel)
        date_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Date range label
        ttk.Label(date_frame, text="Date Range:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        # Start date entry
        ttk.Label(date_frame, text="From:").grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        # Start date components
        start_date_frame = ttk.Frame(date_frame)
        start_date_frame.grid(row=0, column=2, padx=5, pady=5, sticky="w")
        
        self.start_year_var = tk.StringVar(value=str(self.start_date.year))
        self.start_month_var = tk.StringVar(value=str(self.start_date.month).zfill(2))
        self.start_day_var = tk.StringVar(value=str(self.start_date.day).zfill(2))
        
        ttk.Entry(start_date_frame, textvariable=self.start_year_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(start_date_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(start_date_frame, textvariable=self.start_month_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(start_date_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(start_date_frame, textvariable=self.start_day_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # End date entry
        ttk.Label(date_frame, text="To:").grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        # End date components
        end_date_frame = ttk.Frame(date_frame)
        end_date_frame.grid(row=0, column=4, padx=5, pady=5, sticky="w")
        
        self.end_year_var = tk.StringVar(value=str(self.end_date.year))
        self.end_month_var = tk.StringVar(value=str(self.end_date.month).zfill(2))
        self.end_day_var = tk.StringVar(value=str(self.end_date.day).zfill(2))
        
        ttk.Entry(end_date_frame, textvariable=self.end_year_var, width=6).pack(side=tk.LEFT, padx=2)
        ttk.Label(end_date_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(end_date_frame, textvariable=self.end_month_var, width=4).pack(side=tk.LEFT, padx=2)
        ttk.Label(end_date_frame, text="-").pack(side=tk.LEFT)
        ttk.Entry(end_date_frame, textvariable=self.end_day_var, width=4).pack(side=tk.LEFT, padx=2)
        
        # Quick date buttons
        quick_date_frame = ttk.Frame(date_frame)
        quick_date_frame.grid(row=0, column=5, padx=10, pady=5, sticky="w")
        
        ttk.Button(quick_date_frame, text="Evaluation Period", 
                  command=lambda: self._set_date_range(datetime(2023, 1, 1), datetime.now())).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_date_frame, text="Training Period", 
                  command=lambda: self._set_date_range(datetime(2015, 1, 1), datetime(2022, 12, 31))).pack(side=tk.LEFT, padx=2)
        ttk.Button(quick_date_frame, text="All Data", 
                  command=lambda: self._set_date_range(datetime(2015, 1, 1), datetime.now())).pack(side=tk.LEFT, padx=2)
    
    def _create_metrics_selector(self):
        """Create metrics selector for comparison."""
        metrics_frame = ttk.LabelFrame(self.control_panel, text="Metrics to Compare")
        metrics_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Create checkboxes for metrics
        self.metric_vars = {
            "total_value": tk.BooleanVar(value=True),
            "cumulative_return": tk.BooleanVar(value=True),
            "volatility": tk.BooleanVar(value=True),
            "sharpe_ratio": tk.BooleanVar(value=True),
            "max_drawdown": tk.BooleanVar(value=True)
        }
        
        # Create grid of checkboxes
        ttk.Checkbutton(metrics_frame, text="Portfolio Value", 
                       variable=self.metric_vars["total_value"]).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(metrics_frame, text="Cumulative Return", 
                       variable=self.metric_vars["cumulative_return"]).grid(row=0, column=1, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(metrics_frame, text="Volatility", 
                       variable=self.metric_vars["volatility"]).grid(row=0, column=2, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(metrics_frame, text="Sharpe Ratio", 
                       variable=self.metric_vars["sharpe_ratio"]).grid(row=0, column=3, padx=5, pady=2, sticky="w")
        ttk.Checkbutton(metrics_frame, text="Max Drawdown", 
                       variable=self.metric_vars["max_drawdown"]).grid(row=0, column=4, padx=5, pady=2, sticky="w")
        
    def _create_tabs(self):
        """Create visualization tabs."""
        # Time Series tab (for performance over time)
        self.time_series_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.time_series_tab, text="Time Series")
        
        # Create Matplotlib figure for time series comparison
        self.fig_time_series = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_time_series = self.fig_time_series.add_subplot(111)
        self.canvas_time_series = FigureCanvasTkAgg(self.fig_time_series, self.time_series_tab)
        self.canvas_time_series.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bar Chart tab (for comparing metrics at a point in time)
        self.bar_chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.bar_chart_tab, text="Bar Chart")
        
        # Create Matplotlib figure for bar charts
        self.fig_bar = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_bar = self.fig_bar.add_subplot(111)
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, self.bar_chart_tab)
        self.canvas_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Risk-Return tab (scatter plot)
        self.risk_return_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.risk_return_tab, text="Risk-Return")
        
        # Create Matplotlib figure for risk-return analysis
        self.fig_risk_return = plt.Figure(figsize=(10, 6), dpi=100)
        self.ax_risk_return = self.fig_risk_return.add_subplot(111)
        self.canvas_risk_return = FigureCanvasTkAgg(self.fig_risk_return, self.risk_return_tab)
        self.canvas_risk_return.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Summary Table tab
        self.summary_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_tab, text="Summary Table")
        
        # Create treeview for summary metrics
        self.summary_tree_frame = ttk.Frame(self.summary_tab)
        self.summary_tree_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Scrollbar for tree
        scrollbar = ttk.Scrollbar(self.summary_tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create columns for the treeview
        columns = ("portfolio", "total_return", "volatility", "sharpe_ratio", "max_drawdown", "winning_days")
        self.summary_tree = ttk.Treeview(self.summary_tree_frame, columns=columns, 
                                        show="headings", yscrollcommand=scrollbar.set)
        
        # Configure columns
        self.summary_tree.heading("portfolio", text="Portfolio")
        self.summary_tree.heading("total_return", text="Total Return")
        self.summary_tree.heading("volatility", text="Volatility")
        self.summary_tree.heading("sharpe_ratio", text="Sharpe Ratio")
        self.summary_tree.heading("max_drawdown", text="Max Drawdown")
        self.summary_tree.heading("winning_days", text="Winning Days %")
        
        # Set column widths
        self.summary_tree.column("portfolio", width=150)
        self.summary_tree.column("total_return", width=100)
        self.summary_tree.column("volatility", width=100)
        self.summary_tree.column("sharpe_ratio", width=100)
        self.summary_tree.column("max_drawdown", width=100)
        self.summary_tree.column("winning_days", width=100)
        
        self.summary_tree.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.summary_tree.yview)
    
    def _parse_date(self, year_var, month_var, day_var):
        """Parse date from string variables."""
        try:
            year = int(year_var.get())
            month = int(month_var.get())
            day = int(day_var.get())
            return datetime(year, month, day)
        except (ValueError, TypeError):
            messagebox.showerror("Invalid Date", "Please enter valid date values.")
            return None
    
    def _set_date_range(self, start_date, end_date):
        """Set the date range and update the UI."""
        self.start_date = start_date
        self.end_date = end_date
        
        # Update entry fields
        self.start_year_var.set(str(start_date.year))
        self.start_month_var.set(str(start_date.month).zfill(2))
        self.start_day_var.set(str(start_date.day).zfill(2))
        
        self.end_year_var.set(str(end_date.year))
        self.end_month_var.set(str(end_date.month).zfill(2))
        self.end_day_var.set(str(end_date.day).zfill(2))
    
    def _load_portfolios(self):
        """Load available portfolios from the database."""
        try:
            # Clear existing items
            self.available_list.delete(0, tk.END)
            self.portfolio_ids = {}
            self.portfolio_names = []
            
            # Get portfolios from database
            conn = self.db_manager._get_connection()
            cursor = conn.cursor()
            
            query = """
                SELECT p.portfolio_id, p.name, c.name
                FROM Portfolios p
                JOIN Clients c ON p.client_id = c.client_id
                ORDER BY p.name
            """
            
            cursor.execute(query)
            portfolios = cursor.fetchall()
            
            # Add to portfolio combobox
            for portfolio_id, portfolio_name, client_name in portfolios:
                display_name = f"{portfolio_name} ({client_name})"
                self.portfolio_ids[display_name] = portfolio_id
                self.portfolio_names.append(display_name)
                self.available_list.insert(tk.END, display_name)
            
            if self.db_manager.db_path != ":memory:":
                conn.close()
                
            self.status_var.set(f"Loaded {len(portfolios)} portfolios")
            
        except Exception as e:
            self.logger.error(f"Error loading portfolios: {str(e)}")
            messagebox.showerror("Error", f"Failed to load portfolios: {str(e)}")
            self.status_var.set("Error loading portfolios")
    
    def _add_portfolio(self):
        """Add selected portfolios to the comparison list."""
        # Get selected indices from available list
        selected_indices = self.available_list.curselection()
        if not selected_indices:
            return
            
        # Add selected portfolios to the selected list
        for idx in selected_indices:
            portfolio_name = self.available_list.get(idx)
            if portfolio_name not in self.selected_portfolios:
                self.selected_portfolios.append(portfolio_name)
                self.selected_list.insert(tk.END, portfolio_name)
    
    def _remove_portfolio(self):
        """Remove selected portfolios from the comparison list."""
        # Get selected indices from selected list
        selected_indices = self.selected_list.curselection()
        if not selected_indices:
            return
            
        # Remove in reverse order to avoid index shifting
        for idx in sorted(selected_indices, reverse=True):
            portfolio_name = self.selected_list.get(idx)
            self.selected_portfolios.remove(portfolio_name)
            self.selected_list.delete(idx)
    
    def _compare_portfolios(self):
        """Gather data and visualize comparison for selected portfolios."""
        # Check if portfolios are selected
        if not self.selected_portfolios:
            messagebox.showwarning("No Selection", "Please select at least one portfolio to visualize.")
            return
            
        # Parse date range
        start_date = self._parse_date(self.start_year_var, self.start_month_var, self.start_day_var)
        end_date = self._parse_date(self.end_year_var, self.end_month_var, self.end_day_var)
        
        if not (start_date and end_date):
            return
            
        if start_date >= end_date:
            messagebox.showerror("Invalid Date Range", "Start date must be before end date.")
            return
            
        # Update status
        self.status_var.set("Fetching portfolio data...")
        
        # Start comparison thread
        thread = threading.Thread(
            target=self._run_comparison,
            args=(self.selected_portfolios, start_date, end_date)
        )
        thread.daemon = True
        thread.start()
    
    def _run_comparison(self, portfolio_names, start_date, end_date):
        """Run the comparison in a separate thread."""
        try:
            # Collect data for all selected portfolios
            portfolio_data = {}
            
            for portfolio_name in portfolio_names:
                # Get portfolio ID
                portfolio_id = self.portfolio_ids.get(portfolio_name)
                if not portfolio_id:
                    continue
                    
                # Get performance metrics
                perf_query = """
                    SELECT date, total_value, daily_return, cumulative_return, 
                           volatility, sharpe_ratio, max_drawdown
                    FROM PerformanceMetrics
                    WHERE portfolio_id = ? 
                    AND date BETWEEN ? AND ?
                    ORDER BY date
                """
                
                conn = self.db_manager._get_connection()
                
                # Convert dates to string for SQLite
                start_str = start_date.strftime('%Y-%m-%d')
                end_str = end_date.strftime('%Y-%m-%d')
                
                perf_df = pd.read_sql_query(
                    perf_query, 
                    conn, 
                    params=(portfolio_id, start_str, end_str),
                    parse_dates=['date']
                )
                
                if not perf_df.empty:
                    portfolio_data[portfolio_name] = perf_df
                
                if self.db_manager.db_path != ":memory:":
                    conn.close()
            
            # Check if we have data
            if not portfolio_data:
                self.status_var.set("No performance data found for selected portfolios.")
                messagebox.showinfo("No Data", "No performance data found for the selected date range.")
                return
                
            # Update UI with comparison data
            self.after(0, lambda: self._update_comparison_visualizations(portfolio_data))
            
        except Exception as e:
            self.logger.error(f"Error in portfolio comparison: {str(e)}")
            self.status_var.set(f"Error: {str(e)}")
            messagebox.showerror("Error", f"Failed to compare portfolios: {str(e)}")
    
    def _update_comparison_visualizations(self, portfolio_data):
        """Update all visualization tabs with comparison data."""
        # Update time series chart
        self._update_time_series_chart(portfolio_data)
        
        # Update bar chart
        self._update_bar_chart(portfolio_data)
        
        # Update risk-return chart
        self._update_risk_return_chart(portfolio_data)
        
        # Update summary table
        self._update_summary_table(portfolio_data)
        
        # Set status
        self.status_var.set(f"Comparison complete for {len(portfolio_data)} portfolios")
    
    def _update_time_series_chart(self, portfolio_data):
        """Update the time series comparison chart."""
        # Clear previous plot
        self.ax_time_series.clear()
        
        # Plot the selected metrics for each portfolio
        for portfolio_name, data in portfolio_data.items():
            if data.empty:
                continue
                
            # Determine which metric to plot based on checkbox selection
            if self.metric_vars["total_value"].get() and 'total_value' in data.columns:
                self.ax_time_series.plot(data['date'], data['total_value'], label=f"{portfolio_name} - Value")
            elif self.metric_vars["cumulative_return"].get() and 'cumulative_return' in data.columns:
                self.ax_time_series.plot(data['date'], data['cumulative_return'], label=f"{portfolio_name} - Return")
        
        # Set labels and title
        metric_name = "Portfolio Value" if self.metric_vars["total_value"].get() else "Cumulative Return"
        self.ax_time_series.set_title(f"Portfolio Comparison: {metric_name} Over Time")
        self.ax_time_series.set_xlabel("Date")
        self.ax_time_series.set_ylabel(metric_name)
        
        # Format x-axis as dates
        self.ax_time_series.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        self.fig_time_series.autofmt_xdate()
        
        # Add legend
        self.ax_time_series.legend()
        
        # Add grid
        self.ax_time_series.grid(True, linestyle='--', alpha=0.7)
        
        # Update the canvas
        self.fig_time_series.tight_layout()
        self.canvas_time_series.draw()
    
    def _update_bar_chart(self, portfolio_data):
        """Update the bar chart comparison."""
        # Clear previous plot
        self.ax_bar.clear()
        
        # Prepare data for bar chart
        portfolios = list(portfolio_data.keys())
        
        # Get metrics to compare
        metrics_to_compare = []
        metric_labels = []
        
        if self.metric_vars["cumulative_return"].get():
            metrics_to_compare.append("total_return")
            metric_labels.append("Total Return")
        
        if self.metric_vars["volatility"].get():
            metrics_to_compare.append("volatility")
            metric_labels.append("Volatility")
            
        if self.metric_vars["sharpe_ratio"].get():
            metrics_to_compare.append("sharpe_ratio")
            metric_labels.append("Sharpe Ratio")
            
        if self.metric_vars["max_drawdown"].get():
            metrics_to_compare.append("max_drawdown")
            metric_labels.append("Max Drawdown")
            
        if not metrics_to_compare:
            # Default to cumulative return if nothing selected
            metrics_to_compare = ["total_return"]
            metric_labels = ["Total Return"]
        
        # Get the values for each portfolio and metric
        metric_values = {metric: [] for metric in metrics_to_compare}
        
        for portfolio_name in portfolios:
            data = portfolio_data[portfolio_name]
            
            for metric in metrics_to_compare:
                # For max_drawdown, we want the minimum (worst) value
                if metric == "max_drawdown" and "max_drawdown" in data.columns:
                    value = data["max_drawdown"].min() if not data.empty else 0
                # For total_return, get the last cumulative return
                elif metric == "total_return" and "cumulative_return" in data.columns:
                    value = data["cumulative_return"].iloc[-1] if not data.empty else 0
                # For other metrics, get the last value
                elif metric in data.columns:
                    value = data[metric].iloc[-1] if not data.empty else 0
                else:
                    value = 0
                    
                metric_values[metric].append(value)
        
        # Set up bar positions
        x = np.arange(len(portfolios))
        width = 0.8 / len(metrics_to_compare)
        
        # Plot bars for each metric
        for i, metric in enumerate(metrics_to_compare):
            positions = x + i * width - (len(metrics_to_compare) - 1) * width / 2
            self.ax_bar.bar(positions, metric_values[metric], width, label=metric_labels[i])
        
        # Set labels and title
        self.ax_bar.set_title("Portfolio Metrics Comparison")
        self.ax_bar.set_xlabel("Portfolio")
        self.ax_bar.set_ylabel("Value")
        
        # Set x-tick labels
        self.ax_bar.set_xticks(x)
        # Shorten portfolio names for better display
        short_names = [name.split()[0] for name in portfolios]
        self.ax_bar.set_xticklabels(short_names, rotation=45, ha="right")
        
        # Add legend
        self.ax_bar.legend()
        
        # Add grid
        self.ax_bar.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Update the canvas
        self.fig_bar.tight_layout()
        self.canvas_bar.draw()
    
    def _update_risk_return_chart(self, portfolio_data):
        """Update the risk-return scatter plot."""
        # Clear previous plot
        self.ax_risk_return.clear()
        
        # Prepare data for scatter plot
        risk_values = []
        return_values = []
        portfolio_names = []
        
        for portfolio_name, data in portfolio_data.items():
            if data.empty or 'volatility' not in data.columns or 'cumulative_return' not in data.columns:
                continue
                
            # Get the final risk and return values
            risk = data['volatility'].iloc[-1]
            ret = data['cumulative_return'].iloc[-1]
            
            risk_values.append(risk)
            return_values.append(ret)
            portfolio_names.append(portfolio_name)
        
        # Create scatter plot
        scatter = self.ax_risk_return.scatter(risk_values, return_values, s=100, alpha=0.7)
        
        # Add labels for each point
        for i, name in enumerate(portfolio_names):
            self.ax_risk_return.annotate(name.split()[0], 
                                        (risk_values[i], return_values[i]),
                                        xytext=(5, 5),
                                        textcoords='offset points')
        
        # Add a risk-free line (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        max_risk = max(risk_values) if risk_values else 0.3
        self.ax_risk_return.plot([0, max_risk], [risk_free_rate, risk_free_rate], 
                               'k--', alpha=0.3, label='Risk-Free Rate (2%)')
        
        # Set labels and title
        self.ax_risk_return.set_title("Risk-Return Analysis")
        self.ax_risk_return.set_xlabel("Risk (Volatility)")
        self.ax_risk_return.set_ylabel("Return")
        
        # Add grid
        self.ax_risk_return.grid(True, linestyle='--', alpha=0.7)
        
        # Set axis limits
        self.ax_risk_return.set_xlim(left=0)
        
        # Add legend
        self.ax_risk_return.legend()
        
        # Update the canvas
        self.fig_risk_return.tight_layout()
        self.canvas_risk_return.draw()
    
    def _update_summary_table(self, portfolio_data):
        """Update the summary metrics table."""
        # Clear previous data
        for item in self.summary_tree.get_children():
            self.summary_tree.delete(item)
            
        # Add data for each portfolio
        for portfolio_name, data in portfolio_data.items():
            if data.empty:
                continue
                
            # Calculate metrics
            try:
                # Total return is the last cumulative return
                total_return = data['cumulative_return'].iloc[-1] if 'cumulative_return' in data.columns else 0
                
                # Volatility is the last value
                volatility = data['volatility'].iloc[-1] if 'volatility' in data.columns else 0
                
                # Sharpe ratio is the last value
                sharpe_ratio = data['sharpe_ratio'].iloc[-1] if 'sharpe_ratio' in data.columns else 0
                
                # Max drawdown is the minimum value
                max_drawdown = data['max_drawdown'].min() if 'max_drawdown' in data.columns else 0
                
                # Winning days percentage
                if 'daily_return' in data.columns:
                    winning_days = (data['daily_return'] > 0).mean() * 100
                else:
                    winning_days = 0
                    
                # Format values for display
                total_return_str = f"{total_return*100:.2f}%"
                volatility_str = f"{volatility*100:.2f}%"
                sharpe_ratio_str = f"{sharpe_ratio:.2f}"
                max_drawdown_str = f"{max_drawdown*100:.2f}%"
                winning_days_str = f"{winning_days:.1f}%"
                
                # Add to treeview
                self.summary_tree.insert("", "end", values=(
                    portfolio_name,
                    total_return_str,
                    volatility_str,
                    sharpe_ratio_str,
                    max_drawdown_str,
                    winning_days_str
                ))
                
            except Exception as e:
                self.logger.error(f"Error calculating metrics for {portfolio_name}: {str(e)}")
                # Add with error indicators
                self.summary_tree.insert("", "end", values=(
                    portfolio_name,
                    "Error",
                    "Error",
                    "Error",
                    "Error",
                    "Error"
                )) 