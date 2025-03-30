"""
Portfolio Management GUI - Main Application
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Add the project root directory to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Import project modules
from src.data_management.database_manager import DatabaseManager
from src.data_management.data_collector import DataCollector
from src.portfolio_optimization.strategies import create_strategy
from src.visualization.portfolio_visualizer import PortfolioVisualizer

# Import GUI modules
from gui.components.client_management import ClientManagementFrame
from gui.components.portfolio_creation import PortfolioCreationFrame
from gui.components.data_fetching import DataFetchingFrame
from gui.components.portfolio_construction import PortfolioConstructionFrame
from gui.components.portfolio_comparison import PortfolioComparisonFrame

class PortfolioManagerApp:
    """Main application class for the Portfolio Manager GUI."""
    
    def __init__(self, root):
        """Initialize the application."""
        self.root = root
        self.root.title("Portfolio Management System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set up the status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        self.status_bar = ttk.Label(self.root, relief=tk.SUNKEN, anchor=tk.W, textvariable=self.status_var)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize database and data collectors
        self.db_manager = DatabaseManager(project_root + "/outputs/database/database.db")
        self.data_collector = DataCollector()
        self.visualizer = PortfolioVisualizer()
        
        # Create the main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create the tab control
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_tabs()
        
    def create_tabs(self):
        """Create the application tabs."""
        # Create tab control
        self.tab_control = ttk.Notebook(self.main_frame)
        self.tab_control.pack(expand=1, fill="both")
        
        # Create individual tabs
        self.portfolio_tab = ttk.Frame(self.tab_control)
        self.data_tab = ttk.Frame(self.tab_control)
        self.portfolio_construction_tab = ttk.Frame(self.tab_control)
        self.comparison_tab = ttk.Frame(self.tab_control)
        
        # Add tabs to notebook
        self.tab_control.add(self.portfolio_tab, text="Portfolio")
        self.tab_control.add(self.data_tab, text="Data")
        self.tab_control.add(self.portfolio_construction_tab, text="Construction")
        self.tab_control.add(self.comparison_tab, text="Comparison")
        
        # Create tab contents
        self.portfolio_frame = PortfolioCreationFrame(self.portfolio_tab, self.db_manager, self.status_var)
        self.portfolio_frame.pack(fill="both", expand=True)
        
        self.data_frame = DataFetchingFrame(self.data_tab, self.db_manager, self.data_collector, self.status_var)
        self.data_frame.pack(fill="both", expand=True)
        
        self.construction_frame = PortfolioConstructionFrame(self.portfolio_construction_tab, self.db_manager, self.data_collector, self.status_var)
        self.construction_frame.pack(fill="both", expand=True)
        
        self.comparison_frame = PortfolioComparisonFrame(self.comparison_tab, self.db_manager, self.visualizer, self.status_var)
        self.comparison_frame.pack(fill="both", expand=True)

def main():
    """Main function to run the application."""
    root = tk.Tk()
    
    # Set up styles
    style = ttk.Style()
    try:
        style.theme_use("clam")  # Use clam theme for a more modern look
    except:
        pass  # If theme is not available, use default
    
    app = PortfolioManagerApp(root)
    root.mainloop() 