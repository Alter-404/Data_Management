"""
Data management module for the Multi-Asset Portfolio Manager.
Handles data collection, processing, and storage.
"""

from .database_manager import DatabaseManager
from .data_collector import DataCollector, DataProvider, YahooFinanceProvider
from .data_processing import DataProcessor

__all__ = [
    'DatabaseManager',
    'DataCollector',
    'DataProvider',
    'YahooFinanceProvider',
    'DataProcessor'
]
