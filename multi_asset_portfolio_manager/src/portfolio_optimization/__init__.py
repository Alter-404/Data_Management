import sys
import os

# Ajoute src/ au PYTHONPATH pour éviter les erreurs d'import + import des modules
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from data_management.data_processor import YahooFinanceDataProcessor
from data_management.database_manager import TechnicalIndicators