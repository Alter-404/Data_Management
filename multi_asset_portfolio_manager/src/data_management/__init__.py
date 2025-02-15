import sys
import os

# Ajoute src/ au PYTHONPATH pour éviter les erreurs d'import + import des modules
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))
from data_management.data_processor import YahooFinanceDataProcessor
from data_management.database_manager import TechnicalIndicators

# Fonction principale
def run_data_processing():
    tickers = ["AAPL", "GOOGL", "MSFT"]
    start_date='2021-01-01'

    #Récupéré les données de Yahoo Finance
    processor = YahooFinanceDataProcessor(tickers,start_date)
    print("Donnée Yahoo récupéré avec succès.")

    # Calcul des indicateurs pour chaque ticker
    indicators = TechnicalIndicators()
    for ticker in tickers:
        indicators.process_ticker(ticker)
    print("Indicateur récupéré avec succès.")

# Point d'entrée
if __name__ == "__main__":
    run_data_processing()