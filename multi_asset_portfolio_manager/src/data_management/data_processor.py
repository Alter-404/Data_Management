import sqlite3
import yfinance as yf
import pandas as pd
from datetime import datetime
import os

class YahooFinanceDataProcessor:
    def __init__(self, tickers, startdate,db_path="financial_data.db"):
        """
        Initialise la connexion à la base de données SQLite.
        :param db_path: Chemin vers la base de données SQLite.
        :param tickers: Liste des symboles boursiers.
        :param start_date: Date de début.

        """
        script_dir = os.path.dirname('./')
        self.db_path = os.path.join(script_dir, db_path)  # Utilisez db_path ici
        print(f"📁 La base de données sera enregistrée ici : {self.db_path}")  # Debug
        self._initialize_database()
        self.mybase={}
        self.start_date=startdate
        self.mytickers=tickers
        self.fetch_data()
        self.save_to_database()

    def _initialize_database(self):
        """Crée la table dans la base de données si elle n'existe pas."""
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stock_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                date TEXT,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        connection.commit()
        connection.close()

    def fetch_data(self, end_date=None):
        """
        Récupère les données de Yahoo Finance pour une liste de tickers.
        :param tickers: Liste des symboles boursiers.
        :param start_date: Date de début.
        :param end_date: Date de fin (par défaut, aujourd'hui).
        :return: Dictionnaire de DataFrames.
        """
        if end_date is None:
            end_date = datetime.today().strftime('%Y-%m-%d')

        print(f"Récupération des données pour {self.mytickers}...")
        df = yf.download(self.mytickers, start=self.start_date, end=end_date)[['Open', 'Close','High','Low','Volume']]  # Sélectionner uniquement les colonnes utiles

        if df.empty:
            print("Aucune donnée trouvée pour les tickers sélectionnés.")
            return

        # Transformer le MultiIndex en plusieurs DataFrames avec un index simple
        self.mybase = {}

        for ticker in self.mytickers:
            if (("Open", ticker) in df.columns) and (("Close", ticker) in df.columns):  # Vérification que le ticker est bien dans le DF
                df_ticker = df.loc[:, [("Open", ticker), ("Close", ticker),('High',ticker),('Low',ticker),("Volume", ticker)]].copy()
                df_ticker.columns = ['Open', 'Close','High','Low','Volume']
                df_ticker['Ticker']=ticker
                df_ticker.reset_index(inplace=True)  # Convertir l'index Date en colonne
                self.mybase[ticker] = df_ticker  # Stocker dans le dictionnaire

    def save_to_database(self):
        """
        Sauvegarde les données dans la base SQLite.
        :param data: Dictionnaire contenant les DataFrames des tickers.
        """
        connection = sqlite3.connect(self.db_path)
        cursor = connection.cursor()
        
        for ticker, df in self.mybase.items():
            # Renommer les colonnes
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume',
                'Ticker': 'ticker'  # Assurez-vous que cette colonne est bien renommée
            })

            # Vérifier et inclure uniquement les colonnes existantes
            expected_columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'ticker']
            existing_columns = [col for col in expected_columns if col in df.columns]
            df = df[existing_columns]

            # Convertir les colonnes au bon format
            if 'date' in df.columns:
                df['date'] = df['date'].astype(str)  # S'assurer que c'est bien une string
            if 'ticker' in df.columns:
                df['ticker'] = df['ticker'].astype(str)  # Assurer que c'est une string
            for col in ['open', 'high', 'low', 'close']:
                if col in df.columns:
                    df[col] = df[col].astype(float)        
            if 'volume' in df.columns:
                df['volume'] = df['volume'].fillna(0).astype(int)  # Gérer les NaN en les mettant à 0
            
            # Insérer les données dans la base de données
            df.to_sql('stock_data', connection, if_exists='append', index=False)
            print(f"Données pour {ticker} insérées dans la base de données.")
        connection.commit()
        connection.close()
