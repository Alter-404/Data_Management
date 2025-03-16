import sqlite3
import pandas as pd
import ta
import logging

# Configuration du logging pour capturer les erreurs et les messages d'information
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class TechnicalIndicators:
    def __init__(self, db_path="financial_data.db"):
        """
        Initialise la connexion à la base de données.
        """
        self.db_path = db_path

    def _connect_db(self):
        """
        Établit une connexion avec la base de données SQLite.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            return conn
        except sqlite3.Error as e:
            logging.error(f"Erreur de connexion à la base de données : {e}")
            return None

    def get_stock_data(self, ticker):
        """
        Récupère les données de marché pour un ticker donné.
        """
        conn = self._connect_db()
        if conn is None:
            return None

        query = """
        SELECT date, open, high, low, close, volume 
        FROM stock_data 
        WHERE ticker = ? 
        ORDER BY date;
        """

        try:
            df = pd.read_sql_query(query, conn, params=(ticker,))
            conn.close()
            if df.empty:
                logging.warning(f"Aucune donnée trouvée pour {ticker}.")
                return None
            return df
        except sqlite3.Error as e:
            logging.error(f"Erreur lors de la récupération des données pour {ticker} : {e}")
            conn.close()
            return None
        
    def fill_missing_data(self, ticker):
        """
        Remplit les valeurs NULL dans la base avec la dernière valeur disponible.
        """
        conn = self._connect_db()
        if conn is None:
            return
        try:
            cursor = conn.cursor()
            # Remplir les valeurs manquantes avec la dernière valeur connue (ORDER BY date)
            query = f"""
            UPDATE stock_data
            SET open = COALESCE(open, (
                SELECT open FROM stock_data AS s2 
                WHERE s2.ticker = stock_data.ticker 
                AND s2.date < stock_data.date 
                AND s2.open IS NOT NULL
                ORDER BY s2.date DESC
                LIMIT 1
            )),
            high = COALESCE(high, (
                SELECT high FROM stock_data AS s2 
                WHERE s2.ticker = stock_data.ticker 
                AND s2.date < stock_data.date 
                AND s2.high IS NOT NULL
                ORDER BY s2.date DESC
                LIMIT 1
            )),
            low = COALESCE(low, (
                SELECT low FROM stock_data AS s2 
                WHERE s2.ticker = stock_data.ticker 
                AND s2.date < stock_data.date 
                AND s2.low IS NOT NULL
                ORDER BY s2.date DESC
                LIMIT 1
            )),
            close = COALESCE(close, (
                SELECT close FROM stock_data AS s2 
                WHERE s2.ticker = stock_data.ticker 
                AND s2.date < stock_data.date 
                AND s2.close IS NOT NULL
                ORDER BY s2.date DESC
                LIMIT 1
            )),
            volume = COALESCE(volume, (
                SELECT volume FROM stock_data AS s2 
                WHERE s2.ticker = stock_data.ticker 
                AND s2.date < stock_data.date 
                AND s2.volume IS NOT NULL
                ORDER BY s2.date DESC
                LIMIT 1
            ))
            WHERE ticker = ?
            """
            cursor.execute(query, (ticker,))
            conn.commit()

            logging.info(f"Valeurs manquantes corrigées pour {ticker}.")
        except sqlite3.Error as e:
            logging.error(f"Erreur lors de la correction des valeurs NULL : {e}")
        finally:
            conn.close()

    def calculate_indicators(self, df):
        """
        Calcule les indicateurs techniques sur un DataFrame contenant les prix historiques.
        """
        try:
            df["date"] = pd.to_datetime(df["date"])
            df.set_index("date", inplace=True)

            # Calcul des indicateurs
            df["SMA20"] = ta.trend.sma_indicator(df["close"], window=20) #Tendance
            df["SMA50"] = ta.trend.sma_indicator(df["close"], window=50) #Tendance
            df["SMA100"] = ta.trend.sma_indicator(df["close"], window=100) #Tendance
            df["SMA200"] = ta.trend.sma_indicator(df["close"], window=200) #Tendance
            df["RSI"] = ta.momentum.rsi(df["close"], window=14) #Momentum
            df["ATR"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14) #Volatilité
            df["MFI"] = ta.volume.money_flow_index(df["high"], df["low"], df["close"], df["volume"], window=14) #Volume

            # Suppression des valeurs NaN
            df.dropna(inplace=True)
            return df
        except Exception as e:
            logging.error(f"Erreur lors du calcul des indicateurs : {e}")
            return None

    def save_to_database(self, df, ticker):
        """
        Enregistre les indicateurs calculés dans la base de données.
        """
        conn = self._connect_db()
        if conn is None:
            return

        try:
            cursor = conn.cursor()
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS stock_indicators (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                SMA20 REAL, 
                SMA50 REAL, 
                SMA100 REAL, 
                SMA200 REAL,
                RSI REAL, 
                ATR REAL, 
                MFI REAL,
                UNIQUE(ticker, date) ON CONFLICT REPLACE
            );
            """)

            # Insérer les indicateurs dans la base de données
            for index, row in df.iterrows():
                cursor.execute("""
                INSERT INTO stock_indicators (ticker, date, SMA20, SMA50, SMA100, SMA200, RSI, ATR, MFI)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (ticker, index.strftime("%Y-%m-%d"), row["SMA20"], row["SMA50"], row["SMA100"],
                      row["SMA200"], row["RSI"], row["ATR"], row["MFI"]))

            conn.commit()
            conn.close()
            logging.info(f"Les indicateurs de {ticker} ont été enregistrés avec succès.")
        except sqlite3.Error as e:
            logging.error(f"Erreur lors de l'enregistrement des indicateurs pour {ticker} : {e}")
            conn.close()

    def process_ticker(self, ticker):
        """
        Processus complet : récupère les données, calcule les indicateurs et enregistre les résultats.
        """
        logging.info(f"Traitement des indicateurs pour {ticker}...")
        self.fill_missing_data(ticker) # Remplir les valeurs manquantes
        df = self.get_stock_data(ticker)
        if df is None:
            return

        df = self.calculate_indicators(df)
        if df is None or df.empty:
            logging.warning(f"Impossible de calculer les indicateurs pour {ticker}.")
            return

        self.save_to_database(df, ticker)