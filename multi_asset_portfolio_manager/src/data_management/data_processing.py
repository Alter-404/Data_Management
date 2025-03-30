"""
Data processing module for cleaning and preparing market data.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    """Class for processing and preparing market data."""
    
    def __init__(self):
        """Initialize the data processor."""
        self.scaler = StandardScaler()
        self.feature_columns = []
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare market data."""
        try:
            # Create a copy to avoid modifying the original
            cleaned_df = df.copy()
            
            # Convert date column to datetime if it's not already
            if not pd.api.types.is_datetime64_any_dtype(cleaned_df['date']):
                cleaned_df['date'] = pd.to_datetime(cleaned_df['date'])
            
            # Sort by date
            cleaned_df = cleaned_df.sort_values('date')
            
            # Handle missing values
            cleaned_df = self._handle_missing_values(cleaned_df)
            
            # Remove outliers
            cleaned_df = self._remove_outliers(cleaned_df)
            
            return cleaned_df
            
        except Exception as e:
            raise Exception(f"Error cleaning data: {str(e)}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset."""
        # Forward fill missing values
        df = df.ffill()
        
        # Backward fill any remaining missing values
        df = df.bfill()
        
        # If there are still missing values, fill with mean
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        """Remove outliers using the Z-score method."""
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
            df[column] = df[column].mask(z_scores > threshold)
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for analysis."""
        try:
            # Create a copy to avoid modifying the original
            feature_df = df.copy()
            
            # Get unique symbols from column names
            symbols = set(col.split('_')[0] for col in df.columns if col != 'date')
            
            # Calculate technical indicators for each symbol
            for symbol in symbols:
                # Calculate returns
                feature_df[f'{symbol}_returns'] = feature_df[f'{symbol}_close'].pct_change(fill_method=None)
                
                # Calculate moving averages
                feature_df[f'{symbol}_ma_20'] = feature_df[f'{symbol}_close'].rolling(window=20).mean()
                feature_df[f'{symbol}_ma_50'] = feature_df[f'{symbol}_close'].rolling(window=50).mean()
                
                # Calculate volatility
                feature_df[f'{symbol}_volatility'] = feature_df[f'{symbol}_returns'].rolling(window=20).std()
                
                # Calculate RSI
                delta = feature_df[f'{symbol}_close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                feature_df[f'{symbol}_rsi'] = 100 - (100 / (1 + rs))
            
            # Store feature columns
            self.feature_columns = [col for col in feature_df.columns if col not in df.columns]
            
            # Scale features
            feature_df[self.feature_columns] = self.scaler.fit_transform(feature_df[self.feature_columns])
            
            return feature_df
            
        except Exception as e:
            raise Exception(f"Error preparing features: {str(e)}")
    
    def calculate_portfolio_features(
        self,
        df: pd.DataFrame,
        portfolio: Dict[str, float]
    ) -> pd.DataFrame:
        """Calculate portfolio-level features."""
        try:
            # Create a copy to avoid modifying the original
            portfolio_df = df.copy()
            
            # Calculate portfolio returns
            portfolio_df['portfolio_returns'] = 0
            for symbol, weight in portfolio.items():
                if f'{symbol}_returns' in portfolio_df.columns:
                    portfolio_df['portfolio_returns'] += weight * portfolio_df[f'{symbol}_returns']
            
            # Calculate portfolio metrics
            portfolio_df['portfolio_volatility'] = portfolio_df['portfolio_returns'].rolling(window=20).std()
            portfolio_df['portfolio_sharpe'] = (
                portfolio_df['portfolio_returns'].rolling(window=20).mean() /
                portfolio_df['portfolio_volatility']
            )
            
            return portfolio_df
            
        except Exception as e:
            raise Exception(f"Error calculating portfolio features: {str(e)}")
    
    def calculate_risk_metrics(
        self,
        df: pd.DataFrame,
        portfolio: Dict[str, float]
    ) -> Dict[str, float]:
        """Calculate risk metrics for the portfolio."""
        try:
            # Calculate portfolio returns
            portfolio_returns = 0
            for symbol, weight in portfolio.items():
                if f'{symbol}_returns' in df.columns:
                    portfolio_returns += weight * df[f'{symbol}_returns']
            
            # Calculate risk metrics
            risk_metrics = {
                'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
                'sharpe_ratio': (
                    portfolio_returns.mean() * 252 /
                    (portfolio_returns.std() * np.sqrt(252))
                ),
                'max_drawdown': (
                    (portfolio_returns.cummax() - portfolio_returns) /
                    portfolio_returns.cummax()
                ).max(),
                'var_95': np.percentile(portfolio_returns, 5),
                'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
            }
            
            return risk_metrics
            
        except Exception as e:
            raise Exception(f"Error calculating risk metrics: {str(e)}")
