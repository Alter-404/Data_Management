"""
Portfolio optimization module using machine learning agents.
"""

from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from .strategies import PortfolioStrategy, create_strategy
import tensorflow as tf
from tensorflow import keras

class PortfolioOptimizer:
    """Portfolio optimizer using machine learning agents."""
    
    def __init__(self, strategy: PortfolioStrategy):
        """Initialize the portfolio optimizer."""
        self.strategy = strategy
        self.model = self._build_optimization_model()
        
    def _build_optimization_model(self) -> keras.Model:
        """Build the deep learning model for portfolio optimization."""
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(20,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')
        ])
        
        model.compile(optimizer='adam',
                     loss='mse',
                     metrics=['mae'])
        return model
    
    def _prepare_optimization_features(
        self,
        market_data: pd.DataFrame,
        portfolio_data: pd.DataFrame,
        strategy_weights: Dict[str, float]
    ) -> np.ndarray:
        """Prepare features for the optimization model."""
        # Get unique symbols
        symbols = list(set(col.split('_')[0] for col in market_data.columns if '_returns' in col))
        
        # Initialize aggregated features
        agg_features = {
            'price_mean': [],
            'price_std': [],
            'volume_mean': [],
            'returns_mean': [],
            'returns_std': [],
            'sma_ratio': [],
            'ema_ratio': [],
            'volatility': [],
            'momentum': [],
            'rsi_mean': [],
            'macd_mean': [],
            'macd_signal_mean': [],
            'bb_width_mean': [],
            'bb_position': [],
            'atr_mean': [],
            'weight_concentration': [],
            'correlation_mean': [],
            'turnover': [],
            'sharpe_ratio': [],
            'max_drawdown': []
        }
        
        # Calculate features for each symbol and aggregate
        returns_data = pd.DataFrame()
        for symbol in symbols:
            # Store returns for correlation calculation
            returns_data[symbol] = market_data[f'{symbol}_returns']
            
            # Price features
            close = market_data[f'{symbol}_close']
            agg_features['price_mean'].append(close.mean())
            agg_features['price_std'].append(close.std())
            agg_features['volume_mean'].append(market_data[f'{symbol}_volume'].mean())
            
            # Returns features
            returns = market_data[f'{symbol}_returns']
            agg_features['returns_mean'].append(returns.mean())
            agg_features['returns_std'].append(returns.std())
            
            # Moving averages
            sma_20 = close.rolling(20).mean()
            sma_50 = close.rolling(50).mean()
            ema_12 = close.ewm(span=12, adjust=False).mean()
            ema_26 = close.ewm(span=26, adjust=False).mean()
            
            agg_features['sma_ratio'].append(sma_20.iloc[-1] / sma_50.iloc[-1] - 1)
            agg_features['ema_ratio'].append(ema_12.iloc[-1] / ema_26.iloc[-1] - 1)
            
            # Volatility and momentum
            agg_features['volatility'].append(returns.rolling(20).std().iloc[-1])
            agg_features['momentum'].append(close.pct_change(20, fill_method=None).iloc[-1])
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            agg_features['rsi_mean'].append(rsi.iloc[-1])
            
            # MACD
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9, adjust=False).mean()
            agg_features['macd_mean'].append(macd.iloc[-1])
            agg_features['macd_signal_mean'].append(macd_signal.iloc[-1])
            
            # Bollinger Bands
            sma_20 = close.rolling(20).mean()
            std_20 = close.rolling(20).std()
            bb_upper = sma_20 + (2 * std_20)
            bb_lower = sma_20 - (2 * std_20)
            bb_width = (bb_upper - bb_lower) / sma_20
            bb_position = (close - sma_20) / (2 * std_20)
            
            agg_features['bb_width_mean'].append(bb_width.iloc[-1])
            agg_features['bb_position'].append(bb_position.iloc[-1])
            
            # ATR
            high = market_data[f'{symbol}_high']
            low = market_data[f'{symbol}_low']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            agg_features['atr_mean'].append(atr.iloc[-1])
        
        # Calculate portfolio-level features
        weights = np.array([strategy_weights.get(symbol, 0.0) for symbol in symbols])
        agg_features['weight_concentration'].append(np.sum(weights ** 2))  # Herfindahl index
        
        # Average pairwise correlation
        corr_matrix = returns_data.corr()
        corr_mean = (corr_matrix.sum().sum() - len(symbols)) / (len(symbols) * (len(symbols) - 1))
        agg_features['correlation_mean'].append(corr_mean)
        
        # Portfolio turnover (if we have portfolio data)
        if not portfolio_data.empty:
            turnover = abs(portfolio_data['portfolio_returns']).mean()
            sharpe = portfolio_data['portfolio_returns'].mean() / portfolio_data['portfolio_returns'].std()
            max_dd = (portfolio_data['portfolio_returns'].cummax() - portfolio_data['portfolio_returns']).max()
        else:
            turnover = 0.0
            sharpe = 0.0
            max_dd = 0.0
            
        agg_features['turnover'].append(turnover)
        agg_features['sharpe_ratio'].append(sharpe)
        agg_features['max_drawdown'].append(max_dd)
        
        # Create final feature vector by taking means of aggregated features
        features = []
        for feature_name, values in agg_features.items():
            features.append(np.mean(values))
        
        # Fill NaN values with 0
        features = np.array([0.0 if pd.isna(x) else x for x in features])
        
        return features.reshape(1, -1)
    
    def optimize_weights(
        self,
        market_data: pd.DataFrame,
        portfolio_data: pd.DataFrame,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """Optimize portfolio weights using the strategy and ML model."""
        try:
            # Get initial weights from strategy
            strategy_weights = self.strategy.generate_signals(market_data, portfolio_data)
            
            # Prepare features for optimization
            features = self._prepare_optimization_features(
                market_data, portfolio_data, strategy_weights
            )
            
            # Get optimization scores from model
            scores = self.model.predict(features)[0]
            
            # Adjust weights based on optimization scores
            symbols = list(strategy_weights.keys())
            adjusted_weights = {}
            
            for symbol in symbols:
                base_weight = strategy_weights[symbol]
                score = float(scores[0])  # Convert to float
                
                # Adjust weight based on score (0.5 is neutral)
                if score > 0.5:
                    # Increase weight
                    adjusted_weights[symbol] = base_weight * (1 + (score - 0.5))
                else:
                    # Decrease weight
                    adjusted_weights[symbol] = base_weight * (2 * score)
            
            # Normalize weights to sum to 1
            total_weight = sum(adjusted_weights.values())
            if total_weight > 0:
                normalized_weights = {
                    symbol: weight / total_weight 
                    for symbol, weight in adjusted_weights.items()
                }
            else:
                # Fallback to strategy weights if optimization fails
                normalized_weights = strategy_weights
            
            # Apply constraints if provided
            if constraints:
                normalized_weights = self._apply_constraints(normalized_weights, constraints)
            
            return normalized_weights
            
        except Exception as e:
            raise Exception(f"Error optimizing portfolio weights: {str(e)}")
    
    def _apply_constraints(
        self,
        weights: Dict[str, float],
        constraints: Dict
    ) -> Dict[str, float]:
        """Apply constraints to the optimized weights."""
        constrained_weights = weights.copy()
        
        # Apply minimum weight constraint
        if 'min_weight' in constraints:
            min_weight = constraints['min_weight']
            for symbol in constrained_weights:
                if constrained_weights[symbol] < min_weight:
                    constrained_weights[symbol] = min_weight
        
        # Apply maximum weight constraint
        if 'max_weight' in constraints:
            max_weight = constraints['max_weight']
            for symbol in constrained_weights:
                if constrained_weights[symbol] > max_weight:
                    constrained_weights[symbol] = max_weight
        
        # Normalize weights after applying constraints
        total_weight = sum(constrained_weights.values())
        if total_weight > 0:
            constrained_weights = {
                symbol: weight / total_weight 
                for symbol, weight in constrained_weights.items()
            }
        
        return constrained_weights
    
    def generate_trades(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        min_trade_size: float = 0.01
    ) -> List[Tuple[str, float]]:
        """Generate trades to rebalance from current to target weights."""
        trades = []
        
        # Calculate required trades
        for symbol in set(current_weights.keys()) | set(target_weights.keys()):
            current = current_weights.get(symbol, 0.0)
            target = target_weights.get(symbol, 0.0)
            trade_size = target - current
            
            # Only generate trades above minimum size
            if abs(trade_size) >= min_trade_size:
                trades.append((symbol, trade_size))
        
        return sorted(trades, key=lambda x: abs(x[1]), reverse=True)
    
    def update_model(self, features: np.ndarray, rewards: np.ndarray):
        """Update the optimization model with new training data."""
        try:
            # Train the model with new data
            self.model.fit(
                features,
                rewards,
                epochs=10,
                batch_size=32,
                verbose=0
            )
        except Exception as e:
            raise Exception(f"Error updating optimization model: {str(e)}")

class PortfolioManager:
    """Portfolio manager that handles strategy execution and optimization."""
    
    def __init__(self, risk_profile: str):
        """Initialize the portfolio manager."""
        self.strategy = create_strategy(risk_profile)
        self.optimizer = PortfolioOptimizer(self.strategy)
        self.is_model_trained = False
        self.training_end_date = datetime(2022, 12, 31)
        
    def get_portfolio_update(
        self,
        market_data: pd.DataFrame,
        portfolio_data: pd.DataFrame,
        current_weights: Dict[str, float],
        constraints: Optional[Dict] = None
    ) -> List[Tuple[str, float]]:
        """Get updated portfolio allocation."""
        try:
            # Get current date from market data
            current_date = pd.to_datetime(market_data.index[-1])
            
            # Check if we need to train the model
            if not self.is_model_trained:
                self._train_model(market_data, portfolio_data)
                
            # Generate target weights based on the current date
            if current_date <= self.training_end_date:
                # During the training period, use the strategy to generate signals
                target_weights = self.strategy.generate_signals(market_data, portfolio_data)
            else:
                # During the evaluation period, use the trained strategy
                target_weights = self.strategy.generate_signals(market_data, portfolio_data)
                
                # Apply optimizer if available
                if self.is_model_trained:
                    # Fine-tune weights with optimizer
                    target_weights = self.optimizer.optimize_weights(
                        market_data, portfolio_data, constraints
                    )
            
            # Generate trades
            trades = self.optimizer.generate_trades(
                current_weights, target_weights
            )
            
            return trades
            
        except Exception as e:
            raise Exception(f"Error getting portfolio update: {str(e)}")
            
    def _train_model(self, market_data: pd.DataFrame, portfolio_data: pd.DataFrame):
        """Train the underlying model using historical data."""
        try:
            # Filter market data for training period
            training_data = market_data[market_data.index <= self.training_end_date]
            
            # Only train if we have enough data
            if len(training_data) >= 100:  # At least 100 data points for training
                # Train the strategy
                self.strategy.train(training_data, portfolio_data)
                self.is_model_trained = True
        except Exception as e:
            print(f"Error training model: {str(e)}")
            # Continue without training
    
    def evaluate_performance(
        self,
        portfolio_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Evaluate portfolio performance."""
        try:
            return self.strategy.calculate_portfolio_metrics(
                portfolio_data['portfolio_returns']
            )
        except Exception as e:
            raise Exception(f"Error evaluating performance: {str(e)}")
    
    def update_models(
        self,
        market_data: pd.DataFrame,
        portfolio_data: pd.DataFrame,
        performance_metrics: Dict[str, float]
    ):
        """Update models with new market data and performance metrics."""
        try:
            # Prepare features and rewards for model update
            features = self.optimizer._prepare_optimization_features(
                market_data,
                portfolio_data,
                {}  # Empty weights as we're using historical data
            )
            
            # Use Sharpe ratio as reward signal
            rewards = np.array([performance_metrics['sharpe_ratio']])
            
            # Update the optimization model
            self.optimizer.update_model(features, rewards)
            
        except Exception as e:
            raise Exception(f"Error updating models: {str(e)}")
