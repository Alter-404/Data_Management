"""
Visualization module for portfolio analysis and reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime
import logging

class PortfolioVisualizer:
    """Class for creating portfolio visualizations.."""
    
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        self.logger = logging.getLogger(__name__)
        self._setup_style()
    
    def _setup_style(self):
        """Set up the visualization style."""
        sns.set_style("whitegrid")  # This will set up seaborn's style
        sns.set_palette("husl")
        
        # Set default figure size and DPI
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100
        
        # Set font sizes
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['axes.labelsize'] = 10
    
    def plot_portfolio_value(self, portfolio_data: pd.DataFrame, title: str = "Portfolio Value Over Time", 
                           show_periods: bool = True) -> plt.Figure:
        """Plot portfolio value over time."""
        try:
            fig, ax = plt.subplots()
            
            # Plot portfolio value
            ax.plot(portfolio_data['date'], portfolio_data['portfolio_value'], 
                   label='Portfolio Value', linewidth=2)
            
            # Add training/evaluation period separation if requested
            if show_periods:
                # Add vertical line at 2022-12-31
                training_end = datetime(2022, 12, 31)
                if min(portfolio_data['date']) <= training_end <= max(portfolio_data['date']):
                    ax.axvline(x=training_end, color='red', linestyle='--', alpha=0.7)
                    ax.annotate('Training / Evaluation Split', 
                               xy=(training_end, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.5),
                               xytext=(15, 0), textcoords='offset points',
                               rotation=90, va='center', ha='left',
                               color='red', alpha=0.7)
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            
            # Format y-axis with dollar signs
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting portfolio value: {str(e)}")
            raise
    
    def plot_portfolio_returns(self, portfolio_data: pd.DataFrame, 
                             title: str = "Portfolio Returns Over Time") -> plt.Figure:
        """Plot portfolio returns over time."""
        try:
            fig, ax = plt.subplots()
            
            # Plot returns
            ax.plot(portfolio_data['date'], portfolio_data['portfolio_returns'], 
                   label='Returns', linewidth=2)
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Returns (%)')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting portfolio returns: {str(e)}")
            raise
    
    def plot_asset_weights(self, weights_data: pd.DataFrame, 
                         title: str = "Portfolio Asset Weights") -> plt.Figure:
        """Plot portfolio asset weights."""
        try:
            fig, ax = plt.subplots()
            
            # Get the latest weights
            latest_weights = weights_data.iloc[-1].drop('date')
            
            # Create pie chart
            ax.pie(latest_weights, labels=latest_weights.index, autopct='%1.1f%%',
                  startangle=90)
            
            # Add title
            ax.set_title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting asset weights: {str(e)}")
            raise
    
    def plot_risk_metrics(self, portfolio_data: pd.DataFrame, 
                         title: str = "Portfolio Risk Metrics") -> plt.Figure:
        """Plot portfolio risk metrics over time."""
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot volatility
            ax1.plot(portfolio_data['date'], portfolio_data['portfolio_volatility'],
                    label='Volatility', linewidth=2)
            ax1.set_title('Portfolio Volatility')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Volatility (%)')
            ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            ax1.grid(True, alpha=0.3)
            
            # Plot Sharpe ratio
            ax2.plot(portfolio_data['date'], portfolio_data['portfolio_sharpe'],
                    label='Sharpe Ratio', linewidth=2)
            ax2.set_title('Sharpe Ratio')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Sharpe Ratio')
            ax2.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            for ax in [ax1, ax2]:
                plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Add title
            fig.suptitle(title)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting risk metrics: {str(e)}")
            raise
    
    def plot_drawdown(self, portfolio_data: pd.DataFrame, 
                     title: str = "Portfolio Drawdown") -> plt.Figure:
        """Plot portfolio drawdown over time."""
        try:
            fig, ax = plt.subplots()
            
            # Calculate drawdown
            portfolio_value = portfolio_data['portfolio_value']
            rolling_max = portfolio_value.expanding().max()
            drawdown = (portfolio_value - rolling_max) / rolling_max
            
            # Plot drawdown
            ax.fill_between(portfolio_data['date'], drawdown, 0, 
                          color='red', alpha=0.3, label='Drawdown')
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown (%)')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting drawdown: {str(e)}")
            raise
    
    def plot_correlation_matrix(self, returns_data: pd.DataFrame, 
                              title: str = "Asset Correlation Matrix") -> plt.Figure:
        """Plot correlation matrix for portfolio assets."""
        try:
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Calculate correlation matrix
            corr_matrix = returns_data.corr()
            
            # Create heatmap
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       fmt='.2f', square=True, ax=ax)
            
            # Add title
            ax.set_title(title)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting correlation matrix: {str(e)}")
            raise
    
    def plot_risk_contribution(self, risk_contribution: pd.DataFrame, 
                             title: str = "Risk Contribution by Asset") -> plt.Figure:
        """Plot risk contribution by asset."""
        try:
            fig, ax = plt.subplots()
            
            # Create bar plot
            sns.barplot(x='symbol', y='risk_contribution', data=risk_contribution, ax=ax)
            
            # Add title and labels
            ax.set_title(title)
            ax.set_xlabel('Asset')
            ax.set_ylabel('Risk Contribution (%)')
            
            # Format y-axis as percentage
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            # Rotate x-axis labels
            plt.xticks(rotation=45)
            
            # Adjust layout
            plt.tight_layout()
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error plotting risk contribution: {str(e)}")
            raise
    
    def generate_performance_report(self, portfolio_data: pd.DataFrame, 
                                  weights_data: pd.DataFrame,
                                  risk_contribution: pd.DataFrame,
                                  returns_data: pd.DataFrame) -> Dict[str, List[plt.Figure]]:
        """Generate a comprehensive performance report."""
        try:
            # Create combined report
            combined_figures = []
            
            # Add portfolio value chart
            combined_figures.append(self.plot_portfolio_value(portfolio_data))
            
            # Add returns chart
            combined_figures.append(self.plot_portfolio_returns(portfolio_data))
            
            # Add risk metrics chart
            combined_figures.append(self.plot_risk_metrics(portfolio_data))
            
            # Add drawdown chart
            combined_figures.append(self.plot_drawdown(portfolio_data))
            
            # Add asset weights chart
            combined_figures.append(self.plot_asset_weights(weights_data))
            
            # Add correlation matrix
            combined_figures.append(self.plot_correlation_matrix(returns_data))
            
            # Add risk contribution chart
            combined_figures.append(self.plot_risk_contribution(risk_contribution))
            
            # Now create separate reports for training and evaluation periods
            training_end = datetime(2022, 12, 31)
            
            # Split data into training and evaluation periods
            training_portfolio = portfolio_data[portfolio_data['date'] <= training_end].copy() if not portfolio_data.empty else pd.DataFrame()
            evaluation_portfolio = portfolio_data[portfolio_data['date'] > training_end].copy() if not portfolio_data.empty else pd.DataFrame()
            
            training_weights = weights_data[weights_data['date'] <= training_end].copy() if not weights_data.empty else pd.DataFrame()
            evaluation_weights = weights_data[weights_data['date'] > training_end].copy() if not weights_data.empty else pd.DataFrame()
            
            training_returns = returns_data[returns_data.index <= training_end].copy() if not returns_data.empty else pd.DataFrame()
            evaluation_returns = returns_data[returns_data.index > training_end].copy() if not returns_data.empty else pd.DataFrame()
            
            # Create training period report
            training_figures = []
            if not training_portfolio.empty:
                training_figures.append(self.plot_portfolio_value(training_portfolio, title="Training Period Portfolio Value (2015-2022)", show_periods=False))
                training_figures.append(self.plot_portfolio_returns(training_portfolio, title="Training Period Returns (2015-2022)"))
                training_figures.append(self.plot_risk_metrics(training_portfolio, title="Training Period Risk Metrics (2015-2022)"))
                training_figures.append(self.plot_drawdown(training_portfolio, title="Training Period Drawdown (2015-2022)"))
                
                if not training_weights.empty:
                    training_figures.append(self.plot_asset_weights(training_weights, title="Training Period Asset Weights (2015-2022)"))
                
                if not training_returns.empty:
                    training_figures.append(self.plot_correlation_matrix(training_returns, title="Training Period Correlation Matrix (2015-2022)"))
            
            # Create evaluation period report
            evaluation_figures = []
            if not evaluation_portfolio.empty:
                evaluation_figures.append(self.plot_portfolio_value(evaluation_portfolio, title="Evaluation Period Portfolio Value (2023-2024)", show_periods=False))
                evaluation_figures.append(self.plot_portfolio_returns(evaluation_portfolio, title="Evaluation Period Returns (2023-2024)"))
                evaluation_figures.append(self.plot_risk_metrics(evaluation_portfolio, title="Evaluation Period Risk Metrics (2023-2024)"))
                evaluation_figures.append(self.plot_drawdown(evaluation_portfolio, title="Evaluation Period Drawdown (2023-2024)"))
                
                if not evaluation_weights.empty:
                    evaluation_figures.append(self.plot_asset_weights(evaluation_weights, title="Evaluation Period Asset Weights (2023-2024)"))
                
                if not evaluation_returns.empty:
                    evaluation_figures.append(self.plot_correlation_matrix(evaluation_returns, title="Evaluation Period Correlation Matrix (2023-2024)"))
            
            return {
                'combined': combined_figures,
                'training': training_figures,
                'evaluation': evaluation_figures
            }
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {str(e)}")
            raise
    
    def save_figures(self, figures: Union[List[plt.Figure], Dict[str, List[plt.Figure]]], output_dir: str):
        """Save figures to the output directory."""
        try:
            import os
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Handle both list and dictionary formats
            if isinstance(figures, dict):
                # Handle dictionary of figure lists
                for period, period_figures in figures.items():
                    # Create subdirectory for each period
                    period_dir = os.path.join(output_dir, period)
                    os.makedirs(period_dir, exist_ok=True)
                    
                    # Save figures in the period subdirectory
                    for i, fig in enumerate(period_figures):
                        output_path = os.path.join(period_dir, f'figure_{i+1}.png')
                        fig.savefig(output_path, dpi=300, bbox_inches='tight')
                        
                    self.logger.info(f"Saved {len(period_figures)} figures to {period_dir}")
            else:
                # Handle list of figures (original format)
                for i, fig in enumerate(figures):
                    output_path = os.path.join(output_dir, f'figure_{i+1}.png')
                    fig.savefig(output_path, dpi=300, bbox_inches='tight')
                    
                self.logger.info(f"Saved {len(figures)} figures to {output_dir}")
                
        except Exception as e:
            self.logger.error(f"Error saving figures: {str(e)}")
            raise
    
    def plot_portfolio_periods(self, portfolio_data: pd.DataFrame) -> Tuple[plt.Figure, plt.Figure]:
        """Plot portfolio performance separately for training and evaluation periods."""
        try:
            # Split data into training and evaluation periods
            training_end = datetime(2022, 12, 31)
            training_data = portfolio_data[portfolio_data['date'] <= training_end].copy()
            evaluation_data = portfolio_data[portfolio_data['date'] > training_end].copy()
            
            # Create figures for both periods
            training_fig = None
            evaluation_fig = None
            
            if not training_data.empty:
                training_fig, training_ax = plt.subplots()
                training_ax.plot(training_data['date'], training_data['portfolio_value'], 
                               linewidth=2, color='blue')
                training_ax.set_title('Training Period Performance (2015-2022)')
                training_ax.set_xlabel('Date')
                training_ax.set_ylabel('Portfolio Value ($)')
                training_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                training_ax.grid(True, alpha=0.3)
                plt.setp(training_ax.get_xticklabels(), rotation=45)
                plt.tight_layout()
            
            if not evaluation_data.empty:
                evaluation_fig, evaluation_ax = plt.subplots()
                evaluation_ax.plot(evaluation_data['date'], evaluation_data['portfolio_value'], 
                                linewidth=2, color='green')
                evaluation_ax.set_title('Evaluation Period Performance (2023-2024)')
                evaluation_ax.set_xlabel('Date')
                evaluation_ax.set_ylabel('Portfolio Value ($)')
                evaluation_ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
                evaluation_ax.grid(True, alpha=0.3)
                plt.setp(evaluation_ax.get_xticklabels(), rotation=45)
                plt.tight_layout()
            
            return training_fig, evaluation_fig
            
        except Exception as e:
            self.logger.error(f"Error plotting portfolio periods: {str(e)}")
            raise 