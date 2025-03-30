# Multi-Asset Portfolio Manager

## Overview

The Multi-Asset Portfolio Manager project is a comprehensive solution designed to optimize multi-asset portfolio strategies using machine learning. This project fetches financial data from Yahoo Finance, processes and cleans the data, stores it in a SQLite3 database, and uses machine learning models to propose optimized portfolio strategies with defined risk exposure.

This application provides a complete suite of tools for portfolio management, allowing one to construct optimized portfolios based on various investment strategies, fetch market data, track performance metrics, and compare different portfolios.

## Features

- **Portfolio Creation**: Create and manage portfolios with different investment strategies and asset universes
- **Data Management**: Fetch and store market data for various asset classes (equities, cryptocurrencies, commodities, ETFs)
- **Portfolio Construction**: Construct portfolios using advanced optimization strategies:
  - Low Risk Strategy: Focuses on minimizing volatility
  - Low Turnover Strategy: Limits trading to a maximum of two trades per month
  - High Yield Equity Strategy: Maximizes returns without constraints on volatility or turnover
- **Portfolio Comparison**: Visualize and compare multiple portfolios using various metrics:
  - Time series performance charts
  - Risk-return analysis
  - Bar chart comparisons of key metrics
  - Summary tables with performance statistics
- **Database Integration**: Store and retrieve portfolio data, trades, and performance metrics
- **Backtesting**: Test strategies against historical data with separate training and evaluation periods

## Installation

### Prerequisites

- Python 3.8+
- SQLite3

### Dependencies

Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

The requirements include:
- pandas>=1.3.5
- numpy>=1.20.3
- matplotlib>=3.5.1
- yfinance>=0.2.12
- scikit-learn>=1.0.2
- seaborn>=0.11.2
- tensorflow>=2.10.0
- keras>=2.10.0

## Project Structure
```
multi_asset_portfolio_manager/
│
├── gui/                    # GUI components
│   ├── components/         # Individual UI components
│   └── app.py              # Main application entry point
├── src/                    # Core functionality
│   ├── data_management/    # Data fetching and storage
│   ├── portfolio_optimization/ # Portfolio strategies and optimization
│   └── visualization/      # Visualization tools
├── outputs/                # Output files and database
│   └── database/           # SQLite database
├── main.py                 # Main script to run the project
├── requirements.txt        # Project dependencies
```

## Getting Started

1. **Clone the Repository**:
```bash
    git clone https://github.com/Alter-404/Data_Management.git
    cd Data_Management/multi_asset_portfolio_manager/
```

2. **Install Dependencies**:
```bash
    pip install -r requirements.txt
```
3. **Run the Project:**:
```bash
    python main.py
```

## Components

### GUI Components

- **Portfolio Creation**: Create and configure portfolios
- **Data Fetching**: Fetch market data for various assets
- **Portfolio Construction**: Construct and optimize portfolios
- **Portfolio Comparison**: Compare multiple portfolios across different metrics

### Core Modules

- **Database Manager**: Handles database operations for storing and retrieving data
- **Data Collector**: Fetches market data from external sources
- **Portfolio Optimizer**: Implements optimization algorithms for portfolio construction
- **Backtester**: Tests strategies against historical data
- **Portfolio Visualizer**: Creates visualizations of portfolio performance

## Portfolio Strategies

### Low Risk Strategy

Focuses on minimizing portfolio volatility by selecting assets with lower historical volatility and favorable risk-adjusted returns. Ideal for conservative investors prioritizing capital preservation.

### Low Turnover Strategy

Limits trading to a maximum of two trades per month, reducing transaction costs and tax implications. Uses momentum-based selection to prioritize the most promising trades when under constraints.

### High Yield Equity Strategy

Maximizes returns without constraints on volatility or turnover. Focuses on risk-adjusted momentum to allocate capital to the most promising assets, avoiding negative-momentum investments.

## Database Schema

The system uses an SQLite database with the following key tables:

- **Products**: Stores asset information
- **Deals**: Records of portfolio trades
- **Managers**: Portfolio manager information
- **Portfolios**: Portfolio configurations
- **PerformanceMetrics**: Daily and summary performance statistics
- **Returns**: Asset return data
- **Positions**: Current portfolio positions

## License
This project is licensed under the MIT License

## Contact
For any questions or suggestions, please contact the project authors:
- [Mariano BENJAMIN](mailto:mariano.benjamin@dauphine.eu)
- [Rayan BIBILONI](mailto:rayan.bibiloni@dauphine.eu)
- [Clément HUBERT](mailto:clement.hubert@dauphine.eu)
