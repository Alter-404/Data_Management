# Multi-Asset Portfolio Manager

## Overview

The Multi-Asset Portfolio Manager project is a comprehensive solution designed to optimize multi-asset portfolio strategies using machine learning. This project fetches financial data from Yahoo Finance, processes and cleans the data, stores it in a SQLite3 database, and uses machine learning models to propose optimized portfolio strategies with defined risk exposure.

## Features

- **Data Management**: Fetches, cleans, and processes financial data using the Yahoo Finance API.
- **Database Integration**: Stores processed data in a SQLite3 database for efficient querying and management.
- **Machine Learning**: Utilizes machine learning models to optimize portfolio strategies for maximized returns with defined risk exposure.
- **Visualization**: Provides visualizations and reports to illustrate portfolio performance and strategy recommendations.
- **Graphical User Interface (GUI)**: An optional dashboard for interactive data exploration and strategy visualization.

## Project Structure
```
multi_asset_portfolio_manager/
│
├── main.py                  # Main script to run the project
│
├── config/                  # Configuration and settings
│   ├── __init__.py          
│   ├── assets_list.py       # List of assets to be considered
│   └── settings.py          # General settings and configurations
│
├── src/                     # Source code directory
│   ├── __init__.py          
│   ├── data_management/     # Data handling (import, cleaning, database)
│   │   ├── __init__.py      
│   │   ├── data_processing.py # Functions for data importing and cleaning
│   │   ├── database_manager.py # Database operations
│   │   └── database/        # Database schema and storage 
│   │
│   ├── portfolio_optimization/  # Portfolio optimization & risk management
│   │   ├── __init__.py      
│   │   ├── backtester.py   # Functions for backtesting strategies
│   │   ├── optimizer.py     # Machine learning agent for optimization
│   │   └── risk_manager.py  # Functions for managing risk exposure
│   │
│   ├── visualization/       # Data visualization
│   │   ├── __init__.py      
│   │   └── plot_results.py  # Functions for plotting results
│
├── outputs/                 # Store generated outputs
│   ├── logs/                # Log files for debugging and monitoring    
│   │   └── model_training_logs/
│   ├── results/             # Results from model predictions
│   ├── visualizations/      # Visualizations of results
│   ├── database/            # Database storage     
│   └── models/              # Trained ML models 
│
├── tests/                   # Unit tests
│   ├── __init__.py          
│   ├── test_backtesting.py   # Unit tests for backtesting functions
│   ├── test_data_management.py # Unit tests for data management functions
│   └── test_optimizer.py    # Unit tests for optimizer functions
│
├── docs/                    # Documentation
│
├── gui/                     # User Interface (optional)
│   ├── __init__.py          
│   ├── app.py               # Main dashboard file
│   ├── components/         # Directory for GUI components
│   │   ├── __init__.py      
│   │   ├── dashboard.py    # Dashboard component
│   │   └── plot_viewer.py   # Component for viewing plots
│   └── styles/              # Directory for CSS or styling files
│       └── styles.css       
│
└── requirements.txt         # Required Python packages
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

## Configuration
Modify
- `config/settings.py` to adjust project parameters (risk constraints and optimization settings)
- `config/assets_list.py` to change the Assets list available

## Results & Outputs
- Logs: Stored in `outputs/logs/`
- Visualizations: Stored in `outputs/visualizations/`
- Model Predictions: Stored in `outputs/results/`
- Trained Models: Saved in `outputs/models/`

## License
This project is licensed under the MIT License

## Contact
For any questions or suggestions, please contact the project authors:
- [Mariano BENJAMIN](mailto:mariano.benjamin@dauphine.eu)
- [Rayan BIBILONI](mailto:rayan.bibiloni@dauphine.eu)
- [Clément HUBERT](mailto:clement.hubert@dauphine.eu)
