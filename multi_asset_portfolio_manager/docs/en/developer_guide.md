# Developer Guide

This guide provides information for developers who want to understand, extend, or contribute to the Multi-Asset Portfolio Manager project.

## Architecture Overview

The application follows a modular architecture with clear separation of concerns:

```
multi_asset_portfolio_manager/
├── gui/                    # GUI components (Tkinter-based)
│   ├── components/         # Individual UI components 
│   └── app.py              # Main application entry point
├── src/                    # Core functionality
│   ├── data_management/    # Data fetching and storage
│   ├── portfolio_optimization/ # Portfolio strategies and optimization
│   └── visualization/      # Visualization tools
└── outputs/                # Output files and database
```

### Core Components

1. **Database Manager** (`src/data_management/database_manager.py`):
   - Handles all database operations
   - Manages the SQLite database schema
   - Provides methods for CRUD operations on portfolios, assets, and trades

2. **Data Collector** (`src/data_management/data_collector.py`):
   - Fetches market data from external sources (Yahoo Finance)
   - Preprocesses and cleans the data
   - Calculates returns and stores them in the database

3. **Portfolio Optimization** (`src/portfolio_optimization/`):
   - Implements various portfolio optimization strategies
   - Includes backtesting functionality
   - Provides portfolio construction and evaluation tools

4. **GUI Components** (`gui/components/`):
   - Implements the user interface using Tkinter
   - Provides separate frames for different functionalities
   - Handles user interactions and displays results

## Adding New Features

### Adding a New Strategy

To add a new portfolio optimization strategy:

1. Create a new class in `src/portfolio_optimization/strategies.py` that inherits from `PortfolioStrategy`:

```python
class YourNewStrategy(PortfolioStrategy):
    def __init__(self):
        super().__init__('Your Strategy Name')
        # Initialize strategy-specific parameters
        
    def generate_signals(self, market_data, portfolio_data):
        # Implement your strategy logic
        # Return a dictionary of asset weights
        return weights_dict
        
    def train(self, market_data, portfolio_data):
        # Implement training logic for your strategy
        self.is_trained = True
        self.training_data = market_data.copy()
```

2. Update the `create_strategy` function to include your new strategy:

```python
def create_strategy(risk_profile):
    if risk_profile == 'Your Strategy Name':
        return YourNewStrategy()
    # Existing strategies...
```

3. Update the UI to include your new strategy in the dropdown menu in `gui/components/portfolio_creation.py`:

```python
self.strategy_combo['values'] = ('Low Risk', 'Low Turnover', 'High Yield Equity', 'Your Strategy Name')
```

### Adding a New Data Source

To add a new data source:

1. Create a new provider class in `src/data_management/data_collector.py`:

```python
class YourDataProvider(DataProvider):
    def __init__(self, api_key=None):
        super().__init__('Your Provider Name')
        self.api_key = api_key
        
    def fetch_market_data(self, symbol, start_date, end_date):
        # Implement API calls to your data source
        # Process the data into a pandas DataFrame
        # Return the DataFrame
```

2. Update the `DataCollector` class to use your new provider:

```python
def __init__(self, provider_name='your_provider'):
    # Add your provider to the available providers
    if provider_name == 'your_provider':
        self.provider = YourDataProvider()
```

### Adding a New Visualization

To add a new visualization:

1. Add a new method to `src/visualization/portfolio_visualizer.py`:

```python
def your_new_visualization(self, data, **kwargs):
    # Create your visualization using matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    # Implement your visualization
    # Return the figure
    return fig
```

2. Update the GUI to include your new visualization in the appropriate component.

## Database Schema

The database schema is defined in `src/data_management/database_manager.py`. To modify the schema:

1. Update the `_initialize_database` method to include your new tables or modify existing ones
2. Add appropriate methods for CRUD operations on your new tables
3. Consider creating a migration script if making changes to an existing database

## Testing

The project uses Python's built-in `unittest` framework for testing. To add tests:

1. Create test files in the `tests/` directory
2. Run tests using the command: `python -m unittest discover tests`

Follow these guidelines for testing:
- Test each component independently
- Use mock objects for external dependencies
- Create test fixtures for common test setups
- Cover both success and failure cases

## Code Style Guidelines

- Follow PEP 8 coding standards
- Use docstrings for all classes and methods
- Add type hints using Python's typing module
- Keep methods short and focused on a single responsibility
- Use meaningful variable and function names

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Submit a pull request

Before submitting a pull request:
- Ensure your code is well-tested
- Update documentation to reflect your changes
- Make sure all tests pass
- Review your code for quality and style consistency 