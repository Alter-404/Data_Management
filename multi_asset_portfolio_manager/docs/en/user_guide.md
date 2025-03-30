# User Guide

This guide provides detailed instructions on how to use the Multi-Asset Portfolio Manager application effectively.

## Getting Started

After installing dependencies, you can start the application by running:

```bash
python -m main
```

The application will open with the main interface containing four tabs:
- Portfolio
- Data
- Construction
- Comparison

## Portfolio Tab

The Portfolio tab allows you to create and manage investment portfolios.

### Creating a New Portfolio

1. Click the "New Portfolio" button
2. Fill in the required information:
   - Portfolio Name: A unique name for your portfolio
   - Client: Select an existing client or create a new one
   - Strategy: Choose from Low Risk, Low Turnover, or High Yield Equity
   - Asset Universe: Select from available asset universes
3. Click "Create" to create the portfolio

### Managing Portfolios

- **View Portfolio**: Select a portfolio from the dropdown list to view its details
- **Edit Portfolio**: Click the "Edit" button to modify the portfolio settings
- **Delete Portfolio**: Click the "Delete" button to remove a portfolio

## Data Tab

The Data tab is used to fetch and manage market data for different assets.

### Fetching Market Data

1. Select the Asset Universe from the dropdown (e.g., US Equities, Global Equities, Cryptocurrencies)
2. Select the specific assets you want to fetch data for
3. Set the date range using the Date Range selector
4. Click "Fetch Data" to retrieve market data from external sources

### Viewing Market Data

- The fetched data is displayed in a table with dates and price information
- Charts show the price history and returns for the selected assets
- The Returns tab shows calculated daily and cumulative returns

### Data Management

- Data is automatically stored in the database for future use
- You can refresh existing data using the "Refresh" button
- Export data using the "Export" button (CSV format)

## Construction Tab

The Construction tab allows you to build and optimize portfolios based on selected strategies.

### Building a Portfolio

1. Select a portfolio from the dropdown list
2. Choose the construction parameters:
   - Initial Capital: Starting investment amount
   - Training Period: Date range for strategy training
   - Evaluation Period: Date range for strategy testing
   - Max Positions: Maximum number of assets to hold
3. Click "Construct Portfolio" to run the optimization

### Portfolio Analysis

After construction, the following information is displayed:

- **Performance**: Portfolio value over time with key metrics
- **Holdings**: Current portfolio holdings with weights
- **Trades**: Trade history showing buys and sells
- **Returns**: Daily and cumulative returns
- **Metrics**: Performance metrics like Sharpe ratio, volatility, and drawdown

### Saving Results

- Results are automatically saved to the database
- You can manually save specific results using the "Save Results" button

## Comparison Tab

The Comparison tab allows you to compare multiple portfolios and their performance metrics.

### Selecting Portfolios for Comparison

1. From the Available Portfolios list, select one or more portfolios
2. Click "Add >" to move them to the Selected Portfolios list
3. Set the date range for comparison
4. Select the metrics you want to compare
5. Click "Compare Portfolios" to generate the comparison

### Comparison Views

- **Time Series**: Shows the portfolio value or returns over time for all selected portfolios
- **Bar Chart**: Compares specific metrics like total return, volatility, Sharpe ratio
- **Risk-Return**: Scatter plot showing the risk-return profile of each portfolio
- **Summary Table**: Detailed metrics table for all portfolios

## Portfolio Strategies

The application supports three main portfolio optimization strategies:

### Low Risk Strategy

- Focuses on minimizing portfolio volatility
- Prioritizes assets with lower historical volatility
- Suitable for conservative investors

### Low Turnover Strategy

- Limits trading to a maximum of two trades per month
- Uses momentum-based selection to prioritize trades
- Reduces transaction costs and tax implications

### High Yield Equity Strategy

- Maximizes returns without constraints on volatility or turnover
- Focuses on risk-adjusted momentum for asset selection
- Suitable for aggressive investors seeking high returns

## Tips and Best Practices

1. **Start with Data**: Always ensure you have sufficient market data before constructing portfolios
2. **Compare Strategies**: Use the Comparison tab to evaluate different strategies
3. **Regular Updates**: Refresh market data regularly for accurate portfolio optimization
4. **Strategy Selection**: Choose strategies that align with your investment goals:
   - Low Risk for capital preservation
   - Low Turnover for tax efficiency
   - High Yield Equity for maximum returns
5. **Date Ranges**: Use appropriate training and evaluation periods:
   - Training: Longer periods capture more market cycles
   - Evaluation: Recent periods test strategy performance in current conditions 