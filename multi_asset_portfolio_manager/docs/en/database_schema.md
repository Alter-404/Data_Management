# Database Schema Documentation

This document provides a comprehensive overview of the database schema used in the Multi-Asset Portfolio Manager application. The application uses SQLite as its database management system.

## Overview

The database consists of multiple tables that store information about products (assets), clients, portfolio managers, portfolios, deals (trades), portfolio holdings, and performance metrics. The tables are designed to maintain referential integrity through foreign key relationships.

## Table Definitions

### Products (formerly Assets)

Stores information about the financial products available for trading in the system.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the product |
| symbol | TEXT | NOT NULL | Ticker symbol for the product |
| name | TEXT | | Full name of the product |
| type | TEXT | | Type of product (e.g., Stock, Bond, ETF) |
| region | TEXT | | Geographic region of the product |
| sector | TEXT | | Industry sector of the product |
| is_active | INTEGER | DEFAULT 1 | Whether the product is active and available for trading |

### Managers (formerly PortfolioManagers)

Stores information about portfolio managers who manage client portfolios.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the manager |
| name | TEXT | NOT NULL | Name of the portfolio manager |
| email | TEXT | | Email address of the portfolio manager |
| phone | TEXT | | Phone number of the portfolio manager |
| date_joined | TEXT | | Date when the manager joined |

### Clients

Stores information about clients who own portfolios.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the client |
| name | TEXT | NOT NULL | Name of the client |
| risk_profile | TEXT | | Risk tolerance level of the client |
| email | TEXT | | Email address of the client |
| phone | TEXT | | Phone number of the client |

### Portfolios

Stores information about investment portfolios managed within the system.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the portfolio |
| client_id | INTEGER | FOREIGN KEY | Reference to the client who owns the portfolio |
| manager_id | INTEGER | FOREIGN KEY | Reference to the manager who manages the portfolio |
| name | TEXT | NOT NULL | Name of the portfolio |
| strategy | TEXT | | Investment strategy applied to the portfolio |
| asset_universe | TEXT | | Universe of assets available for this portfolio |
| creation_date | TEXT | | Date when the portfolio was created |
| last_updated | TEXT | | Date when the portfolio was last updated |
| cash_balance | REAL | DEFAULT 0.0 | Current cash balance in the portfolio |

### Deals (formerly Trades)

Stores information about all trades executed in the system.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the deal |
| portfolio_id | INTEGER | FOREIGN KEY | Reference to the portfolio where the deal was executed |
| product_id | INTEGER | FOREIGN KEY | Reference to the product being traded |
| date | TEXT | NOT NULL | Date when the deal was executed |
| action | TEXT | NOT NULL | Buy or Sell |
| shares | REAL | NOT NULL | Number of shares traded |
| price | REAL | NOT NULL | Price per share at the time of the deal |
| amount | REAL | NOT NULL | Total monetary value of the deal |
| period | TEXT | | Trading period (e.g., "training", "evaluation") |

### MarketData

Stores historical market data for products.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the market data record |
| product_id | INTEGER | FOREIGN KEY | Reference to the product |
| date | TEXT | NOT NULL | Date of the market data record |
| open | REAL | | Opening price |
| high | REAL | | Highest price during the period |
| low | REAL | | Lowest price during the period |
| close | REAL | | Closing price |
| volume | INTEGER | | Trading volume |
| adjusted_close | REAL | | Adjusted closing price |

### PerformanceMetrics

Stores performance metrics for portfolios over time.

| Column Name | Data Type | Constraints | Description |
|-------------|-----------|-------------|-------------|
| id | INTEGER | PRIMARY KEY | Unique identifier for the performance metrics record |
| portfolio_id | INTEGER | FOREIGN KEY | Reference to the portfolio |
| date | TEXT | NOT NULL | Date of the performance record |
| total_value | REAL | | Total portfolio value including cash |
| daily_return | REAL | | Daily return percentage |
| cumulative_return | REAL | | Cumulative return percentage since inception |
| volatility | REAL | | Portfolio volatility (standard deviation of returns) |
| sharpe_ratio | REAL | | Sharpe ratio (risk-adjusted return) |
| max_drawdown | REAL | | Maximum drawdown (largest peak-to-trough drop) |
| winning_days | INTEGER | | Number of days with positive returns |

## Relationships

The following diagram illustrates the relationships between tables:

```
Clients (1) ---> (N) Portfolios (1) ---> (N) Deals
                     |
                     | (1)
                     v
                     (N)
Managers (1) -----> Portfolios (1) ---> (N) PerformanceMetrics
                     |
                     | (N)
                     v
Products (1) -------> (N) Deals
     |
     | (1)
     v
     (N)
MarketData
```

## Example SQL Queries

### Creating a New Portfolio

```sql
INSERT INTO Portfolios (client_id, manager_id, name, strategy, asset_universe, creation_date, cash_balance)
VALUES (1, 2, 'Growth Portfolio', 'High Yield Equity', 'US Equities', '2023-01-15', 100000.0);
```

### Retrieving Portfolio Performance

```sql
SELECT p.name, pm.date, pm.total_value, pm.daily_return, pm.cumulative_return, pm.volatility, pm.sharpe_ratio
FROM Portfolios p
JOIN PerformanceMetrics pm ON p.id = pm.portfolio_id
WHERE p.id = 1
ORDER BY pm.date DESC;
```

### Retrieving All Deals for a Portfolio

```sql
SELECT d.date, pr.symbol, d.action, d.shares, d.price, d.amount
FROM Deals d
JOIN Products pr ON d.product_id = pr.id
WHERE d.portfolio_id = 1
ORDER BY d.date DESC;
```

### Calculating Portfolio Allocation

```sql
SELECT pr.symbol, pr.type, pr.sector,
       SUM(CASE WHEN d.action = 'BUY' THEN d.shares ELSE -d.shares END) as total_shares,
       SUM(CASE WHEN d.action = 'BUY' THEN d.amount ELSE -d.amount END) as total_invested
FROM Deals d
JOIN Products pr ON d.product_id = pr.id
WHERE d.portfolio_id = 1
GROUP BY pr.id
HAVING total_shares > 0;
```

### Getting Portfolio Manager's Portfolios

```sql
SELECT p.id, p.name, c.name as client_name, p.strategy, p.asset_universe, p.cash_balance
FROM Portfolios p
JOIN Clients c ON p.client_id = c.id
WHERE p.manager_id = 1;
```

## Database Initialization

The database is initialized with predefined products, including common stock tickers like AAPL, MSFT, GOOGL, etc. The initialization creates all the necessary tables with appropriate foreign key constraints and populates them with initial data as needed.

## Migrations

When modifying the database schema, it's important to create migration scripts that handle:

1. Creating new tables
2. Adding new columns to existing tables
3. Renaming tables or columns
4. Updating foreign key relationships
5. Migrating data from old schema to new schema

This ensures that existing data is preserved when the schema changes.

## Performance Considerations

For optimal performance, the database includes indexes on:

- `symbol` in the Products table
- `portfolio_id` and `date` in the Deals table
- `product_id` and `date` in the MarketData table
- `portfolio_id` and `date` in the PerformanceMetrics table

These indexes improve query performance for common operations like retrieving historical data, calculating portfolio metrics, and generating reports. 