# Time Series Forecasting for Portfolio Management Optimization

## Project Overview

This project implements advanced time series forecasting and portfolio optimization for GMF Investments, focusing on three key assets: TSLA, BND, and SPY. The project demonstrates comprehensive financial data analysis, time series modeling, and Modern Portfolio Theory implementation.

## Project Structure

```
b5w11-Time-Series-Forecasting-for-Portfolio-Management-Optimization/
├── src/                           # Source code modules
│   ├── data_loader.py            # YFinance data fetching and validation
│   ├── preprocessing.py           # Data cleaning and feature engineering
│   ├── financial_metrics.py      # Risk metrics calculation (VaR, Sharpe, etc.)
│   ├── eda.py                    # Exploratory data analysis and visualization
│   ├── time_series_models.py     # ARIMA, SARIMA, and LSTM models (Task 2)
│   ├── portfolio_optimization.py # MPT and efficient frontier (Task 4)
│   └── backtesting.py            # Strategy validation (Task 5)
├── notebooks/                     # Jupyter notebooks
│   ├── task1_analysis.ipynb      # Task 1: Data loading, preprocessing, EDA
│   ├── task2_forecasting.ipynb   # Task 2: Time series modeling (future)
│   ├── task3_trends.ipynb        # Task 3: Future market trends (future)
│   ├── task4_portfolio.ipynb     # Task 4: Portfolio optimization (future)
│   └── task5_backtesting.ipynb   # Task 5: Strategy backtesting (future)
├── data/                         # Data storage
├── results/                      # Generated plots and reports
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Features

### Task 1: Data Loading and Preprocessing ✅
- **Efficient Data Loading**: YFinance integration with retry logic and error handling
- **Advanced Preprocessing**: Missing value handling, feature engineering, data validation
- **Comprehensive EDA**: Price analysis, return distributions, volatility analysis
- **Financial Metrics**: VaR, Sharpe Ratio, Maximum Drawdown, stationarity tests

### Task 2: Time Series Forecasting (Future)
- ARIMA/SARIMA models for statistical forecasting
- LSTM neural networks for deep learning approach
- Model comparison and validation

### Task 3: Market Trend Analysis (Future)
- Future price predictions with confidence intervals
- Trend analysis and risk assessment

### Task 4: Portfolio Optimization (Future)
- Modern Portfolio Theory implementation
- Efficient frontier generation
- Optimal portfolio weights calculation

### Task 5: Strategy Backtesting (Future)
- Historical strategy validation
- Performance comparison with benchmarks

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd b5w11-Time-Series-Forecasting-for-Portfolio-Management-Optimization
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**:
   ```bash
   python -c "import yfinance, pandas, numpy, matplotlib, seaborn; print('All packages installed successfully!')"
   ```

## Usage

### Quick Start (Task 1)

1. **Run the main analysis notebook**:
   ```bash
   cd notebooks
   jupyter notebook task1_analysis.ipynb
   ```

2. **Or run the data loader directly**:
   ```bash
   cd src
   python data_loader.py
   ```

### Data Sources

- **TSLA**: Tesla Inc. - High-growth, high-risk stock
- **BND**: Vanguard Total Bond Market ETF - Low-risk, stable
- **SPY**: SPDR S&P 500 ETF - Moderate-risk, diversified

### Key Metrics Calculated

- **Risk Measures**: VaR (95%, 99%), Maximum Drawdown, Volatility
- **Return Metrics**: Sharpe Ratio, Sortino Ratio, Cumulative Returns
- **Statistical Tests**: ADF Stationarity, Normality Tests, Correlation Analysis

## Project Timeline

- **Interim Submission**: Sunday 10 Aug 2025, 20:00 UTC (Task 1)
- **Final Submission**: Tuesday 12 Aug 2025, 20:00 UTC (All Tasks)

## Technical Details

### Data Processing
- **Time Period**: July 1, 2015 - July 31, 2025
- **Frequency**: Daily data
- **Features**: OHLCV + 30+ engineered features
- **Missing Data**: Advanced imputation strategies

### Model Architecture
- **Statistical Models**: ARIMA, SARIMA with auto-parameter optimization
- **Deep Learning**: LSTM with attention mechanisms
- **Validation**: Time series cross-validation, out-of-sample testing

### Portfolio Optimization
- **Framework**: Modern Portfolio Theory (MPT)
- **Risk Metrics**: Variance, VaR, Conditional VaR
- **Constraints**: Long-only, budget constraints, sector limits

## Contributing

This project is developed for the 10 Academy KAIM program. For questions or collaboration, use the `#all-week11` tag.

## License

This project is for educational purposes as part of the 10 Academy curriculum.

## Acknowledgments

- **Data Source**: YFinance (Yahoo Finance API)

## Contact

For project-related questions, please refer to the course materials and use the designated communication channels.

