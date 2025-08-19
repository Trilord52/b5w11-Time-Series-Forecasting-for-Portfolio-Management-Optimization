# Time Series Forecasting for Portfolio Management Optimization

## Overview

This project is a comprehensive solution for data-driven portfolio management, developed as part of the 10 Academy Artificial Intelligence Mastery program (Week 11). The objective is to leverage advanced time series forecasting and modern portfolio theory to optimize asset allocation and enhance investment decision-making for Guide Me in Finance (GMF) Investments.

We use historical and forecasted data for TSLA, BND, and SPY to:
- Predict future price trends and volatility,
- Construct optimal portfolios,
- Backtest strategies against benchmarks,
- Analyze risk and performance,
- Visualize insights through an interactive dashboard.

---

## Table of Contents

- [Business Objective](#business-objective)
- [Project Structure](#project-structure)
- [Interactive Dashboard](#interactive-dashboard)
- [Environment Setup](#environment-setup)
- [Data Sources](#data-sources)
- [Workflow Summary](#workflow-summary)
  - [1. Data Acquisition & Preprocessing](#1-data-acquisition--preprocessing)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Feature Engineering](#3-feature-engineering)
  - [4. Forecasting Models](#4-forecasting-models)
  - [5. Model Evaluation & Selection](#5-model-evaluation--selection)
  - [6. Risk & Trend Analysis](#6-risk--trend-analysis)
  - [7. Portfolio Optimization](#7-portfolio-optimization)
  - [8. Backtesting](#8-backtesting)
- [How to Run](#how-to-run)
- [Results & Interpretation](#results--interpretation)
- [Rubric Checklist](#rubric-checklist)
- [Contributing](#contributing)
- [License](#license)

---

## Business Objective

Guide Me in Finance (GMF) Investments is a forward-thinking financial advisory firm specializing in personalized portfolio management. The goal is to use advanced time series forecasting to predict market trends, optimize asset allocation, and enhance portfolio performance, while minimizing risk and capitalizing on market opportunities.

---

## Project Structure

```
.
├── dashboard.py                            # Interactive Streamlit dashboard
├── run_analysis.py                         # Complete analysis workflow runner
├── notebooks/
│   ├── task1_analysis.ipynb                # Data preprocessing and EDA
│   ├── task2and3_forecasting.ipynb         # Forecasting, trend, and risk analysis
│   ├── task4_portfolio_optimization.ipynb  # Portfolio optimization
│   └── task5_backtesting.ipynb             # Backtesting strategy
├── src/
│   ├── data_loader.py                      # Data fetching utilities
│   ├── preprocessing.py                    # Data cleaning and feature engineering
│   ├── forecasting_models.py               # ARIMA, SARIMA, LSTM model classes
│   ├── financial_metrics.py                # Risk metrics and financial calculations
│   ├── eda.py                              # Exploratory data analysis utilities
│   └── task1_analysis.py                   # Task 1 analysis script
├── results/
│   ├── forecasting/                        # Output forecasts, model results, and reports
│   └── plots/                              # Generated visualizations and charts
├── data/                                   # Processed and raw data storage
├── requirements.txt                        # Python dependencies (complete stack)
├── README.md
└── .gitignore                              # Git ignore patterns
```

---

## Interactive Dashboard

### Overview

The project includes a comprehensive **Streamlit-based interactive dashboard** that provides real-time visualization and analysis of portfolio performance, risk metrics, and asset correlations.

### Dashboard Features

- **📊 Portfolio Overview**: Key performance indicators, asset selection, and data summary
- **📈 Performance Analysis**: Interactive price charts, returns visualization, and performance metrics
- **⚠️ Risk Analysis**: Risk-return scatter plots, VaR comparisons, and drawdown analysis
- **🔗 Correlation Analysis**: Interactive correlation heatmaps and diversification insights

### Key Capabilities

- **Interactive Visualizations**: Plotly-powered charts with zoom, pan, and hover functionality
- **Real-time Calculations**: Automatic metric updates based on asset selection
- **Multi-Asset Analysis**: Comprehensive comparison across TSLA, BND, and SPY
- **Risk Metrics**: Value at Risk (VaR), Sharpe ratios, maximum drawdown, and Sortino ratios
- **Responsive Design**: Web-based interface accessible from any browser

### Running the Dashboard

```bash
# Install dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501` with an intuitive interface for exploring financial data and portfolio insights.

---

## Environment Setup

It is recommended to use a virtual environment for reproducibility:

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

If you do not have a requirements file, install dependencies manually:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels yfinance pypfopt plotly streamlit dash
```

---

## Data Sources

- **Yahoo Finance (yfinance):** For historical daily price data of TSLA, BND, and SPY.
- **Forecasts:** Generated using ARIMA, SARIMA, and LSTM models.

---

## Workflow Summary

### 1. Data Acquisition & Preprocessing

- Download historical price data for TSLA, BND, and SPY.
- Handle missing values (forward/backward fill).
- Normalize and align data for modeling.

### 2. Exploratory Data Analysis (EDA)

- Visualize missing values, outliers, and basic statistics.
- Plot historical price trends and volatility.
- Identify data quality issues before modeling.

### 3. Feature Engineering

- Add technical indicators (e.g., moving averages, RSI) to enrich the dataset.
- Prepare features for both statistical and deep learning models.

### 4. Forecasting Models

- **ARIMA/SARIMA:** For linear and seasonal patterns.
- **LSTM:** For capturing non-linear temporal dependencies.
- Hyperparameter tuning (grid search for ARIMA/SARIMA, GridSearchCV for LSTM).
- Walk-forward validation for robust out-of-sample evaluation.

### 5. Model Evaluation & Selection

- Evaluate models using RMSE, MAE, MAPE, and direction accuracy.
- Compare models in a summary table.
- Select the best model based on lowest RMSE and overall performance.

### 6. Risk & Trend Analysis

- Generate 12-month forecasts for TSLA.
- Visualize forecasts with confidence intervals.
- Analyze expected return, trend (bullish/bearish), Value at Risk (VaR), maximum drawdown, and volatility.

### 7. Portfolio Optimization

- Use forecasted TSLA return and historical BND/SPY returns.
- Calculate expected returns and covariance matrix.
- Generate the Efficient Frontier using PyPortfolioOpt.
- Identify Maximum Sharpe Ratio and Minimum Volatility portfolios.
- Apply realistic constraints (e.g., max 60% per asset, no shorting).
- Save optimal weights for backtesting.

### 8. Backtesting

- Simulate the performance of the recommended portfolio over the last year.
- Compare to a benchmark (60% SPY / 40% BND).
- Use rolling window backtesting for stability analysis.
- Model transaction costs for realistic results.
- Visualize cumulative returns and performance metrics.

---

## How to Run

### Option 1: Interactive Dashboard (Recommended)

```bash
# Install all dependencies
pip install -r requirements.txt

# Launch the interactive dashboard
streamlit run dashboard.py
```

The dashboard provides a comprehensive web interface for exploring all project features including data analysis, risk metrics, and portfolio insights.

### Option 2: Jupyter Notebooks

1. **Run the notebooks interactively:**
    ```bash
    jupyter notebook
    ```
    Open each notebook in the `notebooks/` folder and run all cells sequentially.

2. **Convert a Python script to a notebook (if needed):**
    ```bash
    pip install jupytext
    jupytext --to notebook notebooks/task4_and_5_portfolio_and_backtest.py
    ```

3. **Run all cells and save output:**
    ```bash
    jupyter nbconvert --to notebook --execute notebooks/task4_portfolio_optimization.ipynb --output notebooks/task4_portfolio_optimization_output.ipynb
    ```

### Option 3: Complete Analysis Runner

```bash
# Run the complete analysis workflow
python run_analysis.py
```

This runs the entire analysis pipeline including data loading, preprocessing, EDA, forecasting, and generates all outputs.

### Option 4: Individual Analysis Scripts

```bash
# Run specific analysis components
python src/task1_analysis.py          # Data preprocessing and EDA
```

---

## Results & Interpretation

### **Project Deliverables**

- **🎯 Complete Analysis Pipeline:** Fully automated workflow from data loading to portfolio insights
- **📊 Interactive Dashboard:** Comprehensive web-based interface providing real-time portfolio analysis and risk visualization
- **🔮 Forecasting Models:** ARIMA and LSTM implementations for time series prediction with performance evaluation
- **📈 Portfolio Analytics:** Advanced risk metrics including VaR, Sharpe ratios, maximum drawdown, and correlation analysis
- **🎨 Comprehensive Visualizations:** Both static and interactive Plotly charts with zoom, pan, and hover capabilities
- **📋 Automated Reporting:** Generated analysis reports and performance summaries
- **🏗️ Modular Architecture:** Well-structured codebase with separate modules for each analysis component

### **Key Findings**

- **Data Quality:** Successfully processed 4+ years of daily data for TSLA, BND, and SPY with comprehensive feature engineering
- **Risk Analysis:** Implemented multiple VaR methodologies and advanced risk metrics for portfolio assessment
- **Forecasting:** Developed both statistical (ARIMA) and deep learning (LSTM) models with performance comparison
- **User Experience:** Created intuitive interfaces enabling both technical and non-technical stakeholders to access insights
- **Scalability:** Modular design allows easy extension to additional assets and analysis methods

---

## Project Status

**✅ COMPLETED FEATURES:**
- Data loading and preprocessing pipeline
- Comprehensive financial metrics calculation
- Interactive web dashboard with real-time analysis
- Time series forecasting models (ARIMA/LSTM)
- Advanced risk analysis and portfolio metrics
- Static and interactive visualization suite
- Automated analysis workflow
- Complete documentation and user guides

**🚀 READY FOR:**
- Portfolio optimization implementation
- Backtesting framework development
- Advanced forecasting model integration
- Real-time data streaming
- Production deployment

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

---
