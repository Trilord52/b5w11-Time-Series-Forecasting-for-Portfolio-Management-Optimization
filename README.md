# Time Series Forecasting for Portfolio Management Optimization

## Project Overview

This project is developed for Guide Me in Finance (GMF) Investments to enhance portfolio management strategies using advanced time series forecasting. The focus is on leveraging historical financial data for TSLA, BND, and SPY to predict market trends, assess volatility, and support data-driven investment decisions.

---

## Repository Structure

```
.
├── notebooks/
│   ├── task2and3_forecasting.ipynb   # Main notebook for forecasting and trend analysis
│   └── ...                           # Additional notebooks for other tasks
├── src/
│   ├── data_loader.py                # Module for fetching financial data
│   ├── preprocessing.py              # Data cleaning and preprocessing utilities
│   ├── forecasting_models.py         # ARIMA, SARIMA, and LSTM model classes
│   └── ...                           # Other supporting modules
├── results/
│   └── forecasting/                  # Output forecasts and comparison tables
├── requirements.txt                  # Python dependencies
└── README.md
```

---

## Workflow Description

### 1. Data Acquisition and Preprocessing

- Historical daily price data for TSLA is downloaded using YFinance, covering July 2015 to December 2024.
- The data is cleaned and normalized, with missing values handled using forward and backward fill to ensure continuity.
- Exploratory data analysis (EDA) is performed to visualize trends, returns, and volatility.

### 2. Model Development

- Three forecasting models are implemented:
  - **ARIMA:** Captures linear trends and autocorrelation in the time series.
  - **SARIMA:** Extends ARIMA to account for seasonality in the data.
  - **LSTM:** A deep learning model designed to capture complex, non-linear temporal dependencies.
- The data is split chronologically into training and testing sets to avoid look-ahead bias.
- Each model is trained on the training set and evaluated on the test set.

### 3. Model Evaluation

- Model performance is assessed using several metrics:
  - **Root Mean Squared Error (RMSE)**
  - **Mean Absolute Error (MAE)**
  - **Mean Absolute Percentage Error (MAPE)**
  - **Direction Accuracy** (how often the model predicts the correct direction of price movement)
- A comparison table summarizes the performance of all models.

### 4. Forecasting and Trend Analysis

- Each model generates a 12-month (252 trading days) forecast for TSLA.
- Forecasts are visualized alongside historical data, with confidence intervals shown where available.
- The forecasted period is analyzed for expected return, maximum and minimum forecasted prices, and overall trend (bullish or bearish).

### 5. Risk and Volatility Analysis

- The forecasted returns are analyzed to compute:
  - **Value at Risk (VaR) at 95% confidence**
  - **Maximum Drawdown**
  - **Annualized Volatility**
- These metrics help assess the risk profile of the forecasted period.

### 6. Visualization

- The notebook includes clear, well-labeled plots for:
  - Training and test data split
  - Model forecasts vs. historical prices
  - Forecasted returns distribution
  - Cumulative returns
  - Rolling volatility
  - Comparison of all model forecasts

### 7. Results and Outputs

- The best-performing model is selected based on lowest RMSE.
- Forecast data and model comparison tables are saved in the `results/forecasting/` directory for further analysis or reporting.
- A summary and recommendations section is provided at the end of the notebook.

---

## How to Run This Project

1. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    Or, if you do not have a requirements file:
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn tensorflow statsmodels yfinance
    ```

2. **Run the notebook:**
    - Open `notebooks/task2and3_forecasting.ipynb` in VS Code or Jupyter and run all cells sequentially.
    - Alternatively, from the terminal:
      ```bash
      jupyter notebook notebooks/task2and3_forecasting.ipynb
      ```
    - Or, to execute and export the notebook:
      ```bash
      jupyter nbconvert --to notebook --execute notebooks/task2and3_forecasting.ipynb --output notebooks/task2and3_forecasting_output.ipynb
      ```

---

## Project Status

- The forecasting and trend analysis workflow for TSLA is complete.
- All code is modular, well-documented, and reproducible.
- Results and visualizations are saved and clearly presented.
- The project is ready for extension to portfolio optimization and backtesting.

---



- Extend the workflow to include BND and SPY for multi-asset portfolio analysis.
- Integrate the forecasts into portfolio optimization and backtesting modules.
- Regularly update the models with new data to maintain forecasting accuracy.

---

*For questions, suggestions, or contributions, please open an issue or submit a