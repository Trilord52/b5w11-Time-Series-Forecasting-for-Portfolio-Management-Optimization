"""
Financial Data Preprocessing Module

This module handles data cleaning, feature engineering, and preparation for time series analysis.
It includes advanced handling of missing data, calculation of financial metrics, and data validation.

Author: Financial Analyst Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import warnings

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class FinancialDataPreprocessor:
    """
    A comprehensive class for preprocessing financial data.
    
    This class handles:
    - Missing data imputation with financial logic
    - Feature engineering for financial analysis
    - Data quality checks and validation
    - Time series specific preprocessing
    - Market calendar adjustments
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize the FinancialDataPreprocessor.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate (default: 2% for US Treasury)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1  # Daily equivalent
        
        # Define market holidays (major US market holidays)
        self.market_holidays = [
            '2025-01-01', '2025-01-20', '2025-02-17', '2025-04-18',
            '2025-05-26', '2025-06-19', '2025-07-04', '2025-09-01',
            '2025-11-27', '2025-12-25'
        ]
        
        logger.info(f"FinancialDataPreprocessor initialized with risk-free rate: {risk_free_rate:.2%}")
    
    def preprocess_asset_data(self, raw_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Preprocess all asset data with comprehensive cleaning and feature engineering.
        
        Parameters:
        -----------
        raw_data : Dict[str, pd.DataFrame]
            Dictionary of raw asset data
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary of preprocessed asset data
        """
        logger.info("Starting comprehensive data preprocessing...")
        
        processed_data = {}
        
        for symbol, data in raw_data.items():
            logger.info(f"Preprocessing {symbol}...")
            
            try:
                # Deep copy to avoid modifying original data
                asset_data = data.copy()
                
                # Step 1: Basic data cleaning
                asset_data = self._clean_basic_data(asset_data, symbol)
                
                # Step 2: Handle missing values with advanced techniques
                asset_data = self._handle_missing_values(asset_data, symbol)
                
                # Step 3: Feature engineering
                asset_data = self._engineer_features(asset_data, symbol)
                
                # Step 4: Data validation
                asset_data = self._validate_processed_data(asset_data, symbol)
                
                processed_data[symbol] = asset_data
                logger.info(f"✓ {symbol} preprocessing completed successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to preprocess {symbol}: {str(e)}")
                raise
        
        logger.info("All asset preprocessing completed successfully!")
        return processed_data
    
    def _clean_basic_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Perform basic data cleaning operations.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Raw asset data
        symbol : str
            Asset symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Cleaned data
        """
        logger.info(f"  - Basic cleaning for {symbol}")
        
        # Remove any rows with all NaN values
        initial_rows = len(data)
        data = data.dropna(how='all')
        
        if len(data) < initial_rows:
            logger.info(f"    Removed {initial_rows - len(data)} completely empty rows")
        
        # Ensure proper data types
        numeric_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in numeric_columns:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Sort by date to ensure chronological order
        data = data.sort_index()
        
        # Remove any duplicate dates (keep last occurrence)
        duplicates = data.index.duplicated(keep='last')
        if duplicates.any():
            logger.warning(f"    Found {duplicates.sum()} duplicate dates in {symbol}, keeping last occurrence")
            data = data[~duplicates]
        
        return data
    
    def _handle_missing_values(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Handle missing values using advanced financial techniques.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Asset data with potential missing values
        symbol : str
            Asset symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Data with missing values handled
        """
        logger.info(f"  - Handling missing values for {symbol}")
        
        # Check for missing values
        missing_summary = data.isnull().sum()
        if missing_summary.sum() == 0:
            logger.info(f"    No missing values found in {symbol}")
            return data
        
        logger.info(f"    Missing values found: {missing_summary.to_dict()}")
        
        # Handle missing values column by column with appropriate strategies
        for column in data.columns:
            if data[column].isnull().sum() > 0:
                missing_count = data[column].isnull().sum()
                logger.info(f"    Handling {missing_count} missing values in {column}")
                
                if column in ['Open', 'High', 'Low', 'Close']:
                    # For price data, use forward fill then backward fill
                    # This preserves the last known price for market holidays/weekends
                    data[column] = data[column].fillna(method='ffill').fillna(method='bfill')
                    
                elif column == 'Volume':
                    # For volume, use 0 for missing values (no trading)
                    data[column] = data[column].fillna(0)
                    
                else:
                    # For other columns, use forward fill
                    data[column] = data[column].fillna(method='ffill')
        
        # Final check for any remaining missing values
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"    {remaining_missing} missing values remain after handling")
        else:
            logger.info(f"    All missing values successfully handled for {symbol}")
        
        return data
    
    def _engineer_features(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Engineer financial features for analysis.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Clean asset data
        symbol : str
            Asset symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Data with engineered features
        """
        logger.info(f"  - Engineering features for {symbol}")
        
        # 1. Price-based features
        data['Price_Range'] = data['High'] - data['Low']
        data['Price_Range_Pct'] = data['Price_Range'] / data['Close'] * 100
        
        # 2. Returns (percentage changes)
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # 3. Volatility features (rolling)
        for window in [5, 10, 20, 60]:
            data[f'Volatility_{window}d'] = data['Daily_Return'].rolling(window=window).std() * np.sqrt(252)
            data[f'Rolling_Mean_{window}d'] = data['Close'].rolling(window=window).mean()
            data[f'Rolling_Std_{window}d'] = data['Close'].rolling(window=window).std()
        
        # 4. Volume features
        data['Volume_MA_20'] = data['Volume'].rolling(window=20).mean()
        data['Volume_Ratio'] = data['Volume'] / data['Volume_MA_20']
        
        # 5. Technical indicators
        # Moving averages
        for period in [5, 10, 20, 50, 200]:
            data[f'MA_{period}'] = data['Close'].rolling(window=period).mean()
        
        # RSI (Relative Strength Index)
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        bb_upper, bb_lower = self._calculate_bollinger_bands(data['Close'])
        data['BB_Upper'] = bb_upper
        data['BB_Lower'] = bb_lower
        data['BB_Width'] = (bb_upper - bb_lower) / data['Close'] * 100
        
        # 6. Risk metrics
        data['VaR_95_20d'] = data['Daily_Return'].rolling(window=20).quantile(0.05)
        data['CVaR_95_20d'] = data['Daily_Return'].rolling(window=20).apply(
            lambda x: x[x <= x.quantile(0.05)].mean()
        )
        
        # 7. Market regime indicators
        data['Trend_Indicator'] = np.where(data['Close'] > data['MA_50'], 1, -1)
        data['Volatility_Regime'] = np.where(data['Volatility_20d'] > data['Volatility_20d'].rolling(60).mean(), 'High', 'Low')
        
        logger.info(f"    ✓ {len([col for col in data.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume']])} features engineered")
        
        return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            RSI period (default: 14)
            
        Returns:
        --------
        pd.Series
            RSI values
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series
        period : int
            Moving average period (default: 20)
        std_dev : float
            Standard deviation multiplier (default: 2)
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            Upper and lower Bollinger Bands
        """
        ma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper_band = ma + (std * std_dev)
        lower_band = ma - (std * std_dev)
        return upper_band, lower_band
    
    def _validate_processed_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Validate the processed data for quality and consistency.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Processed asset data
        symbol : str
            Asset symbol for logging
            
        Returns:
        --------
        pd.DataFrame
            Validated data
        """
        logger.info(f"  - Validating processed data for {symbol}")
        
        # Check for infinite values
        inf_count = np.isinf(data.select_dtypes(include=[np.number])).sum().sum()
        if inf_count > 0:
            logger.warning(f"    Found {inf_count} infinite values, replacing with NaN")
            data = data.replace([np.inf, -np.inf], np.nan)
        
        # Check for extreme outliers in returns (beyond 3 standard deviations)
        returns = data['Daily_Return'].dropna()
        if len(returns) > 0:
            mean_return = returns.mean()
            std_return = returns.std()
            extreme_outliers = returns[(returns < mean_return - 3*std_return) | 
                                     (returns > mean_return + 3*std_return)]
            
            if len(extreme_outliers) > 0:
                logger.info(f"    Found {len(extreme_outliers)} extreme return outliers (beyond 3σ)")
                logger.info(f"    Extreme returns range: {extreme_outliers.min():.4f} to {extreme_outliers.max():.4f}")
        
        # Check data consistency
        if len(data) < 100:
            logger.warning(f"    {symbol} has only {len(data)} data points, which may be insufficient for analysis")
        
        # Final data quality summary
        total_features = len(data.columns)
        total_observations = len(data)
        missing_values = data.isnull().sum().sum()
        
        logger.info(f"    ✓ Validation complete: {total_features} features, {total_observations} observations, {missing_values} missing values")
        
        return data
    
    def get_preprocessing_summary(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate a summary of the preprocessing results.
        
        Parameters:
        -----------
        processed_data : Dict[str, pd.DataFrame]
            Dictionary of processed asset data
            
        Returns:
        --------
        pd.DataFrame
            Preprocessing summary
        """
        summary_data = []
        
        for symbol, data in processed_data.items():
            # Count engineered features
            base_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            engineered_features = len([col for col in data.columns if col not in base_columns])
            
            # Data quality metrics
            missing_values = data.isnull().sum().sum()
            total_values = data.size
            data_quality = (1 - missing_values / total_values) * 100
            
            summary = {
                'Symbol': symbol,
                'Total_Features': len(data.columns),
                'Engineered_Features': engineered_features,
                'Observations': len(data),
                'Missing_Values': missing_values,
                'Data_Quality_%': round(data_quality, 2),
                'Date_Range': f"{data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}"
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_processed_data(self, processed_data: Dict[str, pd.DataFrame], output_dir: str = "data") -> None:
        """
        Save the processed data to CSV files.
        
        Parameters:
        -----------
        processed_data : Dict[str, pd.DataFrame]
            Dictionary of processed asset data
        output_dir : str
            Directory to save the processed data
        """
        import os
        
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, data in processed_data.items():
            filename = f"{output_dir}/{symbol}_processed_data.csv"
            data.to_csv(filename)
            logger.info(f"Saved processed {symbol} data to {filename}")

def fill_missing(df):
    return df.fillna(method='ffill').fillna(method='bfill')

# filepath: tests/test_preprocessing.py
import pandas as pd
from preprocessing import fill_missing

def test_fill_missing():
    df = pd.DataFrame({'A': [1, None, 3]})
    filled = fill_missing(df)
    assert filled.isnull().sum().sum() == 0

def main():
    """
    Main function to demonstrate the preprocessor functionality.
    """
    # This would typically be called after data loading
    print("FinancialDataPreprocessor module loaded successfully!")
    print("Use this module to preprocess financial data after loading from YFinance.")


if __name__ == "__main__":
    main()
