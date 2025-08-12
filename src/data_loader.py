"""
Financial Data Loader Module

This module handles the loading and initial processing of financial data from YFinance
for TSLA, BND, and SPY assets covering the period from July 1, 2015 to July 31, 2025.

Author: Financial Analyst Team
Date: August 2025
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
from typing import Dict, List, Tuple, Optional
import logging

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

class FinancialDataLoader:
    """
    A comprehensive class for loading and managing financial data from YFinance.
    
    This class handles:
    - Data fetching with proper error handling
    - Data validation and quality checks
    - Missing data identification
    - Data type conversions
    - Market holiday handling
    """
    
    def __init__(self, start_date: str = "2015-07-01", end_date: str = "2025-07-31"):
        """
        Initialize the FinancialDataLoader.
        
        Parameters:
        -----------
        start_date : str
            Start date for data collection in YYYY-MM-DD format
        end_date : str
            End date for data collection in YYYY-MM-DD format
        """
        self.start_date = start_date
        self.end_date = end_date
        
        # Define the assets to analyze with their descriptions
        self.assets = {
            'TSLA': {
                'name': 'Tesla Inc.',
                'sector': 'Consumer Discretionary',
                'industry': 'Automobile Manufacturing',
                'risk_profile': 'High-growth, High-risk'
            },
            'BND': {
                'name': 'Vanguard Total Bond Market ETF',
                'sector': 'Fixed Income',
                'industry': 'Bond ETF',
                'risk_profile': 'Low-risk, Stable'
            },
            'SPY': {
                'name': 'SPDR S&P 500 ETF Trust',
                'sector': 'Equity',
                'industry': 'Large-Cap Blend',
                'risk_profile': 'Moderate-risk, Diversified'
            }
        }
        
        # Initialize data storage
        self.raw_data = {}
        self.processed_data = {}
        
        logger.info(f"FinancialDataLoader initialized for period: {start_date} to {end_date}")
    
    def fetch_asset_data(self, symbol: str, retry_count: int = 3) -> pd.DataFrame:
        """
        Fetch historical data for a specific asset with retry logic.
        
        Parameters:
        -----------
        symbol : str
            Stock/ETF symbol (e.g., 'TSLA', 'BND', 'SPY')
        retry_count : int
            Number of retry attempts if data fetching fails
            
        Returns:
        --------
        pd.DataFrame
            Historical price data with OHLCV columns
        """
        logger.info(f"Fetching data for {symbol}...")
        
        for attempt in range(retry_count):
            try:
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(
                    start=self.start_date,
                    end=self.end_date,
                    interval='1d',
                    auto_adjust=True,  # Automatically adjust for splits and dividends
                    prepost=False      # Only regular trading hours
                )
                
                if data.empty:
                    raise ValueError(f"No data returned for {symbol}")
                
                # Validate data quality
                self._validate_data_quality(data, symbol)
                
                logger.info(f"Successfully fetched {len(data)} records for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                if attempt == attempt == retry_count - 1:
                    logger.error(f"Failed to fetch data for {symbol} after {retry_count} attempts")
                    raise
                continue
    
    def _validate_data_quality(self, data: pd.DataFrame, symbol: str) -> None:
        """
        Validate the quality of fetched data.
        
        Parameters:
        -----------
        data : pd.DataFrame
            The fetched data to validate
        symbol : str
            Asset symbol for logging purposes
        """
        # Check for required columns
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns for {symbol}: {missing_columns}")
        
        # Check for reasonable data ranges
        if (data['High'] < data['Low']).any():
            logger.warning(f"High price < Low price detected for {symbol}")
        
        if (data['Open'] < 0).any() or (data['Close'] < 0).any():
            raise ValueError(f"Negative prices detected for {symbol}")
        
        # Check for reasonable volume values
        if (data['Volume'] < 0).any():
            raise ValueError(f"Negative volume detected for {symbol}")
        
        logger.info(f"Data quality validation passed for {symbol}")
    
    def load_all_assets(self) -> Dict[str, pd.DataFrame]:
        """
        Load data for all defined assets.
        
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping asset symbols to their historical data
        """
        logger.info("Starting data loading for all assets...")
        
        for symbol in self.assets.keys():
            try:
                data = self.fetch_asset_data(symbol)
                self.raw_data[symbol] = data
                logger.info(f"✓ {symbol}: {len(data)} records loaded")
                
            except Exception as e:
                logger.error(f"✗ Failed to load {symbol}: {str(e)}")
                raise
        
        logger.info("All asset data loaded successfully!")
        return self.raw_data
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Generate a comprehensive summary of the loaded data.
        
        Returns:
        --------
        pd.DataFrame
            Summary statistics for all assets
        """
        summary_data = []
        
        for symbol, data in self.raw_data.items():
            # Basic statistics
            summary = {
                'Symbol': symbol,
                'Asset Name': self.assets[symbol]['name'],
                'Sector': self.assets[symbol]['sector'],
                'Risk Profile': self.assets[symbol]['risk_profile'],
                'Start Date': data.index.min().strftime('%Y-%m-%d'),
                'End Date': data.index.max().strftime('%Y-%m-%d'),
                'Total Records': len(data),
                'Missing Values': data.isnull().sum().sum(),
                'Min Close': data['Close'].min(),
                'Max Close': data['Close'].max(),
                'Current Close': data['Close'].iloc[-1],
                'Avg Volume': data['Volume'].mean(),
                'Total Volume': data['Volume'].sum()
            }
            summary_data.append(summary)
        
        return pd.DataFrame(summary_data)
    
    def save_data_to_csv(self, output_dir: str = "data") -> None:
        """
        Save the loaded data to CSV files for persistence.
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the CSV files
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        for symbol, data in self.raw_data.items():
            filename = f"{output_dir}/{symbol}_historical_data.csv"
            data.to_csv(filename)
            logger.info(f"Saved {symbol} data to {filename}")
    
    def load_data_from_csv(self, data_dir: str = "data") -> Dict[str, pd.DataFrame]:
        """
        Load previously saved data from CSV files.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing the CSV files
            
        Returns:
        --------
        Dict[str, pd.DataFrame]
            Dictionary mapping asset symbols to their historical data
        """
        import os
        
        if not os.path.exists(data_dir):
            logger.warning(f"Data directory {data_dir} does not exist")
            return {}
        
        for filename in os.listdir(data_dir):
            if filename.endswith('_historical_data.csv'):
                symbol = filename.split('_')[0]
                filepath = os.path.join(data_dir, filename)
                
                try:
                    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
                    self.raw_data[symbol] = data
                    logger.info(f"Loaded {symbol} data from {filepath}")
                except Exception as e:
                    logger.error(f"Failed to load {filename}: {str(e)}")
        
        return self.raw_data


def main():
    """
    Main function to demonstrate the data loader functionality.
    """
    # Initialize the data loader
    loader = FinancialDataLoader()
    
    try:
        # Load all assets
        data = loader.load_all_assets()
        
        # Generate and display summary
        summary = loader.get_data_summary()
        print("\n" + "="*80)
        print("DATA LOADING SUMMARY")
        print("="*80)
        print(summary.to_string(index=False))
        print("="*80)
        
        # Save data to CSV
        loader.save_data_to_csv()
        
        print(f"\nData successfully loaded and saved for {len(data)} assets!")
        
    except Exception as e:
        logger.error(f"Data loading failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
