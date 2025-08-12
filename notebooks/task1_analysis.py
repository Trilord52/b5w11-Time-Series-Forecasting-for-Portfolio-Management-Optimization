#!/usr/bin/env python3
"""
Task 1 Analysis: Time Series Forecasting for Portfolio Management Optimization
Guide Me in Finance (GMF) Investments

This script demonstrates the complete Task 1 workflow including:
- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Financial metrics calculations (VaR, Sharpe Ratio, etc.)
- Data visualization and insights

Author: GMF Investment Team
Date: August 2025
"""

import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Standard libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from data_loader import FinancialDataLoader
from preprocessing import FinancialDataPreprocessor
from financial_metrics import FinancialMetricsCalculator
from eda import FinancialEDA

def setup_plotting_style():
    """Set up consistent plotting style for all visualizations."""
    plt.style.use('ggplot')
    sns.set_style('whitegrid')
    plt.rcParams['figure.figsize'] = (14, 8)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['ytick.labelsize'] = 12
    print("Plotting style configured successfully.")

def main():
    """Main execution function for Task 1 analysis."""
    print("=" * 80)
    print("TASK 1: TIME SERIES FORECASTING FOR PORTFOLIO MANAGEMENT OPTIMIZATION")
    print("Guide Me in Finance (GMF) Investments")
    print("=" * 80)
    
    # Step 1: Setup and Configuration
    print("\n1. SETTING UP ANALYSIS ENVIRONMENT")
    print("-" * 50)
    setup_plotting_style()
    
    # Define analysis parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    risk_free_rate = 0.02  # 2% annual risk-free rate
    confidence_level = 0.95
    
    print(f"Analysis Period: {start_date} to {end_date}")
    print(f"Risk-free Rate: {risk_free_rate:.1%}")
    print(f"Confidence Level: {confidence_level:.0%}")
    
    # Step 2: Data Loading
    print("\n2. LOADING FINANCIAL DATA")
    print("-" * 50)
    
    try:
        # Initialize data loader
        data_loader = FinancialDataLoader(start_date, end_date)
        
        # Load all assets
        print("Fetching data for TSLA, BND, and SPY...")
        asset_data = data_loader.load_all_assets()
        
        if asset_data is None:
            print("ERROR: Failed to load asset data. Exiting.")
            return
        
        # Display data summary
        print("\nData Loading Summary:")
        data_summary = data_loader.get_data_summary()
        print(data_summary)
        
        # Save raw data
        data_loader.save_data_to_csv('data/raw_asset_data.csv')
        print("Raw data saved to data/raw_asset_data.csv")
        
    except Exception as e:
        print(f"ERROR in data loading: {str(e)}")
        return
    
    # Step 3: Data Preprocessing
    print("\n3. PREPROCESSING FINANCIAL DATA")
    print("-" * 50)
    
    try:
        # Initialize preprocessor
        preprocessor = FinancialDataPreprocessor(risk_free_rate)
        
        # Preprocess each asset
        processed_data = {}
        for asset_name, asset_df in asset_data.items():
            print(f"Preprocessing {asset_name}...")
            processed_asset = preprocessor.preprocess_asset_data(asset_df, asset_name)
            processed_data[asset_name] = processed_asset
            
            # Display preprocessing summary
            preprocess_summary = preprocessor.get_preprocessing_summary(processed_asset, asset_name)
            print(f"  - {asset_name}: {len(processed_asset)} rows, {len(processed_asset.columns)} features")
        
        # Save processed data
        for asset_name, asset_df in processed_data.items():
            preprocessor.save_processed_data(asset_df, f'data/processed_{asset_name.lower()}_data.csv')
        
        print("All assets preprocessed and saved successfully.")
        
    except Exception as e:
        print(f"ERROR in data preprocessing: {str(e)}")
        return
    
    # Step 4: Financial Metrics Calculation
    print("\n4. CALCULATING FINANCIAL METRICS")
    print("-" * 50)
    
    try:
        # Initialize metrics calculator
        metrics_calc = FinancialMetricsCalculator(risk_free_rate, confidence_level)
        
        # Calculate all metrics for each asset
        all_metrics = {}
        for asset_name, asset_df in processed_data.items():
            print(f"Calculating metrics for {asset_name}...")
            
            # Calculate returns for metrics
            returns = asset_df['Daily_Return'].dropna()
            
            # Calculate key metrics
            var_historical = metrics_calc.calculate_var(returns, method='historical')
            var_parametric = metrics_calc.calculate_var(returns, method='parametric')
            sharpe_ratio = metrics_calc.calculate_sharpe_ratio(returns)
            max_drawdown = metrics_calc.calculate_maximum_drawdown(returns)
            sortino_ratio = metrics_calc.calculate_sortino_ratio(returns)
            
            # Test stationarity
            adf_result = metrics_calc.test_stationarity(returns, test_type='adf')
            kpss_result = metrics_calc.test_stationarity(returns, test_type='kpss')
            
            # Store results
            all_metrics[asset_name] = {
                'VaR_Historical': var_historical,
                'VaR_Parametric': var_parametric,
                'Sharpe_Ratio': sharpe_ratio,
                'Max_Drawdown': max_drawdown['max_drawdown'],
                'Sortino_Ratio': sortino_ratio,
                'ADF_Statistic': adf_result['statistic'],
                'ADF_p_value': adf_result['p_value'],
                'KPSS_Statistic': kpss_result['statistic'],
                'KPSS_p_value': kpss_result['p_value']
            }
            
            print(f"  - {asset_name}: VaR={var_historical:.4f}, Sharpe={sharpe_ratio:.4f}")
        
        # Create metrics summary DataFrame
        metrics_df = pd.DataFrame(all_metrics).T
        print("\nFinancial Metrics Summary:")
        print(metrics_df.round(4))
        
        # Save metrics
        metrics_df.to_csv('results/financial_metrics_summary.csv')
        print("Financial metrics saved to results/financial_metrics_summary.csv")
        
    except Exception as e:
        print(f"ERROR in metrics calculation: {str(e)}")
        return
    
    # Step 5: Exploratory Data Analysis
    print("\n5. EXPLORATORY DATA ANALYSIS (EDA)")
    print("-" * 50)
    
    try:
        # Initialize EDA module
        eda_analyzer = FinancialEDA()
        
        # Create comprehensive EDA plots
        print("Generating EDA visualizations...")
        
        # Price analysis plots
        eda_analyzer.create_price_analysis_plots(processed_data, save_path='results/')
        print("  - Price analysis plots created")
        
        # Return distribution plots
        eda_analyzer.create_return_distribution_plots(processed_data, save_path='results/')
        print("  - Return distribution plots created")
        
        # Correlation analysis
        eda_analyzer.create_correlation_analysis(processed_data, save_path='results/')
        print("  - Correlation analysis created")
        
        # Risk metrics summary
        eda_analyzer.create_risk_metrics_summary(processed_data, save_path='results/')
        print("  - Risk metrics summary created")
        
        # Generate comprehensive EDA report
        eda_report = eda_analyzer.generate_eda_report(processed_data, all_metrics)
        
        # Save EDA report
        with open('results/eda_report.txt', 'w') as f:
            f.write(eda_report)
        print("  - EDA report saved to results/eda_report.txt")
        
        print("All EDA visualizations and reports generated successfully.")
        
    except Exception as e:
        print(f"ERROR in EDA generation: {str(e)}")
        return
    
    # Step 6: Portfolio Analysis
    print("\n6. PORTFOLIO ANALYSIS")
    print("-" * 50)
    
    try:
        # Prepare data for portfolio analysis
        print("Preparing portfolio analysis...")
        
        # Create returns matrix for all assets
        returns_matrix = pd.DataFrame()
        for asset_name, asset_df in processed_data.items():
            returns_matrix[asset_name] = asset_df['Daily_Return']
        
        returns_matrix = returns_matrix.dropna()
        
        # Calculate portfolio statistics
        portfolio_stats = {
            'Total_Assets': len(returns_matrix.columns),
            'Total_Observations': len(returns_matrix),
            'Date_Range': f"{returns_matrix.index[0].strftime('%Y-%m-%d')} to {returns_matrix.index[-1].strftime('%Y-%m-%d')}",
            'Annualized_Returns': returns_matrix.mean() * 252,
            'Annualized_Volatility': returns_matrix.std() * np.sqrt(252),
            'Correlation_Matrix': returns_matrix.corr()
        }
        
        # Calculate portfolio-level metrics
        portfolio_returns = returns_matrix.mean(axis=1)
        portfolio_var = metrics_calc.calculate_var(portfolio_returns, method='historical')
        portfolio_sharpe = metrics_calc.calculate_sharpe_ratio(portfolio_returns)
        
        portfolio_stats.update({
            'Portfolio_VaR': portfolio_var,
            'Portfolio_Sharpe': portfolio_sharpe
        })
        
        print("Portfolio Analysis Summary:")
        print(f"  - Total Assets: {portfolio_stats['Total_Assets']}")
        print(f"  - Total Observations: {portfolio_stats['Total_Observations']}")
        print(f"  - Portfolio VaR: {portfolio_var:.4f}")
        print(f"  - Portfolio Sharpe Ratio: {portfolio_sharpe:.4f}")
        
        # Save portfolio analysis
        portfolio_summary = pd.DataFrame({
            'Metric': ['Total_Assets', 'Total_Observations', 'Portfolio_VaR', 'Portfolio_Sharpe'],
            'Value': [portfolio_stats['Total_Assets'], portfolio_stats['Total_Observations'], 
                     portfolio_stats['Portfolio_VaR'], portfolio_stats['Portfolio_Sharpe']]
        })
        portfolio_summary.to_csv('results/portfolio_summary.csv', index=False)
        
        # Save correlation matrix
        portfolio_stats['Correlation_Matrix'].to_csv('results/asset_correlation_matrix.csv')
        
        print("Portfolio analysis saved to results/")
        
    except Exception as e:
        print(f"ERROR in portfolio analysis: {str(e)}")
        return
    
    # Step 7: Summary and Next Steps
    print("\n7. TASK 1 COMPLETION SUMMARY")
    print("-" * 50)
    
    print("✅ Task 1 completed successfully!")
    print("\nWhat was accomplished:")
    print("  - Data loaded for TSLA, BND, and SPY (2020-2024)")
    print("  - Advanced preprocessing with feature engineering")
    print("  - Comprehensive financial metrics calculation")
    print("  - Detailed EDA with visualizations")
    print("  - Portfolio-level analysis")
    print("\nFiles generated:")
    print("  - data/raw_asset_data.csv")
    print("  - data/processed_*.csv (for each asset)")
    print("  - results/financial_metrics_summary.csv")
    print("  - results/eda_report.txt")
    print("  - results/portfolio_summary.csv")
    print("  - results/asset_correlation_matrix.csv")
    print("  - Multiple visualization plots in results/")
    
    print("\nNext steps for Task 2:")
    print("  - Implement ARIMA/SARIMA models")
    print("  - Develop LSTM neural networks")
    print("  - Model evaluation and comparison")
    print("  - Forecasting implementation")
    
    print("\n" + "=" * 80)
    print("TASK 1 ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()

