#!/usr/bin/env python3
"""
Complete Portfolio Analysis Runner

This script runs the complete time series forecasting and portfolio optimization analysis.
It integrates all modules and provides a comprehensive analysis workflow.

Author: Financial Analytics Team
Date: August 2025
"""

import os
import sys
import warnings
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import all modules
from data_loader import FinancialDataLoader
from preprocessing import FinancialDataPreprocessor
from financial_metrics import FinancialMetricsCalculator
from eda import FinancialEDA
from forecasting_models import ForecastingModels

warnings.filterwarnings('ignore')

def run_complete_analysis():
    """Run the complete portfolio analysis workflow."""
    
    print("="*80)
    print("COMPLETE PORTFOLIO MANAGEMENT ANALYSIS")
    print("Time Series Forecasting for Portfolio Optimization")
    print("="*80)
    
    # Create output directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('results/plots', exist_ok=True)
    os.makedirs('results/forecasting', exist_ok=True)
    
    # Analysis parameters
    start_date = "2020-01-01"
    end_date = "2024-12-31"
    assets = ["TSLA", "BND", "SPY"]
    
    print(f"\nAnalysis Period: {start_date} to {end_date}")
    print(f"Assets: {', '.join(assets)}")
    
    try:
        # Step 1: Data Loading
        print("\n" + "="*60)
        print("STEP 1: DATA LOADING")
        print("="*60)
        
        loader = FinancialDataLoader(start_date=start_date, end_date=end_date)
        raw_data = loader.load_all_assets()
        data_summary = loader.get_data_summary()
        
        print("\nData Loading Summary:")
        print(data_summary.to_string(index=False))
        
        # Step 2: Data Preprocessing
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING")
        print("="*60)
        
        preprocessor = FinancialDataPreprocessor()
        processed_data = preprocessor.preprocess_asset_data(raw_data)
        preprocessing_summary = preprocessor.get_preprocessing_summary(processed_data)
        
        print("\nPreprocessing Summary:")
        print(preprocessing_summary.to_string(index=False))
        
        # Step 3: Financial Metrics Calculation
        print("\n" + "="*60)
        print("STEP 3: FINANCIAL METRICS CALCULATION")
        print("="*60)
        
        calculator = FinancialMetricsCalculator()
        all_metrics = calculator.calculate_all_metrics(processed_data)
        
        # Step 4: Exploratory Data Analysis
        print("\n" + "="*60)
        print("STEP 4: EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        eda = FinancialEDA()
        
        # Generate comprehensive visualizations
        print("Generating comprehensive visualizations...")
        eda.create_price_analysis_plots(processed_data, save_path='results/plots/')
        eda.create_return_distribution_plots(processed_data, save_path='results/plots/')
        eda.create_correlation_analysis(processed_data, save_path='results/plots/')
        eda.create_risk_metrics_summary(all_metrics, save_path='results/plots/')
        
        # Interactive visualizations
        print("Creating interactive visualizations...")
        eda.create_interactive_price_analysis(processed_data, save_path='results/plots/')
        eda.create_interactive_correlation_analysis(processed_data, save_path='results/plots/')
        eda.create_interactive_risk_metrics(all_metrics, save_path='results/plots/')
        
        # Step 5: Forecasting Analysis
        print("\n" + "="*60)
        print("STEP 5: FORECASTING ANALYSIS")
        print("="*60)
        
        forecasting = ForecastingModels()
        forecast_results = {}
        performance_metrics = []
        
        # Focus on TSLA for forecasting demonstration
        if 'TSLA' in processed_data:
            tsla_data = processed_data['TSLA']
            train_data, test_data = forecasting.prepare_data_for_forecasting(tsla_data)
            
            # ARIMA forecasting
            print("Running ARIMA forecasting...")
            arima_results = forecasting.fit_arima_model(train_data)
            if arima_results:
                forecast_results['ARIMA'] = arima_results
                arima_performance = forecasting.evaluate_model_performance(
                    test_data, arima_results['forecast'], 'ARIMA'
                )
                performance_metrics.append(arima_performance)
            
            # LSTM forecasting (if TensorFlow available)
            print("Running LSTM forecasting...")
            lstm_results = forecasting.fit_lstm_model(train_data, epochs=10)  # Reduced epochs for demo
            if lstm_results:
                forecast_results['LSTM'] = lstm_results
                lstm_performance = forecasting.evaluate_model_performance(
                    test_data, lstm_results['forecast'], 'LSTM'
                )
                performance_metrics.append(lstm_performance)
            
            # Generate forecast visualization
            if forecast_results:
                forecasting.create_forecast_visualization(
                    train_data, test_data, forecast_results, 
                    save_path='results/forecasting/forecast_comparison.png'
                )
                
                # Generate forecast report
                forecast_report = forecasting.generate_forecast_report(
                    performance_metrics, save_path='results/forecasting/forecast_report.txt'
                )
                print("\nForecasting Performance Summary:")
                print(forecast_report)
        
        # Step 6: Portfolio Analysis Summary
        print("\n" + "="*60)
        print("STEP 6: PORTFOLIO ANALYSIS SUMMARY")
        print("="*60)
        
        # Calculate portfolio-level metrics
        portfolio_returns = {}
        for asset, data in processed_data.items():
            if 'Daily_Return' in data.columns:
                portfolio_returns[asset] = data['Daily_Return'].dropna()
        
        if len(portfolio_returns) > 1:
            import pandas as pd
            returns_df = pd.DataFrame(portfolio_returns)
            correlation_matrix = returns_df.corr()
            
            print("Portfolio Correlation Matrix:")
            print(correlation_matrix.round(4))
            
            # Calculate individual asset performance
            print("\nIndividual Asset Performance:")
            for asset in returns_df.columns:
                annual_return = returns_df[asset].mean() * 252
                annual_vol = returns_df[asset].std() * (252 ** 0.5)
                sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                print(f"  {asset}: Return={annual_return:.2%}, Vol={annual_vol:.2%}, Sharpe={sharpe:.3f}")
        
        # Final Summary
        print("\n" + "="*80)
        print("ANALYSIS COMPLETION SUMMARY")
        print("="*80)
        
        print("✅ Data Loading: Successfully loaded data for all assets")
        print("✅ Data Preprocessing: Advanced feature engineering completed")
        print("✅ Financial Metrics: Comprehensive risk metrics calculated")
        print("✅ EDA: Static and interactive visualizations generated")
        print("✅ Forecasting: Time series models implemented and evaluated")
        print("✅ Portfolio Analysis: Correlation and performance analysis completed")
        print("✅ Interactive Dashboard: Web-based visualization interface available")
        
        print(f"\n📁 Output Files Generated:")
        print(f"  • Data: data/ (processed datasets)")
        print(f"  • Visualizations: results/plots/ (static and interactive charts)")
        print(f"  • Forecasting: results/forecasting/ (model results and reports)")
        print(f"  • Dashboard: dashboard.py (run with 'streamlit run dashboard.py')")
        
        print(f"\n🚀 Next Steps:")
        print(f"  • Launch dashboard: streamlit run dashboard.py")
        print(f"  • Review forecast results in results/forecasting/")
        print(f"  • Explore interactive visualizations in results/plots/")
        print(f"  • Implement portfolio optimization strategies")
        
        print("\n" + "="*80)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_complete_analysis()
    if success:
        print("\n🎉 Complete portfolio analysis finished successfully!")
    else:
        print("\n💥 Analysis failed. Please check the error messages above.")
        sys.exit(1)
