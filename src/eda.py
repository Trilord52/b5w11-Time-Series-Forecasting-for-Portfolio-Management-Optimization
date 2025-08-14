"""
Exploratory Data Analysis Module for Financial Data

This module provides comprehensive EDA capabilities including:
- Price and return visualizations
- Volatility analysis
- Statistical distributions
- Correlation analysis
- Outlier detection
- Trend analysis and seasonality
- Advanced statistical tests

Author: Financial Analyst Team
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional
import logging
import warnings
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import calendar

# Configure logging and styling
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class FinancialEDA:
    """
    Comprehensive EDA class for financial data analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize FinancialEDA.
        
        Parameters:
        -----------
        figsize : Tuple[int, int]
            Default figure size for plots
        """
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        logger.info("FinancialEDA initialized successfully")
    
    def create_price_analysis_plots(self, asset_data: Dict[str, pd.DataFrame], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create comprehensive price analysis plots.
        """
        logger.info("Creating price analysis plots")
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Comprehensive Price Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Price evolution
        for i, (symbol, data) in enumerate(asset_data.items()):
            axes[0, 0].plot(data.index, data['Close'], label=symbol, color=self.colors[i], linewidth=1.5)
        axes[0, 0].set_title('Price Evolution Over Time')
        axes[0, 0].set_ylabel('Price ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Daily returns
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                axes[0, 1].plot(data.index, data['Daily_Return'], label=symbol, color=self.colors[i], alpha=0.7)
        axes[0, 1].set_title('Daily Returns')
        axes[0, 1].set_ylabel('Daily Return')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility (rolling)
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Volatility_20d' in data.columns:
                axes[1, 0].plot(data.index, data['Volatility_20d'], label=symbol, color=self.colors[i], linewidth=1.5)
        axes[1, 0].set_title('20-Day Rolling Volatility')
        axes[1, 0].set_ylabel('Annualized Volatility')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Volume analysis
        for i, (symbol, data) in enumerate(asset_data.items()):
            axes[1, 1].plot(data.index, data['Volume'], label=symbol, color=self.colors[i], alpha=0.7)
        axes[1, 1].set_title('Trading Volume')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/price_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Price analysis plots saved to {save_path}")
        
        plt.show()
    
    def create_trend_and_seasonality_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                            save_path: Optional[str] = None) -> None:
        """
        Create trend and seasonality analysis plots.
        """
        logger.info("Creating trend and seasonality analysis")
        
        # Focus on TSLA for detailed analysis as per requirements
        tsla_data = asset_data.get('TSLA')
        if tsla_data is None or 'Close' not in tsla_data.columns:
            logger.warning("TSLA data not available for trend analysis")
            return
        
        # Prepare data for decomposition
        tsla_prices = tsla_data['Close'].dropna()
        if len(tsla_prices) < 50:
            logger.warning("Insufficient data for seasonal decomposition")
            return
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(tsla_prices, period=252, extrapolate_trend='freq')
            
            fig, axes = plt.subplots(4, 1, figsize=(15, 12))
            fig.suptitle('TSLA: Trend and Seasonality Analysis', fontsize=16, fontweight='bold')
            
            # Original data
            axes[0].plot(tsla_prices.index, tsla_prices, color='#1f77b4', linewidth=1.5)
            axes[0].set_title('Original Price Data')
            axes[0].set_ylabel('Price ($)')
            axes[0].grid(True, alpha=0.3)
            
            # Trend
            axes[1].plot(decomposition.trend.index, decomposition.trend, color='#ff7f0e', linewidth=1.5)
            axes[1].set_title('Trend Component')
            axes[1].set_ylabel('Trend')
            axes[1].grid(True, alpha=0.3)
            
            # Seasonal
            axes[2].plot(decomposition.seasonal.index, decomposition.seasonal, color='#2ca02c', linewidth=1.5)
            axes[2].set_title('Seasonal Component')
            axes[2].set_ylabel('Seasonal')
            axes[2].grid(True, alpha=0.3)
            
            # Residual
            axes[3].plot(decomposition.resid.index, decomposition.resid, color='#d62728', linewidth=1.5)
            axes[3].set_title('Residual Component')
            axes[3].set_ylabel('Residual')
            axes[3].set_xlabel('Date')
            axes[3].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(f"{save_path}/trend_seasonality_analysis.png", dpi=300, bbox_inches='tight')
                logger.info(f"Trend and seasonality analysis saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
    
    def create_volatility_clustering_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                            save_path: Optional[str] = None) -> None:
        """
        Create volatility clustering and regime analysis.
        """
        logger.info("Creating volatility clustering analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Volatility Clustering and Regime Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Volatility clustering (squared returns)
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                squared_returns = returns ** 2
                axes[0, 0].plot(returns.index, squared_returns, label=symbol, 
                               color=self.colors[i], alpha=0.7)
        axes[0, 0].set_title('Volatility Clustering (Squared Returns)')
        axes[0, 0].set_ylabel('Squared Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Rolling volatility comparison
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Volatility_20d' in data.columns:
                axes[0, 1].plot(data.index, data['Volatility_20d'], label=symbol, 
                               color=self.colors[i], linewidth=1.5)
        axes[0, 1].set_title('20-Day Rolling Volatility Comparison')
        axes[0, 1].set_ylabel('Annualized Volatility')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Volatility regime (TSLA focus)
        tsla_data = asset_data.get('TSLA')
        if tsla_data is not None and 'Volatility_Regime' in tsla_data.columns:
            regime_colors = ['green' if x == 'Low' else 'orange' if x == 'Medium' else 'red' 
                           for x in tsla_data['Volatility_Regime']]
            axes[1, 0].scatter(tsla_data.index, tsla_data['Close'], 
                              c=regime_colors, alpha=0.7, s=10)
            axes[1, 0].set_title('TSLA: Volatility Regime Analysis')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor='green', label='Low Volatility'),
                             Patch(facecolor='orange', label='Medium Volatility'),
                             Patch(facecolor='red', label='High Volatility')]
            axes[1, 0].legend(handles=legend_elements)
        
        # Plot 4: Volume vs Volatility relationship
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns and 'Volume' in data.columns:
                returns = data['Daily_Return'].dropna()
                volume = data['Volume'].loc[returns.index]
                axes[1, 1].scatter(abs(returns), volume, label=symbol, 
                                 color=self.colors[i], alpha=0.6, s=20)
        axes[1, 1].set_title('Volume vs Absolute Returns')
        axes[1, 1].set_xlabel('Absolute Daily Return')
        axes[1, 1].set_ylabel('Volume')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/volatility_clustering_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Volatility clustering analysis saved to {save_path}")
        
        plt.show()
    
    def create_outlier_analysis(self, asset_data: Dict[str, pd.DataFrame],
                               save_path: Optional[str] = None) -> None:
        """
        Create comprehensive outlier analysis using multiple methods.
        """
        logger.info("Creating outlier analysis")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Outlier Detection and Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Z-score based outliers
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                z_scores = np.abs(stats.zscore(returns))
                outliers = returns[z_scores > 3]
                
                axes[0, 0].plot(returns.index, returns, label=f'{symbol} (Normal)', 
                               color=self.colors[i], alpha=0.7)
                if len(outliers) > 0:
                    axes[0, 0].scatter(outliers.index, outliers, 
                                      color='red', s=50, alpha=0.8, 
                                      label=f'{symbol} (Outliers)')
        
        axes[0, 0].set_title('Z-Score Outlier Detection (|Z| > 3)')
        axes[0, 0].set_ylabel('Daily Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: IQR based outliers
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                Q1 = returns.quantile(0.25)
                Q3 = returns.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
                
                axes[0, 1].plot(returns.index, returns, label=f'{symbol} (Normal)', 
                               color=self.colors[i], alpha=0.7)
                if len(outliers) > 0:
                    axes[0, 1].scatter(outliers.index, outliers, 
                                      color='red', s=50, alpha=0.8, 
                                      label=f'{symbol} (Outliers)')
        
        axes[0, 1].set_title('IQR Outlier Detection (1.5 * IQR)')
        axes[0, 1].set_ylabel('Daily Returns')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Return distribution with outlier bounds
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                axes[1, 0].hist(returns, bins=50, alpha=0.7, label=symbol, 
                               color=self.colors[i], density=True)
                
                # Add normal distribution curve
                x = np.linspace(returns.min(), returns.max(), 100)
                mu, sigma = returns.mean(), returns.std()
                y = stats.norm.pdf(x, mu, sigma)
                axes[1, 0].plot(x, y, 'k--', alpha=0.8, label=f'{symbol} Normal')
        
        axes[1, 0].set_title('Return Distribution with Normal Fit')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Outlier summary statistics
        outlier_summary = []
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                z_scores = np.abs(stats.zscore(returns))
                z_outliers = len(returns[z_scores > 3])
                
                Q1, Q3 = returns.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                iqr_outliers = len(returns[(returns < Q1 - 1.5 * IQR) | 
                                         (returns > Q3 + 1.5 * IQR)])
                
                outlier_summary.append([symbol, z_outliers, iqr_outliers, len(returns)])
        
        if outlier_summary:
            summary_df = pd.DataFrame(outlier_summary, 
                                    columns=['Asset', 'Z-Score Outliers', 'IQR Outliers', 'Total Observations'])
            
            # Create bar plot
            x = np.arange(len(summary_df))
            width = 0.35
            
            axes[1, 1].bar(x - width/2, summary_df['Z-Score Outliers'], width, 
                           label='Z-Score Outliers', color='red', alpha=0.7)
            axes[1, 1].bar(x + width/2, summary_df['IQR Outliers'], width, 
                           label='IQR Outliers', color='orange', alpha=0.7)
            
            axes[1, 1].set_title('Outlier Summary by Detection Method')
            axes[1, 1].set_ylabel('Number of Outliers')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(summary_df['Asset'])
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/outlier_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Outlier analysis saved to {save_path}")
        
        plt.show()
    
    def create_statistical_tests_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                        save_path: Optional[str] = None) -> None:
        """
        Create statistical tests analysis including stationarity tests.
        """
        logger.info("Creating statistical tests analysis")
        
        # Focus on TSLA for detailed analysis
        tsla_data = asset_data.get('TSLA')
        if tsla_data is None or 'Daily_Return' not in tsla_data.columns:
            logger.warning("TSLA data not available for statistical tests")
            return
        
        returns = tsla_data['Daily_Return'].dropna()
        if len(returns) < 50:
            logger.warning("Insufficient data for statistical tests")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Statistical Tests and Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: ACF plot
        plot_acf(returns, lags=40, ax=axes[0, 0], alpha=0.05)
        axes[0, 0].set_title('Autocorrelation Function (ACF)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: PACF plot
        plot_pacf(returns, lags=40, ax=axes[0, 1], alpha=0.05)
        axes[0, 1].set_title('Partial Autocorrelation Function (PACF)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Q-Q plot for normality
        stats.probplot(returns, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot for Normality Test')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Histogram with normal fit
        axes[1, 1].hist(returns, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
        
        # Add normal distribution curve
        x = np.linspace(returns.min(), returns.max(), 100)
        mu, sigma = returns.mean(), returns.std()
        y = stats.norm.pdf(x, mu, sigma)
        axes[1, 1].plot(x, y, 'r-', linewidth=2, label=f'Normal(μ={mu:.4f}, σ={sigma:.4f})')
        
        axes[1, 1].set_title('Return Distribution with Normal Fit')
        axes[1, 1].set_xlabel('Daily Return')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/statistical_tests_analysis.png", dpi=300, bbox_inches='tight')
            logger.info(f"Statistical tests analysis saved to {save_path}")
        
        plt.show()
        
        # Perform and log statistical tests
        self._perform_statistical_tests(returns)
    
    def _perform_statistical_tests(self, returns: pd.Series) -> None:
        """
        Perform and log key statistical tests.
        """
        logger.info("Performing statistical tests on returns data")
        
        # Augmented Dickey-Fuller test for stationarity
        try:
            adf_result = adfuller(returns)
            logger.info(f"Augmented Dickey-Fuller Test Results:")
            logger.info(f"  ADF Statistic: {adf_result[0]:.6f}")
            logger.info(f"  p-value: {adf_result[1]:.6f}")
            logger.info(f"  Critical values: {adf_result[4]}")
            logger.info(f"  Series is {'stationary' if adf_result[1] < 0.05 else 'non-stationary'}")
        except Exception as e:
            logger.error(f"Error in ADF test: {e}")
        
        # KPSS test for stationarity
        try:
            kpss_result = kpss(returns, regression='c')
            logger.info(f"KPSS Test Results:")
            logger.info(f"  KPSS Statistic: {kpss_result[0]:.6f}")
            logger.info(f"  p-value: {kpss_result[1]:.6f}")
            logger.info(f"  Critical values: {kpss_result[3]}")
            logger.info(f"  Series is {'stationary' if kpss_result[1] > 0.05 else 'non-stationary'}")
        except Exception as e:
            logger.error(f"Error in KPSS test: {e}")
        
        # Jarque-Bera test for normality
        try:
            jb_stat, jb_pvalue = stats.jarque_bera(returns)
            logger.info(f"Jarque-Bera Test Results:")
            logger.info(f"  JB Statistic: {jb_stat:.6f}")
            logger.info(f"  p-value: {jb_pvalue:.6f}")
            logger.info(f"  Returns are {'normal' if jb_pvalue > 0.05 else 'non-normal'}")
        except Exception as e:
            logger.error(f"Error in Jarque-Bera test: {e}")
        
        # Ljung-Box test for autocorrelation (fixed)
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb_stat, lb_pvalue = acorr_ljungbox(returns, lags=10, return_df=False)
            logger.info(f"Ljung-Box Test Results:")
            logger.info(f"  Test Statistic: {lb_stat[-1]:.6f}")  # Use last lag value
            logger.info(f"  p-value: {lb_pvalue[-1]:.6f}")
            logger.info(f"  Returns show {'no' if lb_pvalue[-1] > 0.05 else 'significant'} autocorrelation")
        except Exception as e:
            logger.error(f"Error in Ljung-Box test: {e}")
    
    def create_return_distribution_plots(self, asset_data: Dict[str, pd.DataFrame],
                                       save_path: Optional[str] = None) -> None:
        """
        Create return distribution analysis plots.
        """
        logger.info("Creating return distribution plots")
        
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('Return Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Return histograms
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                axes[0, 0].hist(returns, bins=50, alpha=0.7, label=symbol, color=self.colors[i])
        axes[0, 0].set_title('Daily Returns Distribution')
        axes[0, 0].set_xlabel('Daily Return')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Q-Q plots
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                stats.probplot(returns, dist="norm", plot=axes[0, 1])
        axes[0, 1].set_title('Q-Q Plot (Normal Distribution)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Box plots
        return_data = []
        labels = []
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                return_data.append(data['Daily_Return'].dropna())
                labels.append(symbol)
        
        if return_data:
            axes[1, 0].boxplot(return_data, labels=labels)
            axes[1, 0].set_title('Return Distribution Box Plots')
            axes[1, 0].set_ylabel('Daily Return')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Cumulative returns
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                cumulative_returns = (1 + data['Daily_Return']).cumprod()
                axes[1, 1].plot(data.index, cumulative_returns, label=symbol, color=self.colors[i], linewidth=1.5)
        axes[1, 1].set_title('Cumulative Returns')
        axes[1, 1].set_ylabel('Cumulative Return')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/return_distribution.png", dpi=300, bbox_inches='tight')
            logger.info(f"Return distribution plots saved to {save_path}")
        
        plt.show()
    
    def create_correlation_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                  save_path: Optional[str] = None) -> None:
        """
        Create correlation analysis plots.
        """
        logger.info("Creating correlation analysis")
        
        # Prepare returns data for correlation
        returns_df = pd.DataFrame()
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns_df[symbol] = data['Daily_Return']
        
        if returns_df.empty:
            logger.warning("No return data available for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.title('Asset Returns Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/correlation_matrix.png", dpi=300, bbox_inches='tight')
            logger.info(f"Correlation matrix saved to {save_path}")
        
        plt.show()
        
        # Print correlation insights
        logger.info("Correlation Analysis Results:")
        for i, asset1 in enumerate(correlation_matrix.columns):
            for j, asset2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicate pairs
                    corr_value = correlation_matrix.loc[asset1, asset2]
                    logger.info(f"{asset1} vs {asset2}: {corr_value:.4f}")
    
    def create_risk_metrics_summary(self, metrics_data: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> None:
        """
        Create risk metrics summary visualization.
        """
        logger.info("Creating risk metrics summary")
        
        # Extract key metrics for visualization
        symbols = list(metrics_data.keys())
        
        # Prepare data for plotting - using the actual data structure from financial metrics
        sharpe_ratios = []
        volatilities = []
        max_drawdowns = []
        
        for symbol in symbols:
            # Extract values from the actual structure
            if 'Sharpe_Ratio' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Sharpe_Ratio'], dict):
                sharpe_value = metrics_data[symbol]['Sharpe_Ratio'].get('Sharpe_Ratio', 0)
                # Convert numpy.float64 to regular float
                sharpe_ratios.append(float(sharpe_value))
            else:
                sharpe_ratios.append(0)
            
            if 'Basic_Statistics' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Basic_Statistics'], dict):
                # Convert daily std to annual volatility
                daily_std = metrics_data[symbol]['Basic_Statistics'].get('Std_Return', 0)
                annual_vol = float(daily_std) * np.sqrt(252)  # Annualize and convert to float
                volatilities.append(annual_vol)
            else:
                volatilities.append(0)
            
            if 'Maximum_Drawdown' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Maximum_Drawdown'], dict):
                max_dd_pct = metrics_data[symbol]['Maximum_Drawdown'].get('Max_Drawdown_Pct', 0)
                # Convert to absolute value and decimal (e.g., -73.6% -> 0.736)
                max_drawdowns.append(abs(float(max_dd_pct)) / 100)
            else:
                max_drawdowns.append(0)
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Risk Metrics Summary', fontsize=16, fontweight='bold')
        
        # Plot 1: Sharpe Ratios
        bars1 = axes[0].bar(symbols, sharpe_ratios, color=self.colors[:len(symbols)])
        axes[0].set_title('Sharpe Ratio Comparison')
        axes[0].set_ylabel('Sharpe Ratio')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, sharpe_ratios):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 2: Volatility
        bars2 = axes[1].bar(symbols, volatilities, color=self.colors[:len(symbols)])
        axes[1].set_title('Annualized Volatility')
        axes[1].set_ylabel('Volatility')
        axes[1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, volatilities):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Maximum Drawdown
        bars3 = axes[2].bar(symbols, max_drawdowns, color=self.colors[:len(symbols)])
        axes[2].set_title('Maximum Drawdown')
        axes[2].set_ylabel('Maximum Drawdown')
        axes[2].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, max_drawdowns):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/risk_metrics_summary.png", dpi=300, bbox_inches='tight')
            logger.info(f"Risk metrics summary saved to {save_path}")
        
        plt.show()
    
    def generate_eda_report(self, asset_data: Dict[str, pd.DataFrame],
                           metrics_data: Dict[str, Dict],
                           save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive EDA report with actionable insights.
        """
        logger.info("Generating comprehensive EDA report")
        
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("Time Series Forecasting for Portfolio Management Optimization")
        report.append("Guide Me in Finance (GMF) Investments")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Executive Summary
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 40)
        report.append("This report provides comprehensive analysis of TSLA, BND, and SPY assets")
        report.append("covering the period from 2015 to 2024. Key findings include volatility")
        report.append("patterns, risk metrics, and statistical properties essential for portfolio")
        report.append("optimization and time series forecasting.")
        report.append("")
        
        # Data Overview
        report.append("DATA OVERVIEW")
        report.append("-" * 40)
        for symbol, data in asset_data.items():
            report.append(f"{symbol}:")
            report.append(f"  - Date Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
            report.append(f"  - Total Observations: {len(data):,}")
            report.append(f"  - Features: {len(data.columns)}")
            report.append(f"  - Missing Values: {data.isnull().sum().sum():,}")
            
            # Price statistics
            if 'Close' in data.columns:
                close_prices = data['Close'].dropna()
                report.append(f"  - Price Range: ${close_prices.min():.2f} - ${close_prices.max():.2f}")
                report.append(f"  - Current Price: ${close_prices.iloc[-1]:.2f}")
            
            report.append("")
        
        # Statistical Analysis
        report.append("STATISTICAL ANALYSIS")
        report.append("-" * 40)
        
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                report.append(f"{symbol} Returns Statistics:")
                report.append(f"  - Mean: {returns.mean():.4f}")
                report.append(f"  - Median: {returns.median():.4f}")
                report.append(f"  - Std Dev: {returns.std():.4f}")
                report.append(f"  - Skewness: {returns.skew():.4f}")
                report.append(f"  - Kurtosis: {returns.kurtosis():.4f}")
                report.append(f"  - Min: {returns.min():.4f}")
                report.append(f"  - Max: {returns.max():.4f}")
                report.append("")
        
        # Volatility Analysis
        report.append("VOLATILITY ANALYSIS")
        report.append("-" * 40)
        
        volatilities = {}
        for symbol, data in asset_data.items():
            if 'Volatility_20d' in data.columns:
                avg_vol = data['Volatility_20d'].mean()
                max_vol = data['Volatility_20d'].max()
                min_vol = data['Volatility_20d'].min()
                volatilities[symbol] = avg_vol
                report.append(f"{symbol}:")
                report.append(f"  - Average 20-day Volatility: {avg_vol:.2%}")
                report.append(f"  - Maximum Volatility: {max_vol:.2%}")
                report.append(f"  - Minimum Volatility: {min_vol:.2%}")
                report.append("")
        
        if volatilities:
            most_volatile = max(volatilities, key=volatilities.get)
            least_volatile = min(volatilities, key=volatilities.get)
            report.append(f"Volatility Ranking:")
            report.append(f"  - Most Volatile: {most_volatile} ({volatilities[most_volatile]:.2%})")
            report.append(f"  - Least Volatile: {least_volatile} ({volatilities[least_volatile]:.2%})")
            report.append("")
        
        # Risk Metrics Summary
        report.append("RISK METRICS SUMMARY")
        report.append("-" * 40)
        
        for symbol, metrics in metrics_data.items():
            report.append(f"{symbol}:")
            
            # Sharpe Ratio
            if 'Sharpe_Ratio' in metrics:
                sharpe = metrics['Sharpe_Ratio'].get('Sharpe_Ratio', 'N/A')
                if isinstance(sharpe, (int, float)):
                    report.append(f"  - Sharpe Ratio: {sharpe:.3f}")
                    if sharpe > 1.0:
                        report.append("    * Excellent risk-adjusted returns")
                    elif sharpe > 0.5:
                        report.append("    * Good risk-adjusted returns")
                    elif sharpe > 0:
                        report.append("    * Positive risk-adjusted returns")
                    else:
                        report.append("    * Poor risk-adjusted returns")
                else:
                    report.append(f"  - Sharpe Ratio: {sharpe}")
            
            # Maximum Drawdown
            if 'Maximum_Drawdown' in metrics:
                max_dd = metrics['Maximum_Drawdown'].get('Max_Drawdown_Pct', 'N/A')
                if isinstance(max_dd, (int, float)):
                    report.append(f"  - Max Drawdown: {max_dd:.2f}%")
                    if abs(max_dd) < 10:
                        report.append("    * Low risk asset")
                    elif abs(max_dd) < 25:
                        report.append("    * Moderate risk asset")
                    else:
                        report.append("    * High risk asset")
                else:
                    report.append(f"  - Max Drawdown: {max_dd}")
            
            # VaR
            if 'VaR' in metrics and 'Historical_95' in metrics['VaR']:
                var_95 = metrics['VaR']['Historical_95'].get('VaR_Annual', 'N/A')
                if isinstance(var_95, (int, float)):
                    report.append(f"  - VaR (95%): {var_95:.3f}")
                else:
                    report.append(f"  - VaR (95%): {var_95}")
            
            report.append("")
        
        # Correlation Analysis
        report.append("CORRELATION ANALYSIS")
        report.append("-" * 40)
        
        # Calculate correlation matrix
        returns_df = pd.DataFrame()
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns_df[symbol] = data['Daily_Return']
        
        if not returns_df.empty:
            correlation_matrix = returns_df.corr()
            report.append("Daily Returns Correlation Matrix:")
            for i, asset1 in enumerate(correlation_matrix.columns):
                for j, asset2 in enumerate(correlation_matrix.columns):
                    if i < j:  # Avoid duplicate pairs
                        corr_value = correlation_matrix.loc[asset1, asset2]
                        report.append(f"  {asset1} vs {asset2}: {corr_value:.4f}")
                        
                        # Interpret correlation
                        if abs(corr_value) > 0.7:
                            report.append("    * Strong correlation")
                        elif abs(corr_value) > 0.4:
                            report.append("    * Moderate correlation")
                        else:
                            report.append("    * Weak correlation")
            report.append("")
        
        # Outlier Analysis
        report.append("OUTLIER ANALYSIS")
        report.append("-" * 40)
        
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                
                # Z-score outliers
                z_scores = np.abs(stats.zscore(returns))
                z_outliers = len(returns[z_scores > 3])
                z_outlier_pct = (z_outliers / len(returns)) * 100
                
                # IQR outliers
                Q1, Q3 = returns.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                iqr_outliers = len(returns[(returns < Q1 - 1.5 * IQR) | 
                                         (returns > Q3 + 1.5 * IQR)])
                iqr_outlier_pct = (iqr_outliers / len(returns)) * 100
                
                report.append(f"{symbol}:")
                report.append(f"  - Z-Score Outliers: {z_outliers} ({z_outlier_pct:.2f}%)")
                report.append(f"  - IQR Outliers: {iqr_outliers} ({iqr_outlier_pct:.2f}%)")
                
                if z_outlier_pct > 5:
                    report.append("    * High number of extreme returns")
                elif z_outlier_pct > 2:
                    report.append("    * Moderate number of extreme returns")
                else:
                    report.append("    * Low number of extreme returns")
                report.append("")
        
        # Stationarity Analysis
        report.append("STATIONARITY ANALYSIS")
        report.append("-" * 40)
        
        # Focus on TSLA for detailed analysis
        tsla_data = asset_data.get('TSLA')
        if tsla_data is not None and 'Daily_Return' in tsla_data.columns:
            tsla_returns = tsla_data['Daily_Return'].dropna()
            
            try:
                # ADF test
                adf_result = adfuller(tsla_returns)
                report.append("TSLA Returns - Augmented Dickey-Fuller Test:")
                report.append(f"  - ADF Statistic: {adf_result[0]:.6f}")
                report.append(f"  - p-value: {adf_result[1]:.6f}")
                report.append(f"  - Critical values: {adf_result[4]}")
                
                if adf_result[1] < 0.05:
                    report.append("  - Conclusion: Series is STATIONARY")
                    report.append("    * Suitable for ARIMA models without differencing")
                else:
                    report.append("  - Conclusion: Series is NON-STATIONARY")
                    report.append("    * Differencing required for ARIMA models")
                
                # KPSS test
                kpss_result = kpss(tsla_returns, regression='c')
                report.append("")
                report.append("TSLA Returns - KPSS Test:")
                report.append(f"  - KPSS Statistic: {kpss_result[0]:.6f}")
                report.append(f"  - p-value: {kpss_result[1]:.6f}")
                
                if kpss_result[1] > 0.05:
                    report.append("  - Conclusion: Series is STATIONARY")
                else:
                    report.append("  - Conclusion: Series is NON-STATIONARY")
                
            except Exception as e:
                report.append(f"Error in stationarity tests: {e}")
            
            report.append("")
        
        # Market Regime Analysis
        report.append("MARKET REGIME ANALYSIS")
        report.append("-" * 40)
        
        for symbol, data in asset_data.items():
            if 'Volatility_Regime' in data.columns:
                regime_counts = data['Volatility_Regime'].value_counts()
                report.append(f"{symbol} Volatility Regimes:")
                for regime, count in regime_counts.items():
                    percentage = (count / len(data)) * 100
                    report.append(f"  - {regime}: {count} days ({percentage:.1f}%)")
                report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        
        # Volatility insights
        if volatilities:
            vol_ratio = volatilities['TSLA'] / volatilities['BND']
            report.append(f"1. TSLA is approximately {vol_ratio:.1f}x more volatile than BND")
            report.append("   - This highlights the risk-return trade-off between growth and stability")
        
        # Correlation insights
        if not returns_df.empty:
            tsla_bnd_corr = returns_df['TSLA'].corr(returns_df['BND'])
            if abs(tsla_bnd_corr) < 0.3:
                report.append("2. TSLA and BND show low correlation, providing diversification benefits")
            else:
                report.append("2. TSLA and BND show moderate correlation, limiting diversification")
        
        report.append("3. SPY provides balanced market exposure with moderate volatility")
        report.append("4. Consider volatility clustering for risk management strategies")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. RISK MANAGEMENT:")
        report.append("   - Use BND for portfolio stability and income generation")
        report.append("   - Limit TSLA exposure based on risk tolerance")
        report.append("   - Monitor volatility regimes for dynamic allocation")
        report.append("")
        report.append("2. PORTFOLIO CONSTRUCTION:")
        report.append("   - Consider equal-weight or risk-parity approaches")
        report.append("   - Rebalance during regime changes")
        report.append("   - Use SPY for core market exposure")
        report.append("")
        report.append("3. FORECASTING CONSIDERATIONS:")
        report.append("   - Account for non-stationarity in returns")
        report.append("   - Model volatility clustering effects")
        report.append("   - Consider seasonal patterns in trading")
        report.append("")
        
        # Next Steps
        report.append("NEXT STEPS")
        report.append("-" * 40)
        report.append("1. Task 2: Develop Time Series Forecasting Models")
        report.append("   - Build ARIMA/SARIMA models for TSLA")
        report.append("   - Implement LSTM neural networks")
        report.append("   - Compare model performance")
        report.append("")
        report.append("2. Task 3: Forecast Future Market Trends")
        report.append("   - Generate 6-12 month forecasts")
        report.append("   - Analyze confidence intervals")
        report.append("   - Assess forecast uncertainty")
        report.append("")
        report.append("3. Task 4: Portfolio Optimization")
        report.append("   - Implement Modern Portfolio Theory")
        report.append("   - Generate Efficient Frontier")
        report.append("   - Identify optimal portfolios")
        report.append("")
        
        report.append("=" * 80)
        report.append("END OF REPORT")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(f"{save_path}/eda_report.txt", 'w') as f:
                f.write(report_text)
            logger.info(f"EDA report saved to {save_path}")
        
        return report_text

    def create_interactive_price_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                        save_path: Optional[str] = None) -> None:
        """
        Create interactive Plotly price analysis plots with zoom/pan controls.
        """
        logger.info("Creating interactive price analysis plots")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Price Evolution Over Time', 'Daily Returns', 
                           '20-Day Rolling Volatility', 'Trading Volume'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Price evolution
        for i, (symbol, data) in enumerate(asset_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['Close'],
                    mode='lines',
                    name=symbol,
                    line=dict(color=self.colors[i], width=2),
                    hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Price: $%{{y:.2f}}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Plot 2: Daily returns
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['Daily_Return'],
                        mode='lines',
                        name=f'{symbol} Returns',
                        line=dict(color=self.colors[i], width=1),
                        opacity=0.7,
                        hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=2
                )
        
        # Plot 3: Volatility (rolling)
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Volatility_20d' in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index, 
                        y=data['Volatility_20d'],
                        mode='lines',
                        name=f'{symbol} Volatility',
                        line=dict(color=self.colors[i], width=2),
                        hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Volatility: %{{y:.2%}}<extra></extra>'
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Volume analysis
        for i, (symbol, data) in enumerate(asset_data.items()):
            fig.add_trace(
                go.Scatter(
                    x=data.index, 
                    y=data['Volume'],
                    mode='lines',
                    name=f'{symbol} Volume',
                    line=dict(color=self.colors[i], width=1),
                    opacity=0.7,
                    hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Volume: %{{y:,.0f}}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout with interactive features
        fig.update_layout(
            title='Interactive Price Analysis Dashboard',
            height=800,
            showlegend=True,
            hovermode='x unified',
            # Add zoom and pan controls
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date',
                title='Date'
            ),
            xaxis2=dict(
                rangeslider=dict(visible=True),
                type='date',
                title='Date'
            ),
            xaxis3=dict(
                rangeslider=dict(visible=True),
                type='date',
                title='Date'
            ),
            xaxis4=dict(
                rangeslider=dict(visible=True),
                type='date',
                title='Date'
            ),
            # Add modebar with zoom/pan tools
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)'
            )
        )
        
        # Update y-axis labels
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return", row=1, col=2)
        fig.update_yaxes(title_text="Annualized Volatility", row=2, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=2)
        
        # Add buttons for different time ranges
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="1M", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(months=1), pd.Timestamp.now()]}]),
                        dict(label="3M", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(months=3), pd.Timestamp.now()]}]),
                        dict(label="6M", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(months=6), pd.Timestamp.now()]}]),
                        dict(label="1Y", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(years=1), pd.Timestamp.now()]}]),
                        dict(label="All", method="relayout", args=[{"xaxis.range": [None, None]}]),
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        
        if save_path:
            fig.write_html(f"{save_path}/interactive_price_analysis.html")
            logger.info(f"Interactive price analysis saved to {save_path}")
        
        # Show the plot
        fig.show()
    
    def create_interactive_risk_metrics(self, metrics_data: Dict[str, Dict],
                                       save_path: Optional[str] = None) -> None:
        """
        Create interactive Plotly risk metrics visualization with zoom/pan controls.
        """
        logger.info("Creating interactive risk metrics visualization")
        
        # Extract key metrics for visualization
        symbols = list(metrics_data.keys())
        
        # Prepare data for plotting
        sharpe_ratios = []
        volatilities = []
        max_drawdowns = []
        
        for symbol in symbols:
            # Extract values from the actual structure
            if 'Sharpe_Ratio' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Sharpe_Ratio'], dict):
                sharpe_value = metrics_data[symbol]['Sharpe_Ratio'].get('Sharpe_Ratio', 0)
                sharpe_ratios.append(float(sharpe_value))
            else:
                sharpe_ratios.append(0)
            
            if 'Basic_Statistics' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Basic_Statistics'], dict):
                daily_std = metrics_data[symbol]['Basic_Statistics'].get('Std_Return', 0)
                annual_vol = float(daily_std) * np.sqrt(252)
                volatilities.append(annual_vol)
            else:
                volatilities.append(0)
            
            if 'Maximum_Drawdown' in metrics_data[symbol] and isinstance(metrics_data[symbol]['Maximum_Drawdown'], dict):
                max_dd_pct = metrics_data[symbol]['Maximum_Drawdown'].get('Max_Drawdown_Pct', 0)
                max_drawdowns.append(abs(float(max_dd_pct)) / 100)
            else:
                max_drawdowns.append(0)
        
        # Create interactive bar charts
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Sharpe Ratio Comparison', 'Annualized Volatility', 'Maximum Drawdown'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sharpe Ratio bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=sharpe_ratios,
                name='Sharpe Ratio',
                marker_color=self.colors[:len(symbols)],
                hovertemplate='Asset: %{x}<br>Sharpe Ratio: %{y:.3f}<extra></extra>',
                text=[f'{val:.3f}' for val in sharpe_ratios],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # Volatility bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=volatilities,
                name='Volatility',
                marker_color=self.colors[:len(symbols)],
                hovertemplate='Asset: %{x}<br>Volatility: %{y:.3f}<extra></extra>',
                text=[f'{val:.3f}' for val in volatilities],
                textposition='auto'
            ),
            row=1, col=2
        )
        
        # Maximum Drawdown bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=max_drawdowns,
                name='Max Drawdown',
                marker_color=self.colors[:len(symbols)],
                hovertemplate='Asset: %{x}<br>Max Drawdown: %{y:.4f}<extra></extra>',
                text=[f'{val:.4f}' for val in max_drawdowns],
                textposition='auto'
            ),
            row=1, col=3
        )
        
        # Update layout with interactive features
        fig.update_layout(
            title='Interactive Risk Metrics Dashboard',
            height=500,
            showlegend=False,
            # Add zoom and pan controls
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)'
            )
        )
        
        # Update axes
        fig.update_xaxes(title_text="Assets", row=1, col=1)
        fig.update_xaxes(title_text="Assets", row=1, col=2)
        fig.update_xaxes(title_text="Assets", row=1, col=3)
        
        fig.update_yaxes(title_text="Sharpe Ratio", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=1, col=2)
        fig.update_yaxes(title_text="Max Drawdown", row=1, col=3)
        
        # Add buttons for different views
        fig.update_layout(
            updatemenus=[
                dict(
                    buttons=list([
                        dict(label="All Metrics", method="restyle", args=[{"visible": [True, True, True]}]),
                        dict(label="Sharpe Only", method="restyle", args=[{"visible": [True, False, False]}]),
                        dict(label="Volatility Only", method="restyle", args=[{"visible": [False, True, False]}]),
                        dict(label="Drawdown Only", method="restyle", args=[{"visible": [False, False, True]}]),
                    ]),
                    direction="down",
                    showactive=True,
                    x=0.1,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                ),
            ]
        )
        
        if save_path:
            fig.write_html(f"{save_path}/interactive_risk_metrics.html")
            logger.info(f"Interactive risk metrics saved to {save_path}")
        
        # Show the plot
        fig.show()

    def create_interactive_correlation_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                              save_path: Optional[str] = None) -> None:
        """
        Create interactive correlation analysis with zoom/pan controls.
        """
        logger.info("Creating interactive correlation analysis")
        
        # Prepare returns data for correlation
        returns_df = pd.DataFrame()
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns_df[symbol] = data['Daily_Return']
        
        if returns_df.empty:
            logger.warning("No return data available for correlation analysis")
            return
        
        # Calculate correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Create interactive correlation heatmap
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values.round(4),
            texttemplate="%{text}",
            textfont={"size": 12},
            hovertemplate='Asset 1: %{y}<br>Asset 2: %{x}<br>Correlation: %{z:.4f}<extra></extra>'
        ))
        
        # Update layout with interactive features
        fig.update_layout(
            title='Interactive Asset Returns Correlation Matrix',
            height=600,
            width=600,
            # Add zoom and pan controls
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)'
            ),
            xaxis=dict(title="Assets"),
            yaxis=dict(title="Assets")
        )
        
        if save_path:
            fig.write_html(f"{save_path}/interactive_correlation_analysis.html")
            logger.info(f"Interactive correlation analysis saved to {save_path}")
        
        # Show the plot
        fig.show()
    
    def create_interactive_outlier_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                          save_path: Optional[str] = None) -> None:
        """
        Create interactive outlier analysis with zoom/pan controls.
        """
        logger.info("Creating interactive outlier analysis")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Z-Score Outlier Detection', 'IQR Outlier Detection', 
                           'Return Distribution', 'Outlier Summary'],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Plot 1: Z-score based outliers
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                z_scores = np.abs(stats.zscore(returns))
                outliers = returns[z_scores > 3]
                
                # Normal returns
                normal_returns = returns[z_scores <= 3]
                fig.add_trace(
                    go.Scatter(
                        x=normal_returns.index,
                        y=normal_returns,
                        mode='markers',
                        name=f'{symbol} (Normal)',
                        marker=dict(color=self.colors[i], size=3, opacity=0.7),
                        hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Outliers
                if len(outliers) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=outliers.index,
                            y=outliers,
                            mode='markers',
                            name=f'{symbol} (Outliers)',
                            marker=dict(color='red', size=8, symbol='x'),
                            hovertemplate=f'{symbol} OUTLIER<br>Date: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>'
                        ),
                        row=1, col=1
                    )
        
        # Plot 2: IQR based outliers
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                Q1 = returns.quantile(0.25)
                Q3 = returns.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = returns[(returns < lower_bound) | (returns > upper_bound)]
                normal_returns = returns[(returns >= lower_bound) & (returns <= upper_bound)]
                
                # Normal returns
                fig.add_trace(
                    go.Scatter(
                        x=normal_returns.index,
                        y=normal_returns,
                        mode='markers',
                        name=f'{symbol} (Normal)',
                        marker=dict(color=self.colors[i], size=3, opacity=0.7),
                        hovertemplate=f'{symbol}<br>Date: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=1, col=2
                )
                
                # Outliers
                if len(outliers) > 0:
                    fig.add_trace(
                        go.Scatter(
                            x=outliers.index,
                            y=outliers,
                            mode='markers',
                            name=f'{symbol} (Outliers)',
                            marker=dict(color='red', size=8, symbol='x'),
                            hovertemplate=f'{symbol} OUTLIER<br>Date: %{{x}}<br>Return: %{{y:.4f}}<extra></extra>',
                            showlegend=False
                        ),
                        row=1, col=2
                    )
        
        # Plot 3: Return distribution with normal fit
        for i, (symbol, data) in enumerate(asset_data.items()):
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                
                # Histogram
                fig.add_trace(
                    go.Histogram(
                        x=returns,
                        name=f'{symbol}',
                        nbinsx=50,
                        opacity=0.7,
                        marker_color=self.colors[i],
                        hovertemplate=f'{symbol}<br>Return: %{{x:.4f}}<br>Count: %{{y}}<extra></extra>'
                    ),
                    row=2, col=1
                )
                
                # Normal distribution curve
                x = np.linspace(returns.min(), returns.max(), 100)
                mu, sigma = returns.mean(), returns.std()
                y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) / 50
                
                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        name=f'{symbol} Normal',
                        line=dict(color=self.colors[i], width=2),
                        hovertemplate=f'{symbol} Normal<br>Return: %{{x:.4f}}<br>Density: %{{y:.2f}}<extra></extra>',
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        # Plot 4: Outlier summary
        outlier_summary = []
        for symbol, data in asset_data.items():
            if 'Daily_Return' in data.columns:
                returns = data['Daily_Return'].dropna()
                z_scores = np.abs(stats.zscore(returns))
                z_outliers = len(returns[z_scores > 3])
                
                Q1, Q3 = returns.quantile([0.25, 0.75])
                IQR = Q3 - Q1
                iqr_outliers = len(returns[(returns < Q1 - 1.5 * IQR) | 
                                         (returns > Q3 + 1.5 * IQR)])
                
                outlier_summary.append([symbol, z_outliers, iqr_outliers])
        
        if outlier_summary:
            symbols_list = [item[0] for item in outlier_summary]
            z_outliers_list = [item[1] for item in outlier_summary]
            iqr_outliers_list = [item[2] for item in outlier_summary]
            
            fig.add_trace(
                go.Bar(
                    x=symbols_list,
                    y=z_outliers_list,
                    name='Z-Score Outliers',
                    marker_color='red',
                    opacity=0.7,
                    hovertemplate='Asset: %{x}<br>Z-Score Outliers: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
            
            fig.add_trace(
                go.Bar(
                    x=symbols_list,
                    y=iqr_outliers_list,
                    name='IQR Outliers',
                    marker_color='orange',
                    opacity=0.7,
                    hovertemplate='Asset: %{x}<br>IQR Outliers: %{y}<extra></extra>'
                ),
                row=2, col=2
            )
        
        # Update layout with interactive features
        fig.update_layout(
            title='Interactive Outlier Analysis Dashboard',
            height=800,
            showlegend=True,
            # Add zoom and pan controls
            modebar=dict(
                orientation='v',
                bgcolor='rgba(255,255,255,0.7)'
            )
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Date", row=1, col=1)
        fig.update_xaxes(title_text="Date", row=1, col=2)
        fig.update_xaxes(title_text="Daily Return", row=2, col=1)
        fig.update_xaxes(title_text="Assets", row=2, col=2)
        
        fig.update_yaxes(title_text="Daily Return", row=1, col=1)
        fig.update_yaxes(title_text="Daily Return", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_yaxes(title_text="Number of Outliers", row=2, col=2)
        
        if save_path:
            fig.write_html(f"{save_path}/interactive_outlier_analysis.html")
            logger.info(f"Interactive outlier analysis saved to {save_path}")
        
        # Show the plot
        fig.show()
    
    def create_interactive_trend_analysis(self, asset_data: Dict[str, pd.DataFrame],
                                        save_path: Optional[str] = None) -> None:
        """
        Create interactive trend and seasonality analysis with zoom/pan controls.
        """
        logger.info("Creating interactive trend and seasonality analysis")
        
        # Focus on TSLA for detailed analysis
        tsla_data = asset_data.get('TSLA')
        if tsla_data is None or 'Close' not in tsla_data.columns:
            logger.warning("TSLA data not available for trend analysis")
            return
        
        # Prepare data for decomposition
        tsla_prices = tsla_data['Close'].dropna()
        if len(tsla_prices) < 50:
            logger.warning("Insufficient data for seasonal decomposition")
            return
        
        # Seasonal decomposition
        try:
            decomposition = seasonal_decompose(tsla_prices, period=252, extrapolate_trend='freq')
            
            # Create interactive subplots
            fig = make_subplots(
                rows=4, cols=1,
                subplot_titles=['Original Price Data', 'Trend Component', 'Seasonal Component', 'Residual Component'],
                specs=[[{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}],
                       [{"secondary_y": False}]]
            )
            
            # Original data
            fig.add_trace(
                go.Scatter(
                    x=tsla_prices.index,
                    y=tsla_prices,
                    mode='lines',
                    name='TSLA Price',
                    line=dict(color='#1f77b4', width=2),
                    hovertemplate='Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # Trend
            fig.add_trace(
                go.Scatter(
                    x=decomposition.trend.index,
                    y=decomposition.trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='#ff7f0e', width=2),
                    hovertemplate='Date: %{x}<br>Trend: $%{y:.2f}<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Seasonal
            fig.add_trace(
                go.Scatter(
                    x=decomposition.seasonal.index,
                    y=decomposition.seasonal,
                    mode='lines',
                    name='Seasonal',
                    line=dict(color='#2ca02c', width=2),
                    hovertemplate='Date: %{x}<br>Seasonal: $%{y:.2f}<extra></extra>'
                ),
                row=3, col=1
            )
            
            # Residual
            fig.add_trace(
                go.Scatter(
                    x=decomposition.resid.index,
                    y=decomposition.resid,
                    mode='lines',
                    name='Residual',
                    line=dict(color='#d62728', width=2),
                    hovertemplate='Date: %{x}<br>Residual: $%{y:.2f}<extra></extra>'
                ),
                row=4, col=1
            )
            
            # Update layout with interactive features
            fig.update_layout(
                title='Interactive TSLA: Trend and Seasonality Analysis',
                height=1000,
                showlegend=True,
                # Add zoom and pan controls
                modebar=dict(
                    orientation='v',
                    bgcolor='rgba(255,255,255,0.7)'
                )
            )
            
            # Add range slider for all subplots
            fig.update_layout(
                xaxis=dict(rangeslider=dict(visible=True)),
                xaxis2=dict(rangeslider=dict(visible=True)),
                xaxis3=dict(rangeslider=dict(visible=True)),
                xaxis4=dict(rangeslider=dict(visible=True))
            )
            
            # Add buttons for different time ranges
            fig.update_layout(
                updatemenus=[
                    dict(
                        buttons=list([
                            dict(label="1Y", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(years=1), pd.Timestamp.now()]}]),
                            dict(label="2Y", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(years=2), pd.Timestamp.now()]}]),
                            dict(label="5Y", method="relayout", args=[{"xaxis.range": [pd.Timestamp.now() - pd.DateOffset(years=5), pd.Timestamp.now()]}]),
                            dict(label="All", method="relayout", args=[{"xaxis.range": [None, None]}]),
                        ]),
                        direction="down",
                        showactive=True,
                        x=0.1,
                        xanchor="left",
                        y=1.1,
                        yanchor="top"
                    ),
                ]
            )
            
            # Update y-axis labels
            fig.update_yaxes(title_text="Price ($)", row=1, col=1)
            fig.update_yaxes(title_text="Trend", row=2, col=1)
            fig.update_yaxes(title_text="Seasonal", row=3, col=1)
            fig.update_yaxes(title_text="Residual", row=4, col=1)
            
            if save_path:
                fig.write_html(f"{save_path}/interactive_trend_analysis.html")
                logger.info(f"Interactive trend analysis saved to {save_path}")
            
            # Show the plot
            fig.show()
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")


def main():
    """
    Main function to demonstrate the EDA module.
    """
    print("FinancialEDA module loaded successfully!")
    print("Use this module for comprehensive financial data exploration and visualization.")


if __name__ == "__main__":
    main()

