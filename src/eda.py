"""
Exploratory Data Analysis Module for Financial Data

This module provides comprehensive EDA capabilities including:
- Price and return visualizations
- Volatility analysis
- Statistical distributions
- Correlation analysis
- Outlier detection

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
        
        # Prepare data for plotting
        sharpe_ratios = []
        var_95_values = []
        max_drawdowns = []
        
        for symbol in symbols:
            if 'Sharpe_Ratio' in metrics_data[symbol]:
                sharpe_ratios.append(metrics_data[symbol]['Sharpe_Ratio'].get('Sharpe_Ratio', 0))
            else:
                sharpe_ratios.append(0)
            
            if 'VaR' in metrics_data[symbol] and 'Historical_95' in metrics_data[symbol]['VaR']:
                var_95_values.append(metrics_data[symbol]['VaR']['Historical_95'].get('VaR_Annual', 0))
            else:
                var_95_values.append(0)
            
            if 'Maximum_Drawdown' in metrics_data[symbol]:
                max_drawdowns.append(metrics_data[symbol]['Maximum_Drawdown'].get('Max_Drawdown_Pct', 0))
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
        
        # Plot 2: VaR (95%)
        bars2 = axes[1].bar(symbols, var_95_values, color=self.colors[:len(symbols)])
        axes[1].set_title('Value at Risk (95% Confidence)')
        axes[1].set_ylabel('Annual VaR')
        axes[1].grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, var_95_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 3: Maximum Drawdown
        bars3 = axes[2].bar(symbols, max_drawdowns, color=self.colors[:len(symbols)])
        axes[2].set_title('Maximum Drawdown')
        axes[2].set_ylabel('Maximum Drawdown (%)')
        axes[2].grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, max_drawdowns):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/risk_metrics_summary.png", dpi=300, bbox_inches='tight')
            logger.info(f"Risk metrics summary saved to {save_path}")
        
        plt.show()
    
    def generate_eda_report(self, asset_data: Dict[str, pd.DataFrame],
                           metrics_data: Dict[str, Dict],
                           save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive EDA report.
        """
        logger.info("Generating comprehensive EDA report")
        
        report = []
        report.append("=" * 80)
        report.append("EXPLORATORY DATA ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
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
            report.append("")
        
        # Key Insights
        report.append("KEY INSIGHTS")
        report.append("-" * 40)
        
        # Volatility comparison
        volatilities = {}
        for symbol, data in asset_data.items():
            if 'Volatility_20d' in data.columns:
                avg_vol = data['Volatility_20d'].mean()
                volatilities[symbol] = avg_vol
                report.append(f"{symbol} Average 20-day Volatility: {avg_vol:.2%}")
        
        if volatilities:
            most_volatile = max(volatilities, key=volatilities.get)
            least_volatile = min(volatilities, key=volatilities.get)
            report.append(f"Most Volatile Asset: {most_volatile}")
            report.append(f"Least Volatile Asset: {least_volatile}")
        
        report.append("")
        
        # Risk Metrics Summary
        report.append("RISK METRICS SUMMARY")
        report.append("-" * 40)
        for symbol, metrics in metrics_data.items():
            report.append(f"{symbol}:")
            if 'Sharpe_Ratio' in metrics:
                sharpe = metrics['Sharpe_Ratio'].get('Sharpe_Ratio', 'N/A')
                report.append(f"  - Sharpe Ratio: {sharpe:.3f}" if isinstance(sharpe, (int, float)) else f"  - Sharpe Ratio: {sharpe}")
            
            if 'Maximum_Drawdown' in metrics:
                max_dd = metrics['Maximum_Drawdown'].get('Max_Drawdown_Pct', 'N/A')
                report.append(f"  - Max Drawdown: {max_dd:.2f}%" if isinstance(max_dd, (int, float)) else f"  - Max Drawdown: {max_dd}")
            
            if 'VaR' in metrics and 'Historical_95' in metrics['VaR']:
                var_95 = metrics['VaR']['Historical_95'].get('VaR_Annual', 'N/A')
                report.append(f"  - VaR (95%): {var_95:.3f}" if isinstance(var_95, (int, float)) else f"  - VaR (95%): {var_95}")
            
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("1. Consider the high volatility of TSLA for risk management")
        report.append("2. BND provides stability and diversification benefits")
        report.append("3. SPY offers balanced market exposure")
        report.append("4. Monitor correlation changes during market stress periods")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(f"{save_path}/eda_report.txt", 'w') as f:
                f.write(report_text)
            logger.info(f"EDA report saved to {save_path}")
        
        return report_text


def main():
    """
    Main function to demonstrate the EDA module.
    """
    print("FinancialEDA module loaded successfully!")
    print("Use this module for comprehensive financial data exploration and visualization.")


if __name__ == "__main__":
    main()

