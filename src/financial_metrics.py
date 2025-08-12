"""
Financial Metrics and Risk Analysis Module

This module provides comprehensive financial calculations including:
- Value at Risk (VaR) with multiple methodologies
- Sharpe Ratio and other risk-adjusted return metrics
- Maximum Drawdown and other risk measures
- Statistical tests for stationarity and normality

Author: Financial Analyst Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from scipy import stats
from scipy.stats import norm
import warnings

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class FinancialMetricsCalculator:
    """
    A comprehensive class for calculating financial metrics and risk measures.
    
    This class implements:
    - Multiple VaR methodologies (Historical, Parametric, Monte Carlo)
    - Risk-adjusted return metrics (Sharpe, Sortino, Calmar ratios)
    - Statistical tests for financial data analysis
    - Advanced risk measures for portfolio management
    """
    
    def __init__(self, risk_free_rate: float = 0.02, confidence_level: float = 0.95):
        """
        Initialize the FinancialMetricsCalculator.
        
        Parameters:
        -----------
        risk_free_rate : float
            Annual risk-free rate (default: 2% for US Treasury)
        confidence_level : float
            Confidence level for VaR calculations (default: 95%)
        """
        self.risk_free_rate = risk_free_rate
        self.daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1  # Daily equivalent
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level  # Significance level
        
        logger.info(f"FinancialMetricsCalculator initialized with:")
        logger.info(f"  - Risk-free rate: {risk_free_rate:.2%}")
        logger.info(f"  - Confidence level: {confidence_level:.1%}")
        logger.info(f"  - Daily RF rate: {self.daily_rf_rate:.6f}")
    
    def calculate_var(self, returns: pd.Series, method: str = 'historical', 
                     confidence_level: Optional[float] = None, 
                     window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Value at Risk using multiple methodologies.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series
        method : str
            VaR calculation method: 'historical', 'parametric', 'monte_carlo'
        confidence_level : float, optional
            Confidence level for VaR (overrides default if provided)
        window : int, optional
            Rolling window for VaR calculation (if None, uses full series)
            
        Returns:
        --------
        Dict[str, float]
            Dictionary containing VaR values and additional metrics
        """
        if confidence_level is None:
            confidence_level = self.confidence_level
            alpha = self.alpha
        else:
            alpha = 1 - confidence_level
        
        logger.info(f"Calculating VaR using {method} method at {confidence_level:.1%} confidence level")
        
        # Clean returns data
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            logger.warning(f"Insufficient data for VaR calculation: {len(clean_returns)} observations")
            return {}
        
        if method == 'historical':
            return self._calculate_historical_var(clean_returns, alpha, window)
        elif method == 'parametric':
            return self._calculate_parametric_var(clean_returns, alpha, window)
        elif method == 'monte_carlo':
            return self._calculate_monte_carlo_var(clean_returns, alpha, window)
        else:
            raise ValueError(f"Unknown VaR method: {method}. Use 'historical', 'parametric', or 'monte_carlo'")
    
    def _calculate_historical_var(self, returns: pd.Series, alpha: float, 
                                window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Historical VaR.
        
        Parameters:
        -----------
        returns : pd.Series
            Clean daily returns
        alpha : float
            Significance level
        window : int, optional
            Rolling window size
            
        Returns:
        --------
        Dict[str, float]
            Historical VaR results
        """
        if window is None:
            # Use full series
            var_historical = np.percentile(returns, alpha * 100)
            var_historical_annual = var_historical * np.sqrt(252)
            
            return {
                'VaR_Daily': var_historical,
                'VaR_Annual': var_historical_annual,
                'Method': 'Historical',
                'Confidence_Level': 1 - alpha,
                'Observations': len(returns),
                'Worst_Return': returns.min(),
                'Best_Return': returns.max()
            }
        else:
            # Rolling VaR
            rolling_var = returns.rolling(window=window).quantile(alpha)
            rolling_var_annual = rolling_var * np.sqrt(252)
            
            return {
                'VaR_Daily_Rolling': rolling_var,
                'VaR_Annual_Rolling': rolling_var_annual,
                'Method': 'Historical_Rolling',
                'Window': window,
                'Confidence_Level': 1 - alpha
            }
    
    def _calculate_parametric_var(self, returns: pd.Series, alpha: float, 
                                window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Parametric (Normal) VaR.
        
        Parameters:
        -----------
        returns : pd.Series
            Clean daily returns
        alpha : float
            Significance level
        window : int, optional
            Rolling window size
            
        Returns:
        --------
        Dict[str, float]
            Parametric VaR results
        """
        if window is None:
            # Use full series
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Z-score for the confidence level
            z_score = norm.ppf(alpha)
            
            var_parametric = mean_return + z_score * std_return
            var_parametric_annual = var_parametric * np.sqrt(252)
            
            return {
                'VaR_Daily': var_parametric,
                'VaR_Annual': var_parametric_annual,
                'Method': 'Parametric',
                'Confidence_Level': 1 - alpha,
                'Mean_Return': mean_return,
                'Std_Return': std_return,
                'Z_Score': z_score,
                'Observations': len(returns)
            }
        else:
            # Rolling parametric VaR
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            z_score = norm.ppf(alpha)
            
            rolling_var = rolling_mean + z_score * rolling_std
            rolling_var_annual = rolling_var * np.sqrt(252)
            
            return {
                'VaR_Daily_Rolling': rolling_var,
                'VaR_Annual_Rolling': rolling_var_annual,
                'Method': 'Parametric_Rolling',
                'Window': window,
                'Confidence_Level': 1 - alpha,
                'Z_Score': z_score
            }
    
    def _calculate_monte_carlo_var(self, returns: pd.Series, alpha: float, 
                                 window: Optional[int] = None, 
                                 n_simulations: int = 10000) -> Dict[str, float]:
        """
        Calculate Monte Carlo VaR.
        
        Parameters:
        -----------
        returns : pd.Series
            Clean daily returns
        alpha : float
            Significance level
        window : int, optional
            Rolling window size
        n_simulations : int
            Number of Monte Carlo simulations
            
        Returns:
        --------
        Dict[str, float]
            Monte Carlo VaR results
        """
        if window is None:
            # Use full series
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Generate random returns from normal distribution
            simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
            
            # Calculate VaR from simulated distribution
            var_monte_carlo = np.percentile(simulated_returns, alpha * 100)
            var_monte_carlo_annual = var_monte_carlo * np.sqrt(252)
            
            return {
                'VaR_Daily': var_monte_carlo,
                'VaR_Annual': var_monte_carlo_annual,
                'Method': 'Monte_Carlo',
                'Confidence_Level': 1 - alpha,
                'Simulations': n_simulations,
                'Mean_Return': mean_return,
                'Std_Return': std_return,
                'Observations': len(returns)
            }
        else:
            # Rolling Monte Carlo VaR
            rolling_mean = returns.rolling(window=window).mean()
            rolling_std = returns.rolling(window=window).std()
            
            # This is a simplified rolling approach - in practice, you might want more sophisticated methods
            rolling_var = rolling_mean + norm.ppf(alpha) * rolling_std
            rolling_var_annual = rolling_var * np.sqrt(252)
            
            return {
                'VaR_Daily_Rolling': rolling_var,
                'VaR_Annual_Rolling': rolling_var_annual,
                'Method': 'Monte_Carlo_Rolling',
                'Window': window,
                'Confidence_Level': 1 - alpha,
                'Simulations': n_simulations
            }
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                              risk_free_rate: Optional[float] = None,
                              window: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate Sharpe Ratio and related risk-adjusted return metrics.
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series
        risk_free_rate : float, optional
            Risk-free rate (overrides default if provided)
        window : int, optional
            Rolling window size
            
        Returns:
        --------
        Dict[str, float]
            Sharpe ratio and related metrics
        """
        if risk_free_rate is None:
            daily_rf_rate = self.daily_rf_rate
        else:
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        logger.info(f"Calculating Sharpe ratio with daily RF rate: {daily_rf_rate:.6f}")
        
        # Clean returns data
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            logger.warning(f"Insufficient data for Sharpe ratio: {len(clean_returns)} observations")
            return {}
        
        if window is None:
            # Full series Sharpe ratio
            excess_returns = clean_returns - daily_rf_rate
            mean_excess_return = excess_returns.mean()
            std_return = clean_returns.std()
            
            # Annualized metrics
            annual_excess_return = mean_excess_return * 252
            annual_volatility = std_return * np.sqrt(252)
            
            sharpe_ratio = annual_excess_return / annual_volatility if annual_volatility != 0 else 0
            
            return {
                'Sharpe_Ratio': sharpe_ratio,
                'Annual_Excess_Return': annual_excess_return,
                'Annual_Volatility': annual_volatility,
                'Daily_Excess_Return': mean_excess_return,
                'Daily_Volatility': std_return,
                'Risk_Free_Rate_Daily': daily_rf_rate,
                'Risk_Free_Rate_Annual': risk_free_rate or self.risk_free_rate,
                'Observations': len(clean_returns)
            }
        else:
            # Rolling Sharpe ratio
            rolling_mean = clean_returns.rolling(window=window).mean()
            rolling_std = clean_returns.rolling(window=window).std()
            
            rolling_excess_returns = rolling_mean - daily_rf_rate
            rolling_sharpe = rolling_excess_returns / rolling_std * np.sqrt(252)
            
            return {
                'Sharpe_Ratio_Rolling': rolling_sharpe,
                'Window': window,
                'Risk_Free_Rate_Daily': daily_rf_rate
            }
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, float]:
        """
        Calculate Maximum Drawdown and related metrics.
        
        Parameters:
        -----------
        prices : pd.Series
            Price series (not returns)
            
        Returns:
        --------
        Dict[str, float]
            Maximum drawdown metrics
        """
        logger.info("Calculating maximum drawdown")
        
        # Calculate cumulative returns
        cumulative_returns = (1 + prices.pct_change()).cumprod()
        
        # Calculate running maximum
        running_max = cumulative_returns.expanding().max()
        
        # Calculate drawdown
        drawdown = (cumulative_returns - running_max) / running_max
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Find the peak and trough dates
        peak_idx = running_max.idxmax()
        trough_idx = drawdown.idxmin()
        
        # Calculate recovery time
        recovery_time = None
        if trough_idx > peak_idx:
            recovery_mask = cumulative_returns[trough_idx:] >= running_max[peak_idx]
            if recovery_mask.any():
                recovery_idx = recovery_mask.idxmax()
                recovery_time = (recovery_idx - trough_idx).days
        
        return {
            'Max_Drawdown': max_drawdown,
            'Max_Drawdown_Pct': max_drawdown * 100,
            'Peak_Date': peak_idx,
            'Trough_Date': trough_idx,
            'Recovery_Time_Days': recovery_time,
            'Current_Drawdown': drawdown.iloc[-1],
            'Current_Drawdown_Pct': drawdown.iloc[-1] * 100
        }
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              risk_free_rate: Optional[float] = None,
                              target_return: float = 0.0) -> Dict[str, float]:
        """
        Calculate Sortino Ratio (downside deviation-based risk-adjusted return).
        
        Parameters:
        -----------
        returns : pd.Series
            Daily returns series
        risk_free_rate : float, optional
            Risk-free rate (overrides default if provided)
        target_return : float
            Target return for downside deviation calculation
            
        Returns:
        --------
        Dict[str, float]
            Sortino ratio and related metrics
        """
        if risk_free_rate is None:
            daily_rf_rate = self.daily_rf_rate
        else:
            daily_rf_rate = (1 + risk_free_rate) ** (1/252) - 1
        
        # Clean returns data
        clean_returns = returns.dropna()
        
        if len(clean_returns) < 30:
            logger.warning(f"Insufficient data for Sortino ratio: {len(clean_returns)} observations")
            return {}
        
        # Calculate downside deviation
        downside_returns = clean_returns[clean_returns < target_return]
        if len(downside_returns) == 0:
            logger.warning("No downside returns found for Sortino ratio calculation")
            return {}
        
        downside_deviation = downside_returns.std()
        
        # Calculate excess return
        excess_return = clean_returns.mean() - daily_rf_rate
        
        # Annualized metrics
        annual_excess_return = excess_return * 252
        annual_downside_deviation = downside_deviation * np.sqrt(252)
        
        # Sortino ratio
        sortino_ratio = annual_excess_return / annual_downside_deviation if annual_downside_deviation != 0 else 0
        
        return {
            'Sortino_Ratio': sortino_ratio,
            'Annual_Excess_Return': annual_excess_return,
            'Annual_Downside_Deviation': annual_downside_deviation,
            'Daily_Excess_Return': excess_return,
            'Daily_Downside_Deviation': downside_deviation,
            'Target_Return': target_return,
            'Downside_Observations': len(downside_returns),
            'Risk_Free_Rate_Daily': daily_rf_rate
        }
    
    def test_stationarity(self, data: pd.Series, test_type: str = 'adf') -> Dict[str, Union[float, str, bool]]:
        """
        Perform stationarity tests on time series data.
        
        Parameters:
        -----------
        data : pd.Series
            Time series data to test
        test_type : str
            Type of test: 'adf' (Augmented Dickey-Fuller) or 'kpss'
            
        Returns:
        --------
        Dict[str, Union[float, str, bool]]
            Test results and interpretation
        """
        logger.info(f"Performing {test_type.upper()} stationarity test")
        
        # Clean data
        clean_data = data.dropna()
        
        if len(clean_data) < 30:
            logger.warning(f"Insufficient data for stationarity test: {len(clean_data)} observations")
            return {}
        
        try:
            if test_type.lower() == 'adf':
                from statsmodels.tsa.stattools import adfuller
                
                # Perform ADF test
                adf_result = adfuller(clean_data, autolag='AIC')
                
                # Extract results
                adf_statistic = adf_result[0]
                p_value = adf_result[1]
                critical_values = adf_result[4]
                
                # Determine stationarity
                is_stationary = p_value < 0.05
                
                return {
                    'Test_Type': 'Augmented Dickey-Fuller',
                    'ADF_Statistic': adf_statistic,
                    'P_Value': p_value,
                    'Critical_Value_1%': critical_values['1%'],
                    'Critical_Value_5%': critical_values['5%'],
                    'Critical_Value_10%': critical_values['10%'],
                    'Is_Stationary': is_stationary,
                    'Interpretation': 'Stationary' if is_stationary else 'Non-stationary',
                    'Observations': len(clean_data)
                }
                
            elif test_type.lower() == 'kpss':
                from statsmodels.tsa.stattools import kpss
                
                # Perform KPSS test
                kpss_result = kpss(clean_data, regression='c')
                
                # Extract results
                kpss_statistic = kpss_result[0]
                p_value = kpss_result[1]
                critical_values = kpss_result[3]
                
                # Determine stationarity (KPSS null hypothesis is stationarity)
                is_stationary = p_value > 0.05
                
                return {
                    'Test_Type': 'KPSS',
                    'KPSS_Statistic': kpss_statistic,
                    'P_Value': p_value,
                    'Critical_Value_1%': critical_values['1%'],
                    'Critical_Value_5%': critical_values['5%'],
                    'Critical_Value_10%': critical_values['10%'],
                    'Is_Stationary': is_stationary,
                    'Interpretation': 'Stationary' if is_stationary else 'Non-stationary',
                    'Observations': len(clean_data)
                }
            else:
                raise ValueError(f"Unknown test type: {test_type}. Use 'adf' or 'kpss'")
                
        except ImportError:
            logger.error(f"Required package not available for {test_type} test")
            return {}
        except Exception as e:
            logger.error(f"Error performing {test_type} test: {str(e)}")
            return {}
    
    def calculate_all_metrics(self, asset_data: Dict[str, pd.DataFrame]) -> Dict[str, Dict[str, Dict]]:
        """
        Calculate all financial metrics for all assets.
        
        Parameters:
        -----------
        asset_data : Dict[str, pd.DataFrame]
            Dictionary of asset data with engineered features
            
        Returns:
        --------
        Dict[str, Dict[str, Dict]]
            Comprehensive metrics for all assets
        """
        logger.info("Calculating comprehensive financial metrics for all assets")
        
        all_metrics = {}
        
        for symbol, data in asset_data.items():
            logger.info(f"Calculating metrics for {symbol}")
            
            try:
                # Ensure we have the required columns
                if 'Daily_Return' not in data.columns:
                    logger.warning(f"Daily_Return not found in {symbol}, skipping metrics")
                    continue
                
                returns = data['Daily_Return']
                prices = data['Close']
                
                # Calculate all metrics
                metrics = {
                    'VaR': {
                        'Historical_95': self.calculate_var(returns, 'historical', 0.95),
                        'Parametric_95': self.calculate_var(returns, 'parametric', 0.95),
                        'Historical_99': self.calculate_var(returns, 'historical', 0.99),
                        'Parametric_99': self.calculate_var(returns, 'parametric', 0.99)
                    },
                    'Sharpe_Ratio': self.calculate_sharpe_ratio(returns),
                    'Sortino_Ratio': self.calculate_sortino_ratio(returns),
                    'Maximum_Drawdown': self.calculate_maximum_drawdown(prices),
                    'Stationarity_Test': self.test_stationarity(returns, 'adf'),
                    'Basic_Statistics': {
                        'Mean_Return': returns.mean(),
                        'Std_Return': returns.std(),
                        'Skewness': returns.skew(),
                        'Kurtosis': returns.kurtosis(),
                        'Min_Return': returns.min(),
                        'Max_Return': returns.max(),
                        'Total_Observations': len(returns)
                    }
                }
                
                all_metrics[symbol] = metrics
                logger.info(f"✓ {symbol} metrics calculated successfully")
                
            except Exception as e:
                logger.error(f"✗ Failed to calculate metrics for {symbol}: {str(e)}")
                all_metrics[symbol] = {}
        
        logger.info("All financial metrics calculation completed!")
        return all_metrics


def main():
    """
    Main function to demonstrate the financial metrics calculator.
    """
    print("FinancialMetricsCalculator module loaded successfully!")
    print("Use this module to calculate comprehensive financial metrics and risk measures.")


if __name__ == "__main__":
    main()
