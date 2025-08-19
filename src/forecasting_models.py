"""
Forecasting Models Module

This module implements time series forecasting models including ARIMA, SARIMA, and LSTM
for financial data prediction and portfolio optimization.

Author: Financial Analytics Team
Date: August 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
import warnings
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

class ForecastingModels:
    """
    A comprehensive class for time series forecasting using multiple methodologies.
    
    This class implements:
    - ARIMA/SARIMA models for linear patterns
    - LSTM models for non-linear dependencies
    - Model evaluation and comparison
    - Forecast visualization and interpretation
    """
    
    def __init__(self):
        """Initialize the ForecastingModels class."""
        self.models = {}
        self.forecasts = {}
        self.model_performance = {}
        
        logger.info("ForecastingModels initialized")
    
    def prepare_data_for_forecasting(self, data: pd.DataFrame, target_column: str = 'Close') -> Tuple[pd.Series, pd.Series]:
        """
        Prepare data for forecasting by splitting into train and test sets.
        
        Parameters:
        -----------
        data : pd.DataFrame
            Input data with time series
        target_column : str
            Column to forecast
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            Training and testing data
        """
        logger.info(f"Preparing data for forecasting on {target_column}")
        
        # Use 80% for training, 20% for testing
        split_point = int(len(data) * 0.8)
        
        train_data = data[target_column].iloc[:split_point]
        test_data = data[target_column].iloc[split_point:]
        
        logger.info(f"Training data: {len(train_data)} points, Testing data: {len(test_data)} points")
        
        return train_data, test_data
    
    def fit_arima_model(self, train_data: pd.Series, order: Tuple[int, int, int] = (1, 1, 1)) -> Dict:
        """
        Fit ARIMA model to training data.
        
        Parameters:
        -----------
        train_data : pd.Series
            Training data
        order : Tuple[int, int, int]
            ARIMA order (p, d, q)
            
        Returns:
        --------
        Dict
            Model results and metadata
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            logger.info(f"Fitting ARIMA{order} model")
            
            model = ARIMA(train_data, order=order)
            fitted_model = model.fit()
            
            # Generate forecasts
            forecast_steps = len(train_data) // 5  # Forecast 20% of training data length
            forecast = fitted_model.forecast(steps=forecast_steps)
            forecast_ci = fitted_model.get_forecast(steps=forecast_steps).conf_int()
            
            results = {
                'model': fitted_model,
                'forecast': forecast,
                'confidence_intervals': forecast_ci,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'order': order,
                'model_type': 'ARIMA'
            }
            
            logger.info(f"ARIMA model fitted successfully. AIC: {fitted_model.aic:.2f}")
            return results
            
        except ImportError:
            logger.error("statsmodels not available for ARIMA modeling")
            return {}
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {str(e)}")
            return {}
    
    def fit_lstm_model(self, train_data: pd.Series, sequence_length: int = 60, epochs: int = 50) -> Dict:
        """
        Fit LSTM model to training data.
        
        Parameters:
        -----------
        train_data : pd.Series
            Training data
        sequence_length : int
            Length of input sequences
        epochs : int
            Number of training epochs
            
        Returns:
        --------
        Dict
            Model results and metadata
        """
        try:
            import tensorflow as tf
            from tensorflow.keras.models import Sequential
            from tensorflow.keras.layers import LSTM, Dense, Dropout
            
            logger.info(f"Fitting LSTM model with sequence length {sequence_length}")
            
            # Prepare data for LSTM
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(train_data.values.reshape(-1, 1))
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(scaled_data)):
                X.append(scaled_data[i-sequence_length:i, 0])
                y.append(scaled_data[i, 0])
            
            X, y = np.array(X), np.array(y)
            X = np.reshape(X, (X.shape[0], X.shape[1], 1))
            
            # Build LSTM model
            model = Sequential([
                LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)),
                Dropout(0.2),
                LSTM(units=50, return_sequences=True),
                Dropout(0.2),
                LSTM(units=50),
                Dropout(0.2),
                Dense(units=1)
            ])
            
            model.compile(optimizer='adam', loss='mean_squared_error')
            
            # Train model
            history = model.fit(X, y, epochs=epochs, batch_size=32, verbose=0)
            
            # Generate forecasts
            forecast_steps = len(train_data) // 5
            last_sequence = scaled_data[-sequence_length:]
            forecasts = []
            
            for _ in range(forecast_steps):
                next_pred = model.predict(last_sequence.reshape(1, sequence_length, 1), verbose=0)
                forecasts.append(next_pred[0, 0])
                last_sequence = np.append(last_sequence[1:], next_pred)
            
            # Inverse transform forecasts
            forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
            
            results = {
                'model': model,
                'scaler': scaler,
                'forecast': pd.Series(forecasts),
                'training_loss': history.history['loss'][-1],
                'sequence_length': sequence_length,
                'model_type': 'LSTM'
            }
            
            logger.info(f"LSTM model fitted successfully. Final loss: {history.history['loss'][-1]:.6f}")
            return results
            
        except ImportError:
            logger.error("TensorFlow not available for LSTM modeling")
            return {}
        except Exception as e:
            logger.error(f"Error fitting LSTM model: {str(e)}")
            return {}
    
    def evaluate_model_performance(self, actual: pd.Series, predicted: pd.Series, model_name: str) -> Dict:
        """
        Evaluate model performance using multiple metrics.
        
        Parameters:
        -----------
        actual : pd.Series
            Actual values
        predicted : pd.Series
            Predicted values
        model_name : str
            Name of the model
            
        Returns:
        --------
        Dict
            Performance metrics
        """
        # Align series for comparison
        min_length = min(len(actual), len(predicted))
        actual_aligned = actual.iloc[:min_length]
        predicted_aligned = predicted.iloc[:min_length] if hasattr(predicted, 'iloc') else predicted[:min_length]
        
        # Calculate metrics
        mse = mean_squared_error(actual_aligned, predicted_aligned)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(actual_aligned, predicted_aligned)
        mape = np.mean(np.abs((actual_aligned - predicted_aligned) / actual_aligned)) * 100
        
        # Direction accuracy
        actual_direction = np.sign(actual_aligned.diff().dropna())
        predicted_direction = np.sign(pd.Series(predicted_aligned).diff().dropna())
        direction_accuracy = np.mean(actual_direction == predicted_direction) * 100
        
        metrics = {
            'model_name': model_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'direction_accuracy': direction_accuracy,
            'observations': len(actual_aligned)
        }
        
        logger.info(f"{model_name} Performance - RMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%")
        
        return metrics
    
    def create_forecast_visualization(self, train_data: pd.Series, test_data: pd.Series, 
                                    forecasts: Dict, save_path: str = None) -> None:
        """
        Create comprehensive forecast visualization.
        
        Parameters:
        -----------
        train_data : pd.Series
            Training data
        test_data : pd.Series
            Testing data
        forecasts : Dict
            Dictionary of model forecasts
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=(15, 10))
        
        # Plot training data
        plt.subplot(2, 1, 1)
        plt.plot(train_data.index, train_data.values, label='Training Data', color='blue', alpha=0.7)
        plt.plot(test_data.index, test_data.values, label='Actual Test Data', color='green', linewidth=2)
        
        colors = ['red', 'orange', 'purple', 'brown']
        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
            if 'forecast' in forecast_data:
                forecast = forecast_data['forecast']
                # Create forecast index
                forecast_index = pd.date_range(start=test_data.index[0], periods=len(forecast), freq='D')
                plt.plot(forecast_index, forecast, label=f'{model_name} Forecast', 
                        color=colors[i % len(colors)], linestyle='--', linewidth=2)
        
        plt.title('Time Series Forecasting Comparison')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot residuals
        plt.subplot(2, 1, 2)
        for i, (model_name, forecast_data) in enumerate(forecasts.items()):
            if 'forecast' in forecast_data:
                forecast = forecast_data['forecast']
                min_length = min(len(test_data), len(forecast))
                residuals = test_data.iloc[:min_length] - forecast[:min_length]
                plt.plot(residuals, label=f'{model_name} Residuals', alpha=0.7)
        
        plt.title('Forecast Residuals')
        plt.xlabel('Time')
        plt.ylabel('Residual')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Forecast visualization saved to {save_path}")
        
        plt.show()
    
    def generate_forecast_report(self, performance_metrics: List[Dict], save_path: str = None) -> str:
        """
        Generate a comprehensive forecast report.
        
        Parameters:
        -----------
        performance_metrics : List[Dict]
            List of performance metrics for each model
        save_path : str, optional
            Path to save the report
            
        Returns:
        --------
        str
            Formatted report
        """
        report = []
        report.append("="*80)
        report.append("TIME SERIES FORECASTING PERFORMANCE REPORT")
        report.append("="*80)
        report.append("")
        
        # Create performance comparison table
        df_metrics = pd.DataFrame(performance_metrics)
        df_metrics = df_metrics.round(4)
        
        report.append("MODEL PERFORMANCE COMPARISON")
        report.append("-"*50)
        report.append(df_metrics.to_string(index=False))
        report.append("")
        
        # Best model analysis
        if not df_metrics.empty:
            best_rmse_model = df_metrics.loc[df_metrics['rmse'].idxmin(), 'model_name']
            best_mape_model = df_metrics.loc[df_metrics['mape'].idxmin(), 'model_name']
            best_direction_model = df_metrics.loc[df_metrics['direction_accuracy'].idxmax(), 'model_name']
            
            report.append("BEST PERFORMING MODELS")
            report.append("-"*30)
            report.append(f"Lowest RMSE: {best_rmse_model}")
            report.append(f"Lowest MAPE: {best_mape_model}")
            report.append(f"Best Direction Accuracy: {best_direction_model}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS")
        report.append("-"*20)
        report.append("• Use ensemble methods combining multiple models for better accuracy")
        report.append("• Consider external factors (market sentiment, economic indicators)")
        report.append("• Implement rolling window validation for robust performance assessment")
        report.append("• Monitor model performance regularly and retrain as needed")
        report.append("")
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            logger.info(f"Forecast report saved to {save_path}")
        
        return report_text


def main():
    """
    Main function to demonstrate the forecasting models functionality.
    """
    print("ForecastingModels module loaded successfully!")
    print("Use this module to implement ARIMA, SARIMA, and LSTM forecasting models.")


if __name__ == "__main__":
    main()
