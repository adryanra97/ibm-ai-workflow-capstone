"""
Abstract base class for time series models

Author: Adryan R A
"""

import os
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class BaseTimeSeriesModel(ABC):
    """
    Abstract base class for time series forecasting models
    """
    
    def __init__(self, country: str, model_name: str):
        """
        Initialize base model
        
        Args:
            country: Country for which the model is trained
            model_name: Name of the model (e.g., 'prophet', 'arima')
        """
        self.country = country
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        self.training_data = None
        self.training_metrics = {}
        self.model_version = "1.0.0"
        self.created_at = None
        self.last_updated = None
        
        # Model file path
        self.model_path = os.path.join(
            settings.model_storage_path,
            f"{self.model_name}_{country.lower().replace(' ', '_')}.pkl"
        )
    
    @abstractmethod
    async def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train the time series model
        
        Args:
            data: Training data with columns 'ds' (date) and 'y' (target)
            **kwargs: Additional model-specific parameters
            
        Returns:
            Training metrics dictionary
        """
        pass
    
    @abstractmethod
    async def predict(self, 
                     forecast_days: int,
                     confidence_interval: float = 0.95,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate forecasts
        
        Args:
            forecast_days: Number of days to forecast
            confidence_interval: Confidence interval for predictions
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results with forecasts and confidence intervals
        """
        pass
    
    @abstractmethod
    async def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate model performance on test data
        
        Args:
            test_data: Test data with columns 'ds' and 'y'
            
        Returns:
            Validation metrics
        """
        pass
    
    async def save_model(self) -> bool:
        """
        Save trained model to disk
        
        Returns:
            Success status
        """
        try:
            if not self.is_trained:
                logger.warning(f"Cannot save untrained model: {self.model_name} for {self.country}")
                return False
            
            # Create model directory if it doesn't exist
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            
            # Model metadata
            model_data = {
                'model': self.model,
                'country': self.country,
                'model_name': self.model_name,
                'model_version': self.model_version,
                'is_trained': self.is_trained,
                'training_metrics': self.training_metrics,
                'created_at': self.created_at,
                'last_updated': datetime.now(),
                'training_data_shape': self.training_data.shape if self.training_data is not None else None
            }
            
            # Save model
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            
            logger.info(f"Model saved: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save model {self.model_name} for {self.country}: {e}")
            return False
    
    async def load_model(self) -> bool:
        """
        Load trained model from disk
        
        Returns:
            Success status
        """
        try:
            if not os.path.exists(self.model_path):
                logger.info(f"No saved model found: {self.model_path}")
                return False
            
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
            
            # Restore model state
            self.model = model_data['model']
            self.country = model_data['country']
            self.model_name = model_data['model_name']
            self.model_version = model_data.get('model_version', '1.0.0')
            self.is_trained = model_data.get('is_trained', False)
            self.training_metrics = model_data.get('training_metrics', {})
            self.created_at = model_data.get('created_at')
            self.last_updated = model_data.get('last_updated')
            
            logger.info(f"Model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {self.model_name} for {self.country}: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information
        
        Returns:
            Model information dictionary
        """
        return {
            'country': self.country,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'is_trained': self.is_trained,
            'training_metrics': self.training_metrics,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'model_file_exists': os.path.exists(self.model_path),
            'training_data_shape': self.training_data.shape if self.training_data is not None else None
        }
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate common evaluation metrics
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Metrics dictionary
        """
        try:
            # Remove any NaN values
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            
            if len(y_true_clean) == 0:
                return {'error': 'No valid data points for metric calculation'}
            
            # Calculate metrics
            mae = mean_absolute_error(y_true_clean, y_pred_clean)
            rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
            
            # MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((y_true_clean - y_pred_clean) / (y_true_clean + 1e-8))) * 100
            
            # R-squared
            ss_res = np.sum((y_true_clean - y_pred_clean) ** 2)
            ss_tot = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            
            # MASE (Mean Absolute Scaled Error) using naive forecast
            if len(y_true_clean) > 1:
                naive_forecast = y_true_clean[:-1]
                mae_naive = mean_absolute_error(y_true_clean[1:], naive_forecast)
                mase = mae / (mae_naive + 1e-8)
            else:
                mase = float('inf')
            
            return {
                'mae': float(mae),
                'rmse': float(rmse),
                'mape': float(mape),
                'r2': float(r2),
                'mase': float(mase)
            }
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {'error': str(e)}
    
    def _prepare_training_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for training (common preprocessing)
        
        Args:
            data: Input data
            
        Returns:
            Prepared data
        """
        try:
            # Ensure required columns exist
            if 'ds' not in data.columns or 'y' not in data.columns:
                raise ValueError("Data must contain 'ds' and 'y' columns")
            
            # Sort by date
            data = data.sort_values('ds').reset_index(drop=True)
            
            # Remove any duplicate dates (keep last)
            data = data.drop_duplicates(subset=['ds'], keep='last')
            
            # Ensure datetime format
            data['ds'] = pd.to_datetime(data['ds'])
            
            # Remove any rows with missing target values
            data = data.dropna(subset=['y'])
            
            # Store training data reference
            self.training_data = data.copy()
            
            return data
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def _split_train_test(self, data: pd.DataFrame, test_ratio: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split data into train and test sets (time series split)
        
        Args:
            data: Input data
            test_ratio: Proportion of data for testing
            
        Returns:
            Tuple of (train_data, test_data)
        """
        try:
            n_total = len(data)
            n_test = int(n_total * test_ratio)
            
            if n_test < 1:
                # If data is too small, use all for training
                return data, pd.DataFrame()
            
            split_idx = n_total - n_test
            train_data = data.iloc[:split_idx].copy()
            test_data = data.iloc[split_idx:].copy()
            
            return train_data, test_data
            
        except Exception as e:
            logger.error(f"Error splitting train/test data: {e}")
            return data, pd.DataFrame()
    
    def _validate_forecast_inputs(self, 
                                 forecast_days: int,
                                 confidence_interval: float) -> None:
        """
        Validate forecast input parameters
        
        Args:
            forecast_days: Number of days to forecast
            confidence_interval: Confidence interval
            
        Raises:
            ValueError: If inputs are invalid
        """
        if not self.is_trained:
            raise ValueError(f"Model {self.model_name} for {self.country} is not trained")
        
        if forecast_days <= 0:
            raise ValueError("Forecast days must be positive")
        
        if forecast_days > settings.max_forecast_days:
            raise ValueError(f"Forecast days cannot exceed {settings.max_forecast_days}")
        
        if not 0.5 <= confidence_interval <= 0.99:
            raise ValueError("Confidence interval must be between 0.5 and 0.99")
    
    def _generate_future_dates(self, start_date: date, periods: int) -> pd.DataFrame:
        """
        Generate future dates for forecasting
        
        Args:
            start_date: Start date for forecasting
            periods: Number of periods to generate
            
        Returns:
            DataFrame with future dates
        """
        try:
            future_dates = pd.date_range(
                start=start_date,
                periods=periods,
                freq='D'
            )
            
            future_df = pd.DataFrame({'ds': future_dates})
            return future_df
            
        except Exception as e:
            logger.error(f"Error generating future dates: {e}")
            raise
