"""
Prophet-based time series forecasting model

Author: Adryan R A
"""

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np

try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    Prophet = None

from .base_model import BaseTimeSeriesModel
from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ProphetModel(BaseTimeSeriesModel):
    """
    Prophet-based time series forecasting model
    """
    
    def __init__(self, country: str):
        """
        Initialize Prophet model
        
        Args:
            country: Country for which the model is trained
        """
        super().__init__(country, "prophet")
        
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet is not installed. Install with: pip install prophet")
        
        # Prophet-specific parameters
        self.prophet_params = {
            'growth': 'linear',
            'yearly_seasonality': True,
            'weekly_seasonality': True,
            'daily_seasonality': False,
            'seasonality_mode': 'additive',
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0,
            'holidays_prior_scale': 10.0,
            'mcmc_samples': 0,
            'interval_width': 0.80,
            'uncertainty_samples': 1000
        }
        
        # Cross-validation parameters
        self.cv_params = {
            'initial': '730 days',  # 2 years
            'period': '90 days',    # 3 months
            'horizon': '180 days'   # 6 months
        }
        
        self.holidays = None
        self.additional_regressors = []
    
    async def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train the Prophet model
        
        Args:
            data: Training data with columns 'ds' (date) and 'y' (target)
            **kwargs: Additional Prophet parameters
            
        Returns:
            Training metrics dictionary
        """
        try:
            logger.info(f"Training Prophet model for {self.country}")
            
            # Prepare data
            train_data = self._prepare_training_data(data)
            
            if len(train_data) < 30:
                raise ValueError(f"Insufficient data for training. Need at least 30 data points, got {len(train_data)}")
            
            # Update Prophet parameters with any provided kwargs
            prophet_params = {**self.prophet_params, **kwargs}
            
            # Initialize Prophet model
            self.model = Prophet(
                growth=prophet_params['growth'],
                yearly_seasonality=prophet_params['yearly_seasonality'],
                weekly_seasonality=prophet_params['weekly_seasonality'],
                daily_seasonality=prophet_params['daily_seasonality'],
                seasonality_mode=prophet_params['seasonality_mode'],
                changepoint_prior_scale=prophet_params['changepoint_prior_scale'],
                seasonality_prior_scale=prophet_params['seasonality_prior_scale'],
                holidays_prior_scale=prophet_params['holidays_prior_scale'],
                mcmc_samples=prophet_params['mcmc_samples'],
                interval_width=prophet_params['interval_width'],
                uncertainty_samples=prophet_params['uncertainty_samples']
            )
            
            # Add holidays if available
            if self.holidays is not None:
                self.model.add_country_holidays(country_name=self.country)
            
            # Add additional regressors if any
            for regressor in self.additional_regressors:
                self.model.add_regressor(regressor)
            
            # Train the model
            self.model.fit(train_data)
            
            # Mark as trained
            self.is_trained = True
            self.created_at = datetime.now()
            self.last_updated = datetime.now()
            
            # Calculate training metrics using cross-validation
            training_metrics = await self._calculate_training_metrics(train_data)
            self.training_metrics = training_metrics
            
            logger.info(f"Prophet model training completed for {self.country}")
            return training_metrics
            
        except Exception as e:
            logger.error(f"Failed to train Prophet model for {self.country}: {e}")
            raise
    
    async def predict(self, 
                     forecast_days: int,
                     confidence_interval: float = 0.95,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate forecasts using Prophet
        
        Args:
            forecast_days: Number of days to forecast
            confidence_interval: Confidence interval for predictions
            **kwargs: Additional prediction parameters
            
        Returns:
            Prediction results with forecasts and confidence intervals
        """
        try:
            # Validate inputs
            self._validate_forecast_inputs(forecast_days, confidence_interval)
            
            logger.info(f"Generating {forecast_days} day forecast for {self.country}")
            
            # Create future dataframe
            future = self.model.make_future_dataframe(periods=forecast_days)
            
            # Add additional regressors to future dataframe if needed
            for regressor in self.additional_regressors:
                if regressor not in future.columns:
                    # For simplicity, forward-fill the last known value
                    if regressor in self.training_data.columns:
                        last_value = self.training_data[regressor].iloc[-1]
                        future[regressor] = last_value
                    else:
                        future[regressor] = 0
            
            # Generate forecast
            forecast = self.model.predict(future)
            
            # Extract forecasts for the requested period
            forecast_start_idx = len(self.training_data)
            forecast_data = forecast.iloc[forecast_start_idx:].copy()
            
            # Prepare response
            predictions = []
            for _, row in forecast_data.iterrows():
                pred_dict = {
                    'date': row['ds'].strftime('%Y-%m-%d'),
                    'forecast': float(row['yhat']),
                    'lower_bound': float(row['yhat_lower']),
                    'upper_bound': float(row['yhat_upper']),
                    'trend': float(row['trend']) if 'trend' in row else None,
                    'seasonal': float(row['yearly'] + row['weekly']) if 'yearly' in row and 'weekly' in row else None
                }
                predictions.append(pred_dict)
            
            # Calculate forecast statistics
            forecast_values = forecast_data['yhat'].values
            forecast_stats = {
                'mean_forecast': float(np.mean(forecast_values)),
                'median_forecast': float(np.median(forecast_values)),
                'min_forecast': float(np.min(forecast_values)),
                'max_forecast': float(np.max(forecast_values)),
                'std_forecast': float(np.std(forecast_values)),
                'total_forecast': float(np.sum(forecast_values))
            }
            
            # Trend analysis
            trend_analysis = self._analyze_trend(forecast_data)
            
            return {
                'model': self.model_name,
                'country': self.country,
                'forecast_days': forecast_days,
                'confidence_interval': confidence_interval,
                'predictions': predictions,
                'statistics': forecast_stats,
                'trend_analysis': trend_analysis,
                'model_info': await self.get_model_info(),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate Prophet forecast for {self.country}: {e}")
            raise
    
    async def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate Prophet model performance on test data
        
        Args:
            test_data: Test data with columns 'ds' and 'y'
            
        Returns:
            Validation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before validation")
            
            logger.info(f"Validating Prophet model for {self.country}")
            
            # Prepare test data
            test_data = test_data.copy()
            test_data['ds'] = pd.to_datetime(test_data['ds'])
            test_data = test_data.sort_values('ds').reset_index(drop=True)
            
            # Generate predictions for test period
            future = test_data[['ds']].copy()
            
            # Add additional regressors if needed
            for regressor in self.additional_regressors:
                if regressor in test_data.columns:
                    future[regressor] = test_data[regressor]
                else:
                    future[regressor] = 0
            
            # Predict
            forecast = self.model.predict(future)
            
            # Calculate metrics
            y_true = test_data['y'].values
            y_pred = forecast['yhat'].values
            
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # Add Prophet-specific metrics
            metrics.update({
                'forecast_bias': float(np.mean(y_pred - y_true)),
                'forecast_variance': float(np.var(y_pred)),
                'coverage': self._calculate_coverage(test_data, forecast)
            })
            
            logger.info(f"Prophet model validation completed for {self.country}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to validate Prophet model for {self.country}: {e}")
            raise
    
    async def _calculate_training_metrics(self, train_data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate training metrics using cross-validation
        
        Args:
            train_data: Training data
            
        Returns:
            Training metrics
        """
        try:
            # Perform cross-validation if data is sufficient
            if len(train_data) >= 1095:  # 3 years of data
                df_cv = cross_validation(
                    self.model,
                    initial=self.cv_params['initial'],
                    period=self.cv_params['period'],
                    horizon=self.cv_params['horizon'],
                    parallel="processes"
                )
                
                # Calculate performance metrics
                df_p = performance_metrics(df_cv)
                
                cv_metrics = {
                    'cv_mae': float(df_p['mae'].mean()),
                    'cv_mape': float(df_p['mape'].mean()),
                    'cv_rmse': float(df_p['rmse'].mean()),
                    'cv_coverage': float(df_p['coverage'].mean())
                }
            else:
                cv_metrics = {'cv_note': 'Insufficient data for cross-validation'}
            
            # In-sample metrics
            forecast = self.model.predict(train_data)
            y_true = train_data['y'].values
            y_pred = forecast['yhat'].values
            
            in_sample_metrics = self._calculate_metrics(y_true, y_pred)
            
            # Combine metrics
            training_metrics = {**in_sample_metrics, **cv_metrics}
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Error calculating training metrics: {e}")
            return {'error': str(e)}
    
    def _analyze_trend(self, forecast_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze forecast trend
        
        Args:
            forecast_data: Forecast results
            
        Returns:
            Trend analysis
        """
        try:
            trend_values = forecast_data['trend'].values if 'trend' in forecast_data.columns else forecast_data['yhat'].values
            
            # Calculate trend slope
            x = np.arange(len(trend_values))
            slope = np.polyfit(x, trend_values, 1)[0]
            
            # Trend direction
            if slope > 0.01:
                direction = 'increasing'
            elif slope < -0.01:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Volatility (coefficient of variation)
            volatility = np.std(trend_values) / (np.mean(trend_values) + 1e-8)
            
            return {
                'direction': direction,
                'slope': float(slope),
                'volatility': float(volatility),
                'trend_strength': 'high' if abs(slope) > 1.0 else 'medium' if abs(slope) > 0.1 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing trend: {e}")
            return {'error': str(e)}
    
    def _calculate_coverage(self, test_data: pd.DataFrame, forecast: pd.DataFrame) -> float:
        """
        Calculate prediction interval coverage
        
        Args:
            test_data: Test data
            forecast: Forecast results
            
        Returns:
            Coverage percentage
        """
        try:
            y_true = test_data['y'].values
            lower = forecast['yhat_lower'].values
            upper = forecast['yhat_upper'].values
            
            # Check if actual values fall within prediction intervals
            within_interval = (y_true >= lower) & (y_true <= upper)
            coverage = np.mean(within_interval) * 100
            
            return float(coverage)
            
        except Exception as e:
            logger.error(f"Error calculating coverage: {e}")
            return 0.0
    
    def add_regressor(self, regressor_name: str, prior_scale: float = 10.0, 
                     standardize: bool = True, mode: str = 'additive'):
        """
        Add external regressor to the model
        
        Args:
            regressor_name: Name of the regressor column
            prior_scale: Prior scale for the regressor
            standardize: Whether to standardize the regressor
            mode: 'additive' or 'multiplicative'
        """
        if regressor_name not in self.additional_regressors:
            self.additional_regressors.append(regressor_name)
            
            if self.model is not None:
                self.model.add_regressor(
                    regressor_name,
                    prior_scale=prior_scale,
                    standardize=standardize,
                    mode=mode
                )
    
    def set_holidays(self, country_name: str = None):
        """
        Set holidays for the model
        
        Args:
            country_name: Country name for built-in holidays
        """
        if country_name:
            self.holidays = country_name
        else:
            self.holidays = self.country
    
    def get_components(self) -> Dict[str, Any]:
        """
        Get model components (trend, seasonality, etc.)
        
        Returns:
            Components analysis
        """
        if not self.is_trained or self.training_data is None:
            return {'error': 'Model not trained or no training data available'}
        
        try:
            # Generate components for training data
            forecast = self.model.predict(self.training_data)
            
            components = {
                'trend': forecast['trend'].tolist(),
                'yearly': forecast['yearly'].tolist() if 'yearly' in forecast.columns else None,
                'weekly': forecast['weekly'].tolist() if 'weekly' in forecast.columns else None,
                'dates': self.training_data['ds'].dt.strftime('%Y-%m-%d').tolist(),
                'actual': self.training_data['y'].tolist(),
                'fitted': forecast['yhat'].tolist()
            }
            
            # Add regressor components if any
            for regressor in self.additional_regressors:
                if regressor in forecast.columns:
                    components[regressor] = forecast[regressor].tolist()
            
            return components
            
        except Exception as e:
            logger.error(f"Error getting model components: {e}")
            return {'error': str(e)}
