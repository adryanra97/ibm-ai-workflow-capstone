"""
ARIMA/SARIMA-based time series forecasting model

Author: Adryan R A
"""

import logging
import warnings
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np

try:
    import statsmodels.api as sm
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    ARIMA = None
    SARIMAX = None

from .base_model import BaseTimeSeriesModel
from utils.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Suppress statsmodels warnings
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')


class ARIMAModel(BaseTimeSeriesModel):
    """
    ARIMA/SARIMA-based time series forecasting model
    """
    
    def __init__(self, country: str):
        """
        Initialize ARIMA model
        
        Args:
            country: Country for which the model is trained
        """
        super().__init__(country, "arima")
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is not installed. Install with: pip install statsmodels")
        
        # ARIMA parameters
        self.arima_order = None  # (p, d, q)
        self.seasonal_order = None  # (P, D, Q, s)
        self.use_seasonal = True
        self.seasonal_period = 7  # Weekly seasonality for daily data
        
        # Model selection parameters
        self.max_p = 5
        self.max_d = 2
        self.max_q = 5
        self.max_P = 2
        self.max_D = 1
        self.max_Q = 2
        
        # Auto-selection criteria
        self.information_criterion = 'aic'  # 'aic', 'bic', 'hqic'
        
        # Preprocessing
        self.is_differenced = False
        self.differencing_order = 0
        self.log_transformed = False
        self.original_scale_params = None
        
        # Model diagnostics
        self.model_diagnostics = {}
    
    async def train(self, data: pd.DataFrame, **kwargs) -> Dict[str, float]:
        """
        Train the ARIMA model
        
        Args:
            data: Training data with columns 'ds' (date) and 'y' (target)
            **kwargs: Additional ARIMA parameters
            
        Returns:
            Training metrics dictionary
        """
        try:
            logger.info(f"Training ARIMA model for {self.country}")
            
            # Prepare data
            train_data = self._prepare_training_data(data)
            
            if len(train_data) < 50:
                raise ValueError(f"Insufficient data for ARIMA training. Need at least 50 data points, got {len(train_data)}")
            
            # Extract time series
            ts_data = train_data.set_index('ds')['y']
            
            # Preprocess the data
            processed_data, preprocessing_info = self._preprocess_data(ts_data)
            
            # Auto-select ARIMA parameters if not provided
            if 'arima_order' in kwargs:
                self.arima_order = kwargs['arima_order']
            else:
                self.arima_order = await self._auto_select_arima_order(processed_data)
            
            if 'seasonal_order' in kwargs:
                self.seasonal_order = kwargs['seasonal_order']
            else:
                if self.use_seasonal:
                    self.seasonal_order = await self._auto_select_seasonal_order(processed_data)
                else:
                    self.seasonal_order = (0, 0, 0, 0)
            
            # Train the model
            if self.seasonal_order == (0, 0, 0, 0):
                # Pure ARIMA model
                self.model = ARIMA(processed_data, order=self.arima_order)
            else:
                # SARIMA model
                self.model = SARIMAX(
                    processed_data,
                    order=self.arima_order,
                    seasonal_order=self.seasonal_order,
                    enforce_stationarity=False,
                    enforce_invertibility=False
                )
            
            # Fit the model
            self.model = self.model.fit(disp=False)
            
            # Mark as trained
            self.is_trained = True
            self.created_at = datetime.now()
            self.last_updated = datetime.now()
            
            # Store preprocessing information
            self.original_scale_params = preprocessing_info
            
            # Calculate training metrics
            training_metrics = await self._calculate_training_metrics(processed_data)
            
            # Model diagnostics
            self.model_diagnostics = self._run_model_diagnostics()
            training_metrics.update(self.model_diagnostics)
            
            self.training_metrics = training_metrics
            
            logger.info(f"ARIMA model training completed for {self.country}")
            logger.info(f"Final model: ARIMA{self.arima_order} x SARIMA{self.seasonal_order}")
            
            return training_metrics
            
        except Exception as e:
            logger.error(f"Failed to train ARIMA model for {self.country}: {e}")
            raise
    
    async def predict(self, 
                     forecast_days: int,
                     confidence_interval: float = 0.95,
                     **kwargs) -> Dict[str, Any]:
        """
        Generate forecasts using ARIMA
        
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
            
            logger.info(f"Generating {forecast_days} day ARIMA forecast for {self.country}")
            
            # Generate forecast
            forecast_result = self.model.get_forecast(steps=forecast_days)
            forecast_values = forecast_result.predicted_mean
            
            # Get confidence intervals
            alpha = 1 - confidence_interval
            conf_int = forecast_result.conf_int(alpha=alpha)
            
            # Transform back to original scale if needed
            if self.log_transformed:
                forecast_values = np.exp(forecast_values)
                conf_int = np.exp(conf_int)
            
            # Generate future dates
            last_date = self.training_data['ds'].max()
            future_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=forecast_days,
                freq='D'
            )
            
            # Prepare response
            predictions = []
            for i, date_val in enumerate(future_dates):
                pred_dict = {
                    'date': date_val.strftime('%Y-%m-%d'),
                    'forecast': float(forecast_values.iloc[i]),
                    'lower_bound': float(conf_int.iloc[i, 0]),
                    'upper_bound': float(conf_int.iloc[i, 1])
                }
                predictions.append(pred_dict)
            
            # Calculate forecast statistics
            forecast_stats = {
                'mean_forecast': float(np.mean(forecast_values)),
                'median_forecast': float(np.median(forecast_values)),
                'min_forecast': float(np.min(forecast_values)),
                'max_forecast': float(np.max(forecast_values)),
                'std_forecast': float(np.std(forecast_values)),
                'total_forecast': float(np.sum(forecast_values))
            }
            
            # Trend analysis
            trend_analysis = self._analyze_forecast_trend(forecast_values)
            
            return {
                'model': self.model_name,
                'country': self.country,
                'forecast_days': forecast_days,
                'confidence_interval': confidence_interval,
                'arima_order': self.arima_order,
                'seasonal_order': self.seasonal_order,
                'predictions': predictions,
                'statistics': forecast_stats,
                'trend_analysis': trend_analysis,
                'model_info': await self.get_model_info(),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate ARIMA forecast for {self.country}: {e}")
            raise
    
    async def validate(self, test_data: pd.DataFrame) -> Dict[str, float]:
        """
        Validate ARIMA model performance on test data
        
        Args:
            test_data: Test data with columns 'ds' and 'y'
            
        Returns:
            Validation metrics
        """
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before validation")
            
            logger.info(f"Validating ARIMA model for {self.country}")
            
            # Prepare test data
            test_data = test_data.copy()
            test_data['ds'] = pd.to_datetime(test_data['ds'])
            test_data = test_data.sort_values('ds').reset_index(drop=True)
            
            # Extract time series
            test_ts = test_data.set_index('ds')['y']
            
            # Apply same preprocessing as training
            if self.log_transformed:
                test_ts = np.log(test_ts + 1)
            
            # Generate predictions
            n_test = len(test_ts)
            forecast_result = self.model.get_forecast(steps=n_test)
            y_pred = forecast_result.predicted_mean.values
            
            # Transform back if needed
            if self.log_transformed:
                y_pred = np.exp(y_pred)
                y_true = test_data['y'].values
            else:
                y_true = test_ts.values
            
            # Calculate metrics
            metrics = self._calculate_metrics(y_true, y_pred)
            
            # Add ARIMA-specific metrics
            residuals = y_true - y_pred
            metrics.update({
                'forecast_bias': float(np.mean(residuals)),
                'forecast_variance': float(np.var(y_pred)),
                'residual_autocorr': self._check_residual_autocorrelation(residuals)
            })
            
            logger.info(f"ARIMA model validation completed for {self.country}")
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to validate ARIMA model for {self.country}: {e}")
            raise
    
    def _preprocess_data(self, ts_data: pd.Series) -> Tuple[pd.Series, Dict]:
        """
        Preprocess time series data for ARIMA modeling
        
        Args:
            ts_data: Input time series
            
        Returns:
            Tuple of (processed_data, preprocessing_info)
        """
        try:
            preprocessing_info = {}
            processed_data = ts_data.copy()
            
            # Handle missing values
            if processed_data.isnull().any():
                processed_data = processed_data.interpolate(method='linear')
                preprocessing_info['missing_values_interpolated'] = True
            
            # Check for negative values and apply log transformation if all positive
            if (processed_data > 0).all():
                self.log_transformed = True
                processed_data = np.log(processed_data + 1)
                preprocessing_info['log_transformed'] = True
            
            # Check stationarity
            stationarity_test = self._test_stationarity(processed_data)
            preprocessing_info['initial_stationarity'] = stationarity_test
            
            # Apply differencing if needed
            if not stationarity_test['is_stationary']:
                processed_data, diff_order = self._make_stationary(processed_data)
                self.differencing_order = diff_order
                self.is_differenced = True
                preprocessing_info['differencing_order'] = diff_order
            
            return processed_data, preprocessing_info
            
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise
    
    def _test_stationarity(self, ts_data: pd.Series) -> Dict[str, Any]:
        """
        Test time series stationarity using ADF and KPSS tests
        
        Args:
            ts_data: Time series data
            
        Returns:
            Stationarity test results
        """
        try:
            # Augmented Dickey-Fuller test
            adf_result = adfuller(ts_data.dropna())
            adf_stationary = adf_result[1] < 0.05  # p-value < 0.05
            
            # KPSS test
            kpss_result = kpss(ts_data.dropna(), regression='ct')
            kpss_stationary = kpss_result[1] > 0.05  # p-value > 0.05
            
            # Combined decision
            is_stationary = adf_stationary and kpss_stationary
            
            return {
                'is_stationary': is_stationary,
                'adf_pvalue': adf_result[1],
                'kpss_pvalue': kpss_result[1],
                'adf_stationary': adf_stationary,
                'kpss_stationary': kpss_stationary
            }
            
        except Exception as e:
            logger.error(f"Error testing stationarity: {e}")
            return {'is_stationary': False, 'error': str(e)}
    
    def _make_stationary(self, ts_data: pd.Series, max_diff: int = 2) -> Tuple[pd.Series, int]:
        """
        Make time series stationary through differencing
        
        Args:
            ts_data: Time series data
            max_diff: Maximum differencing order
            
        Returns:
            Tuple of (stationary_data, differencing_order)
        """
        try:
            current_data = ts_data.copy()
            diff_order = 0
            
            for d in range(1, max_diff + 1):
                current_data = current_data.diff().dropna()
                diff_order = d
                
                # Test stationarity
                stationarity_test = self._test_stationarity(current_data)
                
                if stationarity_test['is_stationary']:
                    break
            
            return current_data, diff_order
            
        except Exception as e:
            logger.error(f"Error making series stationary: {e}")
            return ts_data, 0
    
    async def _auto_select_arima_order(self, ts_data: pd.Series) -> Tuple[int, int, int]:
        """
        Auto-select ARIMA order using information criteria
        
        Args:
            ts_data: Preprocessed time series data
            
        Returns:
            Optimal ARIMA order (p, d, q)
        """
        try:
            best_aic = np.inf
            best_order = (1, 0, 1)
            
            logger.info("Auto-selecting ARIMA order...")
            
            for p in range(self.max_p + 1):
                for d in range(self.max_d + 1):
                    for q in range(self.max_q + 1):
                        try:
                            model = ARIMA(ts_data, order=(p, d, q))
                            fitted_model = model.fit(disp=False)
                            
                            if self.information_criterion == 'aic':
                                criterion = fitted_model.aic
                            elif self.information_criterion == 'bic':
                                criterion = fitted_model.bic
                            else:  # hqic
                                criterion = fitted_model.hqic
                            
                            if criterion < best_aic:
                                best_aic = criterion
                                best_order = (p, d, q)
                                
                        except Exception:
                            continue
            
            logger.info(f"Selected ARIMA order: {best_order} with {self.information_criterion.upper()}={best_aic:.2f}")
            return best_order
            
        except Exception as e:
            logger.error(f"Error auto-selecting ARIMA order: {e}")
            return (1, 1, 1)  # Default fallback
    
    async def _auto_select_seasonal_order(self, ts_data: pd.Series) -> Tuple[int, int, int, int]:
        """
        Auto-select seasonal ARIMA order
        
        Args:
            ts_data: Preprocessed time series data
            
        Returns:
            Optimal seasonal order (P, D, Q, s)
        """
        try:
            if len(ts_data) < self.seasonal_period * 2:
                return (0, 0, 0, 0)  # No seasonality for short series
            
            best_aic = np.inf
            best_order = (0, 0, 0, self.seasonal_period)
            
            logger.info("Auto-selecting seasonal order...")
            
            for P in range(self.max_P + 1):
                for D in range(self.max_D + 1):
                    for Q in range(self.max_Q + 1):
                        try:
                            model = SARIMAX(
                                ts_data,
                                order=self.arima_order,
                                seasonal_order=(P, D, Q, self.seasonal_period),
                                enforce_stationarity=False,
                                enforce_invertibility=False
                            )
                            fitted_model = model.fit(disp=False)
                            
                            if self.information_criterion == 'aic':
                                criterion = fitted_model.aic
                            elif self.information_criterion == 'bic':
                                criterion = fitted_model.bic
                            else:  # hqic
                                criterion = fitted_model.hqic
                            
                            if criterion < best_aic:
                                best_aic = criterion
                                best_order = (P, D, Q, self.seasonal_period)
                                
                        except Exception:
                            continue
            
            logger.info(f"Selected seasonal order: {best_order} with {self.information_criterion.upper()}={best_aic:.2f}")
            return best_order
            
        except Exception as e:
            logger.error(f"Error auto-selecting seasonal order: {e}")
            return (0, 0, 0, self.seasonal_period)
    
    async def _calculate_training_metrics(self, processed_data: pd.Series) -> Dict[str, float]:
        """
        Calculate training metrics for ARIMA model
        
        Args:
            processed_data: Processed training data
            
        Returns:
            Training metrics
        """
        try:
            # In-sample fit
            fitted_values = self.model.fittedvalues
            
            # Transform back to original scale if needed
            if self.log_transformed:
                fitted_values = np.exp(fitted_values)
                actual_values = np.exp(processed_data)
            else:
                actual_values = processed_data
            
            # Align data (fitted values might be shorter due to differencing)
            min_len = min(len(fitted_values), len(actual_values))
            fitted_aligned = fitted_values[-min_len:]
            actual_aligned = actual_values[-min_len:]
            
            # Calculate metrics
            metrics = self._calculate_metrics(actual_aligned.values, fitted_aligned.values)
            
            # Add model information criteria
            metrics.update({
                'aic': float(self.model.aic),
                'bic': float(self.model.bic),
                'hqic': float(self.model.hqic),
                'llf': float(self.model.llf)
            })
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating training metrics: {e}")
            return {'error': str(e)}
    
    def _run_model_diagnostics(self) -> Dict[str, Any]:
        """
        Run model diagnostic tests
        
        Returns:
            Diagnostic test results
        """
        try:
            diagnostics = {}
            
            # Residuals
            residuals = self.model.resid
            
            # Ljung-Box test for residual autocorrelation
            lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
            diagnostics['ljung_box_pvalue'] = float(lb_test['lb_pvalue'].min())
            diagnostics['residuals_autocorrelated'] = diagnostics['ljung_box_pvalue'] < 0.05
            
            # Residual statistics
            diagnostics['residual_mean'] = float(np.mean(residuals))
            diagnostics['residual_std'] = float(np.std(residuals))
            diagnostics['residual_skewness'] = float(residuals.skew())
            diagnostics['residual_kurtosis'] = float(residuals.kurtosis())
            
            # Model convergence
            diagnostics['converged'] = getattr(self.model, 'mle_retvals', {}).get('converged', True)
            
            return diagnostics
            
        except Exception as e:
            logger.error(f"Error running model diagnostics: {e}")
            return {'diagnostics_error': str(e)}
    
    def _analyze_forecast_trend(self, forecast_values: pd.Series) -> Dict[str, Any]:
        """
        Analyze forecast trend
        
        Args:
            forecast_values: Forecast values
            
        Returns:
            Trend analysis
        """
        try:
            values = forecast_values.values
            
            # Calculate trend slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Trend direction
            if slope > 0.01:
                direction = 'increasing'
            elif slope < -0.01:
                direction = 'decreasing'
            else:
                direction = 'stable'
            
            # Volatility
            volatility = np.std(values) / (np.mean(values) + 1e-8)
            
            return {
                'direction': direction,
                'slope': float(slope),
                'volatility': float(volatility),
                'trend_strength': 'high' if abs(slope) > 1.0 else 'medium' if abs(slope) > 0.1 else 'low'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing forecast trend: {e}")
            return {'error': str(e)}
    
    def _check_residual_autocorrelation(self, residuals: np.ndarray) -> float:
        """
        Check residual autocorrelation
        
        Args:
            residuals: Model residuals
            
        Returns:
            Autocorrelation significance (p-value)
        """
        try:
            # Ljung-Box test
            lb_test = acorr_ljungbox(residuals, lags=min(10, len(residuals)//4), return_df=True)
            return float(lb_test['lb_pvalue'].min())
            
        except Exception as e:
            logger.error(f"Error checking residual autocorrelation: {e}")
            return 1.0
    
    def get_model_summary(self) -> Dict[str, Any]:
        """
        Get detailed model summary
        
        Returns:
            Model summary information
        """
        if not self.is_trained:
            return {'error': 'Model not trained'}
        
        try:
            summary_info = {
                'arima_order': self.arima_order,
                'seasonal_order': self.seasonal_order,
                'aic': float(self.model.aic),
                'bic': float(self.model.bic),
                'hqic': float(self.model.hqic),
                'log_likelihood': float(self.model.llf),
                'preprocessing': {
                    'log_transformed': self.log_transformed,
                    'differencing_order': self.differencing_order,
                    'is_differenced': self.is_differenced
                },
                'diagnostics': self.model_diagnostics,
                'model_summary': str(self.model.summary())
            }
            
            return summary_info
            
        except Exception as e:
            logger.error(f"Error getting model summary: {e}")
            return {'error': str(e)}
