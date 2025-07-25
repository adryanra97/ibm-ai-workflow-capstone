"""
Model service for orchestrating time series models

Author: Adryan R A
"""

import logging
import asyncio
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np

from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from services.data_service import DataService
from utils.config import get_settings
from utils.validation import validate_country, validate_date, validate_forecast_parameters
from utils.monitoring import ModelMonitor

logger = logging.getLogger(__name__)
settings = get_settings()


class ModelService:
    """
    Service for managing time series forecasting models
    """
    
    def __init__(self):
        """Initialize model service"""
        self.data_service = DataService()
        self.monitor = ModelMonitor()
        self.models = {}  # Cache for loaded models
        
        # Supported model types
        self.supported_models = {
            'prophet': ProphetModel,
            'arima': ARIMAModel
        }
        
        # Model training configurations
        self.training_configs = {
            'prophet': {
                'min_training_days': 30,
                'recommended_training_days': 365,
                'hyperparameters': {
                    'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
                    'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
                    'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0]
                }
            },
            'arima': {
                'min_training_days': 50,
                'recommended_training_days': 180,
                'hyperparameters': {
                    'max_p': [3, 5, 7],
                    'max_q': [3, 5, 7],
                    'seasonal_period': [7, 30]
                }
            }
        }
    
    async def train_model(self, 
                         country: str,
                         model_type: str,
                         start_date: Optional[date] = None,
                         end_date: Optional[date] = None,
                         hyperparameters: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train a time series model for a specific country
        
        Args:
            country: Country name
            model_type: Type of model ('prophet' or 'arima')
            start_date: Start date for training data
            end_date: End date for training data
            hyperparameters: Custom hyperparameters
            
        Returns:
            Training results and metrics
        """
        try:
            # Validate inputs
            validate_country(country)
            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            if start_date:
                validate_date(start_date)
            if end_date:
                validate_date(end_date)
            
            logger.info(f"Starting model training: {model_type} for {country}")
            
            # Get training data
            training_data = await self.data_service.get_training_data(
                country=country,
                start_date=start_date,
                end_date=end_date
            )
            
            if training_data.empty:
                raise ValueError(f"No training data available for {country}")
            
            # Validate training data sufficiency
            min_days = self.training_configs[model_type]['min_training_days']
            if len(training_data) < min_days:
                raise ValueError(
                    f"Insufficient training data for {model_type}. "
                    f"Need at least {min_days} days, got {len(training_data)}"
                )
            
            # Prepare data for model training
            model_data = training_data[['date', 'total_revenue']].copy()
            model_data.columns = ['ds', 'y']
            
            # Initialize model
            model_class = self.supported_models[model_type]
            model = model_class(country)
            
            # Apply hyperparameters if provided
            training_params = hyperparameters or {}
            
            # Train the model
            training_start = datetime.now()
            training_metrics = await model.train(model_data, **training_params)
            training_duration = (datetime.now() - training_start).total_seconds()
            
            # Save the trained model
            save_success = await model.save_model()
            
            # Cache the model
            model_key = f"{model_type}_{country}"
            self.models[model_key] = model
            
            # Log training completion
            await self.monitor.log_training_event(
                country=country,
                model_type=model_type,
                training_metrics=training_metrics,
                training_duration=training_duration,
                data_points=len(training_data)
            )
            
            # Prepare response
            result = {
                'country': country,
                'model_type': model_type,
                'training_status': 'completed',
                'training_duration_seconds': training_duration,
                'training_data_points': len(training_data),
                'training_date_range': {
                    'start': training_data['date'].min().isoformat(),
                    'end': training_data['date'].max().isoformat()
                },
                'metrics': training_metrics,
                'model_saved': save_success,
                'model_info': await model.get_model_info(),
                'trained_at': datetime.now().isoformat()
            }
            
            # Add model-specific information
            if model_type == 'arima':
                arima_info = model.get_model_summary()
                result['arima_details'] = arima_info
            elif model_type == 'prophet':
                prophet_components = model.get_components()
                result['prophet_components'] = prophet_components
            
            logger.info(f"Model training completed successfully: {model_type} for {country}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to train {model_type} model for {country}: {e}")
            
            # Log training failure
            await self.monitor.log_training_event(
                country=country,
                model_type=model_type,
                training_metrics={'error': str(e)},
                training_duration=0,
                data_points=0,
                success=False
            )
            
            raise
    
    async def predict(self,
                     country: str,
                     model_type: str,
                     forecast_days: int = 30,
                     confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Generate predictions using a trained model
        
        Args:
            country: Country name
            model_type: Type of model ('prophet' or 'arima')
            forecast_days: Number of days to forecast
            confidence_interval: Confidence interval for predictions
            
        Returns:
            Prediction results
        """
        try:
            # Validate inputs
            validate_country(country)
            validate_forecast_parameters(forecast_days, confidence_interval)
            
            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Generating predictions: {model_type} for {country}, {forecast_days} days")
            
            # Get or load model
            model = await self._get_model(country, model_type)
            
            if not model.is_trained:
                raise ValueError(f"Model {model_type} for {country} is not trained")
            
            # Generate predictions
            prediction_start = datetime.now()
            predictions = await model.predict(
                forecast_days=forecast_days,
                confidence_interval=confidence_interval
            )
            prediction_duration = (datetime.now() - prediction_start).total_seconds()
            
            # Enhance predictions with business metrics
            enhanced_predictions = await self._enhance_predictions_with_business_metrics(
                predictions, country
            )
            
            # Log prediction event
            await self.monitor.log_prediction_event(
                country=country,
                model_type=model_type,
                forecast_days=forecast_days,
                prediction_duration=prediction_duration,
                mean_forecast=predictions['statistics']['mean_forecast']
            )
            
            # Monitor for potential drift
            drift_analysis = await self.monitor.check_prediction_drift(
                country=country,
                model_type=model_type,
                predictions=predictions['predictions']
            )
            
            enhanced_predictions['drift_analysis'] = drift_analysis
            enhanced_predictions['prediction_duration_seconds'] = prediction_duration
            
            logger.info(f"Predictions generated successfully: {model_type} for {country}")
            return enhanced_predictions
            
        except Exception as e:
            logger.error(f"Failed to generate predictions with {model_type} for {country}: {e}")
            raise
    
    async def compare_models(self,
                           country: str,
                           forecast_days: int = 30,
                           confidence_interval: float = 0.95) -> Dict[str, Any]:
        """
        Compare predictions from multiple models
        
        Args:
            country: Country name
            forecast_days: Number of days to forecast
            confidence_interval: Confidence interval
            
        Returns:
            Model comparison results
        """
        try:
            validate_country(country)
            validate_forecast_parameters(forecast_days, confidence_interval)
            
            logger.info(f"Comparing models for {country}")
            
            comparison_results = {
                'country': country,
                'forecast_days': forecast_days,
                'confidence_interval': confidence_interval,
                'models': {},
                'comparison': {},
                'generated_at': datetime.now().isoformat()
            }
            
            # Generate predictions from all available models
            for model_type in self.supported_models.keys():
                try:
                    model = await self._get_model(country, model_type, load_if_missing=True)
                    
                    if model and model.is_trained:
                        predictions = await model.predict(forecast_days, confidence_interval)
                        comparison_results['models'][model_type] = predictions
                        
                except Exception as e:
                    logger.warning(f"Could not generate predictions with {model_type} for {country}: {e}")
                    comparison_results['models'][model_type] = {'error': str(e)}
            
            # Perform comparison analysis
            if len([m for m in comparison_results['models'].values() if 'error' not in m]) >= 2:
                comparison_analysis = self._analyze_model_comparison(comparison_results['models'])
                comparison_results['comparison'] = comparison_analysis
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Failed to compare models for {country}: {e}")
            raise
    
    async def validate_model(self,
                           country: str,
                           model_type: str,
                           validation_start_date: Optional[date] = None,
                           validation_end_date: Optional[date] = None) -> Dict[str, Any]:
        """
        Validate model performance on held-out data
        
        Args:
            country: Country name
            model_type: Type of model
            validation_start_date: Start date for validation period
            validation_end_date: End date for validation period
            
        Returns:
            Validation results
        """
        try:
            validate_country(country)
            
            if model_type not in self.supported_models:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            logger.info(f"Validating {model_type} model for {country}")
            
            # Get model
            model = await self._get_model(country, model_type)
            
            if not model.is_trained:
                raise ValueError(f"Model {model_type} for {country} is not trained")
            
            # Get validation data
            validation_data = await self.data_service.get_training_data(
                country=country,
                start_date=validation_start_date,
                end_date=validation_end_date
            )
            
            if validation_data.empty:
                raise ValueError(f"No validation data available for {country}")
            
            # Prepare validation data
            model_data = validation_data[['date', 'total_revenue']].copy()
            model_data.columns = ['ds', 'y']
            
            # Validate model
            validation_metrics = await model.validate(model_data)
            
            # Get business metrics for validation period
            business_metrics = await self.data_service.get_business_metrics(
                country=country,
                start_date=validation_start_date,
                end_date=validation_end_date
            )
            
            # Log validation event
            await self.monitor.log_validation_event(
                country=country,
                model_type=model_type,
                validation_metrics=validation_metrics,
                validation_period_days=len(validation_data)
            )
            
            result = {
                'country': country,
                'model_type': model_type,
                'validation_period': {
                    'start': validation_data['date'].min().isoformat(),
                    'end': validation_data['date'].max().isoformat(),
                    'days': len(validation_data)
                },
                'metrics': validation_metrics,
                'business_metrics': business_metrics,
                'model_info': await model.get_model_info(),
                'validated_at': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to validate {model_type} model for {country}: {e}")
            raise
    
    async def get_model_status(self, country: str) -> Dict[str, Any]:
        """
        Get status of all models for a country
        
        Args:
            country: Country name
            
        Returns:
            Model status information
        """
        try:
            validate_country(country)
            
            logger.info(f"Getting model status for {country}")
            
            status = {
                'country': country,
                'models': {},
                'checked_at': datetime.now().isoformat()
            }
            
            for model_type in self.supported_models.keys():
                try:
                    model = await self._get_model(country, model_type, load_if_missing=True)
                    
                    if model:
                        model_info = await model.get_model_info()
                        status['models'][model_type] = {
                            'available': True,
                            'trained': model_info['is_trained'],
                            'info': model_info
                        }
                    else:
                        status['models'][model_type] = {
                            'available': False,
                            'trained': False,
                            'info': None
                        }
                        
                except Exception as e:
                    status['models'][model_type] = {
                        'available': False,
                        'trained': False,
                        'error': str(e)
                    }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get model status for {country}: {e}")
            raise
    
    async def retrain_models(self, 
                           country: str,
                           model_types: Optional[List[str]] = None,
                           force_retrain: bool = False) -> Dict[str, Any]:
        """
        Retrain models based on drift detection or manual trigger
        
        Args:
            country: Country name
            model_types: List of model types to retrain (None for all)
            force_retrain: Force retraining regardless of drift status
            
        Returns:
            Retraining results
        """
        try:
            validate_country(country)
            
            if model_types is None:
                model_types = list(self.supported_models.keys())
            
            logger.info(f"Initiating model retraining for {country}: {model_types}")
            
            retraining_results = {
                'country': country,
                'model_types': model_types,
                'results': {},
                'started_at': datetime.now().isoformat()
            }
            
            for model_type in model_types:
                try:
                    # Check if retraining is needed
                    if not force_retrain:
                        needs_retraining = await self.monitor.check_model_drift(country, model_type)
                        if not needs_retraining:
                            retraining_results['results'][model_type] = {
                                'status': 'skipped',
                                'reason': 'No drift detected'
                            }
                            continue
                    
                    # Retrain model
                    training_result = await self.train_model(country, model_type)
                    retraining_results['results'][model_type] = {
                        'status': 'completed',
                        'result': training_result
                    }
                    
                except Exception as e:
                    logger.error(f"Failed to retrain {model_type} for {country}: {e}")
                    retraining_results['results'][model_type] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            retraining_results['completed_at'] = datetime.now().isoformat()
            return retraining_results
            
        except Exception as e:
            logger.error(f"Failed to retrain models for {country}: {e}")
            raise
    
    async def _get_model(self, 
                        country: str, 
                        model_type: str,
                        load_if_missing: bool = True) -> Optional[Union[ProphetModel, ARIMAModel]]:
        """
        Get model from cache or load from disk
        
        Args:
            country: Country name
            model_type: Type of model
            load_if_missing: Whether to load from disk if not in cache
            
        Returns:
            Model instance or None
        """
        try:
            model_key = f"{model_type}_{country}"
            
            # Check cache first
            if model_key in self.models:
                return self.models[model_key]
            
            # Load from disk if requested
            if load_if_missing:
                model_class = self.supported_models[model_type]
                model = model_class(country)
                
                load_success = await model.load_model()
                if load_success:
                    self.models[model_key] = model
                    return model
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get model {model_type} for {country}: {e}")
            return None
    
    async def _enhance_predictions_with_business_metrics(self,
                                                       predictions: Dict[str, Any],
                                                       country: str) -> Dict[str, Any]:
        """
        Enhance predictions with business context
        
        Args:
            predictions: Model predictions
            country: Country name
            
        Returns:
            Enhanced predictions
        """
        try:
            # Get recent business metrics for context
            recent_metrics = await self.data_service.get_business_metrics(
                country=country,
                days_back=90
            )
            
            # Calculate business insights
            forecast_total = predictions['statistics']['total_forecast']
            recent_total = recent_metrics.get('revenue_90_days', 0)
            
            if recent_total > 0:
                growth_rate = ((forecast_total - recent_total) / recent_total) * 100
            else:
                growth_rate = 0
            
            business_insights = {
                'forecast_vs_recent_90_days': {
                    'forecast_total': forecast_total,
                    'recent_90_days_total': recent_total,
                    'growth_rate_percent': growth_rate
                },
                'business_context': recent_metrics
            }
            
            predictions['business_insights'] = business_insights
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to enhance predictions with business metrics: {e}")
            return predictions
    
    def _analyze_model_comparison(self, model_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze comparison between multiple models
        
        Args:
            model_results: Results from multiple models
            
        Returns:
            Comparison analysis
        """
        try:
            valid_models = {k: v for k, v in model_results.items() if 'error' not in v}
            
            if len(valid_models) < 2:
                return {'error': 'Need at least 2 valid models for comparison'}
            
            comparison = {
                'model_count': len(valid_models),
                'forecast_statistics': {},
                'agreement_analysis': {},
                'recommendations': []
            }
            
            # Extract forecast statistics
            for model_name, results in valid_models.items():
                stats = results.get('statistics', {})
                comparison['forecast_statistics'][model_name] = {
                    'mean_forecast': stats.get('mean_forecast', 0),
                    'total_forecast': stats.get('total_forecast', 0),
                    'std_forecast': stats.get('std_forecast', 0)
                }
            
            # Calculate agreement metrics
            forecast_means = [stats['mean_forecast'] for stats in comparison['forecast_statistics'].values()]
            forecast_totals = [stats['total_forecast'] for stats in comparison['forecast_statistics'].values()]
            
            comparison['agreement_analysis'] = {
                'mean_forecast_range': {
                    'min': min(forecast_means),
                    'max': max(forecast_means),
                    'std': float(np.std(forecast_means))
                },
                'total_forecast_range': {
                    'min': min(forecast_totals),
                    'max': max(forecast_totals),
                    'std': float(np.std(forecast_totals))
                }
            }
            
            # Generate recommendations
            cv_mean = np.std(forecast_means) / (np.mean(forecast_means) + 1e-8)
            if cv_mean < 0.1:
                comparison['recommendations'].append("Models show high agreement - predictions are reliable")
            elif cv_mean < 0.3:
                comparison['recommendations'].append("Models show moderate agreement - consider ensemble approach")
            else:
                comparison['recommendations'].append("Models show significant disagreement - investigate data quality")
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error analyzing model comparison: {e}")
            return {'error': str(e)}
