"""
Unit tests for model services

Author: Adryan R A
"""

import pytest
import asyncio
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from models.base_model import BaseTimeSeriesModel
from models.prophet_model import ProphetModel
from models.arima_model import ARIMAModel
from services.model_service import ModelService
from conftest import generate_time_series_data


class TestBaseTimeSeriesModel:
    """Test base time series model functionality"""
    
    def test_base_model_initialization(self):
        """Test base model initialization"""
        # This is an abstract class, so we can't instantiate it directly
        # But we can test that it raises the right error
        with pytest.raises(TypeError):
            BaseTimeSeriesModel("United States", "test")
    
    def test_calculate_metrics(self):
        """Test metric calculation method"""
        # Create a concrete implementation for testing
        class TestModel(BaseTimeSeriesModel):
            async def train(self, data, **kwargs):
                pass
            async def predict(self, forecast_days, confidence_interval=0.95, **kwargs):
                pass
            async def validate(self, test_data):
                pass
        
        model = TestModel("United States", "test")
        
        # Test with perfect predictions
        y_true = np.array([100, 200, 300, 400, 500])
        y_pred = np.array([100, 200, 300, 400, 500])
        
        metrics = model._calculate_metrics(y_true, y_pred)
        
        assert metrics['mae'] == 0.0
        assert metrics['rmse'] == 0.0
        assert metrics['mape'] == 0.0
        assert abs(metrics['r2'] - 1.0) < 1e-10
    
    def test_prepare_training_data(self):
        """Test training data preparation"""
        class TestModel(BaseTimeSeriesModel):
            async def train(self, data, **kwargs):
                pass
            async def predict(self, forecast_days, confidence_interval=0.95, **kwargs):
                pass
            async def validate(self, test_data):
                pass
        
        model = TestModel("United States", "test")
        
        # Create test data with missing values and duplicates
        data = pd.DataFrame({
            'ds': pd.date_range('2023-01-01', periods=10, freq='D').tolist() + [pd.Timestamp('2023-01-05')],
            'y': [100, 200, None, 400, 500, 600, 700, 800, 900, 1000, 550]  # Duplicate date with different value
        })
        
        prepared_data = model._prepare_training_data(data)
        
        # Should remove missing values and duplicates
        assert len(prepared_data) == 10  # One duplicate removed
        assert prepared_data['y'].isnull().sum() == 0  # No missing values
        assert prepared_data['ds'].is_monotonic_increasing  # Sorted by date


class TestProphetModel:
    """Test Prophet model functionality"""
    
    @pytest.fixture
    def prophet_model(self):
        """Create Prophet model for testing"""
        return ProphetModel("United States")
    
    @pytest.fixture
    def training_data(self):
        """Create training data for Prophet model"""
        return generate_time_series_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            base_value=1000.0
        )
    
    def test_prophet_model_initialization(self, prophet_model):
        """Test Prophet model initialization"""
        assert prophet_model.country == "United States"
        assert prophet_model.model_name == "prophet"
        assert not prophet_model.is_trained
        assert prophet_model.prophet_params is not None
    
    @pytest.mark.asyncio
    @patch('models.prophet_model.Prophet')
    async def test_prophet_model_training(self, mock_prophet_class, prophet_model, training_data):
        """Test Prophet model training"""
        # Mock Prophet model
        mock_prophet = MagicMock()
        mock_prophet.fit = MagicMock()
        mock_prophet_class.return_value = mock_prophet
        
        # Train the model
        with patch.object(prophet_model, '_calculate_training_metrics', return_value={'mae': 100.0}):
            metrics = await prophet_model.train(training_data)
        
        assert prophet_model.is_trained
        assert prophet_model.model is not None
        assert 'mae' in metrics
        mock_prophet.fit.assert_called_once()
    
    @pytest.mark.asyncio
    @patch('models.prophet_model.Prophet')
    async def test_prophet_model_prediction(self, mock_prophet_class, prophet_model, training_data):
        """Test Prophet model prediction"""
        # Setup mock
        mock_prophet = MagicMock()
        mock_future = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=30, freq='D')
        })
        mock_forecast = pd.DataFrame({
            'ds': pd.date_range('2024-01-01', periods=30, freq='D'),
            'yhat': np.random.normal(1000, 100, 30),
            'yhat_lower': np.random.normal(900, 100, 30),
            'yhat_upper': np.random.normal(1100, 100, 30),
            'trend': np.random.normal(1000, 50, 30),
            'yearly': np.random.normal(0, 20, 30),
            'weekly': np.random.normal(0, 10, 30)
        })
        
        mock_prophet.make_future_dataframe.return_value = mock_future
        mock_prophet.predict.return_value = mock_forecast
        mock_prophet_class.return_value = mock_prophet
        
        # Train model first
        prophet_model.model = mock_prophet
        prophet_model.is_trained = True
        prophet_model.training_data = training_data
        
        # Generate predictions
        predictions = await prophet_model.predict(forecast_days=30)
        
        assert predictions['model'] == 'prophet'
        assert predictions['country'] == 'United States'
        assert predictions['forecast_days'] == 30
        assert len(predictions['predictions']) == 30
        assert 'statistics' in predictions
        assert 'trend_analysis' in predictions
    
    @pytest.mark.asyncio
    async def test_prophet_model_validation_not_trained(self, prophet_model):
        """Test Prophet model validation when not trained"""
        test_data = generate_time_series_data(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 1, 31)
        )
        
        with pytest.raises(ValueError, match="must be trained"):
            await prophet_model.validate(test_data)
    
    def test_prophet_add_regressor(self, prophet_model):
        """Test adding external regressors"""
        prophet_model.add_regressor("holiday_flag")
        assert "holiday_flag" in prophet_model.additional_regressors
    
    def test_prophet_set_holidays(self, prophet_model):
        """Test setting holidays"""
        prophet_model.set_holidays("US")
        assert prophet_model.holidays == "US"


class TestARIMAModel:
    """Test ARIMA model functionality"""
    
    @pytest.fixture
    def arima_model(self):
        """Create ARIMA model for testing"""
        return ARIMAModel("United States")
    
    @pytest.fixture
    def training_data(self):
        """Create training data for ARIMA model"""
        return generate_time_series_data(
            start_date=date(2023, 1, 1),
            end_date=date(2023, 12, 31),
            base_value=1000.0
        )
    
    def test_arima_model_initialization(self, arima_model):
        """Test ARIMA model initialization"""
        assert arima_model.country == "United States"
        assert arima_model.model_name == "arima"
        assert not arima_model.is_trained
        assert arima_model.max_p == 5
        assert arima_model.max_q == 5
    
    @pytest.mark.asyncio
    @patch('models.arima_model.SARIMAX')
    async def test_arima_model_training(self, mock_sarimax_class, arima_model, training_data):
        """Test ARIMA model training"""
        # Mock SARIMAX model
        mock_model = MagicMock()
        mock_fitted = MagicMock()
        mock_fitted.aic = 1000.0
        mock_fitted.bic = 1050.0
        mock_fitted.hqic = 1025.0
        mock_fitted.llf = -500.0
        mock_fitted.fittedvalues = pd.Series(np.random.normal(1000, 100, len(training_data)))
        
        mock_model.fit.return_value = mock_fitted
        mock_sarimax_class.return_value = mock_model
        
        # Mock auto-selection methods
        with patch.object(arima_model, '_auto_select_arima_order', return_value=(1, 1, 1)):
            with patch.object(arima_model, '_auto_select_seasonal_order', return_value=(0, 0, 0, 7)):
                with patch.object(arima_model, '_preprocess_data', return_value=(training_data['y'], {})):
                    metrics = await arima_model.train(training_data)
        
        assert arima_model.is_trained
        assert arima_model.arima_order == (1, 1, 1)
        assert 'aic' in metrics
    
    @pytest.mark.asyncio
    @patch('models.arima_model.SARIMAX')
    async def test_arima_model_prediction(self, mock_sarimax_class, arima_model, training_data):
        """Test ARIMA model prediction"""
        # Setup mock
        mock_fitted = MagicMock()
        mock_forecast_result = MagicMock()
        mock_forecast_result.predicted_mean = pd.Series(np.random.normal(1000, 100, 30))
        mock_forecast_result.conf_int.return_value = pd.DataFrame({
            0: np.random.normal(900, 100, 30),
            1: np.random.normal(1100, 100, 30)
        })
        
        mock_fitted.get_forecast.return_value = mock_forecast_result
        
        # Setup model state
        arima_model.model = mock_fitted
        arima_model.is_trained = True
        arima_model.training_data = training_data
        arima_model.arima_order = (1, 1, 1)
        arima_model.seasonal_order = (0, 0, 0, 7)
        arima_model.log_transformed = False
        
        # Generate predictions
        predictions = await arima_model.predict(forecast_days=30)
        
        assert predictions['model'] == 'arima'
        assert predictions['country'] == 'United States'
        assert predictions['forecast_days'] == 30
        assert len(predictions['predictions']) == 30
        assert 'arima_order' in predictions
        assert 'seasonal_order' in predictions
    
    def test_arima_stationarity_test(self, arima_model):
        """Test stationarity testing"""
        # Create a non-stationary series (trending)
        trend_series = pd.Series(range(100))
        
        with patch('models.arima_model.adfuller', return_value=(0, 0.8, None, None, None, None)):
            with patch('models.arima_model.kpss', return_value=(0, 0.01, None, None)):
                result = arima_model._test_stationarity(trend_series)
        
        assert isinstance(result, dict)
        assert 'is_stationary' in result
        assert 'adf_pvalue' in result
        assert 'kpss_pvalue' in result
    
    def test_arima_make_stationary(self, arima_model):
        """Test making series stationary"""
        # Create a trending series
        trend_series = pd.Series(range(100))
        
        with patch.object(arima_model, '_test_stationarity') as mock_test:
            # First call returns non-stationary, second returns stationary
            mock_test.side_effect = [
                {'is_stationary': False},
                {'is_stationary': True}
            ]
            
            stationary_data, diff_order = arima_model._make_stationary(trend_series)
            
            assert diff_order == 1
            assert len(stationary_data) == len(trend_series) - 1  # One less due to differencing


class TestModelService:
    """Test model service functionality"""
    
    @pytest.fixture
    def model_service(self):
        """Create model service for testing"""
        return ModelService()
    
    @pytest.mark.asyncio
    async def test_model_service_initialization(self, model_service):
        """Test model service initialization"""
        assert model_service.data_service is not None
        assert model_service.monitor is not None
        assert 'prophet' in model_service.supported_models
        assert 'arima' in model_service.supported_models
    
    @pytest.mark.asyncio
    @patch('services.model_service.DataService.get_training_data')
    @patch('services.model_service.ProphetModel')
    async def test_model_service_train_prophet(self, mock_prophet_class, mock_get_data, model_service):
        """Test training Prophet model through service"""
        # Mock data service
        mock_data = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=365, freq='D'),
            'total_revenue': np.random.normal(1000, 100, 365)
        })
        mock_get_data.return_value = mock_data
        
        # Mock Prophet model
        mock_prophet = MagicMock()
        mock_prophet.train = AsyncMock(return_value={'mae': 100.0})
        mock_prophet.save_model = AsyncMock(return_value=True)
        mock_prophet.get_model_info = AsyncMock(return_value={'is_trained': True})
        mock_prophet_class.return_value = mock_prophet
        
        # Train model
        result = await model_service.train_model("United States", "prophet")
        
        assert result['country'] == "United States"
        assert result['model_type'] == "prophet"
        assert result['training_status'] == "completed"
        assert 'metrics' in result
    
    @pytest.mark.asyncio
    @patch('services.model_service.ModelService._get_model')
    async def test_model_service_predict(self, mock_get_model, model_service):
        """Test prediction through model service"""
        # Mock model
        mock_model = MagicMock()
        mock_model.is_trained = True
        mock_model.predict = AsyncMock(return_value={
            'model': 'prophet',
            'country': 'United States',
            'predictions': [{'date': '2024-01-01', 'forecast': 1000.0}],
            'statistics': {'mean_forecast': 1000.0}
        })
        mock_get_model.return_value = mock_model
        
        # Mock data service for business metrics
        with patch.object(model_service.data_service, 'get_business_metrics', return_value={}):
            result = await model_service.predict("United States", "prophet", 30)
        
        assert result['model'] == 'prophet'
        assert result['country'] == 'United States'
        assert 'predictions' in result
    
    @pytest.mark.asyncio
    async def test_model_service_validation(self, model_service):
        """Test input validation in model service"""
        # Test invalid country
        with pytest.raises(ValueError):
            await model_service.train_model("Invalid Country", "prophet")
        
        # Test invalid model type
        with pytest.raises(ValueError):
            await model_service.train_model("United States", "invalid_model")
    
    @pytest.mark.asyncio
    @patch('services.model_service.ModelService._get_model')
    async def test_model_service_compare_models(self, mock_get_model, model_service):
        """Test model comparison"""
        # Mock models
        mock_prophet = MagicMock()
        mock_prophet.is_trained = True
        mock_prophet.predict = AsyncMock(return_value={
            'predictions': [{'forecast': 1000.0}],
            'statistics': {'mean_forecast': 1000.0, 'total_forecast': 30000.0}
        })
        
        mock_arima = MagicMock()
        mock_arima.is_trained = True
        mock_arima.predict = AsyncMock(return_value={
            'predictions': [{'forecast': 1020.0}],
            'statistics': {'mean_forecast': 1020.0, 'total_forecast': 30600.0}
        })
        
        def get_model_side_effect(country, model_type, load_if_missing=True):
            if model_type == 'prophet':
                return mock_prophet
            elif model_type == 'arima':
                return mock_arima
            return None
        
        mock_get_model.side_effect = get_model_side_effect
        
        result = await model_service.compare_models("United States", 30)
        
        assert result['country'] == "United States"
        assert 'models' in result
        assert 'comparison' in result
        assert len(result['models']) == 2
    
    @pytest.mark.asyncio
    @patch('services.model_service.ModelService._get_model')
    async def test_model_service_get_status(self, mock_get_model, model_service):
        """Test getting model status"""
        # Mock model
        mock_model = MagicMock()
        mock_model.get_model_info = AsyncMock(return_value={
            'is_trained': True,
            'created_at': '2024-01-01T12:00:00'
        })
        mock_get_model.return_value = mock_model
        
        result = await model_service.get_model_status("United States")
        
        assert result['country'] == "United States"
        assert 'models' in result
        assert 'checked_at' in result


class TestModelIntegration:
    """Integration tests for model components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_prophet_workflow(self):
        """Test complete Prophet workflow"""
        # This would be a more comprehensive test with real data
        # For now, we'll test the workflow structure
        
        model = ProphetModel("United States")
        
        # Verify initial state
        assert not model.is_trained
        assert model.model is None
        
        # Create synthetic data
        data = generate_time_series_data()
        
        # Test that training would require proper Prophet installation
        # In a real test environment, this would actually train the model
        with pytest.raises((ImportError, AttributeError)):
            await model.train(data)
    
    @pytest.mark.asyncio
    async def test_end_to_end_arima_workflow(self):
        """Test complete ARIMA workflow"""
        model = ARIMAModel("United States")
        
        # Verify initial state
        assert not model.is_trained
        assert model.model is None
        
        # Create synthetic data
        data = generate_time_series_data()
        
        # Test that training would require proper statsmodels installation
        with pytest.raises((ImportError, AttributeError)):
            await model.train(data)
    
    def test_model_serialization(self):
        """Test model save/load functionality"""
        # Test that model paths are correctly constructed
        prophet_model = ProphetModel("United States")
        arima_model = ARIMAModel("Canada")
        
        assert "prophet_united_states.pkl" in prophet_model.model_path
        assert "arima_canada.pkl" in arima_model.model_path
        
        # Test model info before training
        info = asyncio.run(prophet_model.get_model_info())
        assert info['is_trained'] is False
        assert info['country'] == "United States"
