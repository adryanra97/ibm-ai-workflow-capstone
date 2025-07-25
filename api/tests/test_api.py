"""
Unit tests for API endpoints

Author: Adryan R A
"""

import pytest
import json
from datetime import date, datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from conftest import TestUtils


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check returns 200"""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
        assert "uptime_seconds" in data


class TestTrainEndpoint:
    """Test model training endpoint"""
    
    def test_train_endpoint_validation(self, client):
        """Test training endpoint input validation"""
        # Test missing required fields
        response = client.post("/train", json={})
        assert response.status_code == 422
        
        # Test invalid country
        response = client.post("/train", json={
            "country": "Invalid Country",
            "model_type": "prophet"
        })
        assert response.status_code == 400
        
        # Test invalid model type
        response = client.post("/train", json={
            "country": "United States",
            "model_type": "invalid_model"
        })
        assert response.status_code == 400
        
        # Test invalid date format
        response = client.post("/train", json={
            "country": "United States",
            "model_type": "prophet",
            "start_date": "invalid-date"
        })
        assert response.status_code == 422
    
    @patch('services.model_service.ModelService.train_model')
    def test_train_endpoint_success(self, mock_train, client):
        """Test successful model training"""
        # Mock successful training
        mock_train.return_value = {
            "country": "United States",
            "model_type": "prophet",
            "training_status": "completed",
            "metrics": {"mae": 100.0, "rmse": 150.0},
            "trained_at": datetime.now().isoformat(),
            "model_info": {"is_trained": True}
        }
        
        response = client.post("/train", json={
            "country": "United States",
            "model_type": "prophet"
        })
        
        assert response.status_code == 200
        data = response.json()
        TestUtils.assert_training_response(data)
        assert data["country"] == "United States"
        assert data["model_type"] == "prophet"
    
    @patch('services.model_service.ModelService.train_model')
    def test_train_endpoint_with_parameters(self, mock_train, client):
        """Test training with custom parameters"""
        mock_train.return_value = {
            "country": "Canada",
            "model_type": "arima",
            "training_status": "completed",
            "metrics": {"mae": 120.0, "rmse": 180.0},
            "trained_at": datetime.now().isoformat(),
            "model_info": {"is_trained": True}
        }
        
        response = client.post("/train", json={
            "country": "Canada",
            "model_type": "arima",
            "start_date": "2023-01-01",
            "end_date": "2023-12-31",
            "hyperparameters": {
                "max_p": 5,
                "max_q": 5
            }
        })
        
        assert response.status_code == 200
        mock_train.assert_called_once()
    
    @patch('services.model_service.ModelService.train_model')
    def test_train_endpoint_failure(self, mock_train, client):
        """Test training failure handling"""
        mock_train.side_effect = ValueError("Insufficient training data")
        
        response = client.post("/train", json={
            "country": "United States",
            "model_type": "prophet"
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "Insufficient training data" in data["error"]


class TestPredictEndpoint:
    """Test prediction endpoint"""
    
    def test_predict_endpoint_validation(self, client):
        """Test prediction endpoint input validation"""
        # Test missing required fields
        response = client.post("/predict", json={})
        assert response.status_code == 422
        
        # Test invalid forecast days
        response = client.post("/predict", json={
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 0
        })
        assert response.status_code == 400
        
        # Test invalid confidence interval
        response = client.post("/predict", json={
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 30,
            "confidence_interval": 1.5
        })
        assert response.status_code == 400
    
    @patch('services.model_service.ModelService.predict')
    def test_predict_endpoint_success(self, mock_predict, client):
        """Test successful prediction"""
        # Mock successful prediction
        mock_predict.return_value = {
            "model": "prophet",
            "country": "United States",
            "forecast_days": 30,
            "predictions": [
                {
                    "date": "2024-01-01",
                    "forecast": 1500.0,
                    "lower_bound": 1300.0,
                    "upper_bound": 1700.0
                }
            ],
            "statistics": {
                "mean_forecast": 1500.0,
                "total_forecast": 45000.0
            },
            "generated_at": datetime.now().isoformat()
        }
        
        response = client.post("/predict", json={
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 30
        })
        
        assert response.status_code == 200
        data = response.json()
        TestUtils.assert_prediction_response(data)
    
    @patch('services.model_service.ModelService.predict')
    def test_predict_endpoint_model_not_trained(self, mock_predict, client):
        """Test prediction with untrained model"""
        mock_predict.side_effect = ValueError("Model is not trained")
        
        response = client.post("/predict", json={
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 30
        })
        
        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "not trained" in data["error"]


class TestLogsEndpoint:
    """Test logs endpoint"""
    
    def test_logs_endpoint_validation(self, client):
        """Test logs endpoint input validation"""
        # Test invalid date format
        response = client.get("/logs?start_date=invalid-date")
        assert response.status_code == 422
        
        # Test invalid limit
        response = client.get("/logs?limit=-1")
        assert response.status_code == 422
        
        # Test invalid offset
        response = client.get("/logs?offset=-1")
        assert response.status_code == 422
    
    @patch('services.log_service.log_service.get_logs')
    def test_logs_endpoint_success(self, mock_get_logs, client):
        """Test successful log retrieval"""
        mock_get_logs.return_value = {
            "logs": [
                {
                    "id": 1,
                    "timestamp": "2024-01-01T12:00:00",
                    "level": "INFO",
                    "category": "api_request",
                    "message": "POST /predict -> 200",
                    "details": {}
                }
            ],
            "total_count": 1,
            "limit": 100,
            "offset": 0,
            "has_more": False
        }
        
        response = client.get("/logs")
        assert response.status_code == 200
        
        data = response.json()
        TestUtils.assert_log_response(data)
    
    @patch('services.log_service.log_service.get_logs')
    def test_logs_endpoint_with_filters(self, mock_get_logs, client):
        """Test log retrieval with filters"""
        mock_get_logs.return_value = {
            "logs": [],
            "total_count": 0,
            "limit": 50,
            "offset": 0,
            "has_more": False
        }
        
        response = client.get("/logs?level=ERROR&country=United States&limit=50")
        assert response.status_code == 200
        
        # Verify the mock was called with correct parameters
        mock_get_logs.assert_called_once()


class TestModelStatusEndpoint:
    """Test model status endpoint"""
    
    @patch('services.model_service.ModelService.get_model_status')
    def test_model_status_success(self, mock_status, client):
        """Test successful model status retrieval"""
        mock_status.return_value = {
            "country": "United States",
            "models": {
                "prophet": {
                    "available": True,
                    "trained": True,
                    "info": {"is_trained": True, "created_at": "2024-01-01T12:00:00"}
                },
                "arima": {
                    "available": False,
                    "trained": False,
                    "info": None
                }
            },
            "checked_at": datetime.now().isoformat()
        }
        
        response = client.get("/models/status/United States")
        assert response.status_code == 200
        
        data = response.json()
        assert data["country"] == "United States"
        assert "models" in data
        assert "prophet" in data["models"]
        assert "arima" in data["models"]
    
    def test_model_status_invalid_country(self, client):
        """Test model status with invalid country"""
        response = client.get("/models/status/Invalid Country")
        assert response.status_code == 400


class TestCompareModelsEndpoint:
    """Test model comparison endpoint"""
    
    @patch('services.model_service.ModelService.compare_models')
    def test_compare_models_success(self, mock_compare, client):
        """Test successful model comparison"""
        mock_compare.return_value = {
            "country": "United States",
            "forecast_days": 30,
            "models": {
                "prophet": {
                    "predictions": [{"date": "2024-01-01", "forecast": 1500.0}],
                    "statistics": {"mean_forecast": 1500.0}
                },
                "arima": {
                    "predictions": [{"date": "2024-01-01", "forecast": 1520.0}],
                    "statistics": {"mean_forecast": 1520.0}
                }
            },
            "comparison": {
                "model_count": 2,
                "agreement_analysis": {"mean_forecast_range": {"std": 10.0}}
            },
            "generated_at": datetime.now().isoformat()
        }
        
        response = client.post("/models/compare", json={
            "country": "United States",
            "forecast_days": 30
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["country"] == "United States"
        assert "models" in data
        assert "comparison" in data
    
    def test_compare_models_validation(self, client):
        """Test model comparison input validation"""
        # Test missing country
        response = client.post("/models/compare", json={
            "forecast_days": 30
        })
        assert response.status_code == 422
        
        # Test invalid forecast days
        response = client.post("/models/compare", json={
            "country": "United States",
            "forecast_days": 0
        })
        assert response.status_code == 400


class TestBackgroundTasks:
    """Test background task endpoints"""
    
    @patch('services.model_service.ModelService.retrain_models')
    def test_retrain_endpoint(self, mock_retrain, client):
        """Test model retraining endpoint"""
        mock_retrain.return_value = {
            "country": "United States",
            "model_types": ["prophet"],
            "results": {
                "prophet": {
                    "status": "completed",
                    "result": {"training_status": "completed"}
                }
            },
            "started_at": datetime.now().isoformat(),
            "completed_at": datetime.now().isoformat()
        }
        
        response = client.post("/models/retrain", json={
            "country": "United States",
            "model_types": ["prophet"],
            "force_retrain": True
        })
        
        assert response.status_code == 200
        data = response.json()
        assert data["country"] == "United States"
        assert "results" in data
    
    @patch('services.log_service.log_service.get_performance_metrics')
    def test_performance_metrics_endpoint(self, mock_metrics, client):
        """Test performance metrics endpoint"""
        mock_metrics.return_value = {
            "endpoint_metrics": [
                {
                    "endpoint": "/predict",
                    "request_count": 100,
                    "avg_response_time": 250.0,
                    "error_rate": 2.5
                }
            ],
            "overall_statistics": {
                "total_requests": 500,
                "overall_avg_response_time": 180.0,
                "overall_error_rate": 1.8
            }
        }
        
        response = client.get("/monitoring/performance")
        assert response.status_code == 200
        
        data = response.json()
        assert "endpoint_metrics" in data
        assert "overall_statistics" in data


class TestErrorHandling:
    """Test error handling across endpoints"""
    
    def test_404_error(self, client):
        """Test 404 error handling"""
        response = client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_405_error(self, client):
        """Test 405 method not allowed"""
        response = client.put("/health")
        assert response.status_code == 405
    
    @patch('services.model_service.ModelService.predict')
    def test_internal_server_error(self, mock_predict, client):
        """Test 500 internal server error handling"""
        mock_predict.side_effect = Exception("Database connection failed")
        
        response = client.post("/predict", json={
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 30
        })
        
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert "Internal server error" in data["error"]


class TestRequestValidation:
    """Test request validation and sanitization"""
    
    def test_large_request_handling(self, client):
        """Test handling of large requests"""
        # Create a very large request
        large_data = {
            "country": "United States",
            "model_type": "prophet",
            "forecast_days": 30,
            "large_field": "x" * 10000  # 10KB of data
        }
        
        response = client.post("/predict", json=large_data)
        # Should still process the request but ignore the large field
        assert response.status_code in [200, 400]  # Either success or validation error
    
    def test_special_characters_handling(self, client):
        """Test handling of special characters in input"""
        response = client.post("/predict", json={
            "country": "Test<script>alert('xss')</script>",
            "model_type": "prophet",
            "forecast_days": 30
        })
        
        # Should reject due to invalid country
        assert response.status_code == 400
    
    def test_sql_injection_protection(self, client):
        """Test protection against SQL injection"""
        response = client.get("/logs?country='; DROP TABLE logs; --")
        
        # Should not cause an error due to proper parameterized queries
        assert response.status_code in [200, 400]  # Either success or validation error
