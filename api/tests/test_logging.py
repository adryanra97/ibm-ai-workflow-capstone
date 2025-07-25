"""
Unit tests for logging service

Author: Adryan R A
"""

import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch, MagicMock

from services.log_service import LogService
from conftest import TestUtils


class TestLogService:
    """Test logging service functionality"""
    
    @pytest.mark.asyncio
    async def test_log_service_initialization(self, test_db_path):
        """Test log service initialization"""
        log_service = LogService(test_db_path)
        assert log_service.db_path == test_db_path
    
    @pytest.mark.asyncio
    async def test_log_training_event(self, test_db_path):
        """Test logging training events"""
        log_service = LogService(test_db_path)
        
        # Log a training event
        await log_service.log_training(
            country="United Kingdom",
            model_type="prophet",
            model_version="1.0.0",
            training_time=120.5,
            metrics={"mae": 100.5, "rmse": 150.3},
            data_version="2023-01-01"
        )
        
        # Verify the log was recorded
        logs = await log_service.get_training_logs(
            country="United Kingdom",
            limit=1
        )
        
        assert len(logs) == 1
        assert logs[0]["country"] == "United Kingdom"
        assert logs[0]["model_type"] == "prophet"
        assert logs[0]["training_time"] == 120.5
    
    @pytest.mark.asyncio
    async def test_log_prediction_event(self, test_db_path):
        """Test logging prediction events"""
        log_service = LogService(test_db_path)
        
        # Log a prediction event
        await log_service.log_prediction(
            country="United Kingdom",
            model_type="prophet",
            model_version="1.0.0",
            prediction_date=datetime.now(),
            num_days=30,
            request_id="test-123"
        )
        
        # Verify the log was recorded
        logs = await log_service.get_prediction_logs(
            country="United Kingdom",
            limit=1
        )
        
        assert len(logs) == 1
        assert logs[0]["country"] == "United Kingdom"
        assert logs[0]["num_days"] == 30
        assert logs[0]["request_id"] == "test-123"
    
    @pytest.mark.asyncio
    async def test_log_api_request(self, test_db_path):
        """Test logging API requests"""
        log_service = LogService(test_db_path)
        
        # Log an API request
        await log_service.log_api_request(
            endpoint="/predict",
            method="POST",
            status_code=200,
            response_time=0.5,
            request_data={"country": "United Kingdom"},
            user_agent="test-client"
        )
        
        # Verify the log was recorded
        logs = await log_service.get_api_logs(limit=1)
        
        assert len(logs) == 1
        assert logs[0]["endpoint"] == "/predict"
        assert logs[0]["status_code"] == 200
        assert logs[0]["response_time"] == 0.5
    
    @pytest.mark.asyncio
    async def test_get_training_logs_with_filters(self, test_db_path):
        """Test getting training logs with various filters"""
        log_service = LogService(test_db_path)
        
        # Log multiple training events
        countries = ["United Kingdom", "Germany", "France"]
        for i, country in enumerate(countries):
            await log_service.log_training(
                country=country,
                model_type="prophet",
                model_version="1.0.0",
                training_time=100 + i * 10,
                metrics={"mae": 100 + i * 5},
                data_version="2023-01-01"
            )
        
        # Test country filter
        uk_logs = await log_service.get_training_logs(
            country="United Kingdom"
        )
        assert len(uk_logs) == 1
        assert uk_logs[0]["country"] == "United Kingdom"
        
        # Test limit
        limited_logs = await log_service.get_training_logs(limit=2)
        assert len(limited_logs) == 2
        
        # Test date range (should get all since they're recent)
        recent_logs = await log_service.get_training_logs(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(hours=1)
        )
        assert len(recent_logs) == 3
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, test_db_path):
        """Test performance metrics calculation"""
        log_service = LogService(test_db_path)
        
        # Log some API requests with different response times
        response_times = [0.1, 0.2, 0.5, 1.0, 0.3]
        for rt in response_times:
            await log_service.log_api_request(
                endpoint="/predict",
                method="POST",
                status_code=200,
                response_time=rt,
                request_data={},
                user_agent="test-client"
            )
        
        # Get performance metrics
        metrics = await log_service.get_performance_metrics(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now() + timedelta(hours=1)
        )
        
        assert "avg_response_time" in metrics
        assert "total_requests" in metrics
        assert "error_rate" in metrics
        assert metrics["total_requests"] == 5
        assert metrics["avg_response_time"] == sum(response_times) / len(response_times)
    
    @pytest.mark.asyncio
    async def test_error_logging(self, test_db_path):
        """Test error logging functionality"""
        log_service = LogService(test_db_path)
        
        # Log an error
        await log_service.log_error(
            error_type="ValidationError",
            error_message="Invalid country parameter",
            endpoint="/train",
            request_data={"country": "InvalidCountry"},
            stack_trace="Traceback: ..."
        )
        
        # Verify error was logged
        error_logs = await log_service.get_error_logs(limit=1)
        
        assert len(error_logs) == 1
        assert error_logs[0]["error_type"] == "ValidationError"
        assert error_logs[0]["error_message"] == "Invalid country parameter"
    
    @pytest.mark.asyncio
    async def test_database_cleanup(self, test_db_path):
        """Test database cleanup and maintenance"""
        log_service = LogService(test_db_path)
        
        # Log old entries (simulate by manually setting old dates)
        old_date = datetime.now() - timedelta(days=100)
        
        # This would require modifying the database directly for testing
        # In a real scenario, you'd have a cleanup method
        
        # Test that cleanup method exists and can be called
        result = await log_service.cleanup_old_logs(days_to_keep=30)
        assert isinstance(result, dict)
        assert "deleted_count" in result


class TestLogServicePerformance:
    """Test logging service performance and stress scenarios"""
    
    @pytest.mark.asyncio
    async def test_concurrent_logging(self, test_db_path):
        """Test concurrent logging operations"""
        log_service = LogService(test_db_path)
        
        # Create multiple concurrent logging tasks
        tasks = []
        for i in range(10):
            task = log_service.log_api_request(
                endpoint=f"/test-{i}",
                method="GET",
                status_code=200,
                response_time=0.1,
                request_data={},
                user_agent="concurrent-test"
            )
            tasks.append(task)
        
        # Execute all tasks concurrently
        await asyncio.gather(*tasks)
        
        # Verify all logs were recorded
        logs = await log_service.get_api_logs(limit=20)
        concurrent_logs = [log for log in logs if log.get("user_agent") == "concurrent-test"]
        assert len(concurrent_logs) == 10
    
    @pytest.mark.asyncio
    async def test_large_data_logging(self, test_db_path):
        """Test logging with large data payloads"""
        log_service = LogService(test_db_path)
        
        # Create a large request data payload
        large_data = {
            "data": ["item_" + str(i) for i in range(1000)],
            "metadata": {"key_" + str(i): "value_" + str(i) for i in range(100)}
        }
        
        # Log the large payload
        await log_service.log_api_request(
            endpoint="/large-data",
            method="POST",
            status_code=200,
            response_time=2.0,
            request_data=large_data,
            user_agent="large-data-test"
        )
        
        # Verify it was logged correctly
        logs = await log_service.get_api_logs(limit=1)
        assert len(logs) == 1
        assert logs[0]["endpoint"] == "/large-data"
