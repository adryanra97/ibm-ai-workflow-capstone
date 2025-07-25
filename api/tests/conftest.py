"""
Test configuration and fixtures

Author: Adryan R A
"""

import pytest
import asyncio
import tempfile
import shutil
import os
from datetime import date, datetime, timedelta
from typing import Dict, Any, AsyncGenerator
import pandas as pd
import sqlite3

from fastapi.testclient import TestClient
from app import app
from services.data_service import DataService
from services.model_service import ModelService
from services.log_service import LogService
from utils.config import get_settings


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def client():
    """Create test client for FastAPI app"""
    return TestClient(app)


@pytest.fixture
def temp_dir():
    """Create temporary directory for test data"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_settings(temp_dir):
    """Override settings for testing"""
    settings = get_settings()
    settings.database_path = os.path.join(temp_dir, "test_data")
    settings.model_storage_path = os.path.join(temp_dir, "test_models")
    settings.log_database_path = os.path.join(temp_dir, "test_logs.db")
    
    # Create directories
    os.makedirs(settings.database_path, exist_ok=True)
    os.makedirs(settings.model_storage_path, exist_ok=True)
    os.makedirs(os.path.dirname(settings.log_database_path), exist_ok=True)
    
    return settings


@pytest.fixture
async def data_service(test_settings):
    """Create test data service with sample data"""
    service = DataService()
    
    # Create test databases with sample data
    await service._create_test_data()
    
    return service


@pytest.fixture
async def model_service(data_service):
    """Create test model service"""
    return ModelService()


@pytest.fixture
async def log_service(test_settings):
    """Create test log service"""
    return LogService()


@pytest.fixture
def sample_invoice_data():
    """Sample invoice data for testing"""
    return {
        "country": "United States",
        "customer_id": "12345",
        "day": "2023-01-15",
        "invoice_number": "INV-001",
        "total_price": 1500.00,
        "streams": [
            {
                "stream_id": "stream_001",
                "times_viewed": 100,
                "price": 1500.00
            }
        ]
    }


@pytest.fixture
def sample_training_data():
    """Sample training data for model testing"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    data = []
    
    for i, date_val in enumerate(dates):
        # Create realistic revenue data with trend and seasonality
        base_revenue = 1000
        trend = i * 2  # Growing trend
        seasonal = 200 * (1 + 0.3 * (i % 7))  # Weekly seasonality
        noise = 50 * (hash(str(date_val)) % 100 / 100)  # Random noise
        
        revenue = base_revenue + trend + seasonal + noise
        
        data.append({
            "date": date_val.date(),
            "total_revenue": revenue,
            "num_invoices": max(1, int(revenue / 100)),
            "unique_customers": max(1, int(revenue / 200))
        })
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_prediction_request():
    """Sample prediction request"""
    return {
        "country": "United States",
        "model_type": "prophet",
        "forecast_days": 30,
        "confidence_interval": 0.95
    }


@pytest.fixture
def sample_train_request():
    """Sample training request"""
    return {
        "country": "United States",
        "model_type": "prophet",
        "start_date": "2023-01-01",
        "end_date": "2023-12-31"
    }


class TestDataService(DataService):
    """Extended data service for testing"""
    
    async def _create_test_data(self):
        """Create test databases with sample data"""
        await self._create_test_l0_database()
        await self._create_test_l3_database()
    
    async def _create_test_l0_database(self):
        """Create test L0 database with sample invoice data"""
        l0_db_path = os.path.join(self.database_path, "file.db")
        
        with sqlite3.connect(l0_db_path) as conn:
            cursor = conn.cursor()
            
            # Create invoices table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS invoices (
                    country TEXT,
                    customer_id TEXT,
                    day TEXT,
                    invoice_number TEXT,
                    total_price REAL,
                    stream_data TEXT
                )
            """)
            
            # Insert sample data
            countries = ["United States", "Canada", "United Kingdom", "Australia"]
            start_date = datetime(2023, 1, 1)
            
            for country in countries:
                for i in range(365):  # Full year of data
                    current_date = start_date + timedelta(days=i)
                    
                    # Generate 1-10 invoices per day per country
                    num_invoices = (hash(f"{country}_{i}") % 10) + 1
                    
                    for j in range(num_invoices):
                        invoice_number = f"INV-{country[:2]}-{i:03d}-{j:02d}"
                        customer_id = f"CUST-{(hash(invoice_number) % 1000):04d}"
                        
                        # Revenue between $100-$5000
                        total_price = 100 + (hash(invoice_number) % 4900)
                        
                        stream_data = [{
                            "stream_id": f"stream_{j:03d}",
                            "times_viewed": (hash(invoice_number) % 100) + 1,
                            "price": total_price
                        }]
                        
                        cursor.execute("""
                            INSERT INTO invoices 
                            (country, customer_id, day, invoice_number, total_price, stream_data)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            country,
                            customer_id,
                            current_date.strftime("%Y-%m-%d"),
                            invoice_number,
                            total_price,
                            str(stream_data)
                        ))
            
            conn.commit()
    
    async def _create_test_l3_database(self):
        """Create test L3 database with aggregated data"""
        l3_db_path = os.path.join(self.database_path, "analytics.db")
        
        with sqlite3.connect(l3_db_path) as conn:
            cursor = conn.cursor()
            
            # Create analytics tables
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS daily_revenue (
                    date TEXT,
                    country TEXT,
                    total_revenue REAL,
                    num_invoices INTEGER,
                    unique_customers INTEGER,
                    avg_invoice_value REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS business_metrics (
                    date TEXT,
                    country TEXT,
                    metric_name TEXT,
                    metric_value REAL
                )
            """)
            
            # Aggregate data from L0 to L3
            l0_db_path = os.path.join(self.database_path, "file.db")
            
            with sqlite3.connect(l0_db_path) as l0_conn:
                l0_cursor = l0_conn.cursor()
                
                # Get aggregated daily data
                l0_cursor.execute("""
                    SELECT 
                        day,
                        country,
                        SUM(total_price) as total_revenue,
                        COUNT(*) as num_invoices,
                        COUNT(DISTINCT customer_id) as unique_customers,
                        AVG(total_price) as avg_invoice_value
                    FROM invoices
                    GROUP BY day, country
                    ORDER BY day, country
                """)
                
                daily_data = l0_cursor.fetchall()
                
                # Insert into L3
                for row in daily_data:
                    cursor.execute("""
                        INSERT INTO daily_revenue 
                        (date, country, total_revenue, num_invoices, unique_customers, avg_invoice_value)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, row)
                    
                    # Add business metrics
                    date_val, country, total_revenue, num_invoices, unique_customers, avg_invoice_value = row
                    
                    # Calculate additional business metrics
                    metrics = [
                        ("revenue_per_customer", total_revenue / max(unique_customers, 1)),
                        ("customer_acquisition_rate", unique_customers / max(num_invoices, 1)),
                        ("daily_growth_rate", 0.02)  # Placeholder
                    ]
                    
                    for metric_name, metric_value in metrics:
                        cursor.execute("""
                            INSERT INTO business_metrics (date, country, metric_name, metric_value)
                            VALUES (?, ?, ?, ?)
                        """, (date_val, country, metric_name, metric_value))
            
            conn.commit()


# Async test helpers
@pytest.fixture
async def async_client():
    """Async test client"""
    from httpx import AsyncClient
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


# Mock data generators
def generate_time_series_data(
    start_date: date = date(2023, 1, 1),
    end_date: date = date(2023, 12, 31),
    base_value: float = 1000.0,
    trend: float = 1.0,
    seasonality: float = 0.3,
    noise: float = 0.1
) -> pd.DataFrame:
    """Generate synthetic time series data for testing"""
    
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    data = []
    
    for i, date_val in enumerate(dates):
        # Base value with trend
        value = base_value + (trend * i)
        
        # Add weekly seasonality
        day_of_week = date_val.weekday()
        seasonal_component = seasonality * base_value * (1 + 0.5 * (day_of_week % 2))
        
        # Add noise
        noise_component = noise * base_value * ((hash(str(date_val)) % 100) / 100 - 0.5)
        
        final_value = value + seasonal_component + noise_component
        
        data.append({
            "ds": date_val,
            "y": max(0, final_value)  # Ensure non-negative values
        })
    
    return pd.DataFrame(data)


# Test utilities
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def assert_api_response_structure(response_data: Dict[str, Any], expected_keys: list):
        """Assert that API response has expected structure"""
        for key in expected_keys:
            assert key in response_data, f"Missing key: {key}"
    
    @staticmethod
    def assert_prediction_response(response_data: Dict[str, Any]):
        """Assert prediction response structure"""
        required_keys = [
            "model", "country", "forecast_days", "predictions", 
            "statistics", "generated_at"
        ]
        TestUtils.assert_api_response_structure(response_data, required_keys)
        
        # Check predictions structure
        assert len(response_data["predictions"]) > 0
        
        prediction = response_data["predictions"][0]
        prediction_keys = ["date", "forecast", "lower_bound", "upper_bound"]
        TestUtils.assert_api_response_structure(prediction, prediction_keys)
    
    @staticmethod
    def assert_training_response(response_data: Dict[str, Any]):
        """Assert training response structure"""
        required_keys = [
            "country", "model_type", "training_status", "metrics", 
            "trained_at", "model_info"
        ]
        TestUtils.assert_api_response_structure(response_data, required_keys)
        
        # Check training status
        assert response_data["training_status"] == "completed"
    
    @staticmethod
    def assert_log_response(response_data: Dict[str, Any]):
        """Assert log response structure"""
        required_keys = ["logs", "total_count", "limit", "offset", "has_more"]
        TestUtils.assert_api_response_structure(response_data, required_keys)


# Performance testing helpers
@pytest.fixture
def performance_test_config():
    """Configuration for performance testing"""
    return {
        "max_response_time_ms": 5000,
        "max_memory_usage_mb": 512,
        "concurrent_requests": 10,
        "test_duration_seconds": 30
    }
