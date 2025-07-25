"""
Configuration management for the Revenue Forecasting API

Author: Adryan R A
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    debug_mode: bool = os.getenv("DEBUG", "false").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    allowed_origins: List[str] = ["*"]  # Configure for production
    
    # Database Configuration
    l0_database_path: str = os.getenv("L0_DB_PATH", "../file.db")
    l3_database_path: str = os.getenv("L3_DB_PATH", "../analytics.db")
    
    # Model Configuration
    model_storage_path: str = os.getenv("MODEL_PATH", "./models")
    supported_countries: List[str] = [
        "United Kingdom", "Germany", "France", "EIRE", "Spain", 
        "Netherlands", "Belgium", "Switzerland", "Portugal", "Australia",
        "Italy", "Austria", "Cyprus", "Japan", "Denmark", "Finland",
        "Norway", "Poland", "Sweden", "Greece"
    ]
    
    # Prophet Model Settings
    prophet_seasonality_mode: str = "multiplicative"
    prophet_yearly_seasonality: bool = True
    prophet_weekly_seasonality: bool = True
    prophet_daily_seasonality: bool = False
    prophet_interval_width: float = 0.95
    
    # ARIMA Model Settings
    arima_max_p: int = 5
    arima_max_d: int = 2
    arima_max_q: int = 5
    arima_seasonal: bool = True
    arima_seasonal_periods: int = 7  # Weekly seasonality
    
    # Training Configuration
    min_training_days: int = 30
    max_training_days: int = 730
    test_split_ratio: float = 0.2
    
    # Prediction Configuration
    max_forecast_days: int = 365
    ensemble_weights: dict = {"prophet": 0.6, "arima": 0.4}
    
    # Monitoring Configuration
    drift_detection_threshold: float = 0.1
    performance_threshold: float = 0.85
    alert_email_enabled: bool = False
    alert_email_recipients: List[str] = []
    
    # Logging Configuration
    log_retention_days: int = 30
    max_log_entries: int = 10000
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings"""
    return Settings()


# Model type mappings
MODEL_TYPES = {
    "prophet": "Facebook Prophet",
    "arima": "ARIMA/SARIMA", 
    "both": "Prophet and ARIMA",
    "ensemble": "Ensemble (Prophet + ARIMA)"
}

# Business metrics mapping
BUSINESS_METRICS = {
    "revenue": "Total Revenue",
    "transactions": "Transaction Count",
    "customers": "Unique Customers",
    "avg_order_value": "Average Order Value",
    "customer_retention": "Customer Retention Rate"
}

# Supported aggregation levels
AGGREGATION_LEVELS = ["daily", "weekly", "monthly"]

# API response status codes
API_STATUS = {
    "SUCCESS": "success",
    "ERROR": "error", 
    "WARNING": "warning",
    "SKIPPED": "skipped"
}

# Model performance metrics
PERFORMANCE_METRICS = [
    "mae",  # Mean Absolute Error
    "mape", # Mean Absolute Percentage Error  
    "rmse", # Root Mean Square Error
    "mase", # Mean Absolute Scaled Error
    "r2"    # R-squared
]
