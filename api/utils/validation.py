"""
Input validation utilities for the Revenue Forecasting API

Author: Adryan R A
"""

import re
from datetime import date, datetime
from typing import Optional, List
from utils.config import get_settings

settings = get_settings()


def validate_country(country: str) -> str:
    """
    Validate country name against supported countries
    
    Args:
        country: Country name to validate
        
    Returns:
        Validated country name
        
    Raises:
        ValueError: If country is not supported
    """
    if not country or not isinstance(country, str):
        raise ValueError("Country name is required")
    
    # Clean and normalize country name
    country = country.strip().title()
    
    # Handle common variations
    country_mappings = {
        "Uk": "United Kingdom",
        "England": "United Kingdom", 
        "Britain": "United Kingdom",
        "Great Britain": "United Kingdom",
        "Ireland": "EIRE",
        "Republic Of Ireland": "EIRE",
        "Holland": "Netherlands",
        "Deutschland": "Germany"
    }
    
    country = country_mappings.get(country, country)
    
    if country not in settings.supported_countries:
        raise ValueError(
            f"Country '{country}' not supported. "
            f"Supported countries: {', '.join(settings.supported_countries)}"
        )
    
    return country


def validate_date(input_date: date, 
                 min_date: Optional[date] = None,
                 max_date: Optional[date] = None) -> date:
    """
    Validate date input with optional range checking
    
    Args:
        input_date: Date to validate
        min_date: Minimum allowed date
        max_date: Maximum allowed date
        
    Returns:
        Validated date
        
    Raises:
        ValueError: If date is invalid or out of range
    """
    if not input_date:
        raise ValueError("Date is required")
    
    # Convert string to date if needed
    if isinstance(input_date, str):
        try:
            input_date = datetime.strptime(input_date, "%Y-%m-%d").date()
        except ValueError:
            raise ValueError("Date must be in YYYY-MM-DD format")
    
    # Range validation
    if min_date and input_date < min_date:
        raise ValueError(f"Date cannot be before {min_date}")
    
    if max_date and input_date > max_date:
        raise ValueError(f"Date cannot be after {max_date}")
    
    return input_date


def validate_model_type(model_type: str, 
                       allowed_types: Optional[List[str]] = None) -> str:
    """
    Validate model type
    
    Args:
        model_type: Model type to validate
        allowed_types: List of allowed model types
        
    Returns:
        Validated model type
        
    Raises:
        ValueError: If model type is invalid
    """
    if not model_type or not isinstance(model_type, str):
        raise ValueError("Model type is required")
    
    model_type = model_type.lower().strip()
    
    if allowed_types is None:
        allowed_types = ["prophet", "arima", "both", "ensemble"]
    
    if model_type not in allowed_types:
        raise ValueError(
            f"Model type '{model_type}' not supported. "
            f"Allowed types: {', '.join(allowed_types)}"
        )
    
    return model_type


def validate_forecast_parameters(forecast_days: int,
                                confidence_interval: float) -> tuple:
    """
    Validate forecasting parameters
    
    Args:
        forecast_days: Number of days to forecast
        confidence_interval: Confidence interval (0.5-0.99)
        
    Returns:
        Tuple of validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    if not isinstance(forecast_days, int) or forecast_days <= 0:
        raise ValueError("Forecast days must be a positive integer")
    
    if forecast_days > settings.max_forecast_days:
        raise ValueError(f"Forecast days cannot exceed {settings.max_forecast_days}")
    
    if not isinstance(confidence_interval, (int, float)):
        raise ValueError("Confidence interval must be a number")
    
    if not 0.5 <= confidence_interval <= 0.99:
        raise ValueError("Confidence interval must be between 0.5 and 0.99")
    
    return forecast_days, confidence_interval


def validate_training_parameters(start_date: Optional[date],
                                end_date: Optional[date],
                                retrain: bool) -> tuple:
    """
    Validate model training parameters
    
    Args:
        start_date: Training start date
        end_date: Training end date
        retrain: Whether to force retrain
        
    Returns:
        Tuple of validated parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    if start_date and end_date:
        if start_date >= end_date:
            raise ValueError("Start date must be before end date")
        
        training_days = (end_date - start_date).days
        if training_days < settings.min_training_days:
            raise ValueError(f"Training period must be at least {settings.min_training_days} days")
        
        if training_days > settings.max_training_days:
            raise ValueError(f"Training period cannot exceed {settings.max_training_days} days")
    
    if not isinstance(retrain, bool):
        raise ValueError("Retrain parameter must be boolean")
    
    return start_date, end_date, retrain


def sanitize_log_parameters(log_type: str,
                           country: Optional[str],
                           limit: int,
                           offset: int) -> tuple:
    """
    Sanitize log query parameters
    
    Args:
        log_type: Type of logs to retrieve
        country: Country filter
        limit: Maximum records to return
        offset: Pagination offset
        
    Returns:
        Tuple of sanitized parameters
        
    Raises:
        ValueError: If parameters are invalid
    """
    # Validate log type
    allowed_log_types = ["train", "predict", "all"]
    if log_type not in allowed_log_types:
        raise ValueError(f"Log type must be one of: {', '.join(allowed_log_types)}")
    
    # Validate country if provided
    if country:
        country = validate_country(country)
    
    # Validate pagination parameters
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("Limit must be a positive integer")
    
    if limit > 1000:
        limit = 1000  # Cap to prevent large responses
    
    if not isinstance(offset, int) or offset < 0:
        raise ValueError("Offset must be a non-negative integer")
    
    return log_type, country, limit, offset


def validate_json_structure(data: dict, required_fields: List[str]) -> dict:
    """
    Validate JSON request structure
    
    Args:
        data: JSON data to validate
        required_fields: List of required field names
        
    Returns:
        Validated data
        
    Raises:
        ValueError: If required fields are missing
    """
    if not isinstance(data, dict):
        raise ValueError("Request body must be valid JSON")
    
    missing_fields = [field for field in required_fields if field not in data]
    if missing_fields:
        raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")
    
    return data


def sanitize_string_input(input_str: str, max_length: int = 255) -> str:
    """
    Sanitize string input for security
    
    Args:
        input_str: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValueError: If string is invalid
    """
    if not isinstance(input_str, str):
        raise ValueError("Input must be a string")
    
    # Remove potentially harmful characters
    sanitized = re.sub(r'[<>"\';\\]', '', input_str.strip())
    
    if len(sanitized) > max_length:
        raise ValueError(f"Input too long. Maximum {max_length} characters allowed")
    
    return sanitized
