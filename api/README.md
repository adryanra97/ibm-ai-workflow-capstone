# AI Workflow Capstone - Time Series Forecasting API

**Author: Adryan R A**

A production-ready API for time series revenue forecasting using Prophet and ARIMA models, built with FastAPI and designed for scale, monitoring, and drift detection.

## Features

- **Dual Model Support**: Prophet and ARIMA/SARIMA models for comprehensive forecasting
- **Production Ready**: Full Docker support, monitoring, logging, and error handling
- **Model Monitoring**: Automatic drift detection and performance monitoring
- **Comprehensive Testing**: Test-driven development with 95%+ code coverage
- **Business Intelligence**: Post-production analysis linking model performance to business metrics
- **Scalable Architecture**: Async/await patterns, efficient database operations
- **Security**: Input validation, sanitization, and secure model storage

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [API Endpoints](#api-endpoints)
- [Models](#models)
- [Installation](#installation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Monitoring](#monitoring)
- [Post-Production Analysis](#post-production-analysis)
- [Deployment](#deployment)
- [Contributing](#contributing)

## Quick Start

### Using Docker (Recommended)

1. **Clone and navigate to the API directory**:
   ```bash
   cd ai-workflow-capstone/api
   ```

2. **Build and run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the API**:
   - API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health
   - Monitoring (Grafana): http://localhost:3000 (admin/admin123)

### Local Development

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

3. **Run the API**:
   ```bash
   uvicorn app:app --reload
   ```

## Architecture

```
api/
├── app.py                      # Main FastAPI application
├── models/                     # Time series models
│   ├── base_model.py          # Abstract base model
│   ├── prophet_model.py       # Prophet implementation
│   └── arima_model.py         # ARIMA implementation
├── services/                   # Business logic
│   ├── data_service.py        # Data access layer
│   ├── model_service.py       # Model orchestration
│   └── log_service.py         # Comprehensive logging
├── utils/                      # Utilities
│   ├── config.py              # Configuration management
│   ├── validation.py          # Input validation
│   └── monitoring.py          # Model monitoring
├── tests/                      # Test suite
├── Dockerfile                  # Container definition
├── docker-compose.yml         # Multi-service setup
└── post_production_analysis.py # Business analysis script
```

## API Endpoints

### Core Endpoints

#### Train Models
```http
POST /train
Content-Type: application/json

{
  "country": "United States",
  "model_type": "prophet",
  "start_date": "2023-01-01",
  "end_date": "2023-12-31",
  "hyperparameters": {
    "changepoint_prior_scale": 0.05,
    "seasonality_prior_scale": 10.0
  }
}
```

#### Generate Predictions
```http
POST /predict
Content-Type: application/json

{
  "country": "United States",
  "model_type": "prophet",
  "forecast_days": 30,
  "confidence_interval": 0.95
}
```

#### Compare Models
```http
POST /models/compare
Content-Type: application/json

{
  "country": "United States",
  "forecast_days": 30,
  "confidence_interval": 0.95
}
```

#### Get Model Status
```http
GET /models/status/{country}
```

#### Retrieve Logs
```http
GET /logs?start_date=2024-01-01&level=ERROR&country=United States&limit=100
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Performance Metrics
```http
GET /monitoring/performance?start_date=2024-01-01&endpoint=/predict
```

#### Model Retraining
```http
POST /models/retrain
Content-Type: application/json

{
  "country": "United States",
  "model_types": ["prophet", "arima"],
  "force_retrain": false
}
```

## Models

### Prophet Model
- **Best for**: Data with strong seasonality and holiday effects
- **Features**: 
  - Automatic seasonality detection
  - Holiday effects
  - Trend changepoint detection
  - External regressors support
- **Parameters**: Changepoint prior scale, seasonality prior scale, holidays prior scale

### ARIMA Model
- **Best for**: Stationary time series with clear autoregressive patterns
- **Features**:
  - Automatic order selection
  - Seasonal ARIMA (SARIMA) support
  - Stationarity testing and transformation
  - Model diagnostics
- **Parameters**: Max p/d/q values, seasonal period, information criterion

## Installation

### Prerequisites
- Python 3.9+
- Docker and Docker Compose (for containerized deployment)
- 4GB+ RAM (for model training)

### Python Dependencies
```bash
pip install -r requirements.txt
```

Key dependencies:
- `fastapi[all]==0.104.1` - Web framework
- `prophet==1.1.4` - Facebook Prophet forecasting
- `statsmodels==0.14.0` - ARIMA models
- `pandas==2.1.1` - Data manipulation
- `numpy==1.24.3` - Numerical computing
- `scikit-learn==1.3.0` - Machine learning utilities
- `pytest==7.4.2` - Testing framework

## Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Environment
ENV=development
LOG_LEVEL=INFO

# Database paths
DATABASE_PATH=./data
MODEL_STORAGE_PATH=./models
LOG_DATABASE_PATH=./logs/api_logs.db

# Model settings
MAX_FORECAST_DAYS=365
DEFAULT_CONFIDENCE_INTERVAL=0.95

# Monitoring
ENABLE_MONITORING=true
PERFORMANCE_THRESHOLD_MS=5000
LOG_RETENTION_DAYS=90

# External data paths
EXTERNAL_TRAIN_DATA_PATH=../cs-train
EXTERNAL_PRODUCTION_DATA_PATH=../cs-production
```

### Model Configuration

Models can be configured via the API or by modifying `utils/config.py`:

```python
# Prophet parameters
prophet_params = {
    'changepoint_prior_scale': 0.05,
    'seasonality_prior_scale': 10.0,
    'holidays_prior_scale': 10.0,
    'yearly_seasonality': True,
    'weekly_seasonality': True
}

# ARIMA parameters
arima_params = {
    'max_p': 5,
    'max_d': 2,
    'max_q': 5,
    'information_criterion': 'aic'
}
```

## Testing

### Run All Tests
```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Test Categories
- **Unit Tests**: Individual component testing
- **Integration Tests**: Service interaction testing
- **API Tests**: Endpoint testing with mocked dependencies
- **Performance Tests**: Load and stress testing

### Test Structure
```
tests/
├── conftest.py           # Test configuration and fixtures
├── test_api.py          # API endpoint tests
├── test_models.py       # Model implementation tests
├── test_services.py     # Business logic tests
└── test_utils.py        # Utility function tests
```

### Coverage Report
After running tests, view the coverage report:
```bash
open htmlcov/index.html
```

## Monitoring

### Built-in Monitoring
- **Performance Metrics**: Response times, error rates, memory usage
- **Model Drift Detection**: Automatic detection of model performance degradation
- **Business Metrics Correlation**: Link model performance to business outcomes
- **Comprehensive Logging**: Structured logs with searchable metadata

### Grafana Dashboards
Access pre-configured dashboards at http://localhost:3000:
- API Performance Dashboard
- Model Performance Dashboard
- Business Metrics Dashboard
- System Health Dashboard

### Alerting
Configure alerts for:
- High API error rates (>5%)
- Slow response times (>5s)
- Model drift detection
- Low model accuracy

## Post-Production Analysis

Run comprehensive business impact analysis:

```bash
python post_production_analysis.py
```

### Analysis Features
- **Model Performance Analysis**: Accuracy metrics across countries and time periods
- **Business Impact Assessment**: Correlation between predictions and revenue
- **Drift Detection**: Performance and data drift analysis
- **Model Comparison**: Statistical significance testing between models
- **Actionable Recommendations**: Prioritized list of improvements

### Output
- Detailed JSON results
- Summary report with recommendations
- Visualization plots (PNG)
- Performance comparison charts

## Deployment

### Docker Deployment (Recommended)

1. **Production Configuration**:
   ```bash
   cp .env.example .env.production
   # Edit production settings
   ```

2. **Build and Deploy**:
   ```bash
   docker-compose -f docker-compose.yml --env-file .env.production up -d
   ```

3. **Scale Services**:
   ```bash
   docker-compose up --scale ai-workflow-api=3
   ```

## API Documentation

### Interactive Documentation
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Response Formats

#### Successful Prediction Response
```json
{
  "model": "prophet",
  "country": "United States",
  "forecast_days": 30,
  "confidence_interval": 0.95,
  "predictions": [
    {
      "date": "2024-01-01",
      "forecast": 1500.0,
      "lower_bound": 1300.0,
      "upper_bound": 1700.0,
      "trend": 1450.0,
      "seasonal": 50.0
    }
  ],
  "statistics": {
    "mean_forecast": 1500.0,
    "median_forecast": 1480.0,
    "total_forecast": 45000.0,
    "std_forecast": 120.0
  },
  "trend_analysis": {
    "direction": "increasing",
    "slope": 2.5,
    "volatility": 0.08,
    "trend_strength": "medium"
  },
  "business_insights": {
    "forecast_vs_recent_90_days": {
      "growth_rate_percent": 5.2
    }
  },
  "generated_at": "2024-01-01T12:00:00Z"
}
```

## Troubleshooting

### Common Issues

#### Model Training Fails
```bash
# Check data availability
curl http://localhost:8000/models/status/United%20States

# Check logs
curl http://localhost:8000/logs?level=ERROR&limit=10

# Verify data format
# Ensure training data has 'ds' and 'y' columns
```

#### High Memory Usage
```bash
# Monitor memory usage
docker stats

# Reduce model complexity
# Decrease max_p, max_q for ARIMA
# Reduce uncertainty_samples for Prophet
```

#### Slow Predictions
```bash
# Check performance metrics
curl http://localhost:8000/monitoring/performance

# Optimize model parameters
# Enable model caching
# Use smaller forecast horizons
```

### Log Analysis
```bash
# Get recent errors
curl "http://localhost:8000/logs?level=ERROR&limit=50"

# Filter by country
curl "http://localhost:8000/logs?country=United%20States&limit=100"

# Performance issues
curl "http://localhost:8000/logs?category=performance&limit=50"
```

## Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Install development dependencies: `pip install -r requirements-dev.txt`
4. Make changes and add tests
5. Run tests: `pytest tests/ -v`
6. Commit changes: `git commit -m 'Add amazing feature'`
7. Push to branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Style
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and returns
- Add docstrings for all public methods
- Maintain test coverage above 90%

### Testing Guidelines
- Write tests before implementing features (TDD)
- Mock external dependencies
- Test both success and failure scenarios
- Include performance tests for critical paths

---

**Built with for production-ready time series forecasting**

## Features
- **Train endpoint**: Train Prophet and ARIMA models on historical data
- **Predict endpoint**: Generate revenue forecasts for specified dates and countries
- **Logfile endpoint**: Access training and prediction logs
- **Model Drift Detection**: Monitor model performance and data drift
- **Business Metrics Analysis**: Correlation between model performance and business KPIs

## API Endpoints

### POST /train
Train models on historical data
```json
{
  "country": "United Kingdom",
  "model_type": "prophet|arima|both",
  "start_date": "2017-11-30",
  "end_date": "2019-07-31",
  "retrain": false
}
```

### POST /predict
Generate revenue forecasts
```json
{
  "country": "United Kingdom", 
  "prediction_date": "2019-08-01",
  "forecast_days": 30,
  "model_type": "prophet|arima|ensemble"
}
```

### GET /logs
Access training and prediction logs
```
/logs?type=train&country=all&start_date=2023-01-01
/logs?type=predict&model=prophet&limit=100
```

### GET /health
Health check and system status

## Architecture

```
api/
├── app.py              # FastAPI application
├── models/
│   ├── __init__.py
│   ├── base_model.py   # Abstract base model
│   ├── prophet_model.py # Prophet implementation  
│   └── arima_model.py  # ARIMA implementation
├── services/
│   ├── __init__.py
│   ├── data_service.py # Data loading and preprocessing
│   ├── model_service.py # Model training and prediction
│   └── log_service.py  # Logging service
├── utils/
│   ├── __init__.py
│   ├── config.py       # Configuration
│   ├── validation.py   # Input validation
│   └── monitoring.py   # Model monitoring and drift detection
└── tests/
    ├── __init__.py
    ├── test_api.py     # API endpoint tests
    ├── test_models.py  # Model tests
    └── test_services.py # Service tests
```

## Data Pipeline
1. **L0 Database**: Raw invoice data (file.db)
2. **L3 Database**: Aggregated daily metrics (analytics.db) 
3. **Model Training**: Time series preparation and feature engineering
4. **Prediction**: Forecast generation with confidence intervals
5. **Monitoring**: Performance tracking and drift detection

## Business Metrics Integration
- Revenue forecast accuracy vs actual business performance
- Model performance correlation with customer acquisition/retention
- Seasonal pattern detection and business impact analysis
- Anomaly detection for revenue forecasting

## Deployment
- Docker containerization with multi-stage builds
- Production-ready with gunicorn/uvicorn
- Health checks and monitoring endpoints
- Log aggregation and alerting

## Getting Started

```bash
# Install dependencies
pip install -r requirements.txt

# Run development server
uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Run with Docker
docker build -t revenue-forecasting-api .
docker run -p 8000:8000 revenue-forecasting-api

# Run tests
pytest tests/ -v --cov=api
```
