"""
Revenue Forecasting API

Production-ready FastAPI application for time series revenue forecasting
using Prophet and ARIMA models with comprehensive monitoring and logging.

Author: Adryan R A
"""

import os
import logging
from datetime import datetime, date
from typing import Optional, Dict, List, Union
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

from services.data_service import DataService
from services.model_service import ModelService
from services.log_service import LogService
from utils.config import get_settings
from utils.validation import validate_country, validate_date
from utils.monitoring import ModelMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration
settings = get_settings()

# Global services
data_service = DataService()
model_service = ModelService()
log_service = LogService()
monitor = ModelMonitor()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Revenue Forecasting API")
    
    # Initialize services
    await data_service.initialize()
    await model_service.initialize()
    await monitor.initialize()
    
    # Load existing models
    await model_service.load_available_models()
    
    logger.info("API initialization complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Revenue Forecasting API")
    await data_service.cleanup()
    await model_service.cleanup()


# Create FastAPI app
app = FastAPI(
    title="Revenue Forecasting API",
    description="Time series revenue forecasting using Prophet and ARIMA models",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for request/response
class TrainRequest(BaseModel):
    country: str = Field(..., description="Country name for model training")
    model_type: str = Field(default="both", regex="^(prophet|arima|both)$")
    start_date: Optional[date] = Field(None, description="Training start date")
    end_date: Optional[date] = Field(None, description="Training end date") 
    retrain: bool = Field(default=False, description="Force retrain existing model")
    
    @validator('country')
    def validate_country_name(cls, v):
        return validate_country(v)
    
    @validator('start_date', 'end_date')
    def validate_dates(cls, v):
        if v:
            return validate_date(v)
        return v


class PredictRequest(BaseModel):
    country: str = Field(..., description="Country for prediction")
    prediction_date: date = Field(..., description="Date to make prediction from")
    forecast_days: int = Field(default=30, ge=1, le=365, description="Days to forecast")
    model_type: str = Field(default="ensemble", regex="^(prophet|arima|ensemble)$")
    confidence_interval: float = Field(default=0.95, ge=0.5, le=0.99)
    
    @validator('country')
    def validate_country_name(cls, v):
        return validate_country(v)
    
    @validator('prediction_date')
    def validate_prediction_date(cls, v):
        return validate_date(v)


class TrainResponse(BaseModel):
    status: str
    message: str
    country: str
    models_trained: List[str]
    training_metrics: Dict[str, Dict[str, float]]
    training_time: float
    model_versions: Dict[str, str]


class PredictResponse(BaseModel):
    status: str
    country: str
    prediction_date: str
    forecast_days: int
    model_type: str
    predictions: Dict[str, Union[List[float], Dict[str, List[float]]]]
    confidence_intervals: Optional[Dict[str, Dict[str, List[float]]]]
    model_performance: Dict[str, float]
    prediction_time: float


class LogsResponse(BaseModel):
    status: str
    log_type: str
    total_records: int
    logs: List[Dict]


class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    version: str
    database_status: str
    models_loaded: Dict[str, List[str]]
    system_metrics: Dict[str, Union[str, float, int]]


# API Endpoints

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Revenue Forecasting API",
        "version": "1.0.0",
        "status": "healthy",
        "documentation": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Check database connectivity
        db_status = await data_service.check_connection()
        
        # Get loaded models
        models_loaded = await model_service.get_loaded_models()
        
        # Get system metrics
        system_metrics = await monitor.get_system_metrics()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            database_status=db_status,
            models_loaded=models_loaded,
            system_metrics=system_metrics
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@app.post("/train", response_model=TrainResponse)
async def train_models(
    request: TrainRequest,
    background_tasks: BackgroundTasks
):
    """Train time series models for revenue forecasting"""
    try:
        logger.info(f"Training request for country: {request.country}, model: {request.model_type}")
        
        # Check if models already exist and retrain is False
        if not request.retrain:
            existing_models = await model_service.get_existing_models(request.country)
            if existing_models and request.model_type in existing_models:
                logger.info(f"Models already exist for {request.country}. Use retrain=true to force retrain.")
                return TrainResponse(
                    status="skipped",
                    message="Models already exist. Use retrain=true to force retrain.",
                    country=request.country,
                    models_trained=[],
                    training_metrics={},
                    training_time=0.0,
                    model_versions={}
                )
        
        # Load and prepare training data
        training_data = await data_service.get_training_data(
            country=request.country,
            start_date=request.start_date,
            end_date=request.end_date
        )
        
        if training_data is None or len(training_data) == 0:
            raise HTTPException(
                status_code=400, 
                detail=f"No training data available for country: {request.country}"
            )
        
        # Train models
        start_time = datetime.now()
        training_result = await model_service.train_models(
            data=training_data,
            country=request.country,
            model_types=request.model_type,
            retrain=request.retrain
        )
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Log training event
        background_tasks.add_task(
            log_service.log_training,
            country=request.country,
            model_type=request.model_type,
            training_metrics=training_result['metrics'],
            training_time=training_time,
            data_size=len(training_data)
        )
        
        # Update model monitoring
        background_tasks.add_task(
            monitor.update_training_metrics,
            country=request.country,
            metrics=training_result['metrics']
        )
        
        return TrainResponse(
            status="success",
            message=f"Successfully trained {len(training_result['models_trained'])} models",
            country=request.country,
            models_trained=training_result['models_trained'],
            training_metrics=training_result['metrics'],
            training_time=training_time,
            model_versions=training_result['versions']
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@app.post("/predict", response_model=PredictResponse)
async def predict_revenue(
    request: PredictRequest,
    background_tasks: BackgroundTasks
):
    """Generate revenue forecasts"""
    try:
        logger.info(f"Prediction request for {request.country} from {request.prediction_date}")
        
        # Validate that models exist for the country
        available_models = await model_service.get_existing_models(request.country)
        if not available_models:
            raise HTTPException(
                status_code=404,
                detail=f"No trained models found for country: {request.country}"
            )
        
        # Get historical data for context
        historical_data = await data_service.get_historical_data(
            country=request.country,
            end_date=request.prediction_date
        )
        
        # Generate predictions
        start_time = datetime.now()
        prediction_result = await model_service.predict(
            country=request.country,
            prediction_date=request.prediction_date,
            forecast_days=request.forecast_days,
            model_type=request.model_type,
            confidence_interval=request.confidence_interval,
            historical_data=historical_data
        )
        prediction_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction event
        background_tasks.add_task(
            log_service.log_prediction,
            country=request.country,
            prediction_date=request.prediction_date,
            model_type=request.model_type,
            prediction_result=prediction_result,
            prediction_time=prediction_time
        )
        
        # Update monitoring
        background_tasks.add_task(
            monitor.update_prediction_metrics,
            country=request.country,
            model_type=request.model_type,
            performance=prediction_result['performance']
        )
        
        return PredictResponse(
            status="success",
            country=request.country,
            prediction_date=request.prediction_date.isoformat(),
            forecast_days=request.forecast_days,
            model_type=request.model_type,
            predictions=prediction_result['forecasts'],
            confidence_intervals=prediction_result.get('confidence_intervals'),
            model_performance=prediction_result['performance'],
            prediction_time=prediction_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/logs", response_model=LogsResponse)
async def get_logs(
    log_type: str = Query(..., regex="^(train|predict|all)$", description="Type of logs to retrieve"),
    country: Optional[str] = Query(None, description="Filter by country"),
    model_type: Optional[str] = Query(None, description="Filter by model type"),
    start_date: Optional[date] = Query(None, description="Start date for log filtering"),
    end_date: Optional[date] = Query(None, description="End date for log filtering"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of records"),
    offset: int = Query(0, ge=0, description="Offset for pagination")
):
    """Retrieve training and prediction logs"""
    try:
        logs = await log_service.get_logs(
            log_type=log_type,
            country=country,
            model_type=model_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset
        )
        
        return LogsResponse(
            status="success",
            log_type=log_type,
            total_records=len(logs),
            logs=logs
        )
        
    except Exception as e:
        logger.error(f"Failed to retrieve logs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve logs: {str(e)}")


@app.get("/models", response_model=Dict[str, Dict])
async def get_model_info():
    """Get information about available models"""
    try:
        model_info = await model_service.get_model_information()
        return {
            "status": "success",
            "models": model_info
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@app.get("/monitoring/drift", response_model=Dict)
async def get_drift_analysis():
    """Get model drift analysis"""
    try:
        drift_analysis = await monitor.get_drift_analysis()
        return {
            "status": "success",
            "drift_analysis": drift_analysis
        }
    except Exception as e:
        logger.error(f"Failed to get drift analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get drift analysis: {str(e)}")


@app.get("/monitoring/performance", response_model=Dict)
async def get_performance_analysis():
    """Get model performance analysis vs business metrics"""
    try:
        performance_analysis = await monitor.get_performance_analysis()
        return {
            "status": "success",
            "performance_analysis": performance_analysis
        }
    except Exception as e:
        logger.error(f"Failed to get performance analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance analysis: {str(e)}")


# Error handlers
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.error(f"ValueError: {exc}")
    return JSONResponse(
        status_code=400,
        content={"status": "error", "detail": str(exc)}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"status": "error", "detail": "Internal server error"}
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug_mode,
        log_level="info"
    )
