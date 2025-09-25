"""FastAPI application for EV charging demand forecasting.

This module provides a production-ready REST API for serving machine learning
predictions with comprehensive error handling, validation, and monitoring.
"""

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import xgboost as xgb
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, validator, ConfigDict, field_validator
from contextlib import asynccontextmanager
from ev_forecast.utils.config import settings
from ev_forecast.utils.logging import get_logger

# Initialize logger
api_logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "api_requests_total", "Total API requests", ["method", "endpoint", "status"]
)
REQUEST_DURATION = Histogram(
    "api_request_duration_seconds", "Request duration in seconds"
)
PREDICTION_COUNT = Counter("predictions_total", "Total predictions made")
MODEL_LOAD_TIME = Histogram("model_load_duration_seconds", "Model loading time")

# FastAPI app configuration


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Context manager for managing the lifespan of the API.
    Initializes the model on startup and cleans up on shutdown.
    """
    api_logger.info("Starting EV Charging Forecast API")
    load_model()
    yield
    api_logger.info("Shutting down EV Charging Forecast API")


app = FastAPI(
    title="EV Charging Demand Forecast API",
    description=(
        "Advanced ML API for predicting EV charging demand at charging stations"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
model_metadata = {}


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""

    site_id: int = Field(..., ge=1, description="Site ID (must be positive integer)")
    timestamp: str = Field(..., description="ISO 8601 timestamp")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(
        ..., ge=0, le=6, description="Day of week (0=Monday, 6=Sunday)"
    )
    is_weekend: int = Field(..., ge=0, le=1, description="Weekend indicator (0 or 1)")
    hour_sin: float = Field(..., ge=-1, le=1, description="Sine encoding of hour")
    hour_cos: float = Field(..., ge=-1, le=1, description="Cosine encoding of hour")
    lag_1: Optional[float] = Field(None, ge=0, description="1-hour lag feature")
    lag_24: Optional[float] = Field(None, ge=0, description="24-hour lag feature")
    rmean_24: Optional[float] = Field(None, ge=0, description="24-hour rolling mean")

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp(cls, v: Any) -> str:
        """Validate timestamp format."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
            return str(v)
        except ValueError:
            raise ValueError("Invalid timestamp format. Use ISO 8601 format.")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "site_id": 1,
                "timestamp": "2024-01-15T14:30:00Z",
                "hour_of_day": 14,
                "day_of_week": 0,
                "is_weekend": 0,
                "hour_sin": 0.5,
                "hour_cos": 0.866,
                "lag_1": 5.2,
                "lag_24": 4.8,
                "rmean_24": 5.0,
            }
        }
    )


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""

    prediction: float = Field(..., description="Predicted charging sessions")
    site_id: int = Field(..., description="Site ID")
    timestamp: str = Field(..., description="Prediction timestamp")
    model_version: str = Field(..., description="Model version used")
    confidence_interval: Optional[Dict[str, float]] = Field(
        None, description="Prediction confidence interval"
    )
    processing_time_ms: float = Field(
        ..., description="Processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""

    status: str = Field(..., description="API status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    model_path: str = Field(..., description="Path to model file")
    model_version: Optional[str] = Field(None, description="Model version")
    uptime_seconds: float = Field(..., description="API uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")


class ErrorResponse(BaseModel):
    """Response model for errors."""

    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")


# Startup time for uptime calculation
startup_time = time.time()


def load_model() -> None:
    """Load the machine learning model."""
    global model, model_metadata

    model_path = settings.api.model_path

    if not os.path.exists(model_path):
        api_logger.warning(f"Model file not found: {model_path}")
        return

    try:
        with MODEL_LOAD_TIME.time():
            api_logger.info(f"Loading model from {model_path}")
            model = joblib.load(model_path)

            # Extract model metadata
            model_metadata = {
                "path": model_path,
                "loaded_at": datetime.now().isoformat(),
                "file_size": os.path.getsize(model_path),
                "version": "1.0.0",  # Could be extracted from model or filename
            }

        api_logger.info("Model loaded successfully")

    except Exception as e:
        api_logger.error(f"Failed to load model: {e}")
        model = None
        model_metadata = {}


@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Any) -> Any:
    """Add processing time and request tracking."""
    start_time = time.time()

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)

    # Update metrics
    REQUEST_DURATION.observe(process_time)
    REQUEST_COUNT.labels(
        method=request.method, endpoint=request.url.path, status=response.status_code
    ).inc()

    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    """Handle HTTP exceptions with detailed error responses."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            error_code=f"HTTP_{exc.status_code}",
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID"),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Handle general exceptions."""
    api_logger.error(f"Unhandled exception: {exc}", exc_info=True)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            timestamp=datetime.now().isoformat(),
            request_id=request.headers.get("X-Request-ID"),
        ).dict(),
    )


@app.get("/", response_model=HealthResponse, tags=["Health"])
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None,
        model_path=settings.api.model_path,
        model_version=str(model_metadata.get("version")),
        uptime_seconds=time.time() - startup_time,
        timestamp=datetime.now().isoformat(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def detailed_health() -> HealthResponse:
    """Detailed health check endpoint."""
    return await health_check()


@app.post("/predict", response_model=PredictResponse, tags=["Predictions"])
async def predict_demand(request: PredictRequest) -> PredictResponse:
    """Predict EV charging demand for a given site and time."""
    start_time = time.time()

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model file and restart the service.",
        )

    try:
        # Prepare input data
        input_data = pd.DataFrame([request.model_dump(exclude={"timestamp"})])

        # Handle missing lag features
        for col in ["lag_1", "lag_24", "rmean_24"]:
            if col not in input_data.columns or input_data[col].isna().any():
                input_data[col] = 0.0  # Default value for missing lags

        # Create XGBoost DMatrix
        dmatrix = xgb.DMatrix(input_data)

        # Make prediction
        prediction = model.predict(dmatrix)[0]

        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000

        # Update metrics
        PREDICTION_COUNT.inc()

        # Log prediction
        api_logger.info(
            f"Prediction made: site_id={request.site_id}, "
            f"prediction={prediction:.3f}, processing_time={processing_time:.2f}ms"
        )

        return PredictResponse(
            prediction=float(prediction),
            site_id=request.site_id,
            timestamp=request.timestamp,
            model_version=str(model_metadata.get("version", "unknown")),
            confidence_interval={
                "lower": float(prediction * 0.9),
                "upper": float(prediction * 1.1),
            },  # Example
            processing_time_ms=processing_time,
        )

    except Exception as e:
        api_logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(requests: List[PredictRequest]) -> Dict[str, Any]:
    """Batch prediction endpoint for multiple requests."""
    if len(requests) > 100:  # Limit batch size
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size too large. Maximum 100 requests per batch.",
        )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please check model file and restart the service.",
        )

    try:
        start_time = time.time()

        # Convert requests to DataFrame
        input_data = pd.DataFrame(
            [req.model_dump(exclude={"timestamp"}) for req in requests]
        )

        # Handle missing lag features
        for col in ["lag_1", "lag_24", "rmean_24"]:
            if col not in input_data.columns:
                input_data[col] = 0.0
            input_data[col] = input_data[col].fillna(0.0)

        # Make batch predictions
        dmatrix = xgb.DMatrix(input_data)
        predictions = model.predict(dmatrix)

        processing_time = (time.time() - start_time) * 1000

        # Format responses
        responses = []
        for i, (req, pred) in enumerate(zip(requests, predictions)):
            responses.append(
                PredictResponse(
                    prediction=float(pred),
                    site_id=req.site_id,
                    timestamp=req.timestamp,
                    model_version=str(model_metadata.get("version", "unknown")),
                    confidence_interval={
                        "lower": float(pred * 0.9),
                        "upper": float(pred * 1.1),
                    },  # Example
                    processing_time_ms=processing_time / len(requests),
                )
            )

        PREDICTION_COUNT.inc(len(requests))

        api_logger.info(
            f"Batch prediction completed: {len(requests)} predictions in "
            f"{processing_time:.2f}ms"
        )

        return {"predictions": responses, "total_count": len(responses)}

    except Exception as e:
        api_logger.error(f"Batch prediction failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/metrics", tags=["Monitoring"])
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/model/reload", tags=["Management"])
async def reload_model() -> Dict[str, Any]:
    """Reload the machine learning model."""
    try:
        load_model()
        if model is not None:
            return {
                "message": "Model reloaded successfully",
                "timestamp": datetime.now().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to reload model",
            )
    except Exception as e:
        api_logger.error(f"Model reload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}",
        )


@app.get("/model/info", tags=["Management"])
async def get_model_info() -> Dict[str, Any]:
    """Get information about the loaded model."""
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No model loaded"
        )

    return {
        "model_metadata": model_metadata,
        "model_type": type(model).__name__,
        "features_count": getattr(model, "num_features", "unknown"),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "ev_forecast.api.app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.api.reload,
        workers=settings.api.workers,
    )
