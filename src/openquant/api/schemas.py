from __future__ import annotations

"""Pydantic schemas for API request/response models."""

from datetime import date as date_type
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    database: str = Field(..., description="Database status")
    redis: str = Field(..., description="Redis status")
    mlflow: str = Field(..., description="MLflow status")


class OHLCVData(BaseModel):
    """OHLCV data point."""

    date: date_type = Field(..., description="Trading date")
    open: float = Field(..., description="Opening price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Closing price")
    volume: int = Field(..., description="Trading volume")


class DataResponse(BaseModel):
    """OHLCV data response."""

    ticker: str = Field(..., description="Stock ticker symbol")
    data: List[OHLCVData] = Field(..., description="OHLCV data points")
    count: int = Field(..., description="Number of data points")


class TickerListResponse(BaseModel):
    """List of available tickers."""

    tickers: List[str] = Field(..., description="List of ticker symbols")
    count: int = Field(..., description="Number of tickers")


class FeatureRequest(BaseModel):
    """Feature computation request."""

    ticker: str = Field(..., description="Stock ticker symbol")
    feature_names: Optional[List[str]] = Field(None, description="List of feature names to compute")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")
    use_cache: bool = Field(True, description="Whether to use Redis cache")


class FeatureData(BaseModel):
    """Feature data point."""

    date: Optional[date_type] = Field(None, description="Trading date")
    features: Dict[str, Optional[float]] = Field(..., description="Feature values")


class FeatureResponse(BaseModel):
    """Feature computation response."""

    ticker: str = Field(..., description="Stock ticker symbol")
    feature_names: List[str] = Field(..., description="List of computed features")
    data: List[FeatureData] = Field(..., description="Feature data points")
    count: int = Field(..., description="Number of data points")
    cached: bool = Field(False, description="Whether data was retrieved from cache")


class ModelTrainRequest(BaseModel):
    """Model training request."""

    ticker: str = Field(..., description="Stock ticker symbol")
    model_type: str = Field(..., description="Model type (xgboost, lightgbm, etc.)")
    feature_names: List[str] = Field(..., description="List of feature names")
    model_params: Optional[Dict[str, Any]] = Field(None, description="Model hyperparameters")
    target: str = Field("returns", description="Target variable name")
    lookback_days: int = Field(1, description="Number of days ahead to predict")
    test_size: float = Field(0.2, description="Test set size (0.0 to 1.0)")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class ModelTrainResponse(BaseModel):
    """Model training response."""

    run_id: str = Field(..., description="MLflow run ID")
    ticker: str = Field(..., description="Stock ticker symbol")
    model_type: str = Field(..., description="Model type")
    train_metrics: Dict[str, float] = Field(..., description="Training metrics")
    test_metrics: Dict[str, float] = Field(..., description="Test metrics")
    feature_names: List[str] = Field(..., description="Features used")


class ModelPredictRequest(BaseModel):
    """Model prediction request."""

    run_id: str = Field(..., description="MLflow run ID")
    ticker: str = Field(..., description="Stock ticker symbol")
    feature_names: Optional[List[str]] = Field(None, description="List of feature names")
    start_date: Optional[str] = Field(None, description="Start date in YYYY-MM-DD format")
    end_date: Optional[str] = Field(None, description="End date in YYYY-MM-DD format")


class PredictionData(BaseModel):
    """Prediction data point."""

    date: Optional[date_type] = Field(None, description="Trading date")
    prediction: float = Field(..., description="Predicted value")


class ModelPredictResponse(BaseModel):
    """Model prediction response."""

    run_id: str = Field(..., description="MLflow run ID")
    ticker: str = Field(..., description="Stock ticker symbol")
    predictions: List[PredictionData] = Field(..., description="Predictions")
    count: int = Field(..., description="Number of predictions")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")

