"""Models router for model training and prediction endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from openquant.api.schemas import (
    ModelPredictRequest,
    ModelPredictResponse,
    ModelTrainRequest,
    ModelTrainResponse,
    PredictionData,
)
from openquant.config.settings import settings
from openquant.features.cache import FeatureCache
from openquant.features.engine import FeatureEngine
from openquant.ingestion.storage import DuckDBStorage
from openquant.models.predictor import ModelPredictor
from openquant.models.trainer import ModelTrainer
from loguru import logger

router = APIRouter(prefix="/models", tags=["models"])

# Initialize components
_storage = DuckDBStorage(settings.DUCKDB_PATH)
_feature_engine = FeatureEngine(storage=_storage)
_feature_cache = FeatureCache() if settings.REDIS_URL else None
_trainer = ModelTrainer(
    storage=_storage,
    feature_engine=_feature_engine,
    feature_cache=_feature_cache,
)
_predictor = ModelPredictor(
    storage=_storage,
    feature_engine=_feature_engine,
    feature_cache=_feature_cache,
)


@router.post("/train", response_model=ModelTrainResponse)
async def train_model(request: ModelTrainRequest) -> ModelTrainResponse:
    """Train a model for a ticker.

    Args:
        request: Model training request.

    Returns:
        Training results with metrics.

    Raises:
        HTTPException: If training fails.
    """
    try:
        logger.info(f"Training {request.model_type} model for {request.ticker}")

        results = _trainer.train(
            ticker=request.ticker,
            model_type=request.model_type,
            feature_names=request.feature_names,
            model_params=request.model_params,
            target=request.target,
            lookback_days=request.lookback_days,
            test_size=request.test_size,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        return ModelTrainResponse(
            run_id=results["run_id"],
            ticker=results["ticker"],
            model_type=results["model_type"],
            train_metrics=results["train_metrics"],
            test_metrics=results["test_metrics"],
            feature_names=results["feature_names"],
        )

    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=f"Error training model: {e}")


@router.post("/predict", response_model=ModelPredictResponse)
async def predict(request: ModelPredictRequest) -> ModelPredictResponse:
    """Make predictions using a trained model.

    Args:
        request: Prediction request.

    Returns:
        Predictions for the specified ticker.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        predictions_df = _predictor.predict(
            run_id=request.run_id,
            ticker=request.ticker,
            feature_names=request.feature_names,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        if predictions_df.empty:
            raise HTTPException(
                status_code=404, detail="No predictions generated"
            )

        # Convert to response format
        prediction_data = [
            PredictionData(
                date=row.get("Date"),
                prediction=float(row["prediction"]),
            )
            for _, row in predictions_df.iterrows()
        ]

        return ModelPredictResponse(
            run_id=request.run_id,
            ticker=request.ticker,
            predictions=prediction_data,
            count=len(prediction_data),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error making predictions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error making predictions: {e}"
        )


@router.get("/predict/{run_id}/{ticker}/latest")
async def predict_latest(
    run_id: str,
    ticker: str,
    n_days: int = 1,
) -> ModelPredictResponse:
    """Make predictions for the latest N days.

    Args:
        run_id: MLflow run ID.
        ticker: Stock ticker symbol.
        n_days: Number of latest days to predict. Defaults to 1.

    Returns:
        Latest predictions.

    Raises:
        HTTPException: If prediction fails.
    """
    try:
        predictions = _predictor.predict_latest(run_id, ticker, n_days=n_days)

        # Create response
        prediction_data = [
            PredictionData(prediction=float(pred)) for pred in predictions
        ]

        return ModelPredictResponse(
            run_id=run_id,
            ticker=ticker,
            predictions=prediction_data,
            count=len(prediction_data),
        )

    except Exception as e:
        logger.error(f"Error making latest predictions: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error making predictions: {e}"
        )

