"""Features router for feature computation endpoints."""

from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException

from openquant.api.schemas import FeatureData, FeatureRequest, FeatureResponse
from openquant.config.settings import settings
from openquant.features.cache import FeatureCache
from openquant.features.engine import FeatureEngine
from openquant.ingestion.storage import DuckDBStorage
from loguru import logger

router = APIRouter(prefix="/features", tags=["features"])

# Initialize components
_storage = DuckDBStorage(settings.DUCKDB_PATH)
_feature_engine = FeatureEngine(storage=_storage)
_feature_cache = FeatureCache() if settings.REDIS_URL else None


@router.post("/compute", response_model=FeatureResponse)
async def compute_features(request: FeatureRequest) -> FeatureResponse:
    """Compute features for a ticker.

    Args:
        request: Feature computation request.

    Returns:
        Computed features.

    Raises:
        HTTPException: If feature computation fails.
    """
    try:
        ticker = request.ticker
        feature_names = request.feature_names
        use_cache = request.use_cache and _feature_cache is not None

        # Check cache first
        cached_df = None
        if use_cache:
            cached_df = _feature_cache.get(
                ticker, feature_names or [], request.start_date, request.end_date
            )

        if cached_df is not None:
            logger.info(f"Retrieved features from cache for {ticker}")
            # Convert to response format
            feature_data = []
            for _, row in cached_df.iterrows():
                features_dict = {
                    col: float(row[col]) if pd.notna(row[col]) else None
                    for col in cached_df.columns
                    if col not in ["Date", "Ticker"]
                }
                feature_data.append(
                    FeatureData(
                        date=row.get("Date"),
                        features=features_dict,
                    )
                )

            return FeatureResponse(
                ticker=ticker,
                feature_names=feature_names or [],
                data=feature_data,
                count=len(feature_data),
                cached=True,
            )

        # Compute features
        features_df = _feature_engine.compute_features_for_ticker(
            ticker, request.start_date, request.end_date, feature_names
        )

        if features_df.empty:
            raise HTTPException(
                status_code=404, detail=f"No features computed for ticker {ticker}"
            )

        # Cache results
        if use_cache:
            _feature_cache.set(
                ticker,
                feature_names or [],
                features_df,
                request.start_date,
                request.end_date,
            )

        # Convert to response format
        feature_data = []
        for _, row in features_df.iterrows():
            features_dict = {
                col: float(row[col]) if pd.notna(row[col]) else None
                for col in features_df.columns
                if col not in ["Date", "Ticker"]
            }
            feature_data.append(
                FeatureData(
                    date=row.get("Date"),
                    features=features_dict,
                )
            )

        # Get feature names from computed features
        computed_feature_names = [
            col for col in features_df.columns if col not in ["Date", "Ticker"]
        ]

        return FeatureResponse(
            ticker=ticker,
            feature_names=computed_feature_names,
            data=feature_data,
            count=len(feature_data),
            cached=False,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error computing features for {request.ticker}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error computing features: {e}"
        )


@router.get("/list", response_model=dict)
async def list_features() -> dict:
    """List all available features.

    Returns:
        List of available feature names.
    """
    try:
        feature_names = _feature_engine.registry.list_features()
        return {"features": feature_names, "count": len(feature_names)}
    except Exception as e:
        logger.error(f"Error listing features: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing features: {e}")

