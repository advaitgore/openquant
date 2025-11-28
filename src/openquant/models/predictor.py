"""Model prediction with MLflow model loading."""

from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
from loguru import logger

from openquant.config.settings import settings
from openquant.features.engine import FeatureEngine
from openquant.features.cache import FeatureCache
from openquant.ingestion.storage import DuckDBStorage


class ModelPredictor:
    """Predictor for loading models and making predictions."""

    def __init__(
        self,
        storage: Optional[DuckDBStorage] = None,
        feature_engine: Optional[FeatureEngine] = None,
        feature_cache: Optional[FeatureCache] = None,
    ) -> None:
        """Initialize the model predictor.

        Args:
            storage: DuckDB storage instance. Defaults to None.
            feature_engine: Feature engine instance. Defaults to None.
            feature_cache: Feature cache instance. Defaults to None.
        """
        self.storage = storage
        self.feature_engine = feature_engine
        self.feature_cache = feature_cache

        # Initialize MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def load_model(self, run_id: str):
        """Load a model from MLflow by run ID.

        Args:
            run_id: MLflow run ID.

        Returns:
            Loaded model.

        Raises:
            ValueError: If model cannot be loaded.
        """
        try:
            model = mlflow.pyfunc.load_model(f"runs:/{run_id}/model")
            logger.info(f"Loaded model from run {run_id}")
            return model
        except Exception as e:
            logger.error(f"Error loading model {run_id}: {e}")
            raise ValueError(f"Failed to load model {run_id}: {e}") from e

    def predict(
        self,
        run_id: str,
        ticker: str,
        feature_names: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Make predictions using a trained model.

        Args:
            run_id: MLflow run ID of the trained model.
            ticker: Stock ticker symbol.
            feature_names: List of feature names. If None, uses features from training.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            DataFrame with predictions and dates.

        Raises:
            ValueError: If prediction fails.
        """
        if self.feature_engine is None or self.storage is None:
            raise ValueError("Feature engine and storage must be initialized")

        # Load model
        model = self.load_model(run_id)

        # Get run info to determine features used
        run = mlflow.get_run(run_id)
        run_params = run.data.params

        # Extract feature names from run params if not provided
        if feature_names is None:
            feature_names = [
                run_params[f"feature_{i}"]
                for i in range(int(run_params.get("n_features", 0)))
            ]

        if not feature_names:
            raise ValueError("No feature names available")

        # Get features
        features_df = self.feature_engine.compute_features_for_ticker(
            ticker, start_date, end_date, feature_names
        )

        if features_df.empty:
            raise ValueError(f"No features found for {ticker}")

        # Prepare feature matrix
        feature_cols = [col for col in feature_names if col in features_df.columns]
        if not feature_cols:
            raise ValueError(f"None of the specified features found: {feature_names}")

        X = features_df[feature_cols].values

        # Make predictions
        predictions = model.predict(X)

        # Create results DataFrame
        result_df = pd.DataFrame({
            "prediction": predictions,
        })

        if "Date" in features_df.columns:
            result_df["Date"] = features_df["Date"].values
        if "Ticker" in features_df.columns:
            result_df["Ticker"] = features_df["Ticker"].values

        logger.info(f"Made {len(predictions)} predictions for {ticker}")
        return result_df

    def predict_latest(
        self,
        run_id: str,
        ticker: str,
        feature_names: Optional[List[str]] = None,
        n_days: int = 1,
    ) -> np.ndarray:
        """Make predictions for the latest N days.

        Args:
            run_id: MLflow run ID of the trained model.
            ticker: Stock ticker symbol.
            feature_names: List of feature names. If None, uses features from training.
            n_days: Number of latest days to predict. Defaults to 1.

        Returns:
            Array of predictions.
        """
        if self.storage is None:
            raise ValueError("Storage must be initialized")

        # Get latest date
        latest_date = self.storage.get_latest_date(ticker)
        if not latest_date:
            raise ValueError(f"No data found for {ticker}")

        # Get data for last N days
        from openquant.utils.time import get_lookback_date

        start_date = get_lookback_date(days=n_days * 2)  # Get extra data for features
        end_date = latest_date

        result_df = self.predict(run_id, ticker, feature_names, start_date, end_date)

        # Return last N predictions
        return result_df["prediction"].tail(n_days).values

