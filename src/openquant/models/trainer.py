"""Model training with MLflow integration."""

from typing import Any, Dict, List, Optional

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from openquant.config.settings import settings
from openquant.features.engine import FeatureEngine
from openquant.features.cache import FeatureCache
from openquant.ingestion.storage import DuckDBStorage
from openquant.models.registry import BaseModel, ModelRegistry, get_registry


class ModelTrainer:
    """Trainer for ML models with MLflow tracking."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        storage: Optional[DuckDBStorage] = None,
        feature_engine: Optional[FeatureEngine] = None,
        feature_cache: Optional[FeatureCache] = None,
    ) -> None:
        """Initialize the model trainer.

        Args:
            registry: Model registry instance. Defaults to global registry.
            storage: DuckDB storage instance. Defaults to None.
            feature_engine: Feature engine instance. Defaults to None.
            feature_cache: Feature cache instance. Defaults to None.
        """
        self.registry = registry or get_registry()
        self.storage = storage
        self.feature_engine = feature_engine
        self.feature_cache = feature_cache

        # Initialize MLflow
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

    def prepare_data(
        self,
        ticker: str,
        feature_names: List[str],
        target: str = "returns",
        lookback_days: int = 1,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> tuple[np.ndarray, np.ndarray, List[str]]:
        """Prepare training data from features.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names to use.
            target: Target variable name (e.g., 'returns', 'Close').
            lookback_days: Number of days to predict ahead. Defaults to 1.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            Tuple of (X, y, feature_names) where X is feature matrix and y is target vector.

        Raises:
            ValueError: If data preparation fails.
        """
        if self.feature_engine is None or self.storage is None:
            raise ValueError("Feature engine and storage must be initialized")

        # Get features
        features_df = self.feature_engine.compute_features_for_ticker(
            ticker, start_date, end_date, feature_names
        )

        if features_df.empty:
            raise ValueError(f"No features found for {ticker}")

        # Handle target variable
        if target == "returns":
            # If returns is already in features, use it directly
            if "returns" in features_df.columns:
                df = features_df.copy()
                df["target"] = df["returns"].shift(-lookback_days)
            else:
                # Compute returns from Close price
                ohlcv_df = self.storage.get_ohlcv(ticker, start_date, end_date)
                if "Date" in features_df.columns and "Date" in ohlcv_df.columns:
                    df = pd.merge(features_df, ohlcv_df[["Date", "Close"]], on="Date", how="inner")
                else:
                    df = pd.concat([features_df, ohlcv_df[["Close"]]], axis=1)
                # Compute returns: (Close_t / Close_{t-1}) - 1
                df["returns"] = df["Close"].pct_change()
                df["target"] = df["returns"].shift(-lookback_days)
        else:
            # For other targets (e.g., "Close"), get from OHLCV
            ohlcv_df = self.storage.get_ohlcv(ticker, start_date, end_date)
            if "Date" in features_df.columns and "Date" in ohlcv_df.columns:
                df = pd.merge(features_df, ohlcv_df[["Date", target]], on="Date", how="inner")
            else:
                df = pd.concat([features_df, ohlcv_df[[target]]], axis=1)
            # For price targets, compute future return
            df["target"] = (df[target].shift(-lookback_days) / df[target] - 1)

        # Drop rows with NaN
        df = df.dropna()

        if df.empty:
            raise ValueError("No valid data after preprocessing")

        # Select feature columns (exclude target if it's in the feature list to avoid using it as both feature and target)
        feature_cols = [col for col in feature_names if col in df.columns and col != target]
        if not feature_cols:
            raise ValueError(f"None of the specified features found in data: {feature_names}")

        X = df[feature_cols].values
        y = df["target"].values

        logger.info(f"Prepared data: {len(X)} samples, {len(feature_cols)} features")
        return X, y, feature_cols

    def train(
        self,
        ticker: str,
        model_type: str,
        feature_names: List[str],
        model_params: Optional[Dict[str, Any]] = None,
        target: str = "returns",
        lookback_days: int = 1,
        test_size: float = 0.2,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        run_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Train a model with MLflow tracking.

        Args:
            ticker: Stock ticker symbol.
            model_type: Type of model to train (e.g., 'xgboost', 'lightgbm').
            feature_names: List of feature names to use.
            model_params: Model hyperparameters. Defaults to None.
            target: Target variable name. Defaults to 'returns'.
            lookback_days: Number of days to predict ahead. Defaults to 1.
            test_size: Proportion of data for testing. Defaults to 0.2.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).
            run_name: MLflow run name. Defaults to None.

        Returns:
            Dictionary with training results including run_id, metrics, and model path.

        Raises:
            ValueError: If training fails.
        """
        model_params = model_params or {}

        # Prepare data
        X, y, feature_cols = self.prepare_data(
            ticker, feature_names, target, lookback_days, start_date, end_date
        )

        # Split data
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Create model
        model = self.registry.create(model_type, **model_params)

        # Start MLflow run
        run_name = run_name or f"{ticker}_{model_type}_{target}"
        with mlflow.start_run(run_name=run_name):
            # Log parameters
            mlflow.log_params({
                "ticker": ticker,
                "model_type": model_type,
                "target": target,
                "lookback_days": lookback_days,
                "test_size": test_size,
                "n_features": len(feature_cols),
                "n_train_samples": len(X_train),
                "n_test_samples": len(X_test),
            })
            mlflow.log_params(model_params)
            mlflow.log_params({f"feature_{i}": feat for i, feat in enumerate(feature_cols)})

            # Train model
            logger.info(f"Training {model_type} model for {ticker}")
            model.fit(X_train, y_train)

            # Evaluate
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_metrics = {
                "train_mae": mean_absolute_error(y_train, y_train_pred),
                "train_mse": mean_squared_error(y_train, y_train_pred),
                "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "train_r2": r2_score(y_train, y_train_pred),
            }

            test_metrics = {
                "test_mae": mean_absolute_error(y_test, y_test_pred),
                "test_mse": mean_squared_error(y_test, y_test_pred),
                "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
                "test_r2": r2_score(y_test, y_test_pred),
            }

            # Log metrics
            mlflow.log_metrics(train_metrics)
            mlflow.log_metrics(test_metrics)

            # Log model
            if model_type == "xgboost":
                mlflow.xgboost.log_model(model.model, "model")
            elif model_type == "lightgbm":
                mlflow.lightgbm.log_model(model.model, "model")
            else:
                mlflow.sklearn.log_model(model.model, "model")

            run_id = mlflow.active_run().info.run_id
            logger.info(f"Training completed. Run ID: {run_id}")
            logger.info(f"Test metrics: {test_metrics}")

            return {
                "run_id": run_id,
                "ticker": ticker,
                "model_type": model_type,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_names": feature_cols,
            }

