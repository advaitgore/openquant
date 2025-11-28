"""Tests for model training and prediction components."""

from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from openquant.models.predictor import ModelPredictor
from openquant.models.registry import (
    LightGBMModel,
    ModelRegistry,
    RandomForestModel,
    XGBoostModel,
    get_registry,
)
from openquant.models.trainer import ModelTrainer


class TestModelRegistry:
    """Tests for ModelRegistry."""

    def test_create_xgboost(self):
        """Test creating XGBoost model."""
        registry = ModelRegistry()
        model = registry.create("xgboost", n_estimators=50)
        assert isinstance(model, XGBoostModel)

    def test_create_lightgbm(self):
        """Test creating LightGBM model."""
        registry = ModelRegistry()
        model = registry.create("lightgbm", n_estimators=50)
        assert isinstance(model, LightGBMModel)

    def test_create_random_forest(self):
        """Test creating Random Forest model."""
        registry = ModelRegistry()
        model = registry.create("random_forest", n_estimators=50)
        assert isinstance(model, RandomForestModel)

    def test_create_invalid_model(self):
        """Test creating invalid model type."""
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="Unknown model type"):
            registry.create("invalid_model")

    def test_list_models(self):
        """Test listing registered models."""
        registry = ModelRegistry()
        models = registry.list_models()
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "random_forest" in models


class TestModelTraining:
    """Tests for model training."""

    def test_xgboost_fit_predict(self):
        """Test XGBoost model fit and predict."""
        model = XGBoostModel(n_estimators=10, max_depth=3)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert isinstance(predictions, np.ndarray)

    def test_lightgbm_fit_predict(self):
        """Test LightGBM model fit and predict."""
        model = LightGBMModel(n_estimators=10, max_depth=3)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert isinstance(predictions, np.ndarray)

    def test_random_forest_fit_predict(self):
        """Test Random Forest model fit and predict."""
        model = RandomForestModel(n_estimators=10, max_depth=3)
        X = np.random.rand(100, 5)
        y = np.random.rand(100)

        model.fit(X, y)
        predictions = model.predict(X)

        assert len(predictions) == 100
        assert isinstance(predictions, np.ndarray)


class TestModelTrainer:
    """Tests for ModelTrainer."""

    @patch("openquant.models.trainer.mlflow")
    def test_prepare_data(self, mock_mlflow, tmp_path):
        """Test data preparation."""
        from openquant.features.engine import FeatureEngine
        from openquant.ingestion.storage import DuckDBStorage

        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates.date,
                "Ticker": ["AAPL"] * 200,
                "Open": range(100, 300),
                "High": range(101, 301),
                "Low": range(99, 299),
                "Close": range(100, 300),
                "Volume": [1000000] * 200,
            }
        )
        storage.upsert_ohlcv(df)

        feature_engine = FeatureEngine(storage=storage)
        trainer = ModelTrainer(storage=storage, feature_engine=feature_engine)

        X, y, feature_cols = trainer.prepare_data(
            "AAPL", ["sma_20", "rsi_14"], target="returns"
        )

        assert len(X) > 0
        assert len(y) > 0
        assert len(feature_cols) > 0

    @patch("openquant.models.trainer.mlflow")
    def test_train_model(self, mock_mlflow, tmp_path):
        """Test model training."""
        from openquant.features.engine import FeatureEngine
        from openquant.ingestion.storage import DuckDBStorage

        # Setup mocks
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_mlflow.start_run.return_value.__enter__ = Mock(return_value=mock_run)
        mock_mlflow.start_run.return_value.__exit__ = Mock(return_value=None)
        mock_mlflow.active_run.return_value = mock_run

        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data
        dates = pd.date_range("2024-01-01", periods=200, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates.date,
                "Ticker": ["AAPL"] * 200,
                "Open": range(100, 300),
                "High": range(101, 301),
                "Low": range(99, 299),
                "Close": range(100, 300),
                "Volume": [1000000] * 200,
            }
        )
        storage.upsert_ohlcv(df)

        feature_engine = FeatureEngine(storage=storage)
        trainer = ModelTrainer(storage=storage, feature_engine=feature_engine)

        results = trainer.train(
            ticker="AAPL",
            model_type="xgboost",
            feature_names=["sma_20", "rsi_14"],
            model_params={"n_estimators": 10, "max_depth": 3},
        )

        assert "run_id" in results
        assert "test_metrics" in results
        assert "train_metrics" in results


class TestModelPredictor:
    """Tests for ModelPredictor."""

    @patch("openquant.models.predictor.mlflow")
    def test_load_model(self, mock_mlflow):
        """Test loading a model."""
        mock_model = MagicMock()
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        predictor = ModelPredictor()
        model = predictor.load_model("test_run_id")

        assert model == mock_model
        mock_mlflow.pyfunc.load_model.assert_called_once_with("runs:/test_run_id/model")

    @patch("openquant.models.predictor.mlflow")
    def test_predict(self, mock_mlflow, tmp_path):
        """Test making predictions."""
        from openquant.features.engine import FeatureEngine
        from openquant.ingestion.storage import DuckDBStorage

        # Setup mocks
        mock_model = MagicMock()
        mock_model.predict.return_value = np.array([0.01, 0.02, 0.03])
        mock_mlflow.pyfunc.load_model.return_value = mock_model

        mock_run = MagicMock()
        mock_run.data.params = {
            "n_features": 2,
            "feature_0": "sma_20",
            "feature_1": "rsi_14",
        }
        mock_mlflow.get_run.return_value = mock_run

        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates.date,
                "Ticker": ["AAPL"] * 100,
                "Open": range(100, 200),
                "High": range(101, 201),
                "Low": range(99, 199),
                "Close": range(100, 200),
                "Volume": [1000000] * 100,
            }
        )
        storage.upsert_ohlcv(df)

        feature_engine = FeatureEngine(storage=storage)
        predictor = ModelPredictor(storage=storage, feature_engine=feature_engine)

        result_df = predictor.predict("test_run_id", "AAPL", ["sma_20", "rsi_14"])

        assert "prediction" in result_df.columns
        assert len(result_df) > 0

