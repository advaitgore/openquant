"""Tests for FastAPI endpoints."""

from unittest.mock import MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from openquant.api.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "database" in data

    def test_readiness_check(self, client):
        """Test readiness check endpoint."""
        response = client.get("/health/ready")
        # May return 200 or 503 depending on database status
        assert response.status_code in [200, 503]

    def test_liveness_check(self, client):
        """Test liveness check endpoint."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json() == {"status": "alive"}


class TestDataEndpoints:
    """Tests for data endpoints."""

    @patch("openquant.api.routers.data._storage")
    def test_list_tickers(self, mock_storage, client):
        """Test listing tickers."""
        mock_storage.get_tickers.return_value = ["AAPL", "MSFT", "GOOGL"]
        response = client.get("/data/tickers")
        assert response.status_code == 200
        data = response.json()
        assert "tickers" in data
        assert len(data["tickers"]) == 3

    @patch("openquant.api.routers.data._storage")
    def test_get_ohlcv(self, mock_storage, client):
        """Test getting OHLCV data."""
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        mock_storage.get_ohlcv.return_value = mock_df

        response = client.get("/data/AAPL")
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert len(data["data"]) == 2

    @patch("openquant.api.routers.data._storage")
    def test_get_ohlcv_not_found(self, mock_storage, client):
        """Test getting OHLCV data when not found."""
        import pandas as pd

        mock_storage.get_ohlcv.return_value = pd.DataFrame()
        response = client.get("/data/INVALID")
        assert response.status_code == 404


class TestFeaturesEndpoints:
    """Tests for features endpoints."""

    @patch("openquant.api.routers.features._feature_engine")
    def test_list_features(self, mock_engine, client):
        """Test listing features."""
        mock_engine.registry.list_features.return_value = ["sma_20", "rsi_14", "macd"]
        response = client.get("/features/list")
        assert response.status_code == 200
        data = response.json()
        assert "features" in data
        assert len(data["features"]) == 3

    @patch("openquant.api.routers.features._feature_cache")
    @patch("openquant.api.routers.features._feature_engine")
    def test_compute_features(self, mock_engine, mock_cache, client):
        """Test computing features."""
        import pandas as pd

        # Mock feature computation
        mock_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01"]).date,
                "sma_20": [100.0],
                "rsi_14": [50.0],
            }
        )
        mock_engine.compute_features_for_ticker.return_value = mock_df
        mock_cache.get.return_value = None  # Cache miss

        request = {
            "ticker": "AAPL",
            "feature_names": ["sma_20", "rsi_14"],
        }
        response = client.post("/features/compute", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert len(data["data"]) == 1


class TestModelsEndpoints:
    """Tests for models endpoints."""

    @patch("openquant.api.routers.models._trainer")
    def test_train_model(self, mock_trainer, client):
        """Test training a model."""
        mock_trainer.train.return_value = {
            "run_id": "test_run_id",
            "ticker": "AAPL",
            "model_type": "xgboost",
            "train_metrics": {"train_r2": 0.8, "train_rmse": 0.1},
            "test_metrics": {"test_r2": 0.75, "test_rmse": 0.12},
            "feature_names": ["sma_20", "rsi_14"],
        }

        request = {
            "ticker": "AAPL",
            "model_type": "xgboost",
            "feature_names": ["sma_20", "rsi_14"],
        }
        response = client.post("/models/train", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test_run_id"
        assert data["ticker"] == "AAPL"

    @patch("openquant.api.routers.models._predictor")
    def test_predict(self, mock_predictor, client):
        """Test making predictions."""
        import pandas as pd

        mock_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01"]).date,
                "prediction": [0.01],
            }
        )
        mock_predictor.predict.return_value = mock_df

        request = {
            "run_id": "test_run_id",
            "ticker": "AAPL",
        }
        response = client.post("/models/predict", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == "test_run_id"
        assert len(data["predictions"]) == 1

    @patch("openquant.api.routers.models._predictor")
    def test_predict_latest(self, mock_predictor, client):
        """Test predicting latest days."""
        import numpy as np

        mock_predictor.predict_latest.return_value = np.array([0.01, 0.02])
        response = client.get("/models/predict/test_run_id/AAPL/latest?n_days=2")
        assert response.status_code == 200
        data = response.json()
        assert len(data["predictions"]) == 2


class TestRootEndpoint:
    """Tests for root endpoint."""

    def test_root(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert data["name"] == "OpenQuant API"

