"""Pytest configuration and shared fixtures."""

import pytest
from unittest.mock import Mock, MagicMock

from openquant.config.settings import Settings


@pytest.fixture
def settings() -> Settings:
    """Provide test settings instance."""
    return Settings(
        DUCKDB_PATH=":memory:",
        REDIS_URL="redis://localhost:6379/1",
        MLFLOW_TRACKING_URI="file:./test_mlruns",
    )


@pytest.fixture
def mock_redis() -> MagicMock:
    """Provide a mock Redis client."""
    redis_mock = MagicMock()
    redis_mock.get.return_value = None
    redis_mock.set.return_value = True
    redis_mock.delete.return_value = 1
    redis_mock.exists.return_value = False
    redis_mock.keys.return_value = []
    return redis_mock


@pytest.fixture
def mock_duckdb() -> MagicMock:
    """Provide a mock DuckDB connection."""
    conn_mock = MagicMock()
    conn_mock.execute.return_value = None
    conn_mock.fetchdf.return_value = None
    conn_mock.close.return_value = None
    return conn_mock


@pytest.fixture
def mock_mlflow() -> MagicMock:
    """Provide a mock MLflow client."""
    mlflow_mock = MagicMock()
    mlflow_mock.create_experiment.return_value = "experiment_id"
    mlflow_mock.start_run.return_value.__enter__ = Mock(return_value=mlflow_mock)
    mlflow_mock.start_run.return_value.__exit__ = Mock(return_value=None)
    mlflow_mock.log_params.return_value = None
    mlflow_mock.log_metrics.return_value = None
    mlflow_mock.log_model.return_value = None
    return mlflow_mock
