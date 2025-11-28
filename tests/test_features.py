"""Tests for feature engineering components."""

from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from openquant.features.cache import FeatureCache
from openquant.features.engine import FeatureEngine
from openquant.features.registry import FeatureRegistry, register_feature


class TestFeatureRegistry:
    """Tests for FeatureRegistry."""

    def test_register_and_get(self):
        """Test registering and retrieving features."""
        registry = FeatureRegistry()

        def test_feature(df: pd.DataFrame) -> pd.Series:
            return df["Close"] * 2

        registry.register("test_feature", test_feature)
        assert registry.has_feature("test_feature")
        assert registry.get("test_feature") is not None

    def test_list_features(self):
        """Test listing all features."""
        registry = FeatureRegistry()
        registry.register("feature1", lambda df: df["Close"])
        registry.register("feature2", lambda df: df["Volume"])

        features = registry.list_features()
        assert "feature1" in features
        assert "feature2" in features

    def test_get_nonexistent_feature(self):
        """Test getting a non-existent feature."""
        registry = FeatureRegistry()
        assert registry.get("nonexistent") is None
        assert not registry.has_feature("nonexistent")


class TestFeatureEngine:
    """Tests for FeatureEngine."""

    def test_compute_features(self):
        """Test computing features from OHLCV data."""
        engine = FeatureEngine()

        # Create sample OHLCV data
        dates = pd.date_range("2024-01-01", periods=100, freq="D")
        df = pd.DataFrame(
            {
                "Date": dates.date,
                "Open": range(100, 200),
                "High": range(101, 201),
                "Low": range(99, 199),
                "Close": range(100, 200),
                "Volume": [1000000] * 100,
            }
        )

        # Compute a subset of features
        features = engine.compute_features(df, feature_names=["sma_20", "rsi_14"])

        assert "sma_20" in features.columns
        assert "rsi_14" in features.columns
        assert len(features) == 100

    def test_compute_features_missing_columns(self):
        """Test error handling for missing columns."""
        engine = FeatureEngine()
        df = pd.DataFrame({"Close": [100, 101, 102]})

        with pytest.raises(ValueError, match="Missing required columns"):
            engine.compute_features(df)

    def test_compute_features_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        engine = FeatureEngine()
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

        result = engine.compute_features(df)
        assert result.empty

    def test_compute_features_for_ticker(self, tmp_path):
        """Test computing features for a ticker from storage."""
        from openquant.ingestion.storage import DuckDBStorage

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

        # Compute features
        engine = FeatureEngine(storage=storage)
        features = engine.compute_features_for_ticker("AAPL", feature_names=["sma_20"])

        assert "sma_20" in features.columns
        assert len(features) == 100

    def test_compute_features_for_ticker_no_storage(self):
        """Test error when storage is not initialized."""
        engine = FeatureEngine()

        with pytest.raises(ValueError, match="Storage not initialized"):
            engine.compute_features_for_ticker("AAPL")


class TestFeatureCache:
    """Tests for FeatureCache."""

    @patch("openquant.features.cache.redis.from_url")
    def test_get_cache_hit(self, mock_redis_from_url):
        """Test retrieving cached features."""
        mock_client = MagicMock()
        mock_redis_from_url.return_value = mock_client

        # Create test data
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "sma_20": [100.0, 101.0],
            }
        )

        import json
        cached_data = {
            "data": df.to_json(orient="records", date_format="iso"),
            "ticker": "AAPL",
            "feature_names": ["sma_20"],
        }
        mock_client.get.return_value = json.dumps(cached_data)
        mock_client.ping.return_value = True

        cache = FeatureCache(redis_url="redis://localhost:6379/0")
        result = cache.get("AAPL", ["sma_20"])

        assert result is not None
        assert "sma_20" in result.columns
        assert len(result) == 2

    @patch("openquant.features.cache.redis.from_url")
    def test_get_cache_miss(self, mock_redis_from_url):
        """Test cache miss scenario."""
        mock_client = MagicMock()
        mock_redis_from_url.return_value = mock_client
        mock_client.get.return_value = None
        mock_client.ping.return_value = True

        cache = FeatureCache(redis_url="redis://localhost:6379/0")
        result = cache.get("AAPL", ["sma_20"])

        assert result is None

    @patch("openquant.features.cache.redis.from_url")
    def test_set_cache(self, mock_redis_from_url):
        """Test caching features."""
        mock_client = MagicMock()
        mock_redis_from_url.return_value = mock_client
        mock_client.setex.return_value = True
        mock_client.ping.return_value = True

        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01"]).date,
                "sma_20": [100.0],
            }
        )

        cache = FeatureCache(redis_url="redis://localhost:6379/0")
        result = cache.set("AAPL", ["sma_20"], df)

        assert result is True
        mock_client.setex.assert_called_once()

    @patch("openquant.features.cache.redis.from_url")
    def test_delete_cache(self, mock_redis_from_url):
        """Test deleting cached features."""
        mock_client = MagicMock()
        mock_redis_from_url.return_value = mock_client
        mock_client.scan_iter.return_value = ["features:AAPL:key1", "features:AAPL:key2"]
        mock_client.delete.return_value = 2
        mock_client.ping.return_value = True

        cache = FeatureCache(redis_url="redis://localhost:6379/0")
        deleted = cache.delete("AAPL")

        assert deleted == 2

    @patch("openquant.features.cache.redis.from_url")
    def test_exists(self, mock_redis_from_url):
        """Test checking if cache exists."""
        mock_client = MagicMock()
        mock_redis_from_url.return_value = mock_client
        mock_client.exists.return_value = 1
        mock_client.ping.return_value = True

        cache = FeatureCache(redis_url="redis://localhost:6379/0")
        exists = cache.exists("AAPL", ["sma_20"])

        assert exists is True

