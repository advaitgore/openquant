"""Tests for data ingestion components."""

from datetime import datetime
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest

from openquant.ingestion.providers import YFinanceProvider, get_provider
from openquant.ingestion.storage import DuckDBStorage
from openquant.ingestion.scheduler import IngestionScheduler


class TestYFinanceProvider:
    """Tests for YFinanceProvider."""

    @patch("openquant.ingestion.providers.yf.Ticker")
    def test_fetch_ohlcv_success(self, mock_ticker_class):
        """Test successful data fetch."""
        # Setup mock
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker

        # Create sample data
        dates = pd.date_range("2024-01-01", periods=5, freq="D")
        mock_df = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
                "High": [101.0, 102.0, 103.0, 104.0, 105.0],
                "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000000, 1100000, 1200000, 1300000, 1400000],
            }
        )
        mock_ticker.history.return_value = mock_df

        # Test
        provider = YFinanceProvider()
        result = provider.fetch_ohlcv("AAPL", "2024-01-01", "2024-01-05")

        # Assertions
        assert len(result) == 5
        assert "Ticker" in result.columns
        assert result["Ticker"].iloc[0] == "AAPL"
        assert all(col in result.columns for col in ["Date", "Open", "High", "Low", "Close", "Volume"])

    @patch("openquant.ingestion.providers.yf.Ticker")
    def test_fetch_ohlcv_empty(self, mock_ticker_class):
        """Test handling of empty data."""
        mock_ticker = MagicMock()
        mock_ticker_class.return_value = mock_ticker
        mock_ticker.history.return_value = pd.DataFrame()

        provider = YFinanceProvider()
        with pytest.raises(ValueError, match="No data returned"):
            provider.fetch_ohlcv("INVALID", "2024-01-01", "2024-01-05")

    def test_get_provider(self):
        """Test provider factory function."""
        provider = get_provider("yfinance")
        assert isinstance(provider, YFinanceProvider)

    def test_get_provider_invalid(self):
        """Test provider factory with invalid name."""
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("invalid_provider")


class TestDuckDBStorage:
    """Tests for DuckDBStorage."""

    def test_initialize_schema(self, tmp_path):
        """Test schema initialization."""
        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Verify schema exists by checking if we can query
        with storage.get_connection() as conn:
            result = conn.execute("SELECT COUNT(*) FROM ohlcv").fetchone()
            assert result[0] == 0

    def test_upsert_ohlcv(self, tmp_path):
        """Test upserting OHLCV data."""
        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Create test data
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "Ticker": ["AAPL", "AAPL"],
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )

        rows = storage.upsert_ohlcv(df)
        assert rows == 2

        # Verify data
        result = storage.get_ohlcv("AAPL")
        assert len(result) == 2

    def test_get_ohlcv(self, tmp_path):
        """Test retrieving OHLCV data."""
        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]).date,
                "Ticker": ["AAPL", "AAPL", "AAPL"],
                "Open": [100.0, 101.0, 102.0],
                "High": [101.0, 102.0, 103.0],
                "Low": [99.0, 100.0, 101.0],
                "Close": [100.5, 101.5, 102.5],
                "Volume": [1000000, 1100000, 1200000],
            }
        )
        storage.upsert_ohlcv(df)

        # Test retrieval
        result = storage.get_ohlcv("AAPL", "2024-01-01", "2024-01-02")
        assert len(result) == 2

    def test_get_latest_date(self, tmp_path):
        """Test getting latest date for a ticker."""
        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime(["2024-01-01", "2024-01-02"]).date,
                "Ticker": ["AAPL", "AAPL"],
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.0, 100.0],
                "Close": [100.5, 101.5],
                "Volume": [1000000, 1100000],
            }
        )
        storage.upsert_ohlcv(df)

        latest = storage.get_latest_date("AAPL")
        assert latest == "2024-01-02"

        # Test with no data
        latest = storage.get_latest_date("INVALID")
        assert latest is None

    def test_get_tickers(self, tmp_path):
        """Test getting list of tickers."""
        db_path = str(tmp_path / "test.duckdb")
        storage = DuckDBStorage(db_path)
        storage.initialize_schema()

        # Insert test data for multiple tickers
        for ticker in ["AAPL", "MSFT"]:
            df = pd.DataFrame(
                {
                    "Date": pd.to_datetime(["2024-01-01"]).date,
                    "Ticker": [ticker],
                    "Open": [100.0],
                    "High": [101.0],
                    "Low": [99.0],
                    "Close": [100.5],
                    "Volume": [1000000],
                }
            )
            storage.upsert_ohlcv(df)

        tickers = storage.get_tickers()
        assert set(tickers) == {"AAPL", "MSFT"}


class TestIngestionScheduler:
    """Tests for IngestionScheduler."""

    def test_add_daily_job(self):
        """Test adding a daily job."""
        scheduler = IngestionScheduler()
        mock_job = Mock()

        scheduler.add_daily_job(mock_job, hour=16, minute=30)

        assert len(scheduler.scheduler.get_jobs()) == 1

    def test_add_interval_job(self):
        """Test adding an interval job."""
        scheduler = IngestionScheduler()
        mock_job = Mock()

        scheduler.add_interval_job(mock_job, hours=1, minutes=0)

        assert len(scheduler.scheduler.get_jobs()) == 1

    def test_is_running(self):
        """Test is_running flag."""
        scheduler = IngestionScheduler()
        assert not scheduler.is_running()
