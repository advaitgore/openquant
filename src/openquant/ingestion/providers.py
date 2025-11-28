"""Data provider implementations for fetching financial data."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

import yfinance as yf


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def fetch_ohlcv(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume.

        Raises:
            ValueError: If data cannot be fetched.
        """
        pass


class YFinanceProvider(DataProvider):
    """yfinance-based data provider."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def fetch_ohlcv(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """Fetch OHLCV data using yfinance.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format.
            end_date: End date in YYYY-MM-DD format.

        Returns:
            DataFrame with columns: Date, Open, High, Low, Close, Volume.

        Raises:
            ValueError: If data cannot be fetched.
        """
        try:
            logger.debug(f"Fetching data for {ticker} from {start_date} to {end_date}")
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data returned for {ticker}")

            # Reset index to make Date a column
            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.date

            # Select and rename columns
            df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]
            df.columns = ["Date", "Open", "High", "Low", "Close", "Volume"]

            # Add ticker column
            df["Ticker"] = ticker

            logger.info(f"Successfully fetched {len(df)} rows for {ticker}")
            return df

        except Exception as e:
            logger.error(f"Error fetching data for {ticker}: {e}")
            raise ValueError(f"Failed to fetch data for {ticker}: {e}") from e


def get_provider(provider_name: str = "yfinance") -> DataProvider:
    """Factory function to get a data provider.

    Args:
        provider_name: Name of the provider to use.

    Returns:
        DataProvider instance.

    Raises:
        ValueError: If provider name is not supported.
    """
    providers = {
        "yfinance": YFinanceProvider,
    }

    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")

    return providers[provider_name]()
