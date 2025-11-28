"""Script to run data ingestion for configured tickers."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.config.settings import settings
from openquant.ingestion.providers import get_provider
from openquant.ingestion.storage import DuckDBStorage
from openquant.utils.logging import setup_logging
from openquant.utils.time import get_lookback_date, today
from loguru import logger


def ingest_ticker(
    ticker: str,
    provider_name: str,
    storage: DuckDBStorage,
    start_date: str,
    end_date: str,
) -> None:
    """Ingest data for a single ticker.

    Args:
        ticker: Stock ticker symbol.
        provider_name: Name of the data provider.
        storage: DuckDB storage instance.
        start_date: Start date in YYYY-MM-DD format.
        end_date: End date in YYYY-MM-DD format.
    """
    try:
        logger.info(f"Starting ingestion for {ticker}")

        # Check latest date in database
        latest_date = storage.get_latest_date(ticker)
        if latest_date and latest_date >= start_date:
            # Only fetch data after the latest date
            fetch_start = latest_date
            logger.info(f"Found existing data up to {latest_date}, fetching from {fetch_start}")
        else:
            fetch_start = start_date

        # Fetch data
        provider = get_provider(provider_name)
        df = provider.fetch_ohlcv(ticker, fetch_start, end_date)

        if df.empty:
            logger.warning(f"No new data for {ticker}")
            return

        # Store data
        storage.upsert_ohlcv(df)
        logger.info(f"Successfully ingested {len(df)} rows for {ticker}")

    except Exception as e:
        logger.error(f"Error ingesting {ticker}: {e}")
        raise


def main() -> None:
    """Run data ingestion for all configured tickers."""
    setup_logging()
    logger.info("Starting data ingestion")

    # Initialize storage
    storage = DuckDBStorage(settings.DUCKDB_PATH)
    storage.initialize_schema()

    # Get date range
    end_date = today()
    start_date = get_lookback_date(days=365)

    logger.info(f"Ingesting data from {start_date} to {end_date}")

    # Ingest each ticker
    tickers = settings.DEFAULT_TICKERS
    provider_name = settings.DATA_PROVIDER

    for ticker in tickers:
        try:
            ingest_ticker(ticker, provider_name, storage, start_date, end_date)
        except Exception as e:
            logger.error(f"Failed to ingest {ticker}: {e}")
            continue

    logger.info("Data ingestion completed")


if __name__ == "__main__":
    main()
