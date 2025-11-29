"""Scheduler for automated data ingestion."""

from datetime import datetime, time
from typing import Callable, Optional

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from loguru import logger

from openquant.config.settings import settings
from openquant.utils.time import get_market_close_time


class IngestionScheduler:
    """Scheduler for automated data ingestion tasks."""

    def __init__(self) -> None:
        """Initialize the ingestion scheduler."""
        self.scheduler = BlockingScheduler()
        self._is_running = False

    def add_daily_job(
        self,
        job_func: Callable,
        hour: int = 16,
        minute: int = 30,
        timezone: Optional[str] = None,
    ) -> None:
        """Add a daily job to the scheduler.

        Args:
            job_func: Function to execute.
            hour: Hour of day (0-23). Defaults to 16 (4 PM).
            minute: Minute of hour (0-59). Defaults to 30.
            timezone: Timezone string (e.g., 'America/New_York'). Defaults to None.
        """
        trigger = CronTrigger(hour=hour, minute=minute, timezone=timezone)
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id="daily_ingestion",
            name="Daily Data Ingestion",
            replace_existing=True,
        )
        logger.info(f"Scheduled daily job at {hour:02d}:{minute:02d}")

    def add_interval_job(
        self,
        job_func: Callable,
        hours: int = 1,
        minutes: int = 0,
    ) -> None:
        """Add an interval-based job to the scheduler.

        Args:
            job_func: Function to execute.
            hours: Number of hours between runs.
            minutes: Number of minutes between runs.
        """
        from apscheduler.triggers.interval import IntervalTrigger

        trigger = IntervalTrigger(hours=hours, minutes=minutes)
        self.scheduler.add_job(
            job_func,
            trigger=trigger,
            id="interval_ingestion",
            name="Interval Data Ingestion",
            replace_existing=True,
        )
        logger.info(f"Scheduled interval job every {hours}h {minutes}m")

    def start(self) -> None:
        """Start the scheduler."""
        if self._is_running:
            logger.warning("Scheduler is already running")
            return

        logger.info("Starting ingestion scheduler")
        self._is_running = True
        try:
            self.scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
            self.stop()
        except Exception as e:
            logger.error(f"Scheduler error: {e}")
            self.stop()
            raise

    def stop(self) -> None:
        """Stop the scheduler."""
        if not self._is_running:
            return

        logger.info("Stopping ingestion scheduler")
        self.scheduler.shutdown()
        self._is_running = False

    def is_running(self) -> bool:
        """Check if scheduler is running.

        Returns:
            True if scheduler is running, False otherwise.
        """
        return self._is_running


def run_ingestion_job() -> None:
    """Run data ingestion for configured tickers."""
    from openquant.ingestion.providers import get_provider
    from openquant.ingestion.storage import DuckDBStorage
    from openquant.utils.logging import setup_logging
    from openquant.utils.time import get_lookback_date, today

    setup_logging()
    logger.info("Running scheduled data ingestion")

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
            logger.info(f"Starting ingestion for {ticker}")

            # Check latest date in database
            latest_date = storage.get_latest_date(ticker)
            if latest_date and latest_date >= start_date:
                fetch_start = latest_date
                logger.info(f"Found existing data up to {latest_date}, fetching from {fetch_start}")
            else:
                fetch_start = start_date

            # Fetch data
            provider = get_provider(provider_name)
            df = provider.fetch_ohlcv(ticker, fetch_start, end_date)

            if df.empty:
                logger.warning(f"No new data for {ticker}")
                continue

            # Store data
            storage.upsert_ohlcv(df)
            logger.info(f"Successfully ingested {len(df)} rows for {ticker}")

        except Exception as e:
            logger.error(f"Error ingesting {ticker}: {e}")
            continue

    logger.info("Scheduled data ingestion completed")


def main() -> None:
    """Main entry point for the scheduler."""
    from openquant.utils.logging import setup_logging

    setup_logging()
    logger.info("Initializing ingestion scheduler")

    # Create scheduler
    scheduler = IngestionScheduler()

    # Schedule daily ingestion at market close (4:30 PM ET)
    scheduler.add_daily_job(
        run_ingestion_job,
        hour=16,
        minute=30,
        timezone="America/New_York",
    )

    # Start the scheduler
    scheduler.start()


if __name__ == "__main__":
    main()