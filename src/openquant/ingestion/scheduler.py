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
