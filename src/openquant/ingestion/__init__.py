"""Data ingestion module for OpenQuant."""

from openquant.ingestion.providers import DataProvider, YFinanceProvider, get_provider
from openquant.ingestion.scheduler import IngestionScheduler
from openquant.ingestion.storage import DuckDBStorage

__all__ = [
    "DataProvider",
    "YFinanceProvider",
    "get_provider",
    "IngestionScheduler",
    "DuckDBStorage",
]
