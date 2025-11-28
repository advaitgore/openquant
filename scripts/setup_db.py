"""Script to initialize the DuckDB database schema."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from openquant.config.settings import settings
from openquant.ingestion.storage import DuckDBStorage
from openquant.utils.logging import setup_logging
from loguru import logger


def main() -> None:
    """Initialize the database schema."""
    setup_logging()
    logger.info("Initializing database schema")

    storage = DuckDBStorage(settings.DUCKDB_PATH)
    storage.initialize_schema()

    logger.info(f"Database initialized at {settings.DUCKDB_PATH}")


if __name__ == "__main__":
    main()
