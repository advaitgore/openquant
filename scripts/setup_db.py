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
    
    # Drop existing table if it has wrong schema (Ticker as INT64 instead of VARCHAR)
    with storage.get_connection() as conn:
        try:
            # Check if table exists and has wrong schema
            result = conn.execute("PRAGMA table_info(ohlcv)").fetchall()
            if result:
                # Check if Ticker column exists and is not VARCHAR
                ticker_col = next((col for col in result if col[1] == "Ticker"), None)
                if ticker_col and "VARCHAR" not in str(ticker_col[2]).upper():
                    logger.warning("Dropping existing table with incorrect schema")
                    conn.execute("DROP TABLE IF EXISTS ohlcv")
                    conn.execute("DROP INDEX IF EXISTS idx_ohlcv_ticker_date")
        except Exception as e:
            logger.info(f"Table check: {e}")
    
    storage.initialize_schema()

    logger.info(f"Database initialized at {settings.DUCKDB_PATH}")


if __name__ == "__main__":
    main()
