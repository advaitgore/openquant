"""DuckDB storage operations for OHLCV data."""

import contextlib
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from loguru import logger

from openquant.config.settings import settings


class DuckDBStorage:
    """DuckDB storage manager for OHLCV data."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        """Initialize DuckDB storage.

        Args:
            db_path: Path to DuckDB database file. Defaults to settings.DUCKDB_PATH.
        """
        self.db_path = db_path or settings.DUCKDB_PATH
        self._ensure_db_directory()

    def _ensure_db_directory(self) -> None:
        """Ensure the database directory exists."""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def get_connection(self):
        """Get a DuckDB connection context manager.

        Yields:
            DuckDB connection object.
        """
        conn = None
        try:
            conn = duckdb.connect(self.db_path)
            yield conn
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def initialize_schema(self) -> None:
        """Initialize the database schema."""
        with self.get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    Date DATE NOT NULL,
                    Ticker VARCHAR NOT NULL,
                    Open DOUBLE NOT NULL,
                    High DOUBLE NOT NULL,
                    Low DOUBLE NOT NULL,
                    Close DOUBLE NOT NULL,
                    Volume BIGINT NOT NULL,
                    PRIMARY KEY (Date, Ticker)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_ohlcv_ticker_date 
                ON ohlcv (Ticker, Date DESC)
                """
            )
            logger.info("Database schema initialized")

    def upsert_ohlcv(self, df: pd.DataFrame) -> int:
        """Upsert OHLCV data into the database.

        Args:
            df: DataFrame with columns: Date, Ticker, Open, High, Low, Close, Volume.

        Returns:
            Number of rows inserted/updated.
        """
        if df.empty:
            logger.warning("Attempted to upsert empty DataFrame")
            return 0

        required_columns = ["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        with self.get_connection() as conn:
            # Register DataFrame as temporary table for upsert
            conn.execute("BEGIN TRANSACTION")
            try:
                # Ensure Ticker column is string type and reorder columns to match table schema
                # Table order: Date, Ticker, Open, High, Low, Close, Volume
                df_ordered = df[["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"]].copy()
                df_ordered["Ticker"] = df_ordered["Ticker"].astype(str)
                
                # Register DataFrame as temporary table
                conn.register("temp_df", df_ordered)
                
                # Delete existing rows for matching tickers and dates
                tickers = df["Ticker"].unique().tolist()
                for ticker in tickers:
                    conn.execute(
                        """
                        DELETE FROM ohlcv 
                        WHERE Ticker = ? 
                        AND Date IN (SELECT Date FROM temp_df WHERE Ticker = ?)
                        """,
                        [ticker, ticker],
                    )
                
                # Insert new data - ensure column order matches table schema
                # Table order: Date, Ticker, Open, High, Low, Close, Volume
                conn.execute("""
                    INSERT INTO ohlcv (Date, Ticker, Open, High, Low, Close, Volume)
                    SELECT Date, Ticker, Open, High, Low, Close, Volume FROM temp_df
                """)
                conn.execute("COMMIT")
                rows_affected = len(df)
                logger.info(f"Upserted {rows_affected} rows for ticker(s): {tickers}")
                return rows_affected
            except Exception as e:
                conn.execute("ROLLBACK")
                logger.error(f"Error upserting data: {e}")
                raise

    def get_ohlcv(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """Get OHLCV data from the database.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            DataFrame with OHLCV data.
        """
        with self.get_connection() as conn:
            query = "SELECT * FROM ohlcv WHERE Ticker = ?"
            params = [ticker]

            if start_date:
                query += " AND Date >= ?"
                params.append(start_date)

            if end_date:
                query += " AND Date <= ?"
                params.append(end_date)

            query += " ORDER BY Date ASC"

            df = conn.execute(query, params).df()
            logger.debug(f"Retrieved {len(df)} rows for {ticker}")
            return df

    def get_latest_date(self, ticker: str) -> Optional[str]:
        """Get the latest date for a ticker in the database.

        Args:
            ticker: Stock ticker symbol.

        Returns:
            Latest date as YYYY-MM-DD string, or None if no data exists.
        """
        with self.get_connection() as conn:
            result = conn.execute(
                "SELECT MAX(Date) as latest_date FROM ohlcv WHERE Ticker = ?",
                [ticker],
            ).fetchone()

            if result and result[0]:
                return result[0].strftime("%Y-%m-%d")
            return None

    def get_tickers(self) -> list[str]:
        """Get list of all tickers in the database.

        Returns:
            List of ticker symbols.
        """
        with self.get_connection() as conn:
            result = conn.execute("SELECT DISTINCT Ticker FROM ohlcv").fetchall()
            return [row[0] for row in result]
