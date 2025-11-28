"""Data router for OHLCV data endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException

from openquant.api.schemas import DataResponse, ErrorResponse, OHLCVData, TickerListResponse
from openquant.config.settings import settings
from openquant.ingestion.storage import DuckDBStorage
from loguru import logger

router = APIRouter(prefix="/data", tags=["data"])

# Initialize storage (could be dependency injection in production)
_storage = DuckDBStorage(settings.DUCKDB_PATH)


@router.get("/tickers", response_model=TickerListResponse)
async def list_tickers() -> TickerListResponse:
    """List all available tickers.

    Returns:
        List of ticker symbols.
    """
    try:
        tickers = _storage.get_tickers()
        return TickerListResponse(tickers=tickers, count=len(tickers))
    except Exception as e:
        logger.error(f"Error listing tickers: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing tickers: {e}")


@router.get("/{ticker}", response_model=DataResponse)
async def get_ohlcv(
    ticker: str,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> DataResponse:
    """Get OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol.
        start_date: Start date in YYYY-MM-DD format (optional).
        end_date: End date in YYYY-MM-DD format (optional).

    Returns:
        OHLCV data for the ticker.

    Raises:
        HTTPException: If data cannot be retrieved.
    """
    try:
        df = _storage.get_ohlcv(ticker, start_date, end_date)

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {ticker}"
            )

        # Convert to response format
        data_points = [
            OHLCVData(
                date=row["Date"],
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for _, row in df.iterrows()
        ]

        return DataResponse(ticker=ticker, data=data_points, count=len(data_points))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving data for {ticker}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving data: {e}"
        )


@router.get("/{ticker}/latest", response_model=DataResponse)
async def get_latest_ohlcv(
    ticker: str,
    n_days: int = 1,
) -> DataResponse:
    """Get latest N days of OHLCV data for a ticker.

    Args:
        ticker: Stock ticker symbol.
        n_days: Number of days to retrieve. Defaults to 1.

    Returns:
        Latest OHLCV data for the ticker.

    Raises:
        HTTPException: If data cannot be retrieved.
    """
    try:
        # Get latest date
        latest_date = _storage.get_latest_date(ticker)
        if not latest_date:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {ticker}"
            )

        # Get data for last N days
        from openquant.utils.time import get_lookback_date

        start_date = get_lookback_date(days=n_days * 2)  # Get extra for safety
        df = _storage.get_ohlcv(ticker, start_date, latest_date)

        if df.empty:
            raise HTTPException(
                status_code=404, detail=f"No data found for ticker {ticker}"
            )

        # Take last N days
        df = df.tail(n_days)

        # Convert to response format
        data_points = [
            OHLCVData(
                date=row["Date"],
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=int(row["Volume"]),
            )
            for _, row in df.iterrows()
        ]

        return DataResponse(ticker=ticker, data=data_points, count=len(data_points))

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving latest data for {ticker}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error retrieving data: {e}"
        )

