"""Time and date utility functions."""

from datetime import datetime, timedelta


def get_market_close_time() -> str:
    """Return US market close time (4:00 PM ET).

    Returns:
        Market close time as string in HH:MM format.
    """
    return "16:00"


def get_lookback_date(days: int = 365) -> str:
    """Get date N days ago in YYYY-MM-DD format.

    Args:
        days: Number of days to look back.

    Returns:
        Date string in YYYY-MM-DD format.
    """
    date = datetime.now() - timedelta(days=days)
    return date.strftime("%Y-%m-%d")


def today() -> str:
    """Get today's date in YYYY-MM-DD format.

    Returns:
        Date string in YYYY-MM-DD format.
    """
    return datetime.now().strftime("%Y-%m-%d")


def parse_date(date_str: str) -> datetime:
    """Parse date string to datetime object.

    Args:
        date_str: Date string in YYYY-MM-DD format.

    Returns:
        Datetime object.

    Raises:
        ValueError: If date string format is invalid.
    """
    return datetime.strptime(date_str, "%Y-%m-%d")
