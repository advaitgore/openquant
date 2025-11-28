"""Redis-based caching for computed features."""

import json
from typing import Optional

import pandas as pd
import redis
from loguru import logger

from openquant.config.settings import settings


class FeatureCache:
    """Redis-based cache for feature computation results."""

    def __init__(self, redis_url: Optional[str] = None, ttl: Optional[int] = None) -> None:
        """Initialize the feature cache.

        Args:
            redis_url: Redis connection URL. Defaults to settings.REDIS_URL.
            ttl: Time-to-live in seconds. Defaults to settings.REDIS_TTL.
        """
        self.redis_url = redis_url or settings.REDIS_URL
        self.ttl = ttl or settings.REDIS_TTL
        self._client: Optional[redis.Redis] = None

    @property
    def client(self) -> redis.Redis:
        """Get or create Redis client.

        Returns:
            Redis client instance.
        """
        if self._client is None:
            try:
                self._client = redis.from_url(self.redis_url, decode_responses=True)
                # Test connection
                self._client.ping()
                logger.debug("Connected to Redis")
            except Exception as e:
                logger.error(f"Failed to connect to Redis: {e}")
                raise
        return self._client

    def _make_key(
        self,
        ticker: str,
        feature_names: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> str:
        """Generate cache key for feature request.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            Cache key string.
        """
        feature_str = "_".join(sorted(feature_names))
        date_str = f"{start_date or 'all'}_{end_date or 'all'}"
        return f"features:{ticker}:{feature_str}:{date_str}"

    def get(
        self,
        ticker: str,
        feature_names: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """Get cached features.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            Cached DataFrame or None if not found.
        """
        try:
            key = self._make_key(ticker, feature_names, start_date, end_date)
            cached_data = self.client.get(key)

            if cached_data is None:
                logger.debug(f"Cache miss for {key}")
                return None

            # Deserialize DataFrame
            data = json.loads(cached_data)
            df = pd.read_json(data["data"], orient="records")
            if "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"]).dt.date

            logger.debug(f"Cache hit for {key}")
            return df

        except Exception as e:
            logger.error(f"Error retrieving from cache: {e}")
            return None

    def set(
        self,
        ticker: str,
        feature_names: list[str],
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Cache computed features.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names.
            df: DataFrame with computed features.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            True if cached successfully, False otherwise.
        """
        try:
            key = self._make_key(ticker, feature_names, start_date, end_date)

            # Serialize DataFrame
            df_copy = df.copy()
            if "Date" in df_copy.columns:
                df_copy["Date"] = df_copy["Date"].astype(str)

            data = {
                "data": df_copy.to_json(orient="records", date_format="iso"),
                "ticker": ticker,
                "feature_names": feature_names,
            }

            self.client.setex(key, self.ttl, json.dumps(data))
            logger.debug(f"Cached features for {key}")
            return True

        except Exception as e:
            logger.error(f"Error caching features: {e}")
            return False

    def delete(
        self,
        ticker: str,
        feature_names: Optional[list[str]] = None,
    ) -> int:
        """Delete cached features for a ticker.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names (optional). If None, deletes all features for ticker.

        Returns:
            Number of keys deleted.
        """
        try:
            if feature_names:
                # Delete specific features
                pattern = f"features:{ticker}:*"
            else:
                # Delete all features for ticker
                pattern = f"features:{ticker}:*"

            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                deleted = self.client.delete(*keys)
                logger.info(f"Deleted {deleted} cache keys for {ticker}")
                return deleted
            return 0

        except Exception as e:
            logger.error(f"Error deleting cache: {e}")
            return 0

    def clear(self) -> bool:
        """Clear all cached features.

        Returns:
            True if successful, False otherwise.
        """
        try:
            pattern = "features:*"
            keys = list(self.client.scan_iter(match=pattern))
            if keys:
                self.client.delete(*keys)
                logger.info(f"Cleared {len(keys)} cache keys")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False

    def exists(
        self,
        ticker: str,
        feature_names: list[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> bool:
        """Check if cached features exist.

        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).

        Returns:
            True if cached, False otherwise.
        """
        try:
            key = self._make_key(ticker, feature_names, start_date, end_date)
            return self.client.exists(key) > 0

        except Exception as e:
            logger.error(f"Error checking cache: {e}")
            return False

