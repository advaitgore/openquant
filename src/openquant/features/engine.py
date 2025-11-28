"""Feature engineering engine for computing features from OHLCV data."""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
from loguru import logger

from openquant.features.registry import FeatureRegistry, get_registry
from openquant.ingestion.storage import DuckDBStorage


class FeatureEngine:
    """Engine for computing features from OHLCV data."""

    def __init__(
        self,
        registry: Optional[FeatureRegistry] = None,
        storage: Optional[DuckDBStorage] = None,
    ) -> None:
        """Initialize the feature engine.

        Args:
            registry: Feature registry instance. Defaults to global registry.
            storage: DuckDB storage instance. Defaults to None.
        """
        self.registry = registry or get_registry()
        self.storage = storage
        self._register_default_features()

    def _register_default_features(self) -> None:
        """Register default feature functions."""
        # Simple moving averages
        self.registry.register("sma_5", lambda df: ta.sma(df["Close"], length=5))
        self.registry.register("sma_10", lambda df: ta.sma(df["Close"], length=10))
        self.registry.register("sma_20", lambda df: ta.sma(df["Close"], length=20))
        self.registry.register("sma_50", lambda df: ta.sma(df["Close"], length=50))

        # Exponential moving averages
        self.registry.register("ema_12", lambda df: ta.ema(df["Close"], length=12))
        self.registry.register("ema_26", lambda df: ta.ema(df["Close"], length=26))

        # RSI
        self.registry.register("rsi_14", lambda df: ta.rsi(df["Close"], length=14))

        # MACD
        def macd_line(df: pd.DataFrame) -> pd.Series:
            macd = ta.macd(df["Close"])
            return macd[f"MACD_{12}_{26}_9"] if isinstance(macd, pd.DataFrame) else pd.Series()

        def macd_signal(df: pd.DataFrame) -> pd.Series:
            macd = ta.macd(df["Close"])
            return macd[f"MACDs_{12}_{26}_9"] if isinstance(macd, pd.DataFrame) else pd.Series()

        def macd_histogram(df: pd.DataFrame) -> pd.Series:
            macd = ta.macd(df["Close"])
            return macd[f"MACDh_{12}_{26}_9"] if isinstance(macd, pd.DataFrame) else pd.Series()

        self.registry.register("macd", macd_line)
        self.registry.register("macd_signal", macd_signal)
        self.registry.register("macd_histogram", macd_histogram)

        # Bollinger Bands
        def bb_upper(df: pd.DataFrame) -> pd.Series:
            bb = ta.bbands(df["Close"], length=20)
            return bb["BBU_20_2.0"] if isinstance(bb, pd.DataFrame) else pd.Series()

        def bb_middle(df: pd.DataFrame) -> pd.Series:
            bb = ta.bbands(df["Close"], length=20)
            return bb["BBM_20_2.0"] if isinstance(bb, pd.DataFrame) else pd.Series()

        def bb_lower(df: pd.DataFrame) -> pd.Series:
            bb = ta.bbands(df["Close"], length=20)
            return bb["BBL_20_2.0"] if isinstance(bb, pd.DataFrame) else pd.Series()

        self.registry.register("bb_upper", bb_upper)
        self.registry.register("bb_middle", bb_middle)
        self.registry.register("bb_lower", bb_lower)

        # Volume features
        self.registry.register("volume_sma_20", lambda df: ta.sma(df["Volume"], length=20))
        self.registry.register("volume_ratio", lambda df: df["Volume"] / ta.sma(df["Volume"], length=20))

        # Price features
        self.registry.register("returns", lambda df: df["Close"].pct_change())
        self.registry.register("log_returns", lambda df: np.log(df["Close"] / df["Close"].shift(1)))
        self.registry.register("high_low_ratio", lambda df: df["High"] / df["Low"])
        self.registry.register("close_open_ratio", lambda df: df["Close"] / df["Open"])

        # Volatility
        self.registry.register("volatility_20", lambda df: df["Close"].rolling(window=20).std())
        self.registry.register("atr_14", lambda df: ta.atr(df["High"], df["Low"], df["Close"], length=14))

        logger.info(f"Registered {len(self.registry.list_features())} default features")

    def compute_features(
        self,
        df: pd.DataFrame,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute features from OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume).
            feature_names: List of feature names to compute. If None, computes all registered features.

        Returns:
            DataFrame with computed features.

        Raises:
            ValueError: If required OHLCV columns are missing or feature is not found.
        """
        required_columns = ["Open", "High", "Low", "Close", "Volume"]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if df.empty:
            logger.warning("Computing features for empty DataFrame")
            return pd.DataFrame()

        # Sort by date to ensure proper time-series order
        if "Date" in df.columns:
            df = df.sort_values("Date").reset_index(drop=True)

        # Select features to compute
        if feature_names is None:
            feature_names = self.registry.list_features()

        features_df = pd.DataFrame(index=df.index)
        if "Date" in df.columns:
            features_df["Date"] = df["Date"]
        if "Ticker" in df.columns:
            features_df["Ticker"] = df["Ticker"]

        # Compute each feature
        for feature_name in feature_names:
            try:
                feature_func = self.registry.get(feature_name)
                if feature_func is None:
                    logger.warning(f"Feature {feature_name} not found, skipping")
                    continue

                feature_values = feature_func(df)
                if isinstance(feature_values, pd.Series):
                    features_df[feature_name] = feature_values
                elif isinstance(feature_values, pd.DataFrame):
                    # If function returns DataFrame, merge all columns
                    for col in feature_values.columns:
                        features_df[f"{feature_name}_{col}"] = feature_values[col]
                else:
                    logger.warning(f"Feature {feature_name} returned unexpected type: {type(feature_values)}")

            except Exception as e:
                logger.error(f"Error computing feature {feature_name}: {e}")
                # Continue with other features
                continue

        logger.debug(f"Computed {len(feature_names)} features for {len(df)} rows")
        return features_df

    def compute_features_for_ticker(
        self,
        ticker: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compute features for a ticker from stored OHLCV data.

        Args:
            ticker: Stock ticker symbol.
            start_date: Start date in YYYY-MM-DD format (optional).
            end_date: End date in YYYY-MM-DD format (optional).
            feature_names: List of feature names to compute (optional).

        Returns:
            DataFrame with computed features.

        Raises:
            ValueError: If storage is not initialized.
        """
        if self.storage is None:
            raise ValueError("Storage not initialized")

        # Get OHLCV data
        ohlcv_df = self.storage.get_ohlcv(ticker, start_date, end_date)
        if ohlcv_df.empty:
            logger.warning(f"No OHLCV data found for {ticker}")
            return pd.DataFrame()

        # Compute features
        return self.compute_features(ohlcv_df, feature_names)

