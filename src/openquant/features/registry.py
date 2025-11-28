"""Feature registry for managing feature computation functions."""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional

import pandas as pd
from loguru import logger


class FeatureFunction(ABC):
    """Abstract base class for feature functions."""

    @abstractmethod
    def compute(self, df: pd.DataFrame) -> pd.Series:
        """Compute feature values from OHLCV DataFrame.

        Args:
            df: DataFrame with OHLCV data (Date, Open, High, Low, Close, Volume).

        Returns:
            Series with computed feature values.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the feature name."""
        pass


class FeatureRegistry:
    """Registry for feature computation functions."""

    def __init__(self) -> None:
        """Initialize the feature registry."""
        self._features: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {}
        self._feature_configs: Dict[str, Dict[str, Any]] = {}

    def register(
        self,
        name: str,
        func: Callable[[pd.DataFrame], pd.Series],
        config: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Register a feature computation function.

        Args:
            name: Feature name.
            func: Function that takes a DataFrame and returns a Series.
            config: Optional configuration dictionary for the feature.
        """
        if name in self._features:
            logger.warning(f"Feature {name} already registered, overwriting")
        self._features[name] = func
        if config:
            self._feature_configs[name] = config
        logger.debug(f"Registered feature: {name}")

    def get(self, name: str) -> Optional[Callable[[pd.DataFrame], pd.Series]]:
        """Get a feature function by name.

        Args:
            name: Feature name.

        Returns:
            Feature function or None if not found.
        """
        return self._features.get(name)

    def list_features(self) -> list[str]:
        """List all registered feature names.

        Returns:
            List of feature names.
        """
        return list(self._features.keys())

    def get_config(self, name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a feature.

        Args:
            name: Feature name.

        Returns:
            Feature configuration dictionary or None.
        """
        return self._feature_configs.get(name)

    def has_feature(self, name: str) -> bool:
        """Check if a feature is registered.

        Args:
            name: Feature name.

        Returns:
            True if feature is registered, False otherwise.
        """
        return name in self._features


# Global feature registry instance
_registry = FeatureRegistry()


def get_registry() -> FeatureRegistry:
    """Get the global feature registry instance.

    Returns:
        FeatureRegistry instance.
    """
    return _registry


def register_feature(
    name: str,
    func: Callable[[pd.DataFrame], pd.Series],
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Register a feature in the global registry.

    Args:
        name: Feature name.
        func: Function that takes a DataFrame and returns a Series.
        config: Optional configuration dictionary.
    """
    _registry.register(name, func, config)

