"""Model registry for managing ML model types."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from loguru import logger
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


class BaseModel(ABC):
    """Abstract base class for ML models."""

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the model.

        Args:
            X: Feature matrix.
            y: Target vector.
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Feature matrix.

        Returns:
            Predicted values.
        """
        pass

    @abstractmethod
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters.

        Returns:
            Dictionary of model parameters.
        """
        pass


class XGBoostModel(BaseModel):
    """XGBoost regression model."""

    def __init__(self, **kwargs) -> None:
        """Initialize XGBoost model.

        Args:
            **kwargs: XGBoost parameters.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.model = xgb.XGBRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the XGBoost model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with XGBoost."""
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Get XGBoost parameters."""
        return self.model.get_params()


class LightGBMModel(BaseModel):
    """LightGBM regression model."""

    def __init__(self, **kwargs) -> None:
        """Initialize LightGBM model.

        Args:
            **kwargs: LightGBM parameters.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.model = lgb.LGBMRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the LightGBM model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM."""
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Get LightGBM parameters."""
        return self.model.get_params()


class RandomForestModel(BaseModel):
    """Random Forest regression model."""

    def __init__(self, **kwargs) -> None:
        """Initialize Random Forest model.

        Args:
            **kwargs: Random Forest parameters.
        """
        default_params = {
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
        }
        default_params.update(kwargs)
        self.model = RandomForestRegressor(**default_params)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Random Forest model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Random Forest."""
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Get Random Forest parameters."""
        return self.model.get_params()


class LinearRegressionModel(BaseModel):
    """Linear Regression model."""

    def __init__(self, **kwargs) -> None:
        """Initialize Linear Regression model.

        Args:
            **kwargs: Linear Regression parameters.
        """
        self.model = LinearRegression(**kwargs)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the Linear Regression model."""
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Linear Regression."""
        return self.model.predict(X)

    def get_params(self) -> Dict[str, Any]:
        """Get Linear Regression parameters."""
        return self.model.get_params()


class ModelRegistry:
    """Registry for ML model types."""

    def __init__(self) -> None:
        """Initialize the model registry."""
        self._models: Dict[str, type[BaseModel]] = {
            "xgboost": XGBoostModel,
            "lightgbm": LightGBMModel,
            "random_forest": RandomForestModel,
            "linear_regression": LinearRegressionModel,
        }

    def register(self, name: str, model_class: type[BaseModel]) -> None:
        """Register a model type.

        Args:
            name: Model type name.
            model_class: Model class that inherits from BaseModel.
        """
        if name in self._models:
            logger.warning(f"Model {name} already registered, overwriting")
        self._models[name] = model_class
        logger.debug(f"Registered model type: {name}")

    def create(self, model_type: str, **kwargs) -> BaseModel:
        """Create a model instance.

        Args:
            model_type: Type of model to create.
            **kwargs: Model parameters.

        Returns:
            Model instance.

        Raises:
            ValueError: If model type is not registered.
        """
        if model_type not in self._models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Filter out invalid parameters for specific model types
        filtered_kwargs = kwargs.copy()
        
        if model_type == "random_forest":
            # RandomForest doesn't support learning_rate
            filtered_kwargs.pop("learning_rate", None)
        elif model_type == "linear_regression":
            # LinearRegression doesn't support tree-based parameters
            filtered_kwargs.pop("learning_rate", None)
            filtered_kwargs.pop("max_depth", None)
            filtered_kwargs.pop("n_estimators", None)
        
        return self._models[model_type](**filtered_kwargs)

    def list_models(self) -> list[str]:
        """List all registered model types.

        Returns:
            List of model type names.
        """
        return list(self._models.keys())


# Global model registry instance
_registry = ModelRegistry()


def get_registry() -> ModelRegistry:
    """Get the global model registry instance.

    Returns:
        ModelRegistry instance.
    """
    return _registry

