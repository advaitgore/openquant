"""Application settings and configuration."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    DUCKDB_PATH: str = "data/openquant.duckdb"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_TTL: int = 86400  # 24 hours

    # MLflow
    MLFLOW_TRACKING_URI: str = "file:./mlruns"
    MLFLOW_EXPERIMENT_NAME: str = "openquant"

    # Data Provider
    DATA_PROVIDER: str = "yfinance"
    DEFAULT_TICKERS: list[str] = [
        "SPY",
        "AAPL",
        "MSFT",
        "GOOGL",
        "AMZN",
        "TSLA",
        "NVDA",
        "META",
    ]

    # Features
    FEATURE_CONFIG_PATH: str = "configs/features.yaml"

    # Models
    MODEL_CONFIG_PATH: str = "configs/model_config.yaml"
    MODEL_REGISTRY_PATH: str = "models/registry.json"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Logging
    LOG_LEVEL: str = "INFO"


settings = Settings()
