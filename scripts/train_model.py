"""Script to train a model for a ticker."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import yaml
from openquant.config.settings import settings
from openquant.features.cache import FeatureCache
from openquant.features.engine import FeatureEngine
from openquant.ingestion.storage import DuckDBStorage
from openquant.models.trainer import ModelTrainer
from openquant.utils.logging import setup_logging
from loguru import logger


def main() -> None:
    """Train a model based on configuration."""
    setup_logging()
    logger.info("Starting model training")

    # Load model configuration
    with open(settings.MODEL_CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)

    # Initialize components
    storage = DuckDBStorage(settings.DUCKDB_PATH)
    feature_engine = FeatureEngine(storage=storage)
    feature_cache = FeatureCache() if config.get("use_cache", True) else None
    trainer = ModelTrainer(
        storage=storage,
        feature_engine=feature_engine,
        feature_cache=feature_cache,
    )

    # Get training parameters
    ticker = config.get("ticker", "SPY")
    model_type = config.get("model_type", "xgboost")
    feature_names = config.get("feature_names", [])
    model_params = config.get("model_params", {})
    target = config.get("target", "returns")
    lookback_days = config.get("lookback_days", 1)
    test_size = config.get("test_size", 0.2)
    start_date = config.get("start_date")
    end_date = config.get("end_date")

    logger.info(f"Training {model_type} model for {ticker}")

    # Train model
    try:
        results = trainer.train(
            ticker=ticker,
            model_type=model_type,
            feature_names=feature_names,
            model_params=model_params,
            target=target,
            lookback_days=lookback_days,
            test_size=test_size,
            start_date=start_date,
            end_date=end_date,
        )

        logger.info(f"Training completed successfully")
        logger.info(f"Run ID: {results['run_id']}")
        logger.info(f"Test R2: {results['test_metrics']['test_r2']:.4f}")
        logger.info(f"Test RMSE: {results['test_metrics']['test_rmse']:.4f}")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()

