"""Health check router."""

import mlflow
import redis
from fastapi import APIRouter, HTTPException

from openquant.api.schemas import HealthResponse
from openquant.config.settings import settings
from openquant.ingestion.storage import DuckDBStorage
from openquant.features.cache import FeatureCache
from loguru import logger

router = APIRouter(prefix="/health", tags=["health"])


def check_database() -> str:
    """Check database connection.

    Returns:
        Status string: "healthy" or "unhealthy"
    """
    try:
        storage = DuckDBStorage(settings.DUCKDB_PATH)
        with storage.get_connection() as conn:
            conn.execute("SELECT 1")
        return "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return "unhealthy"


def check_redis() -> str:
    """Check Redis connection.

    Returns:
        Status string: "healthy" or "unhealthy"
    """
    try:
        cache = FeatureCache()
        cache.client.ping()
        return "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        return "unhealthy"


def check_mlflow() -> str:
    """Check MLflow connection.

    Returns:
        Status string: "healthy" or "unhealthy"
    """
    try:
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
        return "healthy"
    except Exception as e:
        logger.warning(f"MLflow health check failed: {e}")
        return "unhealthy"


@router.get("", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint.

    Returns:
        Health status of all services.
    """
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        database=check_database(),
        redis=check_redis(),
        mlflow=check_mlflow(),
    )


@router.get("/ready")
async def readiness_check() -> dict[str, str]:
    """Readiness check endpoint.

    Returns:
        Ready status.
    """
    db_status = check_database()
    if db_status != "healthy":
        raise HTTPException(status_code=503, detail="Database not ready")

    return {"status": "ready"}


@router.get("/live")
async def liveness_check() -> dict[str, str]:
    """Liveness check endpoint.

    Returns:
        Live status.
    """
    return {"status": "alive"}

