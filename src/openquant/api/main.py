"""FastAPI application main module."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from prometheus_fastapi_instrumentator import Instrumentator

from openquant.api.routers import data, features, health, models
from openquant.config.settings import settings
from openquant.utils.logging import setup_logging

# Custom Prometheus metrics
API_REQUEST_COUNT = Counter(
    "openquant_api_requests_total",
    "Total number of API requests",
    ["method", "endpoint", "status_code"]
)

API_REQUEST_DURATION = Histogram(
    "openquant_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
)

API_ACTIVE_REQUESTS = Gauge(
    "openquant_api_active_requests",
    "Number of active API requests"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    logger.info("Starting OpenQuant API")
    setup_logging()
    yield
    # Shutdown
    logger.info("Shutting down OpenQuant API")


# Create FastAPI app
app = FastAPI(
    title="OpenQuant API",
    description="Open-source MLOps platform for financial time-series",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(data.router)
app.include_router(features.router)
app.include_router(models.router)

# Instrument the FastAPI app with Prometheus
# This automatically exposes /metrics endpoint
Instrumentator().instrument(app).expose(app)

@app.get("/")
async def root() -> dict[str, str]:
    """Root endpoint.

    Returns:
        API information.
    """
    return {
        "name": "OpenQuant API",
        "version": "0.1.0",
        "description": "Open-source MLOps platform for financial time-series",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "openquant.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
    )

