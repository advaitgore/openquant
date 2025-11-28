"""FastAPI application main module."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from openquant.api.routers import data, features, health, models
from openquant.config.settings import settings
from openquant.utils.logging import setup_logging


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

