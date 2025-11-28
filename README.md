# OpenQuant

Open-source MLOps platform for financial time-series data.

## Overview

OpenQuant is a local-first ML platform designed for building and deploying trading strategies on financial time-series data. It provides a complete pipeline from data ingestion to model training and prediction, all containerized and ready to run.

## Features

- **Data Ingestion**: Automated data collection from multiple sources (yfinance)
- **Feature Engineering**: Configurable feature pipeline with Redis caching
- **Model Training**: MLflow-integrated training with XGBoost, LightGBM, and scikit-learn
- **API Service**: FastAPI-based REST API for model inference and data access
- **Dashboard**: Streamlit dashboard for monitoring and exploration
- **Containerized**: Docker Compose setup for easy deployment

## Tech Stack

- **Language**: Python 3.11
- **Database**: DuckDB (historical data), Redis (caching)
- **API**: FastAPI with Pydantic v2
- **ML Tracking**: MLflow
- **Scheduling**: APScheduler
- **Frontend**: Streamlit
- **Testing**: pytest

## Quick Start

### Prerequisites

- Docker and Docker Compose
- Python 3.11+ (for local development)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd openquant

# Start services with Docker Compose
docker-compose up -d
```

### Development Setup

```bash
# Install dependencies (using uv or poetry)
uv pip install -e ".[dev]"

# Or with poetry
poetry install --with dev

# Run tests
pytest

# Run linting
ruff check src/
mypy src/
```

## Project Structure

```
openquant/
├── src/openquant/      # Main source code
├── tests/              # Test suite
├── configs/            # Configuration files
├── scripts/            # Utility scripts
├── data/               # Data storage
├── models/             # Model artifacts
└── docker/             # Dockerfiles
```

## Documentation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

## License

MIT

