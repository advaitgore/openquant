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

- Docker and Docker Compose (v2.0+)
- Python 3.11+ (for local development)
- 4GB+ RAM recommended

### Docker Installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd openquant

# Start all services
docker compose up -d

# Check service status
docker compose ps

# View logs
docker compose logs -f
```

**Services will be available at:**
- API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Dashboard: http://localhost:8501
- Redis: localhost:6379

### Initial Setup

```bash
# Initialize database schema
docker compose exec api python scripts/setup_db.py

# Run initial data ingestion
docker compose exec worker python scripts/run_ingestion.py
```

### Local Development Setup

```bash
# Install Python 3.11
# Using pyenv (recommended)
pyenv install 3.11
pyenv local 3.11

# Install uv (fast Python package manager)
pip install uv

# Install dependencies
uv pip install -e ".[dev]"

# Or using standard pip
pip install -e ".[dev]"

# Initialize database
python scripts/setup_db.py

# Run data ingestion
python scripts/run_ingestion.py

# Start API server
uvicorn openquant.api.main:app --reload

# Start dashboard (in another terminal)
streamlit run src/openquant/dashboard/app.py
```

## Usage

### Data Ingestion

```bash
# Run ingestion for default tickers
python scripts/run_ingestion.py

# Or via Docker
docker compose exec worker python scripts/run_ingestion.py
```

### Model Training

1. **Configure model** in `configs/model_config.yaml`
2. **Train model**:
   ```bash
   python scripts/train_model.py
   
   # Or via Docker
   docker compose exec api python scripts/train_model.py
   ```

### Using the API

```bash
# Health check
curl http://localhost:8000/health

# List tickers
curl http://localhost:8000/data/tickers

# Get OHLCV data
curl http://localhost:8000/data/AAPL?start_date=2024-01-01

# Compute features
curl -X POST http://localhost:8000/features/compute \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "feature_names": ["sma_20", "rsi_14"]}'

# Train model
curl -X POST http://localhost:8000/models/train \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "AAPL",
    "model_type": "xgboost",
    "feature_names": ["sma_20", "rsi_14", "macd"]
  }'
```

### Using the Dashboard

1. Open http://localhost:8501 in your browser
2. Navigate through pages:
   - **System Health**: Check service status
   - **Data Explorer**: View and visualize OHLCV data
   - **Feature Explorer**: Compute and visualize features
   - **Model Playground**: Train models and make predictions

## Project Structure

```
openquant/
├── src/openquant/          # Main source code
│   ├── api/               # FastAPI service
│   ├── config/            # Configuration
│   ├── dashboard/          # Streamlit dashboard
│   ├── features/          # Feature engineering
│   ├── ingestion/         # Data ingestion
│   ├── models/            # Model training/prediction
│   └── utils/             # Utilities
├── tests/                 # Test suite
├── configs/              # Configuration files
│   ├── data_sources.yaml
│   ├── features.yaml
│   └── model_config.yaml
├── scripts/              # Utility scripts
│   ├── setup_db.py
│   ├── run_ingestion.py
│   └── train_model.py
├── docker/                # Dockerfiles
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.worker
├── data/                  # Data storage (DuckDB)
├── models/                # Model artifacts
├── mlruns/                # MLflow runs
├── logs/                  # Application logs
├── docker-compose.yml     # Docker Compose configuration
└── pyproject.toml         # Python project configuration
```

## Configuration

### Environment Variables

Create a `.env` file (or set environment variables):

```bash
# Database
DUCKDB_PATH=data/openquant.duckdb

# Redis
REDIS_URL=redis://localhost:6379/0
REDIS_TTL=86400

# MLflow
MLFLOW_TRACKING_URI=file:./mlruns
MLFLOW_EXPERIMENT_NAME=openquant

# API
API_HOST=0.0.0.0
API_PORT=8000

# Logging
LOG_LEVEL=INFO
```

### Configuration Files

- `configs/data_sources.yaml`: Data source settings
- `configs/features.yaml`: Feature engineering configuration
- `configs/model_config.yaml`: Model training configuration

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/openquant --cov-report=html

# Run specific test file
pytest tests/test_ingestion.py
```

### Linting and Type Checking

```bash
# Run ruff linter
ruff check src/ tests/

# Run mypy type checker
mypy src/

# Auto-fix linting issues
ruff check --fix src/ tests/
```

### Code Quality

```bash
# Format code (if using black)
black src/ tests/

# Sort imports
ruff check --select I --fix src/ tests/
```

## Docker Commands

```bash
# Build images
docker compose build

# Start services
docker compose up -d

# Stop services
docker compose down

# View logs
docker compose logs -f api
docker compose logs -f dashboard
docker compose logs -f worker

# Execute commands in containers
docker compose exec api python scripts/setup_db.py
docker compose exec worker python scripts/run_ingestion.py

# Rebuild specific service
docker compose build api
docker compose up -d api
```

## API Documentation

Once the API is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Services won't start

```bash
# Check Docker Compose logs
docker compose logs

# Check if ports are in use
netstat -an | grep 8000
netstat -an | grep 8501
netstat -an | grep 6379
```

### Database issues

```bash
# Reinitialize database
docker compose exec api python scripts/setup_db.py

# Check database file
ls -lh data/openquant.duckdb
```

### Redis connection issues

```bash
# Test Redis connection
docker compose exec api python -c "import redis; r=redis.from_url('redis://redis:6379/0'); print(r.ping())"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`pytest && ruff check src/`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - Detailed architecture documentation
- API Documentation - Available at http://localhost:8000/docs when API is running

## License

MIT License - see [LICENSE](LICENSE) file for details

## Support

For issues, questions, or contributions, please open an issue on GitHub.

