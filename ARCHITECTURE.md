# OpenQuant Architecture

## Overview

OpenQuant is a local-first MLOps platform designed for financial time-series data. It provides a complete pipeline from data ingestion to model training and prediction, all containerized and ready to run.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     OpenQuant Platform                        │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   Streamlit  │  │   FastAPI    │  │   Worker     │       │
│  │  Dashboard   │  │     API      │  │   Service    │       │
│  │   (8501)     │  │    (8000)    │  │              │       │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘       │
│         │                 │                  │                │
│         └─────────────────┼──────────────────┘                │
│                           │                                   │
│         ┌─────────────────┼──────────────────┐                │
│         │                 │                  │                │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌──────▼──────┐          │
│  │   DuckDB    │  │    Redis    │  │   MLflow    │          │
│  │  (Storage)  │  │   (Cache)   │  │ (Tracking)  │          │
│  └─────────────┘  └─────────────┘  └─────────────┘          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Data Ingestion Layer

**Location**: `src/openquant/ingestion/`

**Components**:
- **Providers** (`providers.py`): Data provider implementations (yfinance)
- **Storage** (`storage.py`): DuckDB storage manager for OHLCV data
- **Scheduler** (`scheduler.py`): APScheduler for automated data ingestion

**Features**:
- Incremental data updates (only fetches new data)
- Retry logic with tenacity
- Transaction support for data integrity
- Schema initialization and management

**Data Flow**:
```
External API (yfinance) → Provider → Storage (DuckDB) → Feature Engine
```

### 2. Feature Engineering

**Location**: `src/openquant/features/`

**Components**:
- **Registry** (`registry.py`): Feature function registry
- **Engine** (`engine.py`): Feature computation engine
- **Cache** (`cache.py`): Redis-based feature caching

**Features**:
- 20+ default technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, etc.)
- Extensible registry for custom features
- Redis caching with configurable TTL
- Integration with pandas-ta library

**Supported Features**:
- Technical indicators: SMA, EMA, RSI, MACD, Bollinger Bands, ATR
- Volume features: Volume SMA, Volume ratio
- Price features: Returns, Log returns, High/Low ratio
- Volatility: Rolling volatility, ATR

### 3. Model Training & Prediction

**Location**: `src/openquant/models/`

**Components**:
- **Registry** (`registry.py`): Model type registry (XGBoost, LightGBM, Random Forest, Linear Regression)
- **Trainer** (`trainer.py`): Model training with MLflow integration
- **Predictor** (`predictor.py`): Model prediction and inference

**Features**:
- MLflow integration for experiment tracking
- Multiple model types support
- Comprehensive metrics (MAE, MSE, RMSE, R²)
- Model versioning and artifact storage

**Training Pipeline**:
```
OHLCV Data → Feature Engineering → Data Preparation → Model Training → MLflow Tracking
```

### 4. API Service

**Location**: `src/openquant/api/`

**Components**:
- **Main** (`main.py`): FastAPI application
- **Routers**: Health, Data, Features, Models
- **Schemas** (`schemas.py`): Pydantic v2 request/response models

**Endpoints**:
- `GET /health` - Health check
- `GET /data/tickers` - List available tickers
- `GET /data/{ticker}` - Get OHLCV data
- `POST /features/compute` - Compute features
- `POST /models/train` - Train a model
- `POST /models/predict` - Make predictions

**Features**:
- RESTful API design
- Automatic API documentation (Swagger/OpenAPI)
- CORS support
- Error handling with HTTP exceptions

### 5. Dashboard

**Location**: `src/openquant/dashboard/`

**Components**:
- **App** (`app.py`): Multi-page Streamlit application

**Pages**:
1. **System Health**: Service status, system information
2. **Data Explorer**: OHLCV data visualization with charts
3. **Feature Explorer**: Feature computation and visualization
4. **Model Playground**: Model training, predictions, and comparison

**Features**:
- Interactive visualizations with Plotly
- Real-time data loading
- Model training interface
- Prediction visualization

### 6. Configuration

**Location**: `configs/`

**Files**:
- `data_sources.yaml`: Data source configuration
- `features.yaml`: Feature engineering configuration
- `model_config.yaml`: Model training configuration

**Settings**: `src/openquant/config/settings.py`
- Environment variable support
- Pydantic settings with validation
- Default values for all configurations

## Data Flow

### Data Ingestion Flow
```
1. Scheduler triggers ingestion job
2. Provider fetches data from external source (yfinance)
3. Storage checks for existing data (incremental update)
4. New data is upserted into DuckDB
5. Logs are written
```

### Feature Computation Flow
```
1. Request features for ticker
2. Check Redis cache
3. If cache miss:
   a. Load OHLCV data from DuckDB
   b. Compute features using Feature Engine
   c. Cache results in Redis
4. Return features
```

### Model Training Flow
```
1. Prepare data (OHLCV + Features)
2. Split into train/test sets
3. Initialize model from registry
4. Train model
5. Evaluate metrics
6. Log to MLflow (parameters, metrics, artifacts)
7. Return training results
```

### Prediction Flow
```
1. Load model from MLflow by run_id
2. Get features for ticker
3. Make predictions
4. Return predictions with dates
```

## Storage

### DuckDB
- **Purpose**: Historical OHLCV data storage
- **Schema**: `ohlcv` table with Date, Ticker, Open, High, Low, Close, Volume
- **Location**: `data/openquant.duckdb` (configurable)
- **Features**: ACID transactions, fast analytical queries

### Redis
- **Purpose**: Feature caching
- **TTL**: 24 hours (configurable)
- **Key Format**: `features:{ticker}:{features}:{dates}`
- **Benefits**: Fast feature retrieval, reduced computation

### MLflow
- **Purpose**: Model tracking and versioning
- **Storage**: File-based (`mlruns/` directory)
- **Tracks**: Parameters, metrics, artifacts, models
- **Experiments**: Organized by experiment name

## Deployment

### Docker Compose Services

1. **redis**: Redis cache service
2. **data**: Data volume manager
3. **api**: FastAPI service (port 8000)
4. **dashboard**: Streamlit dashboard (port 8501)
5. **worker**: Background worker for ingestion/scheduling

### Volumes
- `duckdb-data`: DuckDB database files
- `mlflow-data`: MLflow runs and artifacts
- `models-data`: Model artifacts
- `redis-data`: Redis persistence

### Networking
- Bridge network: `openquant-network`
- Services communicate via service names
- Ports exposed for external access

## Development Workflow

1. **Local Development**:
   ```bash
   uv pip install -e ".[dev]"
   pytest
   ruff check src/
   mypy src/
   ```

2. **Docker Development**:
   ```bash
   docker compose up -d
   docker compose logs -f
   ```

3. **Data Ingestion**:
   ```bash
   python scripts/setup_db.py
   python scripts/run_ingestion.py
   ```

4. **Model Training**:
   ```bash
   python scripts/train_model.py
   ```

## Testing

- **Framework**: pytest
- **Coverage**: pytest-cov
- **Fixtures**: `tests/conftest.py`
- **Test Files**:
  - `test_ingestion.py`: Data ingestion tests
  - `test_features.py`: Feature engineering tests
  - `test_models.py`: Model training/prediction tests
  - `test_api.py`: API endpoint tests

## Logging

- **Framework**: loguru
- **Levels**: DEBUG, INFO, WARNING, ERROR
- **Outputs**:
  - Console (colored, structured)
  - File: `logs/openquant.log` (rotated, 100MB, 10 days retention)

## Security Considerations

- Environment variables for sensitive configuration
- No hardcoded credentials
- Input validation with Pydantic
- Error handling without exposing internals
- Docker security best practices

## Performance Optimizations

- Redis caching for computed features
- Incremental data ingestion
- Connection pooling for Redis
- Efficient DuckDB queries
- MLflow artifact optimization

## Future Enhancements

- Additional data providers (Alpha Vantage, Polygon, etc.)
- More ML models (LSTM, Transformer-based)
- Real-time data streaming
- Model serving with dedicated service
- Advanced feature engineering (feature selection, PCA)
- Backtesting framework
- Portfolio optimization
- Alerting and notifications

## Contributing

See project README for contribution guidelines. The architecture is designed to be extensible - new features, models, and data providers can be added through the registry pattern.

