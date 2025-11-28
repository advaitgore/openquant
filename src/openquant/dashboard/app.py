"""Streamlit dashboard for OpenQuant - Multi-page application."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import mlflow
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from openquant.config.settings import settings
from openquant.features.cache import FeatureCache
from openquant.features.engine import FeatureEngine
from openquant.ingestion.storage import DuckDBStorage
from openquant.models.predictor import ModelPredictor
from openquant.models.trainer import ModelTrainer
from openquant.utils.logging import setup_logging

# Configure page
st.set_page_config(
    page_title="OpenQuant Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize logging
setup_logging()

# Initialize components
@st.cache_resource
def get_storage():
    """Get DuckDB storage instance."""
    return DuckDBStorage(settings.DUCKDB_PATH)


@st.cache_resource
def get_feature_engine():
    """Get feature engine instance."""
    return FeatureEngine(storage=get_storage())


@st.cache_resource
def get_feature_cache():
    """Get feature cache instance."""
    try:
        return FeatureCache()
    except Exception:
        return None


@st.cache_resource
def get_trainer():
    """Get model trainer instance."""
    return ModelTrainer(
        storage=get_storage(),
        feature_engine=get_feature_engine(),
        feature_cache=get_feature_cache(),
    )


@st.cache_resource
def get_predictor():
    """Get model predictor instance."""
    return ModelPredictor(
        storage=get_storage(),
        feature_engine=get_feature_engine(),
        feature_cache=get_feature_cache(),
    )


# Sidebar navigation
st.sidebar.title("📈 OpenQuant")
st.sidebar.markdown("---")

page = st.sidebar.selectbox(
    "Navigate",
    ["System Health", "Data Explorer", "Feature Explorer", "Model Playground"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown(
    "OpenQuant is an open-source MLOps platform for financial time-series data."
)


# Page 1: System Health
if page == "System Health":
    st.title("🏥 System Health")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    # Database status
    with col1:
        try:
            storage = get_storage()
            with storage.get_connection() as conn:
                conn.execute("SELECT 1")
            db_status = "✅ Healthy"
            db_color = "green"
        except Exception as e:
            db_status = f"❌ Error: {str(e)[:30]}"
            db_color = "red"

        st.metric("Database", db_status)

    # Redis status
    with col2:
        try:
            cache = get_feature_cache()
            if cache:
                cache.client.ping()
                redis_status = "✅ Healthy"
                redis_color = "green"
            else:
                redis_status = "⚠️ Not Configured"
                redis_color = "orange"
        except Exception as e:
            redis_status = f"❌ Error: {str(e)[:30]}"
            redis_color = "red"

        st.metric("Redis", redis_status)

    # MLflow status
    with col3:
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)
            mlflow_status = "✅ Healthy"
            mlflow_color = "green"
        except Exception as e:
            mlflow_status = f"❌ Error: {str(e)[:30]}"
            mlflow_color = "red"

        st.metric("MLflow", mlflow_status)

    # API status
    with col4:
        try:
            import urllib.request
            url = f"http://{settings.API_HOST}:{settings.API_PORT}/health"
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status == 200:
                    api_status = "✅ Healthy"
                else:
                    api_status = "⚠️ Unhealthy"
        except Exception:
            api_status = "❌ Not Running"

        st.metric("API", api_status)

    st.markdown("---")

    # System information
    st.subheader("System Information")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Configuration")
        st.json({
            "Database Path": settings.DUCKDB_PATH,
            "Redis URL": settings.REDIS_URL,
            "MLflow URI": settings.MLFLOW_TRACKING_URI,
            "API Host": settings.API_HOST,
            "API Port": settings.API_PORT,
        })

    with col2:
        st.markdown("### Statistics")
        try:
            storage = get_storage()
            tickers = storage.get_tickers()
            st.metric("Available Tickers", len(tickers))
            st.metric("Default Tickers", len(settings.DEFAULT_TICKERS))
        except Exception as e:
            st.error(f"Error loading statistics: {e}")

    # Available tickers
    st.markdown("---")
    st.subheader("Available Tickers")
    try:
        storage = get_storage()
        tickers = storage.get_tickers()
        if tickers:
            st.write(", ".join(tickers))
        else:
            st.info("No tickers found in database. Run data ingestion first.")
    except Exception as e:
        st.error(f"Error loading tickers: {e}")


# Page 2: Data Explorer
elif page == "Data Explorer":
    st.title("📊 Data Explorer")
    st.markdown("---")

    storage = get_storage()

    # Ticker selection
    try:
        available_tickers = storage.get_tickers()
        if not available_tickers:
            st.warning("No tickers available. Please run data ingestion first.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        st.stop()

    col1, col2, col3 = st.columns(3)
    with col1:
        ticker = st.selectbox("Select Ticker", available_tickers)
    with col2:
        start_date = st.date_input("Start Date", value=None)
    with col3:
        end_date = st.date_input("End Date", value=None)

    if st.button("Load Data"):
        try:
            start_str = start_date.strftime("%Y-%m-%d") if start_date else None
            end_str = end_date.strftime("%Y-%m-%d") if end_date else None

            df = storage.get_ohlcv(ticker, start_str, end_str)

            if df.empty:
                st.warning(f"No data found for {ticker}")
            else:
                st.success(f"Loaded {len(df)} rows for {ticker}")

                # Display data
                st.subheader("Data Table")
                st.dataframe(df, use_container_width=True)

                # Charts
                st.subheader("Price Chart")
                fig = go.Figure()

                # Candlestick chart
                fig.add_trace(
                    go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name=ticker,
                    )
                )

                fig.update_layout(
                    title=f"{ticker} OHLCV Chart",
                    xaxis_title="Date",
                    yaxis_title="Price",
                    height=500,
                )

                st.plotly_chart(fig, use_container_width=True)

                # Volume chart
                st.subheader("Volume Chart")
                fig_vol = px.bar(
                    df,
                    x="Date",
                    y="Volume",
                    title=f"{ticker} Volume",
                )
                fig_vol.update_layout(height=300)
                st.plotly_chart(fig_vol, use_container_width=True)

                # Statistics
                st.subheader("Statistics")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Close", f"${df['Close'].mean():.2f}")
                with col2:
                    st.metric("Max Close", f"${df['Close'].max():.2f}")
                with col3:
                    st.metric("Min Close", f"${df['Close'].min():.2f}")
                with col4:
                    st.metric("Total Volume", f"{df['Volume'].sum():,}")

        except Exception as e:
            st.error(f"Error loading data: {e}")


# Page 3: Feature Explorer
elif page == "Feature Explorer":
    st.title("🔧 Feature Explorer")
    st.markdown("---")

    storage = get_storage()
    feature_engine = get_feature_engine()

    # Ticker selection
    try:
        available_tickers = storage.get_tickers()
        if not available_tickers:
            st.warning("No tickers available. Please run data ingestion first.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        ticker = st.selectbox("Select Ticker", available_tickers)
    with col2:
        available_features = feature_engine.registry.list_features()
        selected_features = st.multiselect(
            "Select Features",
            available_features,
            default=available_features[:5] if len(available_features) > 5 else available_features,
        )

    col3, col4 = st.columns(2)
    with col3:
        start_date = st.date_input("Start Date", value=None)
    with col4:
        end_date = st.date_input("End Date", value=None)

    if st.button("Compute Features"):
        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            try:
                start_str = start_date.strftime("%Y-%m-%d") if start_date else None
                end_str = end_date.strftime("%Y-%m-%d") if end_date else None

                with st.spinner("Computing features..."):
                    features_df = feature_engine.compute_features_for_ticker(
                        ticker, start_str, end_str, selected_features
                    )

                if features_df.empty:
                    st.warning(f"No features computed for {ticker}")
                else:
                    st.success(f"Computed {len(features_df)} rows with {len(selected_features)} features")

                    # Display data
                    st.subheader("Feature Data")
                    st.dataframe(features_df, use_container_width=True)

                    # Feature charts
                    st.subheader("Feature Visualizations")
                    for feature in selected_features:
                        if feature in features_df.columns:
                            fig = px.line(
                                features_df,
                                x="Date" if "Date" in features_df.columns else features_df.index,
                                y=feature,
                                title=f"{ticker} - {feature}",
                            )
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)

            except Exception as e:
                st.error(f"Error computing features: {e}")

    # Feature list
    st.markdown("---")
    st.subheader("Available Features")
    st.write(f"Total: {len(available_features)} features")
    st.write(", ".join(available_features))


# Page 4: Model Playground
elif page == "Model Playground":
    st.title("🤖 Model Playground")
    st.markdown("---")

    storage = get_storage()
    feature_engine = get_feature_engine()
    trainer = get_trainer()
    predictor = get_predictor()

    # Ticker selection
    try:
        available_tickers = storage.get_tickers()
        if not available_tickers:
            st.warning("No tickers available. Please run data ingestion first.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading tickers: {e}")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["Train Model", "View Predictions", "Model Comparison"])

    with tab1:
        st.subheader("Train a New Model")
        col1, col2 = st.columns(2)

        with col1:
            train_ticker = st.selectbox("Ticker", available_tickers, key="train_ticker")
            model_type = st.selectbox(
                "Model Type",
                ["xgboost", "lightgbm", "random_forest", "linear_regression"],
            )
            available_features = feature_engine.registry.list_features()
            selected_features = st.multiselect(
                "Features",
                available_features,
                default=available_features[:5] if len(available_features) > 5 else available_features,
            )

        with col2:
            target = st.selectbox("Target", ["returns", "Close"])
            lookback_days = st.number_input("Lookback Days", min_value=1, max_value=30, value=1)
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

        # Model parameters
        st.subheader("Model Parameters")
        n_estimators = st.number_input("N Estimators", min_value=10, max_value=1000, value=100)
        max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=6)
        learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)

        if st.button("Train Model", type="primary"):
            if not selected_features:
                st.warning("Please select at least one feature.")
            else:
                try:
                    with st.spinner("Training model..."):
                        model_params = {
                            "n_estimators": n_estimators,
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                        }

                        results = trainer.train(
                            ticker=train_ticker,
                            model_type=model_type,
                            feature_names=selected_features,
                            model_params=model_params,
                            target=target,
                            lookback_days=lookback_days,
                            test_size=test_size,
                        )

                    st.success("Model trained successfully!")
                    st.subheader("Training Results")

                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("### Training Metrics")
                        st.json(results["train_metrics"])

                    with col2:
                        st.markdown("### Test Metrics")
                        st.json(results["test_metrics"])

                    st.info(f"Run ID: {results['run_id']}")

                except Exception as e:
                    st.error(f"Error training model: {e}")

    with tab2:
        st.subheader("Make Predictions")
        col1, col2 = st.columns(2)

        with col1:
            pred_ticker = st.selectbox("Ticker", available_tickers, key="pred_ticker")
            run_id = st.text_input("MLflow Run ID", placeholder="Enter run ID")

        with col2:
            n_days = st.number_input("Number of Days", min_value=1, max_value=100, value=10)

        if st.button("Get Predictions"):
            if not run_id:
                st.warning("Please enter a run ID.")
            else:
                try:
                    with st.spinner("Making predictions..."):
                        predictions_df = predictor.predict(
                            run_id=run_id,
                            ticker=pred_ticker,
                            start_date=None,
                            end_date=None,
                        )

                    if predictions_df.empty:
                        st.warning("No predictions generated.")
                    else:
                        st.success(f"Generated {len(predictions_df)} predictions")

                        # Display predictions
                        st.subheader("Predictions")
                        st.dataframe(predictions_df, use_container_width=True)

                        # Prediction chart
                        if "Date" in predictions_df.columns:
                            fig = px.line(
                                predictions_df,
                                x="Date",
                                y="prediction",
                                title=f"{pred_ticker} - Predictions",
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error making predictions: {e}")

    with tab3:
        st.subheader("Model Comparison")
        st.info("Compare multiple model runs from MLflow")

        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            experiment = mlflow.get_experiment_by_name(settings.MLFLOW_EXPERIMENT_NAME)

            if experiment:
                runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
                if not runs.empty:
                    st.dataframe(runs[["run_id", "tags.mlflow.runName", "metrics.test_r2", "metrics.test_rmse"]], use_container_width=True)
                else:
                    st.info("No runs found. Train a model first.")
            else:
                st.warning("MLflow experiment not found.")
        except Exception as e:
            st.error(f"Error loading MLflow runs: {e}")

