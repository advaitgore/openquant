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
from openquant.monitoring.drift import ModelDriftDetector
from openquant.utils.logging import setup_logging

import urllib.request
import json
import subprocess
from datetime import datetime, timedelta
from io import StringIO
import re

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
    ["System Health", "Data Explorer", "Feature Explorer", "Model Playground", "Admin Dashboard"],
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
        
        # Only show max_depth for tree-based models
        if model_type in ["xgboost", "lightgbm", "random_forest"]:
            max_depth = st.number_input("Max Depth", min_value=1, max_value=20, value=6)
        
        # Only show learning_rate for gradient boosting models
        if model_type in ["xgboost", "lightgbm"]:
            learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)

        if st.button("Train Model", type="primary"):
            if not selected_features:
                st.warning("Please select at least one feature.")
            else:
                try:
                    with st.spinner("Training model..."):
                        # Build model_params based on model type
                        model_params = {
                            "n_estimators": n_estimators,
                        }
                        
                        # Add max_depth for tree-based models
                        if model_type in ["xgboost", "lightgbm", "random_forest"]:
                            model_params["max_depth"] = max_depth
                        
                        # Add learning_rate only for gradient boosting models
                        if model_type in ["xgboost", "lightgbm"]:
                            model_params["learning_rate"] = learning_rate

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
                        
                        # Add explanation of what predictions mean
                        if "prediction" in predictions_df.columns:
                            st.info("💡 **What are these predictions?**\n\n"
                                  "- Values represent **predicted returns** (as decimals)\n"
                                  "- Example: -0.01 = -1% return (1% loss), 0.02 = +2% return (2% gain)\n"
                                  "- Dates show when the prediction is FOR (future date)")
                        
                        st.dataframe(predictions_df, use_container_width=True)

                        # Prediction chart
                        if "Date" in predictions_df.columns:
                            fig = px.line(
                                predictions_df,
                                x="Date",
                                y="prediction",
                                title=f"{pred_ticker} - Predicted Returns",
                                labels={"prediction": "Predicted Return (decimal)", "Date": "Prediction Date"},
                            )
                            fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                                        annotation_text="Zero return line")
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


# Page 5: Admin Dashboard
elif page == "Admin Dashboard":
    st.title("⚙️ Admin Dashboard")
    st.markdown("---")
    st.info("🔒 **MLOps Monitoring & Operations** - Track system health, performance, and model drift")
    
    tab1, tab2, tab3 = st.tabs(["API Metrics", "System Resources", "Model Drift"])
    
    with tab1:
        st.subheader("📊 API Performance Metrics")
        
        try:
            # Fetch Prometheus metrics
            metrics_url = f"http://{settings.API_HOST}:{settings.API_PORT}/metrics"
            
            with st.spinner("Fetching metrics..."):
                response = urllib.request.urlopen(metrics_url, timeout=5)
                metrics_text = response.read().decode('utf-8')
            
            # Parse Prometheus metrics
            latency_metrics = {}
            request_counts = {}
            
            for line in metrics_text.split('\n'):
                if line.startswith('#') or not line.strip():
                    continue
                
                # Parse request duration metrics
                if 'openquant_api_request_duration_seconds' in line and not line.startswith('#'):
                    # Extract bucket values
                    match = re.search(r'le="([^"]+)"[^}]*}([0-9.]+)', line)
                    if match:
                        bucket = float(match.group(1))
                        count = float(match.group(2))
                        if bucket not in latency_metrics:
                            latency_metrics[bucket] = 0
                        latency_metrics[bucket] += count
                
                # Parse request counts
                if 'openquant_api_requests_total' in line and not line.startswith('#'):
                    match = re.search(r'} ([0-9.]+)', line)
                    if match:
                        count = float(match.group(1))
                        request_counts[line.split('{')[0].split('_')[-1]] = request_counts.get(line.split('{')[0].split('_')[-1], 0) + count
            
            # Calculate P95 and P99 latency
            if latency_metrics:
                sorted_buckets = sorted(latency_metrics.items())
                total_requests = sum(latency_metrics.values())
                
                if total_requests > 0:
                    p95_threshold = total_requests * 0.95
                    p99_threshold = total_requests * 0.99
                    
                    cumulative = 0
                    p95_latency = None
                    p99_latency = None
                    
                    for bucket, count in sorted_buckets:
                        cumulative += count
                        if p95_latency is None and cumulative >= p95_threshold:
                            p95_latency = bucket * 1000  # Convert to ms
                        if p99_latency is None and cumulative >= p99_threshold:
                            p99_latency = bucket * 1000  # Convert to ms
                            break
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("P95 Latency", f"{p95_latency:.2f} ms" if p95_latency else "N/A")
                    with col2:
                        st.metric("P99 Latency", f"{p99_latency:.2f} ms" if p99_latency else "N/A")
                    with col3:
                        st.metric("Total Requests", f"{int(total_requests):,}")
                    
                    # Latency distribution chart
                    if sorted_buckets:
                        buckets_df = pd.DataFrame(sorted_buckets, columns=["Latency (ms)", "Count"])
                        buckets_df["Latency (ms)"] = buckets_df["Latency (ms)"] * 1000
                        
                        fig = px.bar(
                            buckets_df,
                            x="Latency (ms)",
                            y="Count",
                            title="Request Latency Distribution",
                            labels={"Count": "Request Count"},
                        )
                        fig.add_vline(
                            x=p95_latency if p95_latency else 0,
                            line_dash="dash",
                            line_color="orange",
                            annotation_text="P95"
                        )
                        fig.add_vline(
                            x=p99_latency if p99_latency else 0,
                            line_dash="dash",
                            line_color="red",
                            annotation_text="P99"
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No request data available yet. Start making API requests to see metrics.")
            else:
                st.warning("No latency metrics found. Ensure Prometheus instrumentation is enabled.")
                
        except Exception as e:
            st.error(f"Error fetching metrics: {e}")
            st.info("💡 Make sure the API is running and Prometheus metrics are enabled at /metrics")
    
    with tab2:
        st.subheader("💾 System Resources")
        
        try:
            # Get Docker container stats
            containers = ["api", "dashboard", "worker", "redis", "data"]
            
            container_stats = []
            for container in containers:
                try:
                    result = subprocess.run(
                        ["docker", "stats", "--no-stream", "--format", "{{.Container}}\t{{.MemUsage}}\t{{.CPUPerc}}", container],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    
                    if result.returncode == 0 and result.stdout.strip():
                        parts = result.stdout.strip().split('\t')
                        if len(parts) >= 3:
                            mem_parts = parts[1].split('/')
                            mem_used = mem_parts[0].strip()
                            mem_limit = mem_parts[1].strip() if len(mem_parts) > 1 else "N/A"
                            cpu_perc = parts[2].strip().replace('%', '')
                            
                            # Extract numeric values
                            mem_used_mb = float(re.sub(r'[^0-9.]', '', mem_used)) if 'MiB' in mem_used else 0
                            mem_limit_mb = float(re.sub(r'[^0-9.]', '', mem_limit)) if 'MiB' in mem_limit else 0
                            cpu_perc_float = float(re.sub(r'[^0-9.]', '', cpu_perc)) if cpu_perc else 0
                            
                            mem_usage_pct = (mem_used_mb / mem_limit_mb * 100) if mem_limit_mb > 0 else 0
                            
                            container_stats.append({
                                "Container": container,
                                "Memory Used": f"{mem_used_mb:.1f} MB",
                                "Memory Limit": f"{mem_limit_mb:.1f} MB" if mem_limit_mb > 0 else "N/A",
                                "Memory %": f"{mem_usage_pct:.1f}%",
                                "CPU %": f"{cpu_perc_float:.1f}%"
                            })
                except Exception as e:
                    container_stats.append({
                        "Container": container,
                        "Memory Used": "N/A",
                        "Memory Limit": "N/A",
                        "Memory %": "N/A",
                        "CPU %": f"Error: {str(e)[:30]}"
                    })
            
            if container_stats:
                stats_df = pd.DataFrame(container_stats)
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                # Memory usage chart
                memory_data = []
                for stat in container_stats:
                    mem_pct = float(re.sub(r'[^0-9.]', '', stat["Memory %"])) if stat["Memory %"] != "N/A" else 0
                    memory_data.append({
                        "Container": stat["Container"],
                        "Memory Usage %": mem_pct
                    })
                
                if memory_data:
                    mem_df = pd.DataFrame(memory_data)
                    fig = px.bar(
                        mem_df,
                        x="Container",
                        y="Memory Usage %",
                        title="Memory Usage by Container",
                        color="Memory Usage %",
                        color_continuous_scale="RdYlGn_r"
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Could not fetch container stats. Make sure Docker is running and containers are up.")
                
        except FileNotFoundError:
            st.error("Docker command not found. This feature requires Docker to be installed and accessible.")
        except Exception as e:
            st.error(f"Error fetching container stats: {e}")
    
    with tab3:
        st.subheader("📉 Model Drift Detection")
        st.info("Monitor feature distribution shifts to detect model drift")
        
        storage = get_storage()
        feature_engine = get_feature_engine()
        drift_detector = ModelDriftDetector(storage=storage, feature_engine=feature_engine)
        
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
            ticker = st.selectbox("Ticker", available_tickers, key="drift_ticker")
            available_features = feature_engine.registry.list_features()
            selected_features = st.multiselect(
                "Features to Monitor",
                available_features,
                default=["sma_20", "sma_50", "rsi_14", "returns"] if len(available_features) >= 4 else available_features[:4],
                key="drift_features"
            )
        
        with col2:
            # Date ranges
            today = datetime.now().date()
            reference_end = today - timedelta(days=30)
            reference_start = reference_end - timedelta(days=90)
            current_end = today
            current_start = today - timedelta(days=30)
            
            st.markdown("**Reference Period (Baseline)**")
            ref_start = st.date_input("Start", value=reference_start, key="ref_start")
            ref_end = st.date_input("End", value=reference_end, key="ref_end")
            
            st.markdown("**Current Period (To Compare)**")
            curr_start = st.date_input("Start", value=current_start, key="curr_start")
            curr_end = st.date_input("End", value=current_end, key="curr_end")
            
            drift_threshold = st.slider("Drift Threshold", 0.01, 0.20, 0.05, 0.01, help="Percentage change to trigger drift alert")
        
        if st.button("Check for Drift", type="primary"):
            if not selected_features:
                st.warning("Please select at least one feature to monitor.")
            else:
                try:
                    with st.spinner("Analyzing feature distributions..."):
                        drift_results = drift_detector.detect_drift(
                            ticker=ticker,
                            feature_names=selected_features,
                            reference_start=ref_start.strftime("%Y-%m-%d"),
                            reference_end=ref_end.strftime("%Y-%m-%d"),
                            current_start=curr_start.strftime("%Y-%m-%d"),
                            current_end=curr_end.strftime("%Y-%m-%d"),
                            drift_threshold=drift_threshold,
                        )
                    
                    # Display results
                    if drift_results["drift_detected"]:
                        st.error(f"⚠️ **Drift Detected!** {drift_results['message']}")
                    else:
                        st.success(f"✅ {drift_results['message']}")
                    
                    # Feature-level drift details
                    if drift_results.get("features"):
                        st.subheader("Feature-Level Drift Analysis")
                        
                        drift_data = []
                        for feature, stats in drift_results["features"].items():
                            drift_data.append({
                                "Feature": feature,
                                "Drift Detected": "⚠️ Yes" if stats["drift_detected"] else "✅ No",
                                "Max Shift": f"{stats['max_shift']*100:.2f}%",
                                "Mean Shift": f"{stats['mean_shift']*100:.2f}%",
                                "Std Shift": f"{stats['std_shift']*100:.2f}%",
                                "Current Mean": f"{stats['current_mean']:.4f}",
                                "Reference Mean": f"{stats['reference_mean']:.4f}",
                            })
                        
                        drift_df = pd.DataFrame(drift_data)
                        st.dataframe(drift_df, use_container_width=True, hide_index=True)
                        
                        # Visualize drift
                        features_with_drift = [f for f, s in drift_results["features"].items() if s["drift_detected"]]
                        if features_with_drift:
                            st.subheader("Drift Visualization")
                            
                            shift_data = []
                            for feature in features_with_drift:
                                stats = drift_results["features"][feature]
                                shift_data.append({
                                    "Feature": feature,
                                    "Mean Shift %": stats["mean_shift"] * 100,
                                    "Std Shift %": stats["std_shift"] * 100,
                                })
                            
                            shift_df = pd.DataFrame(shift_data)
                            fig = px.bar(
                                shift_df,
                                x="Feature",
                                y=["Mean Shift %", "Std Shift %"],
                                title="Feature Distribution Shifts",
                                barmode="group"
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error detecting drift: {e}")

