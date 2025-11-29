"""Locust load testing configuration for OpenQuant API.

Run with: locust -f locustfile.py --host=http://localhost:8000
Access web UI at: http://localhost:8089
"""

from locust import HttpUser, between, task
import random


class OpenQuantUser(HttpUser):
    """Simulates a user interacting with the OpenQuant API."""
    
    wait_time = between(0.5, 2.0)  # Wait 0.5-2 seconds between tasks
    
    def on_start(self):
        """Called when a simulated user starts."""
        # Get available tickers for use in tests
        try:
            response = self.client.get("/data/tickers", name="Get Tickers")
            if response.status_code == 200:
                data = response.json()
                self.tickers = data.get("tickers", ["SPY", "MSFT", "AAPL", "GOOGL", "TSLA"])
            else:
                self.tickers = ["SPY", "MSFT", "AAPL"]
        except Exception:
            self.tickers = ["SPY", "MSFT", "AAPL"]
    
    @task(3)
    def health_check(self):
        """Health check endpoint - most common request."""
        self.client.get("/health", name="Health Check")
    
    @task(2)
    def list_tickers(self):
        """List available tickers."""
        self.client.get("/data/tickers", name="List Tickers")
    
    @task(2)
    def get_ohlcv_data(self):
        """Get OHLCV data for a random ticker."""
        ticker = random.choice(self.tickers)
        self.client.get(
            f"/data/{ticker}",
            name="Get OHLCV Data",
            params={"start_date": "2024-01-01", "end_date": "2024-12-31"}
        )
    
    @task(1)
    def compute_features(self):
        """Compute features for a ticker."""
        ticker = random.choice(self.tickers)
        payload = {
            "ticker": ticker,
            "feature_names": ["sma_20", "sma_50", "rsi_14"],
            "start_date": "2024-01-01",
            "end_date": "2024-12-31"
        }
        self.client.post("/features/compute", json=payload, name="Compute Features")
    
    @task(1)
    def get_api_info(self):
        """Get API root information."""
        self.client.get("/", name="API Info")

