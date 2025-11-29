"""Model drift detection for monitoring feature distribution changes."""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from openquant.features.engine import FeatureEngine
from openquant.ingestion.storage import DuckDBStorage


class ModelDriftDetector:
    """Detect model drift by comparing feature distributions."""
    
    def __init__(
        self,
        storage: Optional[DuckDBStorage] = None,
        feature_engine: Optional[FeatureEngine] = None,
    ):
        """Initialize drift detector.
        
        Args:
            storage: DuckDB storage instance.
            feature_engine: Feature engine instance.
        """
        self.storage = storage
        self.feature_engine = feature_engine
    
    def detect_drift(
        self,
        ticker: str,
        feature_names: List[str],
        reference_start: str,
        reference_end: str,
        current_start: str,
        current_end: str,
        drift_threshold: float = 0.05,
    ) -> Dict[str, any]:
        """Detect drift by comparing reference and current feature distributions.
        
        Args:
            ticker: Stock ticker symbol.
            feature_names: List of feature names to check.
            reference_start: Reference period start date.
            reference_end: Reference period end date.
            current_start: Current period start date.
            current_end: Current period end date.
            drift_threshold: Threshold for drift detection (default 0.05 = 5%).
            
        Returns:
            Dictionary with drift detection results.
        """
        try:
            # Get reference features
            ref_features = self.feature_engine.compute_features_for_ticker(
                ticker, reference_start, reference_end, feature_names
            )
            
            # Get current features
            current_features = self.feature_engine.compute_features_for_ticker(
                ticker, current_start, current_end, feature_names
            )
            
            if ref_features.empty or current_features.empty:
                return {
                    "drift_detected": False,
                    "message": "Insufficient data for drift detection",
                    "features": {},
                }
            
            drift_results = {}
            any_drift = False
            
            for feature in feature_names:
                if feature not in ref_features.columns or feature not in current_features.columns:
                    continue
                
                ref_data = ref_features[feature].dropna()
                current_data = current_features[feature].dropna()
                
                if len(ref_data) == 0 or len(current_data) == 0:
                    continue
                
                # Calculate distribution statistics
                ref_mean = ref_data.mean()
                ref_std = ref_data.std()
                
                current_mean = current_data.mean()
                current_std = current_data.std()
                
                # Calculate drift metrics
                # 1. Mean shift (percentage change)
                if abs(ref_mean) > 1e-6:
                    mean_shift = abs((current_mean - ref_mean) / ref_mean)
                else:
                    mean_shift = abs(current_mean - ref_mean)
                
                # 2. Standard deviation shift
                if ref_std > 1e-6:
                    std_shift = abs((current_std - ref_std) / ref_std)
                else:
                    std_shift = abs(current_std - ref_std)
                
                # 3. Kolmogorov-Smirnov test statistic (approximate)
                # Simple approximation: compare percentiles
                ref_percentiles = np.percentile(ref_data, [25, 50, 75])
                current_percentiles = np.percentile(current_data, [25, 50, 75])
                percentile_shift = np.mean(np.abs((current_percentiles - ref_percentiles) / (ref_percentiles + 1e-6)))
                
                # Determine if drift detected
                max_shift = max(mean_shift, std_shift, percentile_shift)
                drift_detected = max_shift > drift_threshold
                
                if drift_detected:
                    any_drift = True
                
                drift_results[feature] = {
                    "drift_detected": drift_detected,
                    "mean_shift": float(mean_shift),
                    "std_shift": float(std_shift),
                    "percentile_shift": float(percentile_shift),
                    "max_shift": float(max_shift),
                    "reference_mean": float(ref_mean),
                    "current_mean": float(current_mean),
                    "reference_std": float(ref_std),
                    "current_std": float(current_std),
                }
            
            return {
                "drift_detected": any_drift,
                "message": f"Drift detected in {sum(1 for r in drift_results.values() if r['drift_detected'])}/{len(drift_results)} features" if any_drift else "No significant drift detected",
                "features": drift_results,
                "threshold": drift_threshold,
            }
            
        except Exception as e:
            logger.error(f"Error detecting drift: {e}")
            return {
                "drift_detected": False,
                "message": f"Error: {e}",
                "features": {},
            }

