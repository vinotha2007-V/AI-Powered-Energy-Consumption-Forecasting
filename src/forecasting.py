"""
Forecasting engine module.
Provides single-step prediction and recursive multi-step forecasting
without future data leakage, with input validation and history management.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd

from src.config import (
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    ALL_MODEL_FEATURES,
    TARGET_COL,
    LAG_HOURS,
    TARGET_UNITS,
)
from src.feature_engineering import compute_single_step_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ForecastingEngine:
    """
    Forecasting engine that encapsulates model loading, historical context validation,
    one-step ahead prediction, and recursive multi-step forecasting.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        metadata_path: Optional[Path] = None,
    ):
        self.model_path = Path(model_path) if model_path else BEST_MODEL_FILE
        self.metadata_path = Path(metadata_path) if metadata_path else MODEL_METADATA_FILE
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self.feature_columns: List[str] = ALL_MODEL_FEATURES

        self._load_model_and_metadata()

    def _load_model_and_metadata(self) -> None:
        """Loads the serialized model artifact and corresponding metadata."""
        if not self.model_path.exists():
            logger.warning(f"Model file not found at {self.model_path}. Model remains uninitialized.")
            return

        try:
            self.model = joblib.load(self.model_path)
            logger.info(f"Loaded trained model from {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {self.model_path}: {e}")
            raise RuntimeError(f"Error loading model artifact: {e}") from e

        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                    self.feature_columns = self.metadata.get("features", ALL_MODEL_FEATURES)
            except Exception as e:
                logger.warning(f"Could not read model metadata: {e}")

    def is_ready(self) -> bool:
        """Returns True if the model is loaded and ready for inference."""
        return self.model is not None

    def validate_history(self, history_series: pd.Series) -> None:
        """
        Validates historical context required for constructing lag and rolling features.

        Requirements:
        - Must be a pd.Series indexed by pd.DatetimeIndex (or convertible)
        - Must have at least max(LAG_HOURS) = 168 consecutive hourly observations
        - Must not contain null/NaN values
        """
        if not isinstance(history_series, pd.Series):
            raise TypeError("history_series must be a pandas Series.")

        if not isinstance(history_series.index, pd.DatetimeIndex):
            try:
                history_series.index = pd.to_datetime(history_series.index)
            except Exception as e:
                raise ValueError("history_series index could not be converted to DatetimeIndex.") from e

        if history_series.isna().any():
            nan_count = int(history_series.isna().sum())
            raise ValueError(f"Historical series contains {nan_count} missing (NaN) values.")

        min_required = max(LAG_HOURS)
        if len(history_series) < min_required:
            raise ValueError(
                f"Insufficient historical observations. Provided {len(history_series)} hours, "
                f"but at least {min_required} hours (1 week of continuous measurements) are required "
                f"to support lag_168 and rolling windows."
            )

    def predict_next_hour(
        self,
        history_series: pd.Series,
    ) -> Dict[str, Any]:
        """
        Generates a one-step-ahead forecast for the hour immediately following history_series.

        Args:
            history_series: Past hourly observations ending at timestamp T.

        Returns:
            Dictionary containing prediction timestamp, predicted value, and units.
        """
        if not self.is_ready():
            raise RuntimeError("Forecasting engine is not initialized. Model artifact missing.")

        self.validate_history(history_series)

        last_timestamp = history_series.index[-1]
        target_timestamp = last_timestamp + pd.Timedelta(hours=1)

        # Build feature row strictly from past observations
        features_df = compute_single_step_features(
            history_series=history_series,
            target_timestamp=target_timestamp,
            feature_columns=self.feature_columns,
        )

        pred_val = float(self.model.predict(features_df)[0])
        # Energy cannot be negative
        pred_val = max(0.0, round(pred_val, 4))

        return {
            "forecast_type": "one_step_ahead",
            "timestamp": target_timestamp.isoformat(),
            "predicted_energy_kwh": pred_val,
            "units": TARGET_UNITS,
            "reference_last_history_timestamp": last_timestamp.isoformat(),
            "model_used": self.metadata.get("model_name", "Unknown"),
        }

    def predict_horizon(
        self,
        history_series: pd.Series,
        horizon_hours: int = 24,
    ) -> pd.DataFrame:
        """
        Executes a recursive multi-step forecast over a configurable future horizon.

        RECURSIVE MECHANISM:
        1. Predict t+1 using strictly available actual history.
        2. Append prediction (t+1, y_hat_t+1) to an updated virtual history buffer.
        3. Dynamically recompute calendar, lag, and rolling features for t+2 using the virtual history.
        4. Predict t+2 and repeat recursively until horizon H is satisfied.

        IMPORTANT DISTINCTION:
        Unlike one-step-ahead testing where actual test values are fed at each step,
        recursive multi-step forecasting does NOT leak actual future values.
        Any forecast error in early steps propagates into subsequent steps,
        faithfully simulating real-world operational forecasting.

        Args:
            history_series: Historical series ending at timestamp T.
            horizon_hours: Number of future hours to predict (1 to 168).

        Returns:
            pd.DataFrame indexed by future DatetimeIndex with prediction details.
        """
        if not self.is_ready():
            raise RuntimeError("Forecasting engine is not initialized. Model artifact missing.")

        if horizon_hours < 1 or horizon_hours > 168:
            raise ValueError(
                f"Horizon {horizon_hours} is invalid. Must be between 1 and 168 hours (up to 7 days)."
            )

        self.validate_history(history_series)

        # Clone historical context to avoid mutating caller data
        virtual_history = history_series.copy()
        last_timestamp = virtual_history.index[-1]

        future_timestamps = pd.date_range(
            start=last_timestamp + pd.Timedelta(hours=1),
            periods=horizon_hours,
            freq="1h",
        )

        predictions_list: List[Dict[str, Any]] = []

        for step_idx, step_ts in enumerate(future_timestamps, start=1):
            # Compute features for current step from virtual history buffer
            feat_row = compute_single_step_features(
                history_series=virtual_history,
                target_timestamp=step_ts,
                feature_columns=self.feature_columns,
            )

            pred_val = float(self.model.predict(feat_row)[0])
            pred_val = max(0.0, round(pred_val, 4))

            predictions_list.append(
                {
                    "step": step_idx,
                    "timestamp": step_ts,
                    "predicted_energy_kwh": pred_val,
                    "hour": step_ts.hour,
                    "day_of_week": step_ts.day_name(),
                    "is_weekend": int(step_ts.dayofweek >= 5),
                }
            )

            # Append the newly predicted value to virtual history for subsequent recursive steps
            new_point = pd.Series([pred_val], index=[step_ts])
            virtual_history = pd.concat([virtual_history, new_point])

        forecast_df = pd.DataFrame(predictions_list).set_index("timestamp")
        return forecast_df
