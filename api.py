"""
REST API for AI-Powered Energy Consumption Forecasting System.
Built with Flask, providing model health, metadata, next-hour prediction,
and multi-step recursive forecasting endpoints.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np

from src.config import (
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    PROCESSED_HOURLY_CSV,
    TARGET_COL,
    TARGET_UNITS,
    DEFAULT_FORECAST_HORIZON,
)
from src.forecasting import ForecastingEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Initialize engine globally
engine = ForecastingEngine()


def get_latest_dataset_history() -> pd.Series:
    """Helper to fetch recent history from the processed dataset if available."""
    if not PROCESSED_HOURLY_CSV.exists():
        raise FileNotFoundError("Processed dataset is not available on server.")
    df = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)
    return df[TARGET_COL].iloc[-336:]  # Last 2 weeks of observations


@app.route("/", methods=["GET"])
def root() -> Any:
    """Root endpoint providing service metadata and API directory."""
    return jsonify(
        {
            "service": "AI-Powered Energy Consumption Forecasting API",
            "version": "1.0.0",
            "status": "operational",
            "endpoints": {
                "GET /": "Service overview and endpoint catalog",
                "GET /health": "Healthcheck and model readiness status",
                "GET /model-info": "Model architecture, target details, and training metadata",
                "POST /predict": "One-step-ahead next-hour electricity forecast",
                "POST /forecast": "Recursive multi-step forecast for custom horizon (1-168 hours)",
            },
        }
    ), 200


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Health check reporting server status and model readiness."""
    is_ready = engine.is_ready()
    status_code = 200 if is_ready else 503
    return jsonify(
        {
            "status": "healthy" if is_ready else "degraded",
            "model_loaded": is_ready,
            "model_path_exists": BEST_MODEL_FILE.exists(),
            "message": "Engine ready for inference." if is_ready else "Model artifact not loaded. Train model first.",
        }
    ), status_code


@app.route("/model-info", methods=["GET"])
def model_info() -> Any:
    """Returns training metadata, selected model name, metrics, and feature list."""
    if not engine.is_ready():
        return jsonify({"error": "Model artifact is not loaded. Train model first."}), 503

    if MODEL_METADATA_FILE.exists():
        with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return jsonify(metadata), 200
    else:
        return jsonify(
            {
                "model_name": type(engine.model).__name__,
                "target": TARGET_COL,
                "target_units": TARGET_UNITS,
                "feature_count": len(engine.feature_columns),
                "features": engine.feature_columns,
            }
        ), 200


@app.route("/predict", methods=["POST"])
def predict_next_hour() -> Any:
    """
    POST /predict
    Generates a one-step-ahead next-hour forecast.

    Accepts JSON body:
    {
      "history_kwh": [float, float, ...], # At least 168 hourly values
      "last_timestamp": "2026-10-13T20:00:00" # Optional ISO timestamp
    }
    Or empty body {} to forecast from the latest timestamp in the processed dataset.
    """
    if not engine.is_ready():
        return jsonify({"error": "Forecasting model is not ready. Train model first."}), 503

    data = request.get_json(silent=True) or {}

    try:
        if "history_kwh" in data:
            raw_history = data["history_kwh"]
            if not isinstance(raw_history, list) or len(raw_history) < 168:
                return jsonify(
                    {"error": f"Invalid 'history_kwh'. Must be an array of at least 168 numeric hourly readings (provided {len(raw_history) if isinstance(raw_history, list) else 0})."}
                ), 400

            # Check numeric types
            try:
                numeric_vals = [float(v) for v in raw_history]
            except (ValueError, TypeError):
                return jsonify({"error": "All elements in 'history_kwh' must be valid numeric values."}), 400

            last_ts_str = data.get("last_timestamp")
            if last_ts_str:
                try:
                    last_ts = pd.to_datetime(last_ts_str)
                except Exception:
                    return jsonify({"error": "Invalid 'last_timestamp' format. Must be ISO 8601 string."}), 400
            else:
                last_ts = pd.Timestamp.now().floor("h")

            idx = pd.date_range(end=last_ts, periods=len(numeric_vals), freq="1h")
            history_series = pd.Series(numeric_vals, index=idx)
        else:
            # Fall back to latest processed dataset observations
            try:
                history_series = get_latest_dataset_history()
            except Exception as e:
                return jsonify(
                    {"error": f"No history provided and unable to load dataset history: {str(e)}"}
                ), 400

        result = engine.predict_next_hour(history_series)
        return jsonify(result), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected prediction error: {e}")
        return jsonify({"error": "An internal error occurred while generating the prediction."}), 500


@app.route("/forecast", methods=["POST"])
def forecast_horizon() -> Any:
    """
    POST /forecast
    Generates a recursive multi-step forecast over a horizon of 1 to 168 hours.

    Accepts JSON body:
    {
      "horizon_hours": 24, # Integer between 1 and 168
      "history_kwh": [float, float, ...], # Optional: at least 168 readings
      "last_timestamp": "2026-10-13T20:00:00" # Optional ISO timestamp
    }
    """
    if not engine.is_ready():
        return jsonify({"error": "Forecasting model is not ready. Train model first."}), 503

    data = request.get_json(silent=True) or {}

    horizon = data.get("horizon_hours", DEFAULT_FORECAST_HORIZON)
    if not isinstance(horizon, int) or horizon < 1 or horizon > 168:
        return jsonify(
            {"error": "Invalid 'horizon_hours'. Must be an integer between 1 and 168 (hours)."}
        ), 400

    try:
        if "history_kwh" in data:
            raw_history = data["history_kwh"]
            if not isinstance(raw_history, list) or len(raw_history) < 168:
                return jsonify(
                    {"error": f"Invalid 'history_kwh'. Must be an array of at least 168 numeric readings."}
                ), 400

            numeric_vals = [float(v) for v in raw_history]
            last_ts_str = data.get("last_timestamp")
            if last_ts_str:
                last_ts = pd.to_datetime(last_ts_str)
            else:
                last_ts = pd.Timestamp.now().floor("h")

            idx = pd.date_range(end=last_ts, periods=len(numeric_vals), freq="1h")
            history_series = pd.Series(numeric_vals, index=idx)
        else:
            history_series = get_latest_dataset_history()

        forecast_df = engine.predict_horizon(history_series, horizon_hours=horizon)

        predictions_list = []
        for ts, row in forecast_df.iterrows():
            predictions_list.append(
                {
                    "step": int(row["step"]),
                    "timestamp": ts.isoformat(),
                    "predicted_energy_kwh": float(row["predicted_energy_kwh"]),
                    "hour": int(row["hour"]),
                    "day_of_week": str(row["day_of_week"]),
                    "is_weekend": int(row["is_weekend"]),
                }
            )

        total_kwh = float(forecast_df["predicted_energy_kwh"].sum())

        return jsonify(
            {
                "forecast_type": "recursive_multi_step",
                "horizon_hours": horizon,
                "total_predicted_kwh": round(total_kwh, 4),
                "units": TARGET_UNITS,
                "model_used": engine.metadata.get("model_name", "Unknown"),
                "predictions": predictions_list,
            }
        ), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        logger.error(f"Unexpected forecasting error: {e}")
        return jsonify({"error": "An internal error occurred while generating the horizon forecast."}), 500


if __name__ == "__main__":
    # Safe local development server settings (debug disabled by default)
    logger.info("Starting Energy Forecasting Flask API server on http://127.0.0.1:5000 ...")
    app.run(host="127.0.0.1", port=5000, debug=False)
