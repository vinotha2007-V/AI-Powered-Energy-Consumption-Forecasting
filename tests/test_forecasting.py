"""
Unit tests for forecasting engine, model persistence, and recursive multi-step forecasting.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
import joblib

from src.config import ALL_MODEL_FEATURES, TARGET_COL
from src.feature_engineering import build_full_feature_matrix
from src.evaluate_model import calculate_metrics
from src.forecasting import ForecastingEngine


@pytest.fixture
def trained_toy_model_and_history(tmp_path):
    """
    Creates a small synthetic dataset, trains a fast model, saves artifact to tmp_path,
    and returns (ForecastingEngine, history_series).
    """
    dates = pd.date_range("2026-01-01 00:00:00", periods=400, freq="1h")
    # Sine wave consumption pattern
    hours = dates.hour
    values = 1.5 + 0.8 * np.sin(2 * np.pi * hours / 24.0) + np.random.RandomState(42).normal(0, 0.05, 400)
    df_raw = pd.DataFrame({TARGET_COL: np.maximum(0.1, values)}, index=dates)

    # Feature matrix
    feat_df = build_full_feature_matrix(df_raw, drop_warmup_na=True)

    X = feat_df[ALL_MODEL_FEATURES]
    y = feat_df[TARGET_COL]

    model = HistGradientBoostingRegressor(max_iter=20, random_state=42)
    model.fit(X, y)

    model_file = tmp_path / "toy_model.joblib"
    joblib.dump(model, model_file)

    engine = ForecastingEngine(model_path=model_file)
    history_series = df_raw[TARGET_COL].iloc[-200:]
    return engine, history_series


def test_metric_calculations():
    """Verifies MAE, RMSE, and R2 calculation logic."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.2, 1.8, 3.0, 4.2, 4.8])

    metrics = calculate_metrics(y_true, y_pred)
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    assert np.isclose(metrics["mae"], 0.16, atol=1e-4)
    assert metrics["r2"] > 0.95


def test_predict_next_hour(trained_toy_model_and_history):
    """Verifies one-step-ahead next-hour prediction."""
    engine, history = trained_toy_model_and_history
    res = engine.predict_next_hour(history)

    assert "predicted_energy_kwh" in res
    assert res["predicted_energy_kwh"] > 0.0
    assert "timestamp" in res
    expected_ts = history.index[-1] + pd.Timedelta(hours=1)
    assert res["timestamp"] == expected_ts.isoformat()


def test_predict_horizon_recursive(trained_toy_model_and_history):
    """Verifies recursive multi-step forecasting over 24 hours."""
    engine, history = trained_toy_model_and_history
    forecast_df = engine.predict_horizon(history, horizon_hours=24)

    assert len(forecast_df) == 24
    assert "predicted_energy_kwh" in forecast_df.columns
    assert (forecast_df["predicted_energy_kwh"] >= 0).all()
    # Check that timestamps advance sequentially by 1 hour
    time_diffs = forecast_df.index.to_series().diff().dropna()
    assert (time_diffs == pd.Timedelta(hours=1)).all()


def test_invalid_horizon_rejected(trained_toy_model_and_history):
    """Verifies error handling for invalid forecast horizons."""
    engine, history = trained_toy_model_and_history
    with pytest.raises(ValueError):
        engine.predict_horizon(history, horizon_hours=0)
    with pytest.raises(ValueError):
        engine.predict_horizon(history, horizon_hours=200)


def test_insufficient_history_rejected(trained_toy_model_and_history):
    """Verifies error handling when historical context has fewer than 168 hours."""
    engine, history = trained_toy_model_and_history
    short_history = history.iloc[-50:]  # Only 50 hours
    with pytest.raises(ValueError):
        engine.predict_next_hour(short_history)
