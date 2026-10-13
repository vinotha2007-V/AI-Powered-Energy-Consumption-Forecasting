"""
Unit tests for feature engineering and data leakage prevention.
"""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    create_calendar_features,
    create_lag_features,
    create_rolling_features,
    build_full_feature_matrix,
)
from src.train_model import chronological_split
from src.config import TARGET_COL, LAG_HOURS, ALL_MODEL_FEATURES


@pytest.fixture
def sample_hourly_df():
    """Generates 300 hours of synthetic hourly energy observations."""
    dates = pd.date_range("2026-01-01 00:00:00", periods=300, freq="1h")
    # Deterministic pattern: values increasing gradually
    values = np.linspace(1.0, 5.0, 300)
    df = pd.DataFrame({TARGET_COL: values}, index=dates)
    return df


def test_calendar_features(sample_hourly_df):
    """Verifies cyclical trigonometric properties and calendar flags."""
    df_feat = create_calendar_features(sample_hourly_df)
    assert "hour" in df_feat.columns
    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "is_weekend" in df_feat.columns

    # Check bounds of sine and cosine
    assert df_feat["hour_sin"].min() >= -1.0
    assert df_feat["hour_sin"].max() <= 1.0
    # Check identity: sin^2 + cos^2 == 1
    trig_identity = df_feat["hour_sin"] ** 2 + df_feat["hour_cos"] ** 2
    assert np.allclose(trig_identity, 1.0, atol=1e-5)


def test_lag_features(sample_hourly_df):
    """Verifies that lag features correspond exactly to historical shifted values."""
    df_feat = create_lag_features(sample_hourly_df, target_col=TARGET_COL, lags=[1, 24])
    # For index 25, lag_1 must equal index 24, and lag_24 must equal index 1
    assert df_feat["lag_1"].iloc[25] == sample_hourly_df[TARGET_COL].iloc[24]
    assert df_feat["lag_24"].iloc[25] == sample_hourly_df[TARGET_COL].iloc[1]


def test_no_data_leakage_in_rolling_features(sample_hourly_df):
    """
    CRITICAL LEAKAGE TEST:
    Modifying the target value at timestamp T must NOT alter the rolling features at timestamp T.
    Because rolling features are shifted by 1, feature(T) depends only on values at T-1, T-2, etc.
    """
    df_original = create_rolling_features(sample_hourly_df, target_col=TARGET_COL)

    # Clone and inject a drastic spike at timestamp index 100
    df_modified = sample_hourly_df.copy()
    spike_idx = df_modified.index[100]
    df_modified.loc[spike_idx, TARGET_COL] = 99999.0

    df_modified_feat = create_rolling_features(df_modified, target_col=TARGET_COL)

    # At timestamp index 100, rolling features MUST BE IDENTICAL between original and modified!
    assert df_original["rolling_mean_6h"].iloc[100] == df_modified_feat["rolling_mean_6h"].iloc[100]
    assert df_original["rolling_mean_24h"].iloc[100] == df_modified_feat["rolling_mean_24h"].iloc[100]

    # Only at index 101 should the shifted spike begin to affect rolling features
    assert df_original["rolling_mean_6h"].iloc[101] != df_modified_feat["rolling_mean_6h"].iloc[101]


def test_chronological_split_ordering(sample_hourly_df):
    """Verifies that train, validation, and test splits are strictly chronological."""
    train_df, val_df, test_df = chronological_split(sample_hourly_df, train_ratio=0.6, val_ratio=0.2)

    assert len(train_df) + len(val_df) + len(test_df) == len(sample_hourly_df)
    # Check strict time boundaries
    assert train_df.index.max() < val_df.index.min()
    assert val_df.index.max() < test_df.index.min()
