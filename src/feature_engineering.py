"""
Feature engineering module for time series electricity forecasting.
Constructs leakage-safe calendar, cyclical, lag, and rolling features.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
import pandas as pd

from src.config import (
    TARGET_COL,
    LAG_HOURS,
    CALENDAR_FEATURES,
    LAG_FEATURES,
    ROLLING_FEATURES,
    ALL_MODEL_FEATURES,
    FEATURED_DATA_CSV,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def create_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates temporal and cyclical calendar features from the DataFrame index.

    Features generated:
    - hour: Hour of day (0-23)
    - day_of_week: Day of week (0=Monday, 6=Sunday)
    - day_of_month: Day of month (1-31)
    - month: Month of year (1-12)
    - is_weekend: Binary flag (1 if Saturday or Sunday, 0 otherwise)
    - hour_sin, hour_cos: 24-hour cyclical sine & cosine encodings
    - day_of_week_sin, day_of_week_cos: 7-day cyclical encodings
    - month_sin, month_cos: 12-month annual cyclical encodings
    """
    df_out = df.copy()
    idx = df_out.index

    hour = idx.hour
    day_of_week = idx.dayofweek
    month = idx.month

    df_out["hour"] = hour
    df_out["day_of_week"] = day_of_week
    df_out["day_of_month"] = idx.day
    df_out["month"] = month
    df_out["is_weekend"] = (day_of_week >= 5).astype(int)

    # Cyclical encodings prevent artificial discontinuity between hour 23 and 0
    df_out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df_out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    df_out["day_of_week_sin"] = np.sin(2 * np.pi * day_of_week / 7.0)
    df_out["day_of_week_cos"] = np.cos(2 * np.pi * day_of_week / 7.0)

    df_out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df_out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    return df_out


def create_lag_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    lags: List[int] = LAG_HOURS,
) -> pd.DataFrame:
    """
    Creates historical lag features for the target consumption variable.
    All lags represent measurements strictly prior to the current prediction step.

    - lag_1: t-1 (previous hour consumption)
    - lag_2: t-2 (2 hours ago)
    - lag_3: t-3 (3 hours ago)
    - lag_24: t-24 (same hour previous day - daily seasonality)
    - lag_168: t-168 (same hour previous week - weekly seasonality)
    """
    df_out = df.copy()
    for lag in lags:
        df_out[f"lag_{lag}"] = df_out[target_col].shift(lag)
    return df_out


def create_rolling_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Creates leakage-safe rolling statistics over historical consumption.

    CRITICAL LEAKAGE PREVENTION:
    All rolling windows are strictly preceded by `.shift(1)` so that the
    current timestamp's actual value is never included in the rolling calculation.
    Only strictly historical observations [t-W, ..., t-1] are utilized.

    - rolling_mean_6h: Average consumption over past 6 hours
    - rolling_std_6h: Consumption volatility over past 6 hours
    - rolling_mean_24h: 24-hour moving average
    - rolling_max_24h: 24-hour peak historical consumption
    - rolling_min_24h: 24-hour baseline historical consumption
    """
    df_out = df.copy()
    # Shift strictly by 1 to guarantee past-only context
    shifted_target = df_out[target_col].shift(1)

    df_out["rolling_mean_6h"] = shifted_target.rolling(window=6, min_periods=4).mean()
    df_out["rolling_std_6h"] = shifted_target.rolling(window=6, min_periods=4).std().fillna(0.0)
    df_out["rolling_mean_24h"] = shifted_target.rolling(window=24, min_periods=18).mean()
    df_out["rolling_max_24h"] = shifted_target.rolling(window=24, min_periods=18).max()
    df_out["rolling_min_24h"] = shifted_target.rolling(window=24, min_periods=18).min()

    return df_out


def build_full_feature_matrix(
    df_hourly: pd.DataFrame,
    target_col: str = TARGET_COL,
    drop_warmup_na: bool = True,
    save_output: bool = False,
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline:
    1. Calendar & cyclical features
    2. Historical lags (1, 2, 3, 24, 168)
    3. Shifted rolling statistics (6h, 24h)
    4. Warmup trimming (drops initial rows that lack 168 hours of historical lags)

    Args:
        df_hourly: Clean hourly DataFrame indexed by DatetimeIndex.
        target_col: Name of the target consumption column.
        drop_warmup_na: If True, drops rows containing NaN caused by lag_168 warmup.
        save_output: Whether to persist featured dataset to CSV.
        output_path: Path to destination CSV.

    Returns:
        pd.DataFrame containing feature matrix with target.
    """
    logger.info("Executing full feature engineering pipeline...")
    if not isinstance(df_hourly.index, pd.DatetimeIndex):
        raise TypeError("Input DataFrame must be indexed by pd.DatetimeIndex.")

    df_feat = create_calendar_features(df_hourly)
    df_feat = create_lag_features(df_feat, target_col=target_col)
    df_feat = create_rolling_features(df_feat, target_col=target_col)

    if drop_warmup_na:
        initial_len = len(df_feat)
        # Drop only rows where required model features are NaN
        df_feat = df_feat.dropna(subset=ALL_MODEL_FEATURES + [target_col]).copy()
        dropped_count = initial_len - len(df_feat)
        logger.info(
            f"Dropped {dropped_count} initial warmup rows (max lag = 168h). "
            f"Remaining complete rows: {len(df_feat):,}."
        )

    if save_output:
        out_csv = Path(output_path) if output_path else FEATURED_DATA_CSV
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_feat.to_csv(out_csv, index=True)
        logger.info(f"Saved featured dataset to: {out_csv.resolve()}")

    return df_feat


def compute_single_step_features(
    history_series: pd.Series,
    target_timestamp: pd.Timestamp,
    feature_columns: List[str] = ALL_MODEL_FEATURES,
) -> pd.DataFrame:
    """
    Constructs a single 1-row DataFrame of features for a specific future timestamp,
    using strictly available past observations from history_series.

    Used by the forecasting engine during one-step and recursive multi-step forecasting.

    Args:
        history_series: pd.Series of past energy observations ending before target_timestamp.
        target_timestamp: Future timestamp to predict for.
        feature_columns: Exact list and order of feature columns expected by the model.

    Returns:
        1-row pd.DataFrame matching feature_columns schema.
    """
    feat_dict: Dict[str, float] = {}

    # Calendar features
    hour = target_timestamp.hour
    dow = target_timestamp.dayofweek
    dom = target_timestamp.day
    month = target_timestamp.month

    feat_dict["hour"] = float(hour)
    feat_dict["day_of_week"] = float(dow)
    feat_dict["day_of_month"] = float(dom)
    feat_dict["month"] = float(month)
    feat_dict["is_weekend"] = 1.0 if dow >= 5 else 0.0

    # Cyclical
    feat_dict["hour_sin"] = float(np.sin(2 * np.pi * hour / 24.0))
    feat_dict["hour_cos"] = float(np.cos(2 * np.pi * hour / 24.0))
    feat_dict["day_of_week_sin"] = float(np.sin(2 * np.pi * dow / 7.0))
    feat_dict["day_of_week_cos"] = float(np.cos(2 * np.pi * dow / 7.0))
    feat_dict["month_sin"] = float(np.sin(2 * np.pi * (month - 1) / 12.0))
    feat_dict["month_cos"] = float(np.cos(2 * np.pi * (month - 1) / 12.0))

    # Lags (values in history relative to target_timestamp)
    # history_series is indexed chronologically by pd.Timestamp
    vals = history_series.values
    n = len(vals)

    if n < max(LAG_HOURS):
        raise ValueError(
            f"History series length ({n}) is shorter than maximum required lag ({max(LAG_HOURS)})."
        )

    # lag_1 is the most recent value (index -1)
    feat_dict["lag_1"] = float(vals[-1])
    feat_dict["lag_2"] = float(vals[-2])
    feat_dict["lag_3"] = float(vals[-3])
    feat_dict["lag_24"] = float(vals[-24])
    feat_dict["lag_168"] = float(vals[-168])

    # Rolling statistics over recent past
    past_6 = vals[-6:]
    past_24 = vals[-24:]

    feat_dict["rolling_mean_6h"] = float(np.mean(past_6))
    feat_dict["rolling_std_6h"] = float(np.std(past_6, ddof=1)) if len(past_6) > 1 else 0.0
    feat_dict["rolling_mean_24h"] = float(np.mean(past_24))
    feat_dict["rolling_max_24h"] = float(np.max(past_24))
    feat_dict["rolling_min_24h"] = float(np.min(past_24))

    # Ensure columns match expected feature ordering
    row_df = pd.DataFrame([feat_dict], index=[target_timestamp])[feature_columns]
    return row_df
