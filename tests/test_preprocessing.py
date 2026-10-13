"""
Unit tests for data preprocessing and hourly resampling.
"""

import pytest
import pandas as pd
import numpy as np

from src.preprocessing import clean_minute_data, resample_hourly_energy
from src.config import DATE_COL, TIME_COL, RAW_ACTIVE_POWER_COL, TARGET_COL


@pytest.fixture
def sample_minute_df():
    """Generates 180 minutes (3 hours) of deterministic test data."""
    dates = pd.date_range("2026-01-01 00:00:00", periods=180, freq="1min")
    power = np.full(180, 2.4)  # 2.4 kW constant power

    # Introduce 5 missing values in hour 1 (still >= 50 valid -> valid hour)
    power[10:15] = np.nan

    # In hour 2 (index 120-179), introduce 20 missing values (40 valid < 50 -> insufficient coverage)
    power[130:150] = np.nan

    df = pd.DataFrame({
        DATE_COL: dates.strftime("%d/%m/%Y"),
        TIME_COL: dates.strftime("%H:%M:%S"),
        RAW_ACTIVE_POWER_COL: power,
        "Global_reactive_power": 0.1,
        "Voltage": 230.0,
        "Global_intensity": 10.0,
        "Sub_metering_1": 0.0,
        "Sub_metering_2": 0.0,
        "Sub_metering_3": 0.0,
    })
    return df


def test_clean_minute_data_parsing(sample_minute_df):
    """Verifies timestamp parsing and sorting."""
    cleaned, stats = clean_minute_data(sample_minute_df)
    assert isinstance(cleaned.index, pd.DatetimeIndex)
    assert len(cleaned) == 180
    assert stats["raw_row_count"] == 180
    assert stats["invalid_timestamp_count"] == 0
    assert cleaned.index.is_monotonic_increasing


def test_clean_minute_duplicate_removal(sample_minute_df):
    """Verifies duplicate timestamps are safely detected and removed."""
    duplicated = pd.concat([sample_minute_df.iloc[[0]], sample_minute_df])
    cleaned, stats = clean_minute_data(duplicated)
    assert stats["duplicate_timestamp_count"] == 1
    assert len(cleaned) == 180


def test_hourly_resampling_energy_integration(sample_minute_df):
    """
    Verifies that power in kW correctly integrates to energy in kWh.
    If power is 2.4 kW for an hour, 2.4 kW * 1 hr = 2.4 kWh.
    Hour 0: 60 minutes of 2.4 kW -> 2.4 kWh.
    Hour 1: 55 minutes of 2.4 kW (>= 50 threshold) -> 2.4 kWh.
    Hour 2: 40 minutes of 2.4 kW (< 50 threshold) -> marked insufficient.
    """
    cleaned, _ = clean_minute_data(sample_minute_df)
    hourly_df, stats = resample_hourly_energy(cleaned, min_minutes_per_hour=50)

    # First hour should be exactly 2.4 kWh
    assert TARGET_COL in hourly_df.columns
    first_hour_val = hourly_df.iloc[0][TARGET_COL]
    assert np.isclose(first_hour_val, 2.4, atol=1e-3)

    # Second hour had 55 minutes -> coverage passed -> should still be 2.4 kWh
    second_hour_val = hourly_df.iloc[1][TARGET_COL]
    assert np.isclose(second_hour_val, 2.4, atol=1e-3)


def test_incomplete_hour_coverage_threshold(sample_minute_df):
    """Verifies that an hour with fewer than 50 valid minutes is marked as insufficient."""
    cleaned, _ = clean_minute_data(sample_minute_df)
    hourly_df, stats = resample_hourly_energy(cleaned, min_minutes_per_hour=50)

    # In our fixture, third hour (index 2) had only 40 valid minutes
    assert stats["insufficient_coverage_hours"] >= 1
