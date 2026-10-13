"""
Data preprocessing module for cleaning, timestamp parsing, coverage verification,
and defensible hourly energy (kWh) integration from raw minute power (kW) data.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    TIME_COL,
    TIMESTAMP_COL,
    RAW_ACTIVE_POWER_COL,
    TARGET_COL,
    MIN_MINUTES_PER_HOUR,
    PROCESSED_HOURLY_CSV,
    PREPROCESSING_REPORT_FILE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def clean_minute_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw minute-level household observations:
    - Parses timestamps safely (dayfirst=True)
    - Casts measurement columns to numeric
    - Checks and resolves duplicate timestamps
    - Checks for invalid or physically impossible readings
    - Sorts chronologically

    Args:
        df: Raw DataFrame containing Date, Time, and measurement columns.

    Returns:
        Tuple of (cleaned DataFrame with DatetimeIndex, preliminary cleaning stats dict)
    """
    stats: Dict[str, Any] = {}
    raw_rows = len(df)
    stats["raw_row_count"] = raw_rows

    logger.info("Parsing dates and times into unified UTC-free datetime index...")
    # Date in UCI dataset is dd/mm/yyyy, Time is HH:MM:SS
    # Combining strings before to_datetime is faster and robust
    datetime_str = df[DATE_COL].astype(str) + " " + df[TIME_COL].astype(str)
    parsed_timestamps = pd.to_datetime(datetime_str, format="%d/%m/%Y %H:%M:%S", errors="coerce")

    invalid_timestamps = parsed_timestamps.isna().sum()
    stats["invalid_timestamp_count"] = int(invalid_timestamps)

    df_clean = df.copy()
    df_clean[TIMESTAMP_COL] = parsed_timestamps

    # Drop any unparseable timestamps
    if invalid_timestamps > 0:
        logger.warning(f"Dropping {invalid_timestamps} unparseable timestamps.")
        df_clean = df_clean.dropna(subset=[TIMESTAMP_COL])

    # Convert numeric measurement columns
    numeric_cols = [
        RAW_ACTIVE_POWER_COL,
        "Global_reactive_power",
        "Voltage",
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3",
    ]
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

    missing_active_power = df_clean[RAW_ACTIVE_POWER_COL].isna().sum()
    stats["raw_missing_active_power_count"] = int(missing_active_power)
    stats["raw_missing_active_power_pct"] = round(float(missing_active_power / raw_rows * 100), 3)

    # Sort chronologically
    df_clean = df_clean.sort_values(by=TIMESTAMP_COL)

    # Check for duplicate timestamps
    duplicates = df_clean.duplicated(subset=[TIMESTAMP_COL]).sum()
    stats["duplicate_timestamp_count"] = int(duplicates)
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate timestamps. Keeping first occurrence.")
        df_clean = df_clean.drop_duplicates(subset=[TIMESTAMP_COL], keep="first")

    # Physical plausibility check (active power must be non-negative)
    negative_power = (df_clean[RAW_ACTIVE_POWER_COL] < 0).sum()
    stats["negative_power_count"] = int(negative_power)
    if negative_power > 0:
        logger.warning(f"Found {negative_power} negative power readings. Setting to NaN.")
        df_clean.loc[df_clean[RAW_ACTIVE_POWER_COL] < 0, RAW_ACTIVE_POWER_COL] = np.nan

    # Sub-metering unit conversion if present (Wh to kWh)
    for sub in ["Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]:
        if sub in df_clean.columns:
            # Sub-metering is active energy in watt-hour per minute
            # sub_kwh = sub_wh / 1000
            df_clean[f"{sub}_kwh"] = df_clean[sub] / 1000.0

    df_clean = df_clean.set_index(TIMESTAMP_COL)
    stats["date_range_start"] = str(df_clean.index.min())
    stats["date_range_end"] = str(df_clean.index.max())

    logger.info(
        f"Minute cleaning complete. Date range: {stats['date_range_start']} to {stats['date_range_end']}."
    )
    return df_clean, stats


def resample_hourly_energy(
    df_minute: pd.DataFrame,
    min_minutes_per_hour: int = MIN_MINUTES_PER_HOUR,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resamples minute-level kW power to defensible hourly energy in kWh.

    Defensible hourly energy calculation:
    Hourly energy (kWh) is the integral of power (kW) over time (1 hour = 60 minutes):
    Energy (kWh) = (Mean active power in kW) * 1.0 hour
                 = Sum(kW readings) / 60 when 60 readings are present.

    Data Coverage Threshold:
    If an hour has fewer than min_minutes_per_hour (default 50 out of 60) valid readings,
    it is marked as NaN (insufficiently observed) to avoid false underestimation.
    For hours with >= min_minutes_per_hour valid readings, the hourly energy is computed
    as mean(kW) * 1.0 hour, properly normalizing for the small proportion of missing minutes.

    Args:
        df_minute: Cleaned minute DataFrame indexed by timestamp.
        min_minutes_per_hour: Minimum number of valid minute readings per hour.

    Returns:
        Tuple of (hourly DataFrame, hourly resampling statistics dict)
    """
    stats: Dict[str, Any] = {}
    logger.info(
        f"Resampling minute measurements to hourly energy (min_valid_minutes_threshold={min_minutes_per_hour})..."
    )

    power_series = df_minute[RAW_ACTIVE_POWER_COL]

    # Group by hourly periods
    hourly_group = power_series.resample("1h")

    # Count valid (non-null) readings per hour
    valid_counts = hourly_group.count()
    # Compute mean power in kW for each hour
    mean_power_kw = hourly_group.mean()

    # Determine valid hours based on coverage threshold
    is_valid_coverage = valid_counts >= min_minutes_per_hour

    # Defensible hourly energy in kWh: mean kW * 1 hour
    energy_kwh = mean_power_kw.copy()
    energy_kwh[~is_valid_coverage] = np.nan

    hourly_df = pd.DataFrame(
        {
            TARGET_COL: energy_kwh,
            "mean_power_kw": mean_power_kw,
            "valid_minutes_count": valid_counts,
            "is_valid_coverage": is_valid_coverage,
        },
        index=valid_counts.index,
    )

    # Sub-meterings aggregation if present
    for sub in ["Sub_metering_1_kwh", "Sub_metering_2_kwh", "Sub_metering_3_kwh"]:
        if sub in df_minute.columns:
            # sum of Wh converted to kWh
            hourly_df[sub] = df_minute[sub].resample("1h").sum()

    total_hours = len(hourly_df)
    valid_hours = int(is_valid_coverage.sum())
    insufficient_hours = total_hours - valid_hours

    stats["total_hourly_periods"] = total_hours
    stats["valid_hourly_observations"] = valid_hours
    stats["insufficient_coverage_hours"] = insufficient_hours
    stats["coverage_retention_pct"] = round(float(valid_hours / total_hours * 100), 2)

    logger.info(
        f"Hourly resampling done: {valid_hours:,} valid hours ({stats['coverage_retention_pct']}%), "
        f"{insufficient_hours:,} insufficient coverage hours."
    )

    # Imputation Strategy for Minor Isolated Gaps:
    # Forward-fill isolated gaps of at most 2 consecutive hours to maintain time series continuity
    # without leaking future observations. Long gap periods (> 2 hours) are NOT filled.
    imputed_count = hourly_df[TARGET_COL].isna().sum()
    hourly_df[TARGET_COL] = hourly_df[TARGET_COL].ffill(limit=2)
    remaining_nans = int(hourly_df[TARGET_COL].isna().sum())
    imputed_filled = int(imputed_count - remaining_nans)

    stats["imputed_isolated_hours"] = imputed_filled
    stats["remaining_missing_hours"] = remaining_nans

    # Drop any remaining unfillable missing hours to ensure clean target values
    initial_clean_len = len(hourly_df)
    hourly_clean = hourly_df.dropna(subset=[TARGET_COL]).copy()
    stats["final_processed_hourly_rows"] = len(hourly_clean)
    stats["dropped_unfillable_hours"] = initial_clean_len - len(hourly_clean)

    # Statistical summary of target
    stats["target_min_kwh"] = round(float(hourly_clean[TARGET_COL].min()), 4)
    stats["target_max_kwh"] = round(float(hourly_clean[TARGET_COL].max()), 4)
    stats["target_mean_kwh"] = round(float(hourly_clean[TARGET_COL].mean()), 4)
    stats["target_std_kwh"] = round(float(hourly_clean[TARGET_COL].std()), 4)

    logger.info(
        f"Final clean hourly series: {len(hourly_clean):,} observations. "
        f"Target kWh mean={stats['target_mean_kwh']}, max={stats['target_max_kwh']}."
    )

    return hourly_clean, stats


def run_preprocessing_pipeline(
    raw_df: Optional[pd.DataFrame] = None,
    save_output: bool = True,
    output_path: Optional[Path] = None,
    report_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Executes the complete preprocessing pipeline from raw measurements to hourly energy DataFrame.

    Args:
        raw_df: Optional preloaded raw DataFrame. If None, loaded via load_raw_data().
        save_output: Whether to persist processed CSV.
        output_path: Custom path for processed CSV.
        report_path: Custom path for preprocessing JSON report.

    Returns:
        Tuple of (processed hourly DataFrame, full preprocessing report dictionary)
    """
    from src.data_loader import load_raw_data

    if raw_df is None:
        raw_df = load_raw_data()

    cleaned_minute, min_stats = clean_minute_data(raw_df)
    hourly_clean, hour_stats = resample_hourly_energy(cleaned_minute)

    full_report = {
        "status": "SUCCESS",
        "dataset_name": "Individual Household Electric Power Consumption",
        "target_variable": TARGET_COL,
        "target_units": "kWh",
        "power_raw_units": "kW",
        "cleaning_statistics": min_stats,
        "resampling_statistics": hour_stats,
    }

    if save_output:
        out_csv = Path(output_path) if output_path else PROCESSED_HOURLY_CSV
        rep_json = Path(report_path) if report_path else PREPROCESSING_REPORT_FILE

        out_csv.parent.mkdir(parents=True, exist_ok=True)
        rep_json.parent.mkdir(parents=True, exist_ok=True)

        hourly_clean.to_csv(out_csv, index=True)
        logger.info(f"Saved processed hourly dataset to: {out_csv.resolve()}")

        with open(rep_json, "w", encoding="utf-8") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Saved preprocessing report to: {rep_json.resolve()}")

    return hourly_clean, full_report
