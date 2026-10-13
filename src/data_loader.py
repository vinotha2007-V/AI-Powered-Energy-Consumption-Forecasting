"""
Data loader module for acquiring and reading the household energy dataset.
Provides automated download, extraction, validation, and safe reading.
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Optional, Tuple
import pandas as pd
import requests

from src.config import (
    UCI_DATASET_URL,
    RAW_DATA_DIR,
    RAW_ZIP_FILE,
    RAW_TXT_FILE,
    DATE_COL,
    TIME_COL,
    RAW_ACTIVE_POWER_COL,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def download_dataset(
    url: str = UCI_DATASET_URL,
    raw_dir: Path = RAW_DATA_DIR,
    force: bool = False,
) -> Path:
    """
    Downloads and extracts the UCI Individual Household Electric Power Consumption dataset.

    Args:
        url: Official download URL for the zip archive.
        raw_dir: Directory where raw files should be saved.
        force: If True, redownload even if file exists.

    Returns:
        Path to the extracted text file (household_power_consumption.txt).

    Raises:
        RuntimeError: If download fails and file is absent.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_txt = raw_dir / "household_power_consumption.txt"
    target_zip = raw_dir / "household_power_consumption.zip"

    if target_txt.exists() and not force:
        file_size_mb = target_txt.stat().st_size / (1024 * 1024)
        logger.info(
            f"Dataset already exists at {target_txt} ({file_size_mb:.2f} MB). Skipping download."
        )
        return target_txt

    logger.info(f"Downloading dataset from: {url}")
    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()

        total_bytes = 0
        with open(target_zip, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    total_bytes += len(chunk)
                    if total_bytes % (5 * 1024 * 1024) == 0:
                        logger.info(f"Downloaded {total_bytes / (1024 * 1024):.1f} MB...")

        logger.info(
            f"Download completed successfully ({total_bytes / (1024 * 1024):.2f} MB). Extracting archive..."
        )

        with zipfile.ZipFile(target_zip, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

        if not target_txt.exists():
            # Check if extracted with a different name or subdirectory
            candidates = list(raw_dir.glob("*.txt"))
            if candidates:
                extracted = candidates[0]
                if extracted.name != "household_power_consumption.txt":
                    extracted.rename(target_txt)

        if target_txt.exists():
            logger.info(
                f"Extraction verified: {target_txt} ({target_txt.stat().st_size / (1024 * 1024):.2f} MB)."
            )
            return target_txt
        else:
            raise FileNotFoundError(f"Expected extracted file at {target_txt} not found.")

    except Exception as e:
        logger.error(f"Failed to automatically download/extract dataset: {e}")
        logger.info(
            "\n[MANUAL DOWNLOAD INSTRUCTION]:\n"
            "1. Visit: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption\n"
            "2. Download 'individual+household+electric+power+consumption.zip'\n"
            f"3. Extract 'household_power_consumption.txt' into: {raw_dir.resolve()}\n"
        )
        raise RuntimeError(
            f"Automated download failed: {e}. Please follow manual download instructions."
        ) from e


def load_raw_data(
    file_path: Optional[Path] = None,
    nrows: Optional[int] = None,
) -> pd.DataFrame:
    """
    Safely loads the raw household electric power consumption dataset.
    The UCI dataset uses ';' delimiter and '?' for missing values.

    Args:
        file_path: Path to household_power_consumption.txt.
        nrows: Optional limit on number of rows to read.

    Returns:
        pd.DataFrame containing the raw measurements.
    """
    target_path = Path(file_path) if file_path else RAW_TXT_FILE

    if not target_path.exists():
        raise FileNotFoundError(
            f"Raw dataset file not found at: {target_path}. "
            f"Please run download_dataset() or place the file in {RAW_DATA_DIR}."
        )

    logger.info(f"Loading raw data from: {target_path} (nrows={nrows})")
    df = pd.read_csv(
        target_path,
        sep=";",
        na_values=["?", "NA", ""],
        low_memory=False,
        nrows=nrows,
    )

    expected_cols = [DATE_COL, TIME_COL, RAW_ACTIVE_POWER_COL]
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Raw data missing required columns: {missing_cols}")

    logger.info(f"Successfully loaded raw dataframe: {df.shape[0]:,} rows, {df.shape[1]} columns.")
    return df


def generate_simulated_fallback_dataset(
    output_path: Optional[Path] = None,
    days: int = 90,
) -> pd.DataFrame:
    """
    Generates a deterministic simulated minute-level energy dataset strictly for
    offline demonstrations or unit testing when internet access is unavailable.

    Notice: This is clearly flagged as SIMULATED synthetic data and must never be
    confused with the real UCI dataset.
    """
    import numpy as np

    logger.warning("Generating clearly labeled SYNTHETIC SIMULATION dataset fallback.")
    dates = pd.date_range("2026-01-01 00:00:00", periods=days * 1440, freq="1min")
    rng = np.random.RandomState(42)

    # Base daily pattern + weekly pattern + random noise
    minute_of_day = dates.hour * 60 + dates.minute
    daily_cycle = 0.8 + 1.2 * np.sin(2 * np.pi * (minute_of_day - 360) / 1440) ** 2
    daily_cycle += 0.5 * np.exp(-((minute_of_day - 1140) ** 2) / (2 * 90**2))  # Evening peak
    is_weekend = (dates.dayofweek >= 5).astype(float) * 0.3
    noise = rng.exponential(scale=0.2, size=len(dates))

    active_power = np.clip(daily_cycle + is_weekend + noise, 0.1, 8.0)

    # Inject a realistic 1% missingness
    mask_missing = rng.rand(len(dates)) < 0.01
    active_power[mask_missing] = np.nan

    df = pd.DataFrame(
        {
            DATE_COL: dates.strftime("%d/%m/%Y"),
            TIME_COL: dates.strftime("%H:%M:%S"),
            RAW_ACTIVE_POWER_COL: active_power,
            "Global_reactive_power": np.clip(active_power * 0.15 + rng.normal(0, 0.05, len(dates)), 0.0, 1.5),
            "Voltage": 230.0 + rng.normal(0, 3.0, len(dates)),
            "Global_intensity": active_power * 4.3,
            "Sub_metering_1": 0.0,
            "Sub_metering_2": 0.0,
            "Sub_metering_3": 0.0,
        }
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_p, sep=";", index=False)
        logger.info(f"Simulated fallback saved to {out_p}")

    return df
