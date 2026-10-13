"""
Visualization module for generating publication-quality figures:
- dataset_overview.png
- hourly_usage_patterns.png
- actual_vs_predicted.png
- forecast_next_24_hours.png
- model_comparison.png
- residual_analysis.png
- feature_importance.png
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    FIGURES_DIR,
    IMAGES_DIR,
    METRICS_JSON_FILE,
    PREDICTIONS_DIR,
    TARGET_COL,
    TARGET_UNITS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Set standard clean aesthetic styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 15,
})


def plot_dataset_overview(
    df_hourly: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generates dataset_overview.png showing long-term trends and monthly averages."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # 1. Weekly resampled trend
    weekly_trend = df_hourly[TARGET_COL].resample("1W").mean()
    axes[0].plot(weekly_trend.index, weekly_trend.values, color="#1f77b4", linewidth=1.8, label="Weekly Mean Consumption")
    axes[0].fill_between(weekly_trend.index, weekly_trend.values, color="#1f77b4", alpha=0.15)
    axes[0].set_title("Long-Term Household Electricity Consumption Trend (Weekly Averages)", fontweight="bold")
    axes[0].set_ylabel(f"Consumption ({TARGET_UNITS})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Monthly Boxplot distribution
    monthly_data = [group[TARGET_COL].dropna().values for _, group in df_hourly.groupby(df_hourly.index.month)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    axes[1].boxplot(monthly_data, tick_labels=month_names[:len(monthly_data)], patch_artist=True,
                    boxprops=dict(facecolor="#90caf9", color="#1565c0"),
                    medianprops=dict(color="#d32f2f", linewidth=1.5),
                    flierprops=dict(marker=".", markersize=3, alpha=0.3))
    axes[1].set_title("Seasonal Distribution by Month (Hourly Observations)", fontweight="bold")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel(f"Hourly Consumption ({TARGET_UNITS})")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    file_path = output_dir / "dataset_overview.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_hourly_usage_patterns(
    df_hourly: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generates hourly_usage_patterns.png showing diurnal cycles and weekday vs weekend comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Calculate patterns
    df = df_hourly.copy()
    df["hour"] = df.index.hour
    df["is_weekend"] = df.index.dayofweek >= 5

    # 1. Overall Diurnal Profile
    hourly_stats = df.groupby("hour")[TARGET_COL].agg(["mean", "std", "median"])
    hours = hourly_stats.index
    axes[0].plot(hours, hourly_stats["mean"], color="#0288d1", linewidth=2.5, marker="o", label="Mean Consumption")
    axes[0].fill_between(
        hours,
        np.maximum(0, hourly_stats["mean"] - hourly_stats["std"] * 0.5),
        hourly_stats["mean"] + hourly_stats["std"] * 0.5,
        color="#0288d1",
        alpha=0.2,
        label="±0.5 Std Dev Band"
    )
    axes[0].set_title("Diurnal Profile (Mean Hourly Consumption)", fontweight="bold")
    axes[0].set_xlabel("Hour of Day (0–23)")
    axes[0].set_ylabel(f"Energy Consumption ({TARGET_UNITS})")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].legend(loc="upper left")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # 2. Weekday vs Weekend Comparison
    weekday_mean = df[~df["is_weekend"]].groupby("hour")[TARGET_COL].mean()
    weekend_mean = df[df["is_weekend"]].groupby("hour")[TARGET_COL].mean()

    axes[1].plot(weekday_mean.index, weekday_mean.values, color="#388e3c", linewidth=2.2, marker="s", label="Weekdays (Mon-Fri)")
    axes[1].plot(weekend_mean.index, weekend_mean.values, color="#e64a19", linewidth=2.2, marker="^", label="Weekends (Sat-Sun)")
    axes[1].set_title("Diurnal Pattern: Weekday vs Weekend", fontweight="bold")
    axes[1].set_xlabel("Hour of Day (0–23)")
    axes[1].set_ylabel(f"Mean Energy Consumption ({TARGET_UNITS})")
    axes[1].set_xticks(range(0, 24, 2))
    axes[1].legend(loc="upper left")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    file_path = output_dir / "hourly_usage_patterns.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_actual_vs_predicted(
    predictions_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    sample_hours: int = 168,  # One full week snapshot
) -> Path:
    """Generates actual_vs_predicted.png comparing ground truth and predictions on test data."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)

    # Take a 1-week representative sample slice from the test set for visual clarity
    sample = predictions_df.iloc[-sample_hours:].copy()

    # Upper panel: Actual vs Predicted
    axes[0].plot(sample.index, sample["actual_kwh"], label="Actual Consumption", color="#212121", linewidth=1.8, alpha=0.85)
    axes[0].plot(sample.index, sample["predicted_kwh"], label="Predicted Consumption (ML)", color="#e53935", linewidth=2.0, linestyle="--")
    axes[0].set_title(f"Test Period Evaluation: Actual vs. Predicted Energy Consumption (1-Week Window: {sample_hours} Hours)", fontweight="bold")
    axes[0].set_ylabel(f"Consumption ({TARGET_UNITS})")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Lower panel: Residuals
    residuals = sample["actual_kwh"] - sample["predicted_kwh"]
    axes[1].bar(sample.index, residuals, color=np.where(residuals >= 0, "#1976d2", "#d32f2f"), alpha=0.6, width=0.03)
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_title("Hourly Prediction Error (Actual - Predicted)", fontweight="bold", fontsize=11)
    axes[1].set_xlabel("Timestamp")
    axes[1].set_ylabel(f"Residual ({TARGET_UNITS})")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    file_path = output_dir / "actual_vs_predicted.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_forecast_next_24_hours(
    history_series: pd.Series,
    forecast_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
    history_window_hours: int = 48,
) -> Path:
    """Generates forecast_next_24_hours.png showing recent historical context and recursive forecast."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(14, 6))

    hist_slice = history_series.iloc[-history_window_hours:]

    plt.plot(hist_slice.index, hist_slice.values, color="#1976d2", linewidth=2.0, marker="o", markersize=4, label="Historical Consumption (Observed)")
    plt.plot(forecast_df.index, forecast_df["predicted_energy_kwh"], color="#f57c00", linewidth=2.5, marker="s", markersize=4, linestyle="--", label="Next 24-Hour Forecast (Recursive)")

    # Connect the last history point to the first forecast point
    plt.plot([hist_slice.index[-1], forecast_df.index[0]], [hist_slice.values[-1], forecast_df["predicted_energy_kwh"].iloc[0]], color="#f57c00", linestyle=":", linewidth=2)

    # Shaded forecast horizon region
    plt.axvspan(forecast_df.index[0], forecast_df.index[-1], color="#ffe082", alpha=0.25, label="24-Hour Forecast Window")

    plt.title("Operational 24-Hour Electricity Consumption Forecast", fontweight="bold")
    plt.xlabel("Timestamp")
    plt.ylabel(f"Energy Consumption ({TARGET_UNITS})")
    plt.legend(loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.6)

    file_path = output_dir / "forecast_next_24_hours.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_model_comparison(
    metrics_summary: Dict[str, Any],
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generates model_comparison.png bar chart comparing baselines and candidate models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    val_models = metrics_summary.get("validation_comparisons", {})
    if not val_models:
        logger.warning("No validation models available to plot.")
        return output_dir / "model_comparison.png"

    model_names = list(val_models.keys())
    mae_vals = [val_models[m]["metrics"]["mae"] for m in model_names]
    rmse_vals = [val_models[m]["metrics"]["rmse"] for m in model_names]
    r2_vals = [val_models[m]["metrics"]["r2"] for m in model_names]

    x = np.arange(len(model_names))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(13, 6))

    # Error metrics (MAE & RMSE) on primary axis
    bar1 = ax1.bar(x - width/2, mae_vals, width, label="MAE (kWh)", color="#26a69a")
    bar2 = ax1.bar(x + width/2, rmse_vals, width, label="RMSE (kWh)", color="#ef5350")
    ax1.set_ylabel(f"Error ({TARGET_UNITS})", fontsize=12)
    ax1.set_title("Benchmarking Candidate Models & Baselines (Validation Set)", fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15, ha="right", fontweight="bold")

    # Value labels on bars
    for rect in bar1:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for rect in bar2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3),
                     textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    file_path = output_dir / "model_comparison.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_residual_analysis(
    predictions_df: pd.DataFrame,
    output_dir: Path = FIGURES_DIR,
) -> Path:
    """Generates residual_analysis.png showing residual histogram and residual vs predicted."""
    output_dir.mkdir(parents=True, exist_ok=True)
    residuals = predictions_df["actual_kwh"] - predictions_df["predicted_kwh"]
    predicted = predictions_df["predicted_kwh"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Residual Distribution
    axes[0].hist(residuals, bins=60, color="#5c6bc0", edgecolor="black", alpha=0.7, density=True)
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.5, label="Zero Error Line")
    axes[0].axvline(residuals.mean(), color="black", linestyle=":", linewidth=1.5, label=f"Mean Error: {residuals.mean():.3f}")
    axes[0].set_title("Test Residuals Distribution", fontweight="bold")
    axes[0].set_xlabel(f"Error = Actual - Predicted ({TARGET_UNITS})")
    axes[0].set_ylabel("Density")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Residuals vs Predicted
    axes[1].scatter(predicted, residuals, alpha=0.25, color="#26a69a", s=15)
    axes[1].axhline(0, color="red", linestyle="--", linewidth=1.5)
    axes[1].set_title("Residuals vs. Predicted Values", fontweight="bold")
    axes[1].set_xlabel(f"Predicted Consumption ({TARGET_UNITS})")
    axes[1].set_ylabel(f"Residual ({TARGET_UNITS})")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    file_path = output_dir / "residual_analysis.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def plot_feature_importance(
    feature_importances: Dict[str, float],
    output_dir: Path = FIGURES_DIR,
    top_n: int = 15,
) -> Path:
    """Generates feature_importance.png ranking the top predictive features."""
    output_dir.mkdir(parents=True, exist_ok=True)
    if not feature_importances:
        logger.warning("Feature importances dictionary is empty.")
        return output_dir / "feature_importance.png"

    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    names = [f[0] for f in sorted_feats][::-1]
    scores = [f[1] for f in sorted_feats][::-1]

    plt.figure(figsize=(10, 7))
    bars = plt.barh(names, scores, color="#42a5f5", edgecolor="#1565c0", height=0.6)
    plt.title(f"Top {len(names)} Most Influential Forecasting Features", fontweight="bold")
    plt.xlabel("Importance Score")
    plt.grid(True, linestyle="--", alpha=0.5)

    for bar in bars:
        w = bar.get_width()
        plt.annotate(f"{w:.4f}", xy=(w, bar.get_y() + bar.get_height() / 2), xytext=(5, 0),
                     textcoords="offset points", ha="left", va="center", fontsize=9)

    plt.tight_layout()
    file_path = output_dir / "feature_importance.png"
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(f"Generated: {file_path}")
    return file_path


def copy_figures_to_images(
    figures_dir: Path = FIGURES_DIR,
    images_dir: Path = IMAGES_DIR,
) -> None:
    """Copies all generated figures from outputs/figures/ to images/ for GitHub presentation."""
    images_dir.mkdir(parents=True, exist_ok=True)
    for fig_file in figures_dir.glob("*.png"):
        shutil.copy2(fig_file, images_dir / fig_file.name)
    logger.info(f"Copied figures from {figures_dir} to {images_dir}")
