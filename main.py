"""
Main execution pipeline for the AI-Powered Energy Consumption Forecasting System.
Provides a unified command-line interface to execute the full pipeline or individual phases:
Acquisition -> Preprocessing -> Feature Engineering -> Training -> Evaluation ->
Forecasting -> Visualization -> Simulation.
"""

import argparse
import logging
import sys
from pathlib import Path
import pandas as pd

from src.config import (
    RAW_TXT_FILE,
    PROCESSED_HOURLY_CSV,
    FEATURED_DATA_CSV,
    BEST_MODEL_FILE,
    PREDICTIONS_DIR,
    DEFAULT_FORECAST_HORIZON,
)
from src.data_loader import download_dataset, load_raw_data
from src.preprocessing import run_preprocessing_pipeline
from src.feature_engineering import build_full_feature_matrix
from src.train_model import train_and_compare_models
from src.forecasting import ForecastingEngine
from src.visualization import (
    plot_dataset_overview,
    plot_hourly_usage_patterns,
    plot_actual_vs_predicted,
    plot_forecast_next_24_hours,
    plot_model_comparison,
    plot_residual_analysis,
    plot_feature_importance,
    copy_figures_to_images,
)
from src.simulation import EnergyPlanningSimulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def stage_download() -> Path:
    logger.info("=== STAGE 1: Dataset Acquisition ===")
    txt_path = download_dataset()
    logger.info(f"Dataset ready at {txt_path}")
    return txt_path


def stage_preprocess() -> pd.DataFrame:
    logger.info("=== STAGE 2: Data Preprocessing & Hourly Resampling ===")
    df_raw = load_raw_data()
    hourly_df, report = run_preprocessing_pipeline(df_raw, save_output=True)
    logger.info(
        f"Preprocessing completed. Processed rows: {len(hourly_df):,}. "
        f"Target mean: {report['resampling_statistics']['target_mean_kwh']} kWh."
    )
    return hourly_df


def stage_features(df_hourly: pd.DataFrame = None) -> pd.DataFrame:
    logger.info("=== STAGE 3: Feature Engineering ===")
    if df_hourly is None:
        if not PROCESSED_HOURLY_CSV.exists():
            logger.info("Processed hourly dataset not found. Running preprocessing first...")
            df_hourly = stage_preprocess()
        else:
            df_hourly = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)

    featured_df = build_full_feature_matrix(df_hourly, save_output=True)
    logger.info(f"Feature engineering completed. Matrix shape: {featured_df.shape}")
    return featured_df


def stage_train(df_features: pd.DataFrame = None) -> dict:
    logger.info("=== STAGE 4: Model Training, Benchmarking & Selection ===")
    if df_features is None:
        if not FEATURED_DATA_CSV.exists():
            df_features = stage_features()
        else:
            df_features = pd.read_csv(FEATURED_DATA_CSV, index_col=0, parse_dates=True)

    summary = train_and_compare_models(df_features)
    selected = summary["selected_model"]["model_name"]
    test_mae = summary["selected_model"]["test_metrics"]["metrics"]["mae"]
    test_rmse = summary["selected_model"]["test_metrics"]["metrics"]["rmse"]
    test_r2 = summary["selected_model"]["test_metrics"]["metrics"]["r2"]
    logger.info(
        f"Selected Model: '{selected}' | Test MAE: {test_mae} kWh | "
        f"Test RMSE: {test_rmse} kWh | Test R2: {test_r2}"
    )
    return summary


def stage_forecast(horizon: int = DEFAULT_FORECAST_HORIZON) -> pd.DataFrame:
    logger.info(f"=== STAGE 5: Recursive Multi-Step Forecasting ({horizon} Hours) ===")
    if not PROCESSED_HOURLY_CSV.exists():
        stage_preprocess()

    hourly_df = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)
    engine = ForecastingEngine()

    if not engine.is_ready():
        logger.info("Model artifact not found. Training model first...")
        stage_train()
        engine = ForecastingEngine()

    target_series = hourly_df["energy_kwh"]
    forecast_df = engine.predict_horizon(target_series, horizon_hours=horizon)

    out_csv = PREDICTIONS_DIR / f"forecast_next_{horizon}h.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(out_csv)
    logger.info(f"Generated {horizon}-hour recursive forecast. Saved to: {out_csv.resolve()}")
    return forecast_df


def stage_plots() -> None:
    logger.info("=== STAGE 6: Visualizations Generation ===")
    if not PROCESSED_HOURLY_CSV.exists():
        stage_preprocess()
    hourly_df = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)

    test_pred_file = PREDICTIONS_DIR / "test_predictions.csv"
    if not test_pred_file.exists():
        stage_train()
    test_preds = pd.read_csv(test_pred_file, index_col=0, parse_dates=True)

    forecast_24_file = PREDICTIONS_DIR / f"forecast_next_{DEFAULT_FORECAST_HORIZON}h.csv"
    if not forecast_24_file.exists():
        forecast_df = stage_forecast(DEFAULT_FORECAST_HORIZON)
    else:
        forecast_df = pd.read_csv(forecast_24_file, index_col=0, parse_dates=True)

    # Load metrics & metadata
    import json
    from src.config import METRICS_JSON_FILE, MODEL_METADATA_FILE
    with open(METRICS_JSON_FILE, "r", encoding="utf-8") as f:
        metrics_summary = json.load(f)
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # 1. Dataset Overview
    plot_dataset_overview(hourly_df)
    # 2. Hourly Usage Patterns
    plot_hourly_usage_patterns(hourly_df)
    # 3. Actual vs Predicted
    plot_actual_vs_predicted(test_preds)
    # 4. Forecast Next 24 Hours
    plot_forecast_next_24_hours(hourly_df["energy_kwh"], forecast_df)
    # 5. Model Comparison
    plot_model_comparison(metrics_summary)
    # 6. Residual Analysis
    plot_residual_analysis(test_preds)
    # 7. Feature Importance
    plot_feature_importance(metadata.get("feature_importances", {}))

    # Copy to images/
    copy_figures_to_images()
    logger.info("All 7 figures generated and synced to images/ directory.")


def stage_simulate() -> dict:
    logger.info("=== STAGE 7: Virtual Energy Planning Simulation ===")
    forecast_24_file = PREDICTIONS_DIR / f"forecast_next_{DEFAULT_FORECAST_HORIZON}h.csv"
    if not forecast_24_file.exists():
        forecast_df = stage_forecast(DEFAULT_FORECAST_HORIZON)
    else:
        forecast_df = pd.read_csv(forecast_24_file, index_col=0, parse_dates=True)

    hourly_df = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)
    simulator = EnergyPlanningSimulator()
    sim_report = simulator.run_simulation(
        forecast_df=forecast_df,
        historical_baseline_series=hourly_df["energy_kwh"],
    )
    simulator.save_simulation_report(sim_report)

    costs = sim_report["cost_and_emissions_estimates"]
    consump = sim_report["consumption_summary"]
    logger.info(
        f"Simulation completed. Total 24h Forecast: {consump['total_predicted_kwh']} kWh | "
        f"Flat Cost: ${costs['estimated_cost_flat_tariff_usd']} | "
        f"ToU Cost: ${costs['estimated_cost_tou_tariff_usd']} | "
        f"Est. CO2: {costs['estimated_carbon_emissions_kg_co2']} kg"
    )
    return sim_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AI-Powered Energy Consumption Forecasting System - Execution Pipeline"
    )
    parser.add_argument("--all", action="store_true", help="Execute complete end-to-end pipeline")
    parser.add_argument("--download", action="store_true", help="Download raw dataset")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess minute measurements to hourly energy")
    parser.add_argument("--features", action="store_true", help="Build feature matrix")
    parser.add_argument("--train", action="store_true", help="Train and select forecasting models")
    parser.add_argument("--forecast", action="store_true", help="Generate 24-hour recursive forecast")
    parser.add_argument("--horizon", type=int, default=DEFAULT_FORECAST_HORIZON, help="Forecast horizon in hours (default 24)")
    parser.add_argument("--plots", action="store_true", help="Generate all evaluation and forecasting plots")
    parser.add_argument("--simulate", action="store_true", help="Run virtual energy planning simulation")

    args = parser.parse_args()

    # If no flags passed, display help or default to --all
    if not any([args.all, args.download, args.preprocess, args.features, args.train, args.forecast, args.plots, args.simulate]):
        logger.info("No specific stage flag provided. Running full end-to-end pipeline (--all)...")
        args.all = True

    if args.all:
        logger.info(">>> STARTING FULL END-TO-END PIPELINE <<<")
        stage_download()
        df_hourly = stage_preprocess()
        df_features = stage_features(df_hourly)
        stage_train(df_features)
        stage_forecast(args.horizon)
        stage_plots()
        stage_simulate()
        logger.info(">>> FULL END-TO-END PIPELINE COMPLETED SUCCESSFULLY! <<<")
        return

    if args.download:
        stage_download()
    if args.preprocess:
        stage_preprocess()
    if args.features:
        stage_features()
    if args.train:
        stage_train()
    if args.forecast:
        stage_forecast(args.horizon)
    if args.plots:
        stage_plots()
    if args.simulate:
        stage_simulate()


if __name__ == "__main__":
    main()
