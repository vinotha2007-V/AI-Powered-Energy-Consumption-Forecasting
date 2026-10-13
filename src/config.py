"""
Configuration module for the AI-Powered Energy Consumption Forecasting System.
Defines paths, hyperparameters, data schemas, feature lists, and simulation parameters.
"""

from pathlib import Path

# Base project directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Outputs & Artifacts Paths
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
IMAGES_DIR = PROJECT_ROOT / "images"

# Remote Dataset URLs & Filenames
UCI_DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/235/"
    "individual+household+electric+power+consumption.zip"
)
RAW_ZIP_FILE = RAW_DATA_DIR / "household_power_consumption.zip"
RAW_TXT_FILE = RAW_DATA_DIR / "household_power_consumption.txt"
PROCESSED_HOURLY_CSV = PROCESSED_DATA_DIR / "hourly_energy_consumption.csv"
FEATURED_DATA_CSV = PROCESSED_DATA_DIR / "energy_features.csv"

# Preprocessing & Target Settings
DATE_COL = "Date"
TIME_COL = "Time"
TIMESTAMP_COL = "timestamp"
RAW_ACTIVE_POWER_COL = "Global_active_power"  # Measured in kW (minute averages)
TARGET_COL = "energy_kwh"                     # Hourly energy consumption in kWh
TARGET_UNITS = "kWh"
POWER_UNITS = "kW"

# Data Quality Thresholds
# Minimum number of valid minute readings required within an hour (out of 60)
# to consider the hourly measurement sufficiently observed.
MIN_MINUTES_PER_HOUR = 50
MAX_MINUTES_PER_HOUR = 60

# Train / Validation / Test Chronological Splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Feature Engineering Definitions
LAG_HOURS = [1, 2, 3, 24, 168]  # t-1, t-2, t-3, t-24, t-168 (1 week)
ROLLING_WINDOWS = [6, 24]        # 6-hour and 24-hour rolling windows

CALENDAR_FEATURES = [
    "hour",
    "day_of_week",
    "day_of_month",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "month_sin",
    "month_cos",
]

LAG_FEATURES = [f"lag_{lag}" for lag in LAG_HOURS]

ROLLING_FEATURES = [
    "rolling_mean_6h",
    "rolling_std_6h",
    "rolling_mean_24h",
    "rolling_max_24h",
    "rolling_min_24h",
]

ALL_MODEL_FEATURES = CALENDAR_FEATURES + LAG_FEATURES + ROLLING_FEATURES

# Model Artifact Filenames
BEST_MODEL_FILE = MODELS_DIR / "best_energy_model.joblib"
MODEL_METADATA_FILE = MODELS_DIR / "model_metadata.json"
MODEL_CONFIG_FILE = MODELS_DIR / "model_config.json"
METRICS_JSON_FILE = REPORTS_DIR / "metrics.json"
EVALUATION_REPORT_FILE = REPORTS_DIR / "model_evaluation.md"
PREPROCESSING_REPORT_FILE = REPORTS_DIR / "preprocessing_report.json"
SIMULATION_REPORT_FILE = REPORTS_DIR / "simulation_report.json"

# Energy Planning & Tariff Defaults
DEFAULT_FLAT_TARIFF_PER_KWH = 0.18        # $ / kWh flat rate
DEFAULT_PEAK_TARIFF_PER_KWH = 0.28        # $ / kWh during peak hours (17:00 - 21:00)
DEFAULT_OFF_PEAK_TARIFF_PER_KWH = 0.12    # $ / kWh during off-peak hours
PEAK_HOURS = [17, 18, 19, 20, 21]         # 5 PM to 9 PM
DEFAULT_CARBON_INTENSITY_KG_PER_KWH = 0.40  # kg CO2 / kWh average electric grid intensity
DEFAULT_PEAK_PERCENTILE = 85.0             # Percentile threshold to mark peak load hours
DEFAULT_FORECAST_HORIZON = 24              # Hours
