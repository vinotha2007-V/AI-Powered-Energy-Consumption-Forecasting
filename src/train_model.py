"""
Model training and selection module.
Executes chronological train/validation/test splits, trains candidate regressors
and baselines, performs objective model selection, and persists the winning model artifact.
"""

import json
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    TARGET_COL,
    ALL_MODEL_FEATURES,
    TRAIN_RATIO,
    VAL_RATIO,
    TEST_RATIO,
    RANDOM_SEED,
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    MODEL_CONFIG_FILE,
    PREDICTIONS_DIR,
    TARGET_UNITS,
)
from src.evaluate_model import evaluate_predictions, save_metrics_report

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits time series data chronologically into train, validation, and test sets.
    NEVER shuffles the data to maintain temporal causal order.
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    logger.info(
        f"Chronological split completed: "
        f"Train={len(train_df):,} rows ({train_df.index.min()} to {train_df.index.max()}), "
        f"Val={len(val_df):,} rows ({val_df.index.min()} to {val_df.index.max()}), "
        f"Test={len(test_df):,} rows ({test_df.index.min()} to {test_df.index.max()})."
    )
    return train_df, val_df, test_df


def train_and_compare_models(
    df_features: pd.DataFrame,
    target_col: str = TARGET_COL,
    feature_cols: list = ALL_MODEL_FEATURES,
) -> Dict[str, Any]:
    """
    Trains candidate forecasting models and baselines, compares validation performance,
    evaluates the selected model on the held-out test set, and saves model artifacts.

    Candidates:
    1. Persistence Baseline: y_t = y_{t-1}
    2. Seasonal-Naive Baseline: y_t = y_{t-24}
    3. HistGradientBoostingRegressor (fast, robust gradient boosting tree)
    4. RandomForestRegressor (ensemble bagging trees)
    5. Ridge Regression Pipeline (regularized linear model with StandardScaler)
    """
    logger.info("Initiating model training and benchmarking pipeline...")

    # Chronological Split
    train_df, val_df, test_df = chronological_split(df_features)

    X_train = train_df[feature_cols]
    y_train = train_df[target_col]

    X_val = val_df[feature_cols]
    y_val = val_df[target_col]

    X_test = test_df[feature_cols]
    y_test = test_df[target_col]

    # Baseline predictions
    # Persistence baseline: predict previous hour consumption (lag_1)
    val_pred_persistence = X_val["lag_1"].values
    test_pred_persistence = X_test["lag_1"].values

    # Seasonal naive: predict previous day same hour (lag_24)
    val_pred_seasonal = X_val["lag_24"].values
    test_pred_seasonal = X_test["lag_24"].values

    validation_results = {
        "Persistence (t-1)": evaluate_predictions(y_val, val_pred_persistence, "Persistence Baseline", "validation"),
        "Seasonal Naive (t-24)": evaluate_predictions(y_val, val_pred_seasonal, "Seasonal-Naive Baseline", "validation"),
    }

    test_baselines = {
        "Persistence (t-1)": evaluate_predictions(y_test, test_pred_persistence, "Persistence Baseline", "test"),
        "Seasonal Naive (t-24)": evaluate_predictions(y_test, test_pred_seasonal, "Seasonal-Naive Baseline", "test"),
    }

    # Define Candidate ML Models
    candidates = {
        "HistGradientBoosting": HistGradientBoostingRegressor(
            max_iter=120,
            max_depth=8,
            learning_rate=0.08,
            min_samples_leaf=20,
            random_state=RANDOM_SEED,
        ),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            max_depth=14,
            min_samples_split=6,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "RidgeRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", Ridge(alpha=10.0)),
        ]),
    }

    fitted_models = {}

    for name, model in candidates.items():
        logger.info(f"Training candidate model: {name}...")
        model.fit(X_train, y_train)
        fitted_models[name] = model

        val_preds = model.predict(X_val)
        val_eval = evaluate_predictions(y_val, val_preds, name, "validation")
        validation_results[name] = val_eval

    # Objective Model Selection based on lowest Validation RMSE
    ml_candidates = ["HistGradientBoosting", "RandomForest", "RidgeRegression"]
    best_candidate_name = min(
        ml_candidates,
        key=lambda m: validation_results[m]["metrics"]["rmse"],
    )
    best_model = fitted_models[best_candidate_name]

    logger.info(
        f"Model Selection Outcome: '{best_candidate_name}' selected based on validation RMSE: "
        f"{validation_results[best_candidate_name]['metrics']['rmse']} {TARGET_UNITS}."
    )

    # FINAL EVALUATION ON HELD-OUT TEST SET
    test_preds = best_model.predict(X_test)
    test_eval = evaluate_predictions(y_test, test_preds, best_candidate_name, "test")

    # Save Test Predictions for Visualizations
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    test_pred_df = pd.DataFrame(
        {
            "actual_kwh": y_test.values,
            "predicted_kwh": test_preds,
            "residual_kwh": y_test.values - test_preds,
            "persistence_kwh": test_pred_persistence,
            "seasonal_naive_kwh": test_pred_seasonal,
        },
        index=test_df.index,
    )
    test_pred_df.to_csv(PREDICTIONS_DIR / "test_predictions.csv", index=True)

    # Persist Winning Model Artifact using Joblib
    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, BEST_MODEL_FILE)
    logger.info(f"Saved winning model artifact to: {BEST_MODEL_FILE.resolve()}")

    # Extract Feature Importances if available
    feature_importances = {}
    if hasattr(best_model, "feature_importances_"):
        for col, imp in zip(feature_cols, best_model.feature_importances_):
            feature_importances[col] = round(float(imp), 5)
    elif best_candidate_name == "HistGradientBoosting":
        # Permutation importance or inspect trees
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(best_model, X_val, y_val, n_repeats=5, random_state=RANDOM_SEED)
        for col, imp in zip(feature_cols, perm.importances_mean):
            feature_importances[col] = round(float(imp), 5)
    elif hasattr(best_model.named_steps.get("regressor", None), "coef_"):
        coefs = best_model.named_steps["regressor"].coef_
        for col, c in zip(feature_cols, coefs):
            feature_importances[col] = round(float(abs(c)), 5)

    # Prepare Metadata & Config
    timestamp_str = datetime.datetime.now().isoformat()
    model_metadata = {
        "model_name": best_candidate_name,
        "timestamp": timestamp_str,
        "target_col": target_col,
        "target_units": TARGET_UNITS,
        "features": feature_cols,
        "feature_count": len(feature_cols),
        "split_summary": {
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_period": [str(train_df.index.min()), str(train_df.index.max())],
            "val_period": [str(val_df.index.min()), str(val_df.index.max())],
            "test_period": [str(test_df.index.min()), str(test_df.index.max())],
        },
        "validation_metrics": validation_results[best_candidate_name]["metrics"],
        "test_metrics": test_eval["metrics"],
        "feature_importances": feature_importances,
    }

    with open(MODEL_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(model_metadata, f, indent=2)

    with open(MODEL_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model_name": best_candidate_name,
                "model_parameters": str(best_model.get_params()),
                "feature_names": feature_cols,
            },
            f,
            indent=2,
        )

    # Save Complete Evaluation Report
    full_evaluation_summary = {
        "timestamp": timestamp_str,
        "selected_model": {
            "model_name": best_candidate_name,
            "selection_rationale": (
                f"Selected automatically as the top performer on chronological validation set "
                f"(lowest validation RMSE={validation_results[best_candidate_name]['metrics']['rmse']} kWh)."
            ),
            "validation_metrics": validation_results[best_candidate_name],
            "test_metrics": test_eval,
        },
        "validation_comparisons": validation_results,
        "baselines": test_baselines,
    }

    save_metrics_report(full_evaluation_summary)

    logger.info("Training, validation comparison, test evaluation, and artifact persistence completed.")
    return full_evaluation_summary
