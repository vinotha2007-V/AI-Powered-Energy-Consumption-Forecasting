# Project Architecture & Data Flow

This document details the system design, modular component topology, and end-to-end data pipelines for the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. System Topology Overview

```
+----------------------------------------------------------------------------------------------------+
|                                      DATA ACQUISITION LAYER                                        |
|  - UCI Machine Learning Repository (Individual Household Electric Power Consumption)              |
|  - Automated streaming download & checksum verification -> data/raw/household_power_consumption.txt|
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+--------------------------------------------------+
|                                    PREPROCESSING & INTEGRATION                                     |
|  - Parse timestamps (dd/mm/yyyy HH:MM:SS) & cast numeric metrics                                  |
|  - Check & remove duplicate timestamps                                                             |
|  - Defensible hourly integration: Energy (kWh) = Riemann sum integral of power (kW) over 1 hour   |
|  - Coverage Threshold: Filter hours with < 50 valid minute observations (insufficiently observed) |
|  - Store clean dataset -> data/processed/hourly_energy_consumption.csv                             |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+--------------------------------------------------+
|                                    FEATURE ENGINEERING ENGINE                                      |
|  - Calendar Features: hour, day_of_week, day_of_month, month, is_weekend                          |
|  - Cyclical Trigonometric: hour_sin/cos (24h), dow_sin/cos (7d), month_sin/cos (12m)               |
|  - Multi-Horizon Lags: lag_1 (t-1), lag_2 (t-2), lag_3 (t-3), lag_24 (daily), lag_168 (weekly)   |
|  - Shifted Rolling Statistics: rolling_mean_6h, std_6h, mean_24h, max_24h, min_24h (shifted by 1) |
|  - ZERO DATA LEAKAGE GUARANTEE: strictly historical observations preceding prediction step         |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+--------------------------------------------------+
|                            MODEL TRAINING & BENCHMARKING ENGINE                                    |
|  - Strictly Chronological Split: 70% Train, 15% Validation, 15% Held-Out Test                     |
|  - Baselines: Persistence (t-1), Seasonal-Naive (t-24)                                             |
|  - Candidate ML Regressors: HistGradientBoosting, RandomForest, Ridge Regularized Pipeline         |
|  - Validation Selection: Objective ranking on validation RMSE & MAE                               |
|  - Final Single-Pass Test Evaluation -> outputs/reports/metrics.json                               |
|  - Serialization: Joblib artifact -> models/best_energy_model.joblib                               |
+-------------------------------------------------+--------------------------------------------------+
                                                  |
                  +-------------------------------+-------------------------------+
                  |                                                               |
                  v                                                               v
+-----------------+-------------------------------+             +-----------------+------------------+
|       RECURSIVE MULTI-STEP FORECASTING ENGINE   |             |  VIRTUAL ENERGY PLANNING SIMULATOR |
|  - Predict step t+1 from real past observations |             |  - Configurable Flat & ToU Tariffs |
|  - Append prediction to virtual history buffer  |             |  - Peak demand threshold detection |
|  - Recompute lags & rolling statistics          |             |  - Carbon intensity factor (CO2/kWh|
|  - Predict t+2 ... up to horizon H (24-72 hrs)  |             |  - Demand shaving scenario analysis|
+-----------------+-------------------------------+             +-----------------+------------------+
                  |                                                               |
                  +-------------------------------+-------------------------------+
                                                  |
                                                  v
+-------------------------------------------------+--------------------------------------------------+
|                                    PRESENTATION & SERVING LAYER                                    |
|  1. Streamlit Interactive Dashboard (app.py)                                                       |
|     - Overview KPIs, Diurnal Analytics, Recursive Horizon Slider, Model Evaluation, Tariff Sim    |
|  2. Flask REST Prediction API (api.py)                                                             |
|     - Endpoints: GET /, GET /health, GET /model-info, POST /predict, POST /forecast                |
|  3. Visualizations & Proof-of-Work Assets (outputs/figures/ & images/)                             |
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Component Directory Responsibilities

| Component | File Path | Core Functionality |
|---|---|---|
| **Configuration** | `src/config.py` | Centralized paths, target schema, lag definitions, splits, tariff parameters. |
| **Data Loader** | `src/data_loader.py` | Stream download of UCI archive, unzipping, schema checking, and fallback simulation. |
| **Preprocessing** | `src/preprocessing.py` | Timestamp parsing, missing value handling, Riemann sum kW -> kWh integration, coverage thresholding. |
| **Feature Engineering** | `src/feature_engineering.py` | Calendar signals, cyclical transforms, lags, shifted rolling windows, single-step inference helper. |
| **Training & Selection**| `src/train_model.py` | Chronological splitting, candidate training, validation selection, artifact persistence. |
| **Evaluation** | `src/evaluate_model.py` | Calculation of MAE, RMSE, R², safe MAPE, JSON and Markdown metrics reporting. |
| **Forecasting Engine** | `src/forecasting.py` | History validation, one-step ahead inference, and recursive multi-step forecasting. |
| **Visualization** | `src/visualization.py` | Publication-ready Matplotlib figures for trend, patterns, actual vs predicted, and residuals. |
| **Simulation** | `src/simulation.py` | Virtual building scenario analysis, tariff costs, carbon intensity, and load curtailment. |
| **Dashboard** | `app.py` | Multi-tab Streamlit dashboard with interactive Plotly graphs and scenario toggles. |
| **API** | `api.py` | Production-ready Flask REST service with validation, CORS, and robust error codes. |
| **Main Pipeline** | `main.py` | Unified CLI entry point supporting `--all`, `--download`, `--train`, `--forecast`, etc. |

---

## 3. Data Leakage Prevention Guarantees

In time series machine learning, data leakage can lead to overly optimistic performance that collapses in production. This architecture enforces three foundational guardrails:

1. **Chronological Partitioning:**
   Shuffling is strictly disallowed. The dataset is segmented into:
   - **Training Set (70%):** Earliest sequential period (2006 to 2009).
   - **Validation Set (15%):** Intermediate period used strictly for candidate model comparison and hyperparameter decisions.
   - **Test Set (15%):** Held-out latest period (2010), evaluated only once after model selection is finalized.

2. **Shifted Rolling Windows:**
   Every rolling window feature (`rolling_mean_6h`, `rolling_mean_24h`, etc.) is computed by first applying `.shift(1)` to the target series:
   $$\text{RollingFeature}_t = f(\{y_{t-W}, y_{t-W+1}, \dots, y_{t-1}\})$$
   $y_t$ is strictly excluded from the rolling calculation for step $t$.

3. **Autonomous Multi-Step Recursion:**
   When generating future multi-step forecasts ($t+1 \dots t+24$), the model does not access future actual values. Instead, step $t+1$'s prediction is appended to a virtual history buffer, from which lags and rolling features for step $t+2$ are recalculated.
