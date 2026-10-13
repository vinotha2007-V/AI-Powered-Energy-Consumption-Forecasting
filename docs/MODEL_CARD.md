# Model Card: AI-Powered Hourly Energy Consumption Forecaster

## 1. Model Details
- **Model Name:** `HistGradientBoostingRegressor` (Gradient Boosted Decision Trees)
- **Library:** `scikit-learn` (v1.9.1)
- **Model Type:** Supervised Regression / Tree-Based Ensemble
- **Target Variable:** `energy_kwh` (Hourly electricity consumption in kilowatt-hours)
- **Artifact File:** `models/best_energy_model.joblib`
- **Configuration File:** `models/model_config.json`
- **Metadata File:** `models/model_metadata.json`

---

## 2. Intended Use & Target Audience
- **Primary Use Case:** Forecasting next-hour and recursive 24-hour electricity demand for single-household or commercial building energy monitoring.
- **Secondary Use Cases:** Operational load awareness, time-of-use tariff cost budgeting, and virtual demand response scenario analysis.
- **Out-of-Scope Uses:**
  - Automated autonomous grid balancing or substation switching without human oversight.
  - Life-critical medical facility power backup decisions.
  - Aggregated regional transmission grid management (requires multi-node regional telemetry).

---

## 3. Training & Validation Setup
- **Splitting Strategy:** Chronological temporal partitioning (70% Train, 15% Validation, 15% Test).
- **Training Samples:** 23,802 hourly records (December 23, 2006 to September 15, 2009).
- **Validation Samples:** 5,101 hourly records (September 15, 2009 to April 19, 2010).
- **Held-Out Test Samples:** 5,101 hourly records (April 19, 2010 to November 26, 2010).
- **Hyperparameters:**
  - `max_iter`: 120
  - `max_depth`: 8
  - `learning_rate`: 0.08
  - `min_samples_leaf`: 20
  - `random_state`: 42

---

## 4. Empirical Evaluation Metrics

### Validation Set Model Benchmarking
All candidates were trained on the identical training set and evaluated on the chronological validation set:

| Candidate Model | Val MAE (kWh) | Val RMSE (kWh) | Val R² Score | Selection Status |
|---|---|---|---|---|
| **HistGradientBoosting** | **0.3682** | **0.5353** | **0.6393** | **Selected (Winner)** |
| RandomForestRegressor | 0.3715 | 0.5414 | 0.6310 | Candidate Runner-Up |
| Ridge Regression (StandardScaler) | 0.4234 | 0.5908 | 0.5606 | Linear Comparison |
| Persistence Baseline ($t-1$) | 0.4670 | 0.6950 | 0.3919 | Baseline |
| Seasonal Naive ($t-24$) | 0.6657 | 0.9584 | -0.1563 | Baseline |

### Final Held-Out Test Set Performance (Single Evaluation Pass)
Evaluated only once on the unseen 2010 test period:

| Metric | Selected ML Model | Persistence Baseline | Seasonal Naive | Relative Improvement |
|---|---|---|---|---|
| **MAE** | **0.3213 kWh** | 0.3859 kWh | 0.5109 kWh | **+16.7% error reduction** |
| **RMSE** | **0.4636 kWh** | 0.5843 kWh | 0.7586 kWh | **+20.6% error reduction** |
| **R² Score** | **0.5728** | 0.3215 | -0.1437 | **+78.2% explained variance** |
| **Safe MAPE**| **25.2%** | 31.4% | 42.1% | Significant stability gain |

---

## 5. Feature Importance Analysis
Top predictive signals identified by permutation importance:
1. `lag_1`: Immediate persistence ($y_{t-1}$) — strongest short-term signal.
2. `rolling_mean_6h`: Recent consumption momentum.
3. `lag_24`: Diurnal periodicity (same hour previous day).
4. `hour_sin` / `hour_cos`: Time-of-day cyclical positioning.
5. `lag_168`: Weekly day-of-week periodicity.
6. `rolling_mean_24h`: Daily baseline intensity.

---

## 6. Caveats & Limitations
1. **Household Volatility:** Residential electrical usage exhibits stochastic consumer spikes (e.g., turning on an oven or washing machine) that cannot be perfectly predicted by calendar signals alone.
2. **Recursive Error Accumulation:** Beyond 24 hours, recursive predictions compound small estimation errors into downstream lags.
3. **Exogenous Factors:** The dataset does not include local ambient temperature or solar irradiance measurements, which could further refine seasonal HVAC load estimation.
