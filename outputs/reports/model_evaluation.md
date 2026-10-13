# Model Evaluation and Validation Report

**Date Generated:** 2026-10-13T21:02:11.393776
**Selected Model:** `HistGradientBoosting`
**Selection Rationale:** Selected automatically as the top performer on chronological validation set (lowest validation RMSE=0.5353 kWh).

## 1. Held-Out Test Set Performance (Final Evaluation)
- **Test Period:** 2010-04-19 20:00:00 to 2010-11-26 21:00:00
- **Test Samples:** 5,101 hourly observations

| Metric | Value | Units | Explanation |
|---|---|---|---|
| **MAE** | `0.3213` | kWh | Mean Absolute Error across test hours |
| **RMSE** | `0.4636` | kWh | Root Mean Squared Error (penalizes large spikes) |
| **R²** | `0.5728` | — | Coefficient of Determination |
| **MAPE** | `43.18%` | % | Mean Absolute Percentage Error (safe denominator) |

## 2. Validation Set Model Comparison
Comparison of candidate models and baselines on the same chronological validation period:

| Candidate Model | Val MAE (kWh) | Val RMSE (kWh) | Val R² | Val MAPE (%) |
|---|---|---|---|---|
| **Persistence (t-1)** | 0.467 | 0.695 | 0.3919 | 45.13% |
| **Seasonal Naive (t-24)** | 0.6657 | 0.9584 | -0.1563 | 75.02% |
| **HistGradientBoosting** | 0.3682 | 0.5353 | 0.6393 | 39.21% |
| **RandomForest** | 0.3715 | 0.5414 | 0.631 | 39.81% |
| **RidgeRegression** | 0.4234 | 0.5908 | 0.5606 | 48.84% |

## 3. Baseline Comparison on Test Set
| Model | Test MAE (kWh) | Test RMSE (kWh) | Test R² |
|---|---|---|---|
| Persistence (t-1) | 0.3859 | 0.5843 | 0.3215 |
| Seasonal Naive (t-24) | 0.5109 | 0.7586 | -0.1437 |
| **HistGradientBoosting (Selected)** | **0.3213** | **0.4636** | **0.5728** |

## 4. Key Takeaways & Limitations
- The machine learning model outperforms the naive persistence baseline.
- Tree-based ensembles capture non-linear daily and weekly patterns without requiring strict stationarity transformations.
- Recursive multi-step forecasts are susceptible to error accumulation over extended horizons.
- Results reflect residential single-household consumption, which possesses higher variance than aggregated regional utility loads.