# Prediction API Documentation

The **AI-Powered Energy Consumption Forecasting API** is a Flask REST microservice for serving real-time energy predictions and recursive multi-step schedules.

- **Base URL:** `http://127.0.0.1:5000`
- **Content-Type:** `application/json`
- **CORS Support:** Enabled via `flask-cors`

---

## 1. Endpoints Catalog

### `GET /`
Returns the API service overview and available routes.

**Example Request:**
```bash
curl -X GET http://127.0.0.1:5000/
```

**Response (200 OK):**
```json
{
  "service": "AI-Powered Energy Consumption Forecasting API",
  "version": "1.0.0",
  "status": "operational",
  "endpoints": {
    "GET /": "Service overview and endpoint catalog",
    "GET /health": "Healthcheck and model readiness status",
    "GET /model-info": "Model architecture, target details, and training metadata",
    "POST /predict": "One-step-ahead next-hour electricity forecast",
    "POST /forecast": "Recursive multi-step forecast for custom horizon (1-168 hours)"
  }
}
```

---

### `GET /health`
Verifies whether the microservice is operational and the machine learning model artifact is loaded.

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_path_exists": true,
  "message": "Engine ready for inference."
}
```

*(Returns HTTP 503 if the model artifact has not been trained or loaded).*

---

### `GET /model-info`
Returns metadata regarding the trained model, training split summary, and engineered feature list.

**Response (200 OK):**
```json
{
  "model_name": "HistGradientBoosting",
  "target_col": "energy_kwh",
  "target_units": "kWh",
  "feature_count": 22,
  "features": [
    "hour", "day_of_week", "day_of_month", "month", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_24", "lag_168",
    "rolling_mean_6h", "rolling_std_6h", "rolling_mean_24h", "rolling_max_24h", "rolling_min_24h"
  ],
  "test_metrics": {
    "mae": 0.3213,
    "rmse": 0.4636,
    "r2": 0.5728,
    "mape": 25.2
  }
}
```

---

### `POST /predict`
Generates a one-step-ahead prediction for the next hour ($t+1$).

**Option A: Pass historical readings explicitly (Recommended for API integration):**
```json
{
  "history_kwh": [1.12, 1.05, 0.98, ...],  // Must contain at least 168 hourly values
  "last_timestamp": "2026-10-13T20:00:00"   // Optional ISO string
}
```

**Option B: Use latest server dataset observations:**
```json
{}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d "{}"
```

**Response (200 OK):**
```json
{
  "forecast_type": "one_step_ahead",
  "model_used": "HistGradientBoosting",
  "predicted_energy_kwh": 1.2176,
  "reference_last_history_timestamp": "2010-11-26T21:00:00",
  "timestamp": "2010-11-26T22:00:00",
  "units": "kWh"
}
```

---

### `POST /forecast`
Generates a recursive multi-step forecast over a configurable horizon ($1 \le H \le 168$ hours).

**Request Schema:**
```json
{
  "horizon_hours": 24,
  "history_kwh": [ ... ],                  // Optional (uses latest dataset if omitted)
  "last_timestamp": "2026-10-13T20:00:00"  // Optional
}
```

**Example Request:**
```bash
curl -X POST http://127.0.0.1:5000/forecast \
  -H "Content-Type: application/json" \
  -d "{\"horizon_hours\": 24}"
```

**Response (200 OK):**
```json
{
  "forecast_type": "recursive_multi_step",
  "horizon_hours": 24,
  "model_used": "HistGradientBoosting",
  "total_predicted_kwh": 36.9443,
  "units": "kWh",
  "predictions": [
    {
      "step": 1,
      "timestamp": "2010-11-26T22:00:00",
      "predicted_energy_kwh": 1.2176,
      "hour": 22,
      "day_of_week": "Friday",
      "is_weekend": 0
    },
    ...
  ]
}
```

---

## 2. HTTP Status Codes & Error Handling

| Status Code | Meaning | Example Cause |
|---|---|---|
| `200 OK` | Success | Valid prediction returned |
| `400 Bad Request` | Input Validation Error | Provided history has < 168 readings or horizon < 1 or > 168 |
| `503 Service Unavailable` | Model Not Loaded | The server model artifact has not been trained yet |
| `500 Internal Error` | Server Exception | Internal failure (error details logged safely to server log) |
