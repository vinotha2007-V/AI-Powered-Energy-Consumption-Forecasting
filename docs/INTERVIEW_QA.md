# Technical Interview Preparation & Project Deep-Dive

This guide prepares you to present, explain, and defend the **AI-Powered Energy Consumption Forecasting System** during technical interviews for Data Scientist, Machine Learning Engineer, and Energy Analytics roles.

---

## 1. Project High-Level Overview

### Q1: Can you describe your project in 60 seconds?
> **Answer:**  
> "I built an end-to-end Machine Learning forecasting system for household electricity consumption using 47 months of minute-level telemetry from the UCI Machine Learning Repository.  
> The core challenge in energy time series is converting noisy instantaneous power readings (in kW) into defensible hourly energy totals (in kWh) while preventing data leakage. I developed a leakage-safe feature engineering pipeline with cyclical calendar transforms, multi-horizon lags, and shifted rolling statistics.  
> Using a strictly chronological 70/15/15 split, I benchmarked gradient boosted trees against random forests, regularized linear models, and persistence baselines. My selected HistGradientBoosting model achieved a test MAE of **0.321 kWh** and RMSE of **0.464 kWh** on unseen 2010 data, outperforming the baseline by over 16%.  
> Finally, I deployed an autonomous recursive forecasting engine, an interactive Streamlit analytics dashboard, a Flask REST API, and a virtual energy planning simulation that models Time-of-Use tariff costs and peak demand shaving."

---

## 2. Data Engineering & Domain Physics

### Q2: Why did you convert active power in kW to energy in kWh, and how did you do it?
> **Answer:**  
> "Power is an instantaneous rate of energy flow ($1\text{ kW} = 1000\text{ Joules per second}$), whereas energy is the cumulative quantity of electricity consumed over time ($1\text{ kWh} = 1\text{ kW}$ sustained for 1 hour).  
> Utilities bill customers for *energy in kWh*, not instantaneous power. Because the raw dataset provides 1-minute power samples, I numerically integrated the readings over each 1-hour window:
> $$\text{Energy (kWh)} = \frac{1}{60} \sum_{m=1}^{60} \text{Global\_active\_power}_m (\text{kW})$$
> This equals $\overline{P}_{\text{hour}} \times 1\text{ hour}$ when all 60 minutes are observed. Conflating kW and kWh is a common beginner mistake in energy data science."

### Q3: How did you handle missing minute readings during hourly aggregation?
> **Answer:**  
> "I established an explicit **minimum observation coverage threshold** of 50 valid minute readings per hour (83.3% coverage).  
> If an hour has fewer than 50 readings, treating the sum as the full hour's consumption would artificially bias the target downward. Instead, I marked insufficiently observed hours as NaN.  
> For predominantly complete hours with 50 to 59 readings, the mean power is multiplied by 1.0 hour, which mathematically normalizes for the minor unobserved minutes. Isolated gaps of 1–2 hours were forward-filled to preserve continuity without future observation leakage."

---

## 3. Feature Engineering & Data Leakage Prevention

### Q4: How did you ensure zero data leakage in your time series features?
> **Answer:**  
> "Data leakage in time series usually happens in three places:
> 1. **Random splitting:** Shuffling mixes future observations into the training set. I used strict chronological splitting: earliest 70% for training, middle 15% for validation, latest 15% for test.
> 2. **Rolling window leakage:** If a 6-hour rolling mean at timestamp $t$ includes observation $y_t$, the model gets access to the ground truth it is trying to predict! I enforced a strict `.shift(1)` before applying the rolling window:
>    $$\text{rolling\_mean\_6h}_t = \frac{1}{6} \sum_{i=1}^{6} y_{t-i}$$
> 3. **Preprocessing fitting:** All scalers and encoders were fit strictly on the training partition and applied out-of-sample to validation and test partitions."

### Q5: Why did you use sine and cosine encodings for calendar features?
> **Answer:**  
> "Standard integer encoding for hour (0 to 23) introduces an artificial numerical discontinuity: hour 23 and hour 0 appear 23 units apart despite being consecutive hours.  
> By projecting hour onto circular trigonometric coordinates:
> $$\text{hour\_sin} = \sin\left(\frac{2\pi \cdot \text{hour}}{24}\right), \quad \text{hour\_cos} = \cos\left(\frac{2\pi \cdot \text{hour}}{24}\right)$$
> the distance between 23:00 and 00:00 is mathematically continuous and smooth, allowing tree-based splits and distance metrics to recognize cyclical diurnal periodicity."

---

## 4. Modeling, Benchmarking & Metrics

### Q6: Why did HistGradientBoosting outperform Random Forest and Ridge Regression?
> **Answer:**  
> "Electricity demand has sharp non-linear interactions: an evening hour on a winter weekday behaves very differently from that same hour on a summer weekend.  
> Linear models like Ridge struggle with these multi-feature multiplicative interactions without manual polynomial expansion.  
> HistGradientBoosting uses histogram binning (similar to LightGBM), which speeds up training by an order of magnitude and sequentially focuses tree capacity on residual errors from prior iterations. On validation data, HistGradientBoosting achieved an RMSE of **0.535 kWh**, beating Random Forest (0.541 kWh) and Ridge (0.591 kWh)."

### Q7: Why did you include persistence and seasonal-naive baselines?
> **Answer:**  
> "In time series forecasting, a model that reports $R^2 = 0.50$ is meaningless unless compared to a naive baseline.  
> - **Persistence ($t-1$):** Assumes consumption next hour equals consumption this hour ($y_t = y_{t-1}$).
> - **Seasonal Naive ($t-24$):** Assumes consumption tomorrow at 3 PM equals consumption today at 3 PM ($y_t = y_{t-24}$).  
> The persistence baseline achieved test RMSE of 0.584 kWh. Our ML model reduced this to 0.464 kWh (a 20.6% improvement), proving genuine algorithmic value over simple heuristics."

### Q8: What does your test $R^2$ of ~0.57 mean? Is that good or bad?
> **Answer:**  
> "For an individual residential household, an $R^2$ of 0.57 is strong and realistic.  
> Household electricity has high human stochasticity: whether an occupant decides to bake bread or run a clothes dryer at 8 PM is unpredictable from calendar features alone.  
> In contrast, regional grid operators like PJM achieve $R^2 > 0.95$ because millions of individual loads aggregate together, smoothing variance by the law of large numbers. Anyone claiming $R^2 > 0.90$ on single-household raw data without weather features likely has target leakage."

---

## 5. Forecasting Engine & System Architecture

### Q9: What is the difference between one-step-ahead and recursive multi-step forecasting?
> **Answer:**  
> "In **one-step-ahead prediction**, actual historical observations are available right up to $t-1$ to predict $t$.  
> In **recursive multi-step forecasting** (e.g., 24 hours ahead), future actual values do not exist yet. After predicting step $t+1$, that predicted value is appended to a virtual history buffer to compute the lags (`lag_1`, `lag_2`, etc.) for step $t+2$.  
> The primary limitation is **error accumulation**: any slight over-prediction in early hours propagates into downstream lags, which causes forecasts beyond 24–48 hours to smooth out toward cyclical daily averages."

### Q10: How does this prototype differ from an industrial utility EMS/SCADA system?
> **Answer:**  
> "My system is an academic software prototype operating in batch and REST microservice modes on historical data.  
> An industrial Energy Management System (EMS) requires:
> 1. Ingestion over real-time SCADA protocols like DNP3, Modbus, or IEC 61850.
> 2. Streaming stream-processing (e.g., Apache Flink or Kafka).
> 3. Two-way Automated Demand Response (OpenADR) to physically trigger smart thermostat setpoints or battery discharge.
> 4. Compliance with NERC CIP critical infrastructure cybersecurity standards."

---

## 6. Real-World Industry Applications

- **Smart Buildings:** Scheduling commercial HVAC pre-cooling during off-peak hours to minimize Time-of-Use demand charges.
- **Microgrids & Renewables:** Pairing load forecasts with solar PV generation forecasts to optimize battery charging schedules.
- **Data Centers:** Anticipating compute cooling loads to prevent thermal throttling and budget peak grid interconnection capacity.
