# Project Limitations, Boundaries, and Deployment Caveats

This document outlines the operational boundaries, data limitations, evaluation caveats, and responsible interpretation of the **AI-Powered Energy Consumption Forecasting System**.

---

## 1. Dataset Limitations

1. **Single-Household Geographic Scope:**
   - The dataset originates from a single residential home located in Sceaux, near Paris, France (collected between December 2006 and November 2010).
   - Household consumption behavior in this dataset reflects Western European residential appliance usage patterns of that era. It does not generalize directly to high-density commercial towers, factories, or households in different climate zones without recalibration.

2. **Absence of Modern Distributed Energy Resources (DERs):**
   - The measurements precede widespread residential adoption of Electric Vehicles (EVs), home battery energy storage systems (BESS), and rooftop photovoltaic (PV) solar panels.
   - EV charging creates abrupt high-power spikes (7 kW to 11 kW) that follow distinct charging schedules not present in this dataset.

3. **Absence of Weather Covariates:**
   - Outdoor temperature, humidity, heating degree days (HDD), and cooling degree days (CDD) are the primary physical drivers of seasonal heating and air conditioning.
   - While our model uses calendar signals (month, hour) to capture baseline seasonality, sudden heatwaves or cold snaps cannot be anticipated without real-time meteorological feeds.

---

## 2. Modeling & Evaluation Limitations

1. **Recursive Forecast Error Accumulation:**
   - For horizons beyond 1 hour (such as 24 or 48 hours ahead), earlier model predictions are recursively fed as lag inputs to subsequent steps.
   - Any residual error or bias in steps $t+1$ and $t+2$ compounds downstream into subsequent lag features (`lag_1`, `lag_2`, `lag_3`), causing multi-step predictions to gradually revert toward daily cyclical means.

2. **Single-Household Stochasticity vs. Regional Aggregate Smoothing:**
   - Individual residential power usage is notoriously noisy: an occupant turning on a 2 kW electric kettle or washing machine causes an immediate sharp step-function spike.
   - At the regional transmission level (such as ERCOT or PJM), millions of households average out by the Central Limit Theorem, producing smooth load curves with $R^2 > 0.95$. Expecting $R^2 > 0.90$ on a single stochastic household is unrealistic and typically indicates data leakage. Our test $R^2$ of **0.5728** represents strong, honest predictive power for a single residential dwelling.

---

## 3. Virtual Planning Simulation Boundaries

1. **Tariff and Cost Assumptions:**
   - Electricity tariffs ($0.18/kWh flat, $0.28 peak / $0.12 off-peak) are user-configurable assumptions illustrative of contemporary retail electricity plans. They do not represent live utility API feeds or guaranteed savings.

2. **Carbon Intensity Factor:**
   - Grid emissions are estimated using a fixed average factor ($0.40\text{ kg CO}_2/\text{kWh}$). Real electrical grids experience dynamic marginal carbon intensity fluctuations as coal, natural gas, wind, and nuclear generation dispatch changes minute by minute.

3. **No Automatic Energy or Cost Reductions:**
   - Generating a forecast does **not** automatically reduce power consumption or greenhouse gas emissions. A forecast provides operational awareness; tangible energy savings require physical load shifting, smart thermostat setpoint adjustments, or deliberate consumer behavior change.

---

## 4. Student Prototype vs. Production Grid Control

| Aspect | Portfolio Prototype (This Project) | Production Utility / SCADA System |
|---|---|---|
| **Data Ingestion** | Batch CSV / Public UCI Archive | Real-time IEC 61850 / DNP3 SCADA protocols & AMI smart meters |
| **Compute Architecture** | Local Windows/Mac Python workstation | Distributed cloud streaming (Kafka, Flink, Kubernetes) |
| **Failover / Redundancy**| Standard local file storage | Geographically replicated hot-standby failover clusters |
| **Actuation** | Virtual planning scenario reporting | Automated demand response (ADR), relay tripping, AGC control |
| **Regulatory Compliance** | Open-source MIT educational license | NERC CIP cybersecurity standards, FERC tariff compliance |
