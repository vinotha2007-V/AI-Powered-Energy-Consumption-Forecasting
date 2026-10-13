# Dataset Documentation & Methodology

## 1. Dataset Provenance & Citation

- **Dataset Name:** Individual Household Electric Power Consumption
- **Originating Organization:** EDF R&D (Électricité de France), Clamart, France
- **Donors:** Georges Hébrail, Alice Bérard
- **Repository:** UCI Machine Learning Repository (Dataset ID: 235)
- **Official URL:** [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
- **Direct Download Archive:** [https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip)
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Temporal Coverage:** December 16, 2006 (17:24:00) to November 26, 2010 (21:02:00) (47 consecutive months).
- **Sampling Frequency:** 1-minute intervals (~2,075,259 raw observations).

---

## 2. Raw Schema & Column Definitions

The source archive provides a single semicolon-delimited text file (`household_power_consumption.txt`) with 9 fields:

| Field Name | Raw Type | Missing Marker | Units | Physical Description |
|---|---|---|---|---|
| `Date` | String | None | dd/mm/yyyy | Calendar date of observation |
| `Time` | String | None | HH:MM:SS | Time of observation |
| `Global_active_power` | Float | `?` | **kW** | Household global minute-averaged active power |
| `Global_reactive_power`| Float | `?` | **kW** | Household global minute-averaged reactive power |
| `Voltage` | Float | `?` | **V** | Minute-averaged root-mean-square voltage |
| `Global_intensity` | Float | `?` | **A** | Minute-averaged current intensity |
| `Sub_metering_1` | Float | `?` | **Wh** | Kitchen sub-meter (dishwasher, microwave) |
| `Sub_metering_2` | Float | `?` | **Wh** | Laundry sub-meter (washing machine, dryer, fridge) |
| `Sub_metering_3` | Float | `?` | **Wh** | Climate sub-meter (water heater, air conditioning) |

---

## 3. Power (kW) vs. Energy (kWh) & Hourly Target Engineering

### Critical Distinction: Power vs. Energy
- **Power ($P$, measured in kW):** The instantaneous rate of electricity consumption at a specific moment in time.
- **Energy ($E$, measured in kWh):** The accumulated work or electricity consumed over a finite duration:
  $$E = \int_{0}^{T} P(t) \, dt$$

### Defensible Hourly Target Derivation
For an hourly interval $[t, t+1\text{ hour}]$ sampled at 1-minute intervals ($\Delta t = 1/60\text{ hour}$), the total hourly energy consumed is obtained by integrating valid minute power readings:
$$E_{\text{hourly}} (\text{kWh}) = \sum_{m=1}^{N} P_m (\text{kW}) \times \frac{1}{60} \text{ hours}$$

When all 60 minutes are observed ($N=60$), this is mathematically equivalent to:
$$E_{\text{hourly}} = \left( \frac{1}{60} \sum_{m=1}^{60} P_m \right) \times 1.0\text{ hour} = \overline{P}_{\text{hour}} \times 1\text{ h}$$

### Minimum Observation Coverage Threshold
To prevent false underestimation caused by missing sensor transmissions:
- An hour is classified as **valid** only if it contains at least **50 non-null minute readings** ($N \ge 50$, corresponding to $>83.3\%$ observation coverage).
- For valid hours with $50 \le N < 60$, the consumption is scaled by $(60 / N)$ to reflect the estimated full hour energy:
  $$E_{\text{hourly}} = \overline{P}_{\text{hour}} \times 1.0\text{ hour}$$
- Hours with fewer than 50 observations are marked as **insufficiently observed (NaN)** rather than artificially treated as zero or complete.

---

## 4. Empirical Dataset Statistics

From actual pipeline execution on the official UCI dataset:
- **Raw Rows Ingested:** 2,075,259
- **Raw Missing Power Readings:** 25,979 (~1.25%)
- **Duplicate Timestamps:** 0
- **Total Potential Hourly Windows:** 34,589
- **Valid Hourly Observations:** 34,147 (98.72% retention)
- **Insufficient Coverage Hours:** 442 (1.28%)
- **Imputed Isolated Single Gaps:** 25 hours (forward-filled up to 2 consecutive hours)
- **Final Clean Hourly Observations:** 34,172
- **Energy Target Mean:** `1.0915 kWh`
- **Energy Target Max:** `6.5605 kWh`
- **Energy Target Std Dev:** `1.0183 kWh`
