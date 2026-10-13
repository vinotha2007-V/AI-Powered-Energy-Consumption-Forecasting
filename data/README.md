# Data Directory Structure & Documentation

This directory contains the datasets for the **AI-Powered Energy Consumption Forecasting System**.

## Structure
```
data/
├── raw/         # Raw downloaded dataset files (e.g., household_power_consumption.txt/zip)
├── processed/   # Cleaned, hourly-resampled energy dataset with feature-ready target
└── README.md    # Dataset guidance, metadata, and citation
```

## Dataset Information
- **Dataset Name:** Individual Household Electric Power Consumption
- **Originating Institution:** EDF R&D (Électricité de France), Clamart, France
- **UCI Machine Learning Repository Citation:**
  - URL: [https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption)
  - Donors: Georges Hébrail, Alice Bérard
  - License: Creative Commons Attribution 4.0 International (CC BY 4.0)
- **Time Coverage:** December 16, 2006 (17:24:00) to November 26, 2010 (21:02:00) (47 months, ~2,075,259 minute readings).
- **Sampling Rate:** 1 minute.

## Target Definition & Unit Conversion
- **Raw Metric:** `Global_active_power` measured in **kilowatts (kW)** (instantaneous household active power).
- **Engineered Hourly Target:** Hourly electricity consumption in **kilowatt-hours (kWh)**:
  $$\text{Energy (kWh)} = \frac{1}{60} \sum_{m=1}^{N} \text{Global\_active\_power}_m$$
  where $N$ is the count of valid minute readings in that hour.
- **Data Coverage Threshold:**
  Hours with fewer than **50 valid minute readings** (out of 60, i.e. < 83.3% coverage) are flagged as invalid / missing rather than falsely treated as complete. Valid hours with 50-59 minutes are scaled proportionally to full hour energy.

## Acquisition Methods
1. **Automatic Acquisition (Recommended):**
   ```powershell
   python main.py --download
   ```
2. **Manual Download:**
   If automated download is blocked by a network proxy or firewall:
   1. Download `individual+household+electric+power+consumption.zip` from [UCI Machine Learning Repository](https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip).
   2. Extract the file `household_power_consumption.txt` directly into `data/raw/`.
   3. Run `python main.py --preprocess`.
