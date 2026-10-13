"""
Virtual Energy Planning Simulation module.
Demonstrates scenario analysis for building energy planning, peak demand identification,
tariff-based cost estimation, and carbon emissions estimation without making exaggerated claims.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from src.config import (
    SIMULATION_REPORT_FILE,
    DEFAULT_FLAT_TARIFF_PER_KWH,
    DEFAULT_PEAK_TARIFF_PER_KWH,
    DEFAULT_OFF_PEAK_TARIFF_PER_KWH,
    PEAK_HOURS,
    DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    DEFAULT_PEAK_PERCENTILE,
    TARGET_UNITS,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EnergyPlanningSimulator:
    """
    Simulates operational energy planning for a building or household
    based on machine learning forecast inputs and user-configured tariffs.
    """

    def __init__(
        self,
        flat_tariff: float = DEFAULT_FLAT_TARIFF_PER_KWH,
        peak_tariff: float = DEFAULT_PEAK_TARIFF_PER_KWH,
        off_peak_tariff: float = DEFAULT_OFF_PEAK_TARIFF_PER_KWH,
        peak_hours: list = PEAK_HOURS,
        carbon_intensity_kg_per_kwh: float = DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    ):
        self.flat_tariff = flat_tariff
        self.peak_tariff = peak_tariff
        self.off_peak_tariff = off_peak_tariff
        self.peak_hours = peak_hours
        self.carbon_intensity = carbon_intensity_kg_per_kwh

    def run_simulation(
        self,
        forecast_df: pd.DataFrame,
        historical_baseline_series: Optional[pd.Series] = None,
        custom_peak_threshold_kwh: Optional[float] = None,
        shaving_pct: float = 0.15,  # 15% hypothetical peak curtailment/shift
    ) -> Dict[str, Any]:
        """
        Executes a 24-hour virtual energy planning simulation.

        Args:
            forecast_df: DataFrame of hourly predictions with 'predicted_energy_kwh'.
            historical_baseline_series: Historical series used to derive empirical peak threshold.
            custom_peak_threshold_kwh: Optional custom threshold in kWh to flag peak hours.
            shaving_pct: Hypothetical load shifting/shaving percentage during peak hours.

        Returns:
            Dictionary containing simulation scenario results, cost estimates, and limitations.
        """
        if "predicted_energy_kwh" not in forecast_df.columns:
            raise ValueError("forecast_df must contain 'predicted_energy_kwh' column.")

        preds = forecast_df["predicted_energy_kwh"].values
        timestamps = forecast_df.index
        hours = [ts.hour for ts in timestamps]

        total_forecast_kwh = float(np.sum(preds))
        avg_hourly_kwh = float(np.mean(preds))
        max_hourly_kwh = float(np.max(preds))
        min_hourly_kwh = float(np.min(preds))

        # Peak Threshold Determination
        if custom_peak_threshold_kwh is not None:
            peak_threshold = float(custom_peak_threshold_kwh)
        elif historical_baseline_series is not None and len(historical_baseline_series) > 0:
            peak_threshold = float(np.percentile(historical_baseline_series.dropna().values, DEFAULT_PEAK_PERCENTILE))
        else:
            peak_threshold = float(np.percentile(preds, 80.0))

        # Flag peak hours (exceeding threshold OR during tariff peak hours)
        is_above_threshold = preds >= peak_threshold
        is_tou_peak = np.isin(hours, self.peak_hours)
        peak_hours_count = int(np.sum(is_above_threshold))

        # Cost Calculations
        # 1. Flat tariff
        cost_flat = float(total_forecast_kwh * self.flat_tariff)

        # 2. Time-of-Use (ToU) tariff
        hourly_costs_tou = []
        for p, h in zip(preds, hours):
            rate = self.peak_tariff if h in self.peak_hours else self.off_peak_tariff
            hourly_costs_tou.append(p * rate)
        cost_tou = float(np.sum(hourly_costs_tou))

        # Carbon Emissions (kg CO2)
        emissions_kg_co2 = float(total_forecast_kwh * self.carbon_intensity)

        # Peak-Shaving / Load-Shifting Hypothetical Scenario
        # Model hypothetical 15% reduction during peak hours
        shaved_preds = preds.copy()
        for i, h in enumerate(hours):
            if is_above_threshold[i] or is_tou_peak[i]:
                shaved_preds[i] = preds[i] * (1.0 - shaving_pct)

        shaved_total_kwh = float(np.sum(shaved_preds))
        shaved_kwh_reduction = total_forecast_kwh - shaved_total_kwh
        shaved_cost_tou = float(
            sum(p * (self.peak_tariff if h in self.peak_hours else self.off_peak_tariff) for p, h in zip(shaved_preds, hours))
        )
        hypothetical_cost_saving = cost_tou - shaved_cost_tou
        hypothetical_emissions_saving = shaved_kwh_reduction * self.carbon_intensity

        # Hourly breakdown
        hourly_breakdown = []
        for i, ts in enumerate(timestamps):
            hourly_breakdown.append(
                {
                    "timestamp": str(ts),
                    "hour": int(hours[i]),
                    "forecast_kwh": round(float(preds[i]), 4),
                    "is_peak_demand": bool(is_above_threshold[i]),
                    "is_tou_peak_rate": bool(is_tou_peak[i]),
                    "hourly_cost_tou": round(float(hourly_costs_tou[i]), 4),
                }
            )

        report = {
            "simulation_name": "Virtual 24-Hour Building Energy Planning Scenario",
            "forecast_horizon_hours": len(forecast_df),
            "assumptions_and_inputs": {
                "flat_tariff_usd_per_kwh": self.flat_tariff,
                "tou_peak_tariff_usd_per_kwh": self.peak_tariff,
                "tou_off_peak_tariff_usd_per_kwh": self.off_peak_tariff,
                "tou_peak_hours_schedule": self.peak_hours,
                "grid_carbon_intensity_kg_co2_per_kwh": self.carbon_intensity,
                "peak_demand_threshold_kwh": round(peak_threshold, 4),
                "hypothetical_shaving_percentage": shaving_pct * 100.0,
            },
            "consumption_summary": {
                "total_predicted_kwh": round(total_forecast_kwh, 4),
                "average_hourly_kwh": round(avg_hourly_kwh, 4),
                "peak_hourly_kwh": round(max_hourly_kwh, 4),
                "minimum_hourly_kwh": round(min_hourly_kwh, 4),
                "peak_hours_count": peak_hours_count,
            },
            "cost_and_emissions_estimates": {
                "estimated_cost_flat_tariff_usd": round(cost_flat, 2),
                "estimated_cost_tou_tariff_usd": round(cost_tou, 2),
                "estimated_carbon_emissions_kg_co2": round(emissions_kg_co2, 2),
            },
            "hypothetical_shaving_scenario": {
                "shaved_total_kwh": round(shaved_total_kwh, 4),
                "potential_kwh_reduced_or_shifted": round(shaved_kwh_reduction, 4),
                "potential_tou_cost_avoidance_usd": round(hypothetical_cost_saving, 2),
                "potential_carbon_avoidance_kg_co2": round(hypothetical_emissions_saving, 2),
            },
            "hourly_schedule": hourly_breakdown,
            "disclaimer_and_limitations": (
                "This scenario is a synthetic virtual planning demonstration using user-configured "
                "tariff rates and carbon intensity factors. It does not represent an actual utility bill, "
                "a physical smart grid connection, or guaranteed financial/energy savings."
            ),
        }

        return report

    def save_simulation_report(
        self,
        report_data: Dict[str, Any],
        output_file: Optional[Path] = None,
    ) -> Path:
        """Saves simulation report as structured JSON."""
        out_path = Path(output_file) if output_file else SIMULATION_REPORT_FILE
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        logger.info(f"Saved simulation report to: {out_path.resolve()}")
        return out_path
