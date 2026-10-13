"""
Streamlit Interactive Dashboard for the AI-Powered Energy Consumption Forecasting System.
Features:
1. Executive Overview & Key Performance Metrics
2. Historical Energy Analytics & Diurnal Consumption Patterns
3. Interactive Recursive Forecasting Engine
4. Empirical Model Evaluation & Baseline Comparison
5. Virtual Energy Planning, Tariff Cost Analysis & Carbon Simulation
"""

import json
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.config import (
    PROCESSED_HOURLY_CSV,
    BEST_MODEL_FILE,
    MODEL_METADATA_FILE,
    METRICS_JSON_FILE,
    PREDICTIONS_DIR,
    DEFAULT_FLAT_TARIFF_PER_KWH,
    DEFAULT_PEAK_TARIFF_PER_KWH,
    DEFAULT_OFF_PEAK_TARIFF_PER_KWH,
    DEFAULT_CARBON_INTENSITY_KG_PER_KWH,
    DEFAULT_FORECAST_HORIZON,
    TARGET_UNITS,
)
from src.forecasting import ForecastingEngine
from src.simulation import EnergyPlanningSimulator

# Page Configuration
st.set_page_config(
    page_title="AI Energy Consumption Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a237e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #455a64;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #1976d2;
    }
    .badge-status {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Loads processed hourly dataset with caching."""
    if not PROCESSED_HOURLY_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED_HOURLY_CSV, index_col=0, parse_dates=True)
    return df


@st.cache_data
def load_model_metadata() -> dict:
    """Loads model metadata JSON."""
    if not MODEL_METADATA_FILE.exists():
        return {}
    with open(MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_metrics_report() -> dict:
    """Loads full metrics comparison JSON."""
    if not METRICS_JSON_FILE.exists():
        return {}
    with open(METRICS_JSON_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    """Loads test period actual vs predicted values."""
    pred_path = PREDICTIONS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        return pd.DataFrame()
    return pd.read_csv(pred_path, index_col=0, parse_dates=True)


@st.cache_resource
def get_forecasting_engine() -> ForecastingEngine:
    """Initializes and caches the forecasting engine singleton."""
    return ForecastingEngine()


def main():
    # Header Banner
    st.markdown('<div class="main-header">⚡ AI-Powered Energy Consumption Forecasting System</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'End-to-End Time-Series Analytics, Recursive Machine Learning Forecasting, '
        'and Virtual Building Energy Planning.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls & System Status
    st.sidebar.image("https://img.icons8.com/color/96/electricity.png", width=70)
    st.sidebar.title("System Controls")

    df_hourly = load_processed_data()
    metadata = load_model_metadata()
    metrics_data = load_metrics_report()
    engine = get_forecasting_engine()

    model_ready = engine.is_ready()
    data_ready = not df_hourly.empty

    st.sidebar.subheader("Environment Status")
    if model_ready:
        st.sidebar.markdown('Model: <span class="badge-status">Active (Fitted)</span>', unsafe_allow_html=True)
        st.sidebar.caption(f"Architecture: **{metadata.get('model_name', 'HistGradientBoosting')}**")
    else:
        st.sidebar.error("Model artifact not found. Run `python main.py --all`.")

    if data_ready:
        st.sidebar.markdown(f"Hourly Records: **{len(df_hourly):,}**")
        st.sidebar.caption(f"Span: {df_hourly.index.min().date()} to {df_hourly.index.max().date()}")
    else:
        st.sidebar.warning("Dataset not processed yet.")

    st.sidebar.divider()
    st.sidebar.info(
        "💡 **Academic Prototype Note:**\n"
        "This system operates on public household measurements from the UCI Repository. "
        "Calculated tariffs and emissions are virtual planning scenarios, not real-time smart grid feeds."
    )

    # Main Navigation Tabs
    tab_overview, tab_history, tab_forecast, tab_eval, tab_sim = st.tabs([
        "📊 System Overview",
        "📈 Historical Analytics",
        "🔮 Recursive Forecasting",
        "🎯 Model Evaluation",
        "🌱 Virtual Energy Planning",
    ])

    # -------------------------------------------------------------
    # TAB 1: SYSTEM OVERVIEW
    # -------------------------------------------------------------
    with tab_overview:
        st.subheader("Executive System Summary")

        if not model_ready or not data_ready:
            st.warning("⚠️ Project artifacts are partially generated. Please run `python main.py --all` to train and build all assets.")

        # Top KPIs
        test_m = metadata.get("test_metrics", {})
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(
                label="Selected Model",
                value=metadata.get("model_name", "Tree Ensemble"),
                help="Top-performing model chosen via chronological validation RMSE",
            )
        with col2:
            st.metric(
                label="Test MAE (Mean Absolute Error)",
                value=f"{test_m.get('mae', '—')} {TARGET_UNITS}",
                help="Average absolute difference between actual and predicted hourly consumption",
            )
        with col3:
            st.metric(
                label="Test RMSE",
                value=f"{test_m.get('rmse', '—')} {TARGET_UNITS}",
                help="Root Mean Squared Error on held-out test data",
            )
        with col4:
            st.metric(
                label="Test R² Score",
                value=f"{test_m.get('r2', '—')}",
                help="Variance explained on held-out test set",
            )

        st.markdown("---")

        c_left, c_right = st.columns([3, 2])
        with c_left:
            st.markdown("### 📋 Core Pipeline Capabilities")
            st.markdown(
                """
                - **Rigorous Data Quality & Integration:** Converts 1-minute active power (kW) readings to true energy (kWh) via numerical integration, enforcing a strict minimum observation coverage threshold (50/60 minutes).
                - **Causal Feature Engineering:** Generates cyclical trigonometric calendar signals, multi-horizon lags ($t-1, t-2, t-3, t-24, t-168$), and strictly shifted rolling statistics to ensure zero future data leakage.
                - **Leakage-Safe Multi-Step Engine:** Deploys recursive forecasting for horizons up to 72 hours without accessing actual future values.
                - **Virtual Operational Planning:** Estimates energy costs under flat and Time-of-Use tariffs, calculates carbon footprints, and quantifies peak shaving scenarios.
                """
            )
        with c_right:
            st.markdown("### ⚙️ Pipeline Specifications")
            specs = {
                "Target Variable": "Hourly Electricity Consumption (kWh)",
                "Raw Data Source": "UCI Individual Household Electric Power",
                "Observation Sampling": "1-minute raw resampled to 1-hour",
                "Validation Strategy": "Chronological split (70% Train, 15% Val, 15% Test)",
                "Forecast Horizon": "Configurable (Default 24 Hours)",
                "Leakage Prevention": "Shifted rolling windows & strict causal lags",
            }
            st.table(pd.DataFrame(list(specs.items()), columns=["Attribute", "Value"]).set_index("Attribute"))

    # -------------------------------------------------------------
    # TAB 2: HISTORICAL ANALYTICS
    # -------------------------------------------------------------
    with tab_history:
        st.subheader("Historical Energy Consumption Analytics")
        if data_ready:
            # Date range filter
            min_date = df_hourly.index.min().date()
            max_date = df_hourly.index.max().date()

            col_d1, col_d2 = st.columns(2)
            with col_d1:
                start_filter = st.date_input("Start Date", value=max_date - pd.Timedelta(days=60), min_value=min_date, max_value=max_date)
            with col_d2:
                end_filter = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)

            filtered_df = df_hourly.loc[str(start_filter):str(end_filter)]

            # Interactive Plotly time series
            st.markdown("#### ⚡ Energy Consumption Trend")
            fig_trend = px.line(
                filtered_df,
                y="energy_kwh",
                title=f"Hourly Energy Consumption ({start_filter} to {end_filter})",
                labels={"energy_kwh": f"Consumption ({TARGET_UNITS})", "index": "Timestamp"},
                color_discrete_sequence=["#1976d2"],
            )
            fig_trend.update_layout(height=400, hovermode="x unified")
            st.plotly_chart(fig_trend, use_container_width=True)

            # Diurnal Profiles
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.markdown("#### 🕒 Average Diurnal Profile (by Hour)")
                df_copy = df_hourly.copy()
                df_copy["hour"] = df_copy.index.hour
                hourly_grp = df_copy.groupby("hour")["energy_kwh"].agg(["mean", "std"]).reset_index()

                fig_diurnal = go.Figure()
                fig_diurnal.add_trace(go.Scatter(
                    x=hourly_grp["hour"], y=hourly_grp["mean"],
                    mode="lines+markers", name="Mean Consumption",
                    line=dict(color="#0288d1", width=3)
                ))
                fig_diurnal.update_layout(
                    title="Average Hourly Energy Profile",
                    xaxis_title="Hour of Day (0-23)",
                    yaxis_title=f"Energy ({TARGET_UNITS})",
                    height=350,
                )
                st.plotly_chart(fig_diurnal, use_container_width=True)

            with c_p2:
                st.markdown("#### 📅 Weekday vs. Weekend Patterns")
                df_copy["is_weekend"] = df_copy.index.dayofweek >= 5
                wk_df = df_copy[~df_copy["is_weekend"]].groupby("hour")["energy_kwh"].mean().reset_index()
                we_df = df_copy[df_copy["is_weekend"]].groupby("hour")["energy_kwh"].mean().reset_index()

                fig_wk = go.Figure()
                fig_wk.add_trace(go.Scatter(x=wk_df["hour"], y=wk_df["energy_kwh"], mode="lines", name="Weekday (Mon-Fri)", line=dict(color="#388e3c", width=2.5)))
                fig_wk.add_trace(go.Scatter(x=we_df["hour"], y=we_df["energy_kwh"], mode="lines", name="Weekend (Sat-Sun)", line=dict(color="#e64a19", width=2.5, dash="dash")))
                fig_wk.update_layout(
                    title="Weekday vs Weekend Hourly Demand",
                    xaxis_title="Hour of Day (0-23)",
                    yaxis_title=f"Mean Energy ({TARGET_UNITS})",
                    height=350,
                )
                st.plotly_chart(fig_wk, use_container_width=True)
        else:
            st.info("Please load or preprocess the dataset to display historical analytics.")

    # -------------------------------------------------------------
    # TAB 3: RECURSIVE FORECASTING
    # -------------------------------------------------------------
    with tab_forecast:
        st.subheader("Recursive Multi-Step Electricity Forecast")
        st.markdown(
            "Configure a future forecast horizon. The engine iteratively predicts each hour ahead, "
            "feeding previous predictions into subsequent lag and rolling features without accessing future test data."
        )

        if model_ready and data_ready:
            c_f1, c_f2 = st.columns([1, 3])
            with c_f1:
                horizon_select = st.slider("Forecast Horizon (Hours)", min_value=6, max_value=72, value=24, step=6)
                history_display_window = st.selectbox("Context Window (Hours)", options=[24, 48, 72, 168], index=1)
                run_forecast_btn = st.button("Generate Recursive Forecast 🚀", type="primary", use_container_width=True)

            target_series = df_hourly["energy_kwh"]

            if run_forecast_btn or "cached_forecast" in st.session_state:
                if run_forecast_btn:
                    with st.spinner(f"Generating recursive {horizon_select}-hour forecast..."):
                        forecast_out = engine.predict_horizon(target_series, horizon_hours=horizon_select)
                        st.session_state["cached_forecast"] = forecast_out
                        st.session_state["cached_horizon"] = horizon_select
                else:
                    forecast_out = st.session_state.get("cached_forecast")
                    horizon_select = st.session_state.get("cached_horizon", 24)

                with c_f2:
                    total_kwh = forecast_out["predicted_energy_kwh"].sum()
                    peak_kwh = forecast_out["predicted_energy_kwh"].max()
                    avg_kwh = forecast_out["predicted_energy_kwh"].mean()

                    kpi1, kpi2, kpi3 = st.columns(3)
                    kpi1.metric("Total Forecasted Energy", f"{total_kwh:.2f} kWh")
                    kpi2.metric("Peak Predicted Load", f"{peak_kwh:.2f} kWh")
                    kpi3.metric("Average Hourly Load", f"{avg_kwh:.2f} kWh")

                # Visualizing Forecast with Historical Context
                st.markdown("#### 🔮 Forecast vs. Recent Observations")
                recent_history = target_series.iloc[-history_display_window:]

                fig_fc = go.Figure()
                fig_fc.add_trace(go.Scatter(
                    x=recent_history.index, y=recent_history.values,
                    mode="lines+markers", name="Recent Actual Observations",
                    line=dict(color="#1976d2", width=2), marker=dict(size=4)
                ))
                fig_fc.add_trace(go.Scatter(
                    x=forecast_out.index, y=forecast_out["predicted_energy_kwh"],
                    mode="lines+markers", name=f"Recursive {horizon_select}h Forecast",
                    line=dict(color="#f57c00", width=3, dash="dot"), marker=dict(size=5, symbol="square")
                ))

                # Highlight forecast horizon
                fig_fc.add_vrect(
                    x0=forecast_out.index[0], x1=forecast_out.index[-1],
                    fillcolor="#ffe082", opacity=0.2, line_width=0,
                    annotation_text="Forecast Window", annotation_position="top left"
                )
                fig_fc.update_layout(
                    xaxis_title="Timestamp", yaxis_title=f"Energy Consumption ({TARGET_UNITS})",
                    hovermode="x unified", height=450
                )
                st.plotly_chart(fig_fc, use_container_width=True)

                # Predictions Data Table
                st.markdown("#### 📋 Timestamped Forecast Schedule")
                st.dataframe(forecast_out.style.format({"predicted_energy_kwh": "{:.3f}"}), use_container_width=True)

                csv_data = forecast_out.to_csv().encode("utf-8")
                st.download_button("Download Forecast CSV", data=csv_data, file_name=f"energy_forecast_{horizon_select}h.csv", mime="text/csv")
        else:
            st.warning("Model or data not ready. Train model first.")

    # -------------------------------------------------------------
    # TAB 4: MODEL EVALUATION
    # -------------------------------------------------------------
    with tab_eval:
        st.subheader("Empirical Model Evaluation & Baseline Comparison")

        test_preds = load_test_predictions()
        if not test_preds.empty and metrics_data:
            # Metrics comparison table
            st.markdown("#### 🏁 Validation Benchmark: Baselines vs. ML Candidates")
            val_comp = metrics_data.get("validation_comparisons", {})
            rows = []
            for name, data in val_comp.items():
                m = data.get("metrics", {})
                rows.append({
                    "Model": name,
                    "MAE (kWh)": m.get("mae"),
                    "RMSE (kWh)": m.get("rmse"),
                    "R² Score": m.get("r2"),
                    "MAPE (%)": f"{m.get('mape')}%",
                })
            df_comp = pd.DataFrame(rows)
            st.table(df_comp)

            # Test Period Actual vs Predicted Chart
            st.markdown("#### 🎯 Held-Out Test Set Performance (Actual vs. Predicted)")
            sample_hours_slice = st.slider("Select Test Snapshot Window (Hours)", min_value=48, max_value=336, value=168, step=24)
            sample_eval = test_preds.iloc[-sample_hours_slice:]

            fig_test = go.Figure()
            fig_test.add_trace(go.Scatter(x=sample_eval.index, y=sample_eval["actual_kwh"], mode="lines", name="Actual Energy", line=dict(color="#212121", width=2)))
            fig_test.add_trace(go.Scatter(x=sample_eval.index, y=sample_eval["predicted_kwh"], mode="lines", name="Predicted Energy (ML)", line=dict(color="#e53935", width=2, dash="dash")))
            fig_test.update_layout(xaxis_title="Timestamp", yaxis_title=f"Consumption ({TARGET_UNITS})", height=400, hovermode="x unified")
            st.plotly_chart(fig_test, use_container_width=True)

            # Residual Distribution
            c_r1, c_r2 = st.columns(2)
            with c_r1:
                st.markdown("#### 📉 Residual Distribution")
                residuals = test_preds["residual_kwh"]
                fig_res = px.histogram(residuals, nbins=60, title="Prediction Errors (Actual - Predicted)", labels={"value": f"Residual ({TARGET_UNITS})"}, color_discrete_sequence=["#5c6bc0"])
                fig_res.add_vline(x=0, line_dash="dash", line_color="red")
                fig_res.update_layout(height=350)
                st.plotly_chart(fig_res, use_container_width=True)

            with c_r2:
                st.markdown("#### 🔍 Metric Definitions")
                st.markdown(
                    """
                    - **MAE (Mean Absolute Error):** Measures the average magnitude of errors in identical units (kWh). Intuitive and directly interpretable.
                    - **RMSE (Root Mean Squared Error):** Heavily penalizes large sporadic errors due to the squaring penalty. Essential for peak capacity planning.
                    - **R² (Coefficient of Determination):** Quantifies the proportion of variation explained by the model compared to predicting the historical mean.
                    """
                )
        else:
            st.info("Test evaluation results will appear here after training. Run `python main.py --train`.")

    # -------------------------------------------------------------
    # TAB 5: VIRTUAL ENERGY PLANNING SIMULATION
    # -------------------------------------------------------------
    with tab_sim:
        st.subheader("Virtual Building Energy Planning & Tariff Simulation")
        st.markdown(
            "Simulates cost estimation and carbon abatement scenarios using user-configurable tariffs. "
            "Evaluate flat vs. Time-of-Use rates and test peak shaving demand response."
        )

        if model_ready and data_ready:
            c_s1, c_s2 = st.columns([1, 2])
            with c_s1:
                st.markdown("##### ⚙️ Scenario Configuration")
                flat_rate = st.number_input("Flat Electricity Tariff ($/kWh)", min_value=0.01, max_value=2.00, value=DEFAULT_FLAT_TARIFF_PER_KWH, step=0.01)
                tou_peak_rate = st.number_input("ToU Peak Tariff ($/kWh)", min_value=0.01, max_value=2.00, value=DEFAULT_PEAK_TARIFF_PER_KWH, step=0.01)
                tou_off_peak_rate = st.number_input("ToU Off-Peak Tariff ($/kWh)", min_value=0.01, max_value=2.00, value=DEFAULT_OFF_PEAK_TARIFF_PER_KWH, step=0.01)
                carbon_factor = st.slider("Grid Carbon Intensity (kg CO2 / kWh)", min_value=0.05, max_value=1.00, value=DEFAULT_CARBON_INTENSITY_KG_PER_KWH, step=0.05)
                shaving_slider = st.slider("Peak Demand Curtailment (%)", min_value=5, max_value=40, value=15, step=5)

            # Generate 24h forecast for simulation
            forecast_24 = engine.predict_horizon(df_hourly["energy_kwh"], horizon_hours=24)
            simulator = EnergyPlanningSimulator(
                flat_tariff=flat_rate,
                peak_tariff=tou_peak_rate,
                off_peak_tariff=tou_off_peak_rate,
                carbon_intensity_kg_per_kwh=carbon_factor,
            )
            sim_report = simulator.run_simulation(
                forecast_df=forecast_24,
                historical_baseline_series=df_hourly["energy_kwh"],
                shaving_pct=shaving_slider / 100.0,
            )

            with c_s2:
                st.markdown("##### 💡 Simulation Results (Next 24 Hours)")
                costs = sim_report["cost_and_emissions_estimates"]
                consump = sim_report["consumption_summary"]
                shaved = sim_report["hypothetical_shaving_scenario"]

                sc1, sc2, sc3 = st.columns(3)
                sc1.metric("Est. Total Consumption", f"{consump['total_predicted_kwh']} kWh")
                sc2.metric("Flat Tariff Cost", f"${costs['estimated_cost_flat_tariff_usd']:.2f}")
                sc3.metric("ToU Tariff Cost", f"${costs['estimated_cost_tou_tariff_usd']:.2f}")

                sc4, sc5, sc6 = st.columns(3)
                sc4.metric("Est. Carbon Footprint", f"{costs['estimated_carbon_emissions_kg_co2']} kg CO₂")
                sc5.metric("Peak Hours Identified", f"{consump['peak_hours_count']} hrs")
                sc6.metric(f"Hypothetical {shaving_slider}% Savings", f"${shaved['potential_tou_cost_avoidance_usd']:.2f}")

            # Hourly Cost Breakdown Chart
            st.markdown("#### ⏱️ Hourly Cost & Demand Profile")
            schedule_df = pd.DataFrame(sim_report["hourly_schedule"])
            schedule_df["timestamp"] = pd.to_datetime(schedule_df["timestamp"])

            fig_cost = px.bar(
                schedule_df,
                x="timestamp",
                y="hourly_cost_tou",
                color="is_peak_demand",
                color_discrete_map={True: "#d32f2f", False: "#1976d2"},
                title="Hourly Cost under Time-of-Use Tariff (Red = Peak Demand Hours)",
                labels={"hourly_cost_tou": "Cost ($)", "is_peak_demand": "High Demand Period", "timestamp": "Hour"},
            )
            fig_cost.update_layout(height=380)
            st.plotly_chart(fig_cost, use_container_width=True)

            st.caption(
                "⚠️ **Disclaimer:** All financial figures and emissions metrics are user-configured estimations. "
                "This simulation does not represent actual electrical billing or verified automated grid optimization."
            )
        else:
            st.info("Simulation requires model and dataset ready. Run `python main.py --all`.")


if __name__ == "__main__":
    main()
