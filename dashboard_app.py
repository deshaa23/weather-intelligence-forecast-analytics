from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
CURRENT_CSV = PROCESSED_DIR / "india_weather_forecast.csv"
PREDICTIONS_CSV = REPORTS_DIR / "xgboost_test_predictions.csv"
METRICS_CSV = REPORTS_DIR / "xgboost_metrics.csv"
CITY_METRICS_CSV = REPORTS_DIR / "xgboost_metrics_by_city.csv"
IMPORTANCE_CSV = REPORTS_DIR / "xgboost_feature_importance.csv"

BG = "#F1F7FC"
CARD = "#FFFFFF"
NAVY = "#073763"
NAVY_2 = "#0B4F80"
BLUE = "#1687F8"
CYAN = "#22B8CF"
TEAL = "#0FA3A8"
GREEN = "#16A34A"
ORANGE = "#F59E0B"
TEXT = "#12314D"
MUTED = "#637A91"
BORDER = "#DCE8F2"
CITY_COLORS = {
    "Bengaluru": "#1687F8",
    "Chennai": "#F97316",
    "Delhi": "#EF4444",
    "Hyderabad": "#8B5CF6",
    "Kolkata": "#EC4899",
    "Mumbai": "#10B981",
    "Pune": "#F59E0B",
}

st.set_page_config(
    page_title="Weather Intelligence | Forecast Analytics",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
      html, body, [class*="css"] {{ font-family: Inter, sans-serif; }}
      .stApp {{ background: linear-gradient(180deg, #EAF3FB 0%, {BG} 45%, #F7FAFD 100%); color: {TEXT}; }}
      .block-container {{ padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1800px; }}
      [data-testid="stSidebar"] {{ background: #E4EEF8; border-right: 1px solid {BORDER}; }}
      [data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; padding: 16px 18px; border-radius: 16px; box-shadow: 0 4px 14px rgba(7,55,99,.06); }}
      [data-testid="stMetricLabel"] {{ color: {MUTED}; font-size: .85rem; font-weight: 600; }}
      [data-testid="stMetricValue"] {{ color: {NAVY}; font-weight: 800; }}
      .hero {{ background: linear-gradient(120deg, {NAVY} 0%, {NAVY_2} 65%, #127EB6 100%); color: white; border-radius: 20px; padding: 22px 26px; margin: 0 0 16px 0; box-shadow: 0 8px 22px rgba(7,55,99,.16); }}
      .hero-title {{ font-size: clamp(1.4rem, 2.2vw, 2.2rem); font-weight: 800; line-height: 1.1; letter-spacing: .01em; margin: 0; }}
      .hero-subtitle {{ color: #D9ECFA; font-size: 1rem; margin-top: 8px; }}
      .section-title {{ font-size: 1.35rem; font-weight: 800; color: {NAVY}; margin: 20px 0 4px 0; }}
      .section-subtitle {{ color: {MUTED}; font-size: .9rem; margin: 0 0 12px 0; }}
      .small-note {{ color: {MUTED}; font-size: .78rem; }}
      div[data-testid="stPlotlyChart"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 16px; padding: 6px; box-shadow: 0 4px 14px rgba(7,55,99,.04); }}
      div[data-testid="stDataFrame"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; }}
      .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
      .stTabs [data-baseweb="tab"] {{ background: white; border-radius: 10px 10px 0 0; padding: 9px 16px; }}
      .footer {{ color: {MUTED}; font-size: .75rem; border-top: 1px solid {BORDER}; margin-top: 28px; padding-top: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def fetch_df(conn, query: str, params: tuple | None = None) -> pd.DataFrame:
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
    return pd.DataFrame(rows, columns=columns)


@st.cache_data(ttl=300, show_spinner=False)
def load_postgres_weather() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (latest per-city snapshot, all retained snapshots)."""
    required = ["DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing = [key for key in required if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Missing database settings in .env: " + ", ".join(missing)
        )

    conn = psycopg2.connect(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        connect_timeout=5,
    )
    try:
        fields = """
            w.city_name,
            w.provider,
            w.forecast_issued_at AS snapshot_timestamp,
            (w.forecast_for AT TIME ZONE 'Asia/Kolkata') AS forecast_for_ist,
            w.temperature_forecast_c,
            w.humidity_forecast_pct,
            w.precipitation_probability_pct,
            w.precipitation_forecast_mm,
            w.wind_speed_forecast_kmh,
            w.weather_code
        """
        all_query = f"""
            SELECT {fields}
            FROM weather_forecasts w
            ORDER BY w.city_name, w.forecast_issued_at, w.forecast_for
        """
        all_snapshots = fetch_df(conn, all_query)
        latest_query = f"""
            WITH latest AS (
                SELECT city_name, MAX(forecast_issued_at) AS latest_issued
                FROM weather_forecasts
                GROUP BY city_name
            )
            SELECT {fields}
            FROM weather_forecasts w
            JOIN latest l
              ON l.city_name = w.city_name
             AND l.latest_issued = w.forecast_issued_at
            ORDER BY w.city_name, w.forecast_for
        """
        latest = fetch_df(conn, latest_query)
    finally:
        conn.close()

    for frame in (latest, all_snapshots):
        if not frame.empty:
            frame["forecast_for_ist"] = pd.to_datetime(frame["forecast_for_ist"], errors="coerce")
            frame["snapshot_timestamp"] = pd.to_datetime(frame["snapshot_timestamp"], errors="coerce", utc=True)
            frame["forecast_date"] = frame["forecast_for_ist"].dt.date
            frame["snapshot_date"] = frame["snapshot_timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    return latest, all_snapshots


def load_current_csv_fallback() -> pd.DataFrame:
    if not CURRENT_CSV.exists():
        return pd.DataFrame()
    df = pd.read_csv(CURRENT_CSV)
    if df.empty:
        return df
    df["forecast_for_ist"] = pd.to_datetime(df["forecast_for"], errors="coerce")
    df["snapshot_timestamp"] = pd.to_datetime(df["forecast_issued_at"], errors="coerce", utc=True)
    df["forecast_date"] = df["forecast_for_ist"].dt.date
    df["snapshot_date"] = df["snapshot_timestamp"].dt.tz_convert("Asia/Kolkata").dt.date
    if "provider" not in df:
        df["provider"] = "open-meteo"
    return df


def load_csv(path: Path, date_cols: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    for col in date_cols or []:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def section_heading(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def apply_plot_style(fig: go.Figure, height: int = 350, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=18, r=18, t=45, b=18),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=TEXT, size=11),
        title_font=dict(size=14, color=NAVY),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text="") if legend else dict(visible=False),
        hoverlabel=dict(bgcolor="white", font_color=TEXT),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=BORDER, title_font=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#E8F0F7", zeroline=False, title_font=dict(size=11))
    return fig


def weather_description(code: Any) -> str:
    try:
        code = int(code)
    except (TypeError, ValueError):
        return "Forecast available"
    descriptions = {
        0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
        45: "Fog", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Drizzle",
        55: "Dense drizzle", 61: "Light rain", 63: "Rain", 65: "Heavy rain",
        71: "Light snow", 73: "Snow", 75: "Heavy snow", 80: "Rain showers",
        81: "Rain showers", 82: "Heavy rain showers", 95: "Thunderstorm",
        96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
    }
    return descriptions.get(code, "Variable conditions")


# -----------------------------------------------------------------------------
# Load local/model data
# -----------------------------------------------------------------------------
latest_weather = pd.DataFrame()
all_snapshots = pd.DataFrame()
db_status = ""
try:
    latest_weather, all_snapshots = load_postgres_weather()
    if latest_weather.empty:
        db_status = "PostgreSQL connected, but weather_forecasts has no rows."
except Exception as exc:
    db_status = f"PostgreSQL connection unavailable: {exc}"
    latest_weather = load_current_csv_fallback()
    all_snapshots = latest_weather.copy()

predictions = load_csv(PREDICTIONS_CSV, ["input_timestamp", "target_timestamp"])
metrics = load_csv(METRICS_CSV)
city_metrics = load_csv(CITY_METRICS_CSV)
importance = load_csv(IMPORTANCE_CSV)

if not predictions.empty:
    needed = {"city_name", "target_timestamp", "actual_temperature_c", "xgboost_prediction_c"}
    if not needed.issubset(predictions.columns):
        predictions = pd.DataFrame()

# -----------------------------------------------------------------------------
# Sidebar filters
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ☰ Dashboard Filters")
if latest_weather.empty:
    st.sidebar.warning("No weather data found. Check PostgreSQL or the current forecast CSV.")
    cities_available: list[str] = []
else:
    cities_available = sorted(latest_weather["city_name"].dropna().unique().tolist())

selected_cities = st.sidebar.multiselect(
    "City name",
    options=cities_available,
    default=cities_available,
    help="Choose one or more cities to filter the weather visuals.",
)

filtered_latest = latest_weather.copy()
if selected_cities and not filtered_latest.empty:
    filtered_latest = filtered_latest[filtered_latest["city_name"].isin(selected_cities)]

if not filtered_latest.empty:
    date_values = pd.to_datetime(filtered_latest["forecast_date"], errors="coerce").dropna()
    min_date = date_values.min().date()
    max_date = date_values.max().date()
    date_range = st.sidebar.date_input(
        "Forecast date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range
    precip_range = st.sidebar.slider(
        "Average precipitation probability (%)",
        min_value=0,
        max_value=100,
        value=(0, 100),
    )
else:
    start_date, end_date = None, None
    precip_range = (0, 100)

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div style="display:flex;align-items:center;gap:18px;flex-wrap:wrap;">
        <div style="font-size:2.5rem">🌦️</div>
        <div>
          <div class="hero-title">WEATHER INTELLIGENCE</div>
          <div class="hero-subtitle">Forecast Analytics Platform | 7 Indian Cities</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if db_status:
    st.info(db_status + (" Showing current CSV data where available." if not latest_weather.empty else ""))

if latest_weather.empty:
    st.error(
        "No weather data is available. Ensure the project .env file is configured, "
        "PostgreSQL is running, and weather_forecasts has data."
    )
    st.stop()

# Apply date and probability filters to latest snapshot.
weather_filtered = filtered_latest.copy()
if start_date is not None and end_date is not None:
    weather_filtered = weather_filtered[
        weather_filtered["forecast_date"].apply(lambda d: start_date <= d <= end_date)
    ]
weather_filtered = weather_filtered[
    weather_filtered["precipitation_probability_pct"].fillna(0).between(*precip_range)
]

# KPIs based on selected latest-snapshot rows.
def safe_mean(frame: pd.DataFrame, column: str) -> float | None:
    if frame.empty or column not in frame:
        return None
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    return None if values.empty else float(values.mean())

k1, k2, k3, k4 = st.columns(4)
k1.metric("Avg Humidity (%)", "—" if safe_mean(weather_filtered, "humidity_forecast_pct") is None else f"{safe_mean(weather_filtered, 'humidity_forecast_pct'):.1f}%")
k2.metric("Avg Wind Speed (km/h)", "—" if safe_mean(weather_filtered, "wind_speed_forecast_kmh") is None else f"{safe_mean(weather_filtered, 'wind_speed_forecast_kmh'):.2f}")
k3.metric("Avg Precipitation Probability", "—" if safe_mean(weather_filtered, "precipitation_probability_pct") is None else f"{safe_mean(weather_filtered, 'precipitation_probability_pct'):.1f}%")
k4.metric("Avg Temperature (°C)", "—" if safe_mean(weather_filtered, "temperature_forecast_c") is None else f"{safe_mean(weather_filtered, 'temperature_forecast_c'):.2f}°C")

# -----------------------------------------------------------------------------
# Weather Overview charts
# -----------------------------------------------------------------------------
section_heading("Weather Overview", "Latest stored forecast snapshot · filtered by city, forecast date and precipitation probability")
if weather_filtered.empty:
    st.warning("No forecast rows match the selected filters. Widen the date or precipitation range.")
else:
    daily = (
        weather_filtered.groupby(["forecast_date", "city_name"], as_index=False)
        .agg(
            avg_temperature_c=("temperature_forecast_c", "mean"),
            avg_humidity_pct=("humidity_forecast_pct", "mean"),
            avg_precipitation_probability_pct=("precipitation_probability_pct", "mean"),
            avg_precipitation_mm=("precipitation_forecast_mm", "sum"),
            avg_wind_speed_kmh=("wind_speed_forecast_kmh", "mean"),
        )
    )
    c1, c2, c3 = st.columns([1.7, 1.0, 1.0])

    with c1:
        fig = px.line(
            daily,
            x="forecast_date",
            y="avg_temperature_c",
            color="city_name",
            color_discrete_map=CITY_COLORS,
            markers=True,
            title="7-Day Temperature Forecast by City",
            labels={"forecast_date": "Forecast date", "avg_temperature_c": "Temperature (°C)", "city_name": "City"},
        )
        fig.update_traces(line=dict(width=2.5), marker=dict(size=5))
        st.plotly_chart(apply_plot_style(fig, 365), use_container_width=True)

    city_weather = (
        weather_filtered.groupby("city_name", as_index=False)
        .agg(
            avg_precipitation_probability_pct=("precipitation_probability_pct", "mean"),
            avg_wind_speed_kmh=("wind_speed_forecast_kmh", "mean"),
        )
    )
    with c2:
        precip_fig = px.bar(
            city_weather.sort_values("avg_precipitation_probability_pct"),
            x="avg_precipitation_probability_pct",
            y="city_name",
            orientation="h",
            title="Avg Precipitation Probability (%)",
            labels={"avg_precipitation_probability_pct": "Probability (%)", "city_name": "City"},
            color_discrete_sequence=[BLUE],
        )
        precip_fig.update_traces(texttemplate="%{x:.1f}%", textposition="outside", cliponaxis=False)
        st.plotly_chart(apply_plot_style(precip_fig, 365, legend=False), use_container_width=True)

    with c3:
        wind_fig = px.bar(
            city_weather.sort_values("avg_wind_speed_kmh"),
            x="avg_wind_speed_kmh",
            y="city_name",
            orientation="h",
            title="Avg Wind Speed (km/h)",
            labels={"avg_wind_speed_kmh": "Wind speed (km/h)", "city_name": "City"},
            color_discrete_sequence=[TEAL],
        )
        wind_fig.update_traces(texttemplate="%{x:.1f}", textposition="outside", cliponaxis=False)
        st.plotly_chart(apply_plot_style(wind_fig, 365, legend=False), use_container_width=True)

    section_heading("Daily Weather Summary", "Daily values computed from the selected latest forecast snapshot")
    summary = daily.copy()
    summary = summary.rename(columns={
        "forecast_date": "Forecast Date",
        "city_name": "City",
        "avg_temperature_c": "Avg Temperature (°C)",
        "avg_humidity_pct": "Avg Humidity (%)",
        "avg_precipitation_probability_pct": "Precipitation Probability (%)",
        "avg_precipitation_mm": "Forecast Precipitation (mm)",
        "avg_wind_speed_kmh": "Avg Wind Speed (km/h)",
    })
    st.dataframe(
        summary.sort_values(["Forecast Date", "City"]).style.format({
            "Avg Temperature (°C)": "{:.2f}",
            "Avg Humidity (%)": "{:.1f}",
            "Precipitation Probability (%)": "{:.1f}",
            "Forecast Precipitation (mm)": "{:.2f}",
            "Avg Wind Speed (km/h)": "{:.2f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

# -----------------------------------------------------------------------------
# Historical forecast snapshots
# -----------------------------------------------------------------------------
section_heading("Forecast History", "Compare temperatures saved by different collection runs")
if not all_snapshots.empty:
    history = all_snapshots.copy()
    if selected_cities:
        history = history[history["city_name"].isin(selected_cities)]
    history["forecast_date"] = pd.to_datetime(history["forecast_for_ist"], errors="coerce").dt.date
    history["snapshot_date"] = pd.to_datetime(history["snapshot_timestamp"], errors="coerce", utc=True).dt.tz_convert("Asia/Kolkata").dt.date
    dates_with_history = sorted(history["forecast_date"].dropna().unique())
    if dates_with_history:
        target_date = st.selectbox(
            "Select one forecast date to compare snapshots",
            options=dates_with_history,
            index=min(len(dates_with_history) - 1, len(dates_with_history) - 1),
            key="history_target_date",
        )
        hist_view = history[history["forecast_date"] == target_date].copy()
        if not hist_view.empty:
            # Use the actual precise collection timestamp: runs for each city can be seconds apart.
            hist_view["snapshot_label"] = pd.to_datetime(hist_view["snapshot_timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
            hist_daily = hist_view.groupby(["snapshot_label", "city_name"], as_index=False).agg(
                avg_temperature_c=("temperature_forecast_c", "mean")
            )
            hist_fig = px.line(
                hist_daily,
                x="snapshot_label",
                y="avg_temperature_c",
                color="city_name",
                color_discrete_map=CITY_COLORS,
                markers=True,
                title=f"Forecast revisions for {target_date}",
                labels={"snapshot_label": "Snapshot collection time (IST)", "avg_temperature_c": "Forecast temperature (°C)", "city_name": "City"},
            )
            hist_fig.update_traces(line=dict(width=2.5), marker=dict(size=7))
            st.plotly_chart(apply_plot_style(hist_fig, 360), use_container_width=True)
            st.caption("The snapshot timestamp is when this project collected the forecast; it is not necessarily the provider's model issue time.")
        else:
            st.info("No snapshot rows are available for this forecast date.")
else:
    st.info("No PostgreSQL snapshot history is available. Connect PostgreSQL to see revisions across runs.")

# -----------------------------------------------------------------------------
# Machine Learning / XGBoost section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero" style="margin-top:22px;">
      <div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">
        <div style="font-size:2.1rem">📈</div>
        <div>
          <div class="hero-title" style="font-size:1.65rem">MODEL PERFORMANCE &amp; PREDICTION</div>
          <div class="hero-subtitle">XGBoost next-hour temperature experiment · chronological test set</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if not predictions.empty:
    xgb_row = metrics[metrics["model"].astype(str).str.lower().str.contains("xgboost", na=False)] if not metrics.empty and "model" in metrics.columns else pd.DataFrame()
    baseline_row = metrics[metrics["model"].astype(str).str.lower().str.contains("baseline", na=False)] if not metrics.empty and "model" in metrics.columns else pd.DataFrame()
    xgb_mae = float(xgb_row.iloc[0]["MAE_C"]) if not xgb_row.empty and "MAE_C" in xgb_row else np.nan
    xgb_rmse = float(xgb_row.iloc[0]["RMSE_C"]) if not xgb_row.empty and "RMSE_C" in xgb_row else np.nan
    base_mae = float(baseline_row.iloc[0]["MAE_C"]) if not baseline_row.empty and "MAE_C" in baseline_row else np.nan
    improvement = (base_mae - xgb_mae) / base_mae * 100 if pd.notna(base_mae) and base_mae > 0 and pd.notna(xgb_mae) else np.nan

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("XGBoost MAE", f"{xgb_mae:.4f}°C" if pd.notna(xgb_mae) else "N/A")
    m2.metric("XGBoost RMSE", f"{xgb_rmse:.4f}°C" if pd.notna(xgb_rmse) else "N/A")
    m3.metric("MAE reduction vs baseline", f"{improvement:.2f}%" if pd.notna(improvement) else "N/A")
    m4.metric("Test records", f"{len(predictions):,}")

    pred_cities = sorted(predictions["city_name"].dropna().unique().tolist())

    # Keep the model section consistent with the sidebar city filter.
    # If the user chooses one city in the sidebar, the ML charts use that city.
    # If several cities are selected, the model selector is limited to that set.
    if selected_cities:
        available_model_cities = [city for city in pred_cities if city in selected_cities]
    else:
        available_model_cities = pred_cities

    if not available_model_cities:
        available_model_cities = pred_cities
        st.info(
            "No prediction rows match the selected sidebar city. "
            "Showing available model cities instead."
        )

    if "ml_city" not in st.session_state or st.session_state["ml_city"] not in available_model_cities:
        st.session_state["ml_city"] = available_model_cities[0]

    ml_city = st.selectbox(
        "City for detailed prediction charts",
        options=available_model_cities,
        key="ml_city",
        help="This selector follows the sidebar city filter; when one city is selected there, this selector is restricted to that city.",
    )
    pred_city = predictions[predictions["city_name"] == ml_city].sort_values("target_timestamp").copy()
    row1a, row1b = st.columns([1.35, 1.0])
    with row1a:
        line = go.Figure()
        line.add_trace(go.Scatter(
            x=pred_city["target_timestamp"], y=pred_city["actual_temperature_c"],
            mode="lines", name="Actual ERA5 temperature", line=dict(color=BLUE, width=2),
        ))
        line.add_trace(go.Scatter(
            x=pred_city["target_timestamp"], y=pred_city["xgboost_prediction_c"],
            mode="lines", name="XGBoost prediction", line=dict(color=ORANGE, width=2),
        ))
        line.update_layout(title=f"Actual vs Predicted Temperature Over Time — {ml_city}", xaxis_title="Target timestamp", yaxis_title="Temperature (°C)")
        st.plotly_chart(apply_plot_style(line, 360), use_container_width=True)
    with row1b:
        sample = pred_city
        if len(sample) > 5000:
            sample = sample.sample(5000, random_state=42)
        lower = float(min(sample["actual_temperature_c"].min(), sample["xgboost_prediction_c"].min()))
        upper = float(max(sample["actual_temperature_c"].max(), sample["xgboost_prediction_c"].max()))
        scatter = px.scatter(
            sample,
            x="actual_temperature_c",
            y="xgboost_prediction_c",
            opacity=0.4,
            title=f"Actual vs Predicted — {ml_city}",
            labels={"actual_temperature_c": "Actual ERA5 temperature (°C)", "xgboost_prediction_c": "XGBoost prediction (°C)"},
        )
        scatter.add_trace(go.Scatter(x=[lower, upper], y=[lower, upper], mode="lines", name="Ideal prediction", line=dict(color=ORANGE, dash="dash")))
        st.plotly_chart(apply_plot_style(scatter, 360), use_container_width=True)

    pred_city["prediction_error_c"] = pred_city["xgboost_prediction_c"] - pred_city["actual_temperature_c"]
    row2a, row2b = st.columns([1.0, 1.4])
    with row2a:
        err_fig = px.histogram(
            pred_city,
            x="prediction_error_c",
            nbins=45,
            title=f"Prediction Error Distribution — {ml_city}",
            labels={"prediction_error_c": "Prediction error (°C)", "count": "Test records"},
            color_discrete_sequence=[CYAN],
        )
        err_fig.add_vline(x=0, line_dash="dash", line_color="#EF4444")
        st.plotly_chart(apply_plot_style(err_fig, 340, legend=False), use_container_width=True)

    with row2b:
        if not city_metrics.empty and {"city_name", "baseline_MAE_C", "xgboost_MAE_C"}.issubset(city_metrics.columns):
            city_plot_data = city_metrics.melt(
                id_vars="city_name",
                value_vars=["baseline_MAE_C", "xgboost_MAE_C"],
                var_name="Model",
                value_name="MAE (°C)",
            )
            city_plot_data["Model"] = city_plot_data["Model"].map({
                "baseline_MAE_C": "Persistence baseline",
                "xgboost_MAE_C": "XGBoost",
            })
            perf_fig = px.bar(
                city_plot_data,
                x="city_name",
                y="MAE (°C)",
                color="Model",
                barmode="group",
                title="Model Performance by City (MAE)",
                color_discrete_map={"Persistence baseline": "#77B7F5", "XGBoost": ORANGE},
                labels={"city_name": "City"},
            )
            st.plotly_chart(apply_plot_style(perf_fig, 340), use_container_width=True)
        else:
            st.info("City-level metrics not found. Run src.train_xgboost to regenerate reports/xgboost_metrics_by_city.csv.")

    if not importance.empty and {"feature", "importance"}.issubset(importance.columns):
        top_importance = importance.sort_values("importance", ascending=False).head(12).sort_values("importance")
        imp_fig = px.bar(
            top_importance,
            x="importance",
            y="feature",
            orientation="h",
            title="XGBoost Feature Importance — Top 12",
            labels={"importance": "Normalized importance", "feature": "Feature"},
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(apply_plot_style(imp_fig, 390, legend=False), use_container_width=True)
    else:
        st.info("Feature importance report not found. Run python -m src.analyze_xgboost.")

    st.caption("Model evaluation uses held-out ERA5 reanalysis targets, not independent weather-station observations. Metrics describe this experiment, not guaranteed real-world forecast accuracy.")
else:
    st.warning(
        "XGBoost test predictions were not found or did not have the expected columns. "
        "Run `python -m src.train_xgboost` to create reports/xgboost_test_predictions.csv."
    )

st.markdown(
    '<div class="footer">Weather Intelligence & Forecast Analytics Platform · Python · PostgreSQL · Plotly · Streamlit · XGBoost</div>',
    unsafe_allow_html=True,
)
