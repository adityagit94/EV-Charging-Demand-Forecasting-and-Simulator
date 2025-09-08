import streamlit as st
import pandas as pd
import altair as alt
from ev_forecast.data_pipeline import load_sessions, aggregate_hourly
from ev_forecast.features import add_time_features, add_lag_features

st.set_page_config(layout="wide", page_title="EV Charging Forecast Dashboard")

st.title("EV Charging Demand Forecast - Demo")

st.markdown(
    "This dashboard shows the synthetic dataset and example site forecasts (XGBoost baseline)."
)

if st.button("Generate / load synthetic data"):
    import data.synthetic_generator as gen

    gen.generate_synthetic_data(n_sites=6, n_days=45)

df = load_sessions()
hourly = aggregate_hourly(df)
hourly["hour"] = pd.to_datetime(hourly["hour"])
hourly = add_time_features(hourly, ts_col="hour")
hourly = add_lag_features(hourly, group_col="site_id", target="sessions")
hourly = hourly.dropna().reset_index(drop=True)

site = st.selectbox("Select site", sorted(hourly["site_id"].unique()))
site_df = hourly[hourly["site_id"] == site].copy()
st.subheader(f"Site {site} - timeseries (last 7 days)")
site_df = site_df.sort_values("hour")
chart = alt.Chart(site_df.tail(24 * 7)).mark_line().encode(x="hour:T", y="sessions:Q")
st.altair_chart(chart, use_container_width=True)

st.markdown(
    "Use the `src/models/train_xgboost.py` script to train a model and the FastAPI to request predictions."
)
