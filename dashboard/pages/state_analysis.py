"""State-wise analysis page for EV charging demand in India."""

from datetime import datetime

import pandas as pd
import streamlit as st
from components.india_visualizations import (create_demand_by_region,
                                             create_india_choropleth,
                                             create_peak_hours_heatmap,
                                             prepare_station_data)
from components.visualizations import display_metrics_cards


def get_state_metrics(data: pd.DataFrame) -> dict:
    """Calculate key metrics for state-wise analysis."""
    current_hour = datetime.now().strftime("%H:00")

    return {
        "Total States": data["state"].nunique() if "state" in data.columns else 0,
        "Active Stations": data["site_id"].nunique(),
        f"Current Sessions ({current_hour})": data[
            data["hour"].dt.strftime("%H:00") == current_hour
        ]["sessions"].mean(),
        "Avg Daily Sessions": data.groupby(data["hour"].dt.date)["sessions"]
        .mean()
        .mean(),
    }


def show_state_analysis(data: pd.DataFrame) -> None:
    """Display the state-wise analysis page."""
    st.markdown("## 🗺️ State-wise Analysis")

    # Calculate and display key metrics
    metrics = get_state_metrics(data)
    display_metrics_cards(metrics)

    # Create and display the map
    if "state" in data.columns:
        # Aggregate data by state
        state_data = (
            data.groupby("state")
            .agg({"site_id": "nunique", "sessions": ["mean", "max"]})
            .reset_index()
        )

        state_data.columns = [
            "state",
            "total_stations",
            "mean_sessions",
            "peak_sessions",
        ]

        # Prepare station data
        station_data = prepare_station_data(data)

        # Create choropleth map with station markers
        fig = create_india_choropleth(
            state_data,
            state_column="state",
            value_column="total_stations",
            title="EV Charging Infrastructure by State",
            show_stations=True,
            station_data=station_data,
            hover_data=["mean_sessions", "peak_sessions"],
        )
        st.plotly_chart(fig, use_container_width=True)

        # Add filters and analysis sections
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📊 Demand by State")
            demand_fig = create_demand_by_region(
                state_data,
                region_column="state",
                demand_column="mean_sessions",
                title="Average Daily Sessions by State",
            )
            st.plotly_chart(demand_fig, use_container_width=True)

        with col2:
            st.markdown("### ⚡ Peak Hours by Region")
            # Calculate hourly averages by state
            hourly_data = (
                data.groupby(["state", data["hour"].dt.hour])["sessions"]
                .mean()
                .reset_index()
            )

            peak_hours_fig = create_peak_hours_heatmap(
                hourly_data,
                region_column="state",
                hour_column="hour",
                demand_column="sessions",
                title="Hourly Demand Patterns by State",
            )
            st.plotly_chart(peak_hours_fig, use_container_width=True)

        # Time-based patterns
        st.markdown("### 📅 Temporal Analysis")

        # Time range selector
        date_range = st.date_input(
            "Select Date Range",
            value=(data["hour"].min().date(), data["hour"].max().date()),
        )

        if len(date_range) == 2:
            start_date, end_date = date_range
            mask = (data["hour"].dt.date >= start_date) & (
                data["hour"].dt.date <= end_date
            )
            filtered_data = data[mask]

            # Aggregate daily demand by state
            # Aggregate daily demand by state
            daily_data = (
                filtered_data.groupby([filtered_data["hour"].dt.date, "state"])[
                    "sessions"
                ]
                .sum()
                .reset_index(name="sessions")
            )
            daily_data.rename(
                columns={filtered_data["hour"].dt.date.name: "date"}, inplace=True
            )

            # Create time series plot
            from components.visualizations import create_time_series_plot

            time_fig = create_time_series_plot(
                daily_data,
                x_col="date",
                y_col="sessions",
                color_col="state",
                title="Daily Session Trends by State",
            )
            st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.warning(
            "State information not available in the dataset. Please ensure your data includes state-level information."
        )
