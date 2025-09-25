"""Overview page for the dashboard."""

import pandas as pd
import streamlit as st
from datetime import timedelta
from ..components.visualizations import (
    create_time_series_plot,
    display_metrics_cards,
    create_demand_heatmap,
)


def show_overview_page(data: pd.DataFrame) -> None:
    """Display the overview page."""
    st.markdown("## 📈 Data Overview")

    # Key metrics
    metrics = {
        "Total Sites": data["site_id"].nunique(),
        "Total Hours": len(data),
        "Avg Sessions/Hour": data["sessions"].mean(),
        "Max Sessions/Hour": data["sessions"].max(),
    }

    display_metrics_cards(metrics)

    # Time series overview
    st.markdown("### 📊 Overall Demand Pattern")

    # Aggregate across all sites
    daily_demand = data.groupby(data["hour"].dt.date)["sessions"].sum().reset_index()
    daily_demand.columns = ["date", "total_sessions"]

    fig = create_time_series_plot(
        daily_demand,
        "date",
        "total_sessions",
        title="Daily Total Charging Sessions Across All Sites",
    )
    st.plotly_chart(fig, use_container_width=True)

    # Hourly patterns
    st.markdown("### ⏰ Demand Patterns")

    col1, col2 = st.columns(2)

    with col1:
        # Demand heatmap
        fig = create_demand_heatmap(data, "Average Sessions by Hour and Day")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Peak hours analysis
        st.markdown("#### 🔥 Peak Hours")

        hourly_avg = data.groupby("hour_of_day")["sessions"].mean().reset_index()
        top_hours = hourly_avg.nlargest(5, "sessions")

        for _, row in top_hours.iterrows():
            hour = int(row["hour_of_day"])
            sessions = row["sessions"]
            st.write(f"**{hour:02d}:00** - {sessions:.2f} sessions")

        # Weekend vs Weekday
        st.markdown("#### 📅 Weekend vs Weekday")
        weekend_avg = data[data["is_weekend"] == 1]["sessions"].mean()
        weekday_avg = data[data["is_weekend"] == 0]["sessions"].mean()

        st.write(f"**Weekday Average**: {weekday_avg:.2f}")
        st.write(f"**Weekend Average**: {weekend_avg:.2f}")

        if weekend_avg > weekday_avg:
            st.write("📈 Weekends are busier")
        else:
            st.write("📈 Weekdays are busier")
