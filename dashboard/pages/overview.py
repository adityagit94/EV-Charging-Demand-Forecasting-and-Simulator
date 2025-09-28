"""Overview page for the dashboard."""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from components.visualizations import (create_demand_heatmap,
                                       create_time_series_plot,
                                       display_metrics_cards)
from plotly.subplots import make_subplots


def create_overview_metrics(data: pd.DataFrame) -> dict:
    """Calculate overview metrics with trends."""
    current_date = data["hour"].dt.date.max()
    previous_date = current_date - timedelta(days=1)

    # Current metrics
    current_data = data[data["hour"].dt.date == current_date]
    previous_data = data[data["hour"].dt.date == previous_date]

    return {
        "Active Charging Sites": {
            "value": f"📍 {data['site_id'].nunique()}",
            "change": f"+{len(set(current_data['site_id']) - set(previous_data['site_id']))}",
        },
        "Total Sessions Today": {
            "value": f"🔌 {int(current_data['sessions'].sum()):,}",
            "change": f"{((current_data['sessions'].sum() / previous_data['sessions'].sum() - 1) * 100):.1f}%",
        },
        "Peak Hour Demand": {
            "value": f"⚡ {current_data.groupby(current_data['hour'].dt.hour)['sessions'].sum().max():.0f}",
            "change": "Peak at "
            + str(
                current_data.groupby(current_data["hour"].dt.hour)["sessions"]
                .sum()
                .idxmax()
            )
            + ":00",
        },
        "Network Utilization": {
            "value": f"📊 {(current_data['sessions'].mean() * 100):.1f}%",
            "change": f"{((current_data['sessions'].mean() / previous_data['sessions'].mean() - 1) * 100):.1f}%",
        },
    }


def show_overview_page(data: pd.DataFrame) -> None:
    """Display the enhanced overview page."""
    # Custom CSS
    st.markdown(
        """
        <style>
        .metric-card {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            text-align: center;
            border: 1px solid #e9ecef;
            transition: transform 0.3s ease;
        }
        .metric-card:hover {
            transform: translateY(-5px);
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
            color: #2c3e50;
            margin: 0;
        }
        .metric-label {
            color: #7f8c8d;
            margin: 0.5rem 0;
        }
        .metric-change {
            font-size: 0.9rem;
            margin: 0;
        }
        .positive-change {
            color: #2ecc71;
        }
        .negative-change {
            color: #e74c3c;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Header with current time
    st.markdown(
        f"""
        <div style='text-align: center; margin-bottom: 2rem;'>
            <h1 style='color: #3498db;'>🚗⚡ EV Charging Network Overview</h1>
            <p style='color: #7f8c8d;'>Real-time monitoring and analytics dashboard</p>
            <p style='color: #95a5a6; font-size: 0.9rem;'>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Enhanced metrics display
    metrics = create_overview_metrics(data)

    # Create a row of metric cards
    cols = st.columns(len(metrics))
    for col, (label, metric) in zip(cols, metrics.items()):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <p class="metric-value">{metric['value']}</p>
                    <p class="metric-label">{label}</p>
                    <p class="metric-change {'positive-change' if not metric['change'].startswith('-') else 'negative-change'}">
                        {metric['change']}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

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
