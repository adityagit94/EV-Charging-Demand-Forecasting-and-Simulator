"""Visualization components for the dashboard."""

from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


def create_time_series_plot(
    data: pd.DataFrame,
    x_col: str,
    y_col: str,
    color_col: Optional[str] = None,
    title: str = "Time Series Plot",
) -> go.Figure:
    """Create an interactive time series plot."""
    # Ensure date column is datetime
    if x_col == "date" and not pd.api.types.is_datetime64_any_dtype(data[x_col]):
        data = data.copy()
        data[x_col] = pd.to_datetime(data[x_col])
    
    fig = px.line(data, x=x_col, y=y_col, color=color_col, title=title)
    
    # Update layout
    fig.update_layout(
        height=400,
        hovermode="x unified",
        showlegend=True if color_col else False,
        xaxis_title="Date",
        yaxis_title="Number of Sessions"
    )
    
    # Format date axis if we're plotting dates
    if x_col == "date":
        fig.update_xaxes(
            tickformat="%Y-%m-%d",
            tickmode="auto",
            nticks=10
        )
    
    return fig


def create_correlation_heatmap(
    data: pd.DataFrame, title: str = "Correlation Matrix"
) -> go.Figure:
    """Create a correlation heatmap."""
    # Ensure unique column names
    data = data.loc[:, ~data.columns.duplicated()]
    
    # Calculate correlation matrix
    correlation_matrix = data.corr()

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=correlation_matrix.values,
        x=correlation_matrix.columns,
        y=correlation_matrix.columns,
        colorscale="RdBu_r",
        zmin=-1,
        zmax=1,
        text=np.round(correlation_matrix.values, 2),
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False,
    ))

    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        xaxis=dict(tickangle=45),
        yaxis=dict(tickangle=0),
        width=800
    )

    return fig


def create_feature_importance_plot(
    features: List[str], importance: List[float], title: str = "Feature Importance"
) -> go.Figure:
    """Create a horizontal bar chart for feature importance."""
    df = pd.DataFrame({"feature": features, "importance": importance}).sort_values(
        "importance", ascending=True
    )

    fig = px.bar(df, x="importance", y="feature", orientation="h", title=title)
    fig.update_layout(height=400)
    return fig


def create_demand_heatmap(
    data: pd.DataFrame, title: str = "Demand Heatmap"
) -> go.Figure:
    """Create a heatmap showing demand patterns by hour and day."""
    # Pivot data for heatmap
    heatmap_data = data.pivot_table(
        values="sessions", index="hour_of_day", columns="day_of_week", aggfunc="mean"
    )

    # Day names for columns
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    heatmap_data.columns = [day_names[i] for i in heatmap_data.columns]

    fig = px.imshow(
        heatmap_data,
        title=title,
        labels={"x": "Day of Week", "y": "Hour of Day", "color": "Avg Sessions"},
        color_continuous_scale="Blues",
    )

    fig.update_layout(height=500)
    return fig


def create_distribution_plot(
    data: pd.DataFrame, column: str, title: str = "Distribution Plot"
) -> go.Figure:
    """Create a distribution plot with histogram and box plot."""
    fig = make_subplots(
        rows=2, cols=1, subplot_titles=("Histogram", "Box Plot"), vertical_spacing=0.1
    )

    # Histogram
    fig.add_trace(
        go.Histogram(x=data[column], name="Histogram", nbinsx=30), row=1, col=1
    )

    # Box plot
    fig.add_trace(go.Box(y=data[column], name="Box Plot"), row=2, col=1)

    fig.update_layout(height=600, title_text=title, showlegend=False)
    return fig


def display_metrics_cards(metrics: dict) -> None:
    """Display metrics in a card layout."""
    cols = st.columns(len(metrics))

    for i, (label, value) in enumerate(metrics.items()):
        with cols[i]:
            if isinstance(value, float):
                st.metric(label, f"{value:.2f}")
            else:
                st.metric(label, value)


def create_prediction_comparison_plot(
    actual: List[float],
    predicted: List[float],
    timestamps: Optional[List] = None,
    title: str = "Actual vs Predicted",
) -> go.Figure:
    """Create a comparison plot of actual vs predicted values."""
    if timestamps is None:
        timestamps = list(range(len(actual)))

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=actual,
            mode="lines+markers",
            name="Actual",
            line=dict(color="blue"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=timestamps,
            y=predicted,
            mode="lines+markers",
            name="Predicted",
            line=dict(color="red", dash="dash"),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Sessions",
        height=400,
        hovermode="x unified",
    )

    return fig
