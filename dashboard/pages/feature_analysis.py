"""Feature analysis page for the dashboard."""

from typing import Dict, List
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from components.visualizations import (
    create_correlation_heatmap,
    create_time_series_plot,
    display_metrics_cards
)

def create_feature_importance_plot(data: pd.DataFrame) -> go.Figure:
    """Create an enhanced feature importance visualization."""
    # Calculate feature correlations with sessions
    correlations = data.corr()["sessions"].sort_values(ascending=True)
    correlations = correlations.drop("sessions")  # Remove self-correlation
    
    # Create bar plot
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=correlations.index,
        x=correlations.values,
        orientation="h",
        marker_color=np.where(correlations > 0, "#2ecc71", "#e74c3c"),
        text=correlations.round(3),
        textposition="auto",
    ))
    
    # Update layout
    fig.update_layout(
        title="Feature Importance (Correlation with Sessions)",
        xaxis_title="Correlation Coefficient",
        yaxis_title="Feature",
        height=600,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    # Add vertical line at x=0
    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    return fig

def create_temporal_pattern_plot(data: pd.DataFrame) -> go.Figure:
    """Create temporal pattern visualization."""
    # Create subplots for different temporal patterns
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Hourly Pattern", 
            "Daily Pattern",
            "Monthly Pattern",
            "Day of Week Pattern"
        )
    )
    
    # Hourly pattern
    hourly_avg = data.groupby("hour_of_day")["sessions"].mean()
    fig.add_trace(
        go.Scatter(x=hourly_avg.index, y=hourly_avg.values, name="Hourly",
                  line=dict(color="#3498db")),
        row=1, col=1
    )
    
    # Daily pattern
    daily_avg = data.groupby("day_of_month")["sessions"].mean()
    fig.add_trace(
        go.Scatter(x=daily_avg.index, y=daily_avg.values, name="Daily",
                  line=dict(color="#2ecc71")),
        row=1, col=2
    )
    
    # Monthly pattern
    monthly_avg = data.groupby("month")["sessions"].mean()
    fig.add_trace(
        go.Scatter(x=monthly_avg.index, y=monthly_avg.values, name="Monthly",
                  line=dict(color="#e67e22")),
        row=2, col=1
    )
    
    # Day of week pattern
    dow_avg = data.groupby("day_of_week")["sessions"].mean()
    dow_labels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    fig.add_trace(
        go.Scatter(x=dow_labels, y=dow_avg.values, name="Day of Week",
                  line=dict(color="#9b59b6")),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        title_text="Temporal Patterns in EV Charging Sessions",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def create_feature_distribution_plots(data: pd.DataFrame, features: List[str]) -> go.Figure:
    """Create distribution plots for selected features."""
    fig = make_subplots(
        rows=len(features), cols=1,
        subplot_titles=[f.replace("_", " ").title() for f in features],
        vertical_spacing=0.05
    )
    
    for i, feature in enumerate(features, 1):
        fig.add_trace(
            go.Histogram(
                x=data[feature],
                name=feature,
                nbinsx=30,
                marker_color=px.colors.qualitative.Set3[i % len(px.colors.qualitative.Set3)]
            ),
            row=i, col=1
        )
    
    fig.update_layout(
        height=200 * len(features),
        showlegend=False,
        title_text="Feature Distributions",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    
    return fig

def show_feature_analysis(data: pd.DataFrame) -> None:
    """Display the feature analysis page with enhanced visualizations."""
    st.markdown("## 📊 Feature Analysis")
    
    # Key metrics about features
    metrics = {
        "Total Features": len(data.columns),
        "Temporal Features": len([col for col in data.columns if any(x in col for x in ["hour", "day", "month", "year"])]),
        "Rolling Features": len([col for col in data.columns if "r" in col and any(x in col for x in ["mean", "std", "min", "max"])]),
        "Lag Features": len([col for col in data.columns if "lag" in col])
    }
    
    display_metrics_cards(metrics)
    
    # Feature Importance
    st.markdown("### 📈 Feature Importance")
    st.markdown("""
    This plot shows how strongly each feature correlates with charging session demand.
    - **Positive values (green)** indicate features that increase with demand
    - **Negative values (red)** indicate features that decrease with demand
    """)
    fig_importance = create_feature_importance_plot(data)
    st.plotly_chart(fig_importance, use_container_width=True)
    
    # Temporal Patterns
    st.markdown("### 🕒 Temporal Patterns")
    st.markdown("""
    These plots show how charging demand varies across different time periods:
    - **Hourly**: Patterns within a day
    - **Daily**: Patterns within a month
    - **Monthly**: Seasonal patterns
    - **Day of Week**: Weekly patterns
    """)
    fig_temporal = create_temporal_pattern_plot(data)
    st.plotly_chart(fig_temporal, use_container_width=True)
    
    # Feature Correlations
    st.markdown("### 🔄 Feature Correlations")
    st.markdown("""
    The correlation matrix shows relationships between different features:
    - **Dark Red**: Strong positive correlation
    - **Dark Blue**: Strong negative correlation
    - **White**: No correlation
    """)
    
    # Allow users to filter correlation matrix
    feature_types = {
        "Temporal": ["hour", "day", "month", "year"],
        "Rolling Stats": ["rmean", "rstd", "rmin", "rmax"],
        "Lag": ["lag"],
        "Binary": ["is_"],
    }
    
    selected_types = st.multiselect(
        "Filter Feature Types:",
        options=list(feature_types.keys()),
        default=["Temporal", "Rolling Stats"]
    )
    
    # Filter columns based on selection
    selected_cols = []
    for type_name in selected_types:
        patterns = feature_types[type_name]
        selected_cols.extend([
            col for col in data.columns
            if any(pattern in col for pattern in patterns)
        ])
    
    if selected_cols:
        # Ensure unique columns and add sessions if not already included
        selected_cols = list(dict.fromkeys(selected_cols))  # Remove duplicates while preserving order
        if "sessions" not in selected_cols:
            selected_cols.append("sessions")
            
        fig_corr = create_correlation_heatmap(
            data[selected_cols],
            title="Feature Correlation Matrix"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    # Feature Distributions
    st.markdown("### 📊 Feature Distributions")
    st.markdown("""
    Examine the distribution of key features to understand their patterns and ranges.
    Select features to visualize their distributions.
    """)
    
    selected_features = st.multiselect(
        "Select features to view distributions:",
        options=[col for col in data.columns if col != "sessions"],
        default=["hour_of_day", "day_of_week", "is_weekend"]
    )
    
    if selected_features:
        fig_dist = create_feature_distribution_plots(data, selected_features)
        st.plotly_chart(fig_dist, use_container_width=True)