"""India-specific visualization components for the dashboard."""

from typing import Dict, List, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ev_forecast.utils.state_mapping import SITE_TO_STATE_MAPPING
from ev_forecast.utils.station_coordinates import STATION_COORDINATES


def load_india_geojson() -> gpd.GeoDataFrame:
    """Load and cache India state boundaries GeoJSON data."""

    # Use st.cache_data to prevent reloading on every run
    @st.cache_data
    def _load_geojson():
        # TODO: Replace with actual India GeoJSON data source
        return gpd.read_file(
            "https://raw.githubusercontent.com/Subhash9325/GeoJson-Data-of-Indian-States/master/Indian_States"
        )

    return _load_geojson()


def prepare_station_data(data: pd.DataFrame) -> pd.DataFrame:
    """Prepare charging station data for visualization.

    Args:
        data: DataFrame containing charging station data

    Returns:
        DataFrame with station coordinates and status
    """
    # Create stations dataframe
    stations_df = (
        pd.DataFrame.from_dict(STATION_COORDINATES, orient="index")
        .reset_index()
        .rename(columns={"index": "site_id"})
    )

    # Calculate current status for each station
    if not data.empty:
        current_hour = data["hour"].max()
        current_data = data[data["hour"] == current_hour]

        # Merge with current data
        stations_df = stations_df.merge(
            current_data[["site_id", "sessions"]], on="site_id", how="left"
        )

        # Determine station status based on activity
        stations_df["status"] = np.where(
            stations_df["sessions"] > 0, "Active", "Inactive"
        )

        # Create status color mapping
        stations_df["color"] = np.where(
            stations_df["status"] == "Active",
            "#2ecc71",  # Green for active
            "#e74c3c",  # Red for inactive
        )
    else:
        stations_df["status"] = "Unknown"
        stations_df["color"] = "#95a5a6"  # Gray for unknown

    return stations_df


def create_india_choropleth(
    data: pd.DataFrame,
    state_column: str,
    value_column: str,
    title: str,
    show_stations: bool = True,
    station_data: Optional[pd.DataFrame] = None,
    hover_data: Optional[List[str]] = None,
) -> go.Figure:
    """Create an interactive choropleth map of Indian states with charging stations.

    Args:
        data: DataFrame containing state-wise data
        state_column: Name of the column containing state names
        value_column: Name of the column to use for coloring
        title: Title of the map
        show_stations: Whether to show charging station markers
        station_data: DataFrame containing station information
        hover_data: Additional columns to show in hover tooltip

    Returns:
        Plotly figure object
    """
    india_states = load_india_geojson()

    # Create base choropleth
    fig = px.choropleth(
        data,
        geojson=india_states.__geo_interface__,
        locations=state_column,
        color=value_column,
        hover_name=state_column,
        hover_data=hover_data,
        title=title,
        color_continuous_scale="Viridis",
    )

    # Add charging station markers if requested
    if show_stations and station_data is not None:
        fig.add_trace(
            go.Scattergeo(
                lon=station_data["lon"],
                lat=station_data["lat"],
                text=station_data.apply(
                    lambda x: f"{x['name']}<br>Status: {x['status']}", axis=1
                ),
                mode="markers",
                marker=dict(
                    size=10,
                    color=station_data["color"],
                    symbol="circle",
                    line=dict(width=1, color="white"),
                ),
                name="Charging Stations",
                hoverinfo="text",
            )
        )

    # Update map layout
    fig.update_geos(
        center=dict(lon=78.9629, lat=20.5937),  # Center of India
        projection_scale=4,  # Zoom level
        showcoastlines=True,
        coastlinecolor="Black",
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="lightblue",
    )

    # Customize the map appearance
    fig.update_geos(
        visible=False,
        projection=dict(
            type="mercator",
            scale=1.5,
        ),
        center=dict(lat=20.5937, lon=78.9629),  # Center of India
        showcoastlines=True,
    )

    # Update layout
    fig.update_layout(
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        height=600,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    return fig


def create_station_markers(
    data: pd.DataFrame,
    lat_column: str,
    lon_column: str,
    hover_data: Optional[List[str]] = None,
) -> go.Figure:
    """Create a scatter map of charging stations.

    Args:
        data: DataFrame containing station locations
        lat_column: Name of the column containing latitude
        lon_column: Name of the column containing longitude
        hover_data: Additional columns to show in hover tooltip

    Returns:
        Plotly figure object
    """
    fig = px.scatter_mapbox(
        data,
        lat=lat_column,
        lon=lon_column,
        hover_data=hover_data,
        zoom=4,
        mapbox_style="carto-positron",
    )

    # Center on India
    fig.update_layout(
        mapbox=dict(
            center=dict(lat=20.5937, lon=78.9629),
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        height=600,
    )

    return fig


def create_demand_by_region(
    data: pd.DataFrame,
    region_column: str,
    demand_column: str,
    title: str = "Charging Demand by Region",
) -> go.Figure:
    """Create a bar chart showing demand by region.

    Args:
        data: DataFrame containing regional demand data
        region_column: Name of the column containing region names
        demand_column: Name of the column containing demand values
        title: Title of the chart

    Returns:
        Plotly figure object
    """
    fig = px.bar(
        data.sort_values(demand_column, ascending=True),
        y=region_column,
        x=demand_column,
        orientation="h",
        title=title,
    )

    fig.update_layout(
        xaxis_title="Total Demand (kWh)",
        yaxis_title="Region",
        height=400,
    )

    return fig


def create_peak_hours_heatmap(
    data: pd.DataFrame,
    region_column: str,
    hour_column: str,
    demand_column: str,
    title: str = "Peak Hours by Region",
) -> go.Figure:
    """Create a heatmap showing peak hours by region.

    Args:
        data: DataFrame containing hourly demand data
        region_column: Name of the column containing region names
        hour_column: Name of the column containing hour information
        demand_column: Name of the column containing demand values
        title: Title of the heatmap

    Returns:
        Plotly figure object
    """
    # Pivot data for heatmap
    pivot_data = data.pivot_table(
        values=demand_column,
        index=region_column,
        columns=hour_column,
        aggfunc="mean",
    )

    fig = px.imshow(
        pivot_data,
        title=title,
        color_continuous_scale="Viridis",
        aspect="auto",
    )

    fig.update_layout(
        xaxis_title="Hour of Day",
        yaxis_title="Region",
        height=400,
    )

    return fig
