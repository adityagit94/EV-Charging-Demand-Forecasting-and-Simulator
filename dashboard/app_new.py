"""Advanced Streamlit Dashboard for EV Charging Demand Forecasting."""

import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data_pipeline import DataPipeline
from src.features import FeatureEngineer
from components.visualizations import (
    create_time_series_plot, create_correlation_heatmap,
    create_feature_importance_plot, display_metrics_cards
)
from pages.overview import show_overview_page

# Page configuration
st.set_page_config(
    page_title="EV Charging Demand Forecast",
    page_icon="🚗⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🚗⚡ EV Charging Demand Forecast Dashboard</h1>', 
            unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None

# Sidebar for controls
st.sidebar.markdown("## 📊 Dashboard Controls")

# Data management section
st.sidebar.markdown("### 🔄 Data Management")

if st.sidebar.button("🔄 Generate New Synthetic Data", type="primary"):
    with st.spinner("Generating synthetic data..."):
        try:
            import data.synthetic_generator as gen
            gen.generate_synthetic_data(n_sites=8, n_days=60)
            st.sidebar.success("✅ Synthetic data generated successfully!")
            st.session_state.data_loaded = False
        except Exception as e:
            st.sidebar.error(f"❌ Error generating data: {e}")

if st.sidebar.button("📥 Load Data"):
    with st.spinner("Loading and processing data..."):
        try:
            pipeline = DataPipeline()
            engineer = FeatureEngineer()
            
            # Load and process data
            df = pipeline.load_sessions()
            hourly = pipeline.aggregate_hourly(df)
            hourly['hour'] = pd.to_datetime(hourly['hour'])
            
            # Add features
            hourly = engineer.create_feature_pipeline(hourly)
            
            # Store in session state
            st.session_state.processed_data = hourly
            st.session_state.data_loaded = True
            st.sidebar.success("✅ Data loaded and processed successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading data: {e}")

# Main content
if not st.session_state.data_loaded or st.session_state.processed_data is None:
    st.warning("⚠️ Please load data first using the sidebar controls.")
    
    st.markdown("""
    ## 🚀 Getting Started
    
    1. **Generate Data**: Click "Generate New Synthetic Data" in the sidebar
    2. **Load Data**: Click "Load Data" to process and load the dataset
    3. **Explore**: Use the various tabs below to explore your data and models
    """)
    
else:
    hourly = st.session_state.processed_data
    
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["📈 Overview", "🔍 Site Analysis", "📊 Features"])
    
    with tab1:
        show_overview_page(hourly)
    
    with tab2:
        st.markdown("## 🔍 Site-Specific Analysis")
        
        # Site selection
        selected_sites = st.multiselect(
            "Select Sites",
            options=sorted(hourly['site_id'].unique()),
            default=sorted(hourly['site_id'].unique())[:3]
        )
        
        if selected_sites:
            site_data = hourly[hourly['site_id'].isin(selected_sites)].copy()
            
            # Time series comparison
            fig = create_time_series_plot(
                site_data, 'hour', 'sessions', 'site_id',
                'Charging Sessions Over Time by Site'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## 📊 Feature Analysis")
        
        # Feature correlation
        feature_cols = [col for col in hourly.columns 
                       if col not in ['site_id', 'hour', 'sessions']]
        feature_data = hourly[feature_cols + ['sessions']].select_dtypes(include=['number'])
        
        fig = create_correlation_heatmap(feature_data, 'Feature Correlation Matrix')
        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🚗⚡ EV Charging Demand Forecast Dashboard | Built with Streamlit & ❤️</p>
</div>
""", unsafe_allow_html=True)
