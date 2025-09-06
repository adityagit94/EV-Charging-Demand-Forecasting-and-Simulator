"""Advanced feature engineering for EV charging demand forecasting.

This module provides comprehensive feature engineering capabilities including:
- Temporal features (cyclical encodings, calendar features)
- Lag features (autoregressive patterns)
- Rolling statistics (trends and volatility)
- Spatial features (location-based patterns)
"""

from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

from .utils.config import settings


class FeatureEngineer:
    """Advanced feature engineering class for time series forecasting."""
    
    def __init__(self) -> None:
        """Initialize the feature engineer."""
        self.logger = logger.bind(name=__name__)
        self.feature_columns = []
        
    def add_temporal_features(
        self, 
        df: pd.DataFrame, 
        ts_col: str = 'hour',
        include_cyclical: bool = True,
        include_calendar: bool = True
    ) -> pd.DataFrame:
        """Add comprehensive temporal features.
        
        Args:
            df: Input dataframe
            ts_col: Name of timestamp column
            include_cyclical: Whether to include cyclical encodings
            include_calendar: Whether to include calendar features
            
        Returns:
            Dataframe with added temporal features
        """
        df = df.copy()
        
        # Basic time features
        df['hour_of_day'] = df[ts_col].dt.hour
        df['day_of_week'] = df[ts_col].dt.dayofweek
        df['day_of_month'] = df[ts_col].dt.day
        df['month'] = df[ts_col].dt.month
        df['quarter'] = df[ts_col].dt.quarter
        df['year'] = df[ts_col].dt.year
        
        # Binary features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_monday'] = (df['day_of_week'] == 0).astype(int)
        df['is_friday'] = (df['day_of_week'] == 4).astype(int)
        
        # Business hours
        df['is_business_hours'] = (
            (df['hour_of_day'] >= 9) & 
            (df['hour_of_day'] <= 17) & 
            (df['day_of_week'] < 5)
        ).astype(int)
        
        # Peak hours (morning and evening commute)
        df['is_morning_peak'] = (
            (df['hour_of_day'] >= 7) & (df['hour_of_day'] <= 9)
        ).astype(int)
        df['is_evening_peak'] = (
            (df['hour_of_day'] >= 17) & (df['hour_of_day'] <= 19)
        ).astype(int)
        
        if include_cyclical:
            # Cyclical encodings for periodic features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
        if include_calendar:
            # Holiday indicators (simplified - can be extended with actual holiday calendar)
            df['is_holiday'] = 0  # Placeholder for holiday detection
            
        self.logger.info(f"Added temporal features, shape: {df.shape}")
        return df
        
    def add_lag_features(
        self, 
        df: pd.DataFrame,
        group_col: str = 'site_id',
        target: str = 'sessions',
        lag_periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add lag features for autoregressive patterns.
        
        Args:
            df: Input dataframe
            group_col: Column to group by (e.g., site_id)
            target: Target column to create lags for
            lag_periods: List of lag periods. If None, uses config defaults.
            
        Returns:
            Dataframe with added lag features
        """
        if lag_periods is None:
            lag_periods = settings.features.lag_hours
            
        df = df.copy()
        df = df.sort_values([group_col, 'hour'])
        
        for lag in lag_periods:
            col_name = f'lag_{lag}'
            df[col_name] = df.groupby(group_col)[target].shift(lag)
            
        self.logger.info(f"Added lag features for periods: {lag_periods}")
        return df
        
    def add_rolling_features(
        self,
        df: pd.DataFrame,
        group_col: str = 'site_id',
        target: str = 'sessions',
        windows: Optional[List[int]] = None,
        include_std: bool = True
    ) -> pd.DataFrame:
        """Add rolling statistical features.
        
        Args:
            df: Input dataframe
            group_col: Column to group by
            target: Target column for rolling statistics
            windows: List of rolling window sizes. If None, uses config defaults.
            include_std: Whether to include standard deviation features
            
        Returns:
            Dataframe with added rolling features
        """
        if windows is None:
            windows = settings.features.rolling_windows
            
        df = df.copy()
        df = df.sort_values([group_col, 'hour'])
        
        for window in windows:
            # Rolling mean
            df[f'rmean_{window}'] = (
                df.groupby(group_col)[target]
                .rolling(window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            # Rolling standard deviation
            if include_std:
                df[f'rstd_{window}'] = (
                    df.groupby(group_col)[target]
                    .rolling(window, min_periods=1)
                    .std()
                    .reset_index(0, drop=True)
                )
                
            # Rolling min/max
            df[f'rmin_{window}'] = (
                df.groupby(group_col)[target]
                .rolling(window, min_periods=1)
                .min()
                .reset_index(0, drop=True)
            )
            
            df[f'rmax_{window}'] = (
                df.groupby(group_col)[target]
                .rolling(window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )
            
        self.logger.info(f"Added rolling features for windows: {windows}")
        return df
        
    def add_difference_features(
        self,
        df: pd.DataFrame,
        group_col: str = 'site_id',
        target: str = 'sessions',
        periods: Optional[List[int]] = None
    ) -> pd.DataFrame:
        """Add difference features to capture trends.
        
        Args:
            df: Input dataframe
            group_col: Column to group by
            target: Target column
            periods: List of difference periods
            
        Returns:
            Dataframe with added difference features
        """
        if periods is None:
            periods = [1, 24]  # 1-hour and 24-hour differences
            
        df = df.copy()
        df = df.sort_values([group_col, 'hour'])
        
        for period in periods:
            df[f'diff_{period}'] = df.groupby(group_col)[target].diff(period)
            
        self.logger.info(f"Added difference features for periods: {periods}")
        return df
        
    def add_interaction_features(
        self,
        df: pd.DataFrame,
        feature_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> pd.DataFrame:
        """Add interaction features between existing features.
        
        Args:
            df: Input dataframe
            feature_pairs: List of feature pairs to create interactions for
            
        Returns:
            Dataframe with added interaction features
        """
        if feature_pairs is None:
            feature_pairs = [
                ('hour_of_day', 'is_weekend'),
                ('is_business_hours', 'day_of_week'),
                ('lag_1', 'hour_of_day')
            ]
            
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
        self.logger.info(f"Added {len(feature_pairs)} interaction features")
        return df
        
    def create_feature_pipeline(
        self,
        df: pd.DataFrame,
        ts_col: str = 'hour',
        group_col: str = 'site_id',
        target: str = 'sessions',
        include_interactions: bool = False
    ) -> pd.DataFrame:
        """Create complete feature engineering pipeline.
        
        Args:
            df: Input dataframe
            ts_col: Timestamp column name
            group_col: Grouping column name
            target: Target column name
            include_interactions: Whether to include interaction features
            
        Returns:
            Dataframe with all engineered features
        """
        self.logger.info("Starting feature engineering pipeline")
        
        # Add temporal features
        df = self.add_temporal_features(df, ts_col)
        
        # Add lag features
        df = self.add_lag_features(df, group_col, target)
        
        # Add rolling features
        df = self.add_rolling_features(df, group_col, target)
        
        # Add difference features
        df = self.add_difference_features(df, group_col, target)
        
        # Add interaction features (optional)
        if include_interactions:
            df = self.add_interaction_features(df)
            
        self.logger.info(f"Feature engineering complete, final shape: {df.shape}")
        return df
        
    def get_feature_importance_names(self) -> List[str]:
        """Get list of engineered feature names for model training.
        
        Returns:
            List of feature column names
        """
        base_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'is_business_hours',
            'is_morning_peak', 'is_evening_peak', 'hour_sin', 'hour_cos',
            'day_sin', 'day_cos'
        ]
        
        # Add lag features
        for lag in settings.features.lag_hours:
            base_features.append(f'lag_{lag}')
            
        # Add rolling features
        for window in settings.features.rolling_windows:
            base_features.extend([
                f'rmean_{window}', f'rstd_{window}',
                f'rmin_{window}', f'rmax_{window}'
            ])
            
        return base_features


# Backward compatibility functions
def add_time_features(df: pd.DataFrame, ts_col: str = 'hour') -> pd.DataFrame:
    """Add time features (backward compatibility function)."""
    engineer = FeatureEngineer()
    return engineer.add_temporal_features(df, ts_col)


def add_lag_features(
    df: pd.DataFrame, 
    group_col: str = 'site_id', 
    target: str = 'sessions'
) -> pd.DataFrame:
    """Add lag features (backward compatibility function)."""
    engineer = FeatureEngineer()
    return engineer.add_lag_features(df, group_col, target)
