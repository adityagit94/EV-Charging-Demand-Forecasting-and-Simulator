"""Data pipeline utilities for loading, validating, and processing
charging session logs.

This module provides robust data loading and processing capabilities with
comprehensive error handling, data validation, and type safety.
"""

from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
from loguru import logger

from .utils.config import settings


class DataValidationError(Exception):
    """Custom exception for data validation errors."""

    pass


class DataPipeline:
    """Main data pipeline class for EV charging session data processing."""

    def __init__(self, config_path: Optional[str] = None) -> None:
        """Initialize the data pipeline.

        Args:
            config_path: Optional path to configuration file
        """
        self.logger = logger.bind(name=__name__)
        self.required_columns = ["site_id", "timestamp", "sessions"]

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validate the input dataframe.

        Args:
            df: Input dataframe to validate

        Raises:
            DataValidationError: If data validation fails
        """
        if df.empty:
            raise DataValidationError("Input dataframe is empty")

        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise DataValidationError(f"Missing required columns: {missing_cols}")

        # Check for null values in required columns
        null_counts = df[self.required_columns].isnull().sum()
        if null_counts.any():
            self.logger.warning(f"Found null values: {null_counts.to_dict()}")

        # Validate data types
        if not pd.api.types.is_numeric_dtype(df["site_id"]):
            raise DataValidationError("site_id must be numeric")

        if not pd.api.types.is_numeric_dtype(df["sessions"]):
            raise DataValidationError("sessions must be numeric")

        # Check for negative values
        if (df["sessions"] < 0).any():
            self.logger.warning("Found negative session values, will be clipped to 0")

        self.logger.info(f"Data validation passed for {len(df)} rows")

    def load_sessions(
        self,
        path: Optional[Union[str, Path]] = None,
        parse_dates: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Load charging session data from CSV file.

        Args:
            path: Path to CSV file. If None, uses default from config.
            parse_dates: List of columns to parse as dates

        Returns:
            Loaded and validated dataframe

        Raises:
            FileNotFoundError: If the data file doesn't exist
            DataValidationError: If data validation fails
        """
        if path is None:
            path = settings.data.raw_data_path

        if parse_dates is None:
            parse_dates = ["timestamp"]

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Data file not found: {path}. "
                "Run 'python data/synthetic_generator.py' first."
            )

        try:
            self.logger.info(f"Loading data from {path}")
            df = pd.read_csv(path, parse_dates=parse_dates)

            # Validate loaded data
            self.validate_data(df)

            # Clean data
            df = self.clean_data(df)

            self.logger.info(
                f"Successfully loaded {len(df)} sessions from "
                f"{len(df['site_id'].unique())} sites"
            )
            return df

        except Exception as e:
            self.logger.error(f"Failed to load data from {path}: {e}")
            raise

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data.

        Args:
            df: Input dataframe

        Returns:
            Cleaned dataframe
        """
        df = df.copy()

        # Remove duplicates
        initial_rows = len(df)
        df = df.drop_duplicates(subset=["site_id", "timestamp"])
        if len(df) < initial_rows:
            self.logger.info(f"Removed {initial_rows - len(df)} duplicate rows")

        # Clip negative sessions to 0
        df["sessions"] = df["sessions"].clip(lower=0)

        # Sort by site and timestamp
        df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)

        return df

    def aggregate_hourly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate session data to hourly frequency.

        Args:
            df: Input dataframe with session data

        Returns:
            Hourly aggregated dataframe

        Raises:
            DataValidationError: If aggregation fails
        """
        try:
            # Ensure timestamp is datetime
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Create hourly bins
            df = df.assign(hour=df["timestamp"].dt.floor("H"))

            # Aggregate by site and hour
            hourly_df = (
                df.groupby(["site_id", "hour"])
                .agg(
                    {
                        "sessions": ["sum", "count", "mean"],
                        "timestamp": "first",  # Keep original timestamp for reference
                    }
                )
                .reset_index()
            )

            # Flatten column names
            hourly_df.columns = [
                "site_id",
                "hour",
                "sessions",
                "session_count",
                "avg_sessions",
                "first_timestamp",
            ]

            # Fill missing hours with zeros
            hourly_df = self._fill_missing_hours(hourly_df)

            self.logger.info(f"Aggregated to {len(hourly_df)} hourly observations")
            return hourly_df

        except Exception as e:
            self.logger.error(f"Failed to aggregate data: {e}")
            raise DataValidationError(f"Aggregation failed: {e}")

    def _fill_missing_hours(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing hours with zero sessions.

        Args:
            df: Hourly aggregated dataframe

        Returns:
            Dataframe with filled missing hours
        """
        # Create complete hour range for each site
        site_ids = df["site_id"].unique()
        hour_range = pd.date_range(
            start=df["hour"].min(), end=df["hour"].max(), freq="H"
        )

        # Create complete index
        complete_index = pd.MultiIndex.from_product(
            [site_ids, hour_range], names=["site_id", "hour"]
        )

        # Reindex and fill missing values
        df = (
            df.set_index(["site_id", "hour"])
            .reindex(complete_index, fill_value=0)
            .reset_index()
        )

        return df

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """Generate a summary of the dataset.

        Args:
            df: Input dataframe

        Returns:
            Dictionary with data summary statistics
        """
        summary = {
            "total_rows": len(df),
            "unique_sites": df["site_id"].nunique(),
            "date_range": {
                "start": (
                    df["timestamp"].min().isoformat()
                    if "timestamp" in df.columns
                    else df["hour"].min().isoformat()
                ),
                "end": (
                    df["timestamp"].max().isoformat()
                    if "timestamp" in df.columns
                    else df["hour"].max().isoformat()
                ),
            },
            "sessions_stats": {
                "total": df["sessions"].sum(),
                "mean": df["sessions"].mean(),
                "median": df["sessions"].median(),
                "std": df["sessions"].std(),
                "min": df["sessions"].min(),
                "max": df["sessions"].max(),
            },
            "missing_values": df.isnull().sum().to_dict(),
        }

        return summary


# Convenience functions for backward compatibility
def load_sessions(
    path: str = "data/raw/synthetic_sessions.csv", parse_dates: List[str] = None
) -> pd.DataFrame:
    """Load sessions data (backward compatibility function)."""
    if parse_dates is None:
        parse_dates = ["timestamp"]
    pipeline = DataPipeline()
    return pipeline.load_sessions(path, parse_dates)


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate data to hourly frequency (backward compatibility function)."""
    pipeline = DataPipeline()
    return pipeline.aggregate_hourly(df)
