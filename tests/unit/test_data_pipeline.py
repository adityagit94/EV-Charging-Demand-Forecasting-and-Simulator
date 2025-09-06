"""Unit tests for data pipeline functionality."""

import pandas as pd
import pytest

from src.data_pipeline import DataPipeline, DataValidationError


class TestDataPipeline:
    """Test cases for DataPipeline class."""

    def test_init(self):
        """Test DataPipeline initialization."""
        pipeline = DataPipeline()
        assert pipeline.required_columns == ["site_id", "timestamp", "sessions"]
        assert hasattr(pipeline, "logger")

    def test_validate_data_success(self, data_pipeline, sample_sessions_data):
        """Test successful data validation."""
        # Should not raise any exception
        data_pipeline.validate_data(sample_sessions_data)

    def test_validate_data_empty_dataframe(self, data_pipeline):
        """Test validation with empty dataframe."""
        empty_df = pd.DataFrame()
        with pytest.raises(DataValidationError, match="Input dataframe is empty"):
            data_pipeline.validate_data(empty_df)

    def test_validate_data_missing_columns(self, data_pipeline):
        """Test validation with missing required columns."""
        df = pd.DataFrame(
            {"site_id": [1, 2], "timestamp": ["2024-01-01", "2024-01-02"]}
        )
        with pytest.raises(DataValidationError, match="Missing required columns"):
            data_pipeline.validate_data(df)

    def test_validate_data_invalid_site_id_type(self, data_pipeline):
        """Test validation with invalid site_id type."""
        df = pd.DataFrame(
            {
                "site_id": ["a", "b"],
                "timestamp": ["2024-01-01", "2024-01-02"],
                "sessions": [1.0, 2.0],
            }
        )
        with pytest.raises(DataValidationError, match="site_id must be numeric"):
            data_pipeline.validate_data(df)

    def test_validate_data_invalid_sessions_type(self, data_pipeline):
        """Test validation with invalid sessions type."""
        df = pd.DataFrame(
            {
                "site_id": [1, 2],
                "timestamp": ["2024-01-01", "2024-01-02"],
                "sessions": ["a", "b"],
            }
        )
        with pytest.raises(DataValidationError, match="sessions must be numeric"):
            data_pipeline.validate_data(df)

    def test_clean_data(self, data_pipeline, sample_sessions_data):
        """Test data cleaning functionality."""
        # Add some duplicates and negative values
        dirty_data = sample_sessions_data.copy()
        dirty_data = pd.concat([dirty_data, dirty_data.iloc[[0, 1]]], ignore_index=True)
        dirty_data.loc[0, "sessions"] = -1.0

        cleaned = data_pipeline.clean_data(dirty_data)

        # Check that duplicates are removed
        assert len(cleaned) < len(dirty_data)

        # Check that negative values are clipped to 0
        assert (cleaned["sessions"] >= 0).all()

        # Check that data is sorted
        assert cleaned.equals(
            cleaned.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
        )

    def test_load_sessions_success(self, data_pipeline, temp_data_file):
        """Test successful data loading."""
        df = data_pipeline.load_sessions(temp_data_file)

        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert all(col in df.columns for col in ["site_id", "timestamp", "sessions"])
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_load_sessions_file_not_found(self, data_pipeline):
        """Test loading with non-existent file."""
        with pytest.raises(FileNotFoundError):
            data_pipeline.load_sessions("non_existent_file.csv")

    def test_aggregate_hourly(self, data_pipeline, sample_sessions_data):
        """Test hourly aggregation."""
        hourly_df = data_pipeline.aggregate_hourly(sample_sessions_data)

        assert isinstance(hourly_df, pd.DataFrame)
        assert "hour" in hourly_df.columns
        assert "sessions" in hourly_df.columns
        assert "session_count" in hourly_df.columns
        assert "avg_sessions" in hourly_df.columns

        # Check that aggregation worked
        assert len(hourly_df) <= len(sample_sessions_data)

    def test_get_data_summary(self, data_pipeline, sample_sessions_data):
        """Test data summary generation."""
        summary = data_pipeline.get_data_summary(sample_sessions_data)

        assert isinstance(summary, dict)
        assert "total_rows" in summary
        assert "unique_sites" in summary
        assert "date_range" in summary
        assert "sessions_stats" in summary
        assert "missing_values" in summary

        assert summary["total_rows"] == len(sample_sessions_data)
        assert summary["unique_sites"] == sample_sessions_data["site_id"].nunique()


@pytest.mark.unit
class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions."""

    def test_load_sessions_function(self, temp_data_file):
        """Test load_sessions backward compatibility function."""
        from src.data_pipeline import load_sessions

        df = load_sessions(temp_data_file)
        assert isinstance(df, pd.DataFrame)
        assert not df.empty

    def test_aggregate_hourly_function(self, sample_sessions_data):
        """Test aggregate_hourly backward compatibility function."""
        from src.data_pipeline import aggregate_hourly

        result = aggregate_hourly(sample_sessions_data)
        assert isinstance(result, pd.DataFrame)
        assert "hour" in result.columns
