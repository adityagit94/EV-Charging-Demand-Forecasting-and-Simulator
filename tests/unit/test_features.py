"""Unit tests for feature engineering functionality."""

import pandas as pd
import pytest

from ev_forecast.features import FeatureEngineer


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def test_init(self):
        """Test FeatureEngineer initialization."""
        engineer = FeatureEngineer()
        assert hasattr(engineer, "logger")
        assert engineer.feature_columns == []

    def test_add_temporal_features(self, feature_engineer, sample_hourly_data):
        """Test temporal feature engineering."""
        result = feature_engineer.add_temporal_features(sample_hourly_data)

        # Check that temporal features are added
        expected_features = [
            "hour_of_day",
            "day_of_week",
            "day_of_month",
            "month",
            "quarter",
            "year",
            "is_weekend",
            "is_monday",
            "is_friday",
            "is_business_hours",
            "is_morning_peak",
            "is_evening_peak",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]

        for feature in expected_features:
            assert feature in result.columns

        # Check value ranges
        assert (result["hour_of_day"] >= 0).all() and (
            result["hour_of_day"] <= 23
        ).all()
        assert (result["day_of_week"] >= 0).all() and (result["day_of_week"] <= 6).all()
        assert (result["is_weekend"].isin([0, 1])).all()
        assert (result["hour_sin"] >= -1).all() and (result["hour_sin"] <= 1).all()
        assert (result["hour_cos"] >= -1).all() and (result["hour_cos"] <= 1).all()

    def test_add_temporal_features_no_cyclical(
        self, feature_engineer, sample_hourly_data
    ):
        """Test temporal features without cyclical encoding."""
        result = feature_engineer.add_temporal_features(
            sample_hourly_data, include_cyclical=False
        )

        # Cyclical features should not be present
        cyclical_features = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
        ]
        for feature in cyclical_features:
            assert feature not in result.columns

    def test_add_lag_features(self, feature_engineer, sample_hourly_data):
        """Test lag feature engineering."""
        lag_periods = [1, 24]
        result = feature_engineer.add_lag_features(
            sample_hourly_data, lag_periods=lag_periods
        )

        # Check that lag features are added
        for lag in lag_periods:
            assert f"lag_{lag}" in result.columns

        # Check that lag values are correct (first few should be NaN)
        assert pd.isna(result["lag_1"].iloc[0])  # First row should be NaN
        assert pd.isna(result["lag_24"].iloc[0])  # First row should be NaN

    def test_add_rolling_features(self, feature_engineer, sample_hourly_data):
        """Test rolling statistical feature engineering."""
        windows = [24, 72]
        result = feature_engineer.add_rolling_features(
            sample_hourly_data, windows=windows
        )

        # Check that rolling features are added
        for window in windows:
            expected_features = [
                f"rmean_{window}",
                f"rstd_{window}",
                f"rmin_{window}",
                f"rmax_{window}",
            ]
            for feature in expected_features:
                assert feature in result.columns

        # Check that rolling mean is reasonable
        assert (result["rmean_24"] >= 0).all()

    def test_add_difference_features(self, feature_engineer, sample_hourly_data):
        """Test difference feature engineering."""
        periods = [1, 24]
        result = feature_engineer.add_difference_features(
            sample_hourly_data, periods=periods
        )

        # Check that difference features are added
        for period in periods:
            assert f"diff_{period}" in result.columns

        # First differences should be NaN for first row
        assert pd.isna(result["diff_1"].iloc[0])

    def test_add_interaction_features(self, feature_engineer, sample_hourly_data):
        """Test interaction feature engineering."""
        # First add some base features
        df_with_features = feature_engineer.add_temporal_features(sample_hourly_data)

        feature_pairs = [("hour_of_day", "is_weekend")]
        result = feature_engineer.add_interaction_features(
            df_with_features, feature_pairs=feature_pairs
        )

        # Check that interaction features are added
        assert "hour_of_day_x_is_weekend" in result.columns

        # Check interaction values
        expected_interaction = (
            df_with_features["hour_of_day"] * df_with_features["is_weekend"]
        )
        pd.testing.assert_series_equal(
            result["hour_of_day_x_is_weekend"], expected_interaction, check_names=False
        )

    def test_create_feature_pipeline(self, feature_engineer, sample_hourly_data):
        """Test complete feature engineering pipeline."""
        result = feature_engineer.create_feature_pipeline(sample_hourly_data)

        # Check that result has more columns than input
        assert result.shape[1] > sample_hourly_data.shape[1]

        # Check that key feature types are present
        temporal_features = ["hour_of_day", "day_of_week", "is_weekend"]
        lag_features = ["lag_1", "lag_24", "lag_168"]
        rolling_features = ["rmean_24", "rmean_72", "rmean_168"]

        for feature_list in [temporal_features, lag_features, rolling_features]:
            for feature in feature_list:
                assert feature in result.columns

    def test_get_feature_importance_names(self, feature_engineer):
        """Test feature importance names generation."""
        feature_names = feature_engineer.get_feature_importance_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0

        # Check that important feature categories are included
        assert any("hour_of_day" in name for name in feature_names)
        assert any("lag_" in name for name in feature_names)
        assert any("rmean_" in name for name in feature_names)


@pytest.mark.unit
class TestBackwardCompatibilityFunctions:
    """Test backward compatibility functions."""

    def test_add_time_features_function(self, sample_hourly_data):
        """Test add_time_features backward compatibility function."""
        from ev_forecast.features import add_time_features

        result = add_time_features(sample_hourly_data)

        # Check that basic temporal features are added
        expected_features = [
            "hour_of_day",
            "day_of_week",
            "is_weekend",
            "hour_sin",
            "hour_cos",
        ]
        for feature in expected_features:
            assert feature in result.columns

    def test_add_lag_features_function(self, sample_hourly_data):
        """Test add_lag_features backward compatibility function."""
        from ev_forecast.features import add_lag_features

        result = add_lag_features(sample_hourly_data)

        # Check that lag features are added (using default config)
        assert "lag_1" in result.columns
        assert "lag_24" in result.columns
        assert "lag_168" in result.columns
