"""Test configuration and fixtures for the EV charging forecast system."""

import os
import tempfile
from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ev_forecast.api.app import app
from ev_forecast.data_pipeline import DataPipeline
from ev_forecast.features import FeatureEngineer

# Set test environment
os.environ["ENV"] = "test"
os.environ["DEBUG"] = "true"


@pytest.fixture(scope="session")
def test_client() -> TestClient:
    """Create a test client for the FastAPI application."""
    return TestClient(app)


@pytest.fixture
def sample_sessions_data() -> pd.DataFrame:
    """Create sample sessions data for testing."""
    data = {
        "site_id": [1, 1, 1, 2, 2, 2] * 10,
        "timestamp": pd.date_range("2024-01-01", periods=60, freq="H"),
        "sessions": [5.0, 3.2, 7.8, 4.1, 6.5, 2.9] * 10,
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_hourly_data() -> pd.DataFrame:
    """Create sample hourly aggregated data for testing."""
    data = {
        "site_id": [1, 1, 2, 2] * 12,
        "hour": pd.date_range("2024-01-01", periods=48, freq="H"),
        "sessions": [5.0, 3.2, 7.8, 4.1] * 12,
        "session_count": [1, 1, 1, 1] * 12,
        "avg_sessions": [5.0, 3.2, 7.8, 4.1] * 12,
        "first_timestamp": pd.date_range("2024-01-01", periods=48, freq="H"),
    }
    return pd.DataFrame(data)


@pytest.fixture
def temp_data_file(sample_sessions_data) -> Generator[str, None, None]:
    """Create a temporary data file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_sessions_data.to_csv(f.name, index=False)
        yield f.name

    # Cleanup
    os.unlink(f.name)


@pytest.fixture
def data_pipeline() -> DataPipeline:
    """Create a DataPipeline instance for testing."""
    return DataPipeline()


@pytest.fixture
def feature_engineer() -> FeatureEngineer:
    """Create a FeatureEngineer instance for testing."""
    return FeatureEngineer()


@pytest.fixture
def sample_prediction_request() -> dict:
    """Sample prediction request data."""
    return {
        "site_id": 1,
        "timestamp": "2024-01-15T14:30:00Z",
        "hour_of_day": 14,
        "day_of_week": 0,
        "is_weekend": 0,
        "hour_sin": 0.5,
        "hour_cos": 0.866,
        "lag_1": 5.2,
        "lag_24": 4.8,
        "rmean_24": 5.0,
    }


@pytest.fixture
def mock_model_path() -> Generator[str, None, None]:
    """Create a mock model file path."""
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        # Create a dummy model file (empty for testing)
        import joblib

        joblib.dump({"dummy": "model"}, f.name)
        yield f.name

    # Cleanup
    os.unlink(f.name)


# Test markers
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration
pytest.mark.slow = pytest.mark.slow
