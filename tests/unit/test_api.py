"""Unit tests for FastAPI application."""

from unittest.mock import Mock, patch

import pytest


class TestAPIEndpoints:
    """Test cases for API endpoints."""

    def test_health_check(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "model_loaded" in data
        assert "model_path" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data

        assert data["status"] == "healthy"
        assert isinstance(data["model_loaded"], bool)
        assert isinstance(data["uptime_seconds"], (int, float))

    def test_detailed_health(self, test_client):
        """Test detailed health endpoint."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()

        # Should have same structure as basic health check
        assert "status" in data
        assert "model_loaded" in data
        assert data["status"] == "healthy"

    def test_predict_without_model(self, test_client, sample_prediction_request):
        """Test prediction endpoint when model is not loaded."""
        response = test_client.post("/predict", json=sample_prediction_request)

        # Should return 503 when model is not loaded
        assert response.status_code == 503
        data = response.json()
        assert "error" in data
        assert "Model not loaded" in data["error"]

    @patch("src.api.app.model")
    def test_predict_with_model(
        self, mock_model, test_client, sample_prediction_request
    ):
        """Test prediction endpoint with loaded model."""
        # Mock the model prediction
        mock_model.predict.return_value = [5.5]
        mock_model.__bool__ = Mock(return_value=True)  # Make model truthy

        # Mock the model metadata
        with patch("src.api.app.model_metadata", {"version": "1.0.0"}):
            response = test_client.post("/predict", json=sample_prediction_request)

        assert response.status_code == 200
        data = response.json()

        assert "prediction" in data
        assert "site_id" in data
        assert "timestamp" in data
        assert "model_version" in data
        assert "processing_time_ms" in data

        assert data["prediction"] == 5.5
        assert data["site_id"] == sample_prediction_request["site_id"]
        assert data["model_version"] == "1.0.0"
        assert isinstance(data["processing_time_ms"], (int, float))

    def test_predict_invalid_request(self, test_client):
        """Test prediction endpoint with invalid request data."""
        invalid_request = {
            "site_id": -1,  # Invalid: should be positive
            "timestamp": "invalid-timestamp",  # Invalid format
            "hour_of_day": 25,  # Invalid: should be 0-23
        }

        response = test_client.post("/predict", json=invalid_request)
        assert response.status_code == 422  # Validation error

    def test_predict_missing_fields(self, test_client):
        """Test prediction endpoint with missing required fields."""
        incomplete_request = {
            "site_id": 1,
            # Missing other required fields
        }

        response = test_client.post("/predict", json=incomplete_request)
        assert response.status_code == 422  # Validation error

    @patch("src.api.app.model")
    def test_batch_predict(self, mock_model, test_client, sample_prediction_request):
        """Test batch prediction endpoint."""
        mock_model.predict.return_value = [5.5, 6.2]
        mock_model.__bool__ = Mock(return_value=True)

        batch_request = [sample_prediction_request, sample_prediction_request.copy()]
        batch_request[1]["site_id"] = 2

        with patch("src.api.app.model_metadata", {"version": "1.0.0"}):
            response = test_client.post("/predict/batch", json=batch_request)

        assert response.status_code == 200
        data = response.json()

        assert "predictions" in data
        assert "total_count" in data
        assert data["total_count"] == 2
        assert len(data["predictions"]) == 2

    def test_batch_predict_too_large(self, test_client, sample_prediction_request):
        """Test batch prediction with too many requests."""
        large_batch = [sample_prediction_request] * 101  # Exceeds limit of 100

        response = test_client.post("/predict/batch", json=large_batch)
        assert response.status_code == 400
        data = response.json()
        assert "Batch size too large" in data["error"]

    def test_model_info_no_model(self, test_client):
        """Test model info endpoint when no model is loaded."""
        response = test_client.get("/model/info")
        assert response.status_code == 404

    @patch("src.api.app.model")
    def test_model_info_with_model(self, mock_model, test_client):
        """Test model info endpoint with loaded model."""
        mock_model.__class__.__name__ = "XGBRegressor"
        mock_model.num_features = 10

        with patch(
            "src.api.app.model_metadata", {"version": "1.0.0", "path": "test.joblib"}
        ):
            response = test_client.get("/model/info")

        assert response.status_code == 200
        data = response.json()

        assert "model_metadata" in data
        assert "model_type" in data
        assert "features_count" in data
        assert data["model_type"] == "XGBRegressor"


class TestAPIValidation:
    """Test cases for API request validation."""

    def test_site_id_validation(self, test_client):
        """Test site_id validation."""
        invalid_requests = [
            {"site_id": 0},  # Should be >= 1
            {"site_id": -1},  # Should be positive
            {"site_id": "invalid"},  # Should be integer
        ]

        for req in invalid_requests:
            # Add minimal required fields
            req.update(
                {
                    "timestamp": "2024-01-15T14:30:00Z",
                    "hour_of_day": 14,
                    "day_of_week": 0,
                    "is_weekend": 0,
                    "hour_sin": 0.5,
                    "hour_cos": 0.866,
                }
            )

            response = test_client.post("/predict", json=req)
            assert response.status_code == 422

    def test_hour_validation(self, test_client, sample_prediction_request):
        """Test hour_of_day validation."""
        sample_prediction_request["hour_of_day"] = 25  # Invalid: should be 0-23

        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422

    def test_day_of_week_validation(self, test_client, sample_prediction_request):
        """Test day_of_week validation."""
        sample_prediction_request["day_of_week"] = 7  # Invalid: should be 0-6

        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422

    def test_trigonometric_validation(self, test_client, sample_prediction_request):
        """Test trigonometric feature validation."""
        sample_prediction_request["hour_sin"] = 1.5  # Invalid: should be -1 to 1

        response = test_client.post("/predict", json=sample_prediction_request)
        assert response.status_code == 422

    def test_timestamp_validation(self, test_client, sample_prediction_request):
        """Test timestamp validation."""
        invalid_timestamps = [
            "not-a-timestamp",
            "2024-13-01T14:30:00Z",  # Invalid month
            "2024-01-32T14:30:00Z",  # Invalid day
        ]

        for timestamp in invalid_timestamps:
            sample_prediction_request["timestamp"] = timestamp
            response = test_client.post("/predict", json=sample_prediction_request)
            assert response.status_code == 422


@pytest.mark.unit
class TestAPIModels:
    """Test Pydantic models used in API."""

    def test_predict_request_model(self):
        """Test PredictRequest model validation."""
        from src.api.app import PredictRequest

        valid_data = {
            "site_id": 1,
            "timestamp": "2024-01-15T14:30:00Z",
            "hour_of_day": 14,
            "day_of_week": 0,
            "is_weekend": 0,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
        }

        # Should create successfully
        request = PredictRequest(**valid_data)
        assert request.site_id == 1
        assert request.hour_of_day == 14

    def test_predict_response_model(self):
        """Test PredictResponse model."""
        from src.api.app import PredictResponse

        data = {
            "prediction": 5.5,
            "site_id": 1,
            "timestamp": "2024-01-15T14:30:00Z",
            "model_version": "1.0.0",
            "processing_time_ms": 12.5,
        }

        response = PredictResponse(**data)
        assert response.prediction == 5.5
        assert response.processing_time_ms == 12.5
