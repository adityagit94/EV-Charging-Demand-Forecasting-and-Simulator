"""Configuration management for the EV charging forecast system."""

import os
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """Model configuration settings."""

    name: str = "xgboost"
    max_depth: int = 6
    learning_rate: float = 0.1
    n_estimators: int = 300
    random_state: int = 42
    early_stopping_rounds: Optional[int] = 50
    eval_metric: str = "rmse"


class DataConfig(BaseModel):
    """Data processing configuration."""

    raw_data_path: str = "data/raw/synthetic_sessions.csv"
    processed_data_path: str = "data/processed/"
    test_size: float = 0.2
    validation_size: float = 0.1
    time_column: str = "timestamp"
    target_column: str = "sessions"
    site_column: str = "site_id"


class FeatureConfig(BaseModel):
    """Feature engineering configuration."""

    lag_hours: List[int] = Field(default=[1, 24, 168])
    rolling_windows: List[int] = Field(default=[24, 72, 168])
    temporal_features: bool = True
    spatial_features: bool = False
    weather_features: bool = False


class APIConfig(BaseModel):
    """API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False
    workers: int = 1
    model_path: str = "models/xgboost_baseline.joblib"


class DashboardConfig(BaseModel):
    """Dashboard configuration."""

    port: int = 8501
    title: str = "EV Charging Demand Forecast"
    theme: str = "light"
    cache_ttl: int = 3600  # seconds


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"
    format: str = (
        "{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} | {message}"
    )
    rotation: str = "10 MB"
    retention: str = "1 month"
    log_file: str = "logs/app.log"


class Settings(BaseSettings):
    """Application settings."""

    # Environment
    environment: str = Field(default="development", alias="ENV")
    debug: bool = Field(default=True, alias="DEBUG")

    # Project paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent.parent
    )
    data_dir: Path = Field(default_factory=lambda: Path("data"))
    models_dir: Path = Field(default_factory=lambda: Path("models"))
    logs_dir: Path = Field(default_factory=lambda: Path("logs"))

    # Component configurations
    model: ModelConfig = Field(default_factory=ModelConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_nested_delimiter = "__"


def load_config(config_path: Optional[str] = None) -> Settings:
    """Load configuration from file and environment variables."""
    settings = Settings()

    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Update settings with config file data
        for key, value in config_data.items():
            if hasattr(settings, key):
                if isinstance(getattr(settings, key), BaseModel):
                    # Update nested configuration
                    current_config = getattr(settings, key)
                    updated_config = current_config.model_copy(update=value)
                    setattr(settings, key, updated_config)
                else:
                    setattr(settings, key, value)

    # Ensure directories exist
    for dir_path in [settings.data_dir, settings.models_dir, settings.logs_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    return settings


def get_settings() -> Settings:
    """Get application settings (cached)."""
    return load_config()


# Global settings instance
settings = get_settings()
