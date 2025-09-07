"""Advanced model training orchestration for EV charging demand forecasting."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split

from .data_pipeline import DataPipeline
from .features import FeatureEngineer
from .models.xgboost_model import XGBoostModel


class ModelTrainer:
    """Advanced model training orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the model trainer.

        Args:
            config_path: Path to configuration file
        """
        self.logger = logger.bind(name=__name__)
        self.pipeline = DataPipeline()
        self.feature_engineer = FeatureEngineer()
        self.models: Dict[str, Any] = {}
        self.training_results: Dict[str, Any] = {}

    def prepare_data(
        self,
        data_path: Optional[str] = None,
        test_size: float = 0.2,
        validation_size: float = 0.1,
        time_based_split: bool = True,
    ) -> Tuple[
        pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series
    ]:
        """Prepare data for training.

        Args:
            data_path: Path to data file
            test_size: Proportion of data for testing
            validation_size: Proportion of data for validation
            time_based_split: Whether to use time-based splitting

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        self.logger.info("Preparing data for training...")

        # Load and process data
        if data_path:
            df = self.pipeline.load_sessions(data_path)
        else:
            df = self.pipeline.load_sessions()

        hourly = self.pipeline.aggregate_hourly(df)
        hourly["hour"] = pd.to_datetime(hourly["hour"])

        # Feature engineering
        features_df = self.feature_engineer.create_feature_pipeline(
            hourly, include_interactions=True
        )

        # Remove rows with NaN values (from lag features)
        features_df = features_df.dropna().reset_index(drop=True)

        # Separate features and target
        feature_cols = self.feature_engineer.get_feature_importance_names()

        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in features_df.columns]
        if len(available_features) < len(feature_cols):
            missing = set(feature_cols) - set(available_features)
            self.logger.warning(f"Missing features: {missing}")
            feature_cols = available_features

        X = features_df[feature_cols]
        y = features_df["sessions"]

        if time_based_split:
            # Time-based split to maintain temporal order
            n_total = len(X)
            n_test = int(n_total * test_size)
            n_val = int(n_total * validation_size)
            n_train = n_total - n_test - n_val

            X_train = X.iloc[:n_train]
            X_val = X.iloc[n_train : n_train + n_val]
            X_test = X.iloc[n_train + n_val :]

            y_train = y.iloc[:n_train]
            y_val = y.iloc[n_train : n_train + n_val]
            y_test = y.iloc[n_train + n_val :]

        else:
            # Random split
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            val_size_adjusted = validation_size / (1 - test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp, test_size=val_size_adjusted, random_state=42
            )

        self.logger.info(
            f"Data prepared: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}"
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        hyperparameter_tuning: bool = False,
        cross_validation: bool = True,
        **kwargs: Any,
    ) -> XGBoostModel:
        """Train XGBoost model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            cross_validation: Whether to perform cross-validation
            **kwargs: Additional parameters for XGBoost

        Returns:
            Trained XGBoost model
        """
        self.logger.info("Training XGBoost model...")

        model = XGBoostModel(**kwargs)

        if hyperparameter_tuning:
            self.logger.info("Performing hyperparameter tuning...")
            tuning_results = model.hyperparameter_tuning(X_train, y_train)
            self.logger.info(f"Best parameters: {tuning_results['best_params']}")
            self.training_results["hyperparameter_tuning"] = tuning_results
        else:
            # Regular training
            model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        if cross_validation:
            self.logger.info("Performing cross-validation...")
            cv_results = model.cross_validate(X_train, y_train)
            self.logger.info(
                f"CV Score: {cv_results['mean_score']:.4f} "
                f"(+/- {cv_results['std_score']:.4f})"
            )
            self.training_results["cross_validation"] = cv_results

        self.models["xgboost"] = model
        return model

    def evaluate_model(
        self, model_name: str, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, Any]:
        """Evaluate a trained model.

        Args:
            model_name: Name of the model to evaluate
            X_test: Test features
            y_test: Test targets

        Returns:
            Evaluation metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        metrics = model.evaluate(X_test, y_test)

        self.logger.info(f"{model_name} evaluation metrics:")
        for metric, value in metrics.items():
            self.logger.info(f"  {metric}: {value:.4f}")

        self.training_results[f"{model_name}_evaluation"] = metrics
        return metrics  # type: ignore

    def save_model(
        self,
        model_name: str,
        output_dir: str = "models",
        include_timestamp: bool = True,
    ) -> str:
        """Save a trained model.

        Args:
            model_name: Name of the model to save
            output_dir: Directory to save the model
            include_timestamp: Whether to include timestamp in filename

        Returns:
            Path to saved model
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Generate filename
        if include_timestamp:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_{timestamp}.joblib"
        else:
            filename = f"{model_name}.joblib"

        filepath = os.path.join(output_dir, filename)

        # Save model
        self.models[model_name].save_model(filepath)

        return filepath

    def save_training_results(
        self, output_dir: str = "models", filename: str = "training_results.json"
    ) -> str:
        """Save training results to JSON file.

        Args:
            output_dir: Directory to save results
            filename: Name of the results file

        Returns:
            Path to saved results file
        """
        import json

        Path(output_dir).mkdir(parents=True, exist_ok=True)
        filepath = os.path.join(output_dir, filename)

        # Convert numpy arrays to lists for JSON serialization
        results: Dict[str, Any] = {}
        for key, value in self.training_results.items():
            if isinstance(value, dict):
                results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results[key][k] = v.tolist()
                    else:
                        results[key][k] = v
            else:
                results[key] = value

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Training results saved to {filepath}")
        return filepath

    def run_full_pipeline(
        self,
        data_path: Optional[str] = None,
        model_types: Optional[List[str]] = None,
        hyperparameter_tuning: bool = False,
        save_models: bool = True,
        output_dir: str = "models",
    ) -> Dict[str, Any]:
        """Run the complete training pipeline.

        Args:
            data_path: Path to data file
            model_types: List of model types to train
            hyperparameter_tuning: Whether to perform hyperparameter tuning
            save_models: Whether to save trained models
            output_dir: Directory to save models and results

        Returns:
            Dictionary with training results
        """
        if model_types is None:
            model_types = ["xgboost"]

        self.logger.info("Starting full training pipeline...")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(data_path)

        results = {}

        # Train models
        for model_type in model_types:
            if model_type == "xgboost":
                self.train_xgboost(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    hyperparameter_tuning=hyperparameter_tuning,
                )

                # Evaluate model
                metrics = self.evaluate_model("xgboost", X_test, y_test)
                results["xgboost"] = metrics

                # Save model if requested
                if save_models:
                    model_path = self.save_model("xgboost", output_dir)
                    results["xgboost"]["model_path"] = model_path

        # Save training results
        results_path = self.save_training_results(output_dir)
        results["results_path"] = results_path  # type: ignore

        self.logger.info("Training pipeline completed successfully!")
        return results


def main() -> None:
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train EV charging demand forecasting models"
    )
    parser.add_argument("--data-path", type=str, help="Path to training data")
    parser.add_argument(
        "--model", type=str, default="xgboost", help="Model type to train"
    )
    parser.add_argument(
        "--hyperparameter-tuning",
        action="store_true",
        help="Perform hyperparameter tuning",
    )
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory for models"
    )
    parser.add_argument("--config", type=str, help="Path to configuration file")

    args = parser.parse_args()

    # Initialize trainer
    trainer = ModelTrainer(args.config)

    # Run training pipeline
    results = trainer.run_full_pipeline(
        data_path=args.data_path,
        model_types=[args.model],
        hyperparameter_tuning=args.hyperparameter_tuning,
        output_dir=args.output_dir,
    )

    print("Training completed successfully!")
    print(f"Results: {results}")


if __name__ == "__main__":
    main()
