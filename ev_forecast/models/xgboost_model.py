"""XGBoost model wrapper for EV charging demand forecasting."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import (GridSearchCV, TimeSeriesSplit,
                                     cross_val_score)
from xgboost import XGBRegressor

from ..utils.config import settings


class XGBoostModel:
    """Wrapper around XGBRegressor with training, evaluation, and persistence.

    The wrapper remembers the feature columns used at fit time so that
    predictions can be aligned to the same columns regardless of the
    order (or extra columns) in the incoming dataframe.
    """

    def __init__(self, **params: Any) -> None:
        """Initialize the model with configured defaults.

        Args:
            **params: Overrides for the XGBRegressor parameters.
        """
        self.logger = logger.bind(name=__name__)

        defaults: Dict[str, Any] = {
            "objective": "reg:squarederror",
            "max_depth": settings.model.max_depth,
            "learning_rate": settings.model.learning_rate,
            "n_estimators": settings.model.n_estimators,
            "random_state": settings.model.random_state,
            "eval_metric": settings.model.eval_metric,
        }
        defaults.update(params)

        self.params = defaults
        self.model = XGBRegressor(**self.params)
        self.feature_names: List[str] = []
        self.is_fitted = False

    def __getstate__(self) -> Dict[str, Any]:
        """Drop the logger when pickling; it holds unpicklable handles."""
        state = self.__dict__.copy()
        state.pop("logger", None)
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore state and re-attach a logger after unpickling."""
        self.__dict__.update(state)
        self.logger = logger.bind(name=__name__)

    @property
    def num_features(self) -> int:
        """Number of features the model was trained on."""
        return len(self.feature_names)

    def _align_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder columns to match training, filling missing ones with 0."""
        if not self.feature_names:
            return X
        return X.reindex(columns=self.feature_names, fill_value=0.0)

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
    ) -> "XGBoostModel":
        """Train the model, optionally with early stopping on a validation set.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Optional validation features for early stopping
            y_val: Optional validation targets for early stopping

        Returns:
            The fitted model instance
        """
        self.feature_names = list(X_train.columns)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            early_stopping = settings.model.early_stopping_rounds
            self.model = XGBRegressor(
                **self.params, early_stopping_rounds=early_stopping
            )
            self.model.fit(
                X_train,
                y_train,
                eval_set=[(self._align_features(X_val), y_val)],
                verbose=False,
            )
        else:
            self.model = XGBRegressor(**self.params)
            self.model.fit(X_train, y_train, verbose=False)

        self.is_fitted = True
        self.logger.info(f"Model trained on {len(X_train)} rows")
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict charging demand for the given features.

        Args:
            X: Feature dataframe

        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        return np.asarray(self.model.predict(self._align_features(X)))

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate the model on a test set.

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with MAE, RMSE, MAPE, and R2 metrics
        """
        preds = self.predict(X_test)
        actuals = np.asarray(y_test, dtype=float)

        mae = float(np.mean(np.abs(preds - actuals)))
        rmse = float(np.sqrt(np.mean((preds - actuals) ** 2)))

        # MAPE only over non-zero actuals to avoid division by zero
        nonzero = actuals != 0
        if nonzero.any():
            mape = float(
                np.mean(np.abs((actuals[nonzero] - preds[nonzero]) / actuals[nonzero]))
                * 100
            )
        else:
            mape = float("nan")

        ss_res = float(np.sum((actuals - preds) ** 2))
        ss_tot = float(np.sum((actuals - np.mean(actuals)) ** 2))
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}

    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
    ) -> Dict[str, Any]:
        """Cross-validate with a time-series split.

        Args:
            X: Features
            y: Targets
            n_splits: Number of time-series splits

        Returns:
            Dictionary with per-fold and aggregate MAE scores
        """
        cv = TimeSeriesSplit(n_splits=n_splits)
        estimator = XGBRegressor(**self.params)
        scores = -cross_val_score(
            estimator, X, y, cv=cv, scoring="neg_mean_absolute_error"
        )

        return {
            "scores": scores,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "metric": "mae",
        }

    def hyperparameter_tuning(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Optional[Dict[str, List[Any]]] = None,
        n_splits: int = 3,
    ) -> Dict[str, Any]:
        """Tune hyperparameters with grid search over a time-series split.

        Args:
            X: Features
            y: Targets
            param_grid: Grid of parameters to search. Uses a small default
                grid when omitted.
            n_splits: Number of time-series splits

        Returns:
            Dictionary with the best parameters and score
        """
        if param_grid is None:
            param_grid = {
                "max_depth": [4, 6, 8],
                "learning_rate": [0.05, 0.1],
                "n_estimators": [200, 300],
            }

        cv = TimeSeriesSplit(n_splits=n_splits)
        search = GridSearchCV(
            XGBRegressor(**self.params),
            param_grid,
            cv=cv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
        )
        search.fit(X, y)

        self.params.update(search.best_params_)
        self.model = search.best_estimator_
        self.feature_names = list(X.columns)
        self.is_fitted = True

        return {
            "best_params": search.best_params_,
            "best_score": float(-search.best_score_),
            "metric": "mae",
        }

    def get_feature_importance(self) -> Dict[str, float]:
        """Return feature importances keyed by feature name."""
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        importances = self.model.feature_importances_
        return dict(zip(self.feature_names, importances.tolist()))

    def save_model(self, filepath: Union[str, Path]) -> str:
        """Persist the model to disk.

        Args:
            filepath: Destination path for the joblib artifact

        Returns:
            The path the model was saved to
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        self.logger.info(f"Model saved to {path}")
        return str(path)

    @staticmethod
    def load_model(filepath: Union[str, Path]) -> "XGBoostModel":
        """Load a previously saved model.

        Args:
            filepath: Path to the joblib artifact

        Returns:
            The loaded model instance
        """
        model = joblib.load(filepath)
        if not isinstance(model, XGBoostModel):
            raise TypeError(f"Expected XGBoostModel artifact, got {type(model)}")
        return model
