"""Model monitoring and data drift detection utilities."""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from scipy import stats


class DataDriftDetector:
    """Detect data drift in incoming data."""

    def __init__(self, reference_data: pd.DataFrame, significance_level: float = 0.05):
        """Initialize drift detector.

        Args:
            reference_data: Reference dataset to compare against
            significance_level: Statistical significance level for tests
        """
        self.reference_data = reference_data
        self.significance_level = significance_level
        self.logger = logger.bind(name=__name__)

        # Calculate reference statistics
        self.reference_stats = self._calculate_statistics(reference_data)

    def _calculate_statistics(self, data: pd.DataFrame) -> Dict:
        """Calculate statistical properties of the data."""
        stats_dict = {}

        for column in data.select_dtypes(include=[np.number]).columns:
            stats_dict[column] = {
                "mean": data[column].mean(),
                "std": data[column].std(),
                "min": data[column].min(),
                "max": data[column].max(),
                "median": data[column].median(),
                "q25": data[column].quantile(0.25),
                "q75": data[column].quantile(0.75),
                "skewness": stats.skew(data[column].dropna()),
                "kurtosis": stats.kurtosis(data[column].dropna()),
            }

        return stats_dict

    def detect_drift(self, new_data: pd.DataFrame) -> Dict:
        """Detect drift in new data compared to reference.

        Args:
            new_data: New data to check for drift

        Returns:
            Dictionary with drift detection results
        """
        drift_results = {
            "timestamp": datetime.now().isoformat(),
            "drift_detected": False,
            "features_with_drift": [],
            "statistical_tests": {},
            "summary": {},
        }

        numeric_columns = new_data.select_dtypes(include=[np.number]).columns

        for column in numeric_columns:
            if column not in self.reference_stats:
                continue

            # Kolmogorov-Smirnov test for distribution drift
            ref_values = self.reference_data[column].dropna()
            new_values = new_data[column].dropna()

            if len(new_values) < 30:  # Minimum sample size
                continue

            ks_statistic, ks_p_value = stats.ks_2samp(ref_values, new_values)

            # Mann-Whitney U test for median drift
            mw_statistic, mw_p_value = stats.mannwhitneyu(
                ref_values, new_values, alternative="two-sided"
            )

            # Store test results
            drift_results["statistical_tests"][column] = {
                "ks_test": {
                    "statistic": float(ks_statistic),
                    "p_value": float(ks_p_value),
                    "drift_detected": ks_p_value < self.significance_level,
                },
                "mannwhitney_test": {
                    "statistic": float(mw_statistic),
                    "p_value": float(mw_p_value),
                    "drift_detected": mw_p_value < self.significance_level,
                },
            }

            # Check if drift is detected
            if (
                ks_p_value < self.significance_level
                or mw_p_value < self.significance_level
            ):
                drift_results["features_with_drift"].append(column)
                drift_results["drift_detected"] = True

        # Summary statistics
        drift_results["summary"] = {
            "total_features_tested": len(numeric_columns),
            "features_with_drift": len(drift_results["features_with_drift"]),
            "drift_percentage": len(drift_results["features_with_drift"])
            / len(numeric_columns)
            * 100,
        }

        if drift_results["drift_detected"]:
            self.logger.warning(
                f"Data drift detected in features: {drift_results['features_with_drift']}"
            )
        else:
            self.logger.info("No significant data drift detected")

        return drift_results


class ModelPerformanceMonitor:
    """Monitor model performance over time."""

    def __init__(self, model_name: str, baseline_metrics: Dict[str, float]):
        """Initialize performance monitor.

        Args:
            model_name: Name of the model being monitored
            baseline_metrics: Baseline performance metrics
        """
        self.model_name = model_name
        self.baseline_metrics = baseline_metrics
        self.performance_history = []
        self.logger = logger.bind(name=__name__)

    def add_performance_record(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, float]:
        """Add a new performance record.

        Args:
            predictions: Model predictions
            actuals: Actual values
            timestamp: Timestamp for the record

        Returns:
            Performance metrics for this record
        """
        if timestamp is None:
            timestamp = datetime.now()

        # Calculate metrics
        mae = np.mean(np.abs(predictions - actuals))
        rmse = np.sqrt(np.mean((predictions - actuals) ** 2))
        mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100

        # R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        metrics = {
            "timestamp": timestamp.isoformat(),
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
            "sample_size": len(predictions),
        }

        self.performance_history.append(metrics)

        # Check for performance degradation
        self._check_performance_degradation(metrics)

        return metrics

    def _check_performance_degradation(self, current_metrics: Dict[str, float]):
        """Check if current performance has degraded significantly."""
        degradation_threshold = 0.1  # 10% degradation threshold

        degraded_metrics = []

        for metric in ["mae", "rmse", "mape"]:
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                current_value = current_metrics[metric]

                # For these metrics, higher is worse
                degradation = (current_value - baseline_value) / baseline_value

                if degradation > degradation_threshold:
                    degraded_metrics.append(
                        {
                            "metric": metric,
                            "baseline": baseline_value,
                            "current": current_value,
                            "degradation_pct": degradation * 100,
                        }
                    )

        # Check R-squared (higher is better)
        if "r2" in self.baseline_metrics:
            baseline_r2 = self.baseline_metrics["r2"]
            current_r2 = current_metrics["r2"]

            degradation = (baseline_r2 - current_r2) / baseline_r2

            if degradation > degradation_threshold:
                degraded_metrics.append(
                    {
                        "metric": "r2",
                        "baseline": baseline_r2,
                        "current": current_r2,
                        "degradation_pct": degradation * 100,
                    }
                )

        if degraded_metrics:
            self.logger.warning(
                f"Performance degradation detected for {self.model_name}: {degraded_metrics}"
            )

    def get_performance_summary(self, days: int = 30) -> Dict:
        """Get performance summary for the last N days.

        Args:
            days: Number of days to include in summary

        Returns:
            Performance summary
        """
        cutoff_date = datetime.now() - timedelta(days=days)

        recent_records = [
            record
            for record in self.performance_history
            if datetime.fromisoformat(record["timestamp"]) >= cutoff_date
        ]

        if not recent_records:
            return {"message": "No recent performance data available"}

        # Calculate summary statistics
        metrics = ["mae", "rmse", "mape", "r2"]
        summary = {
            "period_days": days,
            "total_predictions": sum(
                record["sample_size"] for record in recent_records
            ),
            "num_evaluations": len(recent_records),
            "metrics": {},
        }

        for metric in metrics:
            values = [record[metric] for record in recent_records if metric in record]
            if values:
                summary["metrics"][metric] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "trend": self._calculate_trend(values),
                }

        return summary

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction for a metric."""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend
        x = np.arange(len(values))
        slope, _, _, p_value, _ = stats.linregress(x, values)

        if p_value > 0.05:  # Not statistically significant
            return "stable"
        elif slope > 0:
            return "increasing"
        else:
            return "decreasing"

    def save_performance_history(self, filepath: str):
        """Save performance history to file.

        Args:
            filepath: Path to save the history
        """
        with open(filepath, "w") as f:
            json.dump(
                {
                    "model_name": self.model_name,
                    "baseline_metrics": self.baseline_metrics,
                    "performance_history": self.performance_history,
                },
                f,
                indent=2,
            )

        self.logger.info(f"Performance history saved to {filepath}")

    def load_performance_history(self, filepath: str):
        """Load performance history from file.

        Args:
            filepath: Path to load the history from
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        self.model_name = data["model_name"]
        self.baseline_metrics = data["baseline_metrics"]
        self.performance_history = data["performance_history"]

        self.logger.info(f"Performance history loaded from {filepath}")


class AlertManager:
    """Manage alerts for monitoring systems."""

    def __init__(self, config: Optional[Dict] = None):
        """Initialize alert manager.

        Args:
            config: Configuration for alert channels
        """
        self.config = config or {}
        self.logger = logger.bind(name=__name__)

    def send_drift_alert(self, drift_results: Dict):
        """Send alert for data drift detection.

        Args:
            drift_results: Results from drift detection
        """
        if not drift_results["drift_detected"]:
            return

        message = f"""
        🚨 DATA DRIFT ALERT 🚨
        
        Drift detected in {len(drift_results['features_with_drift'])} features:
        {', '.join(drift_results['features_with_drift'])}
        
        Drift percentage: {drift_results['summary']['drift_percentage']:.1f}%
        Timestamp: {drift_results['timestamp']}
        
        Please investigate and consider retraining the model.
        """

        self._send_alert("Data Drift Alert", message)

    def send_performance_alert(self, model_name: str, degraded_metrics: List[Dict]):
        """Send alert for performance degradation.

        Args:
            model_name: Name of the model
            degraded_metrics: List of degraded metrics
        """
        message = f"""
        📉 PERFORMANCE DEGRADATION ALERT 📉
        
        Model: {model_name}
        
        Degraded metrics:
        """

        for metric_info in degraded_metrics:
            message += f"""
        - {metric_info['metric']}: {metric_info['current']:.4f} 
          (baseline: {metric_info['baseline']:.4f}, 
           degradation: {metric_info['degradation_pct']:.1f}%)
        """

        message += "\n\nConsider investigating data quality or retraining the model."

        self._send_alert("Performance Degradation Alert", message)

    def _send_alert(self, subject: str, message: str):
        """Send alert through configured channels.

        Args:
            subject: Alert subject
            message: Alert message
        """
        # Log the alert
        self.logger.warning(f"ALERT: {subject}\n{message}")

        # Here you could add integrations with:
        # - Email services (SMTP, SendGrid, etc.)
        # - Slack webhooks
        # - PagerDuty
        # - SMS services
        # - Custom webhooks

        # Example Slack integration (commented out)
        # if 'slack_webhook' in self.config:
        #     self._send_slack_alert(subject, message)

    def _send_slack_alert(self, subject: str, message: str):
        """Send alert to Slack (example implementation).

        Args:
            subject: Alert subject
            message: Alert message
        """
        # Example Slack webhook implementation
        # import requests
        #
        # webhook_url = self.config['slack_webhook']
        # payload = {
        #     'text': f"*{subject}*\n```{message}```"
        # }
        #
        # try:
        #     response = requests.post(webhook_url, json=payload)
        #     response.raise_for_status()
        #     self.logger.info("Slack alert sent successfully")
        # except Exception as e:
        #     self.logger.error(f"Failed to send Slack alert: {e}")
        pass
