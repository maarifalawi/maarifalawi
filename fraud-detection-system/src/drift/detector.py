"""
Concept Drift Detection Module
Monitors model performance degradation and feature distribution shifts over time.

Implements:
- Population Stability Index (PSI) for feature drift
- Kolmogorov-Smirnov test for distribution comparison
- Performance drift (AUC decay monitoring)
- Prediction distribution shift
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


class DriftDetector:
    """
    Monitors data and model drift for fraud detection systems.

    Drift Types Detected:
    1. Feature Drift: Input feature distributions change (PSI, KS-test)
    2. Prediction Drift: Model output distribution shifts
    3. Performance Drift: Model accuracy degrades (requires labels)
    4. Concept Drift: Relationship between features and target changes

    Thresholds:
    - PSI < 0.1: No drift
    - 0.1 <= PSI < 0.2: Moderate drift (alert)
    - PSI >= 0.2: Significant drift (retrain)
    """

    PSI_THRESHOLD_ALERT = 0.1
    PSI_THRESHOLD_RETRAIN = 0.2
    KS_THRESHOLD = 0.05  # p-value threshold

    def __init__(
        self,
        reference_data: Optional[pd.DataFrame] = None,
        feature_names: Optional[List[str]] = None,
        n_bins: int = 10,
    ):
        self.reference_data = reference_data
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.drift_history: List[Dict] = []
        self.reference_predictions: Optional[np.ndarray] = None

    def set_reference(
        self,
        data: pd.DataFrame,
        predictions: Optional[np.ndarray] = None,
    ):
        """
        Set reference (baseline) data from training/validation period.

        Args:
            data: Reference feature DataFrame
            predictions: Reference model predictions
        """
        self.reference_data = data
        if self.feature_names is None:
            self.feature_names = data.columns.tolist()
        if predictions is not None:
            self.reference_predictions = predictions
        print(f"Reference set: {len(data)} samples, {len(self.feature_names)} features")


    def compute_psi(
        self,
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: Optional[int] = None,
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI measures how much a distribution has shifted from reference.
        PSI = SUM((actual_% - expected_%) * ln(actual_% / expected_%))

        Args:
            reference: Reference distribution values
            current: Current distribution values
            n_bins: Number of bins for discretization

        Returns:
            PSI value (0 = no shift, >0.2 = significant shift)
        """
        if n_bins is None:
            n_bins = self.n_bins

        # Create bins from reference distribution
        breakpoints = np.linspace(
            min(reference.min(), current.min()),
            max(reference.max(), current.max()),
            n_bins + 1,
        )

        # Compute bin percentages
        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        # Normalize to percentages
        ref_pct = ref_counts / max(len(reference), 1)
        cur_pct = cur_counts / max(len(current), 1)

        # Avoid division by zero / log(0)
        ref_pct = np.clip(ref_pct, 1e-6, None)
        cur_pct = np.clip(cur_pct, 1e-6, None)

        # PSI formula
        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))

        return float(psi)

    def compute_ks_test(
        self,
        reference: np.ndarray,
        current: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Compute Kolmogorov-Smirnov test between distributions.

        Args:
            reference: Reference distribution
            current: Current distribution

        Returns:
            Tuple of (KS statistic, p-value)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ks_stat, p_value = stats.ks_2samp(reference, current)
        return float(ks_stat), float(p_value)

    def detect_feature_drift(
        self,
        current_data: pd.DataFrame,
    ) -> Dict:
        """
        Detect drift in individual features using PSI and KS-test.

        Args:
            current_data: Current period feature DataFrame

        Returns:
            Dict with drift results per feature
        """
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference() first.")

        results = {
            "timestamp": datetime.now().isoformat(),
            "n_reference_samples": len(self.reference_data),
            "n_current_samples": len(current_data),
            "features": {},
            "drifted_features": [],
            "alert_features": [],
            "overall_status": "OK",
        }

        features_to_check = [
            f for f in self.feature_names
            if f in self.reference_data.columns and f in current_data.columns
        ]

        for feature in features_to_check:
            ref_values = self.reference_data[feature].dropna().values
            cur_values = current_data[feature].dropna().values

            if len(ref_values) < 10 or len(cur_values) < 10:
                continue

            # Compute PSI
            psi = self.compute_psi(ref_values, cur_values)

            # Compute KS test
            ks_stat, ks_pvalue = self.compute_ks_test(ref_values, cur_values)

            # Determine drift status
            if psi >= self.PSI_THRESHOLD_RETRAIN:
                status = "DRIFT_DETECTED"
                results["drifted_features"].append(feature)
            elif psi >= self.PSI_THRESHOLD_ALERT:
                status = "ALERT"
                results["alert_features"].append(feature)
            else:
                status = "OK"

            results["features"][feature] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pvalue, 4),
                "ks_significant": ks_pvalue < self.KS_THRESHOLD,
                "status": status,
                "ref_mean": round(float(ref_values.mean()), 4),
                "cur_mean": round(float(cur_values.mean()), 4),
                "ref_std": round(float(ref_values.std()), 4),
                "cur_std": round(float(cur_values.std()), 4),
            }

        # Overall status
        if results["drifted_features"]:
            results["overall_status"] = "DRIFT_DETECTED"
        elif results["alert_features"]:
            results["overall_status"] = "ALERT"

        # Store in history
        self.drift_history.append(results)

        return results


    def detect_prediction_drift(
        self,
        current_predictions: np.ndarray,
    ) -> Dict:
        """
        Detect drift in model prediction distribution.

        Args:
            current_predictions: Current period model predictions

        Returns:
            Dict with prediction drift results
        """
        if self.reference_predictions is None:
            raise ValueError("Reference predictions not set.")

        psi = self.compute_psi(self.reference_predictions, current_predictions)
        ks_stat, ks_pvalue = self.compute_ks_test(
            self.reference_predictions, current_predictions
        )

        # Compare prediction statistics
        ref_fraud_rate = (self.reference_predictions >= 0.5).mean()
        cur_fraud_rate = (current_predictions >= 0.5).mean()

        result = {
            "timestamp": datetime.now().isoformat(),
            "psi": round(psi, 4),
            "ks_statistic": round(ks_stat, 4),
            "ks_pvalue": round(ks_pvalue, 4),
            "ref_mean_score": round(float(self.reference_predictions.mean()), 4),
            "cur_mean_score": round(float(current_predictions.mean()), 4),
            "ref_fraud_rate": round(float(ref_fraud_rate), 4),
            "cur_fraud_rate": round(float(cur_fraud_rate), 4),
            "fraud_rate_change": round(float(cur_fraud_rate - ref_fraud_rate), 4),
            "status": "OK",
        }

        if psi >= self.PSI_THRESHOLD_RETRAIN:
            result["status"] = "DRIFT_DETECTED"
        elif psi >= self.PSI_THRESHOLD_ALERT:
            result["status"] = "ALERT"

        return result

    def detect_performance_drift(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        reference_auc: float,
        tolerance: float = 0.05,
    ) -> Dict:
        """
        Detect model performance degradation.

        Args:
            y_true: True labels for current period
            y_pred: Predicted probabilities for current period
            reference_auc: AUC-ROC from training/validation
            tolerance: Acceptable AUC drop before alerting

        Returns:
            Dict with performance drift results
        """
        from sklearn.metrics import roc_auc_score, average_precision_score

        current_auc = roc_auc_score(y_true, y_pred)
        current_ap = average_precision_score(y_true, y_pred)
        auc_drop = reference_auc - current_auc

        result = {
            "timestamp": datetime.now().isoformat(),
            "reference_auc": round(reference_auc, 4),
            "current_auc": round(float(current_auc), 4),
            "auc_drop": round(float(auc_drop), 4),
            "current_average_precision": round(float(current_ap), 4),
            "tolerance": tolerance,
            "status": "OK",
            "recommendation": None,
        }

        if auc_drop >= tolerance * 2:
            result["status"] = "CRITICAL"
            result["recommendation"] = "Immediate model retrain required"
        elif auc_drop >= tolerance:
            result["status"] = "ALERT"
            result["recommendation"] = "Schedule model retrain"
        elif auc_drop >= tolerance * 0.5:
            result["status"] = "WARNING"
            result["recommendation"] = "Monitor closely"

        return result

    def generate_report(self) -> Dict:
        """
        Generate a comprehensive drift report from all checks.

        Returns:
            Dict with full drift analysis report
        """
        report = {
            "generated_at": datetime.now().isoformat(),
            "n_checks_performed": len(self.drift_history),
            "latest_check": self.drift_history[-1] if self.drift_history else None,
            "summary": {
                "total_features_checked": 0,
                "features_with_drift": 0,
                "features_with_alert": 0,
                "overall_recommendation": "No action needed",
            },
        }

        if self.drift_history:
            latest = self.drift_history[-1]
            n_features = len(latest.get("features", {}))
            n_drifted = len(latest.get("drifted_features", []))
            n_alert = len(latest.get("alert_features", []))

            report["summary"]["total_features_checked"] = n_features
            report["summary"]["features_with_drift"] = n_drifted
            report["summary"]["features_with_alert"] = n_alert

            if n_drifted > n_features * 0.3:
                report["summary"]["overall_recommendation"] = (
                    "RETRAIN: >30% of features show significant drift"
                )
            elif n_drifted > 0:
                report["summary"]["overall_recommendation"] = (
                    f"INVESTIGATE: {n_drifted} features drifted"
                )
            elif n_alert > n_features * 0.2:
                report["summary"]["overall_recommendation"] = (
                    "MONITOR: Multiple features showing early drift signs"
                )

        return report

    def save_report(self, path: str = "data/drift_report.json"):
        """Save drift report to file."""
        report = self.generate_report()
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Drift report saved: {output_path}")

    def print_summary(self):
        """Print a formatted drift summary."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("DRIFT DETECTION REPORT")
        print("=" * 60)
        print(f"Generated: {report['generated_at']}")
        print(f"Checks performed: {report['n_checks_performed']}")

        summary = report["summary"]
        print(f"\nFeatures checked: {summary['total_features_checked']}")
        print(f"Features with drift: {summary['features_with_drift']}")
        print(f"Features with alert: {summary['features_with_alert']}")
        print(f"\nRecommendation: {summary['overall_recommendation']}")

        if self.drift_history:
            latest = self.drift_history[-1]
            drifted = latest.get("drifted_features", [])
            if drifted:
                print(f"\nDrifted features:")
                for feat in drifted:
                    info = latest["features"][feat]
                    print(f"  - {feat}: PSI={info['psi']:.4f}")

        print("=" * 60)
