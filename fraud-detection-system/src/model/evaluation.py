"""
Model Evaluation Module
Comprehensive metrics for fraud detection model performance.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class ModelEvaluator:
    """
    Evaluate fraud detection models with metrics optimized for
    imbalanced classification problems.
    """

    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold

    def evaluate(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict:
        """
        Compute comprehensive evaluation metrics.

        Args:
            y_true: True labels (0/1)
            y_prob: Predicted probabilities
            threshold: Decision threshold (uses self.threshold if None)

        Returns:
            Dict with all metrics
        """
        if threshold is None:
            threshold = self.threshold

        y_pred = (y_prob >= threshold).astype(int)

        metrics = {
            "threshold": threshold,
            "n_samples": len(y_true),
            "n_fraud": int(y_true.sum()),
            "n_legit": int((1 - y_true).sum()),
            "fraud_rate": float(y_true.mean()),
            # Core metrics
            "auc_roc": float(roc_auc_score(y_true, y_prob)),
            "auc_pr": float(average_precision_score(y_true, y_prob)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            # Confusion matrix
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
            # Business metrics
            "false_positive_rate": float(
                confusion_matrix(y_true, y_pred)[0, 1] / max(1, (1 - y_true).sum())
            ),
            "false_negative_rate": float(
                confusion_matrix(y_true, y_pred)[1, 0] / max(1, y_true.sum())
            ),
        }

        # Optimal threshold (maximizing F1)
        metrics["optimal_threshold"] = self._find_optimal_threshold(y_true, y_prob)

        return metrics

    def _find_optimal_threshold(
        self, y_true: np.ndarray, y_prob: np.ndarray
    ) -> float:
        """Find threshold that maximizes F1 score."""
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

        # Compute F1 for each threshold
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
        best_idx = np.argmax(f1_scores)

        if best_idx < len(thresholds):
            return float(thresholds[best_idx])
        return 0.5

    def print_report(self, y_true: np.ndarray, y_prob: np.ndarray) -> Dict:
        """Print a formatted evaluation report and return metrics."""
        metrics = self.evaluate(y_true, y_prob)

        print("=" * 60)
        print("FRAUD DETECTION MODEL EVALUATION")
        print("=" * 60)
        print(f"\nDataset: {metrics['n_samples']:,} samples "
              f"({metrics['n_fraud']:,} fraud, {metrics['fraud_rate']*100:.2f}%)")
        print(f"Threshold: {metrics['threshold']}")
        print(f"\n{'Metric':<25} {'Value':<10}")
        print("-" * 35)
        print(f"{'AUC-ROC':<25} {metrics['auc_roc']:.4f}")
        print(f"{'AUC-PR':<25} {metrics['auc_pr']:.4f}")
        print(f"{'Precision':<25} {metrics['precision']:.4f}")
        print(f"{'Recall':<25} {metrics['recall']:.4f}")
        print(f"{'F1 Score':<25} {metrics['f1']:.4f}")
        print(f"{'Accuracy':<25} {metrics['accuracy']:.4f}")
        print(f"{'False Positive Rate':<25} {metrics['false_positive_rate']:.4f}")
        print(f"{'False Negative Rate':<25} {metrics['false_negative_rate']:.4f}")
        print(f"{'Optimal Threshold':<25} {metrics['optimal_threshold']:.4f}")
        print(f"\nConfusion Matrix:")
        cm = np.array(metrics["confusion_matrix"])
        print(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}")
        print(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}")
        print("=" * 60)

        return metrics

    def save_metrics(self, metrics: Dict, path: str):
        """Save metrics to JSON file."""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to: {output_path}")
