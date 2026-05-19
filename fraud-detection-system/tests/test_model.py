"""Tests for ML model components."""

import pytest
import numpy as np

from src.model.evaluation import ModelEvaluator


class TestModelEvaluator:
    """Test model evaluation metrics."""

    def setup_method(self):
        self.evaluator = ModelEvaluator(threshold=0.5)

    def test_perfect_predictions(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.1, 0.9, 0.8, 0.95])

        metrics = self.evaluator.evaluate(y_true, y_prob)

        assert metrics["auc_roc"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_random_predictions(self):
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=1000)
        y_prob = np.random.random(size=1000)

        metrics = self.evaluator.evaluate(y_true, y_prob)

        # Random predictions should give ~0.5 AUC
        assert 0.4 < metrics["auc_roc"] < 0.6
        assert metrics["n_samples"] == 1000

    def test_threshold_effect(self):
        y_true = np.array([0, 0, 1, 1])
        y_prob = np.array([0.3, 0.6, 0.7, 0.9])

        # High threshold = fewer false positives
        metrics_high = self.evaluator.evaluate(y_true, y_prob, threshold=0.8)
        metrics_low = self.evaluator.evaluate(y_true, y_prob, threshold=0.4)

        assert metrics_high["precision"] >= metrics_low["precision"]
        assert metrics_low["recall"] >= metrics_high["recall"]

    def test_find_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])

        threshold = self.evaluator._find_optimal_threshold(y_true, y_prob)

        assert 0.0 < threshold < 1.0
