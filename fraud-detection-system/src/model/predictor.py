"""
Fraud Predictor
Fast inference engine for real-time fraud scoring.
"""

import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


class FraudPredictor:
    """
    Production inference engine for fraud detection.
    Loads trained models and provides fast prediction.

    Optimized for:
    - Sub-5ms single prediction latency
    - Batch prediction for bulk scoring
    - Ensemble predictions with configurable weights
    """

    def __init__(
        self,
        model_dir: str = "data/models/fraud_model_v1",
        xgb_weight: float = 0.5,
        threshold: float = 0.5,
    ):
        self.model_dir = Path(model_dir)
        self.xgb_weight = xgb_weight
        self.lgb_weight = 1.0 - xgb_weight
        self.threshold = threshold

        self.xgb_model = None
        self.lgb_model = None
        self.feature_names: List[str] = []
        self.is_loaded = False

    def load(self):
        """Load models and metadata from disk."""
        print(f"Loading models from: {self.model_dir}")

        # Load XGBoost
        xgb_path = self.model_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                self.xgb_model = pickle.load(f)
            print(f"  XGBoost loaded")

        # Load LightGBM
        lgb_path = self.model_dir / "lightgbm_model.pkl"
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                self.lgb_model = pickle.load(f)
            print(f"  LightGBM loaded")

        # Load metadata
        meta_path = self.model_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            self.feature_names = metadata.get("feature_names", [])
            print(f"  Features: {len(self.feature_names)}")

        if self.xgb_model is None and self.lgb_model is None:
            raise FileNotFoundError(
                f"No models found in {self.model_dir}. "
                "Train first: python -m src.model.trainer"
            )

        self.is_loaded = True
        print("  Models ready for inference!")

    def predict_single(
        self, features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Predict fraud probability for a single transaction.

        Args:
            features: Dict of feature_name -> value

        Returns:
            Dict with prediction results
        """
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()

        # Build feature vector in correct order
        feature_vector = np.array([
            features.get(name, 0.0) for name in self.feature_names
        ]).reshape(1, -1)

        # Get predictions from available models
        probs = []
        weights = []

        if self.xgb_model is not None:
            xgb_prob = self.xgb_model.predict_proba(feature_vector)[0, 1]
            probs.append(xgb_prob)
            weights.append(self.xgb_weight)

        if self.lgb_model is not None:
            lgb_prob = self.lgb_model.predict_proba(feature_vector)[0, 1]
            probs.append(lgb_prob)
            weights.append(self.lgb_weight)

        # Weighted ensemble
        total_weight = sum(weights)
        ensemble_prob = sum(p * w for p, w in zip(probs, weights)) / total_weight

        latency_ms = (time.perf_counter() - start_time) * 1000

        return {
            "fraud_probability": float(ensemble_prob),
            "is_fraud": bool(ensemble_prob >= self.threshold),
            "risk_level": self._get_risk_level(ensemble_prob),
            "threshold": self.threshold,
            "latency_ms": round(latency_ms, 2),
            "models_used": {
                "xgboost": float(probs[0]) if self.xgb_model else None,
                "lightgbm": float(probs[-1]) if self.lgb_model else None,
            },
        }

    def predict_batch(
        self, features_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Predict fraud probability for a batch of transactions.

        Args:
            features_df: DataFrame with feature columns

        Returns:
            DataFrame with predictions added
        """
        if not self.is_loaded:
            self.load()

        start_time = time.perf_counter()

        # Ensure correct feature order
        X = features_df[self.feature_names].values

        probs = []
        weights = []

        if self.xgb_model is not None:
            xgb_probs = self.xgb_model.predict_proba(X)[:, 1]
            probs.append(xgb_probs)
            weights.append(self.xgb_weight)

        if self.lgb_model is not None:
            lgb_probs = self.lgb_model.predict_proba(X)[:, 1]
            probs.append(lgb_probs)
            weights.append(self.lgb_weight)

        # Weighted ensemble
        total_weight = sum(weights)
        ensemble_probs = sum(p * w for p, w in zip(probs, weights)) / total_weight

        latency_ms = (time.perf_counter() - start_time) * 1000

        results = features_df.copy()
        results["fraud_probability"] = ensemble_probs
        results["is_fraud_predicted"] = (ensemble_probs >= self.threshold).astype(int)
        results["risk_level"] = [self._get_risk_level(p) for p in ensemble_probs]

        print(f"  Batch prediction: {len(features_df)} records in {latency_ms:.1f}ms "
              f"({latency_ms/len(features_df):.3f}ms/record)")

        return results

    @staticmethod
    def _get_risk_level(probability: float) -> str:
        """Convert probability to human-readable risk level."""
        if probability >= 0.9:
            return "CRITICAL"
        elif probability >= 0.7:
            return "HIGH"
        elif probability >= 0.4:
            return "MEDIUM"
        elif probability >= 0.2:
            return "LOW"
        else:
            return "MINIMAL"
