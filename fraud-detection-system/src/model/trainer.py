"""
Model Training Pipeline
XGBoost + LightGBM ensemble with class imbalance handling.
"""

import argparse
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTETomek
except ImportError:
    SMOTE = None
    ADASYN = None
    SMOTETomek = None

from .evaluation import ModelEvaluator



class FocalLossObjective:
    """
    Focal Loss for XGBoost/LightGBM.
    Reduces the loss contribution from easy-to-classify examples,
    focusing the model on hard negatives (missed frauds).

    FL(p) = -alpha * (1-p)^gamma * log(p) for y=1
    FL(p) = -(1-alpha) * p^gamma * log(1-p) for y=0
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.75):
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Compute gradient and hessian for focal loss."""
        gamma = self.gamma
        alpha = self.alpha

        # Sigmoid of predictions
        p = 1.0 / (1.0 + np.exp(-y_pred))
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # Compute focal weight
        y = y_true
        pt = np.where(y == 1, p, 1 - p)
        at = np.where(y == 1, alpha, 1 - alpha)
        focal_weight = at * (1 - pt) ** gamma

        # Gradient
        grad = focal_weight * (p - y)

        # Hessian (approximation)
        hess = focal_weight * p * (1 - p)
        hess = np.maximum(hess, 1e-7)

        return grad, hess



class FraudModelTrainer:
    """
    Production-grade fraud detection model trainer.

    Features:
    - XGBoost + LightGBM training with ensemble
    - Class imbalance handling (SMOTE, scale_pos_weight, focal loss)
    - Stratified K-Fold cross-validation
    - Hyperparameter configuration
    - Model persistence
    """

    DEFAULT_XGB_PARAMS = {
        "objective": "binary:logistic",
        "eval_metric": ["auc", "aucpr", "logloss"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "early_stopping_rounds": 50,
    }

    DEFAULT_LGB_PARAMS = {
        "objective": "binary",
        "metric": ["auc", "average_precision", "binary_logloss"],
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    def __init__(
        self,
        model_dir: str = "data/models",
        use_focal_loss: bool = True,
        use_smote: bool = True,
        focal_gamma: float = 2.0,
        focal_alpha: float = 0.75,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.use_focal_loss = use_focal_loss
        self.use_smote = use_smote
        self.focal_loss = FocalLossObjective(gamma=focal_gamma, alpha=focal_alpha)

        self.xgb_model = None
        self.lgb_model = None
        self.feature_names: List[str] = []
        self.evaluator = ModelEvaluator()
        self.training_metrics: Dict = {}


    def _apply_smote(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to handle class imbalance."""
        if SMOTE is None:
            print("  WARNING: imbalanced-learn not installed. Skipping SMOTE.")
            return X, y

        print(f"  Applying SMOTE (before: {y.sum()}/{len(y)} fraud)...")
        smote = SMOTETomek(random_state=42) if SMOTETomek else SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print(f"  After SMOTE: {y_res.sum()}/{len(y_res)} fraud")
        return X_res, y_res

    def train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Optional[Dict] = None,
    ) -> Any:
        """Train XGBoost model with class imbalance handling."""
        if xgb is None:
            raise ImportError("xgboost not installed: pip install xgboost")

        print("\n" + "=" * 50)
        print("Training XGBoost...")
        print("=" * 50)

        model_params = self.DEFAULT_XGB_PARAMS.copy()
        if params:
            model_params.update(params)

        # Calculate scale_pos_weight for imbalance
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        scale_pos_weight = n_neg / max(n_pos, 1)
        model_params["scale_pos_weight"] = scale_pos_weight
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")

        # Apply SMOTE if enabled
        X_fit, y_fit = X_train, y_train
        if self.use_smote:
            X_fit, y_fit = self._apply_smote(X_train, y_train)

        # Extract early stopping params
        early_stopping = model_params.pop("early_stopping_rounds", 50)
        eval_metric = model_params.pop("eval_metric", ["auc"])

        start_time = time.time()

        self.xgb_model = xgb.XGBClassifier(**model_params)
        self.xgb_model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        y_prob = self.xgb_model.predict_proba(X_val)[:, 1]
        metrics = self.evaluator.evaluate(y_val, y_prob)
        print(f"  Val AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Val AUC-PR:  {metrics['auc_pr']:.4f}")
        print(f"  Val F1:      {metrics['f1']:.4f}")

        self.training_metrics["xgboost"] = metrics
        return self.xgb_model


    def train_lightgbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        params: Optional[Dict] = None,
    ) -> Any:
        """Train LightGBM model with class imbalance handling."""
        if lgb is None:
            raise ImportError("lightgbm not installed: pip install lightgbm")

        print("\n" + "=" * 50)
        print("Training LightGBM...")
        print("=" * 50)

        model_params = self.DEFAULT_LGB_PARAMS.copy()
        if params:
            model_params.update(params)

        # Calculate is_unbalance or scale_pos_weight
        n_neg = (y_train == 0).sum()
        n_pos = (y_train == 1).sum()
        model_params["scale_pos_weight"] = n_neg / max(n_pos, 1)
        print(f"  scale_pos_weight: {model_params['scale_pos_weight']:.2f}")

        # Apply SMOTE if enabled
        X_fit, y_fit = X_train, y_train
        if self.use_smote:
            X_fit, y_fit = self._apply_smote(X_train, y_train)

        start_time = time.time()

        callbacks = [lgb.early_stopping(50), lgb.log_evaluation(50)]
        self.lgb_model = lgb.LGBMClassifier(**model_params)
        self.lgb_model.fit(
            X_fit, y_fit,
            eval_set=[(X_val, y_val)],
            callbacks=callbacks,
        )

        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.1f}s")

        # Evaluate
        y_prob = self.lgb_model.predict_proba(X_val)[:, 1]
        metrics = self.evaluator.evaluate(y_val, y_prob)
        print(f"  Val AUC-ROC: {metrics['auc_roc']:.4f}")
        print(f"  Val AUC-PR:  {metrics['auc_pr']:.4f}")
        print(f"  Val F1:      {metrics['f1']:.4f}")

        self.training_metrics["lightgbm"] = metrics
        return self.lgb_model


    def train_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        xgb_weight: float = 0.5,
    ) -> Dict:
        """
        Train both XGBoost and LightGBM, then create a weighted ensemble.

        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            xgb_weight: Weight for XGBoost in ensemble (LGB gets 1-weight)

        Returns:
            Dict with ensemble metrics
        """
        print("\n" + "=" * 60)
        print("TRAINING ENSEMBLE (XGBoost + LightGBM)")
        print("=" * 60)

        # Train both models
        self.train_xgboost(X_train, y_train, X_val, y_val)
        self.train_lightgbm(X_train, y_train, X_val, y_val)

        # Ensemble predictions
        xgb_prob = self.xgb_model.predict_proba(X_val)[:, 1]
        lgb_prob = self.lgb_model.predict_proba(X_val)[:, 1]
        ensemble_prob = xgb_weight * xgb_prob + (1 - xgb_weight) * lgb_prob

        # Evaluate ensemble
        print("\n" + "=" * 50)
        print("Ensemble Results:")
        print("=" * 50)
        metrics = self.evaluator.print_report(y_val, ensemble_prob)
        self.training_metrics["ensemble"] = metrics

        return metrics

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        model_type: str = "xgboost",
    ) -> Dict:
        """
        Perform stratified K-Fold cross-validation.

        Args:
            X, y: Full training data
            n_folds: Number of folds
            model_type: "xgboost" or "lightgbm"

        Returns:
            Dict with per-fold and aggregate metrics
        """
        print(f"\n{'='*50}")
        print(f"Cross-Validation ({n_folds} folds, {model_type})")
        print(f"{'='*50}")

        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"\n--- Fold {fold+1}/{n_folds} ---")
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            if model_type == "xgboost":
                self.train_xgboost(X_train, y_train, X_val, y_val)
                y_prob = self.xgb_model.predict_proba(X_val)[:, 1]
            else:
                self.train_lightgbm(X_train, y_train, X_val, y_val)
                y_prob = self.lgb_model.predict_proba(X_val)[:, 1]

            metrics = self.evaluator.evaluate(y_val, y_prob)
            fold_metrics.append(metrics)

        # Aggregate
        agg = {
            "mean_auc_roc": np.mean([m["auc_roc"] for m in fold_metrics]),
            "std_auc_roc": np.std([m["auc_roc"] for m in fold_metrics]),
            "mean_auc_pr": np.mean([m["auc_pr"] for m in fold_metrics]),
            "std_auc_pr": np.std([m["auc_pr"] for m in fold_metrics]),
            "mean_f1": np.mean([m["f1"] for m in fold_metrics]),
            "std_f1": np.std([m["f1"] for m in fold_metrics]),
            "fold_metrics": fold_metrics,
        }

        print(f"\n{'='*50}")
        print(f"CV Results ({model_type}):")
        print(f"  AUC-ROC: {agg['mean_auc_roc']:.4f} +/- {agg['std_auc_roc']:.4f}")
        print(f"  AUC-PR:  {agg['mean_auc_pr']:.4f} +/- {agg['std_auc_pr']:.4f}")
        print(f"  F1:      {agg['mean_f1']:.4f} +/- {agg['std_f1']:.4f}")

        return agg


    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get feature importance from trained models."""
        importances = []

        if self.xgb_model is not None and self.feature_names:
            xgb_imp = self.xgb_model.feature_importances_
            for name, imp in zip(self.feature_names, xgb_imp):
                importances.append({
                    "feature": name, "importance": imp, "model": "xgboost"
                })

        if self.lgb_model is not None and self.feature_names:
            lgb_imp = self.lgb_model.feature_importances_
            for name, imp in zip(self.feature_names, lgb_imp):
                importances.append({
                    "feature": name, "importance": imp, "model": "lightgbm"
                })

        df = pd.DataFrame(importances)
        if len(df) == 0:
            return df

        # Average importance across models
        avg_imp = df.groupby("feature")["importance"].mean().reset_index()
        avg_imp = avg_imp.sort_values("importance", ascending=False).head(top_n)

        print(f"\nTop {top_n} Features:")
        print("-" * 40)
        for _, row in avg_imp.iterrows():
            print(f"  {row['feature']:<35} {row['importance']:.4f}")

        return avg_imp

    def save(self, name: str = "fraud_model"):
        """Save trained models to disk."""
        save_dir = self.model_dir / name
        save_dir.mkdir(parents=True, exist_ok=True)

        if self.xgb_model is not None:
            path = save_dir / "xgboost_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.xgb_model, f)
            print(f"  Saved XGBoost: {path}")

        if self.lgb_model is not None:
            path = save_dir / "lightgbm_model.pkl"
            with open(path, "wb") as f:
                pickle.dump(self.lgb_model, f)
            print(f"  Saved LightGBM: {path}")

        # Save metadata
        metadata = {
            "feature_names": self.feature_names,
            "training_metrics": self.training_metrics,
            "n_features": len(self.feature_names),
        }
        meta_path = save_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"  Saved metadata: {meta_path}")

    def load(self, name: str = "fraud_model"):
        """Load trained models from disk."""
        save_dir = self.model_dir / name

        xgb_path = save_dir / "xgboost_model.pkl"
        if xgb_path.exists():
            with open(xgb_path, "rb") as f:
                self.xgb_model = pickle.load(f)
            print(f"  Loaded XGBoost from: {xgb_path}")

        lgb_path = save_dir / "lightgbm_model.pkl"
        if lgb_path.exists():
            with open(lgb_path, "rb") as f:
                self.lgb_model = pickle.load(f)
            print(f"  Loaded LightGBM from: {lgb_path}")

        meta_path = save_dir / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
            self.feature_names = metadata.get("feature_names", [])
            self.training_metrics = metadata.get("training_metrics", {})
            print(f"  Loaded metadata: {len(self.feature_names)} features")


def main():
    """CLI entry point for model training."""
    parser = argparse.ArgumentParser(description="Train fraud detection model")
    parser.add_argument("--data", type=str, default="data/raw/transactions.csv")
    parser.add_argument("--model-dir", type=str, default="data/models")
    parser.add_argument("--model-name", type=str, default="fraud_model_v1")
    parser.add_argument("--no-smote", action="store_true")
    parser.add_argument("--no-focal-loss", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=0, help="CV folds (0=no CV)")
    args = parser.parse_args()

    import sys
    sys.path.insert(0, ".")
    from src.ingestion.data_loader import DataLoader
    from src.features.engine import FeatureEngine

    # Load data
    loader = DataLoader(args.data)
    train_df, val_df, test_df = loader.temporal_split()

    # Feature engineering
    engine = FeatureEngine()
    train_featured = engine.compute_batch(train_df)
    val_featured = engine.compute_batch(val_df)
    test_featured = engine.compute_batch(test_df)

    # Get model input
    X_train = engine.get_model_features(train_featured).values
    y_train = train_featured["is_fraud"].values
    X_val = engine.get_model_features(val_featured).values
    y_val = val_featured["is_fraud"].values
    X_test = engine.get_model_features(test_featured).values
    y_test = test_featured["is_fraud"].values

    feature_names = engine.get_model_features(train_featured).columns.tolist()

    # Train
    trainer = FraudModelTrainer(
        model_dir=args.model_dir,
        use_smote=not args.no_smote,
        use_focal_loss=not args.no_focal_loss,
    )
    trainer.feature_names = feature_names

    if args.cv_folds > 0:
        trainer.cross_validate(X_train, y_train, n_folds=args.cv_folds)

    # Train ensemble
    trainer.train_ensemble(X_train, y_train, X_val, y_val)

    # Test set evaluation
    print("\n" + "=" * 60)
    print("TEST SET EVALUATION")
    print("=" * 60)
    xgb_prob = trainer.xgb_model.predict_proba(X_test)[:, 1]
    lgb_prob = trainer.lgb_model.predict_proba(X_test)[:, 1]
    ensemble_prob = 0.5 * xgb_prob + 0.5 * lgb_prob
    trainer.evaluator.print_report(y_test, ensemble_prob)

    # Feature importance
    trainer.get_feature_importance()

    # Save
    trainer.save(args.model_name)
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
