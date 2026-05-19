"""
Feature Engine
Orchestrates real-time and historical feature computation.
"""

import pandas as pd
from typing import Dict, Any, List, Optional

from .realtime import RealtimeFeatures
from .historical import HistoricalFeatures


class FeatureEngine:
    """
    Main feature engineering orchestrator.
    Combines real-time and historical features into a single feature vector.
    """

    # Features to exclude from model input
    EXCLUDE_COLS = [
        "transaction_id", "customer_id", "card_id", "timestamp",
        "currency", "merchant_category", "merchant_country",
        "merchant_city", "is_fraud", "fraud_type",
    ]

    def __init__(self):
        self.realtime = RealtimeFeatures()
        self.historical = HistoricalFeatures()
        self.feature_names: Optional[List[str]] = None


    def compute_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all features for a batch of transactions (training mode).

        Args:
            df: Raw transaction DataFrame

        Returns:
            DataFrame with all features computed
        """
        print("Computing real-time features...")
        df = self.realtime.compute(df)

        print("Computing historical features...")
        df = self.historical.compute(df)

        # Store feature names
        self.feature_names = [
            col for col in df.columns if col not in self.EXCLUDE_COLS
        ]

        return df

    def compute_single(
        self,
        transaction: Dict[str, Any],
        customer_history: Optional[pd.DataFrame] = None,
    ) -> Dict[str, float]:
        """
        Compute features for a single transaction (inference mode).

        Args:
            transaction: Single transaction dict
            customer_history: Optional DataFrame of past transactions

        Returns:
            Dict of feature_name -> value
        """
        # Real-time features
        features = self.realtime.compute_single(transaction)

        # Historical features
        hist_features = self.historical.compute_single(
            transaction, customer_history
        )
        features.update(hist_features)

        return features

    def get_feature_names(self) -> List[str]:
        """Get ordered list of feature names used by the model."""
        if self.feature_names is None:
            raise ValueError(
                "Feature names not set. Run compute_batch first."
            )
        return self.feature_names

    def get_model_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract only the model-input features from a fully featured DF.

        Args:
            df: DataFrame with all columns (including metadata)

        Returns:
            DataFrame with only numeric feature columns
        """
        feature_cols = [
            col for col in df.columns if col not in self.EXCLUDE_COLS
        ]
        # Keep only numeric columns
        numeric_df = df[feature_cols].select_dtypes(include=["number"])
        return numeric_df
