"""
Real-time Feature Engineering
Computes features that can be calculated on-the-fly for each transaction.
These features capture immediate transaction characteristics and short-term patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class RealtimeFeatures:
    """
    Real-time features computed per transaction without requiring historical aggregation.
    These are fast to compute and suitable for streaming inference.
    """

    # High-risk merchant categories
    HIGH_RISK_CATEGORIES = {
        "cash_advance", "gambling", "wire_transfer", "crypto_exchange", "jewelry"
    }

    # High-risk countries (higher fraud rates)
    HIGH_RISK_COUNTRIES = {"NG", "RU", "BR", "CN"}

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all real-time features for a DataFrame of transactions.

        Args:
            df: Transaction DataFrame with columns:
                timestamp, amount, merchant_category, merchant_country, is_online, etc.

        Returns:
            DataFrame with additional feature columns
        """
        features = df.copy()

        # Time-based features
        features = self._time_features(features)

        # Amount-based features
        features = self._amount_features(features)

        # Categorical risk features
        features = self._categorical_features(features)

        # Transaction metadata features
        features = self._metadata_features(features)

        return features

    def compute_single(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute real-time features for a single transaction (streaming mode).

        Args:
            transaction: Single transaction dict

        Returns:
            Dict with computed feature values
        """
        ts = pd.Timestamp(transaction["timestamp"])

        features = {}

        # Time features
        features["hour_of_day"] = ts.hour
        features["day_of_week"] = ts.dayofweek
        features["is_weekend"] = int(ts.dayofweek >= 5)
        features["is_night"] = int(ts.hour >= 22 or ts.hour <= 5)
        features["is_business_hours"] = int(9 <= ts.hour <= 17 and ts.dayofweek < 5)
        features["minute_of_day"] = ts.hour * 60 + ts.minute
        features["day_of_month"] = ts.day
        features["is_month_start"] = int(ts.day <= 3)
        features["is_month_end"] = int(ts.day >= 28)

        # Cyclical time encoding
        features["hour_sin"] = float(np.sin(2 * np.pi * ts.hour / 24))
        features["hour_cos"] = float(np.cos(2 * np.pi * ts.hour / 24))
        features["dow_sin"] = float(np.sin(2 * np.pi * ts.dayofweek / 7))
        features["dow_cos"] = float(np.cos(2 * np.pi * ts.dayofweek / 7))

        # Amount features
        amount = transaction["amount"]
        features["amount_log"] = float(np.log1p(amount))
        features["is_round_amount"] = int(amount == int(amount))
        features["is_small_amount"] = int(amount < 5.0)
        features["is_large_amount"] = int(amount > 1000.0)
        features["is_very_large_amount"] = int(amount > 5000.0)
        features["amount_cents"] = round(amount % 1, 2)
        features["amount_bucket"] = self._get_amount_bucket(amount)

        # Categorical features
        category = transaction.get("merchant_category", "unknown")
        country = transaction.get("merchant_country", "US")
        features["is_high_risk_category"] = int(category in self.HIGH_RISK_CATEGORIES)
        features["is_high_risk_country"] = int(country in self.HIGH_RISK_COUNTRIES)
        features["is_online"] = int(transaction.get("is_online", False))
        features["is_foreign"] = int(country != "US")  # Can be made dynamic

        return features

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features from timestamp."""
        ts = pd.to_datetime(df["timestamp"])

        df["hour_of_day"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)
        df["is_night"] = ((ts.dt.hour >= 22) | (ts.dt.hour <= 5)).astype(int)
        df["is_business_hours"] = (
            (ts.dt.hour >= 9) & (ts.dt.hour <= 17) & (ts.dt.dayofweek < 5)
        ).astype(int)
        df["minute_of_day"] = ts.dt.hour * 60 + ts.dt.minute
        df["day_of_month"] = ts.dt.day
        df["is_month_start"] = (ts.dt.day <= 3).astype(int)
        df["is_month_end"] = (ts.dt.day >= 28).astype(int)

        # Cyclical encoding (helps models understand circular nature of time)
        df["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        df["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        df["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        return df

    def _amount_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract amount-based features."""
        df["amount_log"] = np.log1p(df["amount"])
        df["is_round_amount"] = (df["amount"] == df["amount"].astype(int)).astype(int)
        df["is_small_amount"] = (df["amount"] < 5.0).astype(int)
        df["is_large_amount"] = (df["amount"] > 1000.0).astype(int)
        df["is_very_large_amount"] = (df["amount"] > 5000.0).astype(int)
        df["amount_cents"] = (df["amount"] % 1).round(2)
        df["amount_bucket"] = df["amount"].apply(self._get_amount_bucket)

        return df

    def _categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract categorical risk features."""
        df["is_high_risk_category"] = df["merchant_category"].isin(
            self.HIGH_RISK_CATEGORIES
        ).astype(int)
        df["is_high_risk_country"] = df["merchant_country"].isin(
            self.HIGH_RISK_COUNTRIES
        ).astype(int)
        df["is_foreign"] = (df["merchant_country"] != "US").astype(int)

        return df

    def _metadata_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract metadata-based features."""
        if "is_online" in df.columns:
            df["is_online"] = df["is_online"].astype(int)

        return df

    @staticmethod
    def _get_amount_bucket(amount: float) -> int:
        """Bucket amount into discrete ranges."""
        if amount < 1:
            return 0
        elif amount < 10:
            return 1
        elif amount < 50:
            return 2
        elif amount < 100:
            return 3
        elif amount < 500:
            return 4
        elif amount < 1000:
            return 5
        elif amount < 5000:
            return 6
        else:
            return 7
