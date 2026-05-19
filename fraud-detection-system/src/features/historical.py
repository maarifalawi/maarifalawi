"""
Historical Feature Engineering
Computes aggregated features based on customer transaction history.
These features capture behavioral patterns and deviations from normal behavior.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional


class HistoricalFeatures:
    """
    Historical/aggregated features that require looking at past transactions.
    These capture customer behavior patterns and are critical for detecting anomalies.
    
    Features include:
    - Velocity features (transaction frequency in time windows)
    - Amount deviation from personal baseline
    - Geographic diversity
    - Category diversity
    - Recency features
    """

    # Time windows for aggregation (in hours)
    TIME_WINDOWS = {
        "1h": 1,
        "6h": 6,
        "12h": 12,
        "24h": 24,
        "48h": 48,
        "7d": 168,
        "30d": 720,
    }

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute historical features for all transactions.
        IMPORTANT: Uses only past transactions to avoid data leakage.

        Args:
            df: Sorted (by timestamp) transaction DataFrame

        Returns:
            DataFrame with historical feature columns added
        """
        df = df.sort_values("timestamp").reset_index(drop=True)

        # Group by customer for efficiency
        print("Computing historical features...")

        # Velocity features (transaction count in time windows)
        df = self._velocity_features(df)

        # Amount statistics (relative to customer history)
        df = self._amount_deviation_features(df)

        # Geographic features
        df = self._geographic_features(df)

        # Category features
        df = self._category_features(df)

        # Recency features
        df = self._recency_features(df)

        # Cross-feature interactions
        df = self._interaction_features(df)

        print(f"  Total features added: {len([c for c in df.columns if c not in ['transaction_id', 'customer_id', 'card_id', 'timestamp', 'amount', 'currency', 'merchant_category', 'merchant_country', 'merchant_city', 'is_online', 'is_fraud', 'fraud_type']])}")

        return df

    def compute_single(
        self,
        transaction: Dict,
        history: pd.DataFrame,
    ) -> Dict:
        """
        Compute historical features for a single transaction given customer history.
        Used in streaming/real-time mode.

        Args:
            transaction: Current transaction dict
            history: DataFrame of customer's past transactions

        Returns:
            Dict with computed historical features
        """
        features = {}
        current_ts = pd.Timestamp(transaction["timestamp"])
        amount = transaction["amount"]

        if history is None or len(history) == 0:
            # New customer - return defaults
            features.update(self._default_features())
            return features

        history = history.sort_values("timestamp")
        hist_ts = pd.to_datetime(history["timestamp"])

        # Velocity features
        for window_name, hours in self.TIME_WINDOWS.items():
            window_start = current_ts - pd.Timedelta(hours=hours)
            mask = hist_ts >= window_start
            window_txns = history[mask]

            features[f"txn_count_{window_name}"] = len(window_txns)
            features[f"txn_amount_sum_{window_name}"] = float(window_txns["amount"].sum()) if len(window_txns) > 0 else 0.0
            features[f"txn_amount_mean_{window_name}"] = float(window_txns["amount"].mean()) if len(window_txns) > 0 else 0.0
            features[f"txn_amount_max_{window_name}"] = float(window_txns["amount"].max()) if len(window_txns) > 0 else 0.0

        # Amount deviation
        hist_amounts = history["amount"]
        if len(hist_amounts) > 0:
            features["amount_zscore"] = float((amount - hist_amounts.mean()) / (hist_amounts.std() + 1e-8))
            features["amount_percentile"] = float((hist_amounts < amount).mean())
            features["amount_ratio_to_mean"] = float(amount / (hist_amounts.mean() + 1e-8))
            features["amount_ratio_to_max"] = float(amount / (hist_amounts.max() + 1e-8))
        else:
            features["amount_zscore"] = 0.0
            features["amount_percentile"] = 0.5
            features["amount_ratio_to_mean"] = 1.0
            features["amount_ratio_to_max"] = 1.0

        # Geographic features
        if "merchant_country" in history.columns:
            unique_countries = history["merchant_country"].nunique()
            features["n_unique_countries_hist"] = unique_countries
            features["is_new_country"] = int(
                transaction.get("merchant_country", "") not in history["merchant_country"].values
            )
        else:
            features["n_unique_countries_hist"] = 0
            features["is_new_country"] = 0

        # Category features
        if "merchant_category" in history.columns:
            unique_cats = history["merchant_category"].nunique()
            features["n_unique_categories_hist"] = unique_cats
            features["is_new_category"] = int(
                transaction.get("merchant_category", "") not in history["merchant_category"].values
            )
        else:
            features["n_unique_categories_hist"] = 0
            features["is_new_category"] = 0

        # Recency
        last_ts = hist_ts.max()
        features["hours_since_last_txn"] = float((current_ts - last_ts).total_seconds() / 3600)
        features["total_txn_count"] = len(history)

        return features

    def _velocity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute velocity features: transaction count and amount in time windows.
        Uses rolling window approach per customer.
        """
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"])

        for window_name, hours in self.TIME_WINDOWS.items():
            col_count = f"txn_count_{window_name}"
            col_sum = f"txn_amount_sum_{window_name}"
            col_mean = f"txn_amount_mean_{window_name}"
            col_max = f"txn_amount_max_{window_name}"

            # Initialize with defaults
            df[col_count] = 0
            df[col_sum] = 0.0
            df[col_mean] = 0.0
            df[col_max] = 0.0

            # Group by customer and compute rolling stats
            for customer_id, group in df.groupby("customer_id"):
                indices = group.index
                timestamps = group["timestamp_dt"].values
                amounts = group["amount"].values

                counts = []
                sums = []
                means = []
                maxes = []

                for i, (ts, amt) in enumerate(zip(timestamps, amounts)):
                    window_start = ts - np.timedelta64(hours, "h")
                    # Look at past transactions only (excluding current)
                    mask = (timestamps[:i] >= window_start)
                    past_amounts = amounts[:i][mask]

                    counts.append(len(past_amounts))
                    sums.append(float(past_amounts.sum()) if len(past_amounts) > 0 else 0.0)
                    means.append(float(past_amounts.mean()) if len(past_amounts) > 0 else 0.0)
                    maxes.append(float(past_amounts.max()) if len(past_amounts) > 0 else 0.0)

                df.loc[indices, col_count] = counts
                df.loc[indices, col_sum] = sums
                df.loc[indices, col_mean] = means
                df.loc[indices, col_max] = maxes

        df.drop(columns=["timestamp_dt"], inplace=True, errors="ignore")
        return df

    def _amount_deviation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute how much the current amount deviates from customer's historical pattern.
        """
        df["amount_zscore"] = 0.0
        df["amount_percentile"] = 0.5
        df["amount_ratio_to_mean"] = 1.0
        df["amount_ratio_to_max"] = 1.0

        for customer_id, group in df.groupby("customer_id"):
            indices = group.index
            amounts = group["amount"].values

            zscores = []
            percentiles = []
            ratio_means = []
            ratio_maxes = []

            for i in range(len(amounts)):
                past = amounts[:i]
                current = amounts[i]

                if len(past) > 1:
                    mean = past.mean()
                    std = past.std()
                    zscores.append((current - mean) / (std + 1e-8))
                    percentiles.append(float((past < current).mean()))
                    ratio_means.append(current / (mean + 1e-8))
                    ratio_maxes.append(current / (past.max() + 1e-8))
                else:
                    zscores.append(0.0)
                    percentiles.append(0.5)
                    ratio_means.append(1.0)
                    ratio_maxes.append(1.0)

            df.loc[indices, "amount_zscore"] = zscores
            df.loc[indices, "amount_percentile"] = percentiles
            df.loc[indices, "amount_ratio_to_mean"] = ratio_means
            df.loc[indices, "amount_ratio_to_max"] = ratio_maxes

        return df

    def _geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute geographic diversity and anomaly features."""
        df["n_unique_countries_hist"] = 0
        df["is_new_country"] = 0

        for customer_id, group in df.groupby("customer_id"):
            indices = group.index
            countries = group["merchant_country"].values

            n_unique = []
            is_new = []

            seen_countries = set()
            for i, country in enumerate(countries):
                n_unique.append(len(seen_countries))
                is_new.append(int(country not in seen_countries))
                seen_countries.add(country)

            df.loc[indices, "n_unique_countries_hist"] = n_unique
            df.loc[indices, "is_new_country"] = is_new

        return df

    def _category_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute merchant category diversity and anomaly features."""
        df["n_unique_categories_hist"] = 0
        df["is_new_category"] = 0

        for customer_id, group in df.groupby("customer_id"):
            indices = group.index
            categories = group["merchant_category"].values

            n_unique = []
            is_new = []

            seen_cats = set()
            for i, cat in enumerate(categories):
                n_unique.append(len(seen_cats))
                is_new.append(int(cat not in seen_cats))
                seen_cats.add(cat)

            df.loc[indices, "n_unique_categories_hist"] = n_unique
            df.loc[indices, "is_new_category"] = is_new

        return df

    def _recency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute recency features (time since last transaction)."""
        df["hours_since_last_txn"] = 0.0
        df["total_txn_count"] = 0

        timestamps = pd.to_datetime(df["timestamp"])

        for customer_id, group in df.groupby("customer_id"):
            indices = group.index
            ts_values = timestamps[indices].values

            hours_since = []
            for i in range(len(ts_values)):
                if i == 0:
                    hours_since.append(0.0)
                else:
                    diff = (ts_values[i] - ts_values[i-1]) / np.timedelta64(1, "h")
                    hours_since.append(float(diff))

            df.loc[indices, "hours_since_last_txn"] = hours_since
            df.loc[indices, "total_txn_count"] = range(len(indices))

        return df

    def _interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute interaction features combining multiple signals."""
        # High velocity + high amount = suspicious
        df["velocity_amount_score"] = (
            df.get("txn_count_1h", 0) * df["amount_log"]
        )

        # Night + foreign + large = suspicious
        if "is_night" in df.columns:
            df["night_foreign_large"] = (
                df.get("is_night", 0) *
                df.get("is_foreign", 0) *
                df.get("is_large_amount", 0)
            )

        # New country + large amount
        df["new_country_large_amount"] = (
            df["is_new_country"] * df.get("is_large_amount", 0)
        )

        return df

    def _default_features(self) -> Dict:
        """Return default feature values for new customers."""
        features = {}
        for window_name in self.TIME_WINDOWS:
            features[f"txn_count_{window_name}"] = 0
            features[f"txn_amount_sum_{window_name}"] = 0.0
            features[f"txn_amount_mean_{window_name}"] = 0.0
            features[f"txn_amount_max_{window_name}"] = 0.0

        features["amount_zscore"] = 0.0
        features["amount_percentile"] = 0.5
        features["amount_ratio_to_mean"] = 1.0
        features["amount_ratio_to_max"] = 1.0
        features["n_unique_countries_hist"] = 0
        features["is_new_country"] = 1
        features["n_unique_categories_hist"] = 0
        features["is_new_category"] = 1
        features["hours_since_last_txn"] = 0.0
        features["total_txn_count"] = 0
        features["velocity_amount_score"] = 0.0
        features["night_foreign_large"] = 0
        features["new_country_large_amount"] = 0

        return features
