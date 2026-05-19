"""
Data Loader Module
Handles loading and preprocessing of transaction datasets.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class DataLoader:
    """
    Load and preprocess transaction data for model training.
    Supports CSV files and provides train/test splitting with temporal awareness.
    """

    def __init__(self, data_path: str = "data/raw/transactions.csv"):
        self.data_path = Path(data_path)
        self.df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        """Load transaction data from CSV."""
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.data_path}\n"
                "Run: python -m src.ingestion.simulator --output data/raw/transactions.csv"
            )

        self.df = pd.read_csv(self.data_path, parse_dates=["timestamp"])
        print(f"Loaded {len(self.df):,} transactions from {self.data_path}")
        print(f"  Fraud rate: {self.df['is_fraud'].mean()*100:.2f}%")
        print(f"  Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        return self.df

    def temporal_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data temporally (time-based split to prevent data leakage).
        
        This is critical for fraud detection — random splits would leak
        future information into training data.

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.df is None:
            self.load()

        df_sorted = self.df.sort_values("timestamp").reset_index(drop=True)
        n = len(df_sorted)

        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df_sorted.iloc[:train_end].copy()
        val_df = df_sorted.iloc[train_end:val_end].copy()
        test_df = df_sorted.iloc[val_end:].copy()

        print(f"\nTemporal split:")
        print(f"  Train: {len(train_df):,} records ({train_df['is_fraud'].mean()*100:.2f}% fraud)")
        print(f"  Val:   {len(val_df):,} records ({val_df['is_fraud'].mean()*100:.2f}% fraud)")
        print(f"  Test:  {len(test_df):,} records ({test_df['is_fraud'].mean()*100:.2f}% fraud)")
        print(f"  Train period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
        print(f"  Test period:  {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

        return train_df, val_df, test_df

    def random_split(
        self,
        test_size: float = 0.2,
        val_size: float = 0.1,
        stratify: bool = True,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Standard random split with stratification on fraud label.
        Use temporal_split for production; this is for quick experiments.
        """
        if self.df is None:
            self.load()

        stratify_col = self.df["is_fraud"] if stratify else None

        train_val, test_df = train_test_split(
            self.df, test_size=test_size, stratify=stratify_col, random_state=random_state
        )

        stratify_col2 = train_val["is_fraud"] if stratify else None
        relative_val_size = val_size / (1 - test_size)

        train_df, val_df = train_test_split(
            train_val, test_size=relative_val_size, stratify=stratify_col2, random_state=random_state
        )

        print(f"\nRandom split (stratified={stratify}):")
        print(f"  Train: {len(train_df):,} records ({train_df['is_fraud'].mean()*100:.2f}% fraud)")
        print(f"  Val:   {len(val_df):,} records ({val_df['is_fraud'].mean()*100:.2f}% fraud)")
        print(f"  Test:  {len(test_df):,} records ({test_df['is_fraud'].mean()*100:.2f}% fraud)")

        return train_df, val_df, test_df

    def get_statistics(self) -> dict:
        """Get descriptive statistics about the dataset."""
        if self.df is None:
            self.load()

        stats = {
            "n_transactions": len(self.df),
            "n_customers": self.df["customer_id"].nunique(),
            "n_fraud": int(self.df["is_fraud"].sum()),
            "fraud_rate": float(self.df["is_fraud"].mean()),
            "avg_amount": float(self.df["amount"].mean()),
            "median_amount": float(self.df["amount"].median()),
            "max_amount": float(self.df["amount"].max()),
            "date_range_days": (
                self.df["timestamp"].max() - self.df["timestamp"].min()
            ).days,
            "merchant_categories": self.df["merchant_category"].nunique(),
            "countries": self.df["merchant_country"].nunique(),
        }

        if "fraud_type" in self.df.columns:
            fraud_df = self.df[self.df["is_fraud"] == 1]
            stats["fraud_types"] = fraud_df["fraud_type"].value_counts().to_dict()

        return stats
