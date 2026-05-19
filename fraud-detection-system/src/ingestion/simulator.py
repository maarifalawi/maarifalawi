"""
Transaction Data Simulator
Generates realistic synthetic credit card transaction data with fraud patterns.
"""

import argparse
import hashlib
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


class TransactionSimulator:
    """
    Generates synthetic credit card transactions that mimic real-world patterns.
    
    Fraud patterns implemented:
    - Card testing: Many small transactions in short time
    - Velocity abuse: Sudden spike in transaction count
    - Geographic anomaly: Transaction far from usual location
    - Amount anomaly: Unusually large transaction
    - Midnight fraud: Transactions at unusual hours
    - Account takeover: Change in merchant category patterns
    """

    MERCHANT_CATEGORIES = [
        "grocery", "gas_station", "restaurant", "online_shopping",
        "electronics", "travel", "entertainment", "healthcare",
        "utilities", "clothing", "jewelry", "cash_advance",
        "gambling", "wire_transfer", "crypto_exchange"
    ]

    COUNTRIES = [
        "US", "US", "US", "US", "US",  # weighted towards US
        "GB", "CA", "DE", "FR", "BR",
        "NG", "RU", "CN", "IN", "JP"
    ]

    CITIES = {
        "US": ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix"],
        "GB": ["London", "Manchester", "Birmingham"],
        "CA": ["Toronto", "Vancouver", "Montreal"],
        "DE": ["Berlin", "Munich", "Hamburg"],
        "FR": ["Paris", "Lyon", "Marseille"],
        "BR": ["Sao Paulo", "Rio de Janeiro"],
        "NG": ["Lagos", "Abuja"],
        "RU": ["Moscow", "St Petersburg"],
        "CN": ["Beijing", "Shanghai", "Shenzhen"],
        "IN": ["Mumbai", "Delhi", "Bangalore"],
        "JP": ["Tokyo", "Osaka"],
    }

    def __init__(
        self,
        n_customers: int = 5000,
        fraud_ratio: float = 0.018,  # ~1.8% fraud rate (realistic)
        seed: int = 42,
    ):
        self.n_customers = n_customers
        self.fraud_ratio = fraud_ratio
        self.rng = np.random.default_rng(seed)
        random.seed(seed)

        # Generate customer profiles
        self.customers = self._generate_customer_profiles()

    def _generate_customer_profiles(self) -> pd.DataFrame:
        """Generate realistic customer profiles with spending patterns."""
        customers = []
        for i in range(self.n_customers):
            country = random.choice(self.COUNTRIES)
            city = random.choice(self.CITIES[country])
            avg_amount = self.rng.lognormal(mean=3.5, sigma=1.0)
            preferred_categories = random.sample(
                self.MERCHANT_CATEGORIES, k=random.randint(3, 7)
            )
            customers.append({
                "customer_id": f"CUST_{i:06d}",
                "home_country": country,
                "home_city": city,
                "avg_transaction_amount": avg_amount,
                "std_transaction_amount": avg_amount * 0.4,
                "preferred_categories": preferred_categories,
                "avg_daily_transactions": self.rng.uniform(1, 8),
                "credit_limit": avg_amount * self.rng.uniform(20, 100),
                "account_age_days": int(self.rng.uniform(30, 3650)),
            })
        return pd.DataFrame(customers)

    def _generate_card_id(self, customer_id: str) -> str:
        """Generate a deterministic card ID from customer ID."""
        return hashlib.md5(customer_id.encode()).hexdigest()[:16].upper()

    def _generate_legitimate_transaction(
        self, customer: pd.Series, timestamp: datetime
    ) -> dict:
        """Generate a single legitimate transaction."""
        amount = max(
            0.50,
            self.rng.normal(
                customer["avg_transaction_amount"],
                customer["std_transaction_amount"]
            )
        )
        category = random.choice(customer["preferred_categories"])
        country = customer["home_country"]
        city = random.choice(self.CITIES[country])

        # Small chance of travel (legitimate foreign transaction)
        if self.rng.random() < 0.05:
            country = random.choice(self.COUNTRIES)
            city = random.choice(self.CITIES[country])

        return {
            "transaction_id": hashlib.sha256(
                f"{customer['customer_id']}_{timestamp.isoformat()}_{self.rng.random()}".encode()
            ).hexdigest()[:24],
            "customer_id": customer["customer_id"],
            "card_id": self._generate_card_id(customer["customer_id"]),
            "timestamp": timestamp,
            "amount": round(amount, 2),
            "currency": "USD",
            "merchant_category": category,
            "merchant_country": country,
            "merchant_city": city,
            "is_online": random.random() < 0.35,
            "is_fraud": 0,
            "fraud_type": None,
        }

    def _generate_fraud_transaction(
        self, customer: pd.Series, timestamp: datetime, fraud_type: str
    ) -> dict:
        """Generate a fraudulent transaction based on fraud pattern type."""
        txn = self._generate_legitimate_transaction(customer, timestamp)
        txn["is_fraud"] = 1
        txn["fraud_type"] = fraud_type

        if fraud_type == "card_testing":
            txn["amount"] = round(self.rng.uniform(0.50, 5.00), 2)
            txn["is_online"] = True

        elif fraud_type == "amount_anomaly":
            txn["amount"] = round(
                customer["avg_transaction_amount"] * self.rng.uniform(5, 20), 2
            )

        elif fraud_type == "geographic_anomaly":
            foreign_countries = [c for c in self.COUNTRIES if c != customer["home_country"]]
            country = random.choice(foreign_countries)
            txn["merchant_country"] = country
            txn["merchant_city"] = random.choice(self.CITIES[country])

        elif fraud_type == "midnight_fraud":
            hour = random.randint(1, 4)
            txn["timestamp"] = timestamp.replace(hour=hour, minute=random.randint(0, 59))
            txn["amount"] = round(
                customer["avg_transaction_amount"] * self.rng.uniform(2, 8), 2
            )

        elif fraud_type == "velocity_abuse":
            txn["amount"] = round(self.rng.uniform(50, 500), 2)
            txn["is_online"] = True

        elif fraud_type == "category_anomaly":
            unusual_cats = [
                c for c in self.MERCHANT_CATEGORIES
                if c not in customer["preferred_categories"]
            ]
            if unusual_cats:
                txn["merchant_category"] = random.choice(unusual_cats)
            txn["amount"] = round(
                customer["avg_transaction_amount"] * self.rng.uniform(2, 10), 2
            )

        return txn

    def generate(
        self,
        n_records: int = 100000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Generate a dataset of synthetic transactions.

        Args:
            n_records: Total number of transactions to generate
            start_date: Start date (ISO format), defaults to 90 days ago
            end_date: End date (ISO format), defaults to today

        Returns:
            DataFrame with synthetic transaction data
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        start_dt = datetime.fromisoformat(start_date)
        end_dt = datetime.fromisoformat(end_date)
        time_range_seconds = int((end_dt - start_dt).total_seconds())

        n_fraud = int(n_records * self.fraud_ratio)
        n_legit = n_records - n_fraud

        fraud_types = [
            "card_testing", "amount_anomaly", "geographic_anomaly",
            "midnight_fraud", "velocity_abuse", "category_anomaly"
        ]

        transactions = []

        # Generate legitimate transactions
        print(f"Generating {n_legit:,} legitimate transactions...")
        for _ in range(n_legit):
            customer = self.customers.iloc[self.rng.integers(0, self.n_customers)]
            offset_seconds = self.rng.integers(0, time_range_seconds)
            timestamp = start_dt + timedelta(seconds=int(offset_seconds))

            # Realistic hour distribution (more during business hours)
            hour_weights = np.array([0.02, 0.01, 0.01, 0.01, 0.02, 0.03,
                          0.05, 0.07, 0.08, 0.09, 0.09, 0.09,
                          0.08, 0.07, 0.06, 0.05, 0.04, 0.04,
                          0.03, 0.02, 0.02, 0.01, 0.01, 0.01])
            hour_weights = hour_weights / hour_weights.sum()  # Normalize to sum=1
            hour = self.rng.choice(24, p=hour_weights)
            timestamp = timestamp.replace(
                hour=hour, minute=self.rng.integers(0, 60)
            )

            txn = self._generate_legitimate_transaction(customer, timestamp)
            transactions.append(txn)

        # Generate fraud transactions (with clustering for velocity abuse)
        print(f"Generating {n_fraud:,} fraudulent transactions...")
        fraud_per_type = n_fraud // len(fraud_types)

        for fraud_type in fraud_types:
            for _ in range(fraud_per_type):
                customer = self.customers.iloc[self.rng.integers(0, self.n_customers)]
                offset_seconds = self.rng.integers(0, time_range_seconds)
                timestamp = start_dt + timedelta(seconds=int(offset_seconds))

                txn = self._generate_fraud_transaction(customer, timestamp, fraud_type)
                transactions.append(txn)

        # Handle remaining fraud to reach exact n_fraud
        remaining = n_fraud - (fraud_per_type * len(fraud_types))
        for _ in range(remaining):
            customer = self.customers.iloc[self.rng.integers(0, self.n_customers)]
            offset_seconds = self.rng.integers(0, time_range_seconds)
            timestamp = start_dt + timedelta(seconds=int(offset_seconds))
            fraud_type = random.choice(fraud_types)
            txn = self._generate_fraud_transaction(customer, timestamp, fraud_type)
            transactions.append(txn)

        df = pd.DataFrame(transactions)
        df = df.sort_values("timestamp").reset_index(drop=True)

        print(f"\nDataset generated:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Fraud transactions: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Unique customers: {df['customer_id'].nunique():,}")

        return df

    def generate_stream(self, tps: int = 10):
        """
        Generate transactions as a stream (generator) for Kafka producer.

        Args:
            tps: Transactions per second

        Yields:
            dict: Single transaction event
        """
        import time

        while True:
            customer = self.customers.iloc[self.rng.integers(0, self.n_customers)]
            timestamp = datetime.now()

            # Decide if this transaction is fraud
            is_fraud = self.rng.random() < self.fraud_ratio

            if is_fraud:
                fraud_type = random.choice([
                    "card_testing", "amount_anomaly", "geographic_anomaly",
                    "midnight_fraud", "velocity_abuse", "category_anomaly"
                ])
                txn = self._generate_fraud_transaction(customer, timestamp, fraud_type)
            else:
                txn = self._generate_legitimate_transaction(customer, timestamp)

            # Convert timestamp to ISO string for serialization
            txn["timestamp"] = txn["timestamp"].isoformat()
            yield txn
            time.sleep(1.0 / tps)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data")
    parser.add_argument("--output", type=str, default="data/raw/transactions.csv")
    parser.add_argument("--n-records", type=int, default=100000)
    parser.add_argument("--n-customers", type=int, default=5000)
    parser.add_argument("--fraud-ratio", type=float, default=0.018)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)

    args = parser.parse_args()

    simulator = TransactionSimulator(
        n_customers=args.n_customers,
        fraud_ratio=args.fraud_ratio,
        seed=args.seed,
    )

    df = simulator.generate(
        n_records=args.n_records,
        start_date=args.start_date,
        end_date=args.end_date,
    )

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
