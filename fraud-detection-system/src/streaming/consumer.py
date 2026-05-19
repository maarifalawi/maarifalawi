"""
Kafka Fraud Scoring Consumer
Consumes transactions from Kafka, computes features, and scores for fraud in real-time.
"""

import argparse
import json
import signal
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Deque, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from kafka import KafkaConsumer, KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    KafkaConsumer = None
    KafkaProducer = None
    KafkaError = None


class CustomerHistoryStore:
    """
    In-memory store for customer transaction history.
    Maintains a sliding window of recent transactions per customer.
    
    In production, this would be backed by Redis or a similar
    low-latency key-value store.
    """

    def __init__(self, max_history: int = 1000, max_age_hours: int = 720):
        self.max_history = max_history
        self.max_age_hours = max_age_hours
        self._store: Dict[str, Deque[dict]] = defaultdict(
            lambda: deque(maxlen=max_history)
        )

    def add(self, customer_id: str, transaction: dict):
        """Add a transaction to customer history."""
        self._store[customer_id].append(transaction)

    def get(self, customer_id: str) -> Optional[pd.DataFrame]:
        """Get customer history as DataFrame."""
        history = self._store.get(customer_id)
        if not history or len(history) == 0:
            return None
        return pd.DataFrame(list(history))

    def size(self) -> int:
        """Total transactions stored."""
        return sum(len(h) for h in self._store.values())

    def n_customers(self) -> int:
        """Number of customers tracked."""
        return len(self._store)



class FraudScoringConsumer:
    """
    Kafka consumer that scores transactions for fraud in real-time.

    Pipeline:
    1. Consume transaction from 'transactions' topic
    2. Compute real-time features
    3. Lookup customer history for historical features
    4. Score with ML model
    5. Publish results to 'fraud-scores' topic
    6. Update customer history store

    Performance target: < 100ms end-to-end latency per transaction.
    """

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        input_topic: str = "transactions",
        output_topic: str = "fraud-scores",
        alert_topic: str = "fraud-alerts",
        group_id: str = "fraud-scoring-group",
        model_path: str = "data/models/fraud_model_v1",
        alert_threshold: float = 0.7,
    ):
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic
        self.alert_topic = alert_topic
        self.group_id = group_id
        self.model_path = model_path
        self.alert_threshold = alert_threshold

        self.consumer = None
        self.result_producer = None
        self.history_store = CustomerHistoryStore()
        self.feature_engine = None
        self.predictor = None

        self.running = False
        self.processed = 0
        self.frauds_detected = 0
        self.total_latency_ms = 0.0
        self.latencies: List[float] = []

    def initialize(self):
        """Initialize all components."""
        if KafkaConsumer is None:
            raise ImportError("kafka-python not installed: pip install kafka-python")

        print("Initializing Fraud Scoring Consumer...")

        # Initialize Kafka consumer
        self.consumer = KafkaConsumer(
            self.input_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id=self.group_id,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            max_poll_records=100,
            fetch_max_wait_ms=100,
        )
        print(f"  Consumer connected: {self.input_topic}")

        # Initialize result producer
        self.result_producer = KafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
            compression_type="gzip",
        )
        print(f"  Producer connected: {self.output_topic}")

        # Initialize feature engine
        from src.features.engine import FeatureEngine
        self.feature_engine = FeatureEngine()
        print("  Feature engine ready")

        # Initialize predictor
        from src.model.predictor import FraudPredictor
        self.predictor = FraudPredictor(model_dir=self.model_path)
        self.predictor.load()
        print("  ML model loaded")

        print("\n  All components initialized!")


    def process_transaction(self, transaction: dict) -> dict:
        """
        Score a single transaction for fraud.

        Args:
            transaction: Raw transaction event

        Returns:
            Scoring result dict
        """
        start_time = time.perf_counter()

        customer_id = transaction.get("customer_id", "unknown")

        # Get customer history
        history = self.history_store.get(customer_id)

        # Compute features
        features = self.feature_engine.compute_single(
            transaction, customer_history=history
        )

        # Score with model
        prediction = self.predictor.predict_single(features)

        # Build result
        latency_ms = (time.perf_counter() - start_time) * 1000

        result = {
            "transaction_id": transaction.get("transaction_id"),
            "customer_id": customer_id,
            "timestamp": transaction.get("timestamp"),
            "amount": transaction.get("amount"),
            "fraud_probability": prediction["fraud_probability"],
            "is_fraud_predicted": prediction["is_fraud"],
            "risk_level": prediction["risk_level"],
            "processing_latency_ms": round(latency_ms, 2),
            "scored_at": datetime.now().isoformat(),
        }

        # Update history store (after scoring to avoid self-reference)
        self.history_store.add(customer_id, transaction)

        # Track metrics
        self.processed += 1
        self.latencies.append(latency_ms)
        if prediction["is_fraud"]:
            self.frauds_detected += 1

        return result

    def run(self):
        """
        Main consumer loop. Process transactions continuously.
        """
        if self.consumer is None:
            self.initialize()

        print(f"\nStarting fraud scoring consumer...")
        print(f"  Input topic: {self.input_topic}")
        print(f"  Output topic: {self.output_topic}")
        print(f"  Alert threshold: {self.alert_threshold}")
        print(f"  Press Ctrl+C to stop\n")

        self.running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        stats_interval = time.time()

        while self.running:
            # Poll for messages
            messages = self.consumer.poll(timeout_ms=100)

            for topic_partition, records in messages.items():
                for record in records:
                    transaction = record.value

                    # Score transaction
                    result = self.process_transaction(transaction)

                    # Publish score to output topic
                    self.result_producer.send(
                        self.output_topic,
                        key=result["customer_id"],
                        value=result,
                    )

                    # Publish alert if high risk
                    if result["fraud_probability"] >= self.alert_threshold:
                        self._publish_alert(result, transaction)

            # Print stats every 5 seconds
            if time.time() - stats_interval >= 5.0:
                self._print_stats()
                stats_interval = time.time()

        self._shutdown()


    def _publish_alert(self, result: dict, transaction: dict):
        """Publish fraud alert for high-risk transactions."""
        alert = {
            "alert_type": "FRAUD_DETECTED",
            "severity": result["risk_level"],
            "transaction_id": result["transaction_id"],
            "customer_id": result["customer_id"],
            "amount": result["amount"],
            "fraud_probability": result["fraud_probability"],
            "timestamp": result["timestamp"],
            "merchant_category": transaction.get("merchant_category"),
            "merchant_country": transaction.get("merchant_country"),
            "alerted_at": datetime.now().isoformat(),
        }

        self.result_producer.send(
            self.alert_topic,
            key=result["customer_id"],
            value=alert,
        )

    def _print_stats(self):
        """Print consumer performance statistics."""
        if self.processed == 0:
            print("  Waiting for messages...")
            return

        recent_latencies = self.latencies[-1000:]  # Last 1000
        p50 = np.percentile(recent_latencies, 50)
        p95 = np.percentile(recent_latencies, 95)
        p99 = np.percentile(recent_latencies, 99)

        print(
            f"  Processed: {self.processed:,} | "
            f"Frauds: {self.frauds_detected:,} | "
            f"Latency p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms | "
            f"History: {self.history_store.n_customers()} customers"
        )

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        print("\n  Shutting down consumer...")
        self.running = False

    def _shutdown(self):
        """Clean up resources."""
        if self.consumer:
            self.consumer.close()
        if self.result_producer:
            self.result_producer.flush()
            self.result_producer.close()

        print(f"\n  Consumer shut down.")
        print(f"  Total processed: {self.processed:,}")
        print(f"  Frauds detected: {self.frauds_detected:,}")
        if self.latencies:
            print(f"  Avg latency: {np.mean(self.latencies):.1f}ms")

    def get_metrics(self) -> dict:
        """Get consumer metrics for monitoring."""
        metrics = {
            "processed_total": self.processed,
            "frauds_detected": self.frauds_detected,
            "fraud_rate": self.frauds_detected / max(1, self.processed),
            "history_customers": self.history_store.n_customers(),
            "history_transactions": self.history_store.size(),
        }

        if self.latencies:
            recent = self.latencies[-1000:]
            metrics.update({
                "latency_p50_ms": float(np.percentile(recent, 50)),
                "latency_p95_ms": float(np.percentile(recent, 95)),
                "latency_p99_ms": float(np.percentile(recent, 99)),
                "latency_mean_ms": float(np.mean(recent)),
            })

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Fraud scoring Kafka consumer")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--input-topic", default="transactions")
    parser.add_argument("--output-topic", default="fraud-scores")
    parser.add_argument("--alert-topic", default="fraud-alerts")
    parser.add_argument("--group-id", default="fraud-scoring-group")
    parser.add_argument("--model-path", default="data/models/fraud_model_v1")
    parser.add_argument("--alert-threshold", type=float, default=0.7)
    args = parser.parse_args()

    consumer = FraudScoringConsumer(
        bootstrap_servers=args.bootstrap_servers,
        input_topic=args.input_topic,
        output_topic=args.output_topic,
        alert_topic=args.alert_topic,
        group_id=args.group_id,
        model_path=args.model_path,
        alert_threshold=args.alert_threshold,
    )
    consumer.initialize()
    consumer.run()


if __name__ == "__main__":
    main()
