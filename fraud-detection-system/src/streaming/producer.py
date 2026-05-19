"""
Kafka Transaction Producer
Streams synthetic or real transactions to Kafka for real-time processing.
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from typing import Optional

try:
    from kafka import KafkaProducer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaError = None


class TransactionProducer:
    """
    Kafka producer that streams transaction events.

    Supports:
    - Simulated transactions (using TransactionSimulator)
    - Configurable throughput (transactions per second)
    - Graceful shutdown
    - Message serialization with JSON
    - Partition by customer_id for ordered processing
    """

    DEFAULT_CONFIG = {
        "bootstrap_servers": "localhost:9092",
        "topic": "transactions",
        "client_id": "fraud-detection-producer",
        "acks": "all",
        "retries": 3,
        "batch_size": 16384,
        "linger_ms": 10,
        "compression_type": "gzip",
    }

    def __init__(
        self,
        bootstrap_servers: str = "localhost:9092",
        topic: str = "transactions",
        config: Optional[dict] = None,
    ):
        self.topic = topic
        self.config = self.DEFAULT_CONFIG.copy()
        if config:
            self.config.update(config)
        self.config["bootstrap_servers"] = bootstrap_servers

        self.producer: Optional[KafkaProducer] = None
        self.running = False
        self.messages_sent = 0
        self.errors = 0

    def connect(self):
        """Initialize Kafka producer connection."""
        if KafkaProducer is None:
            raise ImportError(
                "kafka-python not installed: pip install kafka-python"
            )

        print(f"Connecting to Kafka: {self.config['bootstrap_servers']}")
        self.producer = KafkaProducer(
            bootstrap_servers=self.config["bootstrap_servers"],
            client_id=self.config["client_id"],
            acks=self.config["acks"],
            retries=self.config["retries"],
            batch_size=self.config["batch_size"],
            linger_ms=self.config["linger_ms"],
            compression_type=self.config["compression_type"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8") if k else None,
        )
        print(f"  Connected! Topic: {self.topic}")

    def send(self, transaction: dict) -> bool:
        """
        Send a single transaction to Kafka.

        Args:
            transaction: Transaction event dict

        Returns:
            True if sent successfully
        """
        if self.producer is None:
            self.connect()

        try:
            # Use customer_id as key for partition ordering
            key = transaction.get("customer_id", "unknown")

            future = self.producer.send(
                self.topic,
                key=key,
                value=transaction,
            )
            # Don't block - fire and forget for throughput
            future.add_callback(self._on_success)
            future.add_errback(self._on_error)

            self.messages_sent += 1
            return True

        except Exception as e:
            self.errors += 1
            print(f"  ERROR sending message: {e}")
            return False

    def _on_success(self, metadata):
        """Callback for successful sends."""
        pass  # Counted in send()

    def _on_error(self, exc):
        """Callback for failed sends."""
        self.errors += 1
        print(f"  Kafka send error: {exc}")


    def stream_simulated(
        self,
        tps: int = 100,
        n_customers: int = 5000,
        fraud_ratio: float = 0.018,
    ):
        """
        Stream simulated transactions at specified TPS.

        Args:
            tps: Transactions per second
            n_customers: Number of customer profiles
            fraud_ratio: Fraud transaction ratio
        """
        from src.ingestion.simulator import TransactionSimulator

        simulator = TransactionSimulator(
            n_customers=n_customers, fraud_ratio=fraud_ratio
        )

        print(f"\nStreaming transactions at {tps} TPS...")
        print(f"  Fraud ratio: {fraud_ratio*100:.1f}%")
        print(f"  Press Ctrl+C to stop\n")

        self.running = True
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

        start_time = time.time()
        batch_start = time.time()
        batch_count = 0

        for transaction in simulator.generate_stream(tps=tps):
            if not self.running:
                break

            self.send(transaction)
            batch_count += 1

            # Print stats every second
            if time.time() - batch_start >= 1.0:
                elapsed = time.time() - start_time
                actual_tps = self.messages_sent / elapsed
                print(
                    f"  [+{elapsed:.0f}s] Sent: {self.messages_sent:,} | "
                    f"TPS: {actual_tps:.0f} | "
                    f"Errors: {self.errors}"
                )
                batch_start = time.time()
                batch_count = 0

        self._shutdown()

    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown signal."""
        print("\n  Shutting down producer...")
        self.running = False

    def _shutdown(self):
        """Flush and close the producer."""
        if self.producer:
            self.producer.flush()
            self.producer.close()
            print(f"\n  Producer closed. Total sent: {self.messages_sent:,}")

    def get_stats(self) -> dict:
        """Get producer statistics."""
        return {
            "messages_sent": self.messages_sent,
            "errors": self.errors,
            "error_rate": self.errors / max(1, self.messages_sent),
        }


def main():
    parser = argparse.ArgumentParser(description="Stream transactions to Kafka")
    parser.add_argument("--bootstrap-servers", default="localhost:9092")
    parser.add_argument("--topic", default="transactions")
    parser.add_argument("--tps", type=int, default=100, help="Transactions per second")
    parser.add_argument("--n-customers", type=int, default=5000)
    parser.add_argument("--fraud-ratio", type=float, default=0.018)
    parser.add_argument("--mode", choices=["simulate"], default="simulate")
    args = parser.parse_args()

    producer = TransactionProducer(
        bootstrap_servers=args.bootstrap_servers,
        topic=args.topic,
    )
    producer.connect()

    if args.mode == "simulate":
        producer.stream_simulated(
            tps=args.tps,
            n_customers=args.n_customers,
            fraud_ratio=args.fraud_ratio,
        )


if __name__ == "__main__":
    main()
