"""
Prometheus Metrics Collector
Exposes application metrics for Prometheus scraping.
"""

import time
from typing import Optional

try:
    from prometheus_client import (
        Counter,
        Gauge,
        Histogram,
        Info,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


class MetricsCollector:
    """
    Collects and exposes metrics for the fraud detection system.

    Metrics:
    - Prediction counters (total, fraud, legit)
    - Latency histograms (p50, p95, p99)
    - Model performance gauges
    - System health indicators
    - Drift detection alerts
    """

    def __init__(self, service_name: str = "fraud_detection"):
        self.service_name = service_name
        self.start_time = time.time()

        if not PROMETHEUS_AVAILABLE:
            print("WARNING: prometheus_client not installed. Metrics disabled.")
            self._enabled = False
            return

        self._enabled = True

        # Counters
        self.predictions_total = Counter(
            f"{service_name}_predictions_total",
            "Total number of predictions made",
            ["model", "result"],
        )

        self.transactions_processed = Counter(
            f"{service_name}_transactions_processed_total",
            "Total transactions processed",
        )

        self.errors_total = Counter(
            f"{service_name}_errors_total",
            "Total errors encountered",
            ["error_type"],
        )

        # Histograms
        self.prediction_latency = Histogram(
            f"{service_name}_prediction_latency_seconds",
            "Prediction latency in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
        )

        self.transaction_amount = Histogram(
            f"{service_name}_transaction_amount",
            "Transaction amount distribution",
            buckets=[1, 10, 50, 100, 500, 1000, 5000, 10000],
        )

        # Gauges
        self.model_auc = Gauge(
            f"{service_name}_model_auc_roc",
            "Current model AUC-ROC score",
        )

        self.fraud_rate = Gauge(
            f"{service_name}_fraud_rate",
            "Current detected fraud rate",
        )

        self.drift_psi = Gauge(
            f"{service_name}_drift_psi",
            "Current drift PSI score",
            ["feature"],
        )

        self.active_customers = Gauge(
            f"{service_name}_active_customers",
            "Number of active customers in history store",
        )

        self.uptime_seconds = Gauge(
            f"{service_name}_uptime_seconds",
            "Service uptime in seconds",
        )

        # Info
        self.model_info = Info(
            f"{service_name}_model",
            "Model information",
        )

    def record_prediction(
        self,
        model: str,
        is_fraud: bool,
        latency_seconds: float,
        amount: float,
    ):
        """Record a prediction event."""
        if not self._enabled:
            return

        result = "fraud" if is_fraud else "legit"
        self.predictions_total.labels(model=model, result=result).inc()
        self.transactions_processed.inc()
        self.prediction_latency.observe(latency_seconds)
        self.transaction_amount.observe(amount)

    def record_error(self, error_type: str):
        """Record an error event."""
        if not self._enabled:
            return
        self.errors_total.labels(error_type=error_type).inc()

    def update_model_metrics(self, auc_roc: float, fraud_rate: float):
        """Update model performance gauges."""
        if not self._enabled:
            return
        self.model_auc.set(auc_roc)
        self.fraud_rate.set(fraud_rate)

    def update_drift_metrics(self, feature_psi: dict):
        """Update drift PSI gauges per feature."""
        if not self._enabled:
            return
        for feature, psi_value in feature_psi.items():
            self.drift_psi.labels(feature=feature).set(psi_value)

    def update_system_metrics(self, n_customers: int):
        """Update system-level gauges."""
        if not self._enabled:
            return
        self.active_customers.set(n_customers)
        self.uptime_seconds.set(time.time() - self.start_time)

    def set_model_info(self, version: str, n_features: int, trained_at: str):
        """Set model metadata."""
        if not self._enabled:
            return
        self.model_info.info({
            "version": version,
            "n_features": str(n_features),
            "trained_at": trained_at,
        })

    def get_metrics(self) -> Optional[bytes]:
        """Generate Prometheus metrics output."""
        if not self._enabled:
            return None
        return generate_latest()
