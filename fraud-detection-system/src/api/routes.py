"""
API Route Handlers
"""

import time
from datetime import datetime
from typing import List

import numpy as np
from fastapi import APIRouter, HTTPException

from .schemas import (
    BatchScoreResponse,
    BatchTransactionRequest,
    FraudScoreResponse,
    HealthResponse,
    MetricsResponse,
    ModelInfoResponse,
    TransactionRequest,
)

router = APIRouter()

# These will be injected by the main app
_predictor = None
_feature_engine = None
_start_time = time.time()
_prediction_count = 0
_fraud_count = 0
_latencies: List[float] = []


def set_dependencies(predictor, feature_engine):
    """Inject dependencies into route handlers."""
    global _predictor, _feature_engine
    _predictor = predictor
    _feature_engine = feature_engine


@router.post("/predict", response_model=FraudScoreResponse, tags=["Prediction"])
async def predict_fraud(transaction: TransactionRequest):
    """
    Score a single transaction for fraud.
    
    Returns fraud probability, risk level, and prediction latency.
    Target: < 10ms latency.
    """
    global _prediction_count, _fraud_count

    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()

    # Build transaction dict
    txn_dict = transaction.model_dump()
    if txn_dict.get("timestamp") is None:
        txn_dict["timestamp"] = datetime.now().isoformat()

    # Compute features
    features = _feature_engine.compute_single(txn_dict, customer_history=None)

    # Score
    result = _predictor.predict_single(features)

    latency_ms = (time.perf_counter() - start) * 1000
    _latencies.append(latency_ms)
    _prediction_count += 1
    if result["is_fraud"]:
        _fraud_count += 1

    return FraudScoreResponse(
        transaction_id=txn_dict.get("transaction_id"),
        customer_id=txn_dict["customer_id"],
        fraud_probability=result["fraud_probability"],
        is_fraud=result["is_fraud"],
        risk_level=result["risk_level"],
        threshold=result["threshold"],
        latency_ms=round(latency_ms, 2),
        models_used=result["models_used"],
        scored_at=datetime.now().isoformat(),
    )


@router.post("/predict/batch", response_model=BatchScoreResponse, tags=["Prediction"])
async def predict_batch(request: BatchTransactionRequest):
    """
    Score a batch of transactions for fraud.
    Efficient for bulk processing.
    """
    global _prediction_count, _fraud_count

    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    batch_start = time.perf_counter()
    results = []

    for transaction in request.transactions:
        txn_dict = transaction.model_dump()
        if txn_dict.get("timestamp") is None:
            txn_dict["timestamp"] = datetime.now().isoformat()

        features = _feature_engine.compute_single(txn_dict, customer_history=None)
        result = _predictor.predict_single(features)

        _prediction_count += 1
        if result["is_fraud"]:
            _fraud_count += 1

        results.append(FraudScoreResponse(
            transaction_id=txn_dict.get("transaction_id"),
            customer_id=txn_dict["customer_id"],
            fraud_probability=result["fraud_probability"],
            is_fraud=result["is_fraud"],
            risk_level=result["risk_level"],
            threshold=result["threshold"],
            latency_ms=result["latency_ms"],
            models_used=result["models_used"],
            scored_at=datetime.now().isoformat(),
        ))

    batch_latency = (time.perf_counter() - batch_start) * 1000
    n_fraud = sum(1 for r in results if r.is_fraud)
    avg_lat = batch_latency / max(1, len(results))

    return BatchScoreResponse(
        results=results,
        total_processed=len(results),
        total_fraud_detected=n_fraud,
        avg_latency_ms=round(avg_lat, 2),
        batch_latency_ms=round(batch_latency, 2),
    )



@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy" if _predictor and _predictor.is_loaded else "degraded",
        model_loaded=_predictor is not None and _predictor.is_loaded,
        uptime_seconds=round(time.time() - _start_time, 1),
        total_predictions=_prediction_count,
        version="1.0.0",
    )


@router.get("/model/info", response_model=ModelInfoResponse, tags=["System"])
async def model_info():
    """Get information about the loaded model."""
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return ModelInfoResponse(
        model_name="fraud_ensemble_v1",
        n_features=len(_predictor.feature_names),
        feature_names=_predictor.feature_names,
        training_metrics={},
        threshold=_predictor.threshold,
    )


@router.get("/metrics", response_model=MetricsResponse, tags=["Monitoring"])
async def get_metrics():
    """Get system performance metrics."""
    latencies = _latencies[-10000:] if _latencies else [0]

    return MetricsResponse(
        total_predictions=_prediction_count,
        total_frauds_detected=_fraud_count,
        fraud_detection_rate=_fraud_count / max(1, _prediction_count),
        avg_latency_ms=round(float(np.mean(latencies)), 2),
        p95_latency_ms=round(float(np.percentile(latencies, 95)), 2),
        p99_latency_ms=round(float(np.percentile(latencies, 99)), 2),
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@router.get("/metrics/prometheus", tags=["Monitoring"])
async def prometheus_metrics():
    """Prometheus-compatible metrics endpoint."""
    latencies = _latencies[-10000:] if _latencies else [0]

    metrics_text = f"""# HELP fraud_predictions_total Total predictions made
# TYPE fraud_predictions_total counter
fraud_predictions_total {_prediction_count}

# HELP fraud_detected_total Total frauds detected
# TYPE fraud_detected_total counter
fraud_detected_total {_fraud_count}

# HELP fraud_detection_rate Current fraud detection rate
# TYPE fraud_detection_rate gauge
fraud_detection_rate {_fraud_count / max(1, _prediction_count):.6f}

# HELP prediction_latency_ms Prediction latency in milliseconds
# TYPE prediction_latency_ms summary
prediction_latency_ms{{quantile="0.5"}} {float(np.percentile(latencies, 50)):.2f}
prediction_latency_ms{{quantile="0.95"}} {float(np.percentile(latencies, 95)):.2f}
prediction_latency_ms{{quantile="0.99"}} {float(np.percentile(latencies, 99)):.2f}

# HELP uptime_seconds Service uptime in seconds
# TYPE uptime_seconds gauge
uptime_seconds {time.time() - _start_time:.0f}
"""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(content=metrics_text, media_type="text/plain")
