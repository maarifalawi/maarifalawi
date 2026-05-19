"""
API Pydantic Models / Schemas
"""

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionRequest(BaseModel):
    """Single transaction for fraud scoring."""
    transaction_id: Optional[str] = Field(None, description="Unique transaction ID")
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., gt=0, description="Transaction amount in USD")
    merchant_category: str = Field(..., description="Merchant category code")
    merchant_country: str = Field(default="US", description="Merchant country ISO code")
    merchant_city: Optional[str] = Field(None, description="Merchant city")
    is_online: bool = Field(default=False, description="Whether online transaction")
    timestamp: Optional[str] = Field(None, description="ISO timestamp (defaults to now)")
    currency: str = Field(default="USD", description="Currency code")

    class Config:
        json_schema_extra = {
            "example": {
                "customer_id": "CUST_000123",
                "amount": 299.99,
                "merchant_category": "electronics",
                "merchant_country": "US",
                "is_online": True,
                "timestamp": "2025-01-15T14:30:00",
            }
        }


class FraudScoreResponse(BaseModel):
    """Fraud scoring result."""
    transaction_id: Optional[str]
    customer_id: str
    fraud_probability: float = Field(..., ge=0, le=1)
    is_fraud: bool
    risk_level: str = Field(..., description="MINIMAL/LOW/MEDIUM/HIGH/CRITICAL")
    threshold: float
    latency_ms: float
    models_used: Dict[str, Optional[float]]
    scored_at: str


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for scoring."""
    transactions: List[TransactionRequest]


class BatchScoreResponse(BaseModel):
    """Batch scoring results."""
    results: List[FraudScoreResponse]
    total_processed: int
    total_fraud_detected: int
    avg_latency_ms: float
    batch_latency_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    uptime_seconds: float
    total_predictions: int
    version: str


class ModelInfoResponse(BaseModel):
    """Model information."""
    model_name: str
    n_features: int
    feature_names: List[str]
    training_metrics: Dict
    threshold: float


class MetricsResponse(BaseModel):
    """System metrics."""
    total_predictions: int
    total_frauds_detected: int
    fraud_detection_rate: float
    avg_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    uptime_seconds: float
