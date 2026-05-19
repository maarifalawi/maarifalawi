"""Tests for FastAPI endpoints."""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient


class TestAPISchemas:
    """Test API schema validation."""

    def test_transaction_request_valid(self):
        from src.api.schemas import TransactionRequest

        txn = TransactionRequest(
            customer_id="CUST_001",
            amount=150.00,
            merchant_category="electronics",
            merchant_country="US",
        )
        assert txn.customer_id == "CUST_001"
        assert txn.amount == 150.00

    def test_transaction_request_invalid_amount(self):
        from src.api.schemas import TransactionRequest

        with pytest.raises(Exception):
            TransactionRequest(
                customer_id="CUST_001",
                amount=-10.00,  # Negative amount
                merchant_category="electronics",
            )

    def test_fraud_score_response(self):
        from src.api.schemas import FraudScoreResponse

        response = FraudScoreResponse(
            transaction_id="TXN_001",
            customer_id="CUST_001",
            fraud_probability=0.85,
            is_fraud=True,
            risk_level="HIGH",
            threshold=0.5,
            latency_ms=3.2,
            models_used={"xgboost": 0.83, "lightgbm": 0.87},
            scored_at="2025-01-15T14:30:00",
        )
        assert response.is_fraud is True
        assert response.risk_level == "HIGH"
