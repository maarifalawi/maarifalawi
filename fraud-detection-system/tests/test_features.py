"""Tests for feature engineering pipeline."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.features.realtime import RealtimeFeatures
from src.features.historical import HistoricalFeatures
from src.features.engine import FeatureEngine


class TestRealtimeFeatures:
    """Test real-time feature computation."""

    def setup_method(self):
        self.engine = RealtimeFeatures()

    def test_compute_single_basic(self):
        txn = {
            "timestamp": "2025-01-15T14:30:00",
            "amount": 100.50,
            "merchant_category": "electronics",
            "merchant_country": "US",
            "is_online": True,
        }
        features = self.engine.compute_single(txn)

        assert "hour_of_day" in features
        assert features["hour_of_day"] == 14
        assert features["is_online"] == 1
        assert features["amount_log"] == pytest.approx(np.log1p(100.50), rel=1e-5)
        assert features["is_high_risk_category"] == 0

    def test_high_risk_detection(self):
        txn = {
            "timestamp": "2025-01-15T02:30:00",
            "amount": 6000.00,
            "merchant_category": "gambling",
            "merchant_country": "NG",
            "is_online": True,
        }
        features = self.engine.compute_single(txn)

        assert features["is_night"] == 1
        assert features["is_high_risk_category"] == 1
        assert features["is_high_risk_country"] == 1
        assert features["is_very_large_amount"] == 1

    def test_amount_bucket(self):
        assert RealtimeFeatures._get_amount_bucket(0.50) == 0
        assert RealtimeFeatures._get_amount_bucket(5.00) == 1
        assert RealtimeFeatures._get_amount_bucket(25.00) == 2
        assert RealtimeFeatures._get_amount_bucket(75.00) == 3
        assert RealtimeFeatures._get_amount_bucket(200.00) == 4
        assert RealtimeFeatures._get_amount_bucket(800.00) == 5
        assert RealtimeFeatures._get_amount_bucket(2000.00) == 6
        assert RealtimeFeatures._get_amount_bucket(10000.00) == 7


class TestHistoricalFeatures:
    """Test historical feature computation."""

    def setup_method(self):
        self.engine = HistoricalFeatures()

    def test_compute_single_no_history(self):
        txn = {
            "timestamp": "2025-01-15T14:30:00",
            "amount": 100.0,
            "merchant_category": "electronics",
            "merchant_country": "US",
        }
        features = self.engine.compute_single(txn, history=None)
        assert features["total_txn_count"] == 0
        assert features["is_new_country"] == 1

    def test_compute_single_with_history(self):
        history = pd.DataFrame([
            {"timestamp": "2025-01-14T10:00:00", "amount": 50.0,
             "merchant_category": "grocery", "merchant_country": "US"},
            {"timestamp": "2025-01-14T15:00:00", "amount": 30.0,
             "merchant_category": "gas_station", "merchant_country": "US"},
        ])
        txn = {
            "timestamp": "2025-01-15T14:30:00",
            "amount": 500.0,
            "merchant_category": "electronics",
            "merchant_country": "GB",
        }
        features = self.engine.compute_single(txn, history=history)

        assert features["total_txn_count"] == 2
        assert features["is_new_country"] == 1
        assert features["is_new_category"] == 1
        assert features["amount_zscore"] > 2.0  # 500 is way above 40 avg


class TestFeatureEngine:
    """Test the main feature engine."""

    def setup_method(self):
        self.engine = FeatureEngine()

    def test_compute_single(self):
        txn = {
            "timestamp": "2025-01-15T14:30:00",
            "amount": 100.0,
            "merchant_category": "grocery",
            "merchant_country": "US",
            "is_online": False,
        }
        features = self.engine.compute_single(txn)
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many features
        assert "hour_of_day" in features
        assert "amount_log" in features
