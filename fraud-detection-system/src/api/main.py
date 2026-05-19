"""
FastAPI Application - Fraud Detection API
Production-ready REST API for real-time fraud scoring.
"""

import os
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router, set_dependencies

# App configuration
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time fraud detection system with ML-powered scoring. "
        "Provides single and batch transaction scoring with sub-10ms latency."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
async def startup_event():
    """Load model and initialize components on startup."""
    print("=" * 50)
    print("Starting Fraud Detection API...")
    print("=" * 50)

    model_dir = os.getenv("MODEL_DIR", "data/models/fraud_model_v1")

    try:
        from src.features.engine import FeatureEngine
        from src.model.predictor import FraudPredictor

        # Initialize feature engine
        feature_engine = FeatureEngine()
        print("  Feature engine initialized")

        # Load ML model
        predictor = FraudPredictor(model_dir=model_dir)
        predictor.load()
        print("  Model loaded successfully")

        # Inject dependencies
        set_dependencies(predictor, feature_engine)
        print("\n  API ready for predictions!")

    except FileNotFoundError as e:
        print(f"\n  WARNING: {e}")
        print("  API starting in degraded mode (no model)")
        print("  Train a model first: python -m src.model.trainer")

    except Exception as e:
        print(f"\n  ERROR loading model: {e}")
        print("  API starting in degraded mode")

    print("=" * 50)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "service": "Fraud Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
        "predict": "/api/v1/predict",
    }
