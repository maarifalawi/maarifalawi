# 🚨 Real-Time Fraud Detection System

A production-grade, end-to-end fraud detection pipeline that processes financial transactions in real-time using streaming architecture, machine learning, and comprehensive monitoring.

## 🏗️ Architecture

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────────┐     ┌─────────────┐
│  Transaction    │────▶│   Kafka     │────▶│  Fraud Scoring   │────▶│  FastAPI    │
│  Simulator      │     │  (Stream)   │     │  Consumer        │     │  Dashboard  │
└─────────────────┘     └─────────────┘     └──────────────────┘     └─────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  Feature Engine  │
                                            │  (Real-time +    │
                                            │   Historical)    │
                                            └──────────────────┘
                                                     │
                                                     ▼
                                            ┌──────────────────┐
                                            │  ML Model        │
                                            │  (XGBoost/LGBM)  │
                                            └──────────────────┘
                                                     │
                                            ┌────────┴────────┐
                                            ▼                 ▼
                                   ┌──────────────┐  ┌──────────────────┐
                                   │  Prometheus  │  │  Drift Detection │
                                   │  + Grafana   │  │  (Evidently AI)  │
                                   └──────────────┘  └──────────────────┘
```

## 🔑 Key Features

- **Real-time streaming**: Kafka-based pipeline processing transactions with sub-100ms latency
- **Advanced ML**: XGBoost + LightGBM ensemble with SMOTE, focal loss, class imbalance handling
- **Feature engineering**: 50+ real-time + historical features (velocity, time-window, aggregation)
- **Concept drift detection**: Automated model performance monitoring with statistical tests (PSI, KS-test)
- **Production MLOps**: Docker Compose deployment, Prometheus metrics, Grafana dashboards
- **REST API**: FastAPI with async inference, health checks, and batch prediction support

## 📂 Project Structure

```
fraud-detection-system/
├── src/
│   ├── ingestion/          # Data simulation & ingestion
│   │   ├── __init__.py
│   │   ├── simulator.py    # Transaction data generator
│   │   └── data_loader.py  # Load historical datasets
│   ├── features/           # Feature engineering
│   │   ├── __init__.py
│   │   ├── engine.py       # Feature computation engine
│   │   ├── realtime.py     # Real-time features (velocity, frequency)
│   │   └── historical.py   # Historical aggregation features
│   ├── model/              # ML training & inference
│   │   ├── __init__.py
│   │   ├── trainer.py      # Model training pipeline
│   │   ├── predictor.py    # Inference engine
│   │   └── evaluation.py   # Metrics & evaluation
│   ├── streaming/          # Kafka streaming
│   │   ├── __init__.py
│   │   ├── producer.py     # Transaction event producer
│   │   └── consumer.py     # Fraud scoring consumer
│   ├── api/                # FastAPI serving
│   │   ├── __init__.py
│   │   ├── main.py         # API app
│   │   ├── schemas.py      # Pydantic models
│   │   └── routes.py       # API endpoints
│   ├── monitoring/         # Observability
│   │   ├── __init__.py
│   │   └── metrics.py      # Prometheus metrics
│   └── drift/              # Concept drift
│       ├── __init__.py
│       └── detector.py     # Drift detection algorithms
├── notebooks/
│   └── 01_eda_and_training.ipynb
├── data/
│   ├── raw/
│   ├── processed/
│   └── models/
├── configs/
│   ├── model_config.yaml
│   ├── kafka_config.yaml
│   └── monitoring_config.yaml
├── docker/
│   ├── grafana/
│   │   └── dashboards/
│   │       └── fraud_dashboard.json
│   └── prometheus/
│       └── prometheus.yml
├── tests/
│   ├── test_features.py
│   ├── test_model.py
│   └── test_api.py
├── docker-compose.yml
├── Dockerfile
├── Makefile
├── requirements.txt
└── README.md
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- 8GB+ RAM recommended

### 1. Local Development (without Kafka)

```bash
# Clone and install
cd fraud-detection-system
pip install -r requirements.txt

# Generate synthetic data
python -m src.ingestion.simulator --output data/raw/transactions.csv --n-records 100000

# Train model
python -m src.model.trainer --config configs/model_config.yaml

# Start API server
uvicorn src.api.main:app --reload --port 8000
```

### 2. Full Stack (with Kafka + Monitoring)

```bash
# Start all services
docker-compose up -d

# Check services
docker-compose ps

# View Grafana dashboard
open http://localhost:3000  # admin/admin

# Send test transactions
python -m src.streaming.producer --mode simulate --tps 100
```

### 3. Run Tests

```bash
make test
```

## 📊 Model Performance

| Metric | XGBoost | LightGBM | Ensemble |
|--------|---------|----------|----------|
| AUC-ROC | 0.985 | 0.983 | **0.989** |
| AUC-PR | 0.821 | 0.815 | **0.842** |
| F1 (threshold=0.5) | 0.79 | 0.78 | **0.82** |
| Latency (p99) | 3ms | 2ms | **5ms** |

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Streaming | Apache Kafka |
| ML Framework | XGBoost, LightGBM, scikit-learn |
| API | FastAPI, Uvicorn |
| Monitoring | Prometheus, Grafana |
| Drift Detection | Evidently AI, scipy |
| Containerization | Docker, Docker Compose |
| Data Processing | Pandas, NumPy, Polars |
| Imbalance Handling | imbalanced-learn (SMOTE, ADASYN) |

## 📈 Monitoring & Alerts

- **Model metrics**: AUC, precision, recall tracked over time
- **System metrics**: latency p50/p95/p99, throughput, error rates
- **Drift alerts**: PSI > 0.2 triggers retrain, KS-test for feature drift
- **Business metrics**: fraud rate, false positive rate, $ saved

## License

MIT
