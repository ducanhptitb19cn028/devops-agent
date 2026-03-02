# ML Models — Hybrid AI Analysis Pipeline

> Replaces the Claude API with locally-trained ML models for DevOps observability analysis. Designed for PhD research: each model is independently evaluable, the ensemble is benchmarkable against the API baseline, and the full pipeline includes statistical significance testing.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Telemetry Data (from Collector)               │
│         Metrics · Logs · Traces · Events · TSDB Trends          │
└─────────────┬───────────────────────────────────┬───────────────┘
              │                                   │
   ┌──────────▼──────────┐             ┌──────────▼──────────┐
   │   Metric Features   │             │   Log Messages      │
   │  (8 time-series)    │             │  (raw text)         │
   └──┬─────┬────────┬───┘             └──────────┬──────────┘
      │     │        │                            │
      ▼     ▼        ▼                            ▼
   ┌─────┐ ┌──────┐ ┌─────────┐          ┌──────────────┐
   │ IF  │ │ LSTM │ │  LSTM   │          │ Sentence-BERT│
   │     │ │  AE  │ │Forecast │          │  + HDBSCAN   │
   └──┬──┘ └──┬───┘ └────┬────┘          └──────┬───────┘
      │       │          │                       │
      ▼       ▼          ▼                       ▼
   Anomaly  Anomaly   Forecasts &         Log Patterns &
   Scores   Scores    Breach Alerts       Cluster Labels
      │       │          │                       │
      └───┬───┘          │                       │
          ▼              │                       │
   ┌──────────┐          │                       │
   │ XGBoost  │◄─────────┤                       │
   │Root Cause│          │                       │
   └────┬─────┘          │                       │
        │                │                       │
        ▼                ▼                       ▼
   ┌─────────────────────────────────────────────────┐
   │        Phi-3-mini (4-bit quantised)             │
   │   Synthesises structured ML outputs into a      │
   │   natural language JSON analysis report          │
   └──────────────────────┬──────────────────────────┘
                          │
                          ▼
              Dashboard / Agent / Backend
              (same JSON schema as Claude API)
```

## Models

| Model | Task | Architecture | Training | Inference | VRAM |
|-------|------|-------------|----------|-----------|------|
| Anomaly Detector | Detect metric anomalies | Isolation Forest + LSTM Autoencoder ensemble | ~5 min | ~10ms | ~100MB |
| Forecaster | Predict future metric values | LSTM + Multi-head Attention | ~15 min | ~15ms | ~200MB |
| Root Cause Classifier | Identify probable causes | XGBoost (gradient boosted trees) | ~1 min | ~2ms | CPU only |
| Log Clusterer | Group log patterns | Sentence-BERT + UMAP + HDBSCAN | ~3 min | ~50ms/batch | ~400MB |
| Report Generator | Natural language synthesis | Phi-3-mini-4k (4-bit NF4) | Pre-trained | ~3s | ~3.5GB |
| **Total** | | | **~25 min** | **~3.1s** | **~4.2GB** |

## Quick Start

### 1. Install Dependencies

```bash
cd ml-models
pip install -r requirements.txt
```

> For GPU support ensure CUDA 12.1+ and PyTorch with CUDA are installed.

### 2. Generate Training Data

```bash
python -m pipeline.train_all --data-only
```

This creates synthetic telemetry data in `data/` mimicking real TraceFlix cluster behaviour with 7 types of injected anomalies.

### 3. Train All Models

```bash
python -m pipeline.train_all
```

Train individual models:

```bash
python -m pipeline.train_all --model anomaly
python -m pipeline.train_all --model forecasting
python -m pipeline.train_all --model root_cause
python -m pipeline.train_all --model log_clustering
```

Trained artifacts are saved to `trained_models/`.

### 4. Start the Model Server

```bash
python -m serving.model_server
```

Server runs on port 8001. Test it:

```bash
curl http://localhost:8001/health
curl http://localhost:8001/models
```

### 5. Switch the Agent to ML Mode

In the agent's environment, set:

```bash
export ML_SERVER_URL=http://devops-ml-server.devops-agent.svc.cluster.local:8001
```

Then in `agent/agent.py`, replace:

```python
# Before:
# from claude_analyzer import ClaudeAnalyzer
# analyzer = ClaudeAnalyzer()

# After:
import sys; sys.path.insert(0, "/app/ml-models")
from agent_integration import MLAnalyzer
analyzer = MLAnalyzer()
```

The `MLAnalyzer` class has the same `.analyze(context)` interface as `ClaudeAnalyzer`, so the rest of the agent code remains unchanged.

## Project Structure

```
ml-models/
├── config.py                      # Central hyperparameters & paths
├── requirements.txt               # Python dependencies
├── agent_integration.py           # Sync drop-in replacement for ClaudeAnalyzer
│
├── data/
│   └── generators/
│       ├── metric_generator.py    # Synthetic metrics with anomaly injection
│       ├── log_generator.py       # Synthetic logs with pattern templates
│       └── collect_real_data.py   # Pull live telemetry for retraining
│
├── models/
│   ├── anomaly/
│   │   └── detector.py            # IF + LSTM-AE ensemble
│   ├── forecasting/
│   │   └── forecaster.py          # LSTM + Attention forecaster
│   ├── root_cause/
│   │   └── classifier.py          # XGBoost with feature engineering
│   ├── log_clustering/
│   │   └── clusterer.py           # Sentence-BERT + UMAP + HDBSCAN
│   └── nlp/
│       └── report_generator.py    # Phi-3-mini 4-bit report synthesis
│
├── pipeline/
│   ├── train_all.py               # Training orchestrator
│   └── smoke_test.py              # End-to-end validation (~2 min, CPU)
│
├── evaluation/
│   └── benchmark.py               # ML vs API comparison framework
│
├── notebooks/
│   └── research_training.ipynb    # PhD research: training, eval, benchmarks
│
├── serving/
│   ├── model_server.py            # FastAPI inference server
│   └── Dockerfile                 # GPU-enabled container
│
└── trained_models/                # Generated after training (gitignored)
```

## API Reference

### Model Server Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with VRAM usage |
| `/models` | GET | Loaded model info |
| `/predict/anomaly` | POST | Anomaly detection on metric window |
| `/predict/forecast` | POST | Multi-step metric forecasting |
| `/predict/root_cause` | POST | Root cause classification |
| `/predict/log_cluster` | POST | Log pattern matching |
| `/analyse` | POST | Full pipeline — all models + NLP report |

### Full Analysis Request

```json
POST /analyse
{
  "metrics": {
    "service": "movie-service",
    "timestamps": ["2025-01-01T00:00:00Z", ...],
    "metrics": {
      "request_rate": [50.2, 51.1, ...],
      "error_rate": [0.005, 0.008, ...],
      "latency_p99": [120, 135, ...],
      ...
    }
  },
  "logs": {
    "messages": ["Connection refused to postgres-0:5432", ...]
  },
  "stats": {
    "total_logs": 4287,
    "total_errors": 23
  }
}
```

Response matches the existing dashboard JSON schema exactly.

## Evaluation Framework

The benchmark module (`evaluation/benchmark.py`) provides:

**Per-model metrics:**
- Anomaly detection: Precision, Recall, F1, AUC-ROC, FPR
- Forecasting: MAE, RMSE, MAPE, Directional Accuracy (per feature)
- Root cause: Accuracy, F1 (weighted/macro), Top-k Accuracy, per-class report
- Log clustering: Silhouette Score, ARI, NMI, V-measure

**System-level evaluation:**
- End-to-end latency benchmarking (ML pipeline vs Claude API)
- Cost analysis (GPU amortisation vs API token spend)
- Statistical significance: paired t-test, Wilcoxon signed-rank, bootstrap CIs

**For PhD research:**
- Each model can be evaluated independently (separate chapter per model)
- Ensemble vs individual model ablation studies
- ML ensemble vs Claude API head-to-head comparison
- Publication-ready metrics tables

## Hyperparameter Tuning

All hyperparameters are centralised in `config.py`. Key tunables:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `lstm_seq_length` | 30 | Lookback window for anomaly/forecast |
| `lstm_hidden_dim` | 64/128 | LSTM hidden state size |
| `if_contamination` | 0.05 | Expected anomaly ratio |
| `forecast_horizon` | 15 | Steps ahead to predict |
| `xgb_max_depth` | 8 | Tree depth for root cause |
| `hdbscan_min_cluster_size` | 5 | Minimum log cluster size |
| `nlp_temperature` | 0.3 | LLM generation temperature |

## Kubernetes Deployment

```bash
# Build ML server image (with GPU support)
eval $(minikube docker-env)
docker build -t devops-agent/ml-server:latest -f serving/Dockerfile .

# Deploy
kubectl apply -f ../k8s/08-ml-server.yaml

# Copy trained models into the PVC
kubectl cp trained_models/ devops-agent/devops-ml-server-xxx:/app/trained_models/

# Verify
kubectl port-forward svc/devops-ml-server -n devops-agent 8001:8001
curl http://localhost:8001/health
```

## Training Data

The synthetic data generators create realistic telemetry with 7 injected anomaly types:

| Anomaly | Pattern | Affected Metrics |
|---------|---------|-----------------|
| Memory Leak | Monotonic heap increase | `jvm_heap_used`, `memory_usage`, `jvm_gc_pause` |
| CPU Saturation | Sustained high CPU + bursts | `cpu_usage`, `latency_p99` |
| Downstream Timeout | Sudden P99 spike + recovery | `latency_p99`, `latency_p50`, `error_rate` |
| Error Storm | Correlated error rate spike | `error_rate` |
| GC Pressure | Heap oscillation + long pauses | `jvm_gc_pause`, `jvm_heap_used`, `latency_p99` |
| Request Flood | Traffic ramp → saturation | `request_rate`, `cpu_usage`, `error_rate` |
| Deployment Regression | Step-change in baselines | `latency_p50`, `latency_p99`, `cpu_usage` |

For PhD research: the generators are parameterised so you can study detection performance as anomaly characteristics vary (intensity, duration, noise level).

## Extending with Real Data

Once the cluster is running and collecting real telemetry, you can retrain on actual data:

```python
import requests
import pandas as pd

# Fetch real metrics from backend
metrics = requests.get("http://localhost:8000/api/metrics?since_minutes=1440").json()
df = pd.DataFrame(metrics["metrics"])

# Retrain anomaly detector
from models.anomaly.detector import AnomalyDetector
detector = AnomalyDetector()
detector.train_lstm_autoencoder(df)
detector.save()
```

Or use the built-in collector:

```bash
# Collect last 24 hours of data
python -m data.generators.collect_real_data --hours 24

# Collect and auto-retrain with heuristic labelling
python -m data.generators.collect_real_data --hours 168 --retrain
```

The pipeline supports incremental retraining, so models improve as more real data accumulates.

## Smoke Test

Run the full pipeline validation on CPU in ~2 minutes:

```bash
python -m pipeline.smoke_test
```

This validates: config loading, data generators, all 4 model architectures (train + inference + save/load), NLP fallback, evaluation framework, API schemas, and agent integration.
