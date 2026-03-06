# AI DevOps Agent for TraceFlix

An AI-powered observability agent that integrates with the existing TraceFlix Kubernetes microservices platform. It collects real-time telemetry from Prometheus, Loki, and Tempo, streams it through a Redis pub-sub layer, stores it in PostgreSQL, and produces actionable analysis. A dedicated **VictoriaMetrics TSDB** provides long-range metric storage with automatic downsampling, enabling trend analysis, memory leak detection, and capacity forecasting over 1-hour to 24-hour windows.

**Dual Analyzer Mode:** The agent supports two analysis backends:
- **Claude API** — sends structured context to Claude Sonnet for LLM-based analysis (default)
- **ML Pipeline** — uses locally-trained models (Isolation Forest, LSTM Autoencoder, LSTM Forecaster, XGBoost, Sentence-BERT + HDBSCAN, Phi-3-mini) for fully offline analysis — no API dependency, designed for PhD research evaluation

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    on-demand-observability namespace                      │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                   │
│  │ movie-service │──│ actor-service │  │review-service│  (TraceFlix)      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                   │
│         │  OTel Java Agent │                  │                           │
│         ▼                  ▼                  ▼                           │
│  ┌──────────────────────────────────────────────────┐                    │
│  │            OTel Collector (:4317)                 │                    │
│  └──────┬──────────────┬───────────────┬────────────┘                    │
│         ▼              ▼               ▼                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐                             │
│  │  Tempo   │   │Prometheus│   │   Loki   │     (Existing Stack)        │
│  │ (traces) │   │(metrics) │   │  (logs)  │                             │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘                             │
└───────┼──────────────┼──────────────┼────────────────────────────────────┘
        │              │              │
        │              │──── remote_write (all OTel metrics) ────┐
        ▼              ▼              ▼                           ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                       devops-agent namespace                             │
│                                                                          │
│  ┌─────────────────────────────────────────┐                             │
│  │  VictoriaMetrics TSDB (StatefulSet)     │     TSDB: Long-Range       │
│  │  - 30-day retention, auto downsampling  │     Metric Storage          │
│  │  - PromQL-compatible API (:8428)        │                             │
│  │  - 7d→5m, 30d→1h resolution            │                             │
│  └──────────────────┬──────────────────────┘                             │
│                     │ PromQL range queries                               │
│                     ▼                                                    │
│  ┌─────────────────────────────────────────┐     STEP 1: Real-Time      │
│  │  Collector (Deployment)                 │     Data Collection         │
│  │  - Queries Prometheus API /api/v1/query │                             │
│  │  - Queries Loki API /loki/api/v1/...    │                             │
│  │  - Queries Tempo API /api/search        │                             │
│  │  - Watches K8s events + pod status      │                             │
│  │  - Queries VictoriaMetrics for trends   │                             │
│  └──────────────────┬──────────────────────┘                             │
│                     │ Redis Streams (5 streams)                          │
│                     ▼                                                    │
│  ┌─────────────────────────────────────────┐     STEP 2: Pub-Sub        │
│  │  Redis 7.4 (StatefulSet)               │     Message Broker          │
│  │  - stream:metrics                       │                             │
│  │  - stream:logs                          │                             │
│  │  - stream:traces                        │                             │
│  │  - stream:events                        │                             │
│  │  - stream:tsdb_trends  ← NEW           │                             │
│  └──────────────────┬──────────────────────┘                             │
│                     │ Consumer Groups                                    │
│                     ▼                                                    │
│  ┌─────────────────────────────────────────┐     STEP 3: Backend        │
│  │  Backend — FastAPI (Deployment x2)      │     Data Store + API       │
│  │  - Consumes 5 Redis Streams             │                             │
│  │  - Stores in PostgreSQL (+ tsdb_trends) │                             │
│  │  - REST API + WebSocket                 │                             │
│  │  - /api/tsdb/trends/* endpoints ← NEW  │                             │
│  └──────────────────┬──────────────────────┘                             │
│                     │                                                    │
│                     ▼                                                    │
│  ┌─────────────────────────────────────────┐     STEP 4: AI Agent       │
│  │  AI Agent (Deployment)                  │     Analysis + Insights    │
│  │  - Queries backend for telemetry        │                             │
│  │  - Fetches TSDB trend summary           │                             │
│  │  - Builds structured context            │                             │
│  │  - ANALYZER_MODE=claude → Claude API    │                             │
│  │  - ANALYZER_MODE=ml → ML Model Server   │                             │
│  │  - Stores analysis results              │                             │
│  └──────────────┬──────────────────────────┘                             │
│                  │ (when ANALYZER_MODE=ml)                                │
│                  ▼                                                        │
│  ┌─────────────────────────────────────────┐     STEP 4b: ML Server    │
│  │  ML Model Server (Deployment, GPU)      │     Local Inference       │
│  │  - Anomaly: IF + LSTM Autoencoder       │                             │
│  │  - Forecast: LSTM + Attention           │                             │
│  │  - Root Cause: XGBoost                  │                             │
│  │  - Log Clustering: SBERT + HDBSCAN      │                             │
│  │  - NLP Report: Phi-3-mini (4-bit)       │                             │
│  └─────────────────────────────────────────┘                             │
└──────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

These must be running in your cluster before deploying the agent:

- Kubernetes 1.26+ (minikube works)
- The **on-demand-observability** namespace deployed (from `on-demand-observability.yaml`)
- TraceFlix services running (movie-service, actor-service, review-service)
- For Claude mode: An Anthropic API key
- For ML mode: NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better), trained models in `ml-models/trained_models/`

## Quick Start

### 1. Build Docker Images

```bash
# From the devops-agent/ directory
docker build -t devops-agent/collector:latest  ./collector
docker build -t devops-agent/backend:latest    ./backend
docker build -t devops-agent/agent:latest      ./agent
docker build -t devops-agent/dashboard:latest  ./dashboard

# Optional: ML model server (only if using ANALYZER_MODE=ml)
docker build -t devops-agent/ml-server:latest -f ml-models/serving/Dockerfile ./ml-models

# If using minikube, build inside the minikube Docker daemon:
eval $(minikube docker-env)
docker build -t devops-agent/collector:latest  ./collector
docker build -t devops-agent/backend:latest    ./backend
docker build -t devops-agent/agent:latest      ./agent
docker build -t devops-agent/dashboard:latest  ./dashboard
# docker build -t devops-agent/ml-server:latest -f ml-models/serving/Dockerfile ./ml-models
```

### 2. Create Secrets

```bash
kubectl apply -f k8s/01-namespace-rbac.yaml

# ANTHROPIC_API_KEY is only required when using ANALYZER_MODE=claude
kubectl create secret generic devops-secrets \
  --namespace devops-agent \
  --from-literal=ANTHROPIC_API_KEY=sk-ant-api03-your-key \
  --from-literal=POSTGRES_PASSWORD=strong-db-password \
  --from-literal=REDIS_PASSWORD=strong-redis-password
```
<!-- kubectl create secret generic devops-secrets \
    --namespace devops-agent \
    --from-literal=REDIS_PASSWORD=ducanh0312 \
    --from-literal=POSTGRES_PASSWORD=ducanh0312 \
    --from-literal=POSTGRES_USER=postgres \
    --from-literal=ANTHROPIC_API_KEY=sk-ant-... -->

### 3. Deploy All Components

```bash
kubectl apply -f k8s/01-namespace-rbac.yaml
kubectl apply -f k8s/02-redis-pubsub.yaml
kubectl apply -f k8s/03-postgres.yaml
kubectl apply -f k8s/03a-victoriametrics.yaml

# Wait for stateful services to be ready
kubectl wait --for=condition=ready pod -l app=redis -n devops-agent --timeout=120s
kubectl wait --for=condition=ready pod -l app=postgres -n devops-agent --timeout=120s
kubectl wait --for=condition=ready pod -l app=victoriametrics -n devops-agent --timeout=120s

# Patch Prometheus to remote_write into VictoriaMetrics
kubectl apply -f k8s/prometheus-remote-write-patch.yaml
kubectl rollout restart deployment/prometheus -n on-demand-observability

kubectl apply -f k8s/04-collector.yaml
kubectl apply -f k8s/05-backend.yaml
kubectl apply -f k8s/06-agent.yaml
kubectl apply -f k8s/07-dashboard.yaml

# Optional: Deploy ML model server (only if using ANALYZER_MODE=ml)
# kubectl apply -f k8s/08-ml-server.yaml
# Then switch agent to ML mode:
# kubectl set env deploy/devops-ai-agent -n devops-agent ANALYZER_MODE=ml
```

### 4. Verify Deployment

```bash
kubectl get pods -n devops-agent
# Expected:
# devops-collector-xxx     1/1  Running
# devops-backend-xxx       1/1  Running
# devops-backend-yyy       1/1  Running
# devops-ai-agent-xxx      1/1  Running
# devops-dashboard-xxx     1/1  Running
# postgres-0               1/1  Running
# redis-0                  1/1  Running
# victoriametrics-0        1/1  Running
# devops-ml-server-xxx     1/1  Running   (only if ML mode deployed)

# Check collector logs (should show VM trend queries)
kubectl logs -f deploy/devops-collector -n devops-agent

# Check VictoriaMetrics health
kubectl exec -n devops-agent victoriametrics-0 -- wget -qO- http://localhost:8428/health

# Verify remote_write is working (should show increasing samples count)
kubectl exec -n devops-agent victoriametrics-0 -- wget -qO- 'http://localhost:8428/api/v1/query?query=vm_rows_inserted_total'

# Check AI agent analysis
kubectl logs -f deploy/devops-ai-agent -n devops-agent
```

### 5. Access the Dashboard

```bash
kubectl port-forward svc/devops-dashboard -n devops-agent 3000:3000
```

Open http://localhost:3000 in your browser. The dashboard serves a React SPA via nginx, which reverse-proxies `/api/*` and `/ws/*` requests to the backend service inside the cluster. No separate backend port-forward is needed when accessing through the dashboard.

The dashboard auto-refreshes every 15 seconds and connects to the backend WebSocket at `/ws/live` for real-time log streaming. The connection indicator in the top-right corner shows whether the WebSocket link is active.

### 6. Access the Backend API Directly

If you want to query the API outside the dashboard:

```bash
kubectl port-forward svc/devops-backend -n devops-agent 8000:8000

# Health check
curl http://localhost:8000/api/health

# Dashboard stats
curl http://localhost:8000/api/stats

# Latest AI analysis
curl http://localhost:8000/api/analysis/latest

# Query error logs
curl "http://localhost:8000/api/logs?severity=ERROR&since_minutes=60"

# Query slow traces
curl "http://localhost:8000/api/traces?slow_only=true&since_minutes=60"

# TSDB trend summary (degrading / stable / improving)
curl http://localhost:8000/api/tsdb/trends/summary

# TSDB trends for a specific metric
curl "http://localhost:8000/api/tsdb/trends?query_name=latency_p99_1h&since_hours=6"

# Latest snapshot per metric
curl http://localhost:8000/api/tsdb/trends/latest
```

## ML Pipeline (Alternative to Claude API)

The `ml-models/` directory contains a complete hybrid ML pipeline that can replace the Claude API for analysis. See `ml-models/README.md` for full documentation.

### Quick Start — ML Mode

```bash
# 1. Install Python dependencies
cd ml-models
pip install -r requirements.txt

# 2. Generate training data + train all models (~25 min on RTX 3060)
python -m pipeline.train_all

# 3. Start ML model server locally
python -m serving.model_server
# Server runs on port 8001 — test with: curl http://localhost:8001/health

# 4. Switch agent to ML mode
export ANALYZER_MODE=ml
export ML_SERVER_URL=http://localhost:8001
```

### Deploy ML Server in Kubernetes

```bash
# Build ML server image
eval $(minikube docker-env)
docker build -t devops-agent/ml-server:latest -f ml-models/serving/Dockerfile ./ml-models

# Deploy
kubectl apply -f k8s/08-ml-server.yaml

# Copy trained models to PVC
kubectl cp ml-models/trained_models/ devops-agent/devops-ml-server-xxx:/app/trained_models/

# Switch agent to ML mode
kubectl set env deploy/devops-ai-agent -n devops-agent ANALYZER_MODE=ml
```

### ML Models Summary

| Model | Task | Architecture | VRAM |
|-------|------|-------------|------|
| Anomaly Detector | Detect metric anomalies | Isolation Forest + LSTM Autoencoder | ~100MB |
| Forecaster | Predict future metrics | LSTM + Multi-head Attention | ~200MB |
| Root Cause Classifier | Identify causes | XGBoost (gradient boosted trees) | CPU only |
| Log Clusterer | Group log patterns | Sentence-BERT + UMAP + HDBSCAN | ~400MB |
| Report Generator | Natural language synthesis | Phi-3-mini-4k (4-bit NF4) | ~3.5GB |

### Research Notebook

A Jupyter notebook for training, evaluation, and benchmarking is at `ml-models/notebooks/research_training.ipynb`. It includes per-model metrics, confusion matrices, latency benchmarks, cost analysis, and statistical significance testing (paired t-test, Wilcoxon, bootstrap CIs).

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| GET | `/api/stats` | Dashboard summary (errors, latency, events) |
| GET | `/api/logs?service=X&severity=Y&since_minutes=60` | Query logs |
| GET | `/api/metrics?metric_name=X&since_minutes=60` | Query metrics |
| GET | `/api/traces?service=X&slow_only=true&errors_only=true` | Query traces |
| GET | `/api/events?source=kubernetes&reason=OOMKilled` | Query K8s events |
| GET | `/api/analysis` | List AI analysis results |
| GET | `/api/analysis/latest` | Most recent analysis |
| POST | `/api/analysis` | Store analysis (used by agent) |
| GET | `/api/tsdb/trends?query_name=X&since_hours=6` | Query TSDB trend data |
| GET | `/api/tsdb/trends/latest` | Latest snapshot per metric query |
| GET | `/api/tsdb/trends/summary` | Categorised trends: degrading / stable / improving |
| WS | `/ws/live` | Real-time WebSocket stream |

## Data Flow

The system follows a clear pipeline with a dedicated TSDB layer for long-range metrics:

**TSDB — VictoriaMetrics** receives all OTel metrics via Prometheus `remote_write`. It retains 30 days of data with automatic downsampling (full resolution for 7 days, then 5-minute resolution, then 1-hour resolution after 30 days). The collector queries VictoriaMetrics directly for range-based trend analysis across 1-hour and 24-hour windows.

**Step 1 — Collector** polls the existing observability backends (Prometheus for point-in-time metrics, Loki for logs, Tempo for traces), the K8s API for pod events, and VictoriaMetrics for long-range trend data. Each data type is published to its own Redis Stream, including a dedicated `stream:tsdb_trends` for trend analysis results.

**Step 2 — Redis Pub/Sub** provides durable, ordered message delivery using Redis Streams with consumer groups across five streams. This decouples the collector from the backend and ensures no data loss if the backend restarts.

**Step 3 — Backend** consumes from all five Redis Streams using consumer groups, inserts data into PostgreSQL tables (logs, metrics, traces, events, analysis, tsdb_trends), and exposes a REST API plus a WebSocket endpoint for real-time streaming. Three new `/api/tsdb/trends/*` endpoints serve categorised trend data.

**Step 4 — AI Agent** runs on a configurable interval (default 5 minutes). It queries the backend API including the TSDB trend summary, builds a structured context document containing errors, slow traces, K8s events, metrics, and degradation trends, sends it to Claude API, parses the structured JSON response, and stores the analysis result back in the backend. The TSDB trend data enables Claude to detect memory leaks, latency drift, error rate acceleration, and capacity saturation patterns that would be invisible from point-in-time snapshots alone.

## Configuration

All services are configured via environment variables. Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `METRICS_INTERVAL` | 15 | Seconds between Prometheus queries |
| `LOGS_INTERVAL` | 10 | Seconds between Loki queries |
| `TRACES_INTERVAL` | 20 | Seconds between Tempo searches |
| `TSDB_TRENDS_INTERVAL` | 60 | Seconds between VictoriaMetrics trend queries |
| `ANALYSIS_INTERVAL` | 300 | Seconds between AI analysis runs |
| `LOOKBACK_MINUTES` | 30 | How far back each analysis looks |
| `RETENTION_DAYS` | 7 | Data retention in PostgreSQL |
| `CLAUDE_MODEL` | claude-sonnet-4-20250514 | Claude model for analysis |
| `ANALYZER_MODE` | claude | Analysis backend: `claude` or `ml` |
| `ML_SERVER_URL` | http://devops-ml-server...:8001 | ML model server endpoint (when mode=ml) |
| `ML_TIMEOUT` | 30 | ML server request timeout in seconds |
| `RUN_MODE` | continuous | Agent mode: `continuous` or `once` |
| `VICTORIA_METRICS_URL` | http://victoriametrics....:8428 | VictoriaMetrics query endpoint |

VictoriaMetrics-specific settings are configured via container args in `03a-victoriametrics.yaml`:

| Arg | Value | Description |
|-----|-------|-------------|
| `--retentionPeriod` | 30d | How long to keep metric data |
| `--downsampling.period` | 7d:5m,30d:1h | Resolution reduction over time |
| `--dedup.minScrapeInterval` | 15s | Deduplication window for HA Prometheus |
| `--memory.allowedPercent` | 60 | Max memory usage for merge operations |

## What the AI Agent Analyzes

The agent (in either Claude API or ML mode) provides structured JSON analysis covering seven areas specific to the TraceFlix architecture:

1. **Health Status** — overall cluster health (HEALTHY / DEGRADED / CRITICAL)
2. **Anomaly Detection** — error spikes, latency degradation, resource exhaustion
3. **Root Cause Analysis** — correlates traces, logs, and events to identify root causes (particularly the N+1 sequential call pattern in movie-service)
4. **Performance Insights** — per-service latency, JVM memory/GC, thread pools, inter-service bottlenecks
5. **TSDB Trend Analysis** — long-range degradation detection from VictoriaMetrics: memory leak patterns (monotonic heap increase), latency drift over 1h/24h windows, error rate acceleration, and capacity saturation forecasting
6. **Recommendations** — prioritised actionable steps with kubectl commands
7. **Incident Timeline** — chronological reconstruction of any active incidents

## TSDB Trend Queries

The collector runs the following PromQL range queries against VictoriaMetrics every 60 seconds:

| Query Name | Window | Description |
|------------|--------|-------------|
| `request_rate_1h` | 1 hour | Request rate per service |
| `latency_p99_1h` | 1 hour | P99 latency trend per service |
| `latency_p50_1h` | 1 hour | P50 latency trend per service |
| `error_rate_1h` | 1 hour | 5xx error rate per service |
| `jvm_heap_used_1h` | 1 hour | JVM heap memory (leak detection) |
| `jvm_gc_pause_1h` | 1 hour | GC pause rate trend |
| `request_rate_24h` | 24 hours | Daily traffic pattern |
| `latency_p99_24h` | 24 hours | Full-day latency baseline |

Each series is analysed for direction (increasing / stable / decreasing), percentage change between first and second half of the window, and volatility (coefficient of variation). The `/api/tsdb/trends/summary` endpoint categorises all series into degrading (trend_pct > 10%), stable, or improving, which the AI agent uses to prioritise its analysis.

 Run these three commands in order:                                                                                                                               
  # 1. Build the ML server image                                                                                                                                
  make build-ml

  # 2. Load it into the k8s node's containerd (bypasses Docker Hub)
  docker save devops-agent/ml-server:latest | docker exec -i desktop-control-plane ctr images import -

  # 3. Force a new pod so it picks up the now-local image
  kubectl delete pod -l app=devops-ml-server -n devops-agent