"""
Central configuration for the ML training pipeline.

All hyperparameters, feature definitions, model paths, and
training settings are defined here so experiments can be
tracked and reproduced consistently.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict

# ── Paths ────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "trained_models"
LOG_DIR = BASE_DIR / "logs"
ARTIFACT_DIR = BASE_DIR / "artifacts"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, ARTIFACT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Backend connection ───────────────────────────────────────
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ── Device ───────────────────────────────────────────────────
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[config] Using device: {DEVICE}")
if DEVICE == "cuda":
    print(f"[config] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[config] VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")


# ── Feature Definitions ─────────────────────────────────────
# Metrics collected from Prometheus / VictoriaMetrics
METRIC_FEATURES = [
    "request_rate",
    "error_rate",
    "latency_p50",
    "latency_p99",
    "jvm_heap_used",
    "jvm_gc_pause_seconds",
    "cpu_usage",
    "memory_usage",
]

# Services in the TraceFlix cluster
SERVICES = [
    "movie-service",
    "actor-service",
    "review-service",
]

# Anomaly labels for root cause classification
ROOT_CAUSE_LABELS = [
    "memory_leak",
    "cpu_saturation",
    "connection_pool_exhaustion",
    "downstream_timeout",
    "disk_io_bottleneck",
    "jvm_gc_pressure",
    "request_spike",
    "config_error",
    "deployment_regression",
    "network_partition",
    "normal",
]

# Log severity levels
LOG_SEVERITIES = ["DEBUG", "INFO", "WARN", "ERROR", "FATAL"]


@dataclass
class AnomalyDetectionConfig:
    """Isolation Forest + LSTM Autoencoder ensemble."""

    # ── Isolation Forest ─────────────────────────────────────
    if_n_estimators: int = 200
    if_contamination: float = 0.05      # expected anomaly ratio
    if_max_samples: str = "auto"
    if_random_state: int = 42

    # ── LSTM Autoencoder ─────────────────────────────────────
    lstm_input_dim: int = len(METRIC_FEATURES)
    lstm_hidden_dim: int = 64
    lstm_latent_dim: int = 32
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.2
    lstm_seq_length: int = 30           # 30 time-steps lookback
    lstm_batch_size: int = 64
    lstm_epochs: int = 100
    lstm_lr: float = 1e-3
    lstm_patience: int = 15             # early stopping
    lstm_threshold_percentile: float = 95.0  # reconstruction error threshold

    # ── Ensemble ─────────────────────────────────────────────
    ensemble_weights: Dict[str, float] = field(default_factory=lambda: {
        "isolation_forest": 0.4,
        "lstm_autoencoder": 0.6,
    })

    model_path: Path = MODEL_DIR / "anomaly"


@dataclass
class ForecastingConfig:
    """LSTM Forecaster with attention mechanism."""

    input_dim: int = len(METRIC_FEATURES)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    attention_heads: int = 4
    seq_length: int = 60                # 60 time-steps lookback
    forecast_horizon: int = 15          # predict 15 steps ahead
    batch_size: int = 32
    epochs: int = 150
    lr: float = 5e-4
    patience: int = 20
    scheduler_factor: float = 0.5
    scheduler_patience: int = 10

    model_path: Path = MODEL_DIR / "forecasting"


@dataclass
class RootCauseConfig:
    """XGBoost multi-label root cause classifier."""

    n_estimators: int = 300
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 3
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    scale_pos_weight: float = 1.0       # auto-computed during training
    random_state: int = 42

    # Feature engineering
    window_sizes: List[int] = field(default_factory=lambda: [5, 15, 30])
    use_cross_service_features: bool = True
    n_labels: int = len(ROOT_CAUSE_LABELS)

    model_path: Path = MODEL_DIR / "root_cause"


@dataclass
class LogClusteringConfig:
    """Sentence-BERT embeddings + HDBSCAN clustering."""

    # ── Embedding ────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"  # ~90MB, fast
    embedding_dim: int = 384
    batch_size: int = 256

    # ── HDBSCAN ──────────────────────────────────────────────
    min_cluster_size: int = 5
    min_samples: int = 3
    cluster_selection_method: str = "eom"  # excess of mass
    metric: str = "euclidean"

    # ── UMAP (for dimensionality reduction before clustering)
    umap_n_components: int = 15
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1

    # ── Pattern extraction ───────────────────────────────────
    top_k_patterns: int = 20            # keep top K cluster patterns
    pattern_min_frequency: int = 3

    model_path: Path = MODEL_DIR / "log_clustering"


@dataclass
class NLPConfig:
    """Phi-3-mini quantised for report generation."""

    model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    load_in_4bit: bool = True           # 4-bit quantisation via bitsandbytes
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    max_new_tokens: int = 1024
    temperature: float = 0.3            # low temp for structured output
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    do_sample: bool = True

    # Estimated VRAM: ~3.5GB for 4-bit Phi-3-mini
    model_path: Path = MODEL_DIR / "nlp"


@dataclass
class TrainingConfig:
    """Global training orchestration settings."""

    seed: int = 42
    val_split: float = 0.15
    test_split: float = 0.15
    n_synthetic_samples: int = 50000    # for initial training data
    n_synthetic_incidents: int = 2000   # labelled incident scenarios

    # Data generation
    metric_interval_seconds: int = 15   # simulated scrape interval
    incident_duration_range: tuple = (60, 600)  # seconds
    noise_level: float = 0.05

    # MLflow tracking
    mlflow_tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", f"file://{ARTIFACT_DIR}/mlruns")
    experiment_name: str = "devops-ai-agent-ml"


# ── Singleton instances ──────────────────────────────────────
anomaly_config = AnomalyDetectionConfig()
forecast_config = ForecastingConfig()
root_cause_config = RootCauseConfig()
log_cluster_config = LogClusteringConfig()
nlp_config = NLPConfig()
training_config = TrainingConfig()
